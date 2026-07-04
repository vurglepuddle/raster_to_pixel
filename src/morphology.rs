//! Opt-in morphology cleanup for the quantized pixel-art grid (after
//! quantization, before nearest-neighbor scaling). Alpha at this stage is
//! binary (0 or 255) and every color is a palette color, so the passes only
//! ever clear pixels or reuse existing neighbor colors — they never invent
//! new colors. Like `pipeline`, this module uses the approved `image` crate.
//!
//! Passes, in application order:
//! 1. `fill-pinholes`  — fill tiny enclosed transparent holes.
//! 2. `halo-clean`     — drop light, low-chroma matte fringes along edges.
//! 3. `jaggy-clean`    — drop single diagonal-dangling corner nubs.
//! 4. `remove-orphans` — drop tiny isolated opaque specks.
//!
//! All passes decide on a snapshot and then apply, so results are
//! deterministic and independent of scan order. Intentional single-pixel
//! details are protected by default (see `protect_details` on `cleanup`).

use image::RgbaImage;

use crate::color::{oklab_dist2, srgb8_to_oklab};

/// Cleanup aggressiveness. `None` disables everything (pipeline default).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CleanupPreset {
    None,
    /// Pinholes (1px) + isolated 1px orphans.
    Conservative,
    /// + halo cleanup, jaggy cleanup, 2px pinholes.
    Balanced,
    /// Lower halo threshold, two halo passes, 4px pinholes, 2px orphans.
    Aggressive,
}

/// What a cleanup run changed, for diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CleanupStats {
    pub pinholes_filled: usize,
    pub halo_removed: usize,
    pub jaggies_removed: usize,
    pub orphans_removed: usize,
}

impl CleanupStats {
    pub fn total(&self) -> usize {
        self.pinholes_filled + self.halo_removed + self.jaggies_removed + self.orphans_removed
    }
}

/// Run the preset's passes in place on a quantized pixel-art grid.
///
/// `protect_details` keeps isolated single pixels that repeat nearby (dotted
/// patterns, starfields) instead of treating them as noise.
pub fn cleanup(img: &mut RgbaImage, preset: CleanupPreset, protect_details: bool) -> CleanupStats {
    let mut stats = CleanupStats::default();
    let (pinhole_max, halo, halo_passes, orphan_max) = match preset {
        CleanupPreset::None => return stats,
        CleanupPreset::Conservative => (1, None, 0, 1),
        CleanupPreset::Balanced => (2, Some(0.14f32), 1, 1),
        CleanupPreset::Aggressive => (4, Some(0.10f32), 2, 2),
    };

    stats.pinholes_filled = fill_pinholes(img, pinhole_max);
    if let Some(threshold) = halo {
        for _ in 0..halo_passes {
            let removed = halo_clean(img, threshold);
            stats.halo_removed += removed;
            if removed == 0 {
                break;
            }
        }
        stats.jaggies_removed = jaggy_clean(img);
    }
    stats.orphans_removed = remove_orphans(img, orphan_max, protect_details);
    stats
}

fn opaque(img: &RgbaImage, x: usize, y: usize) -> bool {
    img.get_pixel(x as u32, y as u32)[3] > 0
}

fn rgb(img: &RgbaImage, x: usize, y: usize) -> [u8; 3] {
    let p = img.get_pixel(x as u32, y as u32);
    [p[0], p[1], p[2]]
}

fn clear(img: &mut RgbaImage, x: usize, y: usize) {
    img.put_pixel(x as u32, y as u32, image::Rgba([0, 0, 0, 0]));
}

const ORTHO: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const DIAG: [(isize, isize); 4] = [(-1, -1), (1, -1), (-1, 1), (1, 1)];

fn offset(x: usize, y: usize, d: (isize, isize), w: usize, h: usize) -> Option<(usize, usize)> {
    let nx = x as isize + d.0;
    let ny = y as isize + d.1;
    if nx >= 0 && (nx as usize) < w && ny >= 0 && (ny as usize) < h {
        Some((nx as usize, ny as usize))
    } else {
        None
    }
}

/// Fill transparent 4-connected components of at most `max_size` pixels that
/// do not touch the image border, using the most common surrounding opaque
/// color (ties -> lexicographically smallest RGB). Returns pixels filled.
fn fill_pinholes(img: &mut RgbaImage, max_size: usize) -> usize {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut visited = vec![false; w * h];
    let mut filled = 0;

    for sy in 0..h {
        for sx in 0..w {
            if visited[sy * w + sx] || opaque(img, sx, sy) {
                continue;
            }
            // Collect this transparent component (bounded scan: bail once too big).
            let mut component = vec![(sx, sy)];
            let mut touches_border = false;
            visited[sy * w + sx] = true;
            let mut head = 0;
            while head < component.len() {
                let (x, y) = component[head];
                head += 1;
                if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                    touches_border = true;
                }
                for d in ORTHO {
                    if let Some((nx, ny)) = offset(x, y, d, w, h) {
                        if !visited[ny * w + nx] && !opaque(img, nx, ny) {
                            visited[ny * w + nx] = true;
                            component.push((nx, ny));
                        }
                    }
                }
            }
            if touches_border || component.len() > max_size {
                continue;
            }

            // Majority surrounding opaque color, deterministic tie-break.
            let mut colors: Vec<([u8; 3], usize)> = Vec::new();
            for &(x, y) in &component {
                for d in ORTHO.iter().chain(DIAG.iter()) {
                    if let Some((nx, ny)) = offset(x, y, *d, w, h) {
                        if opaque(img, nx, ny) {
                            let c = rgb(img, nx, ny);
                            match colors.iter_mut().find(|(cc, _)| *cc == c) {
                                Some((_, n)) => *n += 1,
                                None => colors.push((c, 1)),
                            }
                        }
                    }
                }
            }
            let Some(&(fill, _)) = colors
                .iter()
                .max_by(|(ca, na), (cb, nb)| na.cmp(nb).then_with(|| cb.cmp(ca)))
            else {
                continue;
            };
            for &(x, y) in &component {
                img.put_pixel(
                    x as u32,
                    y as u32,
                    image::Rgba([fill[0], fill[1], fill[2], 255]),
                );
                filled += 1;
            }
        }
    }
    filled
}

/// Remove light, low-chroma matte fringe pixels: edge pixels (opaque with a
/// transparent 4-neighbor) that look like leftover white/gray matte AND sit
/// far (in Oklab) from every non-edge interior neighbor. Dark outlines are
/// deliberately never touched. Returns pixels cleared.
fn halo_clean(img: &mut RgbaImage, dist_threshold: f32) -> usize {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut is_edge = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if opaque(img, x, y) {
                is_edge[y * w + x] = ORTHO
                    .iter()
                    .any(|&d| offset(x, y, d, w, h).is_none_or(|(nx, ny)| !opaque(img, nx, ny)));
            }
        }
    }

    let threshold2 = dist_threshold * dist_threshold;
    let mut remove = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if !is_edge[y * w + x] {
                continue;
            }
            let [r, g, b] = rgb(img, x, y);
            let lab = srgb8_to_oklab(r, g, b);
            let chroma2 = lab[1] * lab[1] + lab[2] * lab[2];
            if lab[0] < 0.75 || chroma2 > 0.09f32 * 0.09 {
                continue; // Only light, near-neutral matte is a halo candidate.
            }
            // Compare against interior (non-edge) opaque neighbors only:
            // adjacent halo pixels look alike and must not vouch for each other.
            let mut min_d2 = f32::INFINITY;
            for d in ORTHO.iter().chain(DIAG.iter()) {
                if let Some((nx, ny)) = offset(x, y, *d, w, h) {
                    if opaque(img, nx, ny) && !is_edge[ny * w + nx] {
                        let [nr, ng, nb] = rgb(img, nx, ny);
                        min_d2 = min_d2.min(oklab_dist2(lab, srgb8_to_oklab(nr, ng, nb)));
                    }
                }
            }
            if min_d2.is_finite() && min_d2 >= threshold2 {
                remove.push((x, y));
            }
        }
    }
    for &(x, y) in &remove {
        clear(img, x, y);
    }
    remove.len()
}

/// Remove diagonal-dangling corner nubs: an opaque pixel with zero orthogonal
/// opaque neighbors and exactly one diagonal opaque neighbor, where that
/// neighbor is part of a solid body (>= 2 orthogonal opaque neighbors).
/// One-pixel diagonal lines hang off other diagonal-only pixels, so their
/// ends never match and intentional diagonals survive. Returns pixels cleared.
fn jaggy_clean(img: &mut RgbaImage) -> usize {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let ortho_count = |img: &RgbaImage, x: usize, y: usize| {
        ORTHO
            .iter()
            .filter(|&&d| offset(x, y, d, w, h).is_some_and(|(nx, ny)| opaque(img, nx, ny)))
            .count()
    };

    let mut remove = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if !opaque(img, x, y) || ortho_count(img, x, y) != 0 {
                continue;
            }
            let diag: Vec<(usize, usize)> = DIAG
                .iter()
                .filter_map(|&d| offset(x, y, d, w, h))
                .filter(|&(nx, ny)| opaque(img, nx, ny))
                .collect();
            if diag.len() == 1 && ortho_count(img, diag[0].0, diag[0].1) >= 2 {
                remove.push((x, y));
            }
        }
    }
    for &(x, y) in &remove {
        clear(img, x, y);
    }
    remove.len()
}

/// Remove tiny isolated opaque components (8-connected, size <= `max_size`).
/// With `protect_details`, a component survives if the same RGB appears on
/// another pixel within Chebyshev distance 4 (dotted patterns, starfields).
/// Returns pixels cleared.
fn remove_orphans(img: &mut RgbaImage, max_size: usize, protect_details: bool) -> usize {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut visited = vec![false; w * h];
    let mut remove = Vec::new();

    for sy in 0..h {
        for sx in 0..w {
            if visited[sy * w + sx] || !opaque(img, sx, sy) {
                continue;
            }
            let mut component = vec![(sx, sy)];
            visited[sy * w + sx] = true;
            let mut head = 0;
            while head < component.len() {
                let (x, y) = component[head];
                head += 1;
                for d in ORTHO.iter().chain(DIAG.iter()) {
                    if let Some((nx, ny)) = offset(x, y, *d, w, h) {
                        if !visited[ny * w + nx] && opaque(img, nx, ny) {
                            visited[ny * w + nx] = true;
                            component.push((nx, ny));
                        }
                    }
                }
            }
            if component.len() > max_size {
                continue;
            }
            if protect_details && repeats_nearby(img, &component) {
                continue;
            }
            remove.extend_from_slice(&component);
        }
    }
    for &(x, y) in &remove {
        clear(img, x, y);
    }
    remove.len()
}

/// True if any component pixel's exact RGB also appears on an opaque pixel
/// outside the component within Chebyshev distance 4.
fn repeats_nearby(img: &RgbaImage, component: &[(usize, usize)]) -> bool {
    let (w, h) = (img.width() as usize, img.height() as usize);
    for &(x, y) in component {
        let c = rgb(img, x, y);
        for ny in y.saturating_sub(4)..(y + 5).min(h) {
            for nx in x.saturating_sub(4)..(x + 5).min(w) {
                if component.contains(&(nx, ny)) {
                    continue;
                }
                if opaque(img, nx, ny) && rgb(img, nx, ny) == c {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    const RED: [u8; 4] = [200, 30, 30, 255];
    const CLEARP: [u8; 4] = [0, 0, 0, 0];

    fn blank(w: u32, h: u32) -> RgbaImage {
        RgbaImage::new(w, h)
    }

    fn put(img: &mut RgbaImage, x: u32, y: u32, c: [u8; 4]) {
        img.put_pixel(x, y, Rgba(c));
    }

    fn fill_rect(img: &mut RgbaImage, x0: u32, y0: u32, x1: u32, y1: u32, c: [u8; 4]) {
        for y in y0..=y1 {
            for x in x0..=x1 {
                put(img, x, y, c);
            }
        }
    }

    #[test]
    fn orphan_speck_is_removed_but_dotted_pattern_is_protected() {
        let mut img = blank(16, 8);
        fill_rect(&mut img, 0, 0, 4, 7, RED); // solid body
        put(&mut img, 10, 3, [90, 200, 90, 255]); // lone unique speck
        put(&mut img, 13, 2, [250, 250, 250, 255]); // star pair (protected)
        put(&mut img, 15, 4, [250, 250, 250, 255]);

        let stats = cleanup(&mut img, CleanupPreset::Conservative, true);

        assert_eq!(stats.orphans_removed, 1);
        assert_eq!(img.get_pixel(10, 3).0, CLEARP, "unique speck removed");
        assert_eq!(img.get_pixel(13, 2).0[3], 255, "repeating star kept");
        assert_eq!(img.get_pixel(15, 4).0[3], 255);
        assert_eq!(img.get_pixel(2, 3).0, RED, "body untouched");
    }

    #[test]
    fn orphan_speck_goes_when_protection_is_off() {
        let mut img = blank(10, 6);
        put(&mut img, 4, 2, [250, 250, 250, 255]);
        put(&mut img, 6, 3, [250, 250, 250, 255]);

        let stats = cleanup(&mut img, CleanupPreset::Conservative, false);
        assert_eq!(stats.orphans_removed, 2);
    }

    #[test]
    fn pinhole_is_filled_with_surrounding_color_but_border_notch_is_not() {
        let mut img = blank(8, 8);
        fill_rect(&mut img, 1, 1, 6, 6, RED);
        put(&mut img, 3, 3, CLEARP); // enclosed pinhole
        put(&mut img, 1, 4, CLEARP); // notch open to the border ring? no —
                                     // (0,4) is transparent border, so this
                                     // hole connects to the outside.

        let stats = cleanup(&mut img, CleanupPreset::Conservative, true);

        assert_eq!(stats.pinholes_filled, 1);
        assert_eq!(
            img.get_pixel(3, 3).0,
            RED,
            "pinhole filled with neighbor color"
        );
        assert_eq!(
            img.get_pixel(1, 4).0[3],
            0,
            "border-connected notch untouched"
        );
    }

    #[test]
    fn jaggy_corner_nub_is_removed_but_diagonal_line_survives() {
        let mut img = blank(12, 12);
        fill_rect(&mut img, 1, 1, 5, 5, RED);
        put(&mut img, 6, 6, RED); // nub dangling off the body corner
        for i in 0..4u32 {
            put(&mut img, 8 + i, 2 + i, [40, 40, 220, 255]); // 1px diagonal line
        }

        let stats = cleanup(&mut img, CleanupPreset::Balanced, true);

        assert_eq!(img.get_pixel(6, 6).0, CLEARP, "corner nub removed");
        assert!(stats.jaggies_removed >= 1);
        for i in 0..4u32 {
            assert_eq!(
                img.get_pixel(8 + i, 2 + i).0[3],
                255,
                "diagonal line pixel {i} must survive"
            );
        }
    }

    #[test]
    fn light_matte_halo_ring_is_removed_and_dark_outline_is_kept() {
        // Red 4x4 with a light-gray matte ring -> ring goes.
        let mut img = blank(10, 10);
        fill_rect(&mut img, 2, 2, 7, 7, [235, 235, 235, 255]); // matte ring area
        fill_rect(&mut img, 3, 3, 6, 6, RED); // subject overwrites the middle

        let stats = cleanup(&mut img, CleanupPreset::Balanced, true);

        assert!(
            stats.halo_removed >= 12,
            "ring should be cleared: {stats:?}"
        );
        assert_eq!(img.get_pixel(2, 2).0, CLEARP);
        assert_eq!(img.get_pixel(4, 2).0, CLEARP);
        assert_eq!(img.get_pixel(4, 4).0, RED, "subject intact");

        // Dark outline around a bright body -> untouched (halo pass never
        // targets dark pixels).
        let mut outlined = blank(10, 10);
        fill_rect(&mut outlined, 2, 2, 7, 7, [20, 20, 24, 255]);
        fill_rect(&mut outlined, 3, 3, 6, 6, [240, 220, 90, 255]);
        cleanup(&mut outlined, CleanupPreset::Balanced, true);
        assert_eq!(
            outlined.get_pixel(2, 2).0,
            [20, 20, 24, 255],
            "outline kept"
        );
    }

    #[test]
    fn none_preset_changes_nothing() {
        let mut img = blank(6, 6);
        put(&mut img, 3, 3, [90, 200, 90, 255]);
        let before = img.clone();
        let stats = cleanup(&mut img, CleanupPreset::None, true);
        assert_eq!(stats.total(), 0);
        assert_eq!(img.as_raw(), before.as_raw());
    }
}
