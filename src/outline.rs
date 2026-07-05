//! Opt-in outline cleanup for the quantized pixel-art grid (runs after
//! morphology cleanup, before scaling). Detects the sprite's outline color
//! from its silhouette edge and repairs or enforces it. Only ever repaints
//! with a color already present in the output, so the palette is unchanged.
//! Like `pipeline`, this module uses the approved `image` crate.
//!
//! Detection: silhouette edge pixels are opaque pixels with an in-bounds
//! transparent 4-neighbor (fully opaque images therefore have no edge and
//! the pass is a no-op). Among edge pixels, the most frequent dark color
//! (Oklab L <= `OUTLINE_MAX_L`) is the outline color, if it covers at least
//! `MIN_EDGE_SHARE` of the silhouette.

use image::RgbaImage;

use crate::color::srgb8_to_oklab;

/// What outline cleanup should do. `None` disables (pipeline default).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutlineMode {
    None,
    /// Recolor only edge pixels bridging two outline-colored edge pixels
    /// (fills gaps in an existing outline; conservative).
    Repair,
    /// Recolor every silhouette edge pixel to the outline color.
    Enforce,
}

/// What an outline pass did, for diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OutlineStats {
    pub recolored: usize,
    /// The detected outline color, if detection succeeded.
    pub outline_color: Option<[u8; 3]>,
}

/// Outline candidates must be at most this dark (Oklab L).
const OUTLINE_MAX_L: f32 = 0.45;
/// The winning dark color must cover at least this share of edge pixels.
const MIN_EDGE_SHARE: f32 = 0.35;

pub fn apply_outline(img: &mut RgbaImage, mode: OutlineMode) -> OutlineStats {
    if mode == OutlineMode::None {
        return OutlineStats::default();
    }
    let (w, h) = (img.width() as usize, img.height() as usize);
    let opaque = |img: &RgbaImage, x: usize, y: usize| img.get_pixel(x as u32, y as u32)[3] > 0;

    // Silhouette edge = opaque with an in-bounds transparent 4-neighbor.
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if !opaque(img, x, y) {
                continue;
            }
            let mut is_edge = false;
            for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if nx >= 0
                    && ny >= 0
                    && (nx as usize) < w
                    && (ny as usize) < h
                    && !opaque(img, nx as usize, ny as usize)
                {
                    is_edge = true;
                    break;
                }
            }
            if is_edge {
                edges.push((x, y));
            }
        }
    }
    if edges.is_empty() {
        return OutlineStats::default();
    }

    // Most frequent dark edge color; ties -> lexicographically smallest RGB.
    let mut counts: Vec<([u8; 3], usize)> = Vec::new();
    for &(x, y) in &edges {
        let p = img.get_pixel(x as u32, y as u32);
        let c = [p[0], p[1], p[2]];
        if srgb8_to_oklab(c[0], c[1], c[2])[0] > OUTLINE_MAX_L {
            continue;
        }
        match counts.iter_mut().find(|(cc, _)| *cc == c) {
            Some((_, n)) => *n += 1,
            None => counts.push((c, 1)),
        }
    }
    let Some(&(outline, n)) = counts
        .iter()
        .max_by(|(ca, na), (cb, nb)| na.cmp(nb).then_with(|| cb.cmp(ca)))
    else {
        return OutlineStats::default();
    };
    if (n as f32) < MIN_EDGE_SHARE * edges.len() as f32 {
        return OutlineStats::default();
    }

    // Decide on a snapshot of which edge pixels already carry the outline
    // color, so repairs never chain off each other within one pass.
    let is_outline_edge: std::collections::HashSet<(usize, usize)> = edges
        .iter()
        .filter(|&&(x, y)| {
            let p = img.get_pixel(x as u32, y as u32);
            [p[0], p[1], p[2]] == outline
        })
        .copied()
        .collect();

    let mut recolored = 0usize;
    for &(x, y) in &edges {
        if is_outline_edge.contains(&(x, y)) {
            continue;
        }
        let repaint = match mode {
            OutlineMode::None => false,
            OutlineMode::Enforce => true,
            OutlineMode::Repair => {
                // Bridge rule: >= 2 outline-colored edge pixels in the
                // 8-neighborhood means this pixel is a gap in the outline.
                let mut support = 0;
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let (nx, ny) = (x as isize + dx, y as isize + dy);
                        if nx >= 0
                            && ny >= 0
                            && is_outline_edge.contains(&(nx as usize, ny as usize))
                        {
                            support += 1;
                        }
                    }
                }
                support >= 2
            }
        };
        if repaint {
            let p = img.get_pixel_mut(x as u32, y as u32);
            p[0] = outline[0];
            p[1] = outline[1];
            p[2] = outline[2];
            recolored += 1;
        }
    }

    OutlineStats {
        recolored,
        outline_color: Some(outline),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    const BODY: [u8; 4] = [200, 30, 30, 255];
    const INK: [u8; 4] = [20, 20, 24, 255];

    /// 10x10: dark outline ring at 1..=6, red body filling 2..=5.
    fn outlined_sprite() -> RgbaImage {
        let mut img = RgbaImage::new(10, 10);
        for y in 1..=6u32 {
            for x in 1..=6u32 {
                let ring = x == 1 || x == 6 || y == 1 || y == 6;
                img.put_pixel(x, y, Rgba(if ring { INK } else { BODY }));
            }
        }
        img
    }

    #[test]
    fn repair_fills_a_single_gap_between_outline_runs() {
        let mut img = outlined_sprite();
        img.put_pixel(3, 1, Rgba(BODY)); // body color leaking into the outline

        let stats = apply_outline(&mut img, OutlineMode::Repair);

        assert_eq!(stats.outline_color, Some([20, 20, 24]));
        assert_eq!(stats.recolored, 1);
        assert_eq!(img.get_pixel(3, 1).0, INK, "gap repainted");
        assert_eq!(img.get_pixel(3, 2).0, BODY, "interior untouched");
    }

    #[test]
    fn repair_leaves_wide_breaks_alone_but_enforce_closes_them() {
        let mut broken = outlined_sprite();
        broken.put_pixel(3, 1, Rgba(BODY));
        broken.put_pixel(4, 1, Rgba(BODY)); // two-wide break: no bridge

        let mut repaired = broken.clone();
        let r = apply_outline(&mut repaired, OutlineMode::Repair);
        assert_eq!(
            r.recolored, 0,
            "two-wide break has <2 outline neighbors per pixel"
        );
        assert_eq!(repaired.get_pixel(3, 1).0, BODY);

        let mut enforced = broken;
        let e = apply_outline(&mut enforced, OutlineMode::Enforce);
        assert_eq!(e.recolored, 2);
        assert_eq!(enforced.get_pixel(3, 1).0, INK);
        assert_eq!(enforced.get_pixel(4, 1).0, INK);
    }

    #[test]
    fn bright_sprite_without_dark_edge_is_a_no_op() {
        let mut img = RgbaImage::new(8, 8);
        for y in 2..=5u32 {
            for x in 2..=5u32 {
                img.put_pixel(x, y, Rgba([240, 220, 90, 255]));
            }
        }
        let stats = apply_outline(&mut img, OutlineMode::Enforce);
        assert_eq!(stats.recolored, 0);
        assert_eq!(stats.outline_color, None);
        assert_eq!(img.get_pixel(2, 2).0, [240, 220, 90, 255]);
    }

    #[test]
    fn minority_dark_speck_does_not_hijack_the_outline() {
        // Red sprite whose edge has one dark pixel: below MIN_EDGE_SHARE,
        // so nothing is recolored.
        let mut img = RgbaImage::new(8, 8);
        for y in 2..=5u32 {
            for x in 2..=5u32 {
                img.put_pixel(x, y, Rgba(BODY));
            }
        }
        img.put_pixel(2, 2, Rgba(INK));
        let stats = apply_outline(&mut img, OutlineMode::Enforce);
        assert_eq!(stats.recolored, 0);
        assert_eq!(stats.outline_color, None);
    }

    #[test]
    fn fully_opaque_image_has_no_silhouette_and_is_untouched() {
        let mut img = RgbaImage::new(8, 8);
        for p in img.pixels_mut() {
            *p = Rgba(INK);
        }
        let before = img.clone();
        let stats = apply_outline(&mut img, OutlineMode::Enforce);
        assert_eq!(stats.recolored, 0);
        assert_eq!(img.as_raw(), before.as_raw());
    }
}
