//! Alpha and background cleanup applied to the *source* image before grid
//! detection and downsampling. Like `pipeline`, this module uses the approved
//! `image` crate; the algorithms themselves are pure std.
//!
//! Modes (see `AlphaMode`):
//! - `Preserve`: leave the source alpha untouched (pipeline default).
//! - `Binary`: hard 0/255 alpha at the configured threshold.
//! - `BackgroundFill`: flood-fill from the image border with a color
//!   tolerance and make the reached background transparent. Enclosed islands
//!   of the same color are NOT reachable from the border and survive.
//! - `ColorKey`: every pixel within tolerance of a key color goes transparent.
//!
//! Every pixel this module makes (or finds) fully transparent is
//! decontaminated to `[0, 0, 0, 0]` so stray RGB can never tint later math.

use std::collections::VecDeque;

use image::RgbaImage;

/// How source alpha is prepared before conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaMode {
    Preserve,
    Binary,
    BackgroundFill,
    ColorKey,
}

/// What an alpha pre-pass did, for diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AlphaStats {
    /// Pixels newly made transparent by this pass.
    pub removed: usize,
    /// Transparent pixels whose stray RGB was zeroed (includes `removed`).
    pub decontaminated: usize,
}

/// Squared sRGB distance threshold for a normalized tolerance in 0..=1,
/// where 1.0 spans the full RGB cube diagonal.
fn tolerance_dist2(tolerance: f32) -> u32 {
    let t = tolerance.clamp(0.0, 1.0) as f64 * 255.0 * 3f64.sqrt();
    (t * t) as u32
}

fn rgb_dist2(a: [u8; 3], b: [u8; 3]) -> u32 {
    let dr = a[0] as i32 - b[0] as i32;
    let dg = a[1] as i32 - b[1] as i32;
    let db = a[2] as i32 - b[2] as i32;
    (dr * dr + dg * dg + db * db) as u32
}

/// Apply an alpha mode in place. `Preserve` only decontaminates already-
/// transparent pixels; the other modes may zero more alpha first.
pub fn apply_alpha_mode(
    img: &mut RgbaImage,
    mode: AlphaMode,
    alpha_threshold: u8,
    tolerance: f32,
    color_key: Option<[u8; 3]>,
) -> AlphaStats {
    let removed = match mode {
        AlphaMode::Preserve => 0,
        AlphaMode::Binary => binarize_alpha(img, alpha_threshold),
        AlphaMode::BackgroundFill => background_fill(img, tolerance),
        AlphaMode::ColorKey => color_key.map_or(0, |key| apply_color_key(img, key, tolerance)),
    };
    let decontaminated = decontaminate_transparent(img);
    AlphaStats {
        removed,
        decontaminated,
    }
}

/// Force alpha to 0 or 255 at `threshold`. Returns pixels newly transparent.
fn binarize_alpha(img: &mut RgbaImage, threshold: u8) -> usize {
    let mut removed = 0;
    for p in img.pixels_mut() {
        if p[3] < threshold.max(1) {
            if p[3] != 0 {
                removed += 1;
            }
            p[3] = 0;
        } else {
            p[3] = 255;
        }
    }
    removed
}

/// Make every pixel within tolerance of `key` transparent. Returns count.
fn apply_color_key(img: &mut RgbaImage, key: [u8; 3], tolerance: f32) -> usize {
    let tol2 = tolerance_dist2(tolerance);
    let mut removed = 0;
    for p in img.pixels_mut() {
        if p[3] > 0 && rgb_dist2([p[0], p[1], p[2]], key) <= tol2 {
            p[3] = 0;
            removed += 1;
        }
    }
    removed
}

/// Flood-fill background removal from the image border.
///
/// Every opaque border pixel seeds a fill (scan order, deterministic) using
/// its own RGB as the reference color; neighbors join while within tolerance
/// of that reference. Already-transparent pixels are passable connectors.
/// Interior regions enclosed by non-matching colors are never reached.
fn background_fill(img: &mut RgbaImage, tolerance: f32) -> usize {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return 0;
    }
    let tol2 = tolerance_dist2(tolerance);
    let mut visited = vec![false; w * h];
    let mut remove = vec![false; w * h];
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

    let border: Vec<(usize, usize)> = (0..w)
        .map(|x| (x, 0))
        .chain((0..w).map(|x| (x, h - 1)))
        .chain((0..h).map(|y| (0, y)))
        .chain((0..h).map(|y| (w - 1, y)))
        .collect();

    for &(sx, sy) in &border {
        let si = sy * w + sx;
        if visited[si] {
            continue;
        }
        let seed = img.get_pixel(sx as u32, sy as u32);
        if seed[3] == 0 {
            visited[si] = true; // Already background; nothing to seed from.
            continue;
        }
        let reference = [seed[0], seed[1], seed[2]];
        visited[si] = true;
        remove[si] = true;
        queue.push_back((sx, sy));

        while let Some((x, y)) = queue.pop_front() {
            let neighbors = [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ];
            for (nx, ny) in neighbors {
                if nx >= w || ny >= h {
                    continue;
                }
                let ni = ny * w + nx;
                if visited[ni] {
                    continue;
                }
                let p = img.get_pixel(nx as u32, ny as u32);
                if p[3] == 0 {
                    visited[ni] = true; // Passable transparent connector.
                    queue.push_back((nx, ny));
                } else if rgb_dist2([p[0], p[1], p[2]], reference) <= tol2 {
                    visited[ni] = true;
                    remove[ni] = true;
                    queue.push_back((nx, ny));
                }
            }
        }
    }

    let mut removed = 0;
    for (i, p) in img.pixels_mut().enumerate() {
        if remove[i] && p[3] > 0 {
            p[3] = 0;
            removed += 1;
        }
    }
    removed
}

/// Zero the RGB of every fully transparent pixel. Returns pixels changed.
pub fn decontaminate_transparent(img: &mut RgbaImage) -> usize {
    let mut changed = 0;
    for p in img.pixels_mut() {
        if p[3] == 0 && (p[0] != 0 || p[1] != 0 || p[2] != 0) {
            p[0] = 0;
            p[1] = 0;
            p[2] = 0;
            changed += 1;
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn solid(w: u32, h: u32, rgba: [u8; 4]) -> RgbaImage {
        let mut img = RgbaImage::new(w, h);
        for p in img.pixels_mut() {
            *p = Rgba(rgba);
        }
        img
    }

    #[test]
    fn background_fill_removes_border_background_but_keeps_enclosed_island() {
        // White background, a red hollow square ring, and a white interior
        // enclosed by the ring. The interior must survive the flood fill.
        let mut img = solid(7, 7, [255, 255, 255, 255]);
        for i in 1..=5u32 {
            for &(x, y) in &[(i, 1), (i, 5), (1, i), (5, i)] {
                img.put_pixel(x, y, Rgba([200, 30, 30, 255]));
            }
        }

        let stats = apply_alpha_mode(&mut img, AlphaMode::BackgroundFill, 128, 0.08, None);

        assert!(stats.removed > 0);
        assert_eq!(img.get_pixel(0, 0).0, [0, 0, 0, 0], "border bg removed");
        assert_eq!(img.get_pixel(6, 6).0, [0, 0, 0, 0]);
        assert_eq!(
            img.get_pixel(3, 3).0,
            [255, 255, 255, 255],
            "enclosed island must survive"
        );
        assert_eq!(img.get_pixel(1, 1).0, [200, 30, 30, 255], "ring survives");
    }

    #[test]
    fn background_fill_tolerance_controls_matte_halo_removal() {
        // White bg, one-pixel gray matte halo column, red subject column.
        // The matte is interior-only: border matte pixels would seed their own
        // background fill by design.
        let mut img = solid(6, 5, [255, 255, 255, 255]);
        for y in 1..=3u32 {
            img.put_pixel(2, y, Rgba([210, 210, 210, 255])); // matte
            img.put_pixel(3, y, Rgba([200, 30, 30, 255])); // subject
        }

        let mut tight = img.clone();
        apply_alpha_mode(&mut tight, AlphaMode::BackgroundFill, 128, 0.08, None);
        assert_eq!(tight.get_pixel(0, 2).0[3], 0, "white removed");
        assert_eq!(
            tight.get_pixel(2, 2).0[3],
            255,
            "matte outside tolerance stays"
        );
        assert_eq!(tight.get_pixel(3, 2).0[3], 255, "subject stays");

        let mut loose = img;
        apply_alpha_mode(&mut loose, AlphaMode::BackgroundFill, 128, 0.20, None);
        assert_eq!(loose.get_pixel(2, 2).0[3], 0, "matte inside tolerance goes");
        assert_eq!(loose.get_pixel(3, 2).0[3], 255, "subject still stays");
    }

    #[test]
    fn removed_background_rgb_is_decontaminated_to_zero() {
        let mut img = solid(4, 4, [250, 250, 250, 255]);
        img.put_pixel(1, 1, Rgba([10, 200, 10, 255]));
        img.put_pixel(2, 2, Rgba([9, 9, 9, 0])); // pre-existing transparent garbage

        let stats = apply_alpha_mode(&mut img, AlphaMode::BackgroundFill, 128, 0.08, None);

        for (x, y, p) in img.enumerate_pixels() {
            if p[3] == 0 {
                assert_eq!(&p.0[..3], &[0, 0, 0], "tinted transparent at {x},{y}");
            }
        }
        assert!(stats.decontaminated >= stats.removed);
        assert_eq!(img.get_pixel(1, 1).0, [10, 200, 10, 255]);
    }

    #[test]
    fn color_key_removes_only_near_key_pixels() {
        let mut img = solid(3, 1, [0, 255, 0, 255]);
        img.put_pixel(1, 0, Rgba([5, 250, 8, 255])); // near-key
        img.put_pixel(2, 0, Rgba([200, 30, 30, 255])); // subject

        apply_alpha_mode(&mut img, AlphaMode::ColorKey, 128, 0.05, Some([0, 255, 0]));

        assert_eq!(img.get_pixel(0, 0).0, [0, 0, 0, 0]);
        assert_eq!(img.get_pixel(1, 0).0, [0, 0, 0, 0]);
        assert_eq!(img.get_pixel(2, 0).0, [200, 30, 30, 255]);
    }

    #[test]
    fn binary_alpha_hardens_soft_edges_without_touching_opaque_subject() {
        let mut img = RgbaImage::new(3, 1);
        img.put_pixel(0, 0, Rgba([100, 100, 100, 40])); // soft fringe -> gone
        img.put_pixel(1, 0, Rgba([90, 90, 90, 200])); // mostly solid -> opaque
        img.put_pixel(2, 0, Rgba([80, 80, 80, 255]));

        let stats = apply_alpha_mode(&mut img, AlphaMode::Binary, 128, 0.0, None);

        assert_eq!(img.get_pixel(0, 0).0, [0, 0, 0, 0]);
        assert_eq!(img.get_pixel(1, 0).0, [90, 90, 90, 255]);
        assert_eq!(img.get_pixel(2, 0).0, [80, 80, 80, 255]);
        assert_eq!(stats.removed, 1);
    }
}
