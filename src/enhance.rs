//! Opt-in contrast expansion: a source-resolution pre-pass that protects
//! tiny high-contrast details (eye pixels, glints, thin dark outlines) from
//! being averaged away by downsampling. A pixel qualifies as a feature when
//! it is much darker or much lighter (Oklab L) than its opaque neighborhood;
//! its color is then stamped over contrasting neighbors within `radius`, so
//! the enlarged feature can win its cell's median/dominant vote.
//!
//! Runs AFTER the alpha pre-pass and AFTER grid detection (the stamps would
//! otherwise add off-grid edge energy), and only when `radius > 0`. Like
//! `pipeline`, this module uses the approved `image` crate.

use image::RgbaImage;

use crate::color::srgb8_to_oklab;

/// Features must be at least this far below/above their neighborhood mean L.
const MIN_CONTRAST: f32 = 0.18;
/// Dark features must be at most this L; light features at least `LIGHT_MIN_L`.
const DARK_MAX_L: f32 = 0.35;
const LIGHT_MIN_L: f32 = 0.80;
/// A feature needs this many opaque 8-neighbors, so silhouette slivers and
/// lone noise specks never get amplified.
const MIN_OPAQUE_NEIGHBORS: usize = 3;
/// Single-pixel colorful outliers are often generation noise. Neutral
/// singletons may still be eyes/glints; colorful details need nearby support.
const SINGLETON_MAX_CHROMA: f32 = 0.025;
const COLOR_SUPPORT_RADIUS: isize = 4;

/// Expand high-contrast single-pixel features in place. Returns the number
/// of pixels whose color was overwritten. Deterministic: detection runs on a
/// snapshot; stamping happens in scan order (later features win overlaps).
pub fn expand_contrast(img: &mut RgbaImage, radius: u32, alpha_threshold: u8) -> usize {
    if radius == 0 {
        return 0;
    }
    let r = radius.min(4) as isize;
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return 0;
    }

    // Snapshot: opacity, Oklab L, and source colors, all pre-expansion.
    let threshold = alpha_threshold.max(1);
    let mut opaque = vec![false; w * h];
    let mut luma = vec![0f32; w * h];
    let mut chroma = vec![0f32; w * h];
    let mut rgb = vec![[0u8; 3]; w * h];
    for (i, p) in img.pixels().enumerate() {
        opaque[i] = p[3] >= threshold;
        rgb[i] = [p[0], p[1], p[2]];
        if opaque[i] {
            let lab = srgb8_to_oklab(p[0], p[1], p[2]);
            luma[i] = lab[0];
            chroma[i] = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
        }
    }

    // Detect features against the 8-neighbor ring (independent of radius).
    #[derive(Clone, Copy)]
    struct Feature {
        x: usize,
        y: usize,
        l: f32,
        dark: bool,
    }
    let mut features = Vec::new();
    let mut is_feature = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !opaque[i] {
                continue;
            }
            let mut sum = 0f32;
            let mut count = 0usize;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                        continue;
                    }
                    let ni = ny as usize * w + nx as usize;
                    if opaque[ni] {
                        sum += luma[ni];
                        count += 1;
                    }
                }
            }
            if count < MIN_OPAQUE_NEIGHBORS {
                continue;
            }
            let mean = sum / count as f32;
            let l = luma[i];
            let dark = l <= DARK_MAX_L && mean - l >= MIN_CONTRAST;
            let light = l >= LIGHT_MIN_L && l - mean >= MIN_CONTRAST;
            if dark || light {
                if !has_feature_color_support(x, y, w, h, &opaque, &rgb, chroma[i]) {
                    continue;
                }
                features.push(Feature { x, y, l, dark });
                is_feature[i] = true;
            }
        }
    }

    // Stamp each feature over contrasting opaque neighbors within `radius`.
    // Targets must contrast with the feature in the right direction, so dark
    // features only eat lighter background and vice versa; other features
    // are never overwritten.
    let mut changed = 0usize;
    for f in &features {
        let src = rgb[f.y * w + f.x];
        for dy in -r..=r {
            for dx in -r..=r {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = (f.x as isize + dx, f.y as isize + dy);
                if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                if !opaque[ni] || is_feature[ni] {
                    continue;
                }
                let contrast_ok = if f.dark {
                    luma[ni] - f.l >= MIN_CONTRAST * 0.5
                } else {
                    f.l - luma[ni] >= MIN_CONTRAST * 0.5
                };
                if !contrast_ok {
                    continue;
                }
                let p = img.get_pixel_mut(nx as u32, ny as u32);
                if [p[0], p[1], p[2]] != src {
                    p[0] = src[0];
                    p[1] = src[1];
                    p[2] = src[2];
                    changed += 1;
                }
            }
        }
    }
    changed
}

fn has_feature_color_support(
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    opaque: &[bool],
    rgb: &[[u8; 3]],
    feature_chroma: f32,
) -> bool {
    if feature_chroma <= SINGLETON_MAX_CHROMA {
        return true;
    }

    let i = y * w + x;
    let color = rgb[i];
    for dy in -COLOR_SUPPORT_RADIUS..=COLOR_SUPPORT_RADIUS {
        for dx in -COLOR_SUPPORT_RADIUS..=COLOR_SUPPORT_RADIUS {
            if dx == 0 && dy == 0 {
                continue;
            }
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                continue;
            }
            let ni = ny as usize * w + nx as usize;
            if opaque[ni] && rgb[ni] == color {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn field(w: u32, h: u32, rgba: [u8; 4]) -> RgbaImage {
        let mut img = RgbaImage::new(w, h);
        for p in img.pixels_mut() {
            *p = Rgba(rgba);
        }
        img
    }

    #[test]
    fn dark_dot_on_light_field_expands_to_radius() {
        let mut img = field(9, 9, [230, 230, 230, 255]);
        img.put_pixel(4, 4, Rgba([25, 25, 25, 255]));

        let changed = expand_contrast(&mut img, 1, 128);

        assert_eq!(changed, 8, "3x3 stamp minus the feature itself");
        for y in 3..=5u32 {
            for x in 3..=5u32 {
                assert_eq!(img.get_pixel(x, y).0, [25, 25, 25, 255], "at {x},{y}");
            }
        }
        assert_eq!(
            img.get_pixel(2, 4).0,
            [230, 230, 230, 255],
            "outside radius"
        );
    }

    #[test]
    fn light_glint_on_dark_field_expands_too() {
        let mut img = field(7, 7, [30, 30, 40, 255]);
        img.put_pixel(3, 3, Rgba([250, 250, 250, 255]));

        let changed = expand_contrast(&mut img, 1, 128);

        assert_eq!(changed, 8);
        assert_eq!(img.get_pixel(2, 3).0, [250, 250, 250, 255]);
    }

    #[test]
    fn low_contrast_and_midtone_pixels_are_not_features() {
        // Mid-gray on light gray: below both feature thresholds.
        let mut img = field(7, 7, [200, 200, 200, 255]);
        img.put_pixel(3, 3, Rgba([150, 150, 150, 255]));
        assert_eq!(expand_contrast(&mut img, 1, 128), 0);
        assert_eq!(img.get_pixel(2, 3).0, [200, 200, 200, 255]);
    }

    #[test]
    fn radius_zero_is_a_no_op_and_transparent_pixels_are_untouched() {
        let mut img = field(7, 7, [230, 230, 230, 255]);
        img.put_pixel(3, 3, Rgba([25, 25, 25, 255]));
        img.put_pixel(2, 3, Rgba([0, 0, 0, 0]));

        assert_eq!(expand_contrast(&mut img, 0, 128), 0);

        let changed = expand_contrast(&mut img, 1, 128);
        assert!(changed > 0);
        assert_eq!(
            img.get_pixel(2, 3).0,
            [0, 0, 0, 0],
            "transparent never painted"
        );
    }

    #[test]
    fn silhouette_sliver_with_few_opaque_neighbors_is_not_amplified() {
        // A lone dark pixel poking into transparency has < 3 opaque
        // neighbors and must not become a feature.
        let mut img = RgbaImage::new(7, 7);
        img.put_pixel(3, 3, Rgba([25, 25, 25, 255]));
        img.put_pixel(3, 4, Rgba([230, 230, 230, 255]));
        assert_eq!(expand_contrast(&mut img, 2, 128), 0);
    }

    #[test]
    fn adjacent_features_do_not_overwrite_each_other() {
        let mut img = field(9, 9, [230, 230, 230, 255]);
        img.put_pixel(3, 4, Rgba([25, 25, 25, 255]));
        img.put_pixel(5, 4, Rgba([20, 20, 60, 255]));

        expand_contrast(&mut img, 1, 128);

        assert_eq!(img.get_pixel(3, 4).0, [25, 25, 25, 255], "feature A intact");
        assert_eq!(img.get_pixel(5, 4).0, [20, 20, 60, 255], "feature B intact");
    }

    #[test]
    fn colorful_singleton_outlier_is_not_amplified_at_radius_two() {
        let mut img = field(9, 9, [230, 230, 230, 255]);
        img.put_pixel(4, 4, Rgba([210, 20, 150, 255]));

        assert_eq!(expand_contrast(&mut img, 2, 128), 0);
        assert_eq!(img.get_pixel(3, 4).0, [230, 230, 230, 255]);
        assert_eq!(img.get_pixel(4, 4).0, [210, 20, 150, 255]);
    }
}
