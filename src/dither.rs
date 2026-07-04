//! Dithering: ordered Bayer (pixel-art default) and edge-clamped serpentine
//! Floyd–Steinberg (photo preset). Operates on Oklab pixel buffers + a
//! palette; returns palette indices so the encoder can stay indexed.

use crate::kmeans::nearest;

/// Bayer 4x4 threshold matrix, values 0..16.
pub const BAYER4: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

/// Bayer 8x8 threshold matrix, values 0..64 (derived from BAYER4).
pub fn bayer8() -> [[u8; 8]; 8] {
    let mut m = [[0u8; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            m[y][x] = 4 * BAYER4[y % 4][x % 4] + bayer8_offset(x, y);
        }
    }
    m
}
// The recursive construction: M8[y][x] = 4*M4[y%4][x%4] + M2-pattern of the quadrant.
fn bayer8_offset(x: usize, y: usize) -> u8 {
    [[0u8, 2], [3, 1]][y / 4][x / 4]
}

/// Ordered dither in Oklab: perturb L by a Bayer threshold before nearest-
/// palette lookup. `strength` 0.0 = plain mapping, 1.0 = full dither.
/// `amplitude` is the max L perturbation at strength 1 (0.03–0.06 works well).
pub fn ordered_dither(
    pixels: &[[f32; 3]],
    width: usize,
    palette: &[[f32; 3]],
    strength: f32,
    amplitude: f32,
    use_8x8: bool,
) -> Vec<u16> {
    assert!(width > 0 && pixels.len().is_multiple_of(width));
    let m8 = bayer8();
    let mut out = Vec::with_capacity(pixels.len());
    for (i, p) in pixels.iter().enumerate() {
        let (x, y) = (i % width, i / width);
        // Normalize threshold to bin centers in [-0.5, 0.5).
        let t = if use_8x8 {
            (m8[y % 8][x % 8] as f32 + 0.5) / 64.0 - 0.5
        } else {
            (BAYER4[y % 4][x % 4] as f32 + 0.5) / 16.0 - 0.5
        };
        let q = [p[0] + t * amplitude * strength, p[1], p[2]];
        out.push(nearest(palette, q) as u16);
    }
    out
}

/// Serpentine Floyd–Steinberg in Oklab. `edge_mask[i] == true` marks pixels on
/// strong edges: error is NOT propagated out of them, so dither noise cannot
/// smear across silhouettes (PLAN.md bug risk #3). `clamp` bounds each error
/// component (0.1 is a sane default; f32::INFINITY disables).
pub fn floyd_steinberg(
    pixels: &[[f32; 3]],
    width: usize,
    palette: &[[f32; 3]],
    edge_mask: &[bool],
    clamp: f32,
) -> Vec<u16> {
    assert!(width > 0 && pixels.len().is_multiple_of(width));
    assert_eq!(edge_mask.len(), pixels.len());
    let height = pixels.len() / width;
    let mut buf: Vec<[f32; 3]> = pixels.to_vec();
    let mut out = vec![0u16; pixels.len()];

    for y in 0..height {
        let ltr = y % 2 == 0; // serpentine
        for step in 0..width {
            let x = if ltr { step } else { width - 1 - step };
            let i = y * width + x;
            let idx = nearest(palette, buf[i]);
            out[i] = idx as u16;
            if edge_mask[i] {
                continue; // Absorb error at edges.
            }
            let c = palette[idx];
            let err = [
                (buf[i][0] - c[0]).clamp(-clamp, clamp),
                (buf[i][1] - c[1]).clamp(-clamp, clamp),
                (buf[i][2] - c[2]).clamp(-clamp, clamp),
            ];
            let dx: isize = if ltr { 1 } else { -1 };
            let mut spread = |xx: isize, yy: usize, w: f32| {
                if xx >= 0 && (xx as usize) < width && yy < height {
                    let j = yy * width + xx as usize;
                    if !edge_mask[j] {
                        for ch in 0..3 {
                            buf[j][ch] += err[ch] * w / 16.0;
                        }
                    }
                }
            };
            spread(x as isize + dx, y, 7.0);
            spread(x as isize - dx, y + 1, 3.0);
            spread(x as isize, y + 1, 5.0);
            spread(x as isize + dx, y + 1, 1.0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayer8_is_a_permutation_of_0_to_63() {
        let m = bayer8();
        let mut seen = [false; 64];
        for row in m {
            for v in row {
                assert!(!seen[v as usize], "dup {v}");
                seen[v as usize] = true;
            }
        }
    }

    #[test]
    fn ordered_dither_mixes_two_grays_at_midpoint() {
        // A flat 0.5-luma field against a {0.0, 1.0} palette should come out
        // as a roughly 50/50 checkerboard-ish mix, not solid.
        let pal = vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let px = vec![[0.5f32, 0.0, 0.0]; 64];
        let out = ordered_dither(&px, 8, &pal, 1.0, 2.0, false);
        let ones = out.iter().filter(|&&i| i == 1).count();
        assert!(ones == 32, "expected 32 of each, got {ones} ones");
    }

    #[test]
    fn zero_strength_equals_plain_nearest_mapping() {
        let pal = vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let px = vec![[0.4f32, 0.0, 0.0]; 16];
        let out = ordered_dither(&px, 4, &pal, 0.0, 2.0, true);
        assert!(out.iter().all(|&i| i == 0));
    }

    #[test]
    fn fs_error_does_not_cross_edge_mask() {
        // Left half bright-ish, right half exactly 0; a full-column edge mask
        // between them must keep the right half entirely palette-0.
        let pal = vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let (w, h) = (8usize, 4usize);
        let mut px = vec![[0.0f32, 0.0, 0.0]; w * h];
        let mut mask = vec![false; w * h];
        for y in 0..h {
            for x in 0..4 {
                px[y * w + x] = [0.7, 0.0, 0.0];
            }
            mask[y * w + 4] = true; // wall
        }
        let out = floyd_steinberg(&px, w, &pal, &mask, f32::INFINITY);
        for y in 0..h {
            for x in 5..w {
                assert_eq!(out[y * w + x], 0, "leak at ({x},{y})");
            }
        }
    }
}
