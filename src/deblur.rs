//! Cell-level deblurring for `CellMode::Smart`.
//!
//! A blurred or resampled source mixes each art pixel with its neighbours, so
//! even a perfectly placed cell sample is `observed = (1 - 2b) * own + b *
//! left + b * right` per axis (to first order). A one-pixel dark outline on a
//! light fill then comes out several shades too light, and no palette can put
//! it back.
//!
//! The leak `b` is measured on the grid itself. At an isolated step (two flat
//! runs meeting: `L L | R R`), the two cells at the boundary should equal `L`
//! and `R`; under blur both sit a fraction `b` of the way toward the other
//! side, under sharpening both overshoot (`b < 0`). Real art also has genuine
//! one-cell shading bands that look like a leak locally, but blur moves
//! *every* step, so the estimate is the bound that nearly all steps agree on:
//! the 15th percentile if even that is positive, the 85th if even that is
//! negative, else exactly 0 (a sharp source is never touched). Both axes are
//! pooled. The mix is then inverted exactly with a tridiagonal solve per row
//! and per column, in the gamma-encoded space where resizing and blurring
//! happened.
//!
//! Deterministic: f32/f64 add, multiply, divide only.

/// Steps smaller than this (encoded units, 0..1, summed over RGB) are not used
/// to measure the leak.
const MIN_STEP: f32 = 0.25;
/// The flat runs on either side may vary by at most this share of the step.
const FLAT_SHARE: f32 = 0.08;
/// Leaks below this are measurement noise; leave the cells alone.
const MIN_LEAK: f32 = 0.015;
/// Inverting a larger leak amplifies noise too much (gain `1 / (1 - 4b)`).
const MAX_LEAK: f32 = 0.16;
/// Largest overshoot (sharpening) that is undone.
const MAX_OVERSHOOT: f32 = 0.1;
/// Fewer isolated steps than this cannot support an estimate.
const MIN_STEPS: usize = 24;
/// The two boundary cells of a blurred step leak by the same amount; pairs
/// that disagree by more than this are shading, not blur.
const MAX_ASYMMETRY: f32 = 0.06;

/// Measured neighbour leak per axis (0 = sharp).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Leak {
    pub x: f32,
    pub y: f32,
}

/// Undo neighbour mixing in a `width x height` grid of linear RGBA cells
/// (alpha untouched; cells below `alpha_threshold` neither give nor receive).
/// Returns the leak that was measured and inverted.
pub fn deblur_cells(cells: &mut [f32], width: usize, height: usize, alpha_threshold: f32) -> Leak {
    if width < 6 && height < 6 {
        return Leak::default();
    }
    // Work on gamma-encoded values.
    let mut enc: Vec<[f32; 3]> = cells
        .chunks_exact(4)
        .map(|p| [encode(p[0]), encode(p[1]), encode(p[2])])
        .collect();
    let opaque: Vec<bool> = cells
        .chunks_exact(4)
        .map(|p| p[3] >= alpha_threshold)
        .collect();

    let mut leaks: Vec<f32> = Vec::new();
    collect_leaks(&enc, &opaque, width, height, true, &mut leaks);
    collect_leaks(&enc, &opaque, width, height, false, &mut leaks);
    let b = consensus(&mut leaks);
    let leak = Leak { x: b, y: b };
    if b == 0.0 {
        return leak;
    }
    let mut line: Vec<[f32; 3]> = Vec::new();
    let mut work: Vec<f32> = Vec::new();
    for y in 0..height {
        solve_runs(&mut enc, &opaque, y * width, 1, width, b, &mut line, &mut work);
    }
    for x in 0..width {
        solve_runs(&mut enc, &opaque, x, width, height, b, &mut line, &mut work);
    }
    for (i, px) in cells.chunks_exact_mut(4).enumerate() {
        if opaque[i] {
            for ch in 0..3 {
                px[ch] = decode(enc[i][ch]);
            }
        }
    }
    leak
}

fn encode(linear: f32) -> f32 {
    // Continuous sRGB encode via the 8-bit table's neighbours would lose the
    // sub-level precision the cell means carry, so use the transfer curve.
    let v = linear.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * crate::fmath::powf(v, 1.0 / 2.4) - 0.055
    }
}

fn decode(encoded: f32) -> f32 {
    let v = encoded.clamp(0.0, 1.0);
    if v <= 0.040_45 {
        v / 12.92
    } else {
        crate::fmath::powf((v + 0.055) / 1.055, 2.4)
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn l1(a: [f32; 3]) -> f32 {
    a[0].abs() + a[1].abs() + a[2].abs()
}

/// The leak nearly every isolated step agrees on (see the module docs).
fn consensus(leaks: &mut [f32]) -> f32 {
    if leaks.len() < MIN_STEPS {
        return 0.0;
    }
    leaks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let low = leaks[leaks.len() * 15 / 100];
    let high = leaks[leaks.len() * 85 / 100];
    if low >= MIN_LEAK {
        low.min(MAX_LEAK)
    } else if high <= -MIN_LEAK {
        high.max(-MAX_OVERSHOOT)
    } else {
        0.0
    }
}

/// Leak fraction of every isolated step along one axis.
fn collect_leaks(
    enc: &[[f32; 3]],
    opaque: &[bool],
    width: usize,
    height: usize,
    along_x: bool,
    leaks: &mut Vec<f32>,
) {
    let (lines, len, stride, line_stride) = if along_x {
        (height, width, 1, width)
    } else {
        (width, height, width, 1)
    };
    if len < 6 {
        return;
    }
    for line in 0..lines {
        let at = |k: usize| line * line_stride + k * stride;
        // Cells k-2, k-1 | k, k+1 | k+2, k+3: flat, step, flat.
        for k in 2..len - 3 {
            let idx = [at(k - 2), at(k - 1), at(k), at(k + 1), at(k + 2), at(k + 3)];
            if idx.iter().any(|&i| !opaque[i]) {
                continue;
            }
            let (left, right) = (enc[idx[1]], enc[idx[4]]);
            let step = sub(right, left);
            let size = l1(step);
            if size < MIN_STEP {
                continue;
            }
            let flat = FLAT_SHARE * size;
            if l1(sub(enc[idx[0]], left)) > flat || l1(sub(enc[idx[5]], right)) > flat {
                continue;
            }
            // Position of the two boundary cells along the step, 0 = left.
            let norm2 = step[0] * step[0] + step[1] * step[1] + step[2] * step[2];
            let pos = |c: [f32; 3]| {
                let d = sub(c, left);
                (d[0] * step[0] + d[1] * step[1] + d[2] * step[2]) / norm2
            };
            let (a, b) = (pos(enc[idx[2]]), 1.0 - pos(enc[idx[3]]));
            // Each on its own side, and both leaking alike.
            if !(-0.3..0.5).contains(&a)
                || !(-0.3..0.5).contains(&b)
                || (a - b).abs() > MAX_ASYMMETRY
            {
                continue;
            }
            leaks.push(0.5 * (a + b));
        }
    }
}

/// Invert `o[k] = (1 - 2b) t[k] + b t[k-1] + b t[k+1]` over each maximal run of
/// opaque cells on one line (run ends reflect: the missing neighbour is the
/// cell itself). Thomas algorithm; the system is strictly diagonally dominant
/// for `-0.25 < b < 0.25` (negative `b` undoes sharpening overshoot).
#[allow(clippy::too_many_arguments)]
fn solve_runs(
    enc: &mut [[f32; 3]],
    opaque: &[bool],
    start: usize,
    stride: usize,
    len: usize,
    leak: f32,
    line: &mut Vec<[f32; 3]>,
    work: &mut Vec<f32>,
) {
    let mut k = 0;
    while k < len {
        if !opaque[start + k * stride] {
            k += 1;
            continue;
        }
        let run_start = k;
        while k < len && opaque[start + k * stride] {
            k += 1;
        }
        let n = k - run_start;
        if n < 2 {
            continue;
        }
        line.clear();
        line.extend((0..n).map(|j| enc[start + (run_start + j) * stride]));
        // Diagonal is 1 - 2b inside, 1 - b at the two ends; off-diagonals b.
        work.clear();
        work.resize(n, 0.0);
        let diag = |j: usize| {
            if j == 0 || j == n - 1 {
                1.0 - leak
            } else {
                1.0 - 2.0 * leak
            }
        };
        // Forward sweep.
        let mut denom = diag(0);
        work[0] = leak / denom;
        for ch in 0..3 {
            line[0][ch] /= denom;
        }
        for j in 1..n {
            denom = diag(j) - leak * work[j - 1];
            work[j] = leak / denom;
            for ch in 0..3 {
                line[j][ch] = (line[j][ch] - leak * line[j - 1][ch]) / denom;
            }
        }
        // Back substitution.
        for j in (0..n - 1).rev() {
            for ch in 0..3 {
                line[j][ch] -= work[j] * line[j + 1][ch];
            }
        }
        for (j, value) in line.iter().enumerate() {
            enc[start + (run_start + j) * stride] =
                [value[0].clamp(0.0, 1.0), value[1].clamp(0.0, 1.0), value[2].clamp(0.0, 1.0)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::linear_to_srgb;

    fn cell_srgb8(cell: &[f32]) -> [u8; 3] {
        [
            linear_to_srgb(cell[0]),
            linear_to_srgb(cell[1]),
            linear_to_srgb(cell[2]),
        ]
    }

    fn art(width: usize, height: usize) -> Vec<[u8; 3]> {
        // Blocks of flat colour (isolated steps between them) crossed by two
        // one-cell dark lines.
        let fills = [[230u8, 200, 90], [70, 120, 200], [200, 80, 70], [90, 170, 110]];
        (0..width * height)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                if x == 9 || y == 15 {
                    [25, 20, 30]
                } else {
                    fills[(x / 6 + 2 * (y / 6)) % 4]
                }
            })
            .collect()
    }

    fn to_cells(colors: &[[f32; 3]]) -> Vec<f32> {
        colors
            .iter()
            .flat_map(|c| [decode(c[0]), decode(c[1]), decode(c[2]), 1.0])
            .collect()
    }

    fn blur(src: &[[f32; 3]], width: usize, height: usize, b: f32) -> Vec<[f32; 3]> {
        let pass = |src: &[[f32; 3]], dx: usize, dy: usize| -> Vec<[f32; 3]> {
            (0..width * height)
                .map(|i| {
                    let (x, y) = (i % width, i / width);
                    let get = |ox: isize, oy: isize| {
                        let nx = (x as isize + ox).clamp(0, width as isize - 1) as usize;
                        let ny = (y as isize + oy).clamp(0, height as isize - 1) as usize;
                        src[ny * width + nx]
                    };
                    let (m, l, r) = (
                        get(0, 0),
                        get(-(dx as isize), -(dy as isize)),
                        get(dx as isize, dy as isize),
                    );
                    std::array::from_fn(|ch| (1.0 - 2.0 * b) * m[ch] + b * (l[ch] + r[ch]))
                })
                .collect()
        };
        pass(&pass(src, 1, 0), 0, 1)
    }

    #[test]
    fn sharp_grids_are_left_bit_identical() {
        let (w, h) = (30, 24);
        let truth: Vec<[f32; 3]> = art(w, h)
            .iter()
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect();
        let mut cells = to_cells(&truth);
        let before = cells.clone();
        assert_eq!(deblur_cells(&mut cells, w, h, 0.5), Leak::default());
        assert_eq!(cells, before);
    }

    #[test]
    fn measured_leak_matches_and_thin_lines_regain_their_colour() {
        let (w, h) = (36, 30);
        let colors = art(w, h);
        let truth: Vec<[f32; 3]> = colors
            .iter()
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect();
        let mut cells = to_cells(&blur(&truth, w, h, 0.1));
        // Blurred: the dark lines are visibly washed out.
        let line_cell = 3 * w + 9;
        assert!(cell_srgb8(&cells[line_cell * 4..])[2] > 50);
        let leak = deblur_cells(&mut cells, w, h, 0.5);
        assert!((leak.x - 0.1).abs() < 0.01 && (leak.y - 0.1).abs() < 0.01, "{leak:?}");
        for (i, c) in colors.iter().enumerate() {
            let got = cell_srgb8(&cells[i * 4..]);
            for ch in 0..3 {
                assert!((got[ch] as i32 - c[ch] as i32).abs() <= 1, "cell {i}: {got:?} vs {c:?}");
            }
        }
    }

    #[test]
    fn transparent_cells_do_not_take_part() {
        let (w, h) = (36, 30);
        let truth: Vec<[f32; 3]> = art(w, h)
            .iter()
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect();
        let mut cells = to_cells(&blur(&truth, w, h, 0.08));
        for i in 0..w * h {
            if i % w < 3 {
                cells[i * 4 + 3] = 0.0;
            }
        }
        let before = cells.clone();
        let leak = deblur_cells(&mut cells, w, h, 0.5);
        assert!(leak.x > 0.0);
        for i in 0..w * h {
            if i % w < 3 {
                assert_eq!(cells[i * 4..i * 4 + 4], before[i * 4..i * 4 + 4]);
            }
        }
    }
}
