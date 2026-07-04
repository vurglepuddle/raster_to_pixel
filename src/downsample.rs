//! Cell-based downsampling of premultiplied-alpha linear RGBA to a target
//! grid. Fractional cell bounds are handled in f64 (PLAN.md bug risk #4).
//! Sobel edge-aware cell picking (task 4) plugs in here later: compute an
//! edge map on source luma, and for cells straddling a strong edge, restrict
//! the candidate set to pixels on the majority side before applying `mode`.

/// How a cell's pixels collapse to one output pixel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellMode {
    /// Alpha-weighted mean. Smooth; use for the `photo` preset only.
    Box,
    /// Per-channel median of pixels with alpha > 0. Robust to outliers.
    Median,
    /// Dominant for calm cells, median for high-contrast tiny detail.
    Detail,
    /// Most common coarsely-bucketed color (bucket = 32 levels/channel),
    /// represented by the mean of that bucket's members. Crispest; default.
    Dominant,
}

/// Input: linear RGBA rows, `src_w * src_h * 4` f32s (RGB NOT premultiplied;
/// alpha in [0,1]). Output: `dst_w * dst_h * 4` in the same layout.
pub fn downsample(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    mode: CellMode,
) -> Vec<f32> {
    assert_eq!(src.len(), src_w * src_h * 4);
    assert!(dst_w >= 1 && dst_h >= 1 && dst_w <= src_w && dst_h <= src_h);
    let mut out = Vec::with_capacity(dst_w * dst_h * 4);

    for cy in 0..dst_h {
        // f64 bounds so 1000px -> 64 cells has no drift/truncation.
        let y0 = cy as f64 * src_h as f64 / dst_h as f64;
        let y1 = (cy + 1) as f64 * src_h as f64 / dst_h as f64;
        for cx in 0..dst_w {
            let x0 = cx as f64 * src_w as f64 / dst_w as f64;
            let x1 = (cx + 1) as f64 * src_w as f64 / dst_w as f64;
            let cell = collect_cell(src, src_w, x0, x1, y0, y1);
            out.extend_from_slice(&reduce_cell(&cell, mode));
        }
    }
    out
}

fn collect_cell(
    src: &[f32],
    src_w: usize,
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
) -> Vec<WeightedPixel> {
    let ix0 = x0.floor() as usize;
    let ix1 = x1.ceil() as usize;
    let iy0 = y0.floor() as usize;
    let iy1 = y1.ceil() as usize;
    let mut v = Vec::with_capacity((ix1 - ix0) * (iy1 - iy0));
    for y in iy0..iy1 {
        let wy = overlap_1d(y0, y1, y as f64, y as f64 + 1.0);
        for x in ix0..ix1 {
            let wx = overlap_1d(x0, x1, x as f64, x as f64 + 1.0);
            let weight = (wx * wy) as f32;
            if weight <= 0.0 {
                continue;
            }
            let i = (y * src_w + x) * 4;
            v.push(WeightedPixel {
                rgba: [src[i], src[i + 1], src[i + 2], src[i + 3]],
                weight,
            });
        }
    }
    v
}

#[derive(Clone, Copy, Debug)]
struct WeightedPixel {
    rgba: [f32; 4],
    weight: f32,
}

fn overlap_1d(a0: f64, a1: f64, b0: f64, b1: f64) -> f64 {
    (a1.min(b1) - a0.max(b0)).max(0.0)
}

fn reduce_cell(cell: &[WeightedPixel], mode: CellMode) -> [f32; 4] {
    let area_sum: f32 = cell.iter().map(|p| p.weight).sum();
    let a_sum: f32 = cell.iter().map(|p| p.rgba[3] * p.weight).sum();
    let a_mean = a_sum / area_sum;
    if a_sum <= f32::EPSILON {
        return [0.0, 0.0, 0.0, 0.0]; // Fully transparent: RGB is meaningless.
    }
    match mode {
        CellMode::Box => reduce_box(cell, a_sum, a_mean),
        CellMode::Median => reduce_median(cell, a_mean),
        CellMode::Detail => {
            if luma_range(cell) >= 0.08 {
                reduce_median(cell, a_mean)
            } else {
                reduce_dominant(cell, a_mean)
            }
        }
        CellMode::Dominant => reduce_dominant(cell, a_mean),
    }
}

fn reduce_box(cell: &[WeightedPixel], a_sum: f32, a_mean: f32) -> [f32; 4] {
    // Alpha-weighted so transparent garbage never tints the cell
    // (PLAN.md bug risk #2).
    let mut rgb = [0f32; 3];
    for p in cell {
        for (ch, value) in rgb.iter_mut().enumerate() {
            *value += p.rgba[ch] * p.rgba[3] * p.weight;
        }
    }
    [rgb[0] / a_sum, rgb[1] / a_sum, rgb[2] / a_sum, a_mean]
}

fn reduce_median(cell: &[WeightedPixel], a_mean: f32) -> [f32; 4] {
    let candidates: Vec<&WeightedPixel> = cell.iter().filter(|p| p.rgba[3] > 0.0).collect();
    if candidates.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let mut target = [0f32; 3];
    for (ch, value) in target.iter_mut().enumerate() {
        let mut vals: Vec<(f32, f32)> = cell
            .iter()
            .filter(|p| p.rgba[3] > 0.0)
            .map(|p| (p.rgba[ch], p.weight * p.rgba[3]))
            .collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        *value = weighted_median(&vals);
    }

    // Per-channel medians are robust, but can synthesize colors that were never
    // present in the cell (for example red + green + white -> yellow). Snap the
    // median target to the nearest real source color so median/detail modes do
    // not create neon artifacts along fuzzy edges.
    let mut best = candidates[0];
    let mut best_dist = f32::INFINITY;
    let mut best_weight = -1.0f32;
    for p in candidates {
        let dist = p
            .rgba
            .iter()
            .take(3)
            .zip(target)
            .map(|(a, b)| {
                let d = *a - b;
                d * d
            })
            .sum::<f32>();
        let weight = p.weight * p.rgba[3];
        if dist < best_dist || ((dist - best_dist).abs() <= f32::EPSILON && weight > best_weight) {
            best = p;
            best_dist = dist;
            best_weight = weight;
        }
    }

    [best.rgba[0], best.rgba[1], best.rgba[2], a_mean]
}

fn reduce_dominant(cell: &[WeightedPixel], a_mean: f32) -> [f32; 4] {
    // Bucket to 32 levels/channel, find the most populated bucket
    // (ties broken by lowest bucket key -> deterministic), then
    // average that bucket's members for a clean representative.
    let mut counts: std::collections::HashMap<u16, (f32, [f32; 3])> =
        std::collections::HashMap::new();
    for p in cell.iter().filter(|p| p.rgba[3] > 0.0) {
        let q = |v: f32| ((v.clamp(0.0, 1.0) * 31.0) as u16).min(31);
        let key = q(p.rgba[0]) << 10 | q(p.rgba[1]) << 5 | q(p.rgba[2]);
        let e = counts.entry(key).or_insert((0.0, [0.0; 3]));
        let w = p.weight * p.rgba[3];
        e.0 += w;
        for (ch, value) in p.rgba.iter().take(3).enumerate() {
            e.1[ch] += *value * w;
        }
    }
    let (_, &(n, sum)) = counts
        .iter()
        .max_by(|(&a_key, &(a_n, _)), (&b_key, &(b_n, _))| {
            a_n.partial_cmp(&b_n)
                .unwrap()
                .then_with(|| b_key.cmp(&a_key))
        })
        .unwrap();
    [sum[0] / n, sum[1] / n, sum[2] / n, a_mean]
}

fn luma_range(cell: &[WeightedPixel]) -> f32 {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for p in cell.iter().filter(|p| p.rgba[3] > 0.0) {
        let y = 0.2126 * p.rgba[0] + 0.7152 * p.rgba[1] + 0.0722 * p.rgba[2];
        lo = lo.min(y);
        hi = hi.max(y);
    }
    if lo.is_finite() {
        hi - lo
    } else {
        0.0
    }
}

fn weighted_median(sorted_values: &[(f32, f32)]) -> f32 {
    let total: f32 = sorted_values.iter().map(|(_, weight)| weight).sum();
    let midpoint = total * 0.5;
    let mut acc = 0.0;
    for &(value, weight) in sorted_values {
        acc += weight;
        if acc >= midpoint {
            return value;
        }
    }
    sorted_values.last().map_or(0.0, |&(value, _)| value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: usize, h: usize, px: [f32; 4]) -> Vec<f32> {
        (0..w * h).flat_map(|_| px).collect()
    }

    #[test]
    fn non_divisible_grid_covers_every_output_pixel() {
        // 10x10 -> 3x3: no panics, correct length, all cells filled.
        let src = flat(10, 10, [0.5, 0.25, 0.75, 1.0]);
        for mode in [
            CellMode::Box,
            CellMode::Median,
            CellMode::Detail,
            CellMode::Dominant,
        ] {
            let out = downsample(&src, 10, 10, 3, 3, mode);
            assert_eq!(out.len(), 3 * 3 * 4);
            for c in out.chunks(4) {
                assert!(
                    (c[0] - 0.5).abs() < 1e-5 && (c[3] - 1.0).abs() < 1e-6,
                    "{mode:?}"
                );
            }
        }
    }

    #[test]
    fn fractional_cells_weight_boundary_pixels_once() {
        let src = vec![
            0.0, 0.0, 0.0, 1.0, //
            1.0, 1.0, 1.0, 1.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let out = downsample(&src, 3, 1, 2, 1, CellMode::Box);
        assert!((out[0] - (1.0 / 3.0)).abs() < 1e-6, "got {out:?}");
        assert!((out[4] - (1.0 / 3.0)).abs() < 1e-6, "got {out:?}");
    }

    #[test]
    fn transparent_pixels_do_not_tint_box_average() {
        // Cell: one opaque red pixel + three transparent green pixels.
        // Box result must be pure red.
        let mut src = flat(2, 2, [0.0, 1.0, 0.0, 0.0]);
        src[0] = 1.0;
        src[1] = 0.0;
        src[2] = 0.0;
        src[3] = 1.0;
        let out = downsample(&src, 2, 2, 1, 1, CellMode::Box);
        assert!((out[0] - 1.0).abs() < 1e-6 && out[1].abs() < 1e-6);
        assert!((out[3] - 0.25).abs() < 1e-6); // mean alpha preserved
    }

    #[test]
    fn dominant_picks_majority_color_not_blend() {
        // 3 dark + 1 bright pixel: box would blend, dominant must return dark.
        let mut src = flat(2, 2, [0.1, 0.1, 0.1, 1.0]);
        src[0] = 1.0;
        src[1] = 1.0;
        src[2] = 1.0;
        let out = downsample(&src, 2, 2, 1, 1, CellMode::Dominant);
        assert!((out[0] - 0.1).abs() < 1e-5, "got {out:?}");
    }

    #[test]
    fn median_picks_real_color_not_per_channel_artifact() {
        // Old per-channel median produced yellow [1, 1, 0], which was not present.
        let src = vec![
            1.0, 0.0, 0.0, 1.0, //
            0.0, 1.0, 0.0, 1.0, //
            1.0, 1.0, 1.0, 1.0,
        ];
        let out = downsample(&src, 3, 1, 1, 1, CellMode::Median);
        let color = [out[0], out[1], out[2]];
        assert_ne!(color, [1.0, 1.0, 0.0], "got invented yellow: {out:?}");
        assert!(
            color == [1.0, 0.0, 0.0] || color == [0.0, 1.0, 0.0] || color == [1.0, 1.0, 1.0],
            "median should snap to a real cell color, got {out:?}"
        );
    }

    #[test]
    fn fully_transparent_cell_is_zeroed() {
        let src = flat(4, 4, [0.9, 0.9, 0.9, 0.0]);
        let out = downsample(&src, 4, 4, 2, 2, CellMode::Median);
        assert!(out.iter().all(|&v| v == 0.0));
    }
}
