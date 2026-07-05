//! Cell-based downsampling of premultiplied-alpha linear RGBA to a target
//! grid. Fractional cell bounds are handled in f64 (PLAN.md bug risk #4).
//! Sobel edge-aware cell picking (task 4) plugs in here later: compute an
//! edge map on source luma, and for cells straddling a strong edge, restrict
//! the candidate set to pixels on the majority side before applying `mode`.

use crate::color::{linear_to_oklab, oklab_dist2};

/// How a cell's pixels collapse to one output pixel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellMode {
    /// Alpha-weighted mean. Smooth; use for the `photo` preset only.
    Box,
    /// Per-channel median of pixels with alpha > 0. Robust to outliers.
    Median,
    /// Dominant for calm cells, median for high-contrast tiny detail.
    Detail,
    /// Most common coarsely-bucketed color (two shifted 32-level bucket
    /// grids; the stronger winner is used), represented by the real cell
    /// color nearest the winning bucket's weighted mean. Crispest; default.
    Dominant,
    /// Detail's calm/busy split, but busy cells take their color from a
    /// content-adaptive kernel fit (Kopf et al.-style EM-C, run in Oklab)
    /// instead of the cell median. Opt-in, for fuzzy generated pixel art
    /// where median/dominant lose texture and contours. Falls back to
    /// `Detail` when the job exceeds `adaptive_fits_budget`.
    Adaptive,
}

pub const DEFAULT_DOMINANT_THRESHOLD: f32 = 0.25;
pub const DEFAULT_ADAPTIVE_ITERATIONS: u32 = 3;
pub const MAX_ADAPTIVE_ITERATIONS: u32 = 8;

/// Luma spread at which Detail/Adaptive treat a cell as "busy".
const DETAIL_LUMA_SPLIT: f32 = 0.08;

/// EM-C search window half-size, in units of the larger cell dimension. The
/// C-step clamps kernel eigenvalues to (cell/3)^2, so weights past
/// sqrt(2 * 11.5) * cell/3 ~ 1.6 cells are below `ADAPTIVE_EXPONENT_CUTOFF`
/// anyway: this window is lossless, not an approximation.
const ADAPTIVE_RADIUS_CELLS: f64 = 1.6;
/// Cells wider/taller than this make windows explode; fall back to Detail.
const ADAPTIVE_MAX_CELL: f64 = 32.0;
/// Total window-pixel visits allowed per job (~1 megapixel source at the
/// iteration cap). Above this the mode silently degrades to Detail so GUI
/// previews stay responsive.
const ADAPTIVE_VISIT_BUDGET: f64 = 256e6;
/// Skip Gaussian weights below exp(-11.5) ~ 1e-5 (reference uses the same
/// floor on the weight itself).
const ADAPTIVE_EXPONENT_CUTOFF: f32 = 11.5;
/// Smallest kernel eigenvalue: a 0.25 px standard deviation, so kernels can
/// sharpen to near-point samplers but never degenerate.
const ADAPTIVE_LAMBDA_MIN: f32 = 0.0625;

/// Everything `downsample_grid_opts` needs beyond the geometry.
#[derive(Clone, Copy, Debug)]
pub struct DownsampleOptions {
    pub mode: CellMode,
    /// Minimum winning-bucket coverage for dominant/detail cells.
    pub dominant_threshold: f32,
    /// EM refinement passes for `CellMode::Adaptive` (clamped to
    /// `1..=MAX_ADAPTIVE_ITERATIONS`). Ignored by the other modes.
    pub adaptive_iterations: u32,
}

impl Default for DownsampleOptions {
    fn default() -> Self {
        Self {
            mode: CellMode::Detail,
            dominant_threshold: DEFAULT_DOMINANT_THRESHOLD,
            adaptive_iterations: DEFAULT_ADAPTIVE_ITERATIONS,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingGrid {
    pub origin_x: f64,
    pub origin_y: f64,
    pub cell_w: f64,
    pub cell_h: f64,
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
    downsample_with_dominant_threshold(
        src,
        src_w,
        src_h,
        dst_w,
        dst_h,
        mode,
        DEFAULT_DOMINANT_THRESHOLD,
    )
}

pub fn downsample_with_dominant_threshold(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    mode: CellMode,
    dominant_threshold: f32,
) -> Vec<f32> {
    let grid = SamplingGrid {
        origin_x: 0.0,
        origin_y: 0.0,
        cell_w: src_w as f64 / dst_w as f64,
        cell_h: src_h as f64 / dst_h as f64,
    };
    downsample_grid_with_dominant_threshold(
        src,
        src_w,
        src_h,
        dst_w,
        dst_h,
        grid,
        mode,
        dominant_threshold,
    )
}

pub fn downsample_grid(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    grid: SamplingGrid,
    mode: CellMode,
) -> Vec<f32> {
    downsample_grid_with_dominant_threshold(
        src,
        src_w,
        src_h,
        dst_w,
        dst_h,
        grid,
        mode,
        DEFAULT_DOMINANT_THRESHOLD,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn downsample_grid_with_dominant_threshold(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    grid: SamplingGrid,
    mode: CellMode,
    dominant_threshold: f32,
) -> Vec<f32> {
    downsample_grid_opts(
        src,
        src_w,
        src_h,
        dst_w,
        dst_h,
        grid,
        DownsampleOptions {
            mode,
            dominant_threshold,
            ..DownsampleOptions::default()
        },
    )
}

pub fn downsample_grid_opts(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    grid: SamplingGrid,
    opts: DownsampleOptions,
) -> Vec<f32> {
    assert_eq!(src.len(), src_w * src_h * 4);
    assert!(dst_w >= 1 && dst_h >= 1 && dst_w <= src_w && dst_h <= src_h);
    assert!(grid.cell_w > 0.0 && grid.cell_h > 0.0);

    let iterations = opts.adaptive_iterations.clamp(1, MAX_ADAPTIVE_ITERATIONS);
    let mode = if opts.mode == CellMode::Adaptive
        && !adaptive_fits_budget(dst_w, dst_h, grid.cell_w, grid.cell_h, iterations)
    {
        CellMode::Detail
    } else {
        opts.mode
    };
    let kernels = if mode == CellMode::Adaptive {
        Some(fit_adaptive_kernels(
            src, src_w, src_h, dst_w, dst_h, grid, iterations,
        ))
    } else {
        None
    };

    let mut out = Vec::with_capacity(dst_w * dst_h * 4);
    for cy in 0..dst_h {
        // f64 bounds so 1000px -> 64 cells has no drift/truncation.
        let y0 = grid.origin_y + cy as f64 * grid.cell_h;
        let y1 = grid.origin_y + (cy + 1) as f64 * grid.cell_h;
        for cx in 0..dst_w {
            let x0 = grid.origin_x + cx as f64 * grid.cell_w;
            let x1 = grid.origin_x + (cx + 1) as f64 * grid.cell_w;
            let cell = collect_cell(src, src_w, src_h, x0, x1, y0, y1);
            let px = match &kernels {
                Some(kernels) => {
                    reduce_cell_adaptive(&cell, &kernels[cy * dst_w + cx], opts.dominant_threshold)
                }
                None => reduce_cell(&cell, mode, opts.dominant_threshold),
            };
            out.extend_from_slice(&px);
        }
    }
    out
}

/// Whether `CellMode::Adaptive` would actually run for this job, or fall back
/// to `Detail`. Exposed so the pipeline can report the fallback.
pub fn adaptive_fits_budget(
    dst_w: usize,
    dst_h: usize,
    cell_w: f64,
    cell_h: f64,
    iterations: u32,
) -> bool {
    if !cell_w.is_finite() || !cell_h.is_finite() {
        return false;
    }
    if cell_w > ADAPTIVE_MAX_CELL || cell_h > ADAPTIVE_MAX_CELL {
        return false;
    }
    let r = ADAPTIVE_RADIUS_CELLS * cell_w.max(cell_h);
    let win_w = 2.0 * r + 1.0;
    let win_h = 2.0 * r + 1.0;
    // Two Gaussian sweeps per kernel per iteration (E-step + M-step).
    let visits = dst_w as f64 * dst_h as f64 * win_w * win_h * iterations as f64 * 2.0;
    visits <= ADAPTIVE_VISIT_BUDGET
}

fn collect_cell(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
) -> Vec<WeightedPixel> {
    let ix0 = x0.floor().max(0.0) as usize;
    let ix1 = (x1.ceil().min(src_w as f64).max(0.0)) as usize;
    let iy0 = y0.floor().max(0.0) as usize;
    let iy1 = (y1.ceil().min(src_h as f64).max(0.0)) as usize;
    if ix0 >= ix1 || iy0 >= iy1 {
        return Vec::new();
    }
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

fn reduce_cell(cell: &[WeightedPixel], mode: CellMode, dominant_threshold: f32) -> [f32; 4] {
    let area_sum: f32 = cell.iter().map(|p| p.weight).sum();
    let a_sum: f32 = cell.iter().map(|p| p.rgba[3] * p.weight).sum();
    if area_sum <= f32::EPSILON {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let a_mean = a_sum / area_sum;
    if a_sum <= f32::EPSILON {
        return [0.0, 0.0, 0.0, 0.0]; // Fully transparent: RGB is meaningless.
    }
    match mode {
        CellMode::Box => reduce_box(cell, a_sum, a_mean),
        CellMode::Median => reduce_median(cell, a_mean),
        // Adaptive cells are reduced by `reduce_cell_adaptive`; a bare
        // Adaptive here means no kernel is available, so act like Detail.
        CellMode::Detail | CellMode::Adaptive => {
            if luma_range(cell) >= DETAIL_LUMA_SPLIT {
                reduce_median(cell, a_mean)
            } else {
                reduce_dominant(cell, a_sum, a_mean, dominant_threshold)
            }
        }
        CellMode::Dominant => reduce_dominant(cell, a_sum, a_mean, dominant_threshold),
    }
}

/// Reduce one cell in Adaptive mode: calm cells behave exactly like Detail's
/// calm branch (dominant vote), busy cells snap the EM kernel's Oklab mean to
/// the nearest real opaque member so the output never contains synthetic
/// colors (mirrors the median/dominant snaps). Alpha rules are identical to
/// every other mode.
fn reduce_cell_adaptive(
    cell: &[WeightedPixel],
    kernel: &AdaptiveKernel,
    dominant_threshold: f32,
) -> [f32; 4] {
    let area_sum: f32 = cell.iter().map(|p| p.weight).sum();
    let a_sum: f32 = cell.iter().map(|p| p.rgba[3] * p.weight).sum();
    if area_sum <= f32::EPSILON || a_sum <= f32::EPSILON {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let a_mean = a_sum / area_sum;
    if luma_range(cell) < DETAIL_LUMA_SPLIT {
        return reduce_dominant(cell, a_sum, a_mean, dominant_threshold);
    }
    if kernel.weight <= 0.0 {
        // The kernel never saw an opaque pixel (degenerate coverage); Detail's
        // busy branch is the safe answer.
        return reduce_median(cell, a_mean);
    }

    let mut best: Option<[f32; 4]> = None;
    let mut best_dist = f32::INFINITY;
    let mut best_weight = -1.0f32;
    for p in cell.iter().filter(|p| p.rgba[3] > 0.0) {
        let lab = linear_to_oklab(p.rgba[0], p.rgba[1], p.rgba[2]);
        let dist = oklab_dist2(lab, kernel.color);
        let weight = p.weight * p.rgba[3];
        if dist < best_dist || ((dist - best_dist).abs() <= f32::EPSILON && weight > best_weight) {
            best = Some(p.rgba);
            best_dist = dist;
            best_weight = weight;
        }
    }
    match best {
        Some(rgba) => [rgba[0], rgba[1], rgba[2], a_mean],
        None => reduce_median(cell, a_mean),
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

/// Quantize one channel to a 5-bit bucket index. The `shifted` grid places
/// its bucket boundaries half a bucket away from the plain grid, so a color
/// family straddling a plain-grid boundary lands in one shifted bucket.
fn bucket_index(v: f32, shifted: bool) -> u16 {
    let scaled = v.clamp(0.0, 1.0) * 31.0 + if shifted { 0.5 } else { 0.0 };
    (scaled as u16).min(31)
}

fn bucket_key(rgba: [f32; 4], shifted: bool) -> u16 {
    bucket_index(rgba[0], shifted) << 10
        | bucket_index(rgba[1], shifted) << 5
        | bucket_index(rgba[2], shifted)
}

#[derive(Clone, Copy)]
struct BucketWin {
    key: u16,
    shifted: bool,
    weight: f32,
}

/// Strongest bucket of one grid: max alpha-weight, ties broken by lowest
/// bucket key -> deterministic.
fn strongest_bucket(cell: &[WeightedPixel], shifted: bool) -> Option<BucketWin> {
    let mut counts: std::collections::HashMap<u16, f32> = std::collections::HashMap::new();
    for p in cell.iter().filter(|p| p.rgba[3] > 0.0) {
        *counts.entry(bucket_key(p.rgba, shifted)).or_insert(0.0) += p.weight * p.rgba[3];
    }
    counts
        .iter()
        .max_by(|(&a_key, &a_n), (&b_key, &b_n)| {
            a_n.partial_cmp(&b_n)
                .unwrap()
                .then_with(|| b_key.cmp(&a_key))
        })
        .map(|(&key, &weight)| BucketWin {
            key,
            shifted,
            weight,
        })
}

fn reduce_dominant(
    cell: &[WeightedPixel],
    a_sum: f32,
    a_mean: f32,
    dominant_threshold: f32,
) -> [f32; 4] {
    // Two shifted 5-bit bucket grids: a color family that a bucket boundary
    // would split in one grid stays whole in the other. Use whichever grid
    // produces the strongest dominant cluster (ties prefer the plain grid).
    let plain = strongest_bucket(cell, false);
    let shifted = strongest_bucket(cell, true);
    let win = match (plain, shifted) {
        (Some(a), Some(b)) => {
            if b.weight > a.weight {
                b
            } else {
                a
            }
        }
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => return [0.0, 0.0, 0.0, 0.0],
    };
    if win.weight / a_sum < dominant_threshold.clamp(0.0, 1.0) {
        return reduce_box(cell, a_sum, a_mean);
    }

    // Representative: the winning bucket's weighted mean, snapped to the
    // nearest real member so dominant cells never output synthetic colors
    // (mirrors the median snap). Ties -> heavier member -> scan order.
    let mut mean = [0f32; 3];
    for p in cell.iter().filter(|p| p.rgba[3] > 0.0) {
        if bucket_key(p.rgba, win.shifted) == win.key {
            let w = p.weight * p.rgba[3];
            for (ch, value) in p.rgba.iter().take(3).enumerate() {
                mean[ch] += *value * w;
            }
        }
    }
    for value in &mut mean {
        *value /= win.weight;
    }
    let mut best: Option<[f32; 4]> = None;
    let mut best_dist = f32::INFINITY;
    let mut best_weight = -1.0f32;
    for p in cell.iter().filter(|p| p.rgba[3] > 0.0) {
        if bucket_key(p.rgba, win.shifted) != win.key {
            continue;
        }
        let dist = p
            .rgba
            .iter()
            .take(3)
            .zip(mean)
            .map(|(a, b)| {
                let d = *a - b;
                d * d
            })
            .sum::<f32>();
        let weight = p.weight * p.rgba[3];
        if dist < best_dist || ((dist - best_dist).abs() <= f32::EPSILON && weight > best_weight) {
            best = Some(p.rgba);
            best_dist = dist;
            best_weight = weight;
        }
    }
    let rep = best.unwrap_or([0.0, 0.0, 0.0, 0.0]);
    [rep[0], rep[1], rep[2], a_mean]
}

/// One anisotropic Gaussian kernel per output cell (Kopf et al. EM-C,
/// adapted). `mu` is in source coordinates, `sigma` the symmetric 2x2
/// covariance stored as [xx, xy, yy], `color` in Oklab. `weight` is the total
/// soft assignment the kernel won in the last M-step; 0 means it never saw an
/// opaque pixel and its color must not be trusted.
struct AdaptiveKernel {
    mu: [f32; 2],
    sigma: [f32; 3],
    color: [f32; 3],
    weight: f32,
}

/// Fit one kernel per output cell with the EM-C loop: E-step softly assigns
/// window pixels to kernels (weights normalized per kernel, then shared per
/// pixel across kernels), M-step re-estimates position/color/covariance,
/// C-step clamps covariance eigenvalues so kernels may sharpen toward point
/// samplers but never grow past their initial cell-sized footprint. Two
/// deliberate departures from the reference: everything runs in Oklab (not
/// CIELAB), and kernel centers are clamped to their own cell so neighboring
/// kernels can never swap ownership (keeps the fit stable and deterministic
/// at low iteration counts).
fn fit_adaptive_kernels(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    grid: SamplingGrid,
    iterations: u32,
) -> Vec<AdaptiveKernel> {
    // Per-pixel [L, a, b, alpha], computed once.
    let mut plane = Vec::with_capacity(src_w * src_h * 4);
    for px in src.chunks_exact(4) {
        let lab = linear_to_oklab(px[0], px[1], px[2]);
        plane.extend_from_slice(&[lab[0], lab[1], lab[2], px[3]]);
    }

    let init_sx =
        ((grid.cell_w as f32 / 3.0) * (grid.cell_w as f32 / 3.0)).max(ADAPTIVE_LAMBDA_MIN);
    let init_sy =
        ((grid.cell_h as f32 / 3.0) * (grid.cell_h as f32 / 3.0)).max(ADAPTIVE_LAMBDA_MIN);
    let lambda_max = {
        let c = grid.cell_w.max(grid.cell_h) as f32 / 3.0;
        (c * c).max(ADAPTIVE_LAMBDA_MIN)
    };
    // One isotropic radius from the larger cell dimension, because clamped
    // eigenvalues let a kernel stretch that far along either axis.
    let radius = (ADAPTIVE_RADIUS_CELLS * grid.cell_w.max(grid.cell_h)).max(1.0);
    let (rx, ry) = (radius, radius);

    let mut kernels: Vec<AdaptiveKernel> = (0..dst_h)
        .flat_map(|cy| {
            (0..dst_w).map(move |cx| AdaptiveKernel {
                mu: [
                    (grid.origin_x + (cx as f64 + 0.5) * grid.cell_w) as f32,
                    (grid.origin_y + (cy as f64 + 0.5) * grid.cell_h) as f32,
                ],
                sigma: [init_sx, 0.0, init_sy],
                color: [0.5, 0.0, 0.0],
                weight: 0.0,
            })
        })
        .collect();

    // Reused buffers: per-pixel shared-assignment mass and one bounded
    // per-kernel weight scratchpad (window size is capped by the budget).
    let mut gamma = vec![0f32; src_w * src_h];
    let mut scratch: Vec<(u32, f32)> = Vec::new();
    let mut w_sums = vec![0f32; kernels.len()];

    for _ in 0..iterations {
        gamma.fill(1e-9);

        // E-step: normalize each kernel's window weights, then accumulate how
        // much total kernel mass claims every pixel.
        for (k, kernel) in kernels.iter().enumerate() {
            scratch.clear();
            let mut w_sum = 0f32;
            for_each_window_weight(
                kernel.mu,
                kernel.sigma,
                &plane,
                src_w,
                src_h,
                rx,
                ry,
                |i, w| {
                    scratch.push((i, w));
                    w_sum += w;
                },
            );
            w_sums[k] = w_sum;
            if w_sum <= f32::EPSILON {
                continue;
            }
            for &(i, w) in &scratch {
                gamma[i as usize] += w / w_sum;
            }
        }

        // M-step: re-estimate each kernel from its share of every pixel
        // (gamma_ki = normalized window weight / total pixel mass). The
        // Gaussian sweep is recomputed with the identical routine, so the
        // weights match the E-step exactly.
        for (k, kernel) in kernels.iter_mut().enumerate() {
            let w_sum = w_sums[k];
            kernel.weight = 0.0;
            if w_sum <= f32::EPSILON {
                continue;
            }
            let (mut tw, mut sx, mut sy) = (0f64, 0f64, 0f64);
            let (mut sxx, mut sxy, mut syy) = (0f64, 0f64, 0f64);
            let mut sc = [0f64; 3];
            for_each_window_weight(
                kernel.mu,
                kernel.sigma,
                &plane,
                src_w,
                src_h,
                rx,
                ry,
                |i, w| {
                    let i = i as usize;
                    let g = (w / w_sum) as f64 / gamma[i] as f64;
                    let x = (i % src_w) as f64 + 0.5;
                    let y = (i / src_w) as f64 + 0.5;
                    tw += g;
                    sx += g * x;
                    sy += g * y;
                    sxx += g * x * x;
                    sxy += g * x * y;
                    syy += g * y * y;
                    sc[0] += g * plane[i * 4] as f64;
                    sc[1] += g * plane[i * 4 + 1] as f64;
                    sc[2] += g * plane[i * 4 + 2] as f64;
                },
            );
            if tw <= f64::EPSILON {
                continue;
            }
            let mx = sx / tw;
            let my = sy / tw;
            kernel.color = [
                (sc[0] / tw) as f32,
                (sc[1] / tw) as f32,
                (sc[2] / tw) as f32,
            ];
            kernel.weight = tw as f32;

            // Clamp the center into its own cell (and the image).
            let cx = k % dst_w;
            let cy = k / dst_w;
            let x0 = (grid.origin_x + cx as f64 * grid.cell_w).max(0.0);
            let x1 = (grid.origin_x + (cx + 1) as f64 * grid.cell_w).min(src_w as f64);
            let y0 = (grid.origin_y + cy as f64 * grid.cell_h).max(0.0);
            let y1 = (grid.origin_y + (cy + 1) as f64 * grid.cell_h).min(src_h as f64);
            kernel.mu = [
                mx.clamp(x0, x1.max(x0)) as f32,
                my.clamp(y0, y1.max(y0)) as f32,
            ];

            // C-step: central second moments around the unclamped mean, then
            // eigenvalue clamp.
            let cov = [
                (sxx / tw - mx * mx) as f32,
                (sxy / tw - mx * my) as f32,
                (syy / tw - my * my) as f32,
            ];
            kernel.sigma = clamp_sym2x2_eigenvalues(cov, ADAPTIVE_LAMBDA_MIN, lambda_max);
        }
    }
    kernels
}

/// Visit every opaque pixel in the kernel's window with its Gaussian weight
/// (already multiplied by pixel alpha, so transparent pixels never pull a
/// kernel). Shared by the E- and M-steps so both see bit-identical weights.
#[allow(clippy::too_many_arguments)]
fn for_each_window_weight(
    mu: [f32; 2],
    sigma: [f32; 3],
    plane: &[f32],
    src_w: usize,
    src_h: usize,
    rx: f64,
    ry: f64,
    mut visit: impl FnMut(u32, f32),
) {
    let det = sigma[0] * sigma[2] - sigma[1] * sigma[1];
    if det <= 1e-12 {
        return; // Eigenvalue clamps keep sigma positive definite; belt and braces.
    }
    let ia = sigma[2] / det;
    let ib = -sigma[1] / det;
    let id = sigma[0] / det;
    let x_lo = (mu[0] as f64 - rx).floor().max(0.0) as usize;
    let x_hi = ((mu[0] as f64 + rx).ceil().min(src_w as f64)).max(0.0) as usize;
    let y_lo = (mu[1] as f64 - ry).floor().max(0.0) as usize;
    let y_hi = ((mu[1] as f64 + ry).ceil().min(src_h as f64)).max(0.0) as usize;
    for y in y_lo..y_hi {
        let dy = y as f32 + 0.5 - mu[1];
        for x in x_lo..x_hi {
            let i = y * src_w + x;
            let alpha = plane[i * 4 + 3];
            if alpha <= 0.0 {
                continue;
            }
            let dx = x as f32 + 0.5 - mu[0];
            let q = 0.5 * (dx * dx * ia + 2.0 * dx * dy * ib + dy * dy * id);
            if q >= ADAPTIVE_EXPONENT_CUTOFF || q.is_nan() {
                continue;
            }
            visit(i as u32, (-q).exp() * alpha);
        }
    }
}

/// Clamp the eigenvalues of a symmetric 2x2 matrix [xx, xy, yy] via its exact
/// eigendecomposition: theta = atan2(2*xy, xx - yy) / 2 rotates onto the
/// principal axes, the eigenvalues are clamped there, and the matrix is
/// rebuilt as R * diag(l1, l2) * R^T. (The reference's "simplified
/// reconstruction" is not a valid decomposition; this is.)
fn clamp_sym2x2_eigenvalues(sigma: [f32; 3], lambda_min: f32, lambda_max: f32) -> [f32; 3] {
    let [a, b, d] = sigma;
    let theta = 0.5 * (2.0 * b).atan2(a - d);
    let (s, c) = theta.sin_cos();
    let l1 = (c * c * a + 2.0 * c * s * b + s * s * d).clamp(lambda_min, lambda_max);
    let l2 = (s * s * a - 2.0 * c * s * b + c * c * d).clamp(lambda_min, lambda_max);
    [
        c * c * l1 + s * s * l2,
        c * s * (l1 - l2),
        s * s * l1 + c * c * l2,
    ]
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
            CellMode::Adaptive,
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
    fn sampling_grid_can_start_at_detected_phase() {
        let src = vec![
            0.0, 0.0, 0.0, 1.0, //
            0.2, 0.2, 0.2, 1.0, //
            0.2, 0.2, 0.2, 1.0, //
            0.8, 0.8, 0.8, 1.0, //
            0.8, 0.8, 0.8, 1.0, //
            0.4, 0.4, 0.4, 1.0, //
            0.4, 0.4, 0.4, 1.0,
        ];
        let out = downsample_grid(
            &src,
            7,
            1,
            3,
            1,
            SamplingGrid {
                origin_x: 1.0,
                origin_y: 0.0,
                cell_w: 2.0,
                cell_h: 1.0,
            },
            CellMode::Box,
        );
        assert!((out[0] - 0.2).abs() < 1e-6, "got {out:?}");
        assert!((out[4] - 0.8).abs() < 1e-6, "got {out:?}");
        assert!((out[8] - 0.4).abs() < 1e-6, "got {out:?}");
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
    fn dominant_threshold_falls_back_when_no_color_wins() {
        let src = vec![
            1.0, 0.0, 0.0, 1.0, //
            0.0, 1.0, 0.0, 1.0, //
            0.0, 0.0, 1.0, 1.0, //
            1.0, 1.0, 1.0, 1.0,
        ];
        let out = downsample_with_dominant_threshold(&src, 2, 2, 1, 1, CellMode::Dominant, 0.30);
        assert!(
            (out[0] - 0.5).abs() < 1e-6
                && (out[1] - 0.5).abs() < 1e-6
                && (out[2] - 0.5).abs() < 1e-6,
            "expected box fallback, got {out:?}"
        );
    }

    #[test]
    fn dominant_reunites_colors_split_across_a_bucket_boundary() {
        // 0.322 and 0.323 straddle the plain-grid boundary at 10/31≈0.32258:
        // plain grid splits them 3+3, letting the 4-pixel 0.5 block win. The
        // shifted grid keeps them together (weight 6), so the fuzzy family
        // must win and the output must come from it.
        let mut src = Vec::new();
        for &v in &[
            0.322f32, 0.322, 0.322, 0.323, 0.323, 0.323, 0.5, 0.5, 0.5, 0.5,
        ] {
            src.extend_from_slice(&[v, v, v, 1.0]);
        }
        let out = downsample_with_dominant_threshold(&src, 10, 1, 1, 1, CellMode::Dominant, 0.25);
        assert!(
            (out[0] - 0.3225).abs() < 0.002,
            "split color family should win: {out:?}"
        );
        assert!(out[0] != 0.5, "single-grid winner must not survive");
    }

    #[test]
    fn dominant_representative_is_a_real_cell_color() {
        // Members 0.30, 0.30, 0.316 share a plain bucket; the old mean gave
        // the synthetic 0.3053. The representative must now be a member.
        let mut src = Vec::new();
        for &v in &[0.30f32, 0.30, 0.316] {
            src.extend_from_slice(&[v, v, v, 1.0]);
        }
        let out = downsample(&src, 3, 1, 1, 1, CellMode::Dominant);
        assert!(
            (out[0] - 0.30).abs() < 1e-6,
            "expected the nearest real member, got {out:?}"
        );
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
        for mode in [CellMode::Median, CellMode::Adaptive] {
            let out = downsample(&src, 4, 4, 2, 2, mode);
            assert!(out.iter().all(|&v| v == 0.0), "{mode:?}");
        }
    }

    fn adaptive_opts() -> DownsampleOptions {
        DownsampleOptions {
            mode: CellMode::Adaptive,
            ..DownsampleOptions::default()
        }
    }

    fn full_grid(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> SamplingGrid {
        SamplingGrid {
            origin_x: 0.0,
            origin_y: 0.0,
            cell_w: src_w as f64 / dst_w as f64,
            cell_h: src_h as f64 / dst_h as f64,
        }
    }

    #[test]
    fn adaptive_is_deterministic() {
        let src: Vec<f32> = (0..32 * 24)
            .flat_map(|i| {
                let v = (i % 251) as f32 / 250.0;
                [v, (v * 3.0) % 1.0, (v * 7.0) % 1.0, 1.0]
            })
            .collect();
        let grid = full_grid(32, 24, 8, 6);
        let a = downsample_grid_opts(&src, 32, 24, 8, 6, grid, adaptive_opts());
        let b = downsample_grid_opts(&src, 32, 24, 8, 6, grid, adaptive_opts());
        assert_eq!(a, b, "adaptive must be bit-identical across runs");
    }

    #[test]
    fn adaptive_transparent_pixels_do_not_tint_and_alpha_mean_is_preserved() {
        // Calm cell: one opaque red + three transparent greens -> red, alpha
        // 1/4 (identical rules to every other mode).
        let mut src = flat(2, 2, [0.0, 1.0, 0.0, 0.0]);
        src[0] = 1.0;
        src[1] = 0.0;
        src[2] = 0.0;
        src[3] = 1.0;
        let out = downsample_grid_opts(&src, 2, 2, 1, 1, full_grid(2, 2, 1, 1), adaptive_opts());
        assert!(
            (out[0] - 1.0).abs() < 1e-6 && out[1].abs() < 1e-6,
            "{out:?}"
        );
        assert!(
            (out[3] - 0.25).abs() < 1e-6,
            "mean alpha preserved: {out:?}"
        );

        // Busy cell: opaque red + opaque blue + transparent neon green. The
        // snap may only pick an opaque member, never the green.
        let src = vec![
            1.0, 0.0, 0.0, 1.0, //
            1.0, 0.0, 0.0, 1.0, //
            0.0, 0.0, 1.0, 1.0, //
            0.0, 1.0, 0.0, 0.0,
        ];
        let out = downsample_grid_opts(&src, 2, 2, 1, 1, full_grid(2, 2, 1, 1), adaptive_opts());
        let color = [out[0], out[1], out[2]];
        assert!(
            color == [1.0, 0.0, 0.0] || color == [0.0, 0.0, 1.0],
            "must be a real opaque member, got {out:?}"
        );
    }

    #[test]
    fn adaptive_keeps_hard_edges_crisp_through_an_antialiased_strip() {
        // 24x4 -> 6x1 with 4px cells. Dark field | 2px blended strip | light
        // field; the strip straddles cells 2 and 3 as a minority. The EM
        // kernels must resolve those cells to the majority side's real color,
        // never to the blend.
        let mut src = Vec::new();
        for _ in 0..4 {
            for x in 0..24 {
                let v: f32 = if x <= 10 {
                    0.1
                } else if x <= 12 {
                    0.5
                } else {
                    0.9
                };
                src.extend_from_slice(&[v, v, v, 1.0]);
            }
        }
        let out = downsample_grid_opts(&src, 24, 4, 6, 1, full_grid(24, 4, 6, 1), adaptive_opts());
        for (cell, expected) in [(1usize, 0.1f32), (2, 0.1), (3, 0.9), (4, 0.9)] {
            assert!(
                (out[cell * 4] - expected).abs() < 1e-6,
                "cell {cell} should be {expected}, got {:?}",
                &out[cell * 4..cell * 4 + 4]
            );
        }
        assert!(
            out.chunks(4).all(|c| (c[0] - 0.5).abs() > 1e-6),
            "the antialiasing blend must not survive: {out:?}"
        );
    }

    #[test]
    fn adaptive_falls_back_to_detail_when_cells_exceed_the_cap() {
        // 33px cells are over ADAPTIVE_MAX_CELL, so Adaptive must produce
        // exactly what Detail produces.
        let src: Vec<f32> = (0..132 * 4)
            .flat_map(|i| {
                let v = (i % 97) as f32 / 96.0;
                [v, v, v, 1.0]
            })
            .collect();
        let grid = full_grid(132, 4, 4, 1);
        let adaptive = downsample_grid_opts(&src, 132, 4, 4, 1, grid, adaptive_opts());
        let detail = downsample_grid_opts(
            &src,
            132,
            4,
            4,
            1,
            grid,
            DownsampleOptions::default(), // Detail
        );
        assert_eq!(adaptive, detail, "over-budget adaptive must equal Detail");
    }

    #[test]
    fn adaptive_budget_rejects_huge_jobs_and_accepts_preview_sized_ones() {
        assert!(adaptive_fits_budget(16, 16, 4.0, 4.0, 3));
        assert!(adaptive_fits_budget(128, 128, 8.0, 8.0, 3));
        // Cell dimension cap.
        assert!(!adaptive_fits_budget(4, 1, 33.0, 4.0, 1));
        // Visit-count cap (a 16-megapixel job).
        assert!(!adaptive_fits_budget(2048, 2048, 8.0, 8.0, 2));
        // Iterations are part of the cost.
        assert!(adaptive_fits_budget(256, 256, 8.0, 8.0, 1));
        assert!(!adaptive_fits_budget(1024, 1024, 8.0, 8.0, 8));
    }
}
