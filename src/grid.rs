//! Pixel-grid inference: find the lattice (pitch, origin, cell counts) that the
//! source's art pixels sit on, or decide that there is none.
//!
//! Model. An upscaled pixel-art image changes colour only where one art pixel
//! meets the next, so along each axis the *positions* of colour changes are
//! supported on a lattice `origin + k * pitch`. That holds for every upscaling
//! kernel, with a different observable for each:
//!
//! * hard or lightly blurred edges (nearest, JPEG, noise, mild blur, sharpened
//!   bicubic) put the local maxima of the first difference on cell boundaries;
//! * interpolating kernels (bilinear, bicubic, Lanczos) and heavier blur put
//!   the local maxima of the second difference ("kinks") on cell centres.
//!
//! Both observables are reduced to a per-axis position histogram weighted by
//! contrast, and the lattice hypothesis is scored by the coherence
//! `R(f) = |sum_x P[x] e^{-2 pi i f x}| / sum_x P[x]`, which is 1 exactly when
//! all weight sits on one lattice of pitch `1/f` and stays near the random-walk
//! floor `sqrt(sum P^2) / sum P` when there is no lattice. Fractional pitches
//! need no special case, the phase of the peak gives the origin, and because
//! only positions enter (not gradient magnitudes), blur does not attenuate the
//! peak the way it would in a plain power spectrum.
//!
//! Every lattice of pitch `s` is also a lattice of pitch `s/2, s/3, ...`, so the
//! strongest peak may be a harmonic. The fundamental is the longest period
//! whose coherence stays close to the best one: a period twice too long leaves
//! the odd lines unexplained and its coherence collapses.
//!
//! Everything here is integer or plain `f64` add/multiply (own sin/cos, own
//! FFT), so native and WebAssembly builds agree bit for bit.

use image::RgbaImage;

use crate::fmath;

/// Longest pitch considered, in source pixels per art pixel.
const MAX_PITCH: f64 = 64.0;
/// The image must hold at least this many periods of a candidate pitch.
const MIN_PERIODS: f64 = 6.0;
/// Spectrum oversampling: frequency step is `1 / (OVERSAMPLE * n)`.
const OVERSAMPLE: usize = 8;
/// A colour step must exceed this (sum of |delta| over premultiplied RGBA,
/// 8-bit units) to count as an edge or a kink.
const STEP_THRESHOLD: i32 = 10;
/// Minimum share of edge weight a lattice must explain coherently.
const MIN_COHERENCE: f64 = 0.15;
/// Minimum peak-to-floor ratio, combined over both axes in quadrature. Under
/// the no-lattice hypothesis the chance of reaching `z` at one frequency is
/// about `exp(-z^2)`.
const MIN_SIGNIFICANCE: f64 = 4.0;
/// A longer period is preferred while it keeps this share of the best coherence.
const SUBHARMONIC_KEEP: f64 = 0.8;
/// Largest lattice drift (in cells, over the whole axis) for which a nearly
/// integer pitch is snapped to the integer.
const INTEGER_SNAP_CELLS: f64 = 0.25;

/// A fitted lattice. Cell `(cx, cy)` covers source x in
/// `origin_x + cx * pitch_x .. origin_x + (cx + 1) * pitch_x` (same for y).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridFit {
    pub pitch_x: f64,
    pub pitch_y: f64,
    pub origin_x: f64,
    pub origin_y: f64,
    pub cols: u32,
    pub rows: u32,
    /// Coherence of the chosen pitch, 0..=1.
    pub coherence: f32,
    /// Peak-to-floor ratio of the strongest lattice, both axes in quadrature.
    pub significance: f32,
}

/// Outcome of grid inference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GridDecision {
    Lattice(GridFit),
    /// No trustworthy lattice. Carries the best (insufficient) evidence.
    NoGrid { coherence: f32, significance: f32 },
}

/// sin and cos of `turns * 2 pi`, from add/multiply only (deterministic).
fn sin_cos_turn(turns: f64) -> (f64, f64) {
    let t = turns - turns.floor();
    let quarter = t * 4.0;
    let k = (quarter + 0.5).floor();
    let x = (quarter - k) * std::f64::consts::FRAC_PI_2; // -pi/4 ..= pi/4
    let x2 = x * x;
    // Taylor series in Horner form; |x| <= pi/4 keeps the truncation < 1e-17.
    let s = x
        * (1.0
            + x2 * (-1.0 / 6.0
                + x2 * (1.0 / 120.0
                    + x2 * (-1.0 / 5040.0
                        + x2 * (1.0 / 362_880.0
                            + x2 * (-1.0 / 39_916_800.0
                                + x2 * (1.0 / 6_227_020_800.0
                                    + x2 * (-1.0 / 1_307_674_368_000.0))))))));
    let c = 1.0
        + x2 * (-0.5
            + x2 * (1.0 / 24.0
                + x2 * (-1.0 / 720.0
                    + x2 * (1.0 / 40_320.0
                        + x2 * (-1.0 / 3_628_800.0
                            + x2 * (1.0 / 479_001_600.0
                                + x2 * (-1.0 / 87_178_291_200.0
                                    + x2 * (1.0 / 20_922_789_888_000.0))))))));
    match (k as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// In-place radix-2 FFT (forward, `e^{-i...}`), `re.len()` a power of two.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two() && im.len() == n);
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let table: Vec<(f64, f64)> = (0..half)
            .map(|k| {
                let (s, c) = sin_cos_turn(k as f64 / len as f64);
                (c, -s)
            })
            .collect();
        for start in (0..n).step_by(len) {
            for (k, &(wr, wi)) in table.iter().enumerate() {
                let a = start + k;
                let b = a + half;
                let tr = re[b] * wr - im[b] * wi;
                let ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// Position histograms of one axis. Index `i` of `edge` is the boundary
/// between source pixels `i - 1` and `i`; index `i` of `kink` is the centre of
/// source pixel `i` (coordinate `i + 0.5`).
struct AxisProfile {
    edge: Vec<f64>,
    kink: Vec<f64>,
}

/// Premultiplied RGBA bytes, borrowed when the source is fully opaque.
fn premultiplied(src: &RgbaImage) -> std::borrow::Cow<'_, [u8]> {
    let raw = src.as_raw();
    if raw.chunks_exact(4).all(|p| p[3] == 255) {
        return std::borrow::Cow::Borrowed(raw.as_slice());
    }
    let mut out = Vec::with_capacity(raw.len());
    for p in raw.chunks_exact(4) {
        let a = p[3] as u32;
        for &c in &p[..3] {
            out.push(((c as u32 * a + 127) / 255) as u8);
        }
        out.push(p[3]);
    }
    std::borrow::Cow::Owned(out)
}

/// Add the non-maximum-suppressed first and second differences of one scan
/// line (`len` pixels starting at byte `start`, `stride` bytes apart).
fn accumulate_line(
    buf: &[u8],
    start: usize,
    stride: usize,
    len: usize,
    d1: &mut Vec<i32>,
    d2: &mut Vec<i32>,
    profile: &mut AxisProfile,
) {
    d1.clear();
    d2.clear();
    let px = |i: usize| -> &[u8] { &buf[start + i * stride..start + i * stride + 4] };
    for i in 0..len - 1 {
        let (a, b) = (px(i), px(i + 1));
        let mut sum = 0i32;
        for c in 0..4 {
            sum += (b[c] as i32 - a[c] as i32).abs();
        }
        d1.push(sum);
    }
    for i in 0..len.saturating_sub(2) {
        let (a, b, c3) = (px(i), px(i + 1), px(i + 2));
        let mut sum = 0i32;
        for c in 0..4 {
            sum += (c3[c] as i32 - 2 * b[c] as i32 + a[c] as i32).abs();
        }
        d2.push(sum);
    }
    let peaks = |d: &[i32], out: &mut [f64]| {
        for i in 0..d.len() {
            let v = d[i];
            if v <= STEP_THRESHOLD {
                continue;
            }
            let left = if i > 0 { d[i - 1] } else { 0 };
            let right = if i + 1 < d.len() { d[i + 1] } else { 0 };
            if v >= left && v > right {
                out[i + 1] += v as f64;
            }
        }
    };
    peaks(d1, &mut profile.edge);
    peaks(d2, &mut profile.kink);
}

fn axis_profiles(src: &RgbaImage) -> (AxisProfile, AxisProfile) {
    let (w, h) = (src.width() as usize, src.height() as usize);
    let buf = premultiplied(src);
    let mut px = AxisProfile {
        edge: vec![0.0; w + 1],
        kink: vec![0.0; w + 1],
    };
    let mut py = AxisProfile {
        edge: vec![0.0; h + 1],
        kink: vec![0.0; h + 1],
    };
    let (mut d1, mut d2) = (Vec::new(), Vec::new());
    for y in 0..h {
        accumulate_line(&buf, y * w * 4, 4, w, &mut d1, &mut d2, &mut px);
    }
    for x in 0..w {
        accumulate_line(&buf, x * 4, w * 4, h, &mut d1, &mut d2, &mut py);
    }
    (px, py)
}

/// `|FFT| / sum` of a zero-padded profile, plus its no-lattice floor.
fn coherence_spectrum(profile: &[f64], n_fft: usize) -> Option<(Vec<f64>, f64)> {
    let total: f64 = profile.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let mut re = vec![0.0; n_fft];
    let mut im = vec![0.0; n_fft];
    re[..profile.len()].copy_from_slice(profile);
    fft(&mut re, &mut im);
    let mag: Vec<f64> = (0..=n_fft / 2)
        .map(|k| (re[k] * re[k] + im[k] * im[k]).sqrt() / total)
        .collect();
    let floor = profile.iter().map(|v| v * v).sum::<f64>().sqrt() / total;
    Some((mag, floor))
}

/// Complex coherence of `profile` at an arbitrary frequency (cycles/pixel),
/// with sample `i` at coordinate `i + offset`.
fn coherence_at(profile: &[f64], freq: f64, offset: f64) -> (f64, f64) {
    let total: f64 = profile.iter().sum();
    if total <= 0.0 {
        return (0.0, 0.0);
    }
    let (mut re, mut im) = (0.0, 0.0);
    for (i, &v) in profile.iter().enumerate() {
        if v != 0.0 {
            let (s, c) = sin_cos_turn(freq * (i as f64 + offset));
            re += v * c;
            im -= v * s;
        }
    }
    (re / total, im / total)
}

fn magnitude(c: (f64, f64)) -> f64 {
    (c.0 * c.0 + c.1 * c.1).sqrt()
}

/// Lattice origin (first line position, in `0..pitch`) from a coherence value.
fn phase_origin(c: (f64, f64), pitch: f64) -> f64 {
    let m = magnitude(c);
    if m <= 0.0 {
        return 0.0;
    }
    // Rounded through f32 so platform libm differences cannot leak out.
    let angle = fmath::atan2((c.1 / m) as f32, (c.0 / m) as f32) as f64;
    let origin = -angle / std::f64::consts::TAU * pitch;
    origin - (origin / pitch).floor() * pitch
}

/// One axis's lattice: (pitch, origin, cell count).
fn fit_axis(profile: &AxisProfile, len: usize, f0: f64, bin: f64) -> (f64, f64, u32) {
    let at = |f: f64| {
        (
            coherence_at(&profile.edge, f, 0.0),
            coherence_at(&profile.kink, f, 0.5),
        )
    };
    // Which observable carries this axis: boundaries or cell centres.
    let (e0, k0) = at(f0);
    let use_edge = magnitude(e0) >= 0.5 * magnitude(k0);
    let score = |f: f64| {
        let (e, k) = at(f);
        if use_edge {
            magnitude(e)
        } else {
            magnitude(k)
        }
    };
    // Refine the frequency to 1/16 of a spectrum bin.
    let mut best_f = f0;
    let mut best = score(f0);
    for step in -24i32..=24 {
        let f = f0 + step as f64 * bin / 16.0;
        if f <= 0.0 || f > 0.5 {
            continue;
        }
        let v = score(f);
        if v > best {
            best = v;
            best_f = f;
        }
    }
    let mut pitch = 1.0 / best_f;
    let n = len as f64;
    let origin_of = |pitch: f64| {
        let (e, k) = at(1.0 / pitch);
        let o = if use_edge {
            phase_origin(e, pitch)
        } else {
            // Kinks sit on cell centres; boundaries are half a cell away.
            phase_origin(k, pitch) + pitch * 0.5
        };
        o - (o / pitch).floor() * pitch
    };

    let rounded = pitch.round();
    let mut origin;
    if rounded >= 2.0 && (pitch - rounded).abs() * n / (pitch * pitch) < INTEGER_SNAP_CELLS {
        pitch = rounded;
        origin = origin_of(pitch).round();
        if origin >= pitch {
            origin -= pitch;
        }
    } else {
        origin = origin_of(pitch);
        // Whole-image prior: a resized image is an integer number of cells
        // starting at the corner. Take it when the fit agrees to a fraction of
        // a cell; it removes the residual pitch error entirely.
        let cells = (n / pitch).round();
        let edge_distance = origin.min(pitch - origin);
        if cells >= 1.0 && (n / pitch - cells).abs() < 0.2 && edge_distance < 0.3 * pitch {
            pitch = n / cells;
            origin = 0.0;
        }
    }
    // An origin within half a source pixel of a full cell is that cell's edge.
    if origin > pitch - 0.5 {
        origin -= pitch;
    }
    let cells = ((n - origin + 0.5) / pitch).floor().max(1.0);
    (pitch, origin, cells as u32)
}

/// Infer the pixel grid of `src`.
pub fn detect(src: &RgbaImage) -> GridDecision {
    let none = |coherence: f64, significance: f64| GridDecision::NoGrid {
        coherence: coherence as f32,
        significance: significance as f32,
    };
    let (w, h) = (src.width() as usize, src.height() as usize);
    let max_pitch = MAX_PITCH.min(w.min(h) as f64 / MIN_PERIODS);
    if max_pitch < 2.0 {
        return none(0.0, 0.0);
    }
    let (px, py) = axis_profiles(src);
    let n = w.max(h) + 1;
    let n_fft = (n * OVERSAMPLE).next_power_of_two();

    // One observable = the same kind of profile on both axes. Its coherence is
    // the mean over the axes that have any weight; its significance treats
    // the axes as independent evidence for the same pitch.
    type Spectrum = Option<(Vec<f64>, f64)>;
    let observe = |x: &Spectrum, y: &Spectrum, k: usize| -> (f64, f64) {
        let (mut r, mut z2, mut axes) = (0.0, 0.0, 0.0);
        for (mag, floor) in [x, y].into_iter().flatten() {
            r += mag[k];
            z2 += (mag[k] / floor) * (mag[k] / floor);
            axes += 1.0;
        }
        if axes == 0.0 {
            (0.0, 0.0)
        } else {
            (r / axes, z2.sqrt())
        }
    };
    let edge = (
        coherence_spectrum(&px.edge, n_fft),
        coherence_spectrum(&py.edge, n_fft),
    );
    let kink = (
        coherence_spectrum(&px.kink, n_fft),
        coherence_spectrum(&py.kink, n_fft),
    );
    let k_lo = ((n_fft as f64 / max_pitch).ceil() as usize).max(1);
    let k_hi = n_fft / 2;
    if k_lo >= k_hi {
        return none(0.0, 0.0);
    }
    // R(f) = the better observable at each frequency, with its significance.
    let value = |k: usize| -> (f64, f64) {
        let e = observe(&edge.0, &edge.1, k);
        let q = observe(&kink.0, &kink.1, k);
        if q.0 > e.0 {
            q
        } else {
            e
        }
    };
    let mut best_k = k_lo;
    let mut best = (0.0, 0.0);
    for k in k_lo..=k_hi {
        let v = value(k);
        if v.0 > best.0 {
            best = v;
            best_k = k;
        }
    }
    let (r_max, significance) = best;
    if r_max < MIN_COHERENCE || significance < MIN_SIGNIFICANCE {
        return none(r_max, significance);
    }

    // Longest period that still explains (nearly) as much as the best one.
    let mut chosen_k = best_k as f64;
    let mut chosen_r = r_max;
    let mut m = 2usize;
    while best_k / m >= k_lo {
        let centre = best_k as f64 / m as f64;
        let lo = ((centre - 2.0).floor().max(k_lo as f64)) as usize;
        let hi = ((centre + 2.0).ceil() as usize).min(k_hi);
        let mut local = (0.0, lo);
        for k in lo..=hi {
            let v = value(k).0;
            if v > local.0 {
                local = (v, k);
            }
        }
        if local.0 >= SUBHARMONIC_KEEP * r_max {
            chosen_k = local.1 as f64;
            chosen_r = local.0;
        }
        m += 1;
    }

    let bin = 1.0 / n_fft as f64;
    let f0 = chosen_k * bin;
    let (pitch_x, origin_x, cols) = fit_axis(&px, w, f0, bin);
    let (pitch_y, origin_y, rows) = fit_axis(&py, h, f0, bin);
    GridDecision::Lattice(GridFit {
        pitch_x,
        pitch_y,
        origin_x,
        origin_y,
        cols,
        rows,
        coherence: chosen_r as f32,
        significance: significance as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{imageops, Rgba};

    /// Deterministic pseudo-random art: `w x h` cells from a small palette.
    fn art(w: u32, h: u32, seed: u32) -> RgbaImage {
        let palette = [
            [20u8, 24, 40],
            [230, 170, 40],
            [50, 160, 100],
            [245, 240, 220],
            [210, 35, 85],
            [70, 90, 200],
        ];
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
        RgbaImage::from_fn(w, h, |_, _| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let [r, g, b] = palette[(state >> 24) as usize % palette.len()];
            Rgba([r, g, b, 255])
        })
    }

    fn lattice(img: &RgbaImage) -> GridFit {
        match detect(img) {
            GridDecision::Lattice(fit) => fit,
            other => panic!("expected a lattice, got {other:?}"),
        }
    }

    #[test]
    fn sin_cos_turn_matches_std() {
        for i in -2000..2000 {
            let t = i as f64 / 997.0;
            let (s, c) = sin_cos_turn(t);
            let (es, ec) = (t * std::f64::consts::TAU).sin_cos();
            assert!((s - es).abs() < 1e-12 && (c - ec).abs() < 1e-12, "{t}");
        }
    }

    #[test]
    fn integer_nearest_upscales_are_recovered_at_every_scale() {
        for scale in [2u32, 3, 4, 5, 8, 16] {
            let small = art(40, 28, scale);
            let big = imageops::resize(
                &small,
                40 * scale,
                28 * scale,
                imageops::FilterType::Nearest,
            );
            let fit = lattice(&big);
            assert_eq!((fit.cols, fit.rows), (40, 28), "scale {scale}");
            assert_eq!((fit.pitch_x, fit.pitch_y), (scale as f64, scale as f64));
            assert_eq!((fit.origin_x, fit.origin_y), (0.0, 0.0));
        }
    }

    #[test]
    fn phase_offset_is_found_and_partial_cells_are_dropped() {
        let small = art(30, 30, 7);
        let big = imageops::resize(&small, 150, 150, imageops::FilterType::Nearest);
        let cropped = imageops::crop_imm(&big, 3, 1, 140, 144).to_image();
        let fit = lattice(&cropped);
        assert_eq!((fit.pitch_x, fit.pitch_y), (5.0, 5.0));
        assert_eq!((fit.origin_x, fit.origin_y), (2.0, 4.0));
        assert_eq!((fit.cols, fit.rows), (27, 28));
    }

    #[test]
    fn fractional_smooth_upscales_are_recovered() {
        for (scale, filter) in [
            (4.37, imageops::FilterType::Triangle),
            (5.6, imageops::FilterType::CatmullRom),
            (6.25, imageops::FilterType::Lanczos3),
        ] {
            let small = art(48, 36, 3);
            let w = (48.0 * scale as f64).ceil() as u32;
            let h = (36.0 * scale as f64).ceil() as u32;
            let big = imageops::resize(&small, w, h, filter);
            let fit = lattice(&big);
            assert_eq!((fit.cols, fit.rows), (48, 36), "scale {scale}");
            assert!((fit.pitch_x - w as f64 / 48.0).abs() < 1e-9);
            assert_eq!(fit.origin_x, 0.0);
        }
    }

    #[test]
    fn smooth_gradients_and_noise_have_no_grid() {
        let gradient = RgbaImage::from_fn(240, 200, |x, y| {
            Rgba([(x * 255 / 239) as u8, (y * 255 / 199) as u8, 128, 255])
        });
        assert!(matches!(detect(&gradient), GridDecision::NoGrid { .. }));
        let mut state = 99u32;
        let noise = RgbaImage::from_fn(240, 200, |_, _| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = (state >> 24) as u8;
            Rgba([v, v, v, 255])
        });
        assert!(matches!(detect(&noise), GridDecision::NoGrid { .. }));
    }

    #[test]
    fn transparent_sprites_use_alpha_edges() {
        let mut small = art(24, 24, 11);
        for (x, y, p) in small.enumerate_pixels_mut() {
            if (x as i32 - 12).pow(2) + (y as i32 - 12).pow(2) > 100 {
                *p = Rgba([255, 0, 255, 0]);
            }
        }
        let big = imageops::resize(&small, 96, 96, imageops::FilterType::Nearest);
        let fit = lattice(&big);
        assert_eq!((fit.cols, fit.rows, fit.pitch_x), (24, 24, 4.0));
    }
}
