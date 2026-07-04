//! Shared conversion pipeline: one `Config` + one `convert()` that both the CLI
//! (`main.rs`) and the GUI server (`bin/gui.rs`) call, so the two front ends can
//! never drift. This module owns the orchestration (target-grid selection, auto
//! pixel-size detection, downsample glue, quantize, nearest scale, compare sheet);
//! the pure-std leaf algorithms still live in `color`/`kmeans`/`dither`/`downsample`.
//!
//! `image` is an approved, committed dependency, so this module uses `RgbaImage`
//! directly — it does NOT touch the filesystem, though, so callers resolve palette
//! files/text themselves and hand in a `PaletteChoice`.

use image::{imageops::FilterType, Rgba, RgbaImage};

use crate::{
    color::{linear_to_oklab, oklab_to_srgb8, srgb8_to_oklab, srgb_to_linear},
    dither::ordered_dither,
    downsample::{
        downsample_grid_with_dominant_threshold, downsample_with_dominant_threshold, CellMode,
        SamplingGrid, DEFAULT_DOMINANT_THRESHOLD,
    },
    kmeans::{build_palette, nearest},
    palettes,
};

/// Ordered-dither selection, front-end-agnostic (CLI/GUI map their own enums onto this).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dither {
    None,
    Bayer4,
    Bayer8,
}

/// Where the palette comes from. Callers resolve files/UI into one of these; the
/// pipeline never reads from disk.
#[derive(Clone, Debug)]
pub enum PaletteChoice {
    /// Build an adaptive palette of `Config::colors` entries via k-means in Oklab.
    Adaptive,
    /// A built-in palette by name (pico8/gameboy/sweetie16, see `palettes::builtin`).
    Builtin(String),
    /// Raw Lospec-style hex text ("RRGGBB"/"#RRGGBB" per line).
    HexList(String),
}

/// Everything the pipeline needs. Defaults match the CLI's flag defaults.
#[derive(Clone, Debug)]
pub struct Config {
    /// Long side of the pixel-art grid (used unless `pixel_size`/`auto_pixel_size`).
    pub size: u32,
    /// Estimated source pixels per output pixel. Overrides `size` when set.
    pub pixel_size: Option<f64>,
    /// Estimate pixel size from image edges. Ignored when `pixel_size` is set.
    pub auto_pixel_size: bool,
    /// Align exact pixel-size cells to the strongest detected grid phase.
    pub snap_grid: bool,
    /// Adaptive palette size (ignored for fixed palettes).
    pub colors: usize,
    /// Palette source.
    pub palette: PaletteChoice,
    /// Ordered dithering mode.
    pub dither: Dither,
    /// Dither strength, 0.0..=1.0.
    pub dither_strength: f32,
    /// Nearest-neighbor preview scale baked into the output. 1 = raw grid.
    pub scale: u32,
    /// Alpha threshold, 0..=255. Below this a pixel becomes fully transparent.
    pub alpha_threshold: u8,
    /// Cell reduction mode used while downsampling.
    pub cell: CellMode,
    /// Minimum winning-bucket coverage for dominant/detail cells before falling back to mean.
    pub dominant_threshold: f32,
    /// Write an original|result side-by-side comparison sheet.
    pub compare: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            size: 64,
            pixel_size: None,
            auto_pixel_size: false,
            snap_grid: true,
            colors: 16,
            palette: PaletteChoice::Adaptive,
            dither: Dither::None,
            dither_strength: 0.35,
            scale: 1,
            alpha_threshold: 128,
            cell: CellMode::Detail,
            dominant_threshold: DEFAULT_DOMINANT_THRESHOLD,
            compare: false,
        }
    }
}

impl Config {
    /// Front-end-agnostic validation (clap already enforces the CLI's own messages).
    pub fn validate(&self) -> Result<(), String> {
        if self.size == 0 {
            return Err("size must be at least 1".into());
        }
        if let Some(pixel_size) = self.pixel_size {
            if pixel_size < 1.0 || !pixel_size.is_finite() {
                return Err("pixel_size must be a finite number >= 1.0".into());
            }
        }
        if !(1..=512).contains(&self.colors) {
            return Err("colors must be in 1..=512".into());
        }
        if !(0.0..=1.0).contains(&self.dither_strength) || !self.dither_strength.is_finite() {
            return Err("dither_strength must be a finite number in 0.0..=1.0".into());
        }
        if self.scale == 0 {
            return Err("scale must be at least 1".into());
        }
        if !(0.0..=1.0).contains(&self.dominant_threshold) || !self.dominant_threshold.is_finite() {
            return Err("dominant_threshold must be a finite number in 0.0..=1.0".into());
        }
        Ok(())
    }
}

/// Result of a conversion. `out_w`/`out_h` are the logical pixel grid (before `scale`
/// and `compare`); `image` is the final encoded-ready buffer.
pub struct ConvertResult {
    pub image: RgbaImage,
    /// The actual sRGB palette used, luma-sorted (empty if the source is fully transparent).
    pub palette: Vec<[u8; 3]>,
    pub palette_len: usize,
    pub out_w: u32,
    pub out_h: u32,
    pub detected_pixel_size: Option<f64>,
    pub grid_phase: Option<(u32, u32)>,
}

/// Estimate source-pixels-per-output-pixel from image structure, for the GUI to show
/// at upload time (mirrors what `auto_pixel_size` uses internally).
pub fn detect_pixel_size_of(src: &RgbaImage) -> Option<f64> {
    detect_pixel_size(src)
}

/// Run the full pipeline: source RGBA → deliberate pixel-art RGBA.
pub fn convert(src: &RgbaImage, cfg: &Config) -> Result<ConvertResult, String> {
    cfg.validate()?;
    let (src_w, src_h) = src.dimensions();
    let grid = target_grid(src, cfg);
    let fixed_palette = resolve_palette(&cfg.palette)?;

    let linear = rgba8_to_linear(src);
    let small = if let Some(sampling) = grid.sampling {
        downsample_grid_with_dominant_threshold(
            &linear,
            src_w as usize,
            src_h as usize,
            grid.out_w as usize,
            grid.out_h as usize,
            sampling,
            cfg.cell,
            cfg.dominant_threshold,
        )
    } else {
        downsample_with_dominant_threshold(
            &linear,
            src_w as usize,
            src_h as usize,
            grid.out_w as usize,
            grid.out_h as usize,
            cfg.cell,
            cfg.dominant_threshold,
        )
    };
    let options = QuantizeOptions {
        colors: cfg.colors,
        alpha_threshold: cfg.alpha_threshold,
        fixed_palette: fixed_palette.as_deref(),
        dither: cfg.dither,
        dither_strength: cfg.dither_strength,
    };
    let (pixel_art, palette) = quantize_to_rgba8(&small, grid.out_w, grid.out_h, options);
    let palette_len = palette.len();
    let scaled = if cfg.scale == 1 {
        pixel_art
    } else {
        scale_nearest(&pixel_art, cfg.scale)
    };
    let image = if cfg.compare {
        compare_sheet(src, &scaled)
    } else {
        scaled
    };

    Ok(ConvertResult {
        image,
        palette,
        palette_len,
        out_w: grid.out_w,
        out_h: grid.out_h,
        detected_pixel_size: grid.detected_pixel_size,
        grid_phase: grid.phase,
    })
}

/// Resolve a palette choice into Oklab entries, or `None` for adaptive.
fn resolve_palette(choice: &PaletteChoice) -> Result<Option<Vec<[f32; 3]>>, String> {
    let srgb = match choice {
        PaletteChoice::Adaptive => return Ok(None),
        PaletteChoice::Builtin(name) => palettes::builtin(name)
            .map(|p| p.to_vec())
            .ok_or_else(|| format!("unknown built-in palette {name:?}"))?,
        PaletteChoice::HexList(text) => palettes::parse_hex_list(text)?,
    };
    Ok(Some(
        srgb.into_iter()
            .map(|[r, g, b]| srgb8_to_oklab(r, g, b))
            .collect(),
    ))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GridPlan {
    pub out_w: u32,
    pub out_h: u32,
    pub detected_pixel_size: Option<f64>,
    pub phase: Option<(u32, u32)>,
    pub sampling: Option<SamplingGrid>,
}

pub(crate) fn target_grid(src: &RgbaImage, cfg: &Config) -> GridPlan {
    let (src_w, src_h) = src.dimensions();
    if let Some(pixel_size) = cfg.pixel_size {
        grid_plan_from_pixel_size(src, pixel_size, None, cfg.snap_grid)
    } else if cfg.auto_pixel_size {
        if let Some(pixel_size) = detect_pixel_size(src) {
            grid_plan_from_pixel_size(src, pixel_size, Some(pixel_size), cfg.snap_grid)
        } else {
            let size = target_size(src_w, src_h, cfg.size);
            GridPlan {
                out_w: size.0,
                out_h: size.1,
                detected_pixel_size: None,
                phase: None,
                sampling: None,
            }
        }
    } else {
        let size = target_size(src_w, src_h, cfg.size);
        GridPlan {
            out_w: size.0,
            out_h: size.1,
            detected_pixel_size: None,
            phase: None,
            sampling: None,
        }
    }
}

fn grid_plan_from_pixel_size(
    src: &RgbaImage,
    pixel_size: f64,
    detected_pixel_size: Option<f64>,
    snap_grid: bool,
) -> GridPlan {
    let (src_w, src_h) = src.dimensions();
    if snap_grid {
        if let Some(phase) = detect_grid_phase(src, pixel_size) {
            let out_w = snapped_axis_size(src_w, pixel_size, phase.0).max(1);
            let out_h = snapped_axis_size(src_h, pixel_size, phase.1).max(1);
            return GridPlan {
                out_w,
                out_h,
                detected_pixel_size,
                phase: Some(phase),
                sampling: Some(SamplingGrid {
                    origin_x: phase.0 as f64,
                    origin_y: phase.1 as f64,
                    cell_w: pixel_size,
                    cell_h: pixel_size,
                }),
            };
        }
    }

    let size = target_size_from_pixel_size(src_w, src_h, pixel_size);
    GridPlan {
        out_w: size.0,
        out_h: size.1,
        detected_pixel_size,
        phase: None,
        sampling: None,
    }
}

fn snapped_axis_size(src: u32, pixel_size: f64, phase: u32) -> u32 {
    if phase >= src {
        return 1;
    }
    (((src - phase) as f64 / pixel_size).floor() as u32).clamp(1, src)
}

pub(crate) fn target_size(src_w: u32, src_h: u32, requested_long: u32) -> (u32, u32) {
    let long = src_w.max(src_h);
    let target_long = requested_long.min(long).max(1);
    if src_w >= src_h {
        let h = ((src_h as f64 * target_long as f64 / src_w as f64).round() as u32).max(1);
        (target_long, h.min(src_h))
    } else {
        let w = ((src_w as f64 * target_long as f64 / src_h as f64).round() as u32).max(1);
        (w.min(src_w), target_long)
    }
}

pub(crate) fn target_size_from_pixel_size(src_w: u32, src_h: u32, pixel_size: f64) -> (u32, u32) {
    let w = ((src_w as f64 / pixel_size).round() as u32).clamp(1, src_w);
    let h = ((src_h as f64 / pixel_size).round() as u32).clamp(1, src_h);
    (w, h)
}

pub(crate) fn detect_pixel_size(src: &RgbaImage) -> Option<f64> {
    if let Some(pixel_size) = detect_pixel_size_from_runs(src) {
        return Some(pixel_size);
    }

    let (cols, rows) = edge_profiles(src);
    let sx = estimate_profile_step(&cols);
    let sy = estimate_profile_step(&rows);
    let detected = match (sx, sy) {
        (Some(x), Some(y)) => {
            let ratio = x.max(y) / x.min(y);
            if ratio <= 1.8 {
                Some((x + y) * 0.5)
            } else {
                Some(x.min(y))
            }
        }
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }?;

    let upper = (src.width().min(src.height()) as f64 / 2.0).clamp(1.0, MAX_AUTO_PIXEL_SIZE);
    Some(detected.clamp(1.0, upper))
}

pub(crate) fn detect_grid_phase(src: &RgbaImage, pixel_size: f64) -> Option<(u32, u32)> {
    let step = pixel_size.round();
    if step < 2.0 || (pixel_size - step).abs() > 0.2 {
        return None;
    }
    let step = step as usize;
    let (cols, rows) = edge_profiles(src);
    let x = (best_phase_for_step(&cols, step)? + 1) % step;
    let y = (best_phase_for_step(&rows, step)? + 1) % step;
    Some((x as u32, y as u32))
}

fn best_phase_for_step(profile: &[f64], step: usize) -> Option<usize> {
    if step < 2 || profile.len() <= step {
        return None;
    }

    let total: f64 = profile.iter().sum();
    if total <= f64::EPSILON {
        return None;
    }

    let mut best_phase = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for phase in 0..step {
        let mut score = 0.0;
        let mut pos = phase;
        while pos < profile.len() {
            score += profile[pos];
            pos += step;
        }
        if score > best_score {
            best_score = score;
            best_phase = phase;
        }
    }

    if best_score <= f64::EPSILON {
        return None;
    }
    Some(best_phase)
}

const MAX_AUTO_PIXEL_SIZE: f64 = 32.0;
const MIN_AUTO_PIXEL_SIZE: usize = 3;

fn detect_pixel_size_from_runs(src: &RgbaImage) -> Option<f64> {
    let max = MAX_AUTO_PIXEL_SIZE as usize;
    let mut run_lengths = Vec::new();

    for y in 0..src.height() {
        let mut current = coarse_key(src.get_pixel(0, y));
        let mut len = 1usize;
        for x in 1..src.width() {
            let key = coarse_key(src.get_pixel(x, y));
            if key == current {
                len += 1;
            } else {
                push_run_length(&mut run_lengths, len, max);
                current = key;
                len = 1;
            }
        }
        push_run_length(&mut run_lengths, len, max);
    }

    for x in 0..src.width() {
        let mut current = coarse_key(src.get_pixel(x, 0));
        let mut len = 1usize;
        for y in 1..src.height() {
            let key = coarse_key(src.get_pixel(x, y));
            if key == current {
                len += 1;
            } else {
                push_run_length(&mut run_lengths, len, max);
                current = key;
                len = 1;
            }
        }
        push_run_length(&mut run_lengths, len, max);
    }

    if run_lengths.len() < 64 {
        return None;
    }

    let mut best = None;
    let mut best_score = 0.0;
    for candidate in MIN_AUTO_PIXEL_SIZE..=max {
        let mut score = 0.0;
        for &len in &run_lengths {
            let rem = len % candidate;
            let near_multiple = rem <= 1 || candidate - rem <= 1;
            if near_multiple {
                score += len.min(candidate * 3) as f64;
                if (len as isize - candidate as isize).abs() <= 1 {
                    score += candidate as f64 * 2.0;
                }
            }
        }
        score *= (candidate as f64).sqrt();
        if score > best_score {
            best_score = score;
            best = Some(candidate as f64);
        }
    }

    let total: usize = run_lengths.iter().sum();
    let confidence = best_score / total.max(1) as f64;
    if confidence >= 0.7 {
        best
    } else {
        None
    }
}

fn coarse_key(px: &Rgba<u8>) -> u16 {
    if px[3] < 128 {
        return 0;
    }
    let r = (px[0] >> 5) as u16;
    let g = (px[1] >> 5) as u16;
    let b = (px[2] >> 5) as u16;
    1 + (r << 6) + (g << 3) + b
}

fn push_run_length(run_lengths: &mut Vec<usize>, len: usize, max: usize) {
    if (MIN_AUTO_PIXEL_SIZE..=max).contains(&len) {
        run_lengths.push(len);
    }
}

fn edge_profiles(src: &RgbaImage) -> (Vec<f64>, Vec<f64>) {
    let (w, h) = src.dimensions();
    let mut cols = vec![0.0; w as usize];
    let mut rows = vec![0.0; h as usize];

    if w < 3 || h < 3 {
        return (cols, rows);
    }

    for y in 0..h {
        for x in 1..w - 1 {
            cols[x as usize] +=
                (luma(src.get_pixel(x + 1, y)) - luma(src.get_pixel(x - 1, y))).abs();
        }
    }
    for y in 1..h - 1 {
        for x in 0..w {
            rows[y as usize] +=
                (luma(src.get_pixel(x, y + 1)) - luma(src.get_pixel(x, y - 1))).abs();
        }
    }

    (cols, rows)
}

fn luma(px: &Rgba<u8>) -> f64 {
    if px[3] == 0 {
        return 0.0;
    }
    (0.2126 * px[0] as f64 + 0.7152 * px[1] as f64 + 0.0722 * px[2] as f64) * (px[3] as f64 / 255.0)
}

fn estimate_profile_step(profile: &[f64]) -> Option<f64> {
    if profile.len() < 6 {
        return None;
    }

    let max = profile.iter().copied().fold(0.0, f64::max);
    if max <= 0.0 {
        return None;
    }
    let mean = profile.iter().sum::<f64>() / profile.len() as f64;
    let variance = profile
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / profile.len() as f64;
    let std = variance.sqrt();
    let threshold = (mean + std * 0.35).max(max * 0.18);

    let mut peaks = Vec::new();
    for i in 1..profile.len() - 1 {
        if profile[i] >= threshold && profile[i] > profile[i - 1] && profile[i] >= profile[i + 1] {
            if peaks.last().is_none_or(|&last| i - last >= 2) {
                peaks.push(i);
            } else if let Some(last) = peaks.last_mut() {
                if profile[i] > profile[*last] {
                    *last = i;
                }
            }
        }
    }

    if peaks.len() >= 3 {
        let mut diffs: Vec<f64> = peaks
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as f64)
            .filter(|&d| (3.0..=MAX_AUTO_PIXEL_SIZE).contains(&d))
            .collect();
        if !diffs.is_empty() {
            diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = diffs[diffs.len() / 2];
            if periodic_score(profile, median) >= 0.25 {
                return Some(median);
            }
        }
    }

    estimate_profile_step_by_periodicity(profile, std)
}

fn estimate_profile_step_by_periodicity(profile: &[f64], std: f64) -> Option<f64> {
    if std <= f64::EPSILON {
        return None;
    }
    let upper = (profile.len() / 3).clamp(3, MAX_AUTO_PIXEL_SIZE as usize);
    let mut best = None;
    let mut best_score = 0.0;

    for step in 3..=upper {
        let score = periodic_score(profile, step as f64);
        if score > best_score {
            best_score = score;
            best = Some(step as f64);
        }
    }

    if best_score >= 0.25 {
        best
    } else {
        None
    }
}

fn periodic_score(profile: &[f64], step: f64) -> f64 {
    let step = step.round() as usize;
    if step < 2 || step >= profile.len() {
        return 0.0;
    }
    let mean = profile.iter().sum::<f64>() / profile.len() as f64;
    let max = profile.iter().copied().fold(0.0, f64::max);
    if max <= mean {
        return 0.0;
    }

    let mut best = 0.0;
    for phase in 0..step {
        let mut sum = 0.0;
        let mut count = 0usize;
        let mut i = phase;
        while i < profile.len() {
            sum += profile[i];
            count += 1;
            i += step;
        }
        if count > 0 {
            let avg = sum / count as f64;
            if avg > best {
                best = avg;
            }
        }
    }
    ((best - mean) / (max - mean)).clamp(0.0, 1.0)
}

fn rgba8_to_linear(src: &RgbaImage) -> Vec<f32> {
    let mut out = Vec::with_capacity(src.width() as usize * src.height() as usize * 4);
    for p in src.pixels() {
        out.push(srgb_to_linear(p[0]));
        out.push(srgb_to_linear(p[1]));
        out.push(srgb_to_linear(p[2]));
        out.push(p[3] as f32 / 255.0);
    }
    out
}

#[derive(Clone, Copy)]
struct QuantizeOptions<'a> {
    colors: usize,
    alpha_threshold: u8,
    fixed_palette: Option<&'a [[f32; 3]]>,
    dither: Dither,
    dither_strength: f32,
}

fn quantize_to_rgba8(
    linear_rgba: &[f32],
    width: u32,
    height: u32,
    options: QuantizeOptions<'_>,
) -> (RgbaImage, Vec<[u8; 3]>) {
    let threshold = options.alpha_threshold as f32 / 255.0;
    let mut samples = Vec::new();
    for px in linear_rgba.chunks_exact(4) {
        if px[3] >= threshold {
            samples.push(linear_to_oklab(px[0], px[1], px[2]));
        }
    }

    if samples.is_empty() {
        return (RgbaImage::new(width, height), Vec::new());
    }

    let palette = options
        .fixed_palette
        .map(|palette| palette.to_vec())
        .unwrap_or_else(|| build_palette(&samples, options.colors.min(samples.len()), 32));
    let palette_srgb: Vec<[u8; 3]> = palette.iter().map(|&lab| oklab_to_srgb8(lab)).collect();
    let labs: Vec<[f32; 3]> = linear_rgba
        .chunks_exact(4)
        .map(|px| linear_to_oklab(px[0], px[1], px[2]))
        .collect();
    let dithered = match options.dither {
        Dither::None => None,
        Dither::Bayer4 | Dither::Bayer8 => Some(ordered_dither(
            &labs,
            width as usize,
            &palette,
            options.dither_strength,
            0.08,
            options.dither == Dither::Bayer8,
        )),
    };
    let mut out = RgbaImage::new(width, height);

    for (i, (dst, px)) in out
        .pixels_mut()
        .zip(linear_rgba.chunks_exact(4))
        .enumerate()
    {
        if px[3] < threshold {
            *dst = Rgba([0, 0, 0, 0]);
            continue;
        }
        let idx = dithered
            .as_ref()
            .map(|indices| indices[i] as usize)
            .unwrap_or_else(|| nearest(&palette, labs[i]));
        let [r, g, b] = palette_srgb[idx];
        *dst = Rgba([r, g, b, 255]);
    }

    (out, palette_srgb)
}

fn scale_nearest(src: &RgbaImage, scale: u32) -> RgbaImage {
    let mut out = RgbaImage::new(src.width() * scale, src.height() * scale);
    for y in 0..out.height() {
        for x in 0..out.width() {
            let p = src.get_pixel(x / scale, y / scale);
            out.put_pixel(x, y, *p);
        }
    }
    out
}

fn compare_sheet(original: &RgbaImage, result: &RgbaImage) -> RgbaImage {
    let h = result.height().max(1);
    let w = ((original.width() as f64 * h as f64 / original.height() as f64).round() as u32).max(1);
    let original_resized = image::imageops::resize(original, w, h, FilterType::Triangle);
    let gap = 4;
    let mut sheet = RgbaImage::new(w + gap + result.width(), h);

    for (x, y, p) in original_resized.enumerate_pixels() {
        sheet.put_pixel(x, y, *p);
    }
    for x in w..w + gap {
        for y in 0..h {
            sheet.put_pixel(x, y, Rgba([24, 24, 24, 255]));
        }
    }
    for (x, y, p) in result.enumerate_pixels() {
        sheet.put_pixel(w + gap + x, y, *p);
    }

    sheet
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config::default()
    }

    #[test]
    fn target_size_preserves_aspect_ratio_by_long_side() {
        assert_eq!(target_size(400, 300, 64), (64, 48));
        assert_eq!(target_size(300, 400, 64), (48, 64));
    }

    #[test]
    fn pixel_size_estimates_grid_without_cropping_strays() {
        assert_eq!(target_size_from_pixel_size(80, 48, 5.0), (16, 10));
        assert_eq!(target_size_from_pixel_size(103, 77, 5.0), (21, 15));
        assert_eq!(target_size_from_pixel_size(3, 2, 99.0), (1, 1));
    }

    #[test]
    fn detects_regular_fake_pixel_grid() {
        let mut img = RgbaImage::new(60, 40);
        for y in 0..img.height() {
            for x in 0..img.width() {
                let checker = (x / 5 + y / 5) % 2 == 0;
                let color = if checker {
                    Rgba([220, 210, 80, 255])
                } else {
                    Rgba([30, 40, 80, 255])
                };
                img.put_pixel(x, y, color);
            }
        }

        let detected = detect_pixel_size(&img).unwrap();
        assert!(
            (detected - 5.0).abs() <= 1.0,
            "expected about 5, got {detected}"
        );
    }

    #[test]
    fn explicit_pixel_size_wins_over_auto_detection() {
        let mut img = RgbaImage::new(60, 40);
        for p in img.pixels_mut() {
            *p = Rgba([128, 128, 128, 255]);
        }
        let config = Config {
            pixel_size: Some(4.0),
            auto_pixel_size: true,
            ..cfg()
        };

        let grid = target_grid(&img, &config);
        assert_eq!((grid.out_w, grid.out_h), (15, 10));
        assert!(grid.detected_pixel_size.is_none());
        assert!(grid.phase.is_none());
    }

    #[test]
    fn detects_grid_phase_for_offset_pixel_boundaries() {
        let mut img = RgbaImage::new(17, 14);
        for y in 0..img.height() {
            for x in 0..img.width() {
                let checker = ((x.saturating_sub(2) / 3) + (y.saturating_sub(1) / 3)) % 2 == 0;
                let color = if checker {
                    Rgba([230, 220, 90, 255])
                } else {
                    Rgba([20, 35, 80, 255])
                };
                img.put_pixel(x, y, color);
            }
        }

        assert_eq!(detect_grid_phase(&img, 3.0), Some((2, 1)));

        let config = Config {
            pixel_size: Some(3.0),
            snap_grid: true,
            ..cfg()
        };
        let grid = target_grid(&img, &config);
        assert_eq!(grid.phase, Some((2, 1)));
        assert_eq!((grid.out_w, grid.out_h), (5, 4));
    }

    #[test]
    fn supports_more_than_256_colors_with_dither() {
        // A high-variety source at 300 colors + Bayer dither exercises the palette-index
        // path. With u8 indices, entries >=256 would wrap and the output could never hold
        // more than 256 distinct colors; u16 indices make >256 real.
        // r = x*4, g = y*4 gives 64*64 = 4096 genuinely distinct colors.
        let mut img = RgbaImage::new(64, 64);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = Rgba([(x * 4) as u8, (y * 4) as u8, 128, 255]);
        }
        let config = Config {
            size: 64,
            colors: 300,
            dither: Dither::Bayer4,
            dither_strength: 0.5,
            ..cfg()
        };
        let r = convert(&img, &config).unwrap();
        assert!(r.palette_len > 256, "palette_len was {}", r.palette_len);
        let unique: std::collections::HashSet<[u8; 4]> = r.image.pixels().map(|p| p.0).collect();
        assert!(
            unique.len() > 256,
            "only {} unique output colors",
            unique.len()
        );
    }

    #[test]
    fn convert_is_deterministic_and_grid_correct() {
        let mut img = RgbaImage::new(64, 48);
        for (i, p) in img.pixels_mut().enumerate() {
            let v = (i % 251) as u8;
            *p = Rgba([v, v.wrapping_mul(3), v.wrapping_mul(7), 255]);
        }
        let config = Config {
            size: 16,
            colors: 8,
            ..cfg()
        };
        let a = convert(&img, &config).unwrap();
        let b = convert(&img, &config).unwrap();
        assert_eq!((a.out_w, a.out_h), (16, 12));
        assert!(a.palette_len <= 8);
        assert_eq!(
            a.image.as_raw(),
            b.image.as_raw(),
            "convert must be deterministic"
        );
    }
}
