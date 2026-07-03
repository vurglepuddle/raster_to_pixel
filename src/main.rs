use std::{error::Error, fs, path::PathBuf};

use clap::{Parser, ValueEnum};
use image::{imageops::FilterType, ImageReader, Rgba, RgbaImage};
use raster_to_pixel::{
    color::{linear_to_oklab, linear_to_srgb, oklab_to_srgb8, srgb8_to_oklab},
    dither::ordered_dither,
    downsample::{downsample, CellMode},
    kmeans::{build_palette, nearest},
    palettes,
};

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Convert raster images into small, deliberate pixel-art PNGs."
)]
struct Args {
    /// Input image path.
    input: PathBuf,

    /// Output image path. Use .png for now.
    output: PathBuf,

    /// Long side of the pixel-art result.
    #[arg(long, default_value_t = 64)]
    size: u32,

    /// Estimated source pixels per output pixel. Overrides --size.
    #[arg(long)]
    pixel_size: Option<f64>,

    /// Estimate source pixels per output pixel from image edges. Ignored when --pixel-size is set.
    #[arg(long)]
    auto_pixel_size: bool,

    /// Adaptive palette size.
    #[arg(long, default_value_t = 16)]
    colors: usize,

    /// Built-in palette name (pico8, gameboy, sweetie16) or Lospec hex file path.
    #[arg(long)]
    palette: Option<String>,

    /// Ordered dithering mode.
    #[arg(long, value_enum, default_value_t = DitherArg::None)]
    dither: DitherArg,

    /// Dither strength, 0.0..1.0.
    #[arg(long, default_value_t = 0.35)]
    dither_strength: f32,

    /// Nearest-neighbor preview scale. 1 writes the raw pixel grid.
    #[arg(long, default_value_t = 1)]
    scale: u32,

    /// Alpha threshold, 0..255. Below this becomes fully transparent.
    #[arg(long, default_value_t = 128)]
    alpha_threshold: u8,

    /// Cell reduction mode used during downsampling.
    #[arg(long, value_enum, default_value_t = CellModeArg::Detail)]
    cell: CellModeArg,

    /// Write an original/result side-by-side comparison sheet.
    #[arg(long)]
    compare: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CellModeArg {
    Box,
    Median,
    Detail,
    Dominant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DitherArg {
    None,
    Bayer4,
    Bayer8,
}

impl From<CellModeArg> for CellMode {
    fn from(value: CellModeArg) -> Self {
        match value {
            CellModeArg::Box => CellMode::Box,
            CellModeArg::Median => CellMode::Median,
            CellModeArg::Detail => CellMode::Detail,
            CellModeArg::Dominant => CellMode::Dominant,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_args(&args)?;

    let src = ImageReader::open(&args.input)?.decode()?.to_rgba8();
    let (src_w, src_h) = src.dimensions();
    let (dst_w, dst_h, detected_pixel_size) = target_grid(&src, &args);
    let fixed_palette = load_fixed_palette(&args.palette)?;

    if let Some(pixel_size) = detected_pixel_size {
        eprintln!("auto pixel size: {:.2} source px", pixel_size);
    } else if args.pixel_size.is_none() && args.size > src_w.max(src_h) {
        eprintln!(
            "requested --size {} exceeds source long side {}; using {}x{}",
            args.size,
            src_w.max(src_h),
            dst_w,
            dst_h
        );
    }

    let linear = rgba8_to_linear(&src);
    let small = downsample(
        &linear,
        src_w as usize,
        src_h as usize,
        dst_w as usize,
        dst_h as usize,
        args.cell.into(),
    );
    let quantize = QuantizeOptions {
        colors: args.colors,
        alpha_threshold: args.alpha_threshold,
        fixed_palette: fixed_palette.as_deref(),
        dither: args.dither,
        dither_strength: args.dither_strength,
    };
    let (pixel_art, palette_len) = quantize_to_rgba8(&small, dst_w, dst_h, quantize);
    let result = if args.scale == 1 {
        pixel_art
    } else {
        scale_nearest(&pixel_art, args.scale)
    };
    let output = if args.compare {
        compare_sheet(&src, &result)
    } else {
        result
    };

    output.save(&args.output)?;
    eprintln!(
        "wrote {} ({}x{}, {} colors, scale x{})",
        args.output.display(),
        output.width(),
        output.height(),
        palette_len,
        args.scale
    );
    Ok(())
}

fn validate_args(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.size == 0 {
        return Err("--size must be at least 1".into());
    }
    if let Some(pixel_size) = args.pixel_size {
        if pixel_size < 1.0 || !pixel_size.is_finite() {
            return Err("--pixel-size must be a finite number >= 1.0".into());
        }
    }
    if !(1..=256).contains(&args.colors) {
        return Err("--colors must be in 1..=256".into());
    }
    if !(0.0..=1.0).contains(&args.dither_strength) || !args.dither_strength.is_finite() {
        return Err("--dither-strength must be a finite number in 0.0..=1.0".into());
    }
    if args.scale == 0 {
        return Err("--scale must be at least 1".into());
    }
    Ok(())
}

fn load_fixed_palette(choice: &Option<String>) -> Result<Option<Vec<[f32; 3]>>, Box<dyn Error>> {
    let Some(choice) = choice else {
        return Ok(None);
    };

    let palette = if let Some(builtin) = palettes::builtin(choice) {
        builtin.to_vec()
    } else {
        let text = fs::read_to_string(choice)
            .map_err(|e| format!("failed to read palette {choice:?}: {e}"))?;
        palettes::parse_hex_list(&text)
            .map_err(|e| format!("failed to parse palette {choice:?}: {e}"))?
    };

    Ok(Some(
        palette
            .into_iter()
            .map(|[r, g, b]| srgb8_to_oklab(r, g, b))
            .collect(),
    ))
}

fn target_grid(src: &RgbaImage, args: &Args) -> (u32, u32, Option<f64>) {
    let (src_w, src_h) = src.dimensions();
    if let Some(pixel_size) = args.pixel_size {
        let size = target_size_from_pixel_size(src_w, src_h, pixel_size);
        (size.0, size.1, None)
    } else if args.auto_pixel_size {
        if let Some(pixel_size) = detect_pixel_size(src) {
            let size = target_size_from_pixel_size(src_w, src_h, pixel_size);
            (size.0, size.1, Some(pixel_size))
        } else {
            let size = target_size(src_w, src_h, args.size);
            (size.0, size.1, None)
        }
    } else {
        let size = target_size(src_w, src_h, args.size);
        (size.0, size.1, None)
    }
}

fn target_size(src_w: u32, src_h: u32, requested_long: u32) -> (u32, u32) {
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

fn target_size_from_pixel_size(src_w: u32, src_h: u32, pixel_size: f64) -> (u32, u32) {
    let w = ((src_w as f64 / pixel_size).round() as u32).clamp(1, src_w);
    let h = ((src_h as f64 / pixel_size).round() as u32).clamp(1, src_h);
    (w, h)
}

fn detect_pixel_size(src: &RgbaImage) -> Option<f64> {
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
        out.push(raster_to_pixel::color::srgb_to_linear(p[0]));
        out.push(raster_to_pixel::color::srgb_to_linear(p[1]));
        out.push(raster_to_pixel::color::srgb_to_linear(p[2]));
        out.push(p[3] as f32 / 255.0);
    }
    out
}

#[derive(Clone, Copy)]
struct QuantizeOptions<'a> {
    colors: usize,
    alpha_threshold: u8,
    fixed_palette: Option<&'a [[f32; 3]]>,
    dither: DitherArg,
    dither_strength: f32,
}

fn quantize_to_rgba8(
    linear_rgba: &[f32],
    width: u32,
    height: u32,
    options: QuantizeOptions<'_>,
) -> (RgbaImage, usize) {
    let threshold = options.alpha_threshold as f32 / 255.0;
    let mut samples = Vec::new();
    for px in linear_rgba.chunks_exact(4) {
        if px[3] >= threshold {
            samples.push(linear_to_oklab(px[0], px[1], px[2]));
        }
    }

    if samples.is_empty() {
        return (RgbaImage::new(width, height), 0);
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
        DitherArg::None => None,
        DitherArg::Bayer4 | DitherArg::Bayer8 => Some(ordered_dither(
            &labs,
            width as usize,
            &palette,
            options.dither_strength,
            0.08,
            options.dither == DitherArg::Bayer8,
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

    (out, palette.len())
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

#[allow(dead_code)]
fn linear_rgba_to_rgba8(linear_rgba: &[f32], width: u32, height: u32) -> RgbaImage {
    let mut out = RgbaImage::new(width, height);
    for (dst, px) in out.pixels_mut().zip(linear_rgba.chunks_exact(4)) {
        *dst = Rgba([
            linear_to_srgb(px[0]),
            linear_to_srgb(px[1]),
            linear_to_srgb(px[2]),
            (px[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        ]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let args = Args {
            input: PathBuf::from("in.png"),
            output: PathBuf::from("out.png"),
            size: 64,
            pixel_size: Some(4.0),
            auto_pixel_size: true,
            colors: 16,
            palette: None,
            dither: DitherArg::None,
            dither_strength: 0.35,
            scale: 1,
            alpha_threshold: 128,
            cell: CellModeArg::Detail,
            compare: false,
        };

        let (w, h, detected) = target_grid(&img, &args);
        assert_eq!((w, h), (15, 10));
        assert!(detected.is_none());
    }
}
