use std::{error::Error, path::PathBuf};

use clap::{Parser, ValueEnum};
use image::{ImageReader, Rgba, RgbaImage};
use raster_to_pixel::{
    color::{linear_to_oklab, linear_to_srgb, oklab_to_srgb8},
    downsample::{downsample, CellMode},
    kmeans::{build_palette, nearest},
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

    /// Adaptive palette size.
    #[arg(long, default_value_t = 16)]
    colors: usize,

    /// Nearest-neighbor preview scale. 1 writes the raw pixel grid.
    #[arg(long, default_value_t = 1)]
    scale: u32,

    /// Alpha threshold, 0..255. Below this becomes fully transparent.
    #[arg(long, default_value_t = 128)]
    alpha_threshold: u8,

    /// Cell reduction mode used during downsampling.
    #[arg(long, value_enum, default_value_t = CellModeArg::Detail)]
    cell: CellModeArg,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CellModeArg {
    Box,
    Median,
    Detail,
    Dominant,
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
    let (dst_w, dst_h) = target_grid(src_w, src_h, &args);

    if args.pixel_size.is_none() && args.size > src_w.max(src_h) {
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
    let pixel_art = quantize_to_rgba8(&small, dst_w, dst_h, args.colors, args.alpha_threshold);
    let output = if args.scale == 1 {
        pixel_art
    } else {
        scale_nearest(&pixel_art, args.scale)
    };

    output.save(&args.output)?;
    eprintln!(
        "wrote {} ({}x{}, {} colors, scale x{})",
        args.output.display(),
        output.width(),
        output.height(),
        args.colors,
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
    if args.scale == 0 {
        return Err("--scale must be at least 1".into());
    }
    Ok(())
}

fn target_grid(src_w: u32, src_h: u32, args: &Args) -> (u32, u32) {
    if let Some(pixel_size) = args.pixel_size {
        target_size_from_pixel_size(src_w, src_h, pixel_size)
    } else {
        target_size(src_w, src_h, args.size)
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

fn quantize_to_rgba8(
    linear_rgba: &[f32],
    width: u32,
    height: u32,
    colors: usize,
    alpha_threshold: u8,
) -> RgbaImage {
    let threshold = alpha_threshold as f32 / 255.0;
    let mut samples = Vec::new();
    for px in linear_rgba.chunks_exact(4) {
        if px[3] >= threshold {
            samples.push(linear_to_oklab(px[0], px[1], px[2]));
        }
    }

    if samples.is_empty() {
        return RgbaImage::new(width, height);
    }

    let palette = build_palette(&samples, colors.min(samples.len()), 32);
    let palette_srgb: Vec<[u8; 3]> = palette.iter().map(|&lab| oklab_to_srgb8(lab)).collect();
    let mut out = RgbaImage::new(width, height);

    for (dst, px) in out.pixels_mut().zip(linear_rgba.chunks_exact(4)) {
        if px[3] < threshold {
            *dst = Rgba([0, 0, 0, 0]);
            continue;
        }
        let lab = linear_to_oklab(px[0], px[1], px[2]);
        let idx = nearest(&palette, lab);
        let [r, g, b] = palette_srgb[idx];
        *dst = Rgba([r, g, b, 255]);
    }

    out
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
}
