use std::{error::Error, fs, path::PathBuf};

use clap::{Parser, ValueEnum};
use image::ImageReader;
use raster_to_pixel::{
    downsample::CellMode,
    palettes,
    pipeline::{self, Config, Dither, PaletteChoice},
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

    /// Disable edge-based grid phase snapping for --pixel-size/--auto-pixel-size.
    #[arg(long)]
    no_snap_grid: bool,

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

    /// Minimum winning-bucket coverage for dominant/detail cells before falling back to mean.
    #[arg(long, default_value_t = 0.25)]
    dominant_threshold: f32,

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

impl From<DitherArg> for Dither {
    fn from(value: DitherArg) -> Self {
        match value {
            DitherArg::None => Dither::None,
            DitherArg::Bayer4 => Dither::Bayer4,
            DitherArg::Bayer8 => Dither::Bayer8,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_args(&args)?;

    let src = ImageReader::open(&args.input)?.decode()?.to_rgba8();
    let (src_w, src_h) = src.dimensions();
    let cfg = build_config(&args)?;

    let result = pipeline::convert(&src, &cfg)?;

    if let Some(pixel_size) = result.detected_pixel_size {
        eprintln!("auto pixel size: {:.2} source px", pixel_size);
    } else if args.pixel_size.is_none() && args.size > src_w.max(src_h) {
        eprintln!(
            "requested --size {} exceeds source long side {}; using {}x{}",
            args.size,
            src_w.max(src_h),
            result.out_w,
            result.out_h
        );
    }
    if let Some((x, y)) = result.grid_phase {
        eprintln!("grid phase: {x},{y}");
    }

    result.image.save(&args.output)?;
    eprintln!(
        "wrote {} ({}x{}, {} colors, scale x{})",
        args.output.display(),
        result.image.width(),
        result.image.height(),
        result.palette_len,
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
    if !(1..=512).contains(&args.colors) {
        return Err("--colors must be in 1..=512".into());
    }
    if !(0.0..=1.0).contains(&args.dither_strength) || !args.dither_strength.is_finite() {
        return Err("--dither-strength must be a finite number in 0.0..=1.0".into());
    }
    if args.scale == 0 {
        return Err("--scale must be at least 1".into());
    }
    if !(0.0..=1.0).contains(&args.dominant_threshold) || !args.dominant_threshold.is_finite() {
        return Err("--dominant-threshold must be a finite number in 0.0..=1.0".into());
    }
    Ok(())
}

/// Turn parsed CLI args into a pipeline `Config`, reading any palette file from disk.
fn build_config(args: &Args) -> Result<Config, Box<dyn Error>> {
    let palette = match &args.palette {
        None => PaletteChoice::Adaptive,
        Some(choice) if palettes::builtin(choice).is_some() => {
            PaletteChoice::Builtin(choice.clone())
        }
        Some(path) => {
            let text = fs::read_to_string(path)
                .map_err(|e| format!("failed to read palette {path:?}: {e}"))?;
            PaletteChoice::HexList(text)
        }
    };

    Ok(Config {
        size: args.size,
        pixel_size: args.pixel_size,
        auto_pixel_size: args.auto_pixel_size,
        snap_grid: !args.no_snap_grid,
        colors: args.colors,
        palette,
        dither: args.dither.into(),
        dither_strength: args.dither_strength,
        scale: args.scale,
        alpha_threshold: args.alpha_threshold,
        cell: args.cell.into(),
        dominant_threshold: args.dominant_threshold,
        compare: args.compare,
    })
}
