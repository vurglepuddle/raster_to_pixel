use std::{error::Error, fs, path::PathBuf};

use clap::{Parser, ValueEnum};
use image::ImageReader;
use raster_to_pixel::{
    alpha::AlphaMode,
    downsample::CellMode,
    morphology::CleanupPreset,
    outline::OutlineMode,
    palettes,
    pipeline::{
        self, Config, Dither, PaletteChoice, Quantizer, DEFAULT_BG_TOLERANCE,
        DEFAULT_HIGHLIGHT_COLLAPSE, DEFAULT_SHADOW_COLLAPSE,
    },
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

    /// Adaptive palette highlight cleanup. 0 disables; larger values collapse deeper highlights.
    #[arg(long, default_value_t = DEFAULT_HIGHLIGHT_COLLAPSE)]
    highlight_collapse: f32,

    /// Adaptive palette shadow cleanup. 0 disables; larger values collapse deeper shadows.
    #[arg(long, default_value_t = DEFAULT_SHADOW_COLLAPSE)]
    shadow_collapse: f32,

    /// Source alpha preparation before conversion.
    #[arg(long, value_enum, default_value_t = AlphaModeArg::Preserve)]
    alpha_mode: AlphaModeArg,

    /// Color tolerance for background-fill/color-key, 0..1 of the RGB diagonal.
    #[arg(long, default_value_t = DEFAULT_BG_TOLERANCE)]
    bg_tolerance: f32,

    /// Key color (RRGGBB or #RRGGBB) for --alpha-mode color-key.
    #[arg(long)]
    color_key: Option<String>,

    /// Post-quantize morphology cleanup preset.
    #[arg(long, value_enum, default_value_t = CleanupArg::None)]
    cleanup: CleanupArg,

    /// Let cleanup remove isolated single pixels even when they repeat nearby.
    #[arg(long)]
    no_protect_details: bool,

    /// Pick the adaptive color count automatically (overrides --colors).
    #[arg(long)]
    auto_colors: bool,

    /// Manual grid phase override (x, source pixels) for pixel-size modes.
    #[arg(long)]
    phase_x: Option<u32>,

    /// Manual grid phase override (y, source pixels) for pixel-size modes.
    #[arg(long)]
    phase_y: Option<u32>,

    /// Adaptive palette construction algorithm.
    #[arg(long, value_enum, default_value_t = QuantizerArg::Kmeans)]
    quantizer: QuantizerArg,

    /// Merge adaptive palette entries closer than this Oklab distance. 0 disables.
    #[arg(long, default_value_t = 0.0)]
    palette_merge: f32,

    /// Contrast-expansion radius (source px) protecting tiny details. 0 disables, max 4.
    #[arg(long, default_value_t = 0)]
    contrast_expansion: u32,

    /// Outline cleanup on the output grid.
    #[arg(long, value_enum, default_value_t = OutlineArg::None)]
    outline: OutlineArg,

    /// Write grid/palette/cleanup diagnostics as JSON to this path.
    #[arg(long)]
    debug_json: Option<PathBuf>,

    /// Write the source with the sampling grid drawn on it to this path.
    #[arg(long)]
    debug_grid: Option<PathBuf>,

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum AlphaModeArg {
    Preserve,
    Binary,
    BackgroundFill,
    ColorKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CleanupArg {
    None,
    Conservative,
    Balanced,
    Aggressive,
}

impl From<AlphaModeArg> for AlphaMode {
    fn from(value: AlphaModeArg) -> Self {
        match value {
            AlphaModeArg::Preserve => AlphaMode::Preserve,
            AlphaModeArg::Binary => AlphaMode::Binary,
            AlphaModeArg::BackgroundFill => AlphaMode::BackgroundFill,
            AlphaModeArg::ColorKey => AlphaMode::ColorKey,
        }
    }
}

impl From<CleanupArg> for CleanupPreset {
    fn from(value: CleanupArg) -> Self {
        match value {
            CleanupArg::None => CleanupPreset::None,
            CleanupArg::Conservative => CleanupPreset::Conservative,
            CleanupArg::Balanced => CleanupPreset::Balanced,
            CleanupArg::Aggressive => CleanupPreset::Aggressive,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum QuantizerArg {
    Kmeans,
    Wu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum OutlineArg {
    None,
    Repair,
    Enforce,
}

impl From<QuantizerArg> for Quantizer {
    fn from(value: QuantizerArg) -> Self {
        match value {
            QuantizerArg::Kmeans => Quantizer::KMeans,
            QuantizerArg::Wu => Quantizer::Wu,
        }
    }
}

impl From<OutlineArg> for OutlineMode {
    fn from(value: OutlineArg) -> Self {
        match value {
            OutlineArg::None => OutlineMode::None,
            OutlineArg::Repair => OutlineMode::Repair,
            OutlineArg::Enforce => OutlineMode::Enforce,
        }
    }
}

/// Parse a single "RRGGBB"/"#RRGGBB" color.
fn parse_hex_color(text: &str) -> Result<[u8; 3], String> {
    let t = text.trim().trim_start_matches('#');
    if t.len() != 6 || !t.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("invalid color {text:?}; expected RRGGBB"));
    }
    let v = u32::from_str_radix(t, 16).map_err(|e| e.to_string())?;
    Ok([(v >> 16) as u8, (v >> 8) as u8, v as u8])
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
        match result.phase_confidence {
            Some(conf) => eprintln!("grid phase: {x},{y} (confidence {conf:.2})"),
            None => eprintln!("grid phase: {x},{y}"),
        }
    }
    if let Some(chosen) = result.auto_colors {
        eprintln!("auto colors: {chosen}");
    }
    if result.alpha_removed > 0 {
        eprintln!(
            "alpha cleanup removed {} source pixels",
            result.alpha_removed
        );
    }
    if result.cleanup.total() > 0 {
        let c = &result.cleanup;
        eprintln!(
            "cleanup: {} pinholes filled, {} halo, {} jaggy, {} orphan pixels removed",
            c.pinholes_filled, c.halo_removed, c.jaggies_removed, c.orphans_removed
        );
    }
    if result.contrast_expanded > 0 {
        eprintln!(
            "contrast expansion repainted {} source pixels",
            result.contrast_expanded
        );
    }
    if let Some([r, g, b]) = result.outline.outline_color {
        eprintln!(
            "outline {r:02x}{g:02x}{b:02x}: {} edge pixels recolored",
            result.outline.recolored
        );
    }
    if let Some(path) = &args.debug_json {
        fs::write(path, result.diagnostics_json(&cfg))
            .map_err(|e| format!("failed to write {path:?}: {e}"))?;
        eprintln!("wrote diagnostics {}", path.display());
    }
    if let Some(path) = &args.debug_grid {
        pipeline::debug_grid_image(&src, &cfg)?.save(path)?;
        eprintln!("wrote grid debug image {}", path.display());
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
    if !(0.0..=1.0).contains(&args.highlight_collapse) || !args.highlight_collapse.is_finite() {
        return Err("--highlight-collapse must be a finite number in 0.0..=1.0".into());
    }
    if !(0.0..=1.0).contains(&args.shadow_collapse) || !args.shadow_collapse.is_finite() {
        return Err("--shadow-collapse must be a finite number in 0.0..=1.0".into());
    }
    if !(0.0..=1.0).contains(&args.bg_tolerance) || !args.bg_tolerance.is_finite() {
        return Err("--bg-tolerance must be a finite number in 0.0..=1.0".into());
    }
    if args.alpha_mode == AlphaModeArg::ColorKey && args.color_key.is_none() {
        return Err("--alpha-mode color-key requires --color-key RRGGBB".into());
    }
    if !(0.0..=1.0).contains(&args.palette_merge) || !args.palette_merge.is_finite() {
        return Err("--palette-merge must be a finite number in 0.0..=1.0".into());
    }
    if args.contrast_expansion > 4 {
        return Err("--contrast-expansion must be in 0..=4".into());
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

    let color_key = args.color_key.as_deref().map(parse_hex_color).transpose()?;

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
        highlight_collapse: args.highlight_collapse,
        shadow_collapse: args.shadow_collapse,
        alpha_mode: args.alpha_mode.into(),
        bg_tolerance: args.bg_tolerance,
        color_key,
        cleanup: args.cleanup.into(),
        protect_details: !args.no_protect_details,
        auto_colors: args.auto_colors,
        phase_x: args.phase_x,
        phase_y: args.phase_y,
        quantizer: args.quantizer.into(),
        palette_merge: args.palette_merge,
        contrast_expansion: args.contrast_expansion,
        outline: args.outline.into(),
        compare: args.compare,
    })
}
