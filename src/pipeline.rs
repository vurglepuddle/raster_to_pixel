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
    alpha::{apply_alpha_mode, AlphaMode},
    color::{linear_to_oklab, linear_to_srgb, oklab_to_srgb8, srgb8_to_oklab, srgb_to_linear},
    dither::ordered_dither,
    downsample::{
        downsample_grid_with_dominant_threshold, downsample_with_dominant_threshold, CellMode,
        SamplingGrid, DEFAULT_DOMINANT_THRESHOLD,
    },
    enhance::expand_contrast,
    kmeans::{build_palette, merge_close_entries, nearest},
    morphology::{self, CleanupPreset, CleanupStats},
    outline::{apply_outline, OutlineMode, OutlineStats},
    palettes,
    wu::build_palette_wu,
};

/// Adaptive palette construction algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantizer {
    /// Median-cut init + Lloyd k-means (project default).
    KMeans,
    /// Wu 1992 moment quantization on an Oklab lattice.
    Wu,
}

pub const DEFAULT_HIGHLIGHT_COLLAPSE: f32 = 0.03;
pub const DEFAULT_SHADOW_COLLAPSE: f32 = 0.16;
pub const DEFAULT_BG_TOLERANCE: f32 = 0.10;

/// Presets `auto_colors` snaps to (smallest preset >= significant buckets).
pub const AUTO_COLOR_PRESETS: [usize; 5] = [16, 32, 64, 128, 256];

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
    /// Adaptive-palette highlight cleanup. 0.0 disables; larger values collapse deeper highlights.
    pub highlight_collapse: f32,
    /// Adaptive-palette shadow cleanup. 0.0 disables; larger values collapse deeper shadows.
    pub shadow_collapse: f32,
    /// Source alpha preparation (binary, background flood fill, color key).
    pub alpha_mode: AlphaMode,
    /// Color tolerance (0..=1 of the RGB cube diagonal) for `BackgroundFill`/`ColorKey`.
    pub bg_tolerance: f32,
    /// Key color for `AlphaMode::ColorKey` (required by that mode).
    pub color_key: Option<[u8; 3]>,
    /// Post-quantize morphology cleanup preset. `None` disables.
    pub cleanup: CleanupPreset,
    /// Keep isolated single pixels that repeat nearby (dot patterns, stars).
    pub protect_details: bool,
    /// Pick the adaptive color count automatically (ignores `colors`; adaptive palettes only).
    pub auto_colors: bool,
    /// Manual grid phase override for pixel-size modes (x axis, source pixels).
    pub phase_x: Option<u32>,
    /// Manual grid phase override for pixel-size modes (y axis, source pixels).
    pub phase_y: Option<u32>,
    /// Adaptive palette construction algorithm.
    pub quantizer: Quantizer,
    /// Merge adaptive palette entries closer than this Oklab distance. 0 disables.
    pub palette_merge: f32,
    /// Contrast-expansion radius in source pixels. 0 disables; max 4.
    pub contrast_expansion: u32,
    /// Post-quantize outline repair/enforcement.
    pub outline: OutlineMode,
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
            highlight_collapse: DEFAULT_HIGHLIGHT_COLLAPSE,
            shadow_collapse: DEFAULT_SHADOW_COLLAPSE,
            alpha_mode: AlphaMode::Preserve,
            bg_tolerance: DEFAULT_BG_TOLERANCE,
            color_key: None,
            cleanup: CleanupPreset::None,
            protect_details: true,
            auto_colors: false,
            phase_x: None,
            phase_y: None,
            quantizer: Quantizer::KMeans,
            palette_merge: 0.0,
            contrast_expansion: 0,
            outline: OutlineMode::None,
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
        if !(0.0..=1.0).contains(&self.highlight_collapse) || !self.highlight_collapse.is_finite() {
            return Err("highlight_collapse must be a finite number in 0.0..=1.0".into());
        }
        if !(0.0..=1.0).contains(&self.shadow_collapse) || !self.shadow_collapse.is_finite() {
            return Err("shadow_collapse must be a finite number in 0.0..=1.0".into());
        }
        if !(0.0..=1.0).contains(&self.bg_tolerance) || !self.bg_tolerance.is_finite() {
            return Err("bg_tolerance must be a finite number in 0.0..=1.0".into());
        }
        if self.alpha_mode == AlphaMode::ColorKey && self.color_key.is_none() {
            return Err("alpha_mode color-key requires a color_key".into());
        }
        if !(0.0..=1.0).contains(&self.palette_merge) || !self.palette_merge.is_finite() {
            return Err("palette_merge must be a finite number in 0.0..=1.0".into());
        }
        if self.contrast_expansion > 4 {
            return Err("contrast_expansion must be in 0..=4".into());
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
    pub src_w: u32,
    pub src_h: u32,
    pub out_w: u32,
    pub out_h: u32,
    pub detected_pixel_size: Option<f64>,
    pub grid_phase: Option<(u32, u32)>,
    /// Snap-grid phase confidence in 0..=1 (1 = all edge energy on one phase).
    pub phase_confidence: Option<f32>,
    /// The color count `auto_colors` chose (adaptive palettes only).
    pub auto_colors: Option<usize>,
    /// Source pixels made transparent by the alpha pre-pass.
    pub alpha_removed: usize,
    /// What the morphology cleanup preset changed on the output grid.
    pub cleanup: CleanupStats,
    /// Source pixels repainted by the contrast-expansion pre-pass.
    pub contrast_expanded: usize,
    /// What the outline pass did on the output grid.
    pub outline: OutlineStats,
}

impl ConvertResult {
    /// Hand-rolled diagnostics JSON (no serde): grid decisions, confidence,
    /// auto color choice, and cleanup counts, for `--debug-json` and the GUI.
    pub fn diagnostics_json(&self, cfg: &Config) -> String {
        fn opt_f64(v: Option<f64>) -> String {
            v.map_or("null".into(), |v| format!("{v:.4}"))
        }
        fn opt_f32(v: Option<f32>) -> String {
            v.map_or("null".into(), |v| format!("{v:.4}"))
        }
        fn opt_usize(v: Option<usize>) -> String {
            v.map_or("null".into(), |v| v.to_string())
        }
        let (phase_x, phase_y) = match self.grid_phase {
            Some((x, y)) => (x.to_string(), y.to_string()),
            None => ("null".into(), "null".into()),
        };
        format!(
            concat!(
                "{{\n",
                "  \"srcWidth\": {}, \"srcHeight\": {},\n",
                "  \"outWidth\": {}, \"outHeight\": {},\n",
                "  \"requestedPixelSize\": {}, \"detectedPixelSize\": {},\n",
                "  \"snapGrid\": {}, \"gridPhaseX\": {}, \"gridPhaseY\": {}, \"phaseConfidence\": {},\n",
                "  \"paletteLen\": {}, \"autoColors\": {},\n",
                "  \"cellMode\": \"{:?}\", \"alphaMode\": \"{:?}\", \"alphaRemoved\": {},\n",
                "  \"cleanupPreset\": \"{:?}\", \"pinholesFilled\": {}, \"haloRemoved\": {}, ",
                "\"jaggiesRemoved\": {}, \"orphansRemoved\": {},\n",
                "  \"quantizer\": \"{:?}\", \"paletteMerge\": {:.4},\n",
                "  \"contrastExpansion\": {}, \"contrastExpanded\": {},\n",
                "  \"outlineMode\": \"{:?}\", \"outlineRecolored\": {}, \"outlineColor\": {}\n",
                "}}\n"
            ),
            self.src_w,
            self.src_h,
            self.out_w,
            self.out_h,
            opt_f64(cfg.pixel_size),
            opt_f64(self.detected_pixel_size),
            cfg.snap_grid,
            phase_x,
            phase_y,
            opt_f32(self.phase_confidence),
            self.palette_len,
            opt_usize(self.auto_colors),
            cfg.cell,
            cfg.alpha_mode,
            self.alpha_removed,
            cfg.cleanup,
            self.cleanup.pinholes_filled,
            self.cleanup.halo_removed,
            self.cleanup.jaggies_removed,
            self.cleanup.orphans_removed,
            cfg.quantizer,
            cfg.palette_merge,
            cfg.contrast_expansion,
            self.contrast_expanded,
            cfg.outline,
            self.outline.recolored,
            self.outline
                .outline_color
                .map_or("null".into(), |[r, g, b]| format!("\"{r:02x}{g:02x}{b:02x}\"")),
        )
    }
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

    // Alpha pre-pass (binary / background fill / color key) runs BEFORE grid
    // detection so removed backgrounds stop voting on pixel size and phase.
    // `Preserve` skips the clone entirely — the default path stays zero-copy.
    let mut cleaned: Option<RgbaImage> = None;
    let mut alpha_removed = 0usize;
    if cfg.alpha_mode != AlphaMode::Preserve {
        let mut img = src.clone();
        let stats = apply_alpha_mode(
            &mut img,
            cfg.alpha_mode,
            cfg.alpha_threshold,
            cfg.bg_tolerance,
            cfg.color_key,
        );
        alpha_removed = stats.removed;
        cleaned = Some(img);
    }

    // Grid detection runs before contrast expansion: the stamped blobs would
    // otherwise add off-grid edge energy and could sway phase detection.
    let grid = target_grid(cleaned.as_ref().unwrap_or(src), cfg);

    let mut contrast_expanded = 0usize;
    if cfg.contrast_expansion > 0 {
        let mut img = cleaned.take().unwrap_or_else(|| src.clone());
        contrast_expanded = expand_contrast(&mut img, cfg.contrast_expansion, cfg.alpha_threshold);
        cleaned = Some(img);
    }
    let work: &RgbaImage = cleaned.as_ref().unwrap_or(src);

    let fixed_palette = resolve_palette(&cfg.palette)?;

    let linear = rgba8_to_linear(work);
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

    let auto_colors = if cfg.auto_colors && matches!(cfg.palette, PaletteChoice::Adaptive) {
        Some(auto_color_count(&small, cfg.alpha_threshold as f32 / 255.0))
    } else {
        None
    };

    let options = QuantizeOptions {
        colors: auto_colors.unwrap_or(cfg.colors),
        alpha_threshold: cfg.alpha_threshold,
        fixed_palette: fixed_palette.as_deref(),
        dither: cfg.dither,
        dither_strength: cfg.dither_strength,
        highlight_collapse: cfg.highlight_collapse,
        shadow_collapse: cfg.shadow_collapse,
        quantizer: cfg.quantizer,
        palette_merge: cfg.palette_merge,
    };
    let (mut pixel_art, palette) = quantize_to_rgba8(&small, grid.out_w, grid.out_h, options);
    let cleanup = morphology::cleanup(&mut pixel_art, cfg.cleanup, cfg.protect_details);
    let outline = apply_outline(&mut pixel_art, cfg.outline);
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
        src_w,
        src_h,
        out_w: grid.out_w,
        out_h: grid.out_h,
        detected_pixel_size: grid.detected_pixel_size,
        grid_phase: grid.phase,
        phase_confidence: grid.phase_confidence,
        auto_colors,
        alpha_removed,
        cleanup,
        contrast_expanded,
        outline,
    })
}

/// Estimate a good adaptive palette size from the downsampled grid: count
/// significant coarse RGB buckets (4 bits/channel) among opaque pixels, then
/// snap up to the nearest preset in `AUTO_COLOR_PRESETS`.
fn auto_color_count(linear_rgba: &[f32], alpha_threshold: f32) -> usize {
    let mut counts: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();
    let mut total = 0u32;
    for px in linear_rgba.chunks_exact(4) {
        if px[3] < alpha_threshold {
            continue;
        }
        total += 1;
        let key = ((linear_to_srgb(px[0]) >> 4) as u16) << 8
            | ((linear_to_srgb(px[1]) >> 4) as u16) << 4
            | (linear_to_srgb(px[2]) >> 4) as u16;
        *counts.entry(key).or_insert(0) += 1;
    }
    if total == 0 {
        return AUTO_COLOR_PRESETS[0];
    }
    // A bucket is significant at >= 0.1% of opaque pixels (always >= 1 px).
    let min_count = ((total as f64) * 0.001).ceil().max(1.0) as u32;
    let significant = counts.values().filter(|&&c| c >= min_count).count();
    for preset in AUTO_COLOR_PRESETS {
        if significant <= preset {
            return preset;
        }
    }
    AUTO_COLOR_PRESETS[AUTO_COLOR_PRESETS.len() - 1]
}

/// Draw the sampling grid the current config would use over a copy of the
/// source, for `--debug-grid`. Magenta lines mark cell boundaries, so a bad
/// pixel size or phase is visible at a glance.
pub fn debug_grid_image(src: &RgbaImage, cfg: &Config) -> Result<RgbaImage, String> {
    cfg.validate()?;
    let mut work_owned;
    let work: &RgbaImage = if cfg.alpha_mode != AlphaMode::Preserve {
        work_owned = src.clone();
        apply_alpha_mode(
            &mut work_owned,
            cfg.alpha_mode,
            cfg.alpha_threshold,
            cfg.bg_tolerance,
            cfg.color_key,
        );
        &work_owned
    } else {
        src
    };
    let plan = target_grid(work, cfg);
    let (w, h) = src.dimensions();
    let grid = plan.sampling.unwrap_or(SamplingGrid {
        origin_x: 0.0,
        origin_y: 0.0,
        cell_w: w as f64 / plan.out_w as f64,
        cell_h: h as f64 / plan.out_h as f64,
    });

    let mut out = src.clone();
    let line = Rgba([255, 0, 255, 255]);
    for i in 0..=plan.out_w as u64 {
        let x = (grid.origin_x + i as f64 * grid.cell_w).round();
        if x >= 0.0 && (x as u32) < w {
            for y in 0..h {
                out.put_pixel(x as u32, y, line);
            }
        }
    }
    for i in 0..=plan.out_h as u64 {
        let y = (grid.origin_y + i as f64 * grid.cell_h).round();
        if y >= 0.0 && (y as u32) < h {
            for x in 0..w {
                out.put_pixel(x, y as u32, line);
            }
        }
    }
    Ok(out)
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
    pub phase_confidence: Option<f32>,
    pub sampling: Option<SamplingGrid>,
}

fn unsnapped_plan(src_w: u32, src_h: u32, requested_long: u32) -> GridPlan {
    let size = target_size(src_w, src_h, requested_long);
    GridPlan {
        out_w: size.0,
        out_h: size.1,
        detected_pixel_size: None,
        phase: None,
        phase_confidence: None,
        sampling: None,
    }
}

pub(crate) fn target_grid(src: &RgbaImage, cfg: &Config) -> GridPlan {
    let (src_w, src_h) = src.dimensions();
    if let Some(pixel_size) = cfg.pixel_size {
        grid_plan_from_pixel_size(src, pixel_size, None, cfg)
    } else if cfg.auto_pixel_size {
        if let Some(pixel_size) = detect_pixel_size(src) {
            grid_plan_from_pixel_size(src, pixel_size, Some(pixel_size), cfg)
        } else {
            unsnapped_plan(src_w, src_h, cfg.size)
        }
    } else {
        unsnapped_plan(src_w, src_h, cfg.size)
    }
}

fn grid_plan_from_pixel_size(
    src: &RgbaImage,
    pixel_size: f64,
    detected_pixel_size: Option<f64>,
    cfg: &Config,
) -> GridPlan {
    let (src_w, src_h) = src.dimensions();
    let manual = cfg.phase_x.is_some() || cfg.phase_y.is_some();
    let detected = if cfg.snap_grid || manual {
        detect_grid_phase_with_confidence(src, pixel_size)
    } else {
        None
    };

    // Manual phase wins per axis; a missing axis falls back to detection,
    // then 0. Manual values are reduced modulo the cell size — a phase one
    // whole cell over is the same grid.
    let phase = if manual {
        let fallback = detected.map(|(p, _)| p).unwrap_or((0, 0));
        let step = pixel_size.ceil().max(1.0) as u32;
        Some((
            cfg.phase_x.map(|p| p % step).unwrap_or(fallback.0),
            cfg.phase_y.map(|p| p % step).unwrap_or(fallback.1),
        ))
    } else if cfg.snap_grid {
        detected.map(|(p, _)| p)
    } else {
        None
    };

    if let Some(phase) = phase {
        let out_w = snapped_axis_size(src_w, pixel_size, phase.0).max(1);
        let out_h = snapped_axis_size(src_h, pixel_size, phase.1).max(1);
        return GridPlan {
            out_w,
            out_h,
            detected_pixel_size,
            phase: Some(phase),
            phase_confidence: detected.map(|(_, c)| c),
            sampling: Some(SamplingGrid {
                origin_x: phase.0 as f64,
                origin_y: phase.1 as f64,
                cell_w: pixel_size,
                cell_h: pixel_size,
            }),
        };
    }

    let size = target_size_from_pixel_size(src_w, src_h, pixel_size);
    GridPlan {
        out_w: size.0,
        out_h: size.1,
        detected_pixel_size,
        phase: None,
        phase_confidence: None,
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

/// Detect the strongest grid phase and how decisive it was. Confidence is
/// `1 - second_best/best` per axis (0 = flat profile, ~1 = all edge energy on
/// one phase), combined by taking the weaker axis.
pub(crate) fn detect_grid_phase_with_confidence(
    src: &RgbaImage,
    pixel_size: f64,
) -> Option<((u32, u32), f32)> {
    let step = pixel_size.round();
    if step < 2.0 || (pixel_size - step).abs() > 0.2 {
        return None;
    }
    let step = step as usize;
    let (cols, rows) = edge_profiles(src);
    let sx = best_phase_for_step(&cols, step)?;
    let sy = best_phase_for_step(&rows, step)?;
    let x = (sx.phase + 1) % step;
    let y = (sy.phase + 1) % step;
    Some(((x as u32, y as u32), sx.confidence().min(sy.confidence())))
}

struct PhaseScore {
    phase: usize,
    best: f64,
    second: f64,
}

impl PhaseScore {
    fn confidence(&self) -> f32 {
        if self.best <= f64::EPSILON {
            return 0.0;
        }
        (1.0 - self.second / self.best).clamp(0.0, 1.0) as f32
    }
}

fn best_phase_for_step(profile: &[f64], step: usize) -> Option<PhaseScore> {
    if step < 2 || profile.len() <= step {
        return None;
    }

    let total: f64 = profile.iter().sum();
    if total <= f64::EPSILON {
        return None;
    }

    let mut raw = vec![0.0f64; step];
    for (i, v) in profile.iter().enumerate() {
        raw[i % step] += v;
    }

    // Central differences smear one grid boundary's energy across the two
    // adjacent columns, so score phases as adjacent pairs. This keeps the
    // winner stable for razor-sharp grids (where both columns tie) and makes
    // the confidence ratio meaningful.
    let mut best_phase = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    let mut second_score = f64::NEG_INFINITY;
    for phase in 0..step {
        let score = raw[phase] + raw[(phase + 1) % step];
        if score > best_score {
            second_score = best_score;
            best_score = score;
            best_phase = phase;
        } else if score > second_score {
            second_score = score;
        }
    }

    if best_score <= f64::EPSILON {
        return None;
    }
    Some(PhaseScore {
        phase: best_phase,
        best: best_score,
        second: second_score.max(0.0),
    })
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
    highlight_collapse: f32,
    shadow_collapse: f32,
    quantizer: Quantizer,
    palette_merge: f32,
}

fn quantize_to_rgba8(
    linear_rgba: &[f32],
    width: u32,
    height: u32,
    options: QuantizeOptions<'_>,
) -> (RgbaImage, Vec<[u8; 3]>) {
    let threshold = options.alpha_threshold as f32 / 255.0;
    let adaptive_palette = options.fixed_palette.is_none();
    let cleanup = PaletteCleanup {
        highlight: options.highlight_collapse.clamp(0.0, 1.0),
        shadow: options.shadow_collapse.clamp(0.0, 1.0),
    };
    let raw_labs: Vec<[f32; 3]> = linear_rgba
        .chunks_exact(4)
        .map(|px| linear_to_oklab(px[0], px[1], px[2]))
        .collect();
    let dark_anchor = if adaptive_palette {
        darkest_adaptive_source_color(linear_rgba, &raw_labs, threshold, cleanup)
    } else {
        None
    };
    let mut samples = Vec::new();
    for (px, &lab) in linear_rgba.chunks_exact(4).zip(&raw_labs) {
        if px[3] >= threshold {
            samples.push(if adaptive_palette {
                collapse_adaptive_palette_noise(lab, dark_anchor, cleanup)
            } else {
                lab
            });
        }
    }

    if samples.is_empty() {
        return (RgbaImage::new(width, height), Vec::new());
    }

    let palette = options
        .fixed_palette
        .map(|palette| palette.to_vec())
        .unwrap_or_else(|| {
            let k = options.colors.min(samples.len());
            let mut built = match options.quantizer {
                Quantizer::KMeans => build_palette(&samples, k, 32),
                Quantizer::Wu => build_palette_wu(&samples, k),
            };
            if options.palette_merge > 0.0 {
                built = merge_close_entries(&built, &samples, options.palette_merge);
            }
            built
        });
    let palette_srgb: Vec<[u8; 3]> = palette.iter().map(|&lab| oklab_to_srgb8(lab)).collect();
    let labs: Vec<[f32; 3]> = raw_labs
        .iter()
        .map(|&lab| {
            if adaptive_palette {
                collapse_adaptive_palette_noise(lab, dark_anchor, cleanup)
            } else {
                lab
            }
        })
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

#[derive(Clone, Copy)]
struct PaletteCleanup {
    highlight: f32,
    shadow: f32,
}

fn darkest_adaptive_source_color(
    linear_rgba: &[f32],
    labs: &[[f32; 3]],
    alpha_threshold: f32,
    cleanup: PaletteCleanup,
) -> Option<[f32; 3]> {
    linear_rgba
        .chunks_exact(4)
        .zip(labs)
        .filter(|(px, lab)| px[3] >= alpha_threshold && is_adaptive_near_black(**lab, cleanup))
        .map(|(_, &lab)| lab)
        .min_by(|a, b| {
            a[0].partial_cmp(&b[0])
                .unwrap()
                .then_with(|| a[1].partial_cmp(&b[1]).unwrap())
                .then_with(|| a[2].partial_cmp(&b[2]).unwrap())
        })
}

fn collapse_adaptive_palette_noise(
    lab: [f32; 3],
    dark_anchor: Option<[f32; 3]>,
    cleanup: PaletteCleanup,
) -> [f32; 3] {
    if let Some(anchor) = dark_anchor {
        if is_adaptive_near_black(lab, cleanup) {
            return anchor;
        }
    }
    collapse_adaptive_near_white(lab, cleanup)
}

fn is_adaptive_near_black(lab: [f32; 3], cleanup: PaletteCleanup) -> bool {
    cleanup.shadow > 0.0 && lab[0] <= cleanup.shadow
}

fn collapse_adaptive_near_white(lab: [f32; 3], cleanup: PaletteCleanup) -> [f32; 3] {
    if cleanup.highlight <= 0.0 {
        return lab;
    }
    let min_l = 1.0 - cleanup.highlight;
    let max_chroma = 0.006 + cleanup.highlight * 0.04;
    let chroma2 = lab[1] * lab[1] + lab[2] * lab[2];
    if lab[0] >= min_l && chroma2 <= max_chroma * max_chroma {
        [1.0, 0.0, 0.0]
    } else {
        lab
    }
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

    fn default_cleanup() -> PaletteCleanup {
        PaletteCleanup {
            highlight: DEFAULT_HIGHLIGHT_COLLAPSE,
            shadow: DEFAULT_SHADOW_COLLAPSE,
        }
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

        let (phase, confidence) = detect_grid_phase_with_confidence(&img, 3.0).unwrap();
        assert_eq!(phase, (2, 1));
        assert!(
            (0.0..=1.0).contains(&confidence) && confidence > 0.0,
            "got confidence {confidence}"
        );

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
    fn adaptive_palette_collapses_generated_near_white_noise() {
        let mut img = RgbaImage::new(48, 16);
        let whites = [
            Rgba([254, 254, 254, 255]),
            Rgba([252, 251, 251, 255]),
            Rgba([247, 246, 245, 255]),
        ];
        for (x, y, p) in img.enumerate_pixels_mut() {
            if x < 36 {
                *p = whites[((x + y) % 3) as usize];
            } else {
                let g = 80 + ((x * 11 + y * 7) % 120) as u8;
                *p = Rgba([20, g, 35, 255]);
            }
        }

        let config = Config {
            size: 48,
            colors: 16,
            ..cfg()
        };
        let r = convert(&img, &config).unwrap();
        let near_white_count = r
            .palette
            .iter()
            .filter(|&&[r, g, b]| r >= 245 && g >= 245 && b >= 245)
            .count();

        assert_eq!(
            near_white_count, 1,
            "palette should spend one slot on generated white noise: {:?}",
            r.palette
        );
        assert!(
            r.palette.contains(&[255, 255, 255]),
            "canonical white should be present: {:?}",
            r.palette
        );
    }

    #[test]
    fn adaptive_near_white_collapse_preserves_warm_off_white() {
        let near_white =
            collapse_adaptive_near_white(srgb8_to_oklab(247, 246, 245), default_cleanup());
        let warm_off_white =
            collapse_adaptive_near_white(srgb8_to_oklab(245, 242, 235), default_cleanup());

        assert_eq!(oklab_to_srgb8(near_white), [255, 255, 255]);
        assert_ne!(
            oklab_to_srgb8(warm_off_white),
            [255, 255, 255],
            "warm off-white should remain available as an intentional color"
        );
    }

    #[test]
    fn adaptive_highlight_collapse_zero_disables_white_cleanup() {
        let cleanup = PaletteCleanup {
            highlight: 0.0,
            ..default_cleanup()
        };
        let near_white = collapse_adaptive_near_white(srgb8_to_oklab(247, 246, 245), cleanup);

        assert_ne!(oklab_to_srgb8(near_white), [255, 255, 255]);
    }

    #[test]
    fn adaptive_highlight_collapse_can_reach_deeper_neutral_whites() {
        let cleanup = PaletteCleanup {
            highlight: 0.25,
            ..default_cleanup()
        };
        let gray_white = collapse_adaptive_near_white(srgb8_to_oklab(238, 238, 238), cleanup);
        let warm_off_white = collapse_adaptive_near_white(srgb8_to_oklab(245, 242, 235), cleanup);

        assert_eq!(oklab_to_srgb8(gray_white), [255, 255, 255]);
        assert_eq!(oklab_to_srgb8(warm_off_white), [255, 255, 255]);
    }

    #[test]
    fn adaptive_highlight_collapse_preserves_pale_pinks() {
        let cleanup = PaletteCleanup {
            highlight: 0.25,
            ..default_cleanup()
        };
        let pale_pink = collapse_adaptive_near_white(srgb8_to_oklab(255, 209, 223), cleanup);
        let muted_pink = collapse_adaptive_near_white(srgb8_to_oklab(232, 212, 216), cleanup);

        assert_ne!(oklab_to_srgb8(pale_pink), [255, 255, 255]);
        assert_ne!(oklab_to_srgb8(muted_pink), [255, 255, 255]);
    }

    #[test]
    fn adaptive_palette_collapses_generated_near_black_noise_to_source_darkest() {
        let mut img = RgbaImage::new(48, 16);
        let darks = [
            Rgba([3, 1, 8, 255]),
            Rgba([8, 2, 27, 255]),
            Rgba([16, 8, 9, 255]),
        ];
        for (x, y, p) in img.enumerate_pixels_mut() {
            if x < 12 {
                *p = darks[((x + y) % 3) as usize];
            } else {
                let g = 70 + ((x * 13 + y * 5) % 130) as u8;
                *p = Rgba([18, g, 34, 255]);
            }
        }

        let config = Config {
            size: 48,
            colors: 16,
            ..cfg()
        };
        let r = convert(&img, &config).unwrap();
        let near_black_count = r
            .palette
            .iter()
            .filter(|&&[r, g, b]| r <= 20 && g <= 12 && b <= 32)
            .count();

        assert_eq!(
            near_black_count, 1,
            "palette should spend one slot on generated near-black noise: {:?}",
            r.palette
        );
        assert!(
            r.palette.contains(&[3, 1, 8]),
            "dark collapse should preserve the source's darkest color, not invent black: {:?}",
            r.palette
        );
    }

    #[test]
    fn adaptive_near_black_collapse_preserves_readable_dark_colors() {
        let darkest = srgb8_to_oklab(3, 1, 8);
        let noisy_dark = collapse_adaptive_palette_noise(
            srgb8_to_oklab(16, 8, 9),
            Some(darkest),
            default_cleanup(),
        );
        let readable_navy = collapse_adaptive_palette_noise(
            srgb8_to_oklab(29, 43, 83),
            Some(darkest),
            default_cleanup(),
        );

        assert_eq!(oklab_to_srgb8(noisy_dark), [3, 1, 8]);
        assert_eq!(oklab_to_srgb8(readable_navy), [29, 43, 83]);
    }

    #[test]
    fn adaptive_shadow_collapse_zero_disables_dark_cleanup() {
        let cleanup = PaletteCleanup {
            shadow: 0.0,
            ..default_cleanup()
        };
        let darkest = srgb8_to_oklab(3, 1, 8);
        let noisy_dark =
            collapse_adaptive_palette_noise(srgb8_to_oklab(16, 8, 9), Some(darkest), cleanup);

        assert_ne!(oklab_to_srgb8(noisy_dark), [3, 1, 8]);
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

    #[test]
    fn background_fill_mode_removes_flat_background_through_convert() {
        // Red subject on a flat near-white background; background-fill must
        // yield transparent, fully-zeroed background cells and an untinted
        // subject (transparent RGB decontamination is part of the pass).
        let mut img = RgbaImage::new(32, 32);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = if (8..24).contains(&x) && (8..24).contains(&y) {
                Rgba([200, 30, 30, 255])
            } else {
                Rgba([250, 250, 250, 255])
            };
        }
        let config = Config {
            size: 8,
            colors: 4,
            alpha_mode: AlphaMode::BackgroundFill,
            ..cfg()
        };
        let r = convert(&img, &config).unwrap();
        assert!(r.alpha_removed > 0);
        assert_eq!(
            r.image.get_pixel(0, 0).0,
            [0, 0, 0, 0],
            "bg transparent+zeroed"
        );
        let center = r.image.get_pixel(4, 4).0;
        assert_eq!(center[3], 255);
        assert!(
            center[0] > 150 && center[1] < 90 && center[2] < 90,
            "subject must stay red, got {center:?}"
        );
    }

    #[test]
    fn auto_colors_picks_small_preset_for_flat_art_and_reports_choice() {
        let mut img = RgbaImage::new(64, 64);
        let colors = [
            Rgba([200, 30, 30, 255]),
            Rgba([30, 200, 30, 255]),
            Rgba([30, 30, 200, 255]),
            Rgba([220, 220, 60, 255]),
        ];
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = colors[((x / 16) + (y / 16)) as usize % 4];
        }
        let config = Config {
            size: 32,
            auto_colors: true,
            colors: 512, // must be ignored
            ..cfg()
        };
        let r = convert(&img, &config).unwrap();
        assert_eq!(r.auto_colors, Some(16));
        assert!(r.palette_len <= 16);
    }

    #[test]
    fn auto_colors_scales_up_for_high_variety_sources() {
        let mut img = RgbaImage::new(64, 64);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = Rgba([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8, 255]);
        }
        let config = Config {
            size: 64,
            auto_colors: true,
            ..cfg()
        };
        let r = convert(&img, &config).unwrap();
        let chosen = r.auto_colors.unwrap();
        assert!(
            chosen >= 128,
            "gradient should demand many colors, got {chosen}"
        );
        assert!(AUTO_COLOR_PRESETS.contains(&chosen));
    }

    #[test]
    fn manual_phase_overrides_detection() {
        let mut img = RgbaImage::new(60, 40);
        for p in img.pixels_mut() {
            *p = Rgba([128, 128, 128, 255]); // flat: no detectable phase
        }
        let config = Config {
            pixel_size: Some(4.0),
            phase_x: Some(2),
            phase_y: Some(5), // reduced mod 4 -> 1
            ..cfg()
        };
        let grid = target_grid(&img, &config);
        assert_eq!(grid.phase, Some((2, 1)));
        let sampling = grid.sampling.unwrap();
        assert_eq!(sampling.origin_x, 2.0);
        assert_eq!(sampling.origin_y, 1.0);
        // (60-2)/4 = 14, (40-1)/4 = 9
        assert_eq!((grid.out_w, grid.out_h), (14, 9));
    }

    #[test]
    fn snap_phase_reports_confidence_and_diagnostics_json_is_wellformed() {
        let mut img = RgbaImage::new(60, 40);
        for (x, y, p) in img.enumerate_pixels_mut() {
            let checker = (x / 5 + y / 5) % 2 == 0;
            *p = if checker {
                Rgba([220, 210, 80, 255])
            } else {
                Rgba([30, 40, 80, 255])
            };
        }
        let config = Config {
            pixel_size: Some(5.0),
            colors: 4,
            ..cfg()
        };
        let r = convert(&img, &config).unwrap();
        let conf = r.phase_confidence.expect("confidence with snapped phase");
        assert!(
            (0.0..=1.0).contains(&conf) && conf > 0.3,
            "hard grid should be confident, got {conf}"
        );
        let json = r.diagnostics_json(&config);
        assert!(json.contains("\"srcWidth\": 60"));
        assert!(json.contains("\"phaseConfidence\":"));
        assert!(json.contains("\"cleanupPreset\": \"None\""));
        assert_eq!(json.matches('{').count(), json.matches('}').count());
    }

    #[test]
    fn cleanup_preset_is_applied_to_the_output_grid() {
        // A red block plus an isolated unique speck (4x4 source px so the
        // 5px cell at (6,6) stays above the alpha threshold): conservative
        // cleanup must drop the speck from the final grid.
        let mut img = RgbaImage::new(40, 40);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = if x < 15 && y < 15 {
                Rgba([200, 30, 30, 255])
            } else if (30..34).contains(&x) && (30..34).contains(&y) {
                Rgba([40, 220, 40, 255])
            } else {
                Rgba([0, 0, 0, 0])
            };
        }
        let base = Config {
            size: 8,
            colors: 4,
            cell: CellMode::Dominant,
            ..cfg()
        };
        let plain = convert(&img, &base).unwrap();
        let cleaned = convert(
            &img,
            &Config {
                cleanup: CleanupPreset::Conservative,
                ..base
            },
        )
        .unwrap();
        assert_eq!(cleaned.cleanup.orphans_removed, 1);
        assert_eq!(cleaned.image.get_pixel(6, 6).0[3], 0, "speck removed");
        assert_eq!(
            plain.image.get_pixel(6, 6).0[3],
            255,
            "speck present without cleanup"
        );
    }

    #[test]
    fn wu_quantizer_converts_deterministically_and_respects_color_budget() {
        let mut img = RgbaImage::new(64, 48);
        for (i, p) in img.pixels_mut().enumerate() {
            let v = (i % 251) as u8;
            *p = Rgba([v, v.wrapping_mul(3), v.wrapping_mul(7), 255]);
        }
        let config = Config {
            size: 16,
            colors: 8,
            quantizer: Quantizer::Wu,
            ..cfg()
        };
        let a = convert(&img, &config).unwrap();
        let b = convert(&img, &config).unwrap();
        assert!(a.palette_len <= 8 && a.palette_len >= 2);
        assert_eq!(
            a.image.as_raw(),
            b.image.as_raw(),
            "Wu path must be deterministic"
        );

        // And it must differ from the k-means path only in palette choice,
        // not in structure: same dims, same alpha layout.
        let km = convert(
            &img,
            &Config {
                quantizer: Quantizer::KMeans,
                ..config
            },
        )
        .unwrap();
        assert_eq!((a.out_w, a.out_h), (km.out_w, km.out_h));
    }

    #[test]
    fn palette_merge_collapses_near_duplicate_adaptive_entries() {
        // Two nearly identical greens + one red, k=3: with merge on, the two
        // greens must share one slot.
        let mut img = RgbaImage::new(30, 10);
        for (x, _, p) in img.enumerate_pixels_mut() {
            *p = if x < 10 {
                Rgba([30, 200, 30, 255])
            } else if x < 20 {
                Rgba([34, 204, 34, 255])
            } else {
                Rgba([200, 30, 30, 255])
            };
        }
        let base = Config {
            size: 30,
            colors: 3,
            highlight_collapse: 0.0,
            shadow_collapse: 0.0,
            ..cfg()
        };
        let plain = convert(&img, &base).unwrap();
        assert_eq!(plain.palette_len, 3);

        let merged = convert(
            &img,
            &Config {
                palette_merge: 0.05,
                ..base
            },
        )
        .unwrap();
        assert_eq!(merged.palette_len, 2, "{:?}", merged.palette);
        let red_kept = merged.palette.iter().any(|&[r, g, b]| {
            (r as i32 - 200).abs() <= 2 && (g as i32 - 30).abs() <= 2 && (b as i32 - 30).abs() <= 2
        });
        assert!(red_kept, "distinct red kept: {:?}", merged.palette);
    }

    #[test]
    fn contrast_expansion_saves_single_pixel_details_through_downsampling() {
        // 12x12 -> 3x3 with 4px cells; one dark source pixel centered in the
        // middle cell. Without expansion the light field wins the cell; with
        // radius 1 the stamped 3x3 dark block wins the dominant vote.
        let mut img = RgbaImage::new(12, 12);
        for p in img.pixels_mut() {
            *p = Rgba([225, 225, 225, 255]);
        }
        img.put_pixel(5, 5, Rgba([25, 25, 25, 255]));
        let base = Config {
            size: 3,
            colors: 2,
            cell: CellMode::Dominant,
            highlight_collapse: 0.0,
            shadow_collapse: 0.0,
            ..cfg()
        };
        let plain = convert(&img, &base).unwrap();
        assert!(
            plain.image.get_pixel(1, 1).0[0] > 150,
            "detail lost without expansion"
        );

        let saved = convert(
            &img,
            &Config {
                contrast_expansion: 1,
                ..base
            },
        )
        .unwrap();
        assert!(saved.contrast_expanded > 0);
        assert!(
            saved.image.get_pixel(1, 1).0[0] < 90,
            "dark detail must survive: {:?}",
            saved.image.get_pixel(1, 1)
        );
        assert!(
            saved.image.get_pixel(0, 0).0[0] > 150,
            "field cells stay light"
        );
    }

    #[test]
    fn outline_repair_runs_after_cleanup_on_the_output_grid() {
        // Source maps 1:1 to a 12x12 grid: dark ring sprite with one gap.
        let mut img = RgbaImage::new(12, 12);
        for y in 2..=9u32 {
            for x in 2..=9u32 {
                let ring = x == 2 || x == 9 || y == 2 || y == 9;
                img.put_pixel(
                    x,
                    y,
                    Rgba(if ring {
                        [20, 20, 24, 255]
                    } else {
                        [200, 30, 30, 255]
                    }),
                );
            }
        }
        img.put_pixel(5, 2, Rgba([200, 30, 30, 255])); // gap in the outline
        let config = Config {
            size: 12,
            colors: 2,
            outline: OutlineMode::Repair,
            highlight_collapse: 0.0,
            shadow_collapse: 0.0,
            ..cfg()
        };
        let r = convert(&img, &config).unwrap();
        assert_eq!(r.outline.recolored, 1);
        let p = r.image.get_pixel(5, 2).0;
        assert!(
            p[0] < 60 && p[1] < 60,
            "gap must be repainted dark, got {p:?}"
        );
    }

    #[test]
    fn color_key_mode_requires_a_key() {
        let img = RgbaImage::new(4, 4);
        let config = Config {
            alpha_mode: AlphaMode::ColorKey,
            ..cfg()
        };
        assert!(convert(&img, &config).is_err());
    }

    #[test]
    fn debug_grid_image_draws_lines_at_cell_boundaries() {
        let mut img = RgbaImage::new(20, 10);
        for p in img.pixels_mut() {
            *p = Rgba([100, 100, 100, 255]);
        }
        let config = Config {
            pixel_size: Some(5.0),
            snap_grid: false,
            ..cfg()
        };
        let out = debug_grid_image(&img, &config).unwrap();
        assert_eq!(out.dimensions(), (20, 10));
        assert_eq!(
            out.get_pixel(5, 3).0,
            [255, 0, 255, 255],
            "vertical line at x=5"
        );
        assert_eq!(
            out.get_pixel(7, 5).0,
            [255, 0, 255, 255],
            "horizontal line at y=5"
        );
        assert_eq!(
            out.get_pixel(2, 2).0,
            [100, 100, 100, 255],
            "cell interior untouched"
        );
    }
}
