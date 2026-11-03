//! The Pixeline frontend's request handlers, independent of how a request arrives.
//!
//! The native GUI (`bin/gui.rs`, HTTP over `std::net`) and the browser build
//! (`web-wasm`, wasm-bindgen in a Worker) are thin transports over these functions,
//! so settings parsing, response metadata, and export formats cannot drift apart.
//! Settings are form-urlencoded text; `crop=x,y,width,height` is in source pixels.

use std::collections::HashMap;
use std::io::Cursor;

use image::{ImageFormat, Rgba, RgbaImage};

use crate::{
    alpha::AlphaMode,
    downsample::{CellMode, DEFAULT_ADAPTIVE_ITERATIONS, MAX_ADAPTIVE_ITERATIONS},
    morphology::CleanupPreset,
    outline::OutlineMode,
    palettes,
    pipeline::{
        self, Config, Dither, PaletteChoice, Quantizer, DEFAULT_BG_TOLERANCE,
        DEFAULT_HIGHLIGHT_COLLAPSE, DEFAULT_SHADOW_COLLAPSE,
    },
};

const NO_SOURCE: &str = "no image loaded; POST /api/session first";

/// An HTTP-shaped reply. `headers` holds extra `Name: value` lines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiResponse {
    pub status: u16,
    pub content_type: &'static str,
    pub headers: Vec<String>,
    pub body: Vec<u8>,
}

impl ApiResponse {
    fn ok(content_type: &'static str, headers: Vec<String>, body: Vec<u8>) -> Self {
        Self {
            status: 200,
            content_type,
            headers,
            body,
        }
    }

    pub fn error(status: u16, message: &str) -> Self {
        Self {
            status,
            content_type: "application/json",
            headers: Vec::new(),
            body: format!(r#"{{"error":{}}}"#, json_string(message)).into_bytes(),
        }
    }
}

/// Resource limits a transport may impose. The native GUI has none; the browser
/// build must fail cleanly before an export outgrows wasm32 memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Limits {
    pub max_output_pixels: Option<u64>,
}

/// Metadata for a newly loaded source (`/api/session`).
pub fn session_response(img: &RgbaImage) -> ApiResponse {
    let (w, h) = img.dimensions();
    let auto = match pipeline::detect_pixel_size_of(img) {
        Some(v) => format!("{v:.2}"),
        None => "null".to_string(),
    };
    let json = format!(r#"{{"srcWidth":{w},"srcHeight":{h},"autoPixelSize":{auto}}}"#);
    ApiResponse::ok("application/json", Vec::new(), json.into_bytes())
}

/// Convert the source to a PNG (`/api/preview` when `preview`, else `/api/export`).
pub fn convert_response(
    src: Option<&RgbaImage>,
    settings: &str,
    preview: bool,
    limits: Limits,
) -> ApiResponse {
    let form = parse_form(settings);
    let mut cfg = match config_from_form(&form, preview) {
        Ok(cfg) => cfg,
        Err(e) => return ApiResponse::error(400, &e),
    };
    cfg.max_output_pixels = limits.max_output_pixels;
    let Some(src) = src else {
        return ApiResponse::error(409, NO_SOURCE);
    };
    let result = match pipeline::convert(src, &cfg) {
        Ok(r) => r,
        Err(e) => return ApiResponse::error(400, &e),
    };
    let png = match encode_png(&result.image) {
        Ok(png) => png,
        Err(e) => return ApiResponse::error(500, &e),
    };

    let detected = result
        .detected_pixel_size
        .map(|v| format!("{v:.2}"))
        .unwrap_or_default();
    let phase = result
        .grid_phase
        .map(|(x, y)| format!("{x},{y}"))
        .unwrap_or_default();
    let confidence = result
        .phase_confidence
        .map(|v| format!("{v:.2}"))
        .unwrap_or_default();
    let auto_colors = result
        .auto_colors
        .map(|v| v.to_string())
        .unwrap_or_default();
    let palette_hex: String = result
        .palette
        .iter()
        .map(|[r, g, b]| format!("{r:02x}{g:02x}{b:02x}"))
        .collect::<Vec<_>>()
        .join(",");

    let mut headers = vec![
        format!("X-Out-Width: {}", result.out_w),
        format!("X-Out-Height: {}", result.out_h),
        format!("X-Palette-Len: {}", result.palette_len),
        format!("X-Detected-Pixel-Size: {detected}"),
        format!("X-Grid-Phase: {phase}"),
        format!("X-Phase-Confidence: {confidence}"),
        format!("X-Auto-Colors: {auto_colors}"),
        format!("X-Cleanup-Removed: {}", result.cleanup.total()),
        format!(
            "X-Adaptive-Fallback: {}",
            if result.adaptive_fallback { 1 } else { 0 }
        ),
        format!("X-Palette: {palette_hex}"),
    ];
    if !preview {
        headers.push("Content-Disposition: attachment; filename=\"pixeline.png\"".to_string());
    }
    ApiResponse::ok("image/png", headers, png)
}

/// Export the palette the current settings produce (`/api/palette`).
pub fn palette_response(src: Option<&RgbaImage>, settings: &str, limits: Limits) -> ApiResponse {
    let form = parse_form(settings);
    let mut cfg = match config_from_form(&form, true) {
        Ok(cfg) => cfg,
        Err(e) => return ApiResponse::error(400, &e),
    };
    cfg.max_output_pixels = limits.max_output_pixels;
    let format = form
        .get("paletteFormat")
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| "hex".to_string());
    let Some(src) = src else {
        return ApiResponse::error(409, NO_SOURCE);
    };
    let result = match pipeline::convert(src, &cfg) {
        Ok(r) => r,
        Err(e) => return ApiResponse::error(400, &e),
    };

    fn attachment(name: &str) -> Vec<String> {
        vec![format!(
            "Content-Disposition: attachment; filename=\"{name}\""
        )]
    }
    const TEXT: &str = "text/plain; charset=utf-8";
    match format.as_str() {
        "gpl" => ApiResponse::ok(
            TEXT,
            attachment("pixeline_palette.gpl"),
            palettes::format_gpl(&result.palette, "Pixeline").into_bytes(),
        ),
        "png" => match encode_png(&palette_strip_image(&result.palette)) {
            Ok(png) => ApiResponse::ok("image/png", attachment("pixeline_palette.png"), png),
            Err(e) => ApiResponse::error(500, &e),
        },
        "hex" | "" => ApiResponse::ok(
            TEXT,
            attachment("pixeline_palette.hex"),
            palettes::format_hex_list(&result.palette).into_bytes(),
        ),
        _ => ApiResponse::error(400, "paletteFormat must be hex, gpl, or png"),
    }
}

fn encode_png(img: &RgbaImage) -> Result<Vec<u8>, String> {
    let mut png = Vec::new();
    img.write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
        .map_err(|e| format!("encode failed: {e}"))?;
    Ok(png)
}

fn palette_strip_image(palette: &[[u8; 3]]) -> RgbaImage {
    let width = palette.len().max(1) as u32;
    let mut img = RgbaImage::new(width, 1);
    for (x, [r, g, b]) in palette.iter().enumerate() {
        img.put_pixel(x as u32, 0, Rgba([*r, *g, *b, 255]));
    }
    img
}

fn config_from_form(f: &HashMap<String, String>, preview: bool) -> Result<Config, String> {
    let get = |k: &str| f.get(k).map(|s| s.as_str());
    let size_mode = get("sizeMode").unwrap_or("size");
    let (pixel_size, auto_pixel_size) = match size_mode {
        "pixel" => (Some(parse_or::<f64>(get("pixelSize"), 6.0).max(1.0)), false),
        "auto" => (None, true),
        _ => (None, false),
    };
    let palette = match get("palette").unwrap_or("adaptive") {
        "adaptive" => PaletteChoice::Adaptive,
        "custom" => PaletteChoice::HexList(get("customHex").unwrap_or("").to_string()),
        name => PaletteChoice::Builtin(name.to_string()),
    };
    let cell = match get("cell").unwrap_or("detail") {
        "box" => CellMode::Box,
        "median" => CellMode::Median,
        "dominant" => CellMode::Dominant,
        "adaptive" => CellMode::Adaptive,
        _ => CellMode::Detail,
    };
    let dither = match get("dither").unwrap_or("none") {
        "bayer4" => Dither::Bayer4,
        "bayer8" => Dither::Bayer8,
        _ => Dither::None,
    };
    let alpha_mode = match get("alphaMode").unwrap_or("preserve") {
        "binary" => AlphaMode::Binary,
        "bgfill" | "background-fill" => AlphaMode::BackgroundFill,
        "key" | "color-key" => AlphaMode::ColorKey,
        _ => AlphaMode::Preserve,
    };
    let cleanup = match get("cleanup").unwrap_or("none") {
        "conservative" => CleanupPreset::Conservative,
        "balanced" => CleanupPreset::Balanced,
        "aggressive" => CleanupPreset::Aggressive,
        _ => CleanupPreset::None,
    };
    let color_key = get("colorKey").and_then(parse_hex_color);
    let quantizer = match get("quantizer").unwrap_or("kmeans") {
        "wu" => Quantizer::Wu,
        _ => Quantizer::KMeans,
    };
    let outline = match get("outline").unwrap_or("none") {
        "repair" => OutlineMode::Repair,
        "enforce" => OutlineMode::Enforce,
        _ => OutlineMode::None,
    };

    Ok(Config {
        crop: get("crop").map(str::parse).transpose()?,
        size: parse_or::<u32>(get("size"), 64).max(1),
        pixel_size,
        auto_pixel_size,
        snap_grid: !matches!(get("snapGrid"), Some("false" | "0" | "off")),
        colors: (parse_or::<usize>(get("colors"), 16)).clamp(1, 512),
        palette,
        dither,
        dither_strength: parse_or::<f32>(get("ditherStrength"), 0.35).clamp(0.0, 1.0),
        scale: if preview {
            1
        } else {
            parse_or::<u32>(get("scale"), 1).clamp(1, 16)
        },
        alpha_threshold: parse_or::<u16>(get("alphaThreshold"), 128).min(255) as u8,
        cell,
        dominant_threshold: parse_or::<f32>(get("dominantThreshold"), 0.25).clamp(0.0, 1.0),
        adaptive_iterations: parse_or::<u32>(
            get("adaptiveIterations"),
            DEFAULT_ADAPTIVE_ITERATIONS,
        )
        .clamp(1, MAX_ADAPTIVE_ITERATIONS),
        highlight_collapse: parse_or::<f32>(get("highlightCollapse"), DEFAULT_HIGHLIGHT_COLLAPSE)
            .clamp(0.0, 1.0),
        shadow_collapse: parse_or::<f32>(get("shadowCollapse"), DEFAULT_SHADOW_COLLAPSE)
            .clamp(0.0, 1.0),
        alpha_mode,
        bg_tolerance: parse_or::<f32>(get("bgTolerance"), DEFAULT_BG_TOLERANCE).clamp(0.0, 1.0),
        color_key,
        cleanup,
        protect_details: !matches!(get("protectDetails"), Some("false" | "0" | "off")),
        auto_colors: matches!(get("autoColors"), Some("true" | "on" | "1")),
        phase_x: get("phaseX").and_then(|s| s.parse().ok()),
        phase_y: get("phaseY").and_then(|s| s.parse().ok()),
        quantizer,
        palette_merge: parse_or::<f32>(get("paletteMerge"), 0.0).clamp(0.0, 1.0),
        contrast_expansion: parse_or::<u32>(get("contrastExpansion"), 0).min(4),
        outline,
        compare: if preview {
            false
        } else {
            matches!(get("compare"), Some("true" | "on" | "1"))
        },
        max_output_pixels: None,
    })
}

fn parse_or<T: std::str::FromStr>(v: Option<&str>, default: T) -> T {
    v.and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Parse "RRGGBB"/"#RRGGBB"; None on anything malformed.
fn parse_hex_color(text: &str) -> Option<[u8; 3]> {
    let t = text.trim().trim_start_matches('#');
    if t.len() != 6 || !t.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let v = u32::from_str_radix(t, 16).ok()?;
    Some([(v >> 16) as u8, (v >> 8) as u8, v as u8])
}

fn parse_form(body: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for pair in body.split('&') {
        if pair.is_empty() {
            continue;
        }
        let mut it = pair.splitn(2, '=');
        let key = percent_decode(it.next().unwrap_or(""));
        let val = percent_decode(it.next().unwrap_or(""));
        map.insert(key, val);
    }
    map
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => match (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                (Some(h), Some(l)) => {
                    out.push(h * 16 + l);
                    i += 3;
                }
                _ => {
                    out.push(b'%');
                    i += 1;
                }
            },
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Minimal JSON string escaping for error messages.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header<'a>(response: &'a ApiResponse, name: &str) -> Option<&'a str> {
        let prefix = format!("{name}: ");
        response
            .headers
            .iter()
            .find_map(|h| h.strip_prefix(prefix.as_str()))
    }

    fn checkerboard() -> RgbaImage {
        RgbaImage::from_fn(32, 32, |x, y| {
            if (x / 4 + y / 4) % 2 == 0 {
                Rgba([200, 40, 40, 255])
            } else {
                Rgba([0, 0, 0, 0])
            }
        })
    }

    #[test]
    fn crop_settings_apply_to_preview_export_and_palette() {
        let form = parse_form("crop=2%2C3%2C4%2C5&scale=8&compare=true");
        let preview = config_from_form(&form, true).unwrap();
        let export = config_from_form(&form, false).unwrap();
        assert_eq!(preview.crop, export.crop);
        assert_eq!(preview.crop.unwrap().width, 4);
        assert_eq!(preview.scale, 1);
        assert!(!preview.compare);
        assert_eq!(export.scale, 8);
        assert!(export.compare);
    }

    #[test]
    fn malformed_crop_is_an_error_instead_of_processing_the_full_source() {
        assert!(config_from_form(&parse_form("crop=0,0,10"), true).is_err());
        assert!(config_from_form(&parse_form("crop=0,0,0,10"), false).is_err());
        assert!(config_from_form(&HashMap::new(), true)
            .unwrap()
            .crop
            .is_none());
    }

    #[test]
    fn converting_without_a_source_is_a_conflict() {
        let reply = convert_response(None, "size=8", true, Limits::default());
        assert_eq!(reply.status, 409);
        assert_eq!(palette_response(None, "", Limits::default()).status, 409);
    }

    #[test]
    fn bad_settings_are_rejected_before_the_missing_source_check() {
        let reply = convert_response(None, "crop=nope", true, Limits::default());
        assert_eq!(reply.status, 400);
    }

    #[test]
    fn export_reports_grid_metadata_and_marks_the_attachment() {
        let src = checkerboard();
        let settings = "sizeMode=size&size=8&colors=2&scale=3";
        let preview = convert_response(Some(&src), settings, true, Limits::default());
        let export = convert_response(Some(&src), settings, false, Limits::default());
        assert_eq!(preview.status, 200);
        assert_eq!(preview.content_type, "image/png");
        assert_eq!(header(&preview, "X-Out-Width"), Some("8"));
        assert_eq!(header(&preview, "Content-Disposition"), None);
        assert!(header(&export, "Content-Disposition").is_some());
        assert_ne!(preview.body, export.body, "export bakes in the scale");
    }

    #[test]
    fn output_limit_fails_cleanly_for_exports_but_not_previews() {
        let src = checkerboard();
        let settings = "sizeMode=size&size=32&colors=2&scale=16";
        let limits = Limits {
            max_output_pixels: Some(32 * 32),
        };
        assert_eq!(
            convert_response(Some(&src), settings, true, limits).status,
            200
        );
        let export = convert_response(Some(&src), settings, false, limits);
        assert_eq!(export.status, 400);
        let message = String::from_utf8(export.body).unwrap();
        assert!(message.contains("512"), "{message}");
    }

    #[test]
    fn palette_formats_round_trip_and_unknown_formats_error() {
        let src = checkerboard();
        let run = |format: &str| {
            let settings = format!("sizeMode=size&size=8&colors=2&paletteFormat={format}");
            palette_response(Some(&src), &settings, Limits::default())
        };
        assert_eq!(run("hex").content_type, "text/plain; charset=utf-8");
        assert!(run("gpl").body.starts_with(b"GIMP Palette"));
        assert_eq!(run("png").content_type, "image/png");
        assert_eq!(run("bmp").status, 400);
    }

    #[test]
    fn errors_are_json_with_escaped_messages() {
        let reply = ApiResponse::error(400, "bad \"value\"\n");
        assert_eq!(reply.content_type, "application/json");
        assert_eq!(reply.body, br#"{"error":"bad \"value\"\n"}"#);
    }
}
