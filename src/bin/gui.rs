//! Pixeline — a local web GUI for Raster_to_Pixel.
//!
//! Zero new dependencies: a tiny `std::net` HTTP server bound to 127.0.0.1 that serves
//! an embedded single-page frontend and calls the shared `pipeline::convert`. See
//! GUI_PLAN.md for the design and the API contract.
//!
//!   GET  /               -> the app (embedded gui/index.html)
//!   GET  /api/palettes   -> JSON list of palette presets
//!   POST /api/session    -> raw image bytes; caches the source, returns JSON metadata
//!   POST /api/preview    -> form-urlencoded settings; PNG of the raw grid (scale=1, no compare)
//!   POST /api/export     -> form-urlencoded settings; PNG with real scale + compare (download)

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Cursor, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use image::{ImageFormat, RgbaImage};
use raster_to_pixel::{
    downsample::CellMode,
    pipeline::{
        self, Config, Dither, PaletteChoice, DEFAULT_HIGHLIGHT_COLLAPSE, DEFAULT_SHADOW_COLLAPSE,
    },
};

/// The single cached source image (single-user local tool -> one slot behind a Mutex).
/// Conversion requests clone the `Arc` and drop the lock before the expensive work,
/// so aborted previews do not stall newer previews or uploads.
static SESSION: Mutex<Option<Arc<RgbaImage>>> = Mutex::new(None);

fn main() {
    let mut port: u16 = 7878;
    let mut open = true;
    let mut browser = Browser::Default;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--port" => {
                if let Some(p) = args.next().and_then(|v| v.parse().ok()) {
                    port = p;
                }
            }
            "--no-open" => open = false,
            "--chrome" | "--chromium" => browser = Browser::Chromium,
            "-h" | "--help" => {
                println!("pixeline [--port <N>] [--no-open] [--chrome]");
                return;
            }
            _ => {}
        }
    }

    let (listener, port) = match bind(port) {
        Some(v) => v,
        None => {
            eprintln!("could not bind a local port near {port}");
            std::process::exit(1);
        }
    };
    let url = format!("http://127.0.0.1:{port}/");
    println!("Pixeline GUI running at {url}");
    println!("(Ctrl+C to stop)");
    if open {
        open_browser(&url, browser);
    }

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                std::thread::spawn(move || {
                    if let Err(e) = handle(stream) {
                        eprintln!("connection error: {e}");
                    }
                });
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
}

/// Bind 127.0.0.1, trying a few ports up from `start` if the first is taken.
fn bind(start: u16) -> Option<(TcpListener, u16)> {
    for port in start..start.saturating_add(20) {
        if let Ok(l) = TcpListener::bind(("127.0.0.1", port)) {
            return Some((l, port));
        }
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Browser {
    Default,
    Chromium,
}

fn open_browser(url: &str, browser: Browser) {
    if browser == Browser::Chromium {
        for exe in chromium_candidates() {
            if exe.exists() && std::process::Command::new(&exe).arg(url).spawn().is_ok() {
                return;
            }
        }
        for exe in ["chrome", "msedge"] {
            if std::process::Command::new(exe).arg(url).spawn().is_ok() {
                return;
            }
        }
        eprintln!("could not find Chrome/Edge; falling back to the default browser");
    }

    // Windows: `cmd /C start "" <url>`. Best-effort; the URL is printed regardless.
    let _ = std::process::Command::new("cmd")
        .args(["/C", "start", "", url])
        .spawn();
}

fn chromium_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for var in ["ProgramFiles", "ProgramFiles(x86)", "LocalAppData"] {
        if let Some(base) = std::env::var_os(var) {
            out.push(PathBuf::from(&base).join("Google/Chrome/Application/chrome.exe"));
            out.push(PathBuf::from(&base).join("Microsoft/Edge/Application/msedge.exe"));
        }
    }
    out
}

// ── request handling ────────────────────────────────────────────────────────

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn handle(stream: TcpStream) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);

    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let raw_path = parts.next().unwrap_or("/").to_string();
    let path = raw_path.split('?').next().unwrap_or("/").to_string();

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        }
    }

    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;

    let req = Request { method, path, body };
    let mut stream = stream;
    route(&req, &mut stream)
}

fn route(req: &Request, stream: &mut TcpStream) -> std::io::Result<()> {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") => send(
            stream,
            "200 OK",
            "text/html; charset=utf-8",
            &[],
            index_html().as_bytes(),
        ),
        ("GET", "/favicon.ico") => send(stream, "204 No Content", "text/plain", &[], b""),
        ("GET", "/api/palettes") => send(
            stream,
            "200 OK",
            "application/json",
            &[],
            br#"["adaptive","pico8","gameboy","sweetie16"]"#,
        ),
        ("POST", "/api/session") => handle_session(req, stream),
        ("POST", "/api/preview") => handle_convert(req, stream, true),
        ("POST", "/api/export") => handle_convert(req, stream, false),
        _ => send(stream, "404 Not Found", "text/plain", &[], b"not found"),
    }
}

fn handle_session(req: &Request, stream: &mut TcpStream) -> std::io::Result<()> {
    let img = match image::load_from_memory(&req.body) {
        Ok(img) => img.to_rgba8(),
        Err(e) => return send_error(stream, "400 Bad Request", &format!("decode failed: {e}")),
    };
    let (w, h) = img.dimensions();
    let detected = pipeline::detect_pixel_size_of(&img);
    *SESSION.lock().unwrap() = Some(Arc::new(img));

    let auto = match detected {
        Some(v) => format!("{v:.2}"),
        None => "null".to_string(),
    };
    let json = format!(r#"{{"srcWidth":{w},"srcHeight":{h},"autoPixelSize":{auto}}}"#);
    send(stream, "200 OK", "application/json", &[], json.as_bytes())
}

fn handle_convert(req: &Request, stream: &mut TcpStream, preview: bool) -> std::io::Result<()> {
    let form = parse_form(std::str::from_utf8(&req.body).unwrap_or(""));
    let cfg = config_from_form(&form, preview);

    let Some(src) = SESSION.lock().unwrap().clone() else {
        return send_error(
            stream,
            "409 Conflict",
            "no image loaded; POST /api/session first",
        );
    };

    let result = match pipeline::convert(&src, &cfg) {
        Ok(r) => r,
        Err(e) => return send_error(stream, "400 Bad Request", &e),
    };

    let mut png = Vec::new();
    if let Err(e) = result
        .image
        .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
    {
        return send_error(
            stream,
            "500 Internal Server Error",
            &format!("encode failed: {e}"),
        );
    }

    let detected = result
        .detected_pixel_size
        .map(|v| format!("{v:.2}"))
        .unwrap_or_default();
    let phase = result
        .grid_phase
        .map(|(x, y)| format!("{x},{y}"))
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
        format!("X-Palette: {palette_hex}"),
    ];
    if !preview {
        headers.push("Content-Disposition: attachment; filename=\"pixeline.png\"".to_string());
    }
    let header_refs: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
    send(stream, "200 OK", "image/png", &header_refs, &png)
}

// ── settings mapping ────────────────────────────────────────────────────────

fn config_from_form(f: &HashMap<String, String>, preview: bool) -> Config {
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
        _ => CellMode::Detail,
    };
    let dither = match get("dither").unwrap_or("none") {
        "bayer4" => Dither::Bayer4,
        "bayer8" => Dither::Bayer8,
        _ => Dither::None,
    };

    Config {
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
        highlight_collapse: parse_or::<f32>(get("highlightCollapse"), DEFAULT_HIGHLIGHT_COLLAPSE)
            .clamp(0.0, 1.0),
        shadow_collapse: parse_or::<f32>(get("shadowCollapse"), DEFAULT_SHADOW_COLLAPSE)
            .clamp(0.0, 1.0),
        compare: if preview {
            false
        } else {
            matches!(get("compare"), Some("true" | "on" | "1"))
        },
    }
}

fn parse_or<T: std::str::FromStr>(v: Option<&str>, default: T) -> T {
    v.and_then(|s| s.parse().ok()).unwrap_or(default)
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

// ── responses + assets ──────────────────────────────────────────────────────

fn send(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    extra_headers: &[&str],
    body: &[u8],
) -> std::io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n",
        body.len()
    );
    for h in extra_headers {
        head.push_str(h);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn send_error(stream: &mut TcpStream, status: &str, message: &str) -> std::io::Result<()> {
    let body = format!(r#"{{"error":{}}}"#, json_string(message));
    send(stream, status, "application/json", &[], body.as_bytes())
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

/// The embedded frontend, with a dev override so the page can be iterated without
/// recompiling: set RTP_GUI_ASSET_DIR to a folder containing index.html.
fn index_html() -> Cow<'static, str> {
    if let Ok(dir) = std::env::var("RTP_GUI_ASSET_DIR") {
        if let Ok(s) = std::fs::read_to_string(Path::new(&dir).join("index.html")) {
            return Cow::Owned(s);
        }
    }
    Cow::Borrowed(include_str!("../../gui/index.html"))
}
