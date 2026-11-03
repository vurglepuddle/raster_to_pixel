//! Pixeline — a local web GUI for Raster_to_Pixel.
//!
//! Zero new dependencies: a tiny `std::net` HTTP server bound to 127.0.0.1 that serves
//! an embedded single-page frontend over the shared handlers in `api`, the same ones
//! the browser build calls. See GUI_PLAN.md for the design and the API contract.
//!
//!   GET  /               -> the app (embedded gui/index.html)
//!   GET  /api/palettes   -> JSON list of palette presets
//!   POST /api/session    -> raw image bytes; caches the source, returns JSON metadata
//!   POST /api/preview    -> form-urlencoded settings; PNG of the raw grid (scale=1, no compare)
//!   POST /api/export     -> form-urlencoded settings; PNG with real scale + compare (download)
//!   POST /api/palette    -> form-urlencoded settings; resulting palette download
//! Conversion settings accept optional `crop=x,y,width,height` in source pixels.

use std::borrow::Cow;
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use image::RgbaImage;
use raster_to_pixel::{
    api::{self, ApiResponse, Limits},
    image_io,
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
                    if let Err(e) = handle(stream).filter_client_disconnect() {
                        eprintln!("connection error: {e}");
                    }
                });
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
}

trait IgnoreClientDisconnect {
    fn filter_client_disconnect(self) -> std::io::Result<()>;
}

impl IgnoreClientDisconnect for std::io::Result<()> {
    fn filter_client_disconnect(self) -> std::io::Result<()> {
        match self {
            Err(e) if is_expected_disconnect(&e) => Ok(()),
            other => other,
        }
    }
}

fn is_expected_disconnect(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        ErrorKind::BrokenPipe | ErrorKind::ConnectionReset | ErrorKind::ConnectionAborted
    ) || matches!(e.raw_os_error(), Some(10053 | 10054))
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
        // Only the browser build ships a real one; this keeps gui/index.html identical.
        ("GET", "/web_backend.js") => send(stream, "200 OK", "text/javascript", &[], b""),
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
        ("POST", "/api/palette") => handle_palette(req, stream),
        _ => send(stream, "404 Not Found", "text/plain", &[], b"not found"),
    }
}

fn handle_session(req: &Request, stream: &mut TcpStream) -> std::io::Result<()> {
    let img = match image_io::load_rgba_from_memory(&req.body) {
        Ok(img) => img,
        Err(e) => return send_response(stream, &ApiResponse::error(400, &e)),
    };
    let response = api::session_response(&img);
    *SESSION.lock().unwrap() = Some(Arc::new(img));
    send_response(stream, &response)
}

fn handle_convert(req: &Request, stream: &mut TcpStream, preview: bool) -> std::io::Result<()> {
    let settings = std::str::from_utf8(&req.body).unwrap_or("");
    let src = SESSION.lock().unwrap().clone();
    let response = api::convert_response(src.as_deref(), settings, preview, Limits::default());
    send_response(stream, &response)
}

fn handle_palette(req: &Request, stream: &mut TcpStream) -> std::io::Result<()> {
    let settings = std::str::from_utf8(&req.body).unwrap_or("");
    let src = SESSION.lock().unwrap().clone();
    let response = api::palette_response(src.as_deref(), settings, Limits::default());
    send_response(stream, &response)
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

fn send_response(stream: &mut TcpStream, response: &ApiResponse) -> std::io::Result<()> {
    let status = match response.status {
        200 => "200 OK",
        400 => "400 Bad Request",
        409 => "409 Conflict",
        _ => "500 Internal Server Error",
    };
    let headers: Vec<&str> = response.headers.iter().map(String::as_str).collect();
    send(
        stream,
        status,
        response.content_type,
        &headers,
        &response.body,
    )
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
