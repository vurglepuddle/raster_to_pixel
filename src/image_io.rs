//! Image loading helpers shared by the CLI and GUI.
//!
//! Most formats go through the `image` crate. On Windows, JPEGs are decoded
//! through GDI+ because some generated JPEGs use scan/color layouts (baseline with
//! non-interleaved scans) that the current pure-Rust JPEG path decodes as neon green
//! while Windows viewers display them correctly. The browser build has the same
//! problem and solves it the same way: see `web-wasm/www/worker.js`.

use std::path::Path;

pub use image::RgbaImage;

pub fn load_rgba_from_path(path: &Path) -> Result<RgbaImage, String> {
    #[cfg(windows)]
    if is_jpeg_path(path) {
        if let Ok(img) = windows_gdiplus::decode_jpeg_path(path) {
            return Ok(img);
        }
    }

    image::ImageReader::open(path)
        .map_err(|e| format!("failed to open {}: {e}", path.display()))?
        .decode()
        .map_err(|e| format!("failed to decode {}: {e}", path.display()))
        .map(|img| img.to_rgba8())
}

pub fn load_rgba_from_memory(bytes: &[u8]) -> Result<RgbaImage, String> {
    #[cfg(windows)]
    if is_jpeg_bytes(bytes) {
        if let Ok(img) = windows_gdiplus::decode_jpeg_bytes(bytes) {
            return Ok(img);
        }
    }

    image::load_from_memory(bytes)
        .map_err(|e| format!("decode failed: {e}"))
        .map(|img| img.to_rgba8())
}

/// Read a source's dimensions from its header without decoding the pixels.
pub fn probe_dimensions(bytes: &[u8]) -> Result<(u32, u32), String> {
    // PNG's IHDR is at a fixed offset. Read it directly: on 32-bit targets the decoder
    // refuses huge images with a generic limit error before reporting their size.
    const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
    if let (true, Some(size)) = (bytes.starts_with(PNG_SIGNATURE), bytes.get(16..24)) {
        let [w0, w1, w2, w3, h0, h1, h2, h3] = size else {
            unreachable!("a 16..24 slice has 8 bytes")
        };
        return Ok((
            u32::from_be_bytes([*w0, *w1, *w2, *w3]),
            u32::from_be_bytes([*h0, *h1, *h2, *h3]),
        ));
    }
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("decode failed: {e}"))?;
    // Nothing is allocated for the pixels here, and a decode limit would hide the size.
    reader.no_limits();
    reader
        .into_dimensions()
        .map_err(|e| format!("decode failed: {e}"))
}

/// Wrap RGBA8 pixels decoded elsewhere (the browser's JPEG decoder).
pub fn rgba_from_raw(width: u32, height: u32, rgba: Vec<u8>) -> Result<RgbaImage, String> {
    RgbaImage::from_raw(width, height, rgba)
        .ok_or_else(|| format!("decoded pixels do not match a {width}x{height} image"))
}

#[cfg(windows)]
fn is_jpeg_path(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| matches!(s.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
        .unwrap_or(false)
}

#[cfg(windows)]
fn is_jpeg_bytes(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xff, 0xd8])
}

#[cfg(windows)]
mod windows_gdiplus {
    use std::{
        ffi::OsStr,
        os::windows::ffi::OsStrExt,
        path::{Path, PathBuf},
        ptr::null_mut,
        time::{SystemTime, UNIX_EPOCH},
    };

    use image::{Rgba, RgbaImage};

    #[repr(C)]
    struct GdiplusStartupInput {
        gdiplus_version: u32,
        debug_event_callback: *mut std::ffi::c_void,
        suppress_background_thread: i32,
        suppress_external_codecs: i32,
    }

    #[link(name = "gdiplus")]
    unsafe extern "system" {
        fn GdiplusStartup(
            token: *mut usize,
            input: *const GdiplusStartupInput,
            output: *mut std::ffi::c_void,
        ) -> i32;
        fn GdiplusShutdown(token: usize);
        fn GdipCreateBitmapFromFile(
            filename: *const u16,
            bitmap: *mut *mut std::ffi::c_void,
        ) -> i32;
        fn GdipGetImageWidth(image: *mut std::ffi::c_void, width: *mut u32) -> i32;
        fn GdipGetImageHeight(image: *mut std::ffi::c_void, height: *mut u32) -> i32;
        fn GdipBitmapGetPixel(
            bitmap: *mut std::ffi::c_void,
            x: i32,
            y: i32,
            color: *mut u32,
        ) -> i32;
        fn GdipDisposeImage(image: *mut std::ffi::c_void) -> i32;
    }

    pub fn decode_jpeg_path(path: &Path) -> Result<RgbaImage, String> {
        let _gdiplus = Gdiplus::start()?;
        let wide = wide_path(path);
        let mut bitmap = null_mut();
        check(
            unsafe { GdipCreateBitmapFromFile(wide.as_ptr(), &mut bitmap) },
            "GdipCreateBitmapFromFile",
        )?;
        if bitmap.is_null() {
            return Err("GDI+ returned a null bitmap".into());
        }
        let result = unsafe { bitmap_to_rgba(bitmap) };
        unsafe {
            GdipDisposeImage(bitmap);
        }
        result
    }

    pub fn decode_jpeg_bytes(bytes: &[u8]) -> Result<RgbaImage, String> {
        let path = temp_jpeg_path();
        std::fs::write(&path, bytes)
            .map_err(|e| format!("failed to write temporary JPEG {}: {e}", path.display()))?;
        let decoded = decode_jpeg_path(&path);
        let _ = std::fs::remove_file(&path);
        decoded
    }

    struct Gdiplus {
        token: usize,
    }

    impl Gdiplus {
        fn start() -> Result<Self, String> {
            let input = GdiplusStartupInput {
                gdiplus_version: 1,
                debug_event_callback: null_mut(),
                suppress_background_thread: 0,
                suppress_external_codecs: 0,
            };
            let mut token = 0usize;
            check(
                unsafe { GdiplusStartup(&mut token, &input, null_mut()) },
                "GdiplusStartup",
            )?;
            Ok(Self { token })
        }
    }

    impl Drop for Gdiplus {
        fn drop(&mut self) {
            unsafe {
                GdiplusShutdown(self.token);
            }
        }
    }

    unsafe fn bitmap_to_rgba(bitmap: *mut std::ffi::c_void) -> Result<RgbaImage, String> {
        let mut width = 0u32;
        let mut height = 0u32;
        check(GdipGetImageWidth(bitmap, &mut width), "GdipGetImageWidth")?;
        check(
            GdipGetImageHeight(bitmap, &mut height),
            "GdipGetImageHeight",
        )?;
        let mut out = RgbaImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let mut argb = 0u32;
                check(
                    GdipBitmapGetPixel(bitmap, x as i32, y as i32, &mut argb),
                    "GdipBitmapGetPixel",
                )?;
                out.put_pixel(
                    x,
                    y,
                    Rgba([
                        ((argb >> 16) & 0xff) as u8,
                        ((argb >> 8) & 0xff) as u8,
                        (argb & 0xff) as u8,
                        ((argb >> 24) & 0xff) as u8,
                    ]),
                );
            }
        }
        Ok(out)
    }

    fn check(status: i32, what: &str) -> Result<(), String> {
        if status == 0 {
            Ok(())
        } else {
            Err(format!("{what} failed with GDI+ status {status}"))
        }
    }

    fn wide_path(path: &Path) -> Vec<u16> {
        OsStr::new(path)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }

    fn temp_jpeg_path() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "raster_to_pixel_{}_{}.jpg",
            std::process::id(),
            nanos
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reads_huge_png_dimensions_without_decoding() {
        // Header of a 30000x30000 RGBA PNG whose pixel data is far too short.
        let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
        png.extend(30000u32.to_be_bytes());
        png.extend(30000u32.to_be_bytes());
        assert_eq!(probe_dimensions(&png), Ok((30000, 30000)));
    }

    #[test]
    fn probe_rejects_garbage_and_raw_pixels_must_match_their_size() {
        assert!(probe_dimensions(&[1, 2, 3, 4]).is_err());
        assert!(rgba_from_raw(2, 2, vec![0; 16]).is_ok());
        assert!(rgba_from_raw(2, 2, vec![0; 15]).is_err());
    }
}
