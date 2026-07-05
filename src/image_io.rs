//! Image loading helpers shared by the CLI and GUI.
//!
//! Most formats go through the `image` crate. On Windows, JPEGs are decoded
//! through GDI+ because some generated JPEGs use scan/color layouts that the
//! current pure-Rust JPEG path decodes as neon green while Windows viewers
//! display them correctly.

use std::path::Path;

use image::RgbaImage;

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

fn is_jpeg_path(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| matches!(s.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
        .unwrap_or(false)
}

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
