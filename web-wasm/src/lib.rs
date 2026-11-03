//! wasm-bindgen surface of the browser build. All conversion logic lives in the shared
//! `raster_to_pixel` library; this file only stores the source and enforces the limits
//! that wasm32 memory imposes. `www/worker.js` is the only caller.

use std::cell::RefCell;

use raster_to_pixel::{
    api::{self, ApiResponse, Limits},
    image_io::{self, RgbaImage},
};
use wasm_bindgen::prelude::*;

/// Measured, conversion uses about 23 bytes per source pixel when the output grid is much
/// smaller than the source, which fits wasm32's 4 GiB address space at this size. When
/// Auto finds no grid it falls back to a 1:1 output at about 115 bytes per pixel, which
/// runs out between 20 and 36 MP; that fails with an error suggesting Fixed size mode,
/// and the app recovers. Refusing those images outright would also block the sizes and
/// modes that work. The largest single buffer also stays under wasm32's 2 GiB cap.
const MAX_SOURCE_PIXELS: u64 = 64_000_000;
const LIMITS: Limits = Limits {
    max_output_pixels: Some(100_000_000),
};

thread_local! {
    static SOURCE: RefCell<Option<RgbaImage>> = const { RefCell::new(None) };
}

#[wasm_bindgen]
extern "C" {
    /// Defined by worker.js so a crash can be reported with its reason.
    #[wasm_bindgen(js_namespace = globalThis, js_name = pixelinePanic)]
    fn report_panic(message: &str);
}

#[wasm_bindgen(start)]
fn start() {
    std::panic::set_hook(Box::new(|info| report_panic(&info.to_string())));
}

/// An `api::ApiResponse` handed to JavaScript. `headers` is newline-separated.
#[wasm_bindgen]
pub struct Reply(ApiResponse);

#[wasm_bindgen]
impl Reply {
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> u16 {
        self.0.status
    }

    #[wasm_bindgen(getter)]
    pub fn content_type(&self) -> String {
        self.0.content_type.to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn headers(&self) -> String {
        self.0.headers.join("\n")
    }

    /// Moves the body out rather than holding a second copy in wasm memory.
    pub fn take_body(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.0.body)
    }
}

/// An error reply if this encoded image's header announces too many pixels.
#[wasm_bindgen]
pub fn source_size_error(bytes: &[u8]) -> Option<Reply> {
    let (width, height) = image_io::probe_dimensions(bytes).ok()?;
    size_error(width, height).map(Reply)
}

/// Decode with the Rust decoders (PNG, WebP, and JPEG when the browser's can't be used).
#[wasm_bindgen]
pub fn load_encoded(bytes: &[u8]) -> Reply {
    if let Some(reply) = source_size_error(bytes) {
        return reply;
    }
    match image_io::load_rgba_from_memory(bytes) {
        Ok(img) => store(img),
        Err(e) => Reply(ApiResponse::error(400, &e)),
    }
}

/// Accept RGBA8 pixels from the browser's JPEG decoder.
#[wasm_bindgen]
pub fn load_decoded(width: u32, height: u32, rgba: Vec<u8>) -> Reply {
    if let Some(reply) = size_error(width, height) {
        return Reply(reply);
    }
    match image_io::rgba_from_raw(width, height, rgba) {
        Ok(img) => store(img),
        Err(e) => Reply(ApiResponse::error(400, &e)),
    }
}

#[wasm_bindgen]
pub fn preview(settings: &str) -> Reply {
    with_source(|src| api::convert_response(src, settings, true, LIMITS))
}

#[wasm_bindgen]
pub fn export_png(settings: &str) -> Reply {
    with_source(|src| api::convert_response(src, settings, false, LIMITS))
}

#[wasm_bindgen]
pub fn export_palette(settings: &str) -> Reply {
    with_source(|src| api::palette_response(src, settings, LIMITS))
}

fn store(img: RgbaImage) -> Reply {
    let reply = Reply(api::session_response(&img));
    SOURCE.with_borrow_mut(|source| *source = Some(img));
    reply
}

fn with_source(run: impl FnOnce(Option<&RgbaImage>) -> ApiResponse) -> Reply {
    SOURCE.with_borrow(|source| Reply(run(source.as_ref())))
}

fn size_error(width: u32, height: u32) -> Option<ApiResponse> {
    (width as u64 * height as u64 > MAX_SOURCE_PIXELS).then(|| {
        ApiResponse::error(
            400,
            &format!(
                "this image is {width}x{height}; the web version handles up to {} megapixels. \
                 Shrink it first, or use the native app",
                MAX_SOURCE_PIXELS / 1_000_000
            ),
        )
    })
}
