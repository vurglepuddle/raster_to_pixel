//! Core algorithms for Raster_to_Pixel. The leaf modules (`color`, `dither`,
//! `downsample`, `kmeans`, `palettes`) are pure std, deterministic, and operate on
//! plain f32 buffers. `alpha` (source alpha/background cleanup) and `morphology`
//! (post-quantize grid cleanup) use the approved `image` crate, as does `pipeline`,
//! which sits on top and orchestrates a full conversion; the CLI, the GUI server, and
//! the browser build (`web-wasm`) all call it so they can't drift. `api` holds the
//! GUI's request handlers, shared by the two GUI transports, and `fmath` keeps float
//! results identical across platforms. See PLAN.md (algorithms) and GUI_PLAN.md (GUI).

pub mod alpha;
pub mod api;
pub mod color;
pub mod dither;
pub mod downsample;
pub mod enhance;
pub mod fmath;
pub mod image_io;
pub mod kmeans;
pub mod morphology;
pub mod outline;
pub mod palettes;
pub mod pipeline;
pub mod wu;
