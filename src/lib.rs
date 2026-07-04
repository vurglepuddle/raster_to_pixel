//! Core algorithms for Raster_to_Pixel. The leaf modules (`color`, `dither`,
//! `downsample`, `kmeans`, `palettes`) are pure std, deterministic, and operate on
//! plain f32 buffers. `alpha` (source alpha/background cleanup) and `morphology`
//! (post-quantize grid cleanup) use the approved `image` crate, as does `pipeline`,
//! which sits on top and orchestrates a full conversion; both the CLI and the GUI
//! server call it so they can't drift. See PLAN.md (algorithms) and GUI_PLAN.md (GUI).

pub mod alpha;
pub mod color;
pub mod dither;
pub mod downsample;
pub mod enhance;
pub mod kmeans;
pub mod morphology;
pub mod outline;
pub mod palettes;
pub mod pipeline;
pub mod wu;
