//! Core algorithms for Raster_to_Pixel. The leaf modules (`color`, `dither`,
//! `downsample`, `kmeans`, `palettes`) are pure std, deterministic, and operate on
//! plain f32 buffers. `pipeline` sits on top and orchestrates a full conversion using
//! the (approved, committed) `image` crate; both the CLI and the GUI server call it so
//! they can't drift. See PLAN.md (algorithms) and GUI_PLAN.md (GUI).

pub mod color;
pub mod dither;
pub mod downsample;
pub mod kmeans;
pub mod palettes;
pub mod pipeline;
