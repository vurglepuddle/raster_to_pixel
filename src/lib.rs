//! Core algorithms for Raster_to_Pixel. Everything here is pure std,
//! deterministic, and operates on plain f32 buffers so it stays decoupled
//! from whatever image I/O crate gets approved. See PLAN.md.

pub mod color;
pub mod dither;
pub mod downsample;
pub mod kmeans;
pub mod palettes;
