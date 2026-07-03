# Raster to Pixel

Rust CLI for converting fuzzy raster images into deliberate pixel-art PNGs.

## Quick Start

```cmd
cargo run -- input.png output.png --pixel-size 5 --colors 16 --scale 8
```

Useful knobs:

```cmd
--pixel-size 5      :: estimated source pixels per output pixel
--size 64           :: target long side, ignored when --pixel-size is set
--colors 16         :: adaptive palette size
--scale 8           :: nearest-neighbor preview scale
--cell detail       :: default hybrid detail-preserving downsample mode
```
