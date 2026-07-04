# Raster to Pixel

Rust CLI and local GUI for converting fuzzy raster images into deliberate pixel-art PNGs.

Licensed under MIT. See `LICENSE` and `NOTICE`.

## Quick Start

```cmd
cargo run -- input.png output.png --pixel-size 5 --colors 16 --scale 8
```

Local GUI:

```cmd
cargo run --bin gui -- --chrome
```

Auto pixel-size:

```cmd
cargo run -- input.png output.png --auto-pixel-size --colors 16 --scale 8
```

Fixed palette:

```cmd
cargo run -- input.png output.png --pixel-size 5 --palette pico8 --dither bayer4 --scale 8
```

Compare sheet:

```cmd
cargo run -- input.png compare.png --pixel-size 5 --colors 16 --scale 8 --compare
```

Useful knobs:

```cmd
--pixel-size 5      :: estimated source pixels per output pixel
--auto-pixel-size   :: estimate source pixels per output pixel from image structure
--size 64           :: target long side, ignored when --pixel-size is set
--colors 16         :: adaptive palette size
--palette pico8     :: built-in palette: pico8, gameboy, sweetie16
--palette file.hex  :: Lospec-style RRGGBB hex list
--dither none       :: no dithering
--dither bayer4     :: ordered 4x4 Bayer dithering
--dither bayer8     :: ordered 8x8 Bayer dithering
--dither-strength .35
--scale 8           :: nearest-neighbor preview scale
--cell detail       :: box, median, detail, or dominant
--dominant-threshold .25
--highlight-collapse .03
--shadow-collapse .16
--no-snap-grid      :: disable grid phase snapping for pixel-size modes
--compare           :: write original/result side-by-side
```

## How It Works

The pipeline picks a target grid, downsamples in linear RGB, quantizes in Oklab,
optionally applies Bayer dithering, then writes the raw grid or a nearest-neighbor
scaled preview.

The default `detail` cell mode chooses between median and dominant per cell:

- Median uses the per-channel median as a target, then snaps to the nearest real
  source color from that cell. This avoids synthetic fringe colors.
- Dominant groups near colors into 32-level RGB buckets, averages the winning bucket,
  and falls back to a box average when the winner is below `--dominant-threshold`.
- Cells use fractional coverage weights and alpha-weighted color stats, so partial
  source pixels and transparent edges are handled consistently.
- Pixel-size modes can snap the sampling grid to the strongest detected edge phase,
  which helps when the source grid is offset by a few pixels.
- Adaptive palettes merge generated near-white and near-black noise before k-means.
  Near-whites collapse to white; near-blacks collapse to the darkest source color
  instead of inventing pure black.
