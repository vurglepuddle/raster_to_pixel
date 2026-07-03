# Raster to Pixel

Rust CLI for converting fuzzy raster images into deliberate pixel-art PNGs.

## Quick Start

```cmd
cargo run -- input.png output.png --pixel-size 5 --colors 16 --scale 8
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
--size 64           :: target long side, ignored when --pixel-size is set
--colors 16         :: adaptive palette size
--palette pico8     :: built-in palette: pico8, gameboy, sweetie16
--palette file.hex  :: Lospec-style RRGGBB hex list
--dither none       :: no dithering
--dither bayer4     :: ordered 4x4 Bayer dithering
--dither bayer8     :: ordered 8x8 Bayer dithering
--dither-strength .35
--scale 8           :: nearest-neighbor preview scale
--cell detail       :: default hybrid detail-preserving downsample mode
--compare           :: write original/result side-by-side
```
