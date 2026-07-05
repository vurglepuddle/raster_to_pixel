# Raster to Pixel

Rust CLI and local GUI for converting fuzzy raster images into deliberate pixel-art PNGs.

Licensed under MIT. See `LICENSE` and `NOTICE`.

## GUI

```cmd
cargo run --bin gui -- --chrome
```

![Pixeline GUI in dark mode](screenshots/dark_mode.png)

![Pixeline GUI in light mode](screenshots/light_mode.png)

## Quick Start

For most images, start with auto grid and auto colors:

```cmd
cargo run --bin raster_to_pixel -- input.png output.png --auto-pixel-size --auto-colors --scale 8
```

For a folder:

```cmd
cargo run --bin raster_to_pixel -- examples batch_out --auto-pixel-size --auto-colors --scale 8
```

Use the GUI when you want to compare modes visually. Use the CLI for repeatable
settings, batch conversion, palette export, and debug sidecars.

## Useful Options

Pick the output grid:

```cmd
--auto-pixel-size      Guess the source pixel scale. Use this first.
--pixel-size 5         Force the scale: each output pixel represents 5 source pixels.
--size 64              Ignore pixel scale and make the long side 64 output pixels.
--no-snap-grid         Do not shift the sampling grid onto detected pixel boundaries.
--phase-x 2 --phase-y 0
                       Manually shift the sampling grid if detection lands off-phase.
```

Pick colors:

```cmd
--auto-colors          Pick 16/32/64/128/256 colors from the image.
--colors 32            Force an adaptive palette size instead of auto-colors, any integer as argument.
--palette pico8        Use a fixed palette instead of auto-colors/colors: pico8, gameboy, sweetie16, or a file.
--palette file.hex     Use a Lospec .hex list or GIMP .gpl palette.
--palette-out p.hex    Save the result palette as .hex, .gpl, or .png strip.
```

Tune adaptive palettes:

```cmd
--quantizer kmeans     Default adaptive palette builder; stable all-rounder.
--quantizer wu         Oklab Wu quantizer; can separate big color families better.
--palette-merge .04    Merge near-duplicate palette entries after quantization.
--highlight-collapse .03
                       Collapse generated near-white noise into one highlight.
--shadow-collapse .16  Collapse near-black noise into the darkest real source dark.
```

Pick how each source cell becomes one pixel:

```cmd
--cell detail          Default. Median on busy cells, dominant color on calm cells.
--cell median          Good for fuzzy edges; avoids odd dominant-color surprises.
--cell dominant        Crispest; best when each cell has one clear winning color.
--cell box             Simple weighted average; smoothest, least pixel-art-like.
--cell adaptive        Slower content-adaptive fit; sometimes helps very fuzzy art.
--dominant-threshold .25
                       How strong a winner must be before dominant/detail trusts it.
--adaptive-iterations 3
                       Passes for --cell adaptive. 1 is faster; 3-4 can be cleaner.
```

Add dithering or scale the saved PNG:

```cmd
--dither none          Plain nearest palette color.
--dither bayer4        Fine ordered dither.
--dither bayer8        Larger ordered dither pattern.
--dither-strength .35  How visible the ordered dither is.
--scale 8              Save each output pixel as an 8x8 block.
```

Clean alpha, edges, and small defects:

```cmd
--alpha-mode preserve  Keep source alpha as-is.
--alpha-mode binary    Force alpha to fully transparent or fully opaque.
--alpha-mode background-fill
                       Remove border-connected background within --bg-tolerance.
--alpha-mode color-key --color-key 00ff00
                       Remove one keyed color.
--bg-tolerance .10     Color tolerance for background-fill or color-key.
--cleanup balanced     Morphology cleanup: none, conservative, balanced, aggressive.
--no-protect-details   Let cleanup remove repeating single-pixel details too.
--contrast-expansion 1 Preserve tiny high-contrast details before downsampling.
--outline repair       Fill single-pixel gaps in a detected dark outline.
--outline enforce      Repaint the whole silhouette edge with the outline color.
```

Debug and compare:

```cmd
--compare              Write original/result side-by-side.
--debug-json d.json    Save grid, palette, cleanup, and fallback diagnostics.
--debug-grid g.png     Draw the sampling grid over the source image.
```

If `input` is a directory, Pixeline batch-converts supported images from that
folder into the output directory. In batch mode, `--palette-out`, `--debug-json`,
and `--debug-grid` are directories for per-image sidecars.

Custom palettes accept Lospec `.hex` and GIMP `.gpl` text. In the GUI, paste
them into the `Custom...` palette box.

## How It Works

The pipeline picks a target grid, downsamples in linear RGB, builds or applies a
palette in Oklab, optionally dithers, then writes the raw grid or a
nearest-neighbor scaled PNG.

The default `detail` cell mode chooses between median and dominant per cell:

- Median uses the per-channel median as a target, then snaps to the nearest real
  source color from that cell. This avoids synthetic fringe colors.
- Dominant groups near colors into two shifted 32-level RGB bucket grids (so a color
  family straddling a bucket boundary is never split), picks the stronger winner,
  snaps its weighted mean to a real cell color, and falls back to a box average when
  the winner is below `--dominant-threshold`.
- Cells use fractional coverage weights and alpha-weighted color stats, so partial
  source pixels and transparent edges are handled consistently.
- The opt-in `adaptive` cell mode keeps the detail split but lets high-contrast
  cells use a content-adaptive kernel fit (Kopf et al.-style EM in Oklab). It is
  slower than the other modes, still snaps to a real source color, and falls
  back to `detail` past a runtime budget.
- Pixel-size modes can snap the sampling grid to the strongest detected edge phase
  (with a reported confidence), which helps when the source grid is offset by a few
  pixels; `--phase-x`/`--phase-y` override the detection.
- Adaptive palettes merge generated near-white and near-black noise before
  quantization. Near-whites collapse to white; near-blacks collapse to the
  darkest source color instead of inventing pure black.

Optional cleanup stages:

- `--alpha-mode` prepares the source before conversion: `binary` hardens soft alpha,
  `background-fill` flood-fills the background from the image border with
  `--bg-tolerance` (enclosed islands survive), `color-key` keys out one color.
  Everything made transparent is decontaminated to `[0,0,0,0]`.
- `--cleanup` runs opt-in morphology on the finished grid: fill enclosed pinholes,
  drop light matte halos along edges, remove diagonal jaggy nubs, and delete
  isolated specks. Single-pixel details that repeat nearby are protected by default.
- `--quantizer wu` swaps the adaptive palette builder for Wu 1992 moment
  quantization running on an Oklab lattice. It can separate dominant color
  families better at 16-64 colors. `--palette-merge` then collapses surviving
  near-duplicate entries, keeping the heaviest real color of each group.
- `--contrast-expansion N` stamps tiny high-contrast details (eye pixels, glints)
  over their surroundings before downsampling so the cell vote cannot erase them.
- `--outline repair` detects the sprite's dark outline color along the silhouette
  and fills single-pixel gaps; `enforce` repaints the entire silhouette edge.
