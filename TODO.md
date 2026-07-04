# Raster_to_Pixel TODO

Local planning notes. This file is intentionally ignored by git for now.

## Current Status

- Core conversion works: Oklab color math, deterministic k-means, fixed palettes, Bayer dithering, downsampling modes, alpha threshold, compare export, CLI, and local GUI.
- Auto pixel-size detection exists and works on `examples/source_fuzzy.png` as `5.00`.
- Median/detail color artifact was fixed by snapping the median target to a real source color inside the cell.
- Grid phase snapping is implemented for explicit pixel-size and auto-pixel-size modes. It scores edge energy by phase, samples exact pixel-size cells from the detected phase, and can be disabled with `--no-snap-grid` / the GUI Snap grid checkbox.
- Dominant/detail downsampling now has a dominance threshold. Weak winning buckets fall back to mean color; CLI and GUI expose the threshold.
- GUI viewer has pinned overlays, drag-pan, wheel zoom, and a dense bottom palette strip showing up to 256 colors.
- Adaptive palette building collapses very light, low-chroma generated white noise into one canonical white before k-means.
- Adaptive palette building also collapses very dark generated noise into the darkest matching source color before k-means.
- Palette cleanup thresholds are exposed through CLI flags and GUI sliders.
- Fable batch pass (2026-07): alpha modes (`preserve`/`binary`/`background-fill`/`color-key`)
  with edge flood fill + transparent-RGB decontamination in `src/alpha.rs`; opt-in
  morphology presets (pinholes/halo/jaggy/orphans, protect-details default on) in
  `src/morphology.rs`; dominant cells use two shifted 5-bit bucket grids and snap to a
  real cell color; `--auto-colors` with preset clamping; snap-grid phase confidence
  (pairwise phase scoring) + manual `--phase-x`/`--phase-y`; `--debug-json` /
  `--debug-grid`; GUI: alpha mode/tolerance/key controls, Cleanup presets, Auto colors
  chip, conf readout, pixel-grid overlay toggle. NOT yet field-tested on real images.
- Fable batch pass 2 (2026-07): Wu 1992 quantizer on an Oklab lattice (`src/wu.rs`,
  `--quantizer wu`); post-quantize palette merge keeping heaviest anchors
  (`kmeans::merge_close_entries`, `--palette-merge`); opt-in contrast expansion
  protecting 1px details (`src/enhance.rs`, `--contrast-expansion`, GUI "Detail
  guard"); outline repair/enforce with auto-detected dark outline color
  (`src/outline.rs`, `--outline`). All wired through Config + CLI + GUI. NOT yet
  field-tested; tests written but not executed (compile-checked only).
- Codex pass (2026-07): palette import/export supports Lospec `.hex` and GIMP
  `.gpl`; CLI `--palette-out` writes `.hex`, `.gpl`, or a 1-row PNG strip; GUI
  can download the current result palette as `.hex`; batch folder mode writes one
  PNG per supported top-level input image, with optional per-image palette/debug
  sidecar directories.

## Reference Repo Scan

- `proper-pixel-art` - MIT.
  Useful ideas:
  - Canny + Hough grid-line detection.
  - Mesh homogenization from detected line gaps.
  - Vectorized cell maps for repeated frames.
  - Two-offset RGB binning for dominant color to avoid bin-boundary artifacts.
  - Majority-transparent cell rule.
  Caution:
  - Full mesh/Hough approach is heavier and was reportedly bad in older tests. Treat as later/optional.

- `spritefusion-pixel-snapper` - MIT.
  Useful ideas:
  - Quantize first, then compute edge profiles.
  - Estimate step from profile peaks.
  - Walk cuts along edge energy instead of enforcing a perfectly uniform grid.
  - Stabilize x/y axes if one detector goes off.
  Caution:
  - Elastic cut walking can distort geometry. Good reference, but likely not first implementation.

- `unfake` - MIT.
  Useful ideas:
  - Simple grid phase snapping: for a known scale, score each offset modulo scale by summed edge energy.
  - Runs detector plus edge detector fallback.
  - Binary alpha, morphology cleanup, jaggy cleanup, edge flood-fill background removal.
  - Auto color-count heuristic.
  Caution:
  - Uses OpenCV/Python/Numpy; reimplement concepts in Rust/std instead of adding dependencies.

- `pixel-aid` - AGPL-3.0.
  Useful ideas only; do not copy code into this MIT project unless we intentionally relicense/comply.
  - Grid candidates with scale, phase, source rect, confidence, and diagnostics.
  - Sobel tile voting for sparse foreground images.
  - Alpha modes: preserve, binary, color key, background flood fill.
  - Halo/matte cleanup, tiny-hole/component morphology, cleanup variants.
  - Good testing philosophy: fixtures for halos, matte backgrounds, sheets, and palette drift.

## Fable High-Impact Batch Target

Goal: give Fable one large math/algorithm pass that folds in the strongest ideas from the four reference repos without adding dependencies or copying incompatible code.

Constraints:
- Keep the project dependency-light. Do not add crates unless the user explicitly approves.
- Reimplement concepts in Rust/std and existing project style.
- MIT repos can inform implementation; `pixel-aid` is AGPL-3.0, so use ideas only and do not copy code.
- Preserve the shared `pipeline::Config` path so CLI and GUI stay aligned.
- Add focused unit tests for every risky behavior. Prefer synthetic fixtures over depending on ignored example images.

Requested batch:
- Alpha/background cleanup:
  - alpha modes: `preserve`, `binary`, `background-fill`, optional `color-key`.
  - edge/corner flood-fill background removal with tolerance.
  - transparent RGB decontamination to `[0, 0, 0, 0]`.
  - tests for enclosed islands, matte halos, and transparent RGB not tinting results.
- Morphology cleanup:
  - conservative `remove-orphans`.
  - `fill-pinholes`.
  - `jaggy-clean`.
  - `halo-clean` / matte edge cleanup.
  - opt-in presets: Conservative / Balanced / Aggressive.
  - protect intentional single-pixel details by default.
- Better dominant/detail representative:
  - two shifted 5-bit bucket grids.
  - pick the grid with the strongest cluster.
  - representative should be nearest real source color or robust weighted mean/median.
  - tests for bucket-boundary splits and fuzzy edge colors.
- Auto color count:
  - `--auto-colors`.
  - coarse bucket/significance heuristic.
  - clamp to useful presets like 16/32/64/128/256.
  - GUI Auto chip and readout.
- Diagnostics:
  - debug JSON and/or debug grid image showing pixel size, phase, output dims, candidate/confidence scores.
  - GUI grid overlay toggle if cheap.
- Optional grid upgrades:
  - confidence score for current snap-grid.
  - manual `--phase-x` / `--phase-y`.
  - consider elastic cut walking only if it is conservative and does not distort geometry.

Nice-to-have if time:
- Batch CLI flow.
- Export palette file / manifest metadata.
- Performance guardrails for high color counts and large previews.

## Priority Plan

### 1. Grid Phase Snapping - Implemented, Needs Field Testing

Goal: make `--pixel-size 4` or auto-detected sizes sample from the best grid offset instead of always from `(0, 0)`.

- Done:
  - Internal grid phase detection.
  - Phase-aware downsampling grid.
  - CLI `--no-snap-grid`.
  - GUI Snap grid checkbox.
  - `X-Grid-Phase` preview/export header.
  - Synthetic test for offset phase.
  - Manual check: `source_fuzzy.png --pixel-size 4 --colors 32 --cell detail` snaps to phase `0,2` and outputs `100x54`.
- Follow-up:
  - Test on more generated examples.
  - Consider manual `--phase-x` / `--phase-y` if auto phase is wrong.
  - Consider a debug overlay showing dropped stray strips.

### 2. Diagnostics Overlay / Debug Export

Goal: make grid decisions visible when auto/snap feels wrong.

- Add optional debug output for:
  - detected pixel size
  - phase
  - output dimensions
  - edge profile/candidate scores if cheap
- CLI option ideas:
  - `--debug-json path`
  - `--debug-grid path.png`
- GUI:
  - Add a small readout: `px 5.00`, `phase 1,0`, confidence if we compute it.
  - Later: grid overlay toggle on preview.

### 3. Alpha And Background Cleanup

Goal: make fuzzy generated sprites with backgrounds easier to clean before/after pixelation.

- Add alpha modes:
  - `preserve` current behavior.
  - `binary` hard 0/255 alpha.
  - `background-fill` flood-fill from image edges/corners and make matching background transparent.
  - Maybe `color-key #rrggbb`.
- Ensure transparent pixels are decontaminated to `[0, 0, 0, 0]`.
- Add tests for:
  - edge flood fill does not eat enclosed same-color interior islands.
  - binary alpha preserves opaque subject.
  - transparent RGB does not tint box/downsample.
- GUI:
  - Replace plain alpha threshold with alpha mode + threshold/tolerance.

### 4. Better Dominant Cell Representative - Partly Implemented

Goal: reduce weird color choices around fuzzy edges without losing crispness.

- Done:
  - Added `dominant_threshold` defaulting to `0.25`.
  - If the winning bucket is below threshold, Dominant falls back to alpha-weighted mean.
  - Added CLI `--dominant-threshold`.
  - Added GUI Dominance slider for Detail/Dominant.
  - Added regression for weak/tied dominant buckets.
- Done:
  - Collapse very light, low-chroma near-whites before adaptive k-means so `#fefefe`, `#fcfbfb`, and `#f7f6f5` spend one palette slot instead of several.
  - Collapse very dark near-blacks before adaptive k-means to the darkest source color in that range, avoiding invented pure black.
  - Added tests for generated white/dark noise collapse, preserving warmer off-white, and preserving readable dark colors.
  - Add advanced controls for highlight/shadow collapse thresholds:
    `--highlight-collapse`, `--shadow-collapse`, and GUI sliders/toggles. Defaults should match current conservative behavior, but users can collapse more or disable it.
- Remaining:
  - Consider broader palette pre-clustering / significance weighting if other background color families waste palette slots.
  - Upgrade `CellMode::Dominant` from one 5-bit bucket grid to two shifted grids.
  - Pick whichever grid gives the strongest dominant cluster.
  - Choose the representative from the dominant cluster, ideally nearest real color or robust mean/median.
  - Add regression for colors split across a bucket boundary.

### 5. Auto Color Count

Goal: avoid manual color-count guessing for ordinary generated pixel art.

- Add `--auto-colors`.
- Start simple:
  - downsample/blur mentally equivalent by sampling, no new deps.
  - quantize coarse RGB buckets.
  - count significant buckets above a percentage threshold.
  - clamp to sane presets like 16/32/64/128.
- GUI:
  - Add `Auto` chip next to color presets.
  - Display chosen color count.

### 6. Optional Cleanup Passes

Goal: help with the last annoying artifacts without making the default destructive.

- `remove-orphans`: remove isolated alpha pixels, with preserve-single-pixel-detail default on.
- `fill-pinholes`: fill tiny transparent holes in opaque regions.
- `jaggy-clean`: remove isolated diagonal-only pixels.
- `halo-clean`: replace/clear matte-like edge pixels near transparent/background areas.
- Keep these opt-in or exposed as Conservative/Balanced/Aggressive variants.

### 7. Larger/Maybe Later

- Irregular mesh/cut walking based on SpriteFusion/proper-pixel-art ideas.
- Batch processing in CLI and/or GUI.
- Animation/video/spritesheet workflows.
- Export palette files and manifest metadata.
- Performance cap for high color counts or preview k-means iterations.

## Deliberately Unfinished Plug-in Points

- Elastic cut walking (SpriteFusion-style): skipped on purpose — every conservative
  variant still risks distorting geometry. If ever added, hang it off
  `pipeline::GridPlan::sampling` by replacing the uniform `SamplingGrid` with
  per-axis cut lists; `downsample_grid_with_dominant_threshold` would take
  `&[f64]` cut positions instead of origin+step.
- Sobel edge-aware cell picking still plugs into `downsample::reduce_cell` (see
  module doc comment).
- Dark-matte halo cleanup: `morphology::halo_clean` only targets light neutrals so
  intentional dark outlines survive; a user-supplied matte color would generalize it.
- GUI does not expose `--phase-x`/`--phase-y` yet (the server already parses
  `phaseX`/`phaseY` form fields).

## Next Best Step

Field-test the batch pass on real generated images: background-fill tolerances,
cleanup presets on fuzzy sprites, auto-colors choices, and whether the new pairwise
phase scoring still snaps `source_fuzzy.png --pixel-size 4` to phase `0,2`.
