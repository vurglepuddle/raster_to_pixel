# Pixeline web build

The browser version of Pixeline: a static site with no server, no install, and no uploads.
It runs the same Rust conversion code as the native CLI and GUI, compiled to WebAssembly.

Live site: <https://vurglepuddle.github.io/raster_to_pixel/>

## How it fits together

There is one frontend and one implementation. Only the way a request travels differs:

```text
gui/index.html            the one frontend, byte-identical in both builds
   |  api('/api/preview', ...)
   |
   +-- native:  fetch ----------> src/bin/gui.rs (HTTP) ----+
   |                                                        +--> src/api.rs --> pipeline
   +-- browser: www/web_backend.js --> www/worker.js        |
                                       web-wasm/src/lib.rs -+
```

- `src/api.rs` holds the request handlers (settings parsing, response metadata, export
  formats). The native server and this crate are thin transports over it.
- `www/web_backend.js` runs on the page and answers the page's `/api/*` requests with the
  same statuses, headers, and bodies as the native server. The page finds it through a
  `<script>` tag; natively, the server returns that script as an empty file.
- `www/worker.js` runs the WebAssembly in a Web Worker, so conversions never block the page.
  It needs no cross-origin isolation or threads.
- `web-wasm/` is its own Cargo workspace, so nothing here touches the native `Cargo.lock`
  or build profile. `cargo run --bin gui` works exactly as before.

### Rapid changes and failures

A running WebAssembly call cannot be interrupted, so `web_backend.js` schedules around that:

- One job runs at a time. Exports queue in order and always finish.
- Previews are latest-wins: a newer preview replaces one that is still waiting.
- If a superseded preview keeps the worker busy past 1.5 s, the worker is replaced and the
  source reloaded, because terminating it is the only way to stop it.
- Each new image gets a fresh worker, because WebAssembly memory never shrinks.
- A crash or out-of-memory error fails that one job with a message. The next job gets a new
  worker, so the app keeps working.

## Build and test locally

The tools are pinned and checked against SHA-256 hashes in `install-tools.sh`. They install
into the git-ignored `web-wasm/.tools/`; your `PATH` and global tools are left alone. These
scripts need bash, which on Windows means Git Bash.

```sh
web-wasm/install-tools.sh     # Once. Also runs: rustup target add wasm32-unknown-unknown
web-wasm/build.sh             # Writes the site to dist/
```

To try the site, serve `dist/` over HTTP with any static file server, for example
`python -m http.server -d dist`. Opening `index.html` from disk won't work, because browsers
block module workers on `file://`.

To test it against native output (`examples` is optional and adds your real images):

```sh
cargo run --release --example parity_native -- target/parity examples
cargo build --release --bin gui && node web-wasm/tests/native_gui.mjs
node web-wasm/tests/parity.mjs --browser chrome    # Or edge, firefox
```

`parity.mjs` serves `dist/` under `/raster_to_pixel/` (failing on any request outside that
path), runs every native output through the real worker in a headless browser, then drives
the real page through its DOM. Useful options: `--only parity|ui`, `--fixtures <regex>`.
The tests use only Node's standard library: no npm packages.

## Deployment

`.github/workflows/web.yml` runs on every push and pull request: native tests, build, native
GUI test, and browser comparison. Pushes to `main` then publish `dist/` to GitHub Pages.

**One-time setup:** in the repository's Settings > Pages, set Source to "GitHub Actions".

All paths are relative, so the site works under `/raster_to_pixel/` or anywhere else.

## Output compared with the native app

Measured on Windows against the native build in Chrome, Edge, and Firefox, which all gave
identical results. The 7 synthetic images run 23 settings combinations that reach every
pipeline stage; the 16 real images in `examples/` run the 7 most revealing of those. Both
include PNG and palette exports. CI has no real images, so it runs the synthetic set.

| Input | Result |
| --- | --- |
| PNG (8-bit, 16-bit, with alpha) | **Byte-identical** PNG, palette, and metadata |
| WebP (lossless, lossy, with alpha) | **Byte-identical** PNG, palette, and metadata |
| JPEG | **Very close, not identical**: a different decoder (see below) |

CI compares against a Linux native build, where JPEGs use the Rust decoder instead of GDI+.

### Platform differences

**JPEG decoding.** JPEGs go through the browser's decoder, mirroring GDI+ in the native
Windows build, because the pure-Rust decoder has a real bug: baseline JPEGs with
non-interleaved scans (such as `examples/girl_8.jpeg`) come out neon green, and it behaves
the same natively and in WebAssembly. Measured against GDI+:

| Decoder | Normal JPEGs | `girl_8.jpeg` |
| --- | --- | --- |
| Chrome, Edge, Firefox (identical to each other) | 91-94% of pixels identical, at most 3 levels apart | Correct, 96% identical |
| Pure Rust (identical native and in WebAssembly) | 90-94% of pixels identical, at most 3 levels apart | **Neon green** |

Those tiny decoder differences then nudge the palette, so final JPEG output typically
differs from native Windows by a mean of 0.5 out of 255 per channel (worst tested case 1.9).
Most pixels move imperceptibly, and an isolated pixel can land on a neighboring palette
color. Output-grid dimensions matched native on every tested JPEG. Like the native build, the
worker ignores embedded color profiles (`colorSpaceConversion: 'none'`).

Non-JPEG files don't have this difference, because PNG and WebP use the same Rust decoders
everywhere.

**JPEG orientation.** Browsers apply EXIF orientation and the native decoders don't, so a
rotated phone photo comes out upright here and sideways natively. This is deliberate: the
result then matches the original the page displays, which keeps crop and compare aligned.

**Privacy browsers.** Firefox's `privacy.resistFingerprinting` and Brave's fingerprinting
protection scramble canvas readback, which would corrupt browser-decoded JPEGs. The worker
detects this and falls back to the Rust decoder. Normal JPEGs still work; non-interleaved
ones like `girl_8.jpeg` show the neon green bug in those browsers only. In Firefox, adding
the site to `privacy.resistFingerprinting.exemptedDomains` restores the browser decoder.

**Float math.** `f32` functions such as `powf` and `cbrt` call the platform's math library,
and those disagree in the last bit often enough to change a k-means palette. `src/fmath.rs`
computes in `f64` and rounds to `f32`, which gave bit-identical results on Windows and
WebAssembly over 2M inputs per function. This applies to the native build too, so native
output shifted very slightly when it was introduced: 73 of 80 test conversions stayed
byte-identical, the rest changed by a mean of at most 0.026 out of 255.

**Not yet tested:** Safari (it needs 16.4+ for `OffscreenCanvas`; its JPEG decoder is not
libjpeg-turbo, so expect similar small differences) and any phone.

### Limits

Everything has to fit WebAssembly's 4 GiB address space, and phones will run out sooner.

- **Source images:** up to 64 megapixels. Larger ones are refused from their header, before
  any decoding.
- **Exports:** up to 100 megapixels, counting scale and compare. Beyond that, the export
  fails with a message.

Memory and time depend mostly on the output grid. Measured in Chrome using Auto mode, with
other work running on the machine, so treat the times as rough:

| Source | Pixel grid found | No grid found (1:1 output) |
| --- | --- | --- |
| 9 MP | 209 MB, 3 s | 1 GB, about 7 min |
| 20 MP | 469 MB, 7 s | 2.3 GB, about 14 min |
| 36 MP | 832 MB, 13 s | Out of memory |
| 49 MP | 1.1 GB, 18 s | Out of memory |

When Auto finds no grid it falls back to 1:1 output, which runs k-means on every pixel. That
is slow in the native build too. Fixed size mode avoids it: a 64 MP source exported at 1024
pixels took 14 s. The page stays responsive throughout, and changing a setting replaces the
slow job.

If memory does run out, that job fails with a message suggesting Fixed size mode or a crop,
and the app keeps working. This was verified in Chrome and Firefox by forcing real
out-of-memory errors with a build capped at 96 MiB.

**Speed.** On real images, WebAssembly took about 1.5x as long as native. A typical default
conversion finishes in under a second.

## Privacy

Images are read from a local `File` and never sent anywhere; there is no analytics or
telemetry. The safeguards:

- The page's Content-Security-Policy blocks its connections to any other origin. GitHub
  Pages can't set headers, so it is a `<meta>` tag, which doesn't extend to the worker.
- The worker's only network request is loading the `.wasm` file from this site.
- The WebAssembly module's only imports are a panic reporter and an error-throw helper, so
  it has no way to reach the network or the page. A test fails if that list ever grows.
- The tests fail on any cross-origin page request, CSP violation, or request outside the
  site.

## Dependencies

Minimal on purpose, because of supply-chain risk. No frontend framework, bundler, or npm
package anywhere: the frontend is plain HTML and JavaScript using browser APIs.

| What | Version | Why |
| --- | --- | --- |
| `wasm-bindgen` (crate, exact pin) | 0.2.129 | JavaScript interop |
| `wasm-pack` (prebuilt, hash-pinned) | 0.15.0 | Build driver |
| `wasm-bindgen` CLI (prebuilt, hash-pinned) | 0.2.129 | Must match the crate exactly |

`wasm-bindgen` brings in `wasm-bindgen-macro`, `wasm-bindgen-macro-support`,
`wasm-bindgen-shared`, `bumpalo`, `once_cell`, `rustversion`, and a second major version of
`syn`. All are `MIT OR Apache-2.0`, and only this crate's `Cargo.lock` lists them. `wasm-opt`
is disabled, so `wasm-pack` downloads nothing extra. CI pins actions to commit hashes.

To update: pick a release that is at least two weeks old, change the version and both
SHA-256 hashes in `install-tools.sh`, update the exact pin in `Cargo.toml`, and re-run the
tests above.
