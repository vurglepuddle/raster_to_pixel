#!/usr/bin/env bash
# Build the static site into dist/. No bundler: wasm-pack compiles the crate and the
# frontend is copied as-is, so dist/index.html is byte-identical to gui/index.html.
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$PWD/.tools/wasm-pack-0.15.0:$PWD/.tools/wasm-bindgen-0.2.129:$PATH"
# no-install: fail rather than let wasm-pack download a tool that isn't hash-pinned.
wasm-pack build . --mode no-install --target web --release --no-typescript --no-pack -- --locked

dist=../dist
mkdir -p "$dist/pkg"
cp ../gui/index.html www/web_backend.js www/worker.js "$dist/"
cp pkg/web_wasm.js pkg/web_wasm_bg.wasm "$dist/pkg/"
ls -l "$dist" "$dist/pkg"
