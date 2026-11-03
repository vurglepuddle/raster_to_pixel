#!/usr/bin/env bash
# Install the pinned build tools into web-wasm/.tools (git-ignored, PATH untouched).
# Each download is checked against a SHA-256 recorded here before it is unpacked.
# Used by CI and for local builds. Update a version together with its hashes, and
# wasm-bindgen together with the exact pin in Cargo.toml.
set -euo pipefail
cd "$(dirname "$0")"

WASM_PACK=0.15.0
WASM_BINDGEN=0.2.129

case "$(uname -s)" in
  Linux)
    triple=x86_64-unknown-linux-musl
    wasm_pack_sha=c09f971ecaed9a2efc80fdcea7a00ef6b53c7fadc8c57d1f61b53a6aa66b668a
    wasm_bindgen_sha=82d12bb940e2d4e72e0d5605387fc1b8ca179044e012b620f0ce4e7440e8320e
    ;;
  MINGW* | MSYS* | CYGWIN*)
    triple=x86_64-pc-windows-msvc
    wasm_pack_sha=518dc51180c7bc864699c9279b3bc99025bc109123e0249a34ba34d130a509bf
    wasm_bindgen_sha=79e348c169d0c10c0b287647f1e753d0985353ca96800bc29be8f4261baf13e6
    ;;
  *)
    echo "no pinned tool hashes for $(uname -s); add them here" >&2
    exit 1
    ;;
esac

install_tool() { # <dir name> <url> <sha256>
  local dir=".tools/$1"
  [ -d "$dir" ] && return
  mkdir -p "$dir"
  curl -fsSL --proto '=https' --tlsv1.2 -o "$dir/download.tar.gz" "$2"
  if ! echo "$3  $dir/download.tar.gz" | sha256sum -c --quiet -; then
    rm -r "$dir"
    echo "checksum mismatch for $2" >&2
    exit 1
  fi
  tar -xzf "$dir/download.tar.gz" -C "$dir" --strip-components=1
  rm "$dir/download.tar.gz"
}

install_tool "wasm-pack-$WASM_PACK" \
  "https://github.com/wasm-bindgen/wasm-pack/releases/download/v$WASM_PACK/wasm-pack-v$WASM_PACK-$triple.tar.gz" \
  "$wasm_pack_sha"
install_tool "wasm-bindgen-$WASM_BINDGEN" \
  "https://github.com/wasm-bindgen/wasm-bindgen/releases/download/$WASM_BINDGEN/wasm-bindgen-$WASM_BINDGEN-$triple.tar.gz" \
  "$wasm_bindgen_sha"

rustup target add wasm32-unknown-unknown
