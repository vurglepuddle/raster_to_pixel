// Runs the WebAssembly build off the main thread so sliders never freeze the page.
// web_backend.js sends one job at a time; a crashed instance cannot be reused, so any
// trap is reported as fatal and the backend replaces this whole worker.

import init, {
  export_palette,
  export_png,
  load_decoded,
  load_encoded,
  preview,
  source_size_error,
} from './pkg/web_wasm.js';

let panicMessage = null;
globalThis.pixelinePanic = (message) => {
  panicMessage = message;
  console.error(message);
};

const ready = init();

const isJpeg = (bytes) => bytes[0] === 0xff && bytes[1] === 0xd8;

// True if canvas readback returns exactly what was drawn. Fingerprinting protection
// (Firefox resistFingerprinting, Brave) adds noise, which would corrupt the source.
function canvasReadbackIsExact() {
  const context = new OffscreenCanvas(2, 1).getContext('2d', { willReadFrequently: true });
  context.fillStyle = '#123456';
  context.fillRect(0, 0, 1, 1);
  context.fillStyle = '#fedcba';
  context.fillRect(1, 0, 1, 1);
  const px = context.getImageData(0, 0, 2, 1).data;
  return [0x12, 0x34, 0x56, 255, 0xfe, 0xdc, 0xba, 255].every((v, i) => px[i] === v);
}

// JPEGs go through the browser's decoder when possible, mirroring GDI+ in the native
// Windows build: the pure-Rust decoder renders baseline files with non-interleaved
// scans neon green. Returns null to fall back to Rust. See web-wasm/README.md.
async function decodeJpegInBrowser(bytes) {
  let bitmap;
  try {
    if (!canvasReadbackIsExact()) return null;
    bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }), {
      colorSpaceConversion: 'none', // Match native, which ignores embedded ICC profiles.
      premultiplyAlpha: 'none',
    });
    const { width, height } = bitmap;
    const context = new OffscreenCanvas(width, height).getContext('2d', {
      colorSpace: 'srgb',
      willReadFrequently: true,
    });
    context.drawImage(bitmap, 0, 0);
    const { data } = context.getImageData(0, 0, width, height);
    // JPEGs are opaque, so transparency means the draw failed (e.g. iOS canvas limits).
    if (data[3] !== 255 || data[data.length - 1] !== 255) return null;
    return { width, height, rgba: new Uint8Array(data.buffer) };
  } catch (error) {
    console.warn('browser JPEG decode failed; using the Rust decoder', error);
    return null;
  } finally {
    bitmap?.close();
  }
}

async function loadSource(bytes) {
  if (isJpeg(bytes)) {
    const tooLarge = source_size_error(bytes);
    if (tooLarge) return tooLarge;
    const decoded = await decodeJpegInBrowser(bytes);
    if (decoded) return load_decoded(decoded.width, decoded.height, decoded.rgba);
  }
  return load_encoded(bytes);
}

const OPS = { preview, export: export_png, palette: export_palette };

self.onmessage = async ({ data: { id, op, settings, file } }) => {
  try {
    await ready;
    const reply = op === 'load'
      ? await loadSource(new Uint8Array(await file.arrayBuffer()))
      : OPS[op](settings);
    const message = {
      id,
      status: reply.status,
      contentType: reply.content_type,
      headers: reply.headers,
      body: reply.take_body(),
    };
    reply.free();
    self.postMessage(message, [message.body.buffer]);
  } catch (error) {
    // A wasm trap with no panic message is an allocation failure: Rust aborts on those.
    const trapped = error instanceof WebAssembly.RuntimeError;
    self.postMessage({
      id,
      fatal: panicMessage
        || (trapped ? 'out of memory' : String(error?.message ?? error)),
    });
  }
};
