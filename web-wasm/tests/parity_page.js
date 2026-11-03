// Runs inside the browser under test; parity.mjs compares what it reports. Two parts:
//  1. every native output, reproduced through the real backend -> Worker -> WASM path
//  2. the real gui/index.html in an iframe, driven only through its DOM, as a user would

const post = (params, body) => fetch(`report?${new URLSearchParams(params)}`, { method: 'POST', body });
const log = (message) => post({ kind: 'log' }, message);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const fixtureFile = async (name) => new File(
  [await (await fetch(`fixtures/${encodeURIComponent(name)}`)).blob()],
  name,
);

const OP_PATH = (op) => (op === 'preview' ? '/api/preview' : op === 'export' ? '/api/export' : '/api/palette');

async function runParity(fixtures) {
  for (const fixture of fixtures) {
    const session = await window.pixelineApi('/api/session', { body: await fixtureFile(fixture.name) });
    await post({ kind: 'session', fixture: fixture.name }, await session.text());
    for (const output of fixture.outputs) {
      const response = await window.pixelineApi(OP_PATH(output.op), {
        body: new URLSearchParams(output.settings),
      });
      const headers = [...response.headers.entries()]
        .filter(([name]) => name.startsWith('x-') || name === 'content-disposition');
      // Report in the native header order so the two lists compare directly.
      const order = output.headers.split('\n').filter(Boolean).map((line) => line.split(': ')[0]);
      const byName = new Map(headers);
      const text = order
        .map((name) => `${name}: ${byName.get(name.toLowerCase()) ?? '(missing)'}`)
        .join('\n');
      await post({
        kind: 'output',
        fixture: fixture.name,
        case: output.case,
        op: output.op,
        status: response.status,
        headers: text,
      }, await response.arrayBuffer());
    }
  }
}

// ── UI ──────────────────────────────────────────────────────────────────────

async function check(name, run) {
  try {
    const detail = await run();
    await post({ kind: 'ui', name, ok: true, detail: detail ?? '' });
  } catch (error) {
    await post({ kind: 'ui', name, ok: false, detail: String(error?.message ?? error) });
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function waitFor(condition, what, timeout = 60000) {
  const start = performance.now();
  while (performance.now() - start < timeout) {
    const value = condition();
    if (value) return value;
    await sleep(20);
  }
  throw new Error(`timed out waiting for ${what}`);
}

class Page {
  static async open() {
    const page = new Page();
    page.frame = document.createElement('iframe');
    page.frame.style.cssText = 'width:1300px;height:800px;border:0';
    page.frame.src = '../index.html';
    const loaded = new Promise((resolve) => { page.frame.onload = resolve; });
    document.body.append(page.frame);
    await loaded;
    page.win = page.frame.contentWindow;
    page.doc = page.frame.contentDocument;

    page.violations = [];
    page.doc.addEventListener('securitypolicyviolation', (e) => {
      page.violations.push(`${e.violatedDirective}: ${e.blockedURI}`);
    });

    // Observe (never alter) worker traffic and capture downloads instead of saving them.
    page.sent = [];
    page.terminated = 0;
    page.corruptNext = false;
    const proto = page.win.Worker.prototype;
    const postMessage = proto.postMessage;
    const terminate = proto.terminate;
    proto.postMessage = function patched(message, ...rest) {
      page.sent.push(message);
      if (page.corruptNext) {
        page.corruptNext = false;
        return postMessage.call(this, { ...message, op: 'simulated-crash' }, ...rest);
      }
      return postMessage.call(this, message, ...rest);
    };
    proto.terminate = function patched() {
      page.terminated++;
      return terminate.call(this);
    };
    page.downloads = [];
    page.win.HTMLAnchorElement.prototype.click = function capture() {
      page.downloads.push({ name: this.download, bytes: fetch(this.href).then((r) => r.arrayBuffer()) });
    };
    return page;
  }

  $(id) { return this.doc.getElementById(id); }

  async loadFile(file) {
    const transfer = new this.win.DataTransfer();
    transfer.items.add(file);
    const input = this.$('fileInput');
    input.files = transfer.files;
    input.dispatchEvent(new this.win.Event('change'));
  }

  input(id, value) {
    const el = this.$(id);
    el.value = value;
    el.dispatchEvent(new this.win.Event('input', { bubbles: true }));
    el.dispatchEvent(new this.win.Event('change', { bubbles: true }));
  }

  pick(groupId, label) {
    const button = [...this.$(groupId).children].find((b) => b.textContent === label);
    assert(button, `no ${label} in ${groupId}`);
    button.click();
  }

  rendering() { return this.$('renderInd').style.display !== 'none'; }

  // Waits for a preview newer than `previousSrc` and for rendering to settle.
  async nextPreview(previousSrc, timeout = 120000) {
    await waitFor(() => {
      const src = this.$('preview').getAttribute('src');
      return src && src !== previousSrc && !this.rendering();
    }, 'a new preview', timeout);
    const preview = this.$('preview');
    await waitFor(() => preview.complete && preview.naturalWidth > 0, 'the preview to decode');
    return preview.getAttribute('src');
  }

  previewBytes() {
    return fetch(this.$('preview').getAttribute('src')).then((r) => r.arrayBuffer());
  }

  async toast(pattern) {
    return waitFor(() => {
      const toast = this.$('toast');
      return toast.style.display !== 'none' && pattern.test(toast.textContent) ? toast.textContent : null;
    }, `a toast matching ${pattern}`);
  }

  async nextDownload() {
    await waitFor(() => this.downloads.length, 'a download');
    const { name, bytes } = this.downloads.shift();
    return { name, bytes: await bytes };
  }

  // The last settings the page actually sent, so a direct call can reproduce them.
  lastSettings() {
    for (let i = this.sent.length - 1; i >= 0; i--) {
      if (this.sent[i].op === 'preview') return this.sent[i].settings;
    }
    throw new Error('no preview was sent');
  }

  async directPreview(settings) {
    const response = await this.win.pixelineApi('/api/preview', { body: new URLSearchParams(settings) });
    assert(response.ok, `direct preview returned ${response.status}`);
    return response.arrayBuffer();
  }

  close() { this.frame.remove(); }
}

const sameBytes = (a, b) => {
  const [x, y] = [new Uint8Array(a), new Uint8Array(b)];
  return x.length === y.length && x.every((v, i) => v === y[i]);
};

// A noisy opaque PNG made on the spot, large enough to make conversion take real time.
async function noiseFile(size, name = 'noise.png') {
  const canvas = new OffscreenCanvas(size, size);
  const context = canvas.getContext('2d');
  let seed = 12345;
  const next = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed >>> 16; };
  for (let y = 0; y < size; y += 3) {
    for (let x = 0; x < size; x += 3) {
      context.fillStyle = `rgb(${next() % 256},${next() % 256},${next() % 256})`;
      context.fillRect(x, y, 3, 3);
    }
  }
  return new File([await canvas.convertToBlob({ type: 'image/png' })], name);
}

async function runUi(uiFixture) {
  const outputs = Object.fromEntries(uiFixture.outputs.map((o) => [`${o.case}/${o.op}`, o]));
  const page = await Page.open();

  await check('page starts empty, shows the privacy note', () => {
    assert(page.$('empty').textContent.includes('stay on your device'), 'privacy note missing');
    assert(page.$('meta').textContent === 'no image', 'unexpected meta');
    assert(page.$('saveBtn').disabled, 'save enabled with no image');
  });

  const sprite = await fixtureFile(uiFixture.name);
  let src = null;
  await check('opening a file produces a preview', async () => {
    await page.loadFile(sprite);
    src = await page.nextPreview(src);
    assert(page.$('meta').textContent.includes('203×203'), page.$('meta').textContent);
  });
  await post({ kind: 'ui-bytes', name: 'default preview equals native', file: outputs['defaults/preview'].file },
    await page.previewBytes());

  await check('status bar is filled from the conversion', () => {
    assert(/\d+×\d+/.test(page.$('stOut').textContent), `out: ${page.$('stOut').textContent}`);
    assert(page.$('stSwatches').children.length > 0, 'no swatches');
    return `out ${page.$('stOut').textContent}, pal ${page.$('stPal').textContent}`;
  });

  await check('Save PNG downloads the same bytes as native', async () => {
    page.$('saveBtn').click();
    const { name, bytes } = await page.nextDownload();
    assert(/^sprite_alpha_\d+px\.png$/.test(name), `name ${name}`);
    await post({ kind: 'ui-bytes', name: 'exported PNG equals native', file: outputs['defaults/export'].file }, bytes);
  });

  await check('Save HEX downloads the same palette as native', async () => {
    page.$('paletteBtn').click();
    const { name, bytes } = await page.nextDownload();
    assert(/^sprite_alpha_\d+c\.hex$/.test(name), `name ${name}`);
    await post({ kind: 'ui-bytes', name: 'exported palette equals native', file: outputs['defaults/palette_hex'].file }, bytes);
  });

  await check('scale and compare change the export, not the preview', async () => {
    const sentBefore = page.sent.length;
    for (let i = 0; i < 3; i++) page.$('scaleInc').click();
    page.$('compareBtn').click();
    await sleep(300);
    assert(page.sent.length === sentBefore, 'scale should not re-render the preview');
    page.$('saveBtn').click();
    const { bytes } = await page.nextDownload();
    const bitmap = await createImageBitmap(new Blob([bytes]));
    assert(bitmap.height === page.$('preview').naturalHeight * 4, `height ${bitmap.height}`);
    assert(bitmap.width > page.$('preview').naturalWidth * 4, 'no compare sheet');
    for (let i = 0; i < 3; i++) page.$('scaleDec').click();
    page.$('compareBtn').click();
  });

  await check('crop applies and matches native', async () => {
    page.$('cropBtn').click();
    await waitFor(() => page.$('cropControls').style.display === 'block', 'crop controls');
    for (const [id, value] of [['cropX', 8], ['cropY', 12], ['cropW', 100], ['cropH', 80]]) {
      const el = page.$(id);
      el.value = value;
      el.dispatchEvent(new page.win.Event('change'));
    }
    page.$('applyCropBtn').click();
    src = await page.nextPreview(src);
    assert(page.$('meta').textContent.includes('crop 100×80'), page.$('meta').textContent);
    await post({ kind: 'ui-bytes', name: 'cropped preview equals native', file: outputs['crop/preview'].file },
      await page.previewBytes());
    page.$('resetCropBtn').click();
    src = await page.nextPreview(src);
  });

  await check('theme, accent, and view modes still work', () => {
    page.pick('themeGrp', 'Light');
    assert(page.$('app').dataset.theme === 'light', 'light');
    page.pick('themeGrp', 'Dark');
    assert(page.$('app').dataset.theme === 'dark', 'dark');
    page.pick('baGrp', 'Compare');
    assert(page.$('comparePreview').style.display === 'block', 'compare view');
    assert(page.$('compareDivider').style.display === 'block', 'divider');
    page.pick('baGrp', 'Original');
    assert(page.$('preview').style.display === 'none', 'original view');
    page.pick('baGrp', 'Result');
    assert(page.$('preview').style.display === 'block', 'result view');
  });

  await check('rapid setting changes: latest result wins, most work is skipped', async () => {
    const before = page.sent.filter((m) => m.op === 'preview').length;
    let previous = src;
    for (let colors = 2; colors <= 41; colors++) {
      page.input('inColors', String(colors));
      await sleep(5);
    }
    src = await page.nextPreview(previous);
    await sleep(400); // Let anything wrongly still queued surface.
    const sent = page.sent.filter((m) => m.op === 'preview').length - before;
    const settings = page.lastSettings();
    assert(new URLSearchParams(settings).get('colors') === '41', `last settings: ${settings}`);
    assert(page.$('preview').getAttribute('src') === src, 'a stale result replaced the latest');
    const expected = await page.directPreview(settings);
    assert(sameBytes(await page.previewBytes(), expected), 'shown preview is not the latest settings');
    assert(sent < 40, `sent ${sent} conversions for 40 changes`);
    return `${sent} conversions for 40 changes`;
  });

  await check('drag and drop loads a file', async () => {
    const transfer = new page.win.DataTransfer();
    transfer.items.add(new File([await sprite.arrayBuffer()], 'dropped.png', { type: 'image/png' }));
    const event = new page.win.DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: transfer });
    page.doc.dispatchEvent(event);
    await waitFor(() => page.$('meta').textContent.includes('dropped.png 203') || page.$('meta').textContent.includes('dropped.png · 203'), 'dropped file');
    src = await page.nextPreview(null);
  });

  // ── failure handling ──────────────────────────────────────────────────────
  await check('a corrupt file reports an error, then a good file still loads', async () => {
    await page.loadFile(new File([new Uint8Array([1, 2, 3, 4, 5])], 'broken.png'));
    await page.toast(/decode failed/);
    await page.loadFile(sprite);
    src = await page.nextPreview(null);
  });

  await check('an oversized PNG is refused from its header, without decoding', async () => {
    await page.loadFile(new File([await (await fetch('huge.png')).blob()], 'huge.png'));
    const message = await page.toast(/30000x30000/);
    assert(/64 megapixels/.test(message), message);
    await page.loadFile(sprite);
    src = await page.nextPreview(null);
  });

  await check('an oversized JPEG is refused from its header, without decoding', async () => {
    const jpeg = new Uint8Array(await (await fixtureFile('non_interleaved.jpg')).arrayBuffer());
    for (let i = 2; i < jpeg.length - 9; i++) {
      if (jpeg[i] === 0xff && jpeg[i + 1] === 0xc0) {
        // Rewrite SOF0 height and width to 30000 (0x7530).
        jpeg.set([0x75, 0x30, 0x75, 0x30], i + 5);
        break;
      }
    }
    await page.loadFile(new File([jpeg], 'huge.jpg'));
    await page.toast(/30000x30000/);
    await page.loadFile(sprite);
    src = await page.nextPreview(null);
  });

  await check('an EXIF-rotated JPEG is processed the way the browser displays it', async () => {
    const jpeg = new Uint8Array(await (await fixtureFile('gradient.jpg')).arrayBuffer());
    const exif = [
      0xff, 0xe1, 0x00, 0x22, 0x45, 0x78, 0x69, 0x66, 0, 0, // APP1, length 34, "Exif\0\0"
      0x49, 0x49, 0x2a, 0x00, 8, 0, 0, 0, // Little-endian TIFF header
      1, 0, 0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, // Orientation = 6
    ];
    await page.loadFile(new File([jpeg.subarray(0, 2), new Uint8Array(exif), jpeg.subarray(2)], 'rotated.jpg'));
    await page.nextPreview(null);
    assert(page.$('meta').textContent.includes('180×240'), page.$('meta').textContent);
    const original = page.$('sourcePreview');
    await waitFor(() => original.complete && original.naturalWidth > 0, 'the original to decode');
    assert(original.naturalWidth === 180, `displayed original is ${original.naturalWidth} wide`);
    await page.loadFile(sprite);
    src = await page.nextPreview(null);
  });

  await check('an impossible custom palette reports an error and recovers', async () => {
    page.$('inPalette').value = 'custom';
    page.$('inPalette').dispatchEvent(new page.win.Event('change'));
    page.input('inCustom', 'not-a-color');
    await waitFor(() => page.$('customErr').style.display === 'block', 'the palette error');
    page.$('inPalette').value = 'adaptive';
    page.$('inPalette').dispatchEvent(new page.win.Event('change'));
    src = await page.nextPreview(src);
    assert(page.$('customErr').style.display === 'none', 'error did not clear');
  });

  await check('a crash fails one job, then the next preview recovers automatically', async () => {
    const terminatedBefore = page.terminated;
    page.corruptNext = true;
    page.input('inColors', '12');
    const message = await page.toast(/Processing failed/);
    await waitFor(() => page.terminated > terminatedBefore, 'the worker to be replaced');
    page.input('inColors', '13');
    src = await page.nextPreview(src);
    const expected = await page.directPreview(page.lastSettings());
    assert(sameBytes(await page.previewBytes(), expected), 'wrong preview after recovery');
    return message;
  });

  await check('an export past the output limit fails cleanly and leaves the app working', async () => {
    const big = await noiseFile(800, 'big.png');
    await page.loadFile(big);
    src = await page.nextPreview(null);
    page.pick('sizeModeGrp', 'Pixel');
    page.input('inPixel', '1');
    src = await page.nextPreview(src);
    for (let i = 0; i < 15; i++) page.$('scaleInc').click();
    page.$('saveBtn').click();
    const message = await page.toast(/too large to create/);
    for (let i = 0; i < 15; i++) page.$('scaleDec').click();
    page.input('inPixel', '4');
    src = await page.nextPreview(src);
    return message;
  });

  // ── responsiveness ────────────────────────────────────────────────────────
  await check('heavy conversion runs off the main thread', async () => {
    await page.loadFile(await noiseFile(1800));
    src = await page.nextPreview(null);
    let worstGap = 0;
    let last = performance.now();
    const timer = setInterval(() => {
      const now = performance.now();
      worstGap = Math.max(worstGap, now - last);
      last = now;
    }, 10);
    const start = performance.now();
    // A genuinely slow job: a large grid with the biggest k-means palette.
    page.pick('sizeModeGrp', 'Pixel');
    page.input('inPixel', '2');
    page.input('inColors', '512');
    src = await page.nextPreview(src, 300000);
    clearInterval(timer);
    const elapsed = performance.now() - start;
    await post({ kind: 'measure', name: 'heavy preview ms', value: Math.round(elapsed) });
    await post({ kind: 'measure', name: 'worst main-thread gap ms', value: Math.round(worstGap) });
    assert(elapsed > 600, `conversion only took ${elapsed}ms; too fast to prove anything`);
    // Generous for slow CI machines; a blocked main thread would stall for the whole job.
    assert(worstGap < 1000, `main thread stalled ${Math.round(worstGap)}ms during ${Math.round(elapsed)}ms`);
    page.heavySettings = page.lastSettings();
    page.heavyMs = elapsed;
    return `${Math.round(elapsed)}ms conversion, worst gap ${Math.round(worstGap)}ms`;
  });

  await check('a slow superseded preview is stopped so the latest one is not stuck behind it', async () => {
    assert(page.heavySettings, 'needs the previous check');
    if (page.heavyMs < 2500) return `skipped: conversion took only ${Math.round(page.heavyMs)}ms`;
    const terminatedBefore = page.terminated;
    // Start slow work, then replace it with something fast.
    page.input('inColors', '511');
    await sleep(350);
    const start = performance.now();
    page.input('inColors', '8');
    src = await page.nextPreview(src, 180000);
    const elapsed = performance.now() - start;
    assert(page.terminated > terminatedBefore, 'the slow job was not stopped');
    const expected = await page.directPreview(page.lastSettings());
    assert(sameBytes(await page.previewBytes(), expected), 'wrong preview after the restart');
    await post({ kind: 'measure', name: 'time to latest preview after restart ms', value: Math.round(elapsed) });
    return `${Math.round(elapsed)}ms vs ${Math.round(page.heavyMs)}ms for the slow job`;
  });

  await check('an export queued behind a superseded preview still completes', async () => {
    page.input('inColors', '510');
    await sleep(200);
    page.input('inColors', '9');
    await sleep(200);
    page.$('saveBtn').click();
    const { bytes } = await page.nextDownload();
    assert(bytes.byteLength > 0, 'empty export');
    src = await page.nextPreview(src, 180000);
  });

  await check('no network requests outside the site, no CSP violations', () => {
    const resources = [...page.win.performance.getEntriesByType('resource'), ...performance.getEntriesByType('resource')];
    const foreign = resources.map((r) => r.name).filter((n) => !n.startsWith(location.origin) && !n.startsWith('blob:') && !n.startsWith('data:'));
    assert(foreign.length === 0, `foreign requests: ${foreign.join(', ')}`);
    assert(page.violations.length === 0, `CSP: ${page.violations.join('; ')}`);
    return `${resources.length} resources checked`;
  });

  await check('the WebAssembly module can only import its panic and error hooks', async () => {
    const module = await WebAssembly.compileStreaming(fetch('../pkg/web_wasm_bg.wasm'));
    const imports = WebAssembly.Module.imports(module).map((i) => i.name);
    const unexpected = imports.filter((name) => !/^__wbg_(pixelinePanic|__wbindgen_throw)_/.test(name));
    assert(unexpected.length === 0, `unexpected imports: ${unexpected.join(', ')}`);
    return imports.join(', ');
  });

  page.close();
}

async function main() {
  const { only, fixtures, uiFixture } = await (await fetch('manifest.json')).json();
  try {
    if (only !== 'ui') await runParity(fixtures);
    if (only !== 'parity') await runUi(uiFixture);
  } catch (error) {
    await post({ kind: 'ui', name: 'test page crashed', ok: false, detail: `${error?.stack ?? error}` });
  }
  await post({ kind: 'done' });
}

window.addEventListener('error', (e) => log(`page error: ${e.message}`));
window.addEventListener('unhandledrejection', (e) => log(`unhandled rejection: ${e.reason}`));
main();
