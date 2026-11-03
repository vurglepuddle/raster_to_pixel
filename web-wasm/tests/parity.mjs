// Compares the browser build against the native pipeline in a real headless browser.
// No npm packages: Node's standard library serves dist/ under /raster_to_pixel/ (the
// GitHub Pages path), the browser runs parity_page.js, and results are POSTed back.
//
//   cargo run --release --example parity_native -- target/parity [real image dir]
//   node web-wasm/tests/parity.mjs --browser chrome|edge|firefox
//
// Exit code 0 only if every comparison and UI check passed. See web-wasm/README.md.

import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../..');
const PREFIX = '/raster_to_pixel/';
const TEST_PATH = `${PREFIX}__test__/`;
const TIMEOUT_MS = 20 * 60 * 1000;

// JPEG decoders legitimately differ (see README). Output may drift by this much on
// average per channel, on the 0-255 scale, before it counts as a failure.
const JPEG_MEAN_DELTA_LIMIT = 6;
// Expected RGB of the hand-built non-interleaved JPEG, and allowed error per channel.
const NON_INTERLEAVED_RGB = [124, 94, 68];
const NON_INTERLEAVED_TOLERANCE = 6;

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const browserName = option('browser', 'chrome');
const nativeDir = path.resolve(ROOT, option('native-dir', 'target/parity'));
const distDir = path.resolve(ROOT, option('dist', 'dist'));
const only = option('only', 'all');
const fixtureFilter = new RegExp(option('fixtures', '.'), 'i');

const programFiles = [process.env.PROGRAMFILES, process.env['PROGRAMFILES(X86)'], process.env.LOCALAPPDATA];
const candidates = {
  chrome: [
    process.env.CHROME_BIN,
    ...programFiles.map((d) => d && path.join(d, 'Google/Chrome/Application/chrome.exe')),
    '/usr/bin/google-chrome',
  ],
  edge: [
    ...programFiles.map((d) => d && path.join(d, 'Microsoft/Edge/Application/msedge.exe')),
    '/usr/bin/microsoft-edge',
  ],
  firefox: [
    process.env.FIREFOX_BIN,
    ...programFiles.map((d) => d && path.join(d, 'Mozilla Firefox/firefox.exe')),
    '/usr/bin/firefox',
  ],
};
const browserPath = (candidates[browserName] || []).find((p) => p && existsSync(p));
if (!browserPath) {
  console.error(`could not find ${browserName}`);
  process.exit(2);
}

const manifest = JSON.parse(readFileSync(path.join(nativeDir, 'manifest.json'), 'utf8'));
const fixtures = new Map(manifest.fixtures.map((f) => [f.name, f]));

// ── results ─────────────────────────────────────────────────────────────────

const failures = [];
const outsidePrefix = [];
const stats = new Map(); // fixture -> { identical, tolerated, failed, worstMean, worstMax, worstPixels }
const uiResults = [];
const measurements = [];
const jpegDrift = [];

const isJpeg = (name) => /\.jpe?g$/i.test(name);

function fixtureStats(name) {
  if (!stats.has(name)) {
    stats.set(name, { identical: 0, tolerated: 0, failed: 0, worstMean: 0, worstMax: 0, worstPixels: 0 });
  }
  return stats.get(name);
}

function fail(message) {
  failures.push(message);
  console.log(`  FAIL ${message}`);
}

// Decodes the only PNG layout the pipeline writes: 8-bit RGBA, non-interlaced.
function decodePng(buffer) {
  let offset = 8;
  let width = 0;
  let height = 0;
  const idat = [];
  while (offset < buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString('latin1', offset + 4, offset + 8);
    const data = buffer.subarray(offset + 8, offset + 8 + length);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      if (data[8] !== 8 || data[9] !== 6 || data[12] !== 0) throw new Error('unexpected PNG layout');
    } else if (type === 'IDAT') {
      idat.push(data);
    }
    offset += length + 12;
  }
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const stride = width * 4;
  const pixels = Buffer.alloc(stride * height);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = y * (stride + 1) + 1;
    const out = y * stride;
    for (let x = 0; x < stride; x++) {
      const a = x >= 4 ? pixels[out + x - 4] : 0;
      const b = y > 0 ? pixels[out + x - stride] : 0;
      const c = x >= 4 && y > 0 ? pixels[out + x - stride - 4] : 0;
      let predicted = 0;
      if (filter === 1) predicted = a;
      else if (filter === 2) predicted = b;
      else if (filter === 3) predicted = (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c;
        const [pa, pb, pc] = [Math.abs(p - a), Math.abs(p - b), Math.abs(p - c)];
        predicted = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      pixels[out + x] = (raw[line + x] + predicted) & 255;
    }
  }
  return { width, height, pixels };
}

function pixelDifference(actualBytes, expectedBytes) {
  const actual = decodePng(actualBytes);
  const expected = decodePng(expectedBytes);
  if (actual.width !== expected.width || actual.height !== expected.height) {
    return { dimensions: `${actual.width}x${actual.height} vs ${expected.width}x${expected.height}` };
  }
  let differing = 0;
  let total = 0;
  let max = 0;
  for (let i = 0; i < actual.pixels.length; i += 4) {
    let pixelDelta = 0;
    for (let c = 0; c < 4; c++) {
      pixelDelta = Math.max(pixelDelta, Math.abs(actual.pixels[i + c] - expected.pixels[i + c]));
      total += Math.abs(actual.pixels[i + c] - expected.pixels[i + c]);
    }
    if (pixelDelta) differing++;
    max = Math.max(max, pixelDelta);
  }
  const pixelCount = actual.pixels.length / 4;
  return { pixels: differing / pixelCount, max, mean: total / actual.pixels.length };
}

function compareOutput(fixtureName, output, status, headers, body) {
  const label = `${fixtureName} ${output.case}/${output.op}`;
  const record = fixtureStats(fixtureName);
  const expected = readFileSync(path.join(nativeDir, 'native', output.file));
  const jpeg = isJpeg(fixtureName);
  if (status !== output.status) {
    record.failed++;
    return fail(`${label}: status ${status}, native ${output.status}`);
  }
  // Off Windows the native decoder renders this layout neon green, so native is the wrong
  // reference; checkNonInterleaved tests it against the known correct color instead.
  if (fixtureName === 'non_interleaved.jpg' && manifest.nativeJpegDecoder !== 'gdi+') {
    record.tolerated++;
    return;
  }
  // A different decoder can legitimately change JPEG grid detection and palette stats.
  const headersMatch = jpeg || headers === output.headers;
  if (body.equals(expected) && headersMatch) {
    record.identical++;
    return;
  }
  if (!headersMatch) {
    fail(`${label}: headers differ\n   web:    ${headers.slice(0, 300)}\n   native: ${output.headers.slice(0, 300)}`);
    if (body.equals(expected)) {
      record.failed++;
      return;
    }
  }
  if (output.op.startsWith('palette')) {
    // Palettes are luma-sorted, so one shifted entry reorders the file: comparing JPEG
    // palettes entry by entry says nothing. Their effect shows in the image comparison.
    if (jpeg) {
      record.tolerated++;
      return;
    }
    record.failed++;
    return fail(`${label}: palette file differs`);
  }
  const diff = pixelDifference(body, expected);
  if (diff.dimensions) {
    if (jpeg) {
      record.tolerated++;
      console.log(`  note ${label}: ${diff.dimensions} (decoder changed grid detection)`);
      return;
    }
    record.failed++;
    return fail(`${label}: dimensions ${diff.dimensions}`);
  }
  record.worstMean = Math.max(record.worstMean, diff.mean);
  record.worstMax = Math.max(record.worstMax, diff.max);
  record.worstPixels = Math.max(record.worstPixels, diff.pixels);
  jpegDrift.push({ fixture: fixtureName, case: output.case, op: output.op, ...diff });
  if (jpeg && diff.mean <= JPEG_MEAN_DELTA_LIMIT) {
    record.tolerated++;
    return;
  }
  record.failed++;
  fail(`${label}: ${(diff.pixels * 100).toFixed(3)}% of pixels differ, mean ${diff.mean.toFixed(3)}, max ${diff.max}`);
}

function checkNonInterleaved(png) {
  const { pixels } = decodePng(png);
  const mean = [0, 0, 0];
  for (let i = 0; i < pixels.length; i += 4) for (let c = 0; c < 3; c++) mean[c] += pixels[i + c];
  const rgb = mean.map((v) => Math.round(v / (pixels.length / 4)));
  const ok = rgb.every((v, c) => Math.abs(v - NON_INTERLEAVED_RGB[c]) <= NON_INTERLEAVED_TOLERANCE);
  uiResults.push({ name: `non-interleaved JPEG is not neon green (mean rgb ${rgb})`, ok });
  if (!ok) fail(`non-interleaved JPEG decoded to rgb ${rgb}, expected ${NON_INTERLEAVED_RGB}`);
  else console.log(`  ok   non-interleaved JPEG decoded to rgb ${rgb}`);
}

// ── server ──────────────────────────────────────────────────────────────────

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
};

function serveFile(res, file) {
  if (!existsSync(file)) {
    res.writeHead(404).end('not found');
    return;
  }
  res.writeHead(200, {
    'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
    'Cache-Control': 'no-store',
  });
  res.end(readFileSync(file));
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = crc & 1 ? (crc >>> 1) ^ 0xedb88320 : crc >>> 1;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngWithHugeDimensions() {
  const chunk = (type, data) => {
    const body = Buffer.concat([Buffer.from(type, 'latin1'), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(body));
    const length = Buffer.alloc(4);
    length.writeUInt32BE(data.length);
    return Buffer.concat([length, body, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(30000, 0);
  ihdr.writeUInt32BE(30000, 4);
  ihdr.set([8, 6, 0, 0, 0], 8);
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(Buffer.alloc(16))),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

let finish;
const done = new Promise((resolve) => { finish = resolve; });

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const pathname = decodeURIComponent(url.pathname);
  if (!pathname.startsWith(PREFIX)) outsidePrefix.push(pathname);

  if (pathname.startsWith(TEST_PATH)) {
    const rest = pathname.slice(TEST_PATH.length);
    if (rest === 'page.html') {
      res.writeHead(200, { 'Content-Type': MIME['.html'], 'Cache-Control': 'no-store' });
      res.end(`<!doctype html><meta charset="utf-8"><title>parity</title><link rel="icon" href="data:,">
        <script src="../web_backend.js"></script>
        <script type="module" src="parity_page.js"></script>`);
    } else if (rest === 'parity_page.js') {
      serveFile(res, path.join(HERE, 'parity_page.js'));
    } else if (rest === 'manifest.json') {
      res.writeHead(200, { 'Content-Type': MIME['.json'] });
      res.end(JSON.stringify({
        only,
        fixtures: manifest.fixtures.filter((f) => fixtureFilter.test(f.name)),
        uiFixture: manifest.fixtures.find((f) => f.name === 'sprite_alpha.png'),
      }));
    } else if (rest.startsWith('native/')) {
      serveFile(res, path.join(nativeDir, 'native', rest.slice(7)));
    } else if (rest.startsWith('fixtures/')) {
      serveFile(res, path.join(nativeDir, 'fixtures', rest.slice(9)));
    } else if (rest === 'huge.png') {
      res.writeHead(200, { 'Content-Type': 'image/png' });
      res.end(pngWithHugeDimensions());
    } else if (rest === 'report' && req.method === 'POST') {
      const parts = [];
      req.on('data', (d) => parts.push(d));
      req.on('end', () => {
        handleReport(url.searchParams, Buffer.concat(parts));
        res.writeHead(204).end();
      });
    } else {
      res.writeHead(404).end('not found');
    }
    return;
  }

  let relative = pathname.slice(PREFIX.length) || 'index.html';
  if (relative.endsWith('/')) relative += 'index.html';
  const file = path.resolve(distDir, relative);
  if (!file.startsWith(distDir)) {
    res.writeHead(403).end();
    return;
  }
  serveFile(res, file);
});

function handleReport(params, body) {
  const kind = params.get('kind');
  if (kind === 'session') {
    const fixture = fixtures.get(params.get('fixture'));
    const web = body.toString();
    console.log(`${fixture.name}`);
    if (!isJpeg(fixture.name) && web !== fixture.session) {
      fail(`${fixture.name} session: ${web} vs native ${fixture.session}`);
    }
    fixtureStats(fixture.name);
  } else if (kind === 'output') {
    const fixtureName = params.get('fixture');
    const output = fixtures.get(fixtureName).outputs.find(
      (o) => o.case === params.get('case') && o.op === params.get('op'),
    );
    compareOutput(fixtureName, output, Number(params.get('status')), params.get('headers'), body);
    if (fixtureName === 'non_interleaved.jpg' && output.case === 'defaults' && output.op === 'preview') {
      checkNonInterleaved(body);
    }
  } else if (kind === 'ui-bytes') {
    // A byte-level check of what the real page produced against the native file.
    const name = params.get('name');
    const expected = readFileSync(path.join(nativeDir, 'native', params.get('file')));
    const ok = body.equals(expected);
    uiResults.push({ name, ok });
    console.log(`  ${ok ? 'ok  ' : 'FAIL'} ui: ${name}`);
    if (!ok) failures.push(`ui: ${name}: bytes differ from native`);
  } else if (kind === 'ui') {
    const name = params.get('name');
    const ok = params.get('ok') === 'true';
    uiResults.push({ name, ok, detail: params.get('detail') });
    console.log(`  ${ok ? 'ok  ' : 'FAIL'} ui: ${name}${params.get('detail') ? ` (${params.get('detail')})` : ''}`);
    if (!ok) failures.push(`ui: ${name}: ${params.get('detail')}`);
  } else if (kind === 'measure') {
    measurements.push(`${params.get('name')}: ${params.get('value')}`);
    console.log(`  measure ${params.get('name')}: ${params.get('value')}`);
  } else if (kind === 'log') {
    console.log(`  ${body.toString()}`);
  } else if (kind === 'done') {
    finish();
  }
}

// ── browser ─────────────────────────────────────────────────────────────────

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const pageUrl = `http://127.0.0.1:${server.address().port}${TEST_PATH}page.html`;

// Profiles live under target/ and are reused, so the harness never deletes anything.
const profile = path.join(nativeDir, `profile-${browserName}`);
mkdirSync(profile, { recursive: true });
let browserArgs;
if (browserName === 'firefox') {
  writeFileSync(path.join(profile, 'user.js'), [
    'user_pref("browser.shell.checkDefaultBrowser", false);',
    'user_pref("datareporting.policy.dataSubmissionEnabled", false);',
    'user_pref("browser.aboutwelcome.enabled", false);',
    'user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);',
  ].join('\n'));
  browserArgs = ['--headless', '--no-remote', '--profile', profile, pageUrl];
} else {
  browserArgs = [
    '--headless=new', '--no-first-run', '--no-default-browser-check', '--disable-background-networking',
    '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--window-size=1400,900',
    `--user-data-dir=${profile}`, pageUrl,
  ];
  if (process.env.CI) browserArgs.unshift('--no-sandbox');
}
console.log(`browser: ${browserPath}\nnative JPEG decoder: ${manifest.nativeJpegDecoder}\n`);
const browser = spawn(browserPath, browserArgs, { stdio: 'ignore' });

const finished = await Promise.race([
  done.then(() => true),
  new Promise((resolve) => setTimeout(() => resolve(false), TIMEOUT_MS)),
]);
browser.kill();
server.close();
server.closeAllConnections();

if (!finished) failures.push(`timed out after ${TIMEOUT_MS / 1000}s`);
if (outsidePrefix.length) {
  failures.push(`requests escaped ${PREFIX}: ${[...new Set(outsidePrefix)].join(', ')}`);
}

// ── summary ─────────────────────────────────────────────────────────────────

console.log(`\n${browserName}: fixture results (identical = PNG bytes equal to native)`);
for (const [name, s] of stats) {
  const drift = s.tolerated || s.failed
    ? `  worst: ${(s.worstPixels * 100).toFixed(2)}% px, mean ${s.worstMean.toFixed(2)}, max ${s.worstMax}`
    : '';
  console.log(`  ${name.padEnd(38)} identical ${String(s.identical).padStart(3)}  decoder-drift ${String(s.tolerated).padStart(3)}  failed ${String(s.failed).padStart(3)}${drift}`);
}
if (uiResults.length) {
  console.log(`UI checks: ${uiResults.filter((r) => r.ok).length}/${uiResults.length} passed`);
}
writeFileSync(path.join(nativeDir, `report-${browserName}.json`), JSON.stringify({
  browser: browserName,
  nativeJpegDecoder: manifest.nativeJpegDecoder,
  fixtures: Object.fromEntries(stats),
  ui: uiResults,
  measurements,
  jpegDrift,
  failures,
}, null, 2));

if (failures.length) {
  console.log(`\n${failures.length} failure(s):`);
  failures.slice(0, 40).forEach((f) => console.log(`  - ${f}`));
  process.exit(1);
}
console.log('\nAll checks passed.');
process.exit(0);
