// Checks the native GUI server over real HTTP against the manifest from parity_native,
// so the local app is covered alongside the browser build. No npm packages.
//
//   cargo build --release --bin gui
//   node web-wasm/tests/native_gui.mjs

import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const PARITY = path.join(ROOT, 'target/parity');
const FIXTURE = 'sprite_alpha.png';

const exe = path.join(ROOT, 'target/release', process.platform === 'win32' ? 'gui.exe' : 'gui');
const server = spawn(exe, ['--no-open', '--port', '7950'], { stdio: ['ignore', 'pipe', 'inherit'] });
const base = await new Promise((resolve, reject) => {
  server.on('error', reject);
  server.stdout.on('data', (chunk) => {
    const match = /http:\/\/127\.0\.0\.1:\d+\//.exec(chunk.toString());
    if (match) resolve(match[0]);
  });
});

const failures = [];
let checks = 0;
const expect = (ok, message) => {
  checks++;
  if (!ok) failures.push(message);
};
const post = (route, body) => fetch(new URL(route, base), { method: 'POST', body });

try {
  const page = await fetch(base);
  expect(
    Buffer.from(await page.arrayBuffer()).equals(readFileSync(path.join(ROOT, 'gui/index.html'))),
    'GET / is not gui/index.html',
  );
  const webScript = await fetch(new URL('web_backend.js', base));
  expect(webScript.ok && (await webScript.text()) === '', '/web_backend.js should be an empty script');

  const before = await post('api/preview', 'size=8');
  expect(before.status === 409, `preview with no image returned ${before.status}`);
  const corrupt = await post('api/session', new Uint8Array([1, 2, 3]));
  expect(corrupt.status === 400 && /decode failed/.test((await corrupt.json()).error), 'corrupt upload');

  const manifest = JSON.parse(readFileSync(path.join(PARITY, 'manifest.json'), 'utf8'));
  const fixture = manifest.fixtures.find((f) => f.name === FIXTURE);
  const session = await post('api/session', new Blob([readFileSync(path.join(PARITY, 'fixtures', FIXTURE))]));
  expect((await session.text()) === fixture.session, 'session metadata differs');

  for (const output of fixture.outputs) {
    const route = output.op === 'preview' ? 'api/preview' : output.op === 'export' ? 'api/export' : 'api/palette';
    const response = await post(route, output.settings);
    const label = `${output.case}/${output.op}`;
    expect(response.status === output.status, `${label}: status ${response.status}`);
    expect(response.headers.get('content-type') === output.contentType, `${label}: content type`);
    for (const line of output.headers.split('\n').filter(Boolean)) {
      const [name, value] = [line.slice(0, line.indexOf(': ')), line.slice(line.indexOf(': ') + 2)];
      expect(response.headers.get(name) === value, `${label}: header ${name}`);
    }
    const expected = readFileSync(path.join(PARITY, 'native', output.file));
    expect(Buffer.from(await response.arrayBuffer()).equals(expected), `${label}: body differs`);
  }
} finally {
  server.kill();
}

if (failures.length) {
  console.log(`native GUI: ${failures.length} of ${checks} checks failed`);
  failures.forEach((f) => console.log(`  - ${f}`));
  process.exit(1);
}
console.log(`native GUI: all ${checks} checks passed`);
process.exit(0);
