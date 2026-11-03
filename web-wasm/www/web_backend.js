// Browser backend for gui/index.html: answers the page's /api/* requests with the same
// contract as the native server (status, X-* headers, body), computed by WebAssembly in
// a Worker. Nothing leaves the browser; the source stays a local File reference.
//
// Scheduling, since a running wasm call cannot be interrupted:
//  - one job runs at a time; exports queue in order and always finish
//  - previews are latest-wins: a newer one replaces the waiting one
//  - if a superseded preview keeps the worker busy past STALE_PREVIEW_MS, the worker is
//    replaced and the source reloaded, which is the only way to stop it
//  - each new image gets a fresh worker, because wasm memory never shrinks
//  - a crash (out of memory) fails that one job; the next job gets a new worker
(() => {
  const STALE_PREVIEW_MS = 1500;
  const WORKER_URL = new URL('worker.js', document.currentScript.src);
  const OPS = { '/api/preview': 'preview', '/api/export': 'export', '/api/palette': 'palette' };

  let worker = null;
  let source = null; // { file, loaded: whether the current worker holds it }
  let running = null;
  let waitingPreview = null;
  const queue = []; // Exports and source loads, in order.
  let nextId = 1;
  let staleTimer = null;

  const aborted = () => new DOMException('Replaced by a newer request', 'AbortError');

  function jsonError(status, message) {
    return new Response(JSON.stringify({ error: message }), {
      status,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  function discardWorker() {
    worker?.terminate();
    worker = null;
    running = null;
    clearTimeout(staleTimer);
    if (source) source.loaded = false;
  }

  function spawnWorker() {
    worker = new Worker(WORKER_URL, { type: 'module' });
    worker.onmessage = ({ data }) => {
      if (running?.id !== data.id) return;
      const job = running;
      if (data.fatal) {
        discardWorker();
        job.settle(jsonError(500, `Processing failed (${data.fatal}). Try Fixed size mode, a crop, or a smaller image.`));
      } else {
        running = null;
        clearTimeout(staleTimer);
        job.settle(new Response(data.body, {
          status: data.status,
          headers: [
            ['Content-Type', data.contentType],
            ...data.headers.split('\n').filter(Boolean).map((line) => {
              const colon = line.indexOf(': ');
              return [line.slice(0, colon), line.slice(colon + 2)];
            }),
          ],
        }));
      }
      pump();
    };
    // Only reached when the script or module cannot start at all.
    worker.onerror = (event) => {
      event.preventDefault();
      const job = running;
      discardWorker();
      job?.settle(jsonError(500, 'Could not start WebAssembly in this browser. Please update it.'));
      pump();
    };
  }

  function start(job) {
    if (!worker) spawnWorker();
    job.id = nextId++;
    job.startedAt = performance.now();
    running = job;
    worker.postMessage({ id: job.id, op: job.op, settings: job.settings, file: job.file });
  }

  function pump() {
    if (running) return;
    let job = queue[0]?.op === 'load' ? queue.shift() : null;
    if (!job && (queue.length || waitingPreview)) {
      job = source && !source.loaded ? reloadJob() : queue.shift() ?? takeWaitingPreview();
    }
    if (job) start(job);
  }

  function takeWaitingPreview() {
    const job = waitingPreview;
    waitingPreview = null;
    return job;
  }

  // After a worker is replaced, quietly restore the source before the next job.
  function reloadJob() {
    const reloading = source;
    return {
      op: 'load',
      file: reloading.file,
      reject() {},
      settle(response) {
        if (response.ok) {
          reloading.loaded = true;
          return;
        }
        // The source no longer loads, so nothing queued can succeed.
        source = null;
        const failed = [...queue.splice(0), takeWaitingPreview()].filter(Boolean);
        response.json().then(({ error }) => {
          failed.forEach((job) => job.settle(jsonError(500, error)));
        });
      },
    };
  }

  // A superseded preview is only worth stopping if something is waiting for the worker.
  function armStaleTimer() {
    clearTimeout(staleTimer);
    if (!running?.superseded || !(queue.length || waitingPreview)) return;
    const elapsed = performance.now() - running.startedAt;
    staleTimer = setTimeout(() => {
      discardWorker();
      pump();
    }, Math.max(0, STALE_PREVIEW_MS - elapsed));
  }

  function supersede(job) {
    if (job === waitingPreview) {
      waitingPreview = null;
      job.reject(aborted());
    } else if (job === running) {
      job.superseded = true;
      job.settle = () => {};
      job.reject(aborted());
    }
  }

  function loadSource(file, resolve, reject) {
    // Abandon everything from the previous image and start on fresh memory.
    const abandoned = [...queue.splice(0), takeWaitingPreview(), running].filter(Boolean);
    discardWorker();
    abandoned.forEach((job) => job.reject(aborted()));
    source = { file, loaded: false };
    const mine = source;
    queue.push({
      op: 'load',
      file,
      reject,
      settle: (response) => {
        if (source === mine) {
          if (response.ok) mine.loaded = true;
          else source = null;
        }
        resolve(response);
      },
    });
    pump();
  }

  window.pixelineApi = (path, { body, signal } = {}) => new Promise((resolve, reject) => {
    if (path === '/api/session') {
      loadSource(body, resolve, reject);
      return;
    }
    const op = OPS[path];
    if (!op) {
      reject(new Error(`Unknown request ${path}`));
      return;
    }
    if (!source) {
      resolve(jsonError(409, 'no image loaded; POST /api/session first'));
      return;
    }
    if (signal?.aborted) {
      reject(aborted());
      return;
    }
    const job = { op, settings: body.toString(), settle: resolve, reject };
    if (op === 'preview') {
      if (waitingPreview) supersede(waitingPreview);
      waitingPreview = job;
      signal?.addEventListener('abort', () => {
        supersede(job);
        armStaleTimer();
      }, { once: true });
    } else {
      queue.push(job);
    }
    pump();
    armStaleTimer();
  });

  document.addEventListener('DOMContentLoaded', () => {
    const note = document.createElement('span');
    note.textContent = 'Images stay on your device: everything runs in this browser.';
    document.getElementById('empty')?.append(note);
  });
})();
