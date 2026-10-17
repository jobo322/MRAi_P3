#!/usr/bin/env node
/**
 * view2d.mjs — local web viewer for the folder-51 2D J-resolved spectrum,
 * P3 probability map and system-determined peaks.
 *
 * Zero dependencies: node:http + node:fs only (same style as
 * cheminfo/spectrum-generator/scripts/viewDataset.mjs). It serves
 * `viewer2d/index.html` and the binary artifacts produced once by
 * `convert_2d.py` (which borrows the exact loaders of `interactive_2d.py`).
 * If meta.json is missing, it runs the Python conversion automatically.
 *
 * Usage:
 *   node scripts/plotting/view2d.mjs [--port 8787] [--out /tmp/p51_view2d]
 *   node scripts/plotting/view2d.mjs --ft2 X --tab Y --peaks Z --p3 W [--no-convert]
 *
 * API:
 *   GET /              → viewer2d/index.html
 *   GET /api/meta      → meta.json (shape, obs, labels, peaks, sources)
 *   GET /api/bin/:name → raw float32 LE bytes of spectrum|p3|f2ppm|f1hz
 */
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const INDEX_HTML = path.join(scriptDir, 'viewer2d', 'index.html');
const CONVERT_PY = path.join(scriptDir, 'convert_2d.py');
const DEFAULT_PYTHON = '/Users/abolanos/.pyenv/versions/tflow/bin/python';

// Same defaults as interactive_2d.py / convert_2d.py.
const DEFAULTS = {
  port: 8787,
  out: '/tmp/p51_view2d',
  ft2: '/tmp/p51/51.ft2',
  tab: '/tmp/p51_run/51.tab',
  peaks: '/tmp/p51_run/51.peaks',
  p3: '/tmp/p51_run/51.P3',
  python: DEFAULT_PYTHON,
};

function fail(message, code = 1) {
  console.error(`view2d: ${message}`);
  process.exit(code);
}

function parseArgs(argv) {
  const args = { ...DEFAULTS, convert: true };
  const flags = ['port', 'out', 'ft2', 'tab', 'peaks', 'p3', 'python'];
  for (let i = 2; i < argv.length; i++) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (flag === '--help' || flag === '-h') {
      console.log(
        'Usage: node scripts/plotting/view2d.mjs [--port 8787] [--out dir]\n' +
          '       [--ft2 file] [--tab file] [--peaks file] [--p3 file]\n' +
          '       [--python path] [--no-convert]',
      );
      process.exit(0);
    } else if (flag === '--no-convert') {
      args.convert = false;
    } else if (flags.includes(flag.slice(2))) {
      if (!value) fail(`${flag} requires a value`);
      const key = flag.slice(2);
      args[key] = key === 'port' ? Number(value) : value;
      if (key === 'port' && (!Number.isInteger(args.port) || args.port < 0 || args.port > 65535)) {
        fail(`invalid --port value: ${value}`);
      }
      i++;
    } else {
      fail(`unknown flag: ${flag}`);
    }
  }
  return args;
}

const args = parseArgs(process.argv);
const datasetDir = path.resolve(args.out);
const metaPath = path.join(datasetDir, 'meta.json');

async function exists(file) {
  return fs.access(file).then(() => true, () => false);
}

let meta;
if (await exists(metaPath)) {
  meta = JSON.parse(await fs.readFile(metaPath, 'utf8'));
} else {
  if (!args.convert) {
    fail(
      `no meta.json in ${datasetDir} — run\n` +
        `  ${DEFAULT_PYTHON} ${CONVERT_PY} --out ${datasetDir}\n` +
        'or drop --no-convert to let the viewer convert automatically.',
    );
  }
  console.log('meta.json not found — running convert_2d.py once...');
  const res = spawnSync(
    args.python,
    [
      CONVERT_PY,
      '--ft2', args.ft2,
      '--tab', args.tab,
      '--peaks', args.peaks,
      '--p3', args.p3,
      '--out', datasetDir,
    ],
    { stdio: 'inherit' },
  );
  if (res.status !== 0) {
    fail(`conversion failed (exit ${res.status}); convert manually with ${CONVERT_PY}`);
  }
  try {
    meta = JSON.parse(await fs.readFile(metaPath, 'utf8'));
  } catch {
    fail(`conversion reported success but ${metaPath} is unreadable`);
  }
}

// name → file whitelist; guards against path traversal.
const BIN_NAMES = ['spectrum', 'p3', 'f2ppm', 'f1hz'];
for (const name of BIN_NAMES) {
  if (meta[name] && !/^[A-Za-z0-9._-]+$/.test(meta[name])) {
    fail(`meta.${name} has an unsafe file name: ${meta[name]}`);
  }
}

function send(res, status, body, headers = {}) {
  res.writeHead(status, headers);
  res.end(body);
}

function sendText(res, status, message) {
  send(res, status, `${message}\n`, { 'Content-Type': 'text/plain; charset=utf-8' });
}

function sendJSON(res, status, obj) {
  send(res, status, JSON.stringify(obj), { 'Content-Type': 'application/json; charset=utf-8' });
}

async function serveBin(res, name) {
  const file = meta[name];
  if (!file) {
    sendText(res, 404, `no ${name} data (was --p3 missing during conversion?)`);
    return;
  }
  let buffer;
  try {
    buffer = await fs.readFile(path.join(datasetDir, file));
  } catch {
    sendText(res, 404, `file not found: ${file}`);
    return;
  }
  send(res, 200, buffer, {
    'Content-Type': 'application/octet-stream',
    'Content-Length': String(buffer.length),
  });
}

async function handle(req, res) {
  if (req.method !== 'GET') {
    send(res, 405, 'method not allowed\n', {
      'Content-Type': 'text/plain; charset=utf-8',
      Allow: 'GET',
    });
    return;
  }
  const { pathname } = new URL(req.url, 'http://localhost');

  if (pathname === '/' || pathname === '/index.html') {
    try {
      const html = await fs.readFile(INDEX_HTML);
      send(res, 200, html, { 'Content-Type': 'text/html; charset=utf-8' });
    } catch {
      sendText(res, 500, `viewer not found at ${INDEX_HTML}`);
    }
    return;
  }
  if (pathname === '/api/meta') {
    sendJSON(res, 200, meta);
    return;
  }
  const binMatch = pathname.match(/^\/api\/bin\/([A-Za-z0-9_-]+)$/);
  if (binMatch) {
    if (!BIN_NAMES.includes(binMatch[1])) {
      sendText(res, 404, `unknown binary: ${binMatch[1]}`);
      return;
    }
    await serveBin(res, binMatch[1]);
    return;
  }
  sendText(res, 404, `not found: ${pathname}`);
}

const server = http.createServer((req, res) => {
  Promise.resolve(handle(req, res)).catch((err) => {
    console.error('view2d:', err);
    if (!res.headersSent) sendText(res, 500, 'internal server error');
    else res.end();
  });
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(
      `view2d: port ${args.port} is already in use.\n` +
        `  Try: node scripts/plotting/view2d.mjs --port ${Number(args.port) + 1}`,
    );
  } else {
    console.error('view2d:', err.message);
  }
  process.exit(1);
});

server.listen(args.port, '127.0.0.1', () => {
  const peaks = meta.peaks ? meta.peaks.x.length : 0;
  console.log('folder-51 2D contour viewer (spectrum + P3 + peaks)');
  console.log(`  data    : ${datasetDir} (shape ${meta.shape[0]}x${meta.shape[1]}, ${peaks} peaks)`);
  console.log(`  sources : ${meta.source.ft2} | ${meta.source.peakSource}`);
  console.log(`  open    : http://localhost:${args.port}/`);
});