#!/usr/bin/env node
/**
 * view2d.mjs — local web viewer for 2D JRES contour data with two modes.
 *
 * Zero dependencies: node:http + node:fs only (same style as
 * cheminfo/spectrum-generator/scripts/viewDataset.mjs).
 *
 * Mode A — folder-51 (default): serves the artifacts produced once by
 * `convert_2d.py` (exact loaders of `interactive_2d.py`) and the interactive
 * viewer `viewer2d/index.html` (+ the `/circles` copy). If meta.json is
 * missing, it runs the Python conversion automatically.
 *
 * Mode B — training dataset (--dataset <dir>): browses a spectrum-generator
 * style dataset (dataset.json + manifest.jsonl + spectra/<id>.bin +
 * labels/<id>.bin uint8 mask). Serves `viewer2d/dataset.html`, which draws the
 * training labels (mask == 1) as open blue circles and the inference
 * peak-picking results (.tab per id) as red crosses.
 *
 * Usage:
 *   node scripts/plotting/view2d.mjs [--port 8787] [--out /tmp/p51_view2d]
 *   node scripts/plotting/view2d.mjs --ft2 X --tab Y --peaks Z --p3 W [--no-convert]
 *   node scripts/plotting/view2d.mjs --dataset <dir> [--inference <dir>] [--port 8787]
 *
 * API (mode A):
 *   GET /, /circles     → viewer2d/index.html / index-circles.html
 *   GET /api/meta       → meta.json
 *   GET /api/bin/:name  → float32 LE bytes of spectrum|p3|f2ppm|f1hz
 *
 * API (mode B):
 *   GET /                → viewer2d/dataset.html
 *   GET /api/dataset     → dataset.json
 *   GET /api/manifest    → manifest.jsonl as JSON array
 *   GET /api/spectrum/:id → raw spectra/<id padded 4>.bin (float32 LE)
 *   GET /api/labels/:id   → raw labels/<id padded 4>.bin (uint8)
 *   GET /api/inference/:id → parsed peaks from inference dir (.tab) as JSON
 *   GET /api/inference-index → [ids] que tienen .tab (la pagina filtra con esto)
 */
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const VIEWER_DIR = path.join(scriptDir, 'viewer2d');
const INDEX_HTML = path.join(VIEWER_DIR, 'index.html');
const INDEX_CIRCLES = path.join(VIEWER_DIR, 'index-circles.html');
const DATASET_HTML = path.join(VIEWER_DIR, 'dataset.html');
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
  dataset: null,
  inference: null,
};

function fail(message, code = 1) {
  console.error(`view2d: ${message}`);
  process.exit(code);
}

function parseArgs(argv) {
  const args = { ...DEFAULTS, convert: true };
  const valued = ['port', 'out', 'ft2', 'tab', 'peaks', 'p3', 'python', 'dataset', 'inference'];
  for (let i = 2; i < argv.length; i++) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (flag === '--help' || flag === '-h') {
      console.log(
        'Mode A (folder-51):\n' +
          '  node scripts/plotting/view2d.mjs [--port 8787] [--out dir]\n' +
          '      [--ft2 file] [--tab file] [--peaks file] [--p3 file] [--no-convert]\n' +
          'Mode B (training dataset):\n' +
          '  node scripts/plotting/view2d.mjs --dataset <dir> [--inference <dir>] [--port 8787]',
      );
      process.exit(0);
    } else if (flag === '--no-convert') {
      args.convert = false;
    } else if (valued.includes(flag.slice(2))) {
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
const datasetMode = !!args.dataset;

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

async function exists(file) {
  return fs.access(file).then(() => true, () => false);
}

/* ── Mode B: training dataset ─────────────────────────────────────────────── */
let dsDir = null, infDir = null, dsConfig = null, manifest = [], nSpectra = 0;
let infIds = [];            // ids con .tab en el dir de inferencia
const peaksById = new Map();   // id -> peaks del manifiesto (bajo demanda)
const inferenceCache = new Map();

async function setupDatasetMode() {
  dsDir = path.resolve(args.dataset);
  for (const sub of ['dataset.json', 'manifest.jsonl', 'spectra', 'labels']) {
    if (!(await exists(path.join(dsDir, sub)))) {
      fail(`--dataset ${dsDir} is missing ${sub}`);
    }
  }
  dsConfig = JSON.parse(await fs.readFile(path.join(dsDir, 'dataset.json'), 'utf8'));
  // Manifiesto SLIM: el completo pesa ~106 MB (1000 x 256 picos con pixelSet,
  // factor, muXLib...) y la pagina solo usa id, class, r, noise y
  // peaks.{col,row,A,group}. Enviarlo entero congelaba la pestana del navegador
  // (JSON.parse de 106 MB + cientos de miles de objetos) y no dibujaba nada.
  manifest = (await fs.readFile(path.join(dsDir, 'manifest.jsonl'), 'utf8'))
    .split('\n').filter((l) => l.trim()).map((l) => {
      const rec = JSON.parse(l);
      peaksById.set(rec.id, (rec.peaks || []).map((p) => ({
        row: p.row, col: p.col, A: p.A, group: p.group,
      })));
      return {
        id: rec.id, class: rec.class, r: rec.r, noise: rec.noise || null,
        nPeaks: (rec.peaks || []).length,
      };
    });
  nSpectra = manifest.length;
  if (nSpectra === 0) fail('manifest.jsonl is empty');
  // inference results: default <dataset>/inference, or --inference <dir>
  infDir = args.inference ? path.resolve(args.inference) : path.join(dsDir, 'inference');
  if (!(await exists(infDir))) {
    console.log(`  note   : no inference dir at ${infDir} — red crosses will be empty`);
    infDir = null;
  } else {
    // ids que SÍ tienen resultados: la pagina filtra la lista con esto (si no,
    // hay que buscar a mano entre los 1000 espectros del dataset)
    const re = /^(?:tab_)?0*(\d+)\.tab$/;
    infIds = (await fs.readdir(infDir))
      .map((f) => { const m = f.match(re); return m ? Number(m[1]) : null; })
      .filter((v) => v !== null)
      .sort((a, b) => a - b);
    console.log(`  inference: ${infIds.length} ids con .tab`);
  }
}

// Parse one inference .tab (VARS format, like PeakstoTab.py writes).
// Coordinate convention (verified against the manifest ground truth): in the
// synthetic P3 chain the tab axes are descending, so X_PPM = NX - X_AXIS and
// Y_PPM = NY - Y_AXIS hold the MIRRORED pixel coordinate; the grid position is
// col = (NX-1) - X_PPM, row = (NY-1) - Y_PPM. Sanity check id=1: 204/222 peaks
// within 2px of truth (report.json: 203 true positives) using this mapping;
// raw X1/Y1 gave 197/222 with systematic mirror offset, X_PPM raw gave 0.
async function loadInference(id) {
  if (!infDir) return { peaks: [], source: null };
  if (inferenceCache.has(id)) return inferenceCache.get(id);
  const sid = String(id).padStart(4, '0');
  let out = { peaks: [], source: null };
  // `run_train_2d.py` escribe `<id>.tab` (id sin ceros: 1.tab) y `run_51.py`
  // escribe `<name>.tab`; se aceptan las dos formas y las de 4 digitos.
  const raw = String(id);
  for (const cand of [
    path.join(infDir, `${sid}.tab`), path.join(infDir, `${raw}.tab`),
    path.join(infDir, `tab_${sid}.tab`), path.join(infDir, `tab_${raw}.tab`),
  ]) {
    if (await exists(cand)) {
      const vl = [];
      const peaks = [];
      for (const line of (await fs.readFile(cand, 'utf8')).split('\n')) {
        const s = line.trim();
        if (!s) continue;
        const parts = s.split(/\s+/);
        if (parts[0] === 'VARS') { vl.push(...parts.slice(1)); continue; }
        if (!vl.length || !/^\d/.test(s) || parts.length !== vl.length) continue;
        const get = (k) => parts[vl.indexOf(k)];
        const nx = dsConfig?.config?.nbPoints?.x ?? 2048;
        const ny = dsConfig?.config?.nbPoints?.y ?? 256;
        peaks.push({
          index: Number(get('INDEX')),
          col: (nx - 1) - Number(get('X_PPM')),   // mirrored pixel → grid col
          row: (ny - 1) - Number(get('Y_PPM')),   // mirrored pixel → grid row
          xPpm: Number(get('X_PPM')),
          yPpm: Number(get('Y_PPM')),
          height: Number(get('HEIGHT')),
          type: get('TYPE'),
        });
      }
      out = { peaks, source: cand };
      break;
    }
  }
  out.hasP3 = !!(await findP3(id));
  inferenceCache.set(id, out);
  return out;
}

// Mapa P3 inferido: localiza un p3.bin convertido dentro del árbol de
// inferencia (layout `<inference>/view2d/p3.bin` cuando el dir de inferencia
// ES el dir del id — v_pair_1 —, o `<inference>/<id>/view2d/p3.bin`). Valida la
// shape contra el meta.json vecino cuando existe.
const p3PathCache = new Map();
async function findP3(id) {
  if (p3PathCache.has(id)) return p3PathCache.get(id);
  const sid = String(id).padStart(4, '0');
  const raw = String(id);
  const nx = dsConfig?.config?.nbPoints?.x ?? 2048;
  const ny = dsConfig?.config?.nbPoints?.y ?? 256;
  let out = null;
  for (const [dir, needReport] of [
    [infDir, true],                              // dir plano: atribuir vía report.json
    [path.join(infDir, sid), false],             // <infDir>/<0001>/view2d
    [path.join(infDir, raw), false],             // <infDir>/<1>/view2d
  ].filter(([d]) => d)) {
    const bin = path.join(dir, 'view2d', 'p3.bin');
    if (!(await exists(bin))) continue;
    if (needReport) {
      // el dir plano solo vale para el id declarado en report.json
      try {
        const rep = JSON.parse(await fs.readFile(path.join(dir, 'report.json'), 'utf8'));
        if (String(rep.id) !== String(id)) continue;
      } catch { continue; }   // sin report.json no se puede atribuir a ningún id
    }
    let ok = true;
    try {
      const m = JSON.parse(await fs.readFile(path.join(dir, 'view2d', 'meta.json'), 'utf8'));
      ok = m.shape && m.shape[0] === ny && m.shape[1] === nx;
    } catch { /* sin meta.json legible: confía en el bin */ }
    if (ok) { out = bin; break; }
  }
  p3PathCache.set(id, out);
  return out;
}

async function serveAbsBin(res, file) {
  let buffer;
  try {
    buffer = await fs.readFile(file);
  } catch {
    sendText(res, 404, `file not found: ${file}`);
    return;
  }
  send(res, 200, buffer, {
    'Content-Type': 'application/octet-stream',
    'Content-Length': String(buffer.length),
  });
}

async function serveSpectrumFile(res, subdir, id, n) {
  const sid = String(id).padStart(4, '0');
  let buffer;
  try {
    buffer = await fs.readFile(path.join(dsDir, subdir, `${sid}.bin`));
  } catch {
    sendText(res, 404, `file not found: ${subdir}/${sid}.bin`);
    return;
  }
  send(res, 200, buffer, {
    'Content-Type': 'application/octet-stream',
    'Content-Length': String(buffer.length),
  });
}

/* ── Mode A: folder-51 ────────────────────────────────────────────────────── */
let meta = null, datasetDir = null;
const BIN_NAMES = ['spectrum', 'p3', 'f2ppm', 'f1hz'];

async function setupFolder51Mode() {
  datasetDir = path.resolve(args.out);
  const metaPath = path.join(datasetDir, 'meta.json');
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
  for (const name of BIN_NAMES) {
    if (meta[name] && !/^[A-Za-z0-9._-]+$/.test(meta[name])) {
      fail(`meta.${name} has an unsafe file name: ${meta[name]}`);
    }
  }
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

/* ── routing ──────────────────────────────────────────────────────────────── */
async function serveHTML(res, file) {
  try {
    const html = await fs.readFile(file);
    send(res, 200, html, { 'Content-Type': 'text/html; charset=utf-8' });
  } catch {
    sendText(res, 500, `viewer not found at ${file}`);
  }
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

  if (datasetMode) {
    if (pathname === '/' || pathname === '/index.html') {
      await serveHTML(res, DATASET_HTML);
      return;
    }
    if (pathname === '/api/dataset') { sendJSON(res, 200, dsConfig); return; }
    if (pathname === '/api/manifest') { sendJSON(res, 200, manifest); return; }
    if (pathname === '/api/inference-index') { sendJSON(res, 200, infIds); return; }
    if (pathname.startsWith('/api/peaks/')) {
      const id = Number(decodeURIComponent(pathname.slice('/api/peaks/'.length)));
      if (peaksById.has(id)) sendJSON(res, 200, peaksById.get(id));
      else sendText(res, 404, `no peaks for id ${id}`);
      return;
    }
    const m = pathname.match(/^\/api\/(spectrum|labels|inference|p3)\/(\d+)$/);
    if (m) {
      const id = Number(m[2]);
      if (!Number.isInteger(id) || id < 0 || id >= nSpectra) {
        sendText(res, 404, `id out of range (0..${nSpectra - 1})`);
        return;
      }
      if (m[1] === 'inference') { sendJSON(res, 200, await loadInference(id)); return; }
      if (m[1] === 'p3') {
        const bin = await findP3(id);
        if (!bin) {
          sendText(res, 404, `no converted p3.bin for id ${id} in the inference tree`);
          return;
        }
        await serveAbsBin(res, bin);
        return;
      }
      await serveSpectrumFile(res, m[1] === 'spectrum' ? 'spectra' : 'labels', id, nSpectra);
      return;
    }
    sendText(res, 404, `not found: ${pathname}`);
    return;
  }

  // mode A
  if (pathname === '/' || pathname === '/index.html') {
    await serveHTML(res, INDEX_HTML);
    return;
  }
  if (pathname === '/circles' || pathname === '/index-circles.html') {
    await serveHTML(res, INDEX_CIRCLES);
    return;
  }
  if (pathname === '/api/meta') { sendJSON(res, 200, meta); return; }
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

async function main() {
  if (datasetMode) {
    await setupDatasetMode();
  } else {
    await setupFolder51Mode();
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
    if (datasetMode) {
      console.log('training-dataset viewer (labels ○ azules + inferencia × rojas)');
      console.log(`  dataset  : ${dsDir} (${nSpectra} espectros, ${dsConfig?.config?.nbPoints?.x ?? '?'}×${dsConfig?.config?.nbPoints?.y ?? '?'})`);
      console.log(`  inference: ${infDir ?? '(no encontrada — sin cruces rojas)'}`);
      console.log(`  open     : http://localhost:${args.port}/`);
    } else {
      const peaks = meta.peaks ? meta.peaks.x.length : 0;
      console.log('folder-51 2D contour viewer (spectrum + P3 + peaks)');
      console.log(`  data    : ${datasetDir} (shape ${meta.shape[0]}x${meta.shape[1]}, ${peaks} peaks)`);
      console.log(`  sources : ${meta.source.ft2} | ${meta.source.peakSource}`);
      console.log(`  open    : http://localhost:${args.port}/`);
    }
  });
}

main().catch((err) => fail(err.message));