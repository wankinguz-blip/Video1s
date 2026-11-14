// Extracts the official VANKING.UZ mole head (square green glasses) from the brand SVG
// into a flat list of canvas draw operations: src/assets/mascot-head.js
//
//   node scripts/extract-mascot.mjs [assets/brand/vanking-day-uz.svg]
//
// Supported SVG subset (what the brand art uses): g/path/rect/circle/ellipse, transforms,
// inherited fill/stroke/linecap/linejoin, opacity, clip-path (path/rect), linear & radial
// gradients (userSpaceOnUse + objectBoundingBox, gradientTransform, stop-opacity).
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { XMLParser } from 'fast-xml-parser';
import { pathBBox } from '../src/svgpath.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const src = resolve(root, process.argv[2] || 'assets/brand/vanking-day-uz.svg');
const out = resolve(root, 'src/assets/mascot-head.js');

const parser = new XMLParser({ ignoreAttributes: false, attributeNamePrefix: '', preserveOrder: true });
const doc = parser.parse(readFileSync(src, 'utf8'));

// ---------- tree helpers (preserveOrder format: [{tag:[children], ':@':{attrs}}]) ----------
const tagOf = (n) => Object.keys(n).find((k) => k !== ':@');
const attrs = (n) => n[':@'] || {};
const kids = (n) => n[tagOf(n)] || [];
function walk(nodes, fn) { for (const n of nodes) { const t = tagOf(n); if (!t || t === '#text') continue; fn(n, t); walk(kids(n), fn); } }

const byId = new Map();
walk(doc, (n) => { const id = attrs(n).id; if (id) byId.set(id, n); });

// ---------- matrices ----------
const I = [1, 0, 0, 1, 0, 0];
const mul = (m, n) => [
  m[0] * n[0] + m[2] * n[1], m[1] * n[0] + m[3] * n[1],
  m[0] * n[2] + m[2] * n[3], m[1] * n[2] + m[3] * n[3],
  m[0] * n[4] + m[2] * n[5] + m[4], m[1] * n[4] + m[3] * n[5] + m[5],
];
function parseTransform(s) {
  let m = I;
  if (!s) return m;
  const re = /(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)/g;
  let r;
  while ((r = re.exec(s))) {
    const a = r[2].trim().split(/[\s,]+/).map(Number);
    let t;
    switch (r[1]) {
      case 'matrix': t = a; break;
      case 'translate': t = [1, 0, 0, 1, a[0], a[1] || 0]; break;
      case 'scale': t = [a[0], 0, 0, a.length > 1 ? a[1] : a[0], 0, 0]; break;
      case 'rotate': {
        const k = (a[0] * Math.PI) / 180, c = Math.cos(k), s2 = Math.sin(k);
        t = [c, s2, -s2, c, 0, 0];
        if (a.length === 3) t = mul(mul([1, 0, 0, 1, a[1], a[2]], t), [1, 0, 0, 1, -a[1], -a[2]]);
        break;
      }
      case 'skewX': t = [1, 0, Math.tan((a[0] * Math.PI) / 180), 1, 0, 0]; break;
      case 'skewY': t = [1, Math.tan((a[0] * Math.PI) / 180), 0, 1, 0, 0]; break;
    }
    m = mul(m, t);
  }
  return m;
}
const isIdentity = (m) => m.every((v, i) => Math.abs(v - I[i]) < 1e-9);
const round = (v) => Math.round(v * 1e4) / 1e4;

// ---------- shapes -> path data ----------
function shapeToD(tag, a) {
  const n = (k, d = 0) => (a[k] === undefined ? d : Number(a[k]));
  if (tag === 'path') return a.d;
  if (tag === 'circle') { const cx = n('cx'), cy = n('cy'), r = n('r'); return `M${cx - r} ${cy}A${r} ${r} 0 1 0 ${cx + r} ${cy}A${r} ${r} 0 1 0 ${cx - r} ${cy}Z`; }
  if (tag === 'ellipse') { const cx = n('cx'), cy = n('cy'), rx = n('rx'), ry = n('ry'); return `M${cx - rx} ${cy}A${rx} ${ry} 0 1 0 ${cx + rx} ${cy}A${rx} ${ry} 0 1 0 ${cx - rx} ${cy}Z`; }
  if (tag === 'rect') {
    const x = n('x'), y = n('y'), w = n('width'), h = n('height');
    let rx = a.rx !== undefined ? n('rx') : n('ry'), ry = a.ry !== undefined ? n('ry') : rx;
    rx = Math.min(rx, w / 2); ry = Math.min(ry, h / 2);
    if (!rx) return `M${x} ${y}H${x + w}V${y + h}H${x}Z`;
    return `M${x + rx} ${y}H${x + w - rx}A${rx} ${ry} 0 0 1 ${x + w} ${y + ry}V${y + h - ry}A${rx} ${ry} 0 0 1 ${x + w - rx} ${y + h}H${x + rx}A${rx} ${ry} 0 0 1 ${x} ${y + h - ry}V${y + ry}A${rx} ${ry} 0 0 1 ${x + rx} ${y}Z`;
  }
  return null;
}

// ---------- gradients ----------
const gradients = {};
function colorWithOpacity(c, o) {
  if (o === undefined || Number(o) === 1) return c;
  const h = c.replace('#', '');
  const v = h.length === 3 ? h.split('').map((x) => x + x).join('') : h;
  const r = parseInt(v.slice(0, 2), 16), g = parseInt(v.slice(2, 4), 16), b = parseInt(v.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${Number(o)})`;
}
function gradient(id) {
  if (gradients[id]) return id;
  const node = byId.get(id);
  if (!node) throw new Error(`missing gradient ${id}`);
  const t = tagOf(node), a = attrs(node);
  const obb = (a.gradientUnits || 'objectBoundingBox') === 'objectBoundingBox';
  const num = (k, d) => {
    if (a[k] === undefined) return d;
    const s = String(a[k]);
    return s.endsWith('%') ? parseFloat(s) / 100 : Number(s);
  };
  const stops = kids(node).filter((k) => tagOf(k) === 'stop').map((k) => {
    const s = attrs(k);
    return {
      offset: s.offset === undefined ? 0 : String(s.offset).endsWith('%') ? parseFloat(s.offset) / 100 : Number(s.offset),
      color: colorWithOpacity(s['stop-color'] || '#000', s['stop-opacity']),
    };
  });
  const g = { type: t === 'linearGradient' ? 'linear' : 'radial', obb, stops };
  if (g.type === 'linear') Object.assign(g, { x1: num('x1', 0), y1: num('y1', 0), x2: num('x2', 1), y2: num('y2', 0) });
  else {
    const cx = num('cx', 0.5), cy = num('cy', 0.5);
    Object.assign(g, { cx, cy, r: num('r', 0.5), fx: num('fx', cx), fy: num('fy', cy) });
  }
  const gt = parseTransform(a.gradientTransform);
  if (!isIdentity(gt)) g.transform = gt.map(round);
  gradients[id] = g;
  return id;
}
const urlId = (v) => (typeof v === 'string' ? (v.match(/url\(#([^)]+)\)/) || [])[1] : undefined);

// ---------- clip paths ----------
function clipFor(id, ctm) {
  const node = byId.get(id);
  const list = [];
  for (const k of kids(node)) {
    const t = tagOf(k);
    if (!t || t === '#text') continue;
    const a = attrs(k);
    list.push({ d: shapeToD(t, a), m: mul(ctm, parseTransform(a.transform)).map(round) });
  }
  return { id, shapes: list };
}

// ---------- walk head-art ----------
const head = byId.get(process.env.HEAD_ID || 'head-art');
if (!head) throw new Error('head-art not found');
const ops = [];
const INHERIT = ['fill', 'stroke', 'stroke-width', 'stroke-linecap', 'stroke-linejoin'];

function visit(node, ctm, style, clips, groups) {
  const t = tagOf(node);
  if (!t || t === '#text' || t === 'title' || t === 'desc' || t === 'clipPath' || t === 'defs') return;
  const a = attrs(node);
  const m = mul(ctm, parseTransform(a.transform));
  const st = { ...style };
  for (const k of INHERIT) if (a[k] !== undefined) st[k] = a[k];
  const myClips = a['clip-path'] ? [...clips, clipFor(urlId(a['clip-path']), m)] : clips;
  if (t === 'g') {
    const gs = a.id ? [...groups, a.id] : groups;
    if (a.opacity !== undefined && Number(a.opacity) !== 1) console.warn('group opacity ignored', a.id);
    for (const k of kids(node)) visit(k, m, st, myClips, gs);
    return;
  }
  const d = shapeToD(t, a);
  if (!d) return;
  const op = { d };
  if (a.id) op.id = a.id;
  if (groups.length > 1) op.groups = groups.slice(1);
  const fill = st.fill === undefined ? '#000' : st.fill;
  if (fill !== 'none') op.fill = urlId(fill) ? { grad: gradient(urlId(fill)) } : fill;
  if (st.stroke && st.stroke !== 'none') {
    op.stroke = urlId(st.stroke) ? { grad: gradient(urlId(st.stroke)) } : st.stroke;
    op.lw = Number(st['stroke-width'] || 1);
    op.cap = st['stroke-linecap'] || 'butt';
    op.join = st['stroke-linejoin'] || 'miter';
  }
  if (a.opacity !== undefined && Number(a.opacity) !== 1) op.alpha = Number(a.opacity);
  if (a['stroke-opacity'] !== undefined) op.strokeAlpha = Number(a['stroke-opacity']);
  if (!isIdentity(m)) op.m = m.map(round);
  if (myClips.length) op.clips = myClips;
  if ((op.fill && op.fill.grad && gradients[op.fill.grad].obb) || (op.stroke && op.stroke.grad && gradients[op.stroke.grad].obb)) {
    const b = pathBBox(d);
    op.bbox = [b.x, b.y, b.w, b.h].map(round);
  }
  ops.push(op);
}
visit(head, I, {}, [], []);

// ---------- semantic tags used by the animation (mouth, pupils, eyes, whiskers) ----------
const tag = (pred, name) => { const hits = ops.filter(pred); for (const o of hits) o.tag = name; return hits.length; };
const startsWith = (p) => (o) => o.d.replace(/\s+/g, ' ').trim().startsWith(p);
const report = {
  mouthInterior: tag((o) => o.fill === '#683b24', 'mouth'),
  toothL: tag(startsWith('M576 465'), 'tooth'),
  toothR: tag(startsWith('M602 460'), 'tooth'),
  tongue: tag((o) => o.fill === '#b95f45' || (o.stroke === '#d88363'), 'tongue'),
  smile: tag(startsWith('M543 432'), 'smile'),
  dimples: tag((o) => startsWith('M537 434')(o) || startsWith('M675 462')(o), 'dimple'),
  chin: tag(startsWith('M601 423'), 'philtrum'),
  pupils: tag((o) => o.fill && o.fill.grad === 'head-pupil', 'pupil'),
  glints: tag((o) => o.groups && o.groups.some((g) => g.startsWith('eye-')) && o.fill === '#fffef7', 'glint'),
};
const bounds = ops.reduce((b, o) => {
  const bb = pathBBox(o.d);
  return { x0: Math.min(b.x0, bb.x), y0: Math.min(b.y0, bb.y), x1: Math.max(b.x1, bb.x + bb.w), y1: Math.max(b.y1, bb.y + bb.h) };
}, { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity });

mkdirSync(dirname(out), { recursive: true });
writeFileSync(out,
  `// AUTO-GENERATED by scripts/extract-mascot.mjs from ${src.replace(root + '/', '')} — do not edit by hand.\n` +
  `// Official VANKING.UZ mole head (square green glasses). Coordinates are the brand SVG user space.\n` +
  `export const HEAD_BOUNDS = ${JSON.stringify(bounds)};\n` +
  `export const EYE_CENTERS = { left: [548, 333], right: [734, 366] };\n` +
  `export const HEAD_GRADIENTS = ${JSON.stringify(gradients)};\n` +
  `export const HEAD_OPS = ${JSON.stringify(ops)};\n`);
console.log(`wrote ${out}: ${ops.length} ops, ${Object.keys(gradients).length} gradients`, report, bounds);
