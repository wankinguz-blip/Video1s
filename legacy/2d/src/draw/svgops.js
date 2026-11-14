// Renders extracted SVG draw-ops (see scripts/extract-mascot.mjs) onto a Canvas 2D context.
// Works with any Canvas2D implementation (browser, @napi-rs/canvas).

const pathCache = new Map();
let Path2DImpl = typeof Path2D !== 'undefined' ? Path2D : null;
export function setPath2D(P) { Path2DImpl = P; }
export function P(d) {
  let p = pathCache.get(d);
  if (!p) { p = new Path2DImpl(d); pathCache.set(d, p); }
  return p;
}

const invert = (m) => {
  const det = m[0] * m[3] - m[1] * m[2];
  return [m[3] / det, -m[1] / det, -m[2] / det, m[0] / det, (m[2] * m[5] - m[3] * m[4]) / det, (m[1] * m[4] - m[0] * m[5]) / det];
};
const mul = (m, n) => [
  m[0] * n[0] + m[2] * n[1], m[1] * n[0] + m[3] * n[1],
  m[0] * n[2] + m[2] * n[3], m[1] * n[2] + m[3] * n[3],
  m[0] * n[4] + m[2] * n[5] + m[4], m[1] * n[4] + m[3] * n[5] + m[5],
];
const apply = (m, x, y) => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];

function makeGradient(ctx, g) {
  const gr = g.type === 'linear'
    ? ctx.createLinearGradient(g.x1, g.y1, g.x2, g.y2)
    : ctx.createRadialGradient(g.fx, g.fy, 0, g.cx, g.cy, g.r);
  for (const s of g.stops) gr.addColorStop(Math.min(1, Math.max(0, s.offset)), s.color);
  return gr;
}

/** Fill `path` with gradient `g`; handles objectBoundingBox units and gradientTransform. */
function fillGradient(ctx, path, g, bbox) {
  let M = [1, 0, 0, 1, 0, 0];
  if (g.obb) M = [bbox[2], 0, 0, bbox[3], bbox[0], bbox[1]];
  if (g.transform) M = mul(M, g.transform);
  const ident = M[0] === 1 && M[1] === 0 && M[2] === 0 && M[3] === 1 && M[4] === 0 && M[5] === 0;
  if (ident) { ctx.fillStyle = makeGradient(ctx, g); ctx.fill(path); return; }
  // Clip to the shape in user space, then paint a rect in gradient space that covers the shape.
  ctx.save();
  ctx.clip(path);
  ctx.transform(M[0], M[1], M[2], M[3], M[4], M[5]);
  const inv = invert(M);
  const b = bbox || [0, 0, 1200, 1200];
  const corners = [apply(inv, b[0], b[1]), apply(inv, b[0] + b[2], b[1]), apply(inv, b[0], b[1] + b[3]), apply(inv, b[0] + b[2], b[1] + b[3])];
  const xs = corners.map((c) => c[0]), ys = corners.map((c) => c[1]);
  const x0 = Math.min(...xs), y0 = Math.min(...ys), x1 = Math.max(...xs), y1 = Math.max(...ys);
  ctx.fillStyle = makeGradient(ctx, g);
  ctx.fillRect(x0 - 2, y0 - 2, x1 - x0 + 4, y1 - y0 + 4);
  ctx.restore();
}

function applyClips(ctx, clips) {
  // NB: canvas clip regions intersect; an SVG clipPath with several shapes would be a union,
  // but the brand art only uses single-shape clip paths. restore() would also drop the clip,
  // so the clip transform is undone with its inverse instead.
  for (const c of clips) {
    for (const s of c.shapes) {
      const p = P(s.d);
      if (s.m) {
        const m = s.m, inv = invert(m);
        ctx.transform(m[0], m[1], m[2], m[3], m[4], m[5]);
        ctx.clip(p);
        ctx.transform(inv[0], inv[1], inv[2], inv[3], inv[4], inv[5]);
      } else ctx.clip(p);
    }
  }
}

/**
 * Draw one op. `style` overrides: { tint?: (color)=>color }.
 */
export function drawOp(ctx, op, gradients, opts = {}) {
  const needsSave = op.m || op.clips || op.alpha !== undefined;
  if (needsSave) ctx.save();
  if (op.clips) applyClips(ctx, op.clips);
  if (op.m) ctx.transform(op.m[0], op.m[1], op.m[2], op.m[3], op.m[4], op.m[5]);
  if (op.alpha !== undefined) ctx.globalAlpha *= op.alpha;
  const path = P(op.d);
  if (op.fill) {
    if (typeof op.fill === 'string') { ctx.fillStyle = opts.color ? opts.color(op.fill) : op.fill; ctx.fill(path); }
    else fillGradient(ctx, path, gradients[op.fill.grad], op.bbox);
  }
  if (op.stroke) {
    ctx.lineWidth = op.lw;
    ctx.lineCap = op.cap;
    ctx.lineJoin = op.join;
    if (typeof op.stroke === 'string') ctx.strokeStyle = opts.color ? opts.color(op.stroke) : op.stroke;
    else ctx.strokeStyle = makeGradient(ctx, gradients[op.stroke.grad]);
    if (op.strokeAlpha !== undefined) ctx.globalAlpha *= op.strokeAlpha;
    ctx.stroke(path);
  }
  if (needsSave) ctx.restore();
}
