// Minimal SVG path-data utilities (parse, flatten to points, bounding box).
// Used offline for gradient bounding boxes and at runtime for sampling outlines.

const CMD = /([MmLlHhVvCcSsQqTtAaZz])|(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)/g;

export function parsePath(d) {
  const toks = [];
  let m;
  CMD.lastIndex = 0;
  while ((m = CMD.exec(d))) toks.push(m[1] ? m[1] : Number(m[2]));
  const segs = [];
  let i = 0, cmd = null;
  const ARGS = { M: 2, L: 2, H: 1, V: 1, C: 6, S: 4, Q: 4, T: 2, A: 7, Z: 0 };
  while (i < toks.length) {
    if (typeof toks[i] === 'string') cmd = toks[i++];
    const up = cmd.toUpperCase();
    const n = ARGS[up];
    if (n === 0) { segs.push([cmd]); continue; }
    const args = toks.slice(i, i + n);
    i += n;
    segs.push([cmd, ...args]);
    if (cmd === 'M') cmd = 'L';
    else if (cmd === 'm') cmd = 'l';
  }
  return segs;
}

function arcPoints(x1, y1, rx, ry, phi, fa, fs, x2, y2, steps) {
  // SVG arc endpoint -> center parameterization (F.6.5)
  const pts = [];
  if (rx === 0 || ry === 0) return [[x2, y2]];
  rx = Math.abs(rx); ry = Math.abs(ry);
  const c = Math.cos(phi), s = Math.sin(phi);
  const dx = (x1 - x2) / 2, dy = (y1 - y2) / 2;
  const xp = c * dx + s * dy, yp = -s * dx + c * dy;
  const lam = (xp * xp) / (rx * rx) + (yp * yp) / (ry * ry);
  if (lam > 1) { rx *= Math.sqrt(lam); ry *= Math.sqrt(lam); }
  const num = rx * rx * ry * ry - rx * rx * yp * yp - ry * ry * xp * xp;
  const den = rx * rx * yp * yp + ry * ry * xp * xp;
  let k = Math.sqrt(Math.max(0, num / den));
  if (fa === fs) k = -k;
  const cxp = (k * rx * yp) / ry, cyp = (-k * ry * xp) / rx;
  const cx = c * cxp - s * cyp + (x1 + x2) / 2, cy = s * cxp + c * cyp + (y1 + y2) / 2;
  const ang = (ux, uy, vx, vy) => {
    const a = Math.atan2(ux * vy - uy * vx, ux * vx + uy * vy);
    return a;
  };
  const t1 = ang(1, 0, (xp - cxp) / rx, (yp - cyp) / ry);
  let dt = ang((xp - cxp) / rx, (yp - cyp) / ry, (-xp - cxp) / rx, (-yp - cyp) / ry);
  if (!fs && dt > 0) dt -= 2 * Math.PI;
  if (fs && dt < 0) dt += 2 * Math.PI;
  for (let j = 1; j <= steps; j++) {
    const t = t1 + (dt * j) / steps;
    pts.push([cx + rx * Math.cos(t) * c - ry * Math.sin(t) * s, cy + rx * Math.cos(t) * s + ry * Math.sin(t) * c]);
  }
  return pts;
}

/** Flatten path to an array of subpaths, each an array of [x, y] points. */
export function flattenPath(d, steps = 16) {
  const segs = typeof d === 'string' ? parsePath(d) : d;
  const subs = [];
  let cur = null, x = 0, y = 0, sx = 0, sy = 0, lcx = 0, lcy = 0, lq = null, prev = '';
  const push = (px, py) => cur.push([px, py]);
  for (const s of segs) {
    const c = s[0], up = c.toUpperCase(), rel = c !== up;
    const a = s.slice(1);
    const ox = rel ? x : 0, oy = rel ? y : 0;
    switch (up) {
      case 'M': x = a[0] + ox; y = a[1] + oy; sx = x; sy = y; cur = [[x, y]]; subs.push(cur); break;
      case 'L': x = a[0] + ox; y = a[1] + oy; push(x, y); break;
      case 'H': x = a[0] + (rel ? x : 0); push(x, y); break;
      case 'V': y = a[0] + (rel ? y : 0); push(x, y); break;
      case 'C': case 'S': {
        let c1x, c1y, c2x, c2y, ex, ey;
        if (up === 'C') { c1x = a[0] + ox; c1y = a[1] + oy; c2x = a[2] + ox; c2y = a[3] + oy; ex = a[4] + ox; ey = a[5] + oy; }
        else {
          const p = prev.toUpperCase();
          c1x = p === 'C' || p === 'S' ? 2 * x - lcx : x; c1y = p === 'C' || p === 'S' ? 2 * y - lcy : y;
          c2x = a[0] + ox; c2y = a[1] + oy; ex = a[2] + ox; ey = a[3] + oy;
        }
        for (let j = 1; j <= steps; j++) {
          const t = j / steps, u = 1 - t;
          push(u * u * u * x + 3 * u * u * t * c1x + 3 * u * t * t * c2x + t * t * t * ex,
            u * u * u * y + 3 * u * u * t * c1y + 3 * u * t * t * c2y + t * t * t * ey);
        }
        lcx = c2x; lcy = c2y; x = ex; y = ey; break;
      }
      case 'Q': case 'T': {
        let qx, qy, ex, ey;
        if (up === 'Q') { qx = a[0] + ox; qy = a[1] + oy; ex = a[2] + ox; ey = a[3] + oy; }
        else { const p = prev.toUpperCase(); qx = (p === 'Q' || p === 'T') && lq ? 2 * x - lq[0] : x; qy = (p === 'Q' || p === 'T') && lq ? 2 * y - lq[1] : y; ex = a[0] + ox; ey = a[1] + oy; }
        for (let j = 1; j <= steps; j++) {
          const t = j / steps, u = 1 - t;
          push(u * u * x + 2 * u * t * qx + t * t * ex, u * u * y + 2 * u * t * qy + t * t * ey);
        }
        lq = [qx, qy]; x = ex; y = ey; break;
      }
      case 'A': {
        const ex = a[5] + ox, ey = a[6] + oy;
        for (const p of arcPoints(x, y, a[0], a[1], (a[2] * Math.PI) / 180, a[3] ? 1 : 0, a[4] ? 1 : 0, ex, ey, steps)) push(p[0], p[1]);
        x = ex; y = ey; break;
      }
      case 'Z': if (cur) push(sx, sy); x = sx; y = sy; break;
    }
    prev = c;
  }
  return subs;
}

export function pathBBox(d) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const sub of flattenPath(d, 24)) for (const [x, y] of sub) {
    if (x < x0) x0 = x; if (y < y0) y0 = y; if (x > x1) x1 = x; if (y > y1) y1 = y;
  }
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}
