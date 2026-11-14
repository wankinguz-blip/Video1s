// Math helpers. Everything that animates is a pure function of loop time t ∈ [0, LOOP).
export const TAU = Math.PI * 2;
export const DEG = Math.PI / 180;

export const clamp = (v, a = 0, b = 1) => (v < a ? a : v > b ? b : v);
export const lerp = (a, b, u) => a + (b - a) * u;
export const invLerp = (a, b, v) => clamp((v - a) / (b - a));
export const smoothstep = (a, b, v) => { const u = invLerp(a, b, v); return u * u * (3 - 2 * u); };
export const smootherstep = (a, b, v) => { const u = invLerp(a, b, v); return u * u * u * (u * (u * 6 - 15) + 10); };
export const easeOutCubic = (u) => 1 - Math.pow(1 - clamp(u), 3);
export const easeInCubic = (u) => Math.pow(clamp(u), 3);
export const easeInOutSine = (u) => 0.5 - 0.5 * Math.cos(Math.PI * clamp(u));
export const mod = (a, n) => ((a % n) + n) % n;

/** Shortest signed distance from a to b on a circle of circumference n. */
export const wrapDelta = (a, b, n) => { let d = mod(b - a, n); if (d > n / 2) d -= n; return d; };

/** Smooth (C∞-ish) bump centred at c with half-width w on a periodic timeline. */
export function bump(t, c, w, period) {
  const d = period ? wrapDelta(c, t, period) : t - c;
  const u = Math.abs(d) / w;
  if (u >= 1) return 0;
  const k = 1 - u * u;
  return k * k * k;
}

/** Envelope that rises over [a, a+rise], holds, and falls over [b-fall, b] (smootherstep edges). */
export const envelope = (t, a, b, rise, fall) => smootherstep(a, a + rise, t) * (1 - smootherstep(b - fall, b, t));

// ---------- deterministic randomness ----------
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function hash1(n) {
  let x = Math.imul((n | 0) ^ 0x9e3779b9, 0x85ebca6b);
  x ^= x >>> 13; x = Math.imul(x, 0xc2b2ae35); x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}
export const hash2 = (a, b) => hash1(Math.imul(a | 0, 73856093) ^ Math.imul(b | 0, 19349663));

/**
 * Periodic smooth noise: sum of sines with integer harmonics of the loop frequency, so it
 * repeats exactly every `period` seconds with continuous derivatives.
 */
export function loopNoise(t, period, seed, harmonics = [1, 2, 3, 5]) {
  let v = 0, norm = 0;
  for (let i = 0; i < harmonics.length; i++) {
    const h = harmonics[i];
    const amp = 1 / h;
    v += amp * Math.sin((TAU * h * t) / period + TAU * hash2(seed, i));
    norm += amp;
  }
  return v / norm;
}

// ---------- periodic keyframe splines (C1, loop-closed) ----------
/**
 * keys: [[time, value], ...] sorted by time within [0, period). Values may be numbers or arrays.
 * Cardinal (Catmull-Rom, non-uniform aware) interpolation that wraps around the period,
 * giving position AND velocity continuity at the loop boundary.
 */
export function loopSpline(keys, period) {
  const n = keys.length;
  const times = keys.map((k) => k[0]);
  const vals = keys.map((k) => k[1]);
  const isArr = Array.isArray(vals[0]);
  const at = (i) => { const j = mod(i, n); const wraps = Math.floor(i / n); return [times[j] + wraps * period, vals[j]]; };
  const tangent = (i) => {
    const [t0, v0] = at(i - 1), [t2, v2] = at(i + 1);
    const dt = t2 - t0;
    return isArr ? v0.map((_, k) => (v2[k] - v0[k]) / dt) : (v2 - v0) / dt;
  };
  const tangents = keys.map((_, i) => tangent(i));
  return (t) => {
    const tt = mod(t, period);
    let i = n - 1;
    for (let k = 0; k < n; k++) if (times[k] > tt) { i = k - 1; break; }
    let [ta, va] = at(i), [tb, vb] = at(i + 1);
    let x = tt;
    if (i < 0) { [ta, va] = at(-1); [tb, vb] = at(0); }
    if (x < ta) x += period;
    const ma = tangents[mod(i, n)], mb = tangents[mod(i + 1, n)];
    const h = tb - ta, u = (x - ta) / h;
    const h00 = 2 * u ** 3 - 3 * u ** 2 + 1, h10 = u ** 3 - 2 * u ** 2 + u, h01 = -2 * u ** 3 + 3 * u ** 2, h11 = u ** 3 - u ** 2;
    if (!isArr) return h00 * va + h10 * h * ma + h01 * vb + h11 * h * mb;
    return va.map((_, k) => h00 * va[k] + h10 * h * ma[k] + h01 * vb[k] + h11 * h * mb[k]);
  };
}

// ---------- 2D helpers ----------
export const v2 = (x, y) => ({ x, y });
export const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y });
export const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
export const scale = (a, s) => ({ x: a.x * s, y: a.y * s });
export const len = (a) => Math.hypot(a.x, a.y);
export const rot = (a, ang) => { const c = Math.cos(ang), s = Math.sin(ang); return { x: a.x * c - a.y * s, y: a.x * s + a.y * c }; };
export const mix2 = (a, b, u) => ({ x: lerp(a.x, b.x, u), y: lerp(a.y, b.y, u) });

/**
 * Two-bone IK. Returns the joint (elbow/knee) position for root→joint→end with lengths l1, l2.
 * bendSign (+1/-1) picks which side the joint bends to. Target is clamped to reach.
 */
export function ik2(root, target, l1, l2, bendSign) {
  let d = sub(target, root);
  let dist = len(d);
  const maxR = (l1 + l2) * 0.999, minR = Math.abs(l1 - l2) * 1.001 + 1e-3;
  if (dist > maxR) { d = scale(d, maxR / dist); dist = maxR; }
  if (dist < minR) { d = dist < 1e-6 ? { x: minR, y: 0 } : scale(d, minR / dist); dist = minR; }
  const a = Math.acos(clamp((l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist), -1, 1));
  const base = Math.atan2(d.y, d.x);
  const ang = base + bendSign * a;
  const joint = { x: root.x + Math.cos(ang) * l1, y: root.y + Math.sin(ang) * l1 };
  const end = add(root, d);
  return { joint, end };
}

// ---------- colour ----------
export function hexToRgb(h) {
  const v = h.replace('#', '');
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
}
export const rgba = (c, a = 1) => `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a})`;
export const mixRgb = (a, b, u) => [lerp(a[0], b[0], u), lerp(a[1], b[1], u), lerp(a[2], b[2], u)];
