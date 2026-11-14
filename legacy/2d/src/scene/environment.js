// Environment: sky, moon, stars, meadow, distant hills, the three stage hills, grass, fireflies.
// All coordinates are world units (screen height 1000). Animated parts use periods that divide 15 s.
import { LOOP } from '../config.js';
import { TAU, clamp, hash2, hash1, lerp, smoothstep, mulberry32 } from '../math.js';

// ---------------------------------------------------------------- terrain helpers
const terrainNoise = (x, seed) =>
  0.55 * Math.sin(x * 0.00117 + seed * 1.7) + 0.3 * Math.sin(x * 0.00291 + seed * 3.1) + 0.15 * Math.sin(x * 0.0071 + seed * 5.3);

export const HORIZON_Y = 842;
const farY = (x) => 806 + 22 * terrainNoise(x, 1) + 9 * Math.sin(x * 0.0123 + 0.4);
const midY = (x) => 848 + 26 * terrainNoise(x + 3000, 2) + 7 * Math.sin(x * 0.0157 + 1.1);

/** Height of a stage hill's surface at world x (Infinity when outside the hill). */
export function hillY(h, x) {
  const u = (x - h.x) / h.halfW;
  if (Math.abs(u) >= 1) return Infinity;
  return h.base - (h.base - h.top) * Math.pow(1 - u * u, 0.6);
}
/** Surface normal angle (radians, 0 = straight up) of a hill at x. */
function hillSlope(h, x) {
  const d = 1.5;
  return Math.atan2(hillY(h, x + d) - hillY(h, x - d), 2 * d);
}

function hillPath(ctx, h) {
  ctx.beginPath();
  const n = 160;
  for (let i = 0; i <= n; i++) {
    const u = -1 + (2 * i) / n;
    const x = h.x + u * h.halfW;
    const y = h.base - (h.base - h.top) * Math.pow(Math.max(0, 1 - u * u), 0.6);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.lineTo(h.x + h.halfW, 1100);
  ctx.lineTo(h.x - h.halfW, 1100);
  ctx.closePath();
}

function addHillShape(ctx, h) {
  const n = 120;
  for (let i = 0; i <= n; i++) {
    const u = -1 + (2 * i) / n;
    const x = h.x + u * h.halfW;
    const y = h.base - (h.base - h.top) * Math.pow(Math.max(0, 1 - u * u), 0.6);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.lineTo(h.x + h.halfW, 1100);
  ctx.lineTo(h.x - h.halfW, 1100);
  ctx.closePath();
}

/** Firework light falling on the hills and meadow (additive, clipped to the ground). */
export function drawGroundLight(ctx, L, view, lights) {
  const act = lights.filter((l) => l.i > 0.012);
  if (!act.length) return;
  ctx.save();
  ctx.beginPath();
  for (const h of L.hills) if (h.x + h.halfW > view.x0 && h.x - h.halfW < view.x1) addHillShape(ctx, h);
  ctx.moveTo(view.x0 - 10, 1010);
  for (let x = view.x0 - 10; x <= view.x1 + 10; x += 12) ctx.lineTo(x, 905 + 10 * terrainNoise(x + 777, 4));
  ctx.lineTo(view.x1 + 10, 1010);
  ctx.closePath();
  ctx.clip();
  ctx.globalCompositeOperation = 'lighter';
  for (const l of act) {
    const r = l.r * 0.8;
    const g = ctx.createRadialGradient(l.x, l.y, 0, l.x, l.y, r);
    g.addColorStop(0, `rgba(${l.c[0]},${l.c[1]},${l.c[2]},${0.24 * Math.min(1, l.i)})`);
    g.addColorStop(0.45, `rgba(${l.c[0]},${l.c[1]},${l.c[2]},${0.11 * Math.min(1, l.i)})`);
    g.addColorStop(1, `rgba(${l.c[0]},${l.c[1]},${l.c[2]},0)`);
    ctx.fillStyle = g;
    ctx.fillRect(Math.max(view.x0, l.x - r), 600, Math.min(view.x1, l.x + r) - Math.max(view.x0, l.x - r), 420);
  }
  ctx.restore();
}

const hillOrder = (L) => [...L.hills].sort((a, b) => (a.role === 'center') - (b.role === 'center'));
/** Topmost surface y at x over all stage hills (front-most hill wins visually). */
export function groundY(L, x) {
  let y = Infinity;
  for (const h of L.hills) y = Math.min(y, hillY(h, x));
  return y;
}

// ---------------------------------------------------------------- sky
export function drawSky(ctx, L, view) {
  const x0 = view.x0, x1 = view.x1;
  const g = ctx.createLinearGradient(0, 0, 0, 1000);
  g.addColorStop(0.0, '#050822');
  g.addColorStop(0.28, '#0b1442');
  g.addColorStop(0.52, '#1a1d5c');
  g.addColorStop(0.68, '#2e2470');
  g.addColorStop(0.79, '#4a2f7c');
  g.addColorStop(0.86, '#6a3f86');
  g.addColorStop(1.0, '#3a2a60');
  ctx.fillStyle = g;
  ctx.fillRect(x0 - 2, -2, x1 - x0 + 4, 1004);
  // indigo/violet glow hugging the horizon, strongest behind the centre stage
  const cx = L.chars.singer.x;
  const span = Math.max(1600, L.worldW * 0.45);
  for (const [px, r, a] of [[cx, span, 0.34], [L.chars.emo.x, span * 0.6, 0.18], [L.chars.sax.x, span * 0.6, 0.18]]) {
    ctx.save();
    ctx.translate(px, 880);
    ctx.scale(1, 0.32);
    const rg = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
    rg.addColorStop(0, `rgba(150,90,170,${a})`);
    rg.addColorStop(0.5, `rgba(95,60,150,${a * 0.45})`);
    rg.addColorStop(1, 'rgba(60,40,120,0)');
    ctx.fillStyle = rg;
    ctx.fillRect(-r, -r, 2 * r, 2 * r);
    ctx.restore();
  }
}

export function drawMoon(ctx, L) {
  const { x, y, r } = L.moon;
  // wide soft halo
  let rg = ctx.createRadialGradient(x, y, r * 0.8, x, y, r * 9);
  rg.addColorStop(0, 'rgba(190,205,255,0.22)');
  rg.addColorStop(0.35, 'rgba(140,160,240,0.07)');
  rg.addColorStop(1, 'rgba(120,140,230,0)');
  ctx.fillStyle = rg;
  ctx.fillRect(x - r * 9, y - r * 9, r * 18, r * 18);
  rg = ctx.createRadialGradient(x, y, r * 0.9, x, y, r * 2.2);
  rg.addColorStop(0, 'rgba(240,240,255,0.35)');
  rg.addColorStop(1, 'rgba(200,210,255,0)');
  ctx.fillStyle = rg;
  ctx.fillRect(x - r * 2.2, y - r * 2.2, r * 4.4, r * 4.4);
  // disc
  rg = ctx.createRadialGradient(x - r * 0.3, y - r * 0.35, r * 0.1, x, y, r);
  rg.addColorStop(0, '#fffdf2');
  rg.addColorStop(0.7, '#f3ecd2');
  rg.addColorStop(1, '#ddd2b4');
  ctx.fillStyle = rg;
  ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
  ctx.save();
  ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.clip();
  ctx.fillStyle = 'rgba(160,150,125,0.28)';
  for (const [dx, dy, cr] of [[-0.35, -0.1, 0.22], [0.25, 0.3, 0.16], [0.1, -0.42, 0.1], [0.45, -0.15, 0.09], [-0.15, 0.45, 0.12]]) {
    ctx.beginPath(); ctx.arc(x + dx * r, y + dy * r, cr * r, 0, TAU); ctx.fill();
  }
  // terminator shading for a slightly 3-D look
  const sg = ctx.createLinearGradient(x - r, y - r, x + r, y + r);
  sg.addColorStop(0.45, 'rgba(60,60,110,0)');
  sg.addColorStop(1, 'rgba(60,60,110,0.32)');
  ctx.fillStyle = sg;
  ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
  ctx.restore();
}

// ---------------------------------------------------------------- stars
const starCache = new Map();
function starsFor(L) {
  const key = `${L.mode}:${L.worldW.toFixed(1)}`;
  if (starCache.has(key)) return starCache.get(key);
  const cell = 46;
  const stars = [];
  for (let gx = 0; gx * cell < L.worldW; gx++) {
    for (let gy = 0; gy * cell < 840; gy++) {
      const h = hash2(gx * 7 + 1, gy * 13 + 5);
      if (h > 0.5) continue;
      const x = (gx + hash2(gx, gy * 3 + 1)) * cell;
      const y = (gy + hash2(gx * 5 + 2, gy)) * cell;
      const dm = Math.hypot(x - L.moon.x, y - L.moon.y);
      if (dm < L.moon.r * 2.6) continue;
      const m = hash2(gx + 99, gy + 7);
      const size = 0.55 + 1.5 * Math.pow(m, 3) + (m > 0.965 ? 1.4 : 0);
      const k = 1 + Math.floor(hash2(gx + 3, gy + 11) * 9); // twinkle harmonics: periods 15/k s
      const tint = hash2(gx + 17, gy + 29);
      const col = tint < 0.2 ? [200, 215, 255] : tint > 0.88 ? [255, 236, 205] : [246, 246, 255];
      const fade = 1 - smoothstep(520, 830, y);
      if (fade <= 0.02) continue;
      stars.push({ x, y, size, k, ph: TAU * hash2(gx + 41, gy + 3), base: (0.45 + 0.55 * hash2(gx + 5, gy + 61)) * fade, col, glint: m > 0.965 });
    }
  }
  starCache.set(key, stars);
  return stars;
}

export function drawStars(ctx, L, view, t, sprites) {
  const stars = starsFor(L);
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  for (const s of stars) {
    if (s.x < view.x0 - 20 || s.x > view.x1 + 20) continue;
    const tw = 0.62 + 0.38 * Math.sin((TAU * s.k * t) / LOOP + s.ph);
    const a = clamp(s.base * tw);
    const r = s.size * (0.85 + 0.15 * tw);
    ctx.globalAlpha = a * 0.55;
    ctx.drawImage(sprites.glow(s.col), s.x - r * 4, s.y - r * 4, r * 8, r * 8);
    ctx.globalAlpha = a;
    ctx.fillStyle = `rgb(${s.col[0]},${s.col[1]},${s.col[2]})`;
    ctx.beginPath(); ctx.arc(s.x, s.y, r * 0.55, 0, TAU); ctx.fill();
    if (s.glint) {
      const gl = r * (5 + 3 * tw);
      ctx.globalAlpha = a * 0.55;
      ctx.strokeStyle = ctx.fillStyle;
      ctx.lineWidth = 0.6;
      ctx.beginPath(); ctx.moveTo(s.x - gl, s.y); ctx.lineTo(s.x + gl, s.y); ctx.moveTo(s.x, s.y - gl); ctx.lineTo(s.x, s.y + gl); ctx.stroke();
    }
  }
  ctx.restore();
}

// ---------------------------------------------------------------- static landscape layer
function drawRidge(ctx, view, fn, top, bottom, cTop, cBot, rim) {
  const g = ctx.createLinearGradient(0, top, 0, bottom);
  g.addColorStop(0, cTop); g.addColorStop(1, cBot);
  ctx.fillStyle = g;
  ctx.beginPath();
  const step = 6;
  ctx.moveTo(view.x0 - 10, 1010);
  for (let x = view.x0 - 10; x <= view.x1 + 10; x += step) ctx.lineTo(x, fn(x));
  ctx.lineTo(view.x1 + 10, 1010);
  ctx.closePath();
  ctx.fill();
  if (rim) {
    ctx.strokeStyle = rim; ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let x = view.x0 - 10; x <= view.x1 + 10; x += step) { if (x === view.x0 - 10) ctx.moveTo(x, fn(x)); else ctx.lineTo(x, fn(x)); }
    ctx.stroke();
  }
}

function grassTexture(ctx, rnd, x0, x1, yFn, yMax, density, colors, lenRange) {
  for (let x = x0; x < x1; x += 1 / density) {
    const top = yFn(x);
    if (!isFinite(top)) continue;
    const y = top + 6 + Math.pow(rnd(), 1.6) * (yMax - top);
    if (y > 1005) continue;
    const len = lerp(lenRange[0], lenRange[1], rnd()) * (0.6 + 0.4 * (1 - (y - top) / (yMax - top + 1)));
    const lean = (rnd() - 0.5) * 0.9;
    ctx.strokeStyle = colors[(rnd() * colors.length) | 0];
    ctx.lineWidth = 1.1 + rnd() * 1.3;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.quadraticCurveTo(x + lean * len * 0.4, y - len * 0.6, x + lean * len, y - len);
    ctx.stroke();
  }
}

function drawStageHill(ctx, h, L, view, seed) {
  const rnd = mulberry32(seed);
  ctx.save();
  hillPath(ctx, h);
  const g = ctx.createLinearGradient(0, h.top, 0, 1000);
  g.addColorStop(0, h.role === 'center' ? '#3f8f48' : '#37803f');
  g.addColorStop(0.35, '#2b6c36');
  g.addColorStop(1, '#123a21');
  ctx.fillStyle = g;
  ctx.fill();
  ctx.clip();
  // moonlit left flank / shaded right flank (rounded volume)
  const sg = ctx.createLinearGradient(h.x - h.halfW, 0, h.x + h.halfW, 0);
  sg.addColorStop(0, 'rgba(150,210,255,0.10)');
  sg.addColorStop(0.42, 'rgba(150,210,255,0.04)');
  sg.addColorStop(0.62, 'rgba(0,10,30,0)');
  sg.addColorStop(1, 'rgba(0,12,35,0.38)');
  ctx.fillStyle = sg;
  ctx.fillRect(h.x - h.halfW, h.top - 10, 2 * h.halfW, 1100 - h.top);
  // soft dome highlight
  ctx.save();
  ctx.translate(h.x - h.halfW * 0.18, h.top + 60);
  ctx.scale(1, 0.32);
  const rg = ctx.createRadialGradient(0, 0, 0, 0, 0, h.halfW * 0.75);
  rg.addColorStop(0, 'rgba(140,215,120,0.22)');
  rg.addColorStop(1, 'rgba(140,215,120,0)');
  ctx.fillStyle = rg;
  ctx.fillRect(-h.halfW, -h.halfW, 2 * h.halfW, 2 * h.halfW);
  ctx.restore();
  // grass texture
  const xs0 = Math.max(view.x0 - 20, h.x - h.halfW), xs1 = Math.min(view.x1 + 20, h.x + h.halfW);
  grassTexture(ctx, rnd, xs0, xs1, (x) => hillY(h, x), 1000, 0.55, ['rgba(20,60,30,0.55)', 'rgba(25,75,38,0.5)', 'rgba(70,140,70,0.35)', 'rgba(95,165,85,0.28)'], [8, 20]);
  // little meadow flowers
  for (let i = 0; i < (xs1 - xs0) * 0.02; i++) {
    const x = xs0 + rnd() * (xs1 - xs0);
    const top = hillY(h, x);
    const y = top + 18 + Math.pow(rnd(), 1.3) * (1000 - top);
    const c = ['#f6f1ff', '#ffd76a', '#ff9bc8', '#b9a6ff'][(rnd() * 4) | 0];
    ctx.fillStyle = c; ctx.globalAlpha = 0.55 + rnd() * 0.3;
    const r = 1.4 + rnd() * 1.6;
    for (let k = 0; k < 5; k++) { const a = (k / 5) * TAU; ctx.beginPath(); ctx.arc(x + Math.cos(a) * r, y + Math.sin(a) * r * 0.7, r * 0.75, 0, TAU); ctx.fill(); }
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#ffe9a0'; ctx.beginPath(); ctx.arc(x, y, r * 0.5, 0, TAU); ctx.fill();
  }
  ctx.restore();
  // moonlit rim along the crest
  ctx.save();
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'rgba(170,230,170,0.55)';
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  const n = 120;
  for (let i = 0; i <= n; i++) {
    const u = -0.97 + (1.94 * i) / n;
    const x = h.x + u * h.halfW;
    const y = hillY(h, x) + 1.2;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

/** Static landscape (everything below the sky that does not animate). Rendered once per view. */
export function drawLandscape(ctx, L, view) {
  drawRidge(ctx, view, farY, 760, 900, '#22335a', '#141f3e', 'rgba(150,170,230,0.25)');
  drawRidge(ctx, view, midY, 810, 960, '#1d4040', '#0f2a26', 'rgba(130,200,170,0.22)');
  // meadow plain between the hills
  drawRidge(ctx, view, (x) => 905 + 10 * terrainNoise(x + 777, 4), 890, 1000, '#24603a', '#113421', 'rgba(120,200,120,0.25)');
  const rnd = mulberry32(4242);
  grassTexture(ctx, rnd, view.x0 - 10, view.x1 + 10, (x) => 905 + 10 * terrainNoise(x + 777, 4), 1000, 0.25, ['rgba(20,55,30,0.6)', 'rgba(70,140,70,0.3)'], [6, 14]);
  hillOrder(L).forEach((h, i) => {
    if (h.x + h.halfW < view.x0 - 50 || h.x - h.halfW > view.x1 + 50) return;
    // soft contact shadow where this hill meets the meadow / the hill behind it
    ctx.save();
    ctx.translate(h.x, 990);
    ctx.scale(1, 0.12);
    const rg = ctx.createRadialGradient(0, 0, h.halfW * 0.6, 0, 0, h.halfW * 1.15);
    rg.addColorStop(0, 'rgba(0,10,20,0.35)');
    rg.addColorStop(1, 'rgba(0,10,20,0)');
    ctx.fillStyle = rg;
    ctx.fillRect(-h.halfW * 1.2, -h.halfW * 1.2, h.halfW * 2.4, h.halfW * 2.4);
    ctx.restore();
    drawStageHill(ctx, h, L, view, 1000 + i * 77);
  });
}

// ---------------------------------------------------------------- animated grass on crests
const bladeCache = new Map();
function bladesFor(L) {
  const key = `${L.mode}:${L.worldW.toFixed(1)}`;
  if (bladeCache.has(key)) return bladeCache.get(key);
  const blades = [];
  const order = hillOrder(L);
  order.forEach((h, hi) => {
    const rnd = mulberry32(900 + hi * 31);
    for (let x = h.x - h.halfW * 0.985; x < h.x + h.halfW * 0.985; x += 2.6 + rnd() * 2.2) {
      const y = hillY(h, x);
      if (y > 1000) continue;
      // hidden behind a hill drawn later (in front)?
      const front = order.slice(hi + 1).some((o) => hillY(o, x) < y - 1.5);
      if (front) continue;
      blades.push({ x, y: y + 2.5, n: hillSlope(h, x), len: 7 + rnd() * 15, w: 1.3 + rnd() * 1.4, ph: rnd() * TAU, light: rnd() < 0.35 });
    }
  });
  bladeCache.set(key, blades);
  return blades;
}

export function drawCrestGrass(ctx, L, view, t) {
  const blades = bladesFor(L);
  for (const pass of [false, true]) {
    ctx.fillStyle = pass ? 'rgba(120,200,110,0.9)' : '#2d7038';
    ctx.beginPath();
    for (const b of blades) {
      if (b.light !== pass || b.x < view.x0 - 30 || b.x > view.x1 + 30) continue;
      const sway = 0.2 * Math.sin((TAU * t) / 3 + b.x * 0.011 + b.ph * 0.15) + 0.07 * Math.sin((TAU * t) / 1.5 + b.x * 0.027 + b.ph);
      const ang = b.n + sway;
      const tx = b.x + Math.sin(ang) * b.len, ty = b.y - Math.cos(ang) * b.len;
      const nx = Math.cos(b.n) * b.w, ny = Math.sin(b.n) * b.w;
      ctx.moveTo(b.x - nx, b.y - ny);
      ctx.quadraticCurveTo(b.x + Math.sin(ang) * b.len * 0.5 - nx * 0.4, b.y - Math.cos(ang) * b.len * 0.55, tx, ty);
      ctx.quadraticCurveTo(b.x + Math.sin(ang) * b.len * 0.5 + nx * 0.4, b.y - Math.cos(ang) * b.len * 0.45, b.x + nx, b.y + ny);
      ctx.closePath();
    }
    ctx.fill();
  }
}

// ---------------------------------------------------------------- fireflies & foreground
const flyCache = new Map();
function firefliesFor(L) {
  const key = `${L.mode}:${L.worldW.toFixed(1)}`;
  if (flyCache.has(key)) return flyCache.get(key);
  const rnd = mulberry32(777);
  const n = Math.round(L.worldW / 95);
  const flies = [];
  for (let i = 0; i < n; i++) {
    const x = rnd() * L.worldW;
    const ground = Math.min(groundY(L, x), 930);
    flies.push({ x, y: ground - 12 - rnd() * 70, ax: 18 + rnd() * 40, ay: 8 + rnd() * 18, kx: 1 + ((rnd() * 2) | 0), ky: 2 + ((rnd() * 2) | 0), kb: 3 + ((rnd() * 6) | 0), p: [rnd() * TAU, rnd() * TAU, rnd() * TAU], s: 1.2 + rnd() * 1.2 });
  }
  flyCache.set(key, flies);
  return flies;
}

export function drawFireflies(ctx, L, view, t, sprites) {
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  for (const f of firefliesFor(L)) {
    if (f.x < view.x0 - 80 || f.x > view.x1 + 80) continue;
    const x = f.x + f.ax * Math.sin((TAU * f.kx * t) / LOOP + f.p[0]);
    const y = f.y + f.ay * Math.sin((TAU * f.ky * t) / LOOP + f.p[1]);
    const b = Math.pow(0.5 + 0.5 * Math.sin((TAU * f.kb * t) / LOOP + f.p[2]), 2);
    if (b < 0.02) continue;
    ctx.globalAlpha = b * 0.8;
    ctx.drawImage(sprites.glow([200, 255, 120]), x - f.s * 7, y - f.s * 7, f.s * 14, f.s * 14);
    ctx.globalAlpha = b;
    ctx.fillStyle = '#f4ffc8';
    ctx.beginPath(); ctx.arc(x, y, f.s * 0.7, 0, TAU); ctx.fill();
  }
  ctx.restore();
}

export function drawForeground(ctx, L, view, t) {
  // dark, close grass tufts along the bottom edge for depth
  ctx.save();
  const rnd = mulberry32(31337);
  const x0 = Math.floor(view.x0 / 9) * 9;
  ctx.fillStyle = '#0c2414';
  ctx.beginPath();
  for (let x = 0; x < L.worldW + 9; x += 9) {
    const r1 = rnd(), r2 = rnd(), r3 = rnd();
    if (x < x0 - 40 || x > view.x1 + 40) continue;
    const base = 1004;
    const len = 18 + r1 * 34;
    const sway = 0.16 * Math.sin((TAU * t) / 3 + x * 0.006 + r2 * 2) + 0.05 * Math.sin((TAU * t) / 1 + x * 0.02);
    const ang = (r3 - 0.5) * 0.7 + sway;
    const w = 3 + r2 * 3;
    ctx.moveTo(x - w, base);
    ctx.quadraticCurveTo(x + Math.sin(ang) * len * 0.5, base - len * 0.6, x + Math.sin(ang) * len, base - Math.cos(ang) * len);
    ctx.quadraticCurveTo(x + Math.sin(ang) * len * 0.5 + w * 0.3, base - len * 0.4, x + w, base);
    ctx.closePath();
  }
  ctx.fill();
  ctx.restore();
}
