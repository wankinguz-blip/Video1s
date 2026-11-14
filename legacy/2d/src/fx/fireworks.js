// Fireworks → particle text "VANKING". Every particle has a closed-form state as a function of t,
// so any frame can be rendered independently and the loop is exact.
//
// Per cycle (see config.FIREWORK_CYCLES): 3 rockets (blue/red/green) rise from behind the hills,
// burst into spherical showers, a share of each shower flies into the letters while turning white,
// the word holds ~1 s with a shimmer, then crumbles: particles detach, fall, shrink and fade out.
import { FIREWORK_CYCLES, COLORS, CYCLE_COLORS } from '../config.js';
import { TEXT_POINTS, TEXT_WIDTH } from '../assets/vanking-text.js';
import { TAU, clamp, mulberry32, smoothstep, smootherstep, lerp } from '../math.js';

const KD = 2.7;        // air drag (1/s)
const GRAV = 95;       // gravity on burst sparks (units/s²)
const G_FALL = 240;    // gravity on crumbling letters
const RADIUS = 205;    // burst radius (units) → initial speed RADIUS*KD
const EXTRA = 320;     // decorative sparks per burst (not used for letters)
const WHITE = [255, 255, 255];

function drag(tau) { return (1 - Math.exp(-KD * tau)) / KD; }

function buildCycle(L, cyc, ci) {
  const rnd = mulberry32(7000 + ci * 131);
  const cols = CYCLE_COLORS[ci].map((n) => COLORS[n]);
  const bursts = L.bursts.map((b, k) => ({
    x: b.x + (rnd() - 0.5) * 70, y: b.y + (rnd() - 0.5) * 36,
    sx: b.launchX + (rnd() - 0.5) * 90, sy: 1000,
    color: cols[k], curve: (rnd() - 0.5) * 2,
  }));
  const capH = L.text.h, tcx = L.text.cx, tcy = L.text.cy, tw = TEXT_WIDTH * capH;
  const groups = [[], [], []];
  for (const [x, y, li, e] of TEXT_POINTS) {
    const wx = tcx + x * capH, wy = tcy + y * capH;
    const g = clamp(Math.floor(((wx - (tcx - tw / 2)) / tw) * 3), 0, 2);
    groups[g].push({ x: wx, y: wy, li, e });
  }
  const gatherLen = cyc.gather[1] - cyc.gather[0];
  const dissLen = cyc.dissolve[1] - cyc.dissolve[0];
  const P = [];
  bursts.forEach((B, k) => {
    const targets = groups[k];
    const n = targets.length + EXTRA;
    const dirs = [];
    for (let i = 0; i < n; i++) {
      const z = 1 - (2 * (i + 0.5)) / n, r = Math.sqrt(1 - z * z), th = i * 2.399963 + rnd() * 0.3;
      dirs.push({ x: r * Math.cos(th), y: r * Math.sin(th), sp: 0.62 + 0.38 * Math.sqrt(rnd()) });
    }
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [dirs[i], dirs[j]] = [dirs[j], dirs[i]]; }
    const textDirs = dirs.slice(0, targets.length).sort((a, b) => Math.atan2(a.y, a.x) - Math.atan2(b.y, b.x));
    const tSorted = targets.map((tp) => ({ ...tp, a: Math.atan2(tp.y - B.y, tp.x - B.x) })).sort((a, b) => a.a - b.a);
    // rotate the matching so the angular seam lines up best
    tSorted.forEach((tp, i) => {
      const d = textDirs[i];
      const dist = Math.hypot(tp.x - B.x, tp.y - B.y);
      P.push({
        kind: 1, b: k, vx: d.x * d.sp * RADIUS * KD, vy: d.y * d.sp * RADIUS * KD,
        tx: tp.x, ty: tp.y, edge: tp.e, li: tp.li,
        g0: cyc.gather[0] + (1 - clamp(dist / 600)) * gatherLen * 0.22 + rnd() * gatherLen * 0.1,
        swirl: (rnd() - 0.5) * 120,
        d0: cyc.dissolve[0] + (tp.li / 6) * dissLen * 0.22 + rnd() * dissLen * 0.16,
        dLife: dissLen * (0.5 + rnd() * 0.12), dvx: (rnd() - 0.5) * 50, dvy: -20 + rnd() * 30,
        size: tp.e ? 1.05 : 0.9 + rnd() * 0.25, ph: rnd() * TAU, fq: 7 + rnd() * 9,
      });
    });
    for (const d of dirs.slice(targets.length)) {
      const crackle = rnd() < 0.3;
      const life = crackle ? 0.55 + rnd() * 0.5 : 0.9 + rnd() * 0.7;
      P.push({ kind: 0, b: k, vx: d.x * d.sp * RADIUS * KD * 1.05, vy: d.y * d.sp * RADIUS * KD * 1.05, life: life * Math.min(1, 1 / (cyc.speed || 1) + 0.15), crackle, size: 0.7 + rnd() * 0.5, ph: rnd() * TAU, fq: 10 + rnd() * 14 });
    }
  });
  return { ...cyc, bursts, parts: P, end: cyc.dissolve[1] };
}

export function createFireworks(L, platform) {
  const cycles = FIREWORK_CYCLES.map((c, i) => buildCycle(L, c, i));
  let glowCanvas = null, glowCtx = null, blurCanvas = null, blurCtx = null;
  const faceY = L.faceSafeY;

  const active = (t) => cycles.filter((c) => t >= c.launch - 0.01 && t <= c.end + 0.01);

  function particleState(c, p, t, out) {
    const tau = t - c.burst;
    const B = c.bursts[p.b];
    if (tau < 0) return false;
    const f = drag(tau);
    let x = B.x + p.vx * f, y = B.y + p.vy * f + (GRAV / KD) * (tau - f);
    let col = B.color, a, size = p.size;
    if (p.kind === 0) {
      if (tau > p.life) return false;
      a = (0.35 + 0.65 * Math.exp(-1.3 * tau)) * (1 - smoothstep(p.life - 0.35, p.life, tau));
      if (tau > 0.45) a *= 0.65 + 0.35 * Math.sin(p.fq * tau + p.ph);
      if (p.crackle && tau > p.life - 0.12) { col = WHITE; a = 1.2 * (1 - smoothstep(p.life - 0.04, p.life, tau)); size *= 1.25; }
      a *= 1 - smoothstep(faceY - 50, faceY + 10, y);
      out.trail = tau < 0.75;
    } else {
      out.trail = tau < 0.5 && t < p.g0;
      // gather easing: zero velocity at both ends, most of the travel early (settles gently)
      const gu = clamp((t - p.g0) / (c.gather[1] - p.g0));
      const e = 1 - Math.pow(1 - gu, 4) * (1 + 4 * gu);
      if (e > 0) {
        const sw = Math.sin(Math.PI * e) * p.swirl;
        const dx = p.tx - x, dy = p.ty - y, dl = Math.hypot(dx, dy) || 1;
        x = lerp(x, p.tx, e) - (dy / dl) * sw;
        y = lerp(y, p.ty, e) + (dx / dl) * sw;
      }
      const whiten = smoothstep(c.gather[0], c.gather[1] - 0.06, t);
      col = [lerp(col[0], 255, whiten), lerp(col[1], 255, whiten), lerp(col[2], 255, whiten)];
      a = 0.55 + 0.45 * Math.exp(-1.3 * tau);
      a = lerp(a, 1, e);
      if (t >= c.hold[0]) {
        // shimmer, then loosen right before crumbling
        const loosen = smoothstep(c.dissolve[0] - 0.12, p.d0, t);
        const amp = 0.8 + 3.5 * loosen;
        x = p.tx + amp * Math.sin(p.fq * t + p.ph);
        y = p.ty + amp * Math.cos(p.fq * 0.83 * t + p.ph * 1.3);
        a = 0.88 + 0.12 * Math.sin(p.fq * 1.7 * t + p.ph);
      }
      if (t >= p.d0) {
        const age = t - p.d0, u = age / p.dLife;
        if (u >= 1) return false;
        x = p.tx + p.dvx * age;
        y = p.ty + p.dvy * age + 0.5 * G_FALL * age * age;
        size *= Math.pow(1 - u, 0.8);
        a = Math.pow(1 - u, 1.3) * (0.85 + 0.15 * Math.sin(p.fq * 2 * t + p.ph));
        const warm = smoothstep(0.2, 1, u);
        col = [255, lerp(255, 214, warm), lerp(255, 160, warm)];
      }
      if (p.edge) a *= 1.08;
    }
    out.x = x; out.y = y; out.a = clamp(a, 0, 1.3); out.size = size; out.col = col; out.tau = tau;
    return true;
  }

  function rocketPos(c, B, t) {
    const u = clamp((t - c.launch) / (c.burst - c.launch));
    const e = 1 - Math.pow(1 - u, 2.1);
    return { x: lerp(B.sx, B.x, e) + B.curve * 34 * Math.sin(Math.PI * e), y: lerp(B.sy, B.y, e), u };
  }

  function lights(t) {
    const out = [];
    for (const c of active(t)) {
      const tau = t - c.burst;
      for (const B of c.bursts) {
        if (tau < 0) {
          const r = rocketPos(c, B, t);
          out.push({ x: r.x, y: r.y, c: B.color, i: 0.05 + 0.05 * r.u, r: 900 });
        } else {
          out.push({ x: B.x, y: B.y, c: B.color, i: (0.62 * Math.exp(-tau / 0.12) + 0.3 * Math.exp(-tau / 0.6)) * smoothstep(0, 0.04, tau), r: 2600 });
        }
      }
      const txt = smoothstep(c.gather[0] + 0.2, c.hold[0], t) * (1 - smoothstep(c.dissolve[0], c.dissolve[0] + 0.5, t));
      if (txt > 0) out.push({ x: L.text.cx, y: L.text.cy, c: [235, 240, 255], i: 0.22 * txt, r: 2600 });
    }
    return out;
  }

  function drawSky(ctx, t, env) {
    const { k, view, pxW, pxH, sprites } = env;
    const cs = active(t);
    if (!cs.length) return;
    const gs = 1 / 5; // bloom buffer scale
    const gw = Math.ceil(pxW * gs), gh = Math.ceil(pxH * gs);
    if (!glowCanvas || glowCanvas.width !== gw) {
      glowCanvas = platform.createCanvas(gw, gh); glowCtx = glowCanvas.getContext('2d');
      blurCanvas = platform.createCanvas(gw, gh); blurCtx = blurCanvas.getContext('2d');
    }
    glowCtx.setTransform(1, 0, 0, 1, 0, 0);
    glowCtx.clearRect(0, 0, gw, gh);
    glowCtx.setTransform(k * gs, 0, 0, k * gs, -view.x0 * k * gs, 0);
    glowCtx.globalCompositeOperation = 'lighter';

    // sky illumination
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    for (const Lt of lights(t)) {
      if (Lt.i < 0.01) continue;
      const r = Math.min(Lt.r * 0.55, 1500);
      const g = ctx.createRadialGradient(Lt.x, Lt.y, 0, Lt.x, Lt.y, r);
      g.addColorStop(0, `rgba(${Lt.c[0]},${Lt.c[1]},${Lt.c[2]},${0.55 * Lt.i})`);
      g.addColorStop(0.35, `rgba(${Lt.c[0]},${Lt.c[1]},${Lt.c[2]},${0.16 * Lt.i})`);
      g.addColorStop(1, `rgba(${Lt.c[0]},${Lt.c[1]},${Lt.c[2]},0)`);
      ctx.fillStyle = g;
      ctx.fillRect(Lt.x - r, Lt.y - r, 2 * r, 2 * r);
    }
    const st = {};
    for (const c of cs) {
      const tau = t - c.burst;
      // ---- rockets + sparkling trails
      if (tau < 0.02) {
        for (const B of c.bursts) {
          const head = rocketPos(c, B, t);
          const dt = 1 / 110;
          for (let j = 0; ; j++) {
            const te = c.launch + j * dt;
            if (te > t) break;
            const age = t - te;
            if (age > 0.5) continue;
            const r = rocketPos(c, B, te);
            const h = Math.sin(j * 12.9898) * 43758.5453;
            const rr = h - Math.floor(h);
            const x = r.x + (rr - 0.5) * 18 * age, y = r.y + 70 * age * age + 14 * age;
            const a = Math.pow(1 - age / 0.5, 1.6) * (0.55 + 0.45 * Math.sin(j * 1.7 + t * 40));
            const col = rr > 0.7 ? WHITE : [lerp(B.color[0], 255, 0.55), lerp(B.color[1], 255, 0.55), lerp(B.color[2], 255, 0.55)];
            ctx.globalAlpha = clamp(a);
            const s = 2.2 * (1 - age);
            ctx.drawImage(sprites.spark(col), x - s * 2, y - s * 2, s * 4, s * 4);
          }
          if (tau < 0) {
            ctx.globalAlpha = 1;
            ctx.drawImage(sprites.glow(B.color), head.x - 26, head.y - 26, 52, 52);
            ctx.drawImage(sprites.spark(WHITE), head.x - 7, head.y - 7, 14, 14);
            glowCtx.globalAlpha = 1;
            glowCtx.drawImage(sprites.glow(B.color), head.x - 45, head.y - 45, 90, 90);
          }
        }
      }
      if (tau < 0) continue;
      // ---- burst flash
      for (const B of c.bursts) {
        if (tau < 0.32) {
          const fr = 26 + 330 * tau, fa = Math.pow(1 - tau / 0.32, 2.2);
          const g = ctx.createRadialGradient(B.x, B.y, 0, B.x, B.y, fr);
          g.addColorStop(0, `rgba(255,255,255,${0.9 * fa})`);
          g.addColorStop(0.12, `rgba(255,255,255,${0.45 * fa})`);
          g.addColorStop(0.4, `rgba(${B.color[0]},${B.color[1]},${B.color[2]},${0.22 * fa})`);
          g.addColorStop(1, `rgba(${B.color[0]},${B.color[1]},${B.color[2]},0)`);
          ctx.globalAlpha = 1;
          ctx.fillStyle = g;
          ctx.fillRect(B.x - fr, B.y - fr, 2 * fr, 2 * fr);
        }
      }
      // ---- trails (batched per burst)
      if (tau < 1.0) {
        ctx.lineCap = 'round';
        for (let k2 = 0; k2 < c.bursts.length; k2++) {
          const B = c.bursts[k2];
          ctx.beginPath();
          const f1 = drag(tau), f0 = drag(Math.max(0, tau - 0.11));
          for (const p of c.parts) {
            if (p.b !== k2) continue;
            if (p.kind === 1 && t >= p.g0) continue;
            const x1 = B.x + p.vx * f1, y1 = B.y + p.vy * f1 + (GRAV / KD) * (tau - f1);
            if (y1 > faceY) continue;
            ctx.moveTo(B.x + p.vx * f0, B.y + p.vy * f0 + (GRAV / KD) * (Math.max(0, tau - 0.11) - f0));
            ctx.lineTo(x1, y1);
          }
          ctx.globalAlpha = 0.55 * Math.pow(1 - tau / 1.0, 1.6);
          ctx.strokeStyle = `rgb(${B.color[0]},${B.color[1]},${B.color[2]})`;
          ctx.lineWidth = 1.7;
          ctx.stroke();
        }
      }
      // ---- sparks & text particles
      for (const p of c.parts) {
        if (!particleState(c, p, t, st) || st.a <= 0.01) continue;
        if (st.x < view.x0 - 30 || st.x > view.x1 + 30) continue;
        const s = st.size;
        if (p.kind === 1 && t >= c.gather[0]) {
          // letter particles: small twinkling cores so the word visibly consists of sparks
          const tw = 0.72 + 0.28 * Math.sin(p.fq * 1.3 * t + p.ph * 2.1);
          ctx.globalAlpha = clamp(st.a * 0.24);
          ctx.drawImage(sprites.glow(st.col), st.x - 6.5 * s, st.y - 6.5 * s, 13 * s, 13 * s);
          ctx.globalAlpha = clamp(st.a * tw);
          ctx.drawImage(sprites.spark(st.col), st.x - 3 * s, st.y - 3 * s, 6 * s, 6 * s);
          glowCtx.globalAlpha = clamp(st.a * 0.1);
          glowCtx.drawImage(sprites.glow(st.col), st.x - 16 * s, st.y - 16 * s, 32 * s, 32 * s);
          continue;
        } else {
          ctx.globalAlpha = clamp(st.a * 0.42);
          ctx.drawImage(sprites.glow(st.col), st.x - 9 * s, st.y - 9 * s, 18 * s, 18 * s);
          ctx.globalAlpha = clamp(st.a);
          ctx.drawImage(sprites.spark(st.col), st.x - 3.2 * s, st.y - 3.2 * s, 6.4 * s, 6.4 * s);
        }
        glowCtx.globalAlpha = clamp(st.a * 0.5);
        glowCtx.drawImage(sprites.glow(st.col), st.x - 16 * s, st.y - 16 * s, 32 * s, 32 * s);
      }
    }
    ctx.restore();
    // ---- bloom: blur the low-res emissive buffer and add it on top of the sky
    blurCtx.setTransform(1, 0, 0, 1, 0, 0);
    blurCtx.globalCompositeOperation = 'source-over';
    blurCtx.globalAlpha = 1;
    blurCtx.clearRect(0, 0, gw, gh);
    blurCtx.filter = `blur(${Math.max(1.5, 7 * k * gs)}px)`;
    blurCtx.drawImage(glowCanvas, 0, 0);
    blurCtx.filter = 'none';
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'lighter';
    // map the bloom buffer exactly onto the frame (gw/gh were rounded up)
    const sw = pxW * gs, sh = pxH * gs;
    ctx.globalAlpha = 0.9;
    ctx.drawImage(blurCanvas, 0, 0, sw, sh, 0, 0, pxW, pxH);
    ctx.globalAlpha = 0.45;
    ctx.drawImage(glowCanvas, 0, 0, sw, sh, 0, 0, pxW, pxH);
    ctx.restore();
  }

  return { cycles, lights: (t) => lights(t), drawSky, particleState };
}
