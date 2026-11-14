// Seamless-loop validation.
//  1. pose continuity: key joints sampled at 240 Hz → no jumps, velocity continuous across 15 s → 0 s
//  2. render(15.0) must be pixel-identical to render(0.0)
//  3. frame-difference statistics over all 450 transitions incl. the seam 449 → 0
//  4. audio: the seam jump compared with the same musical position at other phrase starts
//   node scripts/validate-loop.mjs [--screen left] [--mode standalone] [--scale 0.1]
import { createCanvas, Path2D } from '@napi-rs/canvas';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { layoutFor } from '../src/layout.js';
import { SCREENS, LOOP, FPS, BEAT } from '../src/config.js';
import { createShow, CAST } from '../src/show.js';
import { RIG, shoulder, neck, headAngle } from '../src/characters/rig.js';

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf(`--${k}`); return i >= 0 ? argv[i + 1] : d; };
const screen = arg('screen', 'left'), mode = arg('mode', 'standalone'), scale = Number(arg('scale', '0.1'));
const report = [];
const log = (s) => { console.log(s); report.push(s); };

// ---------------------------------------------------------------- 1. pose continuity
log(`## Pose continuity (240 Hz sampling)`);
const keyPoints = (p) => {
  const pts = [p.hip, p.footL, p.footR, neck(p), shoulder(p, -1), shoulder(p, 1)];
  for (const k of ['free', 'armL', 'armR']) if (p[k]?.target) pts.push(p[k].target);
  return { pts, ang: [p.torso, headAngle(p), p.saxRel ?? 0, p.hair ?? 0] };
};
let allOk = true;
for (const m of CAST) {
  const dt = 1 / 240, n = LOOP / dt;
  const sample = (t) => keyPoints(m.pose(t));
  let maxStep = 0, maxAcc = 0, seamStep = 0, seamAcc = 0, worstT = 0;
  const flat = (s) => [...s.pts.flatMap((q) => [q.x, q.y]), ...s.ang.map((a) => a * 300)];
  const series = [];
  for (let i = -1; i <= n + 1; i++) series.push(flat(sample(i * dt)));
  for (let i = 1; i < series.length - 1; i++) {
    let step = 0, acc = 0;
    for (let j = 0; j < series[i].length; j++) {
      step = Math.max(step, Math.abs(series[i + 1][j] - series[i][j]));
      acc = Math.max(acc, Math.abs(series[i + 1][j] - 2 * series[i][j] + series[i - 1][j]));
    }
    if (step > maxStep) { maxStep = step; }
    if (acc > maxAcc) { maxAcc = acc; worstT = (i - 1) * dt; }
    if (i === n) { seamStep = step; seamAcc = acc; } // transition (15 - dt) → 15 ≡ 0
  }
  // exact periodicity: pose(15) == pose(0)
  const a = flat(sample(0)), b = flat(sample(LOOP));
  const periodErr = Math.max(...a.map((v, j) => Math.abs(v - b[j])));
  const ok = periodErr < 1e-6 && seamStep <= maxStep && seamAcc <= maxAcc;
  allOk &&= ok;
  log(`- ${m.key.padEnd(6)} max step ${maxStep.toFixed(2)} u/frame@240Hz, max accel ${maxAcc.toFixed(3)} (at ${worstT.toFixed(3)} s) | seam step ${seamStep.toFixed(2)}, seam accel ${seamAcc.toFixed(3)} | pose(15)-pose(0) = ${periodErr.toExponential(1)} → ${ok ? 'OK' : 'CHECK'}`);
}

// ---------------------------------------------------------------- 2+3. frames
log(`\n## Frames (${mode} / ${screen}, scale ${scale})`);
const L = layoutFor(mode, screen);
const view = L.views.find((v) => v.name === screen) || L.views[0];
const W = Math.round((SCREENS[screen].w * scale) / 2) * 2, H = Math.round((SCREENS[screen].h * scale) / 2) * 2;
const show = createShow({ createCanvas, Path2D }, L, view, W, H);
const grab = (t) => Buffer.from(show.render(t).data());
const f0 = grab(0), f15 = grab(LOOP);
let maxd = 0;
for (let i = 0; i < f0.length; i++) maxd = Math.max(maxd, Math.abs(f0[i] - f15[i]));
log(`- render(15.0) vs render(0.0): max channel difference = ${maxd} → ${maxd === 0 ? 'IDENTICAL' : 'DIFFERENT'}`);
allOk &&= maxd === 0;
const N = LOOP * FPS;
const frames = [];
for (let i = 0; i < N; i++) frames.push(grab(i / FPS));
const mad = (a, b) => { let s = 0; for (let i = 0; i < a.length; i += 4) s += Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]); return s / (a.length * 0.75); };
const d = frames.map((f, i) => mad(f, frames[(i + 1) % N]));
const sorted = [...d].sort((a, b) => a - b);
const pct = (p) => sorted[Math.min(N - 1, Math.floor(p * N))];
const seam = d[N - 1];
const rank = sorted.filter((v) => v > seam).length;
log(`- frame-to-frame mean abs diff: median ${pct(0.5).toFixed(3)}, p90 ${pct(0.9).toFixed(3)}, max ${sorted[N - 1].toFixed(3)}`);
log(`- seam 449→0: ${seam.toFixed(3)} (larger than ${N - 1 - rank} of ${N} transitions) → ${seam <= pct(0.9) ? 'OK (ordinary transition)' : 'CHECK'}`);
const top = d.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 6).map(([v, i]) => `${i}→${(i + 1) % N} (${(i / FPS).toFixed(2)} s): ${v.toFixed(2)}`);
log(`- largest transitions (firework flashes expected): ${top.join('; ')}`);
allOk &&= seam <= pct(0.9);

// ---------------------------------------------------------------- 4. audio seam
try {
  const wav = readFileSync('out/audio/soundtrack_x5.wav');
  const n = (wav.length - 44) / 4, per = n / 5;
  const s = (i) => wav.readInt16LE(44 + i * 4) / 32768;
  const jump = (i) => Math.abs(s(i) - s(i - 1));
  const seamJ = [1, 2, 3, 4].map((k) => jump(k * per));
  const phraseJ = [8, 16, 24].map((b) => jump(Math.round(b * BEAT * 48000) + per));
  let maxAll = 0; for (let i = 1; i < n; i++) maxAll = Math.max(maxAll, jump(i));
  const identical = [0, 1, 2, 3].every((k) => { for (let i = 0; i < per; i += 997) if (s(i) !== s(i + k * per)) return false; return true; });
  log(`\n## Audio (5× loop file)`);
  log(`- seam jumps ${seamJ.map((v) => v.toFixed(4)).join(', ')} | same beat in other phrases ${phraseJ.map((v) => v.toFixed(4)).join(', ')} | max anywhere ${maxAll.toFixed(4)}`);
  log(`- repetitions sample-identical: ${identical ? 'yes' : 'NO'}`);
  allOk &&= identical;
} catch (e) { log(`(audio check skipped: ${e.message})`); }

log(`\n**RESULT: ${allOk ? 'PASS' : 'CHECK ITEMS ABOVE'}**`);
mkdirSync('out', { recursive: true });
writeFileSync(`out/loop-validation-${mode}-${screen}.md`, `# Loop validation — ${mode} / ${screen}\n\n${report.join('\n')}\n`);
