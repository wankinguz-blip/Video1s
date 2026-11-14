// Character 3 — the emo Tecktonik dancer (left hill). Black/pink 2000s look, fringe, wristbands, studs.
// Choreography = blend of Tecktonik move cycles scheduled over the 30-beat loop. The move phase is
// time-warped (faster during firework accents) but the warp adds exactly 2 beats per loop, so every
// move (period 1 or 2 beats) closes perfectly at the loop boundary.
import { BEAT, LOOP_BEATS } from '../config.js';
import { TAU, clamp, bump, mod, smoothstep, smootherstep } from '../math.js';
import { RIG, shoulder, ikSoft, drawLimb, drawPaw, drawShoe, legChain, drawHeadAt, torsoPoint, lerpPt } from './rig.js';
import { emoShirt, emoChain, emoHair } from './outfit.js';
import { BURST_BEATS } from '../music/score.js';

const beats = (t) => mod(t / BEAT, LOOP_BEATS);
const bb = (b, c, w) => bump(b, c, w, LOOP_BEATS);
const G = (x) => (x <= -1 ? 0 : x >= 1 ? 1 : smootherstep(-1, 1, x));
/** Warped beat phase: +0.5 beat around each firework burst (4 × 0.5 = 2 beats per loop). */
export function phase(b) {
  let p = b;
  for (const c of BURST_BEATS) p += 0.5 * G((b - c) / 0.75);
  return p;
}

// ---------------------------------------------------------------- moves (hand targets per side s = ±1)
const MOVES = {
  circles(p, s, sh) {
    const th = Math.PI * p + (s > 0 ? 0 : Math.PI);
    return { x: sh.x + s * 92 + s * 150 * Math.cos(th), y: sh.y - 140 + 150 * Math.sin(th), bend: s, curl: 0.15, wrist: 0.3 * Math.sin(th) };
  },
  rolls(p, s, sh, chest) {
    const th = TAU * p + (s > 0 ? 0 : Math.PI);
    return { x: chest.x + s * 34 + 74 * Math.cos(th), y: chest.y + 54 * Math.sin(th), bend: -s, curl: 0.85, wrist: 0 };
  },
  sweep(p, s, sh) {
    const u = 0.5 - 0.5 * Math.cos(Math.PI * p + (s > 0 ? 0 : Math.PI));
    // sweep across the chest, dipping below the chin so the arm stays visible
    const ax = sh.x + s * 235, ay = sh.y + 50, bx = sh.x - s * 205, by = sh.y + 125;
    return { x: ax + (bx - ax) * u, y: ay + (by - ay) * u + 45 * Math.sin(Math.PI * u), bend: s * Math.cos(Math.PI * u), curl: 0.1, wrist: 0.6 * (u - 0.5) };
  },
  pops(p, s, sh) {
    const f = p - Math.floor(p);
    let q = smootherstep(0, 0.3, f) - smootherstep(0.5, 0.8, f);
    if (s < 0) q = 1 - q;
    const x1 = sh.x + s * 200, y1 = sh.y - 150, x2 = sh.x + s * 120, y2 = sh.y + 95;
    return { x: x2 + (x1 - x2) * q, y: y2 + (y1 - y2) * q, bend: s * (2 * q - 1), curl: 0.6 * (1 - q), wrist: -0.6 * q };
  },
  wave(p, s, sh) {
    const ph = Math.PI * p - (s > 0 ? 1.5 : 0);
    return { x: sh.x + s * 300, y: sh.y - 10 + 80 * Math.sin(ph), bend: s, curl: 0, wrist: 0.7 * Math.cos(ph) };
  },
};
const SCHEDULE = [
  [-2, 4, 'circles'], [4, 6, 'sweep'], [6, 8, 'pops'], [8, 12, 'rolls'], [12, 14, 'wave'],
  [14, 20, 'circles'], [20, 22, 'sweep'], [22, 24, 'pops'], [24, 26, 'rolls'], [26, 28, 'wave'], [28, 34, 'circles'],
];
function weights(b) {
  const h = 0.3;
  const w = SCHEDULE.map(([a, e]) => {
    let v = 0;
    for (const k of [-30, 0, 30]) { const x = b + k; v = Math.max(v, smoothstep(a - h, a + h, x) * (1 - smoothstep(e - h, e + h, x))); }
    return v;
  });
  const sum = w.reduce((p, q) => p + q, 0) || 1;
  return w.map((v) => v / sum);
}

export function emoPose(t) {
  const b = beats(t);
  const p = phase(b);
  const sp = Math.sin(Math.PI * p);
  const twist = bb(b, 7, 0.7) - bb(b, 15, 0.7) + bb(b, 23, 0.7) - bb(b, 29.2, 0.6);
  const pose = {
    hip: { x: 18 * Math.sin(Math.PI * p - 0.5), y: RIG.hipY + 13 * (0.5 + 0.5 * Math.cos(TAU * p)) },
    torso: 0.06 * Math.sin(Math.PI * p + 0.8) + 0.12 * twist,
    twistScale: 1 - 0.1 * Math.abs(twist),
    head: 0.05 * Math.sin(Math.PI * p + 1.7) + 0.025 * Math.sin(TAU * p),
    headShift: { x: 18 * twist, y: 0 },
  };
  const lL = Math.max(0, sp) ** 2, lR = Math.max(0, -sp) ** 2;
  pose.footL = { x: -94 + 18 * sp, y: -30 * lL, toe: 0.12 * lL, heel: 0 };
  pose.footR = { x: 94 + 18 * sp, y: -30 * lR, toe: 0.12 * lR, heel: 0 };
  // arms: blend scheduled moves
  const W = weights(b);
  const chest = torsoPoint(pose, 0, -165);
  for (const s of [-1, 1]) {
    const sh = shoulder(pose, s);
    let x = 0, y = 0, bend = 0, curl = 0, wrist = 0;
    SCHEDULE.forEach(([, , m], i) => {
      if (W[i] < 1e-4) return;
      const r = MOVES[m](p, s, sh, chest);
      x += W[i] * r.x; y += W[i] * r.y; bend += W[i] * r.bend; curl += W[i] * r.curl; wrist += W[i] * r.wrist;
    });
    pose[s < 0 ? 'armL' : 'armR'] = { target: { x, y }, bend, curl, wrist };
  }
  let accent = 0;
  for (const c of BURST_BEATS) accent = Math.max(accent, bb(b, c + 0.15, 0.55));
  pose.face = {
    blink: bb(b, 5.1, 0.16) + bb(b, 13.2, 0.16) + bb(b, 21.1, 0.16) + bb(b, 27.6, 0.16),
    lid: 0.22,
    pupil: { x: 9, y: -2 },
    mouth: accent > 0.05 ? { shape: 'o', open: 0.7 * accent } : { shape: 'smile' },
    whisker: 0.06 * Math.sin(Math.PI * p + 0.3),
  };
  pose.hair = 0.13 * Math.sin(Math.PI * p - 0.9) + 0.05 * Math.sin(TAU * p - 1.2);
  pose.chain = 0.32 * Math.sin(Math.PI * p - 1.3);
  pose.beat = b;
  return pose;
}

const JEANS = { light: '#3b3846', base: '#1c1a22', dark: '#060508', line: '#020203' };
const FUR = { light: '#ad8b73', base: '#896653', dark: '#5a4131', line: '#3b2a1f' };
const TEE = { light: '#3d3746', base: '#1d1a22', dark: '#07060a', line: '#020203' };
const BAND = { light: '#3a3a40', base: '#121115', dark: '#030304', line: '#000000' };
const SNEAKER = { upper: '#1b1a22', upperLight: '#4a4756', sole: '#f2f0ee', line: '#050407', toeCap: '#f2f0ee', laces: '#ff3f9a' };

function drawLeg(ctx, pose, side) {
  const { hipJ, knee, ankle } = legChain(pose, side);
  drawLimb(ctx, [hipJ, knee, ankle], [44, 36, 31], JEANS, 7);
  const foot = side < 0 ? pose.footL : pose.footR;
  drawShoe(ctx, ankle, side, SNEAKER, foot.toe, foot.heel);
}

function drawArm(ctx, pose, s) {
  const arm = pose[s < 0 ? 'armL' : 'armR'];
  const sh = shoulder(pose, s);
  const { elbow, hand } = ikSoft(sh, arm.target, RIG.upper, RIG.fore, arm.bend);
  drawLimb(ctx, [sh, elbow, hand], [42, 36, 30], FUR, 6);
  drawLimb(ctx, [sh, lerpPt(sh, elbow, 0.6)], [53, 47], TEE, 6); // short black sleeve
  const wb = lerpPt(elbow, hand, 0.66);
  drawLimb(ctx, [wb, lerpPt(elbow, hand, 0.97)], [35, 34], BAND, 4); // wristband
  const ang = Math.atan2(hand.y - elbow.y, hand.x - elbow.x);
  ctx.fillStyle = '#d6d8e2';
  for (let k = 0; k < 3; k++) {
    const u = 0.72 + k * 0.1;
    const x = elbow.x + (hand.x - elbow.x) * u, y = elbow.y + (hand.y - elbow.y) * u;
    ctx.beginPath(); ctx.arc(x + Math.cos(ang + Math.PI / 2) * 6, y + Math.sin(ang + Math.PI / 2) * 6, 4.2, 0, TAU); ctx.fill();
  }
  drawPaw(ctx, hand.x, hand.y, ang + arm.wrist, { curl: arm.curl, side: s });
}

export function drawEmo(ctx, pose, t) {
  drawLeg(ctx, pose, -1);
  drawLeg(ctx, pose, 1);
  ctx.save();
  ctx.translate(pose.hip.x, pose.hip.y);
  ctx.rotate(pose.torso);
  ctx.scale(pose.twistScale, 1);
  emoShirt(ctx, t, 1);
  emoChain(ctx, pose.chain);
  ctx.restore();
  // arms circle "around" the head: drawn behind it
  drawArm(ctx, pose, -1);
  drawArm(ctx, pose, 1);
  drawHeadAt(ctx, pose, pose.face, { after: (c) => emoHair(c, pose.hair, t) });
}
