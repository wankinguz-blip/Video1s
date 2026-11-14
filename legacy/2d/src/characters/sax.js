// Character 2 — the jazz saxophonist (right hill). Burgundy suit, bow tie, pork-pie hat, golden sax.
import { BEAT, LOOP_BEATS } from '../config.js';
import { TAU, clamp, bump, mod, smoothstep } from '../math.js';
import { RIG, shoulder, headToChar, ikSoft, drawLimb, drawPaw, drawShoe, legChain, drawHeadAt, headAngle, neck } from './rig.js';
import { saxJacket, porkPie, saxophone, SAX_KEYS } from './outfit.js';
import { drawTeeth, HEAD_PIVOT } from './head.js';
import { saxState } from '../music/sync.js';

const beats = (t) => mod(t / BEAT, LOOP_BEATS);
const bb = (b, c, w) => bump(b, c, w, LOOP_BEATS);
const RAISES = [[3.6, 0.9], [11.1, 0.9], [17.0, 1.1], [19.6, 1.0], [22.2, 0.8], [26.0, 0.7], [29.6, 0.6]];
const DIPS = [[7.6, 0.8], [15.6, 0.8], [23.3, 0.9], [29.0, 0.6]];
const BLINKS = [1.2, 5.0, 9.3, 13.0, 21.0, 25.0, 27.6];
const MOUTH = { x: 612, y: 472 }; // mouthpiece tip in head space

export function saxPose(t) {
  const b = beats(t);
  const ss = saxState(t);
  const bounce = 0.5 + 0.5 * Math.cos(TAU * b);
  const sway = Math.sin((TAU * b) / 6);
  let notRaise = 1;
  for (const [c, w] of RAISES) notRaise *= 1 - bb(b, c, w);
  let raise = 1 - notRaise; // smooth union (no kinks)
  const spot = smoothstep(15.4, 16.4, b) * (1 - smoothstep(23.6, 24.6, b)); // 8–12 s: sax takes the lead
  raise = clamp(raise + 0.22 * spot);
  let notDip = 1;
  for (const [c, w] of DIPS) notDip *= 1 - bb(b, c, w);
  const dip = 1 - notDip;
  const pose = {
    hip: { x: 12 * sway, y: RIG.hipY + 6 * bounce + 26 * dip + 7 * ss.longNote },
    torso: 0.055 * Math.sin((TAU * b) / 6 + 0.6) - 0.11 * raise,
    twistScale: 1,
    head: 0.032 * Math.cos(TAU * b) - 0.05 * raise + 0.02 * Math.sin((TAU * b) / 6 + 2),
    headShift: { x: 0, y: 0 },
  };
  const tap = 0.26 * (0.5 - 0.5 * Math.cos(TAU * b));
  pose.footL = { x: -90 + 6 * sway, y: 0, toe: tap, heel: 0 };
  pose.footR = { x: 92 + 6 * sway, y: 0, toe: 0, heel: 0.14 * dip };
  let blink = 0;
  for (const c of BLINKS) blink = Math.max(blink, bb(b, c, 0.16));
  const feel = clamp(ss.longNote * 0.9 + 0.3 * raise);
  pose.face = {
    blink: Math.max(blink, feel * 0.88),
    lid: 0.4,
    pupil: { x: -7, y: 5 },
    mouth: { shape: 'sax', puff: ss.blow },
    whisker: 0.04 * Math.sin((TAU * b) / 2),
  };
  pose.saxRel = 0.4 - 0.27 * raise + 0.022 * Math.sin(TAU * b);
  pose.ss = ss;
  pose.raise = raise;
  return pose;
}

const SUIT = { light: '#b5405f', base: '#7a1d38', dark: '#3a0a1a', line: '#1e040d' };
const SHOE = { upper: '#f1ece2', upperLight: '#ffffff', sole: '#1a1410', line: '#120d0a', toeCap: '#16110e', heelCap: '#16110e', shine: true };

function drawLeg(ctx, pose, side) {
  const { hipJ, knee, ankle } = legChain(pose, side);
  drawLimb(ctx, [hipJ, knee, ankle], [50, 43, 37], SUIT, 7);
  const foot = side < 0 ? pose.footL : pose.footR;
  drawShoe(ctx, ankle, side, SHOE, foot.toe, foot.heel);
}

/** Sax frame in character space: {o: mouthpiece tip, a: angle}. */
export function saxFrame(pose) {
  return { o: headToChar(pose, MOUTH.x, MOUTH.y), a: headAngle(pose) + pose.saxRel };
}
const toChar = (f, x, y) => ({ x: f.o.x + x * Math.cos(f.a) - y * Math.sin(f.a), y: f.o.y + x * Math.sin(f.a) + y * Math.cos(f.a) });

export function drawSax(ctx, pose, t) {
  drawLeg(ctx, pose, -1);
  drawLeg(ctx, pose, 1);
  ctx.save();
  ctx.translate(pose.hip.x, pose.hip.y);
  ctx.rotate(pose.torso);
  saxJacket(ctx, t, 1);
  ctx.restore();
  drawHeadAt(ctx, pose, pose.face, { after: (c) => porkPie(c, -0.04 * pose.raise) });
  // saxophone attached to the mouth
  const f = saxFrame(pose);
  ctx.save();
  ctx.translate(f.o.x, f.o.y);
  ctx.rotate(f.a);
  saxophone(ctx, pose.raise, t);
  ctx.restore();
  // buck teeth resting on the mouthpiece (as real players do)
  const n = neck(pose);
  ctx.save();
  ctx.translate(n.x, n.y); ctx.rotate(headAngle(pose)); ctx.translate(-HEAD_PIVOT.x, -HEAD_PIVOT.y);
  ctx.translate(612, 470); ctx.scale(0.62, 0.62); ctx.translate(-600, -462);
  drawTeeth(ctx);
  ctx.restore();
  // hands on the key stacks
  const fing = pose.ss.fingers;
  const upKey = toChar(f, SAX_KEYS.upper.x, SAX_KEYS.upper.y);
  const loKey = toChar(f, SAX_KEYS.lower.x, SAX_KEYS.lower.y);
  const sR = shoulder(pose, 1), sL = shoulder(pose, -1);
  const upWrist = toChar(f, SAX_KEYS.upper.x + 62, SAX_KEYS.upper.y + 6);
  const loWrist = toChar(f, SAX_KEYS.lower.x - 66, SAX_KEYS.lower.y + 4);
  const armU = ikSoft(sR, upWrist, RIG.upper, RIG.fore, -1);
  const armL = ikSoft(sL, loWrist, RIG.upper, RIG.fore, 1);
  drawLimb(ctx, [sL, armL.elbow, armL.hand], [48, 41, 34], SUIT, 7);
  cuff(ctx, armL);
  drawPaw(ctx, armL.hand.x, armL.hand.y, Math.atan2(loKey.y - armL.hand.y, loKey.x - armL.hand.x), { curl: 0.55, side: 1, press: [fing[3], fing[4], fing[5], fing[5]] });
  drawLimb(ctx, [sR, armU.elbow, armU.hand], [48, 41, 34], SUIT, 7);
  cuff(ctx, armU);
  drawPaw(ctx, armU.hand.x, armU.hand.y, Math.atan2(upKey.y - armU.hand.y, upKey.x - armU.hand.x), { curl: 0.55, side: -1, press: [fing[0], fing[1], fing[2], fing[2]] });
}

function cuff(ctx, arm) {
  // white shirt cuff peeking out of the sleeve
  const d = { x: arm.hand.x - arm.elbow.x, y: arm.hand.y - arm.elbow.y }, l = Math.hypot(d.x, d.y) || 1;
  const a = { x: arm.hand.x - (d.x / l) * 16, y: arm.hand.y - (d.y / l) * 16 };
  drawLimb(ctx, [a, arm.hand], [33, 32], { light: '#ffffff', base: '#f3ead8', dark: '#cbbd9f', line: '#6a5f4a' }, 4);
}
