// Character 1 — the lead singer (centre hill). MJ-inspired stage look, handheld mic.
import { BEAT, LOOP_BEATS } from '../config.js';
import { TAU, clamp, bump, loopSpline, mod, smoothstep } from '../math.js';
import { RIG, shoulder, headToChar, ikSoft, drawLimb, drawPaw, drawShoe, legChain, drawHeadAt, neck, headAngle, torsoPoint } from './rig.js';
import { singerJacket, fedora, microphone } from './outfit.js';
import { vocalState } from '../music/sync.js';

const beats = (t) => mod(t / BEAT, LOOP_BEATS);
const bb = (b, c, w) => bump(b, c, w, LOOP_BEATS);

// Free (gloved) hand, relative to the right shoulder: [dx, dy, curl, point]
const FREE = loopSpline([
  [0.0, [175, -275, 1, 1]], [1.0, [185, -290, 1, 1]],
  [2.0, [310, -90, 0, 0]], [3.0, [270, 30, 0.1, 0]],
  [4.0, [-30, 100, 0.5, 0]], [4.75, [200, -170, 1, 0]],
  [6.0, [90, 220, 0.7, 0]], [7.0, [180, -60, 0.9, 0]],
  [8.0, [175, -275, 1, 1]], [9.0, [185, -290, 1, 1]],
  [10.0, [310, -90, 0, 0]], [11.0, [250, -150, 0, 0]],
  [12.0, [230, -250, 0.4, 0]], [13.0, [230, -250, 0.4, 0]],
  [14.0, [290, -20, 0.1, 0]], [15.0, [100, 210, 0.7, 0]],
  [16.0, [175, -275, 1, 1]], [17.0, [200, -250, 1, 0]], [17.5, [180, -140, 1, 0]],
  [18.0, [270, -230, 0, 0]], [19.0, [-50, -10, 0.3, 0]],
  [20.0, [-30, 100, 0.5, 0]], [20.75, [240, -130, 0.1, 0]],
  [22.0, [230, -250, 0.4, 0]], [23.0, [270, -60, 0.2, 0]],
  [24.0, [175, -275, 1, 1]], [25.0, [310, -90, 0, 0]],
  [26.0, [250, 40, 0.2, 0]], [26.5, [-30, 90, 0.5, 0]], [27.25, [220, -170, 1, 0]],
  [28.5, [90, 210, 0.7, 0]], [29.3, [150, -150, 0.9, 0.5]],
], LOOP_BEATS);

const hatTouch = (b) => bb(b, 12.6, 1.15) + bb(b, 22.5, 0.95);
const slide = (b) => bb(b, 17.2, 1.05) + 0.75 * bb(b, 5.25, 0.8);
const heelToe = (b) => smoothstep(7.6, 8.2, b) * (1 - smoothstep(11.6, 12.0, b)) + smoothstep(19.6, 20.2, b) * (1 - smoothstep(23.6, 24.0, b));
const twist = (b) => bb(b, 2.0, 0.6) - bb(b, 10.0, 0.6) + bb(b, 18.15, 0.7) - bb(b, 25.0, 0.5);
const lookUp = (b) => bb(b, 1.0, 1.1) + bb(b, 9.0, 1.1) + bb(b, 17.0, 1.1) + bb(b, 24.6, 0.8);
const BLINKS = [3.4, 7.6, 11.45, 15.6, 19.4, 23.55, 28.1];

export function singerPose(t) {
  const b = beats(t);
  const bounce = 0.5 + 0.5 * Math.cos(TAU * b);
  const sl = slide(b), ht = hatTouch(b), tw = twist(b), hT = heelToe(b);
  const pose = {
    hip: { x: 8 * Math.sin(Math.PI * b) + 26 * sl, y: RIG.hipY + 9 * bounce + 16 * sl - 4 * Math.abs(tw) },
    torso: 0.05 * Math.sin(Math.PI * b + 0.3) + 0.07 * ht - 0.05 * sl + 0.05 * tw,
    twistScale: 1 - 0.13 * Math.abs(tw),
    head: -0.05 * Math.sin(Math.PI * b + 0.9) + 0.025 * Math.cos(TAU * b) + 0.2 * ht,
    headShift: { x: 24 * tw, y: 0 },
  };
  // feet: right foot taps, slides; left foot heel-toe sections
  const tapR = 0.3 * (0.5 - 0.5 * Math.cos(TAU * b)) * (1 - clamp(sl * 1.5)) * (1 - 0.7 * ht);
  const s2 = Math.sin(TAU * b);
  pose.footL = { x: -86, y: 0, toe: 0.32 * hT * Math.max(0, -s2) ** 2, heel: 0.3 * hT * Math.max(0, s2) ** 2 };
  pose.footR = { x: 88 + 95 * sl, y: 0, toe: tapR, heel: 0 };
  // face
  const vs = vocalState(t);
  let blink = 0;
  for (const c of BLINKS) blink = Math.max(blink, bb(b, c, 0.16));
  pose.face = {
    blink,
    pupil: { x: 2 + 10 * ht, y: -13 * clamp(lookUp(b)) - 7 * ht },
    lid: 0.08,
    mouth: vs.singing > 0.02 ? { shape: vs.shape, open: vs.open } : { shape: 'smile' },
    whisker: 0.05 * Math.sin(TAU * b * 0.5 + 0.4) + 0.08 * vs.open,
  };
  pose.hatTilt = 0.12 * ht;
  pose.shrugR = 34 * ht;
  // free (gloved) hand target, blended toward the hat brim during hat touches
  const f = FREE(b);
  const sR = shoulder(pose, 1);
  let target = { x: sR.x + f[0], y: sR.y + f[1] };
  if (ht > 0.001) {
    const hatA = -0.1 + pose.hatTilt; // fedora transform in head space (see outfit.fedora)
    const bx = 617 + 196 * Math.cos(hatA) - 16 * Math.sin(hatA), by = 196 + 196 * Math.sin(hatA) + 16 * Math.cos(hatA);
    const brim = headToChar(pose, bx, by);
    const w = smoothstep(0.0, 0.85, ht);
    target = { x: target.x + (brim.x - target.x) * w, y: target.y + (brim.y - target.y) * w };
  }
  pose.free = { target, curl: clamp(f[2] * (1 - ht) + 0.35 * ht), point: clamp(f[3] * (1 - ht)) };
  pose.beat = b;
  return pose;
}

const JACKET = { light: '#3d3d4e', base: '#1c1c25', dark: '#060609', line: '#020203' };
const PANTS = { light: '#2e2e3a', base: '#15151b', dark: '#050507', line: '#020203' };
const SOCK = { light: '#ffffff', base: '#eef0f6', dark: '#b9bdcc', line: '#5d6070' };
const SHOE = { upper: '#0b0b0e', upperLight: '#3a3a46', sole: '#020203', line: '#000', shine: true };

function drawLeg(ctx, pose, side) {
  const { hipJ, knee, ankle } = legChain(pose, side);
  const hem = { x: knee.x + (ankle.x - knee.x) * 0.7, y: knee.y + (ankle.y - knee.y) * 0.7 };
  drawLimb(ctx, [hem, ankle], [27, 26], SOCK, 5); // sparkly white socks (MJ signature)
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  for (let i = 0; i < 4; i++) { ctx.beginPath(); ctx.arc(ankle.x + side * (i * 7 - 10), ankle.y - 10 - i * 6, 2.3, 0, TAU); ctx.fill(); }
  drawLimb(ctx, [hipJ, knee, hem], [50, 42, 36], PANTS, 7);
  const foot = side < 0 ? pose.footL : pose.footR;
  drawShoe(ctx, ankle, side, SHOE, foot.toe, foot.heel);
}

export function drawSinger(ctx, pose, t) {
  drawLeg(ctx, pose, -1);
  drawLeg(ctx, pose, 1);
  // torso
  ctx.save();
  ctx.translate(pose.hip.x, pose.hip.y);
  ctx.rotate(pose.torso);
  ctx.scale(pose.twistScale, 1);
  singerJacket(ctx, t, 1);
  ctx.restore();
  // free arm (sleeve) passes BEHIND the head; only the gloved paw is drawn in front (below)
  const sR = shoulder(pose, 1);
  const fa = ikSoft(sR, pose.free.target, RIG.upper, RIG.fore, freeBend(pose.free.target, sR));
  drawLimb(ctx, [sR, fa.elbow, fa.hand], [48, 41, 34], JACKET, 7);
  sparkleSleeve(ctx, [sR, fa.elbow, fa.hand], t, 11);
  // head + fedora
  drawHeadAt(ctx, pose, pose.face, { after: (c) => fedora(c, pose.hatTilt, t) });
  // microphone held to the mouth (rigidly attached to the head pose)
  const n = neck(pose), ha = headAngle(pose);
  const micAngle = 0.34;
  const grilleHead = { x: 548, y: 552 };
  const g = headToChar(pose, grilleHead.x, grilleHead.y);
  const ang = ha + micAngle;
  const dir = { x: -Math.sin(ang), y: Math.cos(ang) };
  const grip = { x: g.x + dir.x * 96, y: g.y + dir.y * 96 };
  const sL = shoulder(pose, -1);
  let arm = ikSoft(sL, grip, RIG.upper, RIG.fore, 1);
  const fd = { x: arm.hand.x - arm.elbow.x, y: arm.hand.y - arm.elbow.y }, fl = Math.hypot(fd.x, fd.y) || 1;
  arm = ikSoft(sL, { x: grip.x - (fd.x / fl) * 26, y: grip.y - (fd.y / fl) * 26 }, RIG.upper, RIG.fore, 1);
  ctx.save();
  ctx.translate(g.x, g.y);
  ctx.rotate(ang);
  microphone(ctx);
  ctx.restore();
  drawLimb(ctx, [sL, arm.elbow, arm.hand], [48, 41, 34], JACKET, 7);
  sparkleSleeve(ctx, [sL, arm.elbow, arm.hand], t, 3);
  drawPaw(ctx, arm.hand.x, arm.hand.y, Math.atan2(arm.hand.y - arm.elbow.y, arm.hand.x - arm.elbow.x), { curl: 1, side: -1 });
  // the single sparkling white glove (in front of everything)
  const wang = Math.atan2(fa.hand.y - fa.elbow.y, fa.hand.x - fa.elbow.x);
  drawPaw(ctx, fa.hand.x, fa.hand.y, wang, { curl: pose.free.curl, point: pose.free.point, side: 1, glove: true, sparkle: t });
}

/** Elbow out & down for side/raised gestures, inward-down when the hand crosses the body. */
function freeBend(target, s) {
  const dx = target.x - s.x, dy = target.y - s.y;
  const across = smoothstep(40, -40, dx); // hand moving to the body's other side
  const low = smoothstep(-40, 120, dy);
  return clamp(1 - 2 * Math.max(across, low * 0.9), -1, 1);
}

function sparkleSleeve(ctx, pts, t, seed) {
  for (let i = 0; i < 7; i++) {
    const seg = i % 2, u = ((i * 0.37 + seed * 0.13) % 1);
    const a = pts[seg], b = pts[seg + 1];
    const x = a.x + (b.x - a.x) * u + Math.sin(i * 7 + seed) * 14, y = a.y + (b.y - a.y) * u + Math.cos(i * 5 + seed) * 14;
    const tw = 0.5 + 0.5 * Math.sin((TAU * ((i % 5) + 2) * t) / 15 + i + seed);
    ctx.fillStyle = `rgba(230,232,255,${0.2 + 0.7 * tw * tw})`;
    ctx.beginPath(); ctx.arc(x, y, 2.4 + 1.6 * tw, 0, TAU); ctx.fill();
  }
}
