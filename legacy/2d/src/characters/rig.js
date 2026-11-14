// Shared VANKING mole rig: skeleton constants, limb/paw/shoe drawing, soft IK, head placement.
// Character space = brand-SVG units; ground at y = 0, character centre x = 0, y grows downward.
import { clamp, lerp, ik2, TAU } from '../math.js';
import { drawHead, HEAD_PIVOT } from './head.js';

export const RIG = {
  hipY: -236,            // rest hip centre height
  hipHalf: 74,           // hip-joint half spacing
  thigh: 104, shin: 100, // leg bones (hip joint → knee → ankle)
  ankleH: 34,            // ankle above the sole
  shoulderX: 150, shoulderY: -282, // torso-local (relative to hip centre)
  neckY: -300,           // torso-local neck pivot
  upper: 190, fore: 180, // arm bones
};

export const LIGHT = { x: -0.62, y: -0.78 }; // moonlight direction (upper-left), unit-ish

// ---------------------------------------------------------------- transforms
export function torsoPoint(pose, lx, ly) {
  const c = Math.cos(pose.torso), s = Math.sin(pose.torso);
  const sx = pose.twistScale ?? 1;
  const x = lx * sx;
  return { x: pose.hip.x + x * c - ly * s, y: pose.hip.y + x * s + ly * c };
}
export const shoulder = (pose, side) => torsoPoint(pose, side * RIG.shoulderX, RIG.shoulderY - (side > 0 ? pose.shrugR || 0 : pose.shrugL || 0));
export const neck = (pose) => torsoPoint(pose, (pose.headShift?.x || 0), RIG.neckY + (pose.headShift?.y || 0));
export const headAngle = (pose) => pose.torso + pose.head;
/** Map a point in head (brand SVG) coordinates to character space. */
export function headToChar(pose, hx, hy) {
  const n = neck(pose), a = headAngle(pose), c = Math.cos(a), s = Math.sin(a);
  const x = hx - HEAD_PIVOT.x, y = hy - HEAD_PIVOT.y;
  return { x: n.x + x * c - y * s, y: n.y + x * s + y * c };
}

// ---------------------------------------------------------------- soft IK
/** Two-bone IK with a continuous bend side: bend ∈ [-1, 1]. Returns {elbow, hand}. */
export function ikSoft(root, target, l1, l2, bend) {
  const a = ik2(root, target, l1, l2, 1).joint;
  const b = ik2(root, target, l1, l2, -1).joint;
  const w = clamp((bend + 1) / 2);
  let dx = w * (a.x - root.x) + (1 - w) * (b.x - root.x);
  let dy = w * (a.y - root.y) + (1 - w) * (b.y - root.y);
  let d = Math.hypot(dx, dy);
  if (d < 1e-3) { dx = target.x - root.x; dy = target.y - root.y; d = Math.hypot(dx, dy) || 1; }
  const elbow = { x: root.x + (dx / d) * l1, y: root.y + (dy / d) * l1 };
  const ex = target.x - elbow.x, ey = target.y - elbow.y, ed = Math.hypot(ex, ey) || 1;
  const reach = Math.min(l2, ed + 0.001);
  const hand = { x: elbow.x + (ex / ed) * reach, y: elbow.y + (ey / ed) * reach };
  return { elbow, hand };
}

// ---------------------------------------------------------------- primitive shapes
/** Append a tapered capsule (circle A radius ra → circle B radius rb) to the current path. */
export function capsule(ctx, ax, ay, bx, by, ra, rb) {
  const dx = bx - ax, dy = by - ay, d = Math.hypot(dx, dy);
  if (d < Math.abs(ra - rb) + 0.01) { // one circle contains the other
    const r = Math.max(ra, rb), cx = ra >= rb ? ax : bx, cy = ra >= rb ? ay : by;
    ctx.moveTo(cx + r, cy); ctx.arc(cx, cy, r, 0, TAU); return;
  }
  const ang = Math.atan2(dy, dx);
  const off = Math.asin(clamp((ra - rb) / d, -1, 1));
  const a1 = ang + Math.PI / 2 + off;
  ctx.moveTo(ax + ra * Math.cos(a1), ay + ra * Math.sin(a1));
  ctx.arc(ax, ay, ra, a1, a1 + Math.PI - 2 * off, false);
  ctx.lineTo(bx + rb * Math.cos(a1 + Math.PI - 2 * off), by + rb * Math.sin(a1 + Math.PI - 2 * off));
  ctx.arc(bx, by, rb, a1 + Math.PI - 2 * off, a1 + 2 * Math.PI, false);
  ctx.closePath();
}

/** Cylinder-shaded gradient across a segment, lit from LIGHT. */
function segGradient(ctx, a, b, r, col) {
  const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1;
  let nx = -dy / d, ny = dx / d;
  if (nx * LIGHT.x + ny * LIGHT.y < 0) { nx = -nx; ny = -ny; }
  const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
  const g = ctx.createLinearGradient(mx + nx * r, my + ny * r, mx - nx * r, my - ny * r);
  g.addColorStop(0, col.light);
  g.addColorStop(0.42, col.base);
  g.addColorStop(1, col.dark);
  return g;
}

/**
 * Draw a limb (chain of points with radii) with an outline: stroke pass then fill pass.
 * col = { base, light, dark, line }, ow = outline width.
 */
export function drawLimb(ctx, pts, radii, col, ow = 7) {
  ctx.beginPath();
  for (let i = 0; i < pts.length - 1; i++) capsule(ctx, pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y, radii[i] + ow, radii[i + 1] + ow);
  ctx.fillStyle = col.line;
  ctx.fill();
  for (let i = 0; i < pts.length - 1; i++) {
    ctx.beginPath();
    capsule(ctx, pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y, radii[i], radii[i + 1]);
    ctx.fillStyle = segGradient(ctx, pts[i], pts[i + 1], Math.max(radii[i], radii[i + 1]), col);
    ctx.fill();
  }
  // fold crease at the middle joint
  if (pts.length === 3) {
    const j = pts[1];
    const a = Math.atan2(pts[0].y - j.y, pts[0].x - j.x), b = Math.atan2(pts[2].y - j.y, pts[2].x - j.x);
    let mid = (a + b) / 2;
    if (Math.abs(a - b) > Math.PI) mid += Math.PI;
    const bendAmt = Math.abs(Math.PI - Math.abs(((a - b + 3 * Math.PI) % TAU) - Math.PI));
    if (bendAmt > 0.35) {
      ctx.strokeStyle = col.line; ctx.globalAlpha = clamp((bendAmt - 0.35) * 1.2) * 0.55; ctx.lineWidth = 4; ctx.lineCap = 'round';
      const r = radii[1] * 0.8;
      ctx.beginPath(); ctx.arc(j.x, j.y, r, mid - 0.45, mid + 0.45); ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }
}

/** Limb outline path (union of capsules) — handy for clipping decorations. */
export function limbPath(ctx, pts, radii) {
  ctx.beginPath();
  for (let i = 0; i < pts.length - 1; i++) capsule(ctx, pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y, radii[i], radii[i + 1]);
}

// ---------------------------------------------------------------- paws
const PAW = { fill0: '#F6CBA2', fill1: '#E7B48C', fill2: '#D6A07A', line: '#8a5f43', crease: '#aa7b56' };
const GLOVE = { fill0: '#FFFFFF', fill1: '#ECEEF6', fill2: '#C9CCDC', line: '#7c8098', crease: '#a3a8bf' };

/**
 * Mole paw at wrist position (x, y), pointing along `angle`.
 * opts: { curl: 0 open … 1 fist, point: 0..1 index extended, side: ±1 (thumb side), glove, sparkle: t,
 *         press: [f0, f1, f2] finger press amounts (sax keys), scale }
 */
export function drawPaw(ctx, x, y, angle, opts = {}) {
  const P = opts.glove ? GLOVE : PAW;
  const curl = clamp(opts.curl ?? 0.5), point = clamp(opts.point ?? 0), side = opts.side ?? 1, s = (opts.scale ?? 1) * 1.2;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.scale(s, s * side);
  const grad = ctx.createLinearGradient(10, -36, 60, 40);
  grad.addColorStop(0, P.fill0); grad.addColorStop(0.55, P.fill1); grad.addColorStop(1, P.fill2);
  const fingers = [-0.42, -0.14, 0.14, 0.42];
  const press = opts.press || [0, 0, 0, 0];
  const shapeFingers = (pad) => {
    ctx.beginPath();
    // palm
    ctx.ellipse(30, 0, 36 + pad, 33 + pad, 0, 0, TAU);
    // thumb (on the -y side after mirroring)
    const ta = -1.15 + curl * 0.55;
    ctx.moveTo(30 + Math.cos(ta) * 26 + 15 + pad, -16 + Math.sin(ta) * 26);
    ctx.ellipse(30 + Math.cos(ta) * 26, -16 + Math.sin(ta) * 26, 15 + pad, 11 + pad, ta, 0, TAU);
    fingers.forEach((fa, i) => {
      const isIndex = i === 0;
      const ext = isIndex ? Math.max(1 - curl, point) : 1 - curl;
      const len = 14 + ext * 22 - press[i % press.length] * 6;
      const spread = fa * (0.35 + 0.65 * ext) * (isIndex && point > 0.5 ? 0.4 : 1);
      const bx = 52 + Math.cos(spread) * (8 + len * 0.5), by = Math.sin(spread) * 26 * (0.75 + 0.25 * ext) + spread * 6;
      ctx.moveTo(bx + len * 0.55 + pad, by);
      ctx.ellipse(bx, by, len * 0.55 + 6 + pad, 10.5 + pad, spread, 0, TAU);
    });
  };
  ctx.fillStyle = P.line;
  shapeFingers(5);
  ctx.fill();
  ctx.fillStyle = grad;
  shapeFingers(0);
  ctx.fill();
  // knuckle creases
  ctx.strokeStyle = P.crease; ctx.lineWidth = 3.2; ctx.lineCap = 'round';
  ctx.beginPath();
  for (const fa of [-0.28, 0, 0.28]) {
    const cx = 52 + 6 * (1 - curl), cy = fa * 46;
    ctx.moveTo(cx - 3, cy - 2); ctx.quadraticCurveTo(cx + 4, cy, cx - 2, cy + 4);
  }
  ctx.stroke();
  // soft highlight
  ctx.fillStyle = 'rgba(255,255,255,0.28)';
  ctx.beginPath(); ctx.ellipse(22, -12, 14, 8, -0.4, 0, TAU); ctx.fill();
  if (opts.glove && opts.sparkle !== undefined) {
    // rhinestones twinkling (periods divide the loop)
    const pts = [[18, -6], [34, 10], [46, -14], [28, 22], [58, 4], [12, 14], [40, -24], [62, -10]];
    pts.forEach(([px, py], i) => {
      const tw = 0.5 + 0.5 * Math.sin(TAU * ((i % 5) + 2) * opts.sparkle / 15 + i * 1.7);
      ctx.fillStyle = `rgba(255,255,255,${0.55 + 0.45 * tw})`;
      ctx.beginPath(); ctx.arc(px, py, 2.6, 0, TAU); ctx.fill();
      if (tw > 0.75) {
        const l = 9 * (tw - 0.75) * 4;
        ctx.strokeStyle = `rgba(255,255,255,${(tw - 0.75) * 3})`; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(px - l, py); ctx.lineTo(px + l, py); ctx.moveTo(px, py - l); ctx.lineTo(px, py + l); ctx.stroke();
      }
    });
  }
  ctx.restore();
}

// ---------------------------------------------------------------- shoes
/**
 * Shoe at the ankle. side = -1 (viewer-left foot, toe points left) or +1.
 * toe: radians lifting the toe (pivot at heel); heel: radians lifting the heel (pivot at toe).
 * style: { upper, upperLight, sole, line, toeCap?, laces?, shine? }
 */
export function drawShoe(ctx, ankle, side, style, toe = 0, heel = 0) {
  ctx.save();
  ctx.translate(ankle.x, ankle.y);
  ctx.scale(side, 1);
  // pivots (in foot space, sole at y = ankleH)
  const H = RIG.ankleH;
  if (toe > 0) { ctx.translate(-36, H); ctx.rotate(-toe); ctx.translate(36, -H); }
  if (heel > 0) { ctx.translate(86, H); ctx.rotate(heel); ctx.translate(-86, -H); }
  const shape = () => {
    ctx.beginPath();
    ctx.moveTo(-40, -14);
    ctx.bezierCurveTo(-50, 6, -50, 26, -42, H);
    ctx.lineTo(74, H);
    ctx.bezierCurveTo(98, H, 104, 18, 92, 4);
    ctx.bezierCurveTo(78, -12, 42, -22, 12, -24);
    ctx.bezierCurveTo(-12, -26, -30, -24, -40, -14);
    ctx.closePath();
  };
  ctx.lineJoin = 'round';
  shape();
  ctx.lineWidth = 12; ctx.strokeStyle = style.line; ctx.stroke();
  const g = ctx.createLinearGradient(0, -26, 0, H);
  g.addColorStop(0, style.upperLight); g.addColorStop(0.6, style.upper); g.addColorStop(1, style.upper);
  ctx.fillStyle = g; ctx.fill();
  ctx.save();
  shape(); ctx.clip();
  if (style.toeCap) { ctx.fillStyle = style.toeCap; ctx.beginPath(); ctx.ellipse(84, 10, 30, 34, -0.2, 0, TAU); ctx.fill(); }
  if (style.heelCap) { ctx.fillStyle = style.heelCap; ctx.beginPath(); ctx.ellipse(-44, 6, 22, 34, 0.1, 0, TAU); ctx.fill(); }
  ctx.fillStyle = style.sole; ctx.fillRect(-60, H - 9, 180, 14);
  if (style.shine) {
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.beginPath(); ctx.ellipse(52, -6, 22, 5, -0.12, 0, TAU); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.beginPath(); ctx.ellipse(-20, -8, 12, 4, 0.1, 0, TAU); ctx.fill();
  }
  if (style.laces) {
    ctx.strokeStyle = style.laces; ctx.lineWidth = 4.5; ctx.lineCap = 'round';
    ctx.beginPath();
    for (let i = 0; i < 3; i++) { const x = 4 + i * 14; ctx.moveTo(x - 5, -18 + i * 2); ctx.lineTo(x + 7, -10 + i * 3); ctx.moveTo(x + 7, -18 + i * 2); ctx.lineTo(x - 5, -10 + i * 3); }
    ctx.stroke();
  }
  ctx.restore();
  ctx.restore();
}

// ---------------------------------------------------------------- legs
/** Knee via IK (knees bend outward in the frontal view). */
export function legChain(pose, side) {
  const hipJ = { x: pose.hip.x + side * RIG.hipHalf, y: pose.hip.y + 14 };
  const foot = side < 0 ? pose.footL : pose.footR;
  const ankle = { x: foot.x, y: foot.y - RIG.ankleH };
  const { elbow: knee, hand: end } = ikSoft(hipJ, ankle, RIG.thigh, RIG.shin, side < 0 ? 1 : -1);
  return { hipJ, knee, ankle: end };
}

// ---------------------------------------------------------------- head placement
export function drawHeadAt(ctx, pose, face, decorate) {
  const n = neck(pose);
  ctx.save();
  ctx.translate(n.x, n.y);
  ctx.rotate(headAngle(pose));
  ctx.translate(-HEAD_PIVOT.x, -HEAD_PIVOT.y);
  decorate?.before?.(ctx);
  drawHead(ctx, face);
  decorate?.after?.(ctx);
  ctx.restore();
}

/** Contact shadow under the feet (drawn on the ground before the character). */
export function drawContactShadow(ctx, pose, spread = 1) {
  const cx = (pose.footL.x + pose.footR.x) / 2;
  const w = Math.abs(pose.footR.x - pose.footL.x) / 2 + 150 * spread;
  const g = ctx.createRadialGradient(cx, 0, 10, cx, 0, w);
  g.addColorStop(0, 'rgba(0,10,15,0.5)');
  g.addColorStop(1, 'rgba(0,10,15,0)');
  ctx.save();
  ctx.translate(0, 6);
  ctx.scale(1, 0.18);
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(cx, 0, w, 0, TAU); ctx.fill();
  ctx.restore();
}

export const lerpPt = (a, b, u) => ({ x: lerp(a.x, b.x, u), y: lerp(a.y, b.y, u) });
