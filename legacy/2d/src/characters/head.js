// Animated official VANKING.UZ mole head, drawn in brand-SVG coordinates.
import { HEAD_OPS, HEAD_GRADIENTS, EYE_CENTERS } from '../assets/mascot-head.js';
import { drawOp, P } from '../draw/svgops.js';
import { clamp, lerp, smoothstep } from '../math.js';

/** Neck pivot of the head art (bottom-centre of the chin) in SVG coordinates. */
export const HEAD_PIVOT = { x: 617, y: 640 };
export const MOUTH_CENTER = { x: 622, y: 470 };

const EYE_OPEN_CLIP = {
  left: 'M494 316C496 295 517 265 540 257C561 250 579 257 590 276C602 297 604 320 595 342C584 368 562 383 540 378C511 376 492 349 494 316Z',
  right: 'M685 344C686 323 704 301 724 298C745 293 765 310 775 331C787 354 787 380 776 401C767 421 751 432 733 427C711 423 693 409 686 389C680 373 681 357 685 344Z',
};
// Upper-lid lash line endpoints (from the brand art) used for sleepy/half-closed lids.
const LASH = {
  left: { a: [496, 302], c: [548, 288], b: [601, 297], bottom: 378, fill: 'head-left-lid', stroke: '#9b745c' },
  right: { a: [686, 330], c: [733, 320], b: [782, 350], bottom: 428, fill: 'head-right-lid', stroke: '#98715a' },
};

const MOUTH_TOP = 'M574 463C583 464 594 458 601 453C619 473 647 482 671 467';
const TOOTH_L = 'M576 465C582 465 590 463 596 460L593 491C582 492 576 482 576 465Z';
const TOOTH_R = 'M602 460C610 466 617 471 625 474C622 486 615 496 606 497C601 498 597 496 594 493Z';
const SMILE = 'M543 432C545 452 560 465 579 463C589 462 596 457 601 452C619 473 648 484 675 465';

function lidGradient(ctx, side) {
  const g = HEAD_GRADIENTS[LASH[side].fill];
  const gr = ctx.createLinearGradient(g.x1, g.y1, g.x2, g.y2);
  for (const s of g.stops) gr.addColorStop(s.offset, s.color);
  return gr;
}

/** Open singing mouth. open ∈ [0,1]; shape: 'a' | 'o' | 'e'. */
function mouthPath(open, shape) {
  const D = 30 + open * (shape === 'e' ? 22 : shape === 'o' ? 58 : 66);
  const pinch = shape === 'o' ? 14 * open : 0; // rounder bottom for "o"
  const y = 470 + D;
  return `${MOUTH_TOP}C${671 - pinch * 0.3} ${467 + D * 0.55} ${652 - pinch} ${y} ${623} ${y}` +
    `C${596 + pinch} ${y} ${575 + pinch * 0.3} ${463 + D * 0.6} 574 463Z`;
}

function drawSingingMouth(ctx, open, shape) {
  const d = mouthPath(open, shape);
  const p = P(d);
  ctx.save();
  ctx.fillStyle = '#5a2f1b';
  ctx.fill(p);
  ctx.clip(p);
  // throat shadow + tongue
  const D = 30 + open * 66;
  ctx.fillStyle = '#3b1d10';
  ctx.beginPath(); ctx.ellipse(622, 468 + D * 0.42, 34, 10 + D * 0.18, 0, 0, Math.PI * 2); ctx.fill();
  const tg = smoothstep(0.02, 0.3, open); // tongue fades in with opening → no pop vs. the closed grin
  ctx.globalAlpha = tg;
  ctx.fillStyle = '#c0634a';
  ctx.beginPath(); ctx.ellipse(626, 476 + D * 0.95, 44, 12 + D * 0.22, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = 'rgba(232,140,110,0.55)';
  ctx.beginPath(); ctx.ellipse(616, 470 + D * 0.88, 16, 5 + D * 0.06, -0.2, 0, Math.PI * 2); ctx.fill();
  ctx.globalAlpha = 1;
  drawTeeth(ctx);
  ctx.restore();
  ctx.lineWidth = 6; ctx.lineJoin = 'round'; ctx.strokeStyle = '#7a452b';
  ctx.stroke(p);
}

/** The iconic buck teeth (brand art), in head coordinates. */
export function drawTeeth(ctx) {
  const g = HEAD_GRADIENTS['head-tooth'];
  const gr = ctx.createLinearGradient(g.x1, g.y1, g.x2, g.y2);
  for (const s of g.stops) gr.addColorStop(s.offset, s.color);
  ctx.fillStyle = gr;
  ctx.fill(P(TOOTH_L));
  ctx.fill(P(TOOTH_R));
  ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(150,120,90,0.45)';
  ctx.stroke(P(TOOTH_L)); ctx.stroke(P(TOOTH_R));
}

/** Small puckered embouchure for the saxophonist (mouthpiece + teeth drawn by the caller). */
function drawSaxMouth(ctx, puff) {
  ctx.save();
  // puffed cheeks: soft volume + highlight
  for (const [x, y, r] of [[528, 470, 62], [716, 486, 64]]) {
    const g = ctx.createRadialGradient(x - 10, y - 14, 4, x, y, r * (1 + puff * 0.18));
    g.addColorStop(0, `rgba(250,206,165,${0.55 + puff * 0.3})`);
    g.addColorStop(1, 'rgba(240,190,150,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, r * (1 + puff * 0.18), 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
  ctx.lineCap = 'round'; ctx.strokeStyle = '#7b472d'; ctx.lineWidth = 7;
  ctx.beginPath(); ctx.moveTo(560, 452); ctx.bezierCurveTo(572, 466, 588, 466, 601, 455); ctx.bezierCurveTo(616, 468, 638, 474, 660, 466); ctx.stroke();
  // pursed lips around the mouthpiece
  ctx.fillStyle = '#6e3a24';
  ctx.beginPath(); ctx.ellipse(612, 476, 34, 19, -0.12, 0, Math.PI * 2); ctx.fill();
  ctx.lineWidth = 6; ctx.strokeStyle = '#c98f6d';
  ctx.beginPath(); ctx.ellipse(612, 476, 36, 21, -0.12, 0, Math.PI * 2); ctx.stroke();
}

function drawLid(ctx, side, drop) {
  if (drop <= 0.001) return;
  const L = LASH[side];
  const top = Math.min(L.a[1], L.b[1], L.c[1]) - 8;
  const y = lerp(top, L.bottom - 6, clamp(drop));
  ctx.save();
  ctx.clip(P(EYE_OPEN_CLIP[side]));
  ctx.fillStyle = lidGradient(ctx, side);
  ctx.beginPath();
  ctx.moveTo(L.a[0] - 30, top - 40);
  ctx.lineTo(L.b[0] + 30, top - 40);
  ctx.lineTo(L.b[0] + 30, y + (L.b[1] - L.a[1]) * 0.5);
  ctx.quadraticCurveTo(L.c[0], y + 16, L.a[0] - 30, y - (L.b[1] - L.a[1]) * 0.5);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
  ctx.save();
  ctx.clip(P(EYE_OPEN_CLIP[side]));
  ctx.strokeStyle = '#3b2b1e'; ctx.lineWidth = 8; ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(L.a[0] - 30, y - (L.b[1] - L.a[1]) * 0.5);
  ctx.quadraticCurveTo(L.c[0], y + 16, L.b[0] + 30, y + (L.b[1] - L.a[1]) * 0.5);
  ctx.stroke();
  ctx.restore();
}

/**
 * face = {
 *   blink: 0..1 (1 = closed) or blinkL/blinkR,
 *   pupil: {x, y} offset in head units (±14),
 *   lid: 0..1 upper-lid droop (sleepy / "feeling the music"),
 *   mouth: { shape: 'smile' | 'a' | 'o' | 'e' | 'sax', open: 0..1, puff: 0..1 },
 *   whisker: radians of whisker wiggle,
 *   colorFn: optional colour mapping for flat fills (lighting)
 * }
 */
export function drawHead(ctx, face = {}) {
  const blinkL = clamp(face.blinkL ?? face.blink ?? 0);
  const blinkR = clamp(face.blinkR ?? face.blink ?? 0);
  const pupil = face.pupil || { x: 0, y: 0 };
  const mouth = face.mouth || { shape: 'smile' };
  const custom = mouth.shape && mouth.shape !== 'smile';
  const wh = face.whisker || 0;
  let mouthDone = false;
  const opts = face.colorFn ? { color: face.colorFn } : {};
  for (const op of HEAD_OPS) {
    const g = op.groups || [];
    if (custom && (op.tag === 'mouth' || op.tag === 'tooth' || op.tag === 'smile' || op.tag === 'dimple' || op.tag === 'tongue')) {
      if (!mouthDone) {
        mouthDone = true;
        if (mouth.shape === 'sax') drawSaxMouth(ctx, mouth.puff || 0);
        else {
          drawSingingMouth(ctx, clamp(mouth.open ?? 0.5), mouth.shape);
          ctx.lineCap = 'round'; ctx.lineWidth = 7; ctx.strokeStyle = '#7b472d'; ctx.stroke(P(SMILE));
        }
      }
      continue;
    }
    if (op.id === 'eye-closed-left' || op.id === 'eye-closed-right') {
      const b = op.id.endsWith('left') ? blinkL : blinkR;
      const a = smoothstep(0.45, 0.85, b);
      if (a > 0) drawOp(ctx, { ...op, alpha: a }, HEAD_GRADIENTS, opts);
      continue;
    }
    const eyeSide = g.includes('eye-left') ? 'left' : g.includes('eye-right') ? 'right' : null;
    const whiskerSide = g.includes('whiskers-left') ? 'left' : g.includes('whiskers-right') ? 'right' : null;
    if (eyeSide) {
      const b = eyeSide === 'left' ? blinkL : blinkR;
      const [cx, cy] = EYE_CENTERS[eyeSide];
      ctx.save();
      ctx.translate(cx, cy); ctx.scale(1, 1 - 0.94 * b); ctx.translate(-cx, -cy);
      if (op.tag === 'pupil' || op.tag === 'glint') {
        // shift pupil inside its clip region: apply clip first, then translate
        const { clips, ...rest } = op;
        if (clips) { ctx.save(); drawClipped(ctx, clips, () => { ctx.translate(pupil.x, pupil.y); drawOp(ctx, rest, HEAD_GRADIENTS, opts); }); ctx.restore(); }
        else { ctx.translate(pupil.x, pupil.y); drawOp(ctx, rest, HEAD_GRADIENTS, opts); }
      } else drawOp(ctx, op, HEAD_GRADIENTS, opts);
      ctx.restore();
      // after the last op of the eye group, draw the sleepy lid on top
      if (op.d.startsWith('M496 302 Q548 288 601 297') || op.d.startsWith('M686 330 Q733 320 782 350')) {
        if (op.stroke) drawLid(ctx, eyeSide, (face.lid || 0) * (1 - b));
      }
      continue;
    }
    if (whiskerSide && wh) {
      const pivot = whiskerSide === 'left' ? [540, 420] : [692, 460];
      ctx.save();
      ctx.translate(pivot[0], pivot[1]); ctx.rotate(whiskerSide === 'left' ? -wh : wh); ctx.translate(-pivot[0], -pivot[1]);
      drawOp(ctx, op, HEAD_GRADIENTS, opts);
      ctx.restore();
      continue;
    }
    drawOp(ctx, op, HEAD_GRADIENTS, opts);
  }
}

function drawClipped(ctx, clips, fn) {
  for (const c of clips) for (const s of c.shapes) ctx.clip(P(s.d));
  fn();
}
