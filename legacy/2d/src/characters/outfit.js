// Outfits & props: torso shapes, jackets, shirts, belts, hats, hair, microphone, saxophone.
// Torso functions draw in torso-local space (origin = hip centre, y up is negative).
// Hat/hair functions draw in head (brand SVG) space.
import { TAU, clamp } from '../math.js';
import { capsule } from './rig.js';

export function torsoPath(ctx, w = 1) {
  ctx.beginPath();
  ctx.moveTo(-150 * w, -6);
  ctx.bezierCurveTo(-170 * w, -80, -178 * w, -190, -168 * w, -262);
  ctx.bezierCurveTo(-162 * w, -300, -130 * w, -322, -90 * w, -338);
  ctx.bezierCurveTo(-50 * w, -354, 50 * w, -354, 90 * w, -338);
  ctx.bezierCurveTo(130 * w, -322, 162 * w, -300, 168 * w, -262);
  ctx.bezierCurveTo(178 * w, -190, 170 * w, -80, 150 * w, -6);
  ctx.bezierCurveTo(100 * w, 18, -100 * w, 18, -150 * w, -6);
  ctx.closePath();
}

/** Fill the torso with a lit cloth gradient and outline. col = {light, base, dark, line} */
export function drawTorsoCloth(ctx, col, w = 1, ow = 8) {
  torsoPath(ctx, w);
  ctx.lineWidth = ow * 2; ctx.lineJoin = 'round'; ctx.strokeStyle = col.line; ctx.stroke();
  const g = ctx.createLinearGradient(-170, -330, 150, 10);
  g.addColorStop(0, col.light); g.addColorStop(0.45, col.base); g.addColorStop(1, col.dark);
  ctx.fillStyle = g; ctx.fill();
  // side shading for volume
  ctx.save();
  torsoPath(ctx, w); ctx.clip();
  const sg = ctx.createLinearGradient(-175, 0, 175, 0);
  sg.addColorStop(0, 'rgba(255,255,255,0.07)'); sg.addColorStop(0.3, 'rgba(255,255,255,0)');
  sg.addColorStop(0.75, 'rgba(0,0,0,0)'); sg.addColorStop(1, 'rgba(0,0,0,0.28)');
  ctx.fillStyle = sg; ctx.fillRect(-180, -360, 360, 380);
  ctx.restore();
}

// ---------------------------------------------------------------- singer: open sequined jacket
export function singerJacket(ctx, t, w) {
  const jacket = { light: '#3a3a4a', base: '#1b1b24', dark: '#07070a', line: '#030304' };
  drawTorsoCloth(ctx, jacket, w);
  ctx.save();
  torsoPath(ctx, w); ctx.clip();
  // white T-shirt in the open front
  const shirt = () => { ctx.beginPath(); ctx.moveTo(-82, -360); ctx.lineTo(82, -360); ctx.bezierCurveTo(70, -250, 58, -120, 56, 20); ctx.lineTo(-56, 20); ctx.bezierCurveTo(-58, -120, -70, -250, -82, -360); ctx.closePath(); };
  shirt();
  const sg = ctx.createLinearGradient(-60, -300, 60, 0);
  sg.addColorStop(0, '#ffffff'); sg.addColorStop(1, '#d9dbe4');
  ctx.fillStyle = sg; ctx.fill();
  ctx.strokeStyle = 'rgba(120,125,150,0.5)'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(-30, -200); ctx.quadraticCurveTo(-10, -150, -26, -90); ctx.moveTo(32, -170); ctx.quadraticCurveTo(14, -120, 30, -60); ctx.stroke();
  // satin lapel edges
  for (const s of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(s * 82, -362); ctx.bezierCurveTo(s * 70, -250, s * 58, -120, s * 56, 22);
    ctx.lineTo(s * 96, 22); ctx.bezierCurveTo(s * 100, -110, s * 110, -230, s * 118, -330); ctx.closePath();
    const lg = ctx.createLinearGradient(s * 60, -300, s * 110, -100);
    lg.addColorStop(0, '#4a4a5e'); lg.addColorStop(0.5, '#23232e'); lg.addColorStop(1, '#0c0c12');
    ctx.fillStyle = lg; ctx.fill();
    ctx.strokeStyle = '#020203'; ctx.lineWidth = 5; ctx.stroke();
    ctx.strokeStyle = 'rgba(200,205,240,0.35)'; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(s * 90, -330); ctx.bezierCurveTo(s * 80, -230, s * 70, -110, s * 68, 10); ctx.stroke();
  }
  // sequins (twinkle with loop-safe periods)
  for (let i = 0; i < 90; i++) {
    const x = ((i * 97.3) % 330) - 165, y = -((i * 61.7) % 345) - 4;
    if (Math.abs(x) < 70 + (y + 360) * -0.02 && y < 15) continue;
    const tw = 0.5 + 0.5 * Math.sin((TAU * ((i % 7) + 2) * t) / 15 + i * 2.3);
    ctx.fillStyle = `rgba(${200 + 55 * tw},${200 + 55 * tw},255,${0.25 + 0.6 * tw * tw})`;
    ctx.beginPath(); ctx.arc(x, y, 2.6 + tw * 1.4, 0, TAU); ctx.fill();
  }
  ctx.restore();
}

// ---------------------------------------------------------------- sax: burgundy suit + bow tie
export function saxJacket(ctx, t, w) {
  const suit = { light: '#b13a5c', base: '#7a1d38', dark: '#3e0b1c', line: '#22050f' };
  drawTorsoCloth(ctx, suit, w);
  ctx.save();
  torsoPath(ctx, w); ctx.clip();
  // shirt V
  ctx.beginPath(); ctx.moveTo(-92, -360); ctx.lineTo(92, -360); ctx.lineTo(0, -150); ctx.closePath();
  const sh = ctx.createLinearGradient(0, -360, 0, -150);
  sh.addColorStop(0, '#fff8ea'); sh.addColorStop(1, '#e2d6bd');
  ctx.fillStyle = sh; ctx.fill();
  // lapels
  for (const s of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(s * 92, -362); ctx.lineTo(s * 4, -150); ctx.lineTo(s * 30, -150);
    ctx.lineTo(s * 128, -300); ctx.lineTo(s * 110, -312); ctx.lineTo(s * 128, -340); ctx.closePath();
    const lg = ctx.createLinearGradient(s * 40, -340, s * 110, -180);
    lg.addColorStop(0, '#9c2a4a'); lg.addColorStop(1, '#4c0e22');
    ctx.fillStyle = lg; ctx.fill();
    ctx.strokeStyle = '#22050f'; ctx.lineWidth = 5; ctx.lineJoin = 'round'; ctx.stroke();
  }
  // buttons
  for (const y of [-118, -62]) {
    ctx.fillStyle = '#e8bf55'; ctx.strokeStyle = '#7a5410'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(0, y, 10, 0, TAU); ctx.fill(); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.6)'; ctx.beginPath(); ctx.arc(-3, y - 3, 3, 0, TAU); ctx.fill();
  }
  ctx.strokeStyle = 'rgba(20,0,8,0.6)'; ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(0, -150); ctx.lineTo(0, 20); ctx.stroke();
  // breast pocket + gold pocket square (wearer's left = viewer's right)
  ctx.fillStyle = '#f1c75a'; ctx.strokeStyle = '#8a6416'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(84, -216); ctx.lineTo(96, -240); ctx.lineTo(106, -222); ctx.lineTo(118, -244); ctx.lineTo(124, -214); ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.strokeStyle = '#22050f'; ctx.lineWidth = 5;
  ctx.beginPath(); ctx.moveTo(78, -212); ctx.lineTo(132, -216); ctx.stroke();
  ctx.restore();
  // bow tie (drawn last so the chin sits just above it)
  ctx.save();
  ctx.translate(0, -258);
  ctx.fillStyle = '#121014'; ctx.strokeStyle = '#000'; ctx.lineWidth = 4; ctx.lineJoin = 'round';
  for (const s of [-1, 1]) {
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.bezierCurveTo(s * 20, -26, s * 52, -26, s * 56, -10); ctx.bezierCurveTo(s * 58, 6, s * 54, 24, s * 48, 26); ctx.bezierCurveTo(s * 30, 22, s * 16, 12, 0, 0); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.strokeStyle = 'rgba(255,255,255,0.22)'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(s * 14, -6); ctx.quadraticCurveTo(s * 34, -18, s * 50, -10); ctx.stroke();
    ctx.strokeStyle = '#000'; ctx.lineWidth = 4;
  }
  ctx.beginPath(); ctx.ellipse(0, 0, 13, 16, 0, 0, TAU); ctx.fill(); ctx.stroke();
  ctx.restore();
}

// ---------------------------------------------------------------- emo: black tee, pink graphic, studded belt
export function emoShirt(ctx, t, w) {
  const tee = { light: '#3a3442', base: '#1d1a22', dark: '#08070a', line: '#020203' };
  drawTorsoCloth(ctx, tee, w);
  ctx.save();
  torsoPath(ctx, w); ctx.clip();
  // pink heart with the brand "</>" mark
  ctx.save();
  ctx.translate(0, -175);
  const hp = () => { ctx.beginPath(); ctx.moveTo(0, 52); ctx.bezierCurveTo(-70, 10, -78, -44, -40, -54); ctx.bezierCurveTo(-18, -60, -4, -46, 0, -30); ctx.bezierCurveTo(4, -46, 18, -60, 40, -54); ctx.bezierCurveTo(78, -44, 70, 10, 0, 52); ctx.closePath(); };
  hp(); ctx.fillStyle = '#ff3f9a'; ctx.fill();
  ctx.lineWidth = 5; ctx.strokeStyle = '#ffd1ea'; ctx.stroke();
  // crack
  ctx.strokeStyle = '#1d1a22'; ctx.lineWidth = 5; ctx.lineJoin = 'miter';
  ctx.beginPath(); ctx.moveTo(0, -30); ctx.lineTo(-8, -8); ctx.lineTo(6, 6); ctx.lineTo(-4, 28); ctx.stroke();
  ctx.strokeStyle = '#b06cff'; ctx.lineWidth = 7; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.beginPath(); ctx.moveTo(-22, -10); ctx.lineTo(-36, 2); ctx.lineTo(-22, 14); ctx.moveTo(22, -10); ctx.lineTo(36, 2); ctx.lineTo(22, 14); ctx.stroke();
  // little stars
  for (const [x, y, r] of [[-92, -70, 9], [96, -40, 7], [-80, 40, 6]]) {
    ctx.fillStyle = '#c38bff'; ctx.beginPath();
    for (let k = 0; k < 10; k++) { const a = (k / 10) * TAU - Math.PI / 2, rr = k % 2 ? r * 0.42 : r; ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr); }
    ctx.closePath(); ctx.fill();
  }
  ctx.restore();
  // studded belt
  ctx.fillStyle = '#0b0a0d'; ctx.fillRect(-180, -46, 360, 32);
  ctx.strokeStyle = '#2d2a33'; ctx.lineWidth = 3; ctx.strokeRect(-180, -46, 360, 32);
  for (let x = -150; x <= 150; x += 25) {
    const g = ctx.createLinearGradient(x - 7, -38, x + 7, -22);
    g.addColorStop(0, '#ffffff'); g.addColorStop(0.5, '#b9bcc8'); g.addColorStop(1, '#6d7080');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.moveTo(x, -40); ctx.lineTo(x + 8, -30); ctx.lineTo(x, -20); ctx.lineTo(x - 8, -30); ctx.closePath(); ctx.fill();
  }
  // buckle
  ctx.strokeStyle = '#d8dbe6'; ctx.lineWidth = 6; ctx.strokeRect(-22, -48, 44, 36);
  ctx.restore();
}

/** Chain loop hanging from the emo belt (drawn after the torso, swings with motion). */
export function emoChain(ctx, swing) {
  ctx.save();
  ctx.translate(-120, -26);
  ctx.rotate(swing);
  ctx.strokeStyle = '#c9ccd8'; ctx.lineWidth = 4;
  for (let i = 0; i < 9; i++) {
    const a = (i / 8) * Math.PI;
    const x = 52 - Math.cos(a) * 52, y = Math.sin(a) * 48;
    ctx.beginPath(); ctx.ellipse(x, y, 7, 4.5, a + Math.PI / 2, 0, TAU); ctx.stroke();
  }
  ctx.restore();
}

// ---------------------------------------------------------------- hats & hair (head space)
export function fedora(ctx, tilt = 0, t = 0) {
  ctx.save();
  ctx.translate(617, 196);
  ctx.rotate(-0.1 + tilt);
  // brim (back half)
  ctx.lineJoin = 'round';
  ctx.fillStyle = '#0d0d12'; ctx.strokeStyle = '#000'; ctx.lineWidth = 8;
  ctx.beginPath(); ctx.ellipse(0, 0, 238, 40, 0, Math.PI, TAU); ctx.fill(); ctx.stroke();
  // crown
  const crown = () => { ctx.beginPath(); ctx.moveTo(-150, 8); ctx.bezierCurveTo(-160, -60, -140, -132, -96, -142); ctx.bezierCurveTo(-50, -150, -26, -124, 0, -122); ctx.bezierCurveTo(26, -124, 50, -150, 96, -142); ctx.bezierCurveTo(140, -132, 160, -60, 150, 8); ctx.closePath(); };
  crown();
  const cg = ctx.createLinearGradient(-150, -140, 150, 10);
  cg.addColorStop(0, '#3b3b4c'); cg.addColorStop(0.4, '#17171f'); cg.addColorStop(1, '#050507');
  ctx.fillStyle = cg; ctx.fill(); ctx.stroke();
  ctx.strokeStyle = 'rgba(160,165,200,0.35)'; ctx.lineWidth = 5;
  ctx.beginPath(); ctx.moveTo(-40, -126); ctx.quadraticCurveTo(0, -80, 40, -126); ctx.stroke();
  // band
  ctx.save(); crown(); ctx.clip();
  ctx.fillStyle = '#e9e9ef'; ctx.fillRect(-170, -36, 340, 32);
  ctx.fillStyle = 'rgba(0,0,0,0.18)'; ctx.fillRect(-170, -12, 340, 8);
  ctx.restore();
  // brim (front half)
  const bg = ctx.createLinearGradient(0, -10, 0, 44);
  bg.addColorStop(0, '#2c2c38'); bg.addColorStop(1, '#060608');
  ctx.fillStyle = bg; ctx.strokeStyle = '#000'; ctx.lineWidth = 8;
  ctx.beginPath(); ctx.ellipse(0, 0, 238, 40, 0, 0, Math.PI); ctx.bezierCurveTo(-200, 18, 200, 18, 238, 0); ctx.fill();
  ctx.beginPath(); ctx.ellipse(0, 0, 238, 40, 0, 0, Math.PI); ctx.stroke();
  ctx.strokeStyle = 'rgba(190,195,230,0.4)'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.ellipse(0, 2, 226, 32, 0, 0.25, Math.PI - 0.25); ctx.stroke();
  ctx.restore();
}

export function porkPie(ctx, tilt = 0) {
  ctx.save();
  ctx.translate(617, 200);
  ctx.rotate(0.06 + tilt);
  ctx.lineJoin = 'round';
  ctx.fillStyle = '#0f0d12'; ctx.strokeStyle = '#000'; ctx.lineWidth = 8;
  ctx.beginPath(); ctx.ellipse(0, 0, 205, 34, 0, Math.PI, TAU); ctx.fill(); ctx.stroke();
  const crown = () => { ctx.beginPath(); ctx.moveTo(-132, 6); ctx.bezierCurveTo(-136, -40, -130, -84, -118, -96); ctx.bezierCurveTo(-60, -108, 60, -108, 118, -96); ctx.bezierCurveTo(130, -84, 136, -40, 132, 6); ctx.closePath(); };
  crown();
  const cg = ctx.createLinearGradient(-130, -100, 130, 10);
  cg.addColorStop(0, '#3d3546'); cg.addColorStop(0.45, '#19151d'); cg.addColorStop(1, '#050406');
  ctx.fillStyle = cg; ctx.fill(); ctx.stroke();
  // flat-top rim
  ctx.strokeStyle = 'rgba(170,160,190,0.4)'; ctx.lineWidth = 4;
  ctx.beginPath(); ctx.ellipse(0, -92, 112, 10, 0, Math.PI * 1.05, TAU - 0.05); ctx.stroke();
  ctx.save(); crown(); ctx.clip();
  ctx.fillStyle = '#8c2140'; ctx.fillRect(-150, -30, 300, 28);
  ctx.fillStyle = '#e8bf55'; ctx.fillRect(-150, -30, 300, 4);
  ctx.restore();
  const bg = ctx.createLinearGradient(0, -6, 0, 36);
  bg.addColorStop(0, '#2e2833'); bg.addColorStop(1, '#060508');
  ctx.fillStyle = bg;
  ctx.beginPath(); ctx.ellipse(0, 0, 205, 34, 0, 0, Math.PI); ctx.bezierCurveTo(-170, 14, 170, 14, 205, 0); ctx.fill();
  ctx.strokeStyle = '#000'; ctx.lineWidth = 8;
  ctx.beginPath(); ctx.ellipse(0, 0, 205, 34, 0, 0, Math.PI); ctx.stroke();
  ctx.restore();
}

/** Asymmetric emo fringe sweeping over the viewer-left lens. sway in radians. */
export function emoHair(ctx, sway = 0, t = 0) {
  ctx.save();
  ctx.translate(640, 150);
  ctx.rotate(sway * 0.5);
  ctx.translate(-640, -150);
  ctx.lineJoin = 'round';
  // back volume with spikes
  const back = () => {
    ctx.beginPath();
    ctx.moveTo(452, 300);
    ctx.bezierCurveTo(430, 230, 452, 150, 520, 112);
    ctx.lineTo(560, 64); ctx.lineTo(588, 104); ctx.lineTo(640, 48); ctx.lineTo(660, 102); ctx.lineTo(726, 66); ctx.lineTo(724, 124);
    ctx.lineTo(792, 120); ctx.lineTo(770, 168);
    ctx.bezierCurveTo(812, 196, 826, 246, 818, 300);
    ctx.bezierCurveTo(790, 240, 740, 214, 690, 214);
    ctx.closePath();
  };
  back();
  ctx.fillStyle = '#0c0b10'; ctx.fill();
  ctx.strokeStyle = '#000'; ctx.lineWidth = 7; ctx.stroke();
  ctx.restore();
  // fringe swings more than the crown
  ctx.save();
  ctx.translate(700, 190);
  ctx.rotate(sway);
  ctx.translate(-700, -190);
  const fringe = () => {
    ctx.beginPath();
    ctx.moveTo(800, 222);
    ctx.bezierCurveTo(740, 170, 610, 160, 520, 214);
    ctx.bezierCurveTo(476, 244, 452, 300, 452, 352);
    ctx.bezierCurveTo(456, 384, 470, 404, 486, 418); // long tip over the left lens
    ctx.bezierCurveTo(486, 380, 498, 344, 520, 318);
    ctx.bezierCurveTo(528, 340, 536, 356, 552, 366);
    ctx.bezierCurveTo(548, 330, 560, 296, 590, 272);
    ctx.bezierCurveTo(640, 238, 722, 238, 800, 222);
    ctx.closePath();
  };
  fringe();
  const fg = ctx.createLinearGradient(480, 180, 700, 380);
  fg.addColorStop(0, '#2a2733'); fg.addColorStop(0.5, '#121017'); fg.addColorStop(1, '#050407');
  ctx.fillStyle = fg; ctx.fill();
  ctx.strokeStyle = '#000'; ctx.lineWidth = 7; ctx.stroke();
  ctx.save(); fringe(); ctx.clip();
  // hot-pink streak + glossy highlights
  ctx.strokeStyle = '#ff3f9a'; ctx.lineWidth = 22; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(760, 214); ctx.bezierCurveTo(660, 200, 560, 230, 500, 330); ctx.stroke();
  ctx.strokeStyle = 'rgba(190,180,230,0.35)'; ctx.lineWidth = 6;
  ctx.beginPath(); ctx.moveTo(720, 222); ctx.bezierCurveTo(640, 212, 560, 240, 520, 290); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(640, 250); ctx.bezierCurveTo(600, 262, 572, 284, 556, 320); ctx.stroke();
  ctx.restore();
  ctx.restore();
}

// ---------------------------------------------------------------- microphone (local: grille at 0,0; handle +y)
export function microphone(ctx) {
  ctx.lineJoin = 'round';
  // handle
  ctx.beginPath(); capsule(ctx, 0, 36, 0, 158, 15, 11);
  const hg = ctx.createLinearGradient(-15, 0, 15, 0);
  hg.addColorStop(0, '#5a5a66'); hg.addColorStop(0.35, '#1c1c22'); hg.addColorStop(1, '#050507');
  ctx.fillStyle = hg; ctx.fill(); ctx.strokeStyle = '#000'; ctx.lineWidth = 4; ctx.stroke();
  // band
  ctx.fillStyle = '#9da1b0'; ctx.fillRect(-17, 26, 34, 10);
  // grille
  const gg = ctx.createRadialGradient(-10, -10, 2, 0, 0, 30);
  gg.addColorStop(0, '#ffffff'); gg.addColorStop(0.4, '#c9ccd6'); gg.addColorStop(1, '#6d717f');
  ctx.fillStyle = gg; ctx.beginPath(); ctx.arc(0, 0, 30, 0, TAU); ctx.fill();
  ctx.strokeStyle = '#2d2f37'; ctx.lineWidth = 4; ctx.stroke();
  ctx.save(); ctx.beginPath(); ctx.arc(0, 0, 29, 0, TAU); ctx.clip();
  ctx.strokeStyle = 'rgba(40,42,50,0.45)'; ctx.lineWidth = 1.6;
  for (let i = -30; i <= 30; i += 7) { ctx.beginPath(); ctx.moveTo(i, -30); ctx.lineTo(i + 14, 30); ctx.moveTo(i, -30); ctx.lineTo(i - 14, 30); ctx.stroke(); }
  ctx.restore();
}

// ---------------------------------------------------------------- saxophone (local: mouthpiece tip at 0,0)
export const SAX_KEYS = { upper: { x: -40, y: 205 }, lower: { x: -42, y: 338 } };

function goldGradient(ctx, x0, y0, x1, y1) {
  const g = ctx.createLinearGradient(x0, y0, x1, y1);
  g.addColorStop(0, '#7a4a0c'); g.addColorStop(0.22, '#c98d22'); g.addColorStop(0.4, '#ffe08a');
  g.addColorStop(0.55, '#f2bf45'); g.addColorStop(0.8, '#b4781a'); g.addColorStop(1, '#5e3608');
  return g;
}

export function saxophone(ctx, glint = 0, t = 0) {
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  const line = '#4a2a05';
  // ---- bell section & bow (behind the body)
  const bow = () => {
    ctx.beginPath();
    ctx.moveTo(-74, 400);
    ctx.bezierCurveTo(-78, 486, 34, 500, 46, 420);
    ctx.lineTo(70, 312);
    ctx.lineTo(18, 300);
    ctx.lineTo(4, 410);
    ctx.bezierCurveTo(-2, 444, -36, 446, -38, 400);
    ctx.closePath();
  };
  bow();
  ctx.fillStyle = goldGradient(ctx, -80, 0, 80, 0); ctx.fill();
  ctx.strokeStyle = line; ctx.lineWidth = 6; ctx.stroke();
  // bell flare + opening
  ctx.save();
  ctx.translate(46, 300);
  ctx.rotate(-0.12);
  ctx.beginPath(); ctx.moveTo(-32, 10); ctx.bezierCurveTo(-46, -10, -70, -26, -78, -34); ctx.lineTo(78, -34); ctx.bezierCurveTo(70, -26, 46, -10, 32, 10); ctx.closePath();
  ctx.fillStyle = goldGradient(ctx, -80, 0, 80, 0); ctx.fill(); ctx.strokeStyle = line; ctx.lineWidth = 6; ctx.stroke();
  ctx.beginPath(); ctx.ellipse(0, -34, 80, 22, 0, 0, TAU);
  ctx.fillStyle = goldGradient(ctx, -80, -50, 80, -20); ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.ellipse(0, -32, 64, 15, 0, 0, TAU);
  const ig = ctx.createRadialGradient(0, -28, 4, 0, -32, 64);
  ig.addColorStop(0, '#3a1f02'); ig.addColorStop(0.7, '#7a4a0c'); ig.addColorStop(1, '#c98d22');
  ctx.fillStyle = ig; ctx.fill();
  ctx.restore();
  // ---- body tube (conical)
  ctx.beginPath(); capsule(ctx, -40, 126, -56, 404, 15, 34);
  ctx.fillStyle = goldGradient(ctx, -92, 0, -10, 0); ctx.fill();
  ctx.strokeStyle = line; ctx.lineWidth = 6; ctx.stroke();
  // key rods & pearl keys
  ctx.strokeStyle = '#8a5a10'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(-26, 150); ctx.lineTo(-30, 392); ctx.moveTo(-62, 160); ctx.lineTo(-80, 392); ctx.stroke();
  const keys = [[-40, 172, 9], [-41, 205, 10], [-42, 238, 10], [-46, 300, 11], [-47, 338, 11], [-49, 374, 11]];
  for (const [x, y, r] of keys) {
    ctx.fillStyle = '#d9a43a'; ctx.beginPath(); ctx.arc(x, y, r + 4, 0, TAU); ctx.fill();
    ctx.strokeStyle = line; ctx.lineWidth = 2.5; ctx.stroke();
    const pg = ctx.createRadialGradient(x - 3, y - 3, 1, x, y, r);
    pg.addColorStop(0, '#ffffff'); pg.addColorStop(1, '#d8d2c4');
    ctx.fillStyle = pg; ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
  }
  // specular streak (travels slowly — loop safe)
  ctx.save();
  ctx.beginPath(); capsule(ctx, -40, 126, -56, 404, 15, 34); ctx.clip();
  ctx.strokeStyle = 'rgba(255,255,240,0.75)'; ctx.lineWidth = 6;
  ctx.beginPath(); ctx.moveTo(-50, 140); ctx.lineTo(-70, 396); ctx.stroke();
  const gy = 140 + ((t * 40) % 300);
  const gl = ctx.createRadialGradient(-56, gy, 0, -56, gy, 50);
  gl.addColorStop(0, `rgba(255,255,255,${0.35 + glint * 0.5})`); gl.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = gl; ctx.fillRect(-110, gy - 50, 110, 100);
  ctx.restore();
  // ---- neck (crook)
  ctx.beginPath();
  ctx.moveTo(-6, 44); ctx.bezierCurveTo(-6, 88, -30, 100, -48, 132);
  ctx.lineTo(-30, 138); ctx.bezierCurveTo(-14, 108, 8, 92, 8, 44); ctx.closePath();
  ctx.fillStyle = goldGradient(ctx, -40, 0, 10, 0); ctx.fill(); ctx.strokeStyle = line; ctx.lineWidth = 5; ctx.stroke();
  // octave key
  ctx.fillStyle = '#e2ad3f'; ctx.beginPath(); ctx.ellipse(-12, 92, 7, 12, 0.5, 0, TAU); ctx.fill(); ctx.stroke();
  // ---- mouthpiece & ligature
  ctx.beginPath(); ctx.moveTo(-7, -4); ctx.lineTo(7, -4); ctx.lineTo(11, 48); ctx.lineTo(-11, 48); ctx.closePath();
  const mg = ctx.createLinearGradient(-11, 0, 11, 0);
  mg.addColorStop(0, '#3b3b44'); mg.addColorStop(0.4, '#0d0d10'); mg.addColorStop(1, '#000');
  ctx.fillStyle = mg; ctx.fill(); ctx.strokeStyle = '#000'; ctx.lineWidth = 3; ctx.stroke();
  ctx.fillStyle = '#c9ccd8'; ctx.fillRect(-12, 30, 24, 9);
}
