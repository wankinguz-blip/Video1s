// The complete show: environment + fireworks + characters, wired into the frame compositor.
import { createRenderer } from './render.js';
import { singerPose, drawSinger } from './characters/singer.js';
import { saxPose, drawSax } from './characters/sax.js';
import { emoPose, drawEmo } from './characters/emo.js';
import { drawContactShadow } from './characters/rig.js';
import { hillY, drawGroundLight } from './scene/environment.js';
import { createFireworks } from './fx/fireworks.js';

export const CAST = [
  { key: 'emo', hill: 'left', pose: emoPose, draw: drawEmo },
  { key: 'singer', hill: 'center', pose: singerPose, draw: drawSinger },
  { key: 'sax', hill: 'right', pose: saxPose, draw: drawSax },
];
// character-space bounding box that contains every pose (hats, raised arms, sax bell)
const BOX = { x0: -600, y0: -1200, x1: 600, y1: 70 };

export function characterPlacement(L, member) {
  const hill = L.hills.find((h) => h.role === member.hill);
  const x = L.chars[member.key].x;
  return { x, y: hillY(hill, x) - 1.5, s: L.charScale };
}

function makeLayers(platform, sk) {
  const w = Math.ceil((BOX.x1 - BOX.x0) * sk) + 2, h = Math.ceil((BOX.y1 - BOX.y0) * sk) + 2;
  const mk = () => { const c = platform.createCanvas(w, h); return { c, x: c.getContext('2d') }; };
  return { w, h, main: mk(), light: mk() };
}

/**
 * Draw one performer into its own layer, light it (night grade, moon rim, firework flashes),
 * then composite onto the frame at a pixel-snapped position.
 */
function drawPerformer(ctx, env, member, t, lights, layers) {
  const { layout: L, view, k } = env;
  const pl = characterPlacement(L, member);
  const sk = pl.s * k;
  const wx0 = pl.x + BOX.x0 * pl.s, wx1 = pl.x + BOX.x1 * pl.s;
  if (wx1 < view.x0 || wx0 > view.x1) return;
  const pose = member.pose(t);
  // contact shadow on the hill
  ctx.save();
  ctx.translate(pl.x, pl.y);
  ctx.scale(pl.s, pl.s);
  drawContactShadow(ctx, pose);
  ctx.restore();

  const ox = (wx0 - view.x0) * k, oy = (pl.y + BOX.y0 * pl.s) * k;
  const oxi = Math.floor(ox), oyi = Math.floor(oy);
  const { main, light, w, h } = layers;
  const lc = main.x;
  lc.setTransform(1, 0, 0, 1, 0, 0);
  lc.globalCompositeOperation = 'source-over';
  lc.globalAlpha = 1;
  lc.clearRect(0, 0, w, h);
  lc.save();
  lc.setTransform(sk, 0, 0, sk, ox - oxi - BOX.x0 * sk, oy - oyi - BOX.y0 * sk);
  member.draw(lc, pose, t);
  lc.restore();

  // ---- lighting, restricted to the character's pixels
  lc.setTransform(1, 0, 0, 1, 0, 0);
  lc.globalCompositeOperation = 'source-atop';
  lc.fillStyle = 'rgba(16,22,70,0.17)'; // night grade
  lc.fillRect(0, 0, w, h);
  const mg = lc.createLinearGradient(0, 0, w * 0.7, h * 0.55);
  mg.addColorStop(0, 'rgba(175,200,255,0.20)'); // cool moonlight from the upper left
  mg.addColorStop(1, 'rgba(175,200,255,0)');
  lc.fillStyle = mg;
  lc.fillRect(0, 0, w, h);
  if (lights.length) {
    const ll = light.x;
    ll.setTransform(1, 0, 0, 1, 0, 0);
    ll.globalCompositeOperation = 'source-over';
    ll.clearRect(0, 0, w, h);
    ll.globalCompositeOperation = 'lighter';
    for (const L0 of lights) {
      if (L0.i < 0.003) continue;
      const lx = (L0.x - view.x0) * k - oxi, ly = L0.y * k - oyi;
      const r = L0.r * k;
      const g = ll.createRadialGradient(lx, ly, 0, lx, ly, r);
      const [cr, cg, cb] = L0.c;
      g.addColorStop(0, `rgba(${cr},${cg},${cb},${0.17 * Math.min(1, L0.i)})`);
      g.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
      ll.fillStyle = g;
      ll.fillRect(0, 0, w, h);
    }
    ll.globalCompositeOperation = 'destination-in';
    ll.drawImage(main.c, 0, 0);
    lc.globalCompositeOperation = 'lighter';
    lc.drawImage(light.c, 0, 0);
  }
  lc.globalCompositeOperation = 'source-over';
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(main.c, oxi, oyi);
  ctx.restore();
}

export function createShow(platform, layout, view, pxW, pxH, opts = {}) {
  const k = pxH / layout.worldH;
  const layers = makeLayers(platform, layout.charScale * k);
  const fx = opts.fireworks === false ? null : createFireworks(layout, platform);
  return createRenderer(platform, layout, view, pxW, pxH, {
    sky: (ctx, t, env) => fx?.drawSky(ctx, t, env),
    stage: (ctx, t, env) => {
      const lights = fx ? fx.lights(t, env) : [];
      drawGroundLight(ctx, layout, env.view, lights);
      for (const m of CAST) drawPerformer(ctx, env, m, t, lights, layers);
    },
    post: (ctx, t, env) => fx?.drawPost?.(ctx, t, env),
  });
}
