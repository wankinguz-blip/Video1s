// Frame compositor: renderFrame(t) draws one complete frame for one view (screen slice).
import { setPath2D } from './draw/svgops.js';
import { makeSprites } from './sprites.js';
import { drawSky, drawMoon, drawStars, drawLandscape, drawCrestGrass, drawFireflies, drawForeground } from './scene/environment.js';
import { LOOP } from './config.js';
import { mod } from './math.js';

/**
 * platform: { createCanvas(w, h), Path2D }
 * layout:   from layout.js;  view: { x0, x1 } world window;  pxW/pxH: output size in pixels
 */
export function createRenderer(platform, layout, view0, pxW, pxH, hooks = {}) {
  setPath2D(platform.Path2D);
  const k = pxH / layout.worldH;
  // the drawn window always spans the whole canvas (preview sizes are rounded to even pixels)
  const view = { ...view0, x1: view0.x0 + pxW / k };
  const sprites = makeSprites(platform.createCanvas);
  const world = (c) => c.setTransform(k, 0, 0, k, -view.x0 * k, 0);

  const land = platform.createCanvas(pxW, pxH);
  const lctx = land.getContext('2d');
  world(lctx);
  drawLandscape(lctx, layout, view);

  const canvas = platform.createCanvas(pxW, pxH);
  const ctx = canvas.getContext('2d');
  const env = { platform, layout, view, k, pxW, pxH, sprites, world };

  return {
    canvas,
    ctx,
    env,
    render(time) {
      const t = mod(time, LOOP);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = 'source-over';
      ctx.filter = 'none';
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, pxW, pxH); // no carry-over from the previous frame
      ctx.save(); // every frame starts from the identical default state (order-independent frames)
      world(ctx);
      drawSky(ctx, layout, view);
      drawMoon(ctx, layout);
      drawStars(ctx, layout, view, t, sprites);
      hooks.sky?.(ctx, t, env);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(land, 0, 0);
      world(ctx);
      drawCrestGrass(ctx, layout, view, t);
      hooks.stage?.(ctx, t, env);
      drawFireflies(ctx, layout, view, t, sprites);
      drawForeground(ctx, layout, view, t);
      hooks.post?.(ctx, t, env);
      ctx.restore();
      return canvas;
    },
  };
}
