// Render still frames for review.
//   node scripts/still.mjs --screen left|center|right [--mode standalone|spanned] [--scale 0.25] [--t 2.5,6] [--out out/stills]
import { createCanvas, Path2D } from '@napi-rs/canvas';
import { mkdirSync, writeFileSync } from 'node:fs';
import { layoutFor } from '../src/layout.js';
import { SCREENS } from '../src/config.js';
import { createShow } from '../src/show.js';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const mode = arg('mode', 'standalone');
const screens = arg('screen', 'left,center,right').split(',');
const scale = Number(arg('scale', '0.25'));
const times = arg('t', '0,2.5').split(',').map(Number);
const outDir = arg('out', 'out/stills');
mkdirSync(outDir, { recursive: true });

for (const name of screens) {
  const L = layoutFor(mode, name);
  const view = L.views.find((v) => v.name === name) || L.views[0];
  const W = Math.round(SCREENS[name].w * scale), H = Math.round(SCREENS[name].h * scale);
  const r = createShow({ createCanvas, Path2D }, L, view, W, H);
  for (const t of times) {
    const t0 = Date.now();
    const c = r.render(t);
    const file = `${outDir}/${mode}-${name}-t${t.toFixed(2)}.png`;
    writeFileSync(file, c.toBuffer('image/png'));
    console.log(file, `${W}x${H}`, `${Date.now() - t0} ms`);
  }
}
