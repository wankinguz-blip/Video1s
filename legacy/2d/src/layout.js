// Scene layout in world units (screen height = 1000). Two modes:
//  * standalone: a complete composition for one screen of a given aspect ratio
//  * spanned:    one continuous panorama across Left | Center | Right, sliced per screen
import { SCREENS, SCREEN_ORDER, WORLD_H } from './config.js';
import { clamp } from './math.js';
import { TEXT_WIDTH } from './assets/vanking-text.js';

const CHAR_SCALE = 0.335; // brand-SVG units → world units (character ≈ 360 units tall)

function baseLayout(worldW, xs, opts = {}) {
  const [xl, xc, xr] = xs;
  const spread = Math.min(xc - xl, xr - xc);
  const hills = [
    { role: 'left', x: xl, top: 772, halfW: clamp(spread * 0.62, 620, 1500), base: 1230 },
    { role: 'right', x: xr, top: 768, halfW: clamp(spread * 0.64, 640, 1520), base: 1230 },
    { role: 'center', x: xc, top: 732, halfW: clamp(spread * 0.72, 760, 1650), base: 1250 },
  ];
  const textH = opts.textH ?? 205;
  const text = { cx: opts.textX ?? xc, cy: 215, h: textH };
  // one burst centred over each third of the word → short, radial paths into the letters
  const burstSpread = (TEXT_WIDTH * textH) / 3;
  const bursts = [
    { x: text.cx - burstSpread, y: 205, launchX: xl + (text.cx - burstSpread - xl) * 0.55 },
    { x: text.cx, y: 178, launchX: xc + 300 },
    { x: text.cx + burstSpread, y: 210, launchX: xr - (xr - text.cx - burstSpread) * 0.55 },
  ];
  return {
    worldW,
    worldH: WORLD_H,
    charScale: CHAR_SCALE,
    hills,
    chars: { emo: { x: xl }, singer: { x: xc }, sax: { x: xr } },
    text,
    bursts,
    moon: opts.moon ?? { x: Math.max(230, xl - spread * 0.42), y: 150, r: 46 },
    faceSafeY: 372, // non-text sparks fade out below this line so faces are never covered
  };
}

/** Full scene composed for a single screen. */
export function standaloneLayout(screen) {
  const worldW = (WORLD_H * screen.w) / screen.h;
  const cx = worldW / 2;
  const spread = clamp(worldW * 0.3, 950, 1750);
  const L = baseLayout(worldW, [cx - spread, cx, cx + spread]);
  return { ...L, mode: 'standalone', views: [{ name: screen.name, x0: 0, x1: worldW, px: screen }] };
}

/** One panorama across the three screens (left | center | right), each screen a slice. */
export function spannedLayout() {
  const widths = SCREEN_ORDER.map((k) => (WORLD_H * SCREENS[k].w) / SCREENS[k].h);
  const worldW = widths.reduce((a, b) => a + b, 0);
  const edges = [0, widths[0], widths[0] + widths[1], worldW];
  const centers = [0, 1, 2].map((i) => (edges[i] + edges[i + 1]) / 2);
  const L = baseLayout(worldW, centers, {
    textX: centers[1],
    moon: { x: centers[0] - 900, y: 150, r: 48 },
  });
  const views = SCREEN_ORDER.map((k, i) => ({ name: k, x0: edges[i], x1: edges[i + 1], px: SCREENS[k] }));
  return { ...L, mode: 'spanned', views };
}

export function layoutFor(mode, screenName) {
  return mode === 'spanned' ? spannedLayout() : standaloneLayout(SCREENS[screenName]);
}
