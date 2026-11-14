// Cached radial-gradient sprites used for glows (stars, sparks, fireflies, text particles).
export function makeSprites(createCanvas) {
  const cache = new Map();
  const make = (key, size, stops) => {
    let c = cache.get(key);
    if (c) return c;
    c = createCanvas(size, size);
    const x = c.getContext('2d');
    const g = x.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    for (const [o, col] of stops) g.addColorStop(o, col);
    x.fillStyle = g;
    x.fillRect(0, 0, size, size);
    cache.set(key, c);
    return c;
  };
  const rgb = (c, a) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;
  const q = (c) => c.map((v) => Math.round(v / 8) * 8);
  return {
    /** Soft glow falling off to transparent. */
    glow(col) {
      const c = q(col);
      return make(`g${c}`, 64, [[0, rgb(c, 1)], [0.18, rgb(c, 0.55)], [0.45, rgb(c, 0.14)], [1, rgb(c, 0)]]);
    },
    /** Bright spark: white-hot centre with a coloured fringe. */
    spark(col) {
      const c = q(col);
      return make(`s${c}`, 32, [[0, 'rgba(255,255,255,1)'], [0.25, rgb(c.map((v) => v * 0.5 + 128), 0.9)], [0.55, rgb(c, 0.35)], [1, rgb(c, 0)]]);
    },
  };
}
