// Animation-side readers of the score: lip-sync for the singer, fingering/puff for the sax.
import { BEAT, LOOP_BEATS } from '../config.js';
import { VOCAL, SAX, DRUMS } from './score.js';
import { clamp, mod, smoothstep, wrapDelta } from '../math.js';

const OPEN = { a: 1.0, o: 0.78, e: 0.62, i: 0.42, u: 0.5 };
const SHAPE = { a: 'a', o: 'o', e: 'e', i: 'e', u: 'o' };

/** Singer mouth state at time t (seconds). */
export function vocalState(t) {
  const b = mod(t / BEAT, LOOP_BEATS);
  let best = null;
  for (const s of VOCAL) {
    const u = wrapDelta(s.t, b, LOOP_BEATS);
    if (u > -0.08 && u < s.d + 0.12) { best = { s, u }; break; }
  }
  if (!best) return { singing: 0, open: 0, shape: 'smile', note: null };
  const { s, u } = best;
  const sec = u * BEAT, dur = s.d * BEAT;
  // attack/release in seconds, consonant closures at onset ('v' lips, 'k' stop) and nasal coda
  let env = smoothstep(-0.03, 0.035, sec) * (1 - smoothstep(dur - 0.02, dur + 0.06, sec));
  if (s.onset === 'v' || s.onset === 'w') env *= 0.35 + 0.65 * smoothstep(0.0, 0.07, sec);
  if (s.onset === 'k') env *= 0.5 + 0.5 * smoothstep(0.01, 0.06, sec);
  if (s.coda) env *= 1 - 0.55 * smoothstep(dur * 0.62, dur * 0.85, sec);
  const vib = 1 + 0.06 * Math.sin(sec * 2 * Math.PI * 5.5) * smoothstep(0.15, 0.3, sec);
  return { singing: env, open: clamp(env * OPEN[s.vowel] * vib), shape: SHAPE[s.vowel], note: s, progress: sec / dur };
}

/** Saxophone state: how hard it is blowing, which fingers are down, current pitch. */
export function saxState(t) {
  const b = mod(t / BEAT, LOOP_BEATS);
  let cur = null, next = null;
  for (const s of SAX) {
    const u = wrapDelta(s.t, b, LOOP_BEATS);
    if (u >= -0.02 && u < s.d + 0.02) cur = { s, u };
    if (u < 0 && u > -0.5) next = s;
  }
  // phrase activity (smooth union of per-note envelopes → no kinks): blowing pressure persists
  // across short gaps inside a lick; longNote rises during sustained notes and eases out after.
  let notBlow = 1, notLong = 1;
  for (const s of SAX) {
    const u = wrapDelta(s.t, b, LOOP_BEATS);
    notBlow *= 1 - smoothstep(-0.12, 0.0, u) * (1 - smoothstep(s.d, s.d + 0.35, u));
    if (s.d >= 0.45) notLong *= 1 - smoothstep(0.1, 0.4, u) * (1 - smoothstep(s.d - 0.05, s.d + 0.3, u));
  }
  const blow = 1 - notBlow;
  const note = cur ? cur.s.n : next ? next.n : 64;
  // 3 fingers per paw; a simple, consistent mapping pitch → key pattern (lower notes = more keys)
  const steps = clamp((84 - note) / 20) * 6;
  const fingers = [0, 1, 2, 3, 4, 5].map((i) => clamp(steps - i));
  const longNote = 1 - notLong;
  return { blow, playing: !!cur, note, fingers, accent: cur ? cur.s.v : 0, longNote, vib: cur?.s.vib ? 1 : 0 };
}

/** Smooth "beat pulse" 0..1 peaking on each kick, useful for accents. */
export function kickPulse(t) {
  const b = mod(t / BEAT, LOOP_BEATS);
  const f = b - Math.floor(b);
  return Math.exp(-f * 6);
}

export const crashBeats = DRUMS.filter((d) => d.inst === 'crash').map((d) => d.t);
