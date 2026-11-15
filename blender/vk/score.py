"""Animation-side readers of the musical score (same data the soundtrack is synthesised from):
lip-sync for the singer, fingering / blowing for the saxophone, beat pulses for accents.
Times in the score are in BEATS (120 BPM → 1 beat = 0.5 s, 30 beats per loop)."""
import json
import math
import os

from .config import BEAT, LOOP_BEATS
from .mathx import clamp, mod, smoothstep, wrap_delta

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
with open(os.path.join(_DATA, "score.json"), "r", encoding="utf-8") as _f:
    SCORE = json.load(_f)

VOCAL = SCORE["vocal"]      # [{t, d, n, text, vowel, onset?, coda?, slideTo?, v}]
SAX = SCORE["sax"]          # [{t, d, n, v, bend?, vib?}]
DRUMS = SCORE["drums"]      # [{t, inst, v}]
BURST_BEATS = SCORE["burstBeats"]     # [2, 10, 18, 25]
LAUNCH_BEATS = SCORE["launchBeats"]   # [0, 8, 16, 24]
CRASH_BEATS = [d["t"] for d in DRUMS if d["inst"] == "crash"]

_OPEN = {"a": 1.0, "o": 0.78, "e": 0.62, "i": 0.42, "u": 0.5}
_SHAPE = {"a": "a", "o": "o", "e": "e", "i": "e", "u": "o"}


def beat_of(t):
    """Loop time in seconds → beat position in [0, 30)."""
    return mod(t / BEAT, LOOP_BEATS)


def vocal_state(t):
    """Singer mouth state at time t (s): dict(singing 0..1, open 0..1, shape 'a'|'o'|'e'|'smile',
    note (syllable dict or None), progress 0..1)."""
    b = beat_of(t)
    best = None
    for s in VOCAL:
        u = wrap_delta(s["t"], b, LOOP_BEATS)
        if -0.08 < u < s["d"] + 0.12:
            best = (s, u)
            break
    if best is None:
        return dict(singing=0.0, open=0.0, shape="smile", note=None, progress=0.0)
    s, u = best
    sec, dur = u * BEAT, s["d"] * BEAT
    env = smoothstep(-0.03, 0.035, sec) * (1 - smoothstep(dur - 0.02, dur + 0.06, sec))
    onset = s.get("onset")
    if onset in ("v", "w"):
        env *= 0.35 + 0.65 * smoothstep(0.0, 0.07, sec)
    if onset == "k":
        env *= 0.5 + 0.5 * smoothstep(0.01, 0.06, sec)
    if s.get("coda"):
        env *= 1 - 0.55 * smoothstep(dur * 0.62, dur * 0.85, sec)
    vib = 1 + 0.06 * math.sin(sec * 2 * math.pi * 5.5) * smoothstep(0.15, 0.3, sec)
    return dict(singing=env, open=clamp(env * _OPEN[s["vowel"]] * vib), shape=_SHAPE[s["vowel"]],
                note=s, progress=sec / dur)


def sax_state(t):
    """Saxophone state at time t (s): dict(blow 0..1, playing bool, note midi, fingers [6 × 0..1],
    accent 0..1, long_note 0..1, vib 0|1)."""
    b = beat_of(t)
    cur = nxt = None
    for s in SAX:
        u = wrap_delta(s["t"], b, LOOP_BEATS)
        if -0.02 <= u < s["d"] + 0.02:
            cur = (s, u)
        if -0.5 < u < 0:
            nxt = s
    not_blow = not_long = 1.0
    for s in SAX:
        u = wrap_delta(s["t"], b, LOOP_BEATS)
        not_blow *= 1 - smoothstep(-0.12, 0.0, u) * (1 - smoothstep(s["d"], s["d"] + 0.35, u))
        if s["d"] >= 0.45:
            not_long *= 1 - smoothstep(0.1, 0.4, u) * (1 - smoothstep(s["d"] - 0.05, s["d"] + 0.3, u))
    note = cur[0]["n"] if cur else nxt["n"] if nxt else 64
    steps = clamp((84 - note) / 20.0) * 6
    fingers = [clamp(steps - i) for i in range(6)]
    return dict(blow=1 - not_blow, playing=cur is not None, note=note, fingers=fingers,
                accent=cur[0]["v"] if cur else 0.0, long_note=1 - not_long,
                vib=1 if (cur and cur[0].get("vib")) else 0)


def kick_pulse(t):
    """Decaying pulse 1 → 0 restarting on every beat (kick drum)."""
    b = beat_of(t)
    return math.exp(-(b - math.floor(b)) * 6)


def clap_pulse(t):
    """Decaying pulse restarting on every off-beat (claps on beats 1, 3, 5 …)."""
    b = beat_of(t) - 1.0
    f = mod(b, 2.0)
    return math.exp(-f * 5)
