"""Jazz saxophonist (right hill).  The saxophone is attached to the head (mouthpiece in the mouth),
both hands hold it in the instrument's own space, so they can never leave the keys; the body
plays the phrases of the score:

  0-2.5    rest      relaxed sway, left toe taps, glance up at the burst on 2
  2.5-4    phrase 1  rising line: dips, then leans back to the left, bell up on the top note (3.5)
  4-5.75   rest      groove, looks over to the word above the singer
  5.75-8   phrase 2  falling line: starts leaning back to the right, sinks into a deep crouch (7.5)
  8-10.5   rest      comes up, sways, glance up at the burst on 10
  10.5-12  phrase 3  arch with a side tilt on the long note (11)
  12-13.75 rest      wider stance for the solo
  13.75-16 solo      side-to-side swing of the horn, pumping knees
  16.75    long high note with vibrato: big lean back to the right, right heel up, eyes closed
  18       burst     drops forward, looks up
  18.5-20  stomps the right foot forward, crouch, second and bigger lean back to the left (19.25)
  21.75-24 funky run: bent forward, horn swings on every note, heel swivels, lowest crouch on 23,
           pops up on 23.5
  25       burst     glance up, short answer phrase 25.5-26.5
  28-30    last falling line that settles back into the opening sway
"""
import bisect
import math

import numpy as np

from .. import propspec, score
from ..mathx import TAU, clamp, smoothstep, wrap_delta
from ..rig import HEAD, Arm, Attach, Pose
from .common import (FLOW, HOLD, NB, Foot, Track, beats, blink, body_key, bounce, fit_hips, hit, pp, pwin, set_torso,
                     sway, to_leg)

NAME = "sax"
B = body_key

# ---------------------------------------------------------------- the score: fingering, accents
_NOTES = sorted((float(s["t"]), float(s["d"]), float(s["n"]), float(s["v"]), bool(s.get("vib"))) for s in score.SAX)
_NT = [n[0] for n in _NOTES]
_STEPS = [clamp((84.0 - n[2]) / 20.0) * 6.0 for n in _NOTES]


_PAT = [[clamp(st - k) for k in range(6)] for st in _STEPS]


def _fingering(b):
    """Six key values 0..1 (3 per hand) — the same mapping as score.sax_state, but eased over
    about two frames around every note start instead of stepping."""
    i = bisect.bisect_right(_NT, b - 0.07) - 1
    j = (i + 1) % len(_NT)
    w = smoothstep(-0.07, 0.07, wrap_delta(_NT[j], b, NB))
    pi, pj = _PAT[i], _PAT[j]
    return [pi[k] + (pj[k] - pi[k]) * w for k in range(6)]


def _pulse(b):
    """Accent 0..1 on every note start (quick attack, slower release)."""
    q = 1.0
    for t0, _d, _n, vel, _vib in _NOTES:
        u = wrap_delta(t0, b, NB)
        if -0.12 < u < 0.4:
            q *= 1.0 - vel * (smoothstep(-0.12, 0.0, u) if u <= 0.0 else 1.0 - smoothstep(0.0, 0.4, u))
    return 1.0 - q


def _vibrato(b):
    """Envelope 0..1 of the two long vibrato notes."""
    v = 0.0
    for t0, d, _n, _vel, vib in _NOTES:
        if vib:
            u = wrap_delta(t0, b, NB)
            v += smoothstep(0.12, 0.3, u) * (1.0 - smoothstep(d - 0.1, d + 0.05, u))
    return v


# ---------------------------------------------------------------- feet
BX, BY, BYAW = 0.15, -0.13, 0.30
FL = Foot("L", BX, BY, BYAW)
FR = Foot("R", -BX, BY, -BYAW)
FL.step(13.15, 13.6, x=0.22, yaw=0.38, lift=0.05)            # wider stance for the solo
FR.set(16.5, 16.75, heel=0.55)                               # lean back #1: right heel up
FR.set(17.55, 17.9, heel=0.0)
FR.step(18.15, 18.5, x=-0.22, y=-0.21, yaw=-0.42, lift=0.08, peak=0.4)   # stomp forward
FL.set(19.0, 19.25, heel=0.55)                               # lean back #2: left heel up
FL.set(20.0, 20.35, heel=0.0)
FR.step(20.7, 21.1, x=-BX, y=BY, yaw=-BYAW, lift=0.05)
FL.step(21.2, 21.6, x=BX, yaw=BYAW, lift=0.05)
FL.twist(21.78, 22.0, 0.62, pop=0.3).twist(22.28, 22.5, 0.02, pop=0.3).twist(22.78, 23.0, BYAW, pop=0.3)
FR.twist(21.78, 22.0, 0.02, pop=0.3).twist(22.28, 22.5, -0.62, pop=0.3).twist(22.78, 23.0, -BYAW, pop=0.3)
FL.close()
FR.close()

# ---------------------------------------------------------------- body
REST_R = B(-0.04, 0, -0.035, (0.0, -0.12, 0.05), (0.05, -0.05, -0.06), (0.05, -0.14, 0.0))
REST_L = B(0.035, 0, -0.035, (0.0, -0.03, -0.05), (0.05, 0.05, 0.06), (0.05, -0.02, 0.0))
UP_1 = B(-0.02, 0, -0.015, (-0.02, -0.10, 0.03), (-0.08, -0.14, 0.0), (-0.30, -0.36, 0.05))
UP_2 = B(0.02, 0, -0.015, (-0.03, -0.06, -0.03), (-0.10, -0.12, 0.03), (-0.32, -0.30, -0.04))
WORD = B(0.03, 0, -0.03, (0.0, -0.10, -0.04), (-0.03, -0.14, 0.04), (-0.22, -0.42, 0.0))

_BODY = Track(pp([
    (0.0, REST_R, dict(flow=True, snap=0.5)),
    (1.0, REST_L, FLOW),
    (2.0, UP_1, dict(dur=0.34, over=0.04)),
    (2.5, B(0.0, 0, -0.075, (0.04, 0.0, 0.0), (0.14, 0.0, 0.0), (0.10, 0.0, 0.0), (0.15, 0.15)), dict(dur=0.3, over=0.0)),
    (3.0, B(0.02, 0, -0.045, (0.0, 0.05, -0.03), (0.0, 0.10, 0.03), (0.0, 0.05, 0.0)), FLOW),
    (3.5, B(0.03, -0.03, -0.02, (-0.06, 0.12, -0.05), (-0.22, 0.20, 0.08), (-0.20, 0.15, 0.08)), dict(dur=0.26, over=0.06)),
    (4.0, B(0.03, -0.03, -0.02, (-0.06, 0.12, -0.05), (-0.22, 0.20, 0.08), (-0.20, 0.15, 0.08)), HOLD),
    (4.5, REST_L, dict(dur=0.42, over=0.0)),
    (5.0, WORD, dict(dur=0.36, over=0.03)),
    (5.75, B(-0.03, -0.03, -0.02, (-0.06, -0.12, 0.05), (-0.20, -0.20, -0.08), (-0.18, -0.15, -0.06)), dict(dur=0.3, over=0.05)),
    (6.5, B(-0.02, 0, -0.05, (0.02, -0.05, 0.03), (0.05, -0.08, -0.03), (0.03, -0.05, 0.0)), FLOW),
    (7.0, B(0.0, 0, -0.085, (0.06, 0.03, 0.0), (0.18, 0.05, 0.03), (0.10, 0.05, 0.0), (0.2, 0.2)), FLOW),
    (7.5, B(0.02, 0.02, -0.125, (0.10, 0.10, -0.03), (0.27, 0.14, 0.06), (0.14, 0.10, 0.03), (0.35, 0.35)), dict(dur=0.22, over=0.04)),
    (8.0, B(0.02, 0.02, -0.12, (0.10, 0.10, -0.03), (0.26, 0.14, 0.06), (0.14, 0.10, 0.03), (0.35, 0.35)), HOLD),
    (8.6, REST_R, dict(dur=0.5, over=0.0)),
    (9.3, REST_L, FLOW),
    (10.0, UP_2, dict(dur=0.34, over=0.04)),
    (10.5, B(-0.02, 0, -0.07, (0.03, -0.05, 0.0), (0.11, -0.08, 0.0), (0.08, -0.05, 0.0), (0.15, 0.15)), dict(dur=0.3, over=0.0)),
    (11.0, B(0.0, -0.02, -0.02, (-0.05, 0.0, -0.06), (-0.18, 0.05, 0.15), (-0.16, 0.0, 0.12)), dict(dur=0.26, over=0.06)),
    (11.5, B(0.0, 0, -0.05, (0.0, 0.0, 0.03), (0.04, 0.0, -0.05), (0.03, 0.0, -0.03)), FLOW),
    (12.0, B(-0.03, 0, -0.065, (0.03, -0.08, 0.04), (0.10, -0.08, -0.05), (0.06, -0.10, 0.0)), dict(dur=0.26, over=0.0)),
    (12.9, B(0.0, 0, -0.03, (0.0, -0.08, 0.0), (-0.02, -0.10, 0.0), (-0.16, -0.38, 0.0)), dict(dur=0.4, over=0.03)),
    (13.75, B(0.06, 0, -0.08, (0.04, 0.10, -0.04), (0.14, 0.14, 0.06), (0.08, 0.10, 0.04), (0.2, 0.2)), dict(dur=0.3, over=0.04)),
    (14.25, B(-0.02, 0, -0.065, (0.04, -0.10, 0.04), (0.12, -0.16, -0.07), (0.06, -0.10, -0.04), (0.2, 0.2)), FLOW),
    (14.75, B(0.08, 0, -0.08, (0.04, 0.10, -0.04), (0.13, 0.15, 0.07), (0.08, 0.10, 0.04), (0.2, 0.2)), FLOW),
    (15.25, B(-0.01, 0, -0.065, (0.04, -0.10, 0.04), (0.11, -0.16, -0.07), (0.06, -0.10, -0.04), (0.2, 0.2)), FLOW),
    (16.0, B(0.04, 0, -0.03, (-0.02, 0.0, 0.0), (-0.05, 0.0, 0.0), (-0.05, 0.0, 0.0)), FLOW),
    (16.5, B(0.04, 0, -0.085, (0.03, 0.08, 0.0), (0.09, 0.12, 0.0), (0.05, 0.08, 0.0), (0.15, 0.15)), dict(dur=0.22, over=0.0)),
    (16.75, B(0.02, -0.05, -0.03, (-0.10, -0.18, 0.05), (-0.30, -0.26, -0.10), (-0.20, -0.18, -0.08), (0.1, 0.3)), dict(dur=0.22, over=0.07)),
    (17.5, B(0.02, -0.055, -0.03, (-0.11, -0.18, 0.05), (-0.31, -0.26, -0.10), (-0.22, -0.18, -0.08), (0.1, 0.3)), HOLD),
    (18.0, B(0.02, 0, -0.075, (0.03, 0.0, 0.0), (0.10, 0.0, 0.0), (-0.20, -0.32, 0.0), (0.1, 0.1)), dict(dur=0.4, over=0.0)),
    (18.5, B(-0.05, -0.04, -0.105, (0.08, -0.12, 0.05), (0.22, -0.15, -0.06), (0.10, -0.10, -0.03), (0.3, 0.3)), dict(dur=0.3, over=0.04)),
    (19.0, B(-0.01, -0.03, -0.07, (0.02, 0.0, 0.0), (0.05, 0.05, 0.0), (0.03, 0.05, 0.0), (0.15, 0.15)), FLOW),
    (19.25, B(0.05, -0.07, -0.02, (-0.10, 0.20, -0.06), (-0.32, 0.28, 0.12), (-0.22, 0.20, 0.10), (0.3, 0.1)), dict(dur=0.2, over=0.07)),
    (19.95, B(0.05, -0.075, -0.02, (-0.11, 0.20, -0.06), (-0.33, 0.28, 0.12), (-0.24, 0.20, 0.10), (0.3, 0.1)), HOLD),
    (20.5, B(0.04, -0.03, -0.045, (0.0, -0.03, -0.05), (0.05, 0.05, 0.06), (0.05, -0.02, 0.0)), dict(dur=0.45, over=0.0)),
    (21.3, REST_R, FLOW),
    (21.75, B(0.03, 0, -0.095, (0.06, 0.12, -0.04), (0.20, 0.16, 0.06), (0.08, 0.10, 0.04), (0.3, 0.3)), dict(dur=0.26, over=0.03)),
    (22.0, B(-0.03, 0, -0.105, (0.06, -0.12, 0.04), (0.20, -0.18, -0.07), (0.08, -0.10, -0.04), (0.3, 0.3)), dict(dur=0.2, over=0.0)),
    (22.5, B(0.03, 0, -0.10, (0.06, 0.12, -0.04), (0.21, 0.16, 0.06), (0.08, 0.10, 0.04), (0.3, 0.3)), dict(dur=0.2, over=0.0)),
    (22.75, B(0.0, 0, -0.09, (0.07, 0.0, 0.0), (0.22, -0.04, 0.0), (0.09, 0.0, 0.0), (0.3, 0.3)), FLOW),
    (23.0, B(-0.01, 0.02, -0.14, (0.10, -0.04, 0.0), (0.29, -0.06, -0.02), (0.12, -0.03, 0.0), (0.4, 0.4)), dict(dur=0.18, over=0.0)),
    (23.5, B(0.0, -0.03, -0.025, (-0.05, 0.0, 0.0), (-0.17, 0.0, 0.0), (-0.14, 0.0, 0.0)), dict(dur=0.24, over=0.07)),
    (24.0, B(0.0, -0.03, -0.025, (-0.05, 0.0, 0.0), (-0.16, 0.0, 0.0), (-0.14, 0.0, 0.0)), HOLD),
    (24.5, REST_R, dict(dur=0.4, over=0.0)),
    (25.0, UP_1, dict(dur=0.32, over=0.04)),
    (25.5, B(-0.01, 0, -0.07, (0.03, 0.0, 0.0), (0.11, 0.0, 0.0), (0.08, 0.0, 0.0), (0.15, 0.15)), dict(dur=0.3, over=0.0)),
    (25.75, B(0.01, -0.02, -0.03, (-0.03, 0.05, -0.03), (-0.12, 0.08, 0.05), (-0.10, 0.05, 0.04)), FLOW),
    (26.0, B(0.02, 0, -0.05, (0.0, 0.0, -0.03), (0.02, 0.05, 0.04), (0.02, 0.0, 0.0)), FLOW),
    (26.6, REST_L, FLOW),
    (27.3, WORD, dict(dur=0.4, over=0.03)),
    (28.0, B(0.02, 0, -0.03, (-0.02, 0.08, 0.0), (-0.08, 0.10, 0.04), (-0.06, 0.08, 0.03)), dict(dur=0.3, over=0.04)),
    (28.5, B(0.0, 0, -0.06, (0.02, 0.0, 0.0), (0.08, 0.0, 0.0), (0.05, 0.0, 0.0)), FLOW),
    (29.0, B(-0.02, 0, -0.10, (0.06, -0.06, 0.03), (0.20, -0.10, -0.04), (0.10, -0.08, 0.0), (0.25, 0.25)), dict(dur=0.22, over=0.0)),
    (29.5, B(-0.035, 0, -0.055, (0.02, -0.10, 0.04), (0.09, -0.08, -0.05), (0.06, -0.12, 0.0)), FLOW),
]))

# saxophone attitude relative to the CHEST: (bend, twist, lean); bend < 0.34 swings the bell
# forward / up, lean > 0 carries the bottom of the horn to the player's right
S_REST = (0.34, 0.0, 0.20)
_SAX = Track(pp([
    (0.0, S_REST, dict(flow=True, snap=0.5)),
    (2.5, (0.30, 0.0, 0.20), dict(dur=0.3)),
    (3.5, (-0.04, 0.0, 0.12), dict(dur=0.3)),
    (4.0, (-0.04, 0.0, 0.12), HOLD),
    (4.5, S_REST, dict(dur=0.4)),
    (5.75, (-0.04, 0.0, 0.26), dict(dur=0.3)),
    (6.5, (0.10, 0.0, 0.22), FLOW),
    (7.5, (0.30, 0.0, 0.12), dict(dur=0.3)),
    (8.6, S_REST, dict(dur=0.5)),
    (10.5, (0.26, 0.0, 0.20), dict(dur=0.3)),
    (11.0, (0.0, 0.0, 0.30), dict(dur=0.26)),
    (12.0, S_REST, dict(dur=0.4)),
    (13.75, (0.24, 0.0, 0.32), dict(dur=0.3)),
    (14.25, (0.24, 0.0, 0.06), FLOW),
    (14.75, (0.22, 0.0, 0.32), FLOW),
    (15.25, (0.22, 0.0, 0.06), FLOW),
    (16.0, (0.14, 0.0, 0.20), FLOW),
    (16.5, (0.28, 0.0, 0.20), dict(dur=0.22)),
    (16.75, (-0.07, 0.0, 0.28), dict(dur=0.22)),
    (17.5, (-0.07, 0.0, 0.28), HOLD),
    (18.0, S_REST, dict(dur=0.4)),
    (18.5, (0.28, 0.0, 0.26), dict(dur=0.3)),
    (19.25, (-0.08, 0.0, 0.10), dict(dur=0.22)),
    (19.95, (-0.08, 0.0, 0.10), HOLD),
    (20.5, S_REST, dict(dur=0.45)),
    (21.75, (0.26, 0.0, 0.32), dict(dur=0.26)),
    (22.0, (0.26, 0.0, 0.06), dict(dur=0.2)),
    (22.5, (0.26, 0.0, 0.32), dict(dur=0.2)),
    (23.0, (0.30, 0.0, 0.18), dict(dur=0.2)),
    (23.5, (0.0, 0.0, 0.20), dict(dur=0.24)),
    (24.0, (0.0, 0.0, 0.20), HOLD),
    (24.5, S_REST, dict(dur=0.4)),
    (25.5, (0.28, 0.0, 0.20), dict(dur=0.3)),
    (25.75, (0.08, 0.0, 0.22), FLOW),
    (26.5, S_REST, dict(dur=0.4)),
    (28.0, (0.12, 0.0, 0.24), dict(dur=0.3)),
    (29.0, (0.30, 0.0, 0.18), dict(dur=0.3)),
], antic=0.0, over=0.05))

_AMP = Track([(0.0, 0.8), (1.6, 0.8), (2.0, 0.2), (2.6, 0.6), (3.3, 0.2), (4.2, 0.2), (4.6, 0.8), (5.5, 0.4),
              (7.3, 0.5), (8.2, 0.3), (8.8, 0.8), (9.7, 0.6), (10.0, 0.2), (12.2, 0.7), (13.6, 1.0), (16.0, 1.0),
              (16.5, 0.0), (17.6, 0.0), (18.6, 0.6), (19.1, 0.0), (20.1, 0.0), (20.6, 0.8), (21.6, 1.0), (23.3, 0.5),
              (24.1, 0.2), (24.6, 0.8), (26.6, 0.8), (27.2, 0.4), (28.0, 0.6), (29.5, 0.8)], snap=1.0)
# elbows lift out on the big notes
_ELBOW = Track([(0.0, 0.0), (3.2, 0.0), (3.5, 0.8), (4.0, 0.8), (4.5, 0.0), (5.5, 0.0), (5.75, 0.8), (6.5, 0.2),
                (10.7, 0.0), (11.0, 0.7), (11.5, 0.0), (16.5, 0.0), (16.75, 1.0), (17.5, 1.0), (18.0, 0.0),
                (19.0, 0.0), (19.25, 1.0), (19.95, 1.0), (20.5, 0.0), (23.3, 0.0), (23.5, 0.8), (24.0, 0.8),
                (24.5, 0.0)], snap=1.0)

_CLEAR = 0.10                                # horn swings forward by this much (rad) when the head is turned
_MOUTH = np.array(HEAD["mouth"])
_UG = np.array(propspec.SAX["upper_grip"])
_LG = np.array(propspec.SAX["lower_grip"])
_BLINKS = (1.4, 4.4, 8.9, 12.3, 20.6, 24.4, 26.8)
_LOOK = Track(pp([(0.0, (0.0, 0.0)), (2.0, (-0.5, 0.6)), (2.6, (0.0, -0.2)), (5.0, (-0.6, 0.5)), (5.8, (0.0, 0.0)),
                  (10.0, (-0.4, 0.6)), (10.6, (0.0, -0.2)), (12.9, (-0.6, 0.5)), (13.75, (0.0, -0.2)),
                  (18.0, (-0.5, 0.6)), (18.6, (0.0, -0.2)), (20.6, (0.0, 0.0)), (25.0, (-0.5, 0.6)), (25.6, (0.0, 0.0)),
                  (27.3, (-0.6, 0.5)), (28.1, (0.0, 0.0))], dur=0.3, antic=0.0, over=0.0))


def pose(t):
    b = beats(t) % NB
    p = Pose()
    st = score.sax_state(t)
    blow, long_note = st["blow"], st["long_note"]
    pulse = _pulse(b)
    vib = _vibrato(b)
    shake = vib * math.sin(TAU * 3.0 * b)                    # 6 Hz, a whole number of cycles per loop

    # ---- body: hips lead, rib cage and head follow
    v0, v1, v2 = _BODY(b), _BODY(b - 0.07), _BODY(b - 0.14)
    amp = _AMP(b)
    bn = bounce(b)
    sw = sway(b, 2.0, 0.25)
    hips_rot = v0[3:6] + np.array([0.015 * amp * bn, 0.0, 0.02 * amp * sw])
    body_rot = v1[6:9] + np.array([0.03 * amp * bounce(b - 0.08) + 0.03 * pulse * blow, 0.02 * amp * sw, -0.03 * amp * sw])
    head_rot = v2[9:12] + np.array([0.07 * amp * bounce(b - 0.15, sharp=1.2) - 0.03 * amp + 0.04 * pulse * blow + 0.012 * shake,
                                    0.0, 0.02 * amp * sway(b - 0.15, 2.0, 0.25) + 0.012 * shake])
    p.hips_pos = np.array([v0[0] + 0.01 * amp * sw, v0[1], v0[2] - 0.035 * amp * bn - 0.012 * pulse * blow])
    set_torso(p, hips_rot, body_rot, head_rot)

    # ---- feet: the left toe taps the beat during the rests
    tap = (pwin(b, 29.9, 32.3, 0.3, 0.3) + pwin(b, 4.3, 5.6, 0.3, 0.3) + pwin(b, 8.5, 10.4, 0.3, 0.3)
           + pwin(b, 12.1, 13.05, 0.2, 0.2) + pwin(b, 24.2, 25.4, 0.3, 0.3) + pwin(b, 26.5, 27.9, 0.3, 0.3)) \
        * 0.36 * (1.0 - bounce(b, sharp=0.9, skew=0.3))
    p.leg_L = to_leg("L", FL(b), knee_out=v0[12], toe=tap)
    p.leg_R = to_leg("R", FR(b), knee_out=v0[13])
    fit_hips(p)

    # ---- the saxophone: authored relative to the chest, attached to the head
    s = _SAX(b)
    hb = p.neck_rot[0] + p.head_rot[0]                       # head relative to the chest (small angles)
    ht = p.neck_rot[1] + p.head_rot[1]
    hl = p.neck_rot[2] + p.head_rot[2]
    # a turned head carries the mouthpiece sideways and back towards the rib cage: a horn that
    # hangs low swings forward a little so the forearms stay clear of the chest / belly
    clear = _CLEAR * smoothstep(0.08, 0.30, abs(ht)) * smoothstep(0.0, 0.20, s[0])
    rot = (s[0] - hb - clear - 0.03 * pulse * blow + 0.02 * shake, s[1] - 0.75 * ht, s[2] - 0.75 * hl + 0.012 * shake)
    p.props["sax"] = Attach("head", pos=(_MOUTH[0], _MOUTH[1] - 0.01, _MOUTH[2]), rot=rot)

    # ---- hands on the keys (instrument space), fingers follow the notes
    fg = _fingering(b)
    trill = 0.05 * blow * math.sin(TAU * 2.0 * b)
    el = _ELBOW(b)
    cl = [0.55] + [0.36 + 0.30 * fg[k] + trill * (1 if k % 2 else -1) for k in range(3)]
    cr = [0.55] + [0.36 + 0.30 * fg[3 + k] - trill * (1 if k % 2 else -1) for k in range(3)]
    p.arm_L = Arm(hand=_UG + np.array([0.048, 0.052, 0.0]), pole=(0.8 + 0.3 * el, 0.5, -0.55 + 0.5 * el), space="prop:sax",
                  fingers=(-0.12, -1.0, 0.05), palm=(-1.0, 0.0, 0.0), curl=cl, shrug=0.015 * el, forward=0.02)
    p.arm_R = Arm(hand=_LG + np.array([-0.058, 0.052, 0.0]), pole=(-0.8 - 0.3 * el, 0.5, -0.55 + 0.5 * el), space="prop:sax",
                  fingers=(0.12, -1.0, 0.05), palm=(1.0, 0.0, 0.0), curl=cr, shrug=0.015 * el, forward=0.03)

    # ---- face
    p.cheeks = clamp(blow * (0.75 + 0.25 * pulse))
    p.smile = 0.55 * (1.0 - blow)
    eyes = max(blink(b, _BLINKS), 0.92 * long_note, 0.5 * blow * pulse)
    p.blink_L = p.blink_R = clamp(eyes)
    lk = _LOOK(b)
    p.look = (float(lk[0]), float(lk[1]))
    br = 0.75 * long_note + 0.5 * hit(b, 2.0, 0.3, 0.6) + 0.5 * hit(b, 10.0, 0.3, 0.6) + 0.5 * hit(b, 18.0, 0.3, 0.6) \
        + 0.5 * hit(b, 25.0, 0.3, 0.6) + 0.25 * blow * pulse - 0.15 * blow
    p.brow_L = clamp(br, -1.0, 1.0)
    p.brow_R = clamp(br * 0.9, -1.0, 1.0)
    return p
