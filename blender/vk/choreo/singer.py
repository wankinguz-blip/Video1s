"""Lead singer (centre hill): a showman routine in the spirit of 80s pop — microphone in the right
hand, the white glove of the left hand does the talking.

Everything is keyed in BEATS (120 BPM, 30 beats per loop) against the score:

  0-2    "Oh, oh, ohh"   groove on the right leg, left toe taps, finger snap, wind-up
  2      "Hey!" + burst  sky point #1 (hips right, long diagonal up to the glove), hold
  4      "Van-"          fist to the chest, crouch, right knee pops in
  4.75   "-king!"        toe stand, glove thrown wide and high, fingers spread
  6-8    sax answers     heel swivels with a shoulder shimmy, snap on 7, step out to the left
  8, 9   "Yeah, yeah"    "come on" reach to the audience, right foot slides in, arm across
  10     "Woo!" + burst  side kick with the right leg, sky point #2
  12     "Van-"          hand on the heart, small bow to the right
  12.75  "-king!"        broad sweep open up to the word in the sky
  14-16  sax solo        tips the fedora: pinches the brim, head dips into the hand, heel taps
  17-18                  full spin on the ball of the left foot
  18     "Hee!" + burst  lands on the toes, sky point #3, the glove flutters down
  20     "Van-"          wide step, crouch, fist
  20.75  "-king!"        side lunge to the left with a punch up
  22-24  sax             heel-and-toe swivel glide to the right and back, snap on 23
  24-25  "Oh, oh"        groove
  25     "Hey!" + burst  sky point #4 turned three-quarters, second pump on 25.5
  26.5   "Van-"          fist, crouch
  27.25  "-king!"        "ta-da": arm wide to the side, palm up, right knee pop
  28.5-30                back into the opening groove (snap on 29)
"""
import math

import numpy as np

from .. import propspec, score
from ..mathx import TAU, clamp, smootherstep, smoothstep, wrap_delta
from ..rig import HEAD, Attach, Pose, xf, xfd
from .common import (FLOW, HOLD, NB, A, ArmTrack, Foot, Track, amix, beats, blink, body_key, bounce, eval_arm, fit_hips, head_in_chest, pbump, pp,
                     pwin, reach, reach_xyz, set_torso, spin_root, sway, to_arm, to_leg, unit)

NAME = "singer"
SLIDES = {"R": [(8.55, 9.0)]}                 # deliberate slide (for dev/check_choreo.py)
CONTACTS = [("fingertip.L", "hat brim", 14.0, 16.2), ("hand.L", "hat brim", 14.0, 16.2),
            ("wrist.L", "hat brim", 14.0, 16.2)]   # the glove pinches the brim

# ---------------------------------------------------------------- lip sync (own, C1; see report)
_OPEN = {"a": 1.0, "o": 0.80, "e": 0.72, "i": 0.55, "u": 0.55}
_ROUND = {"a": 0.0, "o": 0.9, "e": 0.0, "i": 0.0, "u": 1.0}
_SMILE = {"a": 0.45, "o": 0.1, "e": 0.75, "i": 0.8, "u": 0.05}
_VOC = [(float(s["t"]), float(s["d"]), _OPEN[s["vowel"]], _ROUND[s["vowel"]], _SMILE[s["vowel"]],
         s.get("onset", ""), bool(s.get("coda"))) for s in score.VOCAL]


def _mouth(b):
    """-> (jaw, round, smile, singing) at beat b; every syllable is anticipated by about a frame."""
    closed = 1.0
    quiet = 1.0
    rnd = 0.0
    smile = 0.0
    for t0, d, op, ro, sm, onset, coda in _VOC:
        u = wrap_delta(t0, b, NB)
        if u <= -0.15 or u >= d + 0.15:
            continue
        env = smoothstep(-0.14, 0.04, u) * (1.0 - smoothstep(d - 0.05, d + 0.14, u))
        e = env
        if onset in ("v", "w"):                              # lips start together
            e *= 0.15 + 0.85 * smoothstep(0.0, 0.17, u)
        elif onset == "k":
            e *= 0.45 + 0.55 * smoothstep(0.0, 0.12, u)
        if coda:                                             # n / ng close the mouth early
            e *= 1.0 - 0.6 * smoothstep(0.58 * d, 0.92 * d, u)
        if d > 0.7:                                          # a little vibrato on the long notes
            e *= 1.0 + 0.06 * math.sin(TAU * 2.75 * u) * smoothstep(0.3, 0.5, u)
        closed *= 1.0 - clamp(e * op)
        quiet *= 1.0 - env
        rnd += env * ro
        smile += env * sm
    sing = 1.0 - quiet
    return 1.0 - closed, clamp(rnd), clamp(0.78 * quiet + smile), sing


# ---------------------------------------------------------------- feet
BX, BY, BYAW = 0.16, -0.13, 0.25
FL = Foot("L", BX, BY, BYAW)
FR = Foot("R", -BX, BY, -BYAW)

# "Van-king!" #1: right knee pop, then both heels up (toe stand)
FR.set(3.7, 4.0, heel=0.6)
FL.set(4.45, 4.75, heel=0.7)
FR.set(4.45, 4.75, heel=0.7)
FL.set(5.6, 6.0, heel=0.0)
FR.set(5.6, 6.0, heel=0.0)
# heel swivels on the balls of the feet (heels out / in), then the left foot steps out
FL.twist(6.02, 6.42, -0.12).twist(6.52, 6.92, 0.55).twist(7.02, 7.42, -0.05)
FR.twist(6.02, 6.42, 0.12).twist(6.52, 6.92, -0.55).twist(7.02, 7.42, 0.05).twist(7.5, 7.9, -BYAW)
FL.step(7.55, 8.0, x=0.26, yaw=0.32, lift=0.05)
# the right foot slides in, then kicks out to the side on "Woo!"
FR.slide(8.55, 9.0, x=-0.02, yaw=-0.12)
FR.step(9.68, 10.0, x=-0.37, y=-0.15, yaw=-0.55, lift=0.04, up=0.17, toe_down=0.25, peak=0.6)
FR.step(10.4, 10.8, x=-0.19, y=BY, yaw=-BYAW, lift=0.02, up=0.0, peak=0.3)
FL.step(11.1, 11.5, x=BX, yaw=BYAW, lift=0.05)
FR.step(11.6, 11.95, x=-BX, lift=0.04)
# the spin: on the ball of the left foot, the right foot tucked in
FL.set(16.72, 16.95, heel=0.5)
FR.step(16.82, 17.15, x=0.03, y=-0.16, yaw=-0.15, lift=0.03, up=0.15, toe_down=0.3)
FR.step(17.72, 18.04, x=-0.17, y=BY, yaw=-BYAW, lift=0.0, up=0.0, heel=0.7, peak=0.3)
FL.set(17.8, 18.04, heel=0.7)
FL.set(18.45, 18.8, heel=0.0)
FR.set(18.45, 18.8, heel=0.0)
FR.step(19.0, 19.35, x=-BX, lift=0.03)
# wide step for the side lunge, and back
FL.step(19.55, 20.0, x=0.30, yaw=0.42, lift=0.06)
FL.step(21.62, 22.0, x=BX, yaw=BYAW, lift=0.05)
# heel-and-toe swivel glide: 12 cm to the right and back
FL.twist(22.0, 22.5, -0.35, pivot="heel", pop=0.3).twist(22.5, 23.0, 0.25, pop=0.3)
FL.twist(23.0, 23.5, -0.35, pop=0.3).twist(23.5, 24.0, 0.25, pivot="heel", pop=0.3)
FR.twist(22.0, 22.5, 0.35, pop=0.3).twist(22.5, 23.0, -0.25, pivot="heel", pop=0.3)
FR.twist(23.0, 23.5, 0.35, pivot="heel", pop=0.3).twist(23.5, 24.0, -0.25, pop=0.3)
# last "Van-king!": right knee pop
FR.set(26.9, 27.25, heel=0.6)
FR.set(28.2, 28.6, heel=0.0)
FL.close()
FR.close()


# ---------------------------------------------------------------- body
B = body_key
GROOVE_R = B(-0.05, 0, -0.03, (0.0, -0.08, 0.06), (0.03, 0.12, -0.09), (0.0, 0.06, 0.04))
GROOVE_L = B(0.05, 0, -0.03, (0.0, 0.08, -0.06), (0.03, -0.12, 0.09), (0.0, -0.06, -0.04))
GROOVE_C = B(0.0, 0, -0.03, (0.0, 0.0, 0.0), (0.02, 0.0, 0.0), (0.0, 0.0, 0.0))

_BODY = Track(pp([
    (0.0, GROOVE_R, dict(dur=0.4)),
    (1.0, B(-0.06, 0, -0.02, (-0.02, -0.05, 0.07), (-0.05, 0.16, -0.10), (-0.10, 0.12, 0.07)), dict(dur=0.4, over=0.0)),
    (1.68, B(0.01, 0, -0.085, (0.05, -0.12, 0.0), (0.13, -0.18, 0.0), (0.14, -0.15, -0.04)), dict(dur=0.34, over=0.0)),
    (2.0, B(-0.085, 0, 0.0, (-0.03, 0.10, 0.13), (-0.10, 0.16, 0.10), (-0.26, 0.30, 0.12), (0.12, 0.0)), dict(dur=0.26, over=0.08)),
    (3.72, B(0.03, 0, -0.085, (0.06, -0.16, -0.05), (0.15, -0.20, 0.04), (0.14, -0.12, -0.04), (0.0, -0.1)), dict(dur=0.4, over=0.0)),
    (4.0, B(0.035, 0, -0.075, (0.06, -0.16, -0.06), (0.17, -0.22, 0.05), (0.14, -0.10, -0.05), (0.0, -0.5)), dict(dur=0.2, over=0.0)),
    (4.75, B(0.0, 0, 0.10, (-0.05, 0.08, 0.0), (-0.15, 0.16, 0.03), (-0.26, 0.12, 0.03), (0.1, 0.1)), dict(dur=0.28, over=0.06)),
    (6.0, B(0.0, 0, -0.07, (0.03, 0.0, 0.0), (0.06, 0.0, 0.0), (0.04, 0.0, 0.0)), dict(dur=0.4, over=0.0)),
    (6.42, B(0.02, 0, -0.045, (0.0, 0.16, -0.05), (0.04, -0.30, 0.07), (0.0, 0.05, -0.04), (-0.2, -0.2)), FLOW),
    (6.92, B(-0.02, 0, -0.035, (0.0, -0.16, 0.05), (0.04, 0.30, -0.07), (0.0, -0.05, 0.04), (0.25, 0.25)), FLOW),
    (7.42, B(0.02, 0, -0.045, (0.0, 0.16, -0.05), (0.04, -0.30, 0.07), (0.0, 0.05, -0.04), (-0.2, -0.2)), FLOW),
    (8.0, B(0.13, -0.03, -0.06, (0.10, 0.10, -0.07), (0.16, 0.10, 0.07), (-0.06, 0.05, 0.06)), dict(dur=0.4, over=0.05)),
    (9.0, B(0.12, 0, -0.02, (0.0, -0.16, -0.04), (-0.03, -0.26, 0.05), (0.0, 0.10, 0.0)), dict(dur=0.42, over=0.04)),
    (9.68, B(0.15, 0, -0.085, (0.06, -0.10, -0.05), (0.14, -0.22, 0.02), (0.12, -0.10, 0.0)), dict(dur=0.3, over=0.0)),
    (10.0, B(0.20, 0, -0.01, (-0.02, 0.06, 0.16), (-0.10, 0.14, 0.12), (-0.26, 0.26, 0.10), (0.1, 0.0)), dict(dur=0.26, over=0.06)),
    (10.85, B(0.04, 0, -0.07, (0.03, 0.0, 0.03), (0.06, 0.0, -0.03), (0.02, 0.05, 0.0)), dict(dur=0.45, over=0.0)),
    (11.5, B(-0.02, 0, -0.03, (0.0, -0.05, 0.04), (0.02, 0.06, -0.05), (0.0, 0.0, 0.0)), FLOW),
    (12.0, B(-0.06, -0.02, -0.075, (0.11, -0.12, 0.05), (0.24, -0.20, -0.08), (0.20, -0.16, -0.06)), dict(dur=0.36, over=0.04)),
    (12.75, B(0.04, 0, -0.01, (-0.04, 0.10, -0.04), (-0.17, 0.20, 0.07), (-0.32, 0.14, 0.06)), dict(dur=0.34, over=0.06)),
    (14.45, B(-0.04, 0, -0.045, (0.02, -0.06, 0.05), (0.07, 0.06, -0.12), (0.17, 0.46, 0.20)), dict(dur=0.5, over=0.0)),
    (15.0, B(-0.045, 0, -0.085, (0.03, -0.06, 0.05), (0.10, 0.06, -0.12), (0.27, 0.46, 0.22)), dict(dur=0.22, over=0.0)),
    (15.5, B(-0.04, 0, -0.045, (0.02, -0.06, 0.05), (0.06, 0.06, -0.12), (0.15, 0.46, 0.20)), dict(dur=0.3, over=0.0)),
    (16.0, B(0.03, 0, -0.015, (-0.02, 0.10, -0.02), (-0.08, 0.16, 0.05), (-0.12, 0.10, 0.05)), dict(dur=0.26, over=0.06)),
    (16.75, B(0.08, 0, -0.085, (0.05, -0.26, -0.03), (0.10, -0.36, 0.0), (0.06, -0.26, 0.0), (0.0, 0.1)), dict(dur=0.36, over=0.0)),
    (17.2, B(0.125, -0.01, -0.02, (0.0, 0.08, -0.03), (0.02, 0.10, 0.0), (0.0, 0.12, 0.0), (0.0, 0.55)), dict(dur=0.32, over=0.0)),
    (18.0, B(0.0, 0, 0.10, (-0.03, 0.05, 0.05), (-0.11, 0.12, 0.07), (-0.26, 0.26, 0.09), (0.1, 0.1)), dict(dur=0.28, over=0.06)),
    (18.85, B(0.0, 0, -0.075, (0.03, 0.0, 0.0), (0.06, 0.0, 0.0), (0.03, 0.08, 0.0)), dict(dur=0.42, over=0.0)),
    (19.5, GROOVE_C, FLOW),
    (20.0, B(0.05, 0, -0.10, (0.07, -0.10, 0.0), (0.19, -0.18, 0.0), (0.16, -0.10, 0.0), (0.2, 0.1)), dict(dur=0.36, over=0.04)),
    (20.75, B(0.185, 0, -0.055, (0.0, 0.10, -0.11), (-0.09, 0.12, 0.15), (-0.22, 0.20, 0.09), (0.3, 0.0)), dict(dur=0.28, over=0.06)),
    (22.0, B(0.0, 0, -0.055, (0.02, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.0, 0.0)), dict(dur=0.4, over=0.0)),
    (22.5, B(-0.07, 0, -0.045, (0.0, 0.14, 0.05), (0.04, -0.26, -0.07), (0.0, 0.06, 0.05), (-0.25, -0.25)), FLOW),
    (23.0, B(-0.125, 0, -0.055, (0.0, -0.12, 0.05), (0.04, 0.24, -0.08), (0.0, -0.05, 0.03), (0.2, 0.2)), FLOW),
    (23.5, B(-0.07, 0, -0.045, (0.0, 0.14, -0.05), (0.04, -0.26, 0.07), (0.0, 0.06, -0.05), (-0.25, -0.25)), FLOW),
    (24.0, GROOVE_L, dict(dur=0.4, over=0.0)),
    (24.66, B(-0.01, 0, -0.085, (0.05, -0.14, 0.0), (0.13, -0.20, 0.0), (0.14, -0.15, 0.0)), dict(dur=0.34, over=0.0)),
    (25.0, B(-0.075, 0, 0.0, (-0.02, 0.26, 0.10), (-0.09, 0.30, 0.09), (-0.26, 0.16, 0.10), (0.15, 0.0)), dict(dur=0.26, over=0.08)),
    (25.3, B(-0.07, 0, -0.06, (0.0, 0.24, 0.08), (-0.02, 0.26, 0.06), (-0.12, 0.16, 0.06), (0.15, 0.0)), FLOW),
    (25.5, B(-0.075, 0, 0.0, (-0.02, 0.26, 0.10), (-0.10, 0.30, 0.09), (-0.28, 0.16, 0.10), (0.15, 0.0)), dict(dur=0.18, over=0.0, snap=1.0)),
    (26.5, B(-0.03, 0, -0.095, (0.06, 0.12, 0.0), (0.17, 0.18, 0.0), (0.14, 0.06, 0.0)), dict(dur=0.4, over=0.04)),
    (27.25, B(0.05, 0, -0.005, (-0.02, -0.08, -0.07), (-0.11, -0.10, 0.09), (-0.24, 0.05, 0.07), (0.1, -0.5)), dict(dur=0.28, over=0.06)),
    (28.7, GROOVE_C, dict(dur=0.42, over=0.0)),
]))

# how much of the beat bounce is on (0 during held poses and the spin)
_AMP = Track([(0.0, 1.0), (1.5, 1.0), (1.9, 0.25), (3.3, 0.25), (3.7, 0.0), (5.7, 0.0), (6.1, 0.8), (7.6, 0.8),
              (8.0, 0.5), (9.5, 0.5), (9.9, 0.2), (10.8, 0.2), (11.2, 1.0), (11.7, 0.6), (13.7, 0.3), (14.3, 0.7),
              (16.3, 0.7), (16.7, 0.0), (18.3, 0.0), (18.9, 0.6), (19.5, 0.6), (19.9, 0.2), (21.5, 0.2), (22.0, 0.7),
              (24.0, 1.0), (24.5, 0.6), (24.9, 0.2), (26.2, 0.2), (26.5, 0.0), (28.4, 0.0), (28.9, 1.0)], snap=1.0)

# ---------------------------------------------------------------- left arm (the glove)
POINT = [0.85, 0.0, 0.95, 0.95]           # index finger points
SNAP = [0.75, 0.25, 0.8, 0.8]
FIST = [0.9, 1.0, 1.0, 1.0]
L_REST = reach("L", (0.33, 0.10, -0.13), elbow=(0.8, -0.6, -0.2), curl=0.45)
L_REST2 = reach("L", (0.31, 0.16, -0.08), elbow=(0.8, -0.6, -0.2), curl=0.5)
L_SNAP = reach("L", (0.44, 0.20, 0.13), elbow=(0.5, -0.3, -0.8), fingers=(0.35, 0.45, 0.8), palm=(-0.7, 0.6, 0.0), curl=SNAP)
L_SNAP_LOW = reach("L", (0.47, 0.16, -0.03), elbow=(0.6, -0.6, -0.4), fingers=(0.7, 0.5, 0.3), palm=(-0.3, 0.6, 0.6), curl=SNAP)
L_WIND = reach("L", (0.05, 0.25, -0.03), elbow=(0.8, 0.1, -0.5), curl=0.7)
L_SKY = A("L", (0.60, 0.14, 0.78), (0.56, 0.10, 0.82), palm=(0.3, 0.9, -0.2), curl=POINT, shrug=0.045)
L_SKY_OPEN = A("L", (0.56, 0.16, 0.80), (0.50, 0.14, 0.86), palm=(0.2, 0.95, -0.1), curl=0.0, spread=1.0, shrug=0.045)
L_SKY_TURN = A("L", (0.50, 0.32, 0.80), (0.46, 0.28, 0.84), palm=(0.4, 0.85, -0.2), curl=POINT, shrug=0.045)
L_PUMP = reach("L", (0.44, 0.20, 0.36), elbow=(0.8, -0.1, -0.5), fingers=(0.35, 0.2, 0.9), palm=(0.3, 0.9, -0.1), curl=POINT, shrug=0.02)
L_FIST = reach("L", (0.045, 0.235, 0.085), elbow=(0.8, 0.2, -0.45), fingers=(-0.9, 0.1, 0.35), palm=(0.0, -1.0, 0.1), curl=FIST)
L_FIST_HI = reach("L", (0.07, 0.25, 0.13), elbow=(0.9, 0.2, -0.1), fingers=(-0.95, 0.1, 0.15), palm=(0.0, -0.9, -0.3), curl=FIST)
L_HEART = reach("L", (0.06, 0.225, 0.07), elbow=(0.7, 0.2, -0.6), fingers=(-0.85, 0.05, 0.5), palm=(-0.1, -1.0, 0.0), curl=0.12, spread=0.4)
L_WIDE = A("L", (0.88, 0.12, 0.42), (0.78, 0.18, 0.60), palm=(0.1, 0.95, -0.2), curl=-0.1, spread=1.0, shrug=0.03)
L_HIGH_OPEN = A("L", (0.72, 0.20, 0.62), (0.56, 0.28, 0.78), palm=(-0.3, 0.6, 0.7), curl=0.05, spread=0.8, shrug=0.035)
L_FWD = reach("L", (0.24, 0.43, 0.10), elbow=(0.8, -0.2, -0.5), fingers=(0.1, 0.9, 0.35), palm=(-0.3, -0.2, 0.9), curl=0.2, spread=0.3)
L_COMEON = reach("L", (0.30, 0.42, 0.08), elbow=(0.8, -0.2, -0.5), fingers=(0.15, 0.95, 0.2), palm=(-0.2, -0.1, 1.0), curl=0.15, spread=0.5, forward=0.02)
L_BECKON = reach("L", (0.28, 0.36, 0.14), elbow=(0.8, -0.2, -0.5), fingers=(0.0, 0.55, 0.8), palm=(-0.2, -0.75, 0.6), curl=[0.3, 0.9, 0.9, 0.9], forward=0.02)
L_ACROSS = reach("L", (0.0, 0.27, 0.0), elbow=(0.8, 0.2, -0.5), fingers=(-0.9, 0.3, 0.1), palm=(0.0, -0.3, -0.9), curl=0.1, spread=0.6)
L_LOW_OUT = A("L", (0.75, 0.05, -0.65), (0.85, 0.20, -0.48), palm=(0.0, 0.9, 0.3), curl=0.0, spread=1.0)
L_SIDE = A("L", (0.95, 0.12, 0.05), (0.92, 0.22, 0.22), palm=(0.0, 0.5, -0.85), curl=0.1, spread=0.7)
L_NEAR_HAT = reach("L", (0.44, 0.17, 0.50), elbow=(0.7, 0.5, -0.5), fingers=(-0.5, -0.3, 0.8), palm=(-0.4, -0.2, -0.9), curl=0.45, shrug=0.05)
L_FLICK = A("L", (0.86, 0.16, 0.34), (0.74, 0.26, 0.60), palm=(0.2, 0.95, -0.1), curl=-0.1, spread=1.0, shrug=0.02)
L_TUCK = reach("L", (0.09, 0.235, 0.02), elbow=(0.9, 0.0, -0.4), fingers=(-0.9, 0.2, 0.2), palm=(0.0, -1.0, 0.0), curl=0.8)
L_PUNCH = A("L", (0.66, 0.12, 0.74), (0.60, 0.10, 0.79), palm=(-0.8, 0.5, 0.2), curl=FIST, shrug=0.045)
L_TADA = A("L", (0.90, 0.20, -0.08), (0.90, 0.30, 0.12), fingers=(0.92, 0.3, 0.2), palm=(-0.15, -0.2, 0.95), curl=0.0, spread=0.9)
L_SWING = reach("L", (0.41, 0.32, 0.17), elbow=(0.8, -0.2, -0.6), palm=(0.2, 0.8, -0.5), curl=[0.8, 0.2, 0.8, 0.8])       # passing poses
L_SWING_LOW = reach("L", (0.33, 0.30, -0.02), elbow=(0.8, -0.3, -0.5), curl=0.5)
L_SWING_OPEN = reach("L", (0.38, 0.35, 0.10), elbow=(0.8, -0.3, -0.5), palm=(-0.4, 0.3, 0.85), curl=0.2, spread=0.5)
L_SWING_FIST = reach("L", (0.40, 0.30, 0.22), elbow=(0.8, -0.2, -0.6), curl=FIST)
L_DOWN_SIDE = A("L", (0.93, 0.12, -0.34), (0.86, 0.30, -0.40), curl=0.4)
L_SHIM = reach("L", (0.34, 0.24, 0.0), elbow=(0.8, -0.5, -0.3), curl=0.75)
L_SHIM2 = reach("L", (0.36, 0.14, 0.03), elbow=(0.8, -0.5, -0.3), curl=0.75)

_ARM_L = ArmTrack("L", pp([
    (0.0, L_REST, dict(dur=0.4, over=0.0)),
    (1.0, L_SNAP, dict(dur=0.3)),
    (1.62, L_WIND, dict(dur=0.34, over=0.0)),
    (1.85, L_SWING, FLOW),
    (2.0, L_SKY, dict(dur=0.26, over=0.06)),
    (4.0, L_FIST, dict(dur=0.36, over=0.03)),
    (4.6, L_SWING_OPEN, FLOW),
    (4.75, L_WIDE, dict(dur=0.28, over=0.06)),
    (6.0, L_SHIM, dict(dur=0.45, over=0.0)),
    (6.42, L_SHIM2, FLOW),
    (7.0, L_SNAP_LOW, dict(dur=0.28)),
    (8.0, L_COMEON, dict(dur=0.38, over=0.04)),
    (8.5, L_BECKON, dict(dur=0.22, over=0.0)),
    (9.0, L_ACROSS, dict(dur=0.3, over=0.04)),
    (9.62, L_WIND, dict(dur=0.3, over=0.0)),
    (9.85, L_SWING, FLOW),
    (10.0, L_SKY_OPEN, dict(dur=0.26, over=0.06)),
    (11.25, L_LOW_OUT, dict(dur=0.5, over=0.03)),
    (11.8, L_SWING_LOW, FLOW),
    (12.0, L_HEART, dict(dur=0.36, over=0.0)),
    (12.48, L_FWD, FLOW),
    (12.75, L_HIGH_OPEN, dict(dur=0.2, over=0.05)),
    (14.45, L_NEAR_HAT, dict(dur=0.45, over=0.0)),
    (16.0, L_FLICK, dict(dur=0.24, over=0.06)),
    (16.55, L_SWING_OPEN, FLOW),
    (16.75, L_ACROSS, dict(dur=0.36, over=0.0)),
    (17.2, L_TUCK, dict(dur=0.3, over=0.0)),
    (17.84, L_SWING, FLOW),
    (18.0, L_SKY, dict(dur=0.26, over=0.06)),
    (18.5, L_SKY, HOLD),
    (19.05, L_SIDE, FLOW),
    (19.5, L_REST2, dict(dur=0.3, over=0.0)),
    (20.0, L_FIST_HI, dict(dur=0.34, over=0.03)),
    (20.6, L_SWING_FIST, FLOW),
    (20.75, L_PUNCH, dict(dur=0.26, over=0.06)),
    (21.4, L_PUNCH, HOLD),
    (21.75, L_DOWN_SIDE, FLOW),
    (22.0, L_REST2, dict(dur=0.42, over=0.0)),
    (22.5, L_LOW_OUT, FLOW),
    (23.0, L_SNAP, dict(dur=0.28)),
    (23.6, L_LOW_OUT, FLOW),
    (24.0, L_REST, dict(dur=0.3, over=0.0)),
    (24.62, L_WIND, dict(dur=0.34, over=0.0)),
    (24.85, L_SWING, FLOW),
    (25.0, L_SKY_TURN, dict(dur=0.26, over=0.06)),
    (25.3, L_PUMP, FLOW),
    (25.5, L_SKY_TURN, dict(dur=0.18, over=0.0)),
    (25.95, L_SIDE, FLOW),
    (26.3, L_SWING_LOW, FLOW),
    (26.5, L_FIST, dict(dur=0.34, over=0.03)),
    (27.1, L_SWING_OPEN, FLOW),
    (27.25, L_TADA, dict(dur=0.28, over=0.06)),
    (28.7, L_REST2, dict(dur=0.42, over=0.0)),
    (29.0, L_SNAP, dict(dur=0.26)),
]))

# hat pinch (HEAD space): wrist just below / outside HEAD['brim_front_L'], fingers on the brim
_HAT_WRIST = np.array(HEAD["brim_front_L"]) + np.array([0.05, -0.01, -0.085])
_HAT_FINGERS = unit((-0.50, 0.25, 0.83))
_HAT_PALM = unit((-0.45, 0.30, -0.84))
_HAT_POLE = np.array([0.6, -0.5, -0.6])
_HAT_CURL = [0.55, 0.5, 0.55, 0.6]


# ---------------------------------------------------------------- right arm (microphone)
def _mic_arm(m, head_pos, axis, palm_hint, pole, shrug=0.0):
    """Arm vector (chest space) that puts the ball of the mic at `head_pos` with the handle along
    -`axis`; all arguments are in the frame `m` (4×4 relative to the chest, None = chest)."""
    a = unit(axis)
    palm = np.asarray(palm_hint, dtype=np.float64)
    palm = unit(palm - a * float(np.dot(palm, a)))
    fingers = np.cross(-a, palm)
    wrist = np.asarray(head_pos, dtype=np.float64) - 0.125 * a - 0.034 * palm - 0.062 * fingers
    if m is not None:
        wrist, fingers, palm, pole = xf(m, wrist), xfd(m, fingers), xfd(m, palm), xfd(m, pole)
    return reach_xyz("R", wrist, pole, fingers, palm, curl=0.85, shrug=shrug)


_MOUTH = np.array(HEAD["mouth"])
_MIC_AT = _MOUTH + np.array([-0.015, -0.072, -0.045])        # ball of the mic, head space
_MIC_AXIS = (0.34, 0.42, 0.84)
_MIC_PALM = (0.45, 0.85, 0.0)
_MIC_POLE = np.array([-0.9, 0.1, -0.5])

R_CHEST = _mic_arm(None, (-0.17, -0.30, 0.13), (0.15, -0.15, 0.95), (0.5, 0.85, 0.0), (-0.9, 0.2, -0.4))
R_LOW = _mic_arm(None, (-0.37, -0.20, 0.03), (-0.15, -0.35, 0.9), (0.6, 0.6, 0.3), (-0.8, 0.5, -0.2))
R_OUT = _mic_arm(None, (-0.55, -0.22, 0.20), (-0.35, -0.25, 0.9), (0.5, -0.8, 0.0), (-0.3, 0.3, -0.9))
R_CROWD = _mic_arm(None, (-0.40, -0.44, 0.30), (-0.15, -0.75, 0.6), (0.9, 0.0, 0.3), (-0.8, 0.3, -0.5))

_ARM_R = ArmTrack("R", [(2.6, R_CHEST), (3.1, R_CHEST), (5.9, R_LOW), (7.4, R_LOW), (10.6, R_CROWD), (11.5, R_CROWD),
                (13.9, R_LOW), (16.6, R_LOW), (18.6, R_CHEST), (19.2, R_CHEST), (22.0, R_OUT), (23.5, R_OUT),
                (25.6, R_CHEST), (26.0, R_CHEST), (28.4, R_LOW), (29.3, R_LOW)], snap=1.0)
# 1 = microphone at the mouth, 0 = swung away
_W_MIC = Track([(2.5, 1.0), (3.0, 0.3), (3.55, 1.0), (5.8, 1.0), (6.3, 0.0), (7.3, 0.0), (7.8, 1.0), (10.5, 1.0),
                (11.0, 0.0), (11.4, 0.0), (11.9, 1.0), (13.8, 1.0), (14.3, 0.0), (16.45, 0.0), (16.95, 1.0),
                (18.5, 1.0), (19.0, 0.25), (19.6, 1.0), (21.8, 1.0), (22.3, 0.0), (23.3, 0.0), (23.8, 1.0),
                (25.5, 1.0), (25.9, 0.5), (26.3, 1.0), (28.3, 1.0), (28.8, 0.0), (29.2, 0.0), (29.7, 1.0)], snap=1.0)

# ---------------------------------------------------------------- face
_BLINKS = (3.3, 7.6, 11.4, 16.55, 19.3, 22.3, 26.2, 29.2)
_LOOK = Track(pp([(0.0, (0.0, 0.0)), (2.0, (0.35, 0.6)), (3.7, (0.0, -0.1)), (4.75, (0.0, 0.5)), (6.0, (0.0, 0.0)),
                  (10.0, (0.3, 0.6)), (11.0, (0.0, 0.0)), (12.75, (0.1, 0.7)), (14.4, (0.4, -0.3)), (16.0, (0.0, 0.0)),
                  (18.0, (0.3, 0.6)), (19.0, (0.0, 0.0)), (20.75, (0.3, 0.5)), (22.0, (-0.3, 0.0)), (24.0, (0.0, 0.0)),
                  (25.0, (0.3, 0.6)), (26.5, (0.0, -0.1)), (27.25, (0.2, 0.4)), (28.7, (0.0, 0.0))],
                 dur=0.3, antic=0.0, over=0.0))
_BROW = Track(pp([(0.0, 0.15), (2.0, 0.9), (3.7, -0.2), (4.75, 0.8), (6.0, 0.15), (8.0, 0.6), (9.65, 0.0), (10.0, 0.9),
                  (11.0, 0.15), (12.0, -0.3), (12.75, 0.8), (14.45, 0.3), (15.0, -0.2), (16.0, 0.6), (16.75, -0.3),
                  (18.0, 0.9), (19.0, 0.15), (20.0, -0.3), (20.75, 0.8), (22.0, 0.3), (24.66, 0.0), (25.0, 0.9),
                  (26.5, -0.3), (27.25, 0.8), (28.7, 0.15)], dur=0.3, antic=0.0, over=0.0))

_SPIN = (16.92, 17.96)


def pose(t):
    b = beats(t) % NB
    p = Pose()

    # ---- body: hips lead, rib cage and head follow a little later (overlap)
    v0, v1, v2 = _BODY(b), _BODY(b - 0.07), _BODY(b - 0.13)
    amp = _AMP(b)
    bn = bounce(b)
    bn2 = bounce(b, 2.0)
    sw = sway(b, 2.0, 0.25)
    hips_rot = v0[3:6] + np.array([0.02 * amp * bn, 0.0, 0.025 * amp * sw])
    body_rot = v1[6:9] + np.array([0.035 * amp * bounce(b - 0.08), 0.03 * amp * sw, -0.04 * amp * sw])
    head_rot = v2[9:12] + np.array([0.05 * amp * bounce(b - 0.16, sharp=1.2) - 0.02 * amp, 0.0, 0.02 * amp * sway(b - 0.15, 2.0, 0.25)])
    p.hips_pos = np.array([v0[0] + 0.012 * amp * sw, v0[1], v0[2] - 0.035 * amp * bn - 0.008 * amp * bn2])
    set_torso(p, hips_rot, body_rot, head_rot)

    # ---- feet (taps are added on top of the scripted feet)
    tap_l = (pwin(b, 28.75, 31.55, 0.3, 0.3) + pwin(b, 24.05, 24.65, 0.2, 0.2) + pwin(b, 11.5, 11.98, 0.2, 0.2)) \
        * 0.34 * (1.0 - bounce(b, sharp=0.9, skew=0.3))
    tap_r = pwin(b, 14.1, 16.4, 0.3, 0.3) * 0.38 * (1.0 - bounce(b, sharp=0.9, skew=0.3))
    fl, fr = FL(b), FR(b)
    p.leg_L = to_leg("L", fl, knee_out=v0[12], toe=tap_l)
    p.leg_R = to_leg("R", fr, knee_out=v0[13], heel=tap_r)
    fit_hips(p)

    # ---- left arm: keyed gestures, the hat pinch is a target in head space
    al = eval_arm("L", _ARM_L, b, lead=0.05, lag=0.07, drag=0.3)
    al[3:6] += np.array([0.0, 0.0, 0.07 * amp * bn])                     # forearm bobs with the groove
    hic = head_in_chest(p)
    w_hat = pwin(b, 14.12, 15.92, 0.38, 0.3)
    if w_hat > 0.0:
        hat = reach_xyz("L", xf(hic, _HAT_WRIST), _HAT_POLE, xfd(hic, _HAT_FINGERS), xfd(hic, _HAT_PALM),
                        curl=_HAT_CURL, shrug=0.05)
        al = amix(al, hat, w_hat)
    p.arm_L = to_arm("L", al)

    # ---- right arm: microphone at the mouth (head space), swung away between the phrases
    w_mic = _W_MIC(b)
    ar = _mic_arm(hic, _MIC_AT, _MIC_AXIS, _MIC_PALM, _MIC_POLE)
    if w_mic < 1.0:
        away = _ARM_R(b)
        away[3:6] += np.array([0.0, 0.0, 0.06 * amp * bn])
        ar = amix(away, ar, clamp(w_mic))
    p.arm_R = to_arm("R", ar)
    p.props["mic"] = Attach("hand.R", matrix=propspec.MIC_IN_HAND["R"])

    # ---- the spin (whole character about the ball of the left foot)
    if _SPIN[0] < b < _SPIN[1]:
        spin_root(p, TAU * smootherstep(_SPIN[0], _SPIN[1], b), fl["ball"])

    # ---- face
    jaw, rnd, smile, sing = _mouth(b)
    p.jaw, p.mouth_round, p.smile = jaw, rnd, smile
    bl = blink(b, _BLINKS)
    p.blink_L = p.blink_R = bl
    lk = _LOOK(b)
    p.look = (float(lk[0]), float(lk[1]))
    br = float(_BROW(b)[0])
    p.brow_L = br + 0.15 * sing
    p.brow_R = br * 0.85 + 0.15 * sing
    return p
