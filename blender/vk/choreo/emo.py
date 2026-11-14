"""Emo dancer (left hill): a Tecktonik routine staged for a mole - a big head and short arms, so
"around the head" means: the hands orbit the head's silhouette in the frontal plane (up the
midline in front of the chin, out to the temple with the elbow high, down the outside), never
over or through it.

Vocabulary (names used below)
  roll      forearm roll: the fists circle each other in front of the chest
  windmill  the signature alternation: one hand combs up past the face to the temple while the
            other sweeps down the outside and slices across to the opposite hip; every half beat
            they swap, so each hand draws a full circle per beat
  locks     the arms freeze in angular shapes exactly on the beat: "Z" (one forearm up, one
            hanging from a high elbow), goalposts, box (one forearm folded across the chest),
            frame (both hands at the temples, elbows out); the wrists break on the off-beat
  wave      a pulse that travels finger tips - wrist - elbow - shoulder through the chest and out
            through the other arm
  slices    cross-body cuts of a flat hand to the opposite shoulder / hip

Everything is keyed in BEATS (120 BPM, 30 beats per loop) against the score; the firework
bursts on 2, 10, 18 and 25 are the accents:

  29.5-1.75  roll that speeds up and climbs from the belly to the chin, parallel heel swivels
             on every half beat, deeper and deeper into the knees
  1.8-2      wind-up crouch, jump
  2          BURST  feet apart, left arm points up at the sky, right hand slices to the left hip
  2.5-3.5    windmill (right temple - left temple - right temple), quick ball changes
  4, 5       locks: "Z" with a knee pop, wrist break on 4.5, the Z flips on 5
  5.5        both forearms fold in front of the chest, crouch
  6-7.7      arm wave from the right hand through the shoulders to the left hand, the weight
             shifts from the right leg to the left one, the head follows the wave
  8, 8.5     step to the left; slices to the left / right shoulder with a quarter turn of the
             upper body
  9, 9.5     arms cross low in a crouch, fling open
  9.8-10     wind-up, jump
  10         BURST  feet apart, both arms punch up in a V; second pump on 10.5 (feet together)
  11-12      windmill the other way round (left temple first) over a heel-and-toe swivel glide
  12         frame: both hands at the temples, elbows out
  12.5       up on the toes while both forearms drop from the high elbows
  13, 13.5   box left / box right, head in profile, knee pops
  14-15.7    arm wave back from the left hand to the right hand
  16-17.5    jumping feet apart / together / apart / together; the arms open low, cross, open
             wide, tuck in
  18         BURST  feet wide, right arm points up, left hand slices to the right hip
  18.5-19.25 fastest combination: the hands swap temple / cheek on every quarter beat over
             quarter-beat steps
  19.5, 20   "Z" locks, mirrored, knee pops
  20.5       forearms fold, wind-up
  20.9-22    full spin on the ball of the left foot, arms tucked
  22         lands on the toes, arms thrown wide
  22-24      "snake": a wave runs into both arms at once, through the chest and out again
  24.2-24.8  fast roll, heel swivel
  25         BURST  feet apart, both hands point up to the same side of the sky
  25.5-26.5  windmill
  27, 27.5   "Z" lock, flip
  28         frame
  28.5, 29   step out; slices to the left / right shoulder, upper body turns - and the right
             hand drops straight into the roll that opens the loop again
"""
import math
from types import SimpleNamespace

import numpy as np

from ..mathx import TAU, clamp, smootherstep
from ..rig import Pose
from .common import (FLOW, HOLD, NB, A, ArmTrack, Foot, Track, beats, blink, body_key, bounce, eval_arm, fit_hips, hit,
                     pbump, pp, reach, set_torso, spin_root, sway, to_arm, to_leg)

NAME = "emo"
BURSTS = (2.0, 10.0, 18.0, 25.0)

# ---------------------------------------------------------------- feet
BX, BY, BYAW = 0.16, -0.13, 0.25
# the loop starts in the middle of the heel swivels: both feet turned to the left
FL = Foot("L", BX, BY, 0.55)
FR = Foot("R", -BX, BY, 0.10)

# parallel heel swivels on the balls of the feet, then the jump apart on the burst
FL.twist(0.22, 0.5, -0.10, pop=0.3).twist(0.72, 1.0, 0.55, pop=0.3).twist(1.22, 1.5, -0.10, pop=0.3).twist(1.52, 1.76, BYAW, pop=0.3)
FR.twist(0.22, 0.5, -0.55, pop=0.3).twist(0.72, 1.0, 0.10, pop=0.3).twist(1.22, 1.5, -0.55, pop=0.3).twist(1.52, 1.76, -BYAW, pop=0.3)
FL.step(1.8, 2.0, x=0.27, yaw=0.35, lift=0.04)
FR.step(1.8, 2.0, x=-0.27, yaw=-0.35, lift=0.04)
# ball changes under the windmill
FR.step(2.3, 2.5, x=-0.21, yaw=-BYAW, lift=0.05)
FL.step(2.8, 3.0, x=0.20, yaw=BYAW, lift=0.05)
FR.step(3.3, 3.5, x=-BX, lift=0.05)
FL.step(3.8, 4.0, x=BX, lift=0.05)
# knee pops on the locks
FR.set(3.75, 4.0, heel=0.55)
FR.set(4.6, 4.9, heel=0.0)
FL.set(4.75, 5.0, heel=0.55)
FL.set(5.3, 5.5, heel=0.0)
# wave right -> left: the free heel comes up
FL.set(5.75, 6.0, heel=0.5)
FL.set(6.75, 7.0, heel=0.0)
FR.set(7.2, 7.5, heel=0.5)
# step-touch with the slices
FL.step(7.75, 8.0, x=0.30, yaw=0.35, lift=0.05)
FR.step(8.25, 8.5, x=0.06, yaw=-0.15, lift=0.05, heel=0.0)
FR.step(8.75, 9.0, x=-BX, yaw=-BYAW, lift=0.05)
FL.step(9.25, 9.5, x=BX, yaw=BYAW, lift=0.05)
# jump apart on the burst, together on the pump
FL.step(9.8, 10.0, x=0.27, yaw=0.35, lift=0.04)
FR.step(9.8, 10.0, x=-0.27, yaw=-0.35, lift=0.04)
FL.step(10.3, 10.5, x=0.13, yaw=BYAW, lift=0.04)
FR.step(10.3, 10.5, x=-0.13, yaw=-BYAW, lift=0.04)
# heel-and-toe swivel glide: 12 cm to the right and back
FL.twist(11.0, 11.25, -0.35, pivot="heel", pop=0.3).twist(11.25, 11.5, 0.25, pop=0.3)
FL.twist(11.5, 11.75, -0.35, pop=0.3).twist(11.75, 12.0, 0.25, pivot="heel", pop=0.3)
FR.twist(11.0, 11.25, 0.35, pop=0.3).twist(11.25, 11.5, -0.25, pivot="heel", pop=0.3)
FR.twist(11.5, 11.75, 0.35, pivot="heel", pop=0.3).twist(11.75, 12.0, -0.25, pop=0.3)
# goalposts on the toes, knee pops with the boxes
FL.set(12.3, 12.5, heel=0.5)
FR.set(12.3, 12.5, heel=0.5)
FL.set(12.75, 13.0, heel=0.0)
FL.set(13.25, 13.5, heel=0.55)
FR.set(13.25, 13.5, heel=0.0)
FL.set(13.75, 14.0, heel=0.0)
# wave left -> right
FR.set(14.0, 14.25, heel=0.5)
FR.set(14.75, 15.0, heel=0.0)
FL.set(15.0, 15.25, heel=0.5)
FL.set(15.5, 15.7, heel=0.0)
# jumps: apart / together / apart / together / wide on the burst
for _b, _x, _yaw, _lift in ((16.0, 0.30, 0.38, 0.05), (16.5, 0.10, 0.12, 0.05), (17.0, 0.30, 0.38, 0.05),
                            (17.5, 0.10, 0.12, 0.05), (18.0, 0.30, 0.40, 0.05)):
    FL.step(_b - 0.2, _b, x=_x, yaw=_yaw, lift=_lift)
    FR.step(_b - 0.2, _b, x=-_x, yaw=-_yaw, lift=_lift)
# quarter-beat steps
FL.step(18.3, 18.5, x=0.21, yaw=BYAW, lift=0.04)
FR.step(18.55, 18.75, x=-0.21, yaw=-BYAW, lift=0.04)
FL.step(18.8, 19.0, x=BX, lift=0.04)
FR.step(19.05, 19.25, x=-BX, lift=0.04)
FR.set(19.3, 19.5, heel=0.55)
FR.set(19.8, 20.0, heel=0.0)
FL.set(19.8, 20.0, heel=0.55)
FL.set(20.3, 20.5, heel=0.0)
# the spin: on the ball of the left foot, the right foot tucked in; lands on the toes
FL.set(20.72, 20.95, heel=0.5)
FR.step(20.82, 21.15, x=0.03, y=-0.16, yaw=-0.15, lift=0.03, up=0.15, toe_down=0.3)
FR.step(21.72, 22.04, x=-0.22, y=BY, yaw=-0.30, lift=0.0, up=0.0, heel=0.7, peak=0.3)
FL.set(21.8, 22.04, heel=0.7)
FL.set(22.45, 22.8, heel=0.0)
FR.set(22.45, 22.8, heel=0.0)
FR.step(23.0, 23.35, x=-BX, yaw=-BYAW, lift=0.03)
# swivel under the fast roll, jump apart on the burst, ball changes, knee pops
FL.twist(24.22, 24.5, -0.10, pop=0.3).twist(24.52, 24.76, BYAW, pop=0.3)
FR.twist(24.22, 24.5, -0.55, pop=0.3).twist(24.52, 24.76, -BYAW, pop=0.3)
FL.step(24.8, 25.0, x=0.27, yaw=0.35, lift=0.04)
FR.step(24.8, 25.0, x=-0.27, yaw=-0.35, lift=0.04)
FR.step(25.3, 25.5, x=-0.21, yaw=-BYAW, lift=0.05)
FL.step(25.8, 26.0, x=0.20, yaw=BYAW, lift=0.05)
FR.step(26.3, 26.5, x=-BX, lift=0.05)
FL.step(26.8, 27.0, x=BX, lift=0.05)
FR.set(26.75, 27.0, heel=0.55)
FR.set(27.25, 27.5, heel=0.0)
FL.set(27.25, 27.5, heel=0.55)
FL.set(27.75, 28.0, heel=0.0)
# step out and back with the last slices, then into the swivels that open the loop
FL.step(28.25, 28.5, x=0.28, yaw=0.35, lift=0.05)
FL.step(28.75, 29.0, x=BX, yaw=BYAW, lift=0.05)
FL.twist(29.72, 30.0, 0.55, pop=0.3)
FR.twist(29.72, 30.0, 0.10, pop=0.3)
FL.close()
FR.close()


# ---------------------------------------------------------------- body
def B(hx, hz, hips=(0.0, 0.0, 0.0), body=(0.0, 0.0, 0.0), head=(0.0, 0.0, 0.0), ko=(0.0, 0.0)):
    return body_key(hx, 0.0, hz, hips, body, head, ko)


def _mirror(k):
    """Left / right mirror image of a body key."""
    m = k.copy()
    m[0] = -k[0]
    for i in (4, 5, 7, 8, 10, 11):
        m[i] = -k[i]
    m[12], m[13] = k[13], k[12]
    return m


SWAY_L = B(-0.035, -0.045, (0.02, 0.10, 0.03), (0.08, -0.06, -0.04), (0.04, 0.0, 0.03))     # toes to the left
SWAY_R = _mirror(SWAY_L)
WIND = B(0.0, -0.10, (0.06, 0.0, 0.0), (0.16, 0.0, 0.0), (0.14, 0.0, 0.0), (0.2, 0.2))
SKY_L = B(-0.07, 0.0, (-0.02, 0.08, 0.10), (-0.08, 0.14, 0.08), (-0.22, 0.28, 0.08), (0.1, 0.0))
SKY_R = B(0.07, 0.0, (-0.02, -0.08, -0.10), (-0.08, -0.14, -0.08), (-0.22, 0.20, -0.04), (0.0, 0.1))
SKY_C = B(0.0, 0.0, (-0.03, 0.0, 0.0), (-0.10, 0.06, 0.0), (-0.26, 0.25, 0.04), (0.1, 0.1))
MILL_R = B(0.03, -0.05, (0.02, -0.08, -0.03), (0.06, -0.20, -0.05), (0.02, -0.04, 0.07))    # right hand at the temple
MILL_L = _mirror(MILL_R)
Z_R = B(-0.02, -0.07, (0.03, 0.0, 0.03), (0.05, 0.0, -0.04), (0.0, 0.10, 0.05), (0.0, -0.45))  # right knee pops in
Z_L = _mirror(Z_R)
FOLD = B(0.0, -0.10, (0.06, 0.0, 0.0), (0.15, 0.0, 0.0), (0.12, 0.0, 0.0), (0.25, 0.25))
WAVE_R = B(-0.05, -0.02, (0.0, 0.0, 0.05), (-0.03, 0.0, -0.08), (-0.05, -0.30, -0.06))      # leaning to the right hand
WAVE_L = _mirror(WAVE_R)
WAVE_C = B(0.0, -0.03, (0.0, 0.0, 0.0), (-0.06, 0.0, 0.0), (-0.04, 0.0, 0.0))
TURN_L = B(0.05, -0.05, (0.02, 0.18, -0.03), (0.04, 0.38, 0.04), (0.0, 0.35, 0.0))          # quarter turn to the left
TURN_R = B(0.05, -0.05, (0.02, -0.18, 0.03), (0.04, -0.38, -0.04), (0.0, -0.35, 0.0))
OPEN = B(0.0, -0.01, (-0.02, 0.0, 0.0), (-0.07, 0.0, 0.0), (-0.10, 0.0, 0.0))
FRAME = B(0.0, -0.06, (0.03, 0.0, 0.0), (0.05, 0.0, 0.0), (0.06, 0.0, 0.0))
POSTS = B(0.0, 0.03, (-0.02, 0.0, 0.0), (-0.05, 0.0, 0.0), (-0.08, 0.0, 0.0))
BOX_L = B(0.04, -0.05, (0.02, 0.0, -0.04), (0.03, 0.10, 0.05), (0.0, 0.40, 0.0), (0.0, -0.4))
BOX_R = _mirror(BOX_L)
LAND = B(0.0, -0.06, (0.03, 0.0, 0.0), (0.06, 0.0, 0.0), (0.04, 0.0, 0.0), (0.15, 0.15))
AIR = B(0.0, 0.02, (-0.01, 0.0, 0.0), (-0.03, 0.0, 0.0), (-0.03, 0.0, 0.0))
TUCK = B(0.0, -0.08, (0.04, 0.0, 0.0), (0.11, 0.0, 0.0), (0.09, 0.0, 0.0))
QUICK_L = B(0.02, -0.05, (0.02, 0.0, 0.0), (0.05, 0.0, -0.07), (0.0, 0.0, -0.06))           # left hand at the temple
QUICK_R = _mirror(QUICK_L)

_HITB = dict(dur=0.2, over=0.07)
_LOCKB = dict(dur=0.2, over=0.04)
_MILLB = dict(dur=0.3, over=0.03)
_JUMPB = dict(dur=0.2, over=0.0)

_BODY = Track(pp([
    (0.0, SWAY_L, FLOW), (0.5, SWAY_R, FLOW), (1.0, SWAY_L + B(0.0, -0.01), FLOW), (1.5, SWAY_R + B(0.0, -0.015), FLOW),
    (1.8, WIND, dict(dur=0.25, over=0.0)),
    (2.0, SKY_L, _HITB),
    (2.5, MILL_R, _MILLB), (3.0, MILL_L, _MILLB), (3.5, MILL_R, _MILLB),
    (4.0, Z_R, _LOCKB), (5.0, Z_L, _LOCKB), (5.5, FOLD, _LOCKB),
    (6.0, WAVE_R, dict(dur=0.3, over=0.03)), (6.8, WAVE_C, FLOW), (7.5, WAVE_L, FLOW),
    (8.0, TURN_L, _HITB), (8.5, TURN_R, _HITB),
    (9.0, B(0.0, -0.09, (0.05, 0.0, 0.0), (0.15, 0.0, 0.0), (0.12, 0.0, 0.0), (0.2, 0.2)), _HITB),
    (9.5, OPEN, _HITB),
    (9.8, WIND, dict(dur=0.22, over=0.0)),
    (10.0, SKY_C, _HITB),
    (10.3, B(0.0, -0.06, (0.02, 0.0, 0.0), (0.04, 0.04, 0.0), (-0.10, 0.20, 0.0)), FLOW),
    (10.5, SKY_C, dict(dur=0.18, over=0.04)),
    (11.0, MILL_L, _MILLB), (11.5, MILL_R, _MILLB),
    (12.0, FRAME, _LOCKB), (12.5, POSTS, _LOCKB), (13.0, BOX_L, _LOCKB), (13.5, BOX_R, _LOCKB),
    (14.0, WAVE_L, dict(dur=0.3, over=0.03)), (14.8, WAVE_C, FLOW), (15.5, WAVE_R, FLOW),
    (16.0, LAND, _JUMPB), (16.25, AIR, FLOW), (16.5, TUCK, _JUMPB), (16.75, AIR, FLOW),
    (17.0, LAND, _JUMPB), (17.25, AIR, FLOW), (17.5, WIND, _JUMPB),
    (18.0, SKY_R, _HITB),
    (18.5, QUICK_L, _JUMPB), (18.75, QUICK_R, _JUMPB), (19.0, QUICK_L, _JUMPB), (19.25, QUICK_R, _JUMPB),
    (19.5, Z_R, _LOCKB), (20.0, Z_L, _LOCKB),
    (20.5, FOLD, _LOCKB),
    (20.78, B(0.06, -0.09, (0.05, -0.24, -0.03), (0.10, -0.34, 0.0), (0.06, -0.24, 0.0), (0.0, 0.1)), dict(dur=0.24, over=0.0)),
    (21.2, B(0.125, -0.02, (0.0, 0.08, -0.03), (0.02, 0.10, 0.0), (0.0, 0.12, 0.0), (0.0, 0.55)), dict(dur=0.32, over=0.0)),
    (22.0, B(0.0, 0.08, (-0.02, 0.0, 0.0), (-0.06, 0.0, 0.0), (-0.10, 0.0, 0.0), (0.1, 0.1)), dict(dur=0.28, over=0.06)),
    (22.7, B(0.0, -0.04, (0.0, 0.0, 0.0), (-0.06, 0.0, 0.0), (-0.08, 0.0, 0.0)), dict(dur=0.4, over=0.0)),
    (23.0, B(0.0, -0.07, (0.03, 0.0, 0.0), (0.10, 0.0, 0.0), (0.10, 0.0, 0.0)), FLOW),
    (23.4, B(0.0, -0.02, (-0.02, 0.0, 0.0), (-0.08, 0.0, 0.0), (-0.10, 0.0, 0.0)), FLOW),
    (24.0, B(0.0, -0.04, (0.02, 0.0, 0.0), (0.04, 0.0, 0.0), (0.0, 0.0, 0.0)), FLOW),
    (24.5, SWAY_R + B(0.0, -0.015), FLOW),
    (24.8, WIND, dict(dur=0.22, over=0.0)),
    (25.0, SKY_L, _HITB),
    (25.5, MILL_R, _MILLB), (26.0, MILL_L, _MILLB), (26.5, MILL_R, _MILLB),
    (27.0, Z_R, _LOCKB), (27.5, Z_L, _LOCKB), (28.0, FRAME, _LOCKB),
    (28.5, TURN_L, _HITB), (29.0, _mirror(TURN_L), _HITB),
    (29.5, SWAY_R, FLOW),
]))

# how much of the beat bounce is on (0 in the jumps and the spin, little in the holds and waves)
_AMP = Track([(0.0, 1.0), (1.6, 1.0), (1.8, 0.2), (2.2, 0.2), (2.5, 0.8), (3.6, 0.8), (3.9, 0.25), (5.6, 0.25), (6.0, 0.15),
              (7.6, 0.15), (8.0, 0.5), (9.6, 0.5), (9.8, 0.1), (10.6, 0.1), (11.0, 0.8), (11.7, 0.8), (12.0, 0.25),
              (13.7, 0.25), (14.0, 0.15), (15.6, 0.15), (15.9, 0.0), (17.6, 0.0), (18.3, 0.2), (18.5, 0.7), (19.3, 0.7),
              (19.5, 0.25), (20.4, 0.25), (20.7, 0.0), (22.3, 0.0), (22.6, 0.2), (24.0, 0.3), (24.3, 1.0), (24.6, 1.0),
              (24.8, 0.2), (25.2, 0.2), (25.5, 0.8), (26.6, 0.8), (26.9, 0.25), (28.2, 0.25), (28.5, 0.5), (29.3, 0.5),
              (29.6, 1.0)], snap=1.0)

# ---------------------------------------------------------------- arms
POINT = [0.85, 0.0, 0.95, 0.95]           # index finger points
BLADE = 0.05                              # flat hand


def _bump(x):
    return math.cos(0.5 * math.pi * x) ** 2 if abs(x) < 1.0 else 0.0


def _wave(s, x, amp=0.8):
    """Arm held out to the side with a wave pulse at station x: 0 = finger tips, 1 = wrist,
    2 = elbow, 3 = shoulder; beyond -1 / 4 the arm is a flat T."""
    eh = 1.3 * amp * (_bump(x) - _bump(x - 1.0))
    ef = amp * (_bump(x - 1.0) - _bump(x - 2.0))
    eu = 0.8 * amp * (_bump(x - 2.0) - _bump(x - 3.0))
    # the elbow always stays a little bent backwards, so the bend never flips through a straight arm
    return A(s, (math.cos(eu), -0.06, math.sin(eu)), (math.cos(ef), 0.24, math.sin(ef)),
             fingers=(math.cos(eh), 0.24, math.sin(eh)), palm=(math.sin(eh), 0.0, -math.cos(eh)), curl=0.0, spread=0.4,
             shrug=0.05 * _bump(x - 3.0))


def _roll(s, turns, cz, rad=0.08):
    """Forearm roll: the wrist on a circle in front of the chest (mostly in the front / up plane,
    tilted outwards so that it also reads from the front); the two hands are half a turn apart."""
    a = TAU * (turns + (0.0 if s == "L" else 0.5))
    c = math.cos(a)
    return reach(s, (0.04 + 0.045 * c, 0.33 + rad * c, cz + rad * math.sin(a)), elbow=(0.9, -0.1, -0.35),
                 fingers=(-1.0, 0.15, 0.1), palm=(0.0, -0.3, -1.0), curl=0.85)


def _lib(s):
    P = SimpleNamespace()
    P.T = _wave(s, 9.0)
    P.GUARD = reach(s, (0.21, 0.30, 0.03), elbow=(0.8, -0.3, -0.5), curl=0.7)
    P.WIND = reach(s, (0.30, 0.24, -0.12), elbow=(0.7, -0.6, -0.2), curl=0.7)
    P.SWING = reach(s, (0.40, 0.31, 0.17), elbow=(0.8, -0.2, -0.6), curl=0.5)                       # passing pose
    # the windmill circle: LOW - IN - FACE - TEMPLE - OUT - DOWN - LOW
    P.LOW = reach(s, (-0.02, 0.27, -0.14), elbow=(0.8, 0.25, -0.2), curl=BLADE, spread=0.2)
    P.IN = reach(s, (0.05, 0.36, 0.10), elbow=(0.8, -0.1, -0.5), curl=BLADE)
    P.FACE = reach(s, (0.17, 0.40, 0.29), elbow=(0.8, 0.2, -0.1), curl=BLADE, forward=0.03)
    P.TEMPLE = reach(s, (0.40, 0.16, 0.44), elbow=(1.0, 0.3, 0.0), fingers=(-0.15, -0.10, 0.98), palm=(-0.98, 0.0, -0.15),
                     curl=0.1, spread=0.3, shrug=0.03)
    P.OUT = A(s, (0.95, -0.10, 0.12), (0.90, 0.35, 0.04), curl=BLADE, spread=0.3)
    P.DOWN = reach(s, (0.30, 0.26, -0.14), elbow=(0.9, -0.2, -0.2), curl=BLADE)
    P.CHEEK = reach(s, (0.22, 0.38, 0.22), elbow=(0.9, 0.0, -0.3), fingers=(-0.2, 0.1, 0.97), palm=(-0.97, 0.0, -0.2), curl=0.1)
    # big shapes
    P.VUP = A(s, (0.56, 0.10, 0.82), (0.48, 0.10, 0.87), palm=(0.3, 0.9, -0.2), curl=POINT, shrug=0.04)
    P.PUMP = reach(s, (0.44, 0.20, 0.36), elbow=(0.8, -0.1, -0.5), curl=POINT, shrug=0.02)
    P.PTX = reach(s, (-0.04, 0.37, 0.27), elbow=(0.6, 0.3, -0.6), fingers=(-0.6, 0.2, 0.75), curl=POINT, forward=0.03)
    P.LOWOUT = A(s, (0.75, 0.05, -0.65), (0.85, 0.20, -0.48), curl=0.0, spread=1.0)
    P.OPENW = A(s, (0.95, -0.05, -0.15), (0.92, 0.25, 0.25), curl=0.0, spread=1.0)
    P.SWEEPB = A(s, (0.80, 0.30, -0.25), (0.30, 0.90, 0.25), curl=0.4)                               # passing pose
    # locks: the elbow stays out at shoulder height, the forearm snaps up / down / in
    P.ZUP = A(s, (0.97, 0.15, 0.05), (0.05, 0.25, 0.97), curl=0.0, spread=0.2)
    P.ZUPB = A(s, (0.97, 0.15, 0.05), (0.05, 0.25, 0.97), fingers=(1.0, 0.1, 0.25), palm=(-0.25, 0.0, 1.0), curl=0.0, spread=0.2)
    P.ZDN = A(s, (0.97, 0.15, 0.12), (0.0, 0.30, -0.95), curl=0.0, spread=0.2)
    P.ZDNB = A(s, (0.97, 0.15, 0.12), (0.0, 0.30, -0.95), fingers=(-1.0, 0.2, -0.1), palm=(0.1, 0.0, -1.0), curl=0.0, spread=0.2)
    P.ZMID = A(s, (0.95, 0.20, 0.08), (0.75, 0.65, 0.0), curl=0.0)                                  # passing poses
    P.ZFWD = A(s, (0.92, 0.30, 0.03), (0.10, 1.0, 0.10), curl=0.0)
    P.ZIN = A(s, (0.85, 0.50, 0.05), (-0.90, 0.35, 0.05), curl=0.0)
    # slices
    P.SWEEP = A(s, (0.60, 0.75, 0.0), (0.15, 1.0, 0.10), curl=BLADE)                                # passing pose
    P.XSH = reach(s, (-0.10, 0.27, 0.15), elbow=(0.5, 0.7, -0.4), curl=BLADE)
    # arms crossed low: the left forearm above the right one
    P.XLOW = reach(s, (-0.03, 0.32, 0.02) if s == "L" else (-0.02, 0.30, -0.12), elbow=(0.6, 0.6, -0.25), curl=BLADE)
    return P


_MILL = {}
_HIT = dict(dur=0.22, over=0.06, settle=0.2)
_LOCK = dict(dur=0.18, over=0.04, settle=0.15)
_Q = dict(dur=0.2, over=0.0)


def _roll_keys(s, b0, b1, r0, r1, z0, z1):
    """FLOW keys (8 per turn) of a roll from beat b0 to b1 whose rate goes from r0 to r1 turns per
    beat and whose centre climbs from z0 to z1."""
    d = b1 - b0
    n = int(round(0.5 * (r0 + r1) * d * 8))
    acc = 0.5 * (r1 - r0) / d
    out = []
    for k in range(n + 1):
        ph = k / 8.0
        x = ph / r0 if abs(acc) < 1e-9 else (-r0 + math.sqrt(r0 * r0 + 4.0 * acc * ph)) / (2.0 * acc)
        out.append((b0 + x, _roll(s, ph, z0 + (z1 - z0) * x / d), FLOW))
    return out


def _wave_keys(s, b0, x0, x1, station=0.19):
    """FLOW keys of a wave pulse travelling from station x0 (at beat b0) to x1."""
    n = int(round(abs(x1 - x0) * 2))
    return [(b0 + k * 0.5 * station, _wave(s, x0 + (x1 - x0) * k / n), FLOW) for k in range(n + 1)]


def _mill(P, b, up):
    """Half a windmill circle that lands on beat b: up the inside to the temple or down the
    outside to the opposite hip."""
    if up:
        return [(b - 0.34, P.IN, FLOW), (b - 0.16, P.FACE, FLOW), (b, P.TEMPLE, _MILL)]
    return [(b - 0.34, P.OUT, FLOW), (b - 0.16, P.DOWN, FLOW), (b, P.LOW, _MILL)]


def _track(s, events):
    ev = sorted(((b % NB, v, o) for b, v, o in events), key=lambda e: e[0])
    return ArmTrack(s, pp(ev))


L, R = _lib("L"), _lib("R")

_ARM_L = _track("L", (
    _roll_keys("L", 29.5, 31.75, 1.0, 5.0 / 3.0, 0.02, 0.14)
    + [(1.87, L.SWING, FLOW), (2.0, L.VUP, _HIT)]
    + _mill(L, 2.5, False) + _mill(L, 3.0, True) + _mill(L, 3.5, False)
    + [(3.78, L.DOWN, FLOW), (4.0, L.ZDN, _LOCK), (4.5, L.ZDNB, _LOCK), (4.8, L.ZMID, FLOW), (5.0, L.ZUP, _LOCK),
       (5.5, L.ZIN, _LOCK), (5.8, L.ZFWD, FLOW), (6.0, L.T, dict(dur=0.2, over=0.0)), (6.76, L.T, HOLD)]
    + _wave_keys("L", 6.855, 3.5, -1.0)
    + [(8.0, L.T, HOLD), (8.33, L.SWEEP, FLOW), (8.5, L.XSH, _HIT), (9.0, L.XLOW, _HIT), (9.3, L.DOWN, FLOW), (9.5, L.LOWOUT, _HIT),
       (9.78, L.WIND, _Q), (9.88, L.SWING, FLOW), (10.0, L.VUP, _HIT), (10.3, L.PUMP, FLOW),
       (10.5, L.VUP, dict(dur=0.18, over=0.0)), (11.0, L.TEMPLE, _Q)]
    + _mill(L, 11.5, False) + _mill(L, 12.0, True)
    + [(12.3, L.ZMID, FLOW), (12.5, L.ZDN, _LOCK), (12.8, L.ZMID, FLOW), (13.0, L.ZUPB, _LOCK), (13.5, L.ZIN, _LOCK),
       (13.8, L.ZFWD, FLOW)]
    + _wave_keys("L", 14.0, 0.0, 4.0)
    + [(15.7, L.T, HOLD), (16.0, L.LOWOUT, _HIT), (16.3, L.DOWN, FLOW), (16.5, L.XLOW, _HIT), (16.7, L.DOWN, FLOW),
       (16.86, L.LOWOUT, FLOW), (17.0, L.OPENW, _HIT), (17.33, L.SWEEPB, FLOW), (17.5, L.GUARD, _HIT), (18.0, L.LOW, _HIT)]
    + [(18.2, L.IN, FLOW), (18.36, L.FACE, FLOW), (18.5, L.TEMPLE, _MILL), (18.75, L.CHEEK, _Q), (19.0, L.TEMPLE, _Q),
       (19.25, L.CHEEK, _Q), (19.5, L.ZUP, _LOCK), (19.8, L.ZMID, FLOW), (20.0, L.ZDN, _LOCK), (20.5, L.ZIN, _LOCK),
       (21.7, L.ZIN, HOLD), (21.85, L.ZFWD, FLOW), (22.0, _wave("L", 0.0), _HIT)]
    + _wave_keys("L", 22.2, 0.4, 4.0, 0.2)[1:] + _wave_keys("L", 23.0, 4.0, 0.0, 0.225)[1:]
    + [(24.12, L.ZFWD, FLOW)]
    + _roll_keys("L", 24.25, 24.8, 1.0 / 0.55, 1.0 / 0.55, 0.03, 0.13)
    + [(24.88, L.SWING, FLOW), (25.0, L.VUP, _HIT)]
    + _mill(L, 25.5, False) + _mill(L, 26.0, True) + _mill(L, 26.5, False)
    + [(26.78, L.DOWN, FLOW), (27.0, L.ZDN, _LOCK), (27.3, L.ZMID, FLOW), (27.5, L.ZUP, _LOCK), (28.0, L.TEMPLE, _LOCK),
       (28.5, L.OPENW, _HIT), (28.83, L.SWEEPB, FLOW), (29.0, L.XSH, _HIT), (29.3, L.IN, FLOW)]))

_ARM_R = _track("R", (
    _roll_keys("R", 29.5, 31.75, 1.0, 5.0 / 3.0, 0.02, 0.14)
    + [(1.87, R.IN, FLOW), (2.0, R.LOW, _HIT)]
    + _mill(R, 2.5, True) + _mill(R, 3.0, False) + _mill(R, 3.5, True)
    + [(4.0, R.ZUP, _LOCK), (4.5, R.ZUPB, _LOCK), (4.8, R.ZMID, FLOW), (5.0, R.ZDN, _LOCK), (5.5, R.ZIN, _LOCK),
       (5.8, R.ZFWD, FLOW)]
    + _wave_keys("R", 6.0, 0.0, 4.0)
    + [(7.6, R.T, HOLD), (7.83, R.SWEEP, FLOW), (8.0, R.XSH, _HIT), (8.33, R.SWEEP, FLOW), (8.5, R.T, _HIT),
       (8.8, R.DOWN, FLOW), (9.0, R.XLOW, _HIT), (9.3, R.DOWN, FLOW), (9.5, R.LOWOUT, _HIT),
       (9.78, R.WIND, _Q), (9.88, R.SWING, FLOW), (10.0, R.VUP, _HIT), (10.3, R.PUMP, FLOW),
       (10.5, R.VUP, dict(dur=0.18, over=0.0))]
    + _mill(R, 11.0, False) + _mill(R, 11.5, True)
    + [(12.0, R.TEMPLE, HOLD), (12.3, R.ZMID, FLOW), (12.5, R.ZDN, _LOCK), (13.0, R.ZIN, _LOCK), (13.5, R.ZUPB, _LOCK),
       (14.3, R.T, dict(dur=0.35, over=0.0)), (14.76, R.T, HOLD)]
    + _wave_keys("R", 14.855, 3.5, -1.0)
    + [(16.0, R.LOWOUT, _HIT), (16.3, R.DOWN, FLOW), (16.5, R.XLOW, _HIT), (16.7, R.DOWN, FLOW),
       (16.86, R.LOWOUT, FLOW), (17.0, R.OPENW, _HIT), (17.33, R.SWEEPB, FLOW), (17.5, R.GUARD, _HIT), (17.87, R.SWING, FLOW), (18.0, R.VUP, _HIT)]
    + [(18.3, R.SWING, FLOW), (18.5, R.CHEEK, _Q), (18.75, R.TEMPLE, _Q), (19.0, R.CHEEK, _Q), (19.25, R.TEMPLE, _Q),
       (19.38, R.ZMID, FLOW), (19.5, R.ZDN, _LOCK), (19.8, R.ZMID, FLOW), (20.0, R.ZUP, _LOCK), (20.5, R.ZIN, _LOCK),
       (21.7, R.ZIN, HOLD), (21.85, R.ZFWD, FLOW), (22.0, _wave("R", 0.0), _HIT)]
    + _wave_keys("R", 22.2, 0.4, 4.0, 0.2)[1:] + _wave_keys("R", 23.0, 4.0, 0.0, 0.225)[1:]
    + [(24.12, R.ZFWD, FLOW)]
    + _roll_keys("R", 24.25, 24.8, 1.0 / 0.55, 1.0 / 0.55, 0.03, 0.13)
    + [(25.0, R.PTX, _HIT), (25.34, R.FACE, FLOW), (25.5, R.TEMPLE, _MILL)]
    + _mill(R, 26.0, False) + _mill(R, 26.5, True)
    + [(27.0, R.ZUP, _LOCK), (27.3, R.ZMID, FLOW), (27.5, R.ZDN, _LOCK), (27.8, R.ZMID, FLOW), (28.0, R.TEMPLE, _LOCK),
       (28.3, R.FACE, FLOW), (28.5, R.XSH, _HIT), (28.83, R.SWEEP, FLOW), (29.0, R.T, _HIT), (29.28, R.SWING, FLOW)]))

# ---------------------------------------------------------------- face
_BLINKS = (1.2, 4.4, 7.2, 9.3, 12.3, 15.6, 17.4, 20.6, 23.5, 26.7, 29.4)
_SHOUTS = (4.0, 5.0, 12.0, 13.0, 13.5, 19.5, 20.0, 22.0, 27.0, 27.5)       # a short "hey" on the locks
_LOOK = Track(pp([(0.0, (0.0, -0.1)), (2.0, (0.5, 0.6)), (2.6, (0.0, 0.0)), (6.0, (-0.6, 0.1)), (6.9, (0.0, 0.0)),
                  (7.5, (0.6, 0.1)), (8.5, (-0.5, 0.0)), (9.0, (0.0, -0.2)), (10.0, (0.4, 0.7)), (11.0, (0.0, 0.0)),
                  (13.0, (0.5, 0.0)), (13.5, (-0.5, 0.0)), (14.0, (0.6, 0.1)), (14.9, (0.0, 0.0)), (15.5, (-0.6, 0.1)),
                  (16.0, (0.0, 0.0)), (18.0, (0.5, 0.6)), (18.6, (0.0, 0.0)), (22.0, (0.0, 0.3)), (23.0, (0.0, -0.2)),
                  (25.0, (0.5, 0.6)), (25.6, (0.0, 0.0)), (28.5, (0.5, 0.0)), (29.0, (-0.5, 0.0)), (29.5, (0.0, -0.1))],
                 dur=0.3, antic=0.0, over=0.0))

_SPIN = (20.92, 21.96)


def pose(t):
    b = beats(t) % NB
    p = Pose()

    # ---- body: hips lead, rib cage and head follow a little later (overlap)
    v0, v1, v2 = _BODY(b), _BODY(b - 0.06), _BODY(b - 0.12)
    amp = _AMP(b)
    bn = bounce(b)
    bn2 = bounce(b, 0.5)
    sw = sway(b, 2.0, 0.25)
    hips_rot = v0[3:6] + np.array([0.02 * amp * bn, 0.0, 0.02 * amp * sw])
    body_rot = v1[6:9] + np.array([0.035 * amp * bounce(b - 0.08), 0.0, -0.03 * amp * sw])
    head_rot = v2[9:12] + np.array([0.07 * amp * bounce(b - 0.16, sharp=1.2) - 0.03 * amp, 0.0, 0.0])
    step = math.sin(math.pi * b)                                         # > 0: the left heel is the free one
    p.hips_pos = np.array([v0[0] - 0.015 * amp * step, v0[1], v0[2] - 0.04 * amp * bn - 0.012 * amp * bn2])
    set_torso(p, hips_rot, body_rot, head_rot)

    # ---- feet: on top of the scripted steps the heels pump alternately between the beats
    off = 0.4 * amp * (1.0 - bounce(b, sharp=0.9, skew=0.3))
    fl, fr = FL(b), FR(b)
    p.leg_L = to_leg("L", fl, knee_out=v0[12], heel=off * (0.5 + 0.5 * step))
    p.leg_R = to_leg("R", fr, knee_out=v0[13], heel=off * (0.5 - 0.5 * step))
    fit_hips(p)

    # ---- arms
    al = eval_arm("L", _ARM_L, b, lead=0.04, lag=0.06, drag=0.25)
    ar = eval_arm("R", _ARM_R, b, lead=0.04, lag=0.06, drag=0.25)
    bob = np.array([0.0, 0.0, 0.05 * amp * bn])                          # forearms bob with the groove
    al[3:6] += bob
    ar[3:6] += bob
    p.arm_L = to_arm("L", al)
    p.arm_R = to_arm("R", ar)

    # ---- the spin (whole character about the ball of the left foot)
    if _SPIN[0] < b < _SPIN[1]:
        spin_root(p, TAU * smootherstep(_SPIN[0], _SPIN[1], b), fl["ball"])

    # ---- face: cheeky grin, "whoo" at the bursts, a short "hey" on the locks
    whoo = max(hit(b, x, 0.22, 1.4) for x in BURSTS)
    hey = max(pbump(b, x, 0.3) for x in _SHOUTS)
    p.jaw = clamp(0.85 * whoo + 0.4 * hey + 0.08 * amp * bn)
    p.mouth_round = 0.8 * whoo
    p.smile = 0.8 - 0.45 * whoo
    p.blink_L = p.blink_R = blink(b, _BLINKS)
    lk = _LOOK(b)
    p.look = (float(lk[0]), float(lk[1]))
    br = 0.1 + 0.8 * whoo + 0.4 * hey
    p.brow_L = clamp(br + 0.25 * pbump(b, 13.0, 0.6) + 0.25 * pbump(b, 28.5, 0.6), -1.0, 1.0)
    p.brow_R = clamp(0.85 * br, -1.0, 1.0)
    return p
