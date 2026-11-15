"""Shared prop geometry: the contract between modelling (vk.mole / vk.props) and choreography.

Each prop is modelled in its own LOCAL space described here; choreography places it with
rig.Attach(parent, pos, rot / matrix) in pose.props[name].
"""
import math

import numpy as np

from .rig import Rx, TR

# ---------------------------------------------------------------- microphone ('mic', lead singer)
# Local space: origin at the centre of the grip, +Z along the handle towards the head (ball).
MIC = dict(
    handle_radius=0.021,
    handle_z=(-0.085, 0.085),
    head_center=(0.0, 0.0, 0.125),
    head_radius=0.043,
)
# Matrix for Attach('hand.<side>', matrix=MIC_IN_HAND[side]): the handle lies across the palm,
# the head comes out on the thumb side.  Use finger curl ≈ 0.85 for the grip.
MIC_IN_HAND = {
    "R": TR((0.034, 0.062, 0.0)) @ Rx(math.pi),
    "L": TR((-0.034, 0.062, 0.0)) @ Rx(math.pi),
}

# ---------------------------------------------------------------- saxophone ('sax')
# Local space: axes like character space when the sax is held neutrally in front of the player
# (+Z up, the player stands on the +Y side, the bell is forward = -Y).  Origin = tip of the
# mouthpiece (the point that goes into the mouth).
SAX = dict(
    mouthpiece=(0.0, 0.0, 0.0),
    neck_top=(0.0, -0.075, 0.015),           # top of the neck curve
    body_top=(0.0, -0.115, -0.07),
    body_bottom=(0.0, -0.14, -0.50),         # start of the bow
    bow_bottom=(0.0, -0.20, -0.57),          # lowest point
    bell_center=(0.0, -0.30, -0.36),         # centre of the bell opening
    bell_normal=(0.0, -0.55, 0.83),          # direction the bell faces (forward-up)
    bell_radius=0.075,
    body_radius=(0.022, 0.040),              # tube radius at body_top … body_bottom
    upper_grip=(0.0, -0.127, -0.21),         # LEFT hand: point on the tube axis under the upper keys
    lower_grip=(0.0, -0.137, -0.39),         # RIGHT hand: point on the tube axis under the lower keys
    key_side=(0.0, -1.0, 0.0),               # keys are on the front of the tube (away from the player)
)


def sax_point(name):
    return np.array(SAX[name], dtype=np.float64)
