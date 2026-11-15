"""Mole skeleton, pose description and analytic solver.

Why a custom solver instead of Blender IK constraints: every bone transform is a closed-form
function of the Pose, so any frame can be evaluated independently (exact 15 s loop, parallel /
resumable rendering) and limbs can never detach: bone lengths are constant and every limb starts
exactly at its parent joint.

Spaces
------
Character space ("root"): origin on the ground between the feet, +Z up, the character FACES -Y
(towards the camera), its LEFT side is +X (screen right), its RIGHT side is -X (screen left).
Side suffixes: ".L" = character's left (+X), ".R" = character's right (-X).

Local rotations are (bend, twist, lean) in radians:
  bend  > 0  pitch forward (towards -Y, i.e. towards the camera)
  twist > 0  turn towards the character's left (+X, screen right)
  lean  > 0  tilt towards the character's left (+X, screen right)
composed as  R = Rz(twist) · Rx(bend) · Ry(lean).

Frames
------
solve() returns world-space 4×4 frames per bone.  For the spine chain a frame is the joint origin
with accumulated rotation (identity rotation = rest).  For limb segments the frame is at the
joint, +Y along the bone, X = bend axis.  For hands: +Y towards the finger tips, the palm faces
-X on the left hand and +X on the right hand, the thumb is on the -Z side (see hand_axes()).
For feet: character axes rotated by the foot's yaw / pitch, origin at the ankle.
"""
import math

import numpy as np

from .mathx import clamp

# ---------------------------------------------------------------- proportions (metres, character space)
ANKLE_H = 0.07
THIGH = 0.235
SHIN = 0.215
HIP_Z = ANKLE_H + THIGH + SHIN          # 0.52 (legs straight in the rest pose)
HIP_X = 0.11
SPINE_DZ = 0.10                         # hips  → spine
CHEST_DZ = 0.14                         # spine → chest
NECK_DZ = 0.18                          # chest → neck
HEAD_DZ = 0.06                          # neck  → head pivot (base of the skull)
HEAD_Z = HIP_Z + SPINE_DZ + CHEST_DZ + NECK_DZ + HEAD_DZ      # 1.00
HEAD_TOP = 0.56                         # head pivot → top of the head  (total height ≈ 1.56)
SHOULDER = (0.20, 0.0, 0.13)            # chest joint → left shoulder joint (mirror x for right)
UPPER_ARM = 0.25
FOREARM = 0.235
HAND = 0.12                             # wrist → finger tips
ARM_REST_ANGLE = math.radians(32)       # A-pose: arms hang 32° away from vertical
FOOT_HEEL = 0.07                        # ankle → back of the heel (+Y)
FOOT_BALL = 0.14                        # ankle → ball of the foot (-Y)
FOOT_TOE = 0.26                         # ankle → tip of the shoe (-Y)

# Anatomy landmarks in HEAD space (origin = head pivot, character axes).  Modelling and
# choreography both rely on these.
HEAD = dict(
    center=(0.0, -0.02, 0.28),          # centre of the head volume
    radii=(0.31, 0.27, 0.28),           # half width (x), half depth (y), half height (z)
    mouth=(0.0, -0.285, 0.105),         # centre of the mouth opening (front surface)
    nose=(0.0, -0.36, 0.20),            # tip of the nose
    jaw_pivot=(0.0, -0.06, 0.10),       # hinge of the lower jaw
    eye_L=(0.115, -0.25, 0.33),
    eye_R=(-0.115, -0.25, 0.33),
    hat_base=(0.0, -0.01, 0.45),        # centre of a hat's sweat band (hat sits on top of the head)
    brim_front_R=(-0.20, -0.30, 0.47),  # where the right hand pinches a fedora brim
    brim_front_L=(0.20, -0.30, 0.47),
    top=(0.0, -0.02, HEAD_TOP),
)

# finger layout in HAND space: (name, base position, length of each of the 2 segments)
# index 0 = thumb, 1..3 = fingers (index, middle, little)
FINGERS = [
    ("thumb", (0.0, 0.035, -0.040), 0.038),
    ("f1", (0.0, 0.075, -0.028), 0.030),
    ("f2", (0.0, 0.082, 0.000), 0.032),
    ("f3", (0.0, 0.074, 0.028), 0.028),
]

SIDES = ("L", "R")


def side_sign(side):
    return 1.0 if side == "L" else -1.0


def hand_axes(side):
    """Unit axes of a hand frame, in HAND space: fingers, palm normal (direction the palm faces),
    thumb side."""
    return dict(fingers=np.array([0.0, 1.0, 0.0]), palm=np.array([-side_sign(side), 0.0, 0.0]),
                thumb=np.array([0.0, 0.0, -1.0]))


# ---------------------------------------------------------------- small linear algebra (numpy 4×4)
def vec(*a):
    return np.array(a[0] if len(a) == 1 else a, dtype=np.float64)


def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v * 0.0


def T(p):
    m = np.eye(4)
    m[:3, 3] = p
    return m


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def R(rot):
    """(bend, twist, lean) → 4×4 rotation, see module docstring."""
    bend, twist, lean = rot
    return Rz(twist) @ Rx(bend) @ Ry(lean)


def TR(pos, rot=(0.0, 0.0, 0.0)):
    return T(pos) @ R(rot)


def xf(m, p):
    """Transform point p by 4×4 m."""
    return m[:3, :3] @ np.asarray(p, dtype=np.float64) + m[:3, 3]


def xfd(m, d):
    """Transform direction d by 4×4 m."""
    return m[:3, :3] @ np.asarray(d, dtype=np.float64)


def frame(origin, x, y, z):
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, origin
    return m


def aim(origin, y_dir, x_hint):
    """Frame at origin with +Y along y_dir and +X as close as possible to x_hint."""
    y = norm(np.asarray(y_dir, dtype=np.float64))
    x = np.asarray(x_hint, dtype=np.float64)
    x = x - y * float(np.dot(x, y))
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(y, [0.0, 0.0, 1.0])
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0.0, 0.0])
    x = norm(x)
    return frame(origin, x, y, np.cross(x, y))


def rot_between(a, b):
    """4×4 minimal rotation taking unit vector a onto unit vector b."""
    a, b = norm(np.asarray(a, dtype=np.float64)), norm(np.asarray(b, dtype=np.float64))
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    m = np.eye(4)
    if c < -0.999999:
        return m
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    m[:3, :3] = np.eye(3) + vx + vx @ vx / (1.0 + c)
    return m


def inv(m):
    """Inverse of a rigid 4×4."""
    r = m[:3, :3].T
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = -r @ m[:3, 3]
    return out


def soft_reach(dist, max_len, soft=0.06):
    """Soft IK: approaches max_len smoothly instead of snapping straight at full extension."""
    da = max_len * (1.0 - soft)
    if dist <= da:
        return dist
    return da + max_len * soft * (1.0 - math.exp(-(dist - da) / (max_len * soft)))


def two_bone(a, target, l1, l2, pole, soft=0.03):
    """Analytic two-bone IK.  a = root joint, target = desired end, pole = direction the middle
    joint should point to.  Returns (mid, end, bend_axis) with exact bone lengths."""
    a = np.asarray(a, dtype=np.float64)
    d = np.asarray(target, dtype=np.float64) - a
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        d, dist = np.array([0.0, 0.0, -1.0]), 1e-6
    dirv = d / dist
    reach = soft_reach(dist, (l1 + l2) * 0.9995, soft) if soft > 0 else min(dist, l1 + l2)
    reach = max(reach, abs(l1 - l2) * 1.02 + 1e-4)
    p = np.asarray(pole, dtype=np.float64)
    p = p - dirv * float(np.dot(p, dirv))
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(dirv, [1.0, 0.0, 0.0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(dirv, [0.0, 1.0, 0.0])
    p = norm(p)
    cos_a = clamp((l1 * l1 + reach * reach - l2 * l2) / (2 * l1 * reach), -1.0, 1.0)
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    mid = a + dirv * (l1 * cos_a) + p * (l1 * sin_a)
    end = a + dirv * reach
    axis = norm(np.cross(dirv, p))
    return mid, end, axis


# ---------------------------------------------------------------- pose description
class Arm:
    """One arm.  IK form: `hand` = wrist target, `pole` = direction the elbow points to, both in
    `space` ('root' | 'hips' | 'chest' | 'head' | 'world' | 'prop:<name>' | 'hand.L' | 'hand.R').
    Use arm_fk() for a forward-kinematics description (it produces the same structure).

    Hand orientation: by default the hand continues the forearm; `flex` bends the wrist towards
    the palm (radians), `side` deviates towards the thumb, `roll` twists about the forearm.
    Or give explicit `fingers` (direction of the finger tips) and `palm` (direction the palm
    faces) vectors in `space`.
    `curl`: finger curl 0 (open) … 1 (fist) — one number or [thumb, f1, f2, f3].
    `spread`: fan the fingers apart (0 … 1).
    """

    def __init__(self, hand=(0.0, 0.0, 0.0), pole=(0.0, 1.0, -0.5), space="root", flex=0.0, side=0.0, roll=0.0,
                 fingers=None, palm=None, curl=0.25, spread=0.0, shrug=0.0, forward=0.0):
        self.hand = vec(hand)
        self.pole = vec(pole)
        self.space = space
        self.flex, self.side, self.roll = flex, side, roll
        self.fingers = None if fingers is None else vec(fingers)
        self.palm = None if palm is None else vec(palm)
        self.curl = curl
        self.spread = spread
        self.shrug = shrug          # shoulder raise in metres (+ up)
        self.forward = forward      # shoulder push forward in metres (+ towards -Y)

    def curls(self):
        c = self.curl
        return [float(c)] * 4 if not hasattr(c, "__len__") else [float(v) for v in c]


class Leg:
    """One leg, in ROOT space.  `foot` = (x, y) of the ankle's ground projection, `lift` = height
    of the sole above the ground, `yaw` = toe-out angle (+ = toes towards the character's left),
    `heel` = heel-raise angle (pivot at the ball of the foot, toes stay on the ground),
    `toe` = toe-raise angle (pivot at the heel), `roll` = sideways tilt, `knee` = direction the
    knee points to (root space)."""

    def __init__(self, foot=(0.0, 0.0), lift=0.0, yaw=0.0, heel=0.0, toe=0.0, roll=0.0, knee=(0.0, -1.0, 0.0)):
        self.foot = (float(foot[0]), float(foot[1]))
        self.lift, self.yaw, self.heel, self.toe, self.roll = lift, yaw, heel, toe, roll
        self.knee = vec(knee)


class Attach:
    """A prop rigidly attached to a bone frame (or 'world' / 'root'): frame = parent · matrix."""

    def __init__(self, parent, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), matrix=None):
        self.parent = parent
        self.matrix = np.array(matrix, dtype=np.float64) if matrix is not None else TR(pos, rot)


class Pose:
    """Complete pose of one character.  All fields have neutral defaults (relaxed standing)."""

    def __init__(self):
        self.root_pos = vec(0.0, 0.0, 0.0)      # offset of the whole character from its mark (root axes)
        self.root_yaw = 0.0                     # turn of the whole character (+ = to its left)
        self.hips_pos = vec(0.0, 0.0, -0.02)    # offset of the pelvis from its rest position (root space)
        self.hips_rot = (0.0, 0.0, 0.0)         # (bend, twist, lean)
        self.spine_rot = (0.0, 0.0, 0.0)
        self.chest_rot = (0.0, 0.0, 0.0)
        self.neck_rot = (0.0, 0.0, 0.0)
        self.head_rot = (0.0, 0.0, 0.0)
        self.arm_L = arm_fk("L", swing=0.0, raise_=0.12, elbow=0.25)
        self.arm_R = arm_fk("R", swing=0.0, raise_=0.12, elbow=0.25)
        self.leg_L = Leg((HIP_X + 0.02, 0.0), yaw=0.22)
        self.leg_R = Leg((-HIP_X - 0.02, 0.0), yaw=-0.22)
        # face
        self.jaw = 0.0                          # 0 closed … 1 wide open
        self.mouth_round = 0.0                  # 0 wide … 1 rounded "o"
        self.smile = 0.5                        # 0 neutral … 1 big smile
        self.blink_L = 0.0                      # 0 open … 1 closed
        self.blink_R = 0.0
        self.brow_L = 0.0                       # -1 frown … +1 raised
        self.brow_R = 0.0
        self.look = (0.0, 0.0)                  # pupil offset (-1..1 right/left, -1..1 down/up)
        self.cheeks = 0.0                       # 0 … 1 puffed cheeks (saxophonist)
        self.squash = 0.0                       # body squash (+) / stretch (-), about ±0.1
        # props and extras
        self.props = {}                         # name → Attach
        self.extra = {}                         # extra bone name → (bend, twist, lean) or a 4×4 rotation

    def arm(self, side):
        return self.arm_L if side == "L" else self.arm_R

    def leg(self, side):
        return self.leg_L if side == "L" else self.leg_R


def shoulder_rest(side):
    s = side_sign(side)
    return vec(s * SHOULDER[0], SHOULDER[1], SHOULDER[2])      # in chest space


def arm_fk(side, swing=0.0, raise_=0.0, elbow=0.0, twist=0.0, space="chest", **kw):
    """Forward-kinematics arm → Arm (wrist target + elbow pole) in `space` (default chest).

    raise_ : 0 = upper arm hangs straight down, π/2 = horizontal to the side, π = straight up
    swing  : rotates the raised arm about the vertical axis: 0 = in the side plane,
             +π/2 = pointing forward (towards the camera), -π/2 = backward
    elbow  : flexion angle, 0 = straight arm, ~2.4 = fully bent
    twist  : rotation of the forearm about the upper arm: 0 = forearm bends forward/up (hand in
             front of the body), +π/2 = forearm bends inwards across the body, -π/2 = outwards
    Extra keyword arguments go to Arm (flex, side, roll, curl, spread, fingers, palm, shrug …).
    """
    s = side_sign(side)
    sh = shoulder_rest(side)
    # upper-arm direction
    down = vec(0.0, 0.0, -1.0)
    out = vec(s * math.cos(swing), -math.sin(swing), 0.0)
    u = norm(down * math.cos(raise_) + out * math.sin(raise_))
    # reference "forward" bend direction perpendicular to the upper arm
    fwd = vec(0.0, -1.0, 0.0)
    f = fwd - u * float(np.dot(fwd, u))
    if np.linalg.norm(f) < 1e-4:
        f = vec(0.0, 0.0, 1.0) - u * u[2]
    f = norm(f)
    inward = norm(np.cross(u, f)) * s          # points across the body for positive twist
    b = norm(f * math.cos(twist) + inward * math.sin(twist))
    fore = norm(u * math.cos(elbow) + b * math.sin(elbow))
    elbow_p = sh + u * UPPER_ARM
    wrist = elbow_p + fore * FOREARM
    return Arm(hand=wrist, pole=-b, space=space, **kw)


def mix(a, b, u):
    """Blend two numbers / tuples / arrays."""
    if isinstance(a, (int, float)):
        return a + (b - a) * u
    return type(a)(x + (y - x) * u for x, y in zip(a, b)) if isinstance(a, tuple) else np.asarray(a) + (np.asarray(b) - np.asarray(a)) * u


def mix_arm(a, b, u):
    """Blend two Arm descriptions given in the same space (positions and poles are blended, the
    solver then restores exact bone lengths, so blended arms stay attached)."""
    if u <= 0.0:
        return a
    if u >= 1.0:
        return b
    assert a.space == b.space, "mix_arm: both arms must use the same space"
    out = Arm(hand=a.hand + (b.hand - a.hand) * u, pole=norm(norm(a.pole) + (norm(b.pole) - norm(a.pole)) * u),
              space=a.space, flex=mix(a.flex, b.flex, u), side=mix(a.side, b.side, u), roll=mix(a.roll, b.roll, u),
              curl=[x + (y - x) * u for x, y in zip(a.curls(), b.curls())], spread=mix(a.spread, b.spread, u),
              shrug=mix(a.shrug, b.shrug, u), forward=mix(a.forward, b.forward, u))
    if a.fingers is not None and b.fingers is not None:
        out.fingers = norm(a.fingers + (b.fingers - a.fingers) * u)
        pa = a.palm if a.palm is not None else vec(0, 0, -1)
        pb = b.palm if b.palm is not None else vec(0, 0, -1)
        out.palm = norm(pa + (pb - pa) * u)
    elif (b.fingers is not None and u > 0.5) or (a.fingers is not None and u <= 0.5):
        src = b if u > 0.5 else a
        out.fingers, out.palm = src.fingers, src.palm
    return out


# ---------------------------------------------------------------- skeleton + solver
class Skeleton:
    """Rest skeleton (same for all three moles) + optional extra bones (hair, coat tails …).

    extra: list of (name, parent_bone, rest_position_in_character_space[, options]).  An extra
    bone's frame is parent_frame · T(rest offset) · R(pose.extra[name]).  Extra bones that are not
    set by the choreography get automatic follow-through motion (see vk.characters).
    """

    SPINE = ("hips", "spine", "chest", "neck", "head")

    def __init__(self, extra=()):
        self.extra = list(extra)
        self.rest_pose = Pose()
        self.rest_pose.hips_pos = vec(0.0, 0.0, 0.0)
        a = ARM_REST_ANGLE
        for s in SIDES:
            arm = arm_fk(s, raise_=a, elbow=0.0)
            arm.pole = vec(0.0, 1.0, 0.0)
            arm.curl = 0.0
            setattr(self.rest_pose, "arm_" + s, arm)
            setattr(self.rest_pose, "leg_" + s, Leg((side_sign(s) * HIP_X, 0.0)))
        self.rest_pose.smile = 0.0
        self.rest = self.solve(self.rest_pose, np.eye(4), _rest=True)
        self.bone_names = list(self.rest.keys())
        self.rest_inv = {k: inv(v) for k, v in self.rest.items()}

    # -- helpers
    def joint(self, name):
        """Rest position of a bone origin in character space."""
        return self.rest[name][:3, 3].copy()

    def bone_length(self, name):
        base = name.split(".")[0]
        for fname, _pos, seg in FINGERS:
            if base[:-1] == fname:
                return seg
        return _BONE_LEN.get(base, 0.08)

    # -- the solver
    def solve(self, pose, mark=None, _rest=False):
        """Pose → dict of world 4×4 frames (bones, plus 'root', 'prop:<name>' entries).

        mark: 4×4 world matrix of the character's mark (position on the hill + facing)."""
        F = {}
        self._resting = _rest
        M = np.eye(4) if mark is None else np.asarray(mark, dtype=np.float64)
        root = M @ T(pose.root_pos) @ Rz(pose.root_yaw)
        F["root"] = root
        sq = float(pose.squash)
        hips = root @ T(vec(0.0, 0.0, HIP_Z) + pose.hips_pos) @ R(pose.hips_rot)
        F["hips"] = hips
        F["spine"] = hips @ T((0.0, 0.0, SPINE_DZ * (1 - sq))) @ R(pose.spine_rot)
        F["chest"] = F["spine"] @ T((0.0, 0.0, CHEST_DZ * (1 - sq))) @ R(pose.chest_rot)
        F["neck"] = F["chest"] @ T((0.0, 0.0, NECK_DZ * (1 - sq))) @ R(pose.neck_rot)
        F["head"] = F["neck"] @ T((0.0, 0.0, HEAD_DZ)) @ R(pose.head_rot)
        F["jaw"] = F["head"] @ T(HEAD["jaw_pivot"]) @ Rx(0.55 * clamp(pose.jaw, 0.0, 1.2))
        F["world"] = np.eye(4)

        # props attached to anything but hands (hands need the arms, arms may need these props)
        pending = dict(pose.props)
        self._resolve_props(F, pending)

        # arms: first those that do not depend on the other hand
        order = sorted(SIDES, key=lambda s: pose.arm(s).space.startswith("hand."))
        for s in order:
            self._solve_arm(F, pose.arm(s), s)
            self._resolve_props(F, pending)
        if pending and not _rest:
            raise KeyError(f"props with unknown parents: { {k: v.parent for k, v in pending.items()} }")

        for s in SIDES:
            self._solve_leg(F, pose.leg(s), s, root, hips)

        for item in self.extra:
            name, parent, rest_pos = item[0], item[1], item[2]
            parent_rest = F[parent] if _rest else self.rest[parent]
            off = xf(inv(parent_rest), rest_pos)
            r = pose.extra.get(name, (0.0, 0.0, 0.0))
            F[name] = F[parent] @ T(off) @ (R(r) if len(r) == 3 else np.asarray(r, dtype=np.float64))
        del F["world"]
        return F

    @staticmethod
    def _resolve_props(F, pending):
        progress = True
        while progress:
            progress = False
            for name in list(pending):
                a = pending[name]
                if a.parent in F:
                    F["prop:" + name] = F[a.parent] @ a.matrix
                    del pending[name]
                    progress = True

    def _solve_arm(self, F, arm, side):
        s = side_sign(side)
        chest = F["chest"]
        sh_local = shoulder_rest(side) + vec(0.0, -arm.forward, arm.shrug)
        shoulder = xf(chest, sh_local)
        F["shoulder." + side] = frame(shoulder, chest[:3, 0], chest[:3, 1], chest[:3, 2])
        space = F[arm.space]
        target = xf(space, arm.hand)
        pole = xfd(space, arm.pole)
        elbow, wrist, axis = two_bone(shoulder, target, UPPER_ARM, FOREARM, pole, 0.0 if self._resting else 0.02)
        F["upper_arm." + side] = aim(shoulder, elbow - shoulder, axis)
        fore = aim(elbow, wrist - elbow, axis)
        F["forearm." + side] = fore
        if arm.fingers is not None:
            y = norm(xfd(space, arm.fingers))
            palm = xfd(space, arm.palm if arm.palm is not None else (0.0, 0.0, -1.0))
            hand = aim(wrist, y, -s * palm)
        else:
            # default: continue the forearm; palm faces the body side of the bend plane
            base = frame(wrist, fore[:3, 0], fore[:3, 1], fore[:3, 2])
            hand = base @ Ry(arm.roll) @ Rz(s * arm.flex) @ Rx(-arm.side)
        F["hand." + side] = hand
        curls = arm.curls()
        for i, (fname, base_pos, seg) in enumerate(FINGERS):
            c = clamp(curls[i], -0.3, 1.2)
            fan = arm.spread * (0.0 if i == 0 else (i - 2) * 0.28)
            if i == 0:   # thumb: opposes the fingers
                f1 = hand @ T(base_pos) @ Rx(-(0.6 + 0.4 * arm.spread)) @ Rz(s * (0.15 + 0.9 * c))
                f2 = f1 @ T((0.0, seg, 0.0)) @ Rz(s * 0.8 * c)
            else:
                f1 = hand @ T(base_pos) @ Rx(fan) @ Rz(s * 1.35 * c)
                f2 = f1 @ T((0.0, seg, 0.0)) @ Rz(s * 1.45 * c)
            F[f"{fname}a.{side}"] = f1
            F[f"{fname}b.{side}"] = f2

    def _solve_leg(self, F, leg, side, root, hips):
        s = side_sign(side)
        hip = xf(hips, (s * HIP_X, 0.0, 0.0))
        # foot orientation in root space, pivoting at the ball (heel raise) or the heel (toe raise)
        yaw = Rz(leg.yaw)
        ankle0 = vec(leg.foot[0], leg.foot[1], ANKLE_H + leg.lift)
        rot = yaw @ Ry(leg.roll)
        foot_m = T(ankle0) @ rot
        toe_m = None
        if leg.heel > 1e-5:
            ball = xf(foot_m, (0.0, -FOOT_BALL, -ANKLE_H))
            foot_m = T(ball) @ rot @ Rx(leg.heel) @ T((0.0, FOOT_BALL, ANKLE_H))
            toe_m = T(ball) @ rot                      # toes stay flat on the ground
        if leg.toe > 1e-5:
            heel = xf(foot_m, (0.0, FOOT_HEEL, -ANKLE_H))
            r4 = np.eye(4)
            r4[:3, :3] = foot_m[:3, :3]
            foot_m = T(heel) @ r4 @ Rx(-leg.toe) @ T((0.0, -FOOT_HEEL, ANKLE_H))
        foot = root @ foot_m
        ankle = foot[:3, 3]
        knee_dir = xfd(root, leg.knee)
        knee, ankle_s, axis = two_bone(hip, ankle, THIGH, SHIN, knee_dir, 0.0 if self._resting else 0.05)
        # if the target is out of reach the foot follows the leg (never detaches)
        foot = foot.copy()
        foot[:3, 3] = ankle_s
        F["thigh." + side] = aim(hip, knee - hip, axis)
        F["shin." + side] = aim(knee, ankle_s - knee, axis)
        F["foot." + side] = foot
        if toe_m is not None:
            toe = root @ toe_m
            toe[:3, 3] += ankle_s - ankle
        else:
            toe = foot @ T((0.0, -FOOT_BALL, -ANKLE_H))
        F["toe." + side] = toe

    def skin_matrices(self, frames):
        """World frames → per-bone skinning transforms A (rest character space → world)."""
        return {k: frames[k] @ self.rest_inv[k] for k in self.bone_names if k in frames}


_BONE_LEN = dict(root=0.2, hips=SPINE_DZ, spine=CHEST_DZ, chest=NECK_DZ, neck=HEAD_DZ, head=HEAD_TOP, jaw=0.2,
                 shoulder=0.08, upper_arm=UPPER_ARM, forearm=FOREARM, hand=0.07, thigh=THIGH, shin=SHIN,
                 foot=FOOT_BALL, toe=FOOT_TOE - FOOT_BALL)


_LIMB_BONES = ("upper_arm", "forearm", "hand", "thigh", "shin")
_FINGER_NAMES = tuple(f[0] for f in FINGERS)


# ---------------------------------------------------------------- Blender side
class Rig:
    """Blender armature for one character + per-frame application of solved frames.

    The armature object sits at the world origin; bones are parentless and receive world
    transforms directly (matrix_basis = rest⁻¹ · A · rest), so there is no dependency on
    evaluation order.  Meshes are bound with an Armature modifier and vertex groups named like
    the bones (see vk.skin).  Rigid parts and props are driven through attach().
    """

    def __init__(self, name, skeleton, mark_matrix, col):
        import bpy
        from mathutils import Matrix

        self.name = name
        self.sk = skeleton
        self.mark = np.asarray(mark_matrix, dtype=np.float64)
        self.col = col
        self._Matrix = Matrix
        self.attachments = []       # (object, bone, rest matrix in character space)
        self.props = {}             # prop name → [objects]
        self.frames = None
        arm = bpy.data.armatures.new(name + "_rig")
        obj = bpy.data.objects.new(name + "_rig", arm)
        col.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        for bname in skeleton.bone_names:
            f0 = skeleton.rest[bname]
            eb = arm.edit_bones.new(bname)
            eb.head = f0[:3, 3]
            length = max(0.02, skeleton.bone_length(bname))
            base = bname.split(".")[0]
            along_y = base in _LIMB_BONES or base[:-1] in _FINGER_NAMES
            eb.tail = f0[:3, 3] + (f0[:3, 1] if along_y else np.array([0.0, 0.0, 1.0])) * length
            eb.use_deform = True
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.hide_render = True
        obj.display_type = "WIRE"
        self.obj = obj
        self.rest_local = {b.name: np.array(b.matrix_local) for b in arm.bones}
        self.rest_local_inv = {k: np.linalg.inv(v) for k, v in self.rest_local.items()}
        self.pbones = {pb.name: pb for pb in obj.pose.bones}

    # -- binding
    def bind(self, obj, weights=None):
        """Bind a mesh object modelled in CHARACTER REST SPACE to the armature.
        weights: {bone name: per-vertex weight array}; if None the object's existing vertex
        groups are used."""
        if weights:
            n = len(obj.data.vertices)
            for bname, w in weights.items():
                if bname not in self.pbones:
                    raise KeyError(f"unknown bone '{bname}'")
                vg = obj.vertex_groups.get(bname) or obj.vertex_groups.new(name=bname)
                w = np.asarray(w, dtype=np.float64)
                assert len(w) == n
                for i in np.nonzero(w > 1e-4)[0]:
                    vg.add([int(i)], float(w[i]), "REPLACE")
        mod = obj.modifiers.new("Armature", "ARMATURE")
        mod.object = self.obj
        mod.use_deform_preserve_volume = True
        return mod

    def attach(self, obj, bone, rest_matrix=None):
        """Rigidly attach an object modelled / placed in CHARACTER REST SPACE to a bone.
        rest_matrix defaults to the object's current matrix_world."""
        m = np.array(obj.matrix_world) if rest_matrix is None else np.asarray(rest_matrix, dtype=np.float64)
        if bone not in self.sk.rest_inv:
            raise KeyError(f"unknown bone '{bone}'")
        self.attachments.append((obj, bone, m))

    def add_prop(self, name, objects):
        """Objects modelled in the prop's own local space; they follow pose.props[name]."""
        self.props[name] = list(objects)

    # -- per frame
    def apply(self, pose):
        frames = self.sk.solve(pose, self.mark)
        self.frames = frames
        A = self.sk.skin_matrices(frames)
        Matrix = self._Matrix
        for bname, pb in self.pbones.items():
            basis = self.rest_local_inv[bname] @ A[bname] @ self.rest_local[bname]
            pb.matrix_basis = Matrix(basis.tolist())
        for obj, bone, m in self.attachments:
            obj.matrix_world = Matrix((A[bone] @ m).tolist())
        for name, objs in self.props.items():
            key = "prop:" + name
            shown = key in frames
            mw = Matrix(frames[key].tolist()) if shown else None
            for o in objs:
                if o.hide_render == shown:
                    o.hide_render = not shown      # a prop the pose does not place is not drawn
                if shown:
                    o.matrix_world = mw
        return frames


def mark_matrix(pos, face_point=None):
    """World matrix of a mark: origin at pos, character's -Y pointing at face_point (xy only)."""
    x, y, z = pos
    yaw = 0.0
    if face_point is not None:
        dx, dy = face_point[0] - x, face_point[1] - y
        # character forward is -Y;  rotate so that (0,-1) maps onto (dx, dy)
        yaw = math.atan2(dx, -dy)
    return T((x, y, z)) @ Rz(yaw)
