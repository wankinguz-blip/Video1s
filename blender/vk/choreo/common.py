"""Choreography toolkit shared by the three routines.

Everything here is a pure function of time and exactly periodic over the loop (30 beats = 15 s),
so pose(t) == pose(t + 15) and every channel is C1 across the loop point.

Building blocks
---------------
Track        periodic keyframed channel (time in BEATS), Hermite interpolation; every key has a
             `snap` 0..1: 0 = the motion flows through the key (Catmull-Rom tangent), 1 = full
             stop at the key (ease out / ease in) -> pose-to-pose accents.
pp()         pose-to-pose key generator: hold (with a small anticipation drift), a fast move
             that lands ON the beat, a little overshoot and settle.
arm vectors  an arm is authored FK-style as two directions (upper arm, forearm) + explicit hand
             orientation; interpolating directions makes hands travel on arcs and an arm can
             never be over-stretched.  to_arm() converts to a rig.Arm (IK target in chest space,
             so limbs stay attached with exact bone lengths).
Foot         scripted foot: plants, steps (lifted arc), deliberate slides and twists that pivot
             on the ball or on the heel, so a planted foot can never skate.
fit_hips()   lowers the pelvis just enough for both legs to reach their feet.

Arm keys are written in mirrored "OFU" coordinates so one description serves both sides:
    (out, front, up):  out = away from the body's midline on that arm's own side,
                       front = towards the camera (character -Y),  up = +Z.
"""
import bisect
import math

import numpy as np

from ..config import BEAT, LOOP, LOOP_BEATS
from ..mathx import TAU, clamp, mod, smootherstep, smoothstep, wrap_delta
from ..rig import (ANKLE_H, FOOT_BALL, FOOT_HEEL, FOREARM, HEAD_DZ, HIP_X, HIP_Z, NECK_DZ, SHIN, THIGH,
                   UPPER_ARM, Arm, Leg, R, T, shoulder_rest, side_sign)

NB = float(LOOP_BEATS)
assert abs((LOOP / BEAT) - NB) < 1e-9
FOOT_LEN_BH = FOOT_BALL + FOOT_HEEL        # ball -> heel distance
LEG_REACH = (THIGH + SHIN) * 0.955         # comfortable planted-leg extension (below the soft-IK zone)
ARM_LEN = UPPER_ARM + FOREARM


def beats(t):
    """Seconds -> beats (not wrapped)."""
    return t / BEAT


# ---------------------------------------------------------------- small helpers
def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = math.sqrt(float(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def beat_phase(t, n=1.0, offset=0.0):
    """Phase 0..1 of a cycle lasting `n` beats; t in seconds."""
    return ((t / BEAT - offset) / n) % 1.0


def bounce(b, n=1.0, offset=0.0, sharp=1.6, skew=0.18):
    """Groove curve 0..1 with period `n` beats: 1 exactly on the beat (knees bent, the 'down'),
    0 between beats.  `skew` > 0 makes the drop into the beat quicker than the recovery."""
    ph = ((b - offset) / n) % 1.0
    ph = ph + skew * math.sin(TAU * ph) / TAU
    return (0.5 + 0.5 * math.cos(TAU * ph)) ** sharp


def sway(b, n=2.0, offset=0.0):
    """Sine -1..1 with a period of `n` beats (b in beats)."""
    return math.sin(TAU * (b - offset) / n)


def pbump(b, c, w, period=NB):
    """Periodic C1 bump 0..1 centred on beat c with half-width w beats."""
    d = wrap_delta(c, b, period)
    u = abs(d) / w
    if u >= 1.0:
        return 0.0
    k = 1.0 - u * u
    return k * k * k


def pwin(b, a, c, rise=0.5, fall=0.5, period=NB):
    """Periodic window: 0 -> 1 over [a, a+rise], 1 -> 0 over [c-fall, c] (beats, may wrap)."""
    x = a + mod(b - a, period)
    return smootherstep(a, a + rise, x) * (1.0 - smootherstep(c - fall, c, x))


def hit(b, at, attack=0.25, release=1.0, period=NB):
    """Asymmetric accent 0..1: fast ease-in finishing exactly at beat `at`, slow release."""
    d = wrap_delta(at, b, period)
    if d <= 0.0:
        return smootherstep(-attack, 0.0, d)
    return 1.0 - smootherstep(0.0, release, d)


def ring(b, at, freq=1.5, decay=3.0, length=2.0, period=NB):
    """Damped oscillation that starts at beat `at` (0 before it), used for settle / wobble:
    sin(2π·freq·x)·exp(-decay·x), windowed so that it is exactly 0 after `length` beats."""
    x = mod(b - at, period)
    if x >= length:
        return 0.0
    return math.sin(TAU * freq * x) * math.exp(-decay * x) * (1.0 - smootherstep(0.6 * length, length, x)) \
        * smoothstep(0.0, 0.08, x)


def blink(b, at, w=0.32):
    """Eyelid 0..1 for blinks centred on the beats in `at`."""
    v = 0.0
    for c in at:
        v = max(v, pbump(b, c, w))
    return v


def softmin(vals, k=0.008):
    m = min(vals)
    s = sum(math.exp(-(v - m) / k) for v in vals)
    return m - k * math.log(s)


# ---------------------------------------------------------------- periodic key tracks
class Track:
    """Periodic keyframed channel.  keys: [(beat, value[, snap])], value = number or vector.
    Hermite interpolation with weighted (non-uniform) Catmull-Rom tangents scaled by
    (1 - snap) of the key: C1 everywhere, including the loop point."""

    def __init__(self, keys, snap=1.0, period=NB):
        ks = []
        for k in keys:
            s = snap if len(k) < 3 or k[2] is None else k[2]
            ks.append((float(k[0]) % period, np.atleast_1d(np.asarray(k[1], dtype=np.float64)), float(s)))
        ks.sort(key=lambda k: k[0])
        self.period = period
        self.scalar = not hasattr(keys[0][1], "__len__")
        self.t = [k[0] for k in ks]
        n = self.n = len(ks)
        for i in range(1, n):
            if self.t[i] - self.t[i - 1] < 1e-4:
                raise ValueError(f"Track: two keys at beat {self.t[i]:.4f}")
        self.v = np.array([k[1] for k in ks])
        self.m = np.zeros_like(self.v)
        if n > 1:
            for i in range(n):
                h0 = self.t[i] - self.t[i - 1] + (period if i == 0 else 0.0)
                h1 = self.t[(i + 1) % n] - self.t[i] + (period if i == n - 1 else 0.0)
                d0 = (self.v[i] - self.v[i - 1]) / h0
                d1 = (self.v[(i + 1) % n] - self.v[i]) / h1
                self.m[i] = (d0 * h1 + d1 * h0) / (h0 + h1) * (1.0 - ks[i][2])

    def __call__(self, b):
        n = self.n
        if n == 1:
            return float(self.v[0][0]) if self.scalar else self.v[0].copy()
        x = b % self.period
        i = bisect.bisect_right(self.t, x) - 1
        j = (i + 1) % n
        ta = self.t[i]                       # i == -1 -> last key (python indexing), wrapped below
        tb = self.t[j]
        if i < 0:
            ta -= self.period
        elif j == 0:
            tb += self.period
        h = tb - ta
        u = (x - ta) / h
        u2 = u * u
        u3 = u2 * u
        out = (2 * u3 - 3 * u2 + 1) * self.v[i] + (u3 - 2 * u2 + u) * h * self.m[i] \
            + (-2 * u3 + 3 * u2) * self.v[j] + (u3 - u2) * h * self.m[j]
        return float(out[0]) if self.scalar else out


def pp(events, dur=0.3, antic=0.06, over=0.05, settle=0.3, snap=1.0, period=NB):
    """Pose-to-pose key generator.  events: [(beat, value[, opts])] sorted by beat; opts is a
    dict that may override dur / antic / over / settle / snap, or contain flow=True for a key the
    motion simply passes through (no hold; `snap` defaults to 0 for those).

    For a normal event the previous pose is held (drifting `antic` × the coming change in the
    opposite direction = anticipation), then a fast move of `dur` beats lands exactly on the
    beat, overshoots by `over` × the change and settles `settle` beats later."""
    ev = []
    for e in events:
        o = dict(e[2]) if len(e) > 2 and e[2] else {}
        ev.append((float(e[0]), np.atleast_1d(np.asarray(e[1], dtype=np.float64)), o))
    n = len(ev)

    def _dur(i):
        """Start of the fast move into event i (absolute beat, un-wrapped for i == n)."""
        b, _v, o = ev[i % n]
        b += period if i >= n else 0.0
        if o.get("flow"):
            return b
        pb = ev[i - 1][0] - (period if i == 0 else 0.0)
        return b - min(o.get("dur", dur), (b - pb) * 0.85)

    keys = []
    for i in range(n):
        b, v, o = ev[i]
        pb, pv, _po = ev[i - 1]
        if i == 0:
            pb -= period
        if o.get("flow"):
            keys.append((b, v, o.get("snap", 0.0)))
            continue
        start = _dur(i)
        a = o.get("antic", antic)
        ov = o.get("over", over)
        st = o.get("settle", settle)
        sn = o.get("snap", snap)
        delta = v - pv
        prev_flow = bool(_po.get("flow"))
        prev_end = pb if prev_flow else pb + _po.get("settle", settle)
        if not prev_flow and start - prev_end > 0.05 or prev_flow and start - pb > 0.3:
            keys.append((start, pv - a * delta, 1.0))       # hold (with anticipation drift)
        nxt = ev[(i + 1) % n][2]
        nstart = _dur(i + 1) if not nxt.get("flow") else ev[(i + 1) % n][0] + (period if i == n - 1 else 0.0)
        if ov and nstart - (b + st) > 0.05:
            keys.append((b, v + ov * delta, sn))
            keys.append((b + st, v, 1.0))
        else:
            keys.append((b, v, sn))
            ev[i][2]["settle"] = 0.0                        # tell the next event there is no settle key
    return keys


def track_pp(events, **kw):
    return Track(pp(events, **kw))


# ---------------------------------------------------------------- arms as vectors
# layout of an arm vector (all directions in CHEST space of the actual side):
#   0:3 upper-arm direction   3:6 forearm direction   6:9 finger direction   9:12 palm normal
#   12:16 curls (thumb, f1, f2, f3)   16 spread   17 shrug   18 shoulder forward
ARM_N = 19


def _ofu(side, v):
    return np.array([side_sign(side) * v[0], -v[1], v[2]], dtype=np.float64)


def _curls(curl):
    return [float(curl)] * 4 if not hasattr(curl, "__len__") else [float(c) for c in curl]


def natural_palm(side, u, f):
    """Palm normal of a relaxed hand (neutral forearm rotation) for upper arm u / forearm f."""
    ax = np.cross(f, u)
    n = float(np.linalg.norm(ax))
    if n < 0.3:                                             # nearly straight arm: pretend it bends 'forward'
        ref = np.array([0.0, -1.0, 0.0]) - u * (-u[1])
        if np.linalg.norm(ref) < 0.2:
            ref = np.array([0.0, 0.0, 1.0]) - u * u[2]
        ax = ax + (0.3 - n) * unit(np.cross(unit(ref), u))
    return unit(ax) * (-side_sign(side))


def _avec(side, u, f, fingers, palm, flex, roll, curl, spread, shrug, forward):
    s = side_sign(side)
    fg = f.copy() if fingers is None else fingers
    pm = natural_palm(side, u, f) if palm is None else palm
    pm = unit(pm - fg * float(np.dot(pm, fg)))
    if roll:                                                # + = palm turns towards 'front/up' (supination)
        c, sn = math.cos(roll * s), math.sin(roll * s)
        pm = pm * c + np.cross(fg, pm) * sn
    if flex:                                                # + = wrist bends towards the palm
        c, sn = math.cos(flex), math.sin(flex)
        fg, pm = fg * c + pm * sn, pm * c - fg * sn
    return np.concatenate([u, f, fg, pm, _curls(curl), [spread, shrug, forward]])


def A(side, u, f, fingers=None, palm=None, flex=0.0, roll=0.0, curl=0.3, spread=0.0, shrug=0.0, forward=0.0):
    """Arm vector from the directions of the upper arm `u` and the forearm `f` (OFU coordinates).
    `fingers` / `palm` (OFU) override the natural hand orientation; `flex`, `roll` adjust it."""
    u, f = unit(_ofu(side, u)), unit(_ofu(side, f))
    fg = None if fingers is None else unit(_ofu(side, fingers))
    pm = None if palm is None else unit(_ofu(side, palm))
    return _avec(side, u, f, fg, pm, flex, roll, curl, spread, shrug, forward)


def solve_uf(side, wrist, pole, shrug=0.0, forward=0.0, max_ext=0.975):
    """Two-bone solve in chest space (actual coordinates): wrist target + elbow direction hint
    -> (u, f) unit directions.  Targets beyond max_ext × arm length are pulled in."""
    sh = shoulder_rest(side) + np.array([0.0, -forward, shrug])
    d = np.asarray(wrist, dtype=np.float64) - sh
    dist = float(np.linalg.norm(d))
    dirv = d / dist if dist > 1e-6 else np.array([0.0, 0.0, -1.0])
    dist = clamp(dist, abs(UPPER_ARM - FOREARM) * 1.1 + 0.01, ARM_LEN * max_ext)
    p = np.asarray(pole, dtype=np.float64)
    p = p - dirv * float(np.dot(p, dirv))
    if np.linalg.norm(p) < 1e-5:
        p = np.cross(dirv, [1.0, 0.0, 0.0])
    p = unit(p)
    ca = clamp((UPPER_ARM ** 2 + dist ** 2 - FOREARM ** 2) / (2 * UPPER_ARM * dist), -1.0, 1.0)
    sa = math.sqrt(max(0.0, 1.0 - ca * ca))
    u = dirv * ca + p * sa
    f = unit(dirv * dist - u * UPPER_ARM)
    return u, f


def reach(side, wrist, elbow=(0.7, -0.5, -0.6), fingers=None, palm=None, flex=0.0, roll=0.0, curl=0.3,
          spread=0.0, shrug=0.0, forward=0.0):
    """Arm vector from a wrist position (OFU, metres, relative to the CHEST joint: the shoulder is
    at (0.20, 0, 0.13), the mouth of an upright head at (0, 0.285, 0.345)) and an elbow direction
    hint (OFU)."""
    u, f = solve_uf(side, _ofu(side, wrist), _ofu(side, elbow), shrug, forward)
    fg = None if fingers is None else unit(_ofu(side, fingers))
    pm = None if palm is None else unit(_ofu(side, palm))
    return _avec(side, u, f, fg, pm, flex, roll, curl, spread, shrug, forward)


def reach_xyz(side, wrist, pole, fingers, palm, curl=0.3, spread=0.0, shrug=0.0, forward=0.0):
    """Like reach() but everything in actual chest coordinates (for run-time targets that come
    from another space, e.g. the head)."""
    u, f = solve_uf(side, wrist, pole, shrug, forward)
    return _avec(side, u, f, unit(fingers), np.asarray(palm, dtype=np.float64), 0.0, 0.0, curl, spread, shrug, forward)


def amix(a, b, w):
    """Blend two arm vectors (directions are re-normalised by to_arm)."""
    if w <= 0.0:
        return a
    if w >= 1.0:
        return b
    return a + (b - a) * w


def arm_wrist(side, a):
    """Wrist position in chest space of an arm vector."""
    sh = shoulder_rest(side) + np.array([0.0, -a[18], a[17]])
    return sh + unit(a[0:3]) * UPPER_ARM + unit(a[3:6]) * FOREARM


class ArmTrack:
    """Periodic track of arm vectors.  Directions of the upper arm / forearm, curls etc. are
    Hermite-interpolated like in Track; the HAND ORIENTATION (fingers + palm) is interpolated as a
    quaternion, so a hand that has to turn far between two keys rotates the short way round
    instead of collapsing / flipping (which is what blending two opposite direction vectors
    does).  Quaternion signs are made consistent along the chain of keys; if the chain comes back
    with the opposite sign the track is internally laid out over two loops (q and -q are the same
    rotation, so the result is still exactly periodic over one loop)."""

    def __init__(self, side, keys, snap=1.0, period=NB):
        self.side = side
        ks = []
        for k in keys:
            sn = snap if len(k) < 3 or k[2] is None else k[2]
            v = np.asarray(k[1], dtype=np.float64)
            ks.append([float(k[0]) % period, np.concatenate([v[0:6], _hand_quat(side, v[6:9], v[9:12]), v[12:19]]), sn])
        ks.sort(key=lambda k: k[0])
        for i in range(1, len(ks)):
            if float(ks[i][1][6:10] @ ks[i - 1][1][6:10]) < 0.0:
                ks[i][1][6:10] *= -1.0
        if float(ks[0][1][6:10] @ ks[-1][1][6:10]) < 0.0:   # odd parity: second loop with -q
            more = []
            for t_, v, sn in ks:
                w = v.copy()
                w[6:10] *= -1.0
                more.append([t_ + period, w, sn])
            ks = ks + more
            period = 2.0 * period
        self.keys = ks
        self.track = Track([(k[0], k[1], k[2]) for k in ks], period=period)

    def __call__(self, b):
        v = self.track(b)
        fg, pm = _quat_hand(self.side, v[6:10])
        return np.concatenate([v[0:6], fg, pm, v[10:17]])

    def warnings(self, limit=115.0):
        """Key pairs between which the upper arm or the forearm turns by more than `limit`
        degrees (the blend of the two directions then takes an unpredictable short cut)."""
        out = []
        n = len(self.keys)
        for i in range(n):
            a, b = self.keys[i - 1], self.keys[i]
            for nm, sl in (("upper", slice(0, 3)), ("fore", slice(3, 6))):
                c = float(unit(a[1][sl]) @ unit(b[1][sl]))
                ang = math.degrees(math.acos(clamp(c, -1.0, 1.0)))
                if ang > limit:
                    out.append((a[0], b[0], nm, ang))
        return out


def _hand_quat(side, fg, pm):
    """Hand orientation (fingers, palm; chest space) -> quaternion (w, x, y, z) of the hand frame."""
    s = side_sign(side)
    y = unit(fg)
    x = -s * np.asarray(pm, dtype=np.float64)
    x = unit(x - y * float(x @ y))
    z = np.cross(x, y)
    m00, m11, m22 = x[0], y[1], z[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        k = math.sqrt(tr + 1.0) * 2.0
        q = (0.25 * k, (y[2] - z[1]) / k, (z[0] - x[2]) / k, (x[1] - y[0]) / k)
    elif m00 > m11 and m00 > m22:
        k = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = ((y[2] - z[1]) / k, 0.25 * k, (y[0] + x[1]) / k, (z[0] + x[2]) / k)
    elif m11 > m22:
        k = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = ((z[0] - x[2]) / k, (y[0] + x[1]) / k, 0.25 * k, (z[1] + y[2]) / k)
    else:
        k = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = ((x[1] - y[0]) / k, (z[0] + x[2]) / k, (z[1] + y[2]) / k, 0.25 * k)
    return np.array(q)


def _quat_hand(side, q):
    """Quaternion -> (fingers, palm) unit vectors."""
    n = math.sqrt(float(q @ q))
    w, x, y, z = q / n if n > 1e-9 else (1.0, 0.0, 0.0, 0.0)
    xa = np.array([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)])
    ya = np.array([2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)])
    return ya, -side_sign(side) * xa


def eval_arm(side, track, b, lead=0.06, lag=0.08, drag=0.0):
    """Evaluate an arm track with overlap: the upper arm leads by `lead` beats, the hand
    orientation and fingers trail by `lag` beats, and the hand drags behind the wrist's
    velocity (`drag` = lean-back in radians per (m / beat))."""
    a = track(b)
    if lead:
        a[0:3] = track(b + lead)[0:3]
    if lag:
        a[6:16] = track(b - lag)[6:16]
    if drag:
        h = 0.08
        prev = track(b - h)
        if lead:
            prev[0:3] = track(b - h + lead)[0:3]
        v = (arm_wrist(side, a) - arm_wrist(side, prev)) / h
        fg = unit(a[6:9])
        v = v - fg * float(np.dot(v, fg))
        sp = float(np.linalg.norm(v))
        if sp > 1e-6:
            a[6:9] = unit(fg - v / sp * math.tanh(sp * drag))      # saturates at ~45° of lean-back
    return a


_POLE_HINT = {"L": np.array([0.35, 0.75, -0.55]), "R": np.array([-0.35, 0.75, -0.55])}


def to_arm(side, a):
    """Arm vector -> rig.Arm (IK target and pole in chest space, explicit hand orientation).

    The reach is soft-limited to 97.7 % of the arm length, so blended / overshooting keys can
    never ask the IK for a stretched arm."""
    u, f = unit(a[0:3]), unit(a[3:6])
    shrug, forward = float(a[17]), float(a[18])
    sh = shoulder_rest(side) + np.array([0.0, -forward, shrug])
    reach_v = u * UPPER_ARM + f * FOREARM
    r = math.sqrt(float(reach_v @ reach_v))
    d = reach_v / r
    rr = r / ARM_LEN
    if rr > 0.95:
        reach_v = d * ARM_LEN * (0.95 + 0.035 * (1.0 - math.exp(-(rr - 0.95) / 0.035)))
    off = u - d * float(np.dot(u, d))                       # elbow offset from the shoulder-wrist line
    hint = _POLE_HINT[side]
    hint = hint - d * float(np.dot(hint, d))
    pole = off + 0.05 * unit(hint)
    fg = unit(a[6:9])
    pm = a[9:12] - fg * float(np.dot(a[9:12], fg))
    return Arm(hand=sh + reach_v, pole=pole, space="chest", fingers=fg, palm=unit(pm),
               curl=[float(c) for c in a[12:16]], spread=float(a[16]), shrug=shrug, forward=forward)


# ---------------------------------------------------------------- torso
FLOW = dict(flow=True)                       # pp() option: the motion passes through this key
HOLD = dict(flow=True, snap=1.0)             # pp() option: stay in the previous pose until this beat


def body_key(hx=0.0, hy=0.0, hz=-0.03, hips=(0.0, 0.0, 0.0), body=(0.0, 0.0, 0.0), head=(0.0, 0.0, 0.0), ko=(0.0, 0.0)):
    """Body key vector (14): pelvis offset x, y, z | pelvis (bend, twist, lean) | rib cage relative
    to the pelvis | head relative to the stage | knee-out left, right (radians)."""
    return np.array([hx, hy, hz, *hips, *body, *head, *ko], dtype=np.float64)


def set_torso(p, hips=(0.0, 0.0, 0.0), body=(0.0, 0.0, 0.0), head=(0.0, 0.0, 0.0), comp=(0.6, 0.8, 0.6)):
    """Distribute rotations over the spine.  hips: pelvis (bend, twist, lean).  body: rotation of
    the rib cage relative to the pelvis (40 % spine, 60 % chest).  head: attitude of the head
    relative to the *stage*, of which `comp` (per axis) is compensated for what the pelvis and
    the rib cage already do; shared 35 % neck / 65 % head."""
    hips = np.asarray(hips, dtype=np.float64)
    body = np.asarray(body, dtype=np.float64)
    h = np.asarray(head, dtype=np.float64) - np.asarray(comp) * (hips + body)
    p.hips_rot = tuple(hips)
    p.spine_rot = tuple(0.4 * body)
    p.chest_rot = tuple(0.6 * body)
    p.neck_rot = tuple(0.35 * h)
    p.head_rot = tuple(0.65 * h)


def head_in_chest(p):
    """4×4 frame of the head bone relative to the chest bone for the pose's neck / head angles."""
    return T((0.0, 0.0, NECK_DZ * (1.0 - p.squash))) @ R(p.neck_rot) @ T((0.0, 0.0, HEAD_DZ)) @ R(p.head_rot)


# ---------------------------------------------------------------- feet
def _ease(u):
    return u * u * u * (u * (u * 6 - 15) + 10)


class Foot:
    """Scripted foot.  State = position of the BALL of the foot on the ground (x, y in root
    space), yaw, heel-raise and toe-raise angles.  Moves change the state over a beat interval
    with zero velocity at both ends; between moves the foot is planted and cannot skate.

        step(b0, b1, x, y, yaw, lift)       lifted step along an arc
        slide(b0, b1, x, y, yaw)            deliberate slide along the ground (recorded in .slides)
        twist(b0, b1, yaw, pivot)           swivel on the ball ('ball': heel swings, heel lifts a
                                            little) or on the heel ('heel': toes swing, toes lift)
        set(b0, b1, heel=, toe=)            change heel / toe raise only
    A foot is only allowed to change (x, y) while it is in the air (step) or in a declared slide.
    Moves must be given in time order, must not overlap, and the final state has to equal the
    initial one (closed loop) – checked by close()."""

    def __init__(self, side, x, y, yaw=0.0, heel=0.0, toe=0.0):
        self.side = side
        self.init = dict(x=float(x), y=float(y), yaw=float(yaw), heel=float(heel), toe=float(toe), up=0.0)
        self.cur = dict(self.init)
        self.moves = []          # (b0, b1, kind, state0, state1, params)
        self.starts = []
        self.slides = []         # (b0, b1) windows of deliberate sliding

    def _add(self, b0, b1, kind, new, **par):
        assert b1 > b0, "move must have a duration"
        if self.moves:
            assert b0 >= self.moves[-1][1] - 1e-9, f"foot moves overlap at beat {b0}"
        s0 = dict(self.cur)
        s1 = dict(self.cur)
        s1.update({k: float(v) for k, v in new.items() if v is not None})
        self.moves.append((float(b0), float(b1), kind, s0, s1, par))
        self.starts.append(float(b0))
        self.cur = s1
        return self

    def step(self, b0, b1, x=None, y=None, yaw=None, lift=0.06, heel=None, toe=None, peak=0.5, toe_down=0.0, up=None):
        """`lift`: height of the arc on top of the linear path; `peak`: where in the move the arc
        is highest; `toe_down`: extra heel raise in flight (the foot points its toes); `up`: height
        at which the foot stays after the move (0 = lands; > 0 = the leg stays in the air, e.g. a
        kick or a tucked leg, until a later step brings it down with up=0)."""
        return self._add(b0, b1, "step", dict(x=x, y=y, yaw=yaw, heel=heel, toe=toe, up=up), lift=lift, peak=peak,
                         toe_down=toe_down)

    def slide(self, b0, b1, x=None, y=None, yaw=None, heel=None, toe=None):
        self.slides.append((float(b0), float(b1)))
        return self._add(b0, b1, "slide", dict(x=x, y=y, yaw=yaw, heel=heel, toe=toe))

    def twist(self, b0, b1, yaw, pivot="ball", heel=None, toe=None, pop=None):
        """`pop`: extra heel (ball pivot) or toe (heel pivot) raise in the middle of the twist."""
        if pop is None:
            pop = 0.22
        s0 = self.cur
        new = dict(yaw=yaw, heel=heel, toe=toe)
        if pivot == "heel":                                 # the ball travels on an arc around the heel
            hx = s0["x"] - math.sin(s0["yaw"]) * FOOT_LEN_BH
            hy = s0["y"] + math.cos(s0["yaw"]) * FOOT_LEN_BH
            new["x"] = hx + math.sin(yaw) * FOOT_LEN_BH
            new["y"] = hy - math.cos(yaw) * FOOT_LEN_BH
            return self._add(b0, b1, "twist_heel", new, pop=pop, heel_pt=(hx, hy))
        return self._add(b0, b1, "twist_ball", new, pop=pop)

    def set(self, b0, b1, heel=None, toe=None):
        return self._add(b0, b1, "set", dict(heel=heel, toe=toe))

    def close(self):
        for k, v in self.init.items():
            w = self.cur[k]
            if k == "yaw":
                w = v + wrap_delta(v, w, TAU)
            assert abs(v - w) < 1e-6, f"foot {self.side}: '{k}' does not return to its start ({w} vs {v})"
        return self

    def __call__(self, b):
        """-> dict(x, y (ANKLE ground projection), yaw, heel, toe, lift, ball=(x, y), air 0..1)."""
        x = b % NB
        i = bisect.bisect_right(self.starts, x) - 1
        lift = 0.0
        air = 0.0
        st = self.cur if i < 0 else None                    # before the first move = the closed-loop end state
        if i >= 0:
            b0, b1, kind, s0, s1, par = self.moves[i]
            if x >= b1:
                st = s1
            else:
                u = (x - b0) / (b1 - b0)
                e = _ease(u)
                st = {k: s0[k] + (s1[k] - s0[k]) * e for k in s0}
                if kind == "step":
                    pk = par["peak"]
                    w = u / pk * 0.5 if u < pk else 0.5 + (u - pk) / (1.0 - pk) * 0.5
                    arc = math.sin(math.pi * w) ** 2
                    lift = par["lift"] * arc
                    st["heel"] += par["toe_down"] * arc
                    air = arc
                elif kind == "twist_ball":
                    st["heel"] += par["pop"] * math.sin(math.pi * u) ** 2
                elif kind == "twist_heel":
                    hx, hy = par["heel_pt"]
                    st["x"] = hx + math.sin(st["yaw"]) * FOOT_LEN_BH
                    st["y"] = hy - math.cos(st["yaw"]) * FOOT_LEN_BH
                    st["toe"] += par["pop"] * math.sin(math.pi * u) ** 2
        yaw = st["yaw"]
        ax = st["x"] - math.sin(yaw) * FOOT_BALL            # ankle = ball + Rz(yaw)·(0, +FOOT_BALL)
        ay = st["y"] + math.cos(yaw) * FOOT_BALL
        return dict(x=ax, y=ay, yaw=yaw, heel=st["heel"], toe=st["toe"], lift=lift + st["up"], ball=(st["x"], st["y"]),
                    air=air)


def to_leg(side, st, knee_out=0.0, heel=0.0, toe=0.0, lift=0.0, roll=0.0, knee_fwd=None):
    """Foot state (from Foot.__call__) -> rig.Leg.  The knee points where the toes point (70 % of
    the yaw) plus `knee_out` radians outwards (negative = knee drops inwards).  `heel`, `toe`,
    `lift` are added to the scripted values (taps and bounces)."""
    s = side_sign(side)
    ang = 0.7 * st["yaw"] + s * knee_out
    knee = (math.sin(ang), -math.cos(ang), 0.0) if knee_fwd is None else knee_fwd
    return Leg(foot=(st["x"], st["y"]), lift=st["lift"] + lift, yaw=st["yaw"], heel=max(0.0, st["heel"] + heel),
               toe=max(0.0, st["toe"] + toe), roll=roll, knee=knee)


def ankle_pos(leg):
    """Ankle position in root space for a rig.Leg (same construction as the solver, roll ignored)."""
    cy, sy = math.cos(leg.yaw), math.sin(leg.yaw)
    x, y, z = leg.foot[0], leg.foot[1], ANKLE_H + leg.lift
    if leg.heel > 1e-5:                                     # rotate about the ball of the foot
        c, s = math.cos(leg.heel), math.sin(leg.heel)
        ly = -FOOT_BALL + (FOOT_BALL * c - ANKLE_H * s)
        lz = -ANKLE_H + (FOOT_BALL * s + ANKLE_H * c)
        x, y, z = x - sy * ly, y + cy * ly, z + lz
    if leg.toe > 1e-5:                                      # rotate about the back of the heel
        c, s = math.cos(leg.toe), math.sin(leg.toe)
        h = leg.heel
        ch, sh = math.cos(h), math.sin(h)
        # heel point relative to the ankle in the (pitched) foot frame, then un-pitch by toe
        hy, hz = FOOT_HEEL * ch + ANKLE_H * sh, FOOT_HEEL * sh - ANKLE_H * ch
        a_y, a_z = -FOOT_HEEL, ANKLE_H                      # ankle relative to the heel, foot axes
        ry, rz = a_y * c + a_z * s, -a_y * s + a_z * c      # Rx(-toe)
        ry, rz = ry * ch - rz * sh, ry * sh + rz * ch       # Rx(heel)
        ly, lz = hy + ry, hz + rz
        x, y, z = x - sy * ly, y + cy * ly, z + lz
    return np.array([x, y, z])


def fit_hips(p, reach=LEG_REACH, k=0.008):
    """Lower the pelvis (smoothly) until both legs can reach their foot targets with a little
    knee bend left.  Call after hips_pos / hips_rot / legs are set."""
    hp = np.asarray(p.hips_pos, dtype=np.float64)
    rot = R(p.hips_rot)[:3, :3]
    lims = [float(hp[2])]
    for side, leg in (("L", p.leg_L), ("R", p.leg_R)):
        d = rot @ np.array([side_sign(side) * HIP_X, 0.0, 0.0])
        a = ankle_pos(leg)
        dx, dy = hp[0] + d[0] - a[0], hp[1] + d[1] - a[1]
        h2 = reach * reach - dx * dx - dy * dy
        lims.append(a[2] - HIP_Z - d[2] + (math.sqrt(h2) if h2 > 0 else 0.0))
    p.hips_pos = np.array([hp[0], hp[1], softmin(lims, k)])
    return p


def spin_root(p, angle, pivot):
    """Turn the whole character by `angle` about a vertical axis through `pivot` (x, y in root
    space at angle 0), e.g. the ball of the supporting foot during a spin."""
    c, s = math.cos(angle), math.sin(angle)
    px, py = pivot
    p.root_yaw = angle
    p.root_pos = np.array([px - (c * px - s * py), py - (s * px + c * py), 0.0])
    return p
