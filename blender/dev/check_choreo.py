#!/usr/bin/env python3
"""Numeric checker for the choreographies (plain python3 + numpy, no Blender).

    python3 blender/dev/check_choreo.py [singer] [sax] [emo] [-v]

For every character it samples the whole loop with the analytic solver (vk.rig.Skeleton.solve)
and prints a PASS / FAIL table:

  loop        pose(0) vs pose(15) and pose(t) vs pose(t ± 15) (all bone frames)
  jumps       discontinuities in value (C0) of joint positions, bone rotations and face channels
  kinks       discontinuities in velocity (C1) of joint positions
              (found by refining the worst intervals: a real jump does not shrink with the step)
  speed/acc   largest per-frame displacement / second difference of wrists, ankles, head (info)
  reach       IK targets beyond the limb's reach, solved hand / foot not on its target
  collide     wrists, hands, finger tips, forearms, elbows and props inside the head, muzzle, hat,
              torso or thighs; hands / forearms through each other
  feet        planted feet sliding, feet under the ground, feet too far from the mark
  knees       knees pointing backwards or away from the direction of the foot
  props       microphone near the mouth while singing; saxophone in the mouth and in both hands
  cost        time per pose() call

-v prints the time (seconds / beat) of the worst sample for each row.
-c prints the collision timeline (which probe is inside which volume during which beats).
A choreography module may declare deliberate foot slides:  SLIDES = {"L": [(beat0, beat1)], "R": [...]}
and deliberate contacts:  CONTACTS = [(probe name, volume name, beat0, beat1)]  (e.g. the glove on the hat brim).
"""
import importlib
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vk import propspec, rig  # noqa: E402
from vk.config import BEAT, FPS, LOOP  # noqa: E402
from vk.rig import HEAD, xf  # noqa: E402

VERBOSE = "-v" in sys.argv
NAMES = [a for a in sys.argv[1:] if not a.startswith("-")] or ["singer", "sax", "emo"]
FRAMES = int(round(LOOP * FPS))
SUB = 4                                   # fine sampling: SUB samples per frame for the continuity tests
H = 1.0 / (FPS * SUB)

# ---------------------------------------------------------------- collision volumes
# (bone, kind, centre, size)   ellipsoid: size = radii;  cyl: size = (rx, ry, z0, z1) along the bone's Z
HEAD_VOL = [
    ("head", "ell", HEAD["center"], HEAD["radii"], "head"),
    ("head", "ell", (0.0, -0.24, 0.18), (0.11, 0.12, 0.08), "muzzle"),
]
HATS = {
    "singer": [("head", "cyl", HEAD["hat_base"], (0.37, 0.37, -0.02, 0.02), "hat brim"),
               ("head", "cyl", HEAD["hat_base"], (0.22, 0.20, 0.0, 0.22), "hat crown")],
    "sax": [("head", "cyl", HEAD["hat_base"], (0.30, 0.30, -0.02, 0.02), "hat brim"),
            ("head", "cyl", HEAD["hat_base"], (0.21, 0.19, 0.0, 0.18), "hat crown")],
    "emo": [("head", "ell", (-0.10, -0.33, 0.33), (0.20, 0.06, 0.21), "fringe")],      # the hanging part of outfits.emo_hair()
}
TORSO_VOL = [
    ("hips", "cyl", (0.0, 0.0, 0.0), (0.20, 0.17, -0.08, 0.10), "pelvis"),
    ("spine", "cyl", (0.0, 0.0, 0.0), (0.215, 0.185, 0.0, 0.14), "belly"),
    ("chest", "cyl", (0.0, 0.0, 0.0), (0.20, 0.165, 0.0, 0.12), "chest"),
]
TOL = 0.010                               # penetration tolerance (m)


def depth(frames, vol, p, r):
    """Penetration depth (m, > 0 = inside) of a sphere (p, r) into a volume."""
    bone, kind, c, size, _name = vol
    m = frames[bone]
    q = m[:3, :3].T @ (p - m[:3, 3]) - np.asarray(c)
    if kind == "ell":
        rr = np.asarray(size) + r
        k = math.sqrt(float(np.sum((q / rr) ** 2)))
        return (1.0 - k) * float(np.min(rr))
    rx, ry, z0, z1 = size
    k = math.sqrt((q[0] / (rx + r)) ** 2 + (q[1] / (ry + r)) ** 2)
    dr = (1.0 - k) * (min(rx, ry) + r)
    dz = min(q[2] - (z0 - r), (z1 + r) - q[2])
    return min(dr, dz)


def seg_dist(p1, q1, p2, q2):
    """Distance between two segments."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
    t = (b * s + f) / e if e > 1e-12 else 0.0
    if t < 0:
        t, s = 0.0, np.clip(-c / a, 0, 1)
    elif t > 1:
        t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm(p1 + d1 * s - p2 - d2 * t))


# ---------------------------------------------------------------- sampling
def tip(frames, bone, length):
    return xf(frames[bone], (0.0, length, 0.0))


def points(frames):
    """Named world points used for the motion tests."""
    P = {}
    for b in ("hips", "chest", "head"):
        P[b] = frames[b][:3, 3]
    P["head_top"] = xf(frames["head"], HEAD["top"])
    for s in ("L", "R"):
        P["elbow." + s] = frames["forearm." + s][:3, 3]
        P["wrist." + s] = frames["hand." + s][:3, 3]
        P["fingers." + s] = tip(frames, "f2b." + s, 0.032)
        P["knee." + s] = frames["shin." + s][:3, 3]
        P["ankle." + s] = frames["foot." + s][:3, 3]
        P["ball." + s] = frames["toe." + s][:3, 3]
    for k in frames:
        if k.startswith("prop:"):
            P[k] = frames[k][:3, 3]
            P[k + ".z"] = xf(frames[k], (0.0, -0.15, -0.3))
    return P


ROT_BONES = ["hips", "chest", "head"] + [f"{b}.{s}" for b in ("upper_arm", "forearm", "hand", "thigh", "shin", "foot")
                                         for s in ("L", "R")]
FACE = ("jaw", "mouth_round", "smile", "blink_L", "blink_R", "brow_L", "brow_R", "cheeks", "squash")


def face_vec(p):
    return np.array([float(getattr(p, k)) for k in FACE] + [float(p.look[0]), float(p.look[1])])


def rot_angle(a, b):
    c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) * 0.5
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class Row:
    def __init__(self):
        self.rows = []
        self.failed = False

    def add(self, name, value, limit, ok, where=None, unit=""):
        self.rows.append((name, value, limit, ok, where, unit))
        self.failed |= not ok

    def show(self, title):
        print(f"\n== {title} " + "=" * (66 - len(title)))
        for name, value, limit, ok, where, unit in self.rows:
            v = value if isinstance(value, str) else f"{value:.4g}{unit}"
            lim = "" if limit is None else (limit if isinstance(limit, str) else f"<= {limit:g}{unit}")
            w = ""
            if where is not None and (VERBOSE or not ok):
                w = f"  @ {where:.3f}s (beat {where / BEAT:.2f})" if not isinstance(where, str) else "  " + where
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<34}{v:>14}  {lim:<14}{w}")
        print(f"  -> {title}: {'FAIL' if self.failed else 'PASS'}")


def check(name):
    mod = importlib.import_module("vk.choreo." + name)
    sk = rig.Skeleton()
    out = Row()

    def solve(t):
        p = mod.pose(t)
        return p, sk.solve(p)

    # ---- cost
    t0 = time.perf_counter()
    for i in range(300):
        mod.pose(i * 0.05 + 0.013)
    cost = (time.perf_counter() - t0) / 300 * 1000

    # ---- loop closure / periodicity
    worst = 0.0
    for t in (0.0, 0.0333, 3.71, 7.5, 11.04, 14.9):
        _, f0 = solve(t)
        for dt in (LOOP, -LOOP, 2 * LOOP):
            _, f1 = solve(t + dt)
            worst = max(worst, max(float(np.abs(f0[k] - f1[k]).max()) for k in f0))
    out.add("loop: pose(t) vs pose(t±15)", worst, 1e-6, worst < 1e-6, unit=" m")

    # ---- fine sampling for continuity
    n = FRAMES * SUB
    ts = [(i - 1) * H for i in range(n + 3)]                # one sample before 0 … one after 15
    samples = [solve(t) for t in ts]
    names = list(points(samples[0][1]).keys())
    P = np.array([[points(f)[k] for k in names] for _, f in samples])        # (n+3, K, 3)
    RM = [[f[b] for b in ROT_BONES] for _, f in samples]
    FV = np.array([face_vec(p) for p, _ in samples])

    def refine(fn, t_a, t_b, k=8):
        """Largest sub-step change of fn over [t_a, t_b] split into k steps."""
        vals = [fn(t_a + (t_b - t_a) * j / k) for j in range(k + 1)]
        return vals

    # C0 jumps of points
    d1 = np.linalg.norm(P[1:] - P[:-1], axis=2)             # (n+2, K)
    jump, jwhere = 0.0, None
    cand = np.argsort(d1.max(axis=1))[-12:]
    for i in cand:
        kk = int(np.argmax(d1[i]))
        key = names[kk]
        vals = refine(lambda t: points(solve(t)[1])[key], ts[i], ts[i + 1])
        sub = max(float(np.linalg.norm(vals[j + 1] - vals[j])) for j in range(8))
        if sub > 0.5 * d1[i, kk] and sub > jump:
            jump, jwhere = sub, ts[i]
    out.add("jumps: joint positions", jump, 0.002, jump < 0.002, jwhere, " m")

    # C0 jumps of rotations
    rj, rwhere, rname = 0.0, None, ""
    ang = np.array([[rot_angle(RM[i][b], RM[i + 1][b]) for b in range(len(ROT_BONES))] for i in range(n + 2)])
    for i in np.argsort(ang.max(axis=1))[-12:]:
        kk = int(np.argmax(ang[i]))
        bone = ROT_BONES[kk]
        vals = refine(lambda t: solve(t)[1][bone], ts[i], ts[i + 1])
        sub = max(rot_angle(vals[j], vals[j + 1]) for j in range(8))
        if sub > 0.5 * ang[i, kk] and sub > rj:
            rj, rwhere, rname = sub, ts[i], bone
    out.add("jumps: bone rotations" + (f" ({rname})" if rj >= 2.0 else ""), rj, 2.0, rj < 2.0, rwhere, "°")
    max_ang = float(ang.max()) * SUB
    ai = np.unravel_index(int(np.argmax(ang)), ang.shape)
    out.add(f"info: fastest rotation ({ROT_BONES[ai[1]]})", max_ang, None, True, ts[ai[0]], "°/frame")

    # C0 jumps of face channels
    df = np.abs(FV[1:] - FV[:-1])
    fj, fwhere = 0.0, None
    for i in np.argsort(df.max(axis=1))[-8:]:
        kk = int(np.argmax(df[i]))
        vals = refine(lambda t: face_vec(mod.pose(t))[kk], ts[i], ts[i + 1])
        sub = max(abs(vals[j + 1] - vals[j]) for j in range(8))
        if sub > 0.5 * df[i, kk] and sub > fj:
            fj, fwhere = sub, ts[i]
    out.add("jumps: face channels", fj, 0.03, fj < 0.03, fwhere)
    out.add("info: fastest face change", float(df.max()) * SUB, None, True, ts[int(np.argmax(df.max(axis=1)))], "/frame")

    # C1 kinks of points
    d2 = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=2) / H          # ≈ velocity change per step (m/s)
    kink, kwhere, kname = 0.0, None, ""
    for i in np.argsort(d2.max(axis=1))[-16:]:
        kk = int(np.argmax(d2[i]))
        key = names[kk]
        vals = refine(lambda t: points(solve(t)[1])[key], ts[i], ts[i + 2], 16)
        h8 = H / 8
        sub = max(float(np.linalg.norm(vals[j + 2] - 2 * vals[j + 1] + vals[j])) for j in range(15)) / h8
        if sub > 0.5 * d2[i, kk] and sub > kink:
            kink, kwhere, kname = sub, ts[i + 1], key
    out.add("kinks: joint velocity" + (f" ({kname})" if kink >= 0.25 else ""), kink, 0.25, kink < 0.25, kwhere, " m/s")

    # ---- per-frame speed / acceleration (information)
    F30 = P[1:-1:SUB][:FRAMES + 1]                           # frames 0 … 450
    for grp, keys in (("wrists", ("wrist.L", "wrist.R")), ("ankles", ("ankle.L", "ankle.R")), ("head", ("head_top",))):
        idx = [names.index(k) for k in keys]
        v = np.linalg.norm(F30[1:, idx] - F30[:-1, idx], axis=2)
        a = np.linalg.norm(F30[2:, idx] - 2 * F30[1:-1, idx] + F30[:-2, idx], axis=2)
        out.add(f"info: {grp} max speed", float(v.max()), None, True, float(np.argmax(v.max(axis=1))) / FPS, " m/frame")
        out.add(f"info: {grp} max accel", float(a.max()), None, True, (float(np.argmax(a.max(axis=1))) + 1) / FPS,
                " m/frame²")

    # ---- per-frame geometric tests
    vols = HEAD_VOL + HATS.get(name, []) + TORSO_VOL
    head_like = {"head", "muzzle", "hat brim", "hat crown", "fringe"}
    coll = {}                                                # (point, volume) -> (depth, t)
    reach_arm, reach_arm_t, miss_arm, miss_arm_t = 0.0, 0, 0.0, 0
    miss_leg, miss_leg_t, ext_leg = 0.0, 0, 0.0
    slide, slide_t, under, under_t, far, far_t = 0.0, 0, 0.0, 0, 0.0, 0
    knee_bad, knee_bad_t, knee_back, knee_back_t = 0.0, 0, 0.0, 0
    hands_min, hands_t = 9.0, 0
    mic_d, sax_mouth, grip = [], 0.0, {"L": 0.0, "R": 0.0}
    slides = getattr(mod, "SLIDES", {})
    contacts = getattr(mod, "CONTACTS", [])
    prev = None
    frames30 = samples[1:-1:SUB][:FRAMES + 1]

    timeline = {}                                            # (point, volume) -> [beats with a deep penetration]

    def note(key, d, t):
        if d > coll.get(key, (-9, 0))[0]:
            coll[key] = (d, t)
        if d > TOL:
            timeline.setdefault(key, []).append((t / BEAT, d))

    for i, (p, F) in enumerate(frames30):
        t = i / FPS
        beat = (t / BEAT) % (LOOP / BEAT)
        probes = []                                          # (name, position, radius, test torso?, test head?)
        for s in ("L", "R"):
            hand = F["hand." + s]
            wrist = hand[:3, 3]
            elbow = F["forearm." + s][:3, 3]
            probes.append((f"wrist.{s}", wrist, 0.040, True, True))
            probes.append((f"hand.{s}", xf(hand, (0.0, 0.055, 0.0)), 0.035, True, True))
            probes.append((f"forearm.{s}", (wrist + elbow) * 0.5, 0.045, True, True))
            probes.append((f"elbow.{s}", elbow, 0.045, False, True))
            for fb, seg in (("f1b", 0.030), ("f2b", 0.032), ("f3b", 0.028)):
                probes.append((f"fingertip.{s}", tip(F, f"{fb}.{s}", seg), 0.010, True, True))
        if "prop:mic" in F:
            m = F["prop:mic"]
            probes.append(("mic head", xf(m, propspec.MIC["head_center"]), propspec.MIC["head_radius"], True, True))
            probes.append(("mic handle", xf(m, (0.0, 0.0, propspec.MIC["handle_z"][0])), 0.02, True, True))
        if "prop:sax" in F:
            m = F["prop:sax"]
            S = propspec.SAX
            for key, r in (("body_top", 0.022), ("upper_grip", 0.028), ("lower_grip", 0.034), ("body_bottom", 0.040),
                           ("bow_bottom", 0.046), ("bell_center", 0.07)):
                probes.append(("sax " + key, xf(m, S[key]), r, True, key != "body_top"))
        for pname, pos, r, t_torso, t_head in probes:
            for vol in vols:
                is_head = vol[4] in head_like
                if (is_head and not t_head) or (not is_head and not t_torso):
                    continue
                d = depth(F, vol, pos, r)
                if d > 0 and not any(pname == c[0] and vol[4] == c[1] and c[2] <= beat <= c[3] for c in contacts):
                    note((pname, vol[4]), d, t)
            if pname.startswith(("wrist", "hand.", "sax b")):   # thighs
                for s in ("L", "R"):
                    a, b = F["thigh." + s][:3, 3], F["shin." + s][:3, 3]
                    d = 0.075 + r - seg_dist(a, b, pos, pos + 1e-9)
                    if d > 0:
                        note((pname, "thigh"), d, t)
            if pos[2] - r < -0.005:
                note((pname, "ground"), -(pos[2] - r), t)
        # hands / forearms through each other (not for the saxophonist: both hands share the instrument)
        if "prop:sax" not in F:
            d = seg_dist(F["forearm.L"][:3, 3], xf(F["hand.L"], (0, 0.07, 0)), F["forearm.R"][:3, 3], xf(F["hand.R"], (0, 0.07, 0)))
            if d < hands_min:
                hands_min, hands_t = d, t

        # IK reach
        for s in ("L", "R"):
            arm = p.arm(s)
            space = np.eye(4) if arm.space == "world" else F[arm.space]
            target = xf(space, arm.hand)
            sh = F["upper_arm." + s][:3, 3]
            ratio = float(np.linalg.norm(target - sh)) / (rig.UPPER_ARM + rig.FOREARM)
            if ratio > reach_arm:
                reach_arm, reach_arm_t = ratio, t
            miss = float(np.linalg.norm(F["hand." + s][:3, 3] - target))
            if miss > miss_arm:
                miss_arm, miss_arm_t = miss, t
            leg = p.leg(s)
            from vk.choreo.common import ankle_pos
            a_t = xf(F["root"], ankle_pos(leg))
            miss = float(np.linalg.norm(F["foot." + s][:3, 3] - a_t))
            if abs(leg.roll) < 1e-6 and miss > miss_leg:
                miss_leg, miss_leg_t = miss, t
            ext_leg = max(ext_leg, float(np.linalg.norm(F["foot." + s][:3, 3] - F["thigh." + s][:3, 3])) / (rig.THIGH + rig.SHIN))

            # feet
            foot = F["foot." + s]
            ball = F["toe." + s][:3, 3]
            heel = xf(foot, (0.0, rig.FOOT_HEEL, -rig.ANKLE_H))
            toe_tip = xf(F["toe." + s], (0.0, -(rig.FOOT_TOE - rig.FOOT_BALL), 0.0))
            for q in (ball, heel, toe_tip):
                if -q[2] > under:
                    under, under_t = -q[2], t
            for q in (ball, foot[:3, 3]):
                r_ = math.hypot(q[0], q[1])
                if r_ > far:
                    far, far_t = r_, t
            if prev is not None:
                pp_, PF = prev
                pl = pp_.leg(s)
                deliberate = any(b0 - 0.05 <= beat <= b1 + 0.05 for b0, b1 in slides.get(s, []))
                if leg.lift < 1e-4 and pl.lift < 1e-4 and not deliberate:
                    pball = PF["toe." + s][:3, 3]
                    pheel = xf(PF["foot." + s], (0.0, rig.FOOT_HEEL, -rig.ANKLE_H))
                    mv = 0.0
                    if max(leg.toe, pl.toe) < 0.03:
                        mv = max(mv, float(np.linalg.norm((ball - pball)[:2])))
                    if max(leg.heel, pl.heel) < 0.03:
                        mv = max(mv, float(np.linalg.norm((heel - pheel)[:2])))
                    if mv > slide:
                        slide, slide_t = mv, t

            # knees
            hip, knee, ankle = F["thigh." + s][:3, 3], F["shin." + s][:3, 3], foot[:3, 3]
            axis = (ankle - hip) / max(1e-9, float(np.linalg.norm(ankle - hip)))
            off = (knee - hip) - axis * float((knee - hip) @ axis)
            if float(np.linalg.norm(off)) > 0.06:            # clearly bent
                off_h = np.array([off[0], off[1], 0.0])
                if np.linalg.norm(off_h) > 0.03:
                    off_h /= np.linalg.norm(off_h)
                    ffw = -foot[:3, 1]
                    ffw = np.array([ffw[0], ffw[1], 0.0])
                    ffw /= max(1e-9, float(np.linalg.norm(ffw)))
                    a_ = math.degrees(math.acos(max(-1.0, min(1.0, float(off_h @ ffw)))))
                    if a_ > knee_bad:
                        knee_bad, knee_bad_t = a_, t
                    pfw = -F["hips"][:3, 1]
                    back = -float(off_h[0] * pfw[0] + off_h[1] * pfw[1])
                    if back > knee_back:
                        knee_back, knee_back_t = back, t

        # props
        if "prop:mic" in F and p.jaw > 0.2:
            mh = xf(F["prop:mic"], propspec.MIC["head_center"])
            mic_d.append((float(np.linalg.norm(mh - xf(F["head"], HEAD["mouth"]))), t))
        if "prop:sax" in F:
            m = F["prop:sax"]
            sax_mouth = max(sax_mouth, float(np.linalg.norm(m[:3, 3] - xf(F["head"], HEAD["mouth"]))))
            for s, g in (("L", "upper_grip"), ("R", "lower_grip")):
                palm = xf(F["hand." + s], (-rig.side_sign(s) * 0.02, 0.05, 0.0))
                grip[s] = max(grip[s], float(np.linalg.norm(palm - xf(m, propspec.SAX[g]))))
        prev = (p, F)

    out.add("reach: arm target / arm length", reach_arm, 0.99, reach_arm <= 0.99, reach_arm_t)
    out.add("reach: hand off its target", miss_arm, 0.004, miss_arm <= 0.004, miss_arm_t, " m")
    out.add("reach: foot off its target", miss_leg, 0.004, miss_leg <= 0.004, miss_leg_t, " m")
    out.add("info: max leg extension", ext_leg, None, True)
    worst_c = sorted(coll.items(), key=lambda kv: -kv[1][0])
    bad = [(k, v) for k, v in worst_c if v[0] > TOL]
    if bad:
        for (pn, vn), (d, t) in bad[:8]:
            out.add(f"collide: {pn} in {vn}", d, TOL, False, t, " m")
    else:
        d = worst_c[0][1][0] if worst_c else 0.0
        out.add("collide: deepest penetration", d, TOL, True, worst_c[0][1][1] if worst_c else None, " m")
    if "-c" in sys.argv and timeline:
        print(f"\n-- {name}: collision timeline (beats, deepest)")
        for (pn, vn), lst in sorted(timeline.items(), key=lambda kv: kv[1][0][0]):
            runs, start, last, dmax = [], lst[0][0], lst[0][0], lst[0][1]
            for bt, d in lst[1:]:
                if bt - last > 0.1:
                    runs.append((start, last, dmax))
                    start, dmax = bt, d
                last, dmax = bt, max(dmax, d)
            runs.append((start, last, dmax))
            print(f"   {pn:<14} in {vn:<10} " + "  ".join(f"{a:.2f}-{b_:.2f} ({d * 100:.1f} cm)" for a, b_, d in runs))
    if hands_min < 9:
        out.add("collide: arm-to-arm clearance", hands_min, ">= 0.06 m", hands_min >= 0.06, hands_t, " m")
    out.add("feet: planted foot slide", slide, 0.002, slide <= 0.002, slide_t, " m/frame")
    out.add("feet: below the ground", under, 0.005, under <= 0.005, under_t, " m")
    out.add("feet: distance from the mark", far, 0.47, far <= 0.47, far_t, " m")
    out.add("knees: angle knee / foot direction", knee_bad, 70, knee_bad <= 70, knee_bad_t, "°")
    out.add("knees: pointing backwards", knee_back, 0.25, knee_back <= 0.25, knee_back_t)
    if mic_d:
        mx = max(mic_d)
        mn = min(mic_d)
        out.add("props: mic-mouth distance max", mx[0], 0.12, mx[0] <= 0.12, mx[1], " m")
        out.add("props: mic-mouth distance min", mn[0], ">= 0.05 m", mn[0] >= 0.05, mn[1], " m")
    if "sax" in name:
        out.add("props: sax mouthpiece off the mouth", sax_mouth, 0.03, sax_mouth <= 0.03, unit=" m")
        for s in ("L", "R"):
            out.add(f"props: palm.{s} to sax grip", grip[s], 0.075, grip[s] <= 0.075, unit=" m")
    out.add("cost: pose() call", cost, 2.0, cost <= 2.0, unit=" ms")
    out.show(name)
    return not out.failed


if __name__ == "__main__":
    ok = True
    for nm in NAMES:
        ok &= check(nm)
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
