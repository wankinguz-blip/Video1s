"""The cast: three moles on their marks.  Glue between the rig (vk.rig), the model (vk.mole) and
the choreography (vk.choreo.<name>.pose).

Per frame and per character:  pose = choreo.pose(t)  →  automatic follow-through for extra bones
→  rig.apply(pose)  →  mole.update(...) for the face and other non-bone controls.
"""
import importlib

import numpy as np

from . import config as C
from . import mole, rig, terrain

CAST = ("emo", "singer", "sax")


def build(ctx):
    col = ctx.collection("Characters")
    only = ctx.options.get("cast") or CAST
    chars = []
    for name in CAST:
        if name not in only:
            continue
        sk = rig.Skeleton(extra=mole.extra_bones(name))
        mark = rig.mark_matrix(terrain.mark(name), face_point=(C.CAM_X, -C.CAM_DIST))
        r = rig.Rig(name, sk, mark, col)
        state = mole.build(ctx, r, name)
        choreo = importlib.import_module(f"vk.choreo.{name}")
        chars.append(dict(name=name, rig=r, mole=state, choreo=choreo))
    return dict(chars=chars)


def follow_through(sk, mark, pose_fn, pose, t):
    """Fill pose.extra for extra bones the choreography did not set: each such bone trails behind
    its parent (hair, coat tails, bow-tie ends …).  It compares where the bone's tip is now with
    where the parent carried it `lag` seconds ago — a pure function of t, so it loops exactly."""
    todo = [e for e in sk.extra if e[0] not in pose.extra and len(e) > 3 and e[3].get("lag", 0) > 0]
    if not todo:
        return
    now = sk.solve(pose, mark)
    cache = {}
    for name, parent, rest_pos, opt in todo:
        lag = opt["lag"]
        if lag not in cache:
            cache[lag] = sk.solve(pose_fn(t - lag), mark)
        d = np.asarray(opt.get("dir", (0.0, 0.0, -1.0)), dtype=np.float64)
        length = opt.get("len", 0.12)
        tip_rest = np.asarray(rest_pos, dtype=np.float64) + d * length
        a_now = now[parent] @ sk.rest_inv[parent]
        a_old = cache[lag][parent] @ sk.rest_inv[parent]
        delta = rig.xf(a_old, tip_rest) - rig.xf(a_now, tip_rest)            # world drag vector
        local = now[parent][:3, :3].T @ delta
        rest_dir = sk.rest[parent][:3, :3].T @ d                              # dir in the parent frame
        want = rest_dir * length + local * opt.get("gain", 1.0)
        n = np.linalg.norm(want)
        if n < 1e-6:
            continue
        want /= n
        # limit the swing angle
        lim = opt.get("limit", 0.9)
        c = float(np.clip(np.dot(want, rest_dir), -1.0, 1.0))
        ang = np.arccos(c)
        if ang > lim:
            k = lim / ang
            want = rig.norm(rest_dir * (1 - k) + want * k) if ang < 3.0 else rest_dir
        pose.extra[name] = rig.rot_between(rest_dir, want)


def update(state, ctx, t):
    for c in state["chars"]:
        r = c["rig"]
        pose = c["choreo"].pose(t)
        follow_through(r.sk, r.mark, c["choreo"].pose, pose, t)
        frames = r.apply(pose)
        mole.update(c["mole"], r, pose, frames, t)
