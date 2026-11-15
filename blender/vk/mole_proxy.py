"""PROXY mole (simple skinned tubes) — a stable stand-in used to test the rig and choreography.

API used by vk.characters:
    extra_bones(variant)                     -> [(name, parent_bone, rest_pos, options), ...]
    build(ctx, rig, variant)                 -> state
    update(state, rig, pose, frames, t)      -> None
variant is 'singer' | 'sax' | 'emo'.
"""
import numpy as np

from . import propspec, rig as R, skin, util

BROWN = util.srgb("#8a5a3c")
TINT = {"singer": (0.02, 0.02, 0.025), "sax": (0.25, 0.02, 0.06), "emo": (0.03, 0.03, 0.04)}


def extra_bones(variant):
    if variant == "emo":
        return [("fringe", "head", (-0.08, -0.26, 1.40), dict(lag=0.10, dir=(0.0, 0.0, -1.0), len=0.22, gain=1.2, limit=0.7))]
    return []


def _mat(name, color, **kw):
    import bpy
    return bpy.data.materials.get(name) or util.principled(name, color, **kw)


def _obj(rig, name, mesh, mat):
    o = util.mesh_object(f"{rig.name}_{name}", mesh[0], mesh[1], rig.col)
    o.data.materials.append(mat)
    return o


def build(ctx, rig, variant):
    sk = rig.sk
    J = sk.joint
    fur = _mat("proxy_fur", BROWN, roughness=0.8)
    cloth = _mat("proxy_cloth_" + variant, TINT[variant], roughness=0.6)
    white = _mat("proxy_white", (0.8, 0.8, 0.8), roughness=0.5)
    dark = _mat("proxy_dark", (0.01, 0.01, 0.01), roughness=0.4)
    gold = _mat("proxy_gold", (1.0, 0.72, 0.25), roughness=0.25, metallic=1.0)

    # torso (skinned along the spine)
    pts = [J("hips") + [0, 0, -0.06], J("hips"), J("spine"), J("chest"), J("neck"), J("head")]
    rad = [(0.15, 0.13), (0.20, 0.17), (0.215, 0.185), (0.20, 0.165), (0.12, 0.11), (0.09, 0.09)]
    m = skin.tube(pts, rad, sides=20, per_seg=4)
    o = _obj(rig, "torso", m, cloth)
    rig.bind(o, skin.chain_weights(m[0], [pts[0] - [0, 0, 0.3], J("spine"), J("chest"), J("neck"), J("head"), J("head") + [0, 0, 0.3]],
                                   ["hips", "spine", "chest", "neck", "head"], blend=0.05))
    # head, nose, jaw (rigid)
    hc = J("head") + R.HEAD["center"]
    o = _obj(rig, "head", skin.ellipsoid(hc, R.HEAD["radii"], 32, 20), fur)
    rig.attach(o, "head")
    o = _obj(rig, "nose", skin.ellipsoid(J("head") + R.HEAD["nose"] + [0, 0.04, 0], (0.07, 0.06, 0.05)), dark)
    rig.attach(o, "head")
    for s in ("L", "R"):
        o = _obj(rig, "eye" + s, skin.ellipsoid(J("head") + R.HEAD["eye_" + s], (0.04, 0.03, 0.05)), white)
        rig.attach(o, "head")
    o = _obj(rig, "jaw", skin.ellipsoid(J("head") + R.HEAD["mouth"] + [0, 0.05, -0.05], (0.13, 0.09, 0.05)), white)
    rig.attach(o, "jaw")
    # limbs
    for s in ("L", "R"):
        a, b, c = J("upper_arm." + s), J("forearm." + s), J("hand." + s)
        m = skin.tube([a, (a + b) / 2, b, (b + c) / 2, c], [0.062, 0.058, 0.052, 0.048, 0.042], sides=14, per_seg=4)
        o = _obj(rig, "arm" + s, m, cloth)
        rig.bind(o, skin.chain_weights(m[0], [a, b, c], ["upper_arm." + s, "forearm." + s], blend=0.05))
        hf = sk.rest["hand." + s]
        hv, hfaces = skin.ellipsoid((0.0, 0.045, 0.0), (0.028, 0.055, 0.05), 16, 10)
        hv = hv @ hf[:3, :3].T + hf[:3, 3]
        o = _obj(rig, "hand" + s, (hv, hfaces), fur if not (variant == "singer" and s == "L") else white)
        rig.attach(o, "hand." + s)
        for fname, _base, seg in R.FINGERS:
            for part in ("a", "b"):
                bn = f"{fname}{part}.{s}"
                f0 = sk.rest[bn]
                p0 = f0[:3, 3]
                p1 = p0 + f0[:3, 1] * seg
                o = _obj(rig, bn, skin.tube([p0, p1], [0.014, 0.012], sides=8, per_seg=1), fur)
                rig.attach(o, bn)
        a, b, c = J("thigh." + s), J("shin." + s), J("foot." + s)
        m = skin.tube([a + [0, 0, 0.04], a, (a + b) / 2, b, (b + c) / 2, c], [0.085, 0.085, 0.078, 0.07, 0.062, 0.055], sides=14, per_seg=4)
        o = _obj(rig, "leg" + s, m, cloth)
        rig.bind(o, skin.chain_weights(m[0], [a + [0, 0, 0.2], b, c - [0, 0, 0.2]], ["thigh." + s, "shin." + s], blend=0.05))
        fv, ff = skin.ellipsoid(c + [0, -0.035, -0.02], (0.065, 0.105, 0.05), 16, 10)
        o = _obj(rig, "foot" + s, (fv, ff), dark)
        rig.attach(o, "foot." + s)
        tv, tf = skin.ellipsoid(c + [0, -0.19, -0.035], (0.06, 0.075, 0.035), 16, 10)
        o = _obj(rig, "toe" + s, (tv, tf), dark)
        rig.attach(o, "toe." + s)
    # variant specifics
    if variant == "singer":
        hb = J("head") + R.HEAD["hat_base"]
        o = _obj(rig, "hat_brim", skin.ellipsoid(hb, (0.38, 0.38, 0.015), 32, 6), dark)
        rig.attach(o, "head")
        o = _obj(rig, "hat_crown", skin.tube([hb, hb + [0, 0, 0.2]], [(0.22, 0.2), (0.2, 0.18)], sides=24, per_seg=1, cap_end="flat"), dark)
        rig.attach(o, "head")
        mic = propspec.MIC
        o1 = _obj(rig, "mic_handle", skin.tube([(0, 0, mic["handle_z"][0]), (0, 0, mic["handle_z"][1])], [mic["handle_radius"]] * 2, sides=12, per_seg=1), dark)
        o2 = _obj(rig, "mic_head", skin.ellipsoid(mic["head_center"], [mic["head_radius"]] * 3, 16, 10), white)
        rig.add_prop("mic", [o1, o2])
    if variant == "sax":
        S = propspec.SAX
        pts = [S["mouthpiece"], S["neck_top"], S["body_top"], S["upper_grip"], S["lower_grip"], S["body_bottom"], S["bow_bottom"],
               tuple(np.array(S["bell_center"]) - np.array(S["bell_normal"]) * 0.12), S["bell_center"]]
        rad = [0.008, 0.012, 0.022, 0.028, 0.034, 0.040, 0.046, 0.052, S["bell_radius"]]
        o = _obj(rig, "sax", skin.tube(pts, rad, sides=16, per_seg=5, cap_end=None), gold)
        rig.add_prop("sax", [o])
    if variant == "emo":
        p = np.array((-0.08, -0.26, 1.40))
        o = _obj(rig, "fringe", skin.tube([p, p + [-0.03, -0.02, -0.11], p + [-0.05, -0.02, -0.22]], [(0.10, 0.03), (0.09, 0.03), (0.04, 0.02)], sides=12), dark)
        rig.attach(o, "fringe")
    return dict(variant=variant)


def update(state, rig, pose, frames, t):
    pass
