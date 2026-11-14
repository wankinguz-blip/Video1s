#!/usr/bin/env python3
"""Stick-figure contact sheet of a choreography (numpy only, no Blender, no PIL).

    python3 blender/dev/stick_sheet.py emo out/emo_stick.png --from 0 --to 8 --step 0.25 --cols 8
    python3 blender/dev/stick_sheet.py singer out/singer_side.png --view both --step 0.5

Every cell is pose(t) solved with vk.rig.Skeleton and drawn orthographically:
  front view  screen x = character +X (the character's LEFT is on the right of the picture)
  side view   seen from the character's left, the character faces the right edge of the cell
The character's left limbs are red, the right ones blue; the head, muzzle and torso are drawn as
filled silhouettes of the checker's collision volumes and everything is painted back to front, so
a hand that is hidden by the head / body really is behind it.  Thin light curves are the paths
of the wrists since the previous cell.  Cells on a whole beat have a tinted background; the label
is the beat.  --warn also prints ArmTrack.warnings() of the module's arm tracks.
"""
import argparse
import importlib
import math
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vk import rig  # noqa: E402
from vk.config import BEAT  # noqa: E402
from vk.rig import HEAD, xf  # noqa: E402

RED, BLUE = (214, 48, 40), (38, 92, 220)
RED_L, BLUE_L = (240, 150, 140), (140, 170, 240)
DARK, GREY, BODY, SKIN = (40, 40, 40), (150, 150, 150), (226, 226, 226), (246, 240, 228)
X_HALF, Z_LO, Z_HI = 0.78, -0.07, 1.68               # metres shown in a cell

# volumes (bone, centre, radii) – the head ones are those of dev/check_choreo.py, the torso
# cylinders are drawn as ellipsoids of the same width
VOLS = [("hips", (0.0, 0.0, 0.01), (0.20, 0.17, 0.11), BODY), ("spine", (0.0, 0.0, 0.07), (0.215, 0.185, 0.09), BODY),
        ("chest", (0.0, 0.0, 0.06), (0.20, 0.165, 0.09), BODY), ("head", HEAD["center"], HEAD["radii"], SKIN),
        ("head", (0.0, -0.24, 0.18), (0.11, 0.12, 0.08), SKIN)]
FRINGE = ("head", (-0.07, -0.25, 0.33), (0.17, 0.07, 0.14), (90, 90, 100))

_FONT = {"0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
         "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001010010010",
         "8": "111101111101111", "9": "111101111001111", ".": "000000000000010", "-": "000000111000000",
         " ": "000000000000000"}


# ---------------------------------------------------------------- tiny rasteriser
class Canvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.a = np.full((h, w, 3), 255.0)

    def _box(self, x0, y0, x1, y1, pad):
        xa, xb = max(0, int(math.floor(min(x0, x1) - pad))), min(self.w, int(math.ceil(max(x0, x1) + pad)) + 1)
        ya, yb = max(0, int(math.floor(min(y0, y1) - pad))), min(self.h, int(math.ceil(max(y0, y1) + pad)) + 1)
        if xa >= xb or ya >= yb:
            return None
        ys, xs = np.mgrid[ya:yb, xa:xb]
        return ya, yb, xa, xb, xs + 0.5, ys + 0.5

    def _blend(self, ya, yb, xa, xb, alpha, col):
        reg = self.a[ya:yb, xa:xb]
        reg += (np.asarray(col, dtype=np.float64) - reg) * alpha[..., None]

    def rect(self, x0, y0, x1, y1, col):
        self.a[max(0, y0):y1, max(0, x0):x1] = col

    def line(self, p, q, col, width=2.0):
        b = self._box(p[0], p[1], q[0], q[1], width)
        if b is None:
            return
        ya, yb, xa, xb, xs, ys = b
        dx, dy = q[0] - p[0], q[1] - p[1]
        den = dx * dx + dy * dy
        u = np.clip(((xs - p[0]) * dx + (ys - p[1]) * dy) / den, 0.0, 1.0) if den > 1e-9 else 0.0
        d = np.hypot(xs - (p[0] + u * dx), ys - (p[1] + u * dy))
        self._blend(ya, yb, xa, xb, np.clip(width * 0.5 + 0.5 - d, 0.0, 1.0), col)

    def disc(self, p, r, col):
        self.line(p, p, col, 2.0 * r)

    def ellipse(self, c, m2, fill, edge, width=1.3):
        """Filled ellipse {c + u : u' inv(m2) u <= 1} with an outline (m2 = 2x2 covariance, px^2)."""
        ext_x, ext_y = math.sqrt(m2[0, 0]), math.sqrt(m2[1, 1])
        b = self._box(c[0] - ext_x, c[1] - ext_y, c[0] + ext_x, c[1] + ext_y, 2.0)
        if b is None:
            return
        ya, yb, xa, xb, xs, ys = b
        im = np.linalg.inv(m2)
        ux, uy = xs - c[0], ys - c[1]
        k = np.sqrt(np.maximum(im[0, 0] * ux * ux + 2 * im[0, 1] * ux * uy + im[1, 1] * uy * uy, 1e-12))
        dist = (k - 1.0) * np.hypot(ux, uy) / k                # > 0 outside (approximate pixel distance)
        self._blend(ya, yb, xa, xb, np.clip(0.5 - dist, 0.0, 1.0), fill)
        self._blend(ya, yb, xa, xb, np.clip(width * 0.5 + 0.5 - np.abs(dist), 0.0, 1.0), edge)

    def text(self, x, y, s, col, k=2):
        for ch in s:
            bits = _FONT.get(ch, _FONT[" "])
            for i, bit in enumerate(bits):
                if bit == "1":
                    cx, cy = x + (i % 3) * k, y + (i // 3) * k
                    self.a[cy:cy + k, cx:cx + k] = col
            x += 4 * k

    def save(self, path):
        raw = np.clip(self.a + 0.5, 0, 255).astype(np.uint8)
        rows = b"".join(b"\x00" + raw[y].tobytes() for y in range(self.h))

        def chunk(tag, data):
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0))
                    + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


# ---------------------------------------------------------------- one figure
def _views(view):
    """-> list of (projection 2x3 [screen x, screen up], depth axis: larger = farther)."""
    front = (np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), np.array([0.0, 1.0, 0.0]))
    side = (np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([-1.0, 0.0, 0.0]))
    return {"front": [front], "side": [side], "both": [front, side]}[view]


def figure(cv, F, proj, depth_axis, ox, oy, scale, fringe=False, trail=None):
    """Draw solved frames F with the ground point of the mark at pixel (ox, oy)."""
    def px(p):
        s = proj @ np.asarray(p, dtype=np.float64)
        return (ox + s[0] * scale, oy - s[1] * scale)

    def dp(p):
        return float(depth_axis @ np.asarray(p, dtype=np.float64))

    prims = []                                                # (depth, draw function)

    def seg(a, b, col, width):
        prims.append((0.5 * (dp(a) + dp(b)), lambda: cv.line(px(a), px(b), col, width)))

    def dot(a, r, col, bias=0.0):
        prims.append((dp(a) - bias, lambda: cv.disc(px(a), r, col)))

    def vol(bone, c, radii, col, edge=GREY):
        m = F[bone][:3, :3] * np.asarray(radii)
        pm = proj @ m
        m2 = pm @ pm.T * scale * scale
        m2 = np.array([[m2[0, 0], -m2[0, 1]], [-m2[0, 1], m2[1, 1]]])     # screen y points down
        centre = xf(F[bone], c)
        prims.append((dp(centre), lambda: cv.ellipse(px(centre), m2, col, edge)))

    for v in VOLS:
        vol(*v)
    if fringe:
        vol(*FRINGE, edge=DARK)
    head = F["head"]
    dot(xf(head, HEAD["nose"]), 2.2, DARK, 0.01)
    for e in ("eye_L", "eye_R"):
        dot(xf(head, HEAD[e]), 1.6, DARK, 0.01)
    for s, col, col_l in (("L", RED, RED_L), ("R", BLUE, BLUE_L)):
        sh, el, hand = F["upper_arm." + s][:3, 3], F["forearm." + s][:3, 3], F["hand." + s]
        wr = hand[:3, 3]
        seg(sh, el, col, 3.0)
        seg(el, wr, col, 3.0)
        seg(wr, xf(hand, (0.0, rig.HAND, 0.0)), col, 4.6)
        dot(el, 2.6, DARK)
        dot(wr, 2.2, (255, 255, 255), 0.002)
        hip, kn, foot, toe = F["thigh." + s][:3, 3], F["shin." + s][:3, 3], F["foot." + s], F["toe." + s]
        an = foot[:3, 3]
        seg(hip, kn, col_l, 3.4)
        seg(kn, an, col_l, 3.4)
        heel = xf(foot, (0.0, rig.FOOT_HEEL, -rig.ANKLE_H))
        ball = toe[:3, 3]
        seg(an, heel, col, 2.6)
        seg(heel, ball, col, 2.6)
        seg(an, ball, col, 2.6)
        seg(ball, xf(toe, (0.0, -(rig.FOOT_TOE - rig.FOOT_BALL), 0.0)), col, 3.6)
        if trail is not None:
            pts = [px(q) for q in trail[s]]
            for a, b in zip(pts[:-1], pts[1:]):
                cv.line(a, b, col_l, 1.0)
    for _d, fn in sorted(prims, key=lambda t: -t[0]):
        fn()


def sheet(name, out, b0, b1, step, cols, view="front", scale=100.0, trail=True):
    mod = importlib.import_module("vk.choreo." + name)
    sk = rig.Skeleton()
    views = _views(view)
    n = int(round((b1 - b0) / step))
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    vw = int(round(2 * X_HALF * scale))
    cw, ch = vw * len(views) + 2, int(round((Z_HI - Z_LO) * scale)) + 2
    cv = Canvas(cols * cw, rows * ch)
    for i in range(n):
        b = b0 + i * step
        x0, y0 = (i % cols) * cw, (i // cols) * ch
        on_beat = abs(b - round(b)) < 1e-6
        cv.rect(x0, y0, x0 + cw - 1, y0 + ch - 1, (255, 249, 214) if on_beat else (250, 250, 250))
        gy = y0 + ch - 2 + Z_LO * scale
        cv.line((x0, gy), (x0 + cw - 2, gy), GREY, 1.0)
        F = sk.solve(mod.pose(b * BEAT))
        tr = None
        if trail:
            sub = [sk.solve(mod.pose((b - step * (1.0 - j / 6.0)) * BEAT)) for j in range(6)] + [F]
            tr = {s: [f["hand." + s][:3, 3] for f in sub] for s in ("L", "R")}
        for k, (proj, dax) in enumerate(views):
            figure(cv, F, proj, dax, x0 + k * vw + vw * 0.5, gy, scale, fringe=(name == "emo"), trail=tr)
        cv.text(x0 + 4, y0 + 4, f"{b:g}", (170, 30, 30) if on_beat else DARK)
    cv.save(out)
    return cv.w, cv.h, n


def warnings(name):
    from vk.choreo.common import ArmTrack
    mod = importlib.import_module("vk.choreo." + name)
    found = False
    for key, obj in sorted(vars(mod).items()):
        if isinstance(obj, ArmTrack):
            found = True
            ws = obj.warnings()
            print(f"{name}.{key}: {len(ws)} warning(s)")
            for a, b, part, ang in ws:
                print(f"   {part:<5} turns {ang:5.1f} deg between keys at beats {a:.2f} and {b:.2f}")
    if not found:
        print(f"{name}: no ArmTrack at module level")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", help="choreography module: singer | sax | emo")
    ap.add_argument("out", help="PNG to write")
    ap.add_argument("--from", dest="b0", type=float, default=0.0, help="first beat (default 0)")
    ap.add_argument("--to", dest="b1", type=float, default=30.0, help="end beat, exclusive (default 30)")
    ap.add_argument("--step", type=float, default=0.25, help="beats between cells (default 0.25)")
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--view", choices=("front", "side", "both"), default="front")
    ap.add_argument("--scale", type=float, default=100.0, help="pixels per metre (default 100)")
    ap.add_argument("--no-trail", action="store_true", help="do not draw the wrist paths")
    ap.add_argument("--warn", action="store_true", help="print ArmTrack.warnings() of the module")
    a = ap.parse_args()
    w, h, n = sheet(a.name, a.out, a.b0, a.b1, a.step, a.cols, a.view, a.scale, not a.no_trail)
    print(f"{a.out}: {n} cells, {w} x {h} px")
    if a.warn:
        warnings(a.name)
