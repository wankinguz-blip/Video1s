"""Seam check: the three screens side by side must be the same picture as one panorama render.

    blender -b --factory-startup --python blender/dev/compare_seams.py -- LEFT.png CENTER.png RIGHT.png PANO.png

Reports the difference of the whole picture and of the 16 columns around each of the two seams
(8-bit levels).  The renders use different sample noise, so "equal" means: the seam columns do not
differ more than the rest of the picture.  Runs inside Blender's Python (needs no extra packages).
"""
import sys

import bpy
import numpy as np


def load(path):
    img = bpy.data.images.load(path)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(h, w, 4)[:, :, :3]


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    left, center, right, pano = (load(p) for p in args[:4])
    joined = np.concatenate([left, center, right], axis=1)
    if joined.shape != pano.shape:
        print(f"[seams] sizes differ: screens {joined.shape[1]}x{joined.shape[0]}, panorama {pano.shape[1]}x{pano.shape[0]}")
        print("[seams] RESULT: CHECK (use a scale at which all screen widths are whole pixels: 1, 0.5, 0.25, 0.125)")
        return
    d = np.abs(joined - pano) * 255.0
    whole = float(d.mean())
    print(f"[seams] screens vs panorama, whole picture: mean {whole:.3f}  max {float(d.max()):.1f}  (8-bit levels)")
    worst = 0.0
    for name, col in (("left|center", left.shape[1]), ("center|right", left.shape[1] + center.shape[1])):
        band = d[:, col - 8:col + 8]
        # the seam itself: do the two screens continue each other as the panorama does?
        step_screens = np.abs(joined[:, col] - joined[:, col - 1]) * 255.0
        step_pano = np.abs(pano[:, col] - pano[:, col - 1]) * 255.0
        extra = float(np.abs(step_screens - step_pano).mean())
        worst = max(worst, float(band.mean()), extra)
        print(f"[seams] {name:<13} columns {col - 8}..{col + 7}: mean {float(band.mean()):.3f}  max {float(band.max()):.1f}   "
              f"step across the seam: screens {float(step_screens.mean()):.3f}, panorama {float(step_pano.mean()):.3f}")
    ok = worst < max(2.0, 3.0 * whole)
    print(f"[seams] RESULT: {'PASS' if ok else 'CHECK'}")


main()
