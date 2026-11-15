"""Seam check: screens rendered separately and put side by side must be the same picture as one
render of the whole width.

    blender -b --factory-startup --python blender/dev/compare_seams.py -- SCREEN_1.png SCREEN_2.png [...] WHOLE.png

(normally LEFT CENTER RIGHT PANO).  Reports the difference of the whole picture and of the 16
columns around every seam (8-bit levels).  The renders use different sample noise, so "equal"
means: the seam columns do not differ more than the rest of the picture.  Runs inside Blender's
Python (needs no extra packages).
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
    screens = [load(p) for p in args[:-1]]
    pano = load(args[-1])
    joined = np.concatenate(screens, axis=1)
    if joined.shape != pano.shape:
        print(f"[seams] sizes differ: screens {joined.shape[1]}x{joined.shape[0]}, whole picture {pano.shape[1]}x{pano.shape[0]}")
        print("[seams] RESULT: CHECK (use a scale at which all screen widths are whole pixels: 1, 0.5, 0.25, 0.125)")
        return
    d = np.abs(joined - pano) * 255.0
    whole = float(d.mean())
    print(f"[seams] screens vs whole picture: mean {whole:.3f}  max {float(d.max()):.1f}  (8-bit levels)")
    worst = 0.0
    col = 0
    for i, scr in enumerate(screens[:-1]):
        col += scr.shape[1]
        band = d[:, col - 8:col + 8]
        # the seam itself: do the two screens continue each other as the single render does?
        step_screens = np.abs(joined[:, col] - joined[:, col - 1]) * 255.0
        step_pano = np.abs(pano[:, col] - pano[:, col - 1]) * 255.0
        extra = float(np.abs(step_screens - step_pano).mean())
        worst = max(worst, float(band.mean()), extra)
        print(f"[seams] seam {i + 1} at column {col}: columns {col - 8}..{col + 7} mean {float(band.mean()):.3f}  max {float(band.max()):.1f}   "
              f"step across the seam: screens {float(step_screens.mean()):.3f}, whole picture {float(step_pano.mean()):.3f}")
    ok = worst < max(2.0, 3.0 * whole)
    print(f"[seams] RESULT: {'PASS' if ok else 'CHECK'}")


main()
