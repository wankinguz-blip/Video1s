"""Loop check: compares rendered frames (runs inside Blender's Python, needs no extra packages).

    blender -b --factory-startup --python blender/dev/compare_frames.py -- T0.png T15.png T_LAST.png T_FIRST_PLUS_1.png

T0 vs T15: the same moment of the loop, must be identical.
T_LAST (frame 449) -> T0 is the step across the loop point; it must look like an ordinary frame step
(compared with T0 -> T_FIRST_PLUS_1).
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


def diff(a, b):
    d = np.abs(a - b) * 255.0
    return float(d.mean()), float(d.max())


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    t0, t15, last, first1 = (load(p) for p in args[:4])
    same_mean, same_max = diff(t0, t15)
    seam_mean, _ = diff(last, t0)
    step_mean, _ = diff(t0, first1)
    ratio = seam_mean / max(step_mean, 1e-6)
    print(f"[loop] t=0 vs t=15:            mean {same_mean:.4f}  max {same_max:.1f}  (8-bit levels; 0 = identical)")
    print(f"[loop] step 449->0:            mean {seam_mean:.4f}")
    print(f"[loop] ordinary step 0->1:     mean {step_mean:.4f}")
    print(f"[loop] seam / ordinary ratio:  {ratio:.2f}")
    ok = same_mean < 0.05 and 0.4 < ratio < 2.5
    print(f"[loop] RESULT: {'PASS' if ok else 'CHECK'}")


main()
