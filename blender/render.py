"""Render driver for the VANKING.UZ show.

    blender -b --factory-startup --python blender/render.py -- [options]

Options
  --slice NAME      left | center | right | pano | emo | singer | sax | text | X0:X1   (default center)
  --scale S         resolution scale, 1 = full size (default 1)
  --frames A-B      frame range, inclusive (default 0-449);  --step N renders every N-th frame
  --time T[,T…]     render stills at these loop times (seconds) instead of a frame range
  --out DIR         output directory (default out/frames/<slice>)
  --engine E        eevee (default) | cycles
  --samples N       render samples (default: 32 final / 8 draft for EEVEE)
  --quality Q       final (default) | draft  (draft = lighter geometry for quick previews)
  --skip A,B        skip scene modules (environment, lighting, characters, fireworks)
  --grass Q         grass density multiplier (default 1 final / 0.1 draft); lower = faster frames
  --grass-shadows   let the hero grass cast shadows (much slower, slightly richer meadow)
  --env A,B         build only these environment layers (sky, props, grass, flowers, trees, fireflies)
  --cast A,B        build only these characters (emo, singer, sax)
  --overscan PX     extra columns on each side (full-res px), cropped later by the encoder
  --force           re-render frames that already exist (default: resume, skip existing)
  --save-blend F    save the built scene to a .blend file and exit (for inspection in the UI)
"""
import argparse
import os
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.stdout.reconfigure(line_buffering=True)     # progress lines appear at once, also through a pipe

from vk import camera as cam_mod  # noqa: E402
from vk import config as C  # noqa: E402
from vk.show import Show  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="render.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slice", default="center")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--frames", default=f"0-{C.FRAMES - 1}")
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--part", default="1/1", help="I/N: render only every N-th job, starting with job I (parallel workers)")
    p.add_argument("--time", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--engine", default="eevee", choices=["eevee", "cycles"])
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--quality", default="final", choices=["final", "draft"])
    p.add_argument("--skip", default="")
    p.add_argument("--grass", type=float, default=None)
    p.add_argument("--grass-shadows", action="store_true")
    p.add_argument("--env", default=None)
    p.add_argument("--cast", default=None)
    p.add_argument("--overscan", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--save-blend", default=None)
    return p.parse_args(argv)


def main():
    a = parse_args()
    slice_px = cam_mod.parse_slice(a.slice)
    out_dir = os.path.abspath(a.out or os.path.join("out", "frames", a.slice.replace(":", "-")))
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    options = dict(grass=a.grass, grass_shadow=a.grass_shadows,
                   env=set(a.env.split(",")) if a.env is not None else None,
                   cast=a.cast.split(",") if a.cast else None)
    show = Show(engine=a.engine, quality=a.quality, samples=a.samples, skip=[s for s in a.skip.split(",") if s],
                options=options).build()
    print(f"[vk] scene built in {time.time() - t0:.1f}s")
    scene = show.scene

    if a.save_blend:
        show.frame(slice_px, 0.0, a.scale, a.overscan)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save_blend))
        print(f"[vk] saved {a.save_blend}")
        return

    jobs = []   # (output path without extension, loop time, frame number)
    if a.time is not None:
        for s in a.time.split(","):
            t = float(s)
            jobs.append((os.path.join(out_dir, f"t{t:06.3f}".replace(".", "_")), t, int(round(t * C.FPS))))
    else:
        f0, f1 = (int(v) for v in a.frames.split("-")) if "-" in a.frames else (int(a.frames), int(a.frames))
        for f in range(f0, f1 + 1, a.step):
            jobs.append((os.path.join(out_dir, f"f{f:04d}"), f / C.FPS, f))

    part_i, part_n = (int(v) for v in a.part.split("/"))
    if not 1 <= part_i <= part_n:
        raise SystemExit(f"--part {a.part}: expected I/N with 1 <= I <= N")
    jobs = jobs[part_i - 1::part_n]

    done = 0
    t_start = time.time()
    for path, t, frame in jobs:
        png = path + ".png"
        if os.path.exists(png) and not a.force:
            continue
        scene.frame_set(frame % C.FRAMES)
        w, h = show.frame(slice_px, t, a.scale, a.overscan)
        tmp = path + ".tmp"
        scene.render.filepath = tmp
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        os.replace(tmp + ".png", png)      # atomic: an interrupted render never leaves a broken frame
        done += 1
        el = time.time() - t_start
        left = sum(1 for p, _, _ in jobs if not os.path.exists(p + ".png"))
        print(f"[vk] {os.path.basename(png)}  {w}x{h}  t={t:.3f}s  {time.time() - t1:.1f}s/frame  "
              f"remaining {left} (~{left * el / max(done, 1) / 60:.1f} min)", flush=True)
    print(f"[vk] done: {done} frame(s) rendered into {out_dir} in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
