"""Scene statistics without rendering: build time, memory and geometry load of every module.

    blender -b --factory-startup --python blender/dev/scene_stats.py -- [--quality final|draft] [--skip a,b] [--time T]

Prints, per scene module: build time, process memory after the build, number of objects and the
number of triangles the renderer has to draw (instances included), plus the lights of the scene and
the time of one show.update(t).  Use it to see what makes a frame expensive before rendering.
"""
import argparse
import os
import resource
import sys
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

from vk import camera as cam_mod  # noqa: E402
from vk import show as show_mod  # noqa: E402
from vk.show import Show  # noqa: E402


def rss_mb():
    """Current resident memory in MB (Linux: /proc; elsewhere: peak from getrusage)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except OSError:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2 ** 20 if sys.platform == "darwin" else peak / 1024


def geometry_load():
    """{collection name: [objects, instances, triangles drawn]} from the evaluated depsgraph."""
    dg = bpy.context.evaluated_depsgraph_get()
    tri_cache = {}
    out = {}
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type not in {"MESH", "CURVE", "FONT", "SURFACE"}:
            continue
        src = inst.parent.original if inst.is_instance and inst.parent else ob.original
        if src.hide_render:
            continue
        key = ob.data.name if ob.data else ob.name
        if key not in tri_cache:
            try:
                me = ob.data if ob.type == "MESH" else ob.to_mesh()
                tri_cache[key] = sum(len(p.vertices) - 2 for p in me.polygons)
            except Exception:
                tri_cache[key] = 0
        cols = src.users_collection
        name = cols[0].name if cols else "(scene)"
        row = out.setdefault(name, [set(), 0, 0])
        row[0].add(src.name)
        row[1] += 1
        row[2] += tri_cache[key]
    return {k: (len(v[0]), v[1], v[2]) for k, v in out.items()}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="scene_stats.py")
    p.add_argument("--quality", default="final", choices=["final", "draft"])
    p.add_argument("--skip", default="")
    p.add_argument("--time", type=float, default=2.5)
    p.add_argument("--grass", type=float, default=None)
    a = p.parse_args(argv)

    t_all = time.time()
    show = Show(quality=a.quality, skip=[s for s in a.skip.split(",") if s], options=dict(grass=a.grass))
    print(f"[stats] quality={a.quality}  memory before build: {rss_mb():.0f} MB")
    for name in show_mod.MODULES:
        if name in show.ctx.skip:
            continue
        import importlib
        mod = importlib.import_module("vk." + name)
        t0 = time.time()
        state = mod.build(show.ctx)
        show.ctx.states[name] = state
        show.modules.append((name, mod, state))
        print(f"[stats] built {name:<12} {time.time() - t0:6.1f} s   memory {rss_mb():7.0f} MB")
    print(f"[stats] scene built in {time.time() - t_all:.1f} s")

    show.frame(cam_mod.parse_slice("pano"), a.time, 0.125)
    for i in range(2):
        t0 = time.time()
        show.update(a.time + i / 30.0)
        bpy.context.view_layer.update()
        print(f"[stats] update + depsgraph #{i + 1}: {time.time() - t0:.2f} s   memory {rss_mb():.0f} MB")

    total = [0, 0, 0]
    print(f"[stats] {'collection':<22}{'objects':>9}{'instances':>12}{'triangles':>14}")
    for name, (n_obj, n_inst, n_tri) in sorted(geometry_load().items(), key=lambda kv: -kv[1][2]):
        print(f"[stats] {name:<22}{n_obj:>9}{n_inst:>12}{n_tri:>14,}")
        total = [total[0] + n_obj, total[1] + n_inst, total[2] + n_tri]
    print(f"[stats] {'TOTAL':<22}{total[0]:>9}{total[1]:>12}{total[2]:>14,}")
    for ob in bpy.data.objects:
        if ob.type == "LIGHT":
            d = ob.data
            print(f"[stats] light {ob.name:<16} {d.type:<5} energy {d.energy:8.2f}  shadow {d.use_shadow}")
    print(f"[stats] materials {len(bpy.data.materials)}  meshes {len(bpy.data.meshes)}  objects {len(bpy.data.objects)}")


if __name__ == "__main__":
    main()
