"""bpy-side mesh helpers for the characters: fast mesh creation from numpy, the voxel "sculpt"
operator used to fuse primitives into one seamless organic surface, and vertex attributes that
carry rest positions and paint masks into the shaders."""
import bpy
import numpy as np

from . import skin

_tmp_count = 0


def new_mesh(name, verts, faces):
    """Mesh datablock from numpy data.  faces: (M, k) int array (uniform polygons, fast path) or a
    list of index tuples (mixed polygons)."""
    me = bpy.data.meshes.new(name)
    v = np.asarray(verts, dtype=np.float32)
    if isinstance(faces, np.ndarray) and faces.ndim == 2:
        k = faces.shape[1]
        me.vertices.add(len(v))
        me.vertices.foreach_set("co", v.ravel())
        me.loops.add(faces.size)
        me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
        me.polygons.add(len(faces))
        me.polygons.foreach_set("loop_start", np.arange(0, faces.size, k, dtype=np.int32))
        me.polygons.foreach_set("loop_total", np.full(len(faces), k, dtype=np.int32))
        me.update()
    else:
        me.from_pydata(v.tolist(), [], [tuple(int(i) for i in f) for f in faces])
        me.update()
    if len(me.polygons):
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    return me


def new_object(name, mesh, col, material=None, rest=True):
    """Object from a (verts, faces) pair; stores the rest position attribute used by shaders."""
    me = new_mesh(name, mesh[0], mesh[1])
    obj = bpy.data.objects.new(name, me)
    col.objects.link(obj)
    if rest:
        set_vector(me, "rest", mesh[0])
    if material is not None:
        me.materials.append(material)
    return obj


def set_vector(me, name, values):
    a = me.attributes.get(name) or me.attributes.new(name, "FLOAT_VECTOR", "POINT")
    a.data.foreach_set("vector", np.asarray(values, dtype=np.float32).ravel())


def set_fields(me, name, *fields):
    """Store up to 4 per-vertex scalar fields as the channels of a colour attribute `name`
    (read in shaders with an Attribute node: Color → Separate RGB, Alpha)."""
    n = len(me.vertices)
    data = np.zeros((n, 4), dtype=np.float32)
    data[:, 3] = 1.0
    for i, f in enumerate(fields):
        data[:, i] = np.asarray(f, dtype=np.float32)
    a = me.color_attributes.get(name) or me.color_attributes.new(name, "FLOAT_COLOR", "POINT")
    a.data.foreach_set("color", data.ravel())


def read_mesh(me):
    """(verts (N, 3) float64, faces): faces is an (M, k) array when all polygons have k corners."""
    n = len(me.vertices)
    v = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", v)
    tot = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("loop_total", tot)
    idx = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", idx)
    v = v.reshape(n, 3).astype(np.float64)
    if len(tot) and (tot == tot[0]).all():
        return v, idx.reshape(-1, int(tot[0]))
    start = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("loop_start", start)
    return v, [tuple(idx[s:s + t]) for s, t in zip(start, tot)]


def sculpt(parts, voxel=0.008, smooth=6, factor=0.5, cut=(), inflate=0.0):
    """Fuse closed primitive meshes into ONE watertight surface: union (→ optional boolean cuts)
    → voxel remesh → Laplacian smoothing, which rounds every junction into a soft fillet.  This is
    what makes sleeves grow out of the torso and muzzles out of the head without a seam.

    parts / cut: lists of (verts, faces).  Returns (verts, faces (M, 4) int array).
    inflate: metres added along the normal afterwards (compensates the smoothing shrink)."""
    global _tmp_count
    scene_col = bpy.context.scene.collection
    v, f = skin.merge(*parts)
    _tmp_count += 1
    me = new_mesh(f"vk_tmp{_tmp_count}", v, f)
    ob = bpy.data.objects.new(me.name, me)
    scene_col.objects.link(ob)
    cutters = []
    for i, c in enumerate(cut):
        cme = new_mesh(f"{me.name}_cut{i}", c[0], c[1])
        cob = bpy.data.objects.new(cme.name, cme)
        scene_col.objects.link(cob)
        cutters.append(cob)
    if cutters:
        # union first (voxel), then subtract, then remesh again so the cut edges are rounded too
        m = ob.modifiers.new("r0", "REMESH")
        m.mode, m.voxel_size, m.adaptivity = "VOXEL", voxel, 0.0
        for cob in cutters:
            b = ob.modifiers.new("b", "BOOLEAN")
            b.operation, b.object, b.solver = "DIFFERENCE", cob, "EXACT"
    m = ob.modifiers.new("r", "REMESH")
    m.mode, m.voxel_size, m.adaptivity = "VOXEL", voxel, 0.0
    if smooth > 0:
        s = ob.modifiers.new("s", "SMOOTH")
        s.factor, s.iterations = factor, smooth
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    out_v, out_f = read_mesh(ev.to_mesh())
    ev.to_mesh_clear()
    for o in [ob] + cutters:
        data = o.data
        bpy.data.objects.remove(o)
        bpy.data.meshes.remove(data)
    if not isinstance(out_f, np.ndarray):
        out_f = _quads(out_f)
    if inflate:
        out_v = out_v + skin.vertex_normals(out_v, out_f) * inflate
    return out_v, out_f


def _quads(faces):
    """Mixed polygons → (M, 4) array (triangles get a repeated corner)."""
    out = []
    for f in faces:
        f = list(f)
        if len(f) == 3:
            out.append(f + [f[2]])
        elif len(f) == 4:
            out.append(f)
        else:
            for k in range(1, len(f) - 1):
                out.append([f[0], f[k], f[k + 1], f[k + 1]])
    return np.array(out, dtype=np.int32)


def displace(mesh, amount):
    """Move vertices along their normals by a per-vertex amount (metres)."""
    v, f = mesh
    return v + skin.vertex_normals(v, f) * np.asarray(amount)[:, None], f


def tris(faces):
    """Triangle count of a face array / list."""
    if isinstance(faces, np.ndarray):
        return len(faces) * (faces.shape[1] - 2)
    return sum(len(f) - 2 for f in faces)
