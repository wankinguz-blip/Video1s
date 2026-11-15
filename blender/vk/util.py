"""Small bpy helpers shared by all scene modules."""
import bpy
import numpy as np


def collection(name, parent=None):
    """Get or create a collection linked under `parent` (default: scene collection)."""
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(col)
    return col


def link(obj, col):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def mesh_object(name, verts, faces, col, smooth=True, edges=()):
    """Create a mesh object from numpy/py data.  verts: (N, 3); faces: list of index tuples."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in np.asarray(verts)], [tuple(e) for e in edges], [tuple(f) for f in faces])
    me.update()
    if smooth and len(me.polygons):
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    obj = bpy.data.objects.new(name, me)
    col.objects.link(obj)
    return obj


def grid_mesh(name, xs, ys, zfunc, col, smooth=True):
    """Height-field grid: xs, ys are 1-D arrays; zfunc(X, Y) -> Z (numpy)."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    Z = zfunc(X, Y)
    nx, ny = len(xs), len(ys)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    i = np.arange(nx - 1)
    j = np.arange(ny - 1)
    I, J = np.meshgrid(i, j)
    a = (J * nx + I).ravel()
    faces = np.stack([a, a + 1, a + nx + 1, a + nx], axis=1)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(faces.size)
    me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", np.arange(0, faces.size, 4, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(faces), 4, dtype=np.int32))
    me.update()
    me.validate()
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(faces), dtype=bool))
    obj = bpy.data.objects.new(name, me)
    col.objects.link(obj)
    return obj


def material(name):
    """New node material; returns (material, nodes, links, principled_bsdf, output)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    out = nt.nodes.get("Material Output")
    return mat, nt.nodes, nt.links, bsdf, out


def principled(name, color, roughness=0.6, metallic=0.0, emission=None, emission_strength=0.0, **inputs):
    """Quick Principled BSDF material.  color: (r, g, b) linear."""
    mat, nodes, links, bsdf, out = material(name)
    bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission[:3], 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    for k, v in inputs.items():
        bsdf.inputs[k.replace("_", " ")].default_value = v
    return mat


def srgb(r, g=None, b=None):
    """sRGB (0..255 ints, 0..1 floats or '#rrggbb') → linear RGB tuple."""
    if isinstance(r, str):
        h = r.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    if max(r, g, b) > 1.0:
        r, g, b = r / 255.0, g / 255.0, b / 255.0

    def f(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return (f(r), f(g), f(b))


# ---------------------------------------------------------------- loop time for shaders
_TIME_GROUP = "VK_Time"


def time_group():
    """Node group with outputs 'Seconds' (loop time) and 'Phase' (0..1 over the loop).
    Add it to a material with time_node(nodes).  show.update(t) keeps it current, so shaders can
    animate with sin(2π·k·Phase) (integer k) and loop exactly."""
    ng = bpy.data.node_groups.get(_TIME_GROUP)
    if ng is None:
        ng = bpy.data.node_groups.new(_TIME_GROUP, "ShaderNodeTree")
        ng.interface.new_socket("Seconds", in_out="OUTPUT", socket_type="NodeSocketFloat")
        ng.interface.new_socket("Phase", in_out="OUTPUT", socket_type="NodeSocketFloat")
        out = ng.nodes.new("NodeGroupOutput")
        sec = ng.nodes.new("ShaderNodeValue")
        sec.name = "seconds"
        ph = ng.nodes.new("ShaderNodeValue")
        ph.name = "phase"
        ng.links.new(sec.outputs[0], out.inputs["Seconds"])
        ng.links.new(ph.outputs[0], out.inputs["Phase"])
    return ng


def time_node(nodes):
    n = nodes.new("ShaderNodeGroup")
    n.node_tree = time_group()
    return n


def set_time(t, loop):
    ng = time_group()
    ng.nodes["seconds"].outputs[0].default_value = t
    ng.nodes["phase"].outputs[0].default_value = (t % loop) / loop
    # make sure every user of the group is re-evaluated for this frame
    ng.update_tag()
    for mat in bpy.data.materials:
        if mat.use_nodes and any(n.type == "GROUP" and n.node_tree == ng for n in mat.node_tree.nodes):
            mat.update_tag()
    for w in bpy.data.worlds:
        if w.use_nodes and any(n.type == "GROUP" and n.node_tree == ng for n in w.node_tree.nodes):
            w.update_tag()
