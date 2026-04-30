"""Headless Blender script: repair an STL mesh to be watertight.

Run as:
    blender --background --python blender_repair.py -- <in.stl> <out.stl> [voxel_size_mm]

Strategy
  1. Import STL.
  2. Voxel-Remesh modifier (default 0.5 mm voxel) — reconstructs a
     guaranteed-watertight surface from an internal voxel grid.
  3. Export STL.

A merge-by-distance + recalc-normals pre-pass was tested and made things worse
on these meshes (collapsed near-coincident verts created new non-manifold
edges). Voxel remesh alone reliably produces watertight output.

Reports non-manifold edge count before and after, exits non-zero if the final
mesh is still non-manifold.
"""

import sys
import os

import bpy
import bmesh


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        print("ERROR: expected arguments after `--`", flush=True)
        sys.exit(2)
    args = argv[argv.index("--") + 1:]
    if len(args) < 2:
        print("Usage: blender --background --python blender_repair.py -- "
              "<in.stl> <out.stl> [voxel_size_mm]", flush=True)
        sys.exit(2)
    input_path = args[0]
    output_path = args[1]
    voxel_size = float(args[2]) if len(args) >= 3 else 0.5
    return input_path, output_path, voxel_size


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_stl(path):
    print(f"[blender_repair] importing {path}", flush=True)
    try:
        bpy.ops.wm.stl_import(filepath=path)
    except AttributeError:
        bpy.ops.import_mesh.stl(filepath=path)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    return obj


def export_stl(obj, path):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    print(f"[blender_repair] exporting to {path}", flush=True)
    try:
        bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True)
    except AttributeError:
        bpy.ops.export_mesh.stl(filepath=path, use_selection=True)


def count_non_manifold_edges(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()
    n = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return n


def ensure_single_user(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.make_single_user(object=True, obdata=True)


def voxel_remesh(obj, voxel_size):
    print(f"[blender_repair] voxel remesh @ {voxel_size} mm",
          flush=True)
    ensure_single_user(obj)
    mod = obj.modifiers.new(name="Remesh", type='REMESH')
    mod.mode = 'VOXEL'
    mod.voxel_size = voxel_size
    mod.use_remove_disconnected = False
    mod.use_smooth_shade = False
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Remesh")


def triangulate_and_weld(obj):
    """Voxel remesh outputs quads with unique per-face vertices. Triangulate
    to tris and weld coincident verts so the exported STL is a clean tri mesh
    with no T-junctions or duplicate vertices.
    """
    print("[blender_repair] triangulate + weld coincident verts", flush=True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.00001)
    bpy.ops.object.mode_set(mode='OBJECT')


def main():
    input_path, output_path, voxel_size = parse_args()

    clear_scene()
    obj = import_stl(input_path)
    mesh = obj.data
    print(f"[blender_repair] imported: {len(mesh.vertices):,} verts, "
          f"{len(mesh.polygons):,} faces", flush=True)

    n0 = count_non_manifold_edges(mesh)
    print(f"[blender_repair] initial non-manifold edges: {n0:,}", flush=True)

    voxel_remesh(obj, voxel_size)
    n1 = count_non_manifold_edges(obj.data)
    print(f"[blender_repair] after voxel remesh: {n1:,} non-manifold edges",
          flush=True)

    triangulate_and_weld(obj)
    final_n = count_non_manifold_edges(obj.data)
    print(f"[blender_repair] after triangulate+weld: {final_n:,} non-manifold edges",
          flush=True)

    print(f"[blender_repair] final mesh: {len(obj.data.vertices):,} verts, "
          f"{len(obj.data.polygons):,} faces", flush=True)

    export_stl(obj, output_path)

    print(f"[blender_repair] done. final_non_manifold_edges={final_n}",
          flush=True)
    sys.exit(0 if final_n == 0 else 1)


if __name__ == "__main__":
    main()
