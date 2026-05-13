"""
cube_mold.py

Build an open-top "tray" mold sized to receive the rectangular cube generated
by cube_visualize.py.  Cavity matches the cube's footprint with CLEARANCE on
all four side walls; the top is fully open so the cube can be dropped in for a
fit check, and the bottom face of the cube rests on the cavity floor (no
clearance below).

Outputs:
    cube_mold.stl          — solid mesh of the mold
    cube_mold_preview.png  — multi-view preview rendered with matplotlib
"""
import numpy as np
import trimesh
from manifold3d import Manifold, Mesh as MfdMesh

from cube_visualize import LENGTH, WIDTH, HEIGHT, render_views

# ── mold parameters ───────────────────────────────────────────────────────────
CLEARANCE = 0.25   # mm gap between cube and cavity wall on every side face
WALL      = 5.0    # mm mold wall thickness (floor and four sides)

OUTPUT_STL = "cube_mold.stl"
OUTPUT_PNG = "cube_mold_preview.png"


def _to_manifold(m):
    return Manifold(mesh=MfdMesh(
        vert_properties=m.vertices.astype(np.float32),
        tri_verts=m.faces.astype(np.uint32),
    ))


def _from_manifold(m):
    r = m.to_mesh()
    return trimesh.Trimesh(
        vertices=r.vert_properties[:, :3],
        faces=r.tri_verts,
        process=False,
    )


def build_mold():
    """Return the mold as a trimesh.Trimesh: outer block minus through-top cavity."""
    # Outer block: x ∈ [-WALL, LENGTH+WALL], y ∈ [-WALL, WIDTH+WALL], z ∈ [-WALL, HEIGHT]
    out_x = LENGTH + 2.0 * WALL
    out_y = WIDTH  + 2.0 * WALL
    out_z = HEIGHT + WALL                        # bottom slab + cavity-height walls
    outer = trimesh.creation.box(extents=[out_x, out_y, out_z])
    outer.apply_translation([LENGTH / 2.0, WIDTH / 2.0, (HEIGHT - WALL) / 2.0])

    # Cavity: x ∈ [-CL, LENGTH+CL], y ∈ [-CL, WIDTH+CL], z ∈ [0, HEIGHT+CL+overshoot]
    # The +overshoot guarantees the cavity exits cleanly through the top face.
    overshoot = WALL
    cav_x = LENGTH + 2.0 * CLEARANCE
    cav_y = WIDTH  + 2.0 * CLEARANCE
    cav_z = HEIGHT + CLEARANCE + overshoot
    cavity = trimesh.creation.box(extents=[cav_x, cav_y, cav_z])
    cavity.apply_translation([
        LENGTH / 2.0,
        WIDTH  / 2.0,
        (HEIGHT + CLEARANCE + overshoot) / 2.0,
    ])

    return _from_manifold(_to_manifold(outer) - _to_manifold(cavity))


if __name__ == "__main__":
    m = build_mold()
    print(f"Vertices : {len(m.vertices)}")
    print(f"Faces    : {len(m.faces)}")
    print(f"Watertight: {m.is_watertight}")
    print(f"Volume   : {m.volume:.2f} mm³  ({m.volume/1000:.2f} cm³)")
    print(f"Extents  : {m.extents}")
    print(f"Bounds   : x={m.bounds[:,0]}  y={m.bounds[:,1]}  z={m.bounds[:,2]}")
    m.export(OUTPUT_STL)
    print(f"Exported : {OUTPUT_STL}")

    render_views(
        m, OUTPUT_PNG,
        title=(
            f"Cube mold  —  cavity {LENGTH} × {WIDTH} × {HEIGHT} mm "
            f"+ {CLEARANCE} mm clearance, {WALL} mm walls, top open"
        ),
        color="#d97a4a",
        x_lim=(-WALL,  LENGTH + WALL),
        y_lim=(-WALL,  WIDTH  + WALL),
        z_lim=(-WALL,  HEIGHT + 2.0),
    )
    print(f"Rendered : {OUTPUT_PNG}")
