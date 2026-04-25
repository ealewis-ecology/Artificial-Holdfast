"""
cube_visualize.py

Build and visualize a plain rectangular cube modelling a dive weight:
    Length (X) : 103.5 mm
    Width  (Y) :  89.0 mm
    Height (Z) :  25.5 mm

Outputs:
    cube.stl          — solid mesh, openable in any 3D viewer
    cube_preview.png  — multi-view preview rendered with matplotlib
"""
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── shape parameters ──────────────────────────────────────────────────────────
LENGTH = 103.5   # mm, X axis
WIDTH  =  89.0   # mm, Y axis
HEIGHT =  25.5   # mm, Z axis

OUTPUT_STL = "cube.stl"
OUTPUT_PNG = "cube_preview.png"


def build_box():
    """Solid rectangular cube placed with one corner at the origin (0,0,0)."""
    m = trimesh.creation.box(extents=[LENGTH, WIDTH, HEIGHT])
    m.apply_translation([LENGTH / 2.0, WIDTH / 2.0, HEIGHT / 2.0])
    return m


def render_views(mesh, path, title, color="#4a90d9", x_lim=None, y_lim=None, z_lim=None):
    """Render four standard views of mesh (iso, side, top, end) to a PNG."""
    if x_lim is None: x_lim = (0, LENGTH)
    if y_lim is None: y_lim = (0, WIDTH)
    if z_lim is None: z_lim = (0, HEIGHT * 1.1)

    views = [
        ("Isometric",       (25, -60)),
        ("Side  (along Y)", ( 0,  90)),
        ("Top   (along Z)", (89, -90)),
        ("End   (along X)", ( 0,   0)),
    ]
    fig = plt.figure(figsize=(13, 10))
    tri = mesh.vertices[mesh.faces]
    for k, (vname, (elev, azim)) in enumerate(views, 1):
        ax  = fig.add_subplot(2, 2, k, projection="3d")
        col = Poly3DCollection(tri, alpha=0.85, edgecolor="k", linewidths=0.12)
        col.set_facecolor(color)
        ax.add_collection3d(col)
        ax.set_xlim(*x_lim); ax.set_ylim(*y_lim); ax.set_zlim(*z_lim)
        ax.set_box_aspect((x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
        ax.set_title(f"{vname}  (elev={elev}, azim={azim})")
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    m = build_box()
    print(f"Vertices : {len(m.vertices)}")
    print(f"Faces    : {len(m.faces)}")
    print(f"Watertight: {m.is_watertight}")
    print(f"Volume   : {m.volume:.2f} mm³  ({m.volume/1000:.2f} cm³)")
    print(f"Extents  : {m.extents}")
    m.export(OUTPUT_STL)
    print(f"Exported : {OUTPUT_STL}")
    render_views(
        m, OUTPUT_PNG,
        title=f"Rectangular cube  —  {LENGTH} × {WIDTH} × {HEIGHT} mm",
    )
    print(f"Rendered : {OUTPUT_PNG}")
