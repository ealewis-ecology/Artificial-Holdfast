"""pocket_analysis.py — standalone interstitial pocket analysis for haptera STL files.

Usage
-----
    python pocket_analysis.py <mesh.stl> [--voxel-size MM] [--top N] [--debug]

Arguments
---------
mesh.stl        Path to a watertight STL (or any trimesh-readable mesh file).
--voxel-size    Voxel side length in mm (default 0.5).  Smaller = more accurate
                but memory and runtime scale cubically.  Recommended: 0.5 – 2.0.
--top           Number of largest pockets to list in the report (default 20).
--debug         Print per-stage timing.

Output
------
A .txt report is written to the same directory as the input file, with the same
stem and a _pocketsMM suffix (e.g. haptera_…_pockets0.5mm.txt).  The same
content is also printed to stdout.

Method
------
Stage 1 — Voxelisation
  Both the mesh and its convex hull are rasterised onto a common axis-aligned
  grid.  A voxel is void when it lies inside the hull but outside the solid.
  trimesh.voxelized uses a scanline ray-cast strategy (Kaufman 1987).

Stage 2 — Connected-component labelling (CCL)
  scipy.ndimage.label with a 26-connected structuring element labels each
  topologically isolated void region as a separate pocket.
  (Rosenfeld & Pfaltz 1966, JACM 13(4), doi:10.1145/321356.321357)

Stage 3 — Euclidean distance transform (EDT)
  scipy.ndimage.distance_transform_edt gives each void voxel its distance to
  the nearest solid surface (mm).  The maximum EDT value in a pocket is the
  radius of the largest inscribed sphere — i.e. the largest organism that fits.
  (Blum 1967 medial axis; Maurer et al. 2003, IEEE TPAMI 25(2),
   doi:10.1109/TPAMI.2003.1177156)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Interstitial pocket analysis for a closed mesh STL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("mesh",        help="Path to the STL (or other trimesh-readable) file.")
    p.add_argument("--voxel-size", type=float, default=0.5,  metavar="MM",
                   help="Voxel side length in mm (default 0.5).")
    p.add_argument("--top",        type=int,   default=20,   metavar="N",
                   help="Number of largest pockets to list (default 20).")
    p.add_argument("--debug",      action="store_true",
                   help="Print per-stage timing to stderr.")
    p.add_argument("--chain",      action="store_true",
                   help="Find a chain of touching inscribed spheres through the void.")
    p.add_argument("--chain-max",  type=int,   default=50,  metavar="N",
                   help="Maximum number of spheres in the chain (default 50).")
    p.add_argument("--chain-tol",  type=float, default=2.0, metavar="VOXELS",
                   help="Touching tolerance in voxels: |d − (r₁+r₂)| ≤ tol×pitch (default 2.0).")
    p.add_argument("--plot",       action="store_true",
                   help="Show an interactive 3-D plot of the sphere chain (implies --chain).")
    p.add_argument("--plot-save",  metavar="FILE",
                   help="Save the 3-D plot to FILE (.png/.svg/.pdf) instead of displaying it.")
    return p.parse_args()


# ── Tee: write to stdout and file simultaneously ──────────────────────────────

class _Tee:
    def __init__(self, filepath):
        self.file   = open(filepath, "w")
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()
        sys.stdout = self.stdout


# ── voxelisation helper ───────────────────────────────────────────────────────

def _voxelize_to_grid(mesh, bbox_min, dims, pitch):
    """Rasterise *mesh* into a pre-sized boolean numpy array.

    Uses trimesh's VoxelGrid (scanline ray-cast) to find filled voxels, then
    maps their world-space centres onto the common grid defined by bbox_min and
    pitch.  Points outside dims are silently discarded (only possible if the
    mesh extends beyond the hull bounding box + border, which should not occur).

    Parameters
    ----------
    mesh     : trimesh.Trimesh
    bbox_min : (3,) array  — world-space origin of grid voxel [0,0,0]
    dims     : (3,) int array — grid extents in voxels
    pitch    : float — voxel side length (mm)

    Returns
    -------
    grid : boolean ndarray of shape dims
    """
    vg      = mesh.voxelized(pitch)
    centres = vg.points                          # (N, 3) world-space centres
    idx     = np.round((centres - bbox_min) / pitch).astype(int)
    valid   = np.all((idx >= 0) & (idx < dims), axis=1)
    idx     = idx[valid]
    grid    = np.zeros(dims, dtype=bool)
    if len(idx):
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


# ── sphere chain ─────────────────────────────────────────────────────────────

def _find_sphere_chain(void_mask, edt, pitch, bbox_min, max_spheres, tol_voxels):
    """Find a greedy chain of externally-tangent inscribed spheres in the void.

    Starting from the globally largest inscribed sphere (EDT maximum), each
    subsequent sphere is chosen as the largest sphere that:

      1. Is centred at a medial-axis voxel (local EDT maximum in a 3×3×3
         neighbourhood), ensuring it is a valid maximal inscribed sphere.
      2. Is externally tangent to the immediately preceding sphere:
             |dist(c_prev, c_new) − (r_prev + r_new)| ≤ tol_voxels × pitch
      3. Does not intersect any sphere already in the chain:
             dist(c_i, c_new) ≥ r_i + r_new  for all i.

    Parameters
    ----------
    void_mask  : bool ndarray (dims)       — True where space is free
    edt        : float ndarray (dims)      — Euclidean distance transform (mm)
    pitch      : float                     — voxel side length (mm)
    bbox_min   : (3,) array                — world-space origin of voxel [0,0,0]
    max_spheres: int                       — chain length cap
    tol_voxels : float                     — touching tolerance in voxels

    Returns
    -------
    list of dict  {'center': ndarray(3,) mm,  'radius': float mm}
    """
    from scipy import ndimage as ndi

    # Medial-axis candidates: voxels where EDT equals the 3×3×3 neighbourhood
    # maximum, i.e. locally largest inscribed spheres.
    edt_lmax  = ndi.maximum_filter(edt, size=3, mode='constant', cval=0.0)
    cand_mask = void_mask & (edt > 0) & (edt == edt_lmax)

    cand_coords = np.argwhere(cand_mask)          # (M, 3) voxel indices
    if len(cand_coords) == 0:
        return []

    cand_world = cand_coords * pitch + bbox_min   # (M, 3) world-space centres
    cand_r     = edt[cand_mask]                   # (M,)   sphere radii (mm)

    tol = tol_voxels * pitch                      # touching tolerance (mm)

    # ── seed: globally largest inscribed sphere ───────────────────────────────
    start = int(np.argmax(cand_r))
    chain = [{'center': cand_world[start].copy(), 'radius': float(cand_r[start])}]

    # Mark unavailable: candidates whose sphere would intersect the seed.
    # The touching sphere sits exactly at dist = r_seed + r_cand (boundary),
    # so the strict-less-than keeps touching candidates available.
    d0        = np.linalg.norm(cand_world - chain[0]['center'], axis=1)
    available = d0 >= chain[0]['radius'] + cand_r
    available[start] = False  # do not re-select the seed itself

    # ── greedy extension ──────────────────────────────────────────────────────
    for _ in range(max_spheres - 1):
        prev_c = chain[-1]['center']
        prev_r = chain[-1]['radius']

        av_idx = np.where(available)[0]
        if len(av_idx) == 0:
            break

        d    = np.linalg.norm(cand_world[av_idx] - prev_c, axis=1)
        r_av = cand_r[av_idx]

        # Touching condition
        touch = np.abs(d - (prev_r + r_av)) <= tol
        if not touch.any():
            break

        # Pick the touching candidate with the largest radius
        touching_global = av_idx[touch]
        best = touching_global[int(np.argmax(cand_r[touching_global]))]

        new_c = cand_world[best].copy()
        new_r = float(cand_r[best])
        chain.append({'center': new_c, 'radius': new_r})

        # Exclude from future steps: anything that would intersect the new sphere
        d_new = np.linalg.norm(cand_world - new_c, axis=1)
        available[d_new < new_r + cand_r] = False
        available[best] = False

    return chain


# ── visualisation ─────────────────────────────────────────────────────────────

def _visualize_chain(sphere_chain, hull, mesh_path, pitch, save_path=None):
    """Render the sphere chain in a 3-D matplotlib figure.

    Shows the convex hull as a faint semi-transparent context shell, the
    sphere chain as coloured translucent spheres (colour = inscribed radius),
    and a centre-to-centre spine line.

    Parameters
    ----------
    sphere_chain : list of dict  {'center': ndarray(3,), 'radius': float}
    hull         : trimesh.Trimesh  — convex hull for spatial context
    mesh_path    : Path             — used for the figure title
    pitch        : float            — voxel size (mm), shown in title
    save_path    : Path or None     — if given, save there; else plt.show()
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection   # noqa: F401
    except ImportError:
        print("  ! matplotlib not available — install it with:  pip install matplotlib",
              file=sys.stderr)
        return

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # ── convex hull shell (spatial context) ───────────────────────────────────
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    hull_tris = hull.vertices[hull.faces]          # (F, 3, 3)
    hull_poly = Poly3DCollection(
        hull_tris,
        alpha=0.06, facecolor='steelblue',
        edgecolor='steelblue', linewidth=0.15)
    ax.add_collection3d(hull_poly)

    # ── sphere surfaces ───────────────────────────────────────────────────────
    # Low-res parametric sphere grid (fast enough for ≤ 100 spheres)
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0,     np.pi, 18)
    base_x = np.outer(np.cos(u), np.sin(v))   # (28, 18)
    base_y = np.outer(np.sin(u), np.sin(v))
    base_z = np.outer(np.ones_like(u), np.cos(v))

    radii = np.array([s['radius'] for s in sphere_chain])
    r_min, r_max = radii.min(), radii.max()
    cmap  = plt.cm.plasma
    norm  = plt.Normalize(r_min, r_max)

    for s in sphere_chain:
        cx, cy, cz = s['center']
        r = s['radius']
        color = cmap(norm(r))
        ax.plot_surface(
            cx + r * base_x,
            cy + r * base_y,
            cz + r * base_z,
            color=color, alpha=0.40,
            linewidth=0, antialiased=False)

    # ── centre-to-centre spine ────────────────────────────────────────────────
    centers = np.array([s['center'] for s in sphere_chain])
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
            color='k', linewidth=1.4, alpha=0.75, zorder=5)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
               c='k', s=18, zorder=6)

    # Label first and last sphere
    for idx, label in ((0, f"S1\nr={sphere_chain[0]['radius']:.2f} mm"),
                       (-1, f"S{len(sphere_chain)}\nr={sphere_chain[-1]['radius']:.2f} mm")):
        cx, cy, cz = sphere_chain[idx]['center']
        ax.text(cx, cy, cz, label, fontsize=7, ha='center', va='bottom')

    # ── axes, labels, colorbar ────────────────────────────────────────────────
    b = hull.bounds
    ax.set_xlim(b[0, 0], b[1, 0])
    ax.set_ylim(b[0, 1], b[1, 1])
    ax.set_zlim(b[0, 2], b[1, 2])
    ax.set_xlabel('X (mm)', labelpad=6)
    ax.set_ylabel('Y (mm)', labelpad=6)
    ax.set_zlabel('Z (mm)', labelpad=6)
    ax.set_title(
        f'{mesh_path.name} — sphere chain\n'
        f'{len(sphere_chain)} spheres · voxel {pitch} mm',
        fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.10,
                      label='Inscribed radius (mm)')
    cb.ax.tick_params(labelsize=8)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}", file=sys.stderr)
    else:
        plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args  = _parse_args()
    # --plot / --plot-save imply --chain
    if args.plot or args.plot_save:
        args.chain = True
    pitch = float(args.voxel_size)
    debug = args.debug
    dlog  = (lambda msg: print(msg, file=sys.stderr)) if debug else (lambda _: None)

    # ── load mesh ─────────────────────────────────────────────────────────────
    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        sys.exit(f"Error: file not found: {mesh_path}")

    print(f"Loading {mesh_path.name}...")
    t0   = time.perf_counter()
    mesh = trimesh.load(str(mesh_path), force="mesh")
    dlog(f"  loaded  {time.perf_counter()-t0:.2f}s  "
         f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")

    if not mesh.is_watertight:
        print("  warning: mesh is not watertight — volume and voxelisation may be inaccurate.")

    # ── basic mesh measurements ───────────────────────────────────────────────
    t0           = time.perf_counter()
    mesh_volume  = mesh.volume
    mesh_area    = mesh.area
    base_mask    = mesh.face_normals[:, 2] < -0.999
    base_cap_area = float(trimesh.triangles.area(mesh.triangles[base_mask]).sum()) if base_mask.any() else 0.0
    wetted_area  = mesh_area - base_cap_area
    dlog(f"  mesh measurements  {time.perf_counter()-t0:.2f}s")

    t0   = time.perf_counter()
    hull = mesh.convex_hull
    hull_volume  = hull.volume
    hull_area    = hull.area
    hull_base_mask = hull.face_normals[:, 2] < -0.999
    hull_base_area = float(trimesh.triangles.area(hull.triangles[hull_base_mask]).sum()) if hull_base_mask.any() else 0.0
    dlog(f"  convex hull  {time.perf_counter()-t0:.2f}s")

    interstitial_volume = hull_volume - mesh_volume
    sa_to_vol           = wetted_area / interstitial_volume if interstitial_volume > 0 else 0.0
    total_bounding_area = (hull_area - hull_base_area) + (mesh_area - base_cap_area)

    # ── pocket analysis ───────────────────────────────────────────────────────
    print(f"Running pocket analysis at {pitch} mm/voxel...")
    t_analysis = time.perf_counter()

    pocket_analysis_ok = False
    pocket_error       = ""
    pocket_stats       = []
    void_mask          = None
    dims               = None
    n_pockets          = 0
    voxel_vol          = pitch ** 3

    try:
        from scipy import ndimage as ndi

        # Build a common integer grid aligned to the hull bounding box.
        # 2-voxel border ensures no pocket is clipped by the grid edge.
        bbox_min = hull.bounds[0] - pitch * 2.0
        bbox_max = hull.bounds[1] + pitch * 2.0
        dims     = np.ceil((bbox_max - bbox_min) / pitch).astype(int) + 1

        dlog(f"  grid {tuple(dims)}  ({dims.prod():,} voxels)")

        t0          = time.perf_counter()
        solid_grid  = _voxelize_to_grid(mesh, bbox_min, dims, pitch)
        hull_grid   = _voxelize_to_grid(hull, bbox_min, dims, pitch)
        void_mask   = hull_grid & ~solid_grid
        dlog(f"  voxelisation  {time.perf_counter()-t0:.2f}s  "
             f"void_voxels={int(void_mask.sum()):,}")

        # Stage 2: connected-component labelling (26-connectivity)
        t0                   = time.perf_counter()
        struct26             = ndi.generate_binary_structure(3, 3)
        labels, n_pockets    = ndi.label(void_mask, structure=struct26)
        dlog(f"  CCL  {time.perf_counter()-t0:.2f}s  pockets={n_pockets}")

        # Stage 3: Euclidean distance transform
        t0  = time.perf_counter()
        edt = ndi.distance_transform_edt(void_mask, sampling=pitch)
        dlog(f"  EDT  {time.perf_counter()-t0:.2f}s")

        # Per-pocket statistics using find_objects for efficiency
        slices = ndi.find_objects(labels)
        for label_idx, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            region_mask = labels[sl] == label_idx
            region_edt  = edt[sl][region_mask]
            n_vox       = int(region_mask.sum())
            pocket_stats.append({
                'label':            label_idx,
                'voxels':           n_vox,
                'volume':           n_vox * voxel_vol,
                'max_inscribed_r':  float(region_edt.max()),
                'mean_inscribed_r': float(region_edt.mean()),
            })

        pocket_stats.sort(key=lambda p: p['volume'], reverse=True)

        pocket_analysis_ok = True
        dlog(f"  total analysis  {time.perf_counter()-t_analysis:.2f}s")

    except Exception as exc:
        pocket_error = str(exc)

    # ── sphere chain ──────────────────────────────────────────────────────────
    sphere_chain = []
    if args.chain and pocket_analysis_ok:
        print(f"Finding sphere chain (max {args.chain_max} spheres, "
              f"tol {args.chain_tol} voxels)...")
        t0 = time.perf_counter()
        sphere_chain = _find_sphere_chain(
            void_mask, edt, pitch, bbox_min,
            max_spheres=args.chain_max,
            tol_voxels=args.chain_tol,
        )
        dlog(f"  sphere chain  {time.perf_counter()-t0:.2f}s  "
             f"spheres={len(sphere_chain)}")

    # ── organism size bins ────────────────────────────────────────────────────
    size_bin_edges  = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, float('inf')]
    size_bin_labels = ['< 0.5', '0.5 – 1', '1 – 2', '2 – 5', '5 – 10', '10 – 20', '> 20']
    bin_counts  = [0]   * len(size_bin_labels)
    bin_volumes = [0.0] * len(size_bin_labels)
    if pocket_analysis_ok:
        for p in pocket_stats:
            for i, edge in enumerate(size_bin_edges):
                if p['max_inscribed_r'] < edge:
                    bin_counts[i]  += 1
                    bin_volumes[i] += p['volume']
                    break

    # ── redirect stdout → tee ─────────────────────────────────────────────────
    out_path  = mesh_path.with_name(mesh_path.stem + f"_pockets{pitch}mm.txt")
    tee       = _Tee(out_path)
    sys.stdout = tee

    # ── report ────────────────────────────────────────────────────────────────
    print(f"\nMesh     : {mesh_path}")
    print(f"")
    print(f"Mesh")
    print(f"  vertices               : {len(mesh.vertices)}")
    print(f"  faces                  : {len(mesh.faces)}")
    print(f"  watertight             : {mesh.is_watertight}")
    print(f"")
    print(f"Solid")
    print(f"  volume                 : {mesh_volume:.4f} mm³")
    print(f"  surface area (w/ cap)  : {mesh_area:.4f} mm²")
    print(f"  wetted surface area    : {wetted_area:.4f} mm²  (base cap excluded)")
    print(f"  base cap area          : {base_cap_area:.4f} mm²")
    print(f"")
    print(f"Convex hull (bounding envelope)")
    print(f"  hull volume            : {hull_volume:.4f} mm³")
    print(f"  hull surface area      : {hull_area:.4f} mm²")
    print(f"")
    print(f"Interstitial space")
    print(f"  volume                 : {interstitial_volume:.4f} mm³  (hull − solid)")
    print(f"  fraction of hull       : {interstitial_volume / hull_volume:.6f}")
    print(f"  external surface area  : {hull_area:.4f} mm²  (hull envelope)")
    print(f"  internal surface area  : {mesh_area:.4f} mm²  (full mesh)")
    print(f"  total bounding area    : {total_bounding_area:.4f} mm²  (external + internal; ground-contact excluded)")
    print(f"  SA / volume ratio      : {sa_to_vol:.4f} mm⁻¹  (complexity index, wetted SA)")
    print(f"")
    print(f"Interstitial pocket analysis")
    print(f"  method")
    print(f"    voxelisation         : trimesh.voxelized (scanline ray-cast; Kaufman 1987)")
    print(f"    pocket detection     : scipy.ndimage.label, 26-connectivity (Rosenfeld & Pfaltz 1966)")
    print(f"    organism size        : scipy.ndimage.distance_transform_edt, max inscribed sphere")
    print(f"                           radius (Blum 1967 medial axis; Maurer et al. 2003 algorithm)")
    if pocket_analysis_ok:
        total_void_vol_vox = int(void_mask.sum()) * voxel_vol
        print(f"  voxel size             : {pitch} mm  →  {voxel_vol:.4f} mm³/voxel")
        print(f"  grid dimensions        : {dims[0]} × {dims[1]} × {dims[2]} voxels")
        print(f"  void voxels            : {int(void_mask.sum()):,}")
        print(f"  void volume (voxelised): {total_void_vol_vox:.2f} mm³"
              f"  (cf. hull − solid = {interstitial_volume:.2f} mm³)")
        print(f"  isolated pockets       : {n_pockets}")
        print(f"")
        print(f"  Organism size distribution  (max inscribed sphere radius in pocket, mm)")
        print(f"  {'Radius bin (mm)':<16}  {'Pockets':>8}  {'Void vol (mm³)':>16}  {'% void vol':>11}")
        tvv = sum(p['volume'] for p in pocket_stats) or 1.0
        for lbl, cnt, bvol in zip(size_bin_labels, bin_counts, bin_volumes):
            pct = 100.0 * bvol / tvv
            print(f"  {lbl:<16}  {cnt:>8}  {bvol:>16.2f}  {pct:>10.1f}%")
        print(f"")
        top_n = min(args.top, len(pocket_stats))
        print(f"  Largest {top_n} pockets by volume")
        print(f"  {'Rank':<5}  {'Volume (mm³)':>14}  {'Max r (mm)':>11}  {'Mean r (mm)':>12}  {'Voxels':>8}")
        for ri, p in enumerate(pocket_stats[:top_n], 1):
            print(f"  {ri:<5}  {p['volume']:>14.2f}  {p['max_inscribed_r']:>11.3f}"
                  f"  {p['mean_inscribed_r']:>12.3f}  {p['voxels']:>8}")
    else:
        print(f"  ! analysis failed: {pocket_error}")
        print(f"  Ensure scipy is installed:  pip install scipy")

    if args.chain:
        print(f"")
        print(f"Sphere chain  (touching inscribed spheres through the void)")
        print(f"  method       : greedy from global EDT maximum; each sphere externally")
        print(f"                 tangent to the previous, non-intersecting with all prior")
        print(f"  candidates   : medial-axis voxels (3×3×3 local EDT maxima)")
        print(f"  tolerance    : {args.chain_tol} voxels = {args.chain_tol * pitch:.3f} mm")
        if not pocket_analysis_ok:
            print(f"  ! skipped — pocket analysis did not complete")
        elif not sphere_chain:
            print(f"  ! no chain found (void may be empty or fully disconnected)")
        else:
            total_chain_len = sum(
                float(np.linalg.norm(sphere_chain[i+1]['center'] - sphere_chain[i]['center']))
                for i in range(len(sphere_chain) - 1)
            )
            print(f"  spheres found : {len(sphere_chain)}")
            print(f"  path length   : {total_chain_len:.3f} mm  (centre-to-centre sum)")
            print(f"  largest r     : {sphere_chain[0]['radius']:.3f} mm  (sphere 1, seed)")
            print(f"  smallest r    : {min(s['radius'] for s in sphere_chain):.3f} mm")
            print(f"")
            print(f"  {'#':<5}  {'cx (mm)':>10}  {'cy (mm)':>10}  {'cz (mm)':>10}"
                  f"  {'r (mm)':>9}  {'gap to prev (mm)':>17}")
            for i, s in enumerate(sphere_chain):
                cx, cy, cz = s['center']
                if i == 0:
                    gap_str = "       —"
                else:
                    d = float(np.linalg.norm(s['center'] - sphere_chain[i-1]['center']))
                    gap = d - (sphere_chain[i-1]['radius'] + s['radius'])
                    gap_str = f"{gap:+.3f}"
                print(f"  {i+1:<5}  {cx:>10.3f}  {cy:>10.3f}  {cz:>10.3f}"
                      f"  {s['radius']:>9.3f}  {gap_str:>17}")

            # Write CSV alongside the mesh file
            chain_csv = mesh_path.with_name(
                mesh_path.stem + f"_chain{pitch}mm.csv")
            with open(chain_csv, "w") as cf:
                cf.write("sphere,cx_mm,cy_mm,cz_mm,radius_mm,gap_to_prev_mm\n")
                for i, s in enumerate(sphere_chain):
                    cx, cy, cz = s['center']
                    if i == 0:
                        gap = ""
                    else:
                        d = float(np.linalg.norm(s['center'] - sphere_chain[i-1]['center']))
                        gap = f"{d - sphere_chain[i-1]['radius'] - s['radius']:.4f}"
                    cf.write(f"{i+1},{cx:.4f},{cy:.4f},{cz:.4f},"
                             f"{s['radius']:.4f},{gap}\n")
            print(f"")
            print(f"  Chain CSV: {chain_csv}")

    tee.close()
    print(f"\nReport written to: {out_path}", file=sys.stderr)

    # ── 3-D visualisation ─────────────────────────────────────────────────────
    if args.plot or args.plot_save:
        if not sphere_chain:
            print("  ! Nothing to plot — sphere chain is empty.", file=sys.stderr)
        else:
            save = Path(args.plot_save) if args.plot_save else None
            _visualize_chain(sphere_chain, hull, mesh_path, pitch, save_path=save)


if __name__ == "__main__":
    main()
