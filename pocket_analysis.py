"""pocket_analysis.py — GUI application for interstitial pocket analysis of haptera STL files.

Run with:
    python pocket_analysis.py

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

Stage 4 — Greedy non-overlapping inscribed-sphere packing (per pocket)
  For each pocket, the largest inscribed sphere is identified from the EDT and
  carved out; the EDT is recomputed and the next-largest sphere is placed; the
  process repeats until no sphere of at least the cut-off diameter fits.  The
  cumulative size-distribution of the resulting void spheres follows the
  number-size fractal relation N(>=d) ∝ d^(-D_s) (Mandelbrot 1982, "The
  Fractal Geometry of Nature", W.H. Freeman; cf. Frontier 1987, "Applications
  of fractal theory to ecology", in Legendre & Legendre eds., Developments in
  Numerical Ecology, Springer): a least-squares fit on log-log axes gives the
  habitat fractal index D_s.  Reviewed in the context of aquatic habitat
  complexity by Tokeshi & Arakaki 2012, Hydrobiologia 685: 27–47.

Stage 5 — Box-counting fractal dimension (Orland et al. 2016)
  The solid voxel grid is covered with cubic boxes of decreasing edge length r
  (powers of 2 voxels); N(r) = number of boxes intersecting the solid.  The
  fractal dimension is D_b = -d log N / d log r, fit by least squares over the
  scaling range.  This follows Orland, Cameron, Lock-Wah-Hoon, Stephens &
  Erskine 2016, "Application of computer-aided tomography techniques to
  visualize kelp holdfast structure reveals the importance of habitat
  complexity for supporting marine biodiversity" (J. Exp. Mar. Biol. Ecol. 477,
  doi:10.1016/j.jembe.2016.01.003).

Stage 6 — Hidden-space / overhang angle metrics  (per-voxel)
  Tokeshi & Arakaki 2012 review "minimum angle" approaches to overhang and
  refuge quantification but do not pin down a single equation, so four
  candidate per-voxel metrics are computed for side-by-side comparison:
    1. Surface inclination angle  θ_n = ∠(outward normal, +Z) for every solid
       surface voxel.  Values > 90° flag overhanging surfaces (the standard
       reef-rugosity overhang test).
    2. Sky-view factor  SVF = (rays unblocked to outside the hull) / (rays
       cast in the upper hemisphere) for every void voxel.  Direct analogue of
       the urban-canyon / canopy sky-view factor (Oke 1981; Steyn 1980).
    3. Crevice opening solid-angle, expressed as the half-angle α of an
       equivalent circular cone with the same solid angle, for every void
       voxel adjacent to solid.  Sampled with all 26 integer-step directions.
    4. Local dihedral angle = the maximum pairwise angle between outward
       normals of solid voxels in the 3×3×3 neighbourhood of every surface-
       adjacent void voxel — a corner / concavity descriptor.
  Outward normals are computed as the gradient of a Gaussian-smoothed solid
  occupancy field (Sobel filter); ray escape is evaluated by iterative shifted
  cumulative-OR until every voxel's fate is decided.
"""

import json
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import trimesh


# ── Persistent config ─────────────────────────────────────────────────────────

CONFIG_PATH = Path.home() / ".pocket_analysis_config.json"


def _load_config():
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text())
    except Exception:
        pass
    return {}


def _save_config(cfg):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


# ── Voxelisation helper ───────────────────────────────────────────────────────

def _voxelize_to_grid(mesh, bbox_min, dims, pitch):
    """Rasterise *mesh* into a pre-sized boolean numpy array.

    Uses trimesh's VoxelGrid (scanline ray-cast) to find filled voxels, then
    maps their world-space centres onto the common grid defined by bbox_min and
    pitch.
    """
    vg      = mesh.voxelized(pitch)
    centres = vg.points
    idx     = np.round((centres - bbox_min) / pitch).astype(int)
    valid   = np.all((idx >= 0) & (idx < dims), axis=1)
    idx     = idx[valid]
    grid    = np.zeros(dims, dtype=bool)
    if len(idx):
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


# ── Greedy non-overlapping inscribed-sphere packing ──────────────────────────

def _pack_spheres_in_pocket(pocket_local_mask, pitch, min_diameter_mm,
                            max_spheres=10000):
    """Greedy non-overlapping inscribed-sphere packing for one pocket.

    The largest inscribed sphere is repeatedly identified from the EDT,
    carved out of the available void, and its size and centre recorded.
    Subsequent (smaller) spheres fill the residual interstices between the
    earlier ones, building up a size distribution of refuges within the
    pocket.

    Parameters
    ----------
    pocket_local_mask : 3-D bool array — True where pocket voxels lie.
                        Must be padded by ≥1 voxel of False on every side so
                        the EDT sees the pocket boundary.
    pitch             : float — voxel edge length (mm).
    min_diameter_mm   : float — stop when the largest fitting sphere's
                                diameter falls below this value.
    max_spheres       : int — safety cap on iterations.

    Returns
    -------
    list of dicts with keys 'diameter_mm', 'radius_mm', 'center_local_vox'
    (the (i,j,k) index of the sphere centre within pocket_local_mask).
    """
    from scipy import ndimage as ndi

    available = pocket_local_mask.copy()
    spheres   = []
    min_r_mm  = 0.5 * min_diameter_mm

    while len(spheres) < max_spheres:
        if not available.any():
            break
        edt = ndi.distance_transform_edt(available, sampling=pitch)
        max_r = float(edt.max())
        if max_r < min_r_mm:
            break
        idx = np.unravel_index(int(np.argmax(edt)), edt.shape)

        # Carve out sphere of radius max_r centred at idx.
        rv     = max_r / pitch
        rcell  = int(np.ceil(rv))
        zlo, zhi = max(0, idx[0]-rcell), min(available.shape[0], idx[0]+rcell+1)
        ylo, yhi = max(0, idx[1]-rcell), min(available.shape[1], idx[1]+rcell+1)
        xlo, xhi = max(0, idx[2]-rcell), min(available.shape[2], idx[2]+rcell+1)
        zz, yy, xx = np.ogrid[zlo:zhi, ylo:yhi, xlo:xhi]
        mask = (zz - idx[0])**2 + (yy - idx[1])**2 + (xx - idx[2])**2 <= rv**2
        available[zlo:zhi, ylo:yhi, xlo:xhi] &= ~mask

        spheres.append({
            'diameter_mm':      2.0 * max_r,
            'radius_mm':        max_r,
            'center_local_vox': np.array(idx, dtype=int),
        })

    return spheres


# ── Box-counting fractal dimension (Orland et al. 2016) ──────────────────────

def _box_counting_dimension(solid_grid, pitch):
    """Compute the box-counting fractal dimension D_b of a binary 3-D grid.

    The grid is partitioned into cubic boxes of edge length s voxels (s = 1,
    2, 4, 8, …) and N(s) = number of boxes that contain at least one True
    voxel is counted.  D_b is the slope of log N(r) vs log(1/r), where
    r = s * pitch is the box edge in mm, fit by least squares.

    Returns
    -------
    D_b      : float — fractal dimension (NaN if not enough scales)
    box_mm   : 1-D ndarray — box edge length r at each scale (mm)
    counts   : 1-D ndarray — N(r) at each scale
    fit_used : 1-D bool ndarray — which scales were used in the linear fit
    """
    if not solid_grid.any():
        return float('nan'), np.array([]), np.array([]), np.array([], dtype=bool)

    box_vox = []
    counts  = []
    s = 1
    max_s = max(1, min(solid_grid.shape) // 2)
    while s <= max_s:
        if s == 1:
            n = int(solid_grid.sum())
        else:
            pad = (-np.array(solid_grid.shape)) % s
            padded = np.pad(solid_grid, [(0, p) for p in pad],
                            mode='constant', constant_values=False)
            new_shape = np.array(padded.shape) // s
            view = padded.reshape(new_shape[0], s,
                                  new_shape[1], s,
                                  new_shape[2], s)
            n = int(view.any(axis=(1, 3, 5)).sum())
        if n == 0:
            break
        box_vox.append(s)
        counts.append(n)
        s *= 2

    box_vox = np.array(box_vox, dtype=float)
    counts  = np.array(counts,  dtype=float)
    box_mm  = box_vox * pitch

    # Fit only over the linear (scaling) range: drop the largest-box endpoint
    # if there are enough scales — finite-size effects flatten the slope.
    fit_used = np.ones_like(box_mm, dtype=bool)
    if len(box_mm) >= 4:
        fit_used[-1] = False
    if fit_used.sum() >= 2:
        log_invr = np.log(1.0 / box_mm[fit_used])
        log_N    = np.log(counts[fit_used])
        slope, _ = np.polyfit(log_invr, log_N, 1)
        D_b = float(slope)
    else:
        D_b = float('nan')

    return D_b, box_mm, counts, fit_used


# ── Hidden-space / overhang angle metrics ────────────────────────────────────

def _shifted_view(arr, shift, fill=False):
    """Return out where out[v] = arr[v + shift]; out-of-bounds positions = fill.

    `shift` is a 3-tuple aligned with arr.shape (axis 0, 1, 2).
    """
    out = np.full_like(arr, fill)
    src_slices, dst_slices = [], []
    for s, dim in zip(shift, arr.shape):
        if s >= 0:
            src_slices.append(slice(s, dim))
            dst_slices.append(slice(0, dim - s))
        else:
            src_slices.append(slice(0, dim + s))
            dst_slices.append(slice(-s, dim))
    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


def _ray_escapes(solid_grid, hull_grid, direction, max_steps=None):
    """Per-voxel boolean ray-march: True iff a ray from voxel v in `direction`
    leaves the convex hull without first hitting solid.

    `direction` is an integer (axis0, axis1, axis2) step vector.  Iteratively
    consults voxel v + k*direction for k = 1, 2, … until every voxel's fate
    (escape vs. blocked) is decided.
    """
    shape = solid_grid.shape
    if max_steps is None:
        max_steps = int(np.ceil(np.linalg.norm(shape))) + 1
    ones    = np.ones(shape, dtype=bool)
    decided = np.zeros(shape, dtype=bool)
    escapes = np.zeros(shape, dtype=bool)
    for k in range(1, max_steps + 1):
        sh = tuple(int(k * c) for c in direction)
        in_bounds = _shifted_view(ones,       sh, fill=False)
        sh_solid  = _shifted_view(solid_grid, sh, fill=False)
        sh_inhull = _shifted_view(hull_grid,  sh, fill=False)
        # Escaped: ray cell is out of grid, or in grid but outside the hull.
        # Blocked: ray cell is in grid AND in hull AND solid.
        new_escaped = ((~in_bounds) | (in_bounds & ~sh_inhull)) & ~decided
        new_blocked = ( in_bounds  &  sh_inhull & sh_solid)     & ~decided
        escapes |= new_escaped
        decided |= new_escaped | new_blocked
        if decided.all():
            break
    return escapes


# Upper-hemisphere directions: third component (axis 2 = +Z = up) > 0.
_UPPER_DIRS = np.array([
    ( 0,  0,  1),
    ( 1,  0,  1), (-1,  0,  1),
    ( 0,  1,  1), ( 0, -1,  1),
    ( 1,  1,  1), ( 1, -1,  1), (-1,  1,  1), (-1, -1,  1),
], dtype=int)

# All 26 integer-step neighbour directions, used for full-sphere opening.
_ALL_26_DIRS = np.array(
    [(d0, d1, d2) for d0 in (-1, 0, 1)
                  for d1 in (-1, 0, 1)
                  for d2 in (-1, 0, 1)
                  if not (d0 == 0 and d1 == 0 and d2 == 0)],
    dtype=int,
)


def _compute_angle_metrics(solid_grid, hull_grid, void_mask, pitch,
                           log_fn=lambda x: None,
                           max_voxels=80_000_000):
    """Compute four per-voxel hidden-space / overhang angle metrics.

    Returns a dict with one sub-dict per metric, or None if the grid is too
    large for the analysis to be feasible.
    """
    from scipy import ndimage as ndi

    n = int(np.prod(solid_grid.shape))
    if n > max_voxels:
        log_fn(f"  ⚠  Grid has {n:,} voxels (> {max_voxels:,} threshold) — "
               f"skipping angle analysis to keep runtime tractable.")
        return None

    # ── Outward-normal field on the solid (gradient of smoothed occupancy) ──
    log_fn("  Computing outward-normal field…")
    solid_f  = solid_grid.astype(np.float32)
    smoothed = ndi.gaussian_filter(solid_f, sigma=1.0)
    g0 = ndi.sobel(smoothed, axis=0)
    g1 = ndi.sobel(smoothed, axis=1)
    g2 = ndi.sobel(smoothed, axis=2)
    mag = np.sqrt(g0 * g0 + g1 * g1 + g2 * g2) + 1e-12
    # Sobel of the solid-occupancy field points INTO the solid; outward = −grad.
    n0 = (-g0 / mag).astype(np.float32)
    n1 = (-g1 / mag).astype(np.float32)
    n2 = (-g2 / mag).astype(np.float32)
    del solid_f, smoothed, g0, g1, g2, mag

    struct6   = ndi.generate_binary_structure(3, 1)
    non_solid = ~solid_grid

    # === Metric 1 — surface inclination (per surface solid voxel) ═══════════
    surface_mask = solid_grid & ndi.binary_dilation(non_solid, structure=struct6)
    surf_idx     = np.argwhere(surface_mask)
    surf_n2      = n2[surface_mask]
    incl_deg     = np.degrees(np.arccos(np.clip(surf_n2, -1.0, 1.0)))
    log_fn(f"  Metric 1 (inclination): {surf_idx.shape[0]:,} surface voxels, "
           f"overhang fraction (θ > 90°) = "
           f"{100.0 * (incl_deg > 90).mean():.1f}%")

    # === Metric 2 — sky-view factor (per void voxel) ════════════════════════
    log_fn(f"  Metric 2 (sky-view): ray-marching {len(_UPPER_DIRS)} upper-"
           f"hemisphere directions…")
    sky_count = np.zeros(solid_grid.shape, dtype=np.uint8)
    for i, d in enumerate(_UPPER_DIRS, 1):
        log_fn(f"    [{i:2d}/{len(_UPPER_DIRS)}]  d = "
               f"({int(d[0]):+d},{int(d[1]):+d},{int(d[2]):+d})")
        sky_count += _ray_escapes(solid_grid, hull_grid, d).view(np.uint8)
    void_idx       = np.argwhere(void_mask)
    sky_unblocked  = sky_count[void_mask].astype(np.int16)
    svf            = sky_unblocked.astype(np.float32) / float(len(_UPPER_DIRS))

    # === Metric 3 — crevice opening solid-angle → equivalent cone half-angle ═
    log_fn(f"  Metric 3 (opening): ray-marching the remaining "
           f"{len(_ALL_26_DIRS) - len(_UPPER_DIRS)} non-upper directions…")
    upper_set    = {tuple(int(c) for c in d) for d in _UPPER_DIRS}
    other_dirs   = [d for d in _ALL_26_DIRS
                    if tuple(int(c) for c in d) not in upper_set]
    other_count  = np.zeros(solid_grid.shape, dtype=np.uint8)
    for i, d in enumerate(other_dirs, 1):
        log_fn(f"    [{i:2d}/{len(other_dirs)}]  d = "
               f"({int(d[0]):+d},{int(d[1]):+d},{int(d[2]):+d})")
        other_count += _ray_escapes(solid_grid, hull_grid, d).view(np.uint8)
    full_count        = sky_count.astype(np.int16) + other_count.astype(np.int16)
    surface_void_mask = void_mask & ndi.binary_dilation(solid_grid,
                                                        structure=struct6)
    sva_idx        = np.argwhere(surface_void_mask)
    open_unblocked = full_count[surface_void_mask].astype(np.int16)
    open_frac      = open_unblocked.astype(np.float32) / float(len(_ALL_26_DIRS))
    # Cone of solid angle Ω = 2π(1 − cos α) with Ω = open_frac · 4π
    #   ⇒  cos α = 1 − 2·open_frac,  α ∈ [0°, 180°].
    open_half_deg = np.degrees(
        np.arccos(np.clip(1.0 - 2.0 * open_frac, -1.0, 1.0))
    )

    # === Metric 4 — local dihedral angle (max pairwise angle of solid normals)
    log_fn("  Metric 4 (dihedral): scanning 3×3×3 neighbourhoods…")
    dihedral_deg = np.full(sva_idx.shape[0], np.nan, dtype=np.float32)
    n_neigh      = np.zeros(sva_idx.shape[0], dtype=np.int16)
    sZ, sY, sX   = solid_grid.shape
    for i, (z, y, x) in enumerate(sva_idx):
        z0, z1 = max(0, z - 1), min(sZ, z + 2)
        y0, y1 = max(0, y - 1), min(sY, y + 2)
        x0, x1 = max(0, x - 1), min(sX, x + 2)
        block = solid_grid[z0:z1, y0:y1, x0:x1]
        if not block.any():
            continue
        b0 = n0[z0:z1, y0:y1, x0:x1][block]
        b1 = n1[z0:z1, y0:y1, x0:x1][block]
        b2 = n2[z0:z1, y0:y1, x0:x1][block]
        m = b0.size
        n_neigh[i] = m
        if m < 2:
            continue
        norms = np.column_stack([b0, b1, b2])
        cos_pair = np.clip(norms @ norms.T, -1.0, 1.0)
        np.fill_diagonal(cos_pair, 1.0)
        dihedral_deg[i] = np.degrees(np.arccos(cos_pair.min()))
    log_fn(f"  Metric 4 (dihedral): {np.isfinite(dihedral_deg).sum():,} of "
           f"{dihedral_deg.size:,} surface-adjacent void voxels evaluated")

    return {
        'inclination': {
            'voxels':    surf_idx,
            'angle_deg': incl_deg,
            'normal':    np.column_stack([n0[surface_mask],
                                          n1[surface_mask],
                                          n2[surface_mask]]),
        },
        'sky_view': {
            'voxels':       void_idx,
            'svf':          svf,
            'n_unblocked':  sky_unblocked,
            'n_dirs':       int(len(_UPPER_DIRS)),
        },
        'opening': {
            'voxels':         sva_idx,
            'fraction':       open_frac,
            'half_angle_deg': open_half_deg,
            'n_unblocked':    open_unblocked,
            'n_dirs':         int(len(_ALL_26_DIRS)),
        },
        'dihedral': {
            'voxels':       sva_idx,
            'angle_deg':    dihedral_deg,
            'n_neighbours': n_neigh,
        },
    }


def _write_angle_csvs(angles, mesh_path, pitch, bbox_min):
    """Write one CSV per metric (every analysed voxel)."""
    paths = {}

    a = angles['inclination']
    p = mesh_path.with_name(mesh_path.stem + f"_inclination{pitch}mm.csv")
    coords = a['voxels'] * pitch + bbox_min
    rows = ["x_mm,y_mm,z_mm,inclination_deg,normal_x,normal_y,normal_z\n"]
    for c, ang, nrm in zip(coords, a['angle_deg'], a['normal']):
        rows.append(
            f"{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},{ang:.3f},"
            f"{nrm[0]:.4f},{nrm[1]:.4f},{nrm[2]:.4f}\n"
        )
    p.write_text("".join(rows))
    paths['inclination'] = p

    a = angles['sky_view']
    p = mesh_path.with_name(mesh_path.stem + f"_skyview{pitch}mm.csv")
    coords = a['voxels'] * pitch + bbox_min
    nd = a['n_dirs']
    rows = ["x_mm,y_mm,z_mm,sky_view_factor,n_unblocked,n_dirs\n"]
    for c, s, u in zip(coords, a['svf'], a['n_unblocked']):
        rows.append(f"{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},"
                    f"{s:.4f},{int(u)},{nd}\n")
    p.write_text("".join(rows))
    paths['sky_view'] = p

    a = angles['opening']
    p = mesh_path.with_name(mesh_path.stem + f"_opening{pitch}mm.csv")
    coords = a['voxels'] * pitch + bbox_min
    nd = a['n_dirs']
    rows = ["x_mm,y_mm,z_mm,opening_fraction,opening_half_angle_deg,"
            "n_unblocked,n_dirs\n"]
    for c, f, h, u in zip(coords, a['fraction'], a['half_angle_deg'],
                          a['n_unblocked']):
        rows.append(f"{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},"
                    f"{f:.4f},{h:.3f},{int(u)},{nd}\n")
    p.write_text("".join(rows))
    paths['opening'] = p

    a = angles['dihedral']
    p = mesh_path.with_name(mesh_path.stem + f"_dihedral{pitch}mm.csv")
    coords = a['voxels'] * pitch + bbox_min
    rows = ["x_mm,y_mm,z_mm,max_dihedral_deg,n_solid_neighbours\n"]
    for c, ang, n in zip(coords, a['angle_deg'], a['n_neighbours']):
        ang_str = "" if not np.isfinite(ang) else f"{ang:.3f}"
        rows.append(f"{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},"
                    f"{ang_str},{int(n)}\n")
    p.write_text("".join(rows))
    paths['dihedral'] = p

    return paths


def _plot_angle_distributions(angles, mesh_path, pitch):
    """4-panel histogram of all hidden-space / overhang angle metrics."""
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"{mesh_path.name}  —  Hidden-space / overhang angle metrics  "
        f"(voxel {pitch} mm)",
        fontsize=11,
    )

    # 1. Surface inclination
    ax = axes[0, 0]
    a = angles['inclination']['angle_deg']
    ax.hist(a, bins=60, color="#4c72b0", alpha=0.8,
            edgecolor="white", linewidth=0.3)
    ax.axvline(90, color="#c44e52", ls="--", lw=1, label="90° (vertical)")
    ax.set_xlabel("Surface inclination θ_n vs +Z (deg)")
    ax.set_ylabel("Count")
    ax.set_title(f"1. Surface inclination  ({a.size:,} surface voxels)")
    ax.legend(fontsize=9, loc="upper right")
    ax.text(0.02, 0.97,
            f"overhang fraction: {100.0 * (a > 90).mean():.1f}%\n"
            f"median θ: {np.median(a):.1f}°",
            transform=ax.transAxes, va="top", fontsize=9)

    # 2. Sky-view factor
    ax = axes[0, 1]
    s = angles['sky_view']['svf']
    ax.hist(s, bins=50, color="#55a868", alpha=0.8,
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Sky-view factor (fraction of upper-hemisphere rays unblocked)")
    ax.set_ylabel("Count")
    ax.set_title(f"2. Sky-view factor  ({s.size:,} void voxels, "
                 f"{angles['sky_view']['n_dirs']} dirs)")
    ax.text(0.02, 0.97,
            f"hidden (SVF = 0): {100.0 * (s == 0).mean():.1f}%\n"
            f"median SVF: {np.median(s):.2f}",
            transform=ax.transAxes, va="top", fontsize=9)

    # 3. Crevice opening half-angle
    ax = axes[1, 0]
    h = angles['opening']['half_angle_deg']
    ax.hist(h, bins=60, color="#8172b2", alpha=0.8,
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Equivalent crevice opening half-angle α (deg)")
    ax.set_ylabel("Count")
    ax.set_title(f"3. Crevice opening half-angle  "
                 f"({h.size:,} surface-adj. void voxels)")
    ax.text(0.02, 0.97, f"median α: {np.median(h):.1f}°",
            transform=ax.transAxes, va="top", fontsize=9)

    # 4. Local dihedral angle
    ax = axes[1, 1]
    d = angles['dihedral']['angle_deg']
    d_v = d[np.isfinite(d)]
    ax.hist(d_v, bins=60, color="#ccb974", alpha=0.8,
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Max pairwise angle between local solid normals (deg)")
    ax.set_ylabel("Count")
    ax.set_title(f"4. Local dihedral angle  "
                 f"({d_v.size:,} of {d.size:,} valid)")
    if d_v.size:
        ax.text(0.02, 0.97, f"median: {np.median(d_v):.1f}°",
                transform=ax.transAxes, va="top", fontsize=9)

    plt.tight_layout()
    out = mesh_path.with_name(mesh_path.stem + f"_angles{pitch}mm.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Output helpers ───────────────────────────────────────────────────────────

def _write_spheres_csv(pocket_stats, mesh_path, pitch):
    """Write every packed sphere from every pocket to a CSV file.

    One row per sphere; spheres are listed in placement order (largest first
    within each pocket), pockets in descending volume order.
    """
    csv_path = mesh_path.with_name(
        mesh_path.stem + f"_spheres{pitch}mm.csv"
    )
    header = (
        "pocket_rank,pocket_label,pocket_volume_mm3,"
        "sphere_idx,sphere_diameter_mm,sphere_radius_mm,"
        "center_x_mm,center_y_mm,center_z_mm\n"
    )
    rows = []
    for rank, p in enumerate(pocket_stats, start=1):
        for s_idx, s in enumerate(p.get('spheres', []), start=1):
            cx, cy, cz = s['center_mm']
            rows.append(
                f"{rank},{p['label']},{p['volume']:.6f},"
                f"{s_idx},{s['diameter_mm']:.6f},{s['radius_mm']:.6f},"
                f"{cx:.4f},{cy:.4f},{cz:.4f}\n"
            )
    csv_path.write_text(header + "".join(rows))
    return csv_path


def _plot_sphere_distribution(all_diameters, mesh_path, pitch,
                              D_sphere, sphere_fit):
    """Histogram of all packed-sphere diameters + log-log cumulative N(>=d)
    with the Mandelbrot number-size power-law fit overlaid.
    """
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    diameters = np.asarray(all_diameters, dtype=float)
    if diameters.size == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{mesh_path.name}  —  Inscribed-sphere size distribution"
        f"  (voxel {pitch} mm,  n = {diameters.size:,} spheres)",
        fontsize=11,
    )

    # Panel 1 — log-binned histogram of diameters
    ax = axes[0]
    if diameters.min() > 0 and diameters.max() / diameters.min() > 4:
        bins = np.logspace(np.log10(diameters.min()),
                           np.log10(diameters.max()), 40)
        ax.set_xscale("log")
    else:
        bins = 40
    ax.hist(diameters, bins=bins, color="#4c72b0", alpha=0.75,
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Sphere diameter d (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Diameter histogram")

    # Panel 2 — cumulative N(>= d) with Mandelbrot power-law fit
    ax = axes[1]
    sorted_d = np.sort(diameters)
    N_ge     = np.arange(diameters.size, 0, -1)
    ax.loglog(sorted_d, N_ge, "o", ms=3, color="#4c72b0",
              alpha=0.6, label="N(≥ d)")
    if sphere_fit is not None:
        d_fit, N_fit, slope, intercept = sphere_fit
        d_line = np.array([d_fit.min(), d_fit.max()])
        N_line = np.exp(intercept) * d_line ** slope
        ax.loglog(d_line, N_line, "-", color="#c44e52", lw=2,
                  label=f"slope = {slope:.3f}\nD_s = {-slope:.3f}")
    ax.set_xlabel("Sphere diameter d (mm)")
    ax.set_ylabel("N(≥ d)")
    ax.set_title("Cumulative size distribution  N(≥d) ∝ d^(−D_s)  (Mandelbrot 1982)")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = mesh_path.with_name(mesh_path.stem + f"_spheres{pitch}mm.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_box_counting(box_mm, counts, fit_used, D_b, mesh_path, pitch):
    """Log-log plot of N(r) vs 1/r with the box-counting fit (Orland 2016)."""
    if box_mm.size == 0:
        return None
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(1.0 / box_mm, counts, "o", ms=6, color="#4c72b0",
              label="all scales")
    if fit_used.any():
        ax.loglog(1.0 / box_mm[fit_used], counts[fit_used], "o",
                  ms=8, mfc="none", mec="#c44e52", mew=1.5,
                  label="used in fit")
        log_invr = np.log(1.0 / box_mm[fit_used])
        log_N    = np.log(counts[fit_used])
        slope, intercept = np.polyfit(log_invr, log_N, 1)
        xs = np.array([1.0 / box_mm.max(), 1.0 / box_mm.min()])
        ys = np.exp(intercept) * xs ** slope
        ax.loglog(xs, ys, "-", color="#c44e52", lw=2,
                  label=f"D_b = {D_b:.3f}")
    ax.set_xlabel("1 / r  (mm⁻¹)")
    ax.set_ylabel("N(r)")
    ax.set_title(f"{mesh_path.name}  —  Box-counting fractal dimension"
                 f"  (Orland et al. 2016)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = mesh_path.with_name(mesh_path.stem + f"_boxcount{pitch}mm.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _write_csv(pocket_stats, mesh_path, pitch):
    """Write all pockets to a CSV file sorted by volume (largest first)."""
    csv_path = mesh_path.with_name(mesh_path.stem + f"_pockets{pitch}mm.csv")
    header = (
        "rank,label,voxels,volume_mm3,"
        "max_inscribed_r_mm,mean_inscribed_r_mm,"
        "n_packed_spheres,packed_sphere_total_volume_mm3,"
        "centroid_x_mm,centroid_y_mm,centroid_z_mm\n"
    )
    rows = []
    for rank, p in enumerate(pocket_stats, start=1):
        cx, cy, cz = p["centroid_mm"]
        spheres = p.get("spheres", [])
        n_sph   = len(spheres)
        sph_vol = sum((4.0 / 3.0) * np.pi * s["radius_mm"] ** 3 for s in spheres)
        rows.append(
            f"{rank},{p['label']},{p['voxels']},"
            f"{p['volume']:.6f},{p['max_inscribed_r']:.6f},{p['mean_inscribed_r']:.6f},"
            f"{n_sph},{sph_vol:.6f},"
            f"{cx:.4f},{cy:.4f},{cz:.4f}\n"
        )
    csv_path.write_text(header + "".join(rows))
    return csv_path


def _plot_density(pocket_stats, mesh_path, pitch):
    """Save a 2-panel histogram + KDE density plot of pocket metrics."""
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    volumes = np.array([p["volume"]          for p in pocket_stats])
    radii   = np.array([p["max_inscribed_r"] for p in pocket_stats])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{mesh_path.name}  —  Pocket size distributions"
        f"  (voxel {pitch} mm,  n = {len(pocket_stats):,} pockets)",
        fontsize=11,
    )

    def _panel(ax, data, xlabel, use_log):
        if use_log and data.min() > 0:
            bins = np.logspace(np.log10(max(data.min(), 1e-12)),
                               np.log10(data.max()), 50)
            ax.hist(data, bins=bins, color="#4c72b0", alpha=0.65,
                    edgecolor="white", linewidth=0.3)
            ax.set_xscale("log")
            if len(data) >= 4:
                log_d = np.log10(np.clip(data, 1e-12, None))
                kde   = gaussian_kde(log_d, bw_method="scott")
                xs    = np.linspace(log_d.min(), log_d.max(), 400)
                ax2   = ax.twinx()
                ax2.plot(10 ** xs, kde(xs), color="#c44e52", lw=1.8)
                ax2.set_ylabel("KDE density", color="#c44e52", fontsize=9)
                ax2.tick_params(axis="y", labelcolor="#c44e52", labelsize=8)
                ax2.set_ylim(bottom=0)
        else:
            ax.hist(data, bins=50, color="#55a868", alpha=0.65,
                    edgecolor="white", linewidth=0.3)
            if len(data) >= 4:
                kde = gaussian_kde(data, bw_method="scott")
                xs  = np.linspace(data.min(), data.max(), 400)
                ax2 = ax.twinx()
                ax2.plot(xs, kde(xs), color="#c44e52", lw=1.8)
                ax2.set_ylabel("KDE density", color="#c44e52", fontsize=9)
                ax2.tick_params(axis="y", labelcolor="#c44e52", labelsize=8)
                ax2.set_ylim(bottom=0)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Count",  fontsize=10)
        ax.tick_params(labelsize=9)

    vol_log = (volumes.max() / max(float(volumes.min()), 1e-30)) > 100
    _panel(axes[0], volumes, "Pocket volume (mm³)",          vol_log)
    _panel(axes[1], radii,   "Max inscribed radius (mm)",    False)
    axes[0].set_title("Volume distribution",               fontsize=10)
    axes[1].set_title("Max inscribed radius distribution", fontsize=10)

    plt.tight_layout()
    out = mesh_path.with_name(mesh_path.stem + f"_density{pitch}mm.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_3d_pockets(pocket_stats, hull, mesh_path, pitch):
    """Save a 3-D scatter of pocket voxel clouds, each pocket a distinct colour."""
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    n = len(pocket_stats)
    if n == 0:
        return None

    TOP_CLOUD = 50

    fig = plt.figure(figsize=(11, 8))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    # Convex hull as faint context shell
    hull_tris = hull.vertices[hull.faces]
    ax.add_collection3d(
        Poly3DCollection(hull_tris, alpha=0.04, facecolor="steelblue",
                         edgecolor="steelblue", linewidth=0.1)
    )

    cmap = plt.cm.get_cmap("tab20", min(n, 20))
    legend_handles = []

    for rank, p in enumerate(pocket_stats):
        color = cmap(rank % 20)

        if rank < TOP_CLOUD and "sample_world" in p:
            pts = p["sample_world"]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=[color], s=5, alpha=0.55, depthshade=True)
        else:
            c = p["centroid_mm"]
            ax.scatter([c[0]], [c[1]], [c[2]],
                       c=[color], s=18, alpha=0.7, marker="o")

        if rank < 10:
            label = (f"#{rank+1}  {p['volume']:.1f} mm³  "
                     f"r={p['max_inscribed_r']:.2f} mm")
            legend_handles.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=legend_handles, fontsize=7, loc="upper left",
              title="Top 10 pockets", title_fontsize=8, framealpha=0.7)

    b = hull.bounds
    ax.set_xlim(b[0, 0], b[1, 0])
    ax.set_ylim(b[0, 1], b[1, 1])
    ax.set_zlim(b[0, 2], b[1, 2])
    ax.set_xlabel("X (mm)", labelpad=5)
    ax.set_ylabel("Y (mm)", labelpad=5)
    ax.set_zlabel("Z (mm)", labelpad=5)
    ax.set_title(
        f"{mesh_path.name}  —  Pocket 3-D map\n"
        f"{n} pockets · voxel {pitch} mm  "
        f"(top {min(TOP_CLOUD, n)} as voxel clouds, remainder as centroids)",
        fontsize=9,
    )

    plt.tight_layout()
    out = mesh_path.with_name(mesh_path.stem + f"_3d{pitch}mm.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Core analysis (runs in a background thread) ───────────────────────────────

def run_analysis(mesh_path_str, pitch, top_n, debug, log_fn, progress_fn, status_fn):
    """Run the full pocket analysis with progress and logging callbacks.

    Parameters
    ----------
    mesh_path_str : str
    pitch         : float  — voxel size in mm
    top_n         : int    — number of top pockets in report
    debug         : bool   — include per-stage timing
    log_fn        : callable(str)  — append a line to the console
    progress_fn   : callable(int)  — update progress bar 0-100
    status_fn     : callable(str)  — update status label text
    """
    from scipy import ndimage as ndi

    mesh_path = Path(mesh_path_str)
    voxel_vol = pitch ** 3

    def dlog(msg):
        if debug:
            log_fn(f"    [timing] {msg}")

    def sep():
        log_fn("─" * 62)

    t_total = time.perf_counter()
    progress_fn(0)

    # ══════════════════════════════════════════════════════════════
    # Stage 1 — Load mesh
    # ══════════════════════════════════════════════════════════════
    sep()
    log_fn(f"Stage 1 / 9  —  Loading mesh")
    log_fn(f"  File: {mesh_path.name}")
    status_fn("Loading mesh…")
    t0 = time.perf_counter()

    mesh = trimesh.load(str(mesh_path), force="mesh")

    elapsed = time.perf_counter() - t0
    log_fn(f"  Vertices : {len(mesh.vertices):,}")
    log_fn(f"  Faces    : {len(mesh.faces):,}")
    log_fn(f"  Watertight: {mesh.is_watertight}")
    if not mesh.is_watertight:
        log_fn("  ⚠  Mesh is NOT watertight — voxelisation may be inaccurate")
    dlog(f"load  {elapsed:.2f}s")
    log_fn(f"  ✓  Done  ({elapsed:.2f}s)")
    progress_fn(10)

    # ══════════════════════════════════════════════════════════════
    # Stage 2 — Mesh & hull measurements
    # ══════════════════════════════════════════════════════════════
    sep()
    log_fn("Stage 2 / 9  —  Mesh & convex-hull measurements")
    status_fn("Computing measurements…")
    t0 = time.perf_counter()

    mesh_volume   = mesh.volume
    mesh_area     = mesh.area
    base_mask     = mesh.face_normals[:, 2] < -0.999
    base_cap_area = (
        float(trimesh.triangles.area(mesh.triangles[base_mask]).sum())
        if base_mask.any() else 0.0
    )
    wetted_area = mesh_area - base_cap_area

    dlog(f"mesh measurements  {time.perf_counter()-t0:.2f}s")
    log_fn(f"  Mesh volume      : {mesh_volume:.2f} mm³")
    log_fn(f"  Wetted area      : {wetted_area:.2f} mm²")

    hull           = mesh.convex_hull
    hull_volume    = hull.volume
    hull_area      = hull.area
    hull_base_mask = hull.face_normals[:, 2] < -0.999
    hull_base_area = (
        float(trimesh.triangles.area(hull.triangles[hull_base_mask]).sum())
        if hull_base_mask.any() else 0.0
    )
    interstitial_volume = hull_volume - mesh_volume
    sa_to_vol           = wetted_area / interstitial_volume if interstitial_volume > 0 else 0.0
    total_bounding_area = (hull_area - hull_base_area) + (mesh_area - base_cap_area)

    elapsed = time.perf_counter() - t0
    dlog(f"hull  {elapsed:.2f}s")
    log_fn(f"  Hull volume      : {hull_volume:.2f} mm³")
    log_fn(f"  Interstitial vol : {interstitial_volume:.2f} mm³  "
           f"({100*interstitial_volume/hull_volume:.1f}% of hull)")
    log_fn(f"  ✓  Done  ({elapsed:.2f}s)")
    progress_fn(20)

    # ══════════════════════════════════════════════════════════════
    # Stage 3 — Voxelisation
    # ══════════════════════════════════════════════════════════════
    sep()
    log_fn("Stage 3 / 9  —  Voxelisation")
    status_fn("Voxelising…")

    bbox_min = hull.bounds[0] - pitch * 2.0
    bbox_max = hull.bounds[1] + pitch * 2.0
    dims     = np.ceil((bbox_max - bbox_min) / pitch).astype(int) + 1
    total_vox = int(dims.prod())

    log_fn(f"  Voxel size  : {pitch} mm  →  {voxel_vol:.4f} mm³/voxel")
    log_fn(f"  Grid size   : {dims[0]} × {dims[1]} × {dims[2]}  =  {total_vox:,} voxels")
    log_fn(f"  Voxelising solid mesh…")
    t0 = time.perf_counter()

    solid_grid = _voxelize_to_grid(mesh, bbox_min, dims, pitch)
    dlog(f"solid_grid  {time.perf_counter()-t0:.2f}s  "
         f"({int(solid_grid.sum()):,} filled voxels)")
    log_fn(f"    Solid voxels   : {int(solid_grid.sum()):,}  ({time.perf_counter()-t0:.2f}s)")

    log_fn(f"  Voxelising convex hull…")
    t1 = time.perf_counter()
    hull_grid  = _voxelize_to_grid(hull, bbox_min, dims, pitch)
    dlog(f"hull_grid  {time.perf_counter()-t1:.2f}s  "
         f"({int(hull_grid.sum()):,} filled voxels)")
    log_fn(f"    Hull voxels    : {int(hull_grid.sum()):,}  ({time.perf_counter()-t1:.2f}s)")

    void_mask  = hull_grid & ~solid_grid
    elapsed    = time.perf_counter() - t0
    log_fn(f"    Void voxels    : {int(void_mask.sum()):,}")
    log_fn(f"  ✓  Done  ({elapsed:.2f}s)")
    progress_fn(45)

    # ══════════════════════════════════════════════════════════════
    # Stage 4 — Connected-component labelling (CCL)
    # ══════════════════════════════════════════════════════════════
    sep()
    log_fn("Stage 4 / 9  —  Connected-component labelling (CCL, 26-connectivity)")
    status_fn("Labelling pockets…")
    t0 = time.perf_counter()

    struct26          = ndi.generate_binary_structure(3, 3)
    labels, n_pockets = ndi.label(void_mask, structure=struct26)

    elapsed = time.perf_counter() - t0
    log_fn(f"  Isolated pockets found : {n_pockets}")
    log_fn(f"  ✓  Done  ({elapsed:.2f}s)")
    progress_fn(62)

    # ══════════════════════════════════════════════════════════════
    # Stage 5 — Euclidean distance transform (EDT)
    # ══════════════════════════════════════════════════════════════
    sep()
    log_fn("Stage 5 / 9  —  Euclidean distance transform (EDT)")
    log_fn("  (This is often the slowest stage for fine voxel sizes)")
    status_fn("Computing EDT…")
    t0 = time.perf_counter()

    edt = ndi.distance_transform_edt(void_mask, sampling=pitch)

    elapsed = time.perf_counter() - t0
    log_fn(f"  Max inscribed radius : {edt.max():.3f} mm")
    log_fn(f"  ✓  Done  ({elapsed:.2f}s)")
    progress_fn(75)

    # ══════════════════════════════════════════════════════════════
    # Stage 6 — Per-pocket statistics + greedy sphere packing
    # ══════════════════════════════════════════════════════════════
    sep()
    log_fn("Stage 6 / 9  —  Per-pocket statistics + inscribed-sphere packing")
    log_fn("  (refuge size-distribution; reviewed by Tokeshi & Arakaki 2012)")
    status_fn("Computing statistics…")
    t0 = time.perf_counter()

    # Minimum sphere diameter: 1 voxel edge.  Tighter cut-offs would just sample
    # the rasterisation noise floor.
    min_diameter_mm = pitch

    pocket_stats   = []
    slices         = ndi.find_objects(labels)
    skipped_small  = 0
    for label_idx, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        region_mask  = labels[sl] == label_idx
        region_edt   = edt[sl][region_mask]
        n_vox        = int(region_mask.sum())
        sl_start     = np.array([s.start for s in sl])
        centroid_vox = (np.argwhere(region_mask) + sl_start).mean(axis=0)

        # Pack non-overlapping spheres within this pocket (largest first).
        # Pad by 1 voxel so the EDT sees the pocket boundary on every side.
        padded = np.pad(region_mask, 1, mode='constant', constant_values=False)
        spheres_local = _pack_spheres_in_pocket(
            padded, pitch, min_diameter_mm=min_diameter_mm,
        )
        # Convert local (padded) voxel coords to world-space mm.
        # local_vox = global_vox - sl_start + 1   →   global_vox = local_vox + sl_start - 1
        spheres = []
        for s in spheres_local:
            global_vox = s['center_local_vox'] + sl_start - 1
            spheres.append({
                'diameter_mm': s['diameter_mm'],
                'radius_mm':   s['radius_mm'],
                'center_mm':   global_vox * pitch + bbox_min,
            })

        pocket_stats.append({
            'label':            label_idx,
            'voxels':           n_vox,
            'volume':           n_vox * voxel_vol,
            'max_inscribed_r':  float(region_edt.max()),
            'mean_inscribed_r': float(region_edt.mean()),
            'centroid_mm':      centroid_vox * pitch + bbox_min,
            'spheres':          spheres,
        })

    pocket_stats.sort(key=lambda p: p['volume'], reverse=True)

    # Aggregate sphere statistics across all pockets.
    all_diameters = np.array(
        [s['diameter_mm'] for p in pocket_stats for s in p['spheres']],
        dtype=float,
    )

    # Mandelbrot number-size power-law fit: log N(>=d) = -D_s * log d + c.
    # Restrict the fit to [2*pitch, 0.5*max_d] to avoid voxel-scale noise on
    # the low end and finite-sample sparseness on the high end.
    D_sphere   = float('nan')
    sphere_fit = None
    if all_diameters.size >= 8:
        sorted_d = np.sort(all_diameters)
        N_ge     = np.arange(sorted_d.size, 0, -1, dtype=float)
        d_lo     = max(2.0 * pitch, sorted_d.min())
        d_hi     = 0.5 * sorted_d.max()
        sel      = (sorted_d >= d_lo) & (sorted_d <= d_hi)
        if sel.sum() >= 4:
            log_d   = np.log(sorted_d[sel])
            log_N   = np.log(N_ge[sel])
            slope, intercept = np.polyfit(log_d, log_N, 1)
            D_sphere   = -float(slope)
            sphere_fit = (sorted_d[sel], N_ge[sel], slope, intercept)

    elapsed = time.perf_counter() - t0
    log_fn(f"  Processed {len(pocket_stats)} pockets, "
           f"packed {all_diameters.size:,} spheres "
           f"(min Ø {min_diameter_mm:.3f} mm)")
    if not np.isnan(D_sphere):
        log_fn(f"  Sphere number-size fractal index  D_s = {D_sphere:.3f}")
    log_fn(f"  ✓  Done  ({elapsed:.2f}s)")

    # Collect voxel-cloud samples for the top 50 pockets (used by 3-D plot)
    _TOP_3D  = 50
    _MAX_PTS = 400
    _rng     = np.random.default_rng(42)
    for p in pocket_stats[:_TOP_3D]:
        sl = slices[p['label'] - 1]
        if sl is None:
            p['sample_world'] = p['centroid_mm'].reshape(1, 3)
            continue
        region_mask   = (labels[sl] == p['label'])
        sl_start      = np.array([s.start for s in sl])
        global_coords = np.argwhere(region_mask) + sl_start
        if len(global_coords) > _MAX_PTS:
            idx = _rng.choice(len(global_coords), _MAX_PTS, replace=False)
            global_coords = global_coords[idx]
        p['sample_world'] = global_coords * pitch + bbox_min

    progress_fn(82)

    # ══════════════════════════════════════════════════════════════
    # Stage 7 — Box-counting fractal dimension (Orland et al. 2016)
    # ══════════════════════════════════════════════════════════════
    sep()
    log_fn("Stage 7 / 9  —  Box-counting fractal dimension on solid")
    log_fn("  (Orland et al. 2016 — kelp-holdfast habitat complexity)")
    status_fn("Box counting…")
    t0 = time.perf_counter()
    D_box, box_mm, box_counts, box_fit_used = _box_counting_dimension(
        solid_grid, pitch
    )
    elapsed = time.perf_counter() - t0
    log_fn(f"  Scales evaluated  : {len(box_mm)}  "
           f"(box edge {box_mm.min() if box_mm.size else 0:.3f} – "
           f"{box_mm.max() if box_mm.size else 0:.3f} mm)")
    if not np.isnan(D_box):
        log_fn(f"  Box-counting fractal dimension  D_b = {D_box:.3f}")
    log_fn(f"  ✓  Done  ({elapsed:.2f}s)")
    progress_fn(84)

    # ══════════════════════════════════════════════════════════════
    # Stage 8 — Hidden-space / overhang angle metrics
    # ══════════════════════════════════════════════════════════════
    sep()
    log_fn("Stage 8 / 9  —  Hidden-space / overhang angle metrics")
    log_fn("  (four per-voxel descriptors — 'minimum angle' framework,")
    log_fn("   reviewed by Tokeshi & Arakaki 2012)")
    status_fn("Angle metrics…")
    t0 = time.perf_counter()
    try:
        angles = _compute_angle_metrics(
            solid_grid, hull_grid, void_mask, pitch, log_fn=log_fn,
        )
    except Exception as exc:
        log_fn(f"  ⚠  Angle metrics failed: {exc}")
        log_fn(traceback.format_exc())
        angles = None
    elapsed = time.perf_counter() - t0
    log_fn(f"  ✓  Done  ({elapsed:.2f}s)")
    progress_fn(88)

    # ══════════════════════════════════════════════════════════════
    # Stage 9 — Outputs
    # ══════════════════════════════════════════════════════════════
    sep()
    log_fn("Stage 9 / 9  —  Writing outputs")

    # ── Per-pocket CSV ────────────────────────────────────────────────────────
    log_fn("Exporting per-pocket CSV…")
    status_fn("Writing CSV…")
    csv_path = _write_csv(pocket_stats, mesh_path, pitch)
    log_fn(f"  ✓  Saved: {csv_path}")

    # ── Per-sphere CSV (every packed sphere in every pocket) ─────────────────
    log_fn("Exporting per-sphere CSV…")
    status_fn("Writing sphere CSV…")
    spheres_csv = _write_spheres_csv(pocket_stats, mesh_path, pitch)
    log_fn(f"  ✓  Saved: {spheres_csv}")
    progress_fn(89)

    # ── Pocket-volume / max-r density plot ────────────────────────────────────
    log_fn("Generating density plot…")
    status_fn("Density plot…")
    try:
        density_path = _plot_density(pocket_stats, mesh_path, pitch)
        log_fn(f"  ✓  Saved: {density_path}")
    except Exception as exc:
        log_fn(f"  ⚠  Density plot failed: {exc}")
    progress_fn(91)

    # ── Sphere distribution plot (Mandelbrot number-size power-law) ─────────
    log_fn("Generating sphere-distribution plot…")
    status_fn("Sphere distribution plot…")
    try:
        spheres_plot = _plot_sphere_distribution(
            all_diameters, mesh_path, pitch, D_sphere, sphere_fit
        )
        if spheres_plot:
            log_fn(f"  ✓  Saved: {spheres_plot}")
    except Exception as exc:
        log_fn(f"  ⚠  Sphere-distribution plot failed: {exc}")
    progress_fn(93)

    # ── Box-counting plot (Orland) ────────────────────────────────────────────
    log_fn("Generating box-counting plot…")
    status_fn("Box-counting plot…")
    try:
        boxcount_plot = _plot_box_counting(
            box_mm, box_counts, box_fit_used, D_box, mesh_path, pitch
        )
        if boxcount_plot:
            log_fn(f"  ✓  Saved: {boxcount_plot}")
    except Exception as exc:
        log_fn(f"  ⚠  Box-counting plot failed: {exc}")
    progress_fn(93)

    # ── Angle-metric CSVs + histogram plot ────────────────────────────────────
    angle_csv_paths = {}
    angles_plot     = None
    if angles is not None:
        log_fn("Exporting angle-metric CSVs (one per metric)…")
        status_fn("Writing angle CSVs…")
        try:
            angle_csv_paths = _write_angle_csvs(
                angles, mesh_path, pitch, bbox_min
            )
            for k, v in angle_csv_paths.items():
                log_fn(f"  ✓  Saved: {v}")
        except Exception as exc:
            log_fn(f"  ⚠  Angle-CSV export failed: {exc}")
        progress_fn(94)

        log_fn("Generating angle-distribution plot…")
        status_fn("Angle plot…")
        try:
            angles_plot = _plot_angle_distributions(angles, mesh_path, pitch)
            if angles_plot:
                log_fn(f"  ✓  Saved: {angles_plot}")
        except Exception as exc:
            log_fn(f"  ⚠  Angle plot failed: {exc}")
    progress_fn(95)

    # ── 3-D pocket map ────────────────────────────────────────────────────────
    log_fn("Generating 3-D pocket map…")
    status_fn("3D plot…")
    try:
        plot3d_path = _plot_3d_pockets(pocket_stats, hull, mesh_path, pitch)
        if plot3d_path:
            log_fn(f"  ✓  Saved: {plot3d_path}")
    except Exception as exc:
        log_fn(f"  ⚠  3-D plot failed: {exc}")
    progress_fn(96)

    # ── Size bins ──────────────────────────────────────────────────────────────
    size_bin_edges  = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, float('inf')]
    size_bin_labels = ['< 0.5', '0.5 – 1', '1 – 2', '2 – 5', '5 – 10', '10 – 20', '> 20']
    bin_counts  = [0]   * len(size_bin_labels)
    bin_volumes = [0.0] * len(size_bin_labels)
    for p in pocket_stats:
        for i, edge in enumerate(size_bin_edges):
            if p['max_inscribed_r'] < edge:
                bin_counts[i]  += 1
                bin_volumes[i] += p['volume']
                break

    # ── Build report text ──────────────────────────────────────────────────────
    lines = []
    lines.append(f"Mesh     : {mesh_path}")
    lines.append("")
    lines.append("Mesh")
    lines.append(f"  vertices               : {len(mesh.vertices)}")
    lines.append(f"  faces                  : {len(mesh.faces)}")
    lines.append(f"  watertight             : {mesh.is_watertight}")
    lines.append("")
    lines.append("Solid")
    lines.append(f"  volume                 : {mesh_volume:.4f} mm³")
    lines.append(f"  surface area (w/ cap)  : {mesh_area:.4f} mm²")
    lines.append(f"  wetted surface area    : {wetted_area:.4f} mm²  (base cap excluded)")
    lines.append(f"  base cap area          : {base_cap_area:.4f} mm²")
    lines.append("")
    lines.append("Convex hull (bounding envelope)")
    lines.append(f"  hull volume            : {hull_volume:.4f} mm³")
    lines.append(f"  hull surface area      : {hull_area:.4f} mm²")
    lines.append("")
    lines.append("Interstitial space")
    lines.append(f"  volume                 : {interstitial_volume:.4f} mm³  (hull - solid)")
    lines.append(f"  fraction of hull       : {interstitial_volume / hull_volume:.6f}")
    lines.append(f"  external surface area  : {hull_area:.4f} mm²  (hull envelope)")
    lines.append(f"  internal surface area  : {mesh_area:.4f} mm²  (full mesh)")
    lines.append(f"  total bounding area    : {total_bounding_area:.4f} mm²")
    lines.append(f"  SA / volume ratio      : {sa_to_vol:.4f} mm⁻¹  (wetted SA)")
    lines.append("")
    lines.append("Interstitial pocket analysis")
    lines.append("  method")
    lines.append("    voxelisation         : trimesh.voxelized (scanline ray-cast; Kaufman 1987)")
    lines.append("    pocket detection     : scipy.ndimage.label, 26-connectivity")
    lines.append("    organism size        : scipy.ndimage.distance_transform_edt, max inscribed sphere radius")
    lines.append(f"  voxel size             : {pitch} mm  ->  {voxel_vol:.4f} mm³/voxel")
    lines.append(f"  grid dimensions        : {dims[0]} x {dims[1]} x {dims[2]} voxels")
    lines.append(f"  void voxels            : {int(void_mask.sum()):,}")
    lines.append(f"  void volume (voxelised): {int(void_mask.sum()) * voxel_vol:.2f} mm³"
                 f"  (cf. hull - solid = {interstitial_volume:.2f} mm³)")
    lines.append(f"  isolated pockets       : {n_pockets}")
    lines.append("")
    lines.append("  Organism size distribution  (max inscribed sphere radius in pocket, mm)")
    lines.append(f"  {'Radius bin (mm)':<16}  {'Pockets':>8}  {'Void vol (mm³)':>16}  {'% void vol':>11}")
    tvv = sum(p['volume'] for p in pocket_stats) or 1.0
    for lbl, cnt, bvol in zip(size_bin_labels, bin_counts, bin_volumes):
        pct = 100.0 * bvol / tvv
        lines.append(f"  {lbl:<16}  {cnt:>8}  {bvol:>16.2f}  {pct:>10.1f}%")
    lines.append("")
    top_n_actual = min(top_n, len(pocket_stats))
    lines.append(f"  Largest {top_n_actual} pockets by volume")
    lines.append(
        f"  {'Rank':<5}  {'Volume (mm³)':>14}  {'Max r (mm)':>11}  "
        f"{'Mean r (mm)':>12}  {'Voxels':>8}  {'#Spheres':>9}"
    )
    for ri, p in enumerate(pocket_stats[:top_n_actual], 1):
        lines.append(
            f"  {ri:<5}  {p['volume']:>14.2f}  {p['max_inscribed_r']:>11.3f}"
            f"  {p['mean_inscribed_r']:>12.3f}  {p['voxels']:>8}"
            f"  {len(p['spheres']):>9}"
        )

    # ── Sphere packing & fractal indices ──────────────────────────────────────
    lines.append("")
    lines.append("Greedy non-overlapping inscribed-sphere packing")
    lines.append("  method")
    lines.append("    For each pocket, the largest inscribed sphere is identified from")
    lines.append("    the EDT, carved out of the available void, and the EDT recomputed;")
    lines.append("    smaller spheres then fill the residual interstices, building up a")
    lines.append("    refuge size-distribution that connects the larger spheres together.")
    lines.append("    Number-size power law: N(>=d) = C * d^(-D_s)")
    lines.append("      primary: Mandelbrot 1982, The Fractal Geometry of Nature, W.H. Freeman.")
    lines.append("      see also: Frontier 1987, Applications of fractal theory to ecology,")
    lines.append("                in Legendre & Legendre (eds.), Developments in Numerical")
    lines.append("                Ecology, Springer.")
    lines.append("      reviewed: Tokeshi & Arakaki 2012, Hydrobiologia 685: 27–47.")
    lines.append(f"  minimum sphere diameter : {min_diameter_mm:.4f} mm  (= 1 voxel edge)")
    lines.append(f"  total spheres packed    : {all_diameters.size:,}")
    if all_diameters.size:
        lines.append(f"  diameter range          : "
                     f"{all_diameters.min():.3f} – {all_diameters.max():.3f} mm")
        lines.append(f"  mean / median diameter  : "
                     f"{all_diameters.mean():.3f} / {np.median(all_diameters):.3f} mm")
    if not np.isnan(D_sphere):
        d_lo_used = sphere_fit[0].min(); d_hi_used = sphere_fit[0].max()
        lines.append(f"  Number-size fractal index   D_s  =  {D_sphere:.4f}")
        lines.append(f"    fit range             : "
                     f"{d_lo_used:.3f} – {d_hi_used:.3f} mm  "
                     f"({sphere_fit[0].size:,} points)")
    else:
        lines.append("  Number-size fractal index   D_s  =  n/a "
                     "(too few spheres in scaling range)")
    lines.append("")
    lines.append("Box-counting fractal dimension on solid  (Orland et al. 2016)")
    lines.append("  method")
    lines.append("    The solid voxel grid is partitioned into cubic boxes of edge r")
    lines.append("    (r = pitch * 2^k voxels for k = 0,1,2,…).  N(r) = number of boxes")
    lines.append("    intersecting the solid.  D_b = slope of log N vs log(1/r).")
    if box_mm.size:
        lines.append(f"  scales evaluated      : {len(box_mm)}  "
                     f"(box edge {box_mm.min():.3f} – {box_mm.max():.3f} mm)")
        lines.append("  scale  r (mm)   N(r)")
        for r_mm, n_b, used in zip(box_mm, box_counts, box_fit_used):
            mark = " *" if used else "  "
            lines.append(f"   {r_mm:>8.3f}  {int(n_b):>10}{mark}")
        lines.append("  ( * = used in linear fit)")
    if not np.isnan(D_box):
        lines.append(f"  Orland box-counting dimension   D_b  =  {D_box:.4f}")
    else:
        lines.append("  Orland box-counting dimension   D_b  =  n/a "
                     "(too few scales)")

    # ── Hidden-space / overhang angle metrics ─────────────────────────────────
    lines.append("")
    lines.append("Hidden-space / overhang angle metrics  (per-voxel)")
    lines.append("  framework             : 'minimum angle' approaches to overhang /")
    lines.append("                          refuge quantification, reviewed in")
    lines.append("                          Tokeshi & Arakaki 2012, Hydrobiologia 685: 27–47.")
    lines.append("  outward-normal field  : gradient of Gaussian-smoothed solid occupancy")
    lines.append("                          (σ = 1 voxel), sign-flipped to point into void.")
    lines.append("")
    if angles is None:
        lines.append("  (skipped — grid too large, or analysis failed)")
    else:
        a = angles['inclination']['angle_deg']
        lines.append("  1. Surface inclination  θ_n = ∠(outward normal, +Z)")
        lines.append("       evaluated over solid voxels on the interior surface")
        lines.append(f"       voxels              : {a.size:,}")
        lines.append(f"       min / median / max  : "
                     f"{a.min():.2f} / {np.median(a):.2f} / {a.max():.2f}  deg")
        lines.append(f"       mean ± std          : "
                     f"{a.mean():.2f} ± {a.std():.2f}  deg")
        lines.append(f"       overhang fraction   : "
                     f"{100.0 * (a > 90).mean():.2f} %  (θ > 90°)")
        lines.append("")

        s = angles['sky_view']['svf']
        lines.append("  2. Sky-view factor  SVF = unblocked upper-hemisphere rays / total")
        lines.append(f"       rays per voxel      : {angles['sky_view']['n_dirs']}")
        lines.append(f"       void voxels         : {s.size:,}")
        lines.append(f"       min / median / max  : "
                     f"{s.min():.4f} / {np.median(s):.4f} / {s.max():.4f}")
        lines.append(f"       mean ± std          : "
                     f"{s.mean():.4f} ± {s.std():.4f}")
        lines.append(f"       fully hidden (SVF=0): "
                     f"{100.0 * (s == 0).mean():.2f} %")
        lines.append(f"       fully open (SVF=1)  : "
                     f"{100.0 * (s == 1).mean():.2f} %")
        lines.append("")

        h = angles['opening']['half_angle_deg']
        f = angles['opening']['fraction']
        lines.append("  3. Crevice opening  (equivalent cone half-angle α from 26-ray Ω)")
        lines.append(f"       rays per voxel      : {angles['opening']['n_dirs']}")
        lines.append(f"       surface-adj. void   : {h.size:,} voxels")
        lines.append(f"       opening fraction    : "
                     f"{f.min():.4f} / {np.median(f):.4f} / {f.max():.4f}  "
                     f"(min / median / max)")
        lines.append(f"       half-angle α (deg)  : "
                     f"{h.min():.2f} / {np.median(h):.2f} / {h.max():.2f}")
        lines.append(f"       mean α ± std        : "
                     f"{h.mean():.2f} ± {h.std():.2f} deg")
        lines.append("")

        d = angles['dihedral']['angle_deg']
        d_v = d[np.isfinite(d)]
        lines.append("  4. Local dihedral angle  (max pairwise ∠ of solid normals in 3×3×3)")
        lines.append(f"       voxels evaluated    : {d_v.size:,} of {d.size:,}")
        if d_v.size:
            lines.append(f"       min / median / max  : "
                         f"{d_v.min():.2f} / {np.median(d_v):.2f} / "
                         f"{d_v.max():.2f}  deg")
            lines.append(f"       mean ± std          : "
                         f"{d_v.mean():.2f} ± {d_v.std():.2f}  deg")

    report_text = "\n".join(lines)

    # ── Write report file ──────────────────────────────────────────────────────
    out_path = mesh_path.with_name(mesh_path.stem + f"_pockets{pitch}mm.txt")
    out_path.write_text(report_text)

    total_elapsed = time.perf_counter() - t_total
    log_fn(f"  Report saved: {out_path}")
    progress_fn(98)

    # ── Print report to console ────────────────────────────────────────────────
    sep()
    log_fn("REPORT")
    sep()
    for line in lines:
        log_fn(line)

    sep()
    log_fn(f"Total runtime: {total_elapsed:.1f}s")
    log_fn("Analysis complete!")
    sep()
    progress_fn(100)
    status_fn("Done")


# ── GUI Application ───────────────────────────────────────────────────────────

class PocketAnalysisApp:
    def __init__(self, root):
        self.root    = root
        self.config  = _load_config()
        self._running = False
        root.title("Pocket Analysis")
        root.minsize(740, 560)
        self._build_ui()

        # Restore last mesh path
        last = self.config.get("last_mesh", "")
        if last and Path(last).exists():
            self.mesh_path_var.set(last)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root

        # ── File selection ─────────────────────────────────────────────────────
        file_frame = ttk.LabelFrame(root, text="Mesh File", padding=10)
        file_frame.pack(fill="x", padx=12, pady=(12, 6))

        self.mesh_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.mesh_path_var, width=60).pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        ttk.Button(file_frame, text="Browse…", command=self._browse_file).pack(side="right")

        # ── Parameters ────────────────────────────────────────────────────────
        param_frame = ttk.LabelFrame(root, text="Parameters", padding=10)
        param_frame.pack(fill="x", padx=12, pady=6)
        param_frame.columnconfigure(1, weight=1)

        # Voxel size
        ttk.Label(param_frame, text="Voxel size (mm):").grid(
            row=0, column=0, sticky="w", pady=5, padx=(0, 10)
        )
        self.voxel_var = tk.DoubleVar(value=self.config.get("voxel_size", 0.5))
        ttk.Scale(
            param_frame, from_=0.1, to=5.0,
            variable=self.voxel_var, orient="horizontal",
            command=self._on_voxel_change,
        ).grid(row=0, column=1, sticky="ew", padx=4)
        self.voxel_label = ttk.Label(param_frame, text=f"{self.voxel_var.get():.2f}", width=6)
        self.voxel_label.grid(row=0, column=2, sticky="w", padx=(4, 0))

        # Top N
        ttk.Label(param_frame, text="Top N pockets:").grid(
            row=1, column=0, sticky="w", pady=5, padx=(0, 10)
        )
        self.topn_var = tk.IntVar(value=self.config.get("top_n", 20))
        ttk.Scale(
            param_frame, from_=1, to=100,
            variable=self.topn_var, orient="horizontal",
            command=self._on_topn_change,
        ).grid(row=1, column=1, sticky="ew", padx=4)
        self.topn_label = ttk.Label(param_frame, text=str(self.topn_var.get()), width=6)
        self.topn_label.grid(row=1, column=2, sticky="w", padx=(4, 0))

        # Debug toggle
        self.debug_var = tk.BooleanVar(value=self.config.get("debug", False))
        ttk.Checkbutton(
            param_frame, text="Debug mode  (show per-stage timing)",
            variable=self.debug_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 2))

        # ── Run button + status + progress bar ────────────────────────────────
        action_frame = ttk.Frame(root, padding=(12, 4))
        action_frame.pack(fill="x", padx=0)

        self.run_btn = ttk.Button(
            action_frame, text="Run Analysis", command=self._run, width=14
        )
        self.run_btn.pack(side="left")

        self.status_label = ttk.Label(action_frame, text="Ready", width=18)
        self.status_label.pack(side="left", padx=10)

        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(
            action_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(side="right", fill="x", expand=True, padx=(0, 12))

        # ── Console output ────────────────────────────────────────────────────
        console_outer = ttk.LabelFrame(root, text="Console Output", padding=5)
        console_outer.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        self.console = tk.Text(
            console_outer, wrap="none",
            font=("Menlo", 11),
            bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white",
            selectbackground="#264f78",
        )
        vsb = ttk.Scrollbar(console_outer, orient="vertical",   command=self.console.yview)
        hsb = ttk.Scrollbar(console_outer, orient="horizontal", command=self.console.xview)
        self.console.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        self.console.pack(side="left", fill="both", expand=True)

    # ── Slider callbacks ──────────────────────────────────────────────────────

    def _on_voxel_change(self, _=None):
        v = round(float(self.voxel_var.get()), 2)
        self.voxel_label.config(text=f"{v:.2f}")

    def _on_topn_change(self, _=None):
        v = int(round(float(self.topn_var.get())))
        self.topn_var.set(v)
        self.topn_label.config(text=str(v))

    # ── File browser ─────────────────────────────────────────────────────────

    def _browse_file(self):
        initial_dir = None
        last = self.mesh_path_var.get()
        if last:
            p = Path(last)
            if p.parent.exists():
                initial_dir = str(p.parent)

        path = filedialog.askopenfilename(
            title="Select mesh file",
            initialdir=initial_dir,
            filetypes=[
                ("Mesh files", "*.stl *.obj *.ply *.glb *.gltf *.off"),
                ("STL files", "*.stl"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.mesh_path_var.set(path)

    # ── Thread-safe GUI updaters ──────────────────────────────────────────────

    def _log(self, msg):
        def _append():
            self.console.insert("end", msg + "\n")
            self.console.see("end")
        self.root.after(0, _append)

    def _set_progress(self, val):
        def _update():
            self.progress_var.set(int(val))
        self.root.after(0, _update)

    def _set_status(self, text):
        def _update():
            self.status_label.config(text=text)
        self.root.after(0, _update)

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        if self._running:
            return

        mesh_path = self.mesh_path_var.get().strip()
        if not mesh_path:
            self._log("Please select a mesh file first.")
            return
        if not Path(mesh_path).exists():
            self._log(f"File not found: {mesh_path}")
            return

        # Persist settings
        pitch = round(float(self.voxel_var.get()), 2)
        top_n = int(self.topn_var.get())
        debug = bool(self.debug_var.get())
        self.config.update({
            "last_mesh":  mesh_path,
            "voxel_size": pitch,
            "top_n":      top_n,
            "debug":      debug,
        })
        _save_config(self.config)

        # Reset UI
        self.console.delete("1.0", "end")
        self.progress_var.set(0)
        self._set_status("Starting…")
        self._running = True
        self.run_btn.config(state="disabled")

        def _thread():
            try:
                run_analysis(
                    mesh_path, pitch, top_n, debug,
                    self._log, self._set_progress, self._set_status,
                )
            except Exception as exc:
                self._log(f"\nError: {exc}")
                self._log(traceback.format_exc())
                self._set_progress(0)
                self._set_status("Error")
            finally:
                def _done():
                    self._running = False
                    self.run_btn.config(state="normal")
                self.root.after(0, _done)

        threading.Thread(target=_thread, daemon=True).start()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    PocketAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
