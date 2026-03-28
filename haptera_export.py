import numpy as np
import trimesh
import sys
from collections import defaultdict

class Tee:
    """Write to both stdout and a file simultaneously."""
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

# ── configuration ─────────────────────────────────────────────────────────────
TESTING     = False  # True = matplotlib preview only, no mesh or export
FAST_EXPORT = False  # True = export mesh without volume fitting or measurements
DEBUG       = True  # True = print per-function timing diagnostics

DEPTH  = 10 #Number of nodes
K      = 2 #Number of branches per node
SHRINK = 1
OUTPUT      = "haptera_d{}_k{}_s{}.stl".format(DEPTH, K, int(SHRINK * 100))
TEXT_OUTPUT = OUTPUT.replace(".stl", ".txt")

TUBE_SIDES = 10

# ── convergence ───────────────────────────────────────────────────────────────
TARGET_MODE = "volume"      # "volume"       → converge to TARGET_INTERSTITIAL_VOLUME
                            # "surface_area" → converge to TARGET_SURFACE_AREA
TOLERANCE   = 0.001
MAX_ITERS   = 10

# ── cone geometry ─────────────────────────────────────────────────────────────
# Defined here so targets below can reference cone volume.
CONE_H = 10
CONE_R = 10

N_ROOTS        = 20
SEG_LEN        = 0.1
REF_ROOT_R     = 5
STEER_ONSET    = 0.55
STEER_STRENGTH = 1.5
TORSION        = 0.6  # radians of extra branching-plane rotation per depth level (0 = no twist)

_CONE_VOLUME = (1.0 / 3.0) * np.pi * CONE_R**2 * CONE_H
_NOMINAL_VOLUME = 2 * N_ROOTS * np.pi * REF_ROOT_R**2 * SEG_LEN  # original calibration

# ── targets ───────────────────────────────────────────────────────────────────
# "volume" mode — desired void (interstitial) space inside the cone.
# Default matches original formula so existing runs are unaffected.
TARGET_INTERSTITIAL_VOLUME = 600 #_CONE_VOLUME - _NOMINAL_VOLUME   # adjust as needed

# "surface_area" mode — desired wetted haptera surface area (base cap excluded).
# Run once in volume mode first to see what surface area your tree naturally produces,
# then set this to the desired value.
TARGET_SURFACE_AREA = 1000                                  # adjust as needed

# Haptera mesh volume: convergence target in "volume" mode, initial sizing hint in "surface_area" mode.
BASE_VOLUME = _CONE_VOLUME - TARGET_INTERSTITIAL_VOLUME

# ── PRNG ──────────────────────────────────────────────────────────────────────
def make_rng(seed=54321):
    state = [seed & 0xFFFFFFFF]
    def rng():
        state[0] = (1664525 * state[0] + 1013904223) & 0xFFFFFFFF
        return state[0] / 4294967296
    return rng

# ── geometry helpers ──────────────────────────────────────────────────────────
def cone_contains(x, y, z):
    return (z >= 0) and (z <= CONE_H) and (
        np.sqrt(x*x + y*y) <= (CONE_R / CONE_H) * (CONE_H - z)
    )

def wall_proximity(x, y, z):
    r_pos  = np.sqrt(x*x + y*y)
    r_wall = (CONE_R / CONE_H) * (CONE_H - z)
    if r_wall <= 0:
        return 2.0
    return r_pos / r_wall

def inward_direction(x, y):
    r = np.sqrt(x*x + y*y)
    if r < 1e-9:
        return np.array([0.0, 0.0, 0.0])
    return np.array([-x / r, -y / r, 0.0])

def steer(dx, dy, dz, ox, oy, oz):
    prox = wall_proximity(ox, oy, oz)
    if prox < STEER_ONSET:
        return dx, dy, dz
    t = min((prox - STEER_ONSET) / (1.0 - STEER_ONSET), 1.0)
    weight = STEER_STRENGTH * t * t
    inward = inward_direction(ox, oy)
    ndx = dx + inward[0] * weight
    ndy = dy + inward[1] * weight
    ndz = dz
    nl  = np.sqrt(ndx*ndx + ndy*ndy + ndz*ndz)
    return ndx/nl, ndy/nl, ndz/nl

def grow(ox, oy, oz, dx, dy, dz, r, depth, k, seg_len, shrink, rng, max_depth, out):
    if not cone_contains(ox, oy, oz):
        return
    dx, dy, dz = steer(dx, dy, dz, ox, oy, oz)
    if dz < -1e-6:
        seg_len = min(oz / ((-dz) * (depth + 1)), CONE_H)
    ex = ox + dx * seg_len
    ey = oy + dy * seg_len
    ez = oz + dz * seg_len
    if not cone_contains(ex, ey, ez):
        for _ in range(6):
            inward = inward_direction(ex, ey)
            ex += inward[0] * seg_len * 0.25
            ey += inward[1] * seg_len * 0.25
            if cone_contains(ex, ey, ez):
                break
        if not cone_contains(ex, ey, ez):
            return
    actual_len = np.sqrt((ex-ox)**2 + (ey-oy)**2 + (ez-oz)**2)
    if actual_len < 0.02:
        return
    out.append({'start': np.array([ox, oy, oz]),
                'end':   np.array([ex, ey, ez]),
                'r':     r,
                'level': max_depth - depth})
    if depth == 0:
        return
    seg_dir = np.array([ex-ox, ey-oy, ez-oz])
    seg_dir /= np.linalg.norm(seg_dir)
    dx, dy, dz = seg_dir
    phi = np.arctan2(dy, dx) + (max_depth - depth) * TORSION  # parent direction + per-level twist
    for i in range(k):
        angle  = (2 * np.pi * i / k) + phi + rng() * 0.5 - 0.25
        spread = 0.28 + rng() * 0.18
        ndx = dx + np.cos(angle) * spread
        ndy = dy + np.sin(angle) * spread
        ndz = dz + rng() * 0.3 - 0.1
        nl  = np.sqrt(ndx*ndx + ndy*ndy + ndz*ndz)
        grow(ex, ey, ez, ndx/nl, ndy/nl, ndz/nl,
             r, depth-1, k, seg_len*shrink, shrink, rng, max_depth, out)

# ── volume helpers ────────────────────────────────────────────────────────────
def naive_volume(segs):
    return sum(np.pi * s['r']**2 * np.linalg.norm(s['end'] - s['start'])
               for s in segs)

def overlap_volume(segs):
    junction = defaultdict(list)
    for s in segs:
        junction[tuple(np.round(s['start'], 6))].append(s['r'])
        junction[tuple(np.round(s['end'],   6))].append(s['r'])
    overlap = 0.0
    for radii in junction.values():
        if len(radii) < 2:
            continue
        for r in radii:
            overlap += (2.0 / 3.0) * np.pi * r**3
    return overlap

def scale_radii(segs, factor):
    for s in segs:
        s['r'] *= factor

def cubic_correction(V_naive, V_measured, V_target):
    """
    Return the scale factor f so the union volume hits V_target.

    Model: V(f) = f²·V_naive − f³·V_overlap = V_target
      (cylinder naive ∝ r², junction overlap ∝ r³)

    The cubic has a maximum of  V_max = 4·V_naive³ / (27·V_overlap²).
    If V_target > V_max the model has no solution — this happens when tubes
    overlap so heavily near the apex that no radius can reach V_target via the
    cubic relationship.  In that case we fall back to the simple sqrt step,
    which is damped but still convergent.
    """
    if V_naive <= 0 or V_measured <= 0 or V_target <= 0:
        return 1.0
    V_overlap = V_naive - V_measured
    if V_overlap <= 0:                              # no detectable overlap
        return np.sqrt(V_target / V_naive)
    V_max = 4.0 * V_naive**3 / (27.0 * V_overlap**2)
    if V_target > V_max:                            # cubic unsolvable → sqrt fallback
        return np.sqrt(V_target / V_measured)
    f = np.sqrt(V_target / V_measured)              # warm start
    for _ in range(40):
        g  = V_naive * f**2 - V_overlap * f**3 - V_target
        gp = 2 * V_naive * f - 3 * V_overlap * f**2
        if abs(gp) < 1e-14:
            break
        step = g / gp
        f   -= step
        if abs(step) < 1e-10 * abs(f):
            break
    return max(f, 0.1)

def naive_surface_area(segs):
    """Lateral surface area of all tubes (no end-caps, no overlap correction)."""
    return sum(2 * np.pi * s['r'] * np.linalg.norm(s['end'] - s['start'])
               for s in segs)

def quadratic_sa_correction(SA_naive, SA_measured, SA_target):
    """
    Solve f·SA_naive − f²·SA_overlap = SA_target for f.

    Model: lateral area ∝ r (scales as f), junction overlap area ∝ r² (scales as f²).

    The quadratic has a maximum of SA_max = SA_naive² / (4·SA_overlap).
    If SA_target > SA_max the model has no solution; fall back to linear scaling.
    """
    if SA_naive <= 0 or SA_measured <= 0:
        return 1.0
    SA_overlap = SA_naive - SA_measured
    if SA_overlap <= 0:                             # no detectable overlap
        return SA_target / SA_naive
    SA_max = SA_naive**2 / (4.0 * SA_overlap)
    if SA_target > SA_max:                          # quadratic unsolvable → linear fallback
        return SA_target / SA_measured
    f = SA_target / SA_measured                     # linear warm start
    for _ in range(40):
        g  = SA_naive * f - SA_overlap * f**2 - SA_target
        gp = SA_naive - 2 * SA_overlap * f
        if abs(gp) < 1e-14:
            break
        step = g / gp
        f   -= step
        if abs(step) < 1e-10 * abs(f):
            break
    return max(f, 0.1)

def mesh_wetted_area(mesh):
    """Return (wetted_area, cap_area): total mesh area minus the flat attachment base."""
    base_mask = mesh.face_normals[:, 2] < -0.999
    cap = float(trimesh.triangles.area(mesh.triangles[base_mask]).sum()) if base_mask.any() else 0.0
    return mesh.area - cap, cap

# ── tube mesh builder ─────────────────────────────────────────────────────────
def tube_mesh(start, end, radius, sides=8):
    direction = end - start
    length    = np.linalg.norm(direction)
    if length < 1e-6:
        return None
    direction /= length
    z_axis   = np.array([0.0, 0.0, 1.0])
    axis     = np.cross(z_axis, direction)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        R = np.eye(3) if np.dot(z_axis, direction) > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        axis  /= axis_len
        angle  = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        c, s   = np.cos(angle), np.sin(angle)
        t      = 1.0 - c
        ax, ay, az = axis
        R = np.array([
            [t*ax*ax + c,      t*ax*ay - s*az,  t*ax*az + s*ay],
            [t*ax*ay + s*az,   t*ay*ay + c,      t*ay*az - s*ax],
            [t*ax*az - s*ay,   t*ay*az + s*ax,   t*az*az + c   ],
        ])
    geo = trimesh.creation.cylinder(radius=radius, height=length, sections=sides)
    transform         = np.eye(4)
    transform[:3, :3] = R
    transform[:3,  3] = (start + end) / 2.0
    geo.apply_transform(transform)
    return geo

def build_meshes(segs, sides):
    meshes = []
    for s in segs:
        m = tube_mesh(s['start'], s['end'], s['r'], sides=sides)
        if m is not None:
            meshes.append(m)
    return meshes

def _run_manifold_union(meshes, _log):
    """Binary-tree parallel union. Returns the raw Manifold object (no trimesh conversion)."""
    import time
    from manifold3d import Manifold, Mesh
    from concurrent.futures import ThreadPoolExecutor

    def to_m(m):
        return Manifold(mesh=Mesh(
            vert_properties=m.vertices.astype(np.float32),
            tri_verts=m.faces.astype(np.uint32),
        ))

    def union_pair(pair):
        return pair[0] + pair[1]

    _log(f"    [manifold_union] converting {len(meshes)} meshes to Manifold...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor() as ex:
        level = list(ex.map(to_m, meshes))
    _log(f"    [manifold_union] conversion done  {time.perf_counter()-t0:.2f}s")

    tree_level = 1
    while len(level) > 1:
        pairs = list(zip(level[0::2], level[1::2]))
        tail  = [level[-1]] if len(level) % 2 else []
        _log(f"    [manifold_union] tree level {tree_level}: {len(pairs)} pairs ({len(level)} → {len(pairs) + len(tail)})...")
        t0 = time.perf_counter()
        with ThreadPoolExecutor() as ex:
            level = list(ex.map(union_pair, pairs)) + tail
        _log(f"    [manifold_union] level {tree_level} done  {time.perf_counter()-t0:.2f}s")
        tree_level += 1

    return level[0]


def _manifold_to_trimesh(manifold, _log):
    """Convert a Manifold to trimesh.Trimesh. Only called once after convergence.

    process=False and validate=False skip all mesh cleanup and validation — safe
    because manifold3d guarantees a watertight, non-self-intersecting mesh.
    Arrays are passed directly (no extra copies).
    """
    import time
    _log("    [extract_mesh] converting Manifold to trimesh...")
    t0 = time.perf_counter()
    r  = manifold.to_mesh()
    result = trimesh.Trimesh(
        vertices=r.vert_properties,
        faces=r.tri_verts,
        process=False,
        validate=False,
    )
    _log(f"    [extract_mesh] done  {time.perf_counter()-t0:.2f}s  "
         f"({len(result.vertices)} verts, {len(result.faces)} faces)")
    return result


def build_union_mesh(segs, sides):
    """Build and return the boolean union of all tube meshes as a trimesh.Trimesh.

    Used by surface_area mode (needs face normals for wetted-area detection).
    For volume mode use build_manifold() instead — it skips the trimesh
    conversion, measuring volume directly on the Manifold object.
    """
    import time
    try:
        from tqdm import tqdm as _tqdm
        _log = _tqdm.write
    except ImportError:
        _log = print
    _dlog = _log if DEBUG else (lambda _: None)

    _dlog(f"    [build_meshes] building {len(segs)} tube meshes...")
    t0     = time.perf_counter()
    meshes = build_meshes(segs, sides)
    _dlog(f"    [build_meshes] done  {time.perf_counter()-t0:.2f}s")

    manifold = _run_manifold_union(meshes, _dlog)
    return _manifold_to_trimesh(manifold, _dlog)


def build_manifold(segs, sides):
    """Build the boolean union and return the raw Manifold (no trimesh conversion).

    Used by volume mode: Manifold.volume() reads the volume directly in C++,
    so we never pay the trimesh extraction cost during iteration.
    """
    import time
    try:
        from tqdm import tqdm as _tqdm
        _log = _tqdm.write
    except ImportError:
        _log = print
    _dlog = _log if DEBUG else (lambda _: None)

    _dlog(f"    [build_meshes] building {len(segs)} tube meshes...")
    t0     = time.perf_counter()
    meshes = build_meshes(segs, sides)
    _dlog(f"    [build_meshes] done  {time.perf_counter()-t0:.2f}s")

    return _run_manifold_union(meshes, _dlog)

# ── segment builder ───────────────────────────────────────────────────────────
def build_segments(depth, k, shrink):
    root_r = np.sqrt(BASE_VOLUME / (N_ROOTS * np.pi * SEG_LEN * (depth + 1)))
    rng    = make_rng(54321)
    segs   = []
    for i in range(N_ROOTS):
        a  = (2 * np.pi * i / N_ROOTS) + 0.35
        ox, oy, oz = 0.0, 0.0, CONE_H - 0.05
        rdx = np.cos(a) * 0.45
        rdy = np.sin(a) * 0.45
        rdz = -1.0
        rl  = np.sqrt(rdx*rdx + rdy*rdy + rdz*rdz)
        grow(ox, oy, oz, rdx/rl, rdy/rl, rdz/rl,
             root_r, depth, k, SEG_LEN, shrink, rng, depth, segs)
    nv = naive_volume(segs)
    cv = nv - overlap_volume(segs)
    # cv is almost always ≤ 0 for this geometry (r > seg_len means junction
    # hemispheres exceed cylinder volume), so fall back to naive_volume.
    ref = cv if cv > 0 else nv
    if ref > 0:
        scale_radii(segs, np.sqrt(BASE_VOLUME / ref))
    return segs

# ── main ──────────────────────────────────────────────────────────────────────
import time as _time
print(f"Building segments (depth={DEPTH}, k={K}, shrink={SHRINK})...")
_t = _time.perf_counter()
segs = build_segments(DEPTH, K, SHRINK)
if DEBUG:
    print(f"  [build_segments] done  {_time.perf_counter()-_t:.2f}s  {len(segs)} segments")
else:
    print(f"  {len(segs)} segments generated")

# ── testing / preview mode ────────────────────────────────────────────────────
if TESTING:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    print(f"\nTESTING mode: plotting {len(segs)} segments...")
    max_level = max(s['level'] for s in segs)
    cmap      = cm.viridis
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    for s in segs:
        col = cmap(s['level'] / max(max_level, 1))
        ax.plot([s['start'][0], s['end'][0]],
                [s['start'][1], s['end'][1]],
                [s['start'][2], s['end'][2]],
                color=col, linewidth=max(s['r'] * 40, 0.3))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z (up)')
    ax.set_title(f"depth={DEPTH}  k={K}  shrink={SHRINK}  n_roots={N_ROOTS}  seg_len={SEG_LEN}")
    plt.tight_layout()
    plt.show()
    sys.exit(0)

if FAST_EXPORT:
    print("\nFAST_EXPORT mode: building mesh without volume fitting...")
    meshes   = build_meshes(segs, TUBE_SIDES)
    combined = trimesh.util.concatenate(meshes)
    combined.export(OUTPUT)
    print(f"  exported {len(meshes)} tubes → {OUTPUT}")
    print(f"  vertices: {len(combined.vertices)}  faces: {len(combined.faces)}")
    sys.exit(0)

if TARGET_MODE == "surface_area":
    print(f"\nIterating to surface area (target={TARGET_SURFACE_AREA:.4f}, tol={TOLERANCE*100:.2f}%)...")
    try:
        from tqdm import tqdm as _tqdm
        _ibar = _tqdm(total=MAX_ITERS, desc="  SA iters", unit="iter",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}  elapsed {elapsed}  ETA {remaining}  {rate_fmt}{postfix}")
        _log  = _tqdm.write
    except ImportError:
        _ibar = None
        _log  = print
    import time as _time
    _dlog = _log if DEBUG else (lambda _: None)
    final_mesh = None
    for iteration in range(1, MAX_ITERS + 1):
        _dlog(f"  ── iter {iteration} ──────────────────────────────")
        unioned = build_union_mesh(segs, TUBE_SIDES)
        _dlog(f"    [measure_SA] computing wetted surface area...")
        _tm = _time.perf_counter()
        wetted_sa, _ = mesh_wetted_area(unioned)
        _dlog(f"    [measure_SA] done  {_time.perf_counter()-_tm:.2f}s")
        error = abs(wetted_sa - TARGET_SURFACE_AREA) / TARGET_SURFACE_AREA
        msg   = f"  iter {iteration}: SA={wetted_sa:.4f}  error={error*100:.3f}%"
        if error <= TOLERANCE:
            _log(msg + "  ✓ converged")
            if _ibar: _ibar.update(1)
            final_mesh = unioned
            break
        _dlog(f"    [correction] computing quadratic SA correction...")
        _tm = _time.perf_counter()
        correction = quadratic_sa_correction(naive_surface_area(segs), wetted_sa, TARGET_SURFACE_AREA)
        scale_radii(segs, correction)
        _dlog(f"    [correction] done  {_time.perf_counter()-_tm:.4f}s  f={correction:.5f}")
        _log(msg + f"  → ×{correction:.5f}")
        if _ibar: _ibar.update(1)
        if iteration == MAX_ITERS:
            _log("  ! max iterations reached")
            final_mesh = unioned
    if _ibar: _ibar.close()
    combined         = final_mesh
    final_vol        = combined.volume
else:  # TARGET_MODE == "volume"
    print(f"\nIterating to interstitial volume (target={TARGET_INTERSTITIAL_VOLUME:.4f}, tol={TOLERANCE*100:.2f}%)...")
    try:
        from tqdm import tqdm as _tqdm
        _ibar = _tqdm(total=MAX_ITERS, desc="  volume iters", unit="iter",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}  elapsed {elapsed}  ETA {remaining}  {rate_fmt}{postfix}")
        _log  = _tqdm.write
    except ImportError:
        _ibar = None
        _log  = print
    import time as _time
    _dlog = _log if DEBUG else (lambda _: None)
    combined  = None
    final_vol = None
    for iteration in range(1, MAX_ITERS + 1):
        _dlog(f"  ── iter {iteration} ──────────────────────────────")
        manifold = build_manifold(segs, TUBE_SIDES)
        _dlog(f"    [measure_vol] reading volume from Manifold (C++)...")
        _tm = _time.perf_counter()
        measured_vol = manifold.volume()
        _dlog(f"    [measure_vol] done  {_time.perf_counter()-_tm:.4f}s  vol={measured_vol:.4f}")
        combined_iter = _manifold_to_trimesh(manifold, _dlog)
        _dlog(f"    [measure_hull] computing convex hull volume...")
        _tm = _time.perf_counter()
        hull_vol_iter = combined_iter.convex_hull.volume
        _dlog(f"    [measure_hull] done  {_time.perf_counter()-_tm:.4f}s  hull_vol={hull_vol_iter:.4f}")
        interstitial_iter = hull_vol_iter - measured_vol
        error = abs(interstitial_iter - TARGET_INTERSTITIAL_VOLUME) / TARGET_INTERSTITIAL_VOLUME
        msg   = f"  iter {iteration}: interstitial={interstitial_iter:.4f}  haptera={measured_vol:.4f}  error={error*100:.3f}%"
        if error <= TOLERANCE:
            _log(msg + "  ✓ converged")
            if _ibar: _ibar.update(1)
            combined  = combined_iter
            final_vol = measured_vol
            break
        haptera_target = hull_vol_iter - TARGET_INTERSTITIAL_VOLUME
        if haptera_target <= 0:
            _log(f"  ! TARGET_INTERSTITIAL_VOLUME ({TARGET_INTERSTITIAL_VOLUME:.4f}) >= hull volume "
                 f"({hull_vol_iter:.4f}) — target is geometrically infeasible for this tree. "
                 f"Lower TARGET_INTERSTITIAL_VOLUME and re-run.")
            combined  = combined_iter
            final_vol = measured_vol
            break
        _dlog(f"    [correction] computing cubic volume correction (haptera_target={haptera_target:.4f})...")
        _tm = _time.perf_counter()
        correction = cubic_correction(naive_volume(segs), measured_vol, haptera_target)
        scale_radii(segs, correction)
        _dlog(f"    [correction] done  {_time.perf_counter()-_tm:.4f}s  f={correction:.5f}")
        _log(msg + f"  → ×{correction:.5f}")
        if _ibar: _ibar.update(1)
        if iteration == MAX_ITERS:
            _log("  ! max iterations reached")
            combined  = combined_iter
            final_vol = measured_vol
    if _ibar: _ibar.close()
if DEBUG: print(f"[export] writing {OUTPUT}...")
_t = _time.perf_counter()
combined.export(OUTPUT)
if DEBUG: print(f"[export] done  {_time.perf_counter()-_t:.2f}s")
sys.stdout = Tee(TEXT_OUTPUT)

# ── surface area ──────────────────────────────────────────────────────────────
if DEBUG: print(f"[surface_area] computing haptera surface area...")
_t = _time.perf_counter()
haptera_surface_area = combined.area
base_mask = combined.face_normals[:, 2] < -0.999
base_cap_area = float(trimesh.triangles.area(combined.triangles[base_mask]).sum()) if base_mask.any() else 0.0
area_note = "exact (includes flat base cap)"
if DEBUG: print(f"[surface_area] done  {_time.perf_counter()-_t:.2f}s  area={haptera_surface_area:.4f}")

# ── convex hull (bounding envelope) ───────────────────────────────────────────
if DEBUG: print(f"[convex_hull] computing bounding envelope...")
_t = _time.perf_counter()
haptera_hull      = combined.convex_hull
hull_volume       = haptera_hull.volume
hull_surface_area = haptera_hull.area
if DEBUG: print(f"[convex_hull] done  {_time.perf_counter()-_t:.2f}s  hull_vol={hull_volume:.4f}")

# ── cone geometry (analytical, for reference) ─────────────────────────────────
cone_volume       = (1.0 / 3.0) * np.pi * CONE_R**2 * CONE_H
cone_lateral_area = np.pi * CONE_R * np.sqrt(CONE_R**2 + CONE_H**2)
cone_base_area    = np.pi * CONE_R**2

# ── interstitial measurements ─────────────────────────────────────────────────
interstitial_volume = hull_volume - final_vol
total_surface_area  = haptera_surface_area - base_cap_area
total_bounding_area = hull_surface_area + haptera_surface_area
sa_to_vol           = total_surface_area / interstitial_volume if interstitial_volume > 0 else 0

# ── output ────────────────────────────────────────────────────────────────────
print(f"\nExported : {OUTPUT}")
print(f"")
print(f"Parameters")
print(f"  depth                  : {DEPTH}")
print(f"  k                      : {K}")
print(f"  shrink                 : {SHRINK}")
print(f"  n_roots                : {N_ROOTS}")
print(f"  seg_len                : {SEG_LEN}")
print(f"  tube_sides             : {TUBE_SIDES}")
print(f"  steer_onset            : {STEER_ONSET}")
print(f"  steer_strength         : {STEER_STRENGTH}")
print(f"  cone_h                 : {CONE_H}")
print(f"  cone_r                 : {CONE_R}")
print(f"  target_mode            : {TARGET_MODE}")
if TARGET_MODE == "surface_area":
    print(f"  target_surface_area    : {TARGET_SURFACE_AREA:.4f}")
else:
    print(f"  target_interstitial_vol: {TARGET_INTERSTITIAL_VOLUME:.4f}")
    print(f"  base_volume (haptera)  : {BASE_VOLUME:.4f}")
print(f"")
print(f"Mesh")
print(f"  segments               : {len(segs)}")
print(f"  vertices               : {len(combined.vertices)}")
print(f"  faces                  : {len(combined.faces)}")
print(f"")
print(f"Haptera")
print(f"  volume                 : {final_vol:.4f}")
if TARGET_MODE == "volume":
    print(f"  interstitial vol error : {abs(interstitial_volume - TARGET_INTERSTITIAL_VOLUME) / TARGET_INTERSTITIAL_VOLUME * 100:.3f}%")
print(f"  surface area (w/ cap)  : {haptera_surface_area:.4f}  ({area_note})")
print(f"  wetted surface area    : {total_surface_area:.4f}  (base cap excluded)")
if TARGET_MODE == "surface_area":
    print(f"  SA error               : {abs(total_surface_area - TARGET_SURFACE_AREA) / TARGET_SURFACE_AREA * 100:.3f}%")
print(f"  base cap area          : {base_cap_area:.4f}")
print(f"")
print(f"Cone (design reference)")
print(f"  total volume           : {cone_volume:.4f}")
print(f"  lateral surface area   : {cone_lateral_area:.4f}")
print(f"  base area              : {cone_base_area:.4f}  (excluded — open base)")
print(f"")
print(f"Convex hull (actual bounding envelope)")
print(f"  hull volume            : {hull_volume:.4f}")
print(f"  hull surface area      : {hull_surface_area:.4f}")
print(f"")
print(f"Interstitial space")
print(f"  volume                 : {interstitial_volume:.4f}  (hull − haptera)")
print(f"  external surface area  : {hull_surface_area:.4f}  (hull surface)")
print(f"  internal surface area  : {haptera_surface_area:.4f}  (haptera surface)")
print(f"  total bounding area    : {total_bounding_area:.4f}    (Total surface area exlcluding bottom)")
print(f"  SA / volume ratio      : {sa_to_vol:.4f}  (complexity index using Total surface area)")

sys.stdout.close()
