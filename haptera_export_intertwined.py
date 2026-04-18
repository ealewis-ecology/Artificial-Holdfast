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
DEBUG       = False

DEPTH  = 6   # more segments per strand for detail
K      = 2   # branches per node

TUBE_SIDES = 10

# ── convergence ───────────────────────────────────────────────────────────────
TOLERANCE = 0.001
MAX_ITERS = 40

# ── cone geometry ─────────────────────────────────────────────────────────────
CONE_H = 130
CONE_R = 130
VERT_BOSS_R  = 9
VERT_HOLE_R  = 6
HORIZ_BOSS_R = 6
HORIZ_HOLE_R = 4
HORIZ_N      = 2
HORIZ_S      = 60
HORIZ_H      = 30
SIMPLIFY_TARGET = 6000000

# ── intertwining parameters ───────────────────────────────────────────────────
# Fewer roots than the original so individual helical strands are visually distinct.
N_ROOTS          = 10     # number of helical strands
HELIX_TURNS      = 0.25  # full rotations around the cone axis from apex to base
RADIAL_FRACTION  = 0.65  # helix radius as a fraction of the available cone radius at each z

# How strongly each branch is steered toward its helical lane waypoint.
LANE_BIAS        = .1
# Repulsion activates within REPULSION_R_FACTOR × (2 × tube_radius) of another strand.
REPULSION_R_FACTOR  = 3.0
REPULSION_STRENGTH  = 1.2

SEG_LEN        = CONE_H / DEPTH
REF_ROOT_R     = 5
STEER_ONSET    = 0.90
STEER_STRENGTH = 1.1
TORSION        = 0.3   # reduced: helical steering dominates; less random twist needed

_CONE_VOLUME = (1.0 / 3.0) * np.pi * CONE_R**2 * CONE_H
_NOMINAL_VOLUME = 2 * N_ROOTS * np.pi * REF_ROOT_R**2 * SEG_LEN

# ── targets ───────────────────────────────────────────────────────────────────
TARGET_INTERSTITIAL_FRACTION = 0.747920635
BASE_VOLUME = _CONE_VOLUME * (1 - TARGET_INTERSTITIAL_FRACTION)

OUTPUT      = "haptera_intertwined_d{}_k{}_r{}_h{}_f{}.stl".format(
                DEPTH, K, CONE_R, CONE_H, round(TARGET_INTERSTITIAL_FRACTION * 1000))
TEXT_OUTPUT = OUTPUT.replace(".stl", ".txt")


# ── PRNG ──────────────────────────────────────────────────────────────────────
def make_rng(seed=54321):
    state = [seed & 0xFFFFFFFF]
    def rng():
        state[0] = (1664525 * state[0] + 1013904223) & 0xFFFFFFFF
        return state[0] / 4294967296
    return rng

# ── geometry helpers ──────────────────────────────────────────────────────────
def cone_contains(x, y, z, r=0):
    r_pos  = np.sqrt(x*x + y*y)
    r_eff  = max((CONE_R / CONE_H) * (CONE_H - z) - r, 0.0)
    return (z >= 0) and (z <= CONE_H) and (r_pos <= r_eff)

def wall_proximity(x, y, z, r=0):
    r_pos  = np.sqrt(x*x + y*y)
    r_eff  = (CONE_R / CONE_H) * (CONE_H - z) - r
    if r_eff <= 0:
        return 2.0
    return r_pos / r_eff

def inward_direction(x, y):
    r = np.sqrt(x*x + y*y)
    if r < 1e-9:
        return np.array([0.0, 0.0, 0.0])
    slope = CONE_R / CONE_H
    raw = np.array([-x / r, -y / r, -slope])
    return raw / np.linalg.norm(raw)

def steer(dx, dy, dz, ox, oy, oz, r=0):
    prox = wall_proximity(ox, oy, oz, r)
    if prox < STEER_ONSET:
        return dx, dy, dz
    t = min((prox - STEER_ONSET) / (1.0 - STEER_ONSET), 1.0)
    weight = STEER_STRENGTH * t * t
    inward = inward_direction(ox, oy)
    ndx = dx + inward[0] * weight
    ndy = dy + inward[1] * weight
    ndz = dz + inward[2] * weight
    nl  = np.sqrt(ndx*ndx + ndy*ndy + ndz*ndz)
    return ndx/nl, ndy/nl, ndz/nl

# ── intertwining helpers ──────────────────────────────────────────────────────
def helix_target_direction(ox, oy, oz, lane, n_lanes, seg_len):
    """Direction toward the next waypoint on lane's parametric helix."""
    z_target = max(0.0, oz - seg_len)
    z_frac_t = max(0.0, min(1.0, (CONE_H - z_target) / CONE_H))
    phi_t = 2.0 * np.pi * lane / n_lanes + HELIX_TURNS * 2.0 * np.pi * z_frac_t
    cone_r_t = (CONE_R / CONE_H) * (CONE_H - z_target)
    tx = RADIAL_FRACTION * cone_r_t * np.cos(phi_t) - ox
    ty = RADIAL_FRACTION * cone_r_t * np.sin(phi_t) - oy
    tz = z_target - oz
    length = np.sqrt(tx*tx + ty*ty + tz*tz)
    if length < 1e-9:
        return np.array([0.0, 0.0, -1.0])
    return np.array([tx / length, ty / length, tz / length])

def repulsion_vector(ox, oy, oz, placed_entries, r_tube):
    """Outward push away from nearby placed segment midpoints of other lanes."""
    min_sep = REPULSION_R_FACTOR * 2.0 * r_tube
    rep = np.zeros(3)
    for (cx, cy, cz) in placed_entries:
        diff = np.array([ox - cx, oy - cy, oz - cz])
        d = np.linalg.norm(diff)
        if 1e-6 < d < min_sep:
            weight = ((min_sep - d) / min_sep) ** 2
            rep += (diff / d) * weight
    n = np.linalg.norm(rep)
    if n > 1e-9:
        return rep / n
    return rep

# ── growth ────────────────────────────────────────────────────────────────────
def grow(ox, oy, oz, dx, dy, dz, r, depth, k, seg_len, rng, max_depth, out,
         lane, n_lanes, placed_entries):
    """Grow a helical, non-intersecting branch inside the cone.

    Each step blends the current direction with:
      1. Cone-wall steering (prevents escape)
      2. Helical lane target (makes strands spiral around the axis)
      3. Repulsion from already-placed segments (prevents intersection)
    """
    if not cone_contains(ox, oy, oz, r):
        return

    # 1. Cone boundary steering
    dx, dy, dz = steer(dx, dy, dz, ox, oy, oz, r)

    # 2. Helical lane steering
    helix_dir = helix_target_direction(ox, oy, oz, lane, n_lanes, seg_len)

    # 3. Repulsion from nearby segments of other strands
    rep = repulsion_vector(ox, oy, oz, placed_entries, r)

    # Blend all three contributions
    ndx = dx + LANE_BIAS * helix_dir[0] + REPULSION_STRENGTH * rep[0]
    ndy = dy + LANE_BIAS * helix_dir[1] + REPULSION_STRENGTH * rep[1]
    ndz = dz + LANE_BIAS * helix_dir[2] + REPULSION_STRENGTH * rep[2]
    nl  = np.sqrt(ndx*ndx + ndy*ndy + ndz*ndz)
    if nl < 1e-9:
        return
    dx, dy, dz = ndx/nl, ndy/nl, ndz/nl

    if dz < -1e-6:
        seg_len = min(oz / ((-dz) * (depth + 1)), CONE_H)

    ex = ox + dx * seg_len
    ey = oy + dy * seg_len
    ez = oz + dz * seg_len

    if not cone_contains(ex, ey, ez, r):
        for _ in range(6):
            inward = inward_direction(ex, ey)
            ex += inward[0] * seg_len * 0.25
            ey += inward[1] * seg_len * 0.25
            ez += inward[2] * seg_len * 0.25
            if cone_contains(ex, ey, ez, r):
                break
        if not cone_contains(ex, ey, ez, r):
            return

    actual_len = np.sqrt((ex-ox)**2 + (ey-oy)**2 + (ez-oz)**2)
    if actual_len < 0.02:
        return

    out.append({'start': np.array([ox, oy, oz]),
                'end':   np.array([ex, ey, ez]),
                'r':     r,
                'level': max_depth - depth})

    # Register this segment's midpoint so future branches repel from it
    placed_entries.append(((ox+ex)/2, (oy+ey)/2, (oz+ez)/2))

    if depth == 0:
        return

    seg_dir = np.array([ex-ox, ey-oy, ez-oz])
    seg_dir /= np.linalg.norm(seg_dir)
    dx, dy, dz = seg_dir

    phi = np.arctan2(dy, dx) + (max_depth - depth) * TORSION
    for i in range(k):
        # Fan sub-branches within the parent lane's angular sector
        sub_offset = (i / max(k - 1, 1) - 0.5) / n_lanes
        child_lane = lane + sub_offset
        angle  = (2 * np.pi * i / k) + phi + rng() * 0.5 - 0.25
        spread = 0.28 + rng() * 0.18
        cdx = dx + np.cos(angle) * spread
        cdy = dy + np.sin(angle) * spread
        cdz = dz + rng() * 0.3 - 0.1
        nl  = np.sqrt(cdx*cdx + cdy*cdy + cdz*cdz)
        grow(ex, ey, ez, cdx/nl, cdy/nl, cdz/nl,
             r, depth-1, k, seg_len, rng, max_depth, out,
             child_lane, n_lanes, placed_entries)

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
    if V_naive <= 0 or V_measured <= 0 or V_target <= 0:
        return 1.0
    V_overlap = V_naive - V_measured
    if V_overlap <= 0:
        return np.sqrt(V_target / V_naive)
    V_max = 4.0 * V_naive**3 / (27.0 * V_overlap**2)
    if V_target > V_max:
        return np.sqrt(V_target / V_measured)
    f = np.sqrt(V_target / V_measured)
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
    endpoints = {}
    for s in segs:
        for pt, r in ((s['start'], s['r']), (s['end'], s['r'])):
            key = tuple(np.round(pt, 6))
            if key not in endpoints or r > endpoints[key]:
                endpoints[key] = r
    for pt, r in endpoints.items():
        sph = trimesh.creation.icosphere(subdivisions=1, radius=r)
        sph.apply_translation(np.array(pt))
        meshes.append(sph)
    return meshes

def _run_manifold_union(meshes, _log):
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
    import time
    _log("    [extract_mesh] converting Manifold to trimesh...")
    t0 = time.perf_counter()
    r  = manifold.to_mesh()
    result = trimesh.Trimesh(
        vertices=r.vert_properties[:, :3],
        faces=r.tri_verts,
        process=False,
        validate=False,
    )
    _log(f"    [extract_mesh] done  {time.perf_counter()-t0:.2f}s  "
         f"({len(result.vertices)} verts, {len(result.faces)} faces)")
    return result


def build_manifold(segs, sides):
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
def build_segments(depth, k):
    """Generate helical, non-intersecting haptera skeletons.

    Each of the N_ROOTS strands is assigned an angular lane.  Branches steer
    toward a parametric helix waypoint (HELIX_TURNS rotations from apex to base)
    and repel from already-placed midpoints of other strands, so no two strands
    can occupy the same space.  The overall envelope remains cone-shaped.
    """
    root_r = np.sqrt(BASE_VOLUME / (N_ROOTS * np.pi * SEG_LEN * (depth + 1)))
    rng    = make_rng(54321)
    segs   = []
    # Shared repulsion list: every placed segment midpoint is registered here
    # so subsequent strands curve around it.
    placed_entries = []
    for i in range(N_ROOTS):
        ox, oy, oz = 0.0, 0.0, CONE_H - 0.05
        # Initial direction: toward first helix waypoint for this lane
        init_dir = helix_target_direction(ox, oy, oz, i, N_ROOTS, SEG_LEN)
        grow(ox, oy, oz,
             init_dir[0], init_dir[1], init_dir[2],
             root_r, depth, k, SEG_LEN, rng, depth, segs,
             lane=i, n_lanes=N_ROOTS, placed_entries=placed_entries)
    nv = naive_volume(segs)
    cv = nv - overlap_volume(segs)
    ref = cv if cv > 0 else nv
    if ref > 0:
        scale_radii(segs, np.sqrt(BASE_VOLUME / ref))
    return segs

# ── main ──────────────────────────────────────────────────────────────────────
import time as _time
print(f"Building intertwined segments (depth={DEPTH}, k={K}, strands={N_ROOTS}, helix_turns={HELIX_TURNS})...")
_t = _time.perf_counter()
segs = build_segments(DEPTH, K)
if DEBUG:
    print(f"  [build_segments] done  {_time.perf_counter()-_t:.2f}s  {len(segs)} segments")
else:
    print(f"  {len(segs)} segments generated")

# ── hole volume correction ────────────────────────────────────────────────────
hole_volume       = np.pi * VERT_HOLE_R**2 * CONE_H if VERT_HOLE_R > 0 else 0.0
hole_lateral_area = 2 * np.pi * VERT_HOLE_R * CONE_H if VERT_HOLE_R > 0 else 0.0

# ── boss/hole manifolds ───────────────────────────────────────────────────────
from manifold3d import Manifold as _Manifold, Mesh as _MfdMesh

def _trimesh_to_mfd(m):
    return _Manifold(mesh=_MfdMesh(
        vert_properties=m.vertices.astype(np.float32),
        tri_verts=m.faces.astype(np.uint32),
    ))

_vert_boss_manifold  = None
_vert_hole_manifold  = None
_horiz_boss_manifolds = []
_horiz_hole_manifolds = []
if VERT_BOSS_R > 0:
    _boss_cyl = trimesh.creation.cylinder(radius=VERT_BOSS_R, height=CONE_H, sections=64)
    _boss_cyl.apply_translation([0.0, 0.0, CONE_H / 2.0])
    _vert_boss_manifold = _trimesh_to_mfd(_boss_cyl)
if VERT_HOLE_R > 0:
    _hole_cyl = trimesh.creation.cylinder(radius=VERT_HOLE_R, height=CONE_H * 3, sections=64)
    _hole_cyl.apply_translation([0.0, 0.0, CONE_H / 2.0])
    _vert_hole_manifold = _trimesh_to_mfd(_hole_cyl)
if HORIZ_BOSS_R > 0 or HORIZ_HOLE_R > 0:
    _rot_y90 = trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    _cone_r_at_horiz_h = CONE_R * (CONE_H - HORIZ_H) / CONE_H
    _horiz_y_offsets = [0.0] if HORIZ_N == 1 else [+HORIZ_S / 2.0, -HORIZ_S / 2.0]
    for _y_off in _horiz_y_offsets:
        if HORIZ_BOSS_R > 0:
            _boss_half_len = np.sqrt(max(_cone_r_at_horiz_h**2 - _y_off**2, 0.0))
            _outer_cyl = trimesh.creation.cylinder(radius=HORIZ_BOSS_R, height=2.0 * _boss_half_len, sections=64)
            _outer_cyl.apply_transform(_rot_y90)
            _outer_cyl.apply_translation([0.0, _y_off, HORIZ_H])
            _horiz_boss_manifolds.append(_trimesh_to_mfd(_outer_cyl))
        if HORIZ_HOLE_R > 0:
            _inner_cyl = trimesh.creation.cylinder(radius=HORIZ_HOLE_R, height=2.2 * CONE_R, sections=64)
            _inner_cyl.apply_transform(_rot_y90)
            _inner_cyl.apply_translation([0.0, _y_off, HORIZ_H])
            _horiz_hole_manifolds.append(_trimesh_to_mfd(_inner_cyl))

print(f"\nIterating to interstitial fraction (target={TARGET_INTERSTITIAL_FRACTION:.6f} × hull, tol={TOLERANCE*100:.2f}%)...")
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
combined          = None
final_vol         = None
prev_error        = float('inf')
cumulative_factor = 1.0
lo_factor = None
hi_factor = None
damping   = 1.0
for iteration in range(1, MAX_ITERS + 1):
    _dlog(f"  ── iter {iteration} ──────────────────────────────")
    manifold = build_manifold(segs, TUBE_SIDES)
    _dlog(f"    [measure_vol] reading volume from Manifold (C++)...")
    _tm = _time.perf_counter()
    measured_vol = manifold.volume()
    _dlog(f"    [measure_vol] done  {_time.perf_counter()-_tm:.4f}s  vol={measured_vol:.4f}")
    m_final = manifold
    if _vert_boss_manifold is not None:
        m_final = m_final - _vert_boss_manifold
        m_final = m_final + _vert_boss_manifold
    if _vert_hole_manifold is not None:
        m_final = m_final - _vert_hole_manifold
    for _hboss in _horiz_boss_manifolds:
        m_final = m_final - _hboss
        m_final = m_final + _hboss
    for _hhole in _horiz_hole_manifolds:
        m_final = m_final - _hhole
    combined_iter = _manifold_to_trimesh(m_final, _dlog)
    if SIMPLIFY_TARGET > 0 and len(combined_iter.faces) > SIMPLIFY_TARGET:
        _n_before = len(combined_iter.faces)
        _log(f"    [simplify] {_n_before} → ≤{SIMPLIFY_TARGET} faces...")
        _ts = _time.perf_counter()
        try:
            _tr = max(0.0, min(1.0 - 1e-9, 1.0 - SIMPLIFY_TARGET / _n_before))
            combined_iter = combined_iter.simplify_quadric_decimation(_tr)
            if not combined_iter.is_watertight:
                trimesh.repair.fill_holes(combined_iter)
            _log(f"    [simplify] done  {_time.perf_counter()-_ts:.2f}s  "
                 f"({_n_before} → {len(combined_iter.faces)} faces)")
        except Exception as _e:
            _log(f"    [simplify] failed ({_e}); using full-resolution mesh")
    _dlog(f"    [measure_hull] computing convex hull volume...")
    _tm = _time.perf_counter()
    hull_vol_iter = combined_iter.convex_hull.volume
    _dlog(f"    [measure_hull] done  {_time.perf_counter()-_tm:.4f}s  hull_vol={hull_vol_iter:.4f}")
    final_vol_iter    = combined_iter.volume
    interstitial_iter   = hull_vol_iter - final_vol_iter - hole_volume
    _target_interstitial = TARGET_INTERSTITIAL_FRACTION * hull_vol_iter - hole_volume
    error = abs(interstitial_iter - _target_interstitial) / _target_interstitial
    msg   = f"  iter {iteration}: interstitial={interstitial_iter:.4f}  haptera={final_vol_iter:.4f}  error={error*100:.3f}%"
    if error <= TOLERANCE:
        _log(msg + "  ✓ converged")
        if _ibar: _ibar.update(1)
        combined  = combined_iter
        final_vol = final_vol_iter
        break
    haptera_target = hull_vol_iter - _target_interstitial - (m_final.volume() - measured_vol) - hole_volume
    if haptera_target <= 0:
        haptera_target = BASE_VOLUME
    if haptera_target <= 0:
        _log(f"  ! TARGET_INTERSTITIAL_FRACTION ({TARGET_INTERSTITIAL_FRACTION}) × hull "
             f"({hull_vol_iter:.4f}) — target is geometrically infeasible for this tree. "
             f"Lower TARGET_INTERSTITIAL_FRACTION and re-run.")
        combined  = combined_iter
        final_vol = final_vol_iter
        break
    if interstitial_iter > _target_interstitial:
        lo_factor = cumulative_factor
    else:
        hi_factor = cumulative_factor
    error_grew = (prev_error < float('inf') and error > prev_error * 1.001)
    _dlog(f"    [correction] strategy select  prev_err={prev_error*100:.3f}%  grew={error_grew}  d={damping:.3f}  "
          f"bracket=[{lo_factor},{hi_factor}]")
    _tm = _time.perf_counter()
    if lo_factor is not None and hi_factor is not None:
        target_factor = (lo_factor + hi_factor) / 2.0
        correction    = target_factor / cumulative_factor
        strategy      = f"bisect  bracket=[{lo_factor:.4f},{hi_factor:.4f}]→{target_factor:.4f}"
    elif error_grew:
        damping    = max(damping * 0.5, 0.05)
        raw        = cubic_correction(naive_volume(segs), measured_vol, haptera_target)
        correction = 1.0 + (raw - 1.0) * damping
        correction = max(correction, 0.1)
        strategy   = f"damped_cubic  d={damping:.3f}"
    else:
        damping    = min(damping * 1.1, 1.0)
        raw        = cubic_correction(naive_volume(segs), measured_vol, haptera_target)
        correction = 1.0 + (raw - 1.0) * damping
        correction = max(correction, 0.1)
        strategy   = f"cubic  d={damping:.3f}"
    prev_error         = error
    cumulative_factor *= correction
    scale_radii(segs, correction)
    _dlog(f"    [correction] done  {_time.perf_counter()-_tm:.4f}s  f={correction:.5f}")
    _log(msg + f"  → ×{correction:.5f}  [{strategy}]")
    if _ibar: _ibar.update(1)
    if iteration == MAX_ITERS:
        _log("  ! max iterations reached")
        combined  = combined_iter
        final_vol = final_vol_iter
if _ibar: _ibar.close()


if DEBUG: print(f"[export] writing {OUTPUT}...")
_t = _time.perf_counter()
combined.export(OUTPUT)
if DEBUG: print(f"[export] done  {_time.perf_counter()-_t:.2f}s")
sys.stdout = Tee(TEXT_OUTPUT)

final_vol = combined.volume

# ── surface area ──────────────────────────────────────────────────────────────
if DEBUG: print(f"[surface_area] computing haptera surface area...")
_t = _time.perf_counter()
haptera_surface_area = combined.area
base_mask = combined.face_normals[:, 2] < -0.999
base_cap_area = float(trimesh.triangles.area(combined.triangles[base_mask]).sum()) if base_mask.any() else 0.0
area_note = "exact (includes flat base cap)"
if DEBUG: print(f"[surface_area] done  {_time.perf_counter()-_t:.2f}s  area={haptera_surface_area:.4f}")

# ── convex hull ───────────────────────────────────────────────────────────────
if DEBUG: print(f"[convex_hull] computing bounding envelope...")
_t = _time.perf_counter()
haptera_hull      = combined.convex_hull
hull_volume       = haptera_hull.volume
hull_surface_area = haptera_hull.area
hull_base_mask    = haptera_hull.face_normals[:, 2] < -0.999
hull_base_area    = float(trimesh.triangles.area(haptera_hull.triangles[hull_base_mask]).sum()) if hull_base_mask.any() else 0.0
if DEBUG: print(f"[convex_hull] done  {_time.perf_counter()-_t:.2f}s  hull_vol={hull_volume:.4f}")

# ── cone geometry ─────────────────────────────────────────────────────────────
cone_volume       = (1.0 / 3.0) * np.pi * CONE_R**2 * CONE_H
cone_lateral_area = np.pi * CONE_R * np.sqrt(CONE_R**2 + CONE_H**2)
cone_base_area    = np.pi * CONE_R**2

# ── horizontal cylinder measurements ─────────────────────────────────────────
horiz_boss_vol        = 0.0
horiz_bore_disp_vol   = 0.0
horiz_tube_sa         = 0.0
horiz_bore_lateral_sa = 0.0
if HORIZ_BOSS_R > 0 or HORIZ_HOLE_R > 0:
    _cone_r_at_h = CONE_R * (CONE_H - HORIZ_H) / CONE_H
    _y_offsets  = [0.0] if HORIZ_N == 1 else [+HORIZ_S / 2.0, -HORIZ_S / 2.0]
    _chord_lens = [2.0 * np.sqrt(max(_cone_r_at_h**2 - _yo**2, 0.0)) for _yo in _y_offsets]
    if HORIZ_BOSS_R > 0:
        _boss_solid_r_sq = HORIZ_BOSS_R**2 - (HORIZ_HOLE_R**2 if HORIZ_HOLE_R > 0 else 0.0)
        horiz_boss_vol      = sum(np.pi * _boss_solid_r_sq * _cl for _cl in _chord_lens)
        horiz_tube_sa      += sum(
            2 * np.pi * HORIZ_BOSS_R * _cl + 2 * np.pi * _boss_solid_r_sq
            for _cl in _chord_lens
        )
    if HORIZ_HOLE_R > 0:
        horiz_bore_disp_vol    = sum(np.pi * HORIZ_HOLE_R**2 * _cl for _cl in _chord_lens)
        horiz_bore_lateral_sa  = sum(2 * np.pi * HORIZ_HOLE_R * _cl for _cl in _chord_lens)
        horiz_tube_sa         += horiz_bore_lateral_sa

# ── interstitial measurements ─────────────────────────────────────────────────
interstitial_volume = hull_volume - final_vol
total_surface_area  = haptera_surface_area - base_cap_area
bore_wall_sa        = hole_lateral_area + horiz_bore_lateral_sa
external_sa         = hull_surface_area + bore_wall_sa
internal_sa         = haptera_surface_area
total_bounding_area = (external_sa - hull_base_area) + (internal_sa - base_cap_area)
sa_to_vol           = total_surface_area / interstitial_volume if interstitial_volume > 0 else 0
interstitial_haptera_only = hull_volume - final_vol - hole_volume

# ── output ────────────────────────────────────────────────────────────────────
print(f"\nExported : {OUTPUT}")
print(f"")
print(f"Parameters")
print(f"  depth                  : {DEPTH}")
print(f"  k                      : {K}")
print(f"  n_roots                : {N_ROOTS}")
print(f"  helix_turns            : {HELIX_TURNS}")
print(f"  radial_fraction        : {RADIAL_FRACTION}")
print(f"  lane_bias              : {LANE_BIAS}")
print(f"  repulsion_r_factor     : {REPULSION_R_FACTOR}")
print(f"  repulsion_strength     : {REPULSION_STRENGTH}")
print(f"  seg_len                : {SEG_LEN}")
print(f"  tube_sides             : {TUBE_SIDES}")
print(f"  steer_onset            : {STEER_ONSET}")
print(f"  steer_strength         : {STEER_STRENGTH}")
print(f"  cone_h                 : {CONE_H}")
print(f"  cone_r                 : {CONE_R}")
print(f"  vert_boss_r            : {VERT_BOSS_R}")
print(f"  vert_hole_r            : {VERT_HOLE_R}")
print(f"  horiz_boss_r           : {HORIZ_BOSS_R}")
print(f"  horiz_hole_r           : {HORIZ_HOLE_R}")
print(f"  horiz_n                : {HORIZ_N}")
print(f"  horiz_s                : {HORIZ_S}{'  (ignored — single cylinder)' if HORIZ_N == 1 else ''}")
print(f"  horiz_h                : {HORIZ_H}")
print(f"  target_interstitial_frac: {TARGET_INTERSTITIAL_FRACTION:.6f}  (fraction of hull volume)")
print(f"  base_volume (haptera)  : {BASE_VOLUME:.4f}")
print(f"")
print(f"Mesh")
print(f"  segments               : {len(segs)}")
print(f"  vertices               : {len(combined.vertices)}")
print(f"  faces                  : {len(combined.faces)}")
print(f"")
print(f"Haptera")
print(f"  volume                 : {final_vol:.4f}")
_final_target_iv = TARGET_INTERSTITIAL_FRACTION * hull_volume - hole_volume
print(f"  interstitial vol error : {abs(interstitial_haptera_only - _final_target_iv) / _final_target_iv * 100:.3f}%  (haptera-only vs fraction target)")
print(f"  surface area (w/ cap)  : {haptera_surface_area:.4f}  ({area_note})")
print(f"  wetted surface area    : {total_surface_area:.4f}  (base cap excluded; bore walls included)")
print(f"  base cap area          : {base_cap_area:.4f}")
print(f"")
if HORIZ_BOSS_R > 0 or HORIZ_HOLE_R > 0:
    _cyl_label = "single cylinder" if HORIZ_N == 1 else "both cylinders"
    print(f"Horizontal cylinders (analytical, {_cyl_label})")
    if HORIZ_BOSS_R > 0:
        print(f"  boss solid vol added   : {horiz_boss_vol:.4f}  (annular solid unioned into haptera)")
    if HORIZ_HOLE_R > 0:
        print(f"  bore displacement vol  : {horiz_bore_disp_vol:.4f}  (volume drilled out of haptera+boss)")
    print(f"  surface area           : {horiz_tube_sa:.4f}  (boss outer + end caps + bore walls)")
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
print(f"  volume (total)         : {interstitial_volume:.4f}  (hull − haptera; includes all bore voids)")
print(f"  volume (haptera only)  : {interstitial_haptera_only:.4f}  (vert bore excluded; matches calibration target)")
if hole_volume > 0:
    print(f"  vert bore volume       : {hole_volume:.4f}  (VERT_HOLE contribution)")
if HORIZ_HOLE_R > 0:
    print(f"  horiz bore volume      : {horiz_bore_disp_vol:.4f}  (HORIZ_HOLE contribution, analytical)")
print(f"  bore wall area         : {bore_wall_sa:.4f}  (vert + horiz bore lateral walls, analytical)")
print(f"  external surface area  : {external_sa:.4f}  (hull surface + all bore walls)")
print(f"  internal surface area  : {internal_sa:.4f}  (haptera mesh; all bore walls included)")
print(f"  total bounding area    : {total_bounding_area:.4f}  (external + internal; ground-contact faces excluded)")
print(f"  SA / volume ratio      : {sa_to_vol:.4f}  (complexity index using total wetted SA)")

sys.stdout.close()
