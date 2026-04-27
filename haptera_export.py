import numpy as np
import trimesh
import sys
import os
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
DEBUG       = False  # True = print per-function timing diagnostics
INTERTWINED = False  # True = roots spiral down the cone on helical lanes (see intertwining parameters below)

DEPTH  = 9 #Number of nodes
K      = 2 #Number of branches per node

TUBE_SIDES = 10

# ── convergence ───────────────────────────────────────────────────────────────
TOLERANCE = 0.001 #0.1%
MAX_ITERS = 10

# ── radius caching (speeds up re-runs) ────────────────────────────────────────
# The convergence loop multiplies every tube radius by a correction factor each
# iteration.  When the same configuration is run repeatedly, the cumulative
# correction factor is roughly the same too, so caching it lets a re-run start
# at (or very near) the converged radius and skip most iterations.
USE_CACHED_FACTOR  = True   # read CACHE_FILE on startup if it exists and its config matches
SAVE_CACHED_FACTOR = True   # write CACHE_FILE after the iteration loop
INITIAL_FACTOR     = None   # manual multiplier override, e.g. 0.873.  Takes precedence over the cache.  None = use cache or 1.0.

# ── cone geometry ─────────────────────────────────────────────────────────────
# Defined here so targets below can reference cone volume.
CONE_H = 130
CONE_R = 130
HORIZ_BOSS_R = 9   # outer radius of horizontal bossed cylinders that bisect the haptera; set to 0 to disable
HORIZ_HOLE_R = 7   # inner bore radius of horizontal bossed cylinders; set to 0 to disable
HORIZ_N      = 2   # number of horizontal cylinders: 1 = single centered cylinder, 2 = two at ±HORIZ_S/2
HORIZ_S      = 100  # center-to-center spacing in Y between the two cylinders (ignored when HORIZ_N = 1)
HORIZ_H      = 30  # height above the haptera base for horizontal cylinder centers

SIMPLIFY_TARGET = 1000000  # target face count after QEM decimation (e.g. 50000); 0 = disabled

N_ROOTS        = 40
SEG_LEN        = CONE_H / DEPTH  # scales with cone height so branches traverse the full cone at any depth
REF_ROOT_R     = 5
STEER_ONSET    = 0.90  # was 0.55; lower values pull branches inward too early, producing a structure ~55-80% of CONE_R
STEER_STRENGTH = 1.1
TORSION        = 0.6  # radians of extra branching-plane rotation per depth level (0 = no twist)

# ── intertwining (only used when INTERTWINED = True) ──────────────────────────
# Each root traces a helical path from apex to base, steered by a lane-specific
# parametric helix.  NO_INTERSECT (below) still enforces non-overlap between
# lanes, so the result is a set of intertwined, non-intersecting strands.
HELIX_TURNS     = 0.5   # full rotations around the cone axis from apex to base
RADIAL_FRACTION = 0.65  # helix radius as a fraction of the available cone radius
LANE_BIAS       = 0.10  # strength each branch is pulled toward its helical waypoint

# ── inter-haptera collision avoidance ─────────────────────────────────────────
# When NO_INTERSECT is True, each branch checks every already-placed segment
# and deflects laterally to avoid overlap.  Branches that still intersect after
# REPEL_RETRIES attempts are silently discarded.
NO_INTERSECT   = False  # True = haptera steer around each other
REPEL_ONSET    = 1.5    # repulsion zone starts at this multiple of (r_a + r_b)
REPEL_STRENGTH = 2.5    # lateral deflection weight at full proximity
REPEL_RETRIES  = 12     # direction-correction attempts before discarding a branch

_CONE_VOLUME = (1.0 / 3.0) * np.pi * CONE_R**2 * CONE_H
_NOMINAL_VOLUME = 2 * N_ROOTS * np.pi * REF_ROOT_R**2 * SEG_LEN  # original calibration

# ── targets ───────────────────────────────────────────────────────────────────
# Target interstitial as a fraction of the actual convex hull volume each iteration.
TARGET_INTERSTITIAL_FRACTION = 0.6503867

# Haptera mesh volume fallback (used when dynamic target is infeasible):
BASE_VOLUME = _CONE_VOLUME * (1 - TARGET_INTERSTITIAL_FRACTION)

_OUTPUT_BASE = "haptera_{}d{}_k{}_r{}_h{}_f{}".format(
                "intertwined_" if INTERTWINED else "",
                DEPTH, K, CONE_R, CONE_H,
                round(TARGET_INTERSTITIAL_FRACTION * 1000))
_OUTPUT_DIR = os.environ.get('HAPTERA_OUTPUT_DIR', '.')
if _OUTPUT_DIR != '.':
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
def _outpath(name):
    return os.path.join(_OUTPUT_DIR, name) if _OUTPUT_DIR != '.' else name
OUTPUT      = _outpath(f"{_OUTPUT_BASE}.stl")
TEXT_OUTPUT = _outpath(f"{_OUTPUT_BASE}.txt")
CACHE_FILE  = _outpath(f"{_OUTPUT_BASE}.cache.txt")


# ── PRNG ──────────────────────────────────────────────────────────────────────
def make_rng(seed=54321):
    """Create a deterministic pseudo-random number generator using a linear congruential
    algorithm. Returns a closure that yields floats in [0, 1) on each call. Using a
    fixed seed ensures the branching tree is reproducible across runs."""
    state = [seed & 0xFFFFFFFF]
    def rng():
        state[0] = (1664525 * state[0] + 1013904223) & 0xFFFFFFFF
        return state[0] / 4294967296
    return rng

# ── geometry helpers ──────────────────────────────────────────────────────────
def cone_contains(x, y, z, r=0):
    """Return True if the point (x, y, z) is inside the cone, accounting for tube
    radius r. The skeleton centre must remain at least r from the cone wall so the
    full tube cross-section stays within the cone boundary."""
    # r is the tube radius: the skeleton must stay at least r inside the cone wall
    # so the tube surface (skeleton ± r) fits within the cone.
    r_pos  = np.sqrt(x*x + y*y)
    r_eff  = max((CONE_R / CONE_H) * (CONE_H - z) - r, 0.0)
    return (z >= 0) and (z <= CONE_H) and (r_pos <= r_eff)

def wall_proximity(x, y, z, r=0):
    """Return the radial position of the skeleton as a fraction of the available
    wall radius (cone wall radius minus tube radius r). A value of 0 means the
    skeleton is on the axis; 1.0 means the tube surface is touching the cone wall;
    values > 1.0 mean the tube is outside the cone."""
    # Returns skeleton position as a fraction of the effective wall radius (r_wall - r_tube).
    # At prox=1.0 the tube surface is exactly touching the cone wall.
    r_pos  = np.sqrt(x*x + y*y)
    r_eff  = (CONE_R / CONE_H) * (CONE_H - z) - r
    if r_eff <= 0:
        return 2.0
    return r_pos / r_eff

def inward_direction(x, y):
    """Return a unit vector pointing inward along the cone wall normal from the
    point (x, y). Accounts for the cone's slope so that branches redirected by
    this vector will travel along the wall surface toward the apex rather than
    purely horizontally. Returns a zero vector when already on the axis."""
    r = np.sqrt(x*x + y*y)
    if r < 1e-9:
        return np.array([0.0, 0.0, 0.0])
    # True inward normal to the cone wall: gradient of (r_pos + slope*z - CONE_R) points outward,
    # so inward = [-x/r, -y/r, -slope] normalised.  The horizontal-only version ignored the cone
    # slope and failed to redirect branches that were approaching the narrowing apex.
    slope = CONE_R / CONE_H
    raw = np.array([-x / r, -y / r, -slope])
    return raw / np.linalg.norm(raw)

def helix_target_direction(ox, oy, oz, lane, n_lanes, seg_len):
    """Return a unit vector from (ox, oy, oz) toward the next waypoint on lane's
    parametric helix. Each lane traces a helix from the apex (z=CONE_H) to the
    base (z=0), completing HELIX_TURNS full rotations. The helix radius at height
    z is RADIAL_FRACTION × the cone radius at that height, so the helix always
    sits comfortably inside the cone. Used only when INTERTWINED is True."""
    z_target = max(0.0, oz - seg_len)
    z_frac_t = max(0.0, min(1.0, (CONE_H - z_target) / CONE_H))
    phi_t    = 2.0 * np.pi * lane / n_lanes + HELIX_TURNS * 2.0 * np.pi * z_frac_t
    cone_r_t = (CONE_R / CONE_H) * (CONE_H - z_target)
    tx = RADIAL_FRACTION * cone_r_t * np.cos(phi_t) - ox
    ty = RADIAL_FRACTION * cone_r_t * np.sin(phi_t) - oy
    tz = z_target - oz
    length = np.sqrt(tx*tx + ty*ty + tz*tz)
    if length < 1e-9:
        return np.array([0.0, 0.0, -1.0])
    return np.array([tx / length, ty / length, tz / length])

def _seg_seg_closest(p1, p2, p3, p4):
    """Minimum centre-to-centre distance between finite segments p1→p2 and p3→p4.

    Returns (distance, point_on_p1p2, point_on_p3p4).
    Implements Ericson (2005) 'Real-Time Collision Detection' §5.1.9.
    """
    d1  = p2 - p1
    d2  = p4 - p3
    r   = p1 - p3
    a   = np.dot(d1, d1)   # squared length of seg 1
    e   = np.dot(d2, d2)   # squared length of seg 2
    f   = np.dot(d2, r)
    EPS = 1e-10

    if a <= EPS and e <= EPS:          # both point-degenerate
        return float(np.sqrt(max(np.dot(r, r), 0.0))), p1.copy(), p3.copy()

    if a <= EPS:                       # seg 1 degenerate
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = np.dot(d1, r)
        if e <= EPS:                   # seg 2 degenerate
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b     = np.dot(d1, d2)
            denom = a * e - b * b
            s     = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) if abs(denom) > EPS else 0.0
            t     = float(b * s + f) / float(e)
            if t < 0.0:
                t = 0.0;  s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0;  s = float(np.clip((b - c) / a, 0.0, 1.0))

    c1   = p1 + s * d1
    c2   = p3 + t * d2
    diff = c1 - c2
    return float(np.sqrt(max(np.dot(diff, diff), 0.0))), c1, c2


def steer(dx, dy, dz, ox, oy, oz, r=0):
    """Blend the branch direction (dx, dy, dz) toward the cone-wall inward normal
    when the branch is close to the wall. Steering activates beyond STEER_ONSET
    proximity and ramps up quadratically to STEER_STRENGTH at the wall, preventing
    branches from escaping the cone boundary."""
    prox = wall_proximity(ox, oy, oz, r)
    if prox < STEER_ONSET:
        return dx, dy, dz
    t = min((prox - STEER_ONSET) / (1.0 - STEER_ONSET), 1.0)
    weight = STEER_STRENGTH * t * t
    inward = inward_direction(ox, oy)
    ndx = dx + inward[0] * weight
    ndy = dy + inward[1] * weight
    ndz = dz + inward[2] * weight  # was fixed at dz; now uses cone-wall z-component
    nl  = np.sqrt(ndx*ndx + ndy*ndy + ndz*ndz)
    return ndx/nl, ndy/nl, ndz/nl

def _clip_to_cone(ox, oy, oz, dx, dy, dz, seg_len, r):
    """Compute endpoint (ox,oy,oz)+(dx,dy,dz)*seg_len and nudge it back inside
    the cone (up to 6 inward steps of 25 % seg_len each).  Returns (ex, ey, ez)
    or None when the endpoint cannot be recovered."""
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
        else:
            return None
    return ex, ey, ez


def grow(ox, oy, oz, dx, dy, dz, r, depth, k, seg_len, rng, max_depth, out,
         lane=None, n_lanes=None):
    """Recursively grow a branching tree skeleton inside the cone. Each call extends
    a single segment from origin (ox, oy, oz) in direction (dx, dy, dz) with tube
    radius r. On reaching depth 0 the branch terminates; otherwise k child branches
    are spawned at evenly-spaced azimuths with a random perturbation and TORSION
    twist per depth level. Segments that would leave the cone are nudged inward or
    discarded. Results are appended to the out list as dicts with keys start, end,
    r, and level.

    When NO_INTERSECT is True each proposed segment is checked against every
    already-placed segment.  The direction is deflected laterally (forward component
    preserved) away from any tube that is closer than REPEL_ONSET × (r_a + r_b).
    Up to REPEL_RETRIES correction steps are attempted; if the segment still
    hard-intersects after all retries it is silently discarded.

    When INTERTWINED is True and lane is provided, an additional helical-lane
    bias is blended into the direction so each root traces its own spiral from
    apex to base; NO_INTERSECT still prevents overlap between lanes."""
    if not cone_contains(ox, oy, oz, r):
        return
    dx, dy, dz = steer(dx, dy, dz, ox, oy, oz, r)
    if INTERTWINED and lane is not None and n_lanes is not None:
        helix_dir = helix_target_direction(ox, oy, oz, lane, n_lanes, seg_len)
        hdx = dx + LANE_BIAS * helix_dir[0]
        hdy = dy + LANE_BIAS * helix_dir[1]
        hdz = dz + LANE_BIAS * helix_dir[2]
        hl  = np.sqrt(hdx*hdx + hdy*hdy + hdz*hdz)
        if hl > 1e-9:
            dx, dy, dz = hdx/hl, hdy/hl, hdz/hl
    if dz < -1e-6:
        seg_len = min(oz / ((-dz) * (depth + 1)), CONE_H)

    # Initial endpoint (may be nudged by cone clip)
    clipped = _clip_to_cone(ox, oy, oz, dx, dy, dz, seg_len, r)
    if clipped is None:
        return
    ex, ey, ez = clipped

    # ── inter-haptera repulsion ───────────────────────────────────────────────
    if NO_INTERSECT and out:
        origin = np.array([ox, oy, oz])
        ndx, ndy, ndz = (ex - ox) / seg_len, (ey - oy) / seg_len, (ez - oz) / seg_len
        nl = np.sqrt(ndx*ndx + ndy*ndy + ndz*ndz)
        if nl > 1e-9:
            ndx, ndy, ndz = ndx/nl, ndy/nl, ndz/nl

        accepted = False
        for _retry in range(REPEL_RETRIES):
            end_pt = np.array([ox + ndx * seg_len,
                               oy + ndy * seg_len,
                               oz + ndz * seg_len])
            repel        = np.zeros(3)
            hard_overlap = False

            for seg in out:
                # Skip segments that share this junction — the touching endpoint
                # always gives distance 0 and would trigger spurious repulsion.
                if (np.linalg.norm(seg['end']   - origin) < 1e-3 or
                        np.linalg.norm(seg['start'] - origin) < 1e-3):
                    continue

                min_clear = r + seg['r']
                dist, c_new, c_exist = _seg_seg_closest(
                    origin, end_pt, seg['start'], seg['end'])

                if dist >= min_clear * REPEL_ONSET:
                    continue  # comfortably clear

                if dist < min_clear:
                    hard_overlap = True

                # Quadratic weight — stronger as gap shrinks toward zero
                prox_t = 1.0 - dist / (min_clear * REPEL_ONSET)
                weight = REPEL_STRENGTH * prox_t * prox_t

                # Repulsion direction: away from the nearest point on the
                # existing segment.  Project out the forward component so the
                # branch steers sideways rather than reversing.
                repel_dir = c_new - c_exist
                rd_len    = np.linalg.norm(repel_dir)
                if rd_len < 1e-6:
                    # Perfectly coincident closest points — pick a lateral perp
                    fwd  = np.array([ndx, ndy, ndz])
                    perp = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
                    pl   = np.linalg.norm(perp)
                    if pl < 1e-6:
                        perp = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
                        pl   = np.linalg.norm(perp)
                    repel_dir = perp / max(pl, 1e-9)
                else:
                    repel_dir /= rd_len

                # Remove the forward component → purely lateral deflection
                fwd       = np.array([ndx, ndy, ndz])
                repel_dir -= np.dot(repel_dir, fwd) * fwd
                rl2        = np.linalg.norm(repel_dir)
                if rl2 > 1e-9:
                    repel_dir /= rl2

                repel += repel_dir * weight

            if not hard_overlap:
                # No tube centres closer than their combined radii — accept
                ex, ey, ez = end_pt
                accepted = True
                break

            # Apply accumulated lateral push, re-normalise, re-steer, re-clip
            ndx += repel[0];  ndy += repel[1];  ndz += repel[2]
            nl   = np.sqrt(ndx*ndx + ndy*ndy + ndz*ndz)
            if nl < 1e-9:
                ndx, ndy, ndz = dx, dy, dz   # safety reset to original direction
            else:
                ndx, ndy, ndz = ndx/nl, ndy/nl, ndz/nl

            ndx, ndy, ndz = steer(ndx, ndy, ndz, ox, oy, oz, r)

            clipped = _clip_to_cone(ox, oy, oz, ndx, ndy, ndz, seg_len, r)
            if clipped is None:
                break    # repulsion pushed into wall — fall through to binary search
            ex, ey, ez = clipped

        if not accepted:
            # Direction retries exhausted.  Binary-search for the longest
            # sub-segment in the current direction that clears all tubes.
            # This guarantees a segment is placed rather than dropped.
            best_end = None
            lo_t, hi_t = 0.0, 1.0
            for _ in range(14):   # 2^-14 ≈ 0.006 % of seg_len resolution
                mid_t    = (lo_t + hi_t) / 2.0
                test_end = origin + np.array([ndx, ndy, ndz]) * (seg_len * mid_t)
                if not cone_contains(test_end[0], test_end[1], test_end[2], r):
                    hi_t = mid_t
                    continue
                overlap = False
                for seg in out:
                    if (np.linalg.norm(seg['end']   - origin) < 1e-3 or
                            np.linalg.norm(seg['start'] - origin) < 1e-3):
                        continue
                    dist, _, _ = _seg_seg_closest(origin, test_end, seg['start'], seg['end'])
                    if dist < r + seg['r']:
                        overlap = True
                        break
                if overlap:
                    hi_t = mid_t
                else:
                    best_end = test_end.copy()
                    lo_t     = mid_t
            if best_end is None:
                return   # truly no room — discard only as last resort
            ex, ey, ez = best_end[0], best_end[1], best_end[2]

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
        if INTERTWINED and lane is not None and n_lanes is not None:
            # Sibling children target neighbouring helical waypoints instead of
            # the parent's exact lane, so fan-outs don't all chase the same helix.
            sub_offset = (i / max(k - 1, 1) - 0.5) / n_lanes
            child_lane = lane + sub_offset
        else:
            child_lane = None
        grow(ex, ey, ez, ndx/nl, ndy/nl, ndz/nl,
             r, depth-1, k, seg_len, rng, max_depth, out,
             lane=child_lane, n_lanes=n_lanes)

# ── volume helpers ────────────────────────────────────────────────────────────
def naive_volume(segs):
    """Compute the total cylinder volume of all segments, ignoring overlap at
    junctions. Used as a fast upper-bound estimate and as the V_naive term in
    the cubic volume correction model."""
    return sum(np.pi * s['r']**2 * np.linalg.norm(s['end'] - s['start'])
               for s in segs)

def overlap_volume(segs):
    """Estimate the double-counted volume at branch junctions. At each shared
    endpoint, every touching segment contributes a hemisphere (2/3 π r³), so two
    overlapping hemispheres approximate the excess counted by the naive cylinder sum.
    Returns the total estimated overlap to subtract from naive_volume."""
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
    """Multiply every segment's tube radius by factor in-place. Called each
    iteration to converge the haptera volume toward the target."""
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


# ── tube mesh builder ─────────────────────────────────────────────────────────
def tube_mesh(start, end, radius, sides=8):
    """Build a triangulated cylinder mesh aligned from start to end with the given
    radius and number of lateral facets. Computes the rotation matrix that maps the
    default Z-axis cylinder onto the segment direction using Rodrigues' formula.
    Returns None if the segment is degenerate (zero length)."""
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
    """Convert a list of skeleton segments into trimesh geometry. For each segment a
    cylinder mesh is created via tube_mesh, and an icosphere is placed at every
    endpoint. The spheres fill the junction gaps between adjacent cylinders, preventing
    the coincident flat end-caps that cause non-watertight booleans. Returns the list
    of all component meshes ready for union."""
    meshes = []
    for s in segs:
        m = tube_mesh(s['start'], s['end'], s['r'], sides=sides)
        if m is not None:
            meshes.append(m)
    # Sphere caps at every endpoint prevent the degenerate coplanar-cap geometry
    # that causes non-watertight output: adjacent cylinders share exact endpoints so
    # their flat end-caps are coincident, which is a degenerate case for boolean CSG.
    # A sphere at each junction fills the gap and smooths the connection; at leaf tips
    # it closes the open end cleanly.
    endpoints = {}  # rounded-point tuple → max radius of touching segments
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
    Uses to_mesh64() for float64 precision: float32 quantization at radii near
    CONE_R = 130 mm has 1 ULP ≈ 1.5e-5 mm, which is large enough to leave
    nominally-coincident vertices at distinct positions, breaking watertightness
    after the trimesh round-trip even when manifold3d's topology was correct.
    """
    import time
    _log("    [extract_mesh] converting Manifold to trimesh...")
    t0 = time.perf_counter()
    r  = manifold.to_mesh64()
    # np.array(..., copy=True) materializes writable arrays. The manifold3d buffers
    # are read-only views; passing them straight to trimesh would propagate that
    # flag and cause simplify_quadric_decimation (Cython) to fail with
    # "buffer source array is read-only".
    result = trimesh.Trimesh(
        vertices=np.array(r.vert_properties[:, :3], copy=True),  # slice XYZ; newer manifold3d appends extra property channels
        faces=np.array(r.tri_verts, copy=True),
        process=False,
        validate=False,
    )
    _log(f"    [extract_mesh] done  {time.perf_counter()-t0:.2f}s  "
         f"({len(result.vertices)} verts, {len(result.faces)} faces)")
    return result



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
def build_segments(depth, k):
    """Generate the full skeleton of haptera segments for the given branching depth
    and branching factor k. N_ROOTS evenly-spaced root branches are started from
    near the cone apex, each grown recursively with grow(). After tree generation
    the radii are rescaled so the naive volume approximation matches BASE_VOLUME,
    providing a good starting point for the iterative volume convergence loop.

    When INTERTWINED is True each root's initial direction points at its first
    helical waypoint and the lane is passed through grow() so children inherit
    the helical bias; NO_INTERSECT still enforces non-overlap between lanes."""
    root_r = np.sqrt(BASE_VOLUME / (N_ROOTS * np.pi * SEG_LEN * (depth + 1)))
    rng    = make_rng(54321)
    segs   = []
    for i in range(N_ROOTS):
        ox, oy, oz = 0.0, 0.0, CONE_H - 0.05
        if INTERTWINED:
            init_dir = helix_target_direction(ox, oy, oz, i, N_ROOTS, SEG_LEN)
            rdx, rdy, rdz = init_dir[0], init_dir[1], init_dir[2]
            lane_arg, n_lanes_arg = i, N_ROOTS
        else:
            a  = (2 * np.pi * i / N_ROOTS) + 0.35
            rdx = np.cos(a) * 0.45
            rdy = np.sin(a) * 0.45
            rdz = -1.0
            rl  = np.sqrt(rdx*rdx + rdy*rdy + rdz*rdz)
            rdx, rdy, rdz = rdx/rl, rdy/rl, rdz/rl
            lane_arg, n_lanes_arg = None, None
        grow(ox, oy, oz, rdx, rdy, rdz,
             root_r, depth, k, SEG_LEN, rng, depth, segs,
             lane=lane_arg, n_lanes=n_lanes_arg)
    nv = naive_volume(segs)
    cv = nv - overlap_volume(segs)
    # cv is almost always ≤ 0 for this geometry (r > seg_len means junction
    # hemispheres exceed cylinder volume), so fall back to naive_volume.
    ref = cv if cv > 0 else nv
    if ref > 0:
        scale_radii(segs, np.sqrt(BASE_VOLUME / ref))
    return segs

# ── radius-multiplier cache ───────────────────────────────────────────────────
def _current_factor_config():
    """Return the dict of config keys that determine the converged radius
    multiplier.  Used both to validate cached values and to write the cache so
    a future run can detect a stale cache when geometry has changed."""
    return {
        'DEPTH': DEPTH,
        'K': K,
        'N_ROOTS': N_ROOTS,
        'CONE_H': CONE_H,
        'CONE_R': CONE_R,
        'SEG_LEN': SEG_LEN,
        'STEER_ONSET': STEER_ONSET,
        'STEER_STRENGTH': STEER_STRENGTH,
        'TORSION': TORSION,
        'INTERTWINED': INTERTWINED,
        'HELIX_TURNS': HELIX_TURNS,
        'RADIAL_FRACTION': RADIAL_FRACTION,
        'LANE_BIAS': LANE_BIAS,
        'NO_INTERSECT': NO_INTERSECT,
        'REPEL_ONSET': REPEL_ONSET,
        'REPEL_STRENGTH': REPEL_STRENGTH,
        'REPEL_RETRIES': REPEL_RETRIES,
        'TARGET_INTERSTITIAL_FRACTION': TARGET_INTERSTITIAL_FRACTION,
        'BASE_VOLUME': BASE_VOLUME,
        'HORIZ_BOSS_R': HORIZ_BOSS_R,
        'HORIZ_HOLE_R': HORIZ_HOLE_R,
        'HORIZ_N': HORIZ_N,
        'HORIZ_S': HORIZ_S,
        'HORIZ_H': HORIZ_H,
    }

def _parse_cache_value(s):
    s = s.strip()
    if s == 'True':  return True
    if s == 'False': return False
    if s == 'None':  return None
    try: return int(s)
    except ValueError: pass
    try: return float(s)
    except ValueError: pass
    return s

def _load_cached_factor(path, current_config):
    """Read a cached multiplier from path.  Returns the multiplier if the cached
    config matches current_config, else None.  Returns None when the file is
    absent or unreadable."""
    if not os.path.exists(path):
        return None
    cached = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, val = line.partition('=')
                cached[key.strip()] = _parse_cache_value(val)
    except OSError as e:
        print(f"  ! cache read failed ({e}); starting from scratch")
        return None
    multiplier = cached.get('multiplier')
    if multiplier is None:
        return None
    mismatches = []
    for k, v in current_config.items():
        if k not in cached:
            mismatches.append(f"{k} (missing from cache)")
            continue
        cv = cached[k]
        if isinstance(v, float) or isinstance(cv, float):
            try:
                if abs(float(v) - float(cv)) > 1e-9:
                    mismatches.append(f"{k}: cache={cv} current={v}")
            except (TypeError, ValueError):
                mismatches.append(f"{k}: cache={cv!r} current={v!r}")
        elif cv != v:
            mismatches.append(f"{k}: cache={cv} current={v}")
    if mismatches:
        print(f"  ! cache config mismatch — ignoring cached multiplier ({len(mismatches)} differing keys):")
        for m in mismatches[:5]:
            print(f"      {m}")
        if len(mismatches) > 5:
            print(f"      ...and {len(mismatches) - 5} more")
        return None
    return float(multiplier)

def _save_cached_factor(path, current_config, multiplier, final_root_radius,
                        interstitial_fraction, error, converged):
    """Write the converged multiplier and the config that produced it to path."""
    try:
        with open(path, 'w') as f:
            f.write("# auto-generated by haptera_export.py — safe to delete.\n")
            f.write("# 'multiplier' is the cumulative tube-radius scale factor that produced the\n")
            f.write("# converged interstitial fraction below.  Re-runs with USE_CACHED_FACTOR=True\n")
            f.write("# read it back and apply it before iteration starts.\n")
            f.write("# To use it manually instead, copy its value into INITIAL_FACTOR at the top of the script.\n")
            f.write(f"multiplier={multiplier:.10f}\n")
            f.write(f"final_root_radius={final_root_radius:.10f}\n")
            f.write(f"final_interstitial_fraction={interstitial_fraction:.10f}\n")
            f.write(f"final_error={error:.10f}\n")
            f.write(f"converged={converged}\n")
            for k, v in current_config.items():
                f.write(f"{k}={v}\n")
    except OSError as e:
        print(f"  ! cache write failed ({e})")

# ── main ──────────────────────────────────────────────────────────────────────
import time as _time
print(f"Building segments (depth={DEPTH}, k={K})...")
_t = _time.perf_counter()
segs = build_segments(DEPTH, K)
if DEBUG:
    print(f"  [build_segments] done  {_time.perf_counter()-_t:.2f}s  {len(segs)} segments")
else:
    print(f"  {len(segs)} segments generated")

# ── apply initial radius multiplier (manual or cached) ───────────────────────
# Manual INITIAL_FACTOR takes precedence so the user can paste a value from a
# previous run's log; otherwise USE_CACHED_FACTOR reads CACHE_FILE if present
# and config-compatible.  Either path skips most of the iteration loop on a
# warm re-run.
_initial_factor = INITIAL_FACTOR
_factor_source  = "INITIAL_FACTOR"
if _initial_factor is None and USE_CACHED_FACTOR:
    _cached = _load_cached_factor(CACHE_FILE, _current_factor_config())
    if _cached is not None:
        _initial_factor = _cached
        _factor_source  = f"cache ({CACHE_FILE})"
if _initial_factor is not None and abs(_initial_factor - 1.0) > 1e-9:
    print(f"  Applying initial radius multiplier ×{_initial_factor:.5f} (from {_factor_source})")
    scale_radii(segs, _initial_factor)
else:
    _initial_factor = 1.0

# ── boss/hole manifolds (pre-built once, reused each iteration) ───────────────
from manifold3d import Manifold as _Manifold, Mesh as _MfdMesh

def _trimesh_to_mfd(m):
    """Convert a trimesh.Trimesh to a manifold3d Manifold object by copying vertex
    and face arrays. Used to convert the pre-built boss and hole cylinders into the
    Manifold CSG representation for reuse across iterations."""
    return _Manifold(mesh=_MfdMesh(
        vert_properties=m.vertices.astype(np.float32),
        tri_verts=m.faces.astype(np.uint32),
    ))

def _drop_sliver_bodies(mesh, vol_threshold=1e-3):
    """Remove disconnected components whose absolute volume is below the
    threshold (mm³).  manifold3d boolean ops occasionally leave 1–few-triangle
    fragments at the join between operands; they show up as separate bodies
    in `mesh.body_count` and contribute non-manifold edges relative to the
    main solid.  Volumes are computed per-component with the divergence
    theorem so we don't pay for `mesh.split` (which would copy the geometry).
    The largest body is always kept, even if it is below threshold, so we
    never accidentally delete everything."""
    if len(mesh.faces) == 0:
        return 0
    labels = trimesh.graph.connected_component_labels(
        mesh.face_adjacency, node_count=len(mesh.faces))
    unique = np.unique(labels)
    if len(unique) <= 1:
        return 0
    # Per-face signed volume contribution: (v0 · (v1 × v2)) / 6.
    tri        = mesh.triangles
    face_vols  = np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    body_vols  = {int(l): abs(float(face_vols[labels == l].sum())) for l in unique}
    keep       = [l for l, v in body_vols.items() if v >= vol_threshold]
    if not keep:
        keep = [max(body_vols, key=body_vols.get)]
    if len(keep) == len(unique):
        return 0
    dropped = len(unique) - len(keep)
    keep_mask = np.isin(labels, np.array(keep))
    mesh.update_faces(keep_mask)
    mesh.remove_unreferenced_vertices()
    return dropped

def _delete_nonmanifold_faces(mesh, max_fraction=0.1):
    """Delete only faces touching a strictly-non-manifold edge (≥3 faces share).
    Boundary edges (1 face) are left alone — `fill_holes` patches those, while
    deleting their faces just opens larger boundaries that `fill_holes` then
    fails to bridge (the destructive runaway we hit before).

    Returns the number of faces deleted, or -1 if the deletion would exceed
    `max_fraction` of the mesh and was refused for safety."""
    if len(mesh.faces) == 0:
        return 0
    edges     = mesh.edges_sorted                                                  # shape (3F, 2)
    edge_keys = (edges[:, 0].astype(np.int64) << 32) | edges[:, 1].astype(np.int64)
    _, inverse, counts = np.unique(edge_keys, return_inverse=True, return_counts=True)
    edge_is_nm   = counts[inverse] >= 3                                            # shape (3F,)
    face_has_nm  = edge_is_nm.reshape(-1, 3).any(axis=1)                           # shape (F,)
    n_to_delete  = int(face_has_nm.sum())
    if n_to_delete == 0:
        return 0
    if n_to_delete > max_fraction * len(mesh.faces):
        return -1
    mesh.update_faces(~face_has_nm)
    mesh.remove_unreferenced_vertices()
    return n_to_delete

def _fill_loop_holes(mesh):
    """Close arbitrary boundary loops by ear-clipping triangulation.

    `trimesh.repair.fill_holes` only handles 3- and 4-edge holes; CSG output
    from boolean ops can leave longer boundary loops where two surfaces graze
    without quite intersecting cleanly.  This walks the directed boundary
    edges (each appears in exactly one face, giving a consistent traversal
    direction), groups them into closed loops, and seals each loop with ear
    clipping that only commits a triangle when its chord-edge isn't already
    in the mesh — preventing the "stack on existing triangle → 3-faced edge"
    failure that plagued naive fan triangulation.  Loops with no manifold-
    safe ear at any rotation fall back to a unique-offset centroid vertex.

    The walker tracks *used boundary edges* rather than visited vertices, so
    loops that pinch through a shared vertex are still recovered (a global
    visited-vertex set would falsely terminate the second loop's walk as soon
    as it hit the shared vertex).

    Returns the number of triangles added."""
    if mesh.is_watertight or len(mesh.faces) == 0:
        return 0
    edges        = np.asarray(mesh.edges)         # (3F, 2) directed
    edges_sorted = mesh.edges_sorted              # (3F, 2)
    if len(edges) == 0:
        return 0
    edge_keys = (edges_sorted[:, 0].astype(np.int64) << 32) | edges_sorted[:, 1].astype(np.int64)
    _, inverse, counts = np.unique(edge_keys, return_inverse=True, return_counts=True)
    boundary_mask = counts[inverse] == 1
    if not boundary_mask.any():
        return 0
    boundary_edges = edges[boundary_mask]         # directed boundary edges only
    n_edges        = len(boundary_edges)

    # Map source vertex → list of (dest_vertex, boundary_edge_idx).
    next_map = {}
    for idx in range(n_edges):
        v0, v1 = int(boundary_edges[idx, 0]), int(boundary_edges[idx, 1])
        next_map.setdefault(v0, []).append((v1, idx))

    used     = np.zeros(n_edges, dtype=bool)
    loops    = []
    max_walk = n_edges + 4
    for start_idx in range(n_edges):
        if used[start_idx]:
            continue
        start_v = int(boundary_edges[start_idx, 0])
        used[start_idx] = True
        loop    = [start_v, int(boundary_edges[start_idx, 1])]
        current = loop[-1]
        success = False
        for _ in range(max_walk):
            if current == start_v:
                loop.pop()        # drop the duplicate closing vertex
                success = True
                break
            chosen = -1
            for ni, (_nxt_v, nxt_idx) in enumerate(next_map.get(current, ())):
                if not used[nxt_idx]:
                    chosen = ni
                    break
            if chosen < 0:
                break
            nxt_v, nxt_idx = next_map[current][chosen]
            used[nxt_idx]  = True
            current        = nxt_v
            loop.append(current)
        if success and len(loop) >= 3:
            loops.append(loop)

    if not loops:
        return 0

    # Ear-clipping triangulation.  Each clipped ear uses 3 consecutive loop
    # vertices (v_prev, v_cur, v_next); the new face's chord-edge is
    # (v_prev, v_next).  If that chord already exists in the mesh the chord
    # would become 3-faced (existing triangle + new triangle, since the
    # boundary edges v_prev→v_cur and v_cur→v_next get cancelled by the new
    # face's reverse edges).  So we only clip ears whose chord is NOT in the
    # existing edge set.  Ear clipping has *much* better local options than
    # a single fixed-apex fan: we evaluate every candidate ear at every
    # iteration, so as long as *some* manifold-safe ear exists we'll find
    # it.  Falls back to a unique-offset centroid vertex if a loop has no
    # safe ear at any rotation (rare but possible for fully-interlocked
    # CSG residue).
    existing_edges = set()
    for v0, v1 in edges_sorted:
        existing_edges.add((int(v0), int(v1)))

    def _is_safe_ear(loop, i):
        n = len(loop)
        v_prev = loop[(i - 1) % n]
        v_next = loop[(i + 1) % n]
        if v_prev == v_next:
            return False
        key = (v_prev, v_next) if v_prev < v_next else (v_next, v_prev)
        return key not in existing_edges

    # Pre-allocate new-faces buffer for the worst case: a loop of size n
    # contributes n-2 triangles via ear clipping, or n via centroid fallback.
    n_tris_max = sum(max(len(l), 0) for l in loops)
    if n_tris_max == 0:
        return 0
    new_faces  = np.empty((n_tris_max, 3), dtype=mesh.faces.dtype)
    new_verts  = []   # any centroid-fallback vertices we add
    fi         = 0
    n_verts0   = len(mesh.vertices)
    verts      = mesh.vertices
    for li, loop in enumerate(loops):
        if len(loop) < 3:
            continue
        # Ear-clip in place.  Each iteration picks the first safe ear and
        # shrinks the loop by one vertex; bail to centroid fallback if no
        # safe ear is found at any rotation.
        work = list(loop)
        clipped_ok = True
        ear_faces = []
        while len(work) > 3:
            chosen = -1
            for i in range(len(work)):
                if _is_safe_ear(work, i):
                    chosen = i
                    break
            if chosen < 0:
                clipped_ok = False
                break
            n_w    = len(work)
            v_prev = work[(chosen - 1) % n_w]
            v_cur  = work[chosen]
            v_next = work[(chosen + 1) % n_w]
            ear_faces.append((v_prev, v_cur, v_next))
            key = (v_prev, v_next) if v_prev < v_next else (v_next, v_prev)
            existing_edges.add(key)
            work.pop(chosen)
        if clipped_ok and len(work) == 3:
            v_prev, v_cur, v_next = work
            ear_faces.append((v_prev, v_cur, v_next))
            for a, b in ((v_prev, v_next), (v_prev, v_cur), (v_cur, v_next)):
                key = (a, b) if a < b else (b, a)
                existing_edges.add(key)
        if clipped_ok:
            # Boundary edges in the loop walked V_i→V_{i+1}; ear clipping
            # produces triangles (v_prev, v_cur, v_next) whose edges include
            # v_cur→v_prev (reverse of v_prev→v_cur) — the manifold-sealing
            # direction.  fix_winding repairs anything we got backwards.
            for face in ear_faces:
                new_faces[fi] = face
                fi += 1
        else:
            # Centroid fallback: insert a new vertex unique to this loop.
            # Offset by loop index × 1e-3 mm in Z to guarantee no two
            # centroids round to the same digits_vertex=5 grid cell after
            # merge_vertices.
            loop_pts = verts[loop]
            centroid = loop_pts.mean(axis=0).copy()
            centroid[2] += (li + 1) * 1e-3
            new_idx  = n_verts0 + len(new_verts)
            new_verts.append(centroid)
            n_loop = len(loop)
            for i in range(n_loop):
                v_a = loop[i]
                v_b = loop[(i + 1) % n_loop]
                new_faces[fi] = (new_idx, v_b, v_a)
                fi += 1
                key = (new_idx, v_a) if new_idx < v_a else (v_a, new_idx)
                existing_edges.add(key)
                key = (new_idx, v_b) if new_idx < v_b else (v_b, new_idx)
                existing_edges.add(key)

    if fi == 0:
        return 0
    if new_verts:
        mesh.vertices = np.vstack([verts, np.asarray(new_verts, dtype=verts.dtype)])
    mesh.faces = np.vstack([mesh.faces, new_faces[:fi]])
    return fi

def _drop_open_bodies(mesh):
    """Drop *small* open-boundary components, keeping the largest body and any
    watertight components.

    The main haptera body is the largest connected component; if its remaining
    boundary edges weren't sealed by `_fill_loop_holes` we still want to keep
    it (so a downstream repair pass can try again) — dropping it would leave
    only tiny CSG-residue shells.  The complementary case is also handled:
    smaller open bodies (slivers from boolean grazing) are dropped, while
    smaller watertight bodies (legitimate disconnected solids) are preserved.
    """
    if len(mesh.faces) == 0:
        return 0
    labels = trimesh.graph.connected_component_labels(
        mesh.face_adjacency, node_count=len(mesh.faces))
    unique = np.unique(labels)
    if len(unique) <= 1:
        return 0

    # Per-component boundary check: an edge is a boundary edge for the body
    # iff exactly one face in that body uses it.  Combining the label into the
    # edge key gives a per-(body, edge) histogram in one np.unique call.
    edges_sorted   = mesh.edges_sorted                                # (3F, 2)
    edge_keys      = (edges_sorted[:, 0].astype(np.int64) << 32) | edges_sorted[:, 1].astype(np.int64)
    face_labels    = labels                                           # (F,)
    edge_labels    = np.repeat(face_labels, 3)                        # (3F,)
    composite      = (edge_labels.astype(np.int64) << 60) | (edge_keys & ((1 << 60) - 1))
    _, inv, cnts   = np.unique(composite, return_inverse=True, return_counts=True)
    edge_is_open   = cnts[inv] == 1                                   # (3F,)
    face_has_open  = edge_is_open.reshape(-1, 3).any(axis=1)          # (F,)

    body_face_count = {}
    body_open       = {}
    for label in unique:
        mask                    = face_labels == label
        body_face_count[label]  = int(mask.sum())
        body_open[label]        = bool(face_has_open[mask].any())

    largest = max(body_face_count, key=body_face_count.get)
    keep    = {largest}
    # Preserve all watertight components — they're complete by definition and
    # might be legitimate disconnected solids (e.g. an isolated tube).
    for l in unique:
        if not body_open[l]:
            keep.add(l)
    if len(keep) == len(unique):
        return 0
    keep_mask = np.isin(labels, np.array(list(keep)))
    mesh.update_faces(keep_mask)
    mesh.remove_unreferenced_vertices()
    return len(unique) - len(keep)

def _make_watertight(mesh, aggressive=False):
    """Minimal pass: trust manifold3d's topology and only fix orientation.

    manifold3d's boolean output is guaranteed watertight (every edge has
    exactly two faces) and stays watertight through the trimesh round-trip
    when `process=False, validate=False` is used.  Earlier versions of this
    function ran an aggressive merge_vertices + fill_holes + delete-non-
    manifold-faces pipeline that *introduced* boundary edges and non-
    manifold edges by collapsing legitimately-distinct vertices to within
    1e-5 mm tolerance.  Diagnostic logging confirmed the trimesh wrapping
    arrives watertight at every iteration; the corruption was downstream.

    The remaining work after the booleans is purely orientation:
    fix_winding flips any back-facing face groups, and the volume-sign
    check inverts the whole shell when the outward normal came out
    pointing in.  Both are no-ops on already-correct meshes.

    `aggressive` is kept as a parameter for compatibility with existing
    call sites but no longer triggers any extra work — the iteration-loop
    and final-export paths converge on the same result."""
    if len(mesh.faces) == 0:
        return mesh
    if not mesh.is_winding_consistent:
        trimesh.repair.fix_winding(mesh)
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
    return mesh

# Horizontal boss and hole manifolds stored separately so they can be applied
# with the correct CSG operations: boss uses (+boss) to union a clean solid,
# hole uses (-hole) to subtract.  Pre-combining them as an annular tube and then
# subtracting would invert both operations (haptera - (outer-inner) = haptera - outer + inner).
_horiz_boss_manifolds = []
_horiz_hole_manifolds = []
if HORIZ_BOSS_R > 0 or HORIZ_HOLE_R > 0:
    # Horizontal cylinders are oriented along the X-axis, positioned at z = HORIZ_H.
    # HORIZ_N=1 → single cylinder centered at y=0; HORIZ_N=2 → two at y=±HORIZ_S/2.
    #
    # Built directly with manifold3d primitives (Manifold.cylinder) — the
    # trimesh→manifold round-trip used previously produced CSG artifacts at
    # high segment counts (DEPTH≥4): vertex precision loss in the conversion
    # left the boss/cone-clip intersection with boundary curves that didn't
    # match the haptera mesh exactly, leaving a few open edges per iteration.
    # Native primitives keep topology guaranteed-watertight throughout.
    #
    # CONE_CLIP_MARGIN is bumped to 2 mm so the clip surface stays well clear
    # of any haptera tube (tube radii at DEPTH=9 are ≈1–3 mm); 0.5 mm caused
    # near-tangent grazing in the boss/haptera boolean union.
    _CONE_CLIP_MARGIN = 2.0  # mm; > max haptera tube radius near the cone wall
    _cone_clip_manifold = _Manifold.cylinder(
        height=CONE_H + _CONE_CLIP_MARGIN,
        radius_low=CONE_R + _CONE_CLIP_MARGIN,
        radius_high=0.0,
        circular_segments=128,
        center=False,
    )

    _horiz_y_offsets = [0.0] if HORIZ_N == 1 else [+HORIZ_S / 2.0, -HORIZ_S / 2.0]
    for _y_off in _horiz_y_offsets:
        if HORIZ_BOSS_R > 0:
            # Boss is built along Z, then rotated to X.  Manifold.rotate uses
            # Euler angles in degrees applied in the model's z-y'-x" order;
            # a 90° pitch (Y rotation) maps the Z-cylinder onto the +X axis.
            _boss_mfd = _Manifold.cylinder(
                height=2.2 * CONE_R,
                radius_low=HORIZ_BOSS_R,
                circular_segments=64,
                center=True,
            ).rotate([0.0, 90.0, 0.0]).translate([0.0, _y_off, HORIZ_H])
            _boss_mfd_clipped = _boss_mfd ^ _cone_clip_manifold
            _horiz_boss_manifolds.append(_boss_mfd_clipped)
        if HORIZ_HOLE_R > 0:
            # Bore is intentionally oversized (not clipped to the cone) so it punches
            # cleanly through the boss without coplanar end-cap artefacts at the cone wall.
            _inner_mfd = _Manifold.cylinder(
                height=2.2 * CONE_R,
                radius_low=HORIZ_HOLE_R,
                circular_segments=64,
                center=True,
            ).rotate([0.0, 90.0, 0.0]).translate([0.0, _y_off, HORIZ_H])
            _horiz_hole_manifolds.append(_inner_mfd)

# ── below-base trim manifold (always applied to flatten the print bed) ────────
# Spheres at every endpoint extend down to z = endpoint_z - r.  Endpoints
# clipped to the cone base land near z = 0, so their spheres dangle beneath
# z = 0.  Subtracting a half-space at z < TRIM_BOTTOM_EPS removes every
# sphere fragment below the print bed in one CSG op.  Built once and
# reused per iteration.
TRIM_BOTTOM_EPS  = -1e-3
_trim_extent     = max(CONE_R, CONE_H) * 4.0
_trim_below_box  = trimesh.creation.box(extents=[_trim_extent, _trim_extent, _trim_extent])
_trim_below_box.apply_translation([0.0, 0.0, TRIM_BOTTOM_EPS - _trim_extent / 2.0])
_trim_below_manifold = _trimesh_to_mfd(_trim_below_box)

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
converged = False
for iteration in range(1, MAX_ITERS + 1):
    _dlog(f"  ── iter {iteration} ──────────────────────────────")
    manifold = build_manifold(segs, TUBE_SIDES)
    _dlog(f"    [measure_vol] reading volume from Manifold (C++)...")
    _tm = _time.perf_counter()
    measured_vol = manifold.volume()
    _dlog(f"    [measure_vol] done  {_time.perf_counter()-_tm:.4f}s  vol={measured_vol:.4f}")
    # Apply horizontal boss/hole and trim ops with simple +/-.  manifold3d's
    # union drops interior surfaces, so a direct +boss keeps the topology
    # clean without needing a purge-then-add pattern.
    m_final = manifold
    for _hboss in _horiz_boss_manifolds:
        m_final = m_final + _hboss
    for _hhole in _horiz_hole_manifolds:
        m_final = m_final - _hhole
    # Flatten the print bed by trimming everything below z = 0.  This removes
    # sphere fragments dangling from endpoints that were clipped to the cone
    # base.
    m_final = m_final - _trim_below_manifold
    combined_iter = _manifold_to_trimesh(m_final, _dlog)
    # ── watertight repair: collapse duplicate vertices and plug any holes ─────
    _make_watertight(combined_iter)
    # ── simplify before convergence check ─────────────────────────────────────
    if SIMPLIFY_TARGET > 0 and len(combined_iter.faces) > SIMPLIFY_TARGET:
        _n_before = len(combined_iter.faces)
        _log(f"    [simplify] {_n_before} → ≤{SIMPLIFY_TARGET} faces...")
        _ts = _time.perf_counter()
        try:
            _tr = max(0.0, min(1.0 - 1e-9, 1.0 - SIMPLIFY_TARGET / _n_before))
            combined_iter = combined_iter.simplify_quadric_decimation(_tr)
            _make_watertight(combined_iter)
            _log(f"    [simplify] done  {_time.perf_counter()-_ts:.2f}s  "
                 f"({_n_before} → {len(combined_iter.faces)} faces)")
        except Exception as _e:
            _log(f"    [simplify] failed ({_e}); using full-resolution mesh")
    _dlog(f"    [measure_hull] computing convex hull volume...")
    _tm = _time.perf_counter()
    hull_vol_iter = combined_iter.convex_hull.volume
    _dlog(f"    [measure_hull] done  {_time.perf_counter()-_tm:.4f}s  hull_vol={hull_vol_iter:.4f}")
    final_vol_iter    = combined_iter.volume
    interstitial_iter   = hull_vol_iter - final_vol_iter
    _target_interstitial = TARGET_INTERSTITIAL_FRACTION * hull_vol_iter
    error = abs(interstitial_iter - _target_interstitial) / _target_interstitial
    msg   = f"  iter {iteration}: interstitial={interstitial_iter:.4f}  haptera={final_vol_iter:.4f}  error={error*100:.3f}%"
    if error <= TOLERANCE:
        _log(msg + "  ✓ converged")
        if _ibar: _ibar.update(1)
        combined  = combined_iter
        final_vol = final_vol_iter
        converged = True
        break
    # Correction steered on raw haptera volume (before boss/hole) since only tube radii are scaled.
    haptera_target = hull_vol_iter - _target_interstitial - (m_final.volume() - measured_vol)
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
        # Both sides of the root are bracketed — bisect is guaranteed to converge.
        # Never use cubic here: cubic can overshoot the bracket and cause oscillation.
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

# ── persist multiplier for next run ──────────────────────────────────────────
# Save the cumulative tube-radius multiplier (initial × per-iter corrections).
# A re-run with the same config picks this up via USE_CACHED_FACTOR and skips
# most of the iteration loop.  Logged so the value is also available to copy
# into INITIAL_FACTOR by hand.
_total_factor = _initial_factor * cumulative_factor
_final_root_r = float(segs[0]['r']) if segs else 0.0
_iv_frac      = (interstitial_iter / hull_vol_iter) if hull_vol_iter > 0 else 0.0
print(f"  Cumulative radius multiplier: ×{_total_factor:.6f}  (initial ×{_initial_factor:.6f} × loop ×{cumulative_factor:.6f})")
print(f"  Final root radius           : {_final_root_r:.6f}")
if SAVE_CACHED_FACTOR:
    _save_cached_factor(CACHE_FILE, _current_factor_config(), _total_factor,
                        _final_root_r, _iv_frac, error, converged)
    print(f"  Saved multiplier + config to {CACHE_FILE}")


# Capture pre-repair interstitial measurements so the aggressive pass below
# can be compared against the iteration-loop result.  The aggressive repair
# can drop sliver bodies and re-cap holes, which slightly shifts the volume.
_pre_repair_haptera_vol      = combined.volume
_pre_repair_hull_vol         = combined.convex_hull.volume
_pre_repair_interstitial     = _pre_repair_hull_vol - _pre_repair_haptera_vol
_pre_repair_target_iv        = TARGET_INTERSTITIAL_FRACTION * _pre_repair_hull_vol
_pre_repair_error            = abs(_pre_repair_interstitial - _pre_repair_target_iv) / _pre_repair_target_iv if _pre_repair_target_iv > 0 else float('nan')

# Final watertight pass so the exported STLs are sealed regardless of how the
# last iteration terminated (converged, max-iters, or infeasible-target break).
# Aggressive mode drops sliver bodies and surgically removes faces touching
# non-manifold edges; iteration-loop calls stayed lightweight.
_make_watertight(combined, aggressive=True)

# Post-repair interstitial volume error.  Recomputed against the repaired
# mesh's own convex hull so the target tracks any envelope change from the
# repair (sliver removal can shrink the hull).  Reported here to make the
# repair-induced delta visible separately from the final analysis report.
_post_repair_haptera_vol     = combined.volume
_post_repair_hull_vol        = combined.convex_hull.volume
_post_repair_interstitial    = _post_repair_hull_vol - _post_repair_haptera_vol
_post_repair_target_iv       = TARGET_INTERSTITIAL_FRACTION * _post_repair_hull_vol
_post_repair_error           = abs(_post_repair_interstitial - _post_repair_target_iv) / _post_repair_target_iv if _post_repair_target_iv > 0 else float('nan')
_repair_vol_delta            = _post_repair_haptera_vol - _pre_repair_haptera_vol
_repair_iv_delta             = _post_repair_interstitial - _pre_repair_interstitial

print(f"\nInterstitial volume error after mesh repair:")
print(f"  pre-repair  : interstitial={_pre_repair_interstitial:.4f}  haptera={_pre_repair_haptera_vol:.4f}  error={_pre_repair_error*100:.3f}%")
print(f"  post-repair : interstitial={_post_repair_interstitial:.4f}  haptera={_post_repair_haptera_vol:.4f}  error={_post_repair_error*100:.3f}%")
print(f"  repair Δ    : haptera={_repair_vol_delta:+.4f}  interstitial={_repair_iv_delta:+.4f}  "
      f"({'within' if _post_repair_error <= TOLERANCE else 'OUTSIDE'} tol={TOLERANCE*100:.2f}%)")

if DEBUG: print(f"[export] writing {OUTPUT}...")
_t = _time.perf_counter()
combined.export(OUTPUT)
if DEBUG: print(f"[export] done  {_time.perf_counter()-_t:.2f}s")
sys.stdout = Tee(TEXT_OUTPUT)

# Recompute from the final trimesh object so all downstream measurements
# (interstitial volume, error %) reflect the actual exported mesh after
# boss/hole cutting and any decimation.
final_vol = combined.volume

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
hull_base_mask    = haptera_hull.face_normals[:, 2] < -0.999
hull_base_area    = float(trimesh.triangles.area(haptera_hull.triangles[hull_base_mask]).sum()) if hull_base_mask.any() else 0.0
if DEBUG: print(f"[convex_hull] done  {_time.perf_counter()-_t:.2f}s  hull_vol={hull_volume:.4f}")

# ── cone geometry (analytical, for reference) ─────────────────────────────────
cone_volume       = (1.0 / 3.0) * np.pi * CONE_R**2 * CONE_H
cone_lateral_area = np.pi * CONE_R * np.sqrt(CONE_R**2 + CONE_H**2)
cone_base_area    = np.pi * CONE_R**2

# ── horizontal cylinder analytical measurements ───────────────────────────────
# Boss cylinders are ADDED (unioned) to the haptera → they contribute positive solid volume.
# Bore cylinders are SUBTRACTED from the haptera → they displace (remove) material.
# Displacement volume reported here is the net bore volume removed (×2 cylinders).
# Surface area reported is the total new surface introduced: outer boss lateral walls +
# annular end caps of the boss + inner bore lateral walls (the drilled channel surface).
horiz_boss_vol        = 0.0  # solid volume added by boss cylinders
horiz_bore_disp_vol   = 0.0  # volume removed by bore cylinders
horiz_tube_sa         = 0.0  # analytical surface area generated
horiz_bore_lateral_sa = 0.0  # bore-wall-only contribution (lateral cylinder wall of the drilled channel)
if HORIZ_BOSS_R > 0 or HORIZ_HOLE_R > 0:
    _cone_r_at_h = CONE_R * (CONE_H - HORIZ_H) / CONE_H
    # Each cylinder is clipped to the cone, so its effective chord length depends on its Y offset.
    # Chord of the cone circle at height HORIZ_H for a line at y = y_off:
    #   half_chord = sqrt(cone_r_at_h² - y_off²)  (0 if the cylinder misses the cone entirely)
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
# Total interstitial: all void space inside the hull (includes horiz bore channel volumes).
# The bore walls are already in haptera_surface_area (they are mesh surfaces), and
# the bore volumes are already captured in hull_volume - final_vol (final_vol is
# reduced by drilling).
interstitial_volume = hull_volume - final_vol           # includes horiz bore voids
total_surface_area  = haptera_surface_area - base_cap_area  # includes all bore walls
# Bore wall area (analytical): lateral wall of every drilled channel.
# Added to hull_surface_area for external SA so bore channels are not invisible to the hull metric.
bore_wall_sa        = horiz_bore_lateral_sa
external_sa         = hull_surface_area + bore_wall_sa  # hull envelope + all bore walls
internal_sa         = haptera_surface_area              # full haptera mesh (bore walls included)
total_bounding_area = (external_sa - hull_base_area) + (internal_sa - base_cap_area)  # ground-contact faces excluded
sa_to_vol           = total_surface_area / interstitial_volume if interstitial_volume > 0 else 0

# ── output ────────────────────────────────────────────────────────────────────
print(f"\nExported : {OUTPUT}")
print(f"")
print(f"Parameters")
print(f"  depth                  : {DEPTH}")
print(f"  k                      : {K}")
print(f"  n_roots                : {N_ROOTS}")
print(f"  seg_len                : {SEG_LEN}")
print(f"  tube_sides             : {TUBE_SIDES}")
print(f"  steer_onset            : {STEER_ONSET}")
print(f"  steer_strength         : {STEER_STRENGTH}")
print(f"  cone_h                 : {CONE_H}")
print(f"  cone_r                 : {CONE_R}")
print(f"  horiz_boss_r           : {HORIZ_BOSS_R}")
print(f"  horiz_hole_r           : {HORIZ_HOLE_R}")
print(f"  horiz_n                : {HORIZ_N}")
print(f"  horiz_s                : {HORIZ_S}{'  (ignored — single cylinder)' if HORIZ_N == 1 else ''}")
print(f"  horiz_h                : {HORIZ_H}")
print(f"  target_interstitial_frac: {TARGET_INTERSTITIAL_FRACTION:.6f}  (fraction of hull)")
print(f"  base_volume (haptera)  : {BASE_VOLUME:.4f}")
print(f"")
print(f"Mesh")
print(f"  segments               : {len(segs)}")
print(f"  vertices               : {len(combined.vertices)}")
print(f"  faces                  : {len(combined.faces)}")
print(f"")
# ── solidity test ─────────────────────────────────────────────────────────────
# A mesh is "solid" (a printable, valid volume) iff it is watertight, has
# consistent face winding, has positive volume, and has only manifold edges
# (each edge shared by exactly two faces).  trimesh.is_volume bundles these
# checks; we report the individual flags too so a failure points to the cause.
def _solidity_report(label, mesh):
    # Boundary edges (shared by 1 face) → open holes. Non-manifold edges
    # (shared by 3+ faces) → T-junctions / overlapping shells. Both block
    # is_watertight; the counts pinpoint which one is the problem.
    edges = mesh.edges_sorted
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges    = int((edge_counts == 1).sum())
    nonmanifold_edges = int((edge_counts >= 3).sum())
    body_count        = int(mesh.body_count)
    broken_face_count = len(trimesh.repair.broken_faces(mesh))
    checks = {
        "is_volume (solid)"  : bool(mesh.is_volume),
        "watertight"         : bool(mesh.is_watertight),
        "winding consistent" : bool(mesh.is_winding_consistent),
        "positive volume"    : bool(mesh.volume > 0),
        "manifold edges"     : nonmanifold_edges == 0,
        "no open boundary"   : boundary_edges == 0,
    }
    print(f"{label}")
    for name, ok in checks.items():
        print(f"  {name:<22} : {'PASS' if ok else 'FAIL'}")
    print(f"  {'boundary edges':<22} : {boundary_edges}  (>0 = open holes)")
    print(f"  {'non-manifold edges':<22} : {nonmanifold_edges}  (>0 = T-junctions / overlaps)")
    print(f"  {'broken faces':<22} : {broken_face_count}  (faces touching a bad edge)")
    print(f"  {'connected bodies':<22} : {body_count}  (>1 = floating shells)")
    print(f"  {'euler characteristic':<22} : {mesh.euler_number}  (closed genus-g surface = 2 - 2g)")
    print(f"  {'volume':<22} : {mesh.volume:.4f}")
    print(f"")

_solidity_report("Solidity", combined)
print(f"Haptera")
print(f"  volume                 : {final_vol:.4f}")
_final_target_iv = TARGET_INTERSTITIAL_FRACTION * hull_volume
print(f"  interstitial vol error : {abs(interstitial_volume - _final_target_iv) / _final_target_iv * 100:.3f}%  (vs fraction target)")
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
print(f"  volume                 : {interstitial_volume:.4f}  (hull − haptera; includes horiz bore voids)")
if HORIZ_HOLE_R > 0:
    print(f"  horiz bore volume      : {horiz_bore_disp_vol:.4f}  (HORIZ_HOLE contribution, analytical)")
print(f"  bore wall area         : {bore_wall_sa:.4f}  (horiz bore lateral walls, analytical)")
print(f"  external surface area  : {external_sa:.4f}  (hull surface + all bore walls)")
print(f"  internal surface area  : {internal_sa:.4f}  (haptera mesh; all bore walls included)")
print(f"  total bounding area    : {total_bounding_area:.4f}  (external + internal; ground-contact faces excluded)")
print(f"  SA / volume ratio      : {sa_to_vol:.4f}  (complexity index using total wetted SA)")

sys.stdout.close()
