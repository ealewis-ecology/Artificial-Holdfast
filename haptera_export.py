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
DEBUG       = False  # True = print per-function timing diagnostics
INTERTWINED = False  # True = roots spiral down the cone on helical lanes (see intertwining parameters below)

DEPTH  = 6 #Number of nodes
K      = 2 #Number of branches per node

TUBE_SIDES = 10

# ── convergence ───────────────────────────────────────────────────────────────
TOLERANCE = 0.001 #0.1%
MAX_ITERS = 40

# ── cone geometry ─────────────────────────────────────────────────────────────
# Defined here so targets below can reference cone volume.
CONE_H = 130
CONE_R = 130
VERT_BOSS_R  = 0   # radius of solid central vertical boss cylinder (must be > VERT_HOLE_R); set to 0 to disable
VERT_HOLE_R  = 0   # radius of central vertical through-hole drilled through the boss; set to 0 to disable
HORIZ_BOSS_R = 0   # outer radius of horizontal bossed cylinders that bisect the haptera; set to 0 to disable
HORIZ_HOLE_R = 0   # inner bore radius of horizontal bossed cylinders; set to 0 to disable
HORIZ_N      = 0   # number of horizontal cylinders: 1 = single centered cylinder, 2 = two at ±HORIZ_S/2
HORIZ_S      = 60  # center-to-center spacing in Y between the two cylinders (ignored when HORIZ_N = 1)
HORIZ_H      = 30  # height above the haptera base for horizontal cylinder centers
SIMPLIFY_TARGET = 6000000  # target face count after QEM decimation (e.g. 50000); 0 = disabled

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
TARGET_INTERSTITIAL_FRACTION = 0.747920635

# Haptera mesh volume fallback (used when dynamic target is infeasible):
BASE_VOLUME = _CONE_VOLUME * (1 - TARGET_INTERSTITIAL_FRACTION)

OUTPUT      = "haptera_{}d{}_k{}_r{}_h{}_f{}.stl".format(
                "intertwined_" if INTERTWINED else "",
                DEPTH, K, CONE_R, CONE_H,
                round(TARGET_INTERSTITIAL_FRACTION * 1000))
TEXT_OUTPUT = OUTPUT.replace(".stl", ".txt")


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
    Arrays are passed directly (no extra copies).
    """
    import time
    _log("    [extract_mesh] converting Manifold to trimesh...")
    t0 = time.perf_counter()
    r  = manifold.to_mesh()
    result = trimesh.Trimesh(
        vertices=r.vert_properties[:, :3],  # slice XYZ; newer manifold3d appends extra property channels
        faces=r.tri_verts,
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

# ── main ──────────────────────────────────────────────────────────────────────
import time as _time
print(f"Building segments (depth={DEPTH}, k={K})...")
_t = _time.perf_counter()
segs = build_segments(DEPTH, K)
if DEBUG:
    print(f"  [build_segments] done  {_time.perf_counter()-_t:.2f}s  {len(segs)} segments")
else:
    print(f"  {len(segs)} segments generated")

# ── hole volume correction (used in iteration error and final report) ─────────
hole_volume       = np.pi * VERT_HOLE_R**2 * CONE_H if VERT_HOLE_R > 0 else 0.0
hole_lateral_area = 2 * np.pi * VERT_HOLE_R * CONE_H if VERT_HOLE_R > 0 else 0.0

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

_vert_boss_manifold  = None
_vert_hole_manifold  = None
# Horizontal boss and hole manifolds stored separately so they can be applied
# with the correct CSG operations: boss uses (-boss + boss) to union a clean solid,
# hole uses (-hole) to subtract.  Pre-combining them as an annular tube and then
# subtracting would invert both operations (haptera - (outer-inner) = haptera - outer + inner).
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
    # Horizontal cylinders are oriented along the X-axis, positioned at z = HORIZ_H.
    # HORIZ_N=1 → single cylinder centered at y=0; HORIZ_N=2 → two at y=±HORIZ_S/2.
    _rot_y90 = trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    _cone_r_at_horiz_h = CONE_R * (CONE_H - HORIZ_H) / CONE_H
    _horiz_y_offsets = [0.0] if HORIZ_N == 1 else [+HORIZ_S / 2.0, -HORIZ_S / 2.0]
    for _y_off in _horiz_y_offsets:
        if HORIZ_BOSS_R > 0:
            # Boss length = chord of the cone circle at height HORIZ_H for a line at y = y_off:
            #   half_len = sqrt(cone_r² - y_off²)
            # The boss centerline endpoints land exactly on the cone outline — no boolean
            # intersection needed, and no dependency on manifold3d's intersection operator.
            _boss_half_len = np.sqrt(max(_cone_r_at_horiz_h**2 - _y_off**2, 0.0))
            _outer_cyl = trimesh.creation.cylinder(radius=HORIZ_BOSS_R, height=2.0 * _boss_half_len, sections=64)
            _outer_cyl.apply_transform(_rot_y90)
            _outer_cyl.apply_translation([0.0, _y_off, HORIZ_H])
            _horiz_boss_manifolds.append(_trimesh_to_mfd(_outer_cyl))
        if HORIZ_HOLE_R > 0:
            # The bore is intentionally oversized (not clipped to the cone) so it punches
            # cleanly through the boss without coplanar end-cap artefacts at the cone wall.
            # This mirrors the VERT_HOLE_R pattern (3× height cylinder).
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
    # Apply boss/hole to get the final-mesh manifold for this iteration.
    # Subtract boss first to cleanly remove any haptera inside the boss region,
    # then union the boss back in — this prevents interior haptera surfaces from
    # persisting inside the boss, which would inflate the surface area calculation.
    m_final = manifold
    if _vert_boss_manifold is not None:
        m_final = m_final - _vert_boss_manifold  # purge haptera inside boss
        m_final = m_final + _vert_boss_manifold  # add clean boss solid
    if _vert_hole_manifold is not None:
        m_final = m_final - _vert_hole_manifold
    for _hboss in _horiz_boss_manifolds:
        m_final = m_final - _hboss  # purge haptera inside boss region
        m_final = m_final + _hboss  # add clean boss solid
    for _hhole in _horiz_hole_manifolds:
        m_final = m_final - _hhole  # drill the bore through haptera and boss
    combined_iter = _manifold_to_trimesh(m_final, _dlog)
    # ── watertight repair: collapse duplicate vertices and plug any holes ─────
    combined_iter.merge_vertices()
    if not combined_iter.is_watertight:
        trimesh.repair.fill_holes(combined_iter)
    # ── simplify before convergence check ─────────────────────────────────────
    if SIMPLIFY_TARGET > 0 and len(combined_iter.faces) > SIMPLIFY_TARGET:
        _n_before = len(combined_iter.faces)
        _log(f"    [simplify] {_n_before} → ≤{SIMPLIFY_TARGET} faces...")
        _ts = _time.perf_counter()
        try:
            _tr = max(0.0, min(1.0 - 1e-9, 1.0 - SIMPLIFY_TARGET / _n_before))
            combined_iter = combined_iter.simplify_quadric_decimation(_tr)
            combined_iter.merge_vertices()
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
    # Dynamic target: desired haptera-only interstitial = fraction of hull minus vert bore void.
    _target_interstitial = TARGET_INTERSTITIAL_FRACTION * hull_vol_iter - hole_volume
    error = abs(interstitial_iter - _target_interstitial) / _target_interstitial
    msg   = f"  iter {iteration}: interstitial={interstitial_iter:.4f}  haptera={final_vol_iter:.4f}  error={error*100:.3f}%"
    if error <= TOLERANCE:
        _log(msg + "  ✓ converged")
        if _ibar: _ibar.update(1)
        combined  = combined_iter
        final_vol = final_vol_iter
        break
    # Correction steered on raw haptera volume (before boss/hole) since only tube radii are scaled.
    # m_final.volume() = V(haptera+boss) - hole_volume, so (m_final.volume()-measured_vol) already
    # bakes in -hole_volume; subtract it again to recover the correct net boss contribution.
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


# Final watertight pass so the exported STL is sealed regardless of how the
# last iteration terminated (converged, max-iters, or infeasible-target break).
combined.merge_vertices()
if not combined.is_watertight:
    trimesh.repair.fill_holes(combined)

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
# Total interstitial: all void space inside the hull (includes bore channel volumes).
# The bore walls are already in haptera_surface_area (they are mesh surfaces), and
# the bore volumes are already captured in hull_volume - final_vol (final_vol is
# reduced by drilling).  hole_volume / hole_lateral_area are kept only for the
# convergence error check so it stays consistent with the calibration target.
interstitial_volume = hull_volume - final_vol           # includes vert & horiz bore voids
total_surface_area  = haptera_surface_area - base_cap_area  # includes all bore walls
# Bore wall area (analytical): lateral wall of every drilled channel.
# Added to hull_surface_area for external SA so bore channels are not invisible to the hull metric.
bore_wall_sa        = hole_lateral_area + horiz_bore_lateral_sa
external_sa         = hull_surface_area + bore_wall_sa  # hull envelope + all bore walls
internal_sa         = haptera_surface_area              # full haptera mesh (bore walls included)
total_bounding_area = (external_sa - hull_base_area) + (internal_sa - base_cap_area)  # ground-contact faces excluded
sa_to_vol           = total_surface_area / interstitial_volume if interstitial_volume > 0 else 0
# Haptera-only interstitial (excludes vert bore) — used for error vs calibration target.
interstitial_haptera_only = hull_volume - final_vol - hole_volume

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
