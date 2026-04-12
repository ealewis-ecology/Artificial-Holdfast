import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ── cone + growth parameters (must match haptera_viz.py) ─────────────────────
CONE_H      = 5.0
CONE_R      = 2.0
N_ROOTS     = 4
SEG_LEN     = 1.05
REF_ROOT_R  = 0.28
BASE_VOLUME = N_ROOTS * np.pi * REF_ROOT_R**2 * SEG_LEN

STEER_ONSET    = 0.55
STEER_STRENGTH = 3.2

POINTS_PER_UNIT = 20   # how densely to sample each segment for box-counting

# ── PRNG ──────────────────────────────────────────────────────────────────────
def make_rng(seed=54321):
    state = [seed & 0xFFFFFFFF]
    def rng():
        state[0] = (1664525 * state[0] + 1013904223) & 0xFFFFFFFF
        return state[0] / 4294967296
    return rng

# ── geometry ──────────────────────────────────────────────────────────────────
def cone_contains(x, y, z):
    return (y >= 0) and (y <= CONE_H) and (
        np.sqrt(x*x + z*z) <= (CONE_R / CONE_H) * (CONE_H - y))

def wall_proximity(x, y, z):
    r_pos  = np.sqrt(x*x + z*z)
    r_wall = (CONE_R / CONE_H) * (CONE_H - y)
    if r_wall <= 0:
        return 2.0
    return r_pos / r_wall

def inward_direction(x, z):
    r = np.sqrt(x*x + z*z)
    if r < 1e-9:
        return np.array([0.0, 0.0, 0.0])
    return np.array([-x / r, 0.0, -z / r])

def steer(dx, dy, dz, ox, oy, oz):
    prox = wall_proximity(ox, oy, oz)
    if prox < STEER_ONSET:
        return dx, dy, dz
    t = min((prox - STEER_ONSET) / (1.0 - STEER_ONSET), 1.0)
    weight = STEER_STRENGTH * t * t
    inward = inward_direction(ox, oz)
    ndx = dx + inward[0] * weight
    ndy = dy
    ndz = dz + inward[2] * weight
    nl  = np.sqrt(ndx*ndx + ndy*ndy + ndz*ndz)
    return ndx/nl, ndy/nl, ndz/nl

def grow(ox, oy, oz, dx, dy, dz, r, depth, k, seg_len, shrink, rng, max_depth, out):
    if not cone_contains(ox, oy, oz):
        return
    dx, dy, dz = steer(dx, dy, dz, ox, oy, oz)
    ex = ox + dx * seg_len
    ey = oy + dy * seg_len
    ez = oz + dz * seg_len
    if not cone_contains(ex, ey, ez):
        for _ in range(6):
            inward = inward_direction(ex, ez)
            ex += inward[0] * seg_len * 0.25
            ez += inward[2] * seg_len * 0.25
            if cone_contains(ex, ey, ez):
                break
        if not cone_contains(ex, ey, ez):
            return
    actual_len = np.sqrt((ex-ox)**2 + (ey-oy)**2 + (ez-oz)**2)
    if actual_len < 0.02:
        return
    out.append({'start': np.array([ox, oy, oz]),
                'end':   np.array([ex, ey, ez]),
                'r': r, 'level': max_depth - depth})
    if depth == 0:
        return
    seg_dir = np.array([ex-ox, ey-oy, ez-oz])
    seg_dir /= np.linalg.norm(seg_dir)
    dx, dy, dz = seg_dir
    for i in range(k):
        angle  = (2 * np.pi * i / k) + rng() * 0.5 - 0.25
        spread = 0.28 + rng() * 0.18
        ndx = dx + np.cos(angle) * spread
        ndy = dy + rng() * 0.05
        ndz = dz + np.sin(angle) * spread
        nl  = np.sqrt(ndx*ndx + ndy*ndy + ndz*ndz)
        child_r = r / np.sqrt(k * shrink)
        grow(ex, ey, ez, ndx/nl, ndy/nl, ndz/nl,
             child_r, depth-1, k, seg_len*shrink, shrink, rng, max_depth, out)

def build_segments(depth, k, shrink):
    root_r = np.sqrt(BASE_VOLUME / (N_ROOTS * np.pi * SEG_LEN * (depth + 1)))
    rng  = make_rng(54321)
    segs = []
    for i in range(N_ROOTS):
        a  = (2 * np.pi * i / N_ROOTS) + 0.35
        ox, oy, oz = 0.0, CONE_H - 0.05, 0.0
        rdx = np.cos(a) * 0.45
        rdy = -1.0
        rdz = np.sin(a) * 0.45
        rl  = np.sqrt(rdx*rdx + rdy*rdy + rdz*rdz)
        grow(ox, oy, oz, rdx/rl, rdy/rl, rdz/rl,
             root_r, depth, k, SEG_LEN, shrink, rng, depth, segs)
    actual_vol = sum(np.pi * s['r']**2 * np.linalg.norm(s['end'] - s['start'])
                     for s in segs)
    if actual_vol > 0:
        scale = np.sqrt(BASE_VOLUME / actual_vol)
        for s in segs:
            s['r'] *= scale
    return segs

# ── point cloud sampler ───────────────────────────────────────────────────────
def segments_to_points(segs, points_per_unit=POINTS_PER_UNIT):
    """
    Sample points densely and uniformly along every segment.
    More points = more accurate box count at fine scales.
    """
    all_pts = []
    for s in segs:
        length = np.linalg.norm(s['end'] - s['start'])
        n      = max(2, int(length * points_per_unit))
        ts     = np.linspace(0, 1, n)
        pts    = s['start'][None, :] + ts[:, None] * (s['end'] - s['start'])[None, :]
        all_pts.append(pts)
    return np.vstack(all_pts)

# ── box-counting fractal dimension ────────────────────────────────────────────
def box_count(points, n_scales=16):
    """
    Cover the point cloud with a 3D grid of boxes at a range of scales.
    Count occupied boxes N(eps) at each scale eps.
    Fractal dimension D = -d(log N) / d(log eps)
                        = slope of log(N) vs log(1/eps).

    Scale range is chosen to span two orders of magnitude centred on the
    structure, avoiding the trivially large (whole structure = 1 box) and
    trivially small (box = single point) regimes.
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = (maxs - mins).max()

    # eps from ~span/4 down to ~span/200
    epsilons = np.logspace(np.log10(span / 4),
                           np.log10(span / 200),
                           n_scales)
    counts = []
    for eps in epsilons:
        # Bin each point into a voxel index, count unique voxels
        indices = np.floor((points - mins) / eps).astype(int)
        occupied = len(set(map(tuple, indices)))
        counts.append(occupied)

    counts   = np.array(counts, dtype=float)
    log_eps  = np.log(epsilons)
    log_n    = np.log(counts)

    # Linear fit in log-log space — slope is -D
    coeffs   = np.polyfit(log_eps, log_n, 1)
    D        = -coeffs[0]

    # R² of the fit — how well the structure is actually self-similar
    log_n_fit = np.polyval(coeffs, log_eps)
    ss_res    = np.sum((log_n - log_n_fit)**2)
    ss_tot    = np.sum((log_n - log_n.mean())**2)
    r2        = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return D, r2, epsilons, counts, coeffs

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 7), facecolor='#111111')
fig.patch.set_facecolor('#111111')

ax_plot = fig.add_axes([0.08, 0.20, 0.55, 0.72])
ax_plot.set_facecolor('#111111')
ax_plot.tick_params(colors='#888888')
ax_plot.spines[:].set_edgecolor('#333333')
ax_plot.set_xlabel('log(1 / ε)',  color='#888888', fontsize=11)
ax_plot.set_ylabel('log(N boxes)', color='#888888', fontsize=11)
ax_plot.set_title('Box-counting — log/log scaling', color='#888888', fontsize=11)

stats_ax = fig.add_axes([0.68, 0.30, 0.28, 0.50])
stats_ax.set_facecolor('#111111')
stats_ax.axis('off')
stat_text = stats_ax.text(0.05, 0.95, '', transform=stats_ax.transAxes,
    color='#cccccc', fontsize=12, va='top', fontfamily='monospace')

# sliders
ax_depth  = fig.add_axes([0.10, 0.11,  0.50, 0.025], facecolor='#1e1e1e')
ax_k      = fig.add_axes([0.10, 0.075, 0.50, 0.025], facecolor='#1e1e1e')
ax_shrink = fig.add_axes([0.10, 0.035, 0.50, 0.025], facecolor='#1e1e1e')

sl_depth  = Slider(ax_depth,  'Depth',  1, 7,       valinit=4,    valstep=1, color='#1d9e75')
sl_k      = Slider(ax_k,      'k',      2, 4,       valinit=2,    valstep=1, color='#1d9e75')
sl_shrink = Slider(ax_shrink, 'Shrink', 0.50, 0.85, valinit=0.65,            color='#1d9e75')

for sl in [sl_depth, sl_k, sl_shrink]:
    sl.label.set_color('#aaaaaa')
    sl.valtext.set_color('#cccccc')

TEAL = '#1d9e75'
FIT_COLOR = '#f0994a'

def redraw(_=None):
    depth  = int(sl_depth.val)
    k      = int(sl_k.val)
    shrink = float(sl_shrink.val)

    segs   = build_segments(depth, k, shrink)
    pts    = segments_to_points(segs)
    D, r2, epsilons, counts, coeffs = box_count(pts)

    log_eps_inv = np.log(1.0 / epsilons)
    log_n       = np.log(counts)
    fit_line    = -coeffs[0] * np.log(1.0 / epsilons) + (
                   np.log(counts[0]) + coeffs[0] * np.log(1.0 / epsilons[0]))

    ax_plot.cla()
    ax_plot.set_facecolor('#111111')
    ax_plot.tick_params(colors='#888888')
    ax_plot.spines[:].set_edgecolor('#333333')
    ax_plot.set_xlabel('log(1 / ε)',   color='#888888', fontsize=11)
    ax_plot.set_ylabel('log(N boxes)', color='#888888', fontsize=11)
    ax_plot.set_title('Box-counting — log/log scaling', color='#888888', fontsize=11)

    ax_plot.scatter(log_eps_inv, log_n, color=TEAL,      s=30, zorder=3, label='measured')
    ax_plot.plot(   log_eps_inv, fit_line, color=FIT_COLOR, lw=1.5, zorder=2,
                    label=f'fit  D = {D:.3f}')
    ax_plot.legend(facecolor='#1e1e1e', edgecolor='#333333',
                   labelcolor='#cccccc', fontsize=10)

    stat_text.set_text(
        f"depth   {depth}\n"
        f"k       {k}\n"
        f"shrink  {shrink:.2f}\n"
        f"\n"
        f"D       {D:.4f}\n"
        f"R²      {r2:.4f}\n"
        f"\n"
        f"points  {len(pts)}\n"
        f"segs    {len(segs)}"
    )

    fig.canvas.draw_idle()

sl_depth.on_changed(redraw)
sl_k.on_changed(redraw)
sl_shrink.on_changed(redraw)

redraw()

plt.suptitle('Fractal dimension of haptera', color='#888888', fontsize=11, y=0.99)
plt.show()
