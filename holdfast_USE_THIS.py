import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# -- cone parameters ----------------------------------------------------------
CONE_H      = 5.0
CONE_R      = 2.0
N_ROOTS     = 8
SEG_LEN     = 1.05
REF_ROOT_R  = 0.6
BASE_VOLUME = N_ROOTS * np.pi * REF_ROOT_R**2 * SEG_LEN

# -- wall proximity threshold: steering kicks in when branch is this fraction
# of the way from axis to wall (0 = always steering, 1 = only at wall)
STEER_ONSET = 0.55   # start curving at 55% of wall radius
STEER_STRENGTH = 3.2 # how hard the inward push is at the wall

# -- deterministic PRNG -------------------------------------------------------
def make_rng(seed=54321):
    state = [seed & 0xFFFFFFFF]
    def rng():
        state[0] = (1664525 * state[0] + 1013904223) & 0xFFFFFFFF
        return state[0] / 4294967296
    return rng

# -- geometry -----------------------------------------------------------------
def cone_contains(x, y, z):
    return (y >= 0) and (y <= CONE_H) and (
        np.sqrt(x*x + z*z) <= (CONE_R / CONE_H) * (CONE_H - y)
    )

def wall_proximity(x, y, z):
    """0 = on axis, 1 = exactly at wall, >1 = outside."""
    r_pos  = np.sqrt(x*x + z*z)
    r_wall = (CONE_R / CONE_H) * (CONE_H - y)
    if r_wall <= 0:
        return 2.0
    return r_pos / r_wall

def inward_direction(x, z):
    """Unit vector pointing radially inward (toward cone axis) in xz plane."""
    r = np.sqrt(x*x + z*z)
    if r < 1e-9:
        return np.array([0.0, 0.0, 0.0])
    return np.array([-x / r, 0.0, -z / r])

def steer(dx, dy, dz, ox, oy, oz):
    """
    Blend the branch direction with an inward radial component based on how
    close the origin is to the cone wall.  No hard clipping — branches curve.
    """
    prox = wall_proximity(ox, oy, oz)
    if prox < STEER_ONSET:
        return dx, dy, dz

    # How far past the onset threshold (0 at onset, 1 at wall)
    t = min((prox - STEER_ONSET) / (1.0 - STEER_ONSET), 1.0)
    weight = STEER_STRENGTH * t * t   # quadratic ramp — gentle until close

    inward = inward_direction(ox, oz)
    ndx = dx + inward[0] * weight
    ndy = dy
    ndz = dz + inward[2] * weight
    nl  = np.sqrt(ndx*ndx + ndy*ndy + ndz*ndz)
    return ndx/nl, ndy/nl, ndz/nl

# -- recursive branch grower (no clipping) ------------------------------------
def grow(ox, oy, oz, dx, dy, dz, r, depth, k, seg_len, shrink, rng, max_depth, out):
    if not cone_contains(ox, oy, oz):
        return

    # Apply wall steering to direction before stepping
    dx, dy, dz = steer(dx, dy, dz, ox, oy, oz)

    ex = ox + dx * seg_len
    ey = oy + dy * seg_len
    ez = oz + dz * seg_len

    # If the endpoint still escapes (very acute angle), nudge it back
    # with a stronger inward pull rather than hard-clipping
    if not cone_contains(ex, ey, ez):
        for _ in range(6):
            inward = inward_direction(ex, ez)
            ex += inward[0] * seg_len * 0.25
            ez += inward[2] * seg_len * 0.25
            if cone_contains(ex, ey, ez):
                break
        # If still outside after nudging, drop this branch
        if not cone_contains(ex, ey, ez):
            return

    actual_len = np.sqrt((ex-ox)**2 + (ey-oy)**2 + (ez-oz)**2)
    if actual_len < 0.02:
        return

    out.append({
        'start': np.array([ox, oy, oz]),
        'end':   np.array([ex, ey, ez]),
        'r':     r,
        'level': max_depth - depth,
    })

    if depth == 0:
        return

    # Recompute normalised direction from actual start->end
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

# -- segment builder with volume correction -----------------------------------
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

    # Post-hoc volume correction for any residual loss
    actual_vol = sum(np.pi * s['r']**2 * np.linalg.norm(s['end'] - s['start'])
                     for s in segs)
    if actual_vol > 0:
        scale = np.sqrt(BASE_VOLUME / actual_vol)
        for s in segs:
            s['r'] *= scale

    return segs

# -- colour map ---------------------------------------------------------------
TEAL_DARK  = np.array([0.03, 0.31, 0.25])
TEAL_LIGHT = np.array([0.62, 0.88, 0.80])

def seg_color(level, max_depth):
    t = level / max(max_depth, 1)
    return tuple(TEAL_DARK * (1-t) + TEAL_LIGHT * t)

# -- wireframe cone -----------------------------------------------------------
def draw_cone(ax):
    u = np.linspace(0, 2*np.pi, 40)
    for frac in np.linspace(0.1, 1.0, 6):
        r = CONE_R * frac
        y = CONE_H * (1 - frac)
        ax.plot(r*np.cos(u), np.full_like(u, y), r*np.sin(u),
                color='gray', alpha=0.15, linewidth=0.6, zorder=0)
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        ax.plot([0, CONE_R*np.cos(angle)], [CONE_H, 0], [0, CONE_R*np.sin(angle)],
                color='gray', alpha=0.12, linewidth=0.6, zorder=0)

# -- figure setup -------------------------------------------------------------
fig = plt.figure(figsize=(11, 8), facecolor='#111111')
fig.patch.set_facecolor('#111111')

ax = fig.add_axes([0.05, 0.18, 0.60, 0.78], projection='3d')
ax.set_facecolor('#111111')
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#333333')
ax.tick_params(colors='#555555', labelsize=7)
ax.set_xlabel('X', color='#444444', labelpad=2, fontsize=8)
ax.set_ylabel('Y (height)', color='#444444', labelpad=2, fontsize=8)
ax.set_zlabel('Z', color='#444444', labelpad=2, fontsize=8)

stats_ax = fig.add_axes([0.67, 0.35, 0.30, 0.45])
stats_ax.set_facecolor('#111111')
stats_ax.axis('off')
stat_text = stats_ax.text(0.05, 0.95, '', transform=stats_ax.transAxes,
    color='#cccccc', fontsize=11, va='top', fontfamily='monospace')

# -- sliders ------------------------------------------------------------------
ax_depth  = fig.add_axes([0.10, 0.10,  0.50, 0.025], facecolor='#1e1e1e')
ax_k      = fig.add_axes([0.10, 0.065, 0.50, 0.025], facecolor='#1e1e1e')
ax_shrink = fig.add_axes([0.10, 0.030, 0.50, 0.025], facecolor='#1e1e1e')

sl_depth  = Slider(ax_depth,  'Depth',  0, 10,       valinit=4,    valstep=1, color='#1d9e75')
sl_k      = Slider(ax_k,      'k',      0, 8,       valinit=2,    valstep=1, color='#1d9e75')
sl_shrink = Slider(ax_shrink, 'Shrink', 0.50, 0.85, valinit=0.65,            color='#1d9e75')

for sl in [sl_depth, sl_k, sl_shrink]:
    sl.label.set_color('#aaaaaa')
    sl.valtext.set_color('#cccccc')

# -- redraw -------------------------------------------------------------------
def redraw(_=None):
    depth  = int(sl_depth.val)
    k      = int(sl_k.val)
    shrink = float(sl_shrink.val)

    ax.cla()
    ax.set_facecolor('#111111')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#333333')
    ax.tick_params(colors='#555555', labelsize=7)
    ax.set_xlabel('X', color='#444444', labelpad=2, fontsize=8)
    ax.set_ylabel('Y', color='#444444', labelpad=2, fontsize=8)
    ax.set_zlabel('Z', color='#444444', labelpad=2, fontsize=8)

    draw_cone(ax)

    segs   = build_segments(depth, k, shrink)
    max_r  = max(s['r'] for s in segs) if segs else 1.0

    lines   = []
    colors  = []
    lwidths = []
    for s in segs:
        lines.append([s['start'], s['end']])
        colors.append(seg_color(s['level'], depth))
        lwidths.append(max(0.4, 5.0 * s['r'] / max_r))

    lc = Line3DCollection(lines, colors=colors, linewidths=lwidths, alpha=0.85)
    ax.add_collection3d(lc)

    ax.set_xlim(-CONE_R, CONE_R)
    ax.set_ylim(0, CONE_H)
    ax.set_zlim(-CONE_R, CONE_R)

    total_vol = sum(np.pi * s['r']**2 * np.linalg.norm(s['end'] - s['start'])
                    for s in segs)
    root_r  = segs[0]['r'] if segs else 0.0
    tip_r   = min(s['r'] for s in segs) if segs else 0.0
    tip_cnt = sum(1 for s in segs if s['level'] == depth)

    stat_text.set_text(
        f"depth    {depth}\n"
        f"k        {k}\n"
        f"shrink   {shrink:.2f}\n"
        f"\n"
        f"segments {len(segs)}\n"
        f"tips     {tip_cnt}\n"
        f"volume   {total_vol:.3f}\n"
        f"root r   {root_r:.4f}\n"
        f"tip r    {tip_r:.4f}"
    )

    fig.canvas.draw_idle()

sl_depth.on_changed(redraw)
sl_k.on_changed(redraw)
sl_shrink.on_changed(redraw)

redraw()

plt.suptitle('Haptera growth inside cone', color='#888888', fontsize=10, y=0.99)
plt.show()