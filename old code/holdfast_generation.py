import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# -- cone parameters ----------------------------------------------------------
CONE_H      = 5.0
CONE_R      = 2.0
N_ROOTS     = 4
SEG_LEN     = 1.05
REF_ROOT_R  = 0.28
BASE_VOLUME = N_ROOTS * np.pi * REF_ROOT_R**2 * SEG_LEN  # fixed volume target

# -- deterministic PRNG -------------------------------------------------------
def make_rng(seed=54321):
    state = [seed & 0xFFFFFFFF]
    def rng():
        state[0] = (1664525 * state[0] + 1013904223) & 0xFFFFFFFF
        return state[0] / 4294967296
    return rng

# -- geometry helpers ---------------------------------------------------------
def cone_contains(x, y, z):
    return (y >= 0) and (y <= CONE_H) and (
        np.sqrt(x*x + z*z) <= (CONE_R / CONE_H) * (CONE_H - y)
    )

def clip_to_cone(ox, oy, oz, dx, dy, dz, seg_len):
    ex, ey, ez = ox + dx*seg_len, oy + dy*seg_len, oz + dz*seg_len
    if cone_contains(ex, ey, ez):
        return ex, ey, ez
    lo, hi = 0.0, 1.0
    for _ in range(12):
        m = (lo + hi) / 2
        if cone_contains(ox+dx*seg_len*m, oy+dy*seg_len*m, oz+dz*seg_len*m):
            lo = m
        else:
            hi = m
    return ox+dx*seg_len*lo, oy+dy*seg_len*lo, oz+dz*seg_len*lo

# -- recursive branch grower --------------------------------------------------
def grow(ox, oy, oz, dx, dy, dz, r, depth, k, seg_len, shrink, rng, max_depth, out):
    if not cone_contains(ox, oy, oz):
        return
    ex, ey, ez = clip_to_cone(ox, oy, oz, dx, dy, dz, seg_len)
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
    norm = np.sqrt(dx*dx + dy*dy + dz*dz)
    dx, dy, dz = dx/norm, dy/norm, dz/norm
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

# -- segment builder with post-hoc volume correction -------------------------
def build_segments(depth, k, shrink):
    # Initial root radius from analytic formula (ignores clipping)
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

    # Post-hoc correction: cone clipping shortens/drops segments so actual
    # volume is less than BASE_VOLUME. Scale all radii uniformly to compensate.
    # Volume = pi*r^2*L, so to scale volume by factor F, scale r by sqrt(F).
    actual_vol = sum(np.pi * s['r']**2 * np.linalg.norm(s['end'] - s['start'])
                     for s in segs)
    if actual_vol > 0:
        scale = np.sqrt(BASE_VOLUME / actual_vol)
        for s in segs:
            s['r'] *= scale

    return segs

# -- colour map: dark teal root -> light teal tip -----------------------------
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

sl_depth  = Slider(ax_depth,  'Depth',  1, 6,       valinit=4,    valstep=1, color='#1d9e75')
sl_k      = Slider(ax_k,      'k',      2, 4,       valinit=2,    valstep=1, color='#1d9e75')
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
    root_r = segs[0]['r'] if segs else 0.0

    lines   = []
    colors  = []
    lwidths = []
    max_r   = max(s['r'] for s in segs) if segs else 1.0

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
    tip_r   = min(s['r'] for s in segs) if segs else 0
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