"""Visualize how phi is constructed in haptera_export.py grow().

phi = arctan2(dy, dx) + (max_depth - depth) * TORSION

then each child is placed at:
    angle_i = 2*pi*i/k + phi + (rng()*0.5 - 0.25)   # last term is jitter in [-0.25, 0.25]
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc, Wedge

# Match the constants in haptera_export.py
TORSION = 0.6
K = 2

# Pick a sample parent direction in the xy plane (unit length, pointing 30 deg)
phi_base = np.deg2rad(30.0)
dx, dy = np.cos(phi_base), np.sin(phi_base)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for ax in axes:
    ax.set_aspect('equal')
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='lightgray', lw=1)
    # Axes
    ax.axhline(0, color='lightgray', lw=0.5)
    ax.axvline(0, color='lightgray', lw=0.5)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.tick_params(labelsize=8, colors='gray')
    for s in ['top', 'right', 'bottom', 'left']:
        ax.spines[s].set_visible(False)

def arrow(ax, x, y, color, lw=2, label=None, ls='-'):
    ax.annotate('', xy=(x, y), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, ls=ls))
    if label:
        ax.text(x * 1.12, y * 1.12, label, color=color,
                fontsize=10, ha='center', va='center', fontweight='bold')

def angle_arc(ax, theta_start, theta_end, radius, color, label=None, label_offset=1.0):
    arc = Arc((0, 0), 2 * radius, 2 * radius,
              angle=0, theta1=np.rad2deg(theta_start),
              theta2=np.rad2deg(theta_end), color=color, lw=1.5)
    ax.add_patch(arc)
    if label:
        mid = (theta_start + theta_end) / 2
        ax.text(radius * label_offset * np.cos(mid),
                radius * label_offset * np.sin(mid),
                label, color=color, fontsize=10, ha='center', va='center')

# ── Panel 1: phi_base = arctan2(dy, dx) ───────────────────────────────────────
ax = axes[0]
ax.set_title(r'Step 1:  $\varphi_{\rm base} = \mathrm{atan2}(d_y,\,d_x)$',
             fontsize=11, pad=12)
arrow(ax, 1.0, 0.0, 'lightgray', lw=1.2, label='+x')
arrow(ax, dx, dy, 'C0', lw=2.5, label=r'$\hat d_{xy}$')
angle_arc(ax, 0, phi_base, 0.45, 'C3',
          label=r'$\varphi_{\rm base}$' + f'\n{np.rad2deg(phi_base):.0f}°',
          label_offset=1.55)
ax.text(0, -1.28,
        r'project parent direction onto xy plane,' '\n'
        r'measure angle CCW from +x axis',
        ha='center', fontsize=9, color='dimgray')

# ── Panel 2: phi = phi_base + level * TORSION ────────────────────────────────
ax = axes[1]
ax.set_title(r'Step 2:  $\varphi = \varphi_{\rm base} + \ell\cdot\tau$' +
             f'   (τ = {TORSION} rad)', fontsize=11, pad=12)
arrow(ax, dx, dy, 'lightgray', lw=1.2, label=r'$\varphi_{\rm base}$')
levels = [0, 1, 2, 3]
colors = ['C0', 'C2', 'C1', 'C4']
for lvl, c in zip(levels, colors):
    phi_lvl = phi_base + lvl * TORSION
    x, y = np.cos(phi_lvl), np.sin(phi_lvl)
    arrow(ax, x, y, c, lw=2,
          label=rf'$\ell={lvl}$' + f'\n{np.rad2deg(phi_lvl):.0f}°')
angle_arc(ax, phi_base, phi_base + 3 * TORSION, 0.30, 'C3',
          label=r'$3\tau$', label_offset=1.5)
ax.text(0, -1.28,
        r'each depth level rotates the whole branching plane by τ' '\n'
        r'(prevents nested generations from stacking)',
        ha='center', fontsize=9, color='dimgray')

# ── Panel 3: child placement ──────────────────────────────────────────────────
ax = axes[2]
ax.set_title(rf'Step 3:  child $i$ at  $\theta_i = \frac{{2\pi i}}{{k}}+\varphi+\xi$,  $k={K}$',
             fontsize=11, pad=12)
phi = phi_base + 1 * TORSION  # show one level deep, for example
arrow(ax, np.cos(phi), np.sin(phi), 'lightgray', lw=1.2, label=r'$\varphi$')
# k = 2 children at 2*pi*i/k + phi
child_colors = ['C2', 'C1']
jitter_range = 0.25  # rng()*0.5 - 0.25 ∈ [-0.25, +0.25] rad
for i in range(K):
    theta_i = 2 * np.pi * i / K + phi
    x, y = np.cos(theta_i), np.sin(theta_i)
    arrow(ax, x, y, child_colors[i], lw=2.5,
          label=rf'child {i}' + f'\n{np.rad2deg(theta_i):.0f}°')
    # jitter wedge
    wedge = Wedge((0, 0), 1.0,
                  np.rad2deg(theta_i - jitter_range),
                  np.rad2deg(theta_i + jitter_range),
                  facecolor=child_colors[i], alpha=0.20, edgecolor='none')
    ax.add_patch(wedge)
ax.text(0, -1.28,
        rf'k={K} children evenly spaced ($2\pi/k$ apart),' '\n'
        rf'rotated by $\varphi$, jittered by $\xi\in[-0.25,+0.25]$ rad (shaded)',
        ha='center', fontsize=9, color='dimgray')

plt.tight_layout()
out = '/Users/elile/Documents/MLML/Thesis/Artificial Holdfast/Artificial-Holdfast/phi_visualization.png'
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f"saved {out}")
