"""Visualize the Rodrigues rotation formula in four steps.

The formula:
    R v = v_par + cos(θ) v_perp + sin(θ) (a × v)

geometrically:
1. Set up the problem: rotate v around axis a by angle θ.
2. Decompose v = v_par + v_perp.  (v_par stays fixed; v_perp rotates.)
3. In the 2D plane ⊥ a, v_perp rotates by θ.  The basis {v_perp, a×v} is orthogonal.
4. Apply to a cylinder along z to align with a segment direction d (the haptera case).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, start, end, **kwargs):
        super().__init__((0, 0), (0, 0), **kwargs)
        self._s = np.asarray(start, dtype=float)
        self._e = np.asarray(end, dtype=float)
    def do_3d_projection(self, renderer=None):
        xs = (self._s[0], self._e[0])
        ys = (self._s[1], self._e[1])
        zs = (self._s[2], self._e[2])
        x2, y2, _ = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((x2[0], y2[0]), (x2[1], y2[1]))
        return min(self._s[2], self._e[2])


def arr3d(ax, s, e, color, lw=2.5, label=None, label_off=(0, 0, 0),
          ls='-', alpha=1.0, fontsize=12):
    a = Arrow3D(s, e, mutation_scale=14, lw=lw, arrowstyle='-|>',
                color=color, linestyle=ls, alpha=alpha, zorder=10)
    ax.add_artist(a)
    if label:
        ax.text(e[0]+label_off[0], e[1]+label_off[1], e[2]+label_off[2],
                label, color=color, fontsize=fontsize, fontweight='bold')


def line3d(ax, s, e, color='gray', lw=1, ls='-', alpha=1.0):
    ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]],
            color=color, lw=lw, ls=ls, alpha=alpha)


def setup3d(ax, title, lim=1.2):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, pad=8, fontsize=11)
    line3d(ax, (-lim, 0, 0), (lim, 0, 0), 'lightgray', lw=0.5, ls=':', alpha=0.5)
    line3d(ax, (0, -lim, 0), (0, lim, 0), 'lightgray', lw=0.5, ls=':', alpha=0.5)
    line3d(ax, (0, 0, -lim), (0, 0, lim), 'lightgray', lw=0.5, ls=':', alpha=0.5)
    ax.text(lim*0.96, 0, -0.06, 'x', color='gray', fontsize=8)
    ax.text(0, lim*0.96, -0.06, 'y', color='gray', fontsize=8)
    ax.text(0, 0, lim*0.97, 'z', color='gray', fontsize=8)
    ax.view_init(elev=22, azim=35)


def rodrigues(a, theta):
    c, s = np.cos(theta), np.sin(theta)
    K = np.array([[ 0,    -a[2],  a[1]],
                  [ a[2],  0,    -a[0]],
                  [-a[1],  a[0],  0  ]])
    return np.eye(3) + s*K + (1-c) * (K @ K)


def perp_basis(a):
    arbitrary = np.array([1., 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1., 0])
    e1 = np.cross(a, arbitrary); e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2


# ── Example for panels 1-3:  rotate v = ẑ around â = (1,1,1)/√3 by 60° ──────
a_hat = np.array([1., 1., 1.]) / np.sqrt(3)
v = np.array([0., 0., 1.])
theta = np.deg2rad(60.0)
v_par = np.dot(v, a_hat) * a_hat
v_perp = v - v_par
a_cross_v = np.cross(a_hat, v)
v_perp_rot = np.cos(theta) * v_perp + np.sin(theta) * a_cross_v
v_prime = v_par + v_perp_rot
assert np.allclose(rodrigues(a_hat, theta) @ v, v_prime, atol=1e-9)


fig = plt.figure(figsize=(15, 13))


# ── Panel 1: the rotation problem ─────────────────────────────────────────────
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
setup3d(ax1, r'1. Rotate $\vec v$ around axis $\hat a$ by angle $\theta$')

arr3d(ax1, -a_hat * 0.3, a_hat * 1.15, 'C3',
      label=r'$\hat a$', label_off=(0.05, 0, 0.05), lw=2.0)
arr3d(ax1, [0, 0, 0], v, 'C0',
      label=r'$\vec v$', label_off=(0.04, 0.04, 0.06))
arr3d(ax1, [0, 0, 0], v_prime, 'C2',
      label=r"$\vec v\,' = R\vec v$", label_off=(0.04, -0.20, 0.02))

# Full orbit circle (rotating v around a by all angles 0..2π)
phi = np.linspace(0, 2*np.pi, 120)
orbit = np.array([
    np.cos(p) * v_perp + np.sin(p) * a_cross_v + v_par
    for p in phi
])
ax1.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], 'gray', lw=0.9, alpha=0.45)
ax1.text(orbit[15, 0]+0.05, orbit[15, 1]+0.0, orbit[15, 2]+0.05,
         'orbit of $\\vec v$', color='gray', fontsize=9)

# theta arc (small concentric arc inside the orbit, for readability)
n_arc = 30
arc = np.array([
    np.cos(t) * v_perp + np.sin(t) * a_cross_v + v_par
    for t in np.linspace(0, theta, n_arc)
])
arc_inner = (arc - v_par) * 0.50 + v_par
ax1.plot(arc_inner[:, 0], arc_inner[:, 1], arc_inner[:, 2], 'C7', lw=2)
mid = arc_inner[len(arc_inner)//2]
ax1.text(mid[0]+0.02, mid[1]+0.02, mid[2]+0.04,
         rf'$\theta = {np.rad2deg(theta):.0f}°$', color='C7', fontsize=11)


# ── Panel 2: decomposition v = v_par + v_perp ────────────────────────────────
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
setup3d(ax2, r'2. Decompose $\vec v = \vec v_\parallel + \vec v_\perp$' '\n'
              r'$\vec v_\parallel$ is fixed under rotation;  $\vec v_\perp$ rotates')

# axis â (faint extended line)
line3d(ax2, -a_hat*0.5, a_hat*1.2, 'C3', lw=1.5, alpha=0.6)
ax2.text(a_hat[0]*1.25, a_hat[1]*1.25, a_hat[2]*1.25, r'$\hat a$',
         color='C3', fontsize=12, fontweight='bold')

# Plane perpendicular to â passing through tip of v_par
e1_pl, e2_pl = perp_basis(a_hat)
S = 0.75
sgrid = np.array([[-S, S], [-S, S]])
tgrid = np.array([[-S, -S], [S, S]])
Xp = v_par[0] + sgrid * e1_pl[0] + tgrid * e2_pl[0]
Yp = v_par[1] + sgrid * e1_pl[1] + tgrid * e2_pl[1]
Zp = v_par[2] + sgrid * e1_pl[2] + tgrid * e2_pl[2]
ax2.plot_surface(Xp, Yp, Zp, alpha=0.13, color='gray', edgecolor='none')
ax2.text(v_par[0]+e1_pl[0]*0.6+e2_pl[0]*0.6,
         v_par[1]+e1_pl[1]*0.6+e2_pl[1]*0.6,
         v_par[2]+e1_pl[2]*0.6+e2_pl[2]*0.6,
         r'plane $\perp\hat a$', color='gray', fontsize=9)

# v
arr3d(ax2, [0, 0, 0], v, 'C0',
      label=r'$\vec v$', label_off=(0.04, 0.04, 0.06))
# v_par
arr3d(ax2, [0, 0, 0], v_par, 'C4',
      label=r'$\vec v_\parallel$', label_off=(0.06, 0.05, -0.05))
# v_perp at tip of v_par (tip-to-tail addition: v_par + v_perp = v)
arr3d(ax2, v_par, v_par + v_perp, 'C1',
      label=r'$\vec v_\perp$', label_off=(-0.20, -0.06, 0.04))

# parallelogram dashed sides
line3d(ax2, [0, 0, 0], v_perp, 'C1', lw=0.8, ls='--', alpha=0.45)
line3d(ax2, v_perp, v, 'C4', lw=0.8, ls='--', alpha=0.45)


# ── Panel 3: the 2D rotation in the plane ⊥ â ────────────────────────────────
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_aspect('equal')
LIM2 = 1.45
ax3.set_xlim(-LIM2, LIM2); ax3.set_ylim(-LIM2, LIM2)
ax3.set_title(r'3. Inside that plane, the rotation is just 2D'
              '\n'
              r"$\vec v_\perp\,' = \cos\theta\,\vec v_\perp + \sin\theta\,(\hat a\times\vec v)$",
              fontsize=11, pad=8)

ax3.axhline(0, color='lightgray', lw=0.5)
ax3.axvline(0, color='lightgray', lw=0.5)
phi2 = np.linspace(0, 2*np.pi, 100)
ax3.plot(np.cos(phi2), np.sin(phi2), color='lightgray', lw=0.7)

# basis labels
ax3.annotate('', xy=(1.25, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
ax3.text(1.27, -0.07, r'direction of $\vec v_\perp$', color='gray',
         fontsize=9, va='center')
ax3.annotate('', xy=(0, 1.25), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
ax3.text(0.02, 1.30, r'direction of $\hat a\times\vec v$', color='gray', fontsize=9)

# Original v_perp at (1, 0)
ax3.annotate('', xy=(1.0, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='C1', lw=2.6))
ax3.text(0.55, -0.13, r'$\vec v_\perp$', color='C1', fontsize=14, fontweight='bold')

# Rotated v_perp at (cos θ, sin θ)
ct, st = np.cos(theta), np.sin(theta)
ax3.annotate('', xy=(ct, st), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='C2', lw=2.6))
ax3.text(ct-0.18, st+0.08, r"$\vec v_\perp\,'$", color='C2',
         fontsize=14, fontweight='bold')

# component projections
ax3.plot([ct, ct], [0, st], 'C2--', lw=1, alpha=0.6)
ax3.plot([0, ct], [st, st], 'C2--', lw=1, alpha=0.6)
ax3.text(ct/2, -0.12, r'$\cos\theta$', color='C2',
         fontsize=11, ha='center')
ax3.text(-0.05, st/2, r'$\sin\theta$', color='C2',
         fontsize=11, ha='right', va='center')

# theta arc
arc = Arc((0, 0), 0.6, 0.6, angle=0, theta1=0, theta2=np.rad2deg(theta),
          color='C7', lw=1.5)
ax3.add_patch(arc)
ax3.text(0.42*np.cos(theta/2), 0.42*np.sin(theta/2)-0.02,
         rf'$\theta={np.rad2deg(theta):.0f}°$', color='C7', fontsize=11, ha='center')

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)


# ── Panel 4: haptera-specific case — rotate cylinder along ẑ to align with d̂ ─
ax4 = fig.add_subplot(2, 2, 4, projection='3d')

# Sample haptera segment direction
d_hap = np.array([0.5, 0.6, 0.7])
d_hap /= np.linalg.norm(d_hap)
z_hat = np.array([0., 0., 1.])
cross_zd = np.cross(z_hat, d_hap)
sin_t_hap = np.linalg.norm(cross_zd)
a_hat_hap = cross_zd / sin_t_hap
cos_t_hap = float(np.dot(z_hat, d_hap))
theta_hap = np.arccos(cos_t_hap)
R_hap = rodrigues(a_hat_hap, theta_hap)
assert np.allclose(R_hap @ z_hat, d_hap, atol=1e-6)

setup3d(ax4,
        rf'4. Haptera case: rotate cylinder $\hat z\!\to\!\hat d,\;\;\theta={np.rad2deg(theta_hap):.0f}°$'
        '\n'
        r'(here $\hat a = \hat z\times\hat d/\sin\theta$ lies in the $xy$-plane)')

def make_cyl(radius=0.10, length=1.10, n_u=32, n_v=2):
    u = np.linspace(0, 2*np.pi, n_u)
    v_ = np.linspace(0, length, n_v)
    U, V = np.meshgrid(u, v_)
    X = radius * np.cos(U)
    Y = radius * np.sin(U)
    Z = V
    return X, Y, Z

def rotate_surf(X, Y, Z, R):
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0)
    rot = R @ pts
    return (rot[0].reshape(X.shape),
            rot[1].reshape(Y.shape),
            rot[2].reshape(Z.shape))

X, Y, Z = make_cyl()
ax4.plot_surface(X, Y, Z, alpha=0.40, color='C0', edgecolor='none')
Xr, Yr, Zr = rotate_surf(X, Y, Z, R_hap)
ax4.plot_surface(Xr, Yr, Zr, alpha=0.65, color='C2', edgecolor='none')

arr3d(ax4, [0, 0, 0], z_hat*1.18, 'C0',
      label=r'$\hat z$', label_off=(0.02, 0.02, 0.06))
arr3d(ax4, [0, 0, 0], d_hap*1.18, 'C2',
      label=r'$\hat d$', label_off=(0.04, 0.04, 0.04))
arr3d(ax4, -a_hat_hap*0.3, a_hat_hap*0.95, 'C3',
      label=r'$\hat a$', label_off=(-0.12, -0.04, -0.06), lw=1.7)


plt.tight_layout()
out = '/Users/elile/Documents/MLML/Thesis/Artificial Holdfast/Artificial-Holdfast/rodrigues_visualization.png'
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f"saved {out}")
