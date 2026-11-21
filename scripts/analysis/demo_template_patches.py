"""Picture: the April sharp-disc template next to the transfer-function ramp, as they appear in a patch.

Signal only (no CMB), Nside 256, 5 arcmin beam and pixel window, 256 x 256 gnomonic patch at 13 arcmin.
All panels share the Sachs-Wolfe central amplitude z0 = 5e-5.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import collision_template_transfer as ct
from phase2_signal_model import bubble_collision_signal

NSIDE, LMAX, Z0 = 256, 767, 5e-5
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
tr = ct.camb_transfer(lmax=LMAX)
window = ct.gaussian_beam(LMAX, 5.0) * hp.pixwin(NSIDE, lmax=LMAX)
lon, lat = 120.0, 55.0
centre = hp.ang2vec(lon, lat, lonlat=True)
vec = np.array(hp.pix2vec(NSIDE, np.arange(hp.nside2npix(NSIDE))))
ang = np.arccos(np.clip(centre @ vec, -1, 1))

def gnom(m):
    return np.asarray(hp.gnomview(m, rot=(lon, lat), reso=13.0, xsize=256, return_projected_map=True, no_plot=True))

fig, axes = plt.subplots(2, 4, figsize=(19, 9))
theta1d = np.radians(np.linspace(0, 27, 541))
for row, tcd in enumerate((10.0, 20.0)):
    tc = math.radians(tcd)
    a = 1.0 / tr.d_star
    unit = -(a * tr.d_star / 5.0) * (1.0 - math.cos(tc))
    b = ct.b_ell(tr.ells, tr.k, tr.delta_t, tr.d_star * math.cos(tc), a_ramp=a) * (Z0 / unit)
    lmaxfull = int(tr.ells.max())
    coeff = np.zeros(lmaxfull + 1)
    coeff[tr.ells] = (2 * tr.ells + 1) / (4 * math.pi) * b * window[tr.ells]
    new_map = np.polynomial.legendre.legval(np.cos(ang), coeff)
    old0 = hp.smoothing(bubble_collision_signal(ang, Z0, 0.0, tc), fwhm=math.radians(5 / 60), lmax=LMAX)
    old1 = hp.smoothing(bubble_collision_signal(ang, Z0, Z0, tc), fwhm=math.radians(5 / 60), lmax=LMAX)
    panels = [("April template, z_crit = 0", old0), ("April template, z_crit = z0", old1), ("ramp through CAMB transfer functions", new_map)]
    vmax = max(float(np.abs(gnom(m)).max()) for _, m in panels)
    for col, (title, m) in enumerate(panels):
        ax = axes[row, col]
        im = ax.imshow(gnom(m) * 1e5, origin="lower", cmap="RdBu_r", vmin=-vmax * 1e5, vmax=vmax * 1e5, extent=[-27.7, 27.7, -27.7, 27.7])
        ax.add_patch(plt.Circle((0, 0), tcd, fill=False, ec="k", ls="--", lw=1))
        ax.set_title(f"{title}\ntheta_c = {tcd:.0f} deg", fontsize=11)
        ax.set_xlabel("deg")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Delta T / T  (x 1e-5)")
    ax = axes[row, 3]
    ax.plot(np.degrees(theta1d), bubble_collision_signal(theta1d, Z0, 0.0, tc) * 1e5, color="0.5", lw=1.6, label="April, z_crit = 0")
    ax.plot(np.degrees(theta1d), bubble_collision_signal(theta1d, Z0, Z0, tc) * 1e5, color="#c98a1b", lw=1.6, label="April, z_crit = z0")
    ax.plot(np.degrees(theta1d), np.polynomial.legendre.legval(np.cos(theta1d), coeff) * 1e5, color="#1f4e9c", lw=2.0, label="transfer-function ramp")
    ax.axvline(tcd, color="k", ls="--", lw=1)
    ax.set_xlabel("angle from centre (deg)"); ax.set_ylabel("Delta T / T  (x 1e-5)"); ax.grid(alpha=0.3)
    ax.set_title(f"profiles, theta_c = {tcd:.0f} deg", fontsize=11)
    if row == 0:
        ax.legend(frameon=False, fontsize=9)
    print(f"theta_c {tcd:.0f}: peak Delta T/T  April z_crit=0 {np.abs(old0).max():.2e}, April z_crit=z0 {np.abs(old1).max():.2e}, transfer-function ramp {np.abs(new_map).max():.2e}")
fig.suptitle("Same collision, same Sachs-Wolfe central amplitude (5e-5): April sharp-disc template and the transfer-function template", fontsize=13)
fig.tight_layout()
fig.savefig(out / "template_comparison_patches.png", dpi=95)
print("wrote", out / "template_comparison_patches.png")
