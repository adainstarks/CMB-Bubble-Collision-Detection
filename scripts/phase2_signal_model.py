"""
Phase 2 Preview: Bubble Collision Signal Model.

Assumptions
-----------
* `z0` and `zcrit` are dimensionless fractional temperature modulations
  (Delta T / T), not Kelvin or microkelvin amplitudes.
* The implemented template is the linearized spherical-cap profile from
  Feeney, Johnson, Mortlock & Peiris, Phys. Rev. D 84, 043507 (2011),
  arXiv:1012.3667. The short PRL summary arXiv:1012.1995 is not cited as
  the equation source.
* Patch arrays are CMB temperature anisotropies in Kelvin.

Implements the bubble collision temperature modulation from:
    Feeney, Johnson, Mortlock & Peiris (2011), arXiv:1012.3667

The signal model (centered on the north pole) is:

    dT/T = [ (z_crit - z_0 cos(theta_crit)) / (1 - cos(theta_crit))
           + (z_0 - z_crit) / (1 - cos(theta_crit)) * cos(theta) ]
           * Heaviside(theta_crit - theta)

Parameters:
    z_0       : amplitude at the center of the collision disk
    z_crit    : temperature discontinuity at the causal boundary
    theta_crit: angular radius of the collision disk
    theta_0   : galactic longitude of disk center (degrees)
    phi_0     : galactic latitude of disk center (degrees)

Usage (from project root, with cmb conda env activated):
    python scripts/phase2_signal_model.py
"""

import os
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.special import erf

from phase_config import PATCH_PIX, RESO_ARCMIN, T_CMB

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
SMICA_FILE = os.path.join(DATA_DIR, "COM_CMB_IQU-smica_2048_R3.00_full.fits")

RESO_ARCMIN = float(RESO_ARCMIN.to_value("arcmin"))
T_CMB_K = float(T_CMB.to_value("K"))


def causal_boundary_window(theta, theta_crit, edge_sigma_deg=0.0):
    """
    Return a hard or softly smeared causal boundary window.

    For edge_sigma_deg = 0, this reduces to the sharp Heaviside step used in
    Feeney et al. (2011, arXiv:1012.3667). For edge_sigma_deg > 0, we smooth the boundary with
    a Gaussian CDF transition of width sigma to make the training set more
    robust to sub-degree edge smearing.
    """
    if edge_sigma_deg <= 0.0:
        return (theta <= theta_crit).astype(float)

    edge_sigma = np.radians(edge_sigma_deg)
    scaled = (theta_crit - theta) / (np.sqrt(2.0) * edge_sigma)
    return 0.5 * (1.0 + erf(scaled))


def bubble_collision_signal(theta, z0, zcrit, theta_crit, edge_sigma_deg=0.0):
    """
    Feeney et al. (2011, arXiv:1012.3667): temperature modulation from a
    bubble collision.

    Parameters
    ----------
    theta : ndarray
        Angular distance from the collision center (radians).
    z0 : float
        Amplitude of temperature modulation at disk center (dimensionless dT/T).
    zcrit : float
        Temperature discontinuity at the causal boundary.
    theta_crit : float
        Angular radius of the collision disk (radians).

    edge_sigma_deg : float, optional
        Width of Gaussian edge smearing in degrees. A value of 0 uses the
        original sharp causal boundary.

    Returns
    -------
    signal : ndarray
        Fractional temperature modulation dT/T at each point.
    """
    denom = 1.0 - np.cos(theta_crit)
    intercept = (zcrit - z0 * np.cos(theta_crit)) / denom
    slope = (z0 - zcrit) / denom
    modulation = intercept + slope * np.cos(theta)
    window = causal_boundary_window(theta, theta_crit, edge_sigma_deg=edge_sigma_deg)
    return modulation * window


def make_plane_coordinate_grids(npix, reso_arcmin):
    """
    Return tangent-plane x/y coordinates for each patch pixel.

    The coordinates are expressed in gnomonic-plane units, so a point at angle
    alpha from the patch center along one axis sits at tan(alpha) on the plane.
    """
    patch_center = (npix - 1) / 2.0
    iy, ix = np.mgrid[0:npix, 0:npix]
    dx_angle_rad = np.radians((ix - patch_center) * reso_arcmin / 60.0)
    dy_angle_rad = np.radians((iy - patch_center) * reso_arcmin / 60.0)
    return np.tan(dx_angle_rad), np.tan(dy_angle_rad)


def make_angular_distance_grid(npix, reso_arcmin, center_x_pix=None, center_y_pix=None):
    """
    Build a 2D grid of angular distances (radians) from an arbitrary point in a
    gnomonic (tangent-plane) patch.

    When the signal center is not at the patch midpoint, we compute the angular
    separation on the sphere by mapping both the bubble center and every pixel
    back to 3D directions from the same tangent plane.
    """
    patch_center = (npix - 1) / 2.0
    if center_x_pix is None:
        center_x_pix = patch_center
    if center_y_pix is None:
        center_y_pix = patch_center

    x_plane, y_plane = make_plane_coordinate_grids(npix, reso_arcmin)
    center_dx_angle = np.radians((center_x_pix - patch_center) * reso_arcmin / 60.0)
    center_dy_angle = np.radians((center_y_pix - patch_center) * reso_arcmin / 60.0)
    center_x_plane = np.tan(center_dx_angle)
    center_y_plane = np.tan(center_dy_angle)

    pixel_vec = np.stack((x_plane, y_plane, np.ones_like(x_plane)), axis=0)
    pixel_vec /= np.linalg.norm(pixel_vec, axis=0, keepdims=True)

    center_vec = np.array([center_x_plane, center_y_plane, 1.0], dtype=np.float64)
    center_vec /= np.linalg.norm(center_vec)

    cos_theta = np.clip(np.sum(pixel_vec * center_vec[:, None, None], axis=0), -1.0, 1.0)
    return np.arccos(cos_theta)


def inject_signal_into_patch(
    patch,
    z0,
    zcrit,
    theta_crit_deg,
    edge_sigma_deg=0.0,
    center_x_pix=None,
    center_y_pix=None,
):
    """
    Inject a bubble collision signal into a flat-sky patch.
    
    Uses multiplicative injection with a Feeney-style fractional modulation:
        δT = (1 + f(n̂)) * (T0 + δT_cmb) - T0

    The Planck SMICA patch is a temperature-anisotropy map, so we reconstruct the
    full temperature as T0 + δT_cmb, apply the modulation, and subtract T0 again.
    """
    patch = np.asarray(patch, dtype=np.float64)
    theta_grid = make_angular_distance_grid(
        patch.shape[0],
        RESO_ARCMIN,
        center_x_pix=center_x_pix,
        center_y_pix=center_y_pix,
    )
    theta_crit = np.radians(theta_crit_deg)
    signal = bubble_collision_signal(
        theta_grid,
        z0,
        zcrit,
        theta_crit,
        edge_sigma_deg=edge_sigma_deg,
    )
    injected = (1.0 + signal) * (T_CMB_K + patch) - T_CMB_K
    return injected, signal


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── Load SMICA and degrade ─────────────────────────────────────────
    print("Loading Planck SMICA map...")
    smica = hp.read_map(SMICA_FILE, field=0)
    smica_256 = hp.ud_grade(smica, 256)

    # Extract a clean high-latitude patch
    glon, glat = 120.0, 55.0
    print(f"Extracting patch at (l={glon}, b={glat})...")
    plt.figure()
    clean_patch = hp.gnomview(
        smica_256,
        rot=(glon, glat),
        reso=RESO_ARCMIN,
        xsize=PATCH_PIX,
        return_projected_map=True,
        no_plot=True,
    )
    plt.close()

    # ── Figure 1: The signal model itself at three angular scales ──────
    print("Generating signal model figure...")
    theta_1d = np.linspace(0, np.radians(30), 500)
    z0, zcrit = 5e-5, -3e-5

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, tc_deg in zip(axes, [5, 10, 25]):
        sig = bubble_collision_signal(theta_1d, z0, zcrit, np.radians(tc_deg))
        ax.plot(np.degrees(theta_1d), sig * 1e5, color="#2563eb", linewidth=2)
        ax.axvline(tc_deg, color="#dc2626", linestyle="--", alpha=0.7,
                   label=r"$\theta_{\rm crit}$")
        ax.set_xlabel(r"$\theta$ (degrees)", fontsize=12)
        ax.set_title(
            rf"$\theta_{{\rm crit}} = {tc_deg}°$",
            fontsize=13,
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\delta T / T \;\; (\times 10^{-5})$", fontsize=12)
    fig.suptitle(
        r"Feeney et al. (2011) Eq. 1: Bubble Collision Signal"
        f"\n"
        rf"$z_0 = {z0:.0e},\;\; z_{{\rm crit}} = {zcrit:.0e}$",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    path1 = os.path.join(PLOT_DIR, "06_signal_model_profiles.png")
    plt.savefig(path1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path1}")

    # ── Figure 2: Signal injected into real Planck patch ───────────────
    # Sweep amplitude from barely visible to obvious
    print("Generating injection figure...")
    theta_crit_deg = 10.0
    amplitudes = [
        (1e-5, -5e-6,  "Weak: below noise floor"),
        (3e-5, -2e-5,  "Moderate: at detection threshold"),
        (8e-5, -5e-5,  "Strong: clearly visible"),
    ]

    vmin = np.nanpercentile(clean_patch, 1)
    vmax = np.nanpercentile(clean_patch, 99)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Top row: signal templates alone (what the collision looks like in isolation)
    ax = axes[0, 0]
    ax.set_visible(False)  # placeholder for alignment

    for i, (z0_i, zc_i, label) in enumerate(amplitudes):
        ax = axes[0, i + 1]
        _, sig = inject_signal_into_patch(clean_patch, z0_i, zc_i, theta_crit_deg)
        im = ax.imshow(sig, cmap="RdBu_r", origin="lower",
                       extent=[-21.3, 21.3, -21.3, 21.3])
        crit_pix = theta_crit_deg
        circle = Circle((0, 0), crit_pix, fill=False, edgecolor="lime",
                        linewidth=1.5, linestyle="--")
        ax.add_patch(circle)
        ax.set_title(f"Signal only\n{label}", fontsize=11)
        ax.set_xlabel("degrees")
        plt.colorbar(im, ax=ax, shrink=0.8, label=r"$\delta T / T$")

    # Bottom row: clean patch, then three injected patches
    ax = axes[1, 0]
    im = ax.imshow(clean_patch, cmap="RdBu_r", origin="lower",
                   vmin=vmin, vmax=vmax,
                   extent=[-21.3, 21.3, -21.3, 21.3])
    ax.set_title("Clean Planck patch\n(no injection)", fontsize=11)
    ax.set_xlabel("degrees")
    ax.set_ylabel("degrees")
    plt.colorbar(im, ax=ax, shrink=0.8, label="K")

    for i, (z0_i, zc_i, label) in enumerate(amplitudes):
        ax = axes[1, i + 1]
        injected, _ = inject_signal_into_patch(clean_patch, z0_i, zc_i, theta_crit_deg)
        im = ax.imshow(injected, cmap="RdBu_r", origin="lower",
                       vmin=vmin, vmax=vmax,
                       extent=[-21.3, 21.3, -21.3, 21.3])
        circle = Circle((0, 0), theta_crit_deg, fill=False, edgecolor="lime",
                        linewidth=1.5, linestyle="--")
        ax.add_patch(circle)
        ax.set_title(f"Injected\n{label}", fontsize=11)
        ax.set_xlabel("degrees")
        plt.colorbar(im, ax=ax, shrink=0.8, label="K")

    fig.suptitle(
        r"Bubble Collision Injection into Planck SMICA — $\theta_{\rm crit} = 10°$"
        f"\nFeeney et al. (2011) Eq. 1  |  Patch at (l={glon}°, b={glat}°)",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    path2 = os.path.join(PLOT_DIR, "07_signal_injection_demo.png")
    plt.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path2}")

    # ── Figure 3: Parameter space grid (z0 vs zcrit) ──────────────────
    # Recreates the spirit of Feeney Fig. 2 parameter sweeps
    print("Generating parameter space grid...")
    z0_values = [1e-5, 3e-5, 8e-5]
    zc_values = [-5e-6, -3e-5, -8e-5]

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    for row, zc in enumerate(zc_values):
        for col, z0_v in enumerate(z0_values):
            ax = axes[row, col]
            injected, _ = inject_signal_into_patch(
                clean_patch, z0_v, zc, theta_crit_deg
            )
            ax.imshow(injected, cmap="RdBu_r", origin="lower",
                      vmin=vmin, vmax=vmax,
                      extent=[-21.3, 21.3, -21.3, 21.3])
            circle = Circle((0, 0), theta_crit_deg, fill=False,
                            edgecolor="lime", linewidth=1.2, linestyle="--")
            ax.add_patch(circle)
            ax.set_title(
                rf"$z_0 = {z0_v:.0e},\; z_{{\rm crit}} = {zc:.0e}$",
                fontsize=10,
            )
            if col == 0:
                ax.set_ylabel("degrees")
            if row == 2:
                ax.set_xlabel("degrees")

    fig.suptitle(
        r"Parameter Space: $z_0$ vs $z_{\rm crit}$ — "
        r"$\theta_{\rm crit} = 10°$, injected into Planck SMICA",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    path3 = os.path.join(PLOT_DIR, "08_parameter_space_grid.png")
    plt.savefig(path3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path3}")

    print("\nDone! New plots:")
    print(f"  {path1}  — signal profile at three angular scales")
    print(f"  {path2}  — injection demo (weak / moderate / strong)")
    print(f"  {path3}  — parameter space grid (z0 vs zcrit)")


if __name__ == "__main__":
    main()
