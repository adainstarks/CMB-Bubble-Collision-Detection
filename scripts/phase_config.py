"""Shared physical and pipeline constants for the bubble-screening project.

Assumptions
-----------
* This repository implements a CMB bubble-collision candidate screener, not a
  cosmological detection claim or Bayesian evidence calculation.
* Signal amplitudes are dimensionless fractional temperature modulations
  (Delta T / T). Temperatures stored in HDF5 patches are Kelvin.
* Canonical Planck cleaned CMB products are treated as 5 arcmin FWHM maps, as
  described in Planck Collaboration IV (2020), arXiv:1807.06208.
* HEALPix pixel-window handling follows the HEALPix pixel-window convention:
  pixelized maps carry the pixel window in harmonic space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u


T_CMB = 2.7255 * u.K
NSIDE_WORKING = 256
PATCH_PIX = 256
RESO_ARCMIN = 13.0 * u.arcmin
PLANCK_CMB_BEAM_FWHM = 5.0 * u.arcmin

CANONICAL_MASK_THRESHOLD = 0.90
STRESS_MASK_THRESHOLD = 0.50
LEGACY_CLEAN_MASK_THRESHOLD = 0.95

FLOAT_GEOMETRY_DTYPE = np.float64
FLOAT_STORAGE_DTYPE = np.float32

INJECTION_CONVENTION_FEENEY2011 = "feeney2011_full_temperature_modulation"
INJECTION_CONVENTION_MCEWEN2012 = "mcewen2012_first_order_additive"
INJECTION_CONVENTIONS = (
    INJECTION_CONVENTION_FEENEY2011,
    INJECTION_CONVENTION_MCEWEN2012,
)
DEFAULT_INJECTION_CONVENTION = INJECTION_CONVENTION_FEENEY2011
PROVENANCE_SCHEMA_VERSION = "injection_convention_v1"
INJECTION_METADATA_REQUIRED_AFTER_UTC = "2026-04-20T00:00:00Z"
INJECTION_CONVENTION_NOTES = {
    INJECTION_CONVENTION_FEENEY2011: (
        "Apply the Feeney et al. (2011, PRD 84, 043507) full-temperature "
        "modulation: deltaT = (1 + f(n)) * (T_CMB + deltaT_cmb) - T_CMB."
    ),
    INJECTION_CONVENTION_MCEWEN2012: (
        "Apply the McEwen et al. (2012, PRD 85, 103502) first-order additive "
        "template used for harmonic-space matched-filter construction: "
        "deltaT = deltaT_cmb + f(n) * T_CMB."
    ),
}


@dataclass(frozen=True)
class RemediationDefaults:
    """Default policy choices for remediated v1 products."""

    output_subdir: str = "remediated_v1"
    mask_threshold: float = CANONICAL_MASK_THRESHOLD
    stress_mask_threshold: float = STRESS_MASK_THRESHOLD
    beam_fwhm_arcmin: float = PLANCK_CMB_BEAM_FWHM.to_value(u.arcmin)
    nside: int = NSIDE_WORKING
    patch_pix: int = PATCH_PIX
    reso_arcmin: float = RESO_ARCMIN.to_value(u.arcmin)
    pixel_window_policy: str = "synfast_pixwin_true"
    beam_domain: str = "harmonic_sphere"


DEFAULTS = RemediationDefaults()


def beam_fwhm_rad(fwhm_arcmin: float | None = None) -> float:
    """Return a beam FWHM in radians."""

    if fwhm_arcmin is None:
        fwhm_arcmin = DEFAULTS.beam_fwhm_arcmin
    return (float(fwhm_arcmin) * u.arcmin).to_value(u.rad)


def reso_arcmin_value() -> float:
    """Return the working gnomonic pixel scale in arcmin."""

    return RESO_ARCMIN.to_value(u.arcmin)


def patch_half_width_deg() -> float:
    """Approximate half-width of a square gnomonic patch in degrees."""

    return 0.5 * PATCH_PIX * RESO_ARCMIN.to_value(u.deg)


def min_component_area_pixels(theta_min_deg: float = 5.0, fraction: float = 0.01) -> int:
    """Minimum coherent area floor for image-level detection.

    The floor is a fraction of the smallest canonical bubble-disc footprint.
    It prevents isolated hot pixels from defining image-level detections.
    """

    radius_pix = float(theta_min_deg) * 60.0 / RESO_ARCMIN.to_value(u.arcmin)
    area = np.pi * radius_pix * radius_pix
    return max(1, int(np.ceil(float(fraction) * area)))


def validate_patch_temperature_scale(patch: np.ndarray, *, name: str = "patch") -> None:
    """Raise for non-finite or physically implausible CMB temperature patches."""

    arr = np.asarray(patch)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_abs > 1.0:
        raise ValueError(
            f"{name} max |T|={max_abs:.3g} K is not a CMB anisotropy-scale patch. "
            "Expected Kelvin anisotropies, not microkelvin or full-temperature units."
        )
