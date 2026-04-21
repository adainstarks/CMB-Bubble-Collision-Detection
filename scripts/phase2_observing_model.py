"""Remediated observing-model utilities for CMB bubble screening.

Assumptions
-----------
* The signal model is a fractional temperature modulation, following the
  Feeney-Johnson-Mortlock-Peiris template family in Phys. Rev. D 84, 043507
  (2011), arXiv:1012.3667.
* Synthetic Gaussian CMB skies are generated in Kelvin from CAMB TT spectra.
* HEALPix pixel-window convolution is enabled during synthesis, following the
  HEALPix pixel-window convention.
* The canonical Planck cleaned-map beam is represented by a 5 arcmin Gaussian
  FWHM in harmonic space, per Planck Collaboration IV (2020), arXiv:1807.06208.
* Patch-space Gaussian beam smoothing is legacy-only and is not used by the
  remediated production path.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import camb
import healpy as hp
import numpy as np

from phase_config import (
    DEFAULTS,
    DEFAULT_INJECTION_CONVENTION,
    FLOAT_GEOMETRY_DTYPE,
    FLOAT_STORAGE_DTYPE,
    INJECTION_CONVENTION_NOTES,
    NSIDE_WORKING,
    PATCH_PIX,
    PLANCK_CMB_BEAM_FWHM,
    PROVENANCE_SCHEMA_VERSION,
    RESO_ARCMIN,
    beam_fwhm_rad,
    validate_patch_temperature_scale,
)
from phase2_signal_model import add_fractional_signal_to_patch, bubble_collision_signal


PLANCK_2018_BASE_PARAMS = {
    "H0": 67.36,
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "tau": 0.0544,
    "As": 2.0989031673191437e-9,
    "ns": 0.9649,
}


@contextlib.contextmanager
def temporary_numpy_seed(seed: int):
    """Temporarily seed NumPy's legacy RNG for healpy APIs.

    healpy releases used by this project draw random alms through NumPy's
    module-level RNG. This context makes synthesis deterministic while restoring
    the caller's global RNG state immediately afterward.
    """

    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def camb_tt_cls(
    *,
    lmax: int,
    params: dict[str, float] | None = None,
    lens_potential_accuracy: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return lensed scalar TT C_ell values in Kelvin^2.

    The CAMB accuracy settings are explicit so regenerated datasets carry a
    reproducible transfer-function provenance record.
    """

    if lmax < 2:
        raise ValueError("lmax must be at least 2.")
    params = dict(PLANCK_2018_BASE_PARAMS if params is None else params)
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=params["H0"],
        ombh2=params["ombh2"],
        omch2=params["omch2"],
        tau=params["tau"],
    )
    pars.InitPower.set_params(As=params["As"], ns=params["ns"])
    pars.set_for_lmax(int(lmax), lens_potential_accuracy=int(lens_potential_accuracy))
    pars.Accuracy.AccuracyBoost = 1.5
    pars.Accuracy.lSampleBoost = 1.5
    results = camb.get_results(pars)
    cls_tt = results.get_cmb_power_spectra(pars, CMB_unit="K", raw_cl=True)["lensed_scalar"][:, 0]
    provenance = {
        "params": params,
        "lmax": int(lmax),
        "CMB_unit": "K",
        "raw_cl": True,
        "spectrum": "lensed_scalar_TT",
        "lens_potential_accuracy": int(lens_potential_accuracy),
        "AccuracyBoost": float(pars.Accuracy.AccuracyBoost),
        "lSampleBoost": float(pars.Accuracy.lSampleBoost),
    }
    return np.asarray(cls_tt[: int(lmax) + 1], dtype=FLOAT_GEOMETRY_DTYPE), provenance


def synthesize_cmb_maps(
    *,
    num_realizations: int,
    rng: np.random.Generator,
    nside: int = NSIDE_WORKING,
    lmax: int | None = None,
    beam_fwhm_arcmin: float = DEFAULTS.beam_fwhm_arcmin,
    pixwin: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate beam-and-pixel-window-convolved Gaussian CMB maps."""

    if num_realizations <= 0:
        raise ValueError("num_realizations must be positive.")
    if lmax is None:
        lmax = 3 * int(nside) - 1
    cls_tt, camb_provenance = camb_tt_cls(lmax=int(lmax))
    maps = np.empty((int(num_realizations), hp.nside2npix(int(nside))), dtype=FLOAT_STORAGE_DTYPE)
    seeds = rng.integers(0, 2**32 - 1, size=int(num_realizations), dtype=np.uint32)
    fwhm_rad = beam_fwhm_rad(float(beam_fwhm_arcmin)) if beam_fwhm_arcmin > 0.0 else 0.0
    for idx, seed_i in enumerate(seeds):
        with temporary_numpy_seed(int(seed_i)):
            maps[idx] = hp.synfast(
                cls_tt,
                int(nside),
                lmax=int(lmax),
                new=True,
                pixwin=bool(pixwin),
                fwhm=float(fwhm_rad),
            ).astype(FLOAT_STORAGE_DTYPE)
    provenance = {
        "kind": "camb_synfast",
        "nside": int(nside),
        "lmax": int(lmax),
        "beam_fwhm_arcmin": float(beam_fwhm_arcmin),
        "beam_domain": "synfast_harmonic",
        "pixwin": bool(pixwin),
        "pixel_window_policy": "synfast_pixwin_true" if pixwin else "synfast_pixwin_false",
        "seeds": [int(x) for x in seeds.tolist()],
        "camb": camb_provenance,
    }
    return maps, provenance


def galactic_unit_vector(glon_deg: float, glat_deg: float) -> np.ndarray:
    """Return a 3D unit vector for Galactic lon/lat in degrees."""

    lon = np.radians(float(glon_deg))
    lat = np.radians(float(glat_deg))
    return np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        dtype=FLOAT_GEOMETRY_DTYPE,
    )


def offset_signal_vector(
    glon_deg: float,
    glat_deg: float,
    *,
    center_x_pix: float,
    center_y_pix: float,
    patch_pix: int = PATCH_PIX,
    reso_arcmin: float = RESO_ARCMIN.to_value("arcmin"),
) -> np.ndarray:
    """Map a gnomonic patch pixel offset to an approximate sky direction."""

    center = galactic_unit_vector(glon_deg, glat_deg)
    lon = np.radians(float(glon_deg))
    lat = np.radians(float(glat_deg))
    east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=FLOAT_GEOMETRY_DTYPE)
    north = np.array(
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        dtype=FLOAT_GEOMETRY_DTYPE,
    )
    patch_center = (int(patch_pix) - 1) / 2.0
    dx = np.tan(np.radians((float(center_x_pix) - patch_center) * float(reso_arcmin) / 60.0))
    dy = np.tan(np.radians((float(center_y_pix) - patch_center) * float(reso_arcmin) / 60.0))
    vec = center + dx * east + dy * north
    vec /= np.linalg.norm(vec)
    return vec


def inject_signal_on_sphere(
    base_map: np.ndarray,
    *,
    glon_deg: float,
    glat_deg: float,
    z0: float,
    zcrit: float,
    theta_crit_deg: float,
    edge_sigma_deg: float = 0.0,
    center_x_pix: float | None = None,
    center_y_pix: float | None = None,
    injection_convention: str = DEFAULT_INJECTION_CONVENTION,
) -> np.ndarray:
    """Inject a Feeney fractional-temperature signal on the HEALPix sphere.

    The default convention is the Feeney et al. (2011) full-temperature
    modulation recorded by ``DEFAULT_INJECTION_CONVENTION``.
    """

    out = np.asarray(base_map, dtype=FLOAT_GEOMETRY_DTYPE).copy()
    validate_patch_temperature_scale(out, name="base_map")
    nside = hp.get_nside(out)
    if center_x_pix is None:
        center_x_pix = (PATCH_PIX - 1) / 2.0
    if center_y_pix is None:
        center_y_pix = (PATCH_PIX - 1) / 2.0
    signal_vec = offset_signal_vector(
        glon_deg,
        glat_deg,
        center_x_pix=float(center_x_pix),
        center_y_pix=float(center_y_pix),
    )
    radius = np.radians(float(theta_crit_deg))
    pix = hp.query_disc(nside, signal_vec, radius, inclusive=True)
    pix_vec = np.asarray(hp.pix2vec(nside, pix), dtype=FLOAT_GEOMETRY_DTYPE).T
    theta = np.arccos(np.clip(pix_vec @ signal_vec, -1.0, 1.0))
    signal = bubble_collision_signal(
        theta,
        float(z0),
        float(zcrit),
        radius,
        edge_sigma_deg=float(edge_sigma_deg),
    )
    out[pix] = add_fractional_signal_to_patch(
        out[pix],
        signal,
        injection_convention=injection_convention,
    )
    validate_patch_temperature_scale(out, name="injected_map")
    return out.astype(FLOAT_STORAGE_DTYPE)


def smooth_map_harmonic(
    hp_map: np.ndarray,
    *,
    beam_fwhm_arcmin: float = DEFAULTS.beam_fwhm_arcmin,
    lmax: int | None = None,
) -> np.ndarray:
    """Apply a Gaussian beam in harmonic space."""

    if beam_fwhm_arcmin <= 0.0:
        return np.asarray(hp_map, dtype=FLOAT_STORAGE_DTYPE)
    nside = hp.get_nside(hp_map)
    if lmax is None:
        lmax = 3 * int(nside) - 1
    smoothed = hp.smoothing(
        np.asarray(hp_map, dtype=FLOAT_GEOMETRY_DTYPE),
        fwhm=beam_fwhm_rad(float(beam_fwhm_arcmin)),
        lmax=int(lmax),
    )
    return np.asarray(smoothed, dtype=FLOAT_STORAGE_DTYPE)


def project_patch(
    hp_map: np.ndarray,
    glon_deg: float,
    glat_deg: float,
    *,
    validate_temperature_scale: bool = True,
) -> np.ndarray:
    """Extract a gnomonic patch using the canonical projection geometry.

    Parameters
    ----------
    validate_temperature_scale:
        Keep enabled for physical CMB-temperature patches. Disable only for
        derived auxiliary channels, such as matched-filter or Wiener-response
        fields, whose values are not thermodynamic temperatures.
    """

    patch = hp.gnomview(
        hp_map,
        rot=(float(glon_deg), float(glat_deg)),
        reso=RESO_ARCMIN.to_value("arcmin"),
        xsize=PATCH_PIX,
        return_projected_map=True,
        no_plot=True,
    )
    if bool(validate_temperature_scale):
        validate_patch_temperature_scale(patch, name="projected_patch")
    return np.asarray(patch, dtype=FLOAT_STORAGE_DTYPE)


def remove_real_map_low_modes(hp_map: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Remove monopole/dipole from a real cleaned CMB map before patching."""

    work = np.asarray(hp_map, dtype=FLOAT_GEOMETRY_DTYPE).copy()
    if mask is not None:
        masked = np.asarray(mask) <= 0
        work[masked] = hp.UNSEEN
    cleaned = hp.remove_dipole(work, bad=hp.UNSEEN)
    cleaned = hp.remove_monopole(cleaned, bad=hp.UNSEEN)
    if mask is not None:
        cleaned[np.asarray(mask) <= 0] = hp.UNSEEN
    return np.asarray(cleaned, dtype=FLOAT_STORAGE_DTYPE)


def write_observing_model_provenance(path: str | Path, provenance: dict[str, Any]) -> None:
    """Write an observing-model provenance JSON next to generated artifacts."""

    payload = dict(provenance)
    payload["references"] = {
        "Feeney_PRD_2011": "https://arxiv.org/abs/1012.3667",
        "Planck_2018_IV": "https://arxiv.org/abs/1807.06208",
        "HEALPix_synfast": "https://healpix.sourceforge.io/html/fac_synfast.htm",
        "HEALPix_pixel_window": "https://healpix.sourceforge.io/doc/html/sub_pixel_window.htm",
    }
    payload["provenance_schema_version"] = PROVENANCE_SCHEMA_VERSION
    payload["injection_convention"] = DEFAULT_INJECTION_CONVENTION
    payload["injection_convention_note"] = INJECTION_CONVENTION_NOTES[DEFAULT_INJECTION_CONVENTION]
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
