"""Classical full-sky filters for remediated Phase 3 screening.

Assumptions
-----------
* These utilities are candidate screeners. They do not compute Bayesian
  evidence for bubble collisions.
* `wiener_feeney_matched_filter` uses an axisymmetric Feeney linear cap,
  a Gaussian beam, and inverse covariance weighting
  ``B_l s_l / (B_l^2 C_l + N_l)``. That name is not used for the older
  patch-space circular correlator.
* The Feeney cap follows Feeney et al. Phys. Rev. D 84, 043507 (2011),
  arXiv:1012.3667. The harmonic weighting follows the standard spherical
  matched-filter construction used for azimuthal CMB features, with covariance
  weighting as in Osborne, Senatore & Smith, arXiv:1305.1970.
* `smhw_screen` is a scale-space spherical Mexican-hat/Laplacian-of-Gaussian
  context screen, not a needlet or Wiener matched filter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import healpy as hp
import numpy as np
from astropy import units as u

from phase_config import DEFAULTS, FLOAT_GEOMETRY_DTYPE, T_CMB, beam_fwhm_rad
from phase3_method_registry import SMHW_SCREEN, WIENER_FEENEY_MATCHED_FILTER, method_metadata


SIGN_QUADRANTS = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))


def parse_float_list(text: str) -> tuple[float, ...]:
    """Parse a comma-separated float list."""

    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def validate_cmb_map(hp_map: np.ndarray) -> np.ndarray:
    """Return a finite float64 CMB anisotropy map in Kelvin."""

    arr = np.asarray(hp_map, dtype=FLOAT_GEOMETRY_DTYPE)
    finite = np.isfinite(arr) & (arr != hp.UNSEEN)
    if not bool(finite.any()):
        raise ValueError("Input map contains no finite observed pixels.")
    max_abs = float(np.max(np.abs(arr[finite])))
    if max_abs > 1.0:
        raise ValueError(
            f"Input map max |T|={max_abs:.3g} K. Expected CMB anisotropies in Kelvin, "
            "not microkelvin or full thermodynamic temperature."
        )
    work = arr.copy()
    work[~finite] = 0.0
    return work


def feeney_cap_profile_mu(mu: np.ndarray, theta_crit_rad: float, z0: float, zcrit: float) -> np.ndarray:
    """Evaluate the Feeney linear cap profile as fractional Delta T/T.

    The profile is linear in ``cos(theta)`` inside the angular cap, with
    center value ``z0`` and boundary value ``zcrit``. Outside the cap the
    modulation is zero; see Feeney et al. PRD 84, arXiv:1012.3667.
    """

    theta_crit_rad = float(theta_crit_rad)
    if not (0.0 < theta_crit_rad < np.pi):
        raise ValueError("theta_crit_rad must lie in (0, pi).")
    mu = np.asarray(mu, dtype=FLOAT_GEOMETRY_DTYPE)
    mu_crit = float(np.cos(theta_crit_rad))
    denom = 1.0 - mu_crit
    if denom <= 0.0:
        raise ValueError("theta_crit_rad is too small for a stable cap denominator.")
    out = np.zeros_like(mu, dtype=FLOAT_GEOMETRY_DTYPE)
    inside = mu >= mu_crit
    out[inside] = (
        float(z0) * (mu[inside] - mu_crit) / denom
        + float(zcrit) * (1.0 - mu[inside]) / denom
    )
    return out


def feeney_template_l0(
    *,
    theta_crit_deg: float,
    z0: float,
    zcrit: float,
    lmax: int,
    quadrature_order: int = 4096,
) -> np.ndarray:
    """Return north-pole ``a_l0`` coefficients for a Feeney cap.

    The returned coefficients are in Kelvin and use double precision because
    high-ell cancellation can be significant for small cap radii.
    """

    if lmax < 1:
        raise ValueError("lmax must be at least 1.")
    if quadrature_order <= lmax:
        raise ValueError("quadrature_order must exceed lmax.")
    theta_crit_rad = (float(theta_crit_deg) * u.deg).to_value(u.rad)
    if not (0.0 < theta_crit_rad < np.pi):
        raise ValueError("theta_crit_deg must lie in (0, 180).")

    mu_crit = float(np.cos(theta_crit_rad))
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    # Remap Gauss-Legendre nodes from [-1, 1] to [mu_crit, 1].
    mu = 0.5 * (1.0 - mu_crit) * nodes + 0.5 * (1.0 + mu_crit)
    w = 0.5 * (1.0 - mu_crit) * weights
    profile = feeney_cap_profile_mu(mu, theta_crit_rad, z0, zcrit) * T_CMB.to_value(u.K)

    coeffs = np.zeros(int(lmax) + 1, dtype=FLOAT_GEOMETRY_DTYPE)
    p_l_minus_1 = np.ones_like(mu)
    coeffs[0] = 2.0 * np.pi * np.sqrt(1.0 / (4.0 * np.pi)) * float(np.sum(w * profile * p_l_minus_1))
    if lmax == 0:
        return coeffs

    p_l = mu.copy()
    coeffs[1] = 2.0 * np.pi * np.sqrt(3.0 / (4.0 * np.pi)) * float(np.sum(w * profile * p_l))
    for ell in range(2, int(lmax) + 1):
        p_next = ((2 * ell - 1) * mu * p_l - (ell - 1) * p_l_minus_1) / ell
        coeffs[ell] = (
            2.0
            * np.pi
            * np.sqrt((2.0 * ell + 1.0) / (4.0 * np.pi))
            * float(np.sum(w * profile * p_next))
        )
        p_l_minus_1, p_l = p_l, p_next
    return coeffs


def white_noise_cl(lmax: int, noise_sigma_uk_arcmin: float = 0.0) -> np.ndarray:
    """Return white-noise ``N_l`` in Kelvin^2 from microkelvin-arcmin depth."""

    if noise_sigma_uk_arcmin < 0.0:
        raise ValueError("noise_sigma_uk_arcmin must be non-negative.")
    sigma_k_rad = (float(noise_sigma_uk_arcmin) * 1.0e-6 * u.K * u.arcmin).to(u.K * u.rad).value
    return np.full(int(lmax) + 1, sigma_k_rad * sigma_k_rad, dtype=FLOAT_GEOMETRY_DTYPE)


def matched_filter_transfer(
    *,
    template_l0: np.ndarray,
    cmb_cl: np.ndarray,
    beam_l: np.ndarray,
    noise_cl: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return harmonic transfer weights and unit-template normalization."""

    template_l0 = np.asarray(template_l0, dtype=FLOAT_GEOMETRY_DTYPE)
    cmb_cl = np.asarray(cmb_cl, dtype=FLOAT_GEOMETRY_DTYPE)
    beam_l = np.asarray(beam_l, dtype=FLOAT_GEOMETRY_DTYPE)
    noise_cl = np.asarray(noise_cl, dtype=FLOAT_GEOMETRY_DTYPE)
    lmax = len(template_l0) - 1
    if min(len(cmb_cl), len(beam_l), len(noise_cl)) <= lmax:
        raise ValueError("cmb_cl, beam_l, and noise_cl must cover template lmax.")

    ells = np.arange(lmax + 1, dtype=FLOAT_GEOMETRY_DTYPE)
    denom = beam_l[: lmax + 1] ** 2 * cmb_cl[: lmax + 1] + noise_cl[: lmax + 1]
    if not np.all(np.isfinite(denom)) or np.any(denom[2:] <= 0.0):
        raise ValueError("Non-physical covariance: B_l^2 C_l + N_l must be positive for ell>=2.")
    denom[:2] = np.inf
    rotation_factor = np.sqrt(4.0 * np.pi / np.maximum(2.0 * ells + 1.0, 1.0))
    weights = beam_l[: lmax + 1] * template_l0 * rotation_factor / denom
    norm = float(np.sum((beam_l[2 : lmax + 1] ** 2) * (template_l0[2:] ** 2) / denom[2:]))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Matched-filter normalization is non-positive.")
    return weights, norm


def wiener_feeney_matched_filter_maps(
    hp_map: np.ndarray,
    *,
    cmb_cl: np.ndarray,
    theta_grid_deg: tuple[float, ...],
    lmax: int | None = None,
    beam_fwhm_arcmin: float = DEFAULTS.beam_fwhm_arcmin,
    noise_sigma_uk_arcmin: float = 0.0,
    quadrature_order: int = 4096,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Scan Feeney caps with a beam/covariance-weighted spherical filter."""

    work = validate_cmb_map(hp_map)
    nside = hp.get_nside(work)
    if lmax is None:
        lmax = 3 * int(nside) - 1
    lmax = int(lmax)
    if lmax < 2:
        raise ValueError("lmax must be at least 2.")
    if len(cmb_cl) <= lmax:
        raise ValueError("cmb_cl does not cover lmax.")

    data_alm = hp.map2alm(work, lmax=lmax, iter=3)
    beam_l = hp.gauss_beam(beam_fwhm_rad(float(beam_fwhm_arcmin)), lmax=lmax)
    noise_cl = white_noise_cl(lmax, noise_sigma_uk_arcmin)
    score_maps: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}
    for theta_deg in theta_grid_deg:
        for z0_sign, zcrit_sign in SIGN_QUADRANTS:
            template_l0 = feeney_template_l0(
                theta_crit_deg=float(theta_deg),
                z0=float(z0_sign),
                zcrit=float(zcrit_sign),
                lmax=lmax,
                quadrature_order=max(int(quadrature_order), lmax + 8),
            )
            weights, norm = matched_filter_transfer(
                template_l0=template_l0,
                cmb_cl=cmb_cl,
                beam_l=beam_l,
                noise_cl=noise_cl,
            )
            filtered = hp.almxfl(data_alm, weights)
            snr_map = hp.alm2map(filtered, nside, lmax=lmax) / np.sqrt(norm)
            key = f"theta{float(theta_deg):g}_z0{int(z0_sign):+d}_zcrit{int(zcrit_sign):+d}"
            score_maps[key] = np.asarray(snr_map, dtype=np.float32)
            metadata[key] = {
                "method": WIENER_FEENEY_MATCHED_FILTER,
                "theta_crit_deg": float(theta_deg),
                "z0_sign": float(z0_sign),
                "zcrit_sign": float(zcrit_sign),
                "normalization": norm,
                "beam_fwhm_arcmin": float(beam_fwhm_arcmin),
                "noise_sigma_uk_arcmin": float(noise_sigma_uk_arcmin),
            }
    return score_maps, metadata


def smhw_screen_maps(
    hp_map: np.ndarray,
    *,
    scales_deg: tuple[float, ...],
    lmax: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Return scale-space Mexican-hat context maps on the sphere."""

    work = validate_cmb_map(hp_map)
    nside = hp.get_nside(work)
    if lmax is None:
        lmax = 3 * int(nside) - 1
    lmax = int(lmax)
    data_alm = hp.map2alm(work, lmax=lmax, iter=3)
    ells = np.arange(lmax + 1, dtype=FLOAT_GEOMETRY_DTYPE)
    maps: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}
    for scale_deg in scales_deg:
        scale_rad = (float(scale_deg) * u.deg).to_value(u.rad)
        if scale_rad <= 0.0:
            raise ValueError("SMHW scales must be positive.")
        window = ells * (ells + 1.0) * scale_rad * scale_rad * np.exp(
            -0.5 * ells * (ells + 1.0) * scale_rad * scale_rad
        )
        window[:2] = 0.0
        response = hp.alm2map(hp.almxfl(data_alm, window), nside, lmax=lmax)
        key = f"scale{float(scale_deg):g}"
        maps[key] = np.asarray(response, dtype=np.float32)
        metadata[key] = {
            "method": SMHW_SCREEN,
            "scale_deg": float(scale_deg),
            "is_matched_filter": False,
            "wavelet": "scale_normalized_spherical_laplacian_of_gaussian",
        }
    return maps, metadata


def load_map(path: Path) -> np.ndarray:
    """Load a HEALPix map from FITS or NumPy format."""

    if path.suffix.lower() == ".npy":
        return np.load(path)
    return hp.read_map(str(path), field=0)


def load_cl(path: Path, lmax: int) -> np.ndarray:
    """Load a 1D TT C_l array in Kelvin^2."""

    if path.suffix.lower() == ".npy":
        cl = np.load(path)
    else:
        cl = np.loadtxt(path)
        if cl.ndim == 2:
            cl = cl[:, -1]
    cl = np.asarray(cl, dtype=FLOAT_GEOMETRY_DTYPE)
    if len(cl) <= int(lmax):
        raise ValueError(f"C_l file length {len(cl)} does not cover lmax={lmax}.")
    return cl[: int(lmax) + 1]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for full-sky classical filters."""

    parser = argparse.ArgumentParser(
        description="Run remediated full-sky classical CMB bubble-collision screeners.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-map", type=str, required=True)
    parser.add_argument("--cmb-cl", type=str, required=True, help="1D TT C_l array in K^2, .npy or text.")
    parser.add_argument("--output-npz", type=str, required=True)
    parser.add_argument("--theta-grid-deg", type=str, default="5,10,15,20,25")
    parser.add_argument("--smhw-scales-deg", type=str, default="2,5,10,15,20")
    parser.add_argument("--lmax", type=int, default=0)
    parser.add_argument("--beam-fwhm-arcmin", type=float, default=DEFAULTS.beam_fwhm_arcmin)
    parser.add_argument("--noise-sigma-uk-arcmin", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    input_map = load_map(Path(args.input_map).resolve())
    nside = hp.get_nside(input_map)
    lmax = int(args.lmax) if args.lmax else 3 * int(nside) - 1
    cmb_cl = load_cl(Path(args.cmb_cl).resolve(), lmax)
    theta_grid = parse_float_list(args.theta_grid_deg)
    smhw_scales = parse_float_list(args.smhw_scales_deg)

    matched_maps, matched_metadata = wiener_feeney_matched_filter_maps(
        input_map,
        cmb_cl=cmb_cl,
        theta_grid_deg=theta_grid,
        lmax=lmax,
        beam_fwhm_arcmin=float(args.beam_fwhm_arcmin),
        noise_sigma_uk_arcmin=float(args.noise_sigma_uk_arcmin),
    )
    smhw_maps, smhw_metadata = smhw_screen_maps(input_map, scales_deg=smhw_scales, lmax=lmax)
    payload = {
        **{f"{WIENER_FEENEY_MATCHED_FILTER}__{key}": value for key, value in matched_maps.items()},
        **{f"{SMHW_SCREEN}__{key}": value for key, value in smhw_maps.items()},
    }
    output_npz = Path(args.output_npz).resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **payload)
    metadata = {
        "input_map": str(Path(args.input_map).resolve()),
        "cmb_cl": str(Path(args.cmb_cl).resolve()),
        "lmax": int(lmax),
        "methods": {
            WIENER_FEENEY_MATCHED_FILTER: method_metadata(WIENER_FEENEY_MATCHED_FILTER),
            SMHW_SCREEN: method_metadata(SMHW_SCREEN),
        },
        "matched_filter_maps": matched_metadata,
        "smhw_maps": smhw_metadata,
    }
    output_npz.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"scores_npz": str(output_npz), "metadata_json": str(output_npz.with_suffix(".json"))}, indent=2))


if __name__ == "__main__":
    main()
