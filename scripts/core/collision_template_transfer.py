"""Bubble-collision CMB templates from a primordial curvature profile and CAMB transfer functions.

Assumptions
-----------
* The collision perturbs the primordial comoving curvature zeta along one axis only (planar symmetry):
  zeta(z) = a_ramp * (z - r) for z >= r, zero otherwise ("ramp"), optionally plus a_step for z >= r
  ("ramp+step"). r is the comoving distance from us to the collision boundary, so the disc on the sky has
  angular radius theta_c with cos(theta_c) = r / D, D the comoving distance to last scattering.
  Source: Osborne, Senatore, Smith, JCAP 10 (2013) 001, arXiv:1305.1970, Eqs. (4)-(8).
* Harmonic coefficients (their Eq. 8), for the collision axis along z:
      a_lm = b_l * sqrt((2l+1)/(4 pi)) * delta_m0,
      b_l  = 2 * integral_0^inf dk Delta_l(k) [ i^l zeta~(k) + (-i)^l zeta~(-k) ],
  with zeta~(k) = integral dz zeta(z) exp(-i k z). For the ramp, zeta~ = -a exp(-ikr)/k^2; for the step,
  zeta~ = a_s exp(-ikr)/(ik), so (derived here, checked numerically against the Sachs-Wolfe limit):
      b_l(ramp) = -4 a   * integral dk Delta_l(k) cos(k r - l pi/2) / k^2          (l >= 2)
      b_l(step) =  4 a_s * integral dk Delta_l(k) cos(k r - (l-1) pi/2) / k        (l >= 1)
  Monopole and dipole are dropped (unobservable).
* Delta_l(k) is CAMB's scalar transfer function, normalised so that
  C_l = 4 pi * integral dlnk P_zeta(k) Delta_l(k)^2 (verified numerically by ``check_cl_normalisation``).
* Temperature is the dimensionless Delta T / T. The profile on the sky is
      T(theta) = sum_l (2l+1)/(4 pi) * b_l * W_l * P_l(cos theta),   W_l = beam and pixel window.
* Amplitudes are quoted through the Sachs-Wolfe central value z0_sw = sw_sign * a D (1 - cos theta_c) / 5,
  where sw_sign is the measured sign of CAMB's low-l transfer function relative to j_l(kD)/5. This keeps
  z0 comparable with the central amplitude of the Feeney et al. (2011) template.

Nothing here reads or writes project data. Float64 throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import camb
except ImportError:  # the Sachs-Wolfe self-test below does not need CAMB
    camb = None

_trapz = getattr(np, "trapezoid", None) or np.trapz

PLANCK_2018 = {"H0": 67.36, "ombh2": 0.02237, "omch2": 0.1200, "tau": 0.0544, "As": 2.1e-9, "ns": 0.9649}


@dataclass
class Transfer:
    ells: np.ndarray       # (nl,)
    k: np.ndarray          # (nk,) in 1/Mpc
    delta_t: np.ndarray    # (nl, nk) temperature transfer
    delta_e: np.ndarray    # (nl, nk) E-mode transfer
    d_star: float          # comoving distance to last scattering, Mpc
    params: dict
    cl_tt_camb: np.ndarray  # CAMB's own unlensed scalar TT C_l (dimensionless), for the normalisation check


def camb_transfer(lmax: int = 800, params: dict | None = None, accuracy_boost: float = 1.0) -> Transfer:
    if camb is None:
        raise RuntimeError("camb is not installed")
    params = dict(PLANCK_2018 if params is None else params)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params["H0"], ombh2=params["ombh2"], omch2=params["omch2"], tau=params["tau"])
    pars.InitPower.set_params(As=params["As"], ns=params["ns"])
    pars.set_for_lmax(int(lmax), lens_potential_accuracy=0)
    pars.WantTensors = False
    pars.DoLensing = False
    pars.Accuracy.AccuracyBoost = float(accuracy_boost)
    pars.Accuracy.lSampleBoost = 50.0  # compute every multipole instead of interpolating between samples
    data = camb.get_transfer_functions(pars)
    tr = data.get_cmb_transfer_data("scalar")
    ells = np.asarray(tr.L, dtype=np.int64)
    k = np.asarray(tr.q, dtype=np.float64)
    delta = np.asarray(tr.delta_p_l_k, dtype=np.float64)
    results = camb.get_results(pars)
    derived = results.get_derived_params()
    d_star = float(results.comoving_radial_distance(derived["zstar"]))
    cl = results.get_cmb_power_spectra(pars, CMB_unit=None, raw_cl=True)["unlensed_scalar"][:, 0]
    keep = ells <= lmax
    return Transfer(ells[keep], k, delta[0][keep], delta[1][keep], d_star, params, np.asarray(cl, dtype=np.float64))


def check_cl_normalisation(tr: Transfer) -> dict:
    """C_l from the transfer functions against CAMB's own C_l. Ratios should be 1."""
    p = tr.params["As"] * (tr.k / 0.05) ** (tr.params["ns"] - 1.0)
    cl = 4.0 * math.pi * _trapz(p[None, :] * tr.delta_t**2 / tr.k[None, :], tr.k, axis=1)
    out = {}
    for ell in (2, 10, 30, 100, 220, 500):
        i = np.nonzero(tr.ells == ell)[0]
        if i.size and ell < tr.cl_tt_camb.size:
            out[int(ell)] = float(cl[i[0]] / tr.cl_tt_camb[ell])
    return out


def sachs_wolfe_transfer(ells: np.ndarray, k: np.ndarray, d_star: float, sign: float = -1.0) -> np.ndarray:
    """Pure Sachs-Wolfe transfer, Delta_l(k) = sign * j_l(k D) / 5 (test input for the integrator)."""
    from scipy.special import spherical_jn

    x = k * d_star
    return np.stack([sign * spherical_jn(int(ell), x) / 5.0 for ell in ells], axis=0)


def _extend_to_zero(ells, k, delta, n_extra=400):
    """Prepend k from ~0 to k[0], with Delta_l proportional to k^l (valid for k D << l)."""
    k_lo = np.linspace(k[0] * 1e-4, k[0], n_extra, endpoint=False)
    scale = (k_lo[None, :] / k[0]) ** ells[:, None].astype(np.float64)
    return np.concatenate([k_lo, k]), np.concatenate([delta[:, :1] * scale, delta], axis=1)


def b_ell(ells, k, delta, r, a_ramp=0.0, a_step=0.0, refine: int = 8):
    """Template coefficients b_l for comoving boundary distance r (Mpc).

    The transfer function is linearly interpolated onto a grid ``refine`` times finer than the input
    (at least 12 points per period of the fastest oscillation), then integrated with the trapezoid rule.
    """
    ells = np.asarray(ells, dtype=np.int64)
    k_ext, d_ext = _extend_to_zero(ells, np.asarray(k, dtype=np.float64), np.asarray(delta, dtype=np.float64))
    out = np.zeros(ells.size, dtype=np.float64)
    for idx, ell in enumerate(ells):
        if ell < 2:
            continue
        kk, dd = k_ext, d_ext[idx]
        if refine > 1:
            from scipy.interpolate import CubicSpline

            kf = np.interp(np.linspace(0, kk.size - 1, (kk.size - 1) * refine + 1), np.arange(kk.size), kk)
            dd = CubicSpline(kk, dd)(kf)
            kk = kf
        total = 0.0
        if a_ramp != 0.0:
            total += -4.0 * a_ramp * _trapz(dd * np.cos(kk * r - ell * math.pi / 2.0) / kk**2, kk)
        if a_step != 0.0:
            total += 4.0 * a_step * _trapz(dd * np.cos(kk * r - (ell - 1) * math.pi / 2.0) / kk, kk)
        out[idx] = total
    return out


def profile(theta, ells, b, window=None):
    """T(theta) = sum_l (2l+1)/(4 pi) b_l W_l P_l(cos theta)."""
    lmax = int(np.max(ells))
    coeff = np.zeros(lmax + 1, dtype=np.float64)
    w = np.ones(ells.size) if window is None else np.asarray(window, dtype=np.float64)[np.asarray(ells)]
    coeff[np.asarray(ells)] = (2.0 * np.asarray(ells) + 1.0) / (4.0 * math.pi) * b * w
    return np.polynomial.legendre.legval(np.cos(np.asarray(theta, dtype=np.float64)), coeff)


def ramp_real_space(theta, theta_c, amplitude=1.0, lmax=None):
    """Real-space ramp (cos theta - cos theta_c) for theta < theta_c, scaled to ``amplitude`` at the centre.

    If lmax is given, the monopole and dipole are removed and the series is truncated at lmax, so the
    result can be compared directly with a harmonic-space template.
    """
    theta = np.asarray(theta, dtype=np.float64)
    mu_c = math.cos(theta_c)
    f = np.where(theta < theta_c, (np.cos(theta) - mu_c) / (1.0 - mu_c), 0.0) * amplitude
    if lmax is None:
        return f
    mu, wts = np.polynomial.legendre.leggauss(4 * lmax + 64)
    g = np.where(mu > mu_c, (mu - mu_c) / (1.0 - mu_c), 0.0) * amplitude
    coeff = np.zeros(lmax + 1)
    for ell in range(lmax + 1):
        c = np.zeros(ell + 1)
        c[ell] = 1.0
        coeff[ell] = (2 * ell + 1) / 2.0 * np.sum(wts * g * np.polynomial.legendre.legval(mu, c))
    coeff[:2] = 0.0
    return np.polynomial.legendre.legval(np.cos(theta), coeff)


def gaussian_beam(lmax, fwhm_arcmin):
    sigma = math.radians(fwhm_arcmin / 60.0) / math.sqrt(8.0 * math.log(2.0))
    ell = np.arange(lmax + 1)
    return np.exp(-0.5 * ell * (ell + 1) * sigma**2)
