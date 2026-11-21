"""Ideal matched-filter signal-to-noise of three collision templates against the CMB itself.

Assumptions
-----------
* One collision at a known position on the full sky; "noise" is the Gaussian CMB (CAMB unlensed TT, Planck
  2018 parameters) plus 30 uK-arcmin white noise; 5 arcmin beam and Nside 256 pixel window; lmax 767;
  monopole and dipole excluded. SNR^2 = sum_l (2l+1)/(4 pi) b_l^2 W_l^2 / (C_l W_l^2 + N_l).
  This is the per-template ideal, with no search over position or size, so it is an upper bound.
* Templates, all with central amplitude z0 defined as stated in each column:
    A0  April template, z_crit = 0: linear in cos(theta), zero at the edge (pure Sachs-Wolfe ramp shape).
    A1  April template, z_crit = z0: flat disc with a sharp edge.
    TF  ramp in the primordial curvature through CAMB transfer functions.
  "TF same curvature" uses the curvature slope whose Sachs-Wolfe central amplitude is z0 (same underlying
  collision as A0). "TF same peak" rescales TF so that its peak Delta T / T equals z0 (same observed peak).

Usage: python scripts/analysis/template_snr_comparison.py --out-json OUT
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import healpy as hp
import numpy as np
from scipy.special import eval_legendre

import collision_template_transfer as ct

T_CMB_K = 2.7255


def legendre_b(func_mu, lmax, nodes=20000):
    """b_l = 2 pi * integral f(mu) P_l(mu) dmu, so that f(theta) = sum (2l+1)/(4 pi) b_l P_l."""
    mu, w = np.polynomial.legendre.leggauss(nodes)
    f = func_mu(mu)
    b = np.zeros(lmax + 1)
    p_prev, p = np.ones_like(mu), mu.copy()
    b[0] = 2 * math.pi * np.sum(w * f)
    b[1] = 2 * math.pi * np.sum(w * f * p)
    for ell in range(2, lmax + 1):
        p_prev, p = p, ((2 * ell - 1) * mu * p - (ell - 1) * p_prev) / ell
        b[ell] = 2 * math.pi * np.sum(w * f * p)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--lmax", type=int, default=767)
    ap.add_argument("--z0", type=float, default=1e-5)
    args = ap.parse_args()
    lmax, z0 = args.lmax, args.z0

    tr = ct.camb_transfer(lmax=lmax)
    ells = np.arange(lmax + 1)
    window = ct.gaussian_beam(lmax, 5.0) * hp.pixwin(256, lmax=lmax)
    cl = np.zeros(lmax + 1)
    cl[: min(lmax + 1, tr.cl_tt_camb.size)] = tr.cl_tt_camb[: lmax + 1]
    noise = (30e-6 / T_CMB_K * math.radians(1.0 / 60.0)) ** 2 * np.ones(lmax + 1)
    denom = cl * window**2 + noise

    def snr(b_full):
        ell = ells[2:]
        return float(np.sqrt(np.sum((2 * ell + 1) / (4 * math.pi) * b_full[2:] ** 2 * window[2:] ** 2 / denom[2:])))

    theta_fine = np.radians(np.linspace(0, 90, 3601))
    rows = []
    print(f"ideal single-template SNR for central amplitude z0 = {z0:g}; blind-search threshold in April's benchmark was 5.2")
    print(f"{'theta_c':>8s} {'A0 (z_crit=0)':>14s} {'A1 (z_crit=z0)':>15s} {'TF same curvature':>18s} {'TF same peak':>13s} {'TF peak / z0':>13s} | z0 for SNR 5.2:  {'A0':>9s} {'A1':>9s} {'TF(peak)':>9s}")
    for tcd in (5.0, 10.0, 15.0, 20.0, 25.0, 40.0, 70.0):
        tc = math.radians(tcd)
        mu_c = math.cos(tc)
        b_a0 = legendre_b(lambda mu: np.where(mu > mu_c, z0 * (mu - mu_c) / (1 - mu_c), 0.0), lmax)
        b_a1 = np.zeros(lmax + 1)
        l = np.arange(1, lmax + 1)
        b_a1[1:] = 4 * math.pi / (2 * l + 1) * (z0 / 2.0) * (eval_legendre(l - 1, mu_c) - eval_legendre(l + 1, mu_c))
        a = 1.0 / tr.d_star
        unit = -(a * tr.d_star / 5.0) * (1 - mu_c)
        b_tf = np.zeros(lmax + 1)
        b_tf[tr.ells] = ct.b_ell(tr.ells, tr.k, tr.delta_t, tr.d_star * mu_c, a_ramp=a) * (z0 / unit)
        prof = ct.profile(theta_fine, ells[2:], b_tf[2:], window=window)
        peak = float(np.max(np.abs(prof)))
        s0, s1, st = snr(b_a0), snr(b_a1), snr(b_tf)
        st_peak = st * z0 / peak
        row = {"theta_c_deg": tcd, "snr_A0": s0, "snr_A1": s1, "snr_TF_same_curvature": st, "snr_TF_same_peak": st_peak,
               "TF_peak_over_z0": peak / z0, "z0_for_snr5p2": {"A0": 5.2 * z0 / s0, "A1": 5.2 * z0 / s1, "TF_peak": 5.2 * z0 / st_peak}}
        rows.append(row)
        print(f"{tcd:8.0f} {s0:14.2f} {s1:15.2f} {st:18.2f} {st_peak:13.2f} {peak / z0:13.2f} | {'':16s} {5.2 * z0 / s0:9.1e} {5.2 * z0 / s1:9.1e} {5.2 * z0 / st_peak:9.1e}")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"z0": z0, "lmax": lmax, "rows": rows}, indent=1))
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
