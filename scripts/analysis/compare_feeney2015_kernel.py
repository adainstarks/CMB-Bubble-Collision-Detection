"""Reproduce Fig. 6 of Feeney, Elsner, Johnson and Peiris (arXiv:1506.01716) from this project's transfer functions.

Their figure shows r^2 alpha_l(r) for l = 20, temperature and E polarization, where
    alpha_l(r) = (2 / pi) * integral dk k^2 Delta_l(k) j_l(k r)
is the real-space form of the radiation transfer function (their Eq. after 9). A pure Sachs-Wolfe transfer
function gives a spike at the last-scattering distance D*. A Doppler (derivative) term moves weight to
r > D* and leaves a negative lobe inside, which is what puts a plateau on small collision discs.

The published curves were read from the vector figure in the paper's arXiv source
(work/results_2026-11-22/feeney2015_fig6_curves.csv). Their cosmology: Planck 2013 + WP + highL + BAO best fit.
Not pre-registered: run after the comparison with Osborne et al. Fig. 3 (notebook entry (v)).

Usage: python scripts/analysis/compare_feeney2015_kernel.py --curves work/results_2026-11-22/feeney2015_fig6_curves.csv --out-dir OUT
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

import collision_template_transfer as ct

_trapz = getattr(np, "trapezoid", None) or np.trapz
PLANCK_2013_BAO = {"H0": 67.77, "ombh2": 0.022161, "omch2": 0.11889, "tau": 0.0952, "As": 2.2138e-9, "ns": 0.9611}
ELL = 20


def kernel(k, delta, r, refine=8):
    """r^2 alpha_l(r) for l = ELL on the grid r (Mpc)."""
    kf = np.interp(np.linspace(0, k.size - 1, (k.size - 1) * refine + 1), np.arange(k.size), k)
    dd = CubicSpline(k, delta)(kf)
    out = np.empty(r.size)
    for i, rr in enumerate(r):
        out[i] = rr**2 * (2.0 / math.pi) * _trapz(kf**2 * dd * spherical_jn(ELL, kf * rr), kf)
    return out


def describe(name, r, f, d_star):
    pos = np.where(f > 0, f, 0.0)
    neg = np.where(f < 0, f, 0.0)
    m0 = _trapz(f, r)
    info = {"peak_r": float(r[np.argmax(f)]), "peak_minus_Dstar": float(r[np.argmax(f)] - d_star),
            "positive_area": float(_trapz(pos, r)), "negative_area": float(_trapz(neg, r)),
            "negative_over_positive": float(-_trapz(neg, r) / _trapz(pos, r)),
            "first_moment_about_Dstar_over_integral": float(_trapz(f * (r - d_star), r) / m0)}
    print("%s: peak at r = %.0f Mpc (D* %+.0f), negative/positive area %.2f, first moment about D* / integral = %+.0f Mpc" % (
        name, info["peak_r"], info["peak_minus_Dstar"], info["negative_over_positive"], info["first_moment_about_Dstar_over_integral"]))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lmax", type=int, default=2000)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pub = {}
    with open(args.curves, newline="") as f:
        for row in csv.DictReader(f):
            pub.setdefault(row["curve"], []).append((float(row["r_mpc"]), float(row["value"])))
    tr = ct.camb_transfer(lmax=args.lmax, params=PLANCK_2013_BAO)
    i20 = int(np.nonzero(tr.ells == ELL)[0][0])
    print("cosmology: Planck 2013 + WP + highL + BAO; D* = %.1f Mpc; %d k values up to %.3f /Mpc" % (tr.d_star, tr.k.size, tr.k.max()))
    for ell in (2, 20, 100, 500, 1500):
        j = int(np.nonzero(tr.ells == ell)[0][0])
        nz = np.nonzero(np.abs(tr.delta_t[j]) > 1e-12 * np.abs(tr.delta_t[j]).max())[0]
        print("  l = %4d: transfer function non-zero for k from %.2e to %.3f /Mpc" % (ell, tr.k[nz[0]], tr.k[nz[-1]]))
    report = {"D_star": tr.d_star, "ell": ELL}
    r_fine = np.arange(13400.0, 14200.5, 1.0)
    for name, delta in (("temperature", tr.delta_t[i20]), ("polarization", tr.delta_e[i20])):
        p = np.array(sorted(pub[name]))
        mine_fine = kernel(tr.k, delta, r_fine)
        mine = np.interp(p[:, 0], r_fine, mine_fine)
        amp = float(np.dot(mine, p[:, 1]) / np.dot(mine, mine))
        rms = float(np.sqrt(np.mean((p[:, 1] - amp * mine) ** 2)) / np.max(np.abs(p[:, 1])))
        corr = float(np.corrcoef(mine, p[:, 1])[0, 1])
        print("\n%s: %d published points; amplitude fit %.4g; RMS residual / maximum = %.4f; correlation %.4f" % (name, p.shape[0], amp, rms, corr))
        a = describe("  published", p[:, 0], p[:, 1], tr.d_star)
        b = describe("  this work, at the published radii", p[:, 0], amp * mine, tr.d_star)
        c = describe("  this work, 1 Mpc grid", r_fine, amp * mine_fine, tr.d_star)
        print("  r (Mpc):      " + " ".join("%7.0f" % x for x in (13500, 13700, 13800, 13900, 13950, 14000, 14030, 14050, 14060, 14080, 14100, 14150)))
        print("  published:    " + " ".join("%+7.3f" % np.interp(x, p[:, 0], p[:, 1]) for x in (13500, 13700, 13800, 13900, 13950, 14000, 14030, 14050, 14060, 14080, 14100, 14150)))
        print("  this work:    " + " ".join("%+7.3f" % (amp * np.interp(x, r_fine, mine_fine)) for x in (13500, 13700, 13800, 13900, 13950, 14000, 14030, 14050, 14060, 14080, 14100, 14150)))
        report[name] = {"amplitude": amp, "rms_over_max": rms, "correlation": corr, "published": a, "this_work_at_published_radii": b, "this_work_fine": c}
        np.savez_compressed(out_dir / ("feeney2015_kernel_%s.npz" % name), r=r_fine, kernel=amp * mine_fine, published=p)
    with open(out_dir / "compare_feeney2015_kernel_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print("\nwrote", out_dir / "compare_feeney2015_kernel_report.json")


if __name__ == "__main__":
    main()
