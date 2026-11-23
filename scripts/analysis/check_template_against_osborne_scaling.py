"""Check the transfer-function ramp template against a published number that depends on its shape.

Background
----------
Notebook entries (f), (g) and (t) leave one discrepancy open: for an 11.39 degree disc the template built
here has an edge-to-centre ratio of 0.48 and a centre 2.4 times the Sachs-Wolfe value, where Fig. 3 of
Osborne, Senatore and Smith (arXiv:1305.1970) shows a profile close to the smoothed Sachs-Wolfe ramp.
Entry (g) names one way to settle it: "reproduce one of their published numbers that depends on the
profile shape".

The number
----------
Their WMAP analysis (arXiv:1305.1964) uses the same curvature profile, zeta(z) = a (z - r) for z >= r, and
reports its limit in the variable alpha = a * sin(theta_c)^(4/3), where "the 4/3 exponent is empirically
chosen so that two bubbles with the same value of alpha and different angular sizes have roughly equal
statistical significance". Their limit is -4.66e-8 < alpha < 4.73e-8 per Mpc (95%).

So in their analysis the significance per unit slope a scales as sin(theta_c)^(4/3). That scaling depends
on the template: for a pure Sachs-Wolfe ramp the central amplitude is a D (1 - cos theta_c) / 5, which
falls much faster towards small discs than a template whose small discs are brightened by the Doppler and
acoustic terms.

This script computes the ideal signal-to-noise per unit slope, SNR(theta_c) for a = 1 per Mpc, for
  TF  the ramp through CAMB's full temperature transfer function (this project's template), and
  SW  the same ramp through the pure Sachs-Wolfe transfer -j_l(k D)/5,
against the CMB itself (unlensed TT, optional white noise and beam), monopole and dipole excluded, and
reports (1) the fitted exponent n in SNR proportional to sin(theta_c)^n, (2) how flat SNR / sin^(4/3) is
across sizes, and (3) their 95% limit expressed in units of the ideal one-sigma error on a at each size.
A size-independent search threshold should make (3) roughly constant.

Limits: ideal full-sky SNR at a known position (no mask, no search); Planck 2018 parameters where they
used WMAP 7-year ones (D = 13,873 against about 14,166 Mpc; sizes are compared at fixed angle). The
check is on the size scaling, which these do not change much.

Usage: python scripts/analysis/check_template_against_osborne_scaling.py --out-json OUT.json [--plot OUT.png]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import collision_template_transfer as ct  # noqa: E402

ALPHA_LIMIT_95 = 4.7e-8  # per Mpc, Osborne, Senatore, Smith (arXiv:1305.1964), mean of the two sides
T_CMB_UK = 2.7255e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--lmax", type=int, default=1000)
    ap.add_argument("--sizes-deg", default="5,7.5,10,15,20,30,40,50,60,70,80")
    ap.add_argument("--noise-uk-arcmin", type=float, default=0.0, help="white noise; 0 = cosmic-variance limit")
    ap.add_argument("--beam-fwhm-arcmin", type=float, default=0.0)
    args = ap.parse_args()
    lmax = args.lmax

    tr = ct.camb_transfer(lmax=lmax)
    sw = ct.sachs_wolfe_transfer(tr.ells, tr.k, tr.d_star, sign=-1.0)
    ell = np.arange(lmax + 1, dtype=np.float64)
    cl = np.zeros(lmax + 1)
    n = min(lmax + 1, tr.cl_tt_camb.size)
    cl[:n] = tr.cl_tt_camb[:n]
    beam2 = ct.gaussian_beam(lmax, args.beam_fwhm_arcmin) ** 2 if args.beam_fwhm_arcmin > 0 else np.ones(lmax + 1)
    noise = (args.noise_uk_arcmin / T_CMB_UK * math.radians(1.0 / 60.0)) ** 2 / beam2
    denom = cl + noise
    pref = (2 * ell + 1) / (4 * math.pi)

    def snr_per_unit_slope(delta, r):
        b = np.zeros(lmax + 1)
        b[tr.ells] = ct.b_ell(tr.ells, tr.k, delta, r, a_ramp=1.0)
        return math.sqrt(float(np.sum(pref[2:] * b[2:] ** 2 / denom[2:])))

    rows = []
    for size in [float(x) for x in args.sizes_deg.split(",")]:
        tc = math.radians(size)
        r = tr.d_star * math.cos(tc)
        s_tf = snr_per_unit_slope(tr.delta_t, r)
        s_sw = snr_per_unit_slope(sw, r)
        a_lim = ALPHA_LIMIT_95 / math.sin(tc) ** (4.0 / 3.0)
        rows.append({"theta_c_deg": size, "snr_per_unit_slope_TF": s_tf, "snr_per_unit_slope_SW": s_sw,
                     "TF_over_sin43": s_tf / math.sin(tc) ** (4.0 / 3.0), "SW_over_sin43": s_sw / math.sin(tc) ** (4.0 / 3.0),
                     "published_limit_in_ideal_sigma_TF": a_lim * s_tf, "published_limit_in_ideal_sigma_SW": a_lim * s_sw})

    x = np.log(np.sin(np.radians([r["theta_c_deg"] for r in rows])))

    def exponent(key, lo, hi):
        sel = np.array([lo <= r["theta_c_deg"] <= hi for r in rows])
        return float(np.polyfit(x[sel], np.log([r[key] for r, s in zip(rows, sel) if s]), 1)[0])

    fits = {name: {"TF": exponent("snr_per_unit_slope_TF", lo, hi), "SW": exponent("snr_per_unit_slope_SW", lo, hi)}
            for name, (lo, hi) in {"5_to_80_deg": (5, 80), "10_to_70_deg": (10, 70), "5_to_30_deg": (5, 30)}.items()}

    def flatness(key):
        v = np.array([r[key] for r in rows])
        return float(v.max() / v.min())

    summary = {"exponent_fits": fits, "published_exponent": 4.0 / 3.0,
               "max_over_min_of_snr_over_sin43": {"TF": flatness("TF_over_sin43"), "SW": flatness("SW_over_sin43")}}

    print(f"lmax {lmax}, noise {args.noise_uk_arcmin:g} uK-arcmin, beam {args.beam_fwhm_arcmin:g} arcmin, D* = {tr.d_star:.1f} Mpc")
    print(f"{'theta_c':>8s} {'SNR/a TF':>12s} {'SNR/a SW':>12s} {'TF/sin^4/3':>12s} {'SW/sin^4/3':>12s} {'limit in sigma TF':>18s} {'limit in sigma SW':>18s}")
    for r in rows:
        print(f"{r['theta_c_deg']:8.1f} {r['snr_per_unit_slope_TF']:12.4g} {r['snr_per_unit_slope_SW']:12.4g} {r['TF_over_sin43']:12.4g} "
              f"{r['SW_over_sin43']:12.4g} {r['published_limit_in_ideal_sigma_TF']:18.2f} {r['published_limit_in_ideal_sigma_SW']:18.2f}")
    print("\nfitted exponent n in SNR per unit slope ~ sin(theta_c)^n   (published, empirical: 4/3 = 1.33)")
    for name, f in fits.items():
        print(f"  {name:>13s}:  transfer-function template {f['TF']:.2f}    Sachs-Wolfe ramp {f['SW']:.2f}")
    print(f"largest / smallest value of SNR / sin^(4/3) across sizes: transfer-function {summary['max_over_min_of_snr_over_sin43']['TF']:.2f}, "
          f"Sachs-Wolfe {summary['max_over_min_of_snr_over_sin43']['SW']:.2f}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"lmax": lmax, "noise_uk_arcmin": args.noise_uk_arcmin, "beam_fwhm_arcmin": args.beam_fwhm_arcmin,
                               "alpha_limit_95_per_mpc": ALPHA_LIMIT_95, "rows": rows, "summary": summary}, indent=1))
    print("wrote", out)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
        sizes = [r["theta_c_deg"] for r in rows]
        fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=140)
        fig.patch.set_facecolor(surface)
        ax.set_facecolor(surface)
        ax.plot(sizes, [r["published_limit_in_ideal_sigma_TF"] for r in rows], color="#2a78d6", lw=2, marker="o", ms=5,
                label="this project's template (full transfer function)")
        ax.plot(sizes, [r["published_limit_in_ideal_sigma_SW"] for r in rows], color="#eb6834", lw=2, marker="o", ms=5,
                label="Sachs-Wolfe ramp only")
        ax.set_ylim(0, None)
        ax.set_xlabel("disc radius (degrees)", color=muted)
        ax.set_ylabel("published 95% limit, in ideal one-sigma errors", color=muted)
        ax.set_title("The published WMAP limit is a steady 5 to 6 sigma only for the full-transfer template",
                     color=ink, loc="left", fontsize=11.5, pad=12)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c8c2")
        ax.tick_params(colors=muted)
        ax.legend(frameon=False, loc="lower right", labelcolor=ink)
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, facecolor=surface)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
