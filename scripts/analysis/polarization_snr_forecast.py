"""How much can E-mode polarization add to the collision search? Ideal matched-filter SNR in T, E and both.

Assumptions
-----------
* One collision at a known position. Template coefficients b_l^T and b_l^E come from the same primordial
  curvature ramp through CAMB's temperature and E-mode transfer functions (``collision_template_transfer``).
  CAMB's raw E transfer lacks the factor sqrt((l+2)!/(l-2)!); it is applied here and checked: the EE and TE
  spectra rebuilt from the transfer functions must equal CAMB's own (ratio printed, must be 1).
* Noise is the CMB itself (unlensed scalar TT, TE, EE as a 2x2 covariance per multipole) plus instrument
  noise deconvolved by the beam:
      SNR_T^2     = sum_l (2l+1)/(4 pi) bT^2 / (C_TT + N_T)
      SNR_E^2     = sum_l (2l+1)/(4 pi) bE^2 / (C_EE + N_E)
      SNR_joint^2 = sum_l (2l+1)/(4 pi) [bT bE] Cov_l^-1 [bT bE]^T
  "E given T" is sqrt(SNR_joint^2 - SNR_T^2): what polarization adds once temperature is used.
* Amplitudes are per 1e-5 of the observed temperature peak |Delta T / T| (5 arcmin beam), as in the
  November benchmarks. Monopole and dipole excluded.
* The experiment rows are rough stand-ins (white noise level, beam, lowest usable polarization multipole,
  optional low-l noise knee), not instrument models. They bracket the question; they do not forecast a mission.

Nothing here reads or writes project data. Needs camb, numpy, scipy (no healpy).

Usage: python scripts/analysis/polarization_snr_forecast.py --out-json OUT.json [--plot OUT.png] [--lmax 1000]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import camb  # noqa: E402

import collision_template_transfer as ct  # noqa: E402

T_CMB_UK = 2.7255e6
_trapz = getattr(np, "trapezoid", None) or np.trapz

# label, T noise (uK-arcmin), P noise (uK-arcmin), beam FWHM (arcmin), lowest usable E multipole,
# highest multipole used, low-l knee of the P noise (0 = white; negative = usable only above |knee|,
# Planck-like polarization below), knee slope
EXPERIMENTS = [
    ("cosmic-variance limit", 0.0, 0.0, 0.0, 2, None, 0.0, 0.0),
    ("Planck-like, white 52 uK-arcmin polarization down to l=2", 33.0, 52.0, 5.0, 2, None, 0.0, 0.0),
    ("Planck-like, polarization usable from l=30", 33.0, 52.0, 5.0, 30, None, 0.0, 0.0),
    ("Planck-like, polarization noise knee at l=50", 33.0, 52.0, 5.0, 2, None, 50.0, 2.0),
    ("ground wide survey 14 uK-arcmin, l>=300, Planck-like below", 10.0, 14.0, 1.4, 2, None, -300.0, 0.0),
    ("ground wide survey 9 uK-arcmin, knee l=700, from l=30", 6.0, 9.0, 1.4, 30, None, 700.0, 1.4),
    ("ground small aperture 3 uK-arcmin, l=30-300", 33.0, 3.0, 30.0, 30, 300, 50.0, 2.5),
    ("space mission 2.2 uK-arcmin, 30 arcmin beam, l<=600", 33.0, 2.2, 30.0, 2, 600, 0.0, 0.0),
]


def get_transfer(lmax: int, accuracy_boost: float = 1.0) -> dict:
    p = ct.PLANCK_2018
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=p["H0"], ombh2=p["ombh2"], omch2=p["omch2"], tau=p["tau"])
    pars.InitPower.set_params(As=p["As"], ns=p["ns"])
    pars.set_for_lmax(int(lmax), lens_potential_accuracy=0)
    pars.WantTensors = False
    pars.DoLensing = False
    pars.Accuracy.AccuracyBoost = float(accuracy_boost)
    pars.Accuracy.lSampleBoost = 50.0
    data = camb.get_transfer_functions(pars)
    tr = data.get_cmb_transfer_data("scalar")
    ells = np.asarray(tr.L, dtype=np.int64)
    k = np.asarray(tr.q, dtype=np.float64)
    delta = np.asarray(tr.delta_p_l_k, dtype=np.float64)
    results = camb.get_results(pars)
    d_star = float(results.comoving_radial_distance(results.get_derived_params()["zstar"]))
    cls = np.asarray(results.get_cmb_power_spectra(pars, CMB_unit=None, raw_cl=True)["unlensed_scalar"])
    keep = ells <= lmax
    ells = ells[keep]
    ctnorm = (ells - 1.0) * ells * (ells + 1.0) * (ells + 2.0)
    return {"ells": ells, "k": k, "dT": delta[0][keep], "dE": delta[1][keep] * np.sqrt(ctnorm)[:, None],
            "d_star": d_star, "cls": cls, "params": p}


def normalisation_check(tr: dict) -> dict:
    """TT, EE, TE rebuilt from the transfer functions, divided by CAMB's own. All must be 1."""
    pk = tr["params"]["As"] * (tr["k"] / 0.05) ** (tr["params"]["ns"] - 1.0)
    w = pk[None, :] / tr["k"][None, :]
    tt = 4 * math.pi * _trapz(w * tr["dT"] ** 2, tr["k"], axis=1)
    ee = 4 * math.pi * _trapz(w * tr["dE"] ** 2, tr["k"], axis=1)
    te = 4 * math.pi * _trapz(w * tr["dT"] * tr["dE"], tr["k"], axis=1)
    out = {}
    for ell in (2, 10, 30, 100, 300, 600, 900):
        i = np.nonzero(tr["ells"] == ell)[0]
        if i.size and ell < tr["cls"].shape[0]:
            c = tr["cls"][ell]
            out[int(ell)] = {"TT": float(tt[i[0]] / c[0]), "EE": float(ee[i[0]] / c[1]), "TE": float(te[i[0]] / c[3])}
    return out


def white(lmax: int, uk_arcmin: float) -> np.ndarray:
    return np.full(lmax + 1, (uk_arcmin / T_CMB_UK * math.radians(1.0 / 60.0)) ** 2)


def noise_curves(lmax, tn, pn, fwhm, knee, slope):
    b2 = ct.gaussian_beam(lmax, fwhm) ** 2 if fwhm > 0 else np.ones(lmax + 1)
    ell = np.arange(lmax + 1, dtype=np.float64)
    nt = white(lmax, tn) / b2
    if knee < 0:
        n_hi = white(lmax, pn) / b2
        n_lo = white(lmax, 52.0) / ct.gaussian_beam(lmax, 5.0) ** 2
        ne = np.where(ell >= -knee, 1.0 / (1.0 / n_hi + 1.0 / n_lo), n_lo)
    else:
        ne = white(lmax, pn) / b2
        if knee > 0:
            ne = ne * (1.0 + (knee / np.maximum(ell, 1.0)) ** slope)
    return nt, ne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--lmax", type=int, default=1000)
    ap.add_argument("--sizes-deg", default="5,10,15,20,25,40,60")
    args = ap.parse_args()
    lmax = args.lmax

    tr = get_transfer(lmax)
    check = normalisation_check(tr)
    print(f"CAMB: lmax {lmax}, {tr['k'].size} k values, D* = {tr['d_star']:.1f} Mpc")
    print("spectra rebuilt from the transfer functions / CAMB's own (must be 1):")
    for ell, r in check.items():
        print(f"  l={ell:4d}  TT {r['TT']:.4f}  EE {r['EE']:.4f}  TE {r['TE']:.4f}")
    worst = max(abs(v - 1.0) for r in check.values() for v in r.values())
    if worst > 0.01:
        raise SystemExit(f"normalisation check failed (worst deviation {worst:.3g}); refusing to continue")

    ells = tr["ells"]
    cl_tt, cl_ee, cl_te = (np.zeros(lmax + 1) for _ in range(3))
    n = min(lmax + 1, tr["cls"].shape[0])
    cl_tt[:n], cl_ee[:n], cl_te[:n] = tr["cls"][:n, 0], tr["cls"][:n, 1], tr["cls"][:n, 3]
    ell = np.arange(lmax + 1, dtype=np.float64)
    pref = (2 * ell + 1) / (4 * math.pi)
    theta_fine = np.radians(np.linspace(0.0, 120.0, 12001))
    w5 = ct.gaussian_beam(lmax, 5.0)

    rows = []
    for size in [float(x) for x in args.sizes_deg.split(",")]:
        tc = math.radians(size)
        r = tr["d_star"] * math.cos(tc)
        a = 1.0 / tr["d_star"]
        b_t, b_e = np.zeros(lmax + 1), np.zeros(lmax + 1)
        b_t[ells] = ct.b_ell(ells, tr["k"], tr["dT"], r, a_ramp=a)
        b_e[ells] = ct.b_ell(ells, tr["k"], tr["dE"], r, a_ramp=a)
        b_t[:2] = b_e[:2] = 0.0
        peak = float(np.max(np.abs(ct.profile(theta_fine, np.arange(2, lmax + 1), b_t[2:], window=w5))))
        b_t *= 1e-5 / peak
        b_e *= 1e-5 / peak
        rec = {"theta_c_deg": size, "peak_over_sachs_wolfe_centre": peak / (a * tr["d_star"] * (1 - math.cos(tc)) / 5.0),
               "experiments": {}}
        with np.errstate(divide="ignore", invalid="ignore"):
            for label, tn, pn, fwhm, lmin_e, lmax_use, knee, slope in EXPERIMENTS:
                nt, ne = noise_curves(lmax, tn, pn, fwhm, knee, slope)
                top = lmax if lmax_use is None else min(lmax, lmax_use)
                sel = (ell >= 2) & (ell <= top)
                sel_e = sel & (ell >= lmin_e)
                ctt, cee = cl_tt + nt, cl_ee + ne
                s_t = pref * b_t**2 / ctt
                s_e = pref * b_e**2 / cee
                s_j = pref * (b_t**2 * cee - 2 * b_t * b_e * cl_te + b_e**2 * ctt) / (ctt * cee - cl_te**2)
                snr_t = math.sqrt(float(np.sum(s_t[sel])))
                snr_e = math.sqrt(float(np.sum(s_e[sel_e])))
                snr_j = math.sqrt(float(np.sum(s_j[sel_e]) + np.sum(s_t[sel & ~sel_e])))
                rec["experiments"][label] = {"snr_T": snr_t, "snr_E": snr_e, "snr_joint": snr_j,
                                             "snr_E_given_T": math.sqrt(max(snr_j**2 - snr_t**2, 0.0)),
                                             "joint_over_T": snr_j / snr_t}
            s_e_cv = np.where(ell >= 2, pref * b_e**2 / np.maximum(cl_ee, 1e-300), 0.0)
            s_t_cv = np.where(ell >= 2, pref * b_t**2 / np.maximum(cl_tt, 1e-300), 0.0)
        cum_e, cum_t = np.cumsum(s_e_cv), np.cumsum(s_t_cv)
        rec["multipole_holding_half_of_snr2"] = {"T": int(np.searchsorted(cum_t, 0.5 * cum_t[-1])), "E": int(np.searchsorted(cum_e, 0.5 * cum_e[-1]))}
        rec["multipole_holding_90pct_of_E_snr2"] = int(np.searchsorted(cum_e, 0.9 * cum_e[-1]))
        rows.append(rec)

    print("\nSNR per 1e-5 of observed temperature peak")
    for label, *_ in EXPERIMENTS:
        print(f"\n{label}")
        for rec in rows:
            e = rec["experiments"][label]
            print(f"  {rec['theta_c_deg']:4.0f} deg   T {e['snr_T']:.3f}   E {e['snr_E']:.3f}   joint {e['snr_joint']:.3f}   "
                  f"E given T {e['snr_E_given_T']:.3f}   joint/T {e['joint_over_T']:.2f}")
    print("\nmultipole holding half of the cosmic-variance-limited SNR^2 (T, E) and 90% of E:")
    for rec in rows:
        h = rec["multipole_holding_half_of_snr2"]
        print(f"  {rec['theta_c_deg']:4.0f} deg   T {h['T']:3d}   E {h['E']:3d}   E 90% {rec['multipole_holding_90pct_of_E_snr2']:3d}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"lmax": lmax, "normalisation_check": check, "rows": rows}, indent=1))
    print("\nwrote", out)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
        series = [
            ("cosmic-variance limit", "temperature + polarization, best possible", "#1baf7a"),
            ("Planck-like, polarization usable from l=30", "temperature + Planck-like polarization", "#eb6834"),
        ]
        x = [r["theta_c_deg"] for r in rows]
        fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=140)
        fig.patch.set_facecolor(surface)
        ax.set_facecolor(surface)
        y_t = [r["experiments"]["cosmic-variance limit"]["snr_T"] for r in rows]
        ax.plot(x, y_t, color="#2a78d6", lw=2, marker="o", ms=5, label="temperature only")
        for key, name, colour in series:
            ax.plot(x, [r["experiments"][key]["snr_joint"] for r in rows], color=colour, lw=2, marker="o", ms=5, label=name)
        ax.set_ylim(0, 1.6)
        ax.set_xlabel("disc radius (degrees)", color=muted)
        ax.set_ylabel("ideal signal-to-noise per 1e-5 of peak", color=muted)
        ax.set_title("Polarization lifts the search ceiling by at most 10 to 19%", color=ink, loc="left", fontsize=12, pad=12)
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
