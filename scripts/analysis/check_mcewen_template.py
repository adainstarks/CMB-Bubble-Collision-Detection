"""Which collision template stands behind the figures of McEwen et al. (2012, arXiv:1202.2861)?

Their text: profile [c0 + c1 cos(theta)] s(theta; theta_crit), with a smooth "Schwartz" step s, and "we restrict
our attention to z_crit ~ 0 uK". Their public code (COMB, github.com/astro-informatics/comb, file
src/mod/comb_tmplmap_mod.f90, function comb_tmplmap_bubble, dated November 2011) has the defaults

    zc = 0.2 * z0,  theta_0 = 1.1 * theta_c,
    value = (c0 + c1 cos(theta)) * k,  c1 = (z0 - zc) / (1 - cos(theta_c)),  c0 = z0 - c1,
    k = 1 inside theta_c, exp(1 - 1 / (1 - s^2)) with s = (theta - theta_c) / (theta_0 - theta_c) up to theta_0,
        0 outside,

and its object generator (src/prog/comb_objgen.f90) passes 1.0, 0.2, size, 1.1 * size for every bubble.

Three templates are put against three published curves (notebook entries (aj), (ak)):
  text  z_crit = 0, cut at theta_c (``cosine_coefficients``): this project's "cosine" template;
  fit   0.19 z0 at theta_c, smooth cut-off from 1.005 to 1.110 theta_c: fitted to their Fig. 1a before their
        code was found, and the variant of the pre-registered test;
  code  the defaults of their code.
Curves, all read from the figure files in the arXiv source (``read_published_figure_curves.py``):
  Fig. 1a  radial profile of the template for z0 = 100 uK, theta_crit = 10 deg;
  Fig. 1c  power spectrum of that template (the drawn quantity turns out to be |a_l0|^2, not divided by 2l+1)
           and of the CMB;
  Fig. 4a  signal-to-noise against size for z0 = 100 uK, CMB only, no beam: matched filter
           z0 * sqrt(sum_l (2l+1)/(4 pi) t_l^2 / C_l) and unfiltered field (template value at the centre over the
           rms of the CMB), sums from l = 2.
Background spectrum: CAMB with the WMAP7+BAO+H0 mean parameters and, with ``--their-spectrum``, the file
data_in/wmap_lcdm_pl_model_wmap7baoh0_CAMB.dat of their S2FIL repository (l, l(l+1) C_l / 2 pi in uK^2).

Rule fixed in (aj) for Fig. 4a, matched filter: "explained" if the rms deviation over 2 to 90 deg is below
1.5% and no size is off by more than 3%.

Usage (from the project root):
    python scripts/analysis/check_mcewen_template.py --profile-csv mcewen2012_fig1a_profile.csv \\
        --spectra-csv mcewen2012_fig1c_spectra.csv --snr-csv mcewen2012_fig4a_snr_curves.csv \\
        [--their-spectrum wmap_lcdm_pl_model_wmap7baoh0_CAMB.dat] --out report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import fullsky_filter_search as fs

Z0_UK = 100.0
TEMPLATES = {"text": None, "fit": (0.19, (1.005, 1.110, "integral")), "code": (0.2, (1.0, 1.1, "schwartz"))}


def coefficients(name, theta_c, lmax):
    if TEMPLATES[name] is None:
        return fs.cosine_coefficients(theta_c, lmax)
    edge, step = TEMPLATES[name]
    return fs.cosine_edge_coefficients(theta_c, lmax, edge, step)


def profile_value(name, theta_c, theta):
    """The template in real space, per unit z0."""
    mu_c = math.cos(theta_c)
    if TEMPLATES[name] is None:
        return np.where(theta <= theta_c, (np.cos(theta) - mu_c) / (1.0 - mu_c), 0.0)
    edge, step = TEMPLATES[name]
    cut = fs.schwartz_step if step[2] == "schwartz" else fs.smooth_step
    return (1.0 - (1.0 - edge) * (1.0 - np.cos(theta)) / (1.0 - mu_c)) * cut(theta, step[0] * theta_c, step[1] * theta_c)


def read_curves(path, key):
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(r["curve"], []).append((float(r[key[0]]), float(r[key[1]])))
    return {k: np.array(v) for k, v in out.items()}


def check_profile(path):
    d = read_curves(path, ("theta_deg", "delta_t_uK"))["profile"]
    theta, val = d[:, 0], d[:, 1]
    tc = math.radians(10.0)
    shape = (1.0 - np.cos(np.radians(theta))) / (1.0 - math.cos(tc))
    fits = []
    for tmax in (6.0, 7.0, 8.0, 8.5, 9.0, 9.5, 9.9):
        m = theta <= tmax
        edge = float(np.sum((val[m] - Z0_UK * (1.0 - shape[m])) * shape[m]) / np.sum(shape[m] ** 2))
        rms = float(np.sqrt(np.mean((val[m] - (Z0_UK * (1.0 - shape[m]) + edge * shape[m])) ** 2)))
        fits.append({"fit_up_to_deg": tmax, "value_at_10_deg_uK": edge, "rms_uK": rms})
    res = {"n_points": int(theta.size), "straight_line_fits": fits,
           "last_point_above_0p2_uK_deg": float(theta[val > 0.2].max()), "templates": {}}
    print("Fig. 1a, drawn profile (z0 = 100 uK, 10 deg): straight line in cos(theta) inside the disc, value at 10 deg "
          + ", ".join("%.1f" % f["value_at_10_deg_uK"] for f in fits) + " uK for fits up to 6 ... 9.9 deg; reaches zero at %.1f deg" % (
              res["last_point_above_0p2_uK_deg"] + 0.1))
    for name in TEMPLATES:
        model = Z0_UK * profile_value(name, tc, np.radians(theta))
        res["templates"][name] = {"rms_uK": float(np.sqrt(np.mean((val - model) ** 2))), "largest_difference_uK": float(np.max(np.abs(val - model))),
                                  "value_at_5_deg_uK": float(Z0_UK * profile_value(name, tc, np.radians(np.array([5.0])))[0]),
                                  "value_at_9p99_deg_uK": float(Z0_UK * profile_value(name, tc, np.radians(np.array([9.99])))[0])}
        print("   %-4s template against the drawn curve: rms %.2f uK, largest difference %.2f uK" % (
            name, res["templates"][name]["rms_uK"], res["templates"][name]["largest_difference_uK"]))
    return res


def check_spectra(path, spectra):
    d = read_curves(path, ("ell", "value_uK2"))
    lt, vt = d["template"][:, 0], d["template"][:, 1]
    lr = np.round(lt).astype(int)
    lint = np.unique(lr)
    lint = lint[(lint >= 2) & (lint <= 256)]
    drawn = np.array([float(np.exp(np.mean(np.log(vt[lr == l])))) for l in lint])
    tc = math.radians(10.0)
    ell = np.arange(301)
    res = {"drawn_quantity": "|a_l0|^2 in uK^2", "templates": {}, "table": []}
    spec = {name: Z0_UK**2 * coefficients(name, tc, 300) ** 2 * (2 * ell + 1) / (4.0 * math.pi) for name in TEMPLATES}
    # which quantity is drawn: the ratio to |a_l0|^2 / (2l+1) follows 2l+1 for the right template
    low = [int(l) for l in lint if l <= 20]
    res["definition_check"] = [{"l": l, "two_l_plus_1": 2 * l + 1,
                                **{name: float(drawn[list(lint).index(l)] / (spec[name][l] / (2 * l + 1))) for name in TEMPLATES}} for l in low]
    print("Fig. 1c, drawn template spectrum: ratio to |a_l0|^2 / (2l+1) at l = %s" % low)
    print("   2l+1: " + " ".join("%d" % (2 * l + 1) for l in low))
    for name in TEMPLATES:
        print("   %-4s: " % name + " ".join("%.1f" % r[name] for r in res["definition_check"]))
    sel = lint <= 60
    for name, sp in spec.items():
        ratio = np.log10(drawn[sel] / np.maximum(sp[lint[sel]], 1e-300))
        a, b = lint[sel] <= 20, lint[sel] > 20
        bands = {}
        for lo, hi in ((61, 100), (101, 160), (161, 256)):
            m = (lint >= lo) & (lint <= hi)
            bands["%d_%d" % (lo, hi)] = float(drawn[m].max() / sp[lo:hi + 1].max())
        res["templates"][name] = {"log10_drawn_over_model": {
            "l_2_20": {"median": float(np.median(ratio[a])), "p16": float(np.percentile(ratio[a], 16)), "p84": float(np.percentile(ratio[a], 84))},
            "l_21_60": {"median": float(np.median(ratio[b])), "p16": float(np.percentile(ratio[b], 16)), "p84": float(np.percentile(ratio[b], 84))}},
            "largest_value_in_band_drawn_over_model": bands}
        r = res["templates"][name]["log10_drawn_over_model"]
        print("   %-4s |a_l0|^2: log10(drawn / model), l 2-20: median %+.3f (16-84%%: %+.3f to %+.3f); l 21-60: median %+.3f (%+.3f to %+.3f); "
              "largest value in band, drawn / model: %s" % (name, r["l_2_20"]["median"], r["l_2_20"]["p16"], r["l_2_20"]["p84"],
                                                            r["l_21_60"]["median"], r["l_21_60"]["p16"], r["l_21_60"]["p84"],
                                                            " ".join("%s %.2f" % kv for kv in bands.items())))
    res["table"] = [{"l": int(l), "drawn": float(v), **{name: float(spec[name][l]) for name in TEMPLATES}} for l, v in zip(lint, drawn)]
    # the CMB curve of the same figure against the background spectra used here
    lc, vc = d["cmb"][:, 0], d["cmb"][:, 1]
    res["cmb_curve"] = {}
    for sname, cl in spectra.items():
        ratios = [float(np.exp(np.interp(math.log(l), np.log(lc), np.log(vc))) / cl[l]) for l in (2, 3, 5, 10, 20, 50, 100, 150, 200, 220, 250)]
        res["cmb_curve"][sname] = {"drawn_over_C_l_at_l_2_to_250": ratios, "median": float(np.median(ratios))}
        print("   their drawn CMB curve / C_l (%s): median %.3f, range %.3f to %.3f (a dashed line of 82 points)" % (
            sname, np.median(ratios), min(ratios), max(ratios)))
    return res


def check_snr(path, spectra):
    pub = {"matched_filter": {}, "unfiltered": {}}
    for r in csv.DictReader(open(path)):
        pub[r["curve"]][int(round(float(r["theta_crit_deg"])))] = float(r["snr"])
    sizes = [t for t in sorted(pub["matched_filter"]) if t >= 2]
    usizes = [t for t in sorted(pub["unfiltered"]) if t >= 2]
    show = [2, 3, 5, 7, 10, 14, 20, 30, 45, 60, 75, 90]
    res = {"published": {k: {str(t): v for t, v in d.items()} for k, d in pub.items()}, "runs": {},
           "unfiltered_level_1_to_4_deg": float(np.mean([pub["unfiltered"][t] for t in (1, 2, 3, 4) if t in pub["unfiltered"]]))}
    print("Fig. 4a: their unfiltered curve at 1 to 4 deg reads %.3f = 100 uK / %.1f uK" % (
        res["unfiltered_level_1_to_4_deg"], Z0_UK / res["unfiltered_level_1_to_4_deg"]))
    for sname, cl in spectra.items():
        for lmax in (256, 512):
            ells = np.arange(2, lmax + 1)
            wt = (2 * ells + 1) / (4 * math.pi)
            sigma = math.sqrt(float(np.sum(wt * cl[2:lmax + 1])))
            for name in TEMPLATES:
                mf, un = {}, {}
                for t in range(1, 91):
                    t_l = coefficients(name, math.radians(t), lmax)[2:]
                    mf[t] = Z0_UK * math.sqrt(float(np.sum(wt * t_l**2 / cl[2:lmax + 1])))
                    un[t] = Z0_UK * float(np.sum(wt * t_l)) / sigma
                rm = np.array([mf[t] / pub["matched_filter"][t] for t in sizes])
                ru = np.array([un[t] / pub["unfiltered"][t] for t in usizes])
                rms_m = float(np.sqrt(np.mean((rm - 1.0) ** 2)))
                worst = float(np.max(np.abs(rm - 1.0)))
                key = "%s, l <= %d, %s spectrum" % (name, lmax, sname)
                res["runs"][key] = {"template": name, "lmax": lmax, "spectrum": sname, "rms_cmb_uK": sigma,
                                    "matched": {str(t): mf[t] for t in range(1, 91)}, "unfiltered": {str(t): un[t] for t in range(1, 91)},
                                    "matched_over_published": {"mean": float(rm.mean()), "min": float(rm.min()), "max": float(rm.max()),
                                                               "rms_deviation": rms_m, "largest_deviation": worst, "n_sizes": int(rm.size),
                                                               "at": {str(t): mf[t] / pub["matched_filter"][t] for t in show if t in pub["matched_filter"]}},
                                    "unfiltered_over_published": {"mean": float(ru.mean()), "min": float(ru.min()), "max": float(ru.max()),
                                                                  "rms_deviation": float(np.sqrt(np.mean((ru - 1.0) ** 2))), "n_sizes": int(ru.size)},
                                    "explained_by_rule_of_aj": bool(rms_m < 0.015 and worst <= 0.03)}
                print("   %-34s matched / published: %s | rms dev %.1f%%, largest %.1f%% -> %s | unfiltered: mean %.3f, rms dev %.1f%%" % (
                    key, " ".join("%d:%.3f" % (t, mf[t] / pub["matched_filter"][t]) for t in show if t in pub["matched_filter"]),
                    100 * rms_m, 100 * worst, "explained" if res["runs"][key]["explained_by_rule_of_aj"] else "not by the rule",
                    ru.mean(), 100 * res["runs"][key]["unfiltered_over_published"]["rms_deviation"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-csv", required=True)
    ap.add_argument("--spectra-csv", required=True)
    ap.add_argument("--snr-csv", required=True)
    ap.add_argument("--their-spectrum", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cl, _ = fs.camb_tt_cls(lmax=600, params=fs.WMAP7)
    spectra = {"CAMB": np.asarray(cl, dtype=np.float64) * 1.0e12}          # K^2 -> uK^2
    report = {"templates": {"text": "z_crit = 0, cut at theta_c", "fit": "0.19 z0 at theta_c, cut-off 1.005 to 1.110 theta_c (integral of the bump)",
                            "code": "0.2 z0 at theta_c, cut-off exp(1 - 1/(1 - s^2)) from theta_c to 1.1 theta_c (comb_tmplmap_bubble)"}}
    if args.their_spectrum:
        rows = [line.split() for line in open(args.their_spectrum) if line.strip()]
        theirs = np.zeros(int(rows[-1][0]) + 1)
        for r in rows:
            l = int(r[0])
            theirs[l] = float(r[1]) * 2.0 * math.pi / (l * (l + 1.0))
        spectra["their file"] = theirs
        report["their_spectrum_over_camb"] = {str(l): float(theirs[l] / spectra["CAMB"][l]) for l in (2, 5, 10, 20, 50, 100, 200, 256, 400, 512)}
        print("their spectrum file / CAMB: " + " ".join("%s:%.3f" % kv for kv in report["their_spectrum_over_camb"].items()))
    report["fig1a_profile"] = check_profile(args.profile_csv)
    report["fig1c_spectrum"] = check_spectra(args.spectra_csv, spectra)
    report["fig4a_signal_to_noise"] = check_snr(args.snr_csv, spectra)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
