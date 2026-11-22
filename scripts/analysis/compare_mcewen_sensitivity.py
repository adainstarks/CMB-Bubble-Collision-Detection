"""Two like-for-like comparisons of the full-sky filter search with McEwen et al. (2012, arXiv:1202.2861).

Pre-registered in notebook entry (af), checks 2 and 3.

1. Signal-to-noise of the matched filter for a cosine collision of z0 = 100 uK against its size (their Fig. 4a).
   Their curve is for a CMB-only background of the WMAP7+BAO+H0 spectrum without beam (their caption and Sec.
   III C). The same is computed here from the exact Legendre coefficients of the template:
       SNR = z0 * sqrt( sum_{l=2}^{lmax} (2l+1)/(4 pi) t_l^2 / C_l ).
2. Their exclusion and sensitivity regions (Fig. 6, Sec. IV B): collisions at two positions, three simulated
   WMAP W skies each; a (size, amplitude) pair is "exclusion" if all six are detected, "sensitivity" if some,
   "unprobed" if none. Compared with ``fullsky_filter_search.py --stage inject`` on the wmap7w data set at the
   same two positions (detection = criterion (a), cosine template and filters, many simulated skies).
   Correspondence fixed beforehand: all six detected has probability one half at a detection probability of
   0.89, none of six at 0.11. Their exclusion boundary is compared with the amplitude at which the detection
   fraction reaches 0.89 at the worse of the two positions; their lower boundary with the amplitude at 0.11 at
   the better position. "Same" = within one step of their amplitude grid.
   Added afterwards (not pre-registered): for every cell of their grid, the probability of their class under
   the detection probabilities measured here.

The published numbers come from ``read_published_figure_curves.py mcewen-snr`` and ``mcewen-regions``.

Usage (from the project root):
    python scripts/analysis/compare_mcewen_sensitivity.py --inject-report inject_report_mcewen.json \\
        --snr-csv mcewen2012_fig4a_snr_curves.csv --regions-csv mcewen2012_fig6_regions.csv --out report.json
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

T_UK = 2.725e6


def level(z, frac, target):
    above = np.flatnonzero(frac >= target)
    if not above.size:
        return None
    k = int(above[0])
    if k == 0:
        return float(z[0])
    return float(z[k - 1] + (z[k] - z[k - 1]) * (target - frac[k - 1]) / (frac[k] - frac[k - 1]))


def snr_comparison(snr_csv):
    import fullsky_filter_search as fs
    pub = {"matched_filter": {}, "unfiltered": {}}
    for r in csv.DictReader(open(snr_csv)):
        pub[r["curve"]][int(round(float(r["theta_crit_deg"])))] = float(r["snr"])
    cl, _ = fs.camb_tt_cls(lmax=fs.LMAX_SIM, params=fs.WMAP7)
    cl = np.asarray(cl, dtype=np.float64)
    z0 = 100.0e-6
    rows = []
    for theta in sorted(pub["matched_filter"]):
        row = {"theta_deg": theta, "published": pub["matched_filter"][theta]}
        for lmax in (256, 767):
            t_l = fs.cosine_coefficients(math.radians(theta), lmax)
            ells = np.arange(lmax + 1)
            row["mine_l2_%d" % lmax] = z0 * math.sqrt(float(np.sum((2 * ells[2:] + 1) / (4 * math.pi) * t_l[2:] ** 2 / cl[2:lmax + 1])))
        row["ratio_256"] = row["mine_l2_256"] / row["published"]
        rows.append(row)
    ratio = {r["theta_deg"]: r["ratio_256"] for r in rows}
    sel = [v for t, v in ratio.items() if t >= 5]
    summary = {"sizes_from_5_deg": len(sel), "ratio_mean": float(np.mean(sel)), "ratio_min": float(np.min(sel)), "ratio_max": float(np.max(sel)),
               "sizes_outside_5_percent": [t for t, v in ratio.items() if t >= 5 and abs(v - 1.0) > 0.05],
               "ratio_at": {str(t): ratio[t] for t in (2, 3, 5, 6, 8, 10, 20, 30, 45, 60, 75, 90) if t in ratio}}
    print("signal-to-noise of a 100 uK collision, mine (CMB only, l 2..256) / published:")
    print("  " + "  ".join("%d deg %.3f" % (t, ratio[t]) for t in (2, 3, 5, 6, 8, 10, 20, 30, 45, 60, 75, 90) if t in ratio))
    print("  from 5 deg up: mean %.4f, min %.4f, max %.4f; outside 5%%: %s" % (
        summary["ratio_mean"], summary["ratio_min"], summary["ratio_max"], summary["sizes_outside_5_percent"]))
    return {"rows": rows, "summary": summary}


def regions_comparison(report_path, regions_csv):
    rep = json.load(open(report_path))
    assert rep.get("positions_lonlat"), "the inject report must come from a run with --inject-lonlat"
    grid_sizes, grid_amps, cls = [], [], {}
    for r in csv.DictReader(open(regions_csv)):
        s, a = float(r["theta_crit_deg"]), float(r["z0_uK"])
        cls[(s, a)] = r["class"]
        if s not in grid_sizes:
            grid_sizes.append(s)
        if a not in grid_amps:
            grid_amps.append(a)
    grid_amps = sorted(grid_amps)
    amps = np.array(rep["amplitudes"])
    pair = next(p for p in rep["pairs"] if p["injected"] == "cosine" and p["searched"] == "cosine")
    pv = np.array(rep["peak_value"]["cosine"])
    curves = {}
    for i, row in enumerate(pair["by_size"]):
        curves[row["size_deg"]] = (amps / pv[i] * T_UK, np.array(row["candidate"]["fraction_by_position"]))

    def fractions(size, z0):
        """Detection fraction at the two positions for a size and z0 (uK); a size between two of mine is averaged."""
        if size in curves:
            z, fr = curves[size]
            return [float(np.interp(z0, z, fr[p])) for p in range(fr.shape[0])], [size]
        lo = max(s for s in curves if s < size)
        hi = min(s for s in curves if s > size)
        both = [[float(np.interp(z0, curves[q][0], curves[q][1][p])) for p in range(2)] for q in (lo, hi)]
        return [0.5 * (both[0][p] + both[1][p]) for p in range(2)], [lo, hi]

    def boundary(size, target, pick):
        if size in curves:
            z, fr = curves[size]
            return pick(level(z, fr[p], target) for p in range(fr.shape[0])), [size]
        lo = max(s for s in curves if s < size)
        hi = min(s for s in curves if s > size)
        vals = [pick(level(curves[q][0], curves[q][1][p], target) for p in range(2)) for q in (lo, hi)]
        return 0.5 * (vals[0] + vals[1]), [lo, hi]

    out = {"positions_lonlat": rep["positions_lonlat"], "n_skies": rep["n_skies"], "grid_amplitudes_uK": grid_amps, "sizes": []}
    print("exclusion and sensitivity regions (z0 in uK):")
    for size in grid_sizes:
        black = min(a for a in grid_amps if cls[(size, a)] == "exclusion")
        grey = min(a for a in grid_amps if cls[(size, a)] != "unprobed")
        ib, ig = grid_amps.index(black), grid_amps.index(grey)
        mine_hi, used = boundary(size, 0.89, max)
        mine_lo, _ = boundary(size, 0.11, min)
        allowed_hi = ((grid_amps[ib - 2] if ib >= 2 else 0.0), (grid_amps[ib + 1] if ib + 1 < len(grid_amps) else float("inf")))
        allowed_lo = ((grid_amps[ig - 2] if ig >= 2 else 0.0), (grid_amps[ig + 1] if ig + 1 < len(grid_amps) else float("inf")))
        cells = []
        for a in grid_amps:
            ps, _ = fractions(size, a)
            p_all, p_none = ps[0] ** 3 * ps[1] ** 3, (1 - ps[0]) ** 3 * (1 - ps[1]) ** 3
            prob = {"exclusion": p_all, "sensitivity": 1.0 - p_all - p_none, "unprobed": p_none}
            cells.append({"z0_uK": a, "detection_fraction": ps, "published_class": cls[(size, a)], "probability_of_published_class": prob[cls[(size, a)]]})
        row = {"size_deg": size, "my_sizes_used": used,
               "published_exclusion_between_uK": [grid_amps[ib - 1], black], "mine_at_0p89_worse_position_uK": mine_hi,
               "exclusion_within_one_grid_step": bool(allowed_hi[0] <= mine_hi <= allowed_hi[1]), "exclusion_allowed_uK": list(allowed_hi),
               "published_lower_between_uK": [(grid_amps[ig - 1] if ig > 0 else 0.0), grey], "mine_at_0p11_better_position_uK": mine_lo,
               "lower_within_one_grid_step": bool(allowed_lo[0] <= mine_lo <= allowed_lo[1]), "lower_allowed_uK": list(allowed_lo),
               "cells": cells}
        out["sizes"].append(row)
        print("  %4.0f deg | exclusion: published %g-%g, mine %.1f, %s | lower: published %g-%g, mine %.1f, %s | least likely cell: %.3f" % (
            size, grid_amps[ib - 1], black, mine_hi, "inside one step" if row["exclusion_within_one_grid_step"] else "OUTSIDE",
            row["published_lower_between_uK"][0], grey, mine_lo, "inside one step" if row["lower_within_one_grid_step"] else "OUTSIDE",
            min(c["probability_of_published_class"] for c in cells)))
    inner = [r for r in out["sizes"] if 5.0 <= r["size_deg"] <= 60.0]
    out["criterion_sizes_5_to_60"] = {"comparisons": 2 * len(inner),
                                      "inside": sum(r["exclusion_within_one_grid_step"] + r["lower_within_one_grid_step"] for r in inner)}
    all_cells = [c for r in out["sizes"] for c in r["cells"]]
    out["cells"] = {"n": len(all_cells), "published_class_below_0p05": sum(c["probability_of_published_class"] < 0.05 for c in all_cells)}
    print("  sizes 5 to 60 deg: %d of %d boundary comparisons inside one grid step; cells where the published class has probability below 0.05: %d of %d"
          % (out["criterion_sizes_5_to_60"]["inside"], out["criterion_sizes_5_to_60"]["comparisons"],
             out["cells"]["published_class_below_0p05"], out["cells"]["n"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject-report", required=True)
    ap.add_argument("--snr-csv", required=True)
    ap.add_argument("--regions-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {"signal_to_noise_curve": snr_comparison(args.snr_csv), "regions": regions_comparison(args.inject_report, args.regions_csv)}
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
