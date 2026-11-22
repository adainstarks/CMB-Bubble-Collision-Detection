"""End-to-end check of the sensitivity measurement of ``fullsky_filter_search.py --stage inject``.

The sensitivity stage does not run the search on every injected sky. It uses that preparation and filtering
are linear: the filtered response to a unit template is computed once and added, scaled, to the filtered
signal-free sky; and it calls an injection "a candidate" when S reaches the threshold near the true position,
before the step across sizes. This script tests both shortcuts on a set of cases by doing it the long way:
the template is added to the simulated map, and the whole search (preparation, 66 filters, significance,
regions, step across sizes) runs on the sum.

Per case it reports
  * the largest difference between the shortcut's S and the S of the long way, over the pixels and filter
    sizes that the sensitivity stage looks at;
  * whether criterion (a) holds by the shortcut, and whether the full algorithm returns a candidate that the
    match rule of the search accepts (same template, hot centre, size within 5 deg, centre within the larger
    of 3 deg and a quarter of the size); and, if not, whether a candidate of any size sits there.

Cases: sizes 3, 10, 30, 60, 90 deg, the transfer template with the transfer filters and the cosine template
with the cosine filters, two amplitudes per size (the 50% amplitude of criterion (a) and the 95% amplitude of
criterion (b), read from inject_report.json), ``--repeats`` random (position, sky) pairs each. The skies are
new ones (numbers 1000 and up of the inject stream).

Usage (from the project root, after the inject stage):
    python scripts/analysis/fullsky_inject_check.py --dataset planck --out-dir runs/fullsky_search/planck_v1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import healpy as hp
import numpy as np

import fullsky_filter_search as fs

CHECK_SIZES = [3.0, 10.0, 30.0, 60.0, 90.0]


def _case(task):
    case, ti, size, pix, isim, amp, level = task
    G = fs.G
    ds, bank, thr = G["ds"], G["bank"], G["thr"]
    lmax = ds["lmax"]
    v = np.array(hp.pix2vec(fs.NSIDE, int(pix)))
    ells = np.arange(lmax + 1)
    f_true = ti * fs.NS + fs.SIZES.index(size)
    signal = np.polynomial.legendre.legval(np.clip(v @ ds["vec"], -1.0, 1.0), (2 * ells + 1) / (4 * math.pi) * bank["t_l"][f_true])
    sky = fs.simulate(isim, "inject")

    # the long way: the template is in the map before anything else happens
    w_sum = np.empty((fs.NF, fs.NPIX), dtype=np.float32)
    for f, w in fs.filtered(fs.prepare(sky + amp * signal)):
        w_sum[f] = w
    kept, dropped, _, _ = fs.run_algorithm(w_sum)

    # the shortcut of the sensitivity stage
    w_sky = np.empty((fs.NF, fs.NPIX), dtype=np.float32)
    for f, w in fs.filtered(fs.prepare(sky)):
        w_sky[f] = w
    _, _, _, local, out = fs._signal_response((ti, size, int(pix)))
    filt, resp, sig = out[ti]
    s_short = (w_sky[filt[:, None], local[None, :]] + amp * resp) / sig
    s_long = w_sum[filt[:, None], local[None, :]] / sig
    a_short = bool((s_short >= thr[filt][:, None]).any())
    a_long = bool((s_long >= thr[filt][:, None]).any())
    b_short = bool((s_short >= G["s_glob"][ti]).any())

    tol = max(3.0, 0.25 * size)

    def near(d):
        return float(np.degrees(np.arccos(np.clip(np.array(hp.pix2vec(fs.NSIDE, d["peak"])) @ v, -1.0, 1.0)))) <= tol

    same = [d for d in kept if d["template"] == fs.TEMPLATES[ti] and d["amp"] > 0.0 and near(d)]
    match = [d for d in same if abs(d["size"] - size) <= fs.THETA_ADJ]
    lost = [d for d in dropped if d["template"] == fs.TEMPLATES[ti] and d["amp"] > 0.0 and near(d) and abs(d["size"] - size) <= fs.THETA_ADJ]
    best = max(match, key=lambda d: d["S"]) if match else (max(same, key=lambda d: d["S"]) if same else None)
    return {"case": case, "template": fs.TEMPLATES[ti], "size_deg": size, "pixel": int(pix), "sky": int(isim), "amplitude": amp,
            "level": level, "n_local": int(local.size), "max_abs_S_difference": float(np.max(np.abs(s_short - s_long))),
            "largest_S_shortcut": float(s_short.max()), "largest_S_long_way": float(s_long.max()),
            "criterion_a_shortcut": a_short, "criterion_a_long_way": a_long, "criterion_b_shortcut": b_short,
            "algorithm_matching_candidate": bool(match), "algorithm_candidate_any_size": bool(same),
            "dropped_in_step_across_sizes": len(lost),
            "best_candidate": None if best is None else {"size_deg": best["size"], "S": best["S"], "amplitude": best["amp"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("planck", "planck767", "wmap7w"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--step4-amplitude", default="source", choices=("source", "peak"),
                    help="as in fullsky_filter_search.py; the first run of this check (notebook entry (ag)) used peak")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    args = ap.parse_args()
    out = Path(args.out_dir)
    G = fs.G
    G["step4"] = args.step4_amplitude
    G["ds"] = fs.load_dataset(args.dataset, Path(args.data_dir), args.seed)
    G["bank"] = fs.build_bank(G["ds"])
    G["nb"] = hp.get_all_neighbours(fs.NSIDE, np.arange(fs.NPIX)).astype(np.int32)
    G["lookup"] = np.full(fs.NPIX, -1, dtype=np.int32)
    G["sigma"] = np.load(out / "sigma.npy").astype(np.float64)
    G["sigma"][:, ~G["ds"]["good"]] = np.inf
    G["thr"] = np.array([fs.published_threshold(size) for _ in fs.TEMPLATES for size in fs.SIZES])
    null = np.load(out / "null.npz")
    G["s_glob"] = [float(np.percentile(null["max_s"][:, ti * fs.NS:(ti + 1) * fs.NS].max(axis=1), 95.0)) for ti in range(len(fs.TEMPLATES))]
    report = json.load(open(out / "inject_report.json"))
    levels = {}
    for pair in report["pairs"]:
        if pair["injected"] != pair["searched"]:
            continue
        for row in pair["by_size"]:
            levels[(pair["injected"], row["size_deg"])] = (row["candidate"]["a50"], row["stands_out"]["a95"])

    rng = np.random.default_rng(np.random.SeedSequence([args.seed, fs.STAGE_CODE["inject"], 4242]))
    unmasked = np.flatnonzero(G["ds"]["good"])
    tasks = []
    for ti, tname in enumerate(fs.TEMPLATES):
        for size in CHECK_SIZES:
            for level, amp in zip(("a50 of criterion (a)", "a95 of criterion (b)"), levels[(tname, size)]):
                if amp is None:
                    continue
                for _ in range(args.repeats):
                    tasks.append((len(tasks), ti, size, int(rng.choice(unmasked)), 1000 + len(tasks), float(amp), level))
    fs.log("inject check: %d cases" % len(tasks))
    with Pool(args.workers) as pool:
        rows = pool.map(_case, tasks, chunksize=1)

    worst = max(r["max_abs_S_difference"] for r in rows)
    table = {"a_true_match": 0, "a_true_other_size_only": 0, "a_true_no_candidate": 0, "a_false_match": 0, "a_false_no_match": 0}
    for r in rows:
        if r["criterion_a_shortcut"]:
            key = "a_true_match" if r["algorithm_matching_candidate"] else (
                "a_true_other_size_only" if r["algorithm_candidate_any_size"] else "a_true_no_candidate")
        else:
            key = "a_false_match" if r["algorithm_matching_candidate"] else "a_false_no_match"
        table[key] += 1
    agree = sum(r["criterion_a_shortcut"] == r["criterion_a_long_way"] for r in rows)
    fs.log("largest |S shortcut - S long way| over all cases: %.2e" % worst)
    fs.log("criterion (a), shortcut against long way: the same in %d of %d cases" % (agree, len(rows)))
    fs.log("criterion (a) against the full algorithm: " + json.dumps(table))
    for r in rows:
        b = r["best_candidate"]
        fs.log("  %-8s %4.0f deg  A = %5.2fe-5 (%s)  S short %.2f long %.2f  (a) %s  algorithm: %s%s" % (
            r["template"], r["size_deg"], r["amplitude"] * 1e5, r["level"][:3], r["largest_S_shortcut"], r["largest_S_long_way"],
            "yes" if r["criterion_a_shortcut"] else "no ",
            "match" if r["algorithm_matching_candidate"] else ("other size only" if r["algorithm_candidate_any_size"] else "none"),
            "" if b is None else "  (%g deg, S %.2f, amplitude %.2fe-5)" % (b["size_deg"], b["S"], b["amplitude"] * 1e5)))
    name = "inject_check%s.json" % ("_" + args.tag if args.tag else "")
    with open(out / name, "w") as fh:
        json.dump({"n_cases": len(rows), "step4_amplitude": args.step4_amplitude, "largest_S_difference": worst,
                   "criterion_a_same": agree, "table": table, "cases": rows}, fh, indent=1)
    fs.log("wrote %s" % (out / name))


if __name__ == "__main__":
    main()
