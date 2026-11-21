"""Detection probability split by the edge ratio |z_crit| / |z0|.

Assumptions
-----------
* Uses scores that already exist (no model is run). Rows of the scores file line up with rows of the HDF5.
* Each method gets its own threshold at the target false-positive rate on the negatives, through the
  project's ``conformal_threshold_from_scores``, exactly as the original sensitivity report did.
* The question: how much of each detector's sensitivity comes from the temperature jump at the disc edge,
  which post-2011 theory does not predict (notebook entries (b), (c)).

Usage (from the project root):
    python scripts/analysis/zcrit_ratio_split.py --h5 H5 --scores-npz NPZ --out-json OUT [--label NAME]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import h5py
import numpy as np

from phase3_thresholds import conformal_threshold_from_scores


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def find(h5, names):
    for group in ("truth", "stratification", "metadata", "/"):
        g = h5[group] if group in h5 else None
        if g is None:
            continue
        for name in names:
            if name in g:
                return np.asarray(g[name][:]), f"{group}/{name}"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--scores-npz", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--fpr-target", type=float, default=0.05)
    args = ap.parse_args()

    with np.load(args.scores_npz) as z:
        labels = np.asarray(z["labels"], dtype=np.uint8)
        scores = {k[len("score__"):]: np.asarray(z[k], dtype=np.float64) for k in z.files if k.startswith("score__")}

    with h5py.File(args.h5, "r") as h5:
        print("h5 groups:", {k: (list(h5[k].keys())[:40] if isinstance(h5[k], h5py.Group) else h5[k].shape) for k in h5.keys()})
        amp, amp_src = find(h5, ["amplitude", "z0_abs"])
        if amp is None:
            z0, amp_src = find(h5, ["z0"])
            amp = np.abs(z0)
        theta, _ = find(h5, ["theta_crit_deg"])
        ratio, ratio_src = find(h5, ["zcrit_ratio"])
        if ratio is None:
            zc, s1 = find(h5, ["zcrit"])
            z0, s2 = find(h5, ["z0"])
            ratio = np.where(np.abs(z0) > 0, np.abs(zc) / np.where(np.abs(z0) > 0, np.abs(z0), 1), np.nan)
            ratio_src = f"|{s1}|/|{s2}|"
        h5_labels = np.asarray(h5["labels"][:], dtype=np.uint8) if "labels" in h5 else None

    if h5_labels is not None and not np.array_equal(h5_labels, labels):
        raise SystemExit("labels in the scores file do not match labels in the HDF5")
    print(f"rows={labels.size} positives={int(labels.sum())} negatives={int((labels == 0).sum())}")
    print(f"amplitude from {amp_src}; ratio from {ratio_src}")
    pos = labels == 1
    amp = np.asarray(amp, dtype=np.float64)
    ratio = np.round(np.asarray(ratio, dtype=np.float64), 3)
    ratios = sorted(set(ratio[pos][np.isfinite(ratio[pos])].tolist()))
    amps = sorted(set(np.round(amp[pos], 12).tolist()))
    print("edge ratios present:", ratios)
    print("amplitudes present:", amps)

    bands = {
        "mid (1e-5 to 5e-5)": (amp >= 0.99e-5) & (amp <= 5.01e-5),
        "low (<= 5e-6)": amp <= 5.01e-6,
        "high (1e-4)": amp >= 0.99e-4,
        "all": np.ones_like(pos),
    }
    out = {"label": args.label, "h5": args.h5, "scores_npz": args.scores_npz, "fpr_target": args.fpr_target, "methods": {}}
    for name, s in scores.items():
        rec = conformal_threshold_from_scores(s, labels, args.fpr_target)
        thr = float(rec["threshold"])
        m = {"threshold": thr, "negative_fpr": float(rec.get("negative_fpr", float("nan"))), "bands": {}}
        for band, sel in bands.items():
            rows = {}
            for r in ratios + ["pooled"]:
                mask = pos & sel if r == "pooled" else pos & sel & (ratio == r)
                n = int(mask.sum())
                k = int(np.sum(s[mask] > thr))
                lo, hi = wilson(k, n)
                rows[str(r)] = {"k": k, "n": n, "p": (k / n if n else float("nan")), "ci95": [lo, hi]}
            m["bands"][band] = rows
        out["methods"][name] = m

    print(f"\n=== {args.label}: detection probability by edge ratio, threshold at {args.fpr_target:.0%} false positives ===")
    for band in bands:
        print(f"\n-- amplitude band: {band}")
        print(f"{'method':32s}" + "".join(f"{('ratio ' + str(r)):>22s}" for r in ratios) + f"{'pooled':>22s}")
        for name, m in out["methods"].items():
            cells = []
            for r in [str(x) for x in ratios] + ["pooled"]:
                c = m["bands"][band][r]
                cells.append(f"{c['p']:.3f} [{c['ci95'][0]:.3f},{c['ci95'][1]:.3f}]")
            print(f"{name:32s}" + "".join(f"{c:>22s}" for c in cells))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", args.out_json)


if __name__ == "__main__":
    main()
