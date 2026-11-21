"""Compare rerun outputs with the April originals (restore verification).

Usage (from the project root):
    python scripts/analysis/verify_restore_compare.py metrics ORIGINAL_SUMMARY.json NEW_SUMMARY.json
    python scripts/analysis/verify_restore_compare.py scores ORIGINAL.npz NEW_DIR_OR_NPZ METHOD
"""
import glob
import json
import os
import sys

import numpy as np


def metrics(orig_path, new_path):
    o = json.load(open(orig_path))["selected_threshold_metrics"]
    n = json.load(open(new_path))["selected_threshold_metrics"]
    keys = ["threshold", "image_precision", "image_recall", "image_false_positive_rate", "image_f1",
            "image_tp", "image_fp", "image_fn", "image_tn", "hard_dice_pos", "aux_image_recall",
            "aux_image_false_positive_rate"]
    print(f"{'metric':32s} {'April':>12s} {'rerun':>12s} {'diff':>12s}")
    worst = 0.0
    for k in keys:
        if k in o and k in n:
            d = float(n[k]) - float(o[k])
            worst = max(worst, abs(d))
            print(f"{k:32s} {float(o[k]):12.6f} {float(n[k]):12.6f} {d:12.2e}")
    same_counts = all(int(o[k]) == int(n[k]) for k in ("image_tp", "image_fp", "image_fn", "image_tn") if k in o and k in n)
    print("confusion counts identical:", same_counts, "| largest absolute difference:", f"{worst:.2e}")


def scores(orig_npz, new_path, method):
    key = f"score__{method}"
    with np.load(orig_npz) as z:
        a = np.asarray(z[key], dtype=np.float64)
        labels = np.asarray(z["labels"], dtype=np.uint8)
    cands = [new_path] if new_path.endswith(".npz") else sorted(glob.glob(os.path.join(new_path, "**", "*.npz"), recursive=True))
    b = None
    for c in cands:
        with np.load(c) as z:
            for k in z.files:
                if k == key or k.endswith(method) or k == method:
                    b = np.asarray(z[k], dtype=np.float64)
                    print("rerun scores from", c, "key", k)
                    break
        if b is not None:
            break
    if b is None:
        raise SystemExit(f"no array for {method} found under {new_path}: {cands}")
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch {a.shape} vs {b.shape}")
    d = np.abs(a - b)
    neg = np.sort(a[labels == 0])
    thr = neg[int(np.ceil(0.95 * neg.size)) - 1]  # rough 5% threshold on the April scores, for a decision count only
    flips = int(np.sum((a > thr) != (b > thr)))
    print(f"rows {a.size}: max |diff| {d.max():.3e}, median |diff| {np.median(d):.3e}, "
          f"correlation {np.corrcoef(a, b)[0, 1]:.8f}, decisions that change at the 5% threshold: {flips}")


if __name__ == "__main__":
    if sys.argv[1] == "metrics":
        metrics(sys.argv[2], sys.argv[3])
    else:
        scores(sys.argv[2], sys.argv[3], sys.argv[4])
