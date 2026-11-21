"""
Mine false negatives and weak recoveries by physical and nuisance bins.

This script converts "recall is rough" into concrete generator/training targets:
which amplitude, radius, edge, offset, and local-background families are being
missed under a frozen operating point.
"""

from __future__ import annotations
# Bootstrap organized script folders for direct CLI execution.
import sys as _cmb_sys
from pathlib import Path as _cmb_Path

_cmb_scripts_root = _cmb_Path(__file__).resolve().parents[1]
_cmb_paths = [_cmb_scripts_root]
_cmb_paths.extend(
    sorted(
        _cmb_path
        for _cmb_path in _cmb_scripts_root.iterdir()
        if _cmb_path.is_dir() and _cmb_path.name != "__pycache__"
    )
)
for _cmb_path in reversed(_cmb_paths):
    _cmb_path_text = str(_cmb_path)
    if _cmb_path_text not in _cmb_sys.path:
        _cmb_sys.path.insert(0, _cmb_path_text)
del _cmb_Path, _cmb_sys, _cmb_scripts_root, _cmb_paths, _cmb_path, _cmb_path_text

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from phase3_audit_outputs import load_jsonl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Phase 3 positive misses by physics and nuisance bins.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--data-h5", type=str, default="")
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_h5(eval_dir, data_h5_arg):
    if data_h5_arg:
        return Path(data_h5_arg).resolve()
    summary = load_json(Path(eval_dir) / "evaluation_summary.json")
    run_config = load_json(Path(summary["run_dir"]) / "run_config.json")
    return Path(run_config["data_h5"]).resolve()


def bin_value(value, edges, labels):
    for idx in range(len(edges) - 1):
        if edges[idx] <= value < edges[idx + 1]:
            return labels[idx]
    return labels[-1]


def amplitude_bin(z0, zcrit):
    amp = max(abs(float(z0)), abs(float(zcrit)))
    if amp < 1e-5:
        return "1e-6_to_1e-5"
    if amp < 3e-5:
        return "1e-5_to_3e-5"
    if amp < 5e-5:
        return "3e-5_to_5e-5"
    return "5e-5_to_1e-4"


def edge_bin(zcrit):
    z = abs(float(zcrit))
    if z <= 5e-6:
        return "smooth_|zcrit|<=5e-6"
    if z < 3e-5:
        return "weak_5e-6_to_3e-5"
    return "strong_|zcrit|>=3e-5"


def init_bin():
    return {
        "num_positive": 0,
        "detected": 0,
        "missed": 0,
        "mean_score_max": 0.0,
        "mean_patch_std_k": 0.0,
        "mean_mask_fraction": 0.0,
        "sample_misses": [],
    }


def update_bin(acc, record, patch_std):
    acc["num_positive"] += 1
    detected = bool(record.get("has_candidate", False))
    acc["detected"] += int(detected)
    acc["missed"] += int(not detected)
    acc["mean_score_max"] += float(record.get("score_max", 0.0) or 0.0)
    acc["mean_patch_std_k"] += float(patch_std)
    acc["mean_mask_fraction"] += float(record.get("coord_mask_fraction", np.nan))
    if not detected and len(acc["sample_misses"]) < 25:
        acc["sample_misses"].append(int(record["sample_index"]))


def finalize_bins(bins):
    out = {}
    for key, acc in sorted(bins.items()):
        n = max(acc["num_positive"], 1)
        out[key] = {
            "num_positive": int(acc["num_positive"]),
            "detected": int(acc["detected"]),
            "missed": int(acc["missed"]),
            "recall": float(acc["detected"] / n),
            "miss_rate": float(acc["missed"] / n),
            "mean_score_max": float(acc["mean_score_max"] / n),
            "mean_patch_std_k": float(acc["mean_patch_std_k"] / n),
            "mean_mask_fraction": float(acc["mean_mask_fraction"] / n),
            "sample_misses": acc["sample_misses"],
        }
    return out


def mine(eval_dir, data_h5):
    records = load_jsonl(Path(eval_dir) / "candidate_records.jsonl")
    positive_records = [row for row in records if int(row.get("truth_label", 0)) == 1]
    groups = {
        "theta_crit": {},
        "amplitude": {},
        "edge_strength": {},
        "offcenter_distance": {},
        "local_background_std": {},
        "mask_fraction": {},
        "joint_edge_amplitude": {},
    }
    with h5py.File(data_h5, "r") as h5:
        patches = h5["patches"]
        for record in positive_records:
            sample_idx = int(record["sample_index"])
            patch_std = float(np.std(patches[sample_idx]))
            theta = float(record["truth_theta_crit_deg"])
            z0 = float(record["truth_z0"])
            zcrit = float(record["truth_zcrit"])
            offset = float(np.hypot(float(record["truth_signal_center_dx_deg"]), float(record["truth_signal_center_dy_deg"])))
            mask_fraction = float(record.get("coord_mask_fraction", np.nan))

            labels = {
                "theta_crit": bin_value(theta, [5, 10, 15, 20, 25.1], ["5-10deg", "10-15deg", "15-20deg", "20-25deg"]),
                "amplitude": amplitude_bin(z0, zcrit),
                "edge_strength": edge_bin(zcrit),
                "offcenter_distance": bin_value(offset, [0, 2, 5, 10, 99], ["0-2deg", "2-5deg", "5-10deg", "10deg+"]),
                "local_background_std": bin_value(
                    patch_std,
                    [0, 9.7e-5, 1.03e-4, 1.1e-4, 9],
                    ["low_std", "mid_std", "high_std", "extreme_std"],
                ),
                "mask_fraction": bin_value(mask_fraction, [0, 0.97, 0.985, 1.01], ["lower_clean_fraction", "mid_clean_fraction", "highest_clean_fraction"]),
            }
            labels["joint_edge_amplitude"] = f"{labels['edge_strength']}__{labels['amplitude']}"
            for group_name, label in labels.items():
                groups[group_name].setdefault(label, init_bin())
                update_bin(groups[group_name][label], record, patch_std)

    detected = sum(1 for row in positive_records if bool(row.get("has_candidate", False)))
    report = {
        "eval_dir": str(Path(eval_dir).resolve()),
        "data_h5": str(Path(data_h5).resolve()),
        "num_positive": int(len(positive_records)),
        "detected": int(detected),
        "missed": int(len(positive_records) - detected),
        "recall": float(detected / max(len(positive_records), 1)),
        "groups": {name: finalize_bins(group) for name, group in groups.items()},
    }
    return report


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir).resolve()
    data_h5 = resolve_data_h5(eval_dir, args.data_h5)
    report = mine(eval_dir, data_h5)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: report[k] for k in ("eval_dir", "num_positive", "detected", "missed", "recall")}, indent=2))
    print(f"Error-mining report: {output_json}")


if __name__ == "__main__":
    main()
