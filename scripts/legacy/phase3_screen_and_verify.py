"""
Fuse proposal, boundary-verifier, and template-fit outputs into one ranked table.

This is a packaging layer for the two-branch Phase 3 interpretation:
    Phase 3A: broad proposal branch
    Phase 3B: boundary-aware verification branch
    Phase 3C: template-fit handoff for emitted candidates
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
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a ranked candidate table from proposal, boundary-verifier, and template-fit artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--proposal-eval-dir", type=str, required=True)
    parser.add_argument("--verifier-eval-dir", type=str, required=True)
    parser.add_argument("--template-records", type=str, default="")
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-summary", type=str, required=True)
    parser.add_argument(
        "--keep-policy",
        type=str,
        default="union",
        choices=["union", "proposal", "verifier", "both"],
        help="Candidate-emission rule for the table.",
    )
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def by_sample(rows):
    return {int(row["sample_index"]): row for row in rows}


def safe_float(value, default=0.0):
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def pass_policy(proposal_pass, verifier_pass, policy):
    if policy == "proposal":
        return proposal_pass
    if policy == "verifier":
        return verifier_pass
    if policy == "both":
        return proposal_pass and verifier_pass
    return proposal_pass or verifier_pass


def risk_tag(proposal_pass, verifier_pass, template_delta):
    if proposal_pass and verifier_pass:
        return "priority_consensus"
    if verifier_pass and not proposal_pass:
        return "boundary_only_low_null"
    if proposal_pass and not verifier_pass:
        if template_delta is not None and template_delta >= 5e-5:
            return "proposal_only_template_supported"
        return "proposal_only_high_null_risk"
    return "not_emitted"


def ranking_score(proposal, verifier, template):
    proposal_score = safe_float(proposal.get("score_max"))
    verifier_score = safe_float(verifier.get("score_max"))
    template_delta = safe_float(template.get("delta_chi2_vs_plane_null")) if template else 0.0
    template_term = min(template_delta / 1e-4, 3.0)
    consensus_bonus = 0.5 if proposal.get("has_candidate") and verifier.get("has_candidate") else 0.0
    return float(0.35 * proposal_score + 0.45 * verifier_score + 0.20 * template_term + consensus_bonus)


def choose_sky_value(primary, fallback, key):
    value = primary.get(key)
    if value is not None:
        return value
    return fallback.get(key)


def build_rows(args):
    proposal_dir = Path(args.proposal_eval_dir)
    verifier_dir = Path(args.verifier_eval_dir)
    proposal_summary = read_json(proposal_dir / "evaluation_summary.json")
    verifier_summary = read_json(verifier_dir / "evaluation_summary.json")
    proposal_records = by_sample(read_jsonl(proposal_dir / "candidate_records.jsonl"))
    verifier_records = by_sample(read_jsonl(verifier_dir / "candidate_records.jsonl"))
    template_records = by_sample(read_jsonl(args.template_records)) if args.template_records else {}

    sample_indices = sorted(set(proposal_records) | set(verifier_records))
    rows = []
    for sample_idx in sample_indices:
        proposal = proposal_records.get(sample_idx, {})
        verifier = verifier_records.get(sample_idx, {})
        template = template_records.get(sample_idx)
        proposal_pass = bool(proposal.get("has_candidate", False))
        verifier_pass = bool(verifier.get("has_candidate", False))
        if not pass_policy(proposal_pass, verifier_pass, args.keep_policy):
            continue
        chosen = verifier if verifier_pass else proposal
        fallback = proposal if verifier_pass else verifier
        template_delta = (
            safe_float(template.get("delta_chi2_vs_plane_null")) if template is not None else None
        )
        row = {
            "sample_index": sample_idx,
            "rank_score": ranking_score(proposal, verifier, template),
            "proposal_pass": proposal_pass,
            "verifier_pass": verifier_pass,
            "risk_tag": risk_tag(proposal_pass, verifier_pass, template_delta),
            "truth_label": int(chosen.get("truth_label", fallback.get("truth_label", 0))),
            "patch_center_glon_deg": choose_sky_value(chosen, fallback, "patch_center_glon_deg"),
            "patch_center_glat_deg": choose_sky_value(chosen, fallback, "patch_center_glat_deg"),
            "candidate_glon_deg": choose_sky_value(chosen, fallback, "candidate_glon_deg"),
            "candidate_glat_deg": choose_sky_value(chosen, fallback, "candidate_glat_deg"),
            "candidate_dx_deg": choose_sky_value(chosen, fallback, "candidate_dx_deg"),
            "candidate_dy_deg": choose_sky_value(chosen, fallback, "candidate_dy_deg"),
            "radius_est_deg": choose_sky_value(chosen, fallback, "radius_est_deg"),
            "proposal_score_max": safe_float(proposal.get("score_max")),
            "proposal_score_mean": safe_float(proposal.get("score_mean")),
            "proposal_positive_fraction": safe_float(proposal.get("positive_fraction")),
            "proposal_threshold": safe_float(proposal.get("threshold"), proposal_summary["selected_threshold"]),
            "verifier_score_max": safe_float(verifier.get("score_max")),
            "verifier_score_mean": safe_float(verifier.get("score_mean")),
            "verifier_positive_fraction": safe_float(verifier.get("positive_fraction")),
            "verifier_threshold": safe_float(verifier.get("threshold"), verifier_summary["selected_threshold"]),
            "template_fit_status": template.get("fit_status") if template else None,
            "template_delta_chi2_vs_plane_null": template_delta,
            "template_theta_crit_fit_deg": safe_float(template.get("theta_crit_fit_deg")) if template else None,
            "template_z0_fit": safe_float(template.get("z0_fit")) if template else None,
            "template_zcrit_fit": safe_float(template.get("zcrit_fit")) if template else None,
            "truth_theta_crit_deg": safe_float(chosen.get("truth_theta_crit_deg", fallback.get("truth_theta_crit_deg"))),
            "truth_z0": safe_float(chosen.get("truth_z0", fallback.get("truth_z0"))),
            "truth_zcrit": safe_float(chosen.get("truth_zcrit", fallback.get("truth_zcrit"))),
            "truth_edge_sigma_deg": safe_float(chosen.get("truth_edge_sigma_deg", fallback.get("truth_edge_sigma_deg"))),
            "coord_pool_idx": int(chosen.get("coord_pool_idx", fallback.get("coord_pool_idx", -1))),
            "cmb_realization_idx": int(chosen.get("cmb_realization_idx", fallback.get("cmb_realization_idx", -1))),
            "background_id": int(chosen.get("background_id", fallback.get("background_id", 0))),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["rank_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    truth_labels = [
        int(proposal_records[idx].get("truth_label", verifier_records.get(idx, {}).get("truth_label", 0)))
        for idx in sample_indices
    ]
    totals = {
        "num_samples": int(len(sample_indices)),
        "num_truth_positive": int(sum(1 for value in truth_labels if value == 1)),
        "num_truth_negative": int(sum(1 for value in truth_labels if value == 0)),
    }
    return rows, proposal_summary, verifier_summary, totals


def summarize(rows, args, proposal_summary, verifier_summary, totals):
    truth_pos = sum(1 for row in rows if row["truth_label"] == 1)
    truth_neg = sum(1 for row in rows if row["truth_label"] == 0)
    tag_counts = {}
    for row in rows:
        tag_counts[row["risk_tag"]] = tag_counts.get(row["risk_tag"], 0) + 1
    return {
        "keep_policy": args.keep_policy,
        "proposal_eval_dir": str(Path(args.proposal_eval_dir).resolve()),
        "verifier_eval_dir": str(Path(args.verifier_eval_dir).resolve()),
        "template_records": str(Path(args.template_records).resolve()) if args.template_records else "",
        "proposal_selected_threshold": float(proposal_summary["selected_threshold"]),
        "verifier_selected_threshold": float(verifier_summary["selected_threshold"]),
        **totals,
        "num_emitted_candidates": int(len(rows)),
        "num_truth_positive_emitted": int(truth_pos),
        "num_truth_negative_emitted": int(truth_neg),
        "emitted_precision": float(truth_pos / max(len(rows), 1)),
        "emitted_recall": float(truth_pos / max(totals["num_truth_positive"], 1)),
        "emitted_false_positive_rate": float(truth_neg / max(totals["num_truth_negative"], 1)),
        "risk_tag_counts": tag_counts,
    }


def write_outputs(rows, summary, args):
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)
    output_summary = Path(args.output_summary)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    fieldnames = list(rows[0].keys()) if rows else ["rank", "sample_index", "rank_score"]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(output_summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main():
    args = parse_args()
    rows, proposal_summary, verifier_summary, totals = build_rows(args)
    summary = summarize(rows, args, proposal_summary, verifier_summary, totals)
    write_outputs(rows, summary, args)
    print(json.dumps(summary, indent=2))
    print(f"Ranked JSONL: {args.output_jsonl}")
    print(f"Ranked CSV:   {args.output_csv}")
    print(f"Summary:      {args.output_summary}")


if __name__ == "__main__":
    main()
