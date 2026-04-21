"""Select deployment-safe composite policies from remediated audit artifacts.

Assumptions
-----------
* Composite policy recall estimates come from the remediated sensitivity and
  real-background injection grids. They are screening diagnostics, not
  cosmological detection probabilities.
* Held-out null-control FPR is a patch-level false-positive diagnostic.
* Full-sky tile trigger and cluster counts are deployment-burden diagnostics on
  overlapping gnomonic patches. They are not independent-binomial p-values.
* A deployment policy must satisfy both statistical and operational burden
  constraints before recall gain is useful.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_JSON = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_policy_pareto"
    / "policy_pareto.json"
)
DEFAULT_NULL_AUDIT_JSON = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_null_policy_audit"
    / "null_policy_audit.json"
)
DEFAULT_TILE_AUDIT_JSON = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_policy_tile_audit"
    / "policy_tile_audit.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_deployment_policy_decision"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Turn policy-Pareto, null, and tile audits into a deployment decision.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy-json", type=str, default=str(DEFAULT_POLICY_JSON))
    parser.add_argument("--null-audit-json", type=str, default=str(DEFAULT_NULL_AUDIT_JSON))
    parser.add_argument("--tile-audit-json", type=str, default=str(DEFAULT_TILE_AUDIT_JSON))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--cluster-radius-deg",
        type=float,
        default=15.0,
        help="Cluster radius key to use from the full-sky tile audit.",
    )
    parser.add_argument(
        "--max-clusters-any-map",
        type=int,
        default=70,
        help="Maximum allowed clustered candidates on every cleaned map.",
    )
    parser.add_argument(
        "--max-trigger-fraction-any-map",
        type=float,
        default=0.15,
        help="Maximum allowed overlapping-tile trigger fraction on every cleaned map.",
    )
    parser.add_argument(
        "--max-pooled-null-fpr-ci-high",
        type=float,
        default=0.04,
        help="Maximum allowed upper 95% CI for pooled held-out null-control FPR.",
    )
    parser.add_argument(
        "--min-recall-gain-vs-single",
        type=float,
        default=0.0,
        help="Minimum diagnostic recall gain over exact-threshold single-score baseline.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate numerical arguments and input files."""

    if args.cluster_radius_deg <= 0.0:
        raise ValueError("--cluster-radius-deg must be positive.")
    if args.max_clusters_any_map < 0:
        raise ValueError("--max-clusters-any-map must be non-negative.")
    if not (0.0 <= args.max_trigger_fraction_any_map <= 1.0):
        raise ValueError("--max-trigger-fraction-any-map must lie in [0, 1].")
    if not (0.0 <= args.max_pooled_null_fpr_ci_high <= 1.0):
        raise ValueError("--max-pooled-null-fpr-ci-high must lie in [0, 1].")
    for path_text in (args.policy_json, args.null_audit_json, args.tile_audit_json):
        if not Path(path_text).expanduser().exists():
            raise FileNotFoundError(f"Missing input artifact: {path_text}")


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def policy_key(row: dict[str, Any]) -> tuple[float, float]:
    """Return the FPR-budget key shared by policy, null, and tile reports."""

    return (
        float(row["constraint_camb_fpr_max"]),
        float(row["constraint_real_fpr_max"]),
    )


def cluster_key(radius_deg: float) -> str:
    """Return the key used by policy tile audit JSON for a radius."""

    return f"{float(radius_deg):.1f}"


def build_decision_rows(
    policy_report: dict[str, Any],
    null_report: dict[str, Any],
    tile_report: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Merge audit reports into one decision table."""

    policies = {
        policy_key(row): row
        for row in policy_report.get("top_rows", [])
        if int(row.get("rank", -1)) == 1
    }
    nulls = {policy_key(row): row for row in null_report.get("pooled_rows", [])}
    tiles_by_policy: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in tile_report.get("rows", []):
        tiles_by_policy.setdefault(policy_key(row), []).append(row)

    radius_key = cluster_key(float(args.cluster_radius_deg))
    rows = []
    for key, policy in sorted(policies.items()):
        tile_rows = tiles_by_policy.get(key, [])
        if not tile_rows:
            continue
        null = nulls.get(key, {})
        cluster_counts = [
            int(row.get("cluster_summary", {}).get(radius_key, {}).get("n_clusters", -1))
            for row in tile_rows
        ]
        if any(value < 0 for value in cluster_counts):
            raise KeyError(f"Tile audit missing cluster radius {radius_key} for policy key {key}.")
        trigger_fracs = [float(row["trigger_fraction"]) for row in tile_rows]
        maps = [str(row["map"]) for row in tile_rows]
        max_cluster_idx = int(np.argmax(cluster_counts))
        max_trigger_idx = int(np.argmax(trigger_fracs))
        checks = {
            "pooled_null_fpr_ci_high": float(null.get("fpr_ci_high", np.inf))
            <= float(args.max_pooled_null_fpr_ci_high),
            "clusters_any_map": max(cluster_counts) <= int(args.max_clusters_any_map),
            "trigger_fraction_any_map": max(trigger_fracs)
            <= float(args.max_trigger_fraction_any_map),
            "recall_gain_vs_single": float(policy.get("real_recall_gain_vs_best_single", 0.0))
            >= float(args.min_recall_gain_vs_single),
        }
        promotable = all(checks.values())
        failure_reasons = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "policy_key": f"camb<={key[0]:.2f},real<={key[1]:.2f}",
                "policy": policy["policy"],
                "family": policy["family"],
                "diagnostic_real_recall": float(policy["real_recall"]),
                "diagnostic_real_fpr_200": float(policy["real_fpr"]),
                "diagnostic_camb_fpr": float(policy["camb_fpr"]),
                "gain_vs_exact_single": float(
                    policy.get("real_recall_gain_vs_best_single", np.nan)
                ),
                "pooled_null_fpr": float(null.get("false_positive_rate", np.nan)),
                "pooled_null_fpr_ci_high": float(null.get("fpr_ci_high", np.nan)),
                "max_clusters": int(max(cluster_counts)),
                "max_clusters_map": maps[max_cluster_idx],
                "mean_clusters": float(np.mean(cluster_counts)),
                "max_trigger_fraction": float(max(trigger_fracs)),
                "max_trigger_fraction_map": maps[max_trigger_idx],
                "promotable_under_defaults": bool(promotable),
                "failure_reasons": failure_reasons,
            }
        )
    rows.sort(
        key=lambda row: (
            not bool(row["promotable_under_defaults"]),
            -float(row["diagnostic_real_recall"]),
            int(row["max_clusters"]),
            float(row["max_trigger_fraction"]),
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact CSV output."""

    columns = [
        "policy_key",
        "family",
        "diagnostic_real_recall",
        "gain_vs_exact_single",
        "pooled_null_fpr",
        "pooled_null_fpr_ci_high",
        "max_clusters",
        "max_clusters_map",
        "mean_clusters",
        "max_trigger_fraction",
        "max_trigger_fraction_map",
        "promotable_under_defaults",
        "failure_reasons",
        "policy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in columns}
            out["failure_reasons"] = ";".join(row.get("failure_reasons", []))
            writer.writerow(out)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write Markdown report."""

    rows = report["rows"]
    promoted = [row for row in rows if row["promotable_under_defaults"]]
    best = rows[0] if rows else None
    lines = ["# Remediated v1 Deployment Policy Decision", ""]
    lines.append("This report converts diagnostic recall and false-positive audits into an operational deployment decision.")
    lines.append("")
    lines.append("## Constraints")
    lines.append("")
    for key, value in report["constraints"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    if promoted:
        lines.append(
            f"Promotable policies under these constraints: `{len(promoted)}`. "
            "Use the highest-recall promotable row unless a stricter paper-facing burden rule is chosen."
        )
    else:
        lines.append(
            "No composite policy is promotable under the default cross-map burden constraints. "
            "Use these rows as recall-boost candidates only, or loosen the operational burden rule explicitly."
        )
    if best:
        lines.append("")
        lines.append(
            "Best row by recall-first ordering: "
            f"`{best['policy_key']}` with recall `{best['diagnostic_real_recall']:.4f}`, "
            f"max clusters `{best['max_clusters']}` on `{best['max_clusters_map']}`, "
            f"and max trigger fraction `{best['max_trigger_fraction']:.4f}` on "
            f"`{best['max_trigger_fraction_map']}`."
        )
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append(
        "| policy budget | recall | gain | pooled null FPR high | max clusters | max trigger frac | promotable | blockers |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for row in rows:
        blockers = ", ".join(row["failure_reasons"]) if row["failure_reasons"] else "none"
        lines.append(
            f"| `{row['policy_key']}` | {row['diagnostic_real_recall']:.4f} | "
            f"{row['gain_vs_exact_single']:.4f} | {row['pooled_null_fpr_ci_high']:.4f} | "
            f"{row['max_clusters']} ({row['max_clusters_map']}) | "
            f"{row['max_trigger_fraction']:.4f} ({row['max_trigger_fraction_map']}) | "
            f"{row['promotable_under_defaults']} | {blockers} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_report = load_json(Path(args.policy_json).expanduser().resolve())
    null_report = load_json(Path(args.null_audit_json).expanduser().resolve())
    tile_report = load_json(Path(args.tile_audit_json).expanduser().resolve())
    rows = build_decision_rows(policy_report, null_report, tile_report, args)
    report = {
        "inputs": {
            "policy_json": str(Path(args.policy_json).expanduser().resolve()),
            "null_audit_json": str(Path(args.null_audit_json).expanduser().resolve()),
            "tile_audit_json": str(Path(args.tile_audit_json).expanduser().resolve()),
        },
        "constraints": {
            "cluster_radius_deg": float(args.cluster_radius_deg),
            "max_clusters_any_map": int(args.max_clusters_any_map),
            "max_trigger_fraction_any_map": float(args.max_trigger_fraction_any_map),
            "max_pooled_null_fpr_ci_high": float(args.max_pooled_null_fpr_ci_high),
            "min_recall_gain_vs_single": float(args.min_recall_gain_vs_single),
        },
        "num_promotable": int(sum(bool(row["promotable_under_defaults"]) for row in rows)),
        "rows": rows,
    }
    json_path = output_dir / "deployment_policy_decision.json"
    csv_path = output_dir / "deployment_policy_decision.csv"
    md_path = output_dir / "deployment_policy_decision.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
