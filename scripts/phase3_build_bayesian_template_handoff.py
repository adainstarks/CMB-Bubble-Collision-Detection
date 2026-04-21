"""Build a conservative Bayesian/template-fit handoff packet.

Assumptions
-----------
* Input candidates are the frozen, calibration-scored screening outputs from
  the remediated full-sky policy. They are not detections.
* Template-fit records provide deterministic Feeney-style local fit seeds under
  a plane-null nuisance model. They are not posterior samples or evidence
  ratios.
* This script packages provenance, screening calibration, local fit seeds, and
  explicit caution flags so a downstream Bayesian or template-likelihood stage
  can start from a reproducible candidate list without conflating screening
  scores with cosmological inference.
* The projection/clustering audit is used only to attach geometry/systematics
  guardrails. It is not used to rescore candidates.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_JSONL = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_candidate_score_calibration"
    / "calibrated_candidates.jsonl"
)
DEFAULT_TEMPLATE_JSONL = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_template_fit_handoff"
    / "template_fit_records.jsonl"
)
DEFAULT_PROJECTION_AUDIT_JSON = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_projection_clustering_audit"
    / "projection_clustering_audit.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_bayesian_template_handoff"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Merge calibrated candidates and template fits into a Bayesian handoff packet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-jsonl", type=str, default=str(DEFAULT_CANDIDATE_JSONL))
    parser.add_argument("--template-jsonl", type=str, default=str(DEFAULT_TEMPLATE_JSONL))
    parser.add_argument("--projection-audit-json", type=str, default=str(DEFAULT_PROJECTION_AUDIT_JSON))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--screening-envelope-amplitude-min",
        type=float,
        default=1.0e-6,
        help="Current synthetic screening envelope lower amplitude bound |z|.",
    )
    parser.add_argument(
        "--screening-envelope-amplitude-max",
        type=float,
        default=1.0e-4,
        help="Current synthetic screening envelope upper amplitude bound |z|.",
    )
    parser.add_argument(
        "--screening-envelope-theta-min-deg",
        type=float,
        default=5.0,
        help="Current synthetic screening envelope lower theta bound.",
    )
    parser.add_argument(
        "--screening-envelope-theta-max-deg",
        type=float,
        default=25.0,
        help="Current synthetic screening envelope upper theta bound.",
    )
    parser.add_argument(
        "--tier1-bh-q",
        type=float,
        default=0.01,
        help="Descriptive tier-1 screening threshold on pooled BH q-values.",
    )
    parser.add_argument(
        "--tier2-bh-q",
        type=float,
        default=0.05,
        help="Descriptive tier-2 screening threshold on pooled BH q-values.",
    )
    parser.add_argument(
        "--projection-offset-caution-deg",
        type=float,
        default=10.0,
        help="Candidate offset above which projection/systematics caution is attached.",
    )
    parser.add_argument(
        "--projection-theta-caution-deg",
        type=float,
        default=20.0,
        help="Template radius above which projection/systematics caution is attached.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate non-physical arguments and required inputs."""

    for label, value in (
        ("--screening-envelope-amplitude-min", args.screening_envelope_amplitude_min),
        ("--screening-envelope-amplitude-max", args.screening_envelope_amplitude_max),
        ("--screening-envelope-theta-min-deg", args.screening_envelope_theta_min_deg),
        ("--screening-envelope-theta-max-deg", args.screening_envelope_theta_max_deg),
        ("--tier1-bh-q", args.tier1_bh_q),
        ("--tier2-bh-q", args.tier2_bh_q),
        ("--projection-offset-caution-deg", args.projection_offset_caution_deg),
        ("--projection-theta-caution-deg", args.projection_theta_caution_deg),
    ):
        if not np.isfinite(float(value)):
            raise ValueError(f"{label} must be finite.")
    if float(args.screening_envelope_amplitude_min) <= 0.0:
        raise ValueError("--screening-envelope-amplitude-min must be positive.")
    if float(args.screening_envelope_amplitude_max) <= float(args.screening_envelope_amplitude_min):
        raise ValueError("--screening-envelope-amplitude-max must exceed the minimum.")
    if float(args.screening_envelope_theta_min_deg) <= 0.0:
        raise ValueError("--screening-envelope-theta-min-deg must be positive.")
    if float(args.screening_envelope_theta_max_deg) <= float(args.screening_envelope_theta_min_deg):
        raise ValueError("--screening-envelope-theta-max-deg must exceed the minimum.")
    if not (0.0 <= float(args.tier1_bh_q) <= float(args.tier2_bh_q) <= 1.0):
        raise ValueError("Tier q-value thresholds must satisfy 0 <= tier1 <= tier2 <= 1.")
    if float(args.projection_offset_caution_deg) < 0.0:
        raise ValueError("--projection-offset-caution-deg must be non-negative.")
    if float(args.projection_theta_caution_deg) <= 0.0:
        raise ValueError("--projection-theta-caution-deg must be positive.")
    for label, value in (
        ("candidate JSONL", args.candidate_jsonl),
        ("template JSONL", args.template_jsonl),
    ):
        path = Path(str(value)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def candidate_key(map_name: str, patch_index: int) -> tuple[str, int]:
    """Return a stable join key across screening and template-fit artifacts."""

    return str(map_name).lower(), int(patch_index)


def index_template_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    """Index template rows by map and patch/sample index."""

    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = candidate_key(str(row["map"]), int(row["sample_index"]))
        if key in indexed:
            raise ValueError(f"Duplicate template-fit key detected: {key}")
        indexed[key] = row
    return indexed


def screening_tier(q_value: float, args: argparse.Namespace) -> str:
    """Return a descriptive screening tier from pooled BH q-values."""

    if q_value <= float(args.tier1_bh_q):
        return "tier1"
    if q_value <= float(args.tier2_bh_q):
        return "tier2"
    return "tier3"


def finite_float(value: Any) -> float | None:
    """Convert a value to finite float or return None."""

    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def validate_template_seed(template_row: dict[str, Any]) -> list[str]:
    """Return validation errors for the deterministic template-fit seed."""

    errors: list[str] = []
    if str(template_row.get("fit_status", "")) != "fit":
        errors.append(f"fit_status={template_row.get('fit_status')}")
    theta_crit = finite_float(template_row.get("theta_crit_fit_deg"))
    if theta_crit is None or theta_crit <= 0.0:
        errors.append("non_physical_theta_crit_fit")
    for key in ("z0_fit", "zcrit_fit", "delta_chi2_vs_plane_null", "support_radius_deg"):
        if finite_float(template_row.get(key)) is None:
            errors.append(f"non_finite_{key}")
    support_radius = finite_float(template_row.get("support_radius_deg"))
    if support_radius is not None and support_radius <= 0.0:
        errors.append("non_physical_support_radius")
    return errors


def build_row(
    candidate_row: dict[str, Any],
    template_row: dict[str, Any] | None,
    audit_summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Merge one calibrated candidate with its deterministic template-fit seed."""

    pooled_p = finite_float(candidate_row.get("calibration_pooled_survival_p"))
    pooled_q = finite_float(candidate_row.get("calibration_pooled_bh_q"))
    if pooled_p is None or pooled_q is None:
        raise ValueError(
            f"Candidate {candidate_row.get('map')}:{candidate_row.get('patch_index')} "
            "is missing finite pooled calibration scores."
        )

    row: dict[str, Any] = {
        "map": str(candidate_row["map"]).lower(),
        "patch_index": int(candidate_row["patch_index"]),
        "global_cluster_rank": int(candidate_row.get("global_cluster_rank", 0)),
        "cluster_id": int(candidate_row.get("cluster_id", -1)),
        "cluster_radius_deg": finite_float(candidate_row.get("cluster_radius_deg")),
        "cluster_n_members": int(candidate_row.get("cluster_n_members", 0)),
        "candidate_glon_deg": finite_float(candidate_row.get("candidate_glon_deg")),
        "candidate_glat_deg": finite_float(candidate_row.get("candidate_glat_deg")),
        "patch_center_glon_deg": finite_float(candidate_row.get("patch_center_glon_deg")),
        "patch_center_glat_deg": finite_float(candidate_row.get("patch_center_glat_deg")),
        "policy_slug": str(candidate_row.get("policy_slug", "")),
        "policy_margin": finite_float(candidate_row.get("policy_margin")),
        "screening_priority_tier": screening_tier(pooled_q, args),
        "screening_pooled_survival_p": pooled_p,
        "screening_pooled_bh_q": pooled_q,
        "screening_map_survival_p": finite_float(candidate_row.get("calibration_map_survival_p")),
        "screening_null_count": int(candidate_row.get("calibration_pooled_null_count", 0)),
        "screening_notes": [
            "These calibration scores rank screening outputs against the real-map null-control calibration split.",
            "They are not posterior probabilities, discovery p-values, or Bayesian evidence ratios.",
        ],
        "screening_envelope": {
            "amplitude_abs_min": float(args.screening_envelope_amplitude_min),
            "amplitude_abs_max": float(args.screening_envelope_amplitude_max),
            "theta_crit_deg_min": float(args.screening_envelope_theta_min_deg),
            "theta_crit_deg_max": float(args.screening_envelope_theta_max_deg),
            "note": "This is the current synthetic screening envelope, not a cosmological prior.",
        },
        "projection_audit_context": audit_summary,
    }

    if template_row is None:
        row["template_seed_status"] = "missing_template_fit"
        row["template_seed_errors"] = ["missing_template_fit_row"]
        row["followup_route"] = "screening_only_until_template_seed_exists"
        return row

    template_errors = validate_template_seed(template_row)
    offset_dx = finite_float(template_row.get("candidate_dx_deg"))
    offset_dy = finite_float(template_row.get("candidate_dy_deg"))
    offset_deg = None
    if offset_dx is not None and offset_dy is not None:
        offset_deg = float(np.hypot(offset_dx, offset_dy))

    theta_fit = finite_float(template_row.get("theta_crit_fit_deg"))
    projection_caution = bool(
        (theta_fit is not None and theta_fit >= float(args.projection_theta_caution_deg))
        or (offset_deg is not None and offset_deg >= float(args.projection_offset_caution_deg))
    )

    row.update(
        {
            "template_seed_status": "fit_ok" if not template_errors else "fit_needs_review",
            "template_seed_errors": template_errors,
            "template_seed": {
                "seed_glon_deg": finite_float(template_row.get("candidate_glon_deg")) or row["candidate_glon_deg"],
                "seed_glat_deg": finite_float(template_row.get("candidate_glat_deg")) or row["candidate_glat_deg"],
                "theta_crit_deg": theta_fit,
                "z0": finite_float(template_row.get("z0_fit")),
                "zcrit": finite_float(template_row.get("zcrit_fit")),
                "delta_chi2_vs_plane_null": finite_float(template_row.get("delta_chi2_vs_plane_null")),
                "support_radius_deg": finite_float(template_row.get("support_radius_deg")),
                "candidate_offset_deg": offset_deg,
                "radius_seed_available": bool(template_row.get("candidate_radius_seed_available", False)),
            },
            "template_refinement_window": {
                "theta_crit_deg_min": (
                    None
                    if theta_fit is None
                    else float(max(float(args.screening_envelope_theta_min_deg), theta_fit - 3.0))
                ),
                "theta_crit_deg_max": (
                    None
                    if theta_fit is None
                    else float(min(float(args.screening_envelope_theta_max_deg), theta_fit + 3.0))
                ),
                "roi_radius_deg": finite_float(template_row.get("support_radius_deg")),
                "note": (
                    "Use the seed for local template-likelihood initialization; keep amplitude priors broad and physics-driven."
                ),
            },
            "projection_systematics_caution": projection_caution,
            "projection_systematics_note": (
                "Rerun follow-up on a native-sphere or projection-robust extraction if the fitted scale is large or the candidate lies far from the tile center."
                if projection_caution
                else "Current projection audit does not force an immediate geometry escalation for this seed."
            ),
        }
    )
    row["followup_route"] = (
        "bayesian_or_template_likelihood_followup_with_projection_caution"
        if projection_caution
        else "bayesian_or_template_likelihood_followup"
    )
    return row


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Build a compact summary JSON."""

    tier_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    map_counts: dict[str, int] = {}
    caution_count = 0
    template_ok = 0
    top_rows = []
    for row in rows:
        tier = str(row["screening_priority_tier"])
        route = str(row["followup_route"])
        map_name = str(row["map"])
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        route_counts[route] = route_counts.get(route, 0) + 1
        map_counts[map_name] = map_counts.get(map_name, 0) + 1
        caution_count += int(bool(row.get("projection_systematics_caution", False)))
        template_ok += int(str(row.get("template_seed_status")) == "fit_ok")
    for row in rows[:10]:
        seed = row.get("template_seed", {})
        top_rows.append(
            {
                "global_cluster_rank": row.get("global_cluster_rank"),
                "map": row.get("map"),
                "patch_index": row.get("patch_index"),
                "screening_priority_tier": row.get("screening_priority_tier"),
                "screening_pooled_survival_p": row.get("screening_pooled_survival_p"),
                "screening_pooled_bh_q": row.get("screening_pooled_bh_q"),
                "theta_crit_deg": seed.get("theta_crit_deg"),
                "z0": seed.get("z0"),
                "zcrit": seed.get("zcrit"),
                "delta_chi2_vs_plane_null": seed.get("delta_chi2_vs_plane_null"),
                "followup_route": row.get("followup_route"),
            }
        )
    return {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "num_candidates": int(len(rows)),
        "num_template_seed_ok": int(template_ok),
        "num_projection_cautions": int(caution_count),
        "tier_counts": tier_counts,
        "route_counts": route_counts,
        "map_counts": map_counts,
        "screening_envelope": {
            "amplitude_abs_min": float(args.screening_envelope_amplitude_min),
            "amplitude_abs_max": float(args.screening_envelope_amplitude_max),
            "theta_crit_deg_min": float(args.screening_envelope_theta_min_deg),
            "theta_crit_deg_max": float(args.screening_envelope_theta_max_deg),
        },
        "top_rows": top_rows,
        "assumption_notes": [
            "This is a downstream handoff packet for classical/Bayesian follow-up.",
            "It preserves screening calibration and local template-fit seeds without claiming posterior inference.",
            "Use broad science priors in the follow-up sampler; do not recycle screening thresholds as model priors.",
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write a human-readable handoff summary."""

    lines = ["# Bayesian / Template-Fit Handoff", ""]
    lines.append("This artifact packages screened candidates for downstream template-likelihood or Bayesian follow-up.")
    lines.append("It is not a Bayesian evidence calculation.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- `num_candidates`: `{summary['num_candidates']}`")
    lines.append(f"- `num_template_seed_ok`: `{summary['num_template_seed_ok']}`")
    lines.append(f"- `num_projection_cautions`: `{summary['num_projection_cautions']}`")
    lines.append(f"- `tier_counts`: `{summary['tier_counts']}`")
    lines.append(f"- `route_counts`: `{summary['route_counts']}`")
    lines.append("")
    lines.append("## Top Follow-Up Rows")
    lines.append("")
    lines.append("| rank | map | patch | tier | pooled p | q | theta_fit_deg | delta_chi2 | route |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|---|")
    for row in rows[:10]:
        seed = row.get("template_seed", {})
        lines.append(
            f"| {int(row.get('global_cluster_rank', 0))} | {row.get('map')} | {int(row.get('patch_index', 0))} | "
            f"{row.get('screening_priority_tier')} | {float(row.get('screening_pooled_survival_p', 1.0)):.6f} | "
            f"{float(row.get('screening_pooled_bh_q', 1.0)):.6f} | "
            f"{float(seed.get('theta_crit_deg') or 0.0):.2f} | "
            f"{float(seed.get('delta_chi2_vs_plane_null') or 0.0):.3e} | "
            f"{row.get('followup_route')} |"
        )
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- Screening survival p-values and BH q-values are ranking metadata, not posterior probabilities.")
    lines.append("- Template seeds are deterministic local fits; use them only to initialize the downstream likelihood or sampler.")
    lines.append("- If `projection_systematics_caution` is true, rerun the follow-up on a native-sphere or projection-robust extraction before making any parameter statement.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def projection_audit_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Compress the projection audit into a lightweight context block."""

    if not report:
        return {
            "available": False,
            "note": "Projection/clustering audit JSON not supplied.",
        }
    projection_rows = report.get("projection_rows", [])
    clustering_rows = report.get("clustering_rows", [])
    out: dict[str, Any] = {
        "available": True,
        "projection_row_count": int(len(projection_rows)),
        "clustering_row_count": int(len(clustering_rows)),
        "assumption_warnings": list(report.get("assumption_warnings", [])),
    }
    if projection_rows:
        cosine = np.asarray(
            [float(row["cosine_similarity"]) for row in projection_rows],
            dtype=np.float64,
        )
        iou = np.asarray(
            [float(row["support_iou"]) for row in projection_rows],
            dtype=np.float64,
        )
        peak_error = np.asarray(
            [float(row["peak_abs_frac_error"]) for row in projection_rows],
            dtype=np.float64,
        )
        out.update(
            {
                "projection_cosine_min": float(np.min(cosine)),
                "projection_cosine_median": float(np.median(cosine)),
                "projection_support_iou_min": float(np.min(iou)),
                "projection_support_iou_median": float(np.median(iou)),
                "projection_peak_abs_frac_error_median": float(np.median(peak_error)),
            }
        )
    return out


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    validate_args(args)

    candidate_path = Path(args.candidate_jsonl).expanduser().resolve()
    template_path = Path(args.template_jsonl).expanduser().resolve()
    projection_audit_path = Path(args.projection_audit_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = load_jsonl(candidate_path)
    template_rows = load_jsonl(template_path)
    template_index = index_template_rows(template_rows)
    audit_report = load_json(projection_audit_path) if projection_audit_path.exists() else {}
    audit_summary = projection_audit_summary(audit_report)

    rows = []
    for candidate_row in candidate_rows:
        key = candidate_key(str(candidate_row["map"]), int(candidate_row["patch_index"]))
        template_row = template_index.get(key)
        rows.append(build_row(candidate_row, template_row, audit_summary, args))

    rows.sort(
        key=lambda row: (
            float(row["screening_pooled_bh_q"]),
            float(row["screening_pooled_survival_p"]),
            int(row["global_cluster_rank"]),
        )
    )
    summary = summarize_rows(rows, args)

    jsonl_path = output_dir / "bayesian_template_handoff.jsonl"
    summary_path = output_dir / "bayesian_template_handoff_summary.json"
    markdown_path = output_dir / "bayesian_template_handoff.md"
    write_jsonl(jsonl_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(markdown_path, summary, rows)

    print(
        json.dumps(
            {
                "candidate_jsonl": str(candidate_path),
                "template_jsonl": str(template_path),
                "projection_audit_json": str(projection_audit_path) if projection_audit_path.exists() else "",
                "output_jsonl": str(jsonl_path),
                "output_summary": str(summary_path),
                "output_markdown": str(markdown_path),
                "num_rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
