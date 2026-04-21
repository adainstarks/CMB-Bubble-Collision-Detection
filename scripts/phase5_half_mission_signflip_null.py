"""Half-mission sign-flip null calibration for emitted candidates.

Assumptions
-----------
* The input HM1/HM2 maps are matched Planck cleaned CMB half-mission
  anisotropy maps in Kelvin, with the same beam and component-separation
  convention as the deployment scorer.
* The current pipeline is a candidate screener. This script estimates a
  per-candidate conditional noise-robustness p-value, not a Bayesian evidence
  ratio and not a global LambdaCDM detection probability.
* For HM1 = S + n1 and HM2 = S + n2 with independent equal-variance noise,
  ``0.5 * (HM1 - HM2)`` has the same noise variance as the half-mission mean
  ``0.5 * (HM1 + HM2)``. Pixel-wise sign flips randomize this empirical noise
  proxy while preserving the fixed sky realization in the candidate patch.
* The score functions are the frozen remediated-v1 U-Net max-probability
  screeners and the Feeney-style circular-template screen used by
  ``phase3_policy_pareto_search.py``. Feeney et al. PRD 84, 043507 (2011)
  motivates the collision template family; Planck Collaboration IV (2020)
  motivates the component-separated map inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
import torch
from astropy.io import fits
from astropy import units as u
from scipy.signal import fftconvolve

import phase3_train_unet as p3
from phase_config import DEFAULTS, FLOAT_GEOMETRY_DTYPE, FLOAT_STORAGE_DTYPE
from phase2_generate_training import load_mask, project_patch, projected_unmasked_fraction
from phase2_observing_model import remove_real_map_low_modes
from phase2_signal_model import PATCH_PIX
from phase3_remediated_policy_tile_audit import (
    CIRCULAR_METHOD,
    DEFAULT_MODELS,
    DEFAULT_THETA_GRID_DEG,
    ML_METHODS,
    ModelSpec,
    apply_policy,
    circular_kernels,
    parse_model_spec,
    policy_margin,
    slugify,
    standardize_patch_batch,
)
from phase3_evaluate_run import resolve_checkpoint_path
from phase3_sensitivity_curve import build_model_from_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_JSON = (
    PROJECT_ROOT
    / "runs"
    / "phase3_unet"
    / "remediated_v1_tile_constrained_policy_search"
    / "tile_constrained_policy_search.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "phase5_half_mission_signflip_null"
)
DEFAULT_COMMON_MASK = PROJECT_ROOT / "data" / "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"


@dataclass(frozen=True)
class LoadedModel:
    """In-memory scorer state for a one-channel remediated U-Net."""

    spec: ModelSpec
    model: torch.nn.Module
    channel_mean: float
    channel_std: float
    checkpoint_path: str
    checkpoint_label: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run half-mission sign-flip null calibration for emitted "
            "remediated-v1 candidates."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hm1-map", type=str, default="", help="HM1 map path (.fits, .npy, or .npz).")
    parser.add_argument("--hm2-map", type=str, default="", help="HM2 map path (.fits, .npy, or .npz).")
    parser.add_argument(
        "--hm1-map-template",
        type=str,
        default="",
        help=(
            "Optional per-map HM1 template, e.g. "
            "'data/planck_pr3_hm/COM_CMB_IQU-{map}_2048_R3.00_hm1.fits'."
        ),
    )
    parser.add_argument(
        "--hm2-map-template",
        type=str,
        default="",
        help=(
            "Optional per-map HM2 template, e.g. "
            "'data/planck_pr3_hm/COM_CMB_IQU-{map}_2048_R3.00_hm2.fits'."
        ),
    )
    parser.add_argument(
        "--candidate-jsonl",
        action="append",
        required=True,
        help="Candidate JSONL file. Repeat for multiple policy/map candidate lists.",
    )
    parser.add_argument("--policy-json", type=str, default=str(DEFAULT_POLICY_JSON))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", action="append", default=[], help="name:run_dir:checkpoint")
    parser.add_argument("--num-realizations", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--circular-batch-size", type=int, default=64)
    parser.add_argument("--circular-kernel-chunk", type=int, default=4)
    parser.add_argument("--circular-engine", type=str, default="auto", choices=("auto", "torch", "scipy"))
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--fits-field", type=int, default=0)
    parser.add_argument("--target-nside", type=int, default=DEFAULTS.nside)
    parser.add_argument("--mask-threshold", type=float, default=DEFAULTS.mask_threshold)
    parser.add_argument("--common-mask", type=str, default=str(DEFAULT_COMMON_MASK))
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate candidate, policy, model, mask, and HM-map readiness without "
            "loading models or running sign-flip realizations."
        ),
    )
    parser.add_argument("--skip-low-mode-removal", action="store_true")
    parser.add_argument("--min-valid-fraction", type=float, default=DEFAULTS.mask_threshold)
    parser.add_argument("--on-invalid", type=str, default="raise", choices=("raise", "skip"))
    parser.add_argument("--flip-mode", type=str, default="pixel", choices=("pixel", "block"))
    parser.add_argument("--flip-block-pix", type=int, default=4)
    parser.add_argument(
        "--policy-slug",
        type=str,
        default="",
        help="Override policy slug when candidate records do not contain policy_slug.",
    )
    parser.add_argument(
        "--theta-grid-deg",
        type=str,
        default=",".join(f"{x:g}" for x in DEFAULT_THETA_GRID_DEG),
    )
    parser.add_argument("--beam-fwhm-arcmin", type=float, default=DEFAULTS.beam_fwhm_arcmin)
    return parser.parse_args()


def parse_float_list(text: str) -> tuple[float, ...]:
    """Parse comma-separated floats."""

    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one float.")
    return values


def validate_args(args: argparse.Namespace) -> None:
    """Validate numerical and path arguments."""

    if args.num_realizations <= 0:
        raise ValueError("--num-realizations must be positive.")
    if args.batch_size <= 0 or args.circular_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.circular_kernel_chunk <= 0:
        raise ValueError("--circular-kernel-chunk must be positive.")
    if args.target_nside <= 0 or not hp.isnsideok(int(args.target_nside)):
        raise ValueError("--target-nside must be a valid HEALPix Nside.")
    if not (0.0 < args.mask_threshold <= 1.0):
        raise ValueError("--mask-threshold must lie in (0, 1].")
    if not (0.0 <= args.min_valid_fraction <= 1.0):
        raise ValueError("--min-valid-fraction must lie in [0, 1].")
    if args.flip_block_pix <= 0:
        raise ValueError("--flip-block-pix must be positive.")
    if args.beam_fwhm_arcmin < 0.0:
        raise ValueError("--beam-fwhm-arcmin must be non-negative.")
    args.theta_grid_deg = parse_float_list(args.theta_grid_deg)
    if any(theta <= 0.0 for theta in args.theta_grid_deg):
        raise ValueError("Template radii must be positive.")
    args.models = tuple(parse_model_spec(text) for text in (args.model or DEFAULT_MODELS))
    direct_pair = bool(args.hm1_map or args.hm2_map)
    template_pair = bool(args.hm1_map_template or args.hm2_map_template)
    if direct_pair and template_pair:
        raise ValueError("Use either direct HM map paths or per-map templates, not both.")
    if args.hm1_map and not args.hm2_map:
        raise ValueError("--hm2-map is required when --hm1-map is provided.")
    if args.hm2_map and not args.hm1_map:
        raise ValueError("--hm1-map is required when --hm2-map is provided.")
    if args.hm1_map_template and not args.hm2_map_template:
        raise ValueError("--hm2-map-template is required when --hm1-map-template is provided.")
    if args.hm2_map_template and not args.hm1_map_template:
        raise ValueError("--hm1-map-template is required when --hm2-map-template is provided.")
    if not args.preflight_only and not (bool(args.hm1_map and args.hm2_map) or bool(args.hm1_map_template and args.hm2_map_template)):
        raise ValueError(
            "Provide either --hm1-map/--hm2-map or "
            "--hm1-map-template/--hm2-map-template unless --preflight-only is used."
        )
    path_texts = [*args.candidate_jsonl]
    if args.hm1_map:
        path_texts.append(args.hm1_map)
    if args.hm2_map:
        path_texts.append(args.hm2_map)
    for path_text in path_texts:
        if not Path(path_text).expanduser().exists():
            raise FileNotFoundError(f"Input path not found: {path_text}")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_policy_rows(path: Path) -> list[dict[str, Any]]:
    """Load policy rows from Policy-Pareto or tile-constrained reports."""

    payload = load_json(path)
    raw_rows = list(payload.get("top_rows", []))
    if not raw_rows:
        raise ValueError(f"No policy rows found in {path}.")
    if all("constraint_camb_fpr_max" in row and "rank" in row for row in raw_rows):
        rows = [dict(row) for row in raw_rows if int(row.get("rank", -1)) == 1]
        if not rows:
            raise ValueError(f"No rank-1 Policy-Pareto rows found in {path}.")
        return rows
    rows = []
    for rank, row in enumerate(raw_rows, start=1):
        item = dict(row)
        item["_policy_rank_hint"] = int(rank)
        rows.append(item)
    return rows


def policy_slug(row: dict[str, Any]) -> str:
    """Return the candidate-emission slug for a loaded policy row."""

    if "constraint_camb_fpr_max" in row and "constraint_real_fpr_max" in row and "rank" in row:
        return slugify(
            f"camb{row['constraint_camb_fpr_max']}_real{row['constraint_real_fpr_max']}_rank{row['rank']}"
        )
    rank = int(row.get("_policy_rank_hint", row.get("rank", 1)))
    return slugify(f"tile_constrained_rank{rank}_{row.get('family', 'policy')}")


def load_candidates(paths: list[str], limit: int = 0) -> list[dict[str, Any]]:
    """Load candidate JSONL records."""

    rows: list[dict[str, Any]] = []
    for path_text in paths:
        path = Path(path_text).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                record = json.loads(text)
                record.setdefault("candidate_jsonl", str(path))
                record.setdefault("candidate_jsonl_line", int(line_number))
                rows.append(record)
                if limit and len(rows) >= int(limit):
                    return rows
    if not rows:
        raise ValueError("No candidate records loaded.")
    return rows


def required_candidate_maps(candidates: list[dict[str, Any]]) -> tuple[str, ...]:
    """Return the unique candidate map labels in stable order."""

    seen: set[str] = set()
    maps: list[str] = []
    for record in candidates:
        name = str(record.get("map", "")).strip().lower()
        if not name:
            raise KeyError("Candidate record is missing a non-empty map field.")
        if name not in seen:
            seen.add(name)
            maps.append(name)
    return tuple(maps)


def resolve_map_path(
    *,
    direct_path: str,
    template_path: str,
    map_name: str,
) -> str:
    """Return the concrete path to use for one candidate map family."""

    if direct_path:
        return str(direct_path)
    if not template_path:
        return ""
    return str(template_path).format(map=str(map_name).lower())


def optional_resolved_path(path_text: str) -> str:
    """Return a resolved path string or an empty string."""

    if not str(path_text):
        return ""
    return str(Path(path_text).expanduser().resolve())


def candidate_center(record: dict[str, Any]) -> tuple[float, float]:
    """Return the patch center used for patch extraction."""

    for lon_key, lat_key in (("patch_glon_deg", "patch_glat_deg"), ("glon_deg", "glat_deg")):
        if lon_key in record and lat_key in record:
            lon = float(record[lon_key])
            lat = float(record[lat_key])
            if np.isfinite(lon) and np.isfinite(lat) and -90.0 <= lat <= 90.0:
                return lon % 360.0, lat
    raise KeyError(
        "Candidate record must contain patch_glon_deg/patch_glat_deg "
        "or glon_deg/glat_deg."
    )


def resolve_policy_for_candidate(
    record: dict[str, Any],
    policies_by_slug: dict[str, dict[str, Any]],
    override_slug: str = "",
) -> tuple[str, dict[str, Any]]:
    """Find the policy row corresponding to one candidate."""

    slug = override_slug or str(record.get("policy_slug", ""))
    if slug:
        if slug not in policies_by_slug:
            raise KeyError(f"Candidate references unknown policy_slug={slug!r}.")
        return slug, policies_by_slug[slug]
    if len(policies_by_slug) == 1:
        only_slug = next(iter(policies_by_slug))
        return only_slug, policies_by_slug[only_slug]
    raise KeyError(
        "Candidate record has no policy_slug and multiple rank-1 policies are available; "
        "pass --policy-slug."
    )


def load_map(path: Path, field: int) -> np.ndarray:
    """Load a HEALPix map from FITS, NPY, or NPZ in float64."""

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        loaded = np.load(path)
        if "map" in loaded:
            arr = loaded["map"]
        else:
            keys = list(loaded.files)
            if not keys:
                raise ValueError(f"{path} contains no arrays.")
            arr = loaded[keys[0]]
    else:
        arr = hp.read_map(path, field=int(field), dtype=np.float64)
    arr = np.asarray(arr, dtype=FLOAT_GEOMETRY_DTYPE)
    if arr.ndim != 1:
        raise ValueError(f"Expected a one-dimensional HEALPix map in {path}; got shape {arr.shape}.")
    if not hp.isnpixok(arr.size):
        raise ValueError(f"{path} length {arr.size} is not a valid HEALPix map length.")
    if not np.any(np.isfinite(arr)):
        raise ValueError(f"{path} contains no finite pixels.")
    return arr


def add_preflight_issue(
    issues: list[dict[str, Any]],
    severity: str,
    check: str,
    message: str,
    **context: Any,
) -> None:
    """Append a structured preflight issue."""

    issues.append(
        {
            "severity": str(severity),
            "check": str(check),
            "message": str(message),
            "context": context,
        }
    )


def inspect_map_input(
    path_text: str,
    *,
    label: str,
    field: int,
    target_nside: int,
    expect_half_mission: bool,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Inspect HM map metadata without running the expensive calibration."""

    if not path_text:
        add_preflight_issue(
            issues,
            "fail",
            f"{label}_present",
            f"{label} map path is missing.",
        )
        return {"label": label, "present": False}

    path = Path(path_text).expanduser().resolve()
    info: dict[str, Any] = {"label": label, "path": str(path), "present": path.exists()}
    if not path.exists():
        add_preflight_issue(
            issues,
            "fail",
            f"{label}_exists",
            f"{label} map path does not exist.",
            path=str(path),
        )
        return info

    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
            info["format"] = "npy"
            info["shape"] = list(arr.shape)
            if arr.ndim == 1 and hp.isnpixok(int(arr.shape[0])):
                info["nside"] = int(hp.npix2nside(int(arr.shape[0])))
        elif suffix == ".npz":
            with np.load(path) as loaded:
                keys = list(loaded.files)
                if not keys:
                    raise ValueError("NPZ contains no arrays.")
                key = "map" if "map" in keys else keys[0]
                arr = loaded[key]
                info["format"] = "npz"
                info["array_key"] = key
                info["shape"] = list(arr.shape)
                if arr.ndim == 1 and hp.isnpixok(int(arr.shape[0])):
                    info["nside"] = int(hp.npix2nside(int(arr.shape[0])))
        else:
            with fits.open(path, memmap=True) as hdul:
                info["format"] = "fits"
                info["num_hdus"] = len(hdul)
                for hdu in hdul:
                    header = hdu.header
                    if "NSIDE" in header:
                        info["nside"] = int(header["NSIDE"])
                        info["ordering"] = str(header.get("ORDERING", "")).strip()
                        info["coord"] = str(header.get("COORDSYS", header.get("COORD", ""))).strip()
                        info["fields"] = int(header.get("TFIELDS", 0) or 0)
                        info["npix"] = int(header.get("NAXIS2", 0) or 0)
                        break
        nside = info.get("nside")
        if nside is None:
            add_preflight_issue(
                issues,
                "warn",
                f"{label}_nside",
                f"Could not infer {label} HEALPix NSIDE without loading the full map.",
                path=str(path),
            )
        elif not hp.isnsideok(int(nside)):
            add_preflight_issue(
                issues,
                "fail",
                f"{label}_nside_valid",
                f"{label} map has invalid NSIDE.",
                nside=nside,
            )
        else:
            info["will_ud_grade_to_target"] = int(nside) != int(target_nside)
        if field < 0:
            add_preflight_issue(
                issues,
                "fail",
                f"{label}_field",
                "--fits-field must be non-negative.",
                field=field,
            )
        fields = info.get("fields")
        if fields and int(field) >= int(fields):
            add_preflight_issue(
                issues,
                "fail",
                f"{label}_field_exists",
                f"Requested FITS field {field} is outside available field count {fields}.",
                path=str(path),
                field=field,
                fields=fields,
            )
        if expect_half_mission and path.name.lower().find("hm") < 0 and "half" not in path.name.lower():
            add_preflight_issue(
                issues,
                "warn",
                f"{label}_filename",
                (
                    f"{label} filename does not look like a half-mission product. "
                    "Verify this is not a full-mission cleaned map."
                ),
                path=str(path),
            )
    except Exception as exc:
        add_preflight_issue(
            issues,
            "fail",
            f"{label}_inspect",
            f"Could not inspect {label} map metadata: {exc}",
            path=str(path),
        )
    return info


def preflight_models(
    specs: tuple[ModelSpec, ...],
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Check model run directories and checkpoint paths without loading tensors."""

    model_rows = []
    for spec in specs:
        run_dir = spec.run_dir.expanduser().resolve()
        row = {
            "name": spec.name,
            "run_dir": str(run_dir),
            "checkpoint": spec.checkpoint,
        }
        if not run_dir.exists():
            add_preflight_issue(
                issues,
                "fail",
                "model_run_dir_exists",
                "Model run directory does not exist.",
                model=spec.name,
                run_dir=str(run_dir),
            )
            model_rows.append(row)
            continue
        run_config = run_dir / "run_config.json"
        row["run_config_exists"] = run_config.exists()
        if not run_config.exists():
            add_preflight_issue(
                issues,
                "fail",
                "model_run_config_exists",
                "Model run_config.json is missing.",
                model=spec.name,
                run_dir=str(run_dir),
            )
        try:
            checkpoint_path, checkpoint_label = resolve_checkpoint_path(run_dir, spec.checkpoint)
            row["checkpoint_path"] = str(checkpoint_path)
            row["checkpoint_label"] = checkpoint_label
        except Exception as exc:
            add_preflight_issue(
                issues,
                "fail",
                "model_checkpoint_exists",
                f"Could not resolve checkpoint: {exc}",
                model=spec.name,
                run_dir=str(run_dir),
                checkpoint=spec.checkpoint,
            )
        model_rows.append(row)
    return model_rows


def preflight_candidates(
    candidates: list[dict[str, Any]],
    policies_by_slug: dict[str, dict[str, Any]],
    *,
    override_slug: str,
    min_valid_fraction: float,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check candidate JSONL compatibility with the selected policy report."""

    map_counts: dict[str, int] = {}
    slug_counts: dict[str, int] = {}
    missing_score_counts: dict[str, int] = {}
    min_mask_fraction = np.inf
    for idx, record in enumerate(candidates):
        map_name = str(record.get("map", "unknown"))
        map_counts[map_name] = map_counts.get(map_name, 0) + 1
        try:
            lon, lat = candidate_center(record)
            if not (0.0 <= lon < 360.0 and -90.0 <= lat <= 90.0):
                raise ValueError(f"invalid center ({lon}, {lat})")
        except Exception as exc:
            add_preflight_issue(
                issues,
                "fail",
                "candidate_center",
                f"Candidate center is invalid: {exc}",
                candidate_index=idx,
            )
        try:
            slug, policy_row = resolve_policy_for_candidate(
                record,
                policies_by_slug,
                override_slug=override_slug,
            )
            slug_counts[slug] = slug_counts.get(slug, 0) + 1
            for method in policy_row.get("thresholds", {}):
                score_key = f"score__{method}"
                if score_key not in record:
                    missing_score_counts[score_key] = missing_score_counts.get(score_key, 0) + 1
        except Exception as exc:
            add_preflight_issue(
                issues,
                "fail",
                "candidate_policy_slug",
                f"Candidate policy slug cannot be resolved: {exc}",
                candidate_index=idx,
                policy_slug=record.get("policy_slug"),
            )
        if "mask_fraction" in record:
            mask_fraction = float(record["mask_fraction"])
            min_mask_fraction = min(min_mask_fraction, mask_fraction)
            if mask_fraction < float(min_valid_fraction):
                add_preflight_issue(
                    issues,
                    "warn",
                    "candidate_mask_fraction",
                    (
                        "Candidate table mask_fraction is below --min-valid-fraction. "
                        "The projected HM mask check may skip or reject this candidate."
                    ),
                    candidate_index=idx,
                    mask_fraction=mask_fraction,
                    min_valid_fraction=float(min_valid_fraction),
                )
        else:
            add_preflight_issue(
                issues,
                "warn",
                "candidate_mask_fraction_present",
                "Candidate record has no mask_fraction field.",
                candidate_index=idx,
            )
    if missing_score_counts:
        add_preflight_issue(
            issues,
            "warn",
            "candidate_table_scores",
            (
                "Some candidates are missing stored score__ fields used for "
                "table-vs-recomputed score consistency checks."
            ),
            missing_score_counts=missing_score_counts,
        )
    return {
        "num_candidates": int(len(candidates)),
        "map_counts": map_counts,
        "policy_slug_counts": slug_counts,
        "min_candidate_mask_fraction": None
        if not np.isfinite(min_mask_fraction)
        else float(min_mask_fraction),
        "missing_score_counts": missing_score_counts,
    }


def write_preflight_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write the preflight report as Markdown."""

    lines = ["# Half-Mission Sign-Flip Preflight", ""]
    lines.append(f"- `status`: `{report['status']}`")
    lines.append(f"- `num_failures`: `{report['num_failures']}`")
    lines.append(f"- `num_warnings`: `{report['num_warnings']}`")
    lines.append(
        f"- `smallest_possible_empirical_p`: "
        f"`{report['run_settings']['smallest_possible_empirical_p']}`"
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for key, value in report["inputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Candidate Readiness")
    lines.append("")
    for key, value in report["candidate_summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Issues")
    lines.append("")
    if not report["issues"]:
        lines.append("No preflight issues found.")
    else:
        for issue in report["issues"]:
            context = issue.get("context", {})
            context_text = "" if not context else f" Context: `{context}`"
            lines.append(
                f"- `{issue['severity']}` `{issue['check']}`: "
                f"{issue['message']}{context_text}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_preflight(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    """Run non-inference readiness checks for Phase 5 HM calibration."""

    issues: list[dict[str, Any]] = []
    policy_path = Path(args.policy_json).expanduser().resolve()
    policy_rows = load_policy_rows(policy_path)
    policies_by_slug = {policy_slug(row): row for row in policy_rows}
    candidates = load_candidates(list(args.candidate_jsonl), int(args.candidate_limit))
    candidate_maps = required_candidate_maps(candidates)
    candidate_summary = preflight_candidates(
        candidates,
        policies_by_slug,
        override_slug=str(args.policy_slug),
        min_valid_fraction=float(args.min_valid_fraction),
        issues=issues,
    )
    hm1_infos: dict[str, dict[str, Any]] = {}
    hm2_infos: dict[str, dict[str, Any]] = {}
    for map_name in candidate_maps:
        hm1_path = resolve_map_path(
            direct_path=str(args.hm1_map),
            template_path=str(args.hm1_map_template),
            map_name=map_name,
        )
        hm2_path = resolve_map_path(
            direct_path=str(args.hm2_map),
            template_path=str(args.hm2_map_template),
            map_name=map_name,
        )
        hm1_infos[map_name] = inspect_map_input(
            hm1_path,
            label=f"hm1_{map_name}",
            field=int(args.fits_field),
            target_nside=int(args.target_nside),
            expect_half_mission=True,
            issues=issues,
        )
        hm2_infos[map_name] = inspect_map_input(
            hm2_path,
            label=f"hm2_{map_name}",
            field=int(args.fits_field),
            target_nside=int(args.target_nside),
            expect_half_mission=True,
            issues=issues,
        )
        hm1_nside = hm1_infos[map_name].get("nside")
        hm2_nside = hm2_infos[map_name].get("nside")
        if hm1_nside is not None and hm2_nside is not None and int(hm1_nside) != int(hm2_nside):
            add_preflight_issue(
                issues,
                "fail",
                "hm_nside_match",
                "HM1 and HM2 native NSIDE values differ.",
                map=map_name,
                hm1_nside=hm1_nside,
                hm2_nside=hm2_nside,
            )
    map_info = {
        "hm1_by_map": hm1_infos,
        "hm2_by_map": hm2_infos,
        "common_mask": inspect_map_input(
            str(args.common_mask),
            label="common_mask",
            field=0,
            target_nside=int(args.target_nside),
            expect_half_mission=False,
            issues=issues,
        ),
    }
    engine = str(args.circular_engine)
    if engine == "auto":
        engine = "torch" if torch.cuda.is_available() else "scipy"
    if engine == "torch" and not torch.cuda.is_available():
        add_preflight_issue(
            issues,
            "fail",
            "circular_engine_cuda",
            "Torch circular engine was requested but CUDA is unavailable.",
        )
    model_summary = preflight_models(tuple(args.models), issues)
    num_failures = sum(1 for issue in issues if issue["severity"] == "fail")
    num_warnings = sum(1 for issue in issues if issue["severity"] == "warn")
    report = {
        "status": "pass" if num_failures == 0 else "blocked",
        "num_failures": int(num_failures),
        "num_warnings": int(num_warnings),
        "inputs": {
            "hm1_map": str(args.hm1_map),
            "hm2_map": str(args.hm2_map),
            "hm1_map_template": str(args.hm1_map_template),
            "hm2_map_template": str(args.hm2_map_template),
            "common_mask": str(args.common_mask),
            "policy_json": str(policy_path),
            "candidate_jsonl": [
                str(Path(path).expanduser().resolve()) for path in args.candidate_jsonl
            ],
        },
        "run_settings": {
            "num_realizations": int(args.num_realizations),
            "smallest_possible_empirical_p": float(1.0 / (1 + int(args.num_realizations))),
            "target_nside": int(args.target_nside),
            "mask_threshold": float(args.mask_threshold),
            "min_valid_fraction": float(args.min_valid_fraction),
            "flip_mode": str(args.flip_mode),
            "flip_block_pix": int(args.flip_block_pix),
            "circular_engine_resolved": engine,
            "theta_grid_deg": [float(x) for x in args.theta_grid_deg],
        },
        "policy_slugs_available": sorted(policies_by_slug),
        "candidate_summary": candidate_summary,
        "map_info": map_info,
        "model_summary": model_summary,
        "issues": issues,
    }
    json_path = output_dir / "hm_signflip_preflight_report.json"
    md_path = output_dir / "hm_signflip_preflight_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_preflight_markdown(md_path, report)
    return report


def degrade_map_if_needed(hp_map: np.ndarray, target_nside: int) -> np.ndarray:
    """Return a map at the target HEALPix Nside."""

    nside = hp.get_nside(hp_map)
    if nside == int(target_nside):
        return np.asarray(hp_map, dtype=FLOAT_GEOMETRY_DTYPE)
    return np.asarray(
        hp.ud_grade(hp_map, int(target_nside), order_in="RING", order_out="RING"),
        dtype=FLOAT_GEOMETRY_DTYPE,
    )


def load_analysis_mask(args: argparse.Namespace) -> np.ndarray:
    """Load or download the common analysis mask at target Nside."""

    mask_path = Path(str(args.common_mask)).expanduser()
    if mask_path.exists():
        mask = hp.read_map(mask_path, field=0, dtype=np.float64)
        mask = hp.ud_grade(mask, int(args.target_nside), order_in="RING", order_out="RING")
        return np.where(np.asarray(mask) >= float(args.mask_threshold), 1.0, 0.0).astype(np.float32)
    mask_256, _sky_fraction = load_mask(threshold=float(args.mask_threshold))
    if hp.get_nside(mask_256) != int(args.target_nside):
        mask_256 = hp.ud_grade(mask_256, int(args.target_nside), order_in="RING", order_out="RING")
    return np.where(np.asarray(mask_256) >= 0.5, 1.0, 0.0).astype(np.float32)


def prepare_half_mission_maps(
    args: argparse.Namespace,
    *,
    map_name: str,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load, match, low-mode clean, and mask one map family's HM products."""

    hm1_path = resolve_map_path(
        direct_path=str(args.hm1_map),
        template_path=str(args.hm1_map_template),
        map_name=map_name,
    )
    hm2_path = resolve_map_path(
        direct_path=str(args.hm2_map),
        template_path=str(args.hm2_map_template),
        map_name=map_name,
    )
    if not hm1_path or not hm2_path:
        raise FileNotFoundError(
            f"Missing HM paths for map {map_name!r}; provide direct paths or per-map templates."
        )
    hm1 = degrade_map_if_needed(
        load_map(Path(hm1_path).expanduser().resolve(), int(args.fits_field)),
        args.target_nside,
    )
    hm2 = degrade_map_if_needed(
        load_map(Path(hm2_path).expanduser().resolve(), int(args.fits_field)),
        args.target_nside,
    )
    if hm1.shape != hm2.shape:
        raise ValueError(f"HM map shape mismatch: {hm1.shape} vs {hm2.shape}.")
    if hp.get_nside(hm1) != hp.get_nside(hm2):
        raise ValueError("HM maps have inconsistent NSIDE after degradation.")
    if mask.shape != hm1.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match HM maps {hm1.shape}.")
    if not args.skip_low_mode_removal:
        hm1 = remove_real_map_low_modes(hm1, mask=mask)
        hm2 = remove_real_map_low_modes(hm2, mask=mask)
    bad = (mask <= 0.0) | ~np.isfinite(hm1) | ~np.isfinite(hm2)
    hm1 = np.asarray(hm1, dtype=FLOAT_GEOMETRY_DTYPE)
    hm2 = np.asarray(hm2, dtype=FLOAT_GEOMETRY_DTYPE)
    hm1[bad] = 0.0
    hm2[bad] = 0.0
    mean_map = 0.5 * (hm1 + hm2)
    diff_map = 0.5 * (hm1 - hm2)
    if not np.all(np.isfinite(mean_map)) or not np.all(np.isfinite(diff_map)):
        raise ValueError("Prepared half-mission maps contain non-finite values.")
    max_abs = max(float(np.max(np.abs(mean_map))), float(np.max(np.abs(diff_map))))
    if max_abs > 1.0:
        raise ValueError(
            f"Prepared HM maps exceed anisotropy scale: max |T|={max_abs:.3g} K. "
            "Use Kelvin anisotropy maps, not microkelvin or full-temperature maps."
        )
    return mean_map, diff_map


def load_models(specs: tuple[ModelSpec, ...], device: torch.device) -> list[LoadedModel]:
    """Load one-channel remediated U-Net scorers."""

    out: list[LoadedModel] = []
    for spec in specs:
        model, run_config, checkpoint_path, checkpoint_label = build_model_from_run(
            spec.run_dir.resolve(),
            spec.checkpoint,
            device,
        )
        input_config = p3.input_config_from_run_config(run_config)
        if int(input_config["input_channels"]) != 1:
            raise ValueError(
                f"{spec.name} expects {input_config['input_channels']} input channels. "
                "Half-mission in-memory scoring currently supports one-channel models only."
            )
        mean = float(input_config["channel_means"][0])
        std = float(input_config["channel_stds"][0])
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"Invalid normalization for {spec.name}: mean={mean}, std={std}.")
        out.append(
            LoadedModel(
                spec=spec,
                model=model,
                channel_mean=mean,
                channel_std=max(std, 1.0e-8),
                checkpoint_path=checkpoint_path,
                checkpoint_label=checkpoint_label,
            )
        )
    return out


def score_ml_patches(
    patches: np.ndarray,
    loaded: LoadedModel,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Score in-memory Kelvin patches with a frozen U-Net."""

    arr = np.asarray(patches, dtype=FLOAT_STORAGE_DTYPE)
    if arr.ndim != 3 or arr.shape[1:] != (PATCH_PIX, PATCH_PIX):
        raise ValueError(f"Expected patches shaped (N, {PATCH_PIX}, {PATCH_PIX}); got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite patches passed to {loaded.spec.name}.")
    scores = np.zeros(arr.shape[0], dtype=np.float32)
    loaded.model.eval()
    offset = 0
    with torch.no_grad():
        for start in range(0, arr.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), arr.shape[0])
            batch = (arr[start:stop] - loaded.channel_mean) / loaded.channel_std
            tensor = torch.as_tensor(batch[:, None, :, :], dtype=torch.float32, device=device)
            mask_logits, _aux_logits = p3.unpack_model_output(loaded.model(tensor))
            score = torch.sigmoid(mask_logits).flatten(1).max(dim=1).values
            scores[offset : offset + (stop - start)] = score.detach().cpu().numpy()
            offset += stop - start
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite ML scores from {loaded.spec.name}.")
    return scores


def prepare_circular_kernel_fft(kernels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Prepare circular-template kernel FFTs for torch scoring."""

    full_shape = (2 * PATCH_PIX - 1, 2 * PATCH_PIX - 1)
    kernel_tensor = torch.zeros(
        (kernels.shape[0], full_shape[0], full_shape[1]),
        dtype=torch.float32,
        device=device,
    )
    kernel_tensor[:, :PATCH_PIX, :PATCH_PIX] = torch.as_tensor(
        np.ascontiguousarray(kernels),
        dtype=torch.float32,
        device=device,
    )
    return torch.fft.rfft2(kernel_tensor, s=full_shape)


def score_circular_patches_scipy(
    patches: np.ndarray,
    *,
    kernels: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Score patches with the CPU circular-template screen."""

    scores = np.zeros(patches.shape[0], dtype=np.float32)
    for start in range(0, patches.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), patches.shape[0])
        patch_batch = standardize_patch_batch(np.asarray(patches[start:stop], dtype=np.float32))
        best = np.full(patch_batch.shape[0], -np.inf, dtype=np.float32)
        for kernel in kernels:
            response = fftconvolve(
                patch_batch,
                kernel[None, :, :],
                mode="same",
                axes=(-2, -1),
            )
            best = np.maximum(best, np.max(response, axis=(1, 2)).astype(np.float32))
        scores[start:stop] = best
    if not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite circular-template scores.")
    return scores


def score_circular_patches_torch(
    patches: np.ndarray,
    *,
    kernel_fft: torch.Tensor,
    kernel_count: int,
    kernel_chunk: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Score patches with a torch FFT circular-template screen."""

    full_shape = (2 * PATCH_PIX - 1, 2 * PATCH_PIX - 1)
    crop = (PATCH_PIX - 1) // 2
    scores = np.zeros(patches.shape[0], dtype=np.float32)
    for start in range(0, patches.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), patches.shape[0])
        patch_batch = standardize_patch_batch(np.asarray(patches[start:stop], dtype=np.float32))
        batch_tensor = torch.zeros(
            (patch_batch.shape[0], full_shape[0], full_shape[1]),
            dtype=torch.float32,
            device=device,
        )
        batch_tensor[:, :PATCH_PIX, :PATCH_PIX] = torch.as_tensor(
            np.ascontiguousarray(patch_batch),
            dtype=torch.float32,
            device=device,
        )
        batch_fft = torch.fft.rfft2(batch_tensor, s=full_shape)
        best = torch.full((patch_batch.shape[0],), -torch.inf, dtype=torch.float32, device=device)
        for k0 in range(0, int(kernel_count), int(kernel_chunk)):
            k1 = min(k0 + int(kernel_chunk), int(kernel_count))
            conv = torch.fft.irfft2(
                batch_fft[:, None, :, :] * kernel_fft[None, k0:k1, :, :],
                s=full_shape,
            )
            same = conv[:, :, crop : crop + PATCH_PIX, crop : crop + PATCH_PIX]
            best = torch.maximum(best, torch.amax(same, dim=(1, 2, 3)))
            del conv, same
        scores[start:stop] = best.detach().cpu().numpy()
        del batch_tensor, batch_fft, best
    if not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite circular-template scores.")
    return scores


def score_circular_patches(
    patches: np.ndarray,
    *,
    kernels: np.ndarray,
    kernel_fft: torch.Tensor | None,
    kernel_chunk: int,
    batch_size: int,
    device: torch.device,
    engine: str,
) -> np.ndarray:
    """Score patches with the selected circular-template engine."""

    if engine == "auto":
        engine = "torch" if device.type == "cuda" else "scipy"
    if engine == "torch":
        if device.type != "cuda":
            raise ValueError("Torch circular engine requires CUDA.")
        if kernel_fft is None:
            raise ValueError("Torch circular engine requires a prepared kernel FFT.")
        return score_circular_patches_torch(
            patches,
            kernel_fft=kernel_fft,
            kernel_count=int(kernels.shape[0]),
            kernel_chunk=int(kernel_chunk),
            batch_size=int(batch_size),
            device=device,
        )
    return score_circular_patches_scipy(
        patches,
        kernels=kernels,
        batch_size=int(batch_size),
    )


def score_patch_array(
    patches: np.ndarray,
    *,
    loaded_models: list[LoadedModel],
    kernels: np.ndarray,
    kernel_fft: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Compute all policy scores for an in-memory patch array."""

    scores: dict[str, np.ndarray] = {}
    for loaded in loaded_models:
        scores[loaded.spec.name] = score_ml_patches(
            patches,
            loaded,
            batch_size=int(args.batch_size),
            device=device,
        )
    scores[CIRCULAR_METHOD] = score_circular_patches(
        patches,
        kernels=kernels,
        kernel_fft=kernel_fft,
        kernel_chunk=int(args.circular_kernel_chunk),
        batch_size=int(args.circular_batch_size),
        device=device,
        engine=str(args.circular_engine),
    )
    return scores


def patch_from_map(hp_map: np.ndarray, glon_deg: float, glat_deg: float) -> np.ndarray:
    """Project a HEALPix map to the canonical patch geometry in float64."""

    patch = project_patch(hp_map, float(glon_deg), float(glat_deg))
    patch = np.ma.filled(patch, 0.0)
    patch = np.asarray(patch, dtype=FLOAT_GEOMETRY_DTYPE)
    if patch.shape != (PATCH_PIX, PATCH_PIX):
        raise ValueError(f"Projected patch has unexpected shape {patch.shape}.")
    if not np.all(np.isfinite(patch)):
        raise ValueError("Projected patch contains non-finite values.")
    return patch


def sign_field(
    rng: np.random.Generator,
    *,
    mode: str,
    block_pix: int,
) -> np.ndarray:
    """Draw a pixel or block-correlated sign field."""

    if mode == "pixel":
        return rng.choice((-1.0, 1.0), size=(PATCH_PIX, PATCH_PIX)).astype(np.float32)
    coarse_shape = (
        int(np.ceil(PATCH_PIX / int(block_pix))),
        int(np.ceil(PATCH_PIX / int(block_pix))),
    )
    coarse = rng.choice((-1.0, 1.0), size=coarse_shape).astype(np.float32)
    return np.repeat(np.repeat(coarse, int(block_pix), axis=0), int(block_pix), axis=1)[
        :PATCH_PIX,
        :PATCH_PIX,
    ]


def build_null_patches(
    mean_patch: np.ndarray,
    diff_patch: np.ndarray,
    valid_patch: np.ndarray,
    *,
    num_realizations: int,
    rng: np.random.Generator,
    flip_mode: str,
    flip_block_pix: int,
) -> np.ndarray:
    """Build HM sign-flip null patches for one candidate."""

    out = np.empty((int(num_realizations), PATCH_PIX, PATCH_PIX), dtype=FLOAT_STORAGE_DTYPE)
    for idx in range(int(num_realizations)):
        signs = sign_field(rng, mode=flip_mode, block_pix=int(flip_block_pix))
        patch = np.asarray(mean_patch + signs * diff_patch, dtype=FLOAT_GEOMETRY_DTYPE)
        patch[~valid_patch] = 0.0
        out[idx] = np.asarray(patch, dtype=FLOAT_STORAGE_DTYPE)
    return out


def scalar_policy_scores(row: dict[str, Any], scores: dict[str, float]) -> tuple[bool, float]:
    """Apply a policy to scalar scores."""

    arrays = {method: np.asarray([value], dtype=np.float64) for method, value in scores.items()}
    return bool(apply_policy(row, arrays)[0]), float(policy_margin(row, arrays)[0])


def summarize_quantiles(values: np.ndarray) -> dict[str, float]:
    """Return fixed quantiles for a score array."""

    quantiles = np.quantile(np.asarray(values, dtype=np.float64), [0.05, 0.16, 0.5, 0.84, 0.95])
    return {
        "q05": float(quantiles[0]),
        "q16": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q84": float(quantiles[3]),
        "q95": float(quantiles[4]),
    }


def candidate_table_scores(record: dict[str, Any]) -> dict[str, float]:
    """Extract any stored score__ fields from a candidate record."""

    out = {}
    for method in (*ML_METHODS, CIRCULAR_METHOD):
        key = f"score__{method}"
        if key in record:
            out[method] = float(record[key])
    return out


def evaluate_candidate(
    record: dict[str, Any],
    candidate_idx: int,
    *,
    mean_map: np.ndarray,
    diff_map: np.ndarray,
    mask_map: np.ndarray,
    policy_row: dict[str, Any],
    policy_slug_value: str,
    loaded_models: list[LoadedModel],
    kernels: np.ndarray,
    kernel_fft: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one candidate and return summary plus null arrays."""

    glon_deg, glat_deg = candidate_center(record)
    mask_patch = patch_from_map(mask_map, glon_deg, glat_deg)
    valid_patch = np.isfinite(mask_patch) & (mask_patch > 0.5)
    valid_fraction = projected_unmasked_fraction(mask_patch)
    if valid_fraction < float(args.min_valid_fraction):
        message = (
            f"Candidate {candidate_idx} valid fraction {valid_fraction:.4f} "
            f"is below --min-valid-fraction {float(args.min_valid_fraction):.4f}."
        )
        if args.on_invalid == "skip":
            skipped = {
                "candidate_index": int(candidate_idx),
                "status": "skipped_invalid_mask_fraction",
                "reason": message,
                "patch_glon_deg": float(glon_deg),
                "patch_glat_deg": float(glat_deg),
                "policy_slug": policy_slug_value,
            }
            return skipped, {}
        raise ValueError(message)

    mean_patch = patch_from_map(mean_map, glon_deg, glat_deg)
    diff_patch = patch_from_map(diff_map, glon_deg, glat_deg)
    mean_patch[~valid_patch] = 0.0
    diff_patch[~valid_patch] = 0.0
    observed_patch = np.asarray(mean_patch[None, :, :], dtype=FLOAT_STORAGE_DTYPE)
    null_patches = build_null_patches(
        mean_patch,
        diff_patch,
        valid_patch,
        num_realizations=int(args.num_realizations),
        rng=rng,
        flip_mode=str(args.flip_mode),
        flip_block_pix=int(args.flip_block_pix),
    )

    observed_scores = {
        key: float(value[0])
        for key, value in score_patch_array(
            observed_patch,
            loaded_models=loaded_models,
            kernels=kernels,
            kernel_fft=kernel_fft,
            args=args,
            device=device,
        ).items()
    }
    observed_pass, observed_margin = scalar_policy_scores(policy_row, observed_scores)

    null_scores = score_patch_array(
        null_patches,
        loaded_models=loaded_models,
        kernels=kernels,
        kernel_fft=kernel_fft,
        args=args,
        device=device,
    )
    null_pass = apply_policy(policy_row, null_scores).astype(np.uint8)
    null_margin = policy_margin(policy_row, null_scores).astype(np.float32)
    exceed = int(np.count_nonzero(null_margin >= float(observed_margin)))
    empirical_p = float((1 + exceed) / (1 + int(args.num_realizations)))

    table_scores = candidate_table_scores(record)
    table_policy = None
    if set(policy_row["thresholds"]).issubset(table_scores):
        table_pass, table_margin = scalar_policy_scores(policy_row, table_scores)
        table_policy = {
            "policy_pass": bool(table_pass),
            "policy_margin": float(table_margin),
        }

    valid_mean = mean_patch[valid_patch]
    valid_diff = diff_patch[valid_patch]
    diff_rms_uk = (float(np.std(valid_diff, dtype=np.float64)) * u.K).to_value(u.uK)
    mean_std_uk = (float(np.std(valid_mean, dtype=np.float64)) * u.K).to_value(u.uK)
    result = {
        "candidate_index": int(candidate_idx),
        "status": "ok",
        "source_candidate": {
            key: record.get(key)
            for key in (
                "candidate_jsonl",
                "candidate_jsonl_line",
                "map",
                "patch_index",
                "policy_slug",
                "policy",
                "patch_glon_deg",
                "patch_glat_deg",
                "peak_glon_deg",
                "peak_glat_deg",
                "mask_fraction",
            )
            if key in record
        },
        "patch_glon_deg": float(glon_deg),
        "patch_glat_deg": float(glat_deg),
        "mask_valid_fraction": float(valid_fraction),
        "hm_mean_patch_std_uk": float(mean_std_uk),
        "hm_diff_patch_rms_uk": float(diff_rms_uk),
        "policy_slug": policy_slug_value,
        "policy": policy_row["policy"],
        "policy_family": policy_row["family"],
        "policy_thresholds": {key: float(value) for key, value in policy_row["thresholds"].items()},
        "observed_scores": observed_scores,
        "observed_policy_pass": bool(observed_pass),
        "observed_policy_margin": float(observed_margin),
        "candidate_table_scores": table_scores,
        "candidate_table_policy": table_policy,
        "null_num_realizations": int(args.num_realizations),
        "null_policy_pass_count": int(np.count_nonzero(null_pass)),
        "null_policy_pass_fraction": float(np.mean(null_pass)),
        "null_margin_quantiles": summarize_quantiles(null_margin),
        "null_score_quantiles": {
            method: summarize_quantiles(values)
            for method, values in null_scores.items()
        },
        "hm_signflip_empirical_p_value": empirical_p,
        "hm_signflip_exceedance_count": exceed,
        "p_value_interpretation": (
            "Conditional on the fixed HM mean sky patch and this frozen scorer, "
            "using sign-flipped HM1-HM2/2 as an empirical noise proxy."
        ),
    }
    arrays = {
        f"score__{method}": np.asarray(values, dtype=np.float32)
        for method, values in null_scores.items()
    }
    arrays["policy_margin"] = np.asarray(null_margin, dtype=np.float32)
    arrays["policy_pass"] = np.asarray(null_pass, dtype=np.uint8)
    return result, arrays


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact candidate summary CSV."""

    columns = [
        "candidate_index",
        "status",
        "policy_slug",
        "patch_glon_deg",
        "patch_glat_deg",
        "mask_valid_fraction",
        "hm_mean_patch_std_uk",
        "hm_diff_patch_rms_uk",
        "observed_policy_pass",
        "observed_policy_margin",
        "null_policy_pass_count",
        "null_policy_pass_fraction",
        "hm_signflip_empirical_p_value",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a human-readable report."""

    lines = ["# Half-Mission Sign-Flip Null Calibration", ""]
    lines.append("This is a per-candidate conditional noise-robustness calibration.")
    lines.append("It is not a global LambdaCDM detection p-value or Bayesian evidence ratio.")
    lines.append("")
    lines.append("## Assumptions")
    lines.append("")
    for note in report["assumption_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Candidate Summary")
    lines.append("")
    lines.append("| idx | status | policy | observed pass | observed margin | null pass frac | p-value | HM diff RMS uK |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for row in report["candidates"]:
        lines.append(
            f"| {row.get('candidate_index')} | {row.get('status')} | "
            f"`{row.get('policy_slug', '')}` | {row.get('observed_policy_pass', '')} | "
            f"{float(row.get('observed_policy_margin', np.nan)):.4f} | "
            f"{float(row.get('null_policy_pass_fraction', np.nan)):.4f} | "
            f"{float(row.get('hm_signflip_empirical_p_value', np.nan)):.4f} | "
            f"{float(row.get('hm_diff_patch_rms_uk', np.nan)):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        report = run_preflight(args, output_dir)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "num_failures": report["num_failures"],
                    "num_warnings": report["num_warnings"],
                    "json": str(output_dir / "hm_signflip_preflight_report.json"),
                    "markdown": str(output_dir / "hm_signflip_preflight_report.md"),
                },
                indent=2,
            )
        )
        return
    device = p3.resolve_device(args.device)

    policies = load_policy_rows(Path(args.policy_json).expanduser().resolve())
    policies_by_slug = {policy_slug(row): row for row in policies}
    candidates = load_candidates(list(args.candidate_jsonl), int(args.candidate_limit))
    mask_map = load_analysis_mask(args)
    loaded_models = load_models(args.models, device)
    kernels = circular_kernels(tuple(args.theta_grid_deg), float(args.beam_fwhm_arcmin))[:, ::-1, ::-1]
    engine = str(args.circular_engine)
    if engine == "auto":
        engine = "torch" if device.type == "cuda" else "scipy"
    args.circular_engine = engine
    kernel_fft = prepare_circular_kernel_fft(kernels, device) if engine == "torch" else None
    rng = np.random.default_rng(int(args.seed))
    map_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    rows: list[dict[str, Any]] = []
    null_arrays: dict[str, np.ndarray] = {}
    for idx, record in enumerate(candidates):
        slug, policy_row = resolve_policy_for_candidate(
            record,
            policies_by_slug,
            override_slug=str(args.policy_slug),
        )
        map_name = str(record.get("map", "")).strip().lower()
        if not map_name:
            raise KeyError(f"Candidate {idx} is missing a non-empty map field.")
        if map_name not in map_cache:
            map_cache[map_name] = prepare_half_mission_maps(
                args,
                map_name=map_name,
                mask=mask_map,
            )
        mean_map, diff_map = map_cache[map_name]
        result, arrays = evaluate_candidate(
            record,
            idx,
            mean_map=mean_map,
            diff_map=diff_map,
            mask_map=mask_map,
            policy_row=policy_row,
            policy_slug_value=slug,
            loaded_models=loaded_models,
            kernels=kernels,
            kernel_fft=kernel_fft,
            args=args,
            device=device,
            rng=rng,
        )
        rows.append(result)
        for key, value in arrays.items():
            null_arrays[f"candidate{idx:05d}__{key}"] = np.asarray(value)

    report = {
        "metadata": {
            "hm1_map": optional_resolved_path(str(args.hm1_map)),
            "hm2_map": optional_resolved_path(str(args.hm2_map)),
            "hm1_map_template": str(args.hm1_map_template),
            "hm2_map_template": str(args.hm2_map_template),
            "policy_json": str(Path(args.policy_json).expanduser().resolve()),
            "candidate_jsonl": [str(Path(path).expanduser().resolve()) for path in args.candidate_jsonl],
            "num_candidates": int(len(candidates)),
            "candidate_maps": sorted(required_candidate_maps(candidates)),
            "num_realizations": int(args.num_realizations),
            "seed": int(args.seed),
            "target_nside": int(args.target_nside),
            "mask_threshold": float(args.mask_threshold),
            "min_valid_fraction": float(args.min_valid_fraction),
            "flip_mode": str(args.flip_mode),
            "flip_block_pix": int(args.flip_block_pix),
            "circular_engine": str(args.circular_engine),
            "theta_grid_deg": [float(x) for x in args.theta_grid_deg],
            "beam_fwhm_arcmin": float(args.beam_fwhm_arcmin),
            "models": [
                {
                    "name": loaded.spec.name,
                    "run_dir": str(loaded.spec.run_dir),
                    "checkpoint": loaded.spec.checkpoint,
                    "checkpoint_path": loaded.checkpoint_path,
                    "checkpoint_label": loaded.checkpoint_label,
                }
                for loaded in loaded_models
            ],
        },
        "assumption_notes": [
            "HM1/HM2 maps are treated as same-beam, same-component-separation Kelvin anisotropy maps.",
            "T_mean = 0.5 * (HM1 + HM2) preserves the fixed sky realization.",
            "T_diff = 0.5 * (HM1 - HM2) is used as an empirical mean-map noise proxy.",
            "Empirical p-values compare threshold-relative policy margins under sign-flipped T_diff.",
            "These p-values do not marginalize over LambdaCDM CMB skies or foreground residual models.",
        ],
        "candidates": rows,
    }

    json_path = output_dir / "hm_signflip_null_report.json"
    csv_path = output_dir / "hm_signflip_null_summary.csv"
    md_path = output_dir / "hm_signflip_null_report.md"
    npz_path = output_dir / "hm_signflip_null_scores.npz"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(md_path, report)
    if null_arrays:
        np.savez_compressed(npz_path, **null_arrays)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(md_path),
                "npz": str(npz_path) if null_arrays else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
