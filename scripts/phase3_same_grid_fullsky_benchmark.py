"""Run a full-sky same-grid pilot benchmark for ML and classical screeners.

Assumptions
-----------
* This script creates new full-sky benchmark products. It does not pretend that
  an existing patch-space HDF5 contains enough information to run nonlocal
  spherical filters after the fact.
* The default injection convention is the McEwen et al. (2012) first-order
  additive template because the Wiener/SMHW comparison is a filter-design
  benchmark. Use ``feeney2011_full_temperature_modulation`` only when the
  benchmark report explicitly quotes the Feeney-vs-additive cross-term audit.
* CMB realizations are reconstructed from the CAMB/synfast seed provenance in
  the source sensitivity HDF5. Signals are injected on the HEALPix sphere, then
  the map is convolved with the configured Gaussian beam and HEALPix pixel
  window before white noise is added.
* Classical scores are blind local maxima over the filter bank in the same
  candidate region used for positives and negatives. They are not Bayesian
  evidence and are not an OSS masked-sky optimum.
* Full production runs are expensive. A subset run is a scientifically useful
  smoke/pilot artifact, not permission to claim ML superiority.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import healpy as hp
import numpy as np
import torch
from astropy import units as u

import phase3_train_unet as p3
from phase2_observing_model import (
    camb_tt_cls,
    inject_signal_on_sphere,
    offset_signal_vector,
    project_patch,
    temporary_numpy_seed,
)
from phase3_classical_filters import (
    PIXEL_WINDOW_POLICIES,
    apply_precomputed_smhw_bank,
    apply_precomputed_wiener_bank,
    effective_beam_l,
    precompute_smhw_filter_bank,
    precompute_wiener_feeney_filter_bank,
    smhw_screen_maps,
    validate_cmb_map,
    wiener_feeney_matched_filter_maps,
)
from phase3_method_registry import SMHW_SCREEN, WIENER_FEENEY_MATCHED_FILTER, method_metadata
from phase3_sensitivity_curve import (
    DEFAULT_MODELS,
    binomial_ci,
    build_model_from_run,
    parse_model_spec,
    threshold_from_negatives,
)
from phase_config import (
    DEFAULTS,
    FLOAT_GEOMETRY_DTYPE,
    FLOAT_STORAGE_DTYPE,
    INJECTION_CONVENTION_MCEWEN2012,
    INJECTION_CONVENTION_NOTES,
    INJECTION_CONVENTIONS,
    PATCH_PIX,
    PROVENANCE_SCHEMA_VERSION,
    RESO_ARCMIN,
    beam_fwhm_rad,
    patch_half_width_deg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_H5 = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_sensitivity_curve" / "sensitivity_data.h5"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_same_grid_fullsky_pilot"
SCORE_REGIONS = ("candidate_pixel", "patch_disc_max")


@dataclass
class SourceRows:
    """Selected source rows needed to build a same-grid full-sky benchmark."""

    row_index: np.ndarray
    labels: np.ndarray
    glon_deg: np.ndarray
    glat_deg: np.ndarray
    cmb_realization_idx: np.ndarray
    coord_cluster_id: np.ndarray
    background_id: np.ndarray
    amplitude: np.ndarray
    theta_crit_deg: np.ndarray
    z0: np.ndarray
    zcrit: np.ndarray
    zcrit_ratio: np.ndarray
    signal_center_x_pix: np.ndarray
    signal_center_y_pix: np.ndarray
    amplitude_idx: np.ndarray
    theta_idx: np.ndarray
    zcrit_ratio_idx: np.ndarray
    sign_quadrant: np.ndarray
    masks: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate and score full-sky same-grid benchmark maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-h5", type=str, default=str(DEFAULT_SOURCE_H5))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rows", type=str, default="", help="Comma-separated source row indices. Overrides selection.")
    parser.add_argument(
        "--rows-json",
        type=str,
        default="",
        help="JSON file containing a top-level list of row indices or an object with a `rows` field.",
    )
    parser.add_argument(
        "--manifest-json",
        type=str,
        default="",
        help="Manifest JSON from phase3_same_grid_build_manifest.py.",
    )
    parser.add_argument(
        "--manifest-shard-id",
        type=int,
        default=-1,
        help="Shard id from --manifest-json. Use -1 to run the full manifest row list.",
    )
    parser.add_argument("--row-selection", choices=("balanced", "positives", "negatives", "all"), default="balanced")
    parser.add_argument("--max-rows", type=int, default=64, help="0 means all selected rows.")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--nside", type=int, default=0, help="0 means source HDF5 nside.")
    parser.add_argument("--lmax", type=int, default=0, help="0 means 3*nside-1.")
    parser.add_argument("--theta-grid-deg", type=str, default="", help="Comma list; empty means source grid.")
    parser.add_argument("--smhw-scales-deg", type=str, default="2,5,10,15,20")
    parser.add_argument("--beam-fwhm-arcmin", type=float, default=-1.0)
    parser.add_argument("--noise-sigma-uk-arcmin", type=float, default=-1.0)
    parser.add_argument("--quadrature-order", type=int, default=1024)
    parser.add_argument("--cmb-realization-cache", type=int, default=4)
    parser.add_argument(
        "--pixel-window-policy",
        choices=PIXEL_WINDOW_POLICIES,
        default=DEFAULTS.pixel_window_policy,
        help="Effective transfer applied after full-sky signal injection.",
    )
    parser.add_argument(
        "--injection-convention",
        choices=INJECTION_CONVENTIONS,
        default=INJECTION_CONVENTION_MCEWEN2012,
    )
    parser.add_argument("--score-region", choices=SCORE_REGIONS, default="patch_disc_max")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--model", action="append", default=[], help="ML model as name:run_dir:checkpoint.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-ml-data", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def parse_json_attr(value: Any) -> Any:
    """Parse a JSON-encoded HDF5 attribute if needed."""

    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def parse_float_list(text: str) -> tuple[float, ...]:
    """Parse a comma-separated float list."""

    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def load_source_summary(path: Path) -> dict[str, Any]:
    """Load compact source-HDF5 provenance."""

    with h5py.File(path, "r") as h5:
        if "summary" not in h5:
            raise ValueError("Source HDF5 is missing summary attrs.")
        summary = dict(h5["summary"].attrs)
    out = {key: value for key, value in summary.items()}
    for key in ("amplitude_grid", "theta_grid_deg", "zcrit_ratio_grid", "camb_params"):
        if key in out:
            out[key] = parse_json_attr(out[key])
    return out


def validate_args(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    """Validate physical and numerical settings."""

    source_nside = int(summary.get("nside", DEFAULTS.nside))
    nside = int(args.nside) if int(args.nside) else source_nside
    if nside <= 0 or not hp.isnsideok(nside):
        raise ValueError("--nside must be a valid HEALPix Nside.")
    lmax = int(args.lmax) if int(args.lmax) else 3 * nside - 1
    if lmax < 2:
        raise ValueError("--lmax must be at least 2.")
    if int(args.quadrature_order) <= lmax:
        raise ValueError("--quadrature-order must exceed lmax.")
    if int(args.max_rows) < 0:
        raise ValueError("--max-rows must be non-negative.")
    explicit_row_sources = int(bool(args.rows)) + int(bool(args.rows_json)) + int(bool(args.manifest_json))
    if explicit_row_sources > 1:
        raise ValueError("Use only one of --rows, --rows-json, or --manifest-json.")
    if args.row_selection == "balanced" and int(args.max_rows) == 1:
        raise ValueError("--row-selection balanced needs --max-rows 0 or at least 2.")
    if int(args.cmb_realization_cache) <= 0:
        raise ValueError("--cmb-realization-cache must be positive.")
    if int(args.manifest_shard_id) < -1:
        raise ValueError("--manifest-shard-id must be >= -1.")

    beam_fwhm = (
        float(args.beam_fwhm_arcmin)
        if float(args.beam_fwhm_arcmin) >= 0.0
        else float(summary.get("beam_fwhm_arcmin", DEFAULTS.beam_fwhm_arcmin))
    )
    noise_sigma = (
        float(args.noise_sigma_uk_arcmin)
        if float(args.noise_sigma_uk_arcmin) >= 0.0
        else float(summary.get("noise_sigma_uk_arcmin", 0.0))
    )
    if beam_fwhm < 0.0:
        raise ValueError("Beam FWHM must be non-negative.")
    if noise_sigma < 0.0:
        raise ValueError("Noise depth must be non-negative.")

    theta_grid = parse_float_list(args.theta_grid_deg) if args.theta_grid_deg else tuple(
        float(value) for value in summary["theta_grid_deg"]
    )
    smhw_scales = parse_float_list(args.smhw_scales_deg)
    if any(theta <= 0.0 or theta >= 180.0 for theta in theta_grid):
        raise ValueError("All theta values must lie in (0, 180) deg.")
    if any(scale <= 0.0 for scale in smhw_scales):
        raise ValueError("All SMHW scales must be positive.")

    return {
        "source_nside": source_nside,
        "nside": nside,
        "lmax": lmax,
        "theta_grid_deg": theta_grid,
        "smhw_scales_deg": smhw_scales,
        "beam_fwhm_arcmin": beam_fwhm,
        "noise_sigma_uk_arcmin": noise_sigma,
    }


def parse_explicit_rows(text: str) -> np.ndarray:
    """Parse explicit row indices."""

    rows = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not rows:
        raise ValueError("--rows was provided but no row indices were parsed.")
    if any(row < 0 for row in rows):
        raise ValueError("--rows cannot contain negative indices.")
    return np.asarray(rows, dtype=np.int64)


def load_rows_json(path: Path) -> np.ndarray:
    """Load explicit row indices from a JSON list or manifest-like object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        if "rows" in payload:
            rows = payload["rows"]
        else:
            raise ValueError(f"Row JSON {path} has no top-level `rows` field.")
    else:
        raise ValueError(f"Unsupported JSON payload in {path}; expected list or object.")
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError(f"Row JSON {path} must define a one-dimensional row list.")
    if rows.size == 0:
        raise ValueError(f"Row JSON {path} contains no row indices.")
    if np.any(rows < 0):
        raise ValueError(f"Row JSON {path} contains negative row indices.")
    return np.sort(np.unique(rows))


def load_manifest_rows(path: Path, shard_id: int) -> np.ndarray:
    """Load full-manifest rows or one manifest shard."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(shard_id) < 0:
        rows = manifest.get("rows", [])
    else:
        shards = manifest.get("shards", [])
        if int(shard_id) >= len(shards):
            raise ValueError(
                f"--manifest-shard-id={shard_id} is outside manifest shard range 0..{len(shards) - 1}."
            )
        rows = shards[int(shard_id)].get("rows", [])
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError(f"Manifest {path} produced no valid rows for shard_id={shard_id}.")
    if np.any(rows < 0):
        raise ValueError(f"Manifest {path} contains negative row indices.")
    return np.sort(np.unique(rows))


def select_rows(path: Path, args: argparse.Namespace) -> np.ndarray:
    """Select source rows for a reproducible pilot or production run."""

    with h5py.File(path, "r") as h5:
        labels = np.asarray(h5["labels"][:], dtype=np.uint8)
    if args.rows:
        rows = parse_explicit_rows(args.rows)
        if int(rows.max()) >= labels.size:
            raise ValueError("--rows contains an index beyond the source HDF5 length.")
        return np.sort(np.unique(rows))
    if args.rows_json:
        rows = load_rows_json(Path(args.rows_json).expanduser().resolve())
        if int(rows.max()) >= labels.size:
            raise ValueError("--rows-json contains an index beyond the source HDF5 length.")
        return rows
    if args.manifest_json:
        rows = load_manifest_rows(Path(args.manifest_json).expanduser().resolve(), int(args.manifest_shard_id))
        if int(rows.max()) >= labels.size:
            raise ValueError("--manifest-json contains an index beyond the source HDF5 length.")
        return rows

    if args.row_selection == "all":
        candidates = np.arange(labels.size, dtype=np.int64)
    elif args.row_selection == "positives":
        candidates = np.flatnonzero(labels == 1).astype(np.int64)
    elif args.row_selection == "negatives":
        candidates = np.flatnonzero(labels == 0).astype(np.int64)
    else:
        pos = np.flatnonzero(labels == 1).astype(np.int64)
        neg = np.flatnonzero(labels == 0).astype(np.int64)
        if pos.size == 0 or neg.size == 0:
            raise ValueError("Balanced row selection requires both positive and negative rows.")
        max_rows = int(args.max_rows) if int(args.max_rows) else 2 * min(pos.size, neg.size)
        per_class = max(1, max_rows // 2)
        rng = np.random.default_rng(int(args.seed))
        pos = rng.choice(pos, size=min(per_class, pos.size), replace=False)
        neg = rng.choice(neg, size=min(max_rows - pos.size, neg.size), replace=False)
        candidates = np.concatenate([pos, neg])

    max_rows = int(args.max_rows)
    if max_rows and candidates.size > max_rows:
        rng = np.random.default_rng(int(args.seed))
        candidates = rng.choice(candidates, size=max_rows, replace=False)
    return np.sort(np.asarray(candidates, dtype=np.int64))


def read_selected_rows(path: Path, rows: np.ndarray) -> SourceRows:
    """Read selected source rows into memory."""

    with h5py.File(path, "r") as h5:
        labels = np.asarray(h5["labels"][rows], dtype=np.uint8)
        metadata = h5["metadata"]
        truth = h5["truth"]
        strat = h5["stratification"]

        def optional_metadata(name: str, dtype: Any) -> np.ndarray:
            if name in metadata:
                return np.asarray(metadata[name][rows], dtype=dtype)
            return np.zeros(rows.size, dtype=dtype)

        return SourceRows(
            row_index=np.asarray(rows, dtype=np.int64),
            labels=labels,
            glon_deg=np.asarray(metadata["glon_deg"][rows], dtype=np.float64),
            glat_deg=np.asarray(metadata["glat_deg"][rows], dtype=np.float64),
            cmb_realization_idx=np.asarray(metadata["cmb_realization_idx"][rows], dtype=np.int64),
            coord_cluster_id=optional_metadata("coord_cluster_id", np.uint64),
            background_id=optional_metadata("background_id", np.uint64),
            amplitude=np.asarray(truth["amplitude"][rows], dtype=np.float64),
            theta_crit_deg=np.asarray(truth["theta_crit_deg"][rows], dtype=np.float64),
            z0=np.asarray(truth["z0"][rows], dtype=np.float64),
            zcrit=np.asarray(truth["zcrit"][rows], dtype=np.float64),
            zcrit_ratio=np.asarray(truth["zcrit_ratio"][rows], dtype=np.float64),
            signal_center_x_pix=np.asarray(truth["signal_center_x_pix"][rows], dtype=np.float64),
            signal_center_y_pix=np.asarray(truth["signal_center_y_pix"][rows], dtype=np.float64),
            amplitude_idx=np.asarray(strat["amplitude_idx"][rows], dtype=np.int16),
            theta_idx=np.asarray(strat["theta_idx"][rows], dtype=np.int16),
            zcrit_ratio_idx=np.asarray(strat["zcrit_ratio_idx"][rows], dtype=np.int16),
            sign_quadrant=np.asarray(strat["sign_quadrant"][rows], dtype=np.int16),
            masks=np.asarray(h5["masks"][rows], dtype=np.uint8),
        )


class CmbMapCache:
    """Small LRU cache for reconstructed raw CMB maps."""

    def __init__(self, seeds: list[int], *, nside: int, lmax: int, cls_tt: np.ndarray, max_items: int) -> None:
        self.seeds = [int(seed) for seed in seeds]
        self.nside = int(nside)
        self.lmax = int(lmax)
        self.cls_tt = np.asarray(cls_tt, dtype=FLOAT_GEOMETRY_DTYPE)
        self.max_items = int(max_items)
        self.cache: collections.OrderedDict[int, np.ndarray] = collections.OrderedDict()

    def get(self, realization_idx: int) -> np.ndarray:
        """Return a raw unconvolved CMB map for a source realization index."""

        realization_idx = int(realization_idx)
        if realization_idx < 0 or realization_idx >= len(self.seeds):
            raise ValueError(f"cmb_realization_idx={realization_idx} is outside seed provenance.")
        if realization_idx in self.cache:
            value = self.cache.pop(realization_idx)
            self.cache[realization_idx] = value
            return value
        with temporary_numpy_seed(self.seeds[realization_idx]):
            raw_map = hp.synfast(
                self.cls_tt,
                self.nside,
                lmax=self.lmax,
                new=True,
                pixwin=False,
                fwhm=0.0,
            )
        raw_map = np.asarray(raw_map, dtype=FLOAT_GEOMETRY_DTYPE)
        if not np.all(np.isfinite(raw_map)):
            raise ValueError("Reconstructed CMB map contains non-finite values.")
        self.cache[realization_idx] = raw_map
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return raw_map


def apply_observing_transfer(
    hp_map: np.ndarray,
    *,
    nside: int,
    lmax: int,
    beam_fwhm_arcmin: float,
    pixel_window_policy: str,
) -> np.ndarray:
    """Apply the configured beam and pixel window to a full-sky map."""

    alm = hp.map2alm(np.asarray(hp_map, dtype=FLOAT_GEOMETRY_DTYPE), lmax=int(lmax), iter=0)
    transfer = effective_beam_l(
        nside=int(nside),
        lmax=int(lmax),
        beam_fwhm_arcmin=float(beam_fwhm_arcmin),
        pixel_window_policy=str(pixel_window_policy),
    )
    observed = hp.alm2map(hp.almxfl(alm, transfer), int(nside), lmax=int(lmax))
    observed = np.asarray(observed, dtype=FLOAT_GEOMETRY_DTYPE)
    if not np.all(np.isfinite(observed)):
        raise ValueError("Observed map contains non-finite values after transfer.")
    return observed


def fullsky_noise_sigma_k_per_pixel(noise_sigma_uk_arcmin: float, nside: int) -> float:
    """Convert white-noise depth in uK arcmin to HEALPix pixel RMS in Kelvin."""

    if float(noise_sigma_uk_arcmin) <= 0.0:
        return 0.0
    omega_pix_sr = hp.nside2pixarea(int(nside))
    sigma_k_rad = (float(noise_sigma_uk_arcmin) * 1.0e-6 * u.K * u.arcmin).to(u.K * u.rad).value
    sigma_k = sigma_k_rad / float(np.sqrt(omega_pix_sr))
    if not np.isfinite(sigma_k) or sigma_k < 0.0:
        raise ValueError("Computed non-physical full-sky noise pixel RMS.")
    return float(sigma_k)


def row_noise_rng(seed: int, row_index: int) -> np.random.Generator:
    """Return deterministic row-level noise RNG."""

    mixed = (int(seed) * 6364136223846793005 + int(row_index) * 1442695040888963407) & ((1 << 63) - 1)
    return np.random.default_rng(mixed)


def candidate_vector(rows: SourceRows, idx: int) -> np.ndarray:
    """Return the sky vector used for candidate-region scoring."""

    if int(rows.labels[idx]) == 1:
        return offset_signal_vector(
            float(rows.glon_deg[idx]),
            float(rows.glat_deg[idx]),
            center_x_pix=float((PATCH_PIX - 1) - float(rows.signal_center_x_pix[idx])),
            center_y_pix=float(rows.signal_center_y_pix[idx]),
        )
    theta = np.radians(90.0 - float(rows.glat_deg[idx]))
    phi = np.radians(float(rows.glon_deg[idx]))
    return np.asarray(hp.ang2vec(theta, phi), dtype=FLOAT_GEOMETRY_DTYPE)


def score_region_pixels(nside: int, vec: np.ndarray, score_region: str) -> np.ndarray:
    """Return pixels used to reduce full-sky score maps to one row score."""

    if score_region == "candidate_pixel":
        return np.asarray([hp.vec2pix(int(nside), *vec)], dtype=np.int64)
    radius = (patch_half_width_deg() * u.deg).to_value(u.rad)
    pix = hp.query_disc(int(nside), np.asarray(vec, dtype=FLOAT_GEOMETRY_DTYPE), radius, inclusive=True)
    if pix.size == 0:
        raise ValueError("Candidate score region contains no HEALPix pixels.")
    return np.asarray(pix, dtype=np.int64)


def reduce_wiener_score(score_maps: dict[str, np.ndarray], pix: np.ndarray) -> float:
    """Return blind local maximum over Wiener template score maps."""

    best = -np.inf
    for score_map in score_maps.values():
        best = max(best, float(np.max(np.asarray(score_map)[pix])))
    return float(best)


def reduce_smhw_score(score_maps: dict[str, np.ndarray], pix: np.ndarray) -> float:
    """Return blind local maximum absolute SMHW response."""

    best = -np.inf
    for score_map in score_maps.values():
        best = max(best, float(np.max(np.abs(np.asarray(score_map)[pix]))))
    return float(best)


def build_row_map(
    *,
    rows: SourceRows,
    idx: int,
    cache: CmbMapCache,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> np.ndarray:
    """Build one observed full-sky benchmark map."""

    raw_map = cache.get(int(rows.cmb_realization_idx[idx])).copy()
    if int(rows.labels[idx]) == 1:
        raw_map = inject_signal_on_sphere(
            raw_map,
            glon_deg=float(rows.glon_deg[idx]),
            glat_deg=float(rows.glat_deg[idx]),
            z0=float(rows.z0[idx]),
            zcrit=float(rows.zcrit[idx]),
            theta_crit_deg=float(rows.theta_crit_deg[idx]),
            edge_sigma_deg=0.0,
            center_x_pix=float((PATCH_PIX - 1) - float(rows.signal_center_x_pix[idx])),
            center_y_pix=float(rows.signal_center_y_pix[idx]),
            injection_convention=str(args.injection_convention),
        ).astype(FLOAT_GEOMETRY_DTYPE)
    observed = apply_observing_transfer(
        raw_map,
        nside=int(config["nside"]),
        lmax=int(config["lmax"]),
        beam_fwhm_arcmin=float(config["beam_fwhm_arcmin"]),
        pixel_window_policy=str(args.pixel_window_policy),
    )
    sigma_k = fullsky_noise_sigma_k_per_pixel(float(config["noise_sigma_uk_arcmin"]), int(config["nside"]))
    if sigma_k > 0.0:
        rng = row_noise_rng(int(args.seed), int(rows.row_index[idx]))
        observed = observed + rng.normal(0.0, sigma_k, size=observed.shape)
    return np.asarray(observed, dtype=FLOAT_STORAGE_DTYPE)


def write_patch_h5(
    path: Path,
    *,
    patches: np.ndarray,
    rows: SourceRows,
    summary: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Write patches projected from the full-sky benchmark maps."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("patches", data=patches.astype(np.float32), compression="gzip", shuffle=True)
        h5.create_dataset("labels", data=rows.labels.astype(np.uint8), compression="gzip", shuffle=True)
        h5.create_dataset("masks", data=rows.masks.astype(np.uint8), compression="gzip", shuffle=True)

        metadata = h5.create_group("metadata")
        metadata.create_dataset("source_row_index", data=rows.row_index, compression="gzip", shuffle=True)
        metadata.create_dataset("glon_deg", data=rows.glon_deg.astype(np.float32), compression="gzip", shuffle=True)
        metadata.create_dataset("glat_deg", data=rows.glat_deg.astype(np.float32), compression="gzip", shuffle=True)
        metadata.create_dataset("cmb_realization_idx", data=rows.cmb_realization_idx, compression="gzip", shuffle=True)
        metadata.create_dataset("coord_cluster_id", data=rows.coord_cluster_id, compression="gzip", shuffle=True)
        metadata.create_dataset("background_id", data=rows.background_id, compression="gzip", shuffle=True)

        truth = h5.create_group("truth")
        truth.create_dataset("has_signal", data=rows.labels.astype(np.uint8), compression="gzip", shuffle=True)
        truth.create_dataset("amplitude", data=rows.amplitude.astype(np.float32), compression="gzip", shuffle=True)
        truth.create_dataset("theta_crit_deg", data=rows.theta_crit_deg.astype(np.float32), compression="gzip", shuffle=True)
        truth.create_dataset("z0", data=rows.z0.astype(np.float32), compression="gzip", shuffle=True)
        truth.create_dataset("zcrit", data=rows.zcrit.astype(np.float32), compression="gzip", shuffle=True)
        truth.create_dataset("zcrit_ratio", data=rows.zcrit_ratio.astype(np.float32), compression="gzip", shuffle=True)
        truth.create_dataset(
            "signal_center_x_pix",
            data=rows.signal_center_x_pix.astype(np.float32),
            compression="gzip",
            shuffle=True,
        )
        truth.create_dataset(
            "signal_center_y_pix",
            data=rows.signal_center_y_pix.astype(np.float32),
            compression="gzip",
            shuffle=True,
        )

        strat = h5.create_group("stratification")
        strat.create_dataset("amplitude_idx", data=rows.amplitude_idx, compression="gzip", shuffle=True)
        strat.create_dataset("theta_idx", data=rows.theta_idx, compression="gzip", shuffle=True)
        strat.create_dataset("zcrit_ratio_idx", data=rows.zcrit_ratio_idx, compression="gzip", shuffle=True)
        strat.create_dataset("sign_quadrant", data=rows.sign_quadrant, compression="gzip", shuffle=True)

        out_summary = h5.create_group("summary")
        attrs = {
            "num_samples": int(rows.labels.size),
            "num_positive": int(np.count_nonzero(rows.labels == 1)),
            "num_negative": int(np.count_nonzero(rows.labels == 0)),
            "source_h5": str(Path(args.source_h5).resolve()),
            "source_row_selection": str(args.row_selection),
            "source_max_rows": int(args.max_rows),
            "nside": int(config["nside"]),
            "source_nside": int(config["source_nside"]),
            "lmax": int(config["lmax"]),
            "patch_pixels": int(PATCH_PIX),
            "reso_arcmin": float(RESO_ARCMIN.to_value(u.arcmin)),
            "beam_fwhm_arcmin": float(config["beam_fwhm_arcmin"]),
            "noise_sigma_uk_arcmin": float(config["noise_sigma_uk_arcmin"]),
            "pixel_window_policy": str(args.pixel_window_policy),
            "injection_convention": str(args.injection_convention),
            "injection_convention_note": INJECTION_CONVENTION_NOTES[str(args.injection_convention)],
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            "amplitude_grid": json.dumps(summary.get("amplitude_grid", [])),
            "theta_grid_deg": json.dumps(summary.get("theta_grid_deg", [])),
            "zcrit_ratio_grid": json.dumps(summary.get("zcrit_ratio_grid", [])),
            "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        }
        for key, value in attrs.items():
            out_summary.attrs[key] = value


def score_ml_models(
    *,
    h5_path: Path,
    rows: SourceRows,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Score optional ML models on the generated same-grid patches."""

    if args.skip_ml:
        return {}, {"skipped": True}
    specs = [parse_model_spec(text) for text in (args.model or DEFAULT_MODELS)]
    device = p3.resolve_device(args.device)
    scores: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"device": str(device), "models": {}}
    indices = np.arange(rows.labels.size, dtype=np.int64)
    for spec in specs:
        model, run_config, checkpoint_path, checkpoint_label = build_model_from_run(
            spec.run_dir.resolve(),
            spec.checkpoint,
            device,
        )
        dataset = p3.H5BubbleDataset(
            h5_path=str(h5_path),
            indices=indices,
            **p3.dataset_kwargs_from_run_config(run_config),
            augment=False,
            seed=int(run_config["args"]["seed"]) + 909,
            max_translate_pixels=0,
            cache_data=bool(args.cache_ml_data),
        )
        effective_workers = 0 if args.cache_ml_data else int(args.num_workers)
        loader = p3.DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=effective_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=effective_workers > 0,
        )
        method_scores = np.zeros(rows.labels.size, dtype=np.float32)
        offset = 0
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                mask_logits, _ = p3.unpack_model_output(model(images))
                score = torch.sigmoid(mask_logits).flatten(1).max(dim=1).values
                batch_size = int(images.shape[0])
                method_scores[offset : offset + batch_size] = score.detach().cpu().numpy()
                offset += batch_size
        scores[spec.name] = method_scores
        metadata["models"][spec.name] = {
            "checkpoint_path": checkpoint_path,
            "checkpoint_label": checkpoint_label,
            "score": "segmentation_max",
        }
    return scores, metadata


def summarize_scores(scores_by_method: dict[str, np.ndarray], rows: SourceRows, fpr_target: float) -> tuple[list[dict], dict]:
    """Summarize detection probabilities by amplitude/theta cell."""

    labels = np.asarray(rows.labels, dtype=np.uint8)
    amplitude_values = sorted(float(value) for value in np.unique(rows.amplitude[labels == 1]))
    theta_values = sorted(float(value) for value in np.unique(rows.theta_crit_deg[labels == 1]))
    report_rows = []
    thresholds = {}
    for method, scores in scores_by_method.items():
        if np.count_nonzero(labels == 0) == 0:
            thresholds[method] = {"threshold": None, "negative_fp": None, "negative_fpr": None}
            continue
        threshold, neg_fp, neg_fpr = threshold_from_negatives(scores, labels, fpr_target)
        thresholds[method] = {"threshold": threshold, "negative_fp": neg_fp, "negative_fpr": neg_fpr}
        for amp in amplitude_values:
            for theta in theta_values:
                mask = (labels == 1) & np.isclose(rows.amplitude, amp) & np.isclose(rows.theta_crit_deg, theta)
                n = int(mask.sum())
                if n == 0:
                    continue
                k = int(np.sum(np.asarray(scores)[mask] > threshold))
                low, high = binomial_ci(k, n)
                report_rows.append(
                    {
                        "method": method,
                        "amplitude": amp,
                        "theta_crit_deg": theta,
                        "num_positive": n,
                        "detected": k,
                        "p_det": float(k / max(n, 1)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "threshold": threshold,
                        "negative_fp": neg_fp,
                        "negative_fpr": neg_fpr,
                    }
                )
    return report_rows, thresholds


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a compact benchmark report."""

    lines = ["# Same-Grid Full-Sky Benchmark", ""]
    lines.append(f"Status: `{report['benchmark_status']}`")
    lines.append("")
    lines.append("## Assumptions")
    lines.append("")
    for item in report["assumption_warnings"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    for key in (
        "num_rows",
        "num_positive",
        "num_negative",
        "nside",
        "lmax",
        "injection_convention",
        "pixel_window_policy",
        "score_region",
    ):
        lines.append(f"- {key}: `{report.get(key, 'n/a')}`")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    if report["thresholds"]:
        lines.append("| method | threshold | negative FP | realized FPR |")
        lines.append("|---|---:|---:|---:|")
        for method, row in report["thresholds"].items():
            if row["threshold"] is None:
                lines.append(f"| {method} | n/a | n/a | n/a |")
            else:
                lines.append(
                    f"| {method} | {row['threshold']:.8g} | "
                    f"{row['negative_fp']} | {row['negative_fpr']:.4f} |"
                )
    else:
        lines.append("- no score thresholds computed")
    lines.append("")
    lines.append("## Cell Recall")
    lines.append("")
    if report["rows"]:
        lines.append("| method | A | theta_deg | detected / n | P_det | 95% CI |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in report["rows"]:
            lines.append(
                f"| {row['method']} | {row['amplitude']:.3g} | {row['theta_crit_deg']:.1f} | "
                f"{row['detected']} / {row['num_positive']} | {row['p_det']:.3f} | "
                f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] |"
            )
    else:
        lines.append("- no positive recall rows computed")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    source_h5 = Path(args.source_h5).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_source_summary(source_h5)
    config = validate_args(args, summary)
    selected = select_rows(source_h5, args)
    rows = read_selected_rows(source_h5, selected)

    camb_params = summary.get("camb_params", {})
    seeds = camb_params.get("seeds", [])
    camb_meta = camb_params.get("camb", {})
    camb_param_values = camb_meta.get("params")
    lens_accuracy = int(camb_meta.get("lens_potential_accuracy", 1))
    cls_tt, camb_provenance = camb_tt_cls(
        lmax=int(config["lmax"]),
        params=camb_param_values,
        lens_potential_accuracy=lens_accuracy,
    )
    cache = CmbMapCache(
        seeds,
        nside=int(config["nside"]),
        lmax=int(config["lmax"]),
        cls_tt=cls_tt,
        max_items=int(args.cmb_realization_cache),
    )
    wiener_bank = wiener_bank_meta = smhw_bank = smhw_bank_meta = None
    if not args.skip_classical:
        wiener_bank, wiener_bank_meta = precompute_wiener_feeney_filter_bank(
            nside=int(config["nside"]),
            cmb_cl=cls_tt,
            theta_grid_deg=tuple(config["theta_grid_deg"]),
            lmax=int(config["lmax"]),
            beam_fwhm_arcmin=float(config["beam_fwhm_arcmin"]),
            noise_sigma_uk_arcmin=float(config["noise_sigma_uk_arcmin"]),
            pixel_window_policy=str(args.pixel_window_policy),
            quadrature_order=int(args.quadrature_order),
            collapse_sign_pairs=True,
        )
        smhw_bank, smhw_bank_meta = precompute_smhw_filter_bank(
            scales_deg=tuple(config["smhw_scales_deg"]),
            lmax=int(config["lmax"]),
        )

    patches = np.empty((rows.labels.size, PATCH_PIX, PATCH_PIX), dtype=np.float32)
    classical_scores = {
        WIENER_FEENEY_MATCHED_FILTER: np.zeros(rows.labels.size, dtype=np.float32),
        SMHW_SCREEN: np.zeros(rows.labels.size, dtype=np.float32),
    }
    if args.skip_classical:
        classical_scores = {}

    # Process rows grouped by CMB realization so the small raw-map cache has a
    # meaningful hit rate even when the manifest order is effectively random.
    processing_order = np.argsort(rows.cmb_realization_idx.astype(np.int64), kind="stable")
    rows_done = 0
    for idx in processing_order:
        full_map = build_row_map(rows=rows, idx=idx, cache=cache, config=config, args=args)
        patches[idx] = project_patch(full_map, float(rows.glon_deg[idx]), float(rows.glat_deg[idx]))
        if not args.skip_classical:
            vec = candidate_vector(rows, idx)
            pix = score_region_pixels(int(config["nside"]), vec, str(args.score_region))
            data_alm = hp.map2alm(
                validate_cmb_map(full_map),
                lmax=int(config["lmax"]),
                iter=3,
            )
            wiener_maps, _ = apply_precomputed_wiener_bank(
                data_alm,
                nside=int(config["nside"]),
                lmax=int(config["lmax"]),
                weights_bank=wiener_bank,
                metadata=wiener_bank_meta,
            )
            smhw_maps, _ = apply_precomputed_smhw_bank(
                data_alm,
                nside=int(config["nside"]),
                lmax=int(config["lmax"]),
                window_bank=smhw_bank,
                metadata=smhw_bank_meta,
            )
            classical_scores[WIENER_FEENEY_MATCHED_FILTER][idx] = reduce_wiener_score(wiener_maps, pix)
            classical_scores[SMHW_SCREEN][idx] = reduce_smhw_score(smhw_maps, pix)
        rows_done += 1
        if rows_done % 8 == 0 or rows_done == rows.labels.size:
            print(f"  processed {rows_done} / {rows.labels.size} full-sky benchmark rows", flush=True)

    h5_path = output_dir / "same_grid_fullsky_patches.h5"
    write_patch_h5(
        h5_path,
        patches=patches,
        rows=rows,
        summary=summary,
        config=config,
        args=args,
    )
    ml_scores, ml_metadata = score_ml_models(h5_path=h5_path, rows=rows, args=args)
    scores_by_method = {**classical_scores, **ml_scores}
    score_payload = {f"score__{key}": value.astype(np.float32) for key, value in scores_by_method.items()}
    scores_path = output_dir / "same_grid_fullsky_scores.npz"
    np.savez_compressed(scores_path, labels=rows.labels.astype(np.uint8), source_row_index=rows.row_index, **score_payload)

    fpr_target = float(summary.get("fpr_target", 0.05))
    recall_rows, thresholds = summarize_scores(scores_by_method, rows, fpr_target)
    subset_run = bool(int(args.max_rows)) or bool(args.rows) or bool(args.rows_json) or bool(args.manifest_json)
    if args.manifest_json and int(args.manifest_shard_id) >= 0:
        benchmark_status = "manifest_shard"
    elif args.manifest_json:
        benchmark_status = "manifest_run_unmasked_filter"
    else:
        benchmark_status = "pilot_subset" if subset_run else "full_run_unmasked_filter"
    report = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "benchmark_status": benchmark_status,
        "source_h5": str(source_h5),
        "rows_json": str(Path(args.rows_json).expanduser().resolve()) if args.rows_json else "",
        "manifest_json": str(Path(args.manifest_json).expanduser().resolve()) if args.manifest_json else "",
        "manifest_shard_id": int(args.manifest_shard_id),
        "patch_h5": str(h5_path),
        "scores_npz": str(scores_path),
        "num_rows": int(rows.labels.size),
        "num_positive": int(np.count_nonzero(rows.labels == 1)),
        "num_negative": int(np.count_nonzero(rows.labels == 0)),
        "nside": int(config["nside"]),
        "source_nside": int(config["source_nside"]),
        "lmax": int(config["lmax"]),
        "beam_fwhm_arcmin": float(config["beam_fwhm_arcmin"]),
        "noise_sigma_uk_arcmin": float(config["noise_sigma_uk_arcmin"]),
        "pixel_window_policy": str(args.pixel_window_policy),
        "injection_convention": str(args.injection_convention),
        "injection_convention_note": INJECTION_CONVENTION_NOTES[str(args.injection_convention)],
        "score_region": str(args.score_region),
        "theta_grid_deg": list(config["theta_grid_deg"]),
        "smhw_scales_deg": list(config["smhw_scales_deg"]),
        "fpr_target": fpr_target,
        "rows": recall_rows,
        "thresholds": thresholds,
        "method_metadata": {name: method_metadata(name) for name in scores_by_method},
        "ml_metadata": ml_metadata,
        "camb": camb_provenance,
        "assumption_warnings": [
            "Subset/pilot runs do not close the paper-facing same-grid benchmark.",
            "The current classical score is an unmasked full-sky filter with local candidate-region maximization, not OSS masked-sky evidence.",
            "A full_run_unmasked_filter artifact is still not a masked-sky optimality proof; it is the required same-grid Wiener/SMHW comparator for this repository's current classical implementation.",
            "The default additive injection avoids the Feeney cross term; Feeney-modulated runs must cite the phase2 physics-check cross-term audit.",
        ],
    }
    report_path = output_dir / "same_grid_fullsky_report.json"
    md_path = output_dir / "same_grid_fullsky_report.md"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps({"report": str(report_path), "markdown": str(md_path), "scores": str(scores_path)}, indent=2))


if __name__ == "__main__":
    main()
