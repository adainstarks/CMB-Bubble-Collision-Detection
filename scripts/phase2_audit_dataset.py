"""
Audit a Phase 2 HDF5 dataset for scientific and ML shortcut risks.

This is a hard quality gate. It checks the issues that can make Phase 3 metrics
look valid while being scientifically compromised:
    - malformed HDF5 schema
    - train/validation sample overlap
    - coordinate, CMB-realization, background, or event leakage
    - metadata fields that expose injection truth or NaN class shortcuts
    - label/mask/truth inconsistency
    - contained-geometry violations
    - non-finite patches or suspicious normalization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


TRUTH_FIELD_NAMES = {
    "has_signal",
    "is_positive",
    "event_id",
    "theta_crit_deg",
    "z0",
    "zcrit",
    "edge_sigma_deg",
    "signal_center_x_pix",
    "signal_center_y_pix",
    "signal_center_dx_deg",
    "signal_center_dy_deg",
    "geometry_mode_code",
    "fully_contained",
    "target_touches_edge",
    "visible_target_fraction",
    "visible_target_pixels",
    "full_disc_pixels_est",
    "target_edge_contact_pixels",
    "disc_edge_margin_pix",
    "signal_center_in_patch",
}

REQUIRED_ROOT_DATASETS = ("patches", "labels", "masks")
REQUIRED_METADATA_FIELDS = (
    "sample_index",
    "glon_deg",
    "glat_deg",
    "cmb_realization_idx",
)
REQUIRED_TRUTH_FIELDS = (
    "has_signal",
    "theta_crit_deg",
    "z0",
    "zcrit",
    "signal_center_x_pix",
    "signal_center_y_pix",
    "signal_center_dx_deg",
    "signal_center_dy_deg",
    "target_touches_edge",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit a Phase 2 training HDF5 dataset for leakage and scientific consistency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-h5", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Report legacy schema/leakage issues as warnings instead of hard failures where possible.",
    )
    parser.add_argument(
        "--sample-patch-count",
        type=int,
        default=256,
        help="Number of patches to sample for finite-value/statistical checks. Use 0 for all patches.",
    )
    return parser.parse_args()


def as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class Audit:
    def __init__(self, allow_legacy=False):
        self.allow_legacy = allow_legacy
        self.failures = []
        self.warnings = []
        self.metrics = {}

    def fail(self, message):
        if self.allow_legacy:
            self.warnings.append(message)
        else:
            self.failures.append(message)

    def warn(self, message):
        self.warnings.append(message)

    def require(self, condition, message):
        if not condition:
            self.fail(message)

    def add_metric(self, key, value):
        self.metrics[key] = as_jsonable(value)

    def report(self):
        return {
            "status": "pass" if not self.failures else "fail",
            "num_failures": len(self.failures),
            "num_warnings": len(self.warnings),
            "failures": self.failures,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


def read_group_arrays(group):
    return {name: np.asarray(group[name][:]) for name in group.keys()}


def get_summary_attrs(h5):
    if "summary" not in h5:
        return {}
    return {key: as_jsonable(value) for key, value in h5["summary"].attrs.items()}


def get_truth_group(h5):
    if "truth" in h5:
        return h5["truth"], "truth"
    if "metadata" in h5:
        return h5["metadata"], "metadata"
    return None, ""


def finite_fraction(values):
    values = np.asarray(values)
    if values.size == 0:
        return 1.0
    if np.issubdtype(values.dtype, np.number):
        return float(np.isfinite(values).mean())
    return 1.0


def edge_touch_mask(masks):
    masks = np.asarray(masks, dtype=bool)
    return masks[:, 0, :].any(axis=1) | masks[:, -1, :].any(axis=1) | masks[:, :, 0].any(axis=1) | masks[:, :, -1].any(axis=1)


def nonzero_intersection(a, b, ignore_values=()):
    ignore = set(int(x) for x in ignore_values)
    a_set = {int(x) for x in np.asarray(a).ravel() if int(x) not in ignore}
    b_set = {int(x) for x in np.asarray(b).ravel() if int(x) not in ignore}
    return sorted(a_set & b_set)


def audit_schema(h5, audit):
    for name in REQUIRED_ROOT_DATASETS:
        audit.require(name in h5, f"Missing root dataset: {name}")
    audit.require("metadata" in h5, "Missing metadata group")
    audit.require("splits" in h5, "Missing splits group")

    if "metadata" in h5:
        metadata_names = set(h5["metadata"].keys())
        for name in REQUIRED_METADATA_FIELDS:
            audit.require(name in metadata_names, f"Missing metadata field: {name}")

        truth_in_metadata = sorted(metadata_names & (TRUTH_FIELD_NAMES - {"has_signal", "is_positive"}))
        if truth_in_metadata:
            audit.fail(
                "Injection-truth fields are stored in metadata, creating shortcut/leakage risk: "
                + ", ".join(truth_in_metadata)
            )

    truth_group, truth_group_name = get_truth_group(h5)
    audit.add_metric("truth_group", truth_group_name)
    if truth_group is not None:
        truth_names = set(truth_group.keys())
        for name in REQUIRED_TRUTH_FIELDS:
            if name == "has_signal" and "has_signal" not in truth_names and "is_positive" in truth_names:
                continue
            audit.require(name in truth_names, f"Missing truth field: {name}")


def audit_shapes(h5, audit):
    if not all(name in h5 for name in REQUIRED_ROOT_DATASETS):
        return
    patches = h5["patches"]
    labels = h5["labels"]
    masks = h5["masks"]
    n = int(labels.shape[0])

    audit.add_metric("num_samples", n)
    audit.add_metric("patch_shape", list(patches.shape))
    audit.add_metric("mask_shape", list(masks.shape))
    audit.require(len(patches.shape) == 3, "patches must have shape (N, H, W)")
    audit.require(len(masks.shape) == 3, "masks must have shape (N, H, W)")
    audit.require(patches.shape[0] == n, "patches and labels disagree on N")
    audit.require(masks.shape[0] == n, "masks and labels disagree on N")
    audit.require(patches.shape[-2:] == masks.shape[-2:], "patches and masks have different spatial shapes")


def audit_splits(h5, audit):
    if "splits" not in h5 or "labels" not in h5:
        return None, None
    splits = h5["splits"]
    audit.require("train_idx" in splits, "Missing splits/train_idx")
    has_calibration = "calibration_idx" in splits
    audit.require("val_idx" in splits or has_calibration, "Missing splits/val_idx or splits/calibration_idx")
    if "train_idx" not in splits or ("val_idx" not in splits and not has_calibration):
        return None, None

    labels = np.asarray(h5["labels"][:], dtype=np.uint8)
    train_idx = np.asarray(splits["train_idx"][:], dtype=np.int64)
    val_key = "calibration_idx" if has_calibration else "val_idx"
    val_idx = np.asarray(splits[val_key][:], dtype=np.int64)
    test_idx = np.asarray(splits["test_idx"][:], dtype=np.int64) if "test_idx" in splits else None
    n = len(labels)

    audit.require(train_idx.size > 0, "train split is empty")
    audit.require(val_idx.size > 0, "validation split is empty")
    audit.require(np.all((0 <= train_idx) & (train_idx < n)), "train indices out of range")
    audit.require(np.all((0 <= val_idx) & (val_idx < n)), "validation indices out of range")
    overlap = sorted(set(map(int, train_idx)) & set(map(int, val_idx)))
    audit.require(len(overlap) == 0, f"train/calibration sample overlap: {overlap[:10]}")
    if test_idx is not None:
        audit.require(test_idx.size > 0, "test split is empty")
        audit.require(np.all((0 <= test_idx) & (test_idx < n)), "test indices out of range")
        train_test = sorted(set(map(int, train_idx)) & set(map(int, test_idx)))
        cal_test = sorted(set(map(int, val_idx)) & set(map(int, test_idx)))
        audit.require(len(train_test) == 0, f"train/test sample overlap: {train_test[:10]}")
        audit.require(len(cal_test) == 0, f"calibration/test sample overlap: {cal_test[:10]}")

    audit.add_metric("num_train_samples", int(train_idx.size))
    audit.add_metric("num_calibration_samples", int(val_idx.size))
    audit.add_metric("num_val_samples", int(val_idx.size))
    audit.add_metric("num_train_positive", int(labels[train_idx].sum()))
    audit.add_metric("num_calibration_positive", int(labels[val_idx].sum()))
    audit.add_metric("num_val_positive", int(labels[val_idx].sum()))
    audit.add_metric("num_train_negative", int(train_idx.size - labels[train_idx].sum()))
    audit.add_metric("num_calibration_negative", int(val_idx.size - labels[val_idx].sum()))
    audit.add_metric("num_val_negative", int(val_idx.size - labels[val_idx].sum()))
    if test_idx is not None:
        audit.add_metric("num_test_samples", int(test_idx.size))
        audit.add_metric("num_test_positive", int(labels[test_idx].sum()))
        audit.add_metric("num_test_negative", int(test_idx.size - labels[test_idx].sum()))

    return train_idx, val_idx


def audit_split_leakage(h5, audit, train_idx, val_idx):
    if train_idx is None or val_idx is None or "metadata" not in h5:
        return
    metadata = h5["metadata"]

    for field, ignore_values in (
        ("coord_pool_idx", (-1,)),
        ("cmb_realization_idx", (-1,)),
        ("background_id", (0,)),
        ("split_group_id", (0,)),
    ):
        if field not in metadata:
            audit.warn(f"Cannot audit split leakage for missing metadata field: {field}")
            continue
        values = np.asarray(metadata[field][:])
        shared = nonzero_intersection(values[train_idx], values[val_idx], ignore_values=ignore_values)
        audit.add_metric(f"shared_{field}_count", int(len(shared)))
        audit.require(len(shared) == 0, f"Shared {field} values across train/calibration: {shared[:10]}")

    truth_group, _ = get_truth_group(h5)
    if truth_group is not None and "event_id" in truth_group:
        event_id = np.asarray(truth_group["event_id"][:])
        shared_events = nonzero_intersection(event_id[train_idx], event_id[val_idx], ignore_values=(0,))
        audit.add_metric("shared_event_id_count", int(len(shared_events)))
        audit.require(len(shared_events) == 0, f"Shared event_id values across train/calibration: {shared_events[:10]}")


def audit_metadata_shortcuts(h5, audit):
    if "metadata" not in h5 or "labels" not in h5:
        return
    labels = np.asarray(h5["labels"][:], dtype=np.uint8)
    has_both_classes = bool(np.any(labels == 0) and np.any(labels == 1))
    metadata = h5["metadata"]
    for name in metadata.keys():
        values = np.asarray(metadata[name][:])
        audit.add_metric(f"metadata_finite_fraction/{name}", finite_fraction(values))

        if np.issubdtype(values.dtype, np.number):
            nonfinite = ~np.isfinite(values.astype(np.float64, copy=False))
            if has_both_classes:
                if np.array_equal(nonfinite.astype(np.uint8), labels):
                    audit.fail(f"Metadata field has non-finite pattern that exactly predicts positives: {name}")
                if np.array_equal(nonfinite.astype(np.uint8), 1 - labels):
                    audit.fail(f"Metadata field has non-finite pattern that exactly predicts negatives: {name}")

            pos_values = values[labels == 1]
            neg_values = values[labels == 0]
            if pos_values.size and neg_values.size:
                pos_finite = pos_values[np.isfinite(pos_values.astype(np.float64, copy=False))]
                neg_finite = neg_values[np.isfinite(neg_values.astype(np.float64, copy=False))]
                if pos_finite.size and neg_finite.size:
                    pos_constant = np.all(pos_finite == pos_finite[0])
                    neg_constant = np.all(neg_finite == neg_finite[0])
                    if pos_constant and neg_constant and pos_finite[0] != neg_finite[0]:
                        audit.fail(f"Metadata field is class-constant with different values: {name}")


def audit_truth_and_masks(h5, audit):
    if "labels" not in h5 or "masks" not in h5:
        return
    labels = np.asarray(h5["labels"][:], dtype=np.uint8)
    masks = np.asarray(h5["masks"][:], dtype=np.uint8)
    mask_has_pixels = masks.reshape(masks.shape[0], -1).sum(axis=1) > 0

    audit.require(np.array_equal(mask_has_pixels.astype(np.uint8), labels), "labels do not match non-empty masks")

    truth_group, _ = get_truth_group(h5)
    if truth_group is not None:
        truth_names = set(truth_group.keys())
        has_signal_name = "has_signal" if "has_signal" in truth_names else "is_positive" if "is_positive" in truth_names else None
        if has_signal_name:
            has_signal = np.asarray(truth_group[has_signal_name][:], dtype=np.uint8)
            audit.require(np.array_equal(has_signal, labels), f"truth/{has_signal_name} does not match labels")

        for name in truth_group.keys():
            values = np.asarray(truth_group[name][:])
            if np.issubdtype(values.dtype, np.number):
                finite = np.isfinite(values.astype(np.float64, copy=False))
                if name in {"theta_crit_deg", "z0", "zcrit", "edge_sigma_deg", "signal_center_x_pix", "signal_center_y_pix"}:
                    audit.require(bool(finite[labels == 1].all()), f"truth/{name} has non-finite positive entries")

        target_touches = edge_touch_mask(masks)
        audit.add_metric("positive_targets_touching_edge", int(target_touches[labels == 1].sum()))
        if "target_touches_edge" in truth_group:
            truth_touches = np.asarray(truth_group["target_touches_edge"][:], dtype=bool)
            audit.require(np.array_equal(truth_touches, target_touches), "truth/target_touches_edge disagrees with masks")

        if "visible_target_fraction" in truth_group:
            visible_fraction = np.asarray(truth_group["visible_target_fraction"][:], dtype=np.float64)
            positive_visible = visible_fraction[labels == 1]
            negative_visible = visible_fraction[labels == 0]
            audit.require(
                bool(np.all(np.isfinite(positive_visible))),
                "truth/visible_target_fraction has non-finite positive entries",
            )
            audit.require(
                bool(np.all((0.0 < positive_visible) & (positive_visible <= 1.05))),
                "positive truth/visible_target_fraction values must be in (0, 1]",
            )
            if negative_visible.size:
                audit.require(
                    bool(np.all(negative_visible == 0.0)),
                    "negative truth/visible_target_fraction values must be zero",
                )
            audit.add_metric("positive_visible_target_fraction_mean", float(np.mean(positive_visible)))
            audit.add_metric("positive_visible_target_fraction_min", float(np.min(positive_visible)))
            audit.add_metric("positive_visible_target_fraction_max", float(np.max(positive_visible)))

        if "fully_contained" in truth_group:
            fully_contained = np.asarray(truth_group["fully_contained"][:], dtype=bool)
            audit.require(
                bool(np.all(fully_contained[labels == 1] == ~target_touches[labels == 1])),
                "truth/fully_contained disagrees with target edge contact for positives",
            )

        summary = get_summary_attrs(h5)
        geometry_mode = str(summary.get("geometry_mode", ""))
        audit.add_metric("geometry_mode", geometry_mode)
        if geometry_mode == "contained":
            audit.require(
                int(target_touches[labels == 1].sum()) == 0,
                "contained geometry has positive targets touching patch edge",
            )
        if geometry_mode == "truncated":
            audit.require(
                int(target_touches[labels == 1].sum()) == int(labels.sum()),
                "truncated geometry has positive targets that do not touch patch edge",
            )


def audit_patch_values(h5, audit, sample_patch_count):
    if "patches" not in h5 or "labels" not in h5:
        return
    patches = h5["patches"]
    labels = np.asarray(h5["labels"][:], dtype=np.uint8)
    n = len(labels)
    if sample_patch_count <= 0 or sample_patch_count >= n:
        idx = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(12345)
        idx = np.sort(rng.choice(n, size=sample_patch_count, replace=False))

    sample = np.asarray(patches[idx], dtype=np.float32)
    finite = np.isfinite(sample)
    audit.add_metric("patch_sample_count_checked", int(len(idx)))
    audit.add_metric("patch_finite_fraction", float(finite.mean()))
    audit.require(bool(finite.all()), "sampled patches contain non-finite values")
    audit.add_metric("patch_mean_k", float(np.mean(sample)))
    audit.add_metric("patch_std_k", float(np.std(sample)))
    audit.require(float(np.std(sample)) > 0.0, "sampled patches have zero standard deviation")


def audit_coordinate_pool(h5, audit):
    if "coordinate_pool" not in h5:
        audit.warn("Missing coordinate_pool group")
        return
    pool = h5["coordinate_pool"]
    for name in ("glon_deg", "glat_deg", "mask_fraction"):
        audit.require(name in pool, f"coordinate_pool missing {name}")
    if all(name in pool for name in ("glon_deg", "glat_deg", "mask_fraction")):
        glon = np.asarray(pool["glon_deg"][:], dtype=np.float64)
        glat = np.asarray(pool["glat_deg"][:], dtype=np.float64)
        mask_fraction = np.asarray(pool["mask_fraction"][:], dtype=np.float64)
        audit.add_metric("coordinate_pool_size", int(len(glon)))
        audit.add_metric("coordinate_pool_glon_minmax", [float(np.min(glon)), float(np.max(glon))])
        audit.add_metric("coordinate_pool_glat_minmax", [float(np.min(glat)), float(np.max(glat))])
        audit.add_metric("coordinate_pool_mask_fraction_min", float(np.min(mask_fraction)))
        audit.require(bool(np.all((glon >= 0.0) & (glon <= 360.0))), "coordinate_pool longitude out of range")
        audit.require(bool(np.all((glat >= -90.0) & (glat <= 90.0))), "coordinate_pool latitude out of range")
        summary = get_summary_attrs(h5)
        threshold = float(summary.get("mask_threshold", 0.95))
        audit.require(
            bool(np.all(mask_fraction + 1e-8 >= threshold)),
            f"coordinate_pool contains mask_fraction below configured threshold {threshold:.2f}",
        )


def run_audit(data_h5, allow_legacy=False, sample_patch_count=256):
    audit = Audit(allow_legacy=allow_legacy)
    data_h5 = Path(data_h5).resolve()
    audit.add_metric("data_h5", str(data_h5))
    if not data_h5.exists():
        audit.fail(f"Dataset does not exist: {data_h5}")
        return audit.report()

    with h5py.File(data_h5, "r") as h5:
        audit_schema(h5, audit)
        audit_shapes(h5, audit)
        train_idx, val_idx = audit_splits(h5, audit)
        audit_split_leakage(h5, audit, train_idx, val_idx)
        audit_metadata_shortcuts(h5, audit)
        audit_truth_and_masks(h5, audit)
        audit_patch_values(h5, audit, sample_patch_count)
        audit_coordinate_pool(h5, audit)
        audit.add_metric("summary", get_summary_attrs(h5))

    return audit.report()


def main():
    args = parse_args()
    report = run_audit(
        data_h5=args.data_h5,
        allow_legacy=args.allow_legacy,
        sample_patch_count=args.sample_patch_count,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
