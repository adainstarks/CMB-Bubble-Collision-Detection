"""
Batch 3 geometry router: evaluate two-model portfolio policies on cached
Batch 2 transform scores.

This script consumes the npz caches produced by
`scripts/phase3_postprocess_ablation.py` and evaluates multiple portfolio
policies without requiring any forward pass:

    v6_only        baseline v6 score, threshold calibrated per-model on null
    v7_only        baseline v7 score, threshold calibrated per-model on null
    either_OR      either model above its own threshold at joint FPR target
    both_AND       both models above their own thresholds at joint FPR target
    rank_max       percentile-rank vs null, take max, threshold at FPR target
    geometry_routed use v6 score when v7_mf_on_mask - v7_baseline > tau,
                    else use v7 score; compute rank-max on the selected
                    per-patch score

Reports recall at matched real-SMICA null FPR 0.05 / 0.08 / 0.10 per policy,
with per-geometry group breakdowns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH2 = PROJECT_ROOT / "runs" / "phase3_unet" / "batch2_postprocess_ablation_v1"
DEFAULT_GATE_ROOT = PROJECT_ROOT / "runs" / "phase3_unet" / "real_sky_v7_gate_v1"
FPR_TARGETS = (0.05, 0.08, 0.10)
POLICIES = ("v6_only", "v7_only", "either_OR", "both_AND", "rank_max", "geometry_routed", "learned_logistic", "learned_gbt")

FEATURE_NAMES = ("v6_baseline", "v7_baseline", "v6_mf_on_mask", "v7_mf_on_mask", "v6_smooth_multi", "v7_smooth_multi")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch 3 geometry-aware two-model portfolio router")
    parser.add_argument("--batch2-dir", type=str, default=str(DEFAULT_BATCH2))
    parser.add_argument("--gate-root", type=str, default=str(DEFAULT_GATE_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "runs" / "phase3_unet" / "batch3_geometry_router_v1"))
    parser.add_argument(
        "--route-quantile",
        type=float,
        default=0.5,
        help="Null quantile of (v7_mf_on_mask - v7_baseline) above which we route a candidate to v6. 0.5 is median.",
    )
    parser.add_argument(
        "--learned-fit-on",
        type=str,
        default="cross",
        choices=("cross", "in_sample", "kfold"),
        help="How to train the learned router. cross: fit on the other geometry; in_sample: fit and eval on same data (optimistic upper bound); kfold: 5-fold within the eval geometry.",
    )
    parser.add_argument("--learned-seed", type=int, default=20260417)
    return parser.parse_args()


def exact_ci(k, n):
    if n <= 0:
        return [float("nan"), float("nan")]
    r = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return [float(r.low), float(r.high)]


def load_transform_npz(path):
    with np.load(path) as loaded:
        return {k: np.asarray(loaded[k]) for k in loaded.files}


def load_truth(inj_h5):
    with h5py.File(inj_h5, "r") as h5:
        return {
            "fully_contained": np.asarray(h5["truth"]["fully_contained"][:], dtype=np.uint8),
            "signal_center_in_patch": np.asarray(h5["truth"]["signal_center_in_patch"][:], dtype=np.uint8),
            "visible_target_fraction": np.asarray(h5["truth"]["visible_target_fraction"][:], dtype=np.float32),
            "labels": np.asarray(h5["labels"][:], dtype=np.uint8),
        }


def threshold_at_fpr(null_scores, target_fpr):
    null_sorted = np.sort(np.asarray(null_scores, dtype=np.float64))
    n = null_sorted.size
    if n == 0:
        return float("inf"), 0.0
    k_allowed = int(np.floor(target_fpr * n))
    if k_allowed <= 0:
        threshold = float(np.nextafter(null_sorted[-1], np.inf))
    else:
        threshold = float(null_sorted[-k_allowed])
    actual_fpr = float((null_sorted >= threshold).sum()) / float(n)
    return threshold, actual_fpr


def empirical_rank(values, reference):
    """Return percentile rank of each `values` entry against `reference`, in (0, 1]."""
    ref_sorted = np.sort(np.asarray(reference, dtype=np.float64))
    rank = np.searchsorted(ref_sorted, np.asarray(values, dtype=np.float64), side="right").astype(np.float64)
    return rank / float(max(ref_sorted.size, 1))


def group_breakdown(pos_mask, triggered, truth):
    fully_contained = truth["fully_contained"][pos_mask].astype(bool)
    center_in_patch = truth["signal_center_in_patch"][pos_mask].astype(bool)
    visible = truth["visible_target_fraction"][pos_mask]
    detected = triggered[pos_mask]
    total = int(pos_mask.sum())

    def add(groups, name, mask):
        n = int(mask.sum())
        if n == 0:
            return
        hits = int(detected[mask].sum())
        groups[name] = {"n": n, "detected": hits, "recall": hits / float(n), "recall_ci95": exact_ci(hits, n)}

    groups = {}
    add(groups, "all_positive", np.ones(total, dtype=bool))
    add(groups, "geometry_contained", fully_contained)
    add(groups, "geometry_truncated", ~fully_contained)
    add(groups, "center_inside_patch", center_in_patch)
    add(groups, "center_outside_patch", ~center_in_patch)
    add(groups, "visible_fraction_low", (visible > 0) & (visible < 0.35))
    add(groups, "visible_fraction_mid", (visible >= 0.35) & (visible < 0.70))
    add(groups, "visible_fraction_high", visible >= 0.70)
    return groups


def run_single_model_policy(name, inj_scores, null_scores, labels, truth, fpr_targets):
    rows = []
    for fpr_target in fpr_targets:
        threshold, actual_fpr = threshold_at_fpr(null_scores, fpr_target)
        triggered = inj_scores >= threshold
        pos_mask = labels == 1
        hits = int(triggered[pos_mask].sum())
        total = int(pos_mask.sum())
        rows.append({
            "policy": name,
            "fpr_target": float(fpr_target),
            "threshold_v6": float(threshold) if name == "v6_only" else None,
            "threshold_v7": float(threshold) if name == "v7_only" else None,
            "actual_null_fpr": float(actual_fpr),
            "recall_global": hits / float(max(total, 1)),
            "recall_ci95": exact_ci(hits, total),
            "groups": group_breakdown(pos_mask, triggered, truth),
        })
    return rows


def run_pairwise_policy(name, v6_inj, v6_null, v7_inj, v7_null, combine, labels, truth, fpr_targets):
    rows = []
    for fpr_target in fpr_targets:
        t_v6, _ = threshold_at_fpr(v6_null, fpr_target)
        t_v7, _ = threshold_at_fpr(v7_null, fpr_target)
        joint_triggered_null = combine(v6_null >= t_v6, v7_null >= t_v7)
        actual_fpr = float(joint_triggered_null.mean())
        if actual_fpr > fpr_target + 1e-9:
            # Tighten each model's threshold until joint FPR is within target.
            # Simple binary search over a shared scaling of per-model FPR.
            lo, hi = 0.0, fpr_target
            for _ in range(30):
                fpr_try = (lo + hi) / 2.0
                tv6, _ = threshold_at_fpr(v6_null, fpr_try)
                tv7, _ = threshold_at_fpr(v7_null, fpr_try)
                joint_null = combine(v6_null >= tv6, v7_null >= tv7)
                if joint_null.mean() > fpr_target:
                    hi = fpr_try
                else:
                    lo = fpr_try
                    t_v6, t_v7 = tv6, tv7
            actual_fpr = float(combine(v6_null >= t_v6, v7_null >= t_v7).mean())
        triggered = combine(v6_inj >= t_v6, v7_inj >= t_v7)
        pos_mask = labels == 1
        hits = int(triggered[pos_mask].sum())
        total = int(pos_mask.sum())
        rows.append({
            "policy": name,
            "fpr_target": float(fpr_target),
            "threshold_v6": float(t_v6),
            "threshold_v7": float(t_v7),
            "actual_null_fpr": float(actual_fpr),
            "recall_global": hits / float(max(total, 1)),
            "recall_ci95": exact_ci(hits, total),
            "groups": group_breakdown(pos_mask, triggered, truth),
        })
    return rows


def run_rank_max_policy(name, v6_inj, v6_null, v7_inj, v7_null, labels, truth, fpr_targets):
    """Rank each model against its null, take max rank per patch, threshold on joint rank null."""
    inj_rank_v6 = empirical_rank(v6_inj, v6_null)
    null_rank_v6 = empirical_rank(v6_null, v6_null)
    inj_rank_v7 = empirical_rank(v7_inj, v7_null)
    null_rank_v7 = empirical_rank(v7_null, v7_null)
    inj_score = np.maximum(inj_rank_v6, inj_rank_v7)
    null_score = np.maximum(null_rank_v6, null_rank_v7)
    rows = []
    for fpr_target in fpr_targets:
        threshold, actual_fpr = threshold_at_fpr(null_score, fpr_target)
        triggered = inj_score >= threshold
        pos_mask = labels == 1
        hits = int(triggered[pos_mask].sum())
        total = int(pos_mask.sum())
        rows.append({
            "policy": name,
            "fpr_target": float(fpr_target),
            "threshold_rank": float(threshold),
            "actual_null_fpr": float(actual_fpr),
            "recall_global": hits / float(max(total, 1)),
            "recall_ci95": exact_ci(hits, total),
            "groups": group_breakdown(pos_mask, triggered, truth),
        })
    return rows


def stack_features(v6_npz, v7_npz):
    """Stack per-patch features from transform caches into an (N, 6) matrix."""
    return np.stack([
        np.asarray(v6_npz["baseline"], dtype=np.float64),
        np.asarray(v7_npz["baseline"], dtype=np.float64),
        np.asarray(v6_npz["mf_on_mask"], dtype=np.float64),
        np.asarray(v7_npz["mf_on_mask"], dtype=np.float64),
        np.asarray(v6_npz["smooth_multi"], dtype=np.float64),
        np.asarray(v7_npz["smooth_multi"], dtype=np.float64),
    ], axis=1)


def run_learned_policy(name, model_cls, X_train_pos, X_train_null, X_eval_inj, X_eval_null, labels, truth, fpr_targets, fit_kwargs=None, scale=True):
    """
    Fit classifier on (positives + training-null-split), calibrate threshold on held-out null split.

    Args:
        X_train_pos: positive training features from a DIFFERENT geometry than eval (cross-geometry) or the same one (in_sample).
        X_train_null: null training features (a subset of the full null pool; must be DISJOINT from X_eval_null).
        X_eval_inj: eval-geometry injection features (full set, with labels containing both positives and injection-negatives).
        X_eval_null: held-out null features used solely for threshold calibration.
    """
    fit_kwargs = fit_kwargs or {}
    X_train = np.concatenate([X_train_pos, X_train_null], axis=0)
    y_train = np.concatenate([
        np.ones(X_train_pos.shape[0], dtype=np.int64),
        np.zeros(X_train_null.shape[0], dtype=np.int64),
    ], axis=0)

    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_eval_inj_s = scaler.transform(X_eval_inj)
        X_eval_null_s = scaler.transform(X_eval_null)
    else:
        X_train_s = X_train
        X_eval_inj_s = X_eval_inj
        X_eval_null_s = X_eval_null

    model = model_cls(**fit_kwargs).fit(X_train_s, y_train)
    if hasattr(model, "predict_proba"):
        inj_scores = model.predict_proba(X_eval_inj_s)[:, 1]
        null_scores = model.predict_proba(X_eval_null_s)[:, 1]
    else:
        inj_scores = model.decision_function(X_eval_inj_s)
        null_scores = model.decision_function(X_eval_null_s)

    rows = []
    for fpr_target in fpr_targets:
        threshold, actual_fpr = threshold_at_fpr(null_scores, fpr_target)
        triggered = inj_scores >= threshold
        pos_mask = labels == 1
        hits = int(triggered[pos_mask].sum())
        total = int(pos_mask.sum())
        rows.append({
            "policy": name,
            "fpr_target": float(fpr_target),
            "threshold": float(threshold),
            "actual_null_fpr": float(actual_fpr),
            "recall_global": hits / float(max(total, 1)),
            "recall_ci95": exact_ci(hits, total),
            "groups": group_breakdown(pos_mask, triggered, truth),
        })
    return rows, model


def run_geometry_routed_policy(name, v6_inj, v6_null, v7_inj, v7_null,
                               v7_mf_inj, v7_base_inj, v7_mf_null, v7_base_null,
                               labels, truth, fpr_targets, route_quantile):
    """Per-patch selection: if v7_mf - v7_base > null_quantile, trust v6 (disc-like); else v7 (truncated-like)."""
    route_signal_inj = v7_mf_inj - v7_base_inj * np.std(v7_mf_inj) / max(np.std(v7_base_inj), 1e-9)
    route_signal_null = v7_mf_null - v7_base_null * np.std(v7_mf_null) / max(np.std(v7_base_null), 1e-9)
    # Use the null distribution to define the routing boundary so we do not peek at positives.
    route_threshold = float(np.quantile(route_signal_null, route_quantile))
    route_to_v6_inj = route_signal_inj > route_threshold
    route_to_v6_null = route_signal_null > route_threshold

    # Rank-normalize each model against its null, take per-patch selected score.
    inj_rank_v6 = empirical_rank(v6_inj, v6_null)
    null_rank_v6 = empirical_rank(v6_null, v6_null)
    inj_rank_v7 = empirical_rank(v7_inj, v7_null)
    null_rank_v7 = empirical_rank(v7_null, v7_null)

    inj_score = np.where(route_to_v6_inj, inj_rank_v6, inj_rank_v7)
    null_score = np.where(route_to_v6_null, null_rank_v6, null_rank_v7)

    rows = []
    for fpr_target in fpr_targets:
        threshold, actual_fpr = threshold_at_fpr(null_score, fpr_target)
        triggered = inj_score >= threshold
        pos_mask = labels == 1
        hits = int(triggered[pos_mask].sum())
        total = int(pos_mask.sum())
        rows.append({
            "policy": name,
            "fpr_target": float(fpr_target),
            "threshold_rank": float(threshold),
            "route_threshold": route_threshold,
            "frac_routed_to_v6": float(route_to_v6_inj[pos_mask].mean()) if pos_mask.any() else 0.0,
            "actual_null_fpr": float(actual_fpr),
            "recall_global": hits / float(max(total, 1)),
            "recall_ci95": exact_ci(hits, total),
            "groups": group_breakdown(pos_mask, triggered, truth),
        })
    return rows


def load_geometry_bundle(geometry, batch2_dir, gate_root):
    cache_dir = batch2_dir / "score_cache"
    v6_inj_npz = load_transform_npz(cache_dir / f"inj_{geometry}_v6_aux_only_transforms.npz")
    v7_inj_npz = load_transform_npz(cache_dir / f"inj_{geometry}_v7_mixed_ft_transforms.npz")
    v6_null_npz = load_transform_npz(cache_dir / "null_v6_aux_only_transforms.npz")
    v7_null_npz = load_transform_npz(cache_dir / "null_v7_mixed_ft_transforms.npz")

    labels = np.asarray(v6_inj_npz["labels"], dtype=np.uint8)
    assert np.array_equal(labels, np.asarray(v7_inj_npz["labels"], dtype=np.uint8))
    inj_h5 = gate_root / geometry / f"smica_real_sky_injection_{geometry}.h5"
    truth = load_truth(inj_h5)
    assert np.array_equal(labels, truth["labels"])

    X_inj = stack_features(v6_inj_npz, v7_inj_npz)
    X_null = stack_features(v6_null_npz, v7_null_npz)

    return {
        "geometry": geometry,
        "labels": labels,
        "truth": truth,
        "X_inj": X_inj,
        "X_null": X_null,
        "v6_inj_base": v6_inj_npz["baseline"],
        "v7_inj_base": v7_inj_npz["baseline"],
        "v6_null_base": v6_null_npz["baseline"],
        "v7_null_base": v7_null_npz["baseline"],
        "v7_inj_mf": v7_inj_npz["mf_on_mask"],
        "v7_null_mf": v7_null_npz["mf_on_mask"],
    }


def run_for_geometry(bundle, cross_bundle, route_quantile, learned_fit_on, learned_seed):
    geometry = bundle["geometry"]
    labels = bundle["labels"]
    truth = bundle["truth"]
    results = {"geometry": geometry, "policies": {}}

    results["policies"]["v6_only"] = run_single_model_policy(
        "v6_only", bundle["v6_inj_base"], bundle["v6_null_base"], labels, truth, FPR_TARGETS,
    )
    results["policies"]["v7_only"] = run_single_model_policy(
        "v7_only", bundle["v7_inj_base"], bundle["v7_null_base"], labels, truth, FPR_TARGETS,
    )
    results["policies"]["either_OR"] = run_pairwise_policy(
        "either_OR", bundle["v6_inj_base"], bundle["v6_null_base"], bundle["v7_inj_base"], bundle["v7_null_base"],
        combine=np.logical_or, labels=labels, truth=truth, fpr_targets=FPR_TARGETS,
    )
    results["policies"]["both_AND"] = run_pairwise_policy(
        "both_AND", bundle["v6_inj_base"], bundle["v6_null_base"], bundle["v7_inj_base"], bundle["v7_null_base"],
        combine=np.logical_and, labels=labels, truth=truth, fpr_targets=FPR_TARGETS,
    )
    results["policies"]["rank_max"] = run_rank_max_policy(
        "rank_max", bundle["v6_inj_base"], bundle["v6_null_base"], bundle["v7_inj_base"], bundle["v7_null_base"],
        labels, truth, FPR_TARGETS,
    )
    results["policies"]["geometry_routed"] = run_geometry_routed_policy(
        "geometry_routed", bundle["v6_inj_base"], bundle["v6_null_base"], bundle["v7_inj_base"], bundle["v7_null_base"],
        bundle["v7_inj_mf"], bundle["v7_inj_base"], bundle["v7_null_mf"], bundle["v7_null_base"],
        labels, truth, FPR_TARGETS, route_quantile,
    )

    # Learned routers: logistic regression + gradient boosting on (v6, v7, mf, smooth) features.
    # CRITICAL: the null pool is shared across geometries (same 5000 real-SMICA patches),
    # so training on one geometry's nulls and calibrating the threshold on another geometry's
    # nulls would still be circular. Split the nulls 50/50 by a fixed seed and use disjoint
    # halves for training and evaluation.
    X_inj = bundle["X_inj"]
    X_null = bundle["X_null"]
    rng = np.random.default_rng(learned_seed)
    n_null = X_null.shape[0]
    perm = rng.permutation(n_null)
    null_train_idx = perm[: n_null // 2]
    null_eval_idx = perm[n_null // 2 :]
    X_null_train = X_null[null_train_idx]
    X_null_eval = X_null[null_eval_idx]

    if learned_fit_on == "cross" and cross_bundle is not None:
        X_train_pos = cross_bundle["X_inj"][cross_bundle["labels"] == 1]
        train_descr = f"fit on {cross_bundle['geometry']} positives + shared null train half (held-out null eval half)"
    elif learned_fit_on == "in_sample":
        X_train_pos = X_inj[labels == 1]
        train_descr = f"fit on {geometry} positives + shared null train half (in-sample, optimistic upper bound)"
    else:
        raise NotImplementedError(f"learned_fit_on={learned_fit_on} not supported here")

    print(f"  Learned router training: {train_descr}, features: {list(FEATURE_NAMES)}", flush=True)
    print(f"  Null split: train {X_null_train.shape[0]}, eval {X_null_eval.shape[0]} (disjoint, seed {learned_seed})", flush=True)

    logistic_rows, _ = run_learned_policy(
        "learned_logistic", LogisticRegression,
        X_train_pos=X_train_pos, X_train_null=X_null_train,
        X_eval_inj=X_inj, X_eval_null=X_null_eval,
        labels=labels, truth=truth, fpr_targets=FPR_TARGETS,
        fit_kwargs={"max_iter": 200, "C": 1.0, "class_weight": "balanced"},
    )
    for row in logistic_rows:
        row["train_descr"] = train_descr
    results["policies"]["learned_logistic"] = logistic_rows

    gbt_rows, gbt_model = run_learned_policy(
        "learned_gbt", GradientBoostingClassifier,
        X_train_pos=X_train_pos, X_train_null=X_null_train,
        X_eval_inj=X_inj, X_eval_null=X_null_eval,
        labels=labels, truth=truth, fpr_targets=FPR_TARGETS,
        fit_kwargs={"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "random_state": learned_seed},
        scale=False,
    )
    for row in gbt_rows:
        row["train_descr"] = train_descr
    results["policies"]["learned_gbt"] = gbt_rows

    results["learned_feature_importances"] = {
        "features": list(FEATURE_NAMES),
        "gbt_importances": gbt_model.feature_importances_.tolist(),
    }

    return results


def write_markdown(path, reports, route_quantile):
    lines = ["# Batch 3: Two-Model Portfolio Router", ""]
    lines.append(
        f"Evaluates six portfolio policies on the Batch 2 cached real-SMICA "
        f"transform scores. All thresholds are calibrated on the 5000-patch "
        f"real-SMICA null distribution at FPR targets 0.05 / 0.08 / 0.10."
    )
    lines.append("")
    lines.append(f"Geometry-routed policy: route a candidate to v6 when "
                 f"`v7_mf_on_mask - scaled(v7_baseline) > null q={route_quantile:.2f}`, else route to v7.")
    lines.append("")
    for report in reports:
        lines.append(f"## {report['geometry']} geometry")
        lines.append("")
        lines.append("| policy | FPR 0.05 | FPR 0.08 | FPR 0.10 |")
        lines.append("|---|---:|---:|---:|")
        for policy in POLICIES:
            rows = report["policies"][policy]
            r05 = rows[0]
            r08 = rows[1]
            r10 = rows[2]
            lines.append(
                f"| `{policy}` | {r05['recall_global']:.3f} | {r08['recall_global']:.3f} | {r10['recall_global']:.3f} |"
            )
        lines.append("")
        lines.append("Per-geometry-group recall at FPR 0.08:")
        lines.append("")
        lines.append("| policy | all | contained | truncated | center_out | vis_low |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for policy in POLICIES:
            rows = report["policies"][policy]
            r08 = rows[1]
            g = r08["groups"]
            def g_recall(name):
                v = g.get(name)
                return f"{v['recall']:.3f}" if v else "-"
            lines.append(
                f"| `{policy}` | {g_recall('all_positive')} | {g_recall('geometry_contained')} | "
                f"{g_recall('geometry_truncated')} | {g_recall('center_outside_patch')} | {g_recall('visible_fraction_low')} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    batch2_dir = Path(args.batch2_dir).resolve()
    gate_root = Path(args.gate_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = {g: load_geometry_bundle(g, batch2_dir, gate_root) for g in ("contained", "mixed")}
    reports = []
    for geometry, bundle in bundles.items():
        print(f"\n=== {geometry} ===", flush=True)
        cross_geometry = "mixed" if geometry == "contained" else "contained"
        cross_bundle = bundles[cross_geometry] if args.learned_fit_on == "cross" else None
        report = run_for_geometry(bundle, cross_bundle, args.route_quantile, args.learned_fit_on, args.learned_seed)
        reports.append(report)

    json_path = output_dir / "batch3_router_report.json"
    json_path.write_text(json.dumps({"reports": reports, "route_quantile": args.route_quantile}, indent=2), encoding="utf-8")
    md_path = output_dir / "batch3_router_report.md"
    write_markdown(md_path, reports, args.route_quantile)
    print(f"\n=== Saved ===\n  JSON: {json_path}\n  MD:   {md_path}", flush=True)


if __name__ == "__main__":
    main()
