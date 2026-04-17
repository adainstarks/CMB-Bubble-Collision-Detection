"""
Batch 6 analysis: 3-policy cross-map recalibration at paper-grade
statistical power (Nside=32, ~9500 patches per map).

Recomputes the deployment-representative recall for `v6_only`, `gbt_6`, and
`gbt_14` per Planck cleaned map using the Nside=32 tile patches as the null
distribution. Threshold per policy per map is set so exactly 8% of tile
patches trigger; that threshold is then applied to the cached 17500-positive
mixed gate set to read off the deployment-representative recall.

Outputs (under `runs/phase3_unet/batch6_fullsky_nside32_smica/`):
    crossmap_recalibration_nside32.json
    crossmap_recalibration_nside32.md

Designed to run unattended after `batch6_overnight_orchestrator.sh` finishes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from phase3_geometry_router import (  # noqa: E402
    ALL_FEATURE_NAMES, SCORE_FEATURE_NAMES, FEATURE_SOURCES,
    stack_features, load_transform_npz,
)
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "batch2_postprocess_ablation_v1" / "score_cache"
TILE_PATHS = {
    "smica":     PROJECT_ROOT / "runs" / "phase3_unet" / "batch6_fullsky_nside32_smica"     / "tile_features_smica_nside32.npz",
    "nilc":      PROJECT_ROOT / "runs" / "phase3_unet" / "batch6_fullsky_nside32_nilc"      / "tile_features_nilc_nside32.npz",
    "sevem":     PROJECT_ROOT / "runs" / "phase3_unet" / "batch6_fullsky_nside32_sevem"     / "tile_features_sevem_nside32.npz",
    "commander": PROJECT_ROOT / "runs" / "phase3_unet" / "batch6_fullsky_nside32_commander" / "tile_features_commander_nside32.npz",
}
OUTPUT_DIR = PROJECT_ROOT / "runs" / "phase3_unet" / "batch6_fullsky_nside32_smica"
LEARNED_SEED = 20260417
FPR_TARGET = 0.08


def fit_gbt(feature_names, seed=LEARNED_SEED):
    v6_inj_m = load_transform_npz(CACHE_DIR / "inj_mixed_v6_aux_only_transforms.npz")
    v7_inj_m = load_transform_npz(CACHE_DIR / "inj_mixed_v7_mixed_ft_transforms.npz")
    v6_inj_c = load_transform_npz(CACHE_DIR / "inj_contained_v6_aux_only_transforms.npz")
    v7_inj_c = load_transform_npz(CACHE_DIR / "inj_contained_v7_mixed_ft_transforms.npz")
    v6_null = load_transform_npz(CACHE_DIR / "null_v6_aux_only_transforms.npz")
    v7_null = load_transform_npz(CACHE_DIR / "null_v7_mixed_ft_transforms.npz")
    labels_m = np.asarray(v6_inj_m["labels"], dtype=np.uint8)
    labels_c = np.asarray(v6_inj_c["labels"], dtype=np.uint8)
    X_m = stack_features(v6_inj_m, v7_inj_m, feature_names)
    X_c = stack_features(v6_inj_c, v7_inj_c, feature_names)
    X_null = stack_features(v6_null, v7_null, feature_names)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X_null.shape[0])
    null_train = X_null[perm[:2500]]
    X_train_pos = X_c[labels_c == 1]
    X_train = np.concatenate([X_train_pos, null_train])
    y_train = np.concatenate([np.ones(X_train_pos.shape[0]), np.zeros(null_train.shape[0])])
    gbt = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                     random_state=seed).fit(X_train, y_train)
    return {
        "gbt": gbt,
        "mixed_scores": gbt.predict_proba(X_m)[:, 1],
        "mixed_labels": labels_m,
    }


def threshold_at_fpr(scores, target=FPR_TARGET):
    k = max(1, int(np.floor(target * scores.size)))
    return float(np.sort(scores)[-k])


def tile_scores_for_gbt(gbt, feature_names, tile_npz):
    with np.load(tile_npz) as d:
        X = np.stack([
            d[f'v6_{FEATURE_SOURCES[name][1]}'] if FEATURE_SOURCES[name][0] == 'v6'
            else d[f'v7_{FEATURE_SOURCES[name][1]}']
            for name in feature_names
        ], axis=1).astype(np.float64)
    return gbt.predict_proba(X)[:, 1]


def tile_scores_for_v6(tile_npz):
    with np.load(tile_npz) as d:
        return np.asarray(d["v6_baseline"], dtype=np.float64)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference: shipped clean-null thresholds for v6_only, gbt_6, gbt_14
    v6_inj_m = load_transform_npz(CACHE_DIR / "inj_mixed_v6_aux_only_transforms.npz")
    v6_null = load_transform_npz(CACHE_DIR / "null_v6_aux_only_transforms.npz")
    mixed_labels = np.asarray(v6_inj_m["labels"], dtype=np.uint8)
    v6_mixed_scores = np.asarray(v6_inj_m["baseline"], dtype=np.float64)
    v6_null_scores = np.asarray(v6_null["baseline"], dtype=np.float64)
    thr_v6_shipped = threshold_at_fpr(v6_null_scores)

    fit6 = fit_gbt(list(SCORE_FEATURE_NAMES))
    fit14 = fit_gbt(list(ALL_FEATURE_NAMES))

    # Shipped GBT thresholds: same null-eval-half logic as PR #8 / PR #9
    v6_null_npz = load_transform_npz(CACHE_DIR / "null_v6_aux_only_transforms.npz")
    v7_null_npz = load_transform_npz(CACHE_DIR / "null_v7_mixed_ft_transforms.npz")
    rng = np.random.default_rng(LEARNED_SEED)
    perm = rng.permutation(v6_null_npz["baseline"].shape[0])
    null_eval_idx = perm[2500:]
    X_null_eval_6 = stack_features(v6_null_npz, v7_null_npz, list(SCORE_FEATURE_NAMES))[null_eval_idx]
    X_null_eval_14 = stack_features(v6_null_npz, v7_null_npz, list(ALL_FEATURE_NAMES))[null_eval_idx]
    thr_gbt6_shipped = threshold_at_fpr(fit6["gbt"].predict_proba(X_null_eval_6)[:, 1])
    thr_gbt14_shipped = threshold_at_fpr(fit14["gbt"].predict_proba(X_null_eval_14)[:, 1])

    rows = []
    missing_maps = []
    for map_name, tile_path in TILE_PATHS.items():
        if not tile_path.exists():
            print(f"[skip] {map_name}: {tile_path} not found")
            missing_maps.append(map_name)
            continue
        # v6_only
        ts_v6 = tile_scores_for_v6(tile_path)
        thr_v6_tile = threshold_at_fpr(ts_v6)
        rec_v6_ship = float((v6_mixed_scores[mixed_labels == 1] >= thr_v6_shipped).mean())
        rec_v6_tile = float((v6_mixed_scores[mixed_labels == 1] >= thr_v6_tile).mean())
        patch_fpr_v6 = float((ts_v6 >= thr_v6_shipped).mean())

        # gbt_6
        ts_g6 = tile_scores_for_gbt(fit6["gbt"], list(SCORE_FEATURE_NAMES), tile_path)
        thr_g6_tile = threshold_at_fpr(ts_g6)
        rec_g6_ship = float((fit6["mixed_scores"][fit6["mixed_labels"] == 1] >= thr_gbt6_shipped).mean())
        rec_g6_tile = float((fit6["mixed_scores"][fit6["mixed_labels"] == 1] >= thr_g6_tile).mean())
        patch_fpr_g6 = float((ts_g6 >= thr_gbt6_shipped).mean())

        # gbt_14
        ts_g14 = tile_scores_for_gbt(fit14["gbt"], list(ALL_FEATURE_NAMES), tile_path)
        thr_g14_tile = threshold_at_fpr(ts_g14)
        rec_g14_ship = float((fit14["mixed_scores"][fit14["mixed_labels"] == 1] >= thr_gbt14_shipped).mean())
        rec_g14_tile = float((fit14["mixed_scores"][fit14["mixed_labels"] == 1] >= thr_g14_tile).mean())
        patch_fpr_g14 = float((ts_g14 >= thr_gbt14_shipped).mean())

        n_tile = ts_v6.size
        n_pos = int(mixed_labels.sum())
        # FPR-calibration uncertainty (Poisson)
        k_tile = max(1, int(np.floor(FPR_TARGET * n_tile)))
        fpr_uncertainty = float(np.sqrt(k_tile)) / n_tile

        row = {
            "map": map_name, "n_tile": n_tile, "n_pos": n_pos,
            "fpr_calibration_uncertainty": fpr_uncertainty,
            "patch_fpr_v6_at_shipped": patch_fpr_v6,
            "patch_fpr_g6_at_shipped": patch_fpr_g6,
            "patch_fpr_g14_at_shipped": patch_fpr_g14,
            "thr_v6_tile": thr_v6_tile, "rec_v6_tile": rec_v6_tile,
            "thr_g6_tile": thr_g6_tile, "rec_g6_tile": rec_g6_tile,
            "thr_g14_tile": thr_g14_tile, "rec_g14_tile": rec_g14_tile,
            "delta_g6_minus_v6_tile": rec_g6_tile - rec_v6_tile,
            "delta_g14_minus_v6_tile": rec_g14_tile - rec_v6_tile,
            "delta_g14_minus_g6_tile": rec_g14_tile - rec_g6_tile,
        }
        rows.append(row)

    # Cross-map summary
    if rows:
        summary = {
            "fpr_target": FPR_TARGET,
            "learned_seed": LEARNED_SEED,
            "shipped_thresholds": {
                "v6_only": thr_v6_shipped,
                "gbt_6": thr_gbt6_shipped,
                "gbt_14": thr_gbt14_shipped,
            },
            "shipped_mixed_recall_clean_null": {
                "v6_only": rec_v6_ship,
                "gbt_6": rec_g6_ship,
                "gbt_14": rec_g14_ship,
            },
            "per_map": rows,
            "cross_map_mean_tile_recalibrated": {
                "v6_only": float(np.mean([r["rec_v6_tile"] for r in rows])),
                "gbt_6": float(np.mean([r["rec_g6_tile"] for r in rows])),
                "gbt_14": float(np.mean([r["rec_g14_tile"] for r in rows])),
                "delta_g6_minus_v6": float(np.mean([r["delta_g6_minus_v6_tile"] for r in rows])),
                "delta_g14_minus_v6": float(np.mean([r["delta_g14_minus_v6_tile"] for r in rows])),
                "delta_g14_minus_g6": float(np.mean([r["delta_g14_minus_g6_tile"] for r in rows])),
            },
            "missing_maps": missing_maps,
        }
        out_json = OUTPUT_DIR / "crossmap_recalibration_nside32.json"
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Markdown report
        md = ["# Batch 6: Nside=32 cross-map deployment recalibration", "",
              f"FPR target: {FPR_TARGET}  ",
              f"GBT seed: {LEARNED_SEED}  ",
              f"Tile statistical power: ~{rows[0]['n_tile']} patches per map "
              f"(FPR-calibration uncertainty ~+/- {rows[0]['fpr_calibration_uncertainty']:.4f})",
              ""]
        if missing_maps:
            md.append(f"Missing maps (tile run failed or absent): {', '.join(missing_maps)}")
            md.append("")
        md.append("## Patch-level FPR at shipped (clean-null) thresholds")
        md.append("")
        md.append("| map | n_tile | v6_only | gbt_6 | gbt_14 |")
        md.append("|---|---:|---:|---:|---:|")
        for r in rows:
            md.append(f"| {r['map']} | {r['n_tile']} | {r['patch_fpr_v6_at_shipped']:.4f} | "
                      f"{r['patch_fpr_g6_at_shipped']:.4f} | {r['patch_fpr_g14_at_shipped']:.4f} |")
        md.append("")
        md.append(f"For reference, the shipped FPR target is {FPR_TARGET}. Inflation factor = "
                  f"observed / target.")
        md.append("")
        md.append("## Tile-recalibrated mixed recall at FPR 0.08")
        md.append("")
        md.append("| map | v6_only | gbt_6 | gbt_14 | gbt6-v6 | gbt14-v6 | gbt14-gbt6 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            md.append(
                f"| {r['map']} | {r['rec_v6_tile']:.4f} | {r['rec_g6_tile']:.4f} | "
                f"{r['rec_g14_tile']:.4f} | {r['delta_g6_minus_v6_tile']:+.4f} | "
                f"{r['delta_g14_minus_v6_tile']:+.4f} | {r['delta_g14_minus_g6_tile']:+.4f} |"
            )
        cm = summary["cross_map_mean_tile_recalibrated"]
        md.append(
            f"| **mean** | **{cm['v6_only']:.4f}** | **{cm['gbt_6']:.4f}** | "
            f"**{cm['gbt_14']:.4f}** | **{cm['delta_g6_minus_v6']:+.4f}** | "
            f"**{cm['delta_g14_minus_v6']:+.4f}** | **{cm['delta_g14_minus_g6']:+.4f}** |"
        )
        md.append("")
        md.append("## Comparison vs Batch 5 Nside=8 numbers")
        md.append("")
        md.append("Batch 5 (Nside=8, 700 patches per map) numbers were:  ")
        md.append("v6_only 0.237, gbt_6 0.273, gbt_14 0.257 (cross-map means)  ")
        md.append("g6-v6 +0.036, g14-v6 +0.020, g14-g6 -0.015")
        md.append("")
        md.append("Nside=32 above gives statistically tighter versions of the same comparison. "
                  "If the conclusions hold (gbt_6 wins; gbt_14 retraction stands), they are now "
                  "paper-grade.")

        out_md = OUTPUT_DIR / "crossmap_recalibration_nside32.md"
        out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"Saved {out_json}")
        print(f"Saved {out_md}")
    else:
        print("No tile features found. Did the tile jobs finish?")


if __name__ == "__main__":
    main()
