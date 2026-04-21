"""Finite-sample threshold utilities for Phase 3 score calibration.

Assumptions
-----------
* Calibration scores are exchangeable with future null scores after the chosen
  split, mask, map family, and preprocessing policy are fixed.
* Thresholds are selected from calibration null scores only. Positive labels
  may be present in the same arrays, but they are ignored for threshold
  construction.
* Scores are oriented so larger values are more detection-like.
* We use a strict ``score > threshold`` detection rule, matching the existing
  Phase 3 reports. This is conservative in the presence of tied scores.

The finite-sample construction follows split-conformal quantile calibration:
for ``n`` null calibration scores and target false-positive rate ``alpha``, use
the order statistic ``ceil((n + 1) * (1 - alpha))``. Then, under exchangeability,
the future-null exceedance probability is bounded by the recorded
``finite_sample_fpr_bound`` without a parametric score-distribution assumption.
See Vovk, Gammerman & Shafer (2005) and Angelopoulos & Bates (2023).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def conformal_threshold_from_null_scores(
    null_scores: np.ndarray,
    target_fpr: float,
) -> dict[str, Any]:
    """Return a split-conformal threshold record from null scores only."""

    scores = np.asarray(null_scores, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("No finite null scores available for threshold calibration.")
    alpha = float(target_fpr)
    if not (0.0 < alpha < 1.0):
        raise ValueError("target_fpr must lie in (0, 1).")

    sorted_scores = np.sort(scores)
    n = int(sorted_scores.size)
    rank_1indexed = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank_1indexed > n:
        threshold = float(np.nextafter(sorted_scores[-1], np.inf))
        finite_sample_fpr_bound = 0.0
    else:
        threshold = float(sorted_scores[rank_1indexed - 1])
        finite_sample_fpr_bound = float((n - rank_1indexed + 1) / (n + 1))

    flagged = scores > threshold
    fp = int(np.count_nonzero(flagged))
    realized_fpr = float(fp / n)
    return {
        "threshold": threshold,
        "negative_fp": fp,
        "negative_fpr": realized_fpr,
        "target_fpr": alpha,
        "num_null": n,
        "rank_1indexed": int(rank_1indexed),
        "finite_sample_fpr_bound": finite_sample_fpr_bound,
        "detection_rule": "score > threshold",
        "threshold_method": "split_conformal_null_quantile",
        "assumption": "exchangeable calibration and future null scores",
    }


def conformal_threshold_from_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float,
) -> dict[str, Any]:
    """Return a split-conformal threshold record from mixed labels."""

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have matching one-dimensional shapes.")
    return conformal_threshold_from_null_scores(scores[labels == 0], target_fpr)


def threshold_tuple_from_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float,
) -> tuple[float, int, float]:
    """Compatibility tuple ``(threshold, negative_fp, negative_fpr)``."""

    record = conformal_threshold_from_scores(scores, labels, target_fpr)
    return (
        float(record["threshold"]),
        int(record["negative_fp"]),
        float(record["negative_fpr"]),
    )
