"""Cell-wise panel detection and ranking, with explicit tie handling.

Detection uses the complete measured panel and ``counts > 0`` as truth.
AP averages over cells containing at least one positive gene. Thresholded
and top-k macro scores include every cell and use zero_division=0; the
undefined denominator counts are reported. Fixed top-k metrics average
uniformly over the possible selections at a score-tied cutoff.

Positive-only correlations include *all* truth-positive genes, whether or
not a detector selected them. NDCG uses raw count gains over the full panel;
ties in prediction scores receive their expected discounts. No metric uses
a cell's true number of positive genes to choose a predicted set.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kendalltau

_BLOCK_SIZE = 2048


def _inputs(counts, scores):
    if not hasattr(counts, "shape"):
        counts = np.asarray(counts)
    if not hasattr(scores, "shape"):
        scores = np.asarray(scores)
    if len(counts.shape) != 2 or len(scores.shape) != 2:
        raise ValueError("counts and scores must be two-dimensional cell-by-gene matrices")
    if counts.shape != scores.shape or counts.shape[1] == 0:
        raise ValueError("counts and scores must have the same nonempty gene axis")
    return counts, scores


def _dense(value) -> NDArray[np.float64]:
    return np.asarray(value.toarray() if hasattr(value, "toarray") else value, dtype=np.float64)


def _blocks(counts, scores) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    for start in range(0, counts.shape[0], _BLOCK_SIZE):
        y = _dense(counts[start : start + _BLOCK_SIZE])
        s = _dense(scores[start : start + _BLOCK_SIZE])
        if not np.all(np.isfinite(y)) or np.any(y < 0):
            raise ValueError("counts must be finite and nonnegative")
        if not np.all(np.isfinite(s)):
            raise ValueError("scores must be finite; negative scores are allowed")
        yield y, s


def _beta(beta: float) -> float:
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    return beta


def _ratio(numerator, denominator):
    return np.divide(
        numerator, denominator,
        out=np.zeros_like(numerator, dtype=np.float64), where=denominator != 0,
    )


def _mean(total, n):
    return float(total / n) if n else None


def _group_bounds(sorted_values):
    """Start/end indices of each score tie group in sorted rows."""
    n, g = sorted_values.shape
    positions = np.broadcast_to(np.arange(g), (n, g))
    starts = np.ones((n, g), dtype=bool)
    ends = np.ones((n, g), dtype=bool)
    starts[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    ends[:, :-1] = starts[:, 1:]
    first = np.maximum.accumulate(np.where(starts, positions, 0), axis=1)
    last = np.minimum.accumulate(np.where(ends, positions, g)[:, ::-1], axis=1)[:, ::-1]
    return first, last


def _new_summary():
    return {"n": 0, "tp": 0.0, "pred": 0.0, "truth": 0.0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "fbeta": 0.0,
            "precision_valid": 0, "recall_valid": 0, "fbeta_valid": 0}


def _add_summary(state: dict, tp: np.ndarray, selected: np.ndarray | int,
                 positives: np.ndarray, beta: float):
    selected = np.broadcast_to(np.asarray(selected), tp.shape)
    b2 = beta * beta
    state["n"] += len(tp)
    state["tp"] += float(np.sum(tp, dtype=np.float64))
    state["pred"] += float(np.sum(selected))
    state["truth"] += float(np.sum(positives, dtype=np.float64))
    state["precision"] += float(np.sum(_ratio(tp, selected)))
    state["recall"] += float(np.sum(_ratio(tp, positives)))
    state["f1"] += float(np.sum(_ratio(2 * tp, positives + selected)))
    state["fbeta"] += float(np.sum(_ratio((1 + b2) * tp, b2 * positives + selected)))
    state["precision_valid"] += int(np.sum(selected > 0))
    state["recall_valid"] += int(np.sum(positives > 0))
    state["fbeta_valid"] += int(np.sum((positives + selected) > 0))


def _finish_summary(state: dict, beta: float) -> dict[str, Any]:
    n, tp, selected, positives = state["n"], state["tp"], state["pred"], state["truth"]
    b2 = beta * beta
    result: dict[str, Any] = {key: _mean(state[key], n) for key in ("precision", "recall", "f1", "fbeta")}
    result.update({
        "n_cells": n, "beta": beta,
        "selected_genes_mean": _mean(selected, n),
        "true_positive_genes_mean": _mean(tp, n),
        "false_positive_genes_mean": _mean(selected - tp, n),
        "false_negative_genes_mean": _mean(positives - tp, n),
        "zero_division": 0,
        "micro": {
            "precision": float(tp / selected) if selected else 0.0,
            "recall": float(tp / positives) if positives else 0.0,
            "f1": float(2 * tp / (positives + selected)) if positives + selected else 0.0,
            "fbeta": float((1 + b2) * tp / (b2 * positives + selected))
            if b2 * positives + selected else 0.0,
            "true_positives": tp, "false_positives": selected - tp,
            "false_negatives": positives - tp,
        },
    })
    for metric, source in (("precision", "precision_valid"), ("recall", "recall_valid"),
                           ("f1", "fbeta_valid"), ("fbeta", "fbeta_valid")):
        valid = state[source]
        result[f"{metric}_valid_cells"] = valid
        result[f"{metric}_undefined_cells"] = n - valid
        result[f"{metric}_coverage"] = float(valid / n) if n else 0.0
    return result


def detection_metrics(counts_or_bool, scores, threshold=None, beta=1.2,
                      top_ks=(10, 30, 60, 100)):
    """Summarize exact tie-aware AP and full-panel gene-set selection.

    A supplied threshold selects ``score >= threshold``. Scores need not be
    probabilities and are never clipped. Top-k values are fixed across
    cells, capped at the panel size, and independent of test expression.
    """
    counts, scores = _inputs(counts_or_bool, scores)
    beta = _beta(beta)
    if threshold is not None:
        threshold = float(threshold)
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
    requested_ks = tuple(top_ks)
    ks = tuple(dict.fromkeys(int(k) for k in requested_ks))
    if any(k < 1 or int(k) != k for k in requested_ks):
        raise ValueError("top-k values must be positive integers")
    g = counts.shape[1]
    states = {k: _new_summary() for k in ks}
    threshold_state = _new_summary()
    ap_sum, ap_valid = 0.0, 0
    for y, s in _blocks(counts, scores):
        truth = y > 0
        positives = truth.sum(axis=1)
        order = np.argsort(-s, axis=1, kind="stable")
        ranked_scores = np.take_along_axis(s, order, axis=1)
        ranked_truth = np.take_along_axis(truth, order, axis=1)
        cumulative = np.cumsum(ranked_truth, axis=1, dtype=np.int64)
        first, last = _group_bounds(ranked_scores)
        end_tp = np.take_along_axis(cumulative, last, axis=1)
        ap = _ratio(np.sum(ranked_truth * end_tp / (last + 1), axis=1), positives)
        valid = positives > 0
        ap_sum += float(ap[valid].sum())
        ap_valid += int(valid.sum())
        padded_cumulative = np.pad(cumulative, ((0, 0), (1, 0)))
        rows = np.arange(len(y))
        for k, state in states.items():
            actual_k = min(k, g)
            start, end = first[:, actual_k - 1], last[:, actual_k - 1]
            before = padded_cumulative[rows, start]
            group_tp = padded_cumulative[rows, end + 1] - before
            tp = before + group_tp * (actual_k - start) / (end - start + 1)
            _add_summary(state, tp, actual_k, positives, beta)
        if threshold is not None:
            selected = s >= threshold
            _add_summary(threshold_state, (selected & truth).sum(axis=1),
                         selected.sum(axis=1), positives, beta)
    n = counts.shape[0]
    result = {
        "n_cells": n, "n_genes": g,
        "average_precision": _mean(ap_sum, ap_valid),
        "average_precision_valid_cells": ap_valid,
        "average_precision_undefined_cells": n - ap_valid,
        "average_precision_coverage": float(ap_valid / n) if n else 0.0,
        "average_precision_tie_policy": "precision_at_complete_score_tie_group",
        "top_k": {},
    }
    for k, state in states.items():
        summary = _finish_summary(state, beta)
        summary.update({"k": min(k, g), "requested_k": k,
                        "tie_policy": "expected_uniform_selection_at_boundary"})
        result["top_k"][str(k)] = summary
    if threshold is not None:
        result["threshold"] = threshold
        result["threshold_metrics"] = _finish_summary(threshold_state, beta)
    return result


def _masked_average_ranks(values, positive):
    masked_values = np.where(positive, values, np.inf)
    order = np.argsort(masked_values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(masked_values, order, axis=1)
    first, last = _group_bounds(sorted_values)
    sorted_ranks = (first + last) / 2.0 + 1
    sorted_ranks[~np.isfinite(sorted_values)] = 0
    ranks = np.empty_like(sorted_ranks)
    np.put_along_axis(ranks, order, sorted_ranks, axis=1)
    return ranks


def positives_only_ranking(counts, scores, compute_kendall=True):
    """Positive-only rank correlations and full-panel count-gain NDCG.

    Correlations are undefined for fewer than two positives, constant
    positive counts, or constant positive scores. All such rows are
    excluded from their means and reported in coverage counts. NDCG is
    undefined only for all-zero count rows.
    """
    counts, scores = _inputs(counts, scores)
    g, n = counts.shape[1], counts.shape[0]
    names = ["spearman_positive", "ndcg_10", "ndcg_30", "ndcg_60", "ndcg_all"]
    if compute_kendall:
        names.append("kendall_tau_b_positive")
    sums = dict.fromkeys(names, 0.0)
    valid_counts = dict.fromkeys(names, 0)
    for y, s in _blocks(counts, scores):
        positive = y > 0
        npositive = positive.sum(axis=1)
        mean_rank = (npositive + 1)[:, None] / 2.0
        yr = _masked_average_ranks(y, positive) - mean_rank * positive
        sr = _masked_average_ranks(s, positive) - mean_rank * positive
        covariance = np.sum(yr * sr, axis=1)
        denominator = np.sqrt(np.sum(yr * yr, axis=1) * np.sum(sr * sr, axis=1))
        valid = denominator > 0
        values = _ratio(covariance, denominator)
        sums["spearman_positive"] += float(np.sum(values[valid]))
        valid_counts["spearman_positive"] += int(np.count_nonzero(valid))
        if compute_kendall:
            for i in np.flatnonzero(valid):
                value = float(kendalltau(y[i, positive[i]], s[i, positive[i]], variant="b").statistic)
                if np.isfinite(value):
                    sums["kendall_tau_b_positive"] += value
                    valid_counts["kendall_tau_b_positive"] += 1
        order = np.argsort(-s, axis=1, kind="stable")
        ranked_scores = np.take_along_axis(s, order, axis=1)
        gains = np.take_along_axis(y, order, axis=1)
        first, last = _group_bounds(ranked_scores)
        ideal_gains = np.sort(y, axis=1)[:, ::-1]
        for suffix, k in (("10", min(10, g)), ("30", min(30, g)),
                          ("60", min(60, g)), ("all", g)):
            discounts = np.zeros(g, dtype=np.float64)
            discounts[:k] = 1.0 / np.log2(np.arange(k) + 2)
            prefix = np.r_[0.0, np.cumsum(discounts)]
            expected_discount = (prefix[last + 1] - prefix[first]) / (last - first + 1)
            dcg = np.sum(gains * expected_discount, axis=1)
            ideal = np.sum(ideal_gains * discounts, axis=1)
            valid_ndcg = ideal > 0
            values = _ratio(dcg, ideal)
            name = f"ndcg_{suffix}"
            sums[name] += float(np.sum(values[valid_ndcg]))
            valid_counts[name] += int(np.count_nonzero(valid_ndcg))
    result: dict[str, Any] = {"n_cells": n, "n_genes": g, "kendall_computed": bool(compute_kendall),
              "correlation_gene_set": "all_truth_positive_genes",
              "ndcg_gene_set": "full_panel", "ndcg_gain": "raw_counts",
              "ndcg_tie_policy": "expected_discount_over_complete_score_tie_group"}
    for name in names:
        valid = valid_counts[name]
        result[name] = _mean(sums[name], valid)
        result[f"{name}_valid_cells"] = valid
        result[f"{name}_undefined_cells"] = n - valid
        result[f"{name}_coverage"] = float(valid / n) if n else 0.0
    return result


def _threshold_candidates(scores) -> NDArray[np.float64]:
    """Deterministic score-quantile grid plus all/none selection endpoints."""
    n, g = scores.shape
    # Sample by complete rows, limiting quantile construction to ~200k scores.
    stride = max(1, int(np.ceil(n * g / 200_000)))
    sample = _dense(scores[::stride]).ravel()
    if not np.all(np.isfinite(sample)):
        raise ValueError("scores must be finite")
    minimum, maximum = np.inf, -np.inf
    for start in range(0, n, _BLOCK_SIZE):
        block = _dense(scores[start : start + _BLOCK_SIZE])
        if not np.all(np.isfinite(block)):
            raise ValueError("scores must be finite")
        minimum = min(minimum, float(np.min(block)))
        maximum = max(maximum, float(np.max(block)))
    candidates = np.asarray(np.r_[minimum, np.quantile(sample, np.linspace(0, 1, 101)),
                                  np.nextafter(maximum, np.inf)], dtype=np.float64)
    return np.unique(candidates[np.isfinite(candidates)]).astype(np.float64, copy=False)


def select_threshold(counts, scores, beta=1.2, thresholds=None):
    """Choose one global threshold from source-section validation labels only.

    The objective is macro (row-mean) F-beta with zero_division=0. Exact
    objective ties choose the larger threshold to reduce false positives.
    The default candidates are sampled score quantiles plus finite all/none
    selection endpoints; raw regression scores require no zero cutoff.
    """
    counts, scores = _inputs(counts, scores)
    beta = _beta(beta)
    if counts.shape[0] == 0:
        raise ValueError("threshold selection needs at least one validation cell")
    if thresholds is None:
        candidates = _threshold_candidates(scores)
        method = "validation_score_quantiles_plus_endpoints"
    else:
        candidates = np.asarray(tuple(thresholds), dtype=np.float64)
        if candidates.ndim != 1 or not len(candidates) or not np.all(np.isfinite(candidates)):
            raise ValueError("thresholds must be a nonempty finite one-dimensional sequence")
        candidates = np.unique(candidates).astype(np.float64, copy=False)
        method = "provided_validation_candidates"
    objective_sums = np.zeros(len(candidates))
    g, b2 = counts.shape[1], beta * beta
    for y, s in _blocks(counts, scores):
        truth = y > 0
        positives = truth.sum(axis=1)
        order = np.argsort(s, axis=1, kind="stable")
        sorted_scores = np.take_along_axis(s, order, axis=1)
        cumulative = np.pad(np.cumsum(np.take_along_axis(truth, order, axis=1), axis=1),
                            ((0, 0), (1, 0)))
        shape = (len(y), len(candidates))
        lower = np.zeros(shape, dtype=np.int64)
        upper = np.full(shape, g, dtype=np.int64)
        rows = np.arange(len(y))[:, None]
        # Row-wise searchsorted(side='left'), with memory O(block * thresholds).
        for _ in range((g + 1).bit_length()):
            middle = (lower + upper) // 2
            below = (middle < g) & (sorted_scores[rows, np.minimum(middle, g - 1)] < candidates)
            lower = np.where(below, middle + 1, lower)
            upper = np.where(below, upper, middle)
        selected = g - lower
        tp = positives[:, None] - cumulative[rows, lower]
        objective_sums += _ratio((1 + b2) * tp, b2 * positives[:, None] + selected).sum(axis=0)
    best = np.flatnonzero(objective_sums == objective_sums.max())[-1]
    threshold = float(candidates[best])
    state = _new_summary()
    for y, s in _blocks(counts, scores):
        truth, selected = y > 0, s >= threshold
        _add_summary(state, (truth & selected).sum(axis=1), selected.sum(axis=1),
                     truth.sum(axis=1), beta)
    return {"threshold": threshold, "beta": beta, "selection_metric": "macro_fbeta",
            "candidate_method": method, "candidate_count": len(candidates),
            "tie_break": "larger_threshold", "validation_metrics": _finish_summary(state, beta)}
