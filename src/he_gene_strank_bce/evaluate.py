"""Raw-count metrics for the original STRank+BCE detected-singleton experiment."""

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .metrics import detection_metrics, positives_only_ranking


def evaluate_scores(counts, count_scores, detection_scores, *, threshold=None, beta=1.2,
                    top_ks=(10, 30, 60, 100), kendall=True):
    return {
        "n_cells": len(counts),
        "n_genes": counts.shape[1],
        "composition": composition_losses(counts, count_scores),
        "ranking": positives_only_ranking(counts, count_scores, compute_kendall=kendall),
        "detection": detection_metrics(counts, detection_scores, threshold=threshold,
                                       beta=beta, top_ks=top_ks),
    }


def composition_losses(counts, scores):
    """Compute logits in blocks, then average per-cell losses in their original order."""
    losses = np.empty(len(counts), dtype=np.float64)
    totals = counts.sum(axis=1, dtype=np.float64)
    for start in range(0, len(counts), 2048):
        block = slice(start, start + 2048)
        logits = scores[block].astype(np.float64)
        logp = logits - logsumexp(logits, axis=1, keepdims=True)
        losses[block] = -(counts[block] * logp).sum(axis=1)
    valid = totals > 0
    valid_count = int(np.count_nonzero(valid))
    return {
        "raw_count_nll_mean": float(losses[valid].mean()) if valid_count else None,
        "cell_composition_ce_mean": float((losses[valid] / totals[valid]).mean()) if valid_count else None,
        "positive_total_cells": valid_count,
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
