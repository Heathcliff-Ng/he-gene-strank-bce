"""Regression checks for the two meanings of a single measured molecule."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from he_gene_strank_bce.evaluate import composition_losses
from he_gene_strank_bce.losses import multitask_loss
from he_gene_strank_bce.protocols import neutral_threshold, read_calibration, selection_metrics


def test_composition_report_matches_original_cell_mean_across_blocks():
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 6, size=(5000, 6))
    counts[::19] = 0
    scores = rng.normal(size=counts.shape).astype(np.float32)
    result = composition_losses(counts, scores)
    logits = scores.astype(np.float64)
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    totals = counts.sum(axis=1)
    valid = totals > 0
    nll = -(counts * logp).sum(axis=1)[valid]
    assert result["positive_total_cells"] == np.count_nonzero(valid)
    assert result["raw_count_nll_mean"] == nll.mean()
    assert result["cell_composition_ce_mean"] == (nll / totals[valid]).mean()


def test_singleton_gradients_follow_training_protocol():
    counts = torch.tensor([[0., 1., 3.], [1., 1., 1.]])
    r = torch.tensor([[.1, .2, .3], [.1, .2, .3]], requires_grad=True)
    a = torch.zeros_like(r, requires_grad=True)
    detected, _ = multitask_loss(r, a, counts, count_scale=3, singletons="detected")
    detected.backward()
    assert a.grad[0, 1] < 0
    r.grad = a.grad = None
    neutral, terms = multitask_loss(r, a, counts, count_scale=3, singletons="neutral")
    neutral.backward()
    expected = -3 * torch.log_softmax(r.detach()[0, [0, 2]], dim=0)[1]
    torch.testing.assert_close(terms["count_nll"].detach(), expected)
    assert torch.count_nonzero(a.grad[counts == 1]) == 0
    assert torch.count_nonzero(r.grad[counts == 1]) == 0
    assert a.grad[0, 0] > 0 and a.grad[0, 2] < 0


def test_neutral_selected_stays_in_set_without_adding_cost():
    counts = np.array([[1, 0, 2, 1]])
    probs = np.array([[.9, .8, .7, .1]])
    neutral = selection_metrics(counts, probs, .7, "neutral")
    detected = selection_metrics(counts, probs, .7, "detected")
    assert (neutral["selected"], neutral["selected_ignored"], neutral["tp"], neutral["cost"]) == (3, 1, 1, 1)
    assert (detected["selected"], detected["selected_ignored"], detected["tp"], detected["cost"]) == (3, 0, 2, 3)
    assert neutral_threshold(counts, probs)["threshold"] == .7


def test_frozen_thresholds_are_bound_to_weights():
    root = Path(__file__).resolve().parents[1]
    registry = root / "reports/checkpoints.json"
    record = json.loads(registry.read_text())
    assert {(m["source"], m["seed"]) for m in record["models"]} == {
        (source, seed) for source in ("Rep1", "Rep2") for seed in (42, 43, 44)
    }
    for entry in record["models"]:
        checkpoint = root / entry["path"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == entry["sha256"]
        for mode in ("detected", "neutral"):
            assert read_calibration(registry, checkpoint, mode)["threshold"] == entry[mode + "_threshold"]
    with pytest.raises(ValueError, match="hash"):
        read_calibration(registry, registry, "neutral")
