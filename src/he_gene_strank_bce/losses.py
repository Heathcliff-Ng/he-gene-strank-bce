"""Count likelihood and measured detection losses with explicit reductions."""

import math

import torch
from torch.nn import functional as F


def _validate(logits: torch.Tensor, counts: torch.Tensor):
    if logits.ndim != 2 or counts.shape != logits.shape or not logits.numel():
        raise ValueError("logits and counts must share nonempty shape [cells, genes]")
    if logits.device != counts.device:
        raise ValueError("logits and counts must be on the same device")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(counts).all()):
        raise ValueError("logits and counts must be finite")
    if bool((counts < 0).any()) or not bool((counts == counts.round()).all()):
        raise ValueError("Labels must have raw nonnegative integer count semantics")


def detection_bce(logits: torch.Tensor, counts: torch.Tensor, *, validate: bool = True):
    """Unweighted Bernoulli BCE, mean over cells and all measured genes."""
    if validate:
        _validate(logits, counts)
    return F.binary_cross_entropy_with_logits(logits, (counts > 0).to(logits.dtype))


def multitask_loss(count_logits, detection_logits, counts, *, count_scale: float,
                   detection_weight: float = 1.0, singletons="detected", validate: bool = True):
    """A multi-task surrogate, not independent likelihoods of two datasets.

    count_scale is a fixed positive training-set mean total, not the current
    cell or batch total. Thus it does not change count-risk weighting.
    """
    if not math.isfinite(count_scale) or not math.isfinite(detection_weight) or count_scale <= 0 or detection_weight < 0:
        raise ValueError("count_scale must be positive; detection_weight nonnegative")
    if validate:
        _validate(count_logits, counts)
        _validate(detection_logits, counts)
    if singletons == "detected":
        logp = F.log_softmax(count_logits, dim=-1)
        nll = -(counts.to(logp.dtype) * logp).sum(-1).mean()
        bce = detection_bce(detection_logits, counts, validate=False)
    elif singletons == "neutral":
        valid = counts != 1
        active = valid.any(dim=1)
        # An all-singleton row has no measurable label in this protocol.
        if active.any():
            mask = valid[active]
            masked_logits = count_logits[active].masked_fill(~mask, -torch.inf)
            logp = F.log_softmax(masked_logits, dim=1).masked_fill(~mask, 0)
            nll = -(counts[active] * logp).sum(1).mean()
            per_gene = F.binary_cross_entropy_with_logits(
                detection_logits[active], (counts[active] >= 2).to(detection_logits.dtype),
                reduction="none",
            )
            bce = ((per_gene * mask).sum(1) / mask.sum(1)).mean()
        else:
            nll = count_logits.sum() * 0.0
            bce = detection_logits.sum() * 0.0
    else:
        raise ValueError("singletons must be detected or neutral")
    count_term = nll / count_scale
    total = count_term + detection_weight * bce
    return total, {"count_nll": nll, "scaled_count_nll": count_term, "detection_bce": bce}
