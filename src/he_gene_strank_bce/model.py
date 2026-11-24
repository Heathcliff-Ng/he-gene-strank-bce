"""A frozen-feature predictor with separate composition and detection heads."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    feature_dim: int
    gene_count: int
    hidden_dim: int = 0
    dropout: float = 0.0
    score_mode: str = "count"


class GenePredictor(nn.Module):
    """Forward accepts H&E features only, never counts or cell totals.

    Count logits rank expected measured counts. Sigmoid detection logits
    estimate the probability of a nonzero measurement in the fixed panel.
    """

    feature_mean: torch.Tensor
    feature_scale: torch.Tensor

    def __init__(self, spec: ModelSpec):
        super().__init__()
        if spec.feature_dim < 1 or spec.gene_count < 2 or spec.hidden_dim < 0:
            raise ValueError("Invalid model dimensions")
        if not 0 <= spec.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if spec.score_mode != "count":
            raise ValueError("STRank+BCE requires a count ranking head")
        self.spec = spec
        self.register_buffer("feature_mean", torch.zeros(spec.feature_dim))
        self.register_buffer("feature_scale", torch.ones(spec.feature_dim))
        self.backbone = (
            nn.Sequential(nn.Linear(spec.feature_dim, spec.hidden_dim), nn.GELU(), nn.Dropout(spec.dropout))
            if spec.hidden_dim else nn.Identity()
        )
        dim = spec.hidden_dim or spec.feature_dim
        self.count_head = nn.Linear(dim, spec.gene_count)
        self.detection_head = nn.Linear(dim, spec.gene_count)

    def set_feature_statistics(self, mean: torch.Tensor, scale: torch.Tensor):
        if mean.shape != self.feature_mean.shape or scale.shape != self.feature_scale.shape:
            raise ValueError("Feature statistics have incompatible dimensions")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("Feature statistics must be finite and scales positive")
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(scale)

    def initialize_intercepts(self, counts: torch.Tensor, *, singletons="detected"):
        """Training-only warm start; smoothing is explicitly one half count."""
        with torch.no_grad():
            if singletons == "neutral":
                valid = counts != 1
                pooled = counts.masked_fill(~valid, 0).sum(0) + 0.5
                prevalence = ((counts >= 2).sum(0) + 0.5) / (valid.sum(0) + 1.0)
            elif singletons == "detected":
                pooled = counts.sum(0) + 0.5
                prevalence = ((counts > 0).sum(0) + 0.5) / (len(counts) + 1.0)
            else:
                raise ValueError("singletons must be detected or neutral")
            self.count_head.bias.copy_(torch.log(pooled / pooled.sum()))
            self.detection_head.bias.copy_(torch.logit(prevalence))
            # Feature variation is learned from the pooled-count initialization.
            nn.init.zeros_(self.count_head.weight)
            nn.init.zeros_(self.detection_head.weight)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.spec.feature_dim:
            raise ValueError("features must have shape [cells, feature_dim]")
        h = self.backbone((features - self.feature_mean) / self.feature_scale)
        return self.count_head(h), self.detection_head(h)
