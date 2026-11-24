"""H&E-feature-only inference; no expression labels are read."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.types import Device

from .model import GenePredictor, ModelSpec
from .data import read_index


def resolve_device(requested: str = "auto") -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_checkpoint(path: str | Path, device: Device = "cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    spec = ModelSpec(**payload["model_spec"])
    if len(payload["gene_ids"]) != spec.gene_count or len(set(payload["gene_ids"])) != spec.gene_count:
        raise ValueError("Checkpoint gene identities disagree with model output")
    if len(payload["gene_names"]) != spec.gene_count:
        raise ValueError("Checkpoint gene names disagree with model output")
    if not 0 <= payload["threshold"] <= np.nextafter(1.0, np.inf):
        raise ValueError("Invalid checkpoint probability threshold")
    if payload.get("metadata", {}).get("loss_mode", "multinomial_bce") != "multinomial_bce":
        raise ValueError("Checkpoint is not a STRank+BCE model")
    model = GenePredictor(spec)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


def checkpoint_payload(model, *, gene_ids, gene_names, threshold, source_section, metadata):
    return {
        "format_version": 1,
        "model_spec": asdict(model.spec),
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "gene_ids": list(gene_ids),
        "gene_names": list(gene_names),
        "threshold": float(threshold),
        "source_section": str(source_section),
        "label_definition": ("raw counts >= 2; raw 1 ignored; raw 0 negative"
                             if metadata.get("singletons") == "neutral"
                             else "raw counts > 0 on the fixed measured panel"),
        "metadata": metadata,
    }


@torch.inference_mode()
def predict_features(model, features, *, device=None, batch_size=4096):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = device or next(model.parameters()).device
    model.eval()
    count_scores, probabilities = [], []
    for start in range(0, len(features), batch_size):
        x = features[start:start + batch_size]
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        r, a = model(x)
        count_scores.append(r.cpu().numpy())
        probabilities.append(torch.sigmoid(a).cpu().numpy())
    if not count_scores:
        raise ValueError("Cannot predict an empty cell collection")
    return np.concatenate(count_scores), np.concatenate(probabilities)


def save_predictions(path, *, count_scores, detection_probabilities, threshold, cell_ids, gene_ids, gene_names,
                     section, protocol="detected", training_protocol="detected"):
    """Export all scores and selected genes in descending ranking-score order.

    selected_order contains output-column indices, followed by -1 padding.
    Stable index tie breaks are display-only, not biological ordering.
    """
    if count_scores.shape != detection_probabilities.shape:
        raise ValueError("Prediction heads must share cell/gene dimensions")
    if count_scores.shape != (len(cell_ids), len(gene_ids)):
        raise ValueError("Prediction identity dimensions disagree")
    if len(gene_names) != len(gene_ids) or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("Gene metadata must contain unique identities and aligned names")
    if not np.isfinite(count_scores).all() or not np.isfinite(detection_probabilities).all():
        raise ValueError("Predictions must be finite")
    if not 0 <= threshold <= np.nextafter(1.0, np.inf) or np.any((detection_probabilities < 0) | (detection_probabilities > 1)):
        raise ValueError("Invalid detection probability or threshold")
    selected = detection_probabilities.astype(np.float64, copy=False) >= threshold
    order = np.argsort(-count_scores, axis=1, kind="stable")
    chosen = np.take_along_axis(selected, order, axis=1)
    # Move selected entries to the beginning without changing their rank order.
    permutation = np.argsort(~chosen, axis=1, kind="stable")
    selected_order = np.take_along_axis(order, permutation, axis=1).astype(np.int32)
    lengths = selected.sum(1)
    selected_order[np.arange(order.shape[1])[None, :] >= lengths[:, None]] = -1
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, count_logits=count_scores, ranking_scores=count_scores, detection_probabilities=detection_probabilities,
        format_version=2, loss_mode="multinomial_bce", protocol=protocol, training_protocol=training_protocol,
        score_interpretation="count ranking",
        selected=selected, selected_order=selected_order, threshold=float(threshold),
        cell_ids=np.asarray(cell_ids, dtype=np.str_), section=str(section),
        gene_ids=np.asarray(gene_ids, dtype=np.str_), gene_names=np.asarray(gene_names, dtype=np.str_),
    )


def read_features(path):
    """Read H5ad obsm/he or a NumPy feature array without opening counts."""
    path = Path(path)
    if path.suffix == ".h5ad":
        with h5py.File(path, "r") as f:
            features = np.asarray(f["obsm/he"][...], dtype=np.float32)
            cell_ids = read_index(f["obs"])
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as f:
            features = np.asarray(f["features"], dtype=np.float32)
            cell_ids = np.asarray(f["cell_ids"], dtype=np.str_)
    elif path.suffix == ".npy":
        features = np.load(path, allow_pickle=False).astype(np.float32)
        cell_ids = np.arange(len(features)).astype(np.str_)
    else:
        raise ValueError("Input must be .h5ad, .npz with features/cell_ids, or .npy")
    if features.ndim != 2 or features.shape[0] != len(cell_ids) or not np.isfinite(features).all():
        raise ValueError("Features must be a finite matrix aligned with cell IDs")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("Cell IDs within the input must be unique")
    return features, cell_ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--section", required=True)
    parser.add_argument("--calibration", help="Use a frozen protocol threshold bound to this checkpoint.")
    parser.add_argument("--protocol", choices=("detected", "neutral"),
                        help="Choose a threshold protocol from the supplied calibration.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    training_protocol = checkpoint["metadata"].get("protocol", "detected")
    protocol = training_protocol
    threshold = checkpoint["threshold"]
    if args.calibration:
        from .protocols import read_calibration
        calibration = read_calibration(args.calibration, args.checkpoint, args.protocol)
        if calibration["source"] != checkpoint["source_section"]:
            parser.error("Calibration and checkpoint have different source sections")
        threshold, protocol = calibration["threshold"], calibration["protocol"]
    elif args.protocol and args.protocol != training_protocol:
        parser.error("Switching threshold protocol requires --calibration")
    features, ids = read_features(args.features)
    r, pi = predict_features(model, features, device=device, batch_size=args.batch_size)
    save_predictions(args.output, count_scores=r, detection_probabilities=pi,
                     threshold=threshold, cell_ids=ids, protocol=protocol, training_protocol=training_protocol,
                     gene_ids=checkpoint["gene_ids"], gene_names=checkpoint["gene_names"], section=args.section)
    print(f"Saved {len(ids)} cell predictions to {args.output}")


if __name__ == "__main__":
    main()
