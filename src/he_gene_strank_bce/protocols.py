"""Calibrate on a source-section holdout or evaluate frozen STRank+BCE predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import SectionData, align_genes, load_section, spatial_split
from .evaluate import evaluate_scores, write_json
from .metrics import select_threshold
from .predict import load_checkpoint, predict_features, resolve_device


def _counts_and_probabilities(counts, probabilities):
    counts = np.asarray(counts)
    scores = np.asarray(probabilities)
    if (counts.ndim != 2 or not counts.size or scores.shape != counts.shape
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts != np.floor(counts))):
        raise ValueError("Expected aligned nonempty [cells, genes] arrays with raw integer counts")
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Detection probabilities must be finite and in [0, 1]")
    return counts, scores


def _threshold(value):
    threshold = float(value)
    if not 0 <= threshold <= np.nextafter(1., np.inf):
        raise ValueError("Invalid detection threshold")
    return threshold


def neutral_threshold(counts, probabilities):
    """Exact source FP+2FN minimization, with ties resolved as in the frozen reevaluation."""
    counts, scores = _counts_and_probabilities(counts, probabilities)
    scores = scores.astype(np.float64, copy=False).ravel()
    order = np.argsort(-scores, kind="stable")
    ranked, truth = scores[order], counts.ravel()[order]
    ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]), len(ranked) - 1]
    thresholds = np.r_[np.nextafter(ranked[0], np.inf), ranked[ends]]
    positive = np.greater_equal(truth, 2)
    fp = np.r_[0, np.cumsum(truth == 0, dtype=np.int64)[ends]]
    tp = np.r_[0, np.cumsum(positive, dtype=np.int64)[ends]]
    fn = np.count_nonzero(positive) - tp
    selected = np.r_[0, ends + 1]
    cost = fp + 2 * fn
    best = np.lexsort((selected, fp, cost))[0]
    return {"threshold": float(thresholds[best]), "criterion": "min FP+2FN",
            "tie_break": "fewer FP, then fewer total selected including singletons",
            "candidate_count": len(thresholds), "score_comparison": ">=",
            "fp": float(fp[best] / len(counts)), "fn": float(fn[best] / len(counts)),
            "cost": float(cost[best] / len(counts)), "selected": float(selected[best] / len(counts))}


def selection_metrics(counts, probabilities, threshold, protocol):
    """Means over cells. Neutral entries remain in the actual selected set."""
    if protocol not in ("detected", "neutral"):
        raise ValueError("Unknown singleton protocol")
    counts, probabilities = _counts_and_probabilities(counts, probabilities)
    selected = probabilities.astype(np.float64, copy=False) >= _threshold(threshold)
    positive = np.greater_equal(counts, 2 if protocol == "neutral" else 1)
    tp = (selected & positive).sum(1)
    fp = (selected & (counts == 0)).sum(1)
    fn = (~selected & positive).sum(1)
    ignored = selected.sum(1) - tp - fp
    return {"tp": float(tp.mean()), "fp": float(fp.mean()), "fn": float(fn.mean()),
            "cost": float((fp + 2 * fn).mean()), "selected": float(selected.sum(1).mean()),
            "selected_ignored": float(ignored.mean()), "errors": float((fp + fn).mean()),
            "precision": float((tp / np.maximum(tp + fp, 1)).mean()),
            "recall": float((tp / np.maximum(tp + fn, 1)).mean())}


def _neutral_ndcg10(counts, count_scores, probabilities, threshold):
    """Rank selected non-neutral genes by count score. Use raw counts as gains."""
    order = np.argsort(-count_scores, axis=1, kind="stable")
    selected = probabilities.astype(np.float64, copy=False) >= threshold
    selected_valid = np.take_along_axis(selected & (counts != 1), order, axis=1)
    # Compact the selected genes, preserving their score order.
    selected_order = np.take_along_axis(order, np.argsort(~selected_valid, axis=1, kind="stable"), axis=1)
    k = min(10, counts.shape[1])
    gains = np.take_along_axis(counts, selected_order[:, :k], axis=1)
    gains = np.where(np.arange(k) < selected_valid.sum(1)[:, None], gains, 0)
    ideal = np.sort(np.where(counts >= 2, counts, 0), axis=1)[:, ::-1][:, :k]
    discounts = 1 / np.log2(np.arange(2, k + 2))
    denominator = (ideal * discounts).sum(axis=1)
    valid = denominator > 0
    discounted_gains = (gains[valid] * discounts).sum(axis=1)
    return float((discounted_gains / denominator[valid]).mean()) if valid.any() else None


def _sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_calibration(path, checkpoint_path, protocol=None):
    """Resolve an individual calibration or the distributed checkpoint/threshold registry.

    The registry defaults to its recalibrated neutral threshold. For an individual
    file the stored protocol is used unless the caller explicitly requires one.
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = _sha256(checkpoint_path)
    if "models" in record:
        matches = [entry for entry in record["models"] if entry["sha256"] == digest]
        if len(matches) != 1:
            raise ValueError("Checkpoint hash is missing or ambiguous in the threshold registry")
        match = matches[0]
        mode = protocol or "neutral"
        if mode not in ("detected", "neutral"):
            raise ValueError("Invalid protocol")
        record = {"protocol": mode, "threshold": match[f"{mode}_threshold"],
                  "checkpoint_sha256": digest, "source": match["source"], "seed": match["seed"]}
    if record.get("protocol") not in ("detected", "neutral") or record.get("checkpoint_sha256") != digest:
        raise ValueError("Invalid protocol or calibration belongs to a different checkpoint")
    if protocol and record["protocol"] != protocol:
        raise ValueError("Calibration uses a different singleton protocol")
    return {**record, "threshold": _threshold(record["threshold"])}


def _rows(section: SectionData, ids_path, eligible=None):
    if ids_path is None:
        return slice(None) if eligible is None else eligible
    names = Path(ids_path).read_text(encoding="utf-8").splitlines()
    if not names or any(not cell for cell in names) or len(set(names)) != len(names):
        raise ValueError("Expected one unique cell ID per line")
    lookup = {cell: i for i, cell in enumerate(section.cell_ids)}
    missing = next((cell for cell in names if cell not in lookup), None)
    if missing is not None:
        raise ValueError(f"Unknown cell ID in {section.section}: {missing!r}")
    rows = np.array([lookup[cell] for cell in names], dtype=np.int64)
    if eligible is not None and not np.isin(rows, eligible).all():
        raise ValueError("Calibration IDs must belong to the source spatial holdout")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for name in ("calibrate", "evaluate"):
        part = subs.add_parser(name)
        part.add_argument("--protocol", required=True, choices=("detected", "neutral"))
        part.add_argument("--checkpoint", required=True)
        part.add_argument("--data", required=True, help="Xenium section .h5ad with raw counts and H&E features")
        part.add_argument("--section", required=True, help="Section ID of --data")
        part.add_argument("--ids", help="Optional ordered cohort, one cell ID per line")
        part.add_argument("--output", required=True)
        part.add_argument("--device", default="auto")
        if name == "evaluate":
            part.add_argument("--calibration", help="Frozen calibration JSON or checkpoint registry")
            part.add_argument("--skip-kendall", action="store_true")
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    source = payload["source_section"]
    metadata = payload["metadata"]
    info = {"protocol": args.protocol, "source": source, "section": args.section,
            "checkpoint_sha256": _sha256(args.checkpoint), "seed": metadata.get("seed"),
            "training_protocol": metadata.get("protocol", "detected")}
    if args.ids:
        info["cohort_ids_sha256"] = _sha256(args.ids)
    if args.action == "calibrate" and (args.section != source or metadata.get("refit_full_source")):
        raise ValueError("Calibration requires the model's source section with a held-out spatial region")

    section = load_section(args.data, args.section)
    if section.gene_ids != list(payload["gene_ids"]):
        section = align_genes(section, payload["gene_ids"])
    eligible = None
    if args.action == "calibrate":
        eligible = spatial_split(section.coordinates, **metadata.get("split", {}))["validation"]
    rows = _rows(section, args.ids, eligible)
    counts = section.counts[rows]
    ranks, probabilities = predict_features(model, section.features[rows], device=device)
    info.update(n_cells=len(counts), n_genes=counts.shape[1])
    if args.action == "calibrate":
        if args.protocol == "neutral":
            choice = neutral_threshold(counts, probabilities)
        else:
            choice = select_threshold(counts, probabilities, beta=metadata.get("beta", 1.2),
                                      thresholds=np.linspace(0, 1, 101))
        report = {**info, **choice}
    else:
        if args.calibration:
            frozen = read_calibration(args.calibration, args.checkpoint, args.protocol)
            if frozen["source"] != source:
                raise ValueError("Calibration source does not match checkpoint")
            threshold = frozen["threshold"]
            info["threshold_source"] = "frozen calibration"
        elif args.protocol == info["training_protocol"]:
            threshold = _threshold(payload["threshold"])
            info["threshold_source"] = "checkpoint"
        else:
            raise ValueError("A separate frozen --calibration is required for this evaluation protocol")
        selection: dict[str, float | None] = dict(selection_metrics(counts, probabilities, threshold, args.protocol))
        report = {**info, "threshold": threshold, "selection": selection}
        if args.protocol == "detected":
            report.update(evaluate_scores(counts, ranks, probabilities, threshold=threshold,
                                          beta=metadata.get("beta", 1.2), kendall=not args.skip_kendall))
        else:
            selection["ndcg10"] = _neutral_ndcg10(counts, ranks, probabilities, threshold)
    write_json(args.output, report)
    print(f"Saved {args.protocol} {args.action} report to {args.output}")


if __name__ == "__main__":
    main()
