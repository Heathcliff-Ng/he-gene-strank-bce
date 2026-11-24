"""Fit the STRank+BCE model on one source section and evaluate on the other."""

import argparse
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml

from .data import SectionData, align_genes, load_section, spatial_split
from .evaluate import evaluate_scores, write_json
from .losses import multitask_loss
from .metrics import detection_metrics, select_threshold
from .model import GenePredictor, ModelSpec
from .predict import checkpoint_payload, predict_features, resolve_device
from .protocols import neutral_threshold, selection_metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_candidate(source: SectionData, split: dict, candidate: dict, settings: dict,
                  device: str, seed: int, singletons: str):
    seed_everything(seed)
    model = GenePredictor(ModelSpec(source.features.shape[1], len(source.gene_ids),
                                    candidate["hidden_dim"], candidate["dropout"])).to(device)
    train = split["train"]
    train_features = source.features[train]
    mean = train_features.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(train_features.std(0, dtype=np.float64).astype(np.float32), 1e-6)
    del train_features
    model.set_feature_statistics(torch.as_tensor(mean, device=device), torch.as_tensor(scale, device=device))
    train_counts = source.counts[train]
    model.initialize_intercepts(torch.as_tensor(train_counts, device=device), singletons=singletons)
    count_scale = float(train_counts.sum(1, dtype=np.float64,
                                         where=train_counts != 1 if singletons == "neutral" else True).mean())
    if count_scale <= 0 or not np.isfinite(count_scale):
        raise ValueError("Source training cells must contain retained positive counts")
    del train_counts
    features = torch.as_tensor(source.features, device=device)
    counts = torch.as_tensor(source.counts, device=device)
    validation = split["validation"]
    train_indices = torch.as_tensor(train, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    history = []
    best_score = -np.inf
    best_weights = None
    best_epoch = 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        shuffled = train_indices[torch.randperm(len(train_indices), device=device)]
        total_loss = 0.0
        for indices in shuffled.split(settings["batch_size"]):
            labels = counts[indices]
            if singletons == "neutral" and (labels == 1).all():
                continue
            count_logits, detection_logits = model(features[indices])
            loss, _ = multitask_loss(count_logits, detection_logits, labels,
                                     count_scale=count_scale, detection_weight=settings["detection_weight"],
                                     singletons=singletons, validate=False)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(indices)

        _, probabilities = predict_features(model, source.features[validation], device=device)
        if singletons == "neutral":
            cost = neutral_threshold(source.counts[validation], probabilities)["cost"]
            score = -cost
            monitor = {"fp_plus_2fn": cost}
            tolerance = 0.0
        else:
            ap = detection_metrics(source.counts[validation], probabilities, top_ks=())["average_precision"]
            if ap is None:
                raise ValueError("Validation AP is undefined on this source split")
            score = ap
            monitor = {"average_precision": ap}
            tolerance = 1e-6
        history.append({"epoch": epoch, "loss": total_loss / len(train), **monitor})
        print(f"{source.section} seed={seed} epoch={epoch} {monitor}", flush=True)
        if score > best_score + tolerance:
            best_score, best_epoch = score, epoch
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= settings["patience"]:
            break
    if best_weights is None:
        raise ValueError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_weights)
    return model.eval(), best_score, {
        "candidate": candidate, "best_epoch": best_epoch, "count_scale": count_scale, "history": history,
    }


def run_fold(source: SectionData, target: SectionData, config: dict, output: Path,
             device: str, seed: int, singletons: str):
    settings, evaluation = config["training"], config["evaluation"]
    split = spatial_split(source.coordinates, **config["split"])
    validation = split["validation"]
    np.savez_compressed(output / "split.npz", cell_ids=source.cell_ids,
                        train=split["train"], validation=validation, buffer=split["buffer"])
    candidates = []
    best = None
    for candidate in settings["candidates"]:
        model, score, report = fit_candidate(source, split, candidate, settings, device, seed, singletons)
        candidates.append(report)
        if best is None or score > best[0]:
            best = (score, model, report)
    if best is None:
        raise ValueError("At least one model candidate is required")
    _, model, report = best
    _, validation_probabilities = predict_features(model, source.features[validation], device=device)
    if singletons == "neutral":
        threshold = neutral_threshold(source.counts[validation], validation_probabilities)
        selection = "minimum source validation FP+2FN with raw=1 ignored"
    else:
        threshold = select_threshold(source.counts[validation], validation_probabilities, beta=settings["beta"],
                                     thresholds=np.linspace(0., 1., evaluation["threshold_grid_size"]))
        selection = "maximum source validation macro AP"

    metadata = {
        "seed": seed, "protocol": singletons, "singletons": singletons, "loss_mode": "multinomial_bce",
        "beta": settings["beta"], "detection_weight": settings["detection_weight"],
        "count_scale": report["count_scale"], "best_epoch": report["best_epoch"],
        "model_selection": selection, "refit_full_source": False, "reference_weight": 0.0,
        "threshold_selection": selection if singletons == "neutral" else "source validation macro F_beta",
        "feature_key": "obsm/he", "source_file": str(source.path), "score_interpretation": "count ranking",
        "split": config["split"],
    }
    payload = checkpoint_payload(model, gene_ids=source.gene_ids, gene_names=source.gene_names,
                                 threshold=threshold["threshold"], source_section=source.section, metadata=metadata)
    torch.save(payload, output / "validation_model.pt")
    scores, probabilities = predict_features(model, target.features, device=device)
    if singletons == "detected":
        metrics = evaluate_scores(target.counts, scores, probabilities, threshold=threshold["threshold"],
                                  beta=settings["beta"], top_ks=evaluation["top_ks"], kendall=evaluation["kendall"])
    else:
        metrics = selection_metrics(target.counts, probabilities, threshold["threshold"], singletons)
    write_json(output / "report.json", {
        "source": source.section, "target": target.section, "metadata": metadata,
        "split": split["metadata"], "candidates": candidates, "threshold": threshold, "metrics": metrics,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml", type=Path)
    parser.add_argument("--rep1", "--rep1-h5ad", type=Path, help="Override the Rep1 H5AD path.")
    parser.add_argument("--rep2", "--rep2-h5ad", type=Path, help="Override the Rep2 H5AD path.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source", choices=["Rep1", "Rep2"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--singletons", choices=["detected", "neutral"])
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--skip-kendall", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = args.config.resolve().parent.parent
    singletons = args.singletons or config.get("singletons", "detected")
    if singletons not in {"detected", "neutral"}:
        parser.error("singletons must be detected or neutral")
    if config["training"].get("refit_full_source", False):
        parser.error("Training uses a held-out validation partition")
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.skip_kendall:
        config["evaluation"]["kendall"] = False
    if any(config["training"][k] < 1 for k in ("epochs", "batch_size", "patience")):
        parser.error("epochs, batch_size and patience must be positive")
    output = args.output_dir or base / config.get("runs_dir", "runs") / singletons
    device = resolve_device(args.device or config.get("device", "auto"))
    config.update(singletons=singletons, output_dir=str(output.resolve()), device=str(device))
    torch.set_num_threads(config.get("threads", 8))
    sections = {name: load_section(override or base / config["sections"][name]["path"], name)
                for name, override in (("Rep1", args.rep1), ("Rep2", args.rep2))}
    panel = sections["Rep1"].gene_ids
    if len(panel) != config["expected_genes"] or set(panel) != set(sections["Rep2"].gene_ids):
        raise ValueError("Input sections must share the fixed gene panel")
    sections = {name: align_genes(data, panel) for name, data in sections.items()}
    seeds = [args.seed] if args.seed is not None else config["seeds"]
    config["seeds"] = seeds
    for name, data in sections.items():
        config["sections"][name]["path"] = str(data.path)
    for name in [args.source] if args.source else sections:
        other = "Rep2" if name == "Rep1" else "Rep1"
        for seed in seeds:
            fold = output / f"{name}_to_{other}" / f"seed_{seed}"
            fold.parent.mkdir(parents=True, exist_ok=True)
            if fold.exists():
                raise FileExistsError(f"Run already exists: {fold}. Select a different --output-dir.")
            with TemporaryDirectory(dir=fold.parent, prefix=f".seed_{seed}_") as temporary:
                stage = Path(temporary)
                run_fold(sections[name], sections[other], config, stage, device, seed, singletons)
                write_json(stage / "config.json", config)
                stage.rename(fold)
            print(f"Saved {fold}", flush=True)


if __name__ == "__main__":
    main()
