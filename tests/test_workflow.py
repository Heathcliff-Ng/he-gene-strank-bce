"""Run both training protocols, source calibration and label-free inference."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from he_gene_strank_bce.cli import main


def write_section(path: Path, section: str):
    n = 20
    with h5py.File(path, "w") as handle:
        strings = h5py.string_dtype()
        obs = handle.create_group("obs")
        obs.create_dataset("_index", dtype=strings, data=[section + str(i) for i in range(n)])
        var = handle.create_group("var")
        genes = ["a", "b", "c"] if section == "Rep1" else ["c", "a", "b"]
        var.create_dataset("_index", dtype=strings, data=genes)
        var.create_dataset("gene_ids", dtype=strings, data=genes)
        var.create_dataset("feature_types", dtype=strings, data=["Gene Expression"] * 3)
        handle["obsm/he"] = np.random.default_rng(4).standard_normal((n, 4)).astype("f4")
        handle["obsm/image_coor"] = np.column_stack((np.arange(n), np.zeros(n)))
        counts = np.stack([np.roll([0, 1, 2], i % 3) for i in range(n)])
        handle["layers/raw"] = counts if section == "Rep1" else counts[:, [2, 0, 1]]


@pytest.mark.parametrize("mode", ["detected", "neutral"])
def test_cli_from_training_to_prediction(tmp_path, mode):
    package_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((package_root / "configs/training.yaml").read_text())
    config["split"]["buffer_pixels"] = 0
    config["expected_genes"] = 3
    config["evaluation"]["top_ks"] = [1, 2]
    config["threads"] = 1
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    source, target = tmp_path / "Rep1.h5ad", tmp_path / "Rep2.h5ad"
    write_section(source, "Rep1")
    write_section(target, "Rep2")
    runs = tmp_path / "runs"
    main(["train", "--config", str(cfg_path), "--rep1", str(source), "--rep2", str(target),
          "--singletons", mode, "--source", "Rep1", "--seed", "42", "--epochs", "1",
          "--device", "cpu", "--output-dir", str(runs), "--skip-kendall"])
    checkpoint = runs / "Rep1_to_Rep2/seed_42/validation_model.pt"
    if mode == "detected":
        original = tmp_path / "detected.json"
        main(["evaluate", "--protocol", "detected", "--checkpoint", str(checkpoint),
              "--device", "cpu", "--data", str(target), "--section", "Rep2", "--output", str(original)])
        detected = json.loads(original.read_text())
        assert detected["threshold_source"] == "checkpoint"
        assert detected["selection"]["selected_ignored"] == 0
        assert detected["composition"]["positive_total_cells"] == 20
        assert 0 <= detected["detection"]["average_precision"] <= 1
    labels = ["--protocol", "neutral", "--checkpoint", str(checkpoint), "--device", "cpu"]
    calibration = tmp_path / "cal.json"
    main(["calibrate", *labels, "--data", str(source), "--section", "Rep1",
          "--output", str(calibration)])
    report_path = tmp_path / "result.json"
    main(["evaluate", *labels, "--data", str(target), "--section", "Rep2",
          "--calibration", str(calibration), "--output", str(report_path)])
    report = json.loads(report_path.read_text())
    assert report["training_protocol"] == mode
    assert report["protocol"] == "neutral"
    assert report["n_cells"] == 20 and report["n_genes"] == 3

    with h5py.File(target, "a") as handle:
        del handle["layers"]
    prediction = tmp_path / "pred.npz"
    main(["predict", *labels, "--features", str(target), "--section", "Rep2",
          "--calibration", str(calibration), "--output", str(prediction)])
    with np.load(prediction) as out:
        assert out["selected"].shape == (20, 3)
        assert out["training_protocol"].item() == mode
        assert out["protocol"].item() == "neutral"
        assert out["selected"].sum(axis=1).mean() == report["selection"]["selected"]
        assert out["gene_ids"].tolist() == ["a", "b", "c"]
