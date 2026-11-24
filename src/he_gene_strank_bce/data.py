"""Load aligned Xenium counts and saved H&E features without AnnData.

H5ad's dataframe indices provide the shared row/column identities. This
loader preserves that ordering and never substitutes normalized ``X`` for
``layers/raw``. Independent feature files need their own identity join.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class SectionData:
    section: str
    path: Path
    features: np.ndarray
    counts: np.ndarray
    coordinates: np.ndarray
    cell_ids: np.ndarray
    gene_ids: list[str]
    gene_names: list[str]
    predictions: dict[str, np.ndarray]
    audit: dict[str, Any]

    @property
    def cell_keys(self) -> list[tuple[str, str]]:
        """Section-qualified keys: cell IDs may repeat across sections."""
        return [(self.section, str(cell_id)) for cell_id in self.cell_ids]


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _strings(node: h5py.Dataset | h5py.Group) -> np.ndarray:
    """Decode plain string arrays and H5ad categorical columns."""
    if isinstance(node, h5py.Dataset):
        values = np.asarray(node[...])
        if values.ndim != 1:
            raise ValueError(f"{node.name} must be a one-dimensional string column")
        if values.dtype.kind not in "OSU":
            raise ValueError(f"{node.name} must contain strings")
        return np.asarray([_text(value) for value in values], dtype=np.str_)
    encoding = _text(node.attrs.get("encoding-type", ""))
    if encoding != "categorical" or not {"codes", "categories"} <= set(node):
        raise ValueError(f"Unsupported string encoding in {node.name}: {encoding!r}")
    categories = _strings(node["categories"])
    codes = np.asarray(node["codes"][...])
    if codes.ndim != 1 or codes.dtype.kind not in "iu":
        raise ValueError(f"Invalid categorical codes in {node.name}")
    if np.any(codes < 0) or np.any(codes >= len(categories)):
        raise ValueError(f"Missing or invalid categorical values in {node.name}")
    return categories[codes]


def read_index(frame: h5py.Group) -> np.ndarray:
    index_key = _text(frame.attrs.get("_index", "_index"))
    if index_key not in frame:
        raise ValueError(f"Missing dataframe index {frame.name}/{index_key}")
    return _strings(frame[index_key])


def _matrix(node: h5py.Dataset | h5py.Group, *, raw: bool = False) -> np.ndarray:
    if isinstance(node, h5py.Dataset):
        values = np.asarray(node[...])
        if values.ndim != 2:
            raise ValueError(f"{node.name} must be a two-dimensional matrix")
        if raw:
            _validate_counts(values, node.name)
        return np.asarray(values, dtype=np.float32)
    encoding = _text(node.attrs.get("encoding-type", ""))
    if encoding not in {"csr_matrix", "csc_matrix"}:
        raise ValueError(f"Unsupported matrix encoding in {node.name}: {encoding!r}")
    if not {"data", "indices", "indptr"} <= set(node) or "shape" not in node.attrs:
        raise ValueError(f"Incomplete sparse matrix {node.name}")
    shape = tuple(int(value) for value in node.attrs["shape"])
    if len(shape) != 2 or any(value <= 0 for value in shape):
        raise ValueError(f"Invalid sparse matrix shape in {node.name}: {shape}")
    values = np.asarray(node["data"][...])
    indices = np.asarray(node["indices"][...])
    indptr = np.asarray(node["indptr"][...])
    if raw:
        _validate_counts(values, node.name)
    if values.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError(f"Sparse storage arrays must be one-dimensional in {node.name}")
    if indices.dtype.kind not in "iu" or indptr.dtype.kind not in "iu":
        raise ValueError(f"Sparse indices must be integers in {node.name}")
    indptr = indptr.astype(np.int64, copy=False)
    major_size, minor_size = shape if encoding == "csr_matrix" else shape[::-1]
    if (
        len(indptr) != major_size + 1
        or len(values) != len(indices)
        or indptr[0] != 0
        or indptr[-1] != len(values)
        or np.any(indptr[1:] < indptr[:-1])
        or np.any(indices < 0)
        or np.any(indices >= minor_size)
    ):
        raise ValueError(f"Invalid sparse storage arrays in {node.name}")
    constructor = sparse.csr_matrix if encoding == "csr_matrix" else sparse.csc_matrix
    matrix = constructor((values, indices, indptr), shape=shape)
    dense = matrix.toarray()
    if raw:
        _validate_counts(dense, node.name)
    return np.asarray(dense, dtype=np.float32)


def _validate_counts(values: np.ndarray, name: str) -> None:
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{name} raw counts must be numeric")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{name} raw counts must be finite and nonnegative")
    if np.any(values != np.rint(values)):
        raise ValueError(f"{name} raw counts must have integer semantics; normalized counts are invalid")


def _unique(values: np.ndarray | list[str], name: str) -> None:
    if any(not str(value).strip() for value in values):
        raise ValueError(f"{name} cannot contain empty identities")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


def load_section(
    path: str | Path,
    section: str,
    prediction_layers: list[str] | None = None,
) -> SectionData:
    """Load validated raw panel counts, UNI features, and optional predictions.

    The loader retains zero-total cells and applies no expression-derived
    filtering. Named binary prediction layers cannot be used for ranking.
    """
    path = Path(path).expanduser().resolve()
    if not isinstance(section, str) or not section.strip():
        raise ValueError("A nonempty section identity is required")
    predictions: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        required = ["obs", "var", "var/gene_ids", "var/feature_types", "obsm/he", "obsm/image_coor", "layers/raw"]
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"Missing required H5ad fields: {', '.join(missing)}")
        cell_ids = read_index(handle["obs"])
        gene_names = read_index(handle["var"]).tolist()
        gene_ids = _strings(handle["var/gene_ids"]).tolist()
        feature_types = _strings(handle["var/feature_types"])
        _unique(cell_ids, "Cell IDs within a section")
        _unique(gene_ids, "Stable gene IDs")
        if len(cell_ids) == 0 or len(gene_ids) == 0:
            raise ValueError("A section must contain cells and measured genes")
        if len(gene_names) != len(gene_ids) or len(feature_types) != len(gene_ids):
            raise ValueError("Gene metadata column lengths disagree")
        if not np.all(feature_types == "Gene Expression"):
            raise ValueError("Panel must contain only biological Gene Expression features; controls are not genes")
        features = _matrix(handle["obsm/he"])
        counts = _matrix(handle["layers/raw"], raw=True)
        coordinates = np.asarray(handle["obsm/image_coor"][...], dtype=np.float64)
        expected = (len(cell_ids), len(gene_ids))
        if counts.shape != expected:
            raise ValueError(f"Raw counts shape {counts.shape} disagrees with cell/gene identities {expected}")
        if features.shape[0] != expected[0] or features.shape[1] == 0:
            raise ValueError("H&E features must have one row per indexed cell and at least one feature")
        if coordinates.shape != (expected[0], 2):
            raise ValueError("Image coordinates must have shape [number of cells, 2]")
        if not np.isfinite(features).all() or not np.isfinite(coordinates).all():
            raise ValueError("H&E features and image coordinates must be finite")
        _validate_counts(counts, "layers/raw after float32 conversion")
        for name in prediction_layers or []:
            if "binary" in name.lower() or name == "raw":
                raise ValueError(f"{name!r} is not a continuous prediction layer")
            key = f"layers/{name}"
            if key not in handle:
                raise ValueError(f"Missing requested prediction layer {key}")
            prediction = _matrix(handle[key])
            if prediction.shape != expected or not np.isfinite(prediction).all():
                raise ValueError(f"Prediction layer {name!r} must be finite and aligned to raw counts {expected}")
            predictions[name] = prediction

    totals = counts.sum(axis=1, dtype=np.float64)
    audit = {
        "section": section,
        "path": str(path),
        "n_cells": len(cell_ids),
        "n_genes": len(gene_ids),
        "feature_dimension": features.shape[1],
        "counts_source": "layers/raw",
        "feature_source": "obsm/he",
        "coordinate_source": "obsm/image_coor",
        "coordinate_order": ["image_col", "image_row"],
        "row_alignment": "shared H5ad obs index; no external feature identity join",
        "zero_total_cells": int(np.count_nonzero(totals == 0)),
        "positive_fraction": float(np.count_nonzero(counts) / counts.size),
        "total_counts": float(totals.sum()),
        "prediction_layers": list(predictions),
    }
    return SectionData(section, path, features, counts, coordinates, cell_ids, gene_ids, gene_names, predictions, audit)


def align_genes(data: SectionData, gene_ids: list[str]) -> SectionData:
    """Reorder measured columns to a fixed panel; never impute absent genes."""
    target = list(gene_ids)
    _unique(target, "Target gene IDs")
    if not target:
        raise ValueError("Target gene panel cannot be empty")
    _unique(data.gene_ids, "Source stable gene IDs")
    lookup = {gene_id: index for index, gene_id in enumerate(data.gene_ids)}
    missing = [gene_id for gene_id in target if gene_id not in lookup]
    if missing:
        raise ValueError(f"Cannot align absent/unmeasured genes: {missing}")
    columns = np.asarray([lookup[gene_id] for gene_id in target], dtype=np.int64)
    counts = data.counts[:, columns]
    totals = counts.sum(axis=1, dtype=np.float64)
    audit = {
        **data.audit,
        "n_genes": len(target),
        "source_n_genes": len(data.gene_ids),
        "gene_alignment": "stable gene IDs",
        "zero_total_cells": int(np.count_nonzero(totals == 0)),
        "positive_fraction": float(np.count_nonzero(counts) / counts.size),
        "total_counts": float(totals.sum()),
    }
    return replace(
        data,
        counts=counts,
        gene_ids=target,
        gene_names=[data.gene_names[index] for index in columns],
        predictions={name: values[:, columns] for name, values in data.predictions.items()},
        audit=audit,
    )


def spatial_split(
    coordinates: np.ndarray,
    validation_fraction: float = 0.2,
    buffer_pixels: float = 64.0,
    axis: int = 0,
) -> dict[str, Any]:
    """Hold out a coordinate tail and exclude a symmetric buffer at its edge.

    Fraction defines the pre-buffer quantile, so the realized validation
    fraction can be smaller. Coordinate distance between the two retained
    regions is at least ``buffer_pixels`` along the selected image axis.
    This function uses coordinates only, without consulting counts.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] < 2:
        raise ValueError("Spatial split requires at least two [col, row] coordinates")
    if not np.isfinite(coordinates).all():
        raise ValueError("Spatial split coordinates must be finite")
    if axis not in (0, 1):
        raise ValueError("Spatial split axis must be 0 or 1")
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not np.isfinite(buffer_pixels) or buffer_pixels < 0:
        raise ValueError("buffer_pixels must be finite and nonnegative")
    positions = np.asarray(coordinates[:, axis], dtype=np.float64)
    cutoff = float(np.quantile(positions, 1.0 - validation_fraction))
    halfbuffer = buffer_pixels / 2.0
    train_mask = positions < cutoff - halfbuffer
    validation_mask = positions >= cutoff + halfbuffer
    train = np.flatnonzero(train_mask).astype(np.int64)
    validation = np.flatnonzero(validation_mask).astype(np.int64)
    buffer = np.flatnonzero(~(train_mask | validation_mask)).astype(np.int64)
    if train.size == 0 or validation.size == 0:
        raise ValueError("Spatial split produced an empty train or validation region; adjust the axis, fraction, or buffer")
    return {
        "train": train,
        "validation": validation,
        "buffer": buffer,
        "metadata": {
            "axis": axis,
            "validation_fraction_requested": float(validation_fraction),
            "buffer_pixels": float(buffer_pixels),
            "cutoff": cutoff,
            "train_upper_boundary": cutoff - halfbuffer,
            "validation_lower_boundary": cutoff + halfbuffer,
            "n_train": int(train.size),
            "n_validation": int(validation.size),
            "n_buffer": int(buffer.size),
            "validation_fraction_realized": float(validation.size / len(coordinates)),
            "minimum_axis_gap_pixels": float(positions[validation].min() - positions[train].max()),
        },
    }
