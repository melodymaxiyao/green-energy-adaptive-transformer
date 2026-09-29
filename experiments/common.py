"""Shared orchestration helpers for the public final-paper experiments."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.panel import (  # noqa: E402
    PanelDataset,
    apply_normalizer,
    fit_train_only_normalizer,
    load_final_panel,
)
from evaluation.metrics import evaluate_predictions  # noqa: E402
from training.train import (  # noqa: E402
    ModelInputs,
    TrainingProtocol,
    TrainingResult,
    resolve_device,
    set_seed,
    train_model,
)

FINAL_PAPER_SEED = 42
INPUT_CONFIGURATIONS = ("daily_only", "weekly_only", "daily_weekly")


@dataclass(frozen=True)
class ExperimentData:
    panel: PanelDataset
    normalized: dict[str, ModelInputs]
    masks: dict[str, np.ndarray]


@dataclass(frozen=True)
class CellResult:
    training: TrainingResult
    y_true: np.ndarray
    y_pred: np.ndarray
    ticker: np.ndarray
    anchor_date: np.ndarray


def prepare_experiment_data(processed_dir: Path) -> ExperimentData:
    """Load the final panel and normalize every input configuration once."""
    panel = load_final_panel(processed_dir)
    statistics = fit_train_only_normalizer(panel)
    normalized = {
        frequency: apply_normalizer(panel, statistics, frequency)
        for frequency in INPUT_CONFIGURATIONS
    }
    masks = {
        split: panel.split == split
        for split in ("train", "val", "test")
    }
    return ExperimentData(panel, normalized, masks)


def subset_inputs(inputs: ModelInputs, mask: np.ndarray) -> ModelInputs:
    if isinstance(inputs, tuple):
        return tuple(values[mask] for values in inputs)
    return inputs[mask]


def inputs_for(
    data: ExperimentData,
    frequency: str,
    split: str,
) -> ModelInputs:
    return subset_inputs(data.normalized[frequency], data.masks[split])


def targets_for(data: ExperimentData, horizon: int, split: str) -> np.ndarray:
    return data.panel.targets[horizon][data.masks[split]]


def _input_tuple(inputs: ModelInputs) -> tuple[np.ndarray | torch.Tensor, ...]:
    return inputs if isinstance(inputs, tuple) else (inputs,)


def predict(
    model: nn.Module,
    inputs: ModelInputs,
    device: str | torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run ordered batched prediction without duplicating evaluation logic."""
    resolved = resolve_device(device)
    tensors = tuple(
        torch.as_tensor(values, dtype=torch.float32)
        for values in _input_tuple(inputs)
    )
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False)
    chunks = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            chunks.append(model(*(value.to(resolved) for value in batch)).cpu())
    return torch.cat(chunks).numpy().astype(np.float32, copy=False)


def predict_with_router(
    model: nn.Module,
    inputs: ModelInputs,
    frequency: str,
    device: str | torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return ordered predictions and per-branch router allocations."""
    resolved = resolve_device(device)
    tensors = tuple(
        torch.as_tensor(values, dtype=torch.float32)
        for values in _input_tuple(inputs)
    )
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False)
    prediction_chunks: list[torch.Tensor] = []
    weight_chunks: dict[str, list[torch.Tensor]] = {
        branch: []
        for branch in (
            ("daily", "weekly")
            if frequency == "daily_weekly"
            else ("daily",) if frequency == "daily_only" else ("weekly",)
        )
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model.forward_with_router(*(value.to(resolved) for value in batch))
            prediction_chunks.append(output[0].cpu())
            for branch, weights in zip(weight_chunks, output[1:]):
                weight_chunks[branch].append(weights.cpu())
    predictions = torch.cat(prediction_chunks).numpy().astype(np.float32, copy=False)
    weights = {
        branch: torch.cat(chunks).numpy().astype(np.float64, copy=False)
        for branch, chunks in weight_chunks.items()
    }
    return predictions, weights


def fit_neural_cell(
    data: ExperimentData,
    frequency: str,
    horizon: int,
    seed: int,
    model_factory: Callable[[], nn.Module],
    protocol: TrainingProtocol,
    device: str | torch.device,
) -> CellResult:
    """Construct, train, restore, and evaluate one final-paper model cell."""
    set_seed(seed)
    model = model_factory()
    training = train_model(
        model,
        inputs_for(data, frequency, "train"),
        targets_for(data, horizon, "train"),
        inputs_for(data, frequency, "val"),
        targets_for(data, horizon, "val"),
        protocol,
        device,
    )
    test_mask = data.masks["test"]
    prediction = predict(
        training.model,
        inputs_for(data, frequency, "test"),
        device,
        protocol.batch_size,
    )
    return CellResult(
        training=training,
        y_true=data.panel.targets[horizon][test_mask],
        y_pred=prediction,
        ticker=data.panel.ticker[test_mask],
        anchor_date=data.panel.anchor_date[test_mask],
    )


def metric_record(
    cell: CellResult,
    model: str,
    frequency: str,
    horizon: int,
    seed: int | None,
) -> dict[str, object]:
    """Create one tidy result row from the clean forecast metrics."""
    metrics = evaluate_predictions(
        cell.y_true,
        cell.y_pred,
        cell.ticker,
        cell.anchor_date,
    )
    return {
        "model": model,
        "frequency_setting": frequency,
        "horizon": f"{horizon}d",
        "seed": seed,
        "best_epoch": cell.training.best_epoch,
        "epochs_trained": cell.training.epochs_trained,
        "best_validation_loss": cell.training.best_validation_loss,
        **metrics,
    }


def prediction_frame(
    cell: CellResult,
    mechanism: str,
    horizon: int,
    seed: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mechanism": mechanism,
            "horizon": f"{horizon}d",
            "seed": seed,
            "ticker": cell.ticker,
            "anchor_date": cell.anchor_date,
            "y_true": cell.y_true,
            "y_pred": cell.y_pred,
        }
    )


def router_weight_frame(
    cell: CellResult,
    weights: dict[str, np.ndarray],
    mechanism: str,
    horizon: int,
    seed: int,
) -> pd.DataFrame:
    """Put model-emitted router weights into the clean diagnostic schema."""
    frames = []
    for frequency, values in weights.items():
        if values.shape != (len(cell.y_true), 4):
            raise ValueError(
                f"Expected four {frequency} weights per test sample; got {values.shape}"
            )
        frame = pd.DataFrame(values, columns=[f"weight_{i}" for i in range(1, 5)])
        frame.insert(0, "frequency", frequency)
        frame.insert(0, "anchor_date", cell.anchor_date)
        frame.insert(0, "ticker", cell.ticker)
        frame.insert(0, "seed", seed)
        frame.insert(0, "horizon", f"{horizon}d")
        frame.insert(0, "mechanism", mechanism)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def save_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
