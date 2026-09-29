"""Reproduce the final-paper benchmark table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    FINAL_PAPER_SEED,
    INPUT_CONFIGURATIONS,
    fit_neural_cell,
    metric_record,
    prepare_experiment_data,
    save_table,
    targets_for,
)
from green_energy_transformer.data.universe import FORECAST_HORIZONS
from green_energy_transformer.evaluation.metrics import evaluate_predictions
from green_energy_transformer.models.baselines import (
    build_lstm_model,
    build_swim_model,
    build_transformer_model,
    fit_ridge_with_validation,
    flatten_ridge_features,
)
from green_energy_transformer.models.multiscale import build_multiscale_model
from green_energy_transformer.training.train import (
    MULTISCALE_PROTOCOL,
    NEURAL_BASELINE_PROTOCOL,
)


def _forecast_row(
    model: str,
    frequency: str,
    horizon: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ticker: np.ndarray,
    anchor_date: np.ndarray,
    **extra,
) -> dict[str, object]:
    return {
        "model": model,
        "frequency_setting": frequency,
        "horizon": f"{horizon}d",
        "seed": None,
        **evaluate_predictions(y_true, y_pred, ticker, anchor_date),
        **extra,
    }


def run_benchmark(processed_dir: Path) -> pd.DataFrame:
    data = prepare_experiment_data(processed_dir)
    test_mask = data.masks["test"]
    rows: list[dict[str, object]] = []

    for horizon in FORECAST_HORIZONS:
        y_train = targets_for(data, horizon, "train")
        y_test = targets_for(data, horizon, "test")
        for model, prediction, value in (
            ("zero", np.zeros_like(y_test), np.nan),
            (
                "train_mean",
                np.full_like(y_test, float(np.mean(y_train))),
                float(np.mean(y_train)),
            ),
        ):
            rows.append(
                _forecast_row(
                    model,
                    "n/a",
                    horizon,
                    y_test,
                    prediction,
                    data.panel.ticker[test_mask],
                    data.panel.anchor_date[test_mask],
                    train_mean_value=value,
                )
            )

    for frequency in INPUT_CONFIGURATIONS:
        flat = flatten_ridge_features(frequency, data.normalized[frequency])
        for horizon in FORECAST_HORIZONS:
            model, alpha, validation_mse = fit_ridge_with_validation(
                flat[data.masks["train"]],
                targets_for(data, horizon, "train"),
                flat[data.masks["val"]],
                targets_for(data, horizon, "val"),
            )
            rows.append(
                _forecast_row(
                    "ridge",
                    frequency,
                    horizon,
                    targets_for(data, horizon, "test"),
                    model.predict(flat[test_mask]),
                    data.panel.ticker[test_mask],
                    data.panel.anchor_date[test_mask],
                    best_alpha=alpha,
                    best_validation_mse=validation_mse,
                )
            )

    neural_models = (
        ("lstm", build_lstm_model, "auto"),
        ("vanilla_transformer", build_transformer_model, "auto"),
        ("swim_transformer", build_swim_model, "cpu"),
    )
    for model_name, builder, device in neural_models:
        for frequency in INPUT_CONFIGURATIONS:
            for horizon in FORECAST_HORIZONS:
                cell = fit_neural_cell(
                    data,
                    frequency,
                    horizon,
                    FINAL_PAPER_SEED,
                    lambda builder=builder, frequency=frequency: builder(frequency),
                    NEURAL_BASELINE_PROTOCOL,
                    device,
                )
                rows.append(
                    metric_record(
                        cell,
                        model_name,
                        frequency,
                        horizon,
                        FINAL_PAPER_SEED,
                    )
                )

    for frequency in INPUT_CONFIGURATIONS:
        for horizon in FORECAST_HORIZONS:
            cell = fit_neural_cell(
                data,
                frequency,
                horizon,
                FINAL_PAPER_SEED,
                lambda frequency=frequency: build_multiscale_model(
                    frequency,
                    scale_mode="adaptive",
                ),
                MULTISCALE_PROTOCOL,
                "cpu",
            )
            rows.append(
                metric_record(
                    cell,
                    "adaptive_multiscale_transformer",
                    frequency,
                    horizon,
                    FINAL_PAPER_SEED,
                )
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.csv"))
    args = parser.parse_args()
    results = run_benchmark(args.processed_dir)
    save_table(results, args.output)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
