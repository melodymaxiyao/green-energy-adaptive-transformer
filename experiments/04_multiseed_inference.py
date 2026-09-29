"""Reproduce five-seed Adaptive-vs-Static robustness and inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    fit_neural_cell,
    inputs_for,
    metric_record,
    predict_with_router,
    prediction_frame,
    prepare_experiment_data,
    router_weight_frame,
    save_table,
)
from data.universe import FORECAST_HORIZONS
from evaluation.inference import (
    SEEDS,
    bootstrap_inference,
    hac_inference,
    loss_difference_by_date,
    loss_difference_by_seed_date,
)
from models.multiscale import build_multiscale_model
from training.train import MULTISCALE_PROTOCOL

MECHANISMS = ("adaptive", "static")


def paired_seed_results(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in FORECAST_HORIZONS:
        label = f"{horizon}d"
        for seed in SEEDS:
            cells = cell_metrics[
                (cell_metrics["horizon"] == label)
                & (cell_metrics["seed"] == seed)
            ].set_index("mechanism")
            adaptive_mse = float(cells.loc["adaptive", "pooled_mse"])
            static_mse = float(cells.loc["static", "pooled_mse"])
            rows.append(
                {
                    "horizon": label,
                    "seed": seed,
                    "adaptive_mse": adaptive_mse,
                    "static_mse": static_mse,
                    "delta_mse": adaptive_mse - static_mse,
                }
            )
    return pd.DataFrame(rows)


def summarize_multiseed(
    cell_metrics: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize MSE with the frozen across-seed ddof=1 convention."""
    rows = []
    for horizon in FORECAST_HORIZONS:
        label = f"{horizon}d"
        cells = cell_metrics[cell_metrics["horizon"] == label]
        adaptive = cells[cells["mechanism"] == "adaptive"]["pooled_mse"]
        static = cells[cells["mechanism"] == "static"]["pooled_mse"]
        delta = pairs[pairs["horizon"] == label]["delta_mse"]
        rows.append(
            {
                "horizon": label,
                "adaptive_mean_mse": float(adaptive.mean()),
                "adaptive_sd_mse": float(adaptive.std(ddof=1)),
                "static_mean_mse": float(static.mean()),
                "static_sd_mse": float(static.std(ddof=1)),
                "mean_delta_mse": float(delta.mean()),
                "sd_delta_mse": float(delta.std(ddof=1)),
                "adaptive_mse_wins": int((delta < 0).sum()),
                "static_mse_wins": int((delta > 0).sum()),
                "mse_ties": int((delta == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def run_multiseed(
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = prepare_experiment_data(processed_dir)
    metric_rows = []
    prediction_tables = []
    router_tables = []
    for mechanism in MECHANISMS:
        for horizon in FORECAST_HORIZONS:
            for seed in SEEDS:
                cell = fit_neural_cell(
                    data,
                    "daily_weekly",
                    horizon,
                    seed,
                    lambda mechanism=mechanism: build_multiscale_model(
                        "daily_weekly",
                        scale_mode=mechanism,
                    ),
                    MULTISCALE_PROTOCOL,
                    "cpu",
                )
                row = metric_record(
                    cell,
                    mechanism,
                    "daily_weekly",
                    horizon,
                    seed,
                )
                row["mechanism"] = mechanism
                metric_rows.append(row)
                prediction_tables.append(
                    prediction_frame(cell, mechanism, horizon, seed)
                )
                router_prediction, weights = predict_with_router(
                    cell.training.model,
                    inputs_for(data, "daily_weekly", "test"),
                    "daily_weekly",
                    "cpu",
                    MULTISCALE_PROTOCOL.batch_size,
                )
                if not (router_prediction == cell.y_pred).all():
                    raise RuntimeError("Router pass changed the restored model predictions")
                router_tables.append(
                    router_weight_frame(cell, weights, mechanism, horizon, seed)
                )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_tables, ignore_index=True),
        pd.concat(router_tables, ignore_index=True),
    )


def infer_from_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_date = loss_difference_by_seed_date(predictions)
    date_difference = loss_difference_by_date(seed_date)
    _, bootstrap = bootstrap_inference(date_difference)
    hac = hac_inference(date_difference)
    return seed_date, date_difference, bootstrap, hac


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    cells, predictions, router_weights = run_multiseed(args.processed_dir)
    pairs = paired_seed_results(cells)
    summary = summarize_multiseed(cells, pairs)
    seed_date, date_difference, bootstrap, hac = infer_from_predictions(predictions)

    outputs = {
        "multiseed_cells.csv": cells,
        "multiseed_predictions.csv": predictions,
        "multiseed_router_weights.csv": router_weights,
        "multiseed_pairs.csv": pairs,
        "multiseed_summary.csv": summary,
        "loss_difference_by_seed_date.csv": seed_date,
        "loss_difference_by_date.csv": date_difference,
        "inference.csv": bootstrap,
        "hac_dm.csv": hac,
    }
    for filename, table in outputs.items():
        save_table(table, args.output_dir / filename)
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(hac.to_string(index=False))


if __name__ == "__main__":
    main()
