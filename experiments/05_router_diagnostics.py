"""Reproduce final-paper router diagnostics from experiment-04 outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import save_table
from green_energy_transformer.data.universe import FINAL_TICKERS, FORECAST_HORIZONS
from green_energy_transformer.evaluation.inference import (
    SEEDS,
    loss_difference_by_seed_date,
)
from green_energy_transformer.evaluation.router import (
    build_router_sample_metrics,
    cross_seed_stability,
    dominant_scale_usage,
    loss_association_by_seed,
    router_cell_summary,
    static_weight_references,
    summarize_cross_seed_stability,
    summarize_loss_associations,
)


def sample_loss_differences(predictions: pd.DataFrame) -> pd.DataFrame:
    """Delegate exact per-sample loss construction to the clean inference API."""
    rows = []
    for horizon in FORECAST_HORIZONS:
        label = f"{horizon}d"
        for seed in SEEDS:
            cell = predictions[
                (predictions["horizon"] == label)
                & (predictions["seed"] == seed)
            ]
            for ticker in FINAL_TICKERS:
                ticker_cell = cell[cell["ticker"] == ticker]
                loss = loss_difference_by_seed_date(
                    ticker_cell,
                    horizons=(label,),
                    seeds=(seed,),
                    expected_tickers=1,
                ).rename(columns={"mean_loss_diff": "loss_diff"})
                loss["ticker"] = ticker
                rows.append(
                    loss[["horizon", "seed", "ticker", "anchor_date", "loss_diff"]]
                )
    return pd.concat(rows, ignore_index=True)


def run_router_diagnostics(
    predictions: pd.DataFrame,
    router_weights: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    losses = sample_loss_differences(predictions)
    static_weights = router_weights[
        router_weights["mechanism"] == "static"
    ].drop(columns="mechanism")
    adaptive_weights = router_weights[
        router_weights["mechanism"] == "adaptive"
    ].drop(columns="mechanism")
    adaptive_weights["anchor_date"] = pd.to_datetime(
        adaptive_weights["anchor_date"]
    )
    adaptive_weights = adaptive_weights.merge(
        losses,
        on=["horizon", "seed", "ticker", "anchor_date"],
        how="left",
        validate="many_to_one",
    )
    if adaptive_weights["loss_diff"].isna().any():
        raise ValueError("Router weights could not be paired with all loss differences")

    references = static_weight_references(static_weights)
    sample_metrics = build_router_sample_metrics(adaptive_weights, references)
    stability = cross_seed_stability(sample_metrics)
    associations = loss_association_by_seed(sample_metrics)
    cell_summary = router_cell_summary(sample_metrics, references)
    leading = ["horizon", "seed", "frequency", "n_samples", "n_tickers", "n_dates"]
    weight_means = [f"mean_weight_{index}" for index in range(1, 5)]
    weight_sds = [f"sd_weight_{index}" for index in range(1, 5)]
    trailing = [
        column
        for column in cell_summary.columns
        if column not in leading + weight_means + weight_sds
    ]
    cell_summary = cell_summary[leading + weight_means + weight_sds + trailing]
    return {
        "router_diagnostics.csv": cell_summary,
        "router_dominant_scale_usage.csv": dominant_scale_usage(sample_metrics),
        "router_cross_seed_stability.csv": summarize_cross_seed_stability(stability),
        "router_loss_association_by_seed.csv": associations,
        "router_loss_association.csv": summarize_loss_associations(associations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/multiseed_predictions.csv"),
    )
    parser.add_argument(
        "--router-weights",
        type=Path,
        default=Path("results/multiseed_router_weights.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions, parse_dates=["anchor_date"])
    router_weights = pd.read_csv(args.router_weights, parse_dates=["anchor_date"])
    outputs = run_router_diagnostics(predictions, router_weights)
    for filename, table in outputs.items():
        save_table(table, args.output_dir / filename)
    for table in outputs.values():
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
