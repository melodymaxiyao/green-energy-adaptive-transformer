"""Reproduce the final-paper Daily+Weekly aggregation-mechanism ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    FINAL_PAPER_SEED,
    fit_neural_cell,
    metric_record,
    prepare_experiment_data,
    save_table,
)
from green_energy_transformer.data.universe import FORECAST_HORIZONS
from green_energy_transformer.models.multiscale import build_multiscale_model
from green_energy_transformer.training.train import MULTISCALE_PROTOCOL

MECHANISMS = ("single", "fixed", "static", "adaptive")
SINGLE_PATCHES = {
    5: (5, 2),
    10: (5, 2),
    20: (5, 2),
}


def build_mechanism_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    """Create the three descriptive comparisons from the cell results."""
    rows = []
    comparisons = (
        ("fixed_vs_single", "fixed", "single"),
        ("static_vs_fixed", "static", "fixed"),
        ("adaptive_vs_static", "adaptive", "static"),
    )
    for horizon in FORECAST_HORIZONS:
        horizon_label = f"{horizon}d"
        cells = results[results["horizon"] == horizon_label].set_index("mechanism")
        for label, mechanism, baseline in comparisons:
            rows.append(
                {
                    "comparison": label,
                    "horizon": horizon_label,
                    "mechanism": mechanism,
                    "baseline_mechanism": baseline,
                    "mechanism_mse": float(cells.loc[mechanism, "pooled_mse"]),
                    "baseline_mse": float(cells.loc[baseline, "pooled_mse"]),
                    "delta_mse_vs_baseline": float(
                        cells.loc[mechanism, "pooled_mse"]
                        - cells.loc[baseline, "pooled_mse"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_mechanism_ablation(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = prepare_experiment_data(processed_dir)
    rows = []
    for mechanism in MECHANISMS:
        for horizon in FORECAST_HORIZONS:
            daily_patch, weekly_patch = SINGLE_PATCHES[horizon]
            cell = fit_neural_cell(
                data,
                "daily_weekly",
                horizon,
                FINAL_PAPER_SEED,
                lambda mechanism=mechanism,
                daily_patch=daily_patch,
                weekly_patch=weekly_patch: build_multiscale_model(
                    "daily_weekly",
                    scale_mode=mechanism,
                    selected_daily_patch=daily_patch,
                    selected_weekly_patch=weekly_patch,
                ),
                MULTISCALE_PROTOCOL,
                "cpu",
            )
            row = metric_record(
                cell,
                mechanism,
                "daily_weekly",
                horizon,
                FINAL_PAPER_SEED,
            )
            row["mechanism"] = mechanism
            row["selected_daily_patch"] = daily_patch if mechanism == "single" else None
            row["selected_weekly_patch"] = weekly_patch if mechanism == "single" else None
            rows.append(row)
    results = pd.DataFrame(rows)
    return results, build_mechanism_comparisons(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mechanism_ablation.csv"),
    )
    parser.add_argument(
        "--comparisons-output",
        type=Path,
        default=Path("results/mechanism_comparisons.csv"),
    )
    args = parser.parse_args()
    results, comparisons = run_mechanism_ablation(args.processed_dir)
    save_table(results, args.output)
    save_table(comparisons, args.comparisons_output)
    print(results.to_string(index=False))
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
