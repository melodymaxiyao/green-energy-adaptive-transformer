"""Reproduce the Adaptive model's Daily/Weekly input comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    FINAL_PAPER_SEED,
    INPUT_CONFIGURATIONS,
    fit_neural_cell,
    metric_record,
    prepare_experiment_data,
    save_table,
)
from green_energy_transformer.data.universe import FORECAST_HORIZONS
from green_energy_transformer.models.multiscale import build_multiscale_model
from green_energy_transformer.training.train import MULTISCALE_PROTOCOL


def run_input_configurations(processed_dir: Path) -> pd.DataFrame:
    data = prepare_experiment_data(processed_dir)
    rows = []
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/input_configurations.csv"),
    )
    args = parser.parse_args()
    results = run_input_configurations(args.processed_dir)
    save_table(results, args.output)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
