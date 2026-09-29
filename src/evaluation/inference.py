"""Final-paper Adaptive-minus-Static loss inference."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

HORIZONS = ("5d", "10d", "20d")
HORIZON_INTS = {"5d": 5, "10d": 10, "20d": 20}
SEEDS = (0, 1, 21, 42, 3407)
BOOTSTRAP_REPLICATIONS = 20_000
BOOTSTRAP_BASE_SEED = 20_260_826
PRIMARY_BLOCK_LENGTHS = {"5d": 5, "10d": 10, "20d": 20}
SENSITIVITY_BLOCK_LENGTHS = {"5d": 10, "10d": 20, "20d": 40}
HAC_MAX_LAGS = {"5d": 4, "10d": 9, "20d": 19}
ALPHA = 0.05


def loss_difference_by_seed_date(
    predictions: pd.DataFrame,
    horizons: tuple[str, ...] = HORIZONS,
    seeds: tuple[int, ...] = SEEDS,
    expected_tickers: int | None = 17,
) -> pd.DataFrame:
    """Build seed/date means of float32 Adaptive-minus-Static squared loss."""
    required = {
        "mechanism",
        "horizon",
        "seed",
        "ticker",
        "anchor_date",
        "y_true",
        "y_pred",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")

    rows = []
    float32_tolerance = np.finfo(np.float32).eps
    for horizon in horizons:
        for seed in seeds:
            cells = {}
            for mechanism in ("adaptive", "static"):
                cell = predictions[
                    (predictions["mechanism"] == mechanism)
                    & (predictions["horizon"] == horizon)
                    & (predictions["seed"] == seed)
                ].copy()
                cell["anchor_date"] = pd.to_datetime(cell["anchor_date"])
                cell = cell.sort_values(
                    ["ticker", "anchor_date"],
                    kind="mergesort",
                ).reset_index(drop=True)
                if cell.empty:
                    raise ValueError(f"Missing {mechanism}/{horizon}/seed={seed} cell")
                if cell.duplicated(["ticker", "anchor_date"]).any():
                    raise ValueError(
                        f"Duplicate ticker/date rows in {mechanism}/{horizon}/seed={seed}"
                    )
                cells[mechanism] = cell

            adaptive = cells["adaptive"]
            static = cells["static"]
            if not adaptive[["ticker", "anchor_date"]].equals(
                static[["ticker", "anchor_date"]]
            ):
                raise ValueError(f"Adaptive/Static key mismatch for {horizon}/seed={seed}")

            y_true = adaptive["y_true"].to_numpy(dtype=np.float32)
            static_y_true = static["y_true"].to_numpy(dtype=np.float32)
            if not np.array_equal(y_true, static_y_true) and not np.allclose(
                y_true,
                static_y_true,
                rtol=0.0,
                atol=float32_tolerance,
                equal_nan=False,
            ):
                raise ValueError(f"Adaptive/Static target mismatch for {horizon}/seed={seed}")

            adaptive_prediction = adaptive["y_pred"].to_numpy(dtype=np.float32)
            static_prediction = static["y_pred"].to_numpy(dtype=np.float32)
            adaptive_error = (y_true - adaptive_prediction) ** 2
            static_error = (y_true - static_prediction) ** 2
            loss_difference = (adaptive_error - static_error).astype(np.float64)
            if not np.isfinite(loss_difference).all():
                raise ValueError(f"Non-finite loss difference for {horizon}/seed={seed}")

            sample = pd.DataFrame(
                {
                    "anchor_date": adaptive["anchor_date"].to_numpy(),
                    "loss_diff": loss_difference,
                }
            )
            grouped = (
                sample.groupby("anchor_date")["loss_diff"]
                .agg(["mean", "count"])
                .reset_index()
            )
            if expected_tickers is not None and not grouped["count"].eq(
                expected_tickers
            ).all():
                raise ValueError(
                    f"Expected {expected_tickers} stocks per date for {horizon}/seed={seed}"
                )
            grouped["horizon"] = horizon
            grouped["seed"] = int(seed)
            grouped = grouped.rename(
                columns={"mean": "mean_loss_diff", "count": "n_tickers"}
            )
            rows.append(
                grouped[
                    ["horizon", "seed", "anchor_date", "mean_loss_diff", "n_tickers"]
                ]
            )

    result = pd.concat(rows, ignore_index=True)
    result["horizon"] = pd.Categorical(
        result["horizon"],
        categories=list(horizons),
        ordered=True,
    )
    result["seed"] = pd.Categorical(
        result["seed"],
        categories=list(seeds),
        ordered=True,
    )
    result = result.sort_values(
        ["horizon", "seed", "anchor_date"]
    ).reset_index(drop=True)
    result["horizon"] = result["horizon"].astype(str)
    result["seed"] = result["seed"].astype(int)
    return result


def loss_difference_by_date(
    seed_date: pd.DataFrame,
    seeds: tuple[int, ...] = SEEDS,
    horizons: tuple[str, ...] = HORIZONS,
) -> pd.DataFrame:
    """Average seed/date loss differences equally across the five seeds."""
    result = (
        seed_date.groupby(["horizon", "anchor_date"])["mean_loss_diff"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_loss_diff", "count": "n_seeds"})
    )
    if not result["n_seeds"].eq(len(seeds)).all():
        raise ValueError(f"Each date must contain exactly {len(seeds)} seeds")
    result["horizon"] = pd.Categorical(
        result["horizon"],
        categories=list(horizons),
        ordered=True,
    )
    result = result.sort_values(["horizon", "anchor_date"]).reset_index(drop=True)
    result["horizon"] = result["horizon"].astype(str)
    return result[["horizon", "anchor_date", "mean_loss_diff", "n_seeds"]]


def moving_block_bootstrap(
    series: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
    replications: int = BOOTSTRAP_REPLICATIONS,
) -> np.ndarray:
    """Return fixed-length moving-block-bootstrap sample means."""
    series = np.asarray(series, dtype=np.float64)
    sample_size = series.shape[0]
    block_length = int(block_length)
    possible_starts = sample_size - block_length + 1
    if possible_starts < 1:
        raise ValueError(
            f"Invalid block length {block_length} for series length {sample_size}"
        )
    blocks_per_replication = math.ceil(sample_size / block_length)
    starts = rng.integers(
        0,
        possible_starts,
        size=(replications, blocks_per_replication),
    )
    offsets = np.arange(block_length)
    indices = starts[:, :, None] + offsets[None, None, :]
    indices = indices.reshape(
        replications,
        blocks_per_replication * block_length,
    )[:, :sample_size]
    return series[indices].mean(axis=1, dtype=np.float64)


def holm_adjust_three(pvalues: dict[str, float]) -> dict[str, float]:
    """Apply the frozen step-down Holm adjustment to exactly three tests."""
    if len(pvalues) != 3:
        raise ValueError("Holm adjustment requires exactly three p-values")
    order = sorted(pvalues, key=lambda key: pvalues[key])
    running_max = 0.0
    adjusted = {}
    for rank, key in enumerate(order, start=1):
        candidate = (len(order) - rank + 1) * pvalues[key]
        running_max = max(running_max, candidate)
        adjusted[key] = min(running_max, 1.0)
    return adjusted


def _mbb_classification(lower: float, upper: float) -> str:
    if lower > 0:
        return "MBB_STATIC_LOWER_LOSS"
    if upper < 0:
        return "MBB_ADAPTIVE_LOWER_LOSS"
    return "MBB_NO_DISTINGUISHABLE_DIFFERENCE"


def bootstrap_inference(
    date_loss_difference: pd.DataFrame,
    replications: int = BOOTSTRAP_REPLICATIONS,
    base_seed: int = BOOTSTRAP_BASE_SEED,
    horizons: tuple[str, ...] = HORIZONS,
    primary_blocks: dict[str, int] = PRIMARY_BLOCK_LENGTHS,
    sensitivity_blocks: dict[str, int] = SENSITIVITY_BLOCK_LENGTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run primary and sensitivity MBB inference for all horizons."""
    replicate_rows = []
    summary_rows = []
    primary_pvalues = {}
    specifications = (
        ("PRIMARY", primary_blocks),
        ("SENSITIVITY", sensitivity_blocks),
    )

    for horizon in horizons:
        series = date_loss_difference[
            date_loss_difference["horizon"] == horizon
        ].sort_values("anchor_date")["mean_loss_diff"].to_numpy(dtype=np.float64)
        if series.size == 0:
            raise ValueError(f"Missing date-level loss series for {horizon}")
        point_estimate = float(series.mean())
        for specification, blocks in specifications:
            block_length = blocks[horizon]
            seed_sequence = np.random.SeedSequence(
                [base_seed, HORIZON_INTS[horizon], block_length]
            )
            bootstrap_means = moving_block_bootstrap(
                series,
                block_length,
                np.random.default_rng(seed_sequence),
                replications,
            )
            bootstrap_mean = float(np.mean(bootstrap_means))
            bootstrap_se = float(np.std(bootstrap_means, ddof=1))
            lower, upper = np.quantile(
                bootstrap_means,
                [0.025, 0.975],
                method="linear",
            )
            centered_pvalue = (
                1
                + int(
                    np.sum(
                        np.abs(bootstrap_means - point_estimate)
                        >= abs(point_estimate)
                    )
                )
            ) / (replications + 1)
            if specification == "PRIMARY":
                primary_pvalues[horizon] = centered_pvalue

            replicate_rows.extend(
                {
                    "horizon": horizon,
                    "block_length": block_length,
                    "specification": specification,
                    "replicate": index,
                    "bootstrap_mean_loss_diff": float(value),
                }
                for index, value in enumerate(bootstrap_means)
            )
            summary_rows.append(
                {
                    "horizon": horizon,
                    "block_length": block_length,
                    "specification": specification,
                    "n_dates": len(series),
                    "n_bootstrap": replications,
                    "theta_hat": point_estimate,
                    "bootstrap_mean": bootstrap_mean,
                    "bootstrap_se": bootstrap_se,
                    "ci_lower_95": float(lower),
                    "ci_upper_95": float(upper),
                    "centered_p_value": centered_pvalue,
                    "classification": _mbb_classification(float(lower), float(upper)),
                }
            )

    adjusted = holm_adjust_three(primary_pvalues)
    for row in summary_rows:
        row["centered_p_value_holm_primary"] = (
            adjusted[row["horizon"]]
            if row["specification"] == "PRIMARY"
            else float("nan")
        )

    replicates = pd.DataFrame(replicate_rows)
    summary = pd.DataFrame(summary_rows)
    specification_order = ["PRIMARY", "SENSITIVITY"]
    for frame in (replicates, summary):
        frame["horizon"] = pd.Categorical(
            frame["horizon"], categories=list(horizons), ordered=True
        )
        frame["specification"] = pd.Categorical(
            frame["specification"],
            categories=specification_order,
            ordered=True,
        )
    replicates = replicates.sort_values(
        ["horizon", "specification", "replicate"]
    ).reset_index(drop=True)
    summary = summary.sort_values(["horizon", "specification"]).reset_index(drop=True)
    for frame in (replicates, summary):
        frame["horizon"] = frame["horizon"].astype(str)
        frame["specification"] = frame["specification"].astype(str)
    return replicates, summary[
        [
            "horizon",
            "block_length",
            "specification",
            "n_dates",
            "n_bootstrap",
            "theta_hat",
            "bootstrap_mean",
            "bootstrap_se",
            "ci_lower_95",
            "ci_upper_95",
            "centered_p_value",
            "centered_p_value_holm_primary",
            "classification",
        ]
    ]


def hac_dm_test(series: np.ndarray, max_lag: int) -> dict[str, float]:
    """Bartlett-weighted HAC mean-loss-differential (DM-style) diagnostic."""
    series = np.asarray(series, dtype=np.float64)
    sample_size = series.shape[0]
    mean = float(series.mean())
    centered = series - mean
    long_run_variance = float(np.sum(centered * centered) / sample_size)
    for lag in range(1, max_lag + 1):
        autocovariance = float(
            np.sum(centered[lag:] * centered[:-lag]) / sample_size
        )
        bartlett_weight = 1.0 - lag / (max_lag + 1)
        long_run_variance += 2.0 * bartlett_weight * autocovariance
    if not np.isfinite(long_run_variance) or long_run_variance <= 0:
        raise ValueError(f"Invalid HAC long-run variance: {long_run_variance}")
    standard_error = math.sqrt(long_run_variance / sample_size)
    if not np.isfinite(standard_error) or standard_error <= 0:
        raise ValueError(f"Invalid HAC standard error: {standard_error}")
    statistic = mean / standard_error
    pvalue = math.erfc(abs(statistic) / math.sqrt(2.0))
    return {
        "n_dates": sample_size,
        "max_lag": int(max_lag),
        "theta_hat": mean,
        "long_run_variance": long_run_variance,
        "se_mean": standard_error,
        "z_stat": statistic,
        "p_value_raw": pvalue,
    }


def _hac_classification(pvalue: float, mean: float) -> str:
    if pvalue < ALPHA and mean > 0:
        return "SECONDARY_HAC_STATIC_LOWER_LOSS"
    if pvalue < ALPHA and mean < 0:
        return "SECONDARY_HAC_ADAPTIVE_LOWER_LOSS"
    return "SECONDARY_HAC_NO_DISTINGUISHABLE_DIFFERENCE"


def hac_inference(
    date_loss_difference: pd.DataFrame,
    horizons: tuple[str, ...] = HORIZONS,
    max_lags: dict[str, int] = HAC_MAX_LAGS,
) -> pd.DataFrame:
    """Run the secondary HAC diagnostic and three-horizon Holm adjustment."""
    rows = []
    raw_pvalues = {}
    for horizon in horizons:
        series = date_loss_difference[
            date_loss_difference["horizon"] == horizon
        ].sort_values("anchor_date")["mean_loss_diff"].to_numpy(dtype=np.float64)
        result = hac_dm_test(series, max_lags[horizon])
        result["horizon"] = horizon
        result["classification_raw"] = _hac_classification(
            result["p_value_raw"],
            result["theta_hat"],
        )
        raw_pvalues[horizon] = result["p_value_raw"]
        rows.append(result)
    adjusted = holm_adjust_three(raw_pvalues)
    for row in rows:
        row["p_value_holm"] = adjusted[row["horizon"]]
    frame = pd.DataFrame(rows)
    frame["horizon"] = pd.Categorical(
        frame["horizon"], categories=list(horizons), ordered=True
    )
    frame = frame.sort_values("horizon").reset_index(drop=True)
    frame["horizon"] = frame["horizon"].astype(str)
    return frame[
        [
            "horizon",
            "n_dates",
            "max_lag",
            "theta_hat",
            "long_run_variance",
            "se_mean",
            "z_stat",
            "p_value_raw",
            "p_value_holm",
            "classification_raw",
        ]
    ]
