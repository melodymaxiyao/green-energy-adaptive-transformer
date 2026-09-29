"""Final-paper router diagnostics."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

HORIZONS = ("5d", "10d", "20d")
SEEDS = (0, 1, 21, 42, 3407)
FREQUENCIES = ("daily", "weekly")
PATCH_SIZES = {
    "daily": (5, 10, 20, 30),
    "weekly": (2, 4, 8, 13),
}
WEIGHT_COLUMNS = ("weight_1", "weight_2", "weight_3", "weight_4")


def total_variation_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Total variation distance between two four-scale allocations."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("TV distance requires matching vector shapes")
    return float(0.5 * np.sum(np.abs(a - b), dtype=np.float64))


def entropy_metrics(weights: np.ndarray) -> tuple[float, float, float, float]:
    """Return entropy, normalized entropy, effective scales, and max weight."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (4,):
        raise ValueError("Entropy metrics require one four-scale vector")
    positive = weights[weights > 0.0]
    entropy = float(-np.sum(positive * np.log(positive), dtype=np.float64))
    normalized_entropy = float(entropy / math.log(4.0))
    effective_scales = float(math.exp(entropy))
    maximum_weight = float(np.max(weights))
    return entropy, normalized_entropy, effective_scales, maximum_weight


def manual_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    """Frozen average-rank Spearman implementation without p-values."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("Spearman inputs must have matching shapes")
    if x.size == 0:
        return float("nan"), False
    rank_x = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    rank_y = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    centered_x = rank_x - np.mean(rank_x, dtype=np.float64)
    centered_y = rank_y - np.mean(rank_y, dtype=np.float64)
    denominator = math.sqrt(
        float(
            np.sum(centered_x ** 2, dtype=np.float64)
            * np.sum(centered_y ** 2, dtype=np.float64)
        )
    )
    if denominator == 0.0:
        return float("nan"), False
    correlation = float(
        np.sum(centered_x * centered_y, dtype=np.float64) / denominator
    )
    if not np.isfinite(correlation):
        return float("nan"), False
    if correlation < -1.0 - 1e-12 or correlation > 1.0 + 1e-12:
        raise ValueError(f"Spearman correlation outside [-1, 1]: {correlation}")
    return correlation, True


def static_weight_references(static_weights: pd.DataFrame) -> pd.DataFrame:
    """Average all persisted Static vectors within each branch/horizon/seed cell."""
    keys = ["horizon", "seed", "frequency"]
    required = set(keys + list(WEIGHT_COLUMNS))
    missing = sorted(required - set(static_weights.columns))
    if missing:
        raise ValueError(f"Static weight table is missing columns: {missing}")
    references = (
        static_weights.groupby(keys, sort=False)[list(WEIGHT_COLUMNS)]
        .mean()
        .reset_index()
    )
    values = references[list(WEIGHT_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Static reference contains non-finite weights")
    if np.any(values < -1e-10) or np.any(values > 1.0 + 1e-10):
        raise ValueError("Static reference contains weights outside [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("Static reference weights do not sum to one")
    return references


def build_router_sample_metrics(
    adaptive_weights: pd.DataFrame,
    static_references: pd.DataFrame,
) -> pd.DataFrame:
    """Build Adaptive sample-level concentration and TV diagnostics."""
    keys = ["horizon", "seed", "frequency"]
    sample_keys = keys + ["ticker", "anchor_date"]
    required = set(sample_keys + list(WEIGHT_COLUMNS) + ["loss_diff"])
    missing = sorted(required - set(adaptive_weights.columns))
    if missing:
        raise ValueError(f"Adaptive weight table is missing columns: {missing}")
    reference_lookup = {
        tuple(row[key] for key in keys): row[list(WEIGHT_COLUMNS)].to_numpy(
            dtype=np.float64
        )
        for _, row in static_references.iterrows()
    }

    records = []
    for cell_key, cell in adaptive_weights.groupby(keys, sort=False):
        if cell_key not in reference_lookup:
            raise ValueError(f"Missing Static reference for {cell_key}")
        cell = cell.sort_values(
            ["ticker", "anchor_date"], kind="mergesort"
        ).reset_index(drop=True)
        weights = cell[list(WEIGHT_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.isfinite(weights).all():
            raise ValueError(f"Non-finite Adaptive weights for {cell_key}")
        cell_mean = np.mean(weights, axis=0, dtype=np.float64)
        static_vector = reference_lookup[cell_key]
        frequency = cell_key[2]
        patch_sizes = PATCH_SIZES[frequency]

        for index, vector in enumerate(weights):
            entropy, normalized, effective, maximum = entropy_metrics(vector)
            dominant_patch = patch_sizes[int(np.argmax(vector))]
            row = {
                "horizon": cell_key[0],
                "seed": int(cell_key[1]),
                "ticker": cell.iloc[index]["ticker"],
                "anchor_date": cell.iloc[index]["anchor_date"],
                "frequency": frequency,
                "patch_1": patch_sizes[0],
                "patch_2": patch_sizes[1],
                "patch_3": patch_sizes[2],
                "patch_4": patch_sizes[3],
                "weight_1": float(vector[0]),
                "weight_2": float(vector[1]),
                "weight_3": float(vector[2]),
                "weight_4": float(vector[3]),
                "entropy": entropy,
                "normalized_entropy": normalized,
                "effective_scales": effective,
                "max_weight": maximum,
                "dominant_patch": int(dominant_patch),
                "tv_to_cell_mean": total_variation_distance(vector, cell_mean),
                "tv_to_static": total_variation_distance(vector, static_vector),
                "loss_diff": float(cell.iloc[index]["loss_diff"]),
            }
            records.append(row)
    result = pd.DataFrame(records)
    horizon_order = {value: index for index, value in enumerate(HORIZONS)}
    frequency_order = {value: index for index, value in enumerate(FREQUENCIES)}
    result["_horizon"] = result["horizon"].map(horizon_order)
    result["_frequency"] = result["frequency"].map(frequency_order)
    return result.sort_values(
        ["_horizon", "seed", "ticker", "anchor_date", "_frequency"],
        kind="mergesort",
    ).drop(columns=["_horizon", "_frequency"]).reset_index(drop=True)


def _distribution_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values, dtype=np.float64)),
        f"{prefix}_sd": (
            float(np.std(values, ddof=1, dtype=np.float64))
            if len(values) > 1
            else float("nan")
        ),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p10": float(np.quantile(values, 0.10, method="linear")),
        f"{prefix}_p90": float(np.quantile(values, 0.90, method="linear")),
    }


def router_cell_summary(
    sample_metrics: pd.DataFrame,
    static_references: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize sample, temporal, cross-sectional, and Static distances."""
    keys = ["horizon", "seed", "frequency"]
    reference_lookup = {
        tuple(row[key] for key in keys): row[list(WEIGHT_COLUMNS)].to_numpy(
            dtype=np.float64
        )
        for _, row in static_references.iterrows()
    }
    rows = []
    for cell_key, cell in sample_metrics.groupby(keys, sort=False):
        weights = cell[list(WEIGHT_COLUMNS)].to_numpy(dtype=np.float64)
        cell_mean = np.mean(weights, axis=0, dtype=np.float64)
        static_vector = reference_lookup[cell_key]

        ticker_tvs = []
        for _, group in cell.groupby("ticker"):
            vectors = group[list(WEIGHT_COLUMNS)].to_numpy(dtype=np.float64)
            ticker_mean = np.mean(vectors, axis=0, dtype=np.float64)
            ticker_tvs.append(
                float(
                    np.mean(
                        [total_variation_distance(vector, ticker_mean) for vector in vectors],
                        dtype=np.float64,
                    )
                )
            )
        date_tvs = []
        for _, group in cell.groupby("anchor_date"):
            vectors = group[list(WEIGHT_COLUMNS)].to_numpy(dtype=np.float64)
            date_mean = np.mean(vectors, axis=0, dtype=np.float64)
            date_tvs.append(
                float(
                    np.mean(
                        [total_variation_distance(vector, date_mean) for vector in vectors],
                        dtype=np.float64,
                    )
                )
            )

        row = {
            "horizon": cell_key[0],
            "seed": int(cell_key[1]),
            "frequency": cell_key[2],
            "n_samples": int(len(cell)),
            "n_tickers": int(cell["ticker"].nunique()),
            "n_dates": int(cell["anchor_date"].nunique()),
        }
        for index, column in enumerate(WEIGHT_COLUMNS, start=1):
            values = cell[column].to_numpy(dtype=np.float64)
            row[f"mean_weight_{index}"] = float(np.mean(values, dtype=np.float64))
            row[f"sd_weight_{index}"] = (
                float(np.std(values, ddof=1, dtype=np.float64))
                if len(values) > 1
                else float("nan")
            )
        row.update(
            {
                "sample_dependence_index": float(
                    np.mean(cell["tv_to_cell_mean"].to_numpy(dtype=np.float64))
                ),
                "within_ticker_temporal_tv": float(
                    np.mean(ticker_tvs, dtype=np.float64)
                ),
                "within_date_crosssectional_tv": float(
                    np.mean(date_tvs, dtype=np.float64)
                ),
            }
        )
        row.update(
            _distribution_summary(
                cell["normalized_entropy"].to_numpy(dtype=np.float64),
                "normalized_entropy",
            )
        )
        row.update(
            _distribution_summary(
                cell["effective_scales"].to_numpy(dtype=np.float64),
                "effective_scales",
            )
        )
        row.update(
            _distribution_summary(
                cell["max_weight"].to_numpy(dtype=np.float64),
                "max_weight",
            )
        )
        for index, value in enumerate(static_vector, start=1):
            row[f"static_weight_{index}"] = float(value)
        tv_to_static = cell["tv_to_static"].to_numpy(dtype=np.float64)
        row["tv_cell_mean_to_static"] = total_variation_distance(
            cell_mean,
            static_vector,
        )
        row.update(_distribution_summary(tv_to_static, "tv_to_static"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def dominant_scale_usage(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    """Count all four dominant patches within each branch/horizon/seed cell."""
    rows = []
    for cell_key, cell in sample_metrics.groupby(
        ["horizon", "seed", "frequency"], sort=False
    ):
        for patch_size in PATCH_SIZES[cell_key[2]]:
            count = int((cell["dominant_patch"] == patch_size).sum())
            rows.append(
                {
                    "horizon": cell_key[0],
                    "seed": int(cell_key[1]),
                    "frequency": cell_key[2],
                    "patch_size": patch_size,
                    "dominant_count": count,
                    "dominant_share": float(count / len(cell)),
                }
            )
    return pd.DataFrame(rows)


def cross_seed_stability(
    sample_metrics: pd.DataFrame,
    seeds: tuple[int, ...] = SEEDS,
) -> pd.DataFrame:
    """Compute ten pairwise-seed TVs and dominant-scale agreement per sample."""
    rows = []
    group_keys = ["horizon", "ticker", "anchor_date", "frequency"]
    for key, group in sample_metrics.groupby(group_keys, sort=False):
        group = group.sort_values("seed", kind="mergesort")
        if tuple(group["seed"].tolist()) != tuple(seeds):
            raise ValueError(f"Seed set mismatch for {key}")
        weights = group[list(WEIGHT_COLUMNS)].to_numpy(dtype=np.float64)
        pairwise = np.asarray(
            [
                total_variation_distance(weights[left], weights[right])
                for left, right in combinations(range(len(seeds)), 2)
            ],
            dtype=np.float64,
        )
        counts = group["dominant_patch"].value_counts()
        agreement = float(counts.max() / len(seeds))
        rows.append(
            {
                "horizon": key[0],
                "ticker": key[1],
                "anchor_date": key[2],
                "frequency": key[3],
                "n_seeds": len(seeds),
                "mean_pairwise_seed_tv": float(np.mean(pairwise, dtype=np.float64)),
                "max_pairwise_seed_tv": float(np.max(pairwise)),
                "dominant_agreement_share": agreement,
                "dominant_unanimous": int(agreement == 1.0),
            }
        )
    return pd.DataFrame(rows)


def summarize_cross_seed_stability(stability: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cross-seed stability separately by horizon and branch."""
    rows = []
    for key, group in stability.groupby(["horizon", "frequency"], sort=False):
        mean_pairwise = group["mean_pairwise_seed_tv"].to_numpy(dtype=np.float64)
        max_pairwise = group["max_pairwise_seed_tv"].to_numpy(dtype=np.float64)
        agreement = group["dominant_agreement_share"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "horizon": key[0],
                "frequency": key[1],
                "n_samples": int(len(group)),
                "mean_pairwise_seed_tv_mean": float(np.mean(mean_pairwise)),
                "mean_pairwise_seed_tv_sd": float(np.std(mean_pairwise, ddof=1)),
                "mean_pairwise_seed_tv_median": float(np.median(mean_pairwise)),
                "mean_pairwise_seed_tv_p90": float(
                    np.quantile(mean_pairwise, 0.90, method="linear")
                ),
                "max_pairwise_seed_tv_mean": float(np.mean(max_pairwise)),
                "max_pairwise_seed_tv_median": float(np.median(max_pairwise)),
                "max_pairwise_seed_tv_p90": float(
                    np.quantile(max_pairwise, 0.90, method="linear")
                ),
                "dominant_agreement_share_mean": float(np.mean(agreement)),
                "dominant_agreement_share_median": float(np.median(agreement)),
                "dominant_unanimous_share": float(
                    np.mean(group["dominant_unanimous"].to_numpy(dtype=np.float64))
                ),
            }
        )
    return pd.DataFrame(rows)


def loss_association_by_seed(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute sample- and date-level router-distance/loss associations."""
    rows = []
    for key, group in sample_metrics.groupby(
        ["horizon", "seed", "frequency"], sort=False
    ):
        sample_rho, sample_defined = manual_spearman(
            group["tv_to_static"].to_numpy(dtype=np.float64),
            group["loss_diff"].to_numpy(dtype=np.float64),
        )
        date_pairs = []
        for _, date_group in group.groupby("anchor_date"):
            date_pairs.append(
                (
                    np.mean(
                        date_group["tv_to_static"].to_numpy(dtype=np.float64),
                        dtype=np.float64,
                    ),
                    np.mean(
                        date_group["loss_diff"].to_numpy(dtype=np.float64),
                        dtype=np.float64,
                    ),
                )
            )
        date_tv = np.asarray([pair[0] for pair in date_pairs], dtype=np.float64)
        date_loss = np.asarray([pair[1] for pair in date_pairs], dtype=np.float64)
        date_rho, date_defined = manual_spearman(
            date_tv,
            date_loss,
        )
        rows.append(
            {
                "horizon": key[0],
                "seed": int(key[1]),
                "frequency": key[2],
                "n_samples": int(len(group)),
                "n_dates": int(len(date_pairs)),
                "sample_level_spearman_rho": sample_rho,
                "sample_level_correlation_defined": bool(sample_defined),
                "date_level_spearman_rho": date_rho,
                "date_level_correlation_defined": bool(date_defined),
            }
        )
    return pd.DataFrame(rows)


def _summarize_rhos(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    count = len(valid)
    if count == 0:
        return {
            "mean": float("nan"),
            "sd": float("nan"),
            "median": float("nan"),
            "n_defined": 0,
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": 0,
        }
    return {
        "mean": float(np.mean(valid, dtype=np.float64)),
        "sd": (
            float(np.std(valid, ddof=1, dtype=np.float64))
            if count > 1
            else float("nan")
        ),
        "median": float(np.median(valid)),
        "n_defined": count,
        "n_positive": int(np.sum(valid > 0.0)),
        "n_negative": int(np.sum(valid < 0.0)),
        "n_zero": int(np.sum(valid == 0.0)),
    }


def summarize_loss_associations(associations: pd.DataFrame) -> pd.DataFrame:
    """Summarize the five descriptive correlations per horizon and branch."""
    rows = []
    for key, group in associations.groupby(["horizon", "frequency"], sort=False):
        sample = _summarize_rhos(
            group["sample_level_spearman_rho"].to_numpy(dtype=np.float64)
        )
        date = _summarize_rhos(
            group["date_level_spearman_rho"].to_numpy(dtype=np.float64)
        )
        row = {"horizon": key[0], "frequency": key[1]}
        for prefix, summary in (("sample_level", sample), ("date_level", date)):
            row[f"{prefix}_rho_mean"] = summary["mean"]
            row[f"{prefix}_rho_sd"] = summary["sd"]
            row[f"{prefix}_rho_median"] = summary["median"]
            row[f"{prefix}_n_defined"] = summary["n_defined"]
            row[f"{prefix}_n_positive"] = summary["n_positive"]
            row[f"{prefix}_n_negative"] = summary["n_negative"]
            row[f"{prefix}_n_zero"] = summary["n_zero"]
        rows.append(row)
    return pd.DataFrame(rows)
