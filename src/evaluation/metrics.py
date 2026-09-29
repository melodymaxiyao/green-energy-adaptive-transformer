"""Final-paper forecast metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _to_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def safe_pearson_corr(x, y) -> float:
    """Pearson correlation, or NaN for short or near-constant inputs."""
    x = np.asarray(_to_numpy(x), dtype=np.float64)
    y = np.asarray(_to_numpy(y), dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank_ic_by_date(y_true, y_pred, ticker, anchor_date) -> pd.DataFrame:
    """Compute cross-sectional Spearman Rank IC separately for each date."""
    frame = pd.DataFrame(
        {
            "date": _to_numpy(anchor_date),
            "ticker": _to_numpy(ticker),
            "y_true": _to_numpy(y_true),
            "y_pred": _to_numpy(y_pred),
        }
    )
    rows = []
    for date, group in frame.groupby("date"):
        count = len(group)
        if (
            count < 3
            or group["y_true"].std() < 1e-12
            or group["y_pred"].std() < 1e-12
        ):
            rank_ic = float("nan")
        else:
            correlation, _ = spearmanr(group["y_pred"], group["y_true"])
            rank_ic = float(correlation) if not np.isnan(correlation) else float("nan")
        rows.append(
            {
                "anchor_date": date,
                "n_tickers": count,
                "rank_ic": rank_ic,
            }
        )
    return pd.DataFrame(rows).sort_values("anchor_date").reset_index(drop=True)


def cross_sectional_rank_ic(y_true, y_pred, ticker, anchor_date) -> float:
    """Average datewise cross-sectional Rank IC over defined dates."""
    values = rank_ic_by_date(y_true, y_pred, ticker, anchor_date)["rank_ic"].dropna()
    return float(values.mean()) if len(values) else float("nan")


def evaluate_predictions(y_true, y_pred, ticker, anchor_date) -> dict[str, float]:
    """Return the forecast metrics used in the final-paper experiments."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    ticker = _to_numpy(ticker)
    anchor_date = _to_numpy(anchor_date)

    difference = y_pred - y_true
    pooled_mse = float(np.mean(difference ** 2))
    pooled_mae = float(np.mean(np.abs(difference)))
    pooled_corr = safe_pearson_corr(y_pred, y_true)
    pooled_da = float((np.sign(y_pred) == np.sign(y_true)).mean())
    pooled_pred_std = float(np.std(y_pred))
    pooled_true_std = float(np.std(y_true))
    pooled_ratio = (
        pooled_pred_std / pooled_true_std
        if pooled_true_std > 1e-12
        else float("nan")
    )

    frame = pd.DataFrame(
        {"ticker": ticker, "y_true": y_true, "y_pred": y_pred}
    )
    per_ticker_rows = []
    for _, group in frame.groupby("ticker"):
        if len(group) < 2:
            continue
        ticker_difference = group["y_pred"].values - group["y_true"].values
        per_ticker_rows.append(
            {
                "mse": np.mean(ticker_difference ** 2),
                "mae": np.mean(np.abs(ticker_difference)),
                "corr": safe_pearson_corr(
                    group["y_pred"].values,
                    group["y_true"].values,
                ),
                "da": (
                    np.sign(group["y_pred"]) == np.sign(group["y_true"])
                ).mean(),
                "pred_std": group["y_pred"].std(),
                "true_std": group["y_true"].std(),
            }
        )
    per_ticker = pd.DataFrame(per_ticker_rows)

    return {
        "pooled_mse": pooled_mse,
        "pooled_mae": pooled_mae,
        "pooled_corr": pooled_corr,
        "pooled_da": pooled_da,
        "pooled_pred_std": pooled_pred_std,
        "pooled_true_std": pooled_true_std,
        "pooled_pred_std_over_true_std": pooled_ratio,
        "cross_sectional_rank_ic": cross_sectional_rank_ic(
            y_true,
            y_pred,
            ticker,
            anchor_date,
        ),
        "ticker_avg_mse": float(per_ticker["mse"].mean()),
        "ticker_avg_mae": float(per_ticker["mae"].mean()),
        "ticker_avg_corr": float(per_ticker["corr"].mean()),
        "ticker_avg_da": float(per_ticker["da"].mean()),
        "ticker_avg_pred_std": float(per_ticker["pred_std"].mean()),
        "ticker_avg_true_std": float(per_ticker["true_std"].mean()),
    }
