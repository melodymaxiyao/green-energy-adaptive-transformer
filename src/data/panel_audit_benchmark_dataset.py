"""
Panel Benchmark Dataset Audit (Phase 2a, Step 0)

This is the mandatory first step before any panel model (Ridge, LSTM, Vanilla
Transformer, PathFormer) is written. It does NOT train anything. It:

  1. Loads the existing Daily + Weekly panel dataset for all 24 tickers in
     `panel_universe.GREEN_ENERGY_UNIVERSE` (already built by
     `panel_build_multiscale_dataset.py`).
  2. Audits per-ticker data availability: date coverage, sample counts,
     target mean/std, NaN/Inf counts.
  3. Reports the common date coverage across all 24 tickers (do not assume
     it) and derives ONE shared panel-wide chronological split (train/val/test
     cut dates), applied identically to every ticker.
  4. Locks the normalization protocol: per-ticker, per-frequency feature
     mean/std computed strictly from each ticker's TRAIN split rows.
  5. Builds one unified sample index (ticker, anchor_date, split, y_5d/10d/20d)
     that every downstream model must reuse unchanged.

Outputs (all under dataset/audit/):
  - panel_benchmark_data_audit.csv   per-ticker data availability audit
  - panel_split_summary.csv          per-ticker split counts/date ranges under the shared cut dates
  - panel_norm_stats.csv             locked train-only per-ticker/per-frequency normalization stats
  - panel_sample_index.csv           unified (ticker, anchor_date, split, y_5d, y_10d, y_20d) index
  - panel_benchmark_data_audit_summary.txt   human-readable summary + leakage checks

No model training happens here. Do not proceed to panel_baseline_ridge.py
until this audit has been reviewed and the leakage checks all pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.python.panel_universe import GREEN_ENERGY_UNIVERSE

PANEL_DIR = ROOT / "dataset" / "multiscale_dataset" / "panel"
AUDIT_DIR = ROOT / "dataset" / "audit"

FEATURES = ["open", "high", "low", "close", "volume"]
HORIZONS = [5, 10, 20]

# Shared panel-wide split fractions (applied over pooled anchor dates, not
# per-ticker row fractions) — see global_split_by_pooled_dates().
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def load_ticker(ticker: str) -> dict | None:
    tdir = PANEL_DIR / ticker
    required = ["X_daily.npy", "X_weekly.npy", "y_5d.npy", "y_10d.npy", "y_20d.npy", "meta.csv"]
    if not all((tdir / f).exists() for f in required):
        return None

    X_daily = np.load(tdir / "X_daily.npy")
    X_weekly = np.load(tdir / "X_weekly.npy")
    y_5d = np.load(tdir / "y_5d.npy")
    y_10d = np.load(tdir / "y_10d.npy")
    y_20d = np.load(tdir / "y_20d.npy")
    meta = pd.read_csv(tdir / "meta.csv", parse_dates=["anchor_date"])

    n = min(len(X_daily), len(X_weekly), len(y_5d), len(y_10d), len(y_20d), len(meta))
    return {
        "ticker": ticker,
        "X_daily": X_daily[:n],
        "X_weekly": X_weekly[:n],
        "y_5d": y_5d[:n],
        "y_10d": y_10d[:n],
        "y_20d": y_20d[:n],
        "anchor_date": meta["anchor_date"].values[:n],
    }


def audit_one_ticker(rec: dict) -> dict:
    ticker = rec["ticker"]
    anchor_date = pd.to_datetime(rec["anchor_date"])
    X_daily, X_weekly = rec["X_daily"], rec["X_weekly"]
    y_5d, y_10d, y_20d = rec["y_5d"], rec["y_10d"], rec["y_20d"]

    n = len(anchor_date)
    # X_daily/X_weekly are built jointly per anchor in panel_build_multiscale_dataset.py,
    # so "daily-only" / "weekly-only" / "daily+weekly" sample counts are identical by
    # construction today. Reported separately anyway so a future data-source change
    # (e.g. an independently-built weekly series) would be caught here.
    n_daily = int(np.sum(~np.isnan(X_daily).any(axis=(1, 2))))
    n_weekly = int(np.sum(~np.isnan(X_weekly).any(axis=(1, 2))))
    n_daily_and_weekly = int(np.sum(
        (~np.isnan(X_daily).any(axis=(1, 2))) & (~np.isnan(X_weekly).any(axis=(1, 2)))
    ))

    return {
        "ticker": ticker,
        "first_anchor_date": anchor_date.min(),
        "last_anchor_date": anchor_date.max(),
        "n_samples": n,
        "n_daily_samples": n_daily,
        "n_weekly_samples": n_weekly,
        "n_daily_and_weekly_samples": n_daily_and_weekly,
        "n_y5d": int(np.sum(~np.isnan(y_5d))),
        "n_y10d": int(np.sum(~np.isnan(y_10d))),
        "n_y20d": int(np.sum(~np.isnan(y_20d))),
        "y5d_mean": float(np.nanmean(y_5d)),
        "y5d_std": float(np.nanstd(y_5d)),
        "y10d_mean": float(np.nanmean(y_10d)),
        "y10d_std": float(np.nanstd(y_10d)),
        "y20d_mean": float(np.nanmean(y_20d)),
        "y20d_std": float(np.nanstd(y_20d)),
        "n_nan_X_daily": int(np.isnan(X_daily).sum()),
        "n_nan_X_weekly": int(np.isnan(X_weekly).sum()),
        "n_inf_X_daily": int(np.isinf(X_daily).sum()),
        "n_inf_X_weekly": int(np.isinf(X_weekly).sum()),
        "n_nan_y5d": int(np.isnan(y_5d).sum()),
        "n_nan_y10d": int(np.isnan(y_10d).sum()),
        "n_nan_y20d": int(np.isnan(y_20d).sum()),
        "n_inf_y5d": int(np.isinf(y_5d).sum()),
        "n_inf_y10d": int(np.isinf(y_10d).sum()),
        "n_inf_y20d": int(np.isinf(y_20d).sum()),
    }


def global_split_by_pooled_dates(all_anchor_dates: np.ndarray) -> tuple[pd.Timestamp, pd.Timestamp]:
    """One shared train/val/test cut date pair, derived from the pooled (all-ticker)
    anchor-date distribution — NOT a per-ticker 70/15/15 split. Every ticker uses
    these same two cut dates."""
    unique_dates = np.sort(np.unique(all_anchor_dates))
    n_dates = len(unique_dates)
    train_cut = unique_dates[int(n_dates * TRAIN_FRAC) - 1]
    val_cut = unique_dates[int(n_dates * (TRAIN_FRAC + VAL_FRAC)) - 1]
    return pd.Timestamp(train_cut), pd.Timestamp(val_cut)


def split_label(anchor_date: pd.Timestamp, train_cut: pd.Timestamp, val_cut: pd.Timestamp) -> str:
    if anchor_date <= train_cut:
        return "train"
    if anchor_date <= val_cut:
        return "val"
    return "test"


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    records = {}
    for ticker in GREEN_ENERGY_UNIVERSE:
        rec = load_ticker(ticker)
        if rec is None:
            print(f"[skip] {ticker}: panel files not found under {PANEL_DIR / ticker}")
            continue
        records[ticker] = rec

    if not records:
        raise RuntimeError(f"No panel tickers found under {PANEL_DIR}. Run panel_build_multiscale_dataset.py first.")

    # ---- 1) Per-ticker data availability audit ----
    audit_rows = [audit_one_ticker(rec) for rec in records.values()]
    audit_df = pd.DataFrame(audit_rows).sort_values("ticker").reset_index(drop=True)
    audit_df.to_csv(AUDIT_DIR / "panel_benchmark_data_audit.csv", index=False)

    # ---- 2) Common date coverage across all tickers (report, do not assume) ----
    common_start = audit_df["first_anchor_date"].max()  # latest of all "first dates"
    common_end = audit_df["last_anchor_date"].min()      # earliest of all "last dates"
    universe_start = audit_df["first_anchor_date"].min()
    universe_end = audit_df["last_anchor_date"].max()

    # ---- 3) One shared chronological split, derived from pooled anchor dates ----
    all_anchor_dates = np.concatenate([pd.to_datetime(rec["anchor_date"]).values for rec in records.values()])
    train_cut, val_cut = global_split_by_pooled_dates(all_anchor_dates)

    # ---- 4) Build unified sample index + per-ticker split counts ----
    sample_index_rows = []
    split_summary_rows = []
    sample_id = 0
    for ticker, rec in records.items():
        anchor_date = pd.to_datetime(rec["anchor_date"])
        splits = np.array([split_label(d, train_cut, val_cut) for d in anchor_date])

        for i in range(len(anchor_date)):
            sample_index_rows.append({
                "sample_id": sample_id,
                "ticker": ticker,
                "anchor_date": anchor_date[i],
                "split": splits[i],
                "y_5d": rec["y_5d"][i],
                "y_10d": rec["y_10d"][i],
                "y_20d": rec["y_20d"][i],
            })
            sample_id += 1

        n_train = int(np.sum(splits == "train"))
        n_val = int(np.sum(splits == "val"))
        n_test = int(np.sum(splits == "test"))

        def date_range(mask):
            if mask.sum() == 0:
                return (pd.NaT, pd.NaT)
            d = anchor_date[mask]
            return (d.min(), d.max())

        train_start, train_end = date_range(splits == "train")
        val_start, val_end = date_range(splits == "val")
        test_start, test_end = date_range(splits == "test")

        warning = ""
        if n_train == 0 or n_val == 0 or n_test == 0:
            warning = "ZERO_SAMPLES_IN_SPLIT"
        # Leakage check: no train date after the train cut, no test date at/before the val cut.
        leakage_ok = True
        if n_train > 0 and train_end > train_cut:
            leakage_ok = False
        if n_test > 0 and test_start <= val_cut:
            leakage_ok = False

        split_summary_rows.append({
            "ticker": ticker,
            "split_train_cut": train_cut,
            "split_val_cut": val_cut,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "train_start_date": train_start,
            "train_end_date": train_end,
            "val_start_date": val_start,
            "val_end_date": val_end,
            "test_start_date": test_start,
            "test_end_date": test_end,
            "warning": warning,
            "leakage_check_passed": leakage_ok,
        })

    sample_index_df = pd.DataFrame(sample_index_rows)
    sample_index_df.to_csv(AUDIT_DIR / "panel_sample_index.csv", index=False)

    split_summary_df = pd.DataFrame(split_summary_rows).sort_values("ticker").reset_index(drop=True)
    split_summary_df.to_csv(AUDIT_DIR / "panel_split_summary.csv", index=False)

    # ---- 5) Lock the normalization protocol: train-only, per-ticker, per-frequency ----
    norm_rows = []
    for ticker, rec in records.items():
        anchor_date = pd.to_datetime(rec["anchor_date"])
        train_mask = anchor_date <= train_cut
        if train_mask.sum() == 0:
            continue

        for freq_name, X in (("daily", rec["X_daily"]), ("weekly", rec["X_weekly"])):
            X_train = X[train_mask]  # (n_train, window, n_features), train rows only
            for f_idx, feature in enumerate(FEATURES):
                vals = X_train[:, :, f_idx]
                norm_rows.append({
                    "ticker": ticker,
                    "frequency": freq_name,
                    "feature": feature,
                    "train_mean": float(np.nanmean(vals)),
                    "train_std": float(np.nanstd(vals)),
                    "n_train_rows": int(train_mask.sum()),
                })

    norm_df = pd.DataFrame(norm_rows)
    norm_df.to_csv(AUDIT_DIR / "panel_norm_stats.csv", index=False)

    # ---- 6) Human-readable summary ----
    n_zero_split = int((split_summary_df["warning"] == "ZERO_SAMPLES_IN_SPLIT").sum())
    n_leakage_fail = int((~split_summary_df["leakage_check_passed"]).sum())

    lines = [
        "PANEL BENCHMARK DATASET AUDIT (Phase 2a, Step 0)",
        "=" * 72,
        f"Tickers audited: {len(records)} / {len(GREEN_ENERGY_UNIVERSE)} in GREEN_ENERGY_UNIVERSE",
        "",
        "Date coverage:",
        f"  Universe-wide range (earliest first_anchor_date to latest last_anchor_date): {universe_start.date()} -> {universe_end.date()}",
        f"  Common coverage across ALL tickers (max-of-mins to min-of-maxes): {common_start.date()} -> {common_end.date()}",
        "",
        "Shared chronological split (applied identically to every ticker):",
        f"  Train:      anchor_date <= {train_cut.date()}",
        f"  Validation: {train_cut.date()} < anchor_date <= {val_cut.date()}",
        f"  Test:       anchor_date > {val_cut.date()}",
        "",
        f"Total pooled samples: {len(sample_index_df)}",
        f"  train: {(sample_index_df['split'] == 'train').sum()}",
        f"  val:   {(sample_index_df['split'] == 'val').sum()}",
        f"  test:  {(sample_index_df['split'] == 'test').sum()}",
        "",
        f"Tickers with a zero-sample split (train/val/test): {n_zero_split}",
        f"Tickers failing the leakage check: {n_leakage_fail}",
    ]
    if n_zero_split > 0:
        bad = split_summary_df[split_summary_df["warning"] == "ZERO_SAMPLES_IN_SPLIT"]["ticker"].tolist()
        lines.append(f"  -> zero-sample tickers: {bad}")
    if n_leakage_fail > 0:
        bad = split_summary_df[~split_summary_df["leakage_check_passed"]]["ticker"].tolist()
        lines.append(f"  -> leakage-check-failed tickers: {bad}")

    total_nan = int(audit_df[[c for c in audit_df.columns if c.startswith("n_nan_")]].sum().sum())
    total_inf = int(audit_df[[c for c in audit_df.columns if c.startswith("n_inf_")]].sum().sum())
    lines += [
        "",
        f"Total NaN cells across all tickers' features/targets: {total_nan}",
        f"Total Inf cells across all tickers' features/targets: {total_inf}",
        "",
        "Outputs written:",
        "  dataset/audit/panel_benchmark_data_audit.csv",
        "  dataset/audit/panel_split_summary.csv",
        "  dataset/audit/panel_norm_stats.csv",
        "  dataset/audit/panel_sample_index.csv",
        "",
        "REVIEW THIS FILE BEFORE WRITING panel_baseline_ridge.py.",
        "If n_zero_split > 0 or n_leakage_fail > 0, the split/protocol must be fixed first.",
    ]

    summary_path = AUDIT_DIR / "panel_benchmark_data_audit_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
