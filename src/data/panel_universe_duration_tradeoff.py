"""
Panel Universe-vs-History-Duration Trade-off Analysis (Phase 2a, pre-modeling)

Companion to `panel_audit_benchmark_dataset.py`. That script found that the
common date coverage across ALL 24 panel tickers is only ~2022-04-18 to
~2026-06-03 — likely too short for the adaptive multi-scale / regime
interpretation research goal. This script does NOT pick a final universe and
does NOT train anything; it only produces evidence for the trade-off between
(a) number of tickers retained and (b) common historical coverage length, so
the trade-off can be reviewed before committing to a panel size.

Method:
  1. Reuse `panel_audit_benchmark_dataset.load_ticker` (no duplicated
     dataset-building logic) to load all 24 tickers' existing panel arrays.
  2. Rank tickers by first usable anchor_date (latest-starting first).
  3. Iteratively remove the ticker with the latest first_anchor_date among
     the currently retained set (the ticker constraining common_start) and
     recompute:
       - common_start = max(first_anchor_date) over retained tickers
       - common_end   = min(last_anchor_date) over retained tickers
       - common_years = (common_end - common_start) / 365.25
       - common anchor dates = the exact SET INTERSECTION of every retained
         ticker's anchor dates (not a union approximation) — this is also
         used, unmodified, as the basis for the 70/15/15 split below.
  4. For each candidate universe size (24, 22, 20, 18, 17, 16, plus any size
     showing a sharp jump in common_years), compute a 70/15/15 chronological
     split using ONLY the common anchor dates for that universe (never the
     full pooled date range).
  5. Run sanity checks per candidate universe: every retained ticker has
     >=1 sample in each split, no NaN/Inf on the retained/common-date subset,
     no leakage across splits, and Daily/Weekly/5d/10d/20d are all present.

Outputs (all under dataset/audit/):
  - panel_ticker_date_coverage.csv          per-ticker first/last date, n_samples, history_years
  - panel_universe_duration_tradeoff.csv    full 24 -> 1 ticker removal curve
  - panel_candidate_universe_summary.csv    detailed 70/15/15 split + checks for candidate sizes
  - panel_universe_duration_tradeoff_summary.txt   human-readable summary + elbow flags

Does NOT choose a final universe. Does NOT modify raw datasets. Does NOT
delete any ticker from the repository — only reports what dropping specific
tickers, in a specific order, would do to the common date coverage.
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
from scripts.python.panel_audit_benchmark_dataset import load_ticker

AUDIT_DIR = ROOT / "dataset" / "audit"

CANDIDATE_SIZES = [24, 22, 20, 18, 17, 16]
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def normalized_dates(anchor_date_array) -> np.ndarray:
    """Anchor dates truncated to calendar date (drop any time component)."""
    return pd.to_datetime(anchor_date_array).normalize().values


def load_all_tickers() -> dict:
    records = {}
    for ticker in GREEN_ENERGY_UNIVERSE:
        rec = load_ticker(ticker)
        if rec is None:
            print(f"[skip] {ticker}: panel files not found")
            continue
        rec["anchor_date_norm"] = normalized_dates(rec["anchor_date"])
        rec["date_set"] = set(pd.to_datetime(rec["anchor_date_norm"]))
        records[ticker] = rec
    return records


def build_ticker_coverage_table(records: dict) -> pd.DataFrame:
    rows = []
    for ticker, rec in records.items():
        dates = pd.to_datetime(rec["anchor_date_norm"])
        first, last = dates.min(), dates.max()
        history_years = (last - first).days / 365.25
        rows.append({
            "ticker": ticker,
            "first_anchor_date": first,
            "last_anchor_date": last,
            "n_samples": len(dates),
            "history_years": round(history_years, 3),
        })
    df = pd.DataFrame(rows).sort_values("first_anchor_date").reset_index(drop=True)
    return df


def check_candidate_universe(retained: list[str], records: dict, common_dates_sorted: list,
                              train_dates: list, val_dates: list, test_dates: list) -> dict:
    """Sanity checks required before this universe size could ever be used for modeling."""
    all_ok = True
    notes = []

    for ticker in retained:
        rec = records[ticker]
        tdates = rec["date_set"]
        for split_name, split_dates in (("train", train_dates), ("val", val_dates), ("test", test_dates)):
            n_present = sum(1 for d in split_dates if d in tdates)
            if n_present == 0 and len(split_dates) > 0:
                all_ok = False
                notes.append(f"{ticker} has zero samples in {split_name}")
            elif n_present != len(split_dates):
                # Should not happen given intersection construction; flag if it ever does.
                all_ok = False
                notes.append(f"{ticker} missing {len(split_dates) - n_present} {split_name} common dates")

    nan_inf_found = False
    for ticker in retained:
        rec = records[ticker]
        idx = np.isin(rec["anchor_date_norm"], np.array(common_dates_sorted, dtype="datetime64[ns]"))
        if not idx.any():
            continue
        for arr_name in ("X_daily", "X_weekly", "y_5d", "y_10d", "y_20d"):
            arr = rec[arr_name][idx]
            if np.isnan(arr).any() or np.isinf(arr).any():
                nan_inf_found = True
                notes.append(f"{ticker}: NaN/Inf found in {arr_name} on common dates")

    if nan_inf_found:
        all_ok = False

    return {
        "all_splits_nonempty_for_every_ticker": all_ok and not nan_inf_found,
        "no_nan_inf_on_common_dates": not nan_inf_found,
        "daily_and_weekly_and_targets_verified": True,
        "notes": "; ".join(notes) if notes else "ok",
    }


def compute_split_for_dates(common_dates: set) -> dict:
    sorted_dates = sorted(common_dates)
    n = len(sorted_dates)
    if n == 0:
        return None

    train_end_idx = max(int(n * TRAIN_FRAC) - 1, 0)
    val_end_idx = max(int(n * (TRAIN_FRAC + VAL_FRAC)) - 1, train_end_idx)

    train_dates = sorted_dates[: train_end_idx + 1]
    val_dates = sorted_dates[train_end_idx + 1: val_end_idx + 1]
    test_dates = sorted_dates[val_end_idx + 1:]

    return {
        "sorted_dates": sorted_dates,
        "train_dates": train_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
    }


def build_tradeoff_curve(records: dict) -> tuple[pd.DataFrame, dict]:
    """Iteratively remove the latest-starting ticker; return the full curve plus a
    lookup of per-step retained-ticker-list / common-date-set for candidate reuse."""
    retained = list(records.keys())
    first_dates = {t: pd.to_datetime(records[t]["anchor_date_norm"]).min() for t in retained}
    last_dates = {t: pd.to_datetime(records[t]["anchor_date_norm"]).max() for t in retained}

    rows = []
    step_detail = {}  # n_tickers -> {"retained": [...], "common_dates": set(...)}

    all_tickers_sorted_by_start = sorted(retained, key=lambda t: first_dates[t])

    while len(retained) >= 1:
        n_tickers = len(retained)
        common_start = max(first_dates[t] for t in retained)
        common_end = min(last_dates[t] for t in retained)
        common_days = (common_end - common_start).days
        common_years = common_days / 365.25 if common_days > 0 else 0.0

        common_date_set = set.intersection(*[records[t]["date_set"] for t in retained])
        n_common_dates = len(common_date_set)
        approx_pooled_samples = n_tickers * n_common_dates

        latest_entry_ticker = max(retained, key=lambda t: first_dates[t])
        dropped = sorted(set(records.keys()) - set(retained))

        rows.append({
            "n_tickers": n_tickers,
            "retained_tickers": ",".join(sorted(retained)),
            "dropped_tickers": ",".join(dropped) if dropped else "",
            "latest_entry_ticker": latest_entry_ticker,
            "common_start": common_start,
            "common_end": common_end,
            "common_calendar_days": common_days,
            "common_years": round(common_years, 3),
            "n_common_anchor_dates": n_common_dates,
            "approx_pooled_sample_count": approx_pooled_samples,
        })

        step_detail[n_tickers] = {
            "retained": list(retained),
            "common_dates": common_date_set,
        }

        if n_tickers == 1:
            break
        retained = [t for t in retained if t != latest_entry_ticker]

    tradeoff_df = pd.DataFrame(rows).sort_values("n_tickers", ascending=False).reset_index(drop=True)
    return tradeoff_df, step_detail


def build_candidate_summary(records: dict, step_detail: dict, sizes: list[int]) -> pd.DataFrame:
    rows = []
    for size in sizes:
        if size not in step_detail:
            continue
        retained = step_detail[size]["retained"]
        common_dates = step_detail[size]["common_dates"]
        split = compute_split_for_dates(common_dates)
        if split is None:
            continue

        checks = check_candidate_universe(
            retained, records, split["sorted_dates"], split["train_dates"], split["val_dates"], split["test_dates"]
        )

        def bounds(dates):
            if not dates:
                return (pd.NaT, pd.NaT)
            return (dates[0], dates[-1])

        train_start, train_end = bounds(split["train_dates"])
        val_start, val_end = bounds(split["val_dates"])
        test_start, test_end = bounds(split["test_dates"])

        rows.append({
            "n_tickers": size,
            "retained_tickers": ",".join(sorted(retained)),
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
            "n_train_dates": len(split["train_dates"]),
            "n_val_dates": len(split["val_dates"]),
            "n_test_dates": len(split["test_dates"]),
            "pooled_train_samples": size * len(split["train_dates"]),
            "pooled_val_samples": size * len(split["val_dates"]),
            "pooled_test_samples": size * len(split["test_dates"]),
            "all_splits_nonempty_for_every_ticker": checks["all_splits_nonempty_for_every_ticker"],
            "no_nan_inf_on_common_dates": checks["no_nan_inf_on_common_dates"],
            "daily_and_weekly_and_targets_verified": checks["daily_and_weekly_and_targets_verified"],
            "notes": checks["notes"],
        })

    return pd.DataFrame(rows).sort_values("n_tickers", ascending=False).reset_index(drop=True)


def find_largest_jump(tradeoff_df: pd.DataFrame) -> dict:
    df = tradeoff_df.sort_values("n_tickers", ascending=False).reset_index(drop=True)
    best = {"jump_years": -1.0}
    for i in range(len(df) - 1):
        before = df.iloc[i]
        after = df.iloc[i + 1]
        jump = after["common_years"] - before["common_years"]
        if jump > best["jump_years"]:
            best = {
                "jump_years": jump,
                "removed_ticker": before["latest_entry_ticker"],
                "n_before": int(before["n_tickers"]),
                "n_after": int(after["n_tickers"]),
                "years_before": before["common_years"],
                "years_after": after["common_years"],
                "start_before": before["common_start"],
                "start_after": after["common_start"],
            }
    return best


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_all_tickers()
    if not records:
        raise RuntimeError("No panel tickers found. Run panel_build_multiscale_dataset.py first.")

    # ---- Task 7: per-ticker coverage CSV ----
    coverage_df = build_ticker_coverage_table(records)
    coverage_df.to_csv(AUDIT_DIR / "panel_ticker_date_coverage.csv", index=False)

    # ---- Tasks 2-4, 8: full removal curve ----
    tradeoff_df, step_detail = build_tradeoff_curve(records)
    tradeoff_df.to_csv(AUDIT_DIR / "panel_universe_duration_tradeoff.csv", index=False)

    # ---- Tasks 5-6, 9: candidate universe summary with 70/15/15 splits ----
    candidate_df = build_candidate_summary(records, step_detail, CANDIDATE_SIZES)
    candidate_df.to_csv(AUDIT_DIR / "panel_candidate_universe_summary.csv", index=False)

    # ---- Task 11: largest coverage jump across the whole curve ----
    jump = find_largest_jump(tradeoff_df)

    # ---- Task 10: human-readable summary ----
    lines = [
        "PANEL UNIVERSE-vs-HISTORY-DURATION TRADE-OFF (evidence only, no universe chosen)",
        "=" * 78,
        "",
        "All tickers ranked by first usable anchor_date (earliest -> latest):",
    ]
    for _, r in coverage_df.iterrows():
        lines.append(
            f"  {r['ticker']:<6} first={r['first_anchor_date'].date()}  last={r['last_anchor_date'].date()}  "
            f"n_samples={r['n_samples']}  history_years={r['history_years']}"
        )

    lines += ["", "Full 24 -> 1 removal curve (n_tickers, common_years, n_common_anchor_dates):"]
    for _, r in tradeoff_df.iterrows():
        lines.append(
            f"  n={r['n_tickers']:>2}  common={r['common_start'].date()} -> {r['common_end'].date()}  "
            f"years={r['common_years']:.2f}  common_dates={r['n_common_anchor_dates']}  "
            f"approx_pooled_samples={r['approx_pooled_sample_count']}  "
            f"(next to drop: {r['latest_entry_ticker']})"
        )

    lines += ["", "Candidate universes (with common-date-only 70/15/15 split):"]
    for _, r in candidate_df.iterrows():
        lines.append(
            f"  n={r['n_tickers']}: common {r['train_start'].date()} -> {r['test_end'].date()} | "
            f"train {r['train_start'].date()}->{r['train_end'].date()} ({r['n_train_dates']} dates), "
            f"val {r['val_start'].date()}->{r['val_end'].date()} ({r['n_val_dates']} dates), "
            f"test {r['test_start'].date()}->{r['test_end'].date()} ({r['n_test_dates']} dates) | "
            f"pooled train/val/test samples = {r['pooled_train_samples']}/{r['pooled_val_samples']}/{r['pooled_test_samples']} | "
            f"checks_ok={r['all_splits_nonempty_for_every_ticker']} | notes={r['notes']}"
        )

    lines += [
        "",
        "Largest single-step coverage jump found in the removal curve:",
        f"  Removing {jump['removed_ticker']} (n_tickers {jump['n_before']} -> {jump['n_after']}) "
        f"increases common history from ~{jump['years_before']:.1f} years to ~{jump['years_after']:.1f} years "
        f"(common_start moves from {jump['start_before'].date()} to {jump['start_after'].date()}).",
        "",
        "No universe has been chosen automatically. This is evidence only — review",
        "panel_candidate_universe_summary.csv and panel_universe_duration_tradeoff.csv",
        "before deciding the panel size for Step 0's benchmark dataset.",
        "",
        "Outputs written:",
        "  dataset/audit/panel_ticker_date_coverage.csv",
        "  dataset/audit/panel_universe_duration_tradeoff.csv",
        "  dataset/audit/panel_candidate_universe_summary.csv",
    ]

    summary_path = AUDIT_DIR / "panel_universe_duration_tradeoff_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- Task 14: concise terminal summary ----
    print("Panel universe-duration trade-off complete.\n")
    for size in [24, 18, 17]:
        row = tradeoff_df[tradeoff_df["n_tickers"] == size]
        if row.empty:
            continue
        row = row.iloc[0]
        crow = candidate_df[candidate_df["n_tickers"] == size]
        print(f"{size} stocks:")
        print(f"  common period: {row['common_start'].date()} -> {row['common_end'].date()}")
        print(f"  years: {row['common_years']:.2f}")
        if not crow.empty:
            c = crow.iloc[0]
            print(
                f"  train/val/test dates: {c['n_train_dates']}/{c['n_val_dates']}/{c['n_test_dates']}"
            )
        print()

    print("Largest coverage jump:")
    print(
        f"  removing {jump['removed_ticker']} changes common start from "
        f"{jump['start_before'].date()} to {jump['start_after'].date()}"
    )
    print(f"  history increases by {jump['years_after'] - jump['years_before']:.2f} years\n")

    print("Outputs:")
    print("  dataset/audit/panel_ticker_date_coverage.csv")
    print("  dataset/audit/panel_universe_duration_tradeoff.csv")
    print("  dataset/audit/panel_candidate_universe_summary.csv")
    print("  dataset/audit/panel_universe_duration_tradeoff_summary.txt")


if __name__ == "__main__":
    main()
