"""
Task (Advisor Pivot) - Download Daily + Weekly OHLCV for the Green Energy Panel

Downloads Yahoo Finance daily and weekly bars for every ticker in
`panel_universe.GREEN_ENERGY_UNIVERSE` and saves them in the same schema used
by the existing FSLR pipeline (datetime,open,high,low,close,volume), so the
existing multi-scale dataset logic can be reused with minimal changes.

Output:
  dataset/finance/panel_raw/<TICKER>_daily.csv
  dataset/finance/panel_raw/<TICKER>_weekly.csv
  dataset/finance/panel_raw/panel_download_manifest.csv

Only Daily + Weekly are downloaded here. Hourly/Half-Day remain FSLR-only
(see project_progress.md, "Advisor Pivot" section) due to Bloomberg-only
Hourly coverage and yfinance's ~730-day Hourly history cap.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.python.panel_universe import GREEN_ENERGY_UNIVERSE

OUT_DIR = ROOT / "dataset" / "finance" / "panel_raw"
MIN_ROWS_DAILY = 500  # roughly 2 years of trading days; below this we still save but flag it


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance's columns and rename to the project schema."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df = df.rename(
        columns={
            "Date": "datetime",
            "Datetime": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def download_ticker(ticker: str) -> dict:
    record = {"ticker": ticker, "daily_rows": 0, "weekly_rows": 0, "status": "ok", "note": ""}
    try:
        daily = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)
        weekly = yf.download(ticker, period="max", interval="1wk", auto_adjust=True, progress=False)
    except Exception as exc:  # noqa: BLE001 - network/vendor errors, log and continue
        record["status"] = "download_error"
        record["note"] = str(exc)
        return record

    if daily is None or daily.empty:
        record["status"] = "empty_daily"
        return record

    daily = _standardize(daily)
    weekly = _standardize(weekly) if weekly is not None and not weekly.empty else pd.DataFrame(
        columns=["datetime", "open", "high", "low", "close", "volume"]
    )

    daily.to_csv(OUT_DIR / f"{ticker}_daily.csv", index=False)
    weekly.to_csv(OUT_DIR / f"{ticker}_weekly.csv", index=False)

    record["daily_rows"] = len(daily)
    record["weekly_rows"] = len(weekly)
    if len(daily) < MIN_ROWS_DAILY:
        record["status"] = "short_history"
        record["note"] = f"only {len(daily)} daily rows (< {MIN_ROWS_DAILY})"
    return record


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for i, ticker in enumerate(GREEN_ENERGY_UNIVERSE, start=1):
        print(f"[{i}/{len(GREEN_ENERGY_UNIVERSE)}] downloading {ticker} ...")
        record = download_ticker(ticker)
        manifest.append(record)
        print(f"    -> {record}")
        time.sleep(0.5)  # be polite to the data vendor

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(OUT_DIR / "panel_download_manifest.csv", index=False)

    print("\nDownload Summary")
    print("-" * 60)
    print(manifest_df.to_string(index=False))
    n_ok = (manifest_df["status"] == "ok").sum()
    n_short = (manifest_df["status"] == "short_history").sum()
    n_fail = manifest_df["status"].isin(["download_error", "empty_daily"]).sum()
    print(f"\nok={n_ok} short_history={n_short} failed={n_fail} total={len(manifest_df)}")


if __name__ == "__main__":
    main()
