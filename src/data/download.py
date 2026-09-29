"""Download the final-paper Daily OHLCV inputs from Yahoo Finance."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from .universe import FEATURE_COLUMNS, FINAL_TICKERS


def standardize_yfinance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a yfinance result to the project's canonical Daily CSV schema."""
    if frame is None or frame.empty:
        raise ValueError("Yahoo Finance returned no rows")

    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [column[0] for column in frame.columns]

    frame = frame.reset_index().rename(
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
    required = ["datetime", *FEATURE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Yahoo Finance result is missing columns: {missing}")

    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise").dt.tz_localize(None)
    return frame[required].sort_values("datetime").reset_index(drop=True)


def download_daily(ticker: str, output_dir: Path) -> Path:
    """Download and save one ticker's complete adjusted Daily OHLCV history."""
    raw = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    daily = standardize_yfinance_frame(raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}_daily.csv"
    daily.to_csv(output_path, index=False)
    return output_path


def download_all(output_dir: Path) -> list[Path]:
    """Download all 17 frozen final-paper tickers, failing on any retrieval error."""
    outputs = []
    for ticker in FINAL_TICKERS:
        try:
            outputs.append(download_daily(ticker, output_dir))
        except Exception as exc:
            raise RuntimeError(f"Failed to download Daily OHLCV for {ticker}") from exc
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for <TICKER>_daily.csv files (default: data/raw)",
    )
    args = parser.parse_args()
    download_all(args.output_dir)


if __name__ == "__main__":
    main()
