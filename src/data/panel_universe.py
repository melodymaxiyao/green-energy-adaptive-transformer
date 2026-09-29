"""
Green Energy Panel - Ticker Universe (Advisor Pivot, 2026-07-08)

Defines the multi-stock panel used for the adaptive multi-scale PathFormer
experiments. FSLR remains the single-stock case study (see project_progress.md);
this universe is used for the main quantitative experiments.

Panel scope decision (per advisor pivot):
- Frequencies used for the panel: Daily + Weekly only.
  (True Bloomberg-quality Hourly/Half-Day data only exists for FSLR; other
  tickers only have Yahoo Finance daily-derived history, and yfinance 1h bars
  are capped at ~730 days, which is too short for this panel.)
- Category: green energy (solar, renewable IPPs, storage, hydrogen/fuel cell).

Tickers are grouped by expected data-history depth so downstream scripts can
warn/skip tickers with insufficient history for the train/val/test split.
"""

# Long history (pre-2016), most similar in depth to the FSLR case study.
LONG_HISTORY = [
    "FSLR",  # First Solar
    "CSIQ",  # Canadian Solar
    "NEE",   # NextEra Energy (renewable-heavy utility)
    "AES",   # The AES Corporation (renewable-heavy utility)
    "ORA",   # Ormat Technologies (geothermal)
    "PLUG",  # Plug Power (hydrogen fuel cell)
    "FCEL",  # FuelCell Energy
    "BLDP",  # Ballard Power Systems
    "JKS",   # JinkoSolar
]

# Medium history (roughly 2012-2016 IPOs).
MEDIUM_HISTORY = [
    "ENPH",  # Enphase Energy
    "SEDG",  # SolarEdge Technologies
    "RUN",   # Sunrun
    "BEP",   # Brookfield Renewable Partners
    "NEP",   # NextEra Energy Partners
    "BE",    # Bloom Energy
    "AY",    # Atlantica Sustainable Infrastructure
    "HASI",  # Hannon Armstrong Sustainable Infrastructure
    "DQ",    # Daqo New Energy
    "CWEN",  # Clearway Energy
    "BLNK",  # Blink Charging
]

# Short history (2019-2021 IPOs) - keep, but expect a tighter test split.
SHORT_HISTORY = [
    "ARRY",  # Array Technologies
    "SHLS",  # Shoals Technologies Group
    "NOVA",  # Sunnova Energy
    "MAXN",  # Maxeon Solar Technologies
    "STEM",  # Stem Inc.
    "FLNC",  # Fluence Energy
    "CHPT",  # ChargePoint Holdings
]

GREEN_ENERGY_UNIVERSE = LONG_HISTORY + MEDIUM_HISTORY + SHORT_HISTORY

# Optional caution list: known corporate actions (reverse splits, bankruptcy,
# restructuring) that can distort return series. Not excluded automatically,
# but flagged for review if they show up as outliers in the data audit.
CAUTION_TICKERS = {
    "SPWR": "Chapter 11 filing / delisting in 2024, excluded from universe.",
}

# Confirmed delisted / unavailable via yfinance as of 2026-07-08 download run
# (panel_download_daily_weekly.py). Removed from the active universe below.
DELISTED_UNAVAILABLE = {
    "NEP": "possibly delisted; no data returned by yfinance",
    "AY": "possibly delisted; no data returned by yfinance",
    "NOVA": "possibly delisted; no data returned by yfinance",
}
for _bad_ticker in DELISTED_UNAVAILABLE:
    if _bad_ticker in GREEN_ENERGY_UNIVERSE:
        GREEN_ENERGY_UNIVERSE.remove(_bad_ticker)

if __name__ == "__main__":
    print(f"Universe size: {len(GREEN_ENERGY_UNIVERSE)}")
    print("Long history  :", LONG_HISTORY)
    print("Medium history:", MEDIUM_HISTORY)
    print("Short history :", SHORT_HISTORY)
    print("Delisted/removed:", list(DELISTED_UNAVAILABLE))

