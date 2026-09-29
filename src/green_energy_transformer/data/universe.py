"""Frozen constants for the final-paper dataset."""

FINAL_TICKERS = (
    "AES",
    "BEP",
    "BLDP",
    "BLNK",
    "CSIQ",
    "CWEN",
    "DQ",
    "ENPH",
    "FCEL",
    "FSLR",
    "HASI",
    "JKS",
    "NEE",
    "ORA",
    "PLUG",
    "RUN",
    "SEDG",
)

FEATURE_COLUMNS = ("open", "high", "low", "close", "volume")
DAILY_LOOKBACK = 90
WEEKLY_LOOKBACK = 26
FORECAST_HORIZONS = (5, 10, 20)
MAX_FORECAST_HORIZON = max(FORECAST_HORIZONS)
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
BOUNDARY_PURGE = 20
