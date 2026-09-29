# Data

Daily OHLCV data are obtained exclusively from Yahoo Finance using the
`yfinance` Python package.

Weekly observations are deterministically aggregated from the daily series
and therefore represent an alternative temporal resolution of the same
underlying market data.

Raw downloaded data are not stored in this repository. The data preparation
pipeline will reproduce the dataset from Yahoo Finance.
