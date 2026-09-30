# Adaptive Multi-Scale Transformer for Green-Energy Stock Return Forecasting

Official implementation of:

**An Adaptive Multi-Scale Transformer for Green-Energy Stock Return Forecasting:  
A Controlled Evaluation of Static Scale Aggregation and Sample-Dependent Routing**

This project develops a PathFormer-inspired multi-scale Transformer for forecasting
green-energy stock returns and evaluates a central modeling question:

> Does sample-dependent scale routing add predictive value beyond a globally
> learned static mixture of temporal scales?

The empirical study uses daily OHLCV data for 17 green-energy-related equities
from Yahoo Finance and evaluates 5-, 10-, and 20-day forward return forecasts.

## Research Workflow

![Research workflow](docs/graphic_abstract.png)

## Key Findings

- **Multi-scale representation shows horizon-dependent value.** In the controlled ablation, multi-scale aggregation provides the clearest descriptive benefit at the 20-day forecasting horizon.

- **Adaptive routing does not improve upon static aggregation.** Across five pre-specified seeds, the Static model achieves approximately 2.5%–5.1% lower mean test MSE than the Adaptive model across the three forecast horizons.

- **The Adaptive–Static difference is robust to serial-dependence-aware inference.** Moving-block bootstrap and HAC/Diebold–Mariano-style analyses consistently favor Static aggregation.

- **Adaptive routing is sample-dependent but diffuse.** Router diagnostics indicate approximately 3.8 effective scales out of four, with routing variation driven more strongly by cross-sectional differences across stocks than by within-stock temporal changes.

- **Absolute return predictability remains weak.** None of the learned configurations in the main benchmark outperforms the Global Train-Mean predictor in pooled test MSE.
