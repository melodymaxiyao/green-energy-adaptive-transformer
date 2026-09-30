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

## Controlled Evaluation

The central experiment separates the value of **multi-scale representation** from the value of **sample-dependent routing** by comparing four aggregation mechanisms under the same scale-expert architecture.

| Variant | Scale experts | Aggregation weights | Sample-dependent |
| --- | --- | --- | --- |
| **Single** | One selected scale per branch | — | No |
| **Fixed** | Four scales per branch | Equal weights | No |
| **Static** | Four scales per branch | Globally learned softmax weights | No |
| **Adaptive** | Four scales per branch | Observation-dependent router weights | Yes |

This design isolates three questions:

- **Fixed vs. Single:** Does access to multiple temporal scales improve forecasting?
- **Static vs. Fixed:** Does learning global scale importance improve over equal weighting?
- **Adaptive vs. Static:** Does making scale weights observation-dependent provide additional predictive value?

The Daily and Weekly branches use separate within-resolution scale aggregation. When both resolutions are used, their branch representations are combined only afterward through late fusion.

## Data and Experimental Setup

The empirical study uses **17 green-energy-related U.S.-listed equities**:

`AES`, `BEP`, `BLDP`, `BLNK`, `CSIQ`, `CWEN`, `DQ`, `ENPH`, `FCEL`, `FSLR`, `HASI`, `JKS`, `NEE`, `ORA`, `PLUG`, `RUN`, and `SEDG`.

Daily OHLCV data are retrieved exclusively from **Yahoo Finance** using `yfinance` with adjusted prices. Weekly observations are reconstructed deterministically from the Daily series rather than downloaded as a separate data source.

### Forecasting setup

| Component | Specification |
| --- | --- |
| Daily input window | 90 trading days |
| Weekly input window | 26 completed weekly bars |
| Daily patch sizes | 5, 10, 20, 30 days |
| Weekly patch sizes | 2, 4, 8, 13 weeks |
| Forecast horizons | 5, 10, 20 trading days |
| Target | Forward adjusted-price log return |
| Panel | Exact common-date intersection across all 17 stocks |
| Split | Chronological 70% / 15% / 15% Train / Validation / Test |
| Boundary purge | Final 20 common dates removed from Train and Validation |
| Normalization | Fit on Train only, separately by ticker, resolution, and feature |

Weekly OHLCV bars are constructed from Daily observations using:

- Open: first
- High: maximum
- Low: minimum
- Close: last
- Volume: sum

Only **completed weeks** are available to the model at each forecast anchor, preventing unfinished-week information from entering the input.

The final aligned panel contains **43,571 stock-date observations**, with identical retained observations across Daily-only, Weekly-only, and Daily+Weekly configurations.

## Model Architecture

The proposed model uses separate **Daily** and **Weekly** branches. Each branch learns representations across multiple temporal scales before the two resolutions are combined through late fusion.

### Multi-scale experts

Each resolution-specific branch contains four scale experts:

| Branch | Candidate patch sizes |
| --- | --- |
| Daily | 5, 10, 20, 30 trading days |
| Weekly | 2, 4, 8, 13 weeks |

Each expert applies:

1. input projection with fixed sinusoidal positional encoding;
2. patchification at its assigned temporal scale;
3. intra-patch multi-head attention;
4. valid-observation-aware pooling;
5. occupancy-aware patch readout;
6. inter-patch attention;
7. feed-forward transformation and layer normalization.

The shared architecture uses:

| Hyperparameter | Value |
| --- | ---: |
| Embedding dimension | 32 |
| Attention heads | 4 |
| Feed-forward dimension | 64 |
| Dropout | 0.1 |

### Adaptive scale routing

For the Adaptive variant, the four expert representations within each branch are concatenated and passed through a dense router:

`128 → 32 → 4 → Softmax`

The resulting weights form a **dense, observation-dependent mixture** of all four scale experts. Routing is performed independently within the Daily and Weekly branches; the router does not choose between the two resolutions.

### Daily–Weekly late fusion

For the Daily+Weekly configuration, the 32-dimensional Daily and Weekly branch representations are concatenated into a 64-dimensional vector and passed through:

`64 → 64 → 1`

with a ReLU hidden layer and a final linear scalar output for return prediction.
