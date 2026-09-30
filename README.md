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

---

## Research Workflow

![Research workflow](docs/graphic_abstract.png)

---

## Key Findings

- **Multi-scale representation shows horizon-dependent value.** In the controlled ablation, multi-scale aggregation provides the clearest descriptive benefit at the 20-day forecasting horizon.

- **Adaptive routing does not improve upon static aggregation.** Across five pre-specified seeds, the Static model achieves approximately 2.5%–5.1% lower mean test MSE than the Adaptive model across the three forecast horizons.

- **The Adaptive–Static difference is robust to serial-dependence-aware inference.** Moving-block bootstrap and HAC/Diebold–Mariano-style analyses consistently favor Static aggregation.

- **Adaptive routing is sample-dependent but diffuse.** Router diagnostics indicate approximately 3.8 effective scales out of four, with routing variation driven more strongly by cross-sectional differences across stocks than by within-stock temporal changes.

- **Absolute return predictability remains weak.** None of the learned configurations in the main benchmark outperforms the Global Train-Mean predictor in pooled test MSE.

---

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

---

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

---

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

```text
128 → 32 → 4 → Softmax
```

The resulting weights form a **dense, observation-dependent mixture** of all four scale experts. Routing is performed independently within the Daily and Weekly branches; the router does not choose between the two resolutions.

### Daily–Weekly late fusion

For the Daily+Weekly configuration, the 32-dimensional Daily and Weekly branch representations are concatenated into a 64-dimensional vector and passed through:

```text
64 → 64 → 1
```

with a ReLU hidden layer and a final linear scalar output for return prediction.

---

## Benchmark Models

The final benchmark includes:

- Zero Predictor
- Global Train-Mean Predictor
- Ridge Regression
- LSTM
- Vanilla Transformer
- SWiM-style Transformer
- Adaptive Multi-Scale Transformer

Learned models are evaluated under Daily-only, Weekly-only, and Daily+Weekly input configurations where applicable.

Forecast performance is assessed using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Pearson correlation
- Direction accuracy
- Cross-sectional Rank IC
- PredStd / TrueStd

---

## Statistical Evaluation

The primary robustness analysis compares **Adaptive** and **Static** aggregation across five pre-specified seeds:

```text
0, 1, 21, 42, 3407
```

The primary loss differential is:

```text
Adaptive squared loss − Static squared loss
```

so positive values indicate lower squared loss for Static.

Inference includes:

- moving-block bootstrap;
- percentile confidence intervals;
- centered bootstrap p-values;
- Holm adjustment across the three forecast horizons;
- block-length sensitivity analysis;
- HAC standard errors with Bartlett weights;
- Diebold–Mariano-style test statistics.

Router diagnostics additionally evaluate routing dispersion, effective number of scales, cross-sectional and temporal variation, cross-seed stability, dominant-scale agreement, and the association between routing behavior and relative forecast loss.

---

## Repository Structure

```text
green-energy-adaptive-transformer/
├── data/
│   └── README.md
├── docs/
│   └── graphic_abstract.png
├── experiments/
│   ├── 01_benchmark.py
│   ├── 02_input_configurations.py
│   ├── 03_mechanism_ablation.py
│   ├── 04_multiseed_inference.py
│   ├── 05_router_diagnostics.py
│   └── common.py
├── results/
│   └── README.md
├── src/
│   └── green_energy_transformer/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── build_dataset.py
│       │   ├── download.py
│       │   ├── panel.py
│       │   └── universe.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── inference.py
│       │   ├── metrics.py
│       │   └── router.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── baselines.py
│       │   └── multiscale.py
│       └── training/
│           ├── __init__.py
│           └── train.py
├── tests/
│   └── test_temporal_integrity.py
├── pyproject.toml
├── .gitignore
└── README.md
```

The repository separates data construction, model definitions, training, evaluation, and experiment orchestration so that each stage of the final-paper pipeline can be inspected independently.

---

## Installation

Python **3.10 or later** is required.

Clone the repository:

```bash
git clone https://github.com/melodymaxiyao/green-energy-adaptive-transformer.git
cd green-energy-adaptive-transformer
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the project in editable mode:

```bash
pip install -e .
```

To include the test dependency:

```bash
pip install -e ".[test]"
```

The core runtime dependencies are:

- NumPy
- pandas
- SciPy
- scikit-learn
- PyTorch
- yfinance

The public package uses the namespace:

```python
import green_energy_transformer
```

---

## Reproducing the Experiments

The public experiment layer mirrors the final-paper experimental structure.

### 1. Main benchmark

```bash
python experiments/01_benchmark.py
```

Compares the naive predictors, Ridge, LSTM, Vanilla Transformer, SWiM-style Transformer, and Adaptive Multi-Scale Transformer.

### 2. Input configurations

```bash
python experiments/02_input_configurations.py
```

Compares Daily-only, Weekly-only, and Daily+Weekly versions of the Adaptive model.

### 3. Mechanism ablation

```bash
python experiments/03_mechanism_ablation.py
```

Runs the controlled:

```text
Single → Fixed → Static → Adaptive
```

comparison under the Daily+Weekly configuration.

### 4. Multi-seed robustness and inference

```bash
python experiments/04_multiseed_inference.py
```

Runs the five-seed Static-versus-Adaptive comparison and computes the moving-block-bootstrap and HAC/DM inference outputs.

### 5. Router diagnostics

```bash
python experiments/05_router_diagnostics.py
```

Evaluates routing concentration, variation, stability, dominant-scale behavior, and loss associations.

Machine-readable outputs are written to:

```text
results/
```

The experiment scripts deliberately avoid publication-specific plotting, artifact manifests, and legacy development outputs.

---

## Testing

Temporal-integrity tests are provided in:

```text
tests/test_temporal_integrity.py
```

The tests cover:

- the frozen 17-stock universe;
- Daily and Weekly window construction;
- Weekly OHLCV aggregation;
- exclusion of unfinished Weekly bars;
- forward-return target construction;
- chronological partitioning;
- Train and Validation boundary purging;
- Train-only normalization;
- shared Daily/Weekly stock-date indexing;
- balanced common dates across all 17 stocks.

Run:

```bash
pytest
```

---

## Reproducibility Notes

The clean public implementation was refactored from the final-paper research pipeline and checked against the authoritative outputs.

The refactor preserves:

- final ticker universe;
- retained stock-date samples;
- Daily and Weekly arrays;
- 5-, 10-, and 20-day targets;
- chronological split and boundary purge;
- Train-only normalization;
- model architecture and parameterization;
- Single / Fixed / Static / Adaptive execution paths;
- training protocol;
- evaluation metrics;
- moving-block-bootstrap and HAC/DM inference;
- router diagnostics.

### Ridge numerical environment

The authoritative final-paper Ridge benchmark used:

```text
scikit-learn 1.5.0
```

Accordingly, the public package pins:

```text
scikit-learn==1.5.0
```

The original frozen Ridge reference artifacts were generated in a Python 3.9.25 research environment. The clean public package requires Python 3.10+ because it uses modern Python typing syntax.

Exact bit-level Ridge reproduction can additionally depend on the surrounding NumPy/SciPy/BLAS numerical stack. The scikit-learn pin therefore identifies the authoritative estimator version but should not be interpreted as guaranteeing bitwise-identical coefficients across every operating system or linear-algebra backend.

The frozen final-paper predictions remain the numerical reference for validation.

---

## Data Availability

Raw market data are not distributed with this repository.

Daily OHLCV observations are obtained from Yahoo Finance through the `yfinance` Python package. Weekly observations are reconstructed from the downloaded Daily series.

See:

```text
data/README.md
```

for the public data-reproduction policy.

---

## License

No license has been added yet.

Until a license is specified, the repository source remains publicly viewable but should not be assumed to grant unrestricted reuse or redistribution rights.

---

## Citation

If you use this repository in academic work, please cite the associated paper.

A formal citation and DOI can be added here once the publication record is finalized.
