
## Data

**`BeijingAQI_data.csv`** — hourly PM2.5 and weather, UCI Beijing PM2.5 data set, 12 Jan to 9 Jun 2014 (*n* = 3548). `aqi_category` follows the HJ 633-2012 cut-points at 35, 75 and 150 µg/m³.

**`energy_data_excerpt.csv`** — hourly load at Ball State University's Foundational Sciences Building, Sep to Nov 2024 (*n* = 2184), used with the permission of Ball State University Facilities Planning and Management. Category labels and covariates only; the kilowatt readings are not redistributed. Cut-points at 370, 415 and 460 kW are the mean and mean ± 0.7 SD of load. This excerpt runs the method but does not reproduce the paper's load results, which use 1 Apr 2024 to 28 Jul 2025. Requests for the meter data should go to Facilities Planning and Management.

## What produces what

| Paper | Script |
|---|---|
| Tables 3–6, Figures 2–5 | `run_experiments.py` (DGP in `simulate_ordinal_ts.py`) |
| Table 9 | `run_beijing_horizon.py` (`launch_beijing.sh`) |
| Table 10 | `run_energy_horizon.py` (`launch_energy.sh`) |
| Table 8 baselines | `logistic_baselines_sweep.R`, `energy_logistic_baselines_sweep.R` |
| Table 11, Figures 8–9 | `real_data_decomposition.ipynb` |
| Figure 6 | `make_ordinal_series_figure.py` |

Figures 1 and 7 are TikZ in the manuscript. `OrdinalNeuralProphet_v2.py` implements the two-stage method, including the rolling-origin loop that chains NeuralProphet's output blocks across a horizon and the Nelder–Mead threshold search.

## Configuration

`L_a = 24` lags, output block `B = 12`, covariate lags `L_x = 24`, daily and weekly seasonality, softplus thresholds, top-*m* averaging at `m = 3`, `S = 200` Nelder–Mead starts (300 on the real series; results are insensitive beyond about 200).

Scripts read settings from the environment. The R baselines take `AR_SPEC=fixed`, as reported in the paper, or `legacy`, which reproduces an earlier frame in which the covariates were misaligned by one step relative to the prediction call.

Each simulation run is determined by its seed, injected into both data generation and model fitting.

## Requirements

Python: `requirements.txt`. R: `nnet`.
