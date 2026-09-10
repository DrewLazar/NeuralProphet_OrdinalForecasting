# Code

## Simulation

`OrdinalNPnotebook.ipynb` produces Tables 3–6 and Figures 2–5. Needs
`OrdinalNeuralProphet.py`, `simulate_ordinal_ts.py` and `run_experiments.py` in
the same directory, plus `../data/BeijingAQI_data.csv`. Run top to bottom;
about an hour.

## Real data

| File | |
|---|---|
| `run_beijing_horizon.py`, `run_energy_horizon.py` | horizon ladders, Tables 9 and 10. See the matching `launch_*.sh` |
| `beijing_logistic_baselines_sweep.R`, `energy_logistic_baselines_sweep.R` | multinomial logistic baselines, Table 8. `AR_SPEC=fixed` is what the paper reports |
| `real_data_decomposition.ipynb` | component SDs (Table 11) and Figures 8–9 |
| `make_ordinal_series_figure.py` | Figure 6 |
| `make_energy_excerpt.R` | builds the public energy excerpt in `../data/` |

The energy cells in `real_data_decomposition.ipynb` need the full load record,
which is not redistributed. The Beijing half runs from the repository as-is.

## Core modules

`OrdinalNeuralProphet.py` implements the two-stage method: the rolling-origin
loop for blind multi-step forecasting, and the Nelder–Mead threshold search
with top-*m* averaging. Constructing the object fits the model, so there is no
separate `.fit()`.

`simulate_ordinal_ts.py` is the data-generating process, including
piecewise-linear trends. `run_experiments.py` holds the reference
configuration (`BASE_SIM`, `BASE_MODEL`) and the multi-seed runners.
