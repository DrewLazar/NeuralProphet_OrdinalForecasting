# Code

## Simulation

`OrdinalNPnotebook.ipynb` runs the simulation study and writes Tables 3–6 and
Figures 2–5. Needs the three modules below in the same directory, and
`../data/BeijingAQI_data.csv` for the smoke test and the noise-ratio cell.
Outputs go to `figures/` and `results/`. Run top to bottom; about an hour.

| File | |
|---|---|
| `OrdinalNeuralProphet.py` | the two-stage method: rolling-origin blind multi-step forecasting, and the Nelder–Mead threshold search with top-*m* averaging |
| `simulate_ordinal_ts.py` | data-generating process, including piecewise-linear trends |
| `run_experiments.py` | reference configuration (`BASE_SIM`, `BASE_MODEL`) and the multi-seed experiment runners |

Constructing `OrdinalNeuralProphet` fits the model, so there is no separate
`.fit()` call.

## Reproducibility

Each run is determined by its seed, injected into both the data generation and
the model fitting. The reference results pool 40 seeds (22–61); the noise sweep
uses 20 per level (42–61), so its σ = 0.25 row differs slightly from Table 4.

Single-seed figure cells can differ from the paper in the third decimal because
the torch fit is not bit-reproducible across runs. The multi-seed tables
reproduce exactly.

Pinned to `neuralprophet==0.4.2` and `torch==1.11.0`; see `../requirements.txt`.
Later NeuralProphet versions change the API and the learning-rate finder, so the
pin is required rather than advisory.create code folder
