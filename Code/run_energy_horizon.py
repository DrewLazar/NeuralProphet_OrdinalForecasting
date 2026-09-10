#!/usr/bin/env python
"""
Ball State energy horizon experiment (fixed-origin ladder), one horizon per job.

Reads energy_data.csv (Beijing format: datetime, load, temp, occupancy binaries,
load_category). Fits the two-stage model at fixed origins for the horizon given
in $HORIZON, pools the test-block predictions, and writes per-fold and pooled
summaries plus a persistence baseline.

Energy specifics: temp is lagged+future; the occupancy binaries are future-only
(needs the PATCHED OrdinalNeuralProphet_v2.py). FIRST_ORIGIN=2000. Because the
series is long (~11.6k hours), the number of non-overlapping origins can be huge
at short horizons, so MAX_ORIGINS caps how many are used (evenly subsampled) to
keep each job's runtime reasonable -- ~100 origins already gives a stable pooled
estimate.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from OrdinalNeuralProphet_v2 import OrdinalNeuralProphet

DATA        = os.environ.get('ENERGY_CSV', 'energy_data.csv')
OUTDIR      = os.environ.get('OUTDIR', 'energy_out')
HORIZON     = int(os.environ.get('HORIZON', '72'))
VAL         = int(os.environ.get('VAL_PERIODS', '500'))
N_LAGS      = int(os.environ.get('N_LAGS', '24'))
N_FC        = int(os.environ.get('N_FORECASTS', '12'))
SEED        = int(os.environ.get('SEED', '22'))
FIRST_ORIGIN= int(os.environ.get('FIRST_ORIGIN', '2000'))
N_STARTS    = int(os.environ.get('N_STARTS', '300'))
EPOCHS      = int(os.environ.get('EPOCHS', '50'))
AR_REG      = float(os.environ.get('AR_REG', '0.3'))
MAX_ORIGINS = int(os.environ.get('MAX_ORIGINS', '100'))   # cap fits per job
TAG         = os.environ.get('TAG', f'H{HORIZON}_seed{SEED}')

WEATHER = ['temp']                                                   # lagged + future
OCCUP   = ['is_not_weekend', 'in_session', 'is_not_holiday', 'is_not_summerbreak']  # future only
FUTURE  = WEATHER + OCCUP

os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(DATA)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')
print(f"loaded {DATA}: {df.shape[0]} rows, cols={list(df.columns)}", flush=True)

H = HORIZON
STEP = H
ALL_ORIGINS = list(range(FIRST_ORIGIN, len(df) - H + 1, STEP))
# evenly subsample if there are too many, to bound runtime
if len(ALL_ORIGINS) > MAX_ORIGINS:
    idx = np.linspace(0, len(ALL_ORIGINS) - 1, MAX_ORIGINS).round().astype(int)
    ORIGINS = [ALL_ORIGINS[i] for i in idx]
else:
    ORIGINS = ALL_ORIGINS
print(f"H={H}: {len(ALL_ORIGINS)} non-overlapping origins available, using {len(ORIGINS)}", flush=True)

y_all = df['load_category'].astype(int).values
rows = []
pool_true, pool_base, pool_opt, pool_persist = [], [], [], []

for origin in ORIGINS:
    sub = df.iloc[:origin + H].copy()
    try:
        m = OrdinalNeuralProphet(
            sub, 'load_category', seed=SEED,
            n_lags=N_LAGS, n_forecasts=N_FC, test_periods=H,
            val_periods=VAL, num_categories=4,
            forecast_only=False, round_intermediate=False,
            future_covariates=FUTURE, covariates=WEATHER,
            freq='h', epochs=EPOCHS, ar_reg=AR_REG,
            val_mode='refit',
        )
        res = m.optimize_thresholds(gap_method='softplus', n_starts=N_STARTS,
                                    kappa_weights='quadratic')
        yt = np.asarray(res['y_test_true'])
        yb = np.asarray(res['y_test_baseline'])
        yo = np.asarray(res['y_test_optimized'])
        yp = np.full(len(yt), y_all[origin - 1], dtype=int)
        pool_true.append(yt); pool_base.append(yb)
        pool_opt.append(yo);  pool_persist.append(yp)
        rows.append({
            'H': H, 'origin': origin, 'n_classes': len(np.unique(yt)),
            'base_acc': (yt == yb).mean(), 'opt_acc': (yt == yo).mean(),
            'persist_acc': (yt == yp).mean(),
            'base_mae': np.abs(yt - yb).mean(), 'opt_mae': np.abs(yt - yo).mean(),
            'persist_mae': np.abs(yt - yp).mean(),
        })
        print(f"  origin={origin} done (opt_acc={(yt==yo).mean():.3f})", flush=True)
    except Exception as e:
        print(f"  origin={origin} FAILED: {e}", flush=True)

if not pool_true:
    print("no successful folds; exiting", flush=True); raise SystemExit(1)

T = np.concatenate(pool_true); B = np.concatenate(pool_base)
O = np.concatenate(pool_opt);  P = np.concatenate(pool_persist)
summary = {
    'H': H, 'n_origins': len(pool_true), 'n_pooled': len(T),
    'persist_kappa': cohen_kappa_score(T, P, weights='quadratic'),
    'base_kappa':    cohen_kappa_score(T, B, weights='quadratic'),
    'opt_kappa':     cohen_kappa_score(T, O, weights='quadratic'),
    'persist_acc': (T == P).mean(), 'base_acc': (T == B).mean(), 'opt_acc': (T == O).mean(),
    'persist_mae': np.abs(T - P).mean(), 'base_mae': np.abs(T - B).mean(), 'opt_mae': np.abs(T - O).mean(),
}
pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, f'energy_horizon_perfold_{TAG}.csv'), index=False)
pd.DataFrame([summary]).to_csv(os.path.join(OUTDIR, f'energy_horizon_summary_{TAG}.csv'), index=False)
print(f"\nH={H} | folds {summary['n_origins']} | "
      f"kappa persist {summary['persist_kappa']:.3f} base {summary['base_kappa']:.3f} "
      f"opt {summary['opt_kappa']:.3f} | gain {summary['opt_kappa']-summary['base_kappa']:+.3f}", flush=True)
print(f"wrote energy_horizon_summary_{TAG}.csv", flush=True)
