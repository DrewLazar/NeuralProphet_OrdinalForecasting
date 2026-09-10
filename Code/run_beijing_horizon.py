#!/usr/bin/env python
"""
Beijing AQI horizon experiment (fixed-origin ladder).

Fits the two-stage ordinal model at a set of fixed origins for each horizon in
HORIZONS, pools the test-block predictions per horizon, and writes per-fold and
pooled-summary CSVs. Intended to run as a batch job (see submit_beijing.slurm).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from OrdinalNeuralProphet import OrdinalNeuralProphet

# --- config ---
DATA = os.environ.get('BEIJING_CSV', 'BeijingAQI_data.csv')
OUTDIR = os.environ.get('OUTDIR', 'beijing_out')
# One horizon per job: read it from env (HORIZON), default 72.
HORIZONS = [int(os.environ.get('HORIZON', '72'))]
VAL     = int(os.environ.get('VAL_PERIODS', '500'))
N_LAGS  = int(os.environ.get('N_LAGS', '24'))
N_FC    = int(os.environ.get('N_FORECASTS', '12'))
SEED    = int(os.environ.get('SEED', '22'))
FIRST_ORIGIN = int(os.environ.get('FIRST_ORIGIN', '2100'))
N_STARTS = int(os.environ.get('N_STARTS', '300'))
EPOCHS   = int(os.environ.get('EPOCHS', '40'))
AR_REG   = float(os.environ.get('AR_REG', '0.5'))
TAG      = os.environ.get('TAG', f'H{HORIZONS[0]}_seed{SEED}')

os.makedirs(OUTDIR, exist_ok=True)

# --- load data fresh (self-contained; do not rely on any prior state) ---
df = pd.read_csv(DATA, parse_dates=['datetime']).set_index('datetime')
print(f"loaded {DATA}: {df.shape[0]} rows, columns={list(df.columns)}", flush=True)

MAX_H = max(HORIZONS)
STEP = MAX_H
ORIGINS = list(range(FIRST_ORIGIN, len(df) - MAX_H + 1, STEP))
print(f"{len(ORIGINS)} origins: {ORIGINS}", flush=True)
print(f"horizons: {HORIZONS}", flush=True)
print(f"total fits: {len(ORIGINS) * len(HORIZONS)}", flush=True)

y_all = df['aqi_category'].astype(int).values
rows = []
horizon_summary = []

for H in HORIZONS:
    pool_true, pool_base, pool_opt, pool_persist = [], [], [], []
    for origin in ORIGINS:
        sub = df.iloc[:origin + H].copy()
        try:
            m = OrdinalNeuralProphet(
                sub, 'aqi_category', seed=SEED,
                n_lags=N_LAGS, n_forecasts=N_FC, test_periods=H,
                val_periods=VAL, num_categories=4,
                forecast_only=False, round_intermediate=False,
                future_covariates=['TEMP', 'DEWP', 'PRES', 'Iws'],
                covariates=['TEMP', 'DEWP', 'PRES', 'Iws'],
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
                'base_acc': (yt == yb).mean(),
                'opt_acc':  (yt == yo).mean(),
                'persist_acc': (yt == yp).mean(),
                'base_mae': np.abs(yt - yb).mean(),
                'opt_mae':  np.abs(yt - yo).mean(),
                'persist_mae': np.abs(yt - yp).mean(),
            })
            print(f"  H={H} origin={origin} done "
                  f"(n_test={len(yt)}, opt_acc={(yt==yo).mean():.3f})", flush=True)
        except Exception as e:
            print(f"  H={H} origin={origin} FAILED: {e}", flush=True)

    if not pool_true:
        print(f"H={H}: no successful folds, skipping summary", flush=True)
        continue

    T = np.concatenate(pool_true); B = np.concatenate(pool_base)
    O = np.concatenate(pool_opt);  P = np.concatenate(pool_persist)
    horizon_summary.append({
        'H': H, 'n_origins': len(pool_true), 'n_pooled': len(T),
        'persist_kappa': cohen_kappa_score(T, P, weights='quadratic'),
        'base_kappa':    cohen_kappa_score(T, B, weights='quadratic'),
        'opt_kappa':     cohen_kappa_score(T, O, weights='quadratic'),
        'persist_acc': (T == P).mean(), 'base_acc': (T == B).mean(), 'opt_acc': (T == O).mean(),
        'persist_mae': np.abs(T - P).mean(), 'base_mae': np.abs(T - B).mean(), 'opt_mae': np.abs(T - O).mean(),
    })
    hs = horizon_summary[-1]
    print(f"H={H:3d} | folds {hs['n_origins']:2d} | "
          f"kappa persist {hs['persist_kappa']:.3f} base {hs['base_kappa']:.3f} opt {hs['opt_kappa']:.3f} | "
          f"acc opt {hs['opt_acc']:.3f} | mae opt {hs['opt_mae']:.3f}", flush=True)

perfold_path = os.path.join(OUTDIR, f'beijing_horizon_perfold_{TAG}.csv')
summary_path = os.path.join(OUTDIR, f'beijing_horizon_summary_{TAG}.csv')
pd.DataFrame(rows).to_csv(perfold_path, index=False)
hsum = pd.DataFrame(horizon_summary)
hsum.to_csv(summary_path, index=False)
print(f"\nwrote {perfold_path} and {summary_path}", flush=True)

if len(hsum):
    print("\n=== DEGRADATION SUMMARY (pooled per H) ===", flush=True)
    print(hsum[['H', 'n_origins', 'n_pooled', 'persist_kappa', 'base_kappa', 'opt_kappa',
                'opt_acc', 'opt_mae']].to_string(index=False), flush=True)
