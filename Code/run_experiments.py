"""
Experiment harness for OrdinalNeuralProphet on simulated data.

One atomic operation underlies every experiment:
    run_once(sim_kwargs, model_kwargs, seed) -> dict of metrics
i.e. simulate data with known truth, fit the model, optimize thresholds,
measure latent recovery, and return a flat row of metrics.

The three experiment "types" are just different ways of varying the inputs:
    - Multi-seed reproducibility -> one config, many seeds
    - Simulation sweep           -> vary sim params (noise, AR, trend, ...)
    - Model ablations            -> vary model params (ar_reg, covariate mode, ...)

Each calls run_experiment(configs, seeds) and then summarize(results).

Drop this in the same folder as simulate_ordinal_ts.py and
OrdinalNeuralProphet_v2.py.
"""

import io
import time
import logging
import warnings
import contextlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from simulate_ordinal_ts import simulate_ordinal_timeseries
from OrdinalNeuralProphet import OrdinalNeuralProphet

warnings.filterwarnings("ignore")
logging.getLogger("NP").setLevel(logging.ERROR)


# =====================================================================
# BASE CONFIGURATIONS
# Your current "known-good" reference settings. Every experiment starts
# from these and overrides only the parameters under investigation.
# =====================================================================
BASE_SIM = dict(
    n=3500,
    start_date='2024-01-01 00:00:00',
    freq='h',
    num_categories=4,
     thresholds=[-0.8, 0.0, 1.0],          # explicit-threshold mode
    trend_slope=-9e-5,
    daily_amplitude=0.6,
    weekly_amplitude=0.3,
    betas=(0.30, 0.20, -0.40, 0.15),
    ar_coefs=(0.5,),
    noise_std=0.25,
    # seed is injected per-run by run_once()
)

BASE_MODEL = dict(
    n_lags=24,
    n_forecasts=12,
    test_periods=72,                    # match Beijing
    val_periods=500,
    num_categories=4,
    forecast_only=False,
    round_intermediate=False,
    freq='h',
    covariates=['X1', 'X2', 'X3', 'X4'],
    future_covariates=['X1', 'X2', 'X3', 'X4'],   # match Beijing
    epochs=50,                          # match Beijing
    #learning_rate=0.03,
    val_mode='refit',
    ar_reg=.5                        # match Beijing
)


def make_config(name, sim=None, model=None):
    """Build a named config by overriding BASE_SIM / BASE_MODEL.

    Example:
        make_config('high_noise', sim=dict(noise_std=0.5))
        make_config('no_reg',     model=dict(ar_reg=None))
    """
    return {
        'name': name,
        'sim': {**BASE_SIM, **(sim or {})},
        'model': {**BASE_MODEL, **(model or {})},
    }


# =====================================================================
# SINGLE RUN
# =====================================================================
def run_once(sim_kwargs, model_kwargs, seed, n_starts=200, quiet=True):
    """Simulate -> fit -> optimize -> measure. Returns a flat metrics dict.

    The same `seed` is injected into BOTH the simulator and the model, so a
    given seed fully determines the run (data generation + NN training).
    """
    sim_kwargs = {**sim_kwargs, 'seed': seed}
    model_kwargs = {**model_kwargs, 'seed': seed}

    # 1. Simulate (df keeps y_latent for the recovery diagnostic)
    df, truth = simulate_ordinal_timeseries(**sim_kwargs)
    df = df.set_index('datetime')

    # 2-4. Fit + baseline + threshold optimization (silence the model's prints)
    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        model = OrdinalNeuralProphet(df, target='aqi_category', **model_kwargs)
        model.assessment()
        results = model.optimize_thresholds(
            gap_method='softplus', n_starts=n_starts, kappa_weights='quadratic',
        )

    # 4b. Compute kappa post-hoc from the same test predictions, for BOTH
    # weightings. This decouples the reported metrics from whichever weighting
    # was used during threshold optimization. If you switch the optimization
    # objective (e.g. kappa_weights='linear' above), the harness still reports
    # both quadratic and linear kappa here.
    y_true = results['y_test_true']
    y_base = results['y_test_baseline']
    y_opt  = results['y_test_optimized']
    quadratic_baseline_kappa  = cohen_kappa_score(y_true, y_base, weights='quadratic')
    quadratic_optimized_kappa = cohen_kappa_score(y_true, y_opt,  weights='quadratic')
    linear_baseline_kappa     = cohen_kappa_score(y_true, y_base, weights='linear')
    linear_optimized_kappa    = cohen_kappa_score(y_true, y_opt,  weights='linear')

    # 5. Latent recovery on the test window: is yhat_continuous monotonic in z?
    ts, te = model.val_end_idx, model.test_end_idx
    yhat = model.forecast_df['yhat_continuous'].iloc[ts:te].values
    z = df['y_latent'].iloc[ts:te].values
    mask = ~np.isnan(yhat)
    yhat, z = yhat[mask], z[mask]
    if len(yhat) > 2:
        pearson = float(np.corrcoef(yhat, z)[0, 1])
        spearman = float(spearmanr(yhat, z).correlation)
    else:
        pearson = spearman = np.nan

    # 6. Flatten everything into one row
    return {
        'seed': seed,
        # quadratic kappa (the optimization objective)
        'baseline_kappa':       results['baseline_kappa'],
        'optimized_kappa':      results['optimized_kappa'],
        'kappa_improvement':    results['optimized_kappa'] - results['baseline_kappa'],
        # quadratic kappa, recomputed by the harness independently of the
        # optimization objective. Equal to baseline_kappa/optimized_kappa when
        # the optimizer uses quadratic; differs if the optimizer switches to
        # linear (or another weighting).
        'quadratic_baseline_kappa':    quadratic_baseline_kappa,
        'quadratic_optimized_kappa':   quadratic_optimized_kappa,
        'quadratic_kappa_improvement': quadratic_optimized_kappa - quadratic_baseline_kappa,
        # linear kappa (reported only, computed post-hoc on the same predictions)
        'linear_baseline_kappa':    linear_baseline_kappa,
        'linear_optimized_kappa':   linear_optimized_kappa,
        'linear_kappa_improvement': linear_optimized_kappa - linear_baseline_kappa,
        # validation-set kappa (the optimization objective, on val not test)
        'val_kappa': results['val_kappa'],
        # other test-set metrics
        'baseline_accuracy':    results['baseline_accuracy'],
        'optimized_accuracy':   results['optimized_accuracy'],
        'baseline_mae':         results['mae_baseline'],
        'optimized_mae':        results['mae_optimized'],
        'within_one_baseline':  results['within_one_baseline'],
        'within_one_optimized': results['within_one_optimized'],
        # latent recovery diagnostic
        'pearson_recovery':  pearson,
        'spearman_recovery': spearman,
        # bookkeeping
        'cat_proportions':      tuple(round(p, 3) for p in truth['category_proportions']),
        'recovered_thresholds': tuple(round(t, 3) for t in results['optimal_thresholds']),
    }


# =====================================================================
# EXPERIMENT LOOP
# =====================================================================
def run_experiment(configs, seeds, n_starts=200, quiet=True,
                   progress=True, save_path=None):
    """Run every (config x seed) combination; return a tidy long DataFrame.

    Failures are recorded as rows with status != 'ok' rather than crashing
    the whole sweep. If save_path is given, results are written after each
    run so a long sweep is crash-resilient.
    """
    rows = []
    total = len(configs) * len(seeds)
    count = 0
    t0 = time.time()

    for config in configs:
        for seed in seeds:
            count += 1
            try:
                metrics = run_once(config['sim'], config['model'], seed,
                                   n_starts=n_starts, quiet=quiet)
                metrics['status'] = 'ok'
            except Exception as e:
                metrics = {'seed': seed, 'status': f'FAILED: {type(e).__name__}: {e}'}
            metrics['config'] = config['name']
            rows.append(metrics)

            if progress:
                elapsed = time.time() - t0
                k = metrics.get('optimized_kappa', float('nan'))
                kstr = f"{k:.3f}" if isinstance(k, float) and not np.isnan(k) else "  -  "
                status = metrics['status'] if metrics['status'] != 'ok' else ''
                print(f"[{count:3d}/{total}] {config['name']:<22} seed={seed} "
                      f"opt_kappa={kstr}  ({elapsed:5.0f}s)  {status}")

            if save_path:
                pd.DataFrame(rows).to_csv(save_path, index=False)

    df = pd.DataFrame(rows)
    # Put identifying columns first
    front = [c for c in ['config', 'seed', 'status'] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


# =====================================================================
# AGGREGATION
# =====================================================================
def summarize(results, metrics=('optimized_kappa', 'baseline_kappa',
                                 'kappa_improvement', 'spearman_recovery')):
    """Mean +/- SD of key metrics across seeds, grouped by config.

    Returns a DataFrame with a 'mean' and 'std' column per metric, plus an
    'n' column counting successful runs per config.
    """
    ok = results[results['status'] == 'ok'].copy()
    if ok.empty:
        print("No successful runs to summarize.")
        return pd.DataFrame()

    agg = ok.groupby('config')[list(metrics)].agg(['mean', 'std'])
    agg['n'] = ok.groupby('config').size()
    # Preserve the order configs first appeared in
    order = list(dict.fromkeys(results['config']))
    agg = agg.reindex([c for c in order if c in agg.index])
    return agg


def pretty_summary(results, metrics=('optimized_kappa', 'baseline_kappa',
                                      'kappa_improvement', 'spearman_recovery')):
    """Print a compact 'mean +/- std (n)' table."""
    ok = results[results['status'] == 'ok'].copy()
    if ok.empty:
        print("No successful runs to summarize.")
        return
    order = list(dict.fromkeys(results['config']))
    header = f"{'config':<22} " + " ".join(f"{m:>22}" for m in metrics) + f" {'n':>4}"
    print(header)
    print("-" * len(header))
    for cfg in order:
        sub = ok[ok['config'] == cfg]
        if sub.empty:
            continue
        cells = []
        for m in metrics:
            mean, sd = sub[m].mean(), sub[m].std()
            cells.append(f"{mean:6.3f} +/- {sd:5.3f}")
        print(f"{cfg:<22} " + " ".join(f"{c:>22}" for c in cells) + f" {len(sub):>4}")


# =====================================================================
# EXAMPLE EXPERIMENT DEFINITIONS
# Each returns (configs, seeds) ready to pass to run_experiment().
# =====================================================================
def experiment_multiseed(seeds=tuple(range(42, 52))):
    """One reference config, many seeds -> reproducibility / variance."""
    configs = [make_config('reference')]
    return configs, list(seeds)


def experiment_noise_sweep(seeds=(42, 43, 44)):
    """Vary noise_std -> how gracefully does performance degrade?"""
    configs = [make_config(f'noise={ns}', sim=dict(noise_std=ns))
               for ns in (0.15, 0.25, 0.40, 0.60, 0.80)]
    return configs, list(seeds)


def experiment_ar_sweep(seeds=(42, 43, 44)):
    """Vary the true AR(1) strength in the DGP -> does the model track it?"""
    configs = [make_config(f'ar={rho}', sim=dict(ar_coefs=(rho,)))
               for rho in (0.0, 0.3, 0.5, 0.7, 0.9)]
    return configs, list(seeds)


def experiment_ablations(seeds=(42, 43, 44)):
    """Model-side ablations holding the DGP fixed."""
    cov = ['X1', 'X2', 'X3', 'X4']
    configs = [
        make_config('full'),                                          # reference
        make_config('no_reg',      model=dict(ar_reg=None)),          # no AR sparsity
        make_config('strong_reg',  model=dict(ar_reg=1.0)),           # aggressive sparsity
        make_config('cov_lags_6',  model=dict(covariate_lags=6)),     # shorter covariate window
        make_config('future_cov',  model=dict(covariates=[],         # contemporaneous only
                                              future_covariates=cov)),
        make_config('both_cov',    model=dict(future_covariates=cov)),# distributed lag
    ]
    return configs, list(seeds)


def experiment_hidden_factor(seeds=(42, 43, 44)):
    """Turn on the unobserved AR factor; compare AR regularization settings.

    With a hidden autocorrelated driver, AR-on-y is the only mechanism that
    can capture it -- so ar_reg that's too aggressive should hurt here.
    """
    hidden = dict(unobserved_ar=0.9, unobserved_sd=0.4)
    configs = [
        make_config('hidden_no_reg',     sim=hidden, model=dict(ar_reg=None)),
        make_config('hidden_mild_reg',   sim=hidden, model=dict(ar_reg=0.1)),
        make_config('hidden_strong_reg', sim=hidden, model=dict(ar_reg=1.0)),
    ]
    return configs, list(seeds)


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    # Pick one experiment to run. Start small (few seeds, fewer epochs) to
    # sanity-check timing before launching a big sweep.
    configs, seeds = experiment_multiseed(seeds=(42, 43, 44))

    results = run_experiment(
        configs, seeds,
        n_starts=200,
        quiet=True,
        progress=True,
        save_path='experiment_results.csv',
    )

    print("\n" + "=" * 70)
    print("SUMMARY (mean +/- std across seeds)")
    print("=" * 70)
    pretty_summary(results)

    print("\nFull results saved to experiment_results.csv")