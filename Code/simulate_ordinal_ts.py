"""
Simulate ordinal time-series data with trend, seasonality, and covariate effects.

Data-generating process (latent-variable framework):
    z_t = sum_j rho_j * z_{t-j}                  # AR(p) on z (optional, off by default)
        + alpha * t                              # trend
        + A_day  * sin(2*pi * t / 24  + phi_day) # daily seasonality
        + A_week * sin(2*pi * t / 168 + phi_week)# weekly seasonality
        + sum_k beta_k * X_{k, t-L}              # covariate effects at lag L (L = covariate_lag)
        + u_t                                    # unobserved AR(1) factor (optional)
        + epsilon_t,    epsilon_t ~ N(0, sigma^2)

    u_t = rho_u * u_{t-1} + eta_t                # hidden AR(1); u_t == 0 if unobserved_sd == 0

    y_t = 1 if z_t <= tau_1
        = 2 if tau_1 < z_t <= tau_2
        = 3 if tau_2 < z_t <= tau_3
        = 4 if z_t >  tau_3                       # for num_categories = 4

Three distinct, independent autocorrelation / lag mechanisms are available:
  * ar_coefs       -- AR(p) directly on z (the response has its own momentum).
                      Observable through y, so NeuralProphet's AR-on-y can learn it.
  * covariate_lag  -- z_t is driven by X_{t-L} rather than X_t. Makes the model's
                      *lagged* regressors the correctly-specified recovery path.
  * unobserved_ar  -- a hidden AR(1) factor u_t added to z but never exposed as a
                      covariate, so lagged y is the ONLY signal that can capture it.
All three default to off, recovering the original contemporaneous DGP.

Because the truth (alpha, A_day, A_week, betas, taus, sigma, rhos, L) is known,
this is ideal for sanity-checking OrdinalNeuralProphet.
"""

import numpy as np
import pandas as pd


def build_trend(n, trend_slope, trend_segments=None):
    """Construct the true trend array: single line or continuous piecewise-linear.

    If trend_segments is None, returns a single linear trend trend_slope * t
    (the original behavior). Otherwise trend_segments is a list of
    (start_index, slope) pairs defining a continuous piecewise-linear trend;
    the first segment must start at index 0, start indices must be strictly
    increasing and < n, and each segment continues until the next segment's
    start (the last runs to n). Segments are joined continuously (no jumps).

    Examples
    --------
    build_trend(n, -9e-5)                                  # single line
    build_trend(n, None, [(0, -4e-4), (1750, 4e-4)])       # down then up
    """
    if trend_segments is None:
        return trend_slope * np.arange(n)
    segs = sorted(trend_segments, key=lambda s: s[0])
    if segs[0][0] != 0:
        raise ValueError("first trend segment must start at index 0")
    if any(segs[j][0] >= segs[j + 1][0] for j in range(len(segs) - 1)):
        raise ValueError("trend segment start indices must be strictly increasing")
    if segs[-1][0] >= n:
        raise ValueError("trend segment start indices must be < n")
    trend = np.zeros(n)
    level = 0.0
    for j, (start, slope) in enumerate(segs):
        end = segs[j + 1][0] if j + 1 < len(segs) else n
        seg_len = end - start
        trend[start:end] = level + slope * np.arange(seg_len)
        level = level + slope * seg_len   # continuity at the joint
    return trend


def simulate_ordinal_timeseries(
    n=3500,
    start_date='2024-01-01 00:00:00',
    freq='h',
    num_categories=4,
    # ---- trend ----
    trend_slope=3e-4,        # latent units per time step
    # Optional piecewise-linear trend. List of (start_index, slope) pairs, e.g.
    # [(0, -4e-4), (1750, 4e-4)] for a down-then-up trend with one change at
    # t=1750. Continuous (segments joined without jumps); first must start at 0.
    # When provided, OVERRIDES trend_slope. None = single linear trend (default).
    trend_segments=None,
    # ---- seasonality (latent space) ----
    daily_amplitude=0.6,
    daily_phase=0.0,         # radians; 0 -> peak around hour 6
    weekly_amplitude=0.3,
    weekly_phase=0.0,
    # ---- covariates ----
    # X1: smooth seasonal (TEMP-like)
    # X2: correlated with X1 (DEWP-like)
    # X3: opposite seasonal (PRES-like)
    # X4: spiky / intermittent (Iws-like)
    betas=(0.30, 0.20, -0.40, 0.15),
    x1_x2_corr=0.7,
    x4_spike_prob=0.05,
    x4_spike_scale=3.0,
    # ---- covariate lag ----
    # If > 0, z_t is driven by X_{t - covariate_lag} instead of X_t. A positive
    # lag makes the model's *lagged* regressors the correctly-specified path to
    # recover the covariate effect. 0 = contemporaneous (original behavior).
    covariate_lag=0,
    # ---- noise ----
    noise_std=0.4,
    # ---- AR(p) on latent z ----
    # If provided, z_t depends on its own past values:
    #     z_t = sum_j ar_coefs[j-1] * z_{t-j} + (trend + season + Xb + u + noise)_t
    # Pass a list/tuple like (0.5,) for AR(1) or (0.5, 0.2, 0.1) for AR(3).
    # Set to None or () to disable (default).
    # NOTE: must be stationary -- the function raises ValueError if not.
    # A simple sufficient condition is sum(|ar_coefs|) < 1.
    ar_coefs=None,
    # ---- unobserved AR(1) factor ----
    # An autocorrelated latent factor u_t added to z but NOT exposed as a
    # covariate, so lagged y is the only signal NeuralProphet's AR-on-y can use
    # to capture it. With unobserved_sd=0 the factor is identically zero.
    # `unobserved_sd` is the STATIONARY standard deviation of u_t (the innovation
    # SD is scaled internally so that SD(u) == unobserved_sd at stationarity).
    unobserved_ar=0.0,           # AR(1) persistence in (-1, 1)
    unobserved_sd=0.0,           # stationary SD of u_t (in latent z units)
    # ---- thresholds in latent space ----
    # If None, thresholds are placed at quantiles producing roughly the given
    # category probabilities. Otherwise pass an explicit list of length
    # (num_categories - 1) that is strictly increasing.
    thresholds=None,
    target_proportions=(0.45, 0.30, 0.18, 0.07),  # imbalanced like AQI data
    # ---- reproducibility ----
    seed=42,
    return_truth=True,
):
    """
    Generate a synthetic ordinal time series for testing OrdinalNeuralProphet.

    Returns
    -------
    df : pd.DataFrame
        Columns: ['datetime', 'X1', 'X2', 'X3', 'X4', 'u_unobserved',
                  'y_latent', 'aqi_category'].
        ('aqi_category' is named to match the Beijing AQI pipeline. 'y_latent'
        and 'u_unobserved' are unobservable truth, for diagnostics only.)
    truth : dict (only if return_truth=True)
        The true parameters used to generate the data, including the thresholds.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    dates = pd.date_range(start=start_date, periods=n, freq=freq)

    # Validate the lag / unobserved-factor parameters early.
    if not isinstance(covariate_lag, (int, np.integer)) or covariate_lag < 0:
        raise ValueError("covariate_lag must be a non-negative integer.")
    if covariate_lag >= n:
        raise ValueError("covariate_lag must be smaller than n.")
    if unobserved_sd > 0 and not (-1.0 < unobserved_ar < 1.0):
        raise ValueError(
            "unobserved_ar must lie in (-1, 1) for a stationary factor "
            f"(got {unobserved_ar})."
        )

    # ---------- 1. Covariates ----------
    # X1: smooth annual + daily wave + AR(1) noise (temperature-like)
    annual_period = 24 * 365  # purely cosmetic on this length, but realistic
    x1 = (
        10.0 * np.sin(2 * np.pi * t / annual_period - np.pi / 2)   # annual
        + 4.0 * np.sin(2 * np.pi * t / 24 - np.pi / 2)             # cooler at night
    )
    # AR(1) shock to make it autocorrelated
    ar_noise = np.zeros(n)
    eps = rng.normal(0, 1.5, size=n)
    phi = 0.85
    for i in range(1, n):
        ar_noise[i] = phi * ar_noise[i - 1] + eps[i]
    x1 = x1 + ar_noise

    # X2: correlated with X1 (DEWP-like)
    indep = rng.normal(0, 1, size=n)
    x2 = x1_x2_corr * (x1 - x1.mean()) / x1.std() + np.sqrt(1 - x1_x2_corr ** 2) * indep
    x2 = x2 * 6.0 - 5.0  # rescale to a TEMP/DEWP-ish range

    # X3: opposite seasonal pattern (pressure-like, anticorrelated with X1)
    x3 = (
        1015.0
        - 0.3 * x1
        + rng.normal(0, 4.0, size=n)
    )

    # X4: spiky / intermittent (wind-speed-like)
    spikes = rng.binomial(1, x4_spike_prob, size=n).astype(float)
    spike_mag = np.abs(rng.normal(0, x4_spike_scale, size=n)) * spikes
    base = np.abs(rng.normal(0, 0.8, size=n))
    x4 = base + spike_mag

    X = np.column_stack([x1, x2, x3, x4])
    # Standardize covariates BEFORE applying betas, so betas are in
    # "effect per standard deviation of X" units (interpretable + comparable).
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    # ---------- 2. Latent components ----------
    trend = build_trend(n, trend_slope, trend_segments)
    daily = daily_amplitude * np.sin(2 * np.pi * t / 24 + daily_phase)
    weekly = weekly_amplitude * np.sin(2 * np.pi * t / 168 + weekly_phase)

    # Covariate effects, optionally applied at a lag: z_t depends on
    # X_{t - covariate_lag}. covariate_lag=0 reproduces the original
    # contemporaneous behavior.
    if covariate_lag > 0:
        X_eff = X_std.copy()
        X_eff[covariate_lag:] = X_std[:-covariate_lag]
        # The first `covariate_lag` rows have no valid lagged value available;
        # they keep their contemporaneous value as a short warm-up region.
    else:
        X_eff = X_std
    cov_effect = X_eff @ np.array(betas)

    # Optional unobserved AR(1) latent factor u_t. Identically zero unless
    # unobserved_sd > 0. The innovation SD is scaled so that the STATIONARY SD
    # of u equals unobserved_sd: Var(u) = sigma_eta^2 / (1 - rho^2).
    u = np.zeros(n)
    if unobserved_sd > 0:
        sigma_eta = unobserved_sd * np.sqrt(1 - unobserved_ar ** 2)
        u_eps = rng.normal(0, sigma_eta, size=n)
        u[0] = rng.normal(0, unobserved_sd)  # initialize from stationary dist
        for i in range(1, n):
            u[i] = unobserved_ar * u[i - 1] + u_eps[i]

    noise = rng.normal(0, noise_std, size=n)

    # Everything that does NOT depend on past z (the "exogenous" part of z)
    exog = trend + daily + weekly + cov_effect + u + noise

    # Optional AR(p) on z: z_t = sum_j ar_coefs[j-1] * z_{t-j} + exog_t
    if ar_coefs is not None and len(ar_coefs) > 0:
        ar_coefs_arr = np.asarray(ar_coefs, dtype=float)
        p = len(ar_coefs_arr)
        # Stationarity check: roots of x^p - rho_1 x^(p-1) - ... - rho_p = 0
        # must all lie strictly inside the unit circle.
        char_poly = np.concatenate([[1.0], -ar_coefs_arr])
        roots = np.roots(char_poly)
        max_abs_root = float(np.abs(roots).max()) if len(roots) > 0 else 0.0
        if max_abs_root >= 1.0:
            raise ValueError(
                f"AR coefficients are non-stationary "
                f"(max |root| = {max_abs_root:.3f} >= 1). "
                f"Try smaller magnitudes; a simple sufficient condition is "
                f"sum(|ar_coefs|) < 1."
            )
        z = np.zeros(n)
        z[:p] = exog[:p]  # warm-up: first p steps use exog only
        for i in range(p, n):
            # Lags in order [z_{t-1}, z_{t-2}, ..., z_{t-p}] dotted with ar_coefs
            z[i] = ar_coefs_arr @ z[i - p:i][::-1] + exog[i]
    else:
        z = exog

    # ---------- 3. Thresholds & categorization ----------
    if thresholds is None:
        # Place thresholds at quantiles matching target_proportions
        if len(target_proportions) != num_categories:
            raise ValueError(
                f"target_proportions must have length {num_categories}, "
                f"got {len(target_proportions)}"
            )
        if not np.isclose(sum(target_proportions), 1.0):
            raise ValueError("target_proportions must sum to 1.")
        cum = np.cumsum(target_proportions)[:-1]
        thresholds = list(np.quantile(z, cum))
    else:
        thresholds = list(thresholds)
        if len(thresholds) != num_categories - 1:
            raise ValueError(
                f"thresholds must have length {num_categories - 1}, "
                f"got {len(thresholds)}"
            )
        if any(thresholds[i] >= thresholds[i + 1] for i in range(len(thresholds) - 1)):
            raise ValueError("thresholds must be strictly increasing.")

    y = np.ones(n, dtype=int)
    for i, tau in enumerate(thresholds):
        y[z > tau] = i + 2
    y = np.clip(y, 1, num_categories)

    # ---------- 4. Assemble DataFrame ----------
    df = pd.DataFrame({
        'datetime': dates,
        'X1': x1,
        'X2': x2,
        'X3': x3,
        'X4': x4,
        'u_unobserved': u,            # hidden AR(1) factor (diagnostics; NOT a model input)
        'y_latent': z,                # underlying continuous z (diagnostics)
        'aqi_category': y,            # named to match Beijing AQI pipeline
    })

    if not return_truth:
        return df

    truth = {
        'n': n,
        'freq': freq,
        'num_categories': num_categories,
        'trend_slope': trend_slope,
        'trend_segments': [list(s) for s in trend_segments] if trend_segments is not None else None,
        'daily_amplitude': daily_amplitude,
        'daily_phase': daily_phase,
        'weekly_amplitude': weekly_amplitude,
        'weekly_phase': weekly_phase,
        'betas': list(betas),
        'covariate_lag': covariate_lag,
        'noise_std': noise_std,
        'ar_coefs': list(ar_coefs) if ar_coefs is not None else [],
        'unobserved_ar': unobserved_ar,
        'unobserved_sd': unobserved_sd,
        'thresholds': thresholds,
        'category_proportions': [float((y == k).mean()) for k in range(1, num_categories + 1)],
        'covariate_means_pre_std': X.mean(axis=0).tolist(),
        'covariate_stds_pre_std': X.std(axis=0).tolist(),
        'seed': seed,
    }
    return df, truth


if __name__ == '__main__':
    df, truth = simulate_ordinal_timeseries()
    print("Simulated data preview:")
    print(df.head())
    print("\nCategory distribution:")
    print(df['aqi_category'].value_counts().sort_index())
    print("\nTrue thresholds:", truth['thresholds'])
