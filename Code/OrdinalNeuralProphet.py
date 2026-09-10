"""
OrdinalNeuralProphet: two-stage ordinal time series forecasting.

Stage 1 fits NeuralProphet to ordinal labels treated as continuous, giving an
interpretable additive decomposition and a continuous forecast. Stage 2 maps that
forecast to categories through K-1 ordered thresholds optimized on a validation
segment, replacing naive rounding.

Blind multi-step forecasting
    NeuralProphet emits a block of `n_forecasts` steps from one input window but does
    not extend beyond it. The rolling-origin loop here chains those blocks to cover
    `test_periods`, feeding the model's own continuous predictions into the
    autoregressive input past the forecast origin. Covariates are exogenous and are
    supplied from observed values throughout; the target's own future is never seen.

Threshold parameterization
    Thresholds are searched with Nelder-Mead over unconstrained gap parameters, so the
    ordering t1 < t2 < ... is satisfied by construction. Three options:

      direct   sorts inside the objective (discontinuous landscape)
      softplus gap = log(1 + exp(b)), gaps in (0, inf)          [used in the paper]
      sigmoid  gap = max_gap * sigmoid(b), gaps in (0, max_gap)

    Multiple random starts are run and the top-m threshold vectors by validation kappa
    are averaged elementwise.

Data split
    |--- training ---|--- validation ---|--- test ---|

    The model is fit on the training and validation blocks combined, and the thresholds
    are then tuned on this fitted model's predictions over the validation block. The
    validation block is therefore used twice: once as data the forecaster is fit on,
    and once to select thresholds. This is deliberate, so that the forecaster can
    condition on the data immediately preceding the test window. It means the tuning
    predictions are in-sample and somewhat more accurate than the test forecasts the
    thresholds are applied to. No test observation is used to fit the model or to
    choose a threshold, so the reported test metrics remain out-of-sample.

Usage
    from OrdinalNeuralProphet import OrdinalNeuralProphet

    model = OrdinalNeuralProphet(
        df,
        'aqi_category',
        n_lags=24,
        n_forecasts=12,
        test_periods=72,
        val_periods=500,
        num_categories=4,
        freq='h',
        covariates=['TEMP', 'DEWP', 'PRES', 'Iws'],          # lagged regressors
        future_covariates=['TEMP', 'DEWP', 'PRES', 'Iws'],   # future regressors
        covariate_lags=24,                                   # <= n_lags
    )

    model.assessment()
    results = model.optimize_thresholds(gap_method='softplus', n_starts=200)
    model.plot_comparison()
"""

import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from sklearn.metrics import cohen_kappa_score, classification_report
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class OrdinalNeuralProphet:
    def __init__(self, dataframe, target, n_lags=24, n_forecasts=1,
                 test_periods=168, val_periods=None, forecast_only=False,
                 num_categories=3, round_intermediate=False, freq='D',
                 covariates=None, future_covariates=None, covariate_lags=None,
                 epochs=30, seed=None, learning_rate=None, trend_reg=None,
                 ar_reg=None, seasonality_reg=None, n_changepoints=10,
                 val_mode='holdout'):
        """
        Ordinal time-series forecasting using NeuralProphet with numeric proxy approach.

        Args:
            dataframe: DataFrame with datetime index and target column (+ optional covariates)
            target: Name of the ordinal target column
            n_lags: Number of lagged values to use for AR
            n_forecasts: Number of steps to forecast at once
            test_periods: Number of periods to hold out for final testing
            val_periods: Number of periods for validation (threshold tuning).
                         If None, defaults to 20% of test_periods.
                         Validation comes BEFORE test in time order.
            forecast_only: If True, forecast beyond the data
            num_categories: Number of ordinal categories (expects 1, 2, ..., num_categories)
            round_intermediate: If True, round predictions before feeding back
            freq: Data frequency ('D' for daily, 'h' for hourly)
            covariates: List of column names to use as lagged regressors. A
                        lagged regressor contributes past covariate values
                        x_{t-1}, ..., x_{t-L} (e.g., ['TEMP', 'PRES']).
            future_covariates: List of column names to use as future
                        regressors. A future regressor contributes the
                        contemporaneous value x_t and therefore requires the
                        covariate's future values to be known at prediction
                        time. A name may appear in both 'covariates' and
                        'future_covariates'; together they form a distributed
                        lag x_{t-L}, ..., x_t for that covariate.
            covariate_lags: Lag count for the lagged regressors. None (default)
                        uses n_lags for every covariate; an int sets a common
                        value for all covariates; a dict sets per-covariate
                        values (covariates absent from the dict fall back to
                        n_lags). Each value must satisfy 1 <= value <= n_lags.
            epochs: Number of training epochs for NeuralProphet (default 30, increase for stability)
            seed: Random seed for reproducibility. When set, seeds numpy,
                  Python's random, and PyTorch (CPU + CUDA). None = unseeded.
            learning_rate: NeuralProphet learning rate. None (default) lets
                  NeuralProphet auto-discover it via its LR range test;
                  setting an explicit value (e.g. 0.03) improves run-to-run
                  reproducibility.
            trend_reg: Regularization strength on trend changepoints. Larger
                  values penalize trend flexibility. None = NeuralProphet default.
            ar_reg: Regularization strength on the AR lag weights. Larger
                  values sparsify the AR component (push more lag weights to
                  zero). None = NeuralProphet default (no AR regularization).
            seasonality_reg: Regularization strength on the Fourier seasonality
                  terms. Larger values damp seasonality. None = NeuralProphet default.
            n_changepoints: Number of potential trend changepoints (default 10,
                  matching NeuralProphet's default). Higher allows more trend flexibility.
            val_mode: How the validation block is used. 'holdout' (default):
                  the model is fit on the training block only; the validation
                  block is used solely to optimize thresholds and is never seen
                  by the model (the original two-stage scheme). 'refit': the
                  model is fit on train+validation combined; that single model
                  supplies both the validation predictions used to tune
                  thresholds (in-sample to the model) and the test forecast, so
                  the forecaster uses all pre-test data and the thresholds are
                  calibrated to the same model that produces the test
                  predictions. Both modes hold out the test set; neither leaks
                  test data. 'refit' costs no extra fits over 'holdout'.
        """
        # Store covariate configuration
        self.covariates = covariates if covariates is not None else []
        self.future_covariates = future_covariates if future_covariates is not None else []

        # Internal column names for future regressors. A '__fut' suffix keeps
        # the future-regressor copy distinct from the lagged-regressor column,
        # so a covariate can serve as both roles without a name collision.
        self._future_col_map = {src: f"{src}__fut" for src in self.future_covariates}

        # All covariate columns that must be pulled from the input dataframe
        # (union of lagged and future covariates, order-preserving, deduped).
        self._all_source_covariates = list(
            dict.fromkeys(self.covariates + self.future_covariates)
        )

        # Resolve per-covariate lag counts for the lagged regressors.
        #   None -> every lagged covariate uses the AR lag count (n_lags)
        #   int  -> every lagged covariate uses this common value
        #   dict -> per-covariate; covariates absent from the dict use n_lags
        if covariate_lags is None:
            self.covariate_lags = {c: n_lags for c in self.covariates}
        elif isinstance(covariate_lags, int):
            self.covariate_lags = {c: covariate_lags for c in self.covariates}
        elif isinstance(covariate_lags, dict):
            self.covariate_lags = {c: covariate_lags.get(c, n_lags) for c in self.covariates}
            unknown = set(covariate_lags) - set(self.covariates)
            if unknown:
                print(f"Warning: covariate_lags keys not in 'covariates' (ignored): {sorted(unknown)}")
        else:
            raise TypeError("covariate_lags must be None, int, or dict")

        # Lagged-regressor lags must not exceed the AR lag count. This keeps
        # the model's maximum input window equal to n_lags, so the iterative
        # forecasting loop (which feeds NeuralProphet a history tail of length
        # n_lags) remains valid. Covariate lags SMALLER than n_lags are fully
        # supported; allowing larger would require widening that window.
        for c, L in self.covariate_lags.items():
            if not isinstance(L, int) or L < 1:
                raise ValueError(f"covariate_lags['{c}'] must be a positive integer, got {L}")
            if L > n_lags:
                raise ValueError(
                    f"covariate_lags['{c}']={L} exceeds n_lags={n_lags}. "
                    f"Covariate lags must be <= n_lags in this version."
                )

        # Validate that all covariate columns exist in the input dataframe
        missing = [c for c in self._all_source_covariates if c not in dataframe.columns]
        if missing:
            raise ValueError(f"Covariate column(s) not found in dataframe: {missing}")

        # Select target and covariate columns
        cols_to_keep = [target] + self._all_source_covariates
        self.dataframe = dataframe[cols_to_keep].copy()

        self.target = target
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.test_periods = test_periods
        self.val_periods = val_periods if val_periods is not None else int(test_periods * 0.2)
        self.forecast_only = forecast_only
        self.num_categories = num_categories
        self.round_intermediate = round_intermediate
        self.freq = freq
        self.epochs = epochs
        self.seed = seed
        self.learning_rate = learning_rate
        self.trend_reg = trend_reg
        self.ar_reg = ar_reg
        self.seasonality_reg = seasonality_reg
        self.n_changepoints = n_changepoints
        # Validation-block handling: 'holdout' (val excluded from fit, original
        # scheme) or 'refit' (val included in fit, supplies its own tuning
        # predictions).
        if val_mode not in ('holdout', 'refit'):
            raise ValueError("val_mode must be 'holdout' or 'refit'")
        self.val_mode = val_mode

        # Seed RNGs for reproducibility (numpy, Python random, PyTorch).
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except ImportError:
                pass

        # Threshold optimization results (populated by optimize_thresholds)
        self.optimal_thresholds = None
        self.optimization_results = None

        # Validate that target contains integer-like values
        unique_vals = self.dataframe[target].dropna().unique()
        if not all(float(v).is_integer() for v in unique_vals):
            print(f"Warning: Target column contains non-integer values: {unique_vals}")

        # Prepare data in NeuralProphet format
        self.prophet_data = self._data_transform()

        # Store split indices for later use
        total_len = len(self.prophet_data)
        self.train_end_idx = total_len - self.test_periods - self.val_periods
        self.val_end_idx = total_len - self.test_periods
        self.test_end_idx = total_len

        print(f"Data split:")
        print(f"  Training:   {self.train_end_idx} samples (indices 0 to {self.train_end_idx - 1})")
        print(f"  Validation: {self.val_periods} samples (indices {self.train_end_idx} to {self.val_end_idx - 1})")
        print(f"  Test:       {self.test_periods} samples (indices {self.val_end_idx} to {self.test_end_idx - 1})")
        if self.val_mode == 'refit':
            print(f"  Val mode:   refit (model fit on train+validation; "
                  f"thresholds tuned on its own validation predictions)")

        # Fit model and generate forecasts
        self.forecast_df, self.fitted_model = self._fit_neuralprophet()

    def _data_transform(self):
        """Transform data to NeuralProphet format (ds, y, + covariates).

        For every covariate used as a future regressor, a duplicated column
        with a '__fut' suffix is created. The original column name feeds the
        lagged regressor; the suffixed copy feeds the future regressor. This
        keeps the two regressor roles on distinct column names so a single
        covariate can serve as both without any naming collision.
        """
        df = self.dataframe.copy()
        df = df.reset_index()

        # Rename: index -> ds, target -> y, keep covariate names
        new_columns = ['ds', 'y'] + self._all_source_covariates
        df.columns = new_columns
        df['ds'] = pd.to_datetime(df['ds'])

        # Duplicate columns for future-regressor use (suffixed to avoid collision)
        for src, fut_col in self._future_col_map.items():
            df[fut_col] = df[src]

        return df

    def _fit_neuralprophet(self):
        """Fit NeuralProphet and run iterative forecasting loop."""
        total_len = len(self.prophet_data)

        if self.forecast_only:
            train_df = self.prophet_data.copy()
            last_date = train_df['ds'].max()

            future_dates = pd.date_range(
                start=last_date + pd.to_timedelta(1, unit=self.freq),
                periods=self.test_periods,
                freq=self.freq
            )
            test_df = pd.DataFrame({'ds': future_dates})
            val_df = None

            if self.covariates or self.future_covariates:
                print("Warning: forecast_only mode with covariates requires future covariate values.")
                print("Lagged and future regressors are not supported in forecast_only mode.")
        else:
            # Split: Training | Validation | Test
            train_df = self.prophet_data.iloc[:self.train_end_idx].copy()
            val_df = self.prophet_data.iloc[self.train_end_idx:self.val_end_idx].copy()
            test_df = self.prophet_data.iloc[self.val_end_idx:].copy()

        # Initialize model. Optional regularization / learning-rate
        # parameters are included only when explicitly set, so omitting
        # them preserves NeuralProphet's own defaults.
        np_kwargs = dict(
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            weekly_seasonality=True,
            daily_seasonality=(self.freq == 'h'),
            epochs=self.epochs,
            n_changepoints=self.n_changepoints,
        )
        if self.learning_rate is not None:
            np_kwargs['learning_rate'] = self.learning_rate
        if self.trend_reg is not None:
            np_kwargs['trend_reg'] = self.trend_reg
        if self.ar_reg is not None:
            np_kwargs['ar_reg'] = self.ar_reg
        if self.seasonality_reg is not None:
            np_kwargs['seasonality_reg'] = self.seasonality_reg

        model = NeuralProphet(**np_kwargs)

        # Add lagged regressors (covariates), each with its own lag count.
        # A lagged regressor contributes the delayed terms x_{t-1},...,x_{t-L}.
        for covar in self.covariates:
            model.add_lagged_regressor(names=covar, n_lags=self.covariate_lags[covar])

        # Add future regressors. A future regressor contributes the
        # contemporaneous term x_t; combined with the lagged regressors above
        # this yields a full distributed lag x_{t-L}, ..., x_t. The '__fut'
        # column holds the (known) future covariate values.
        for src in self.future_covariates:
            model.add_future_regressor(name=self._future_col_map[src])

        # Validation predictions used for threshold tuning. In 'refit' mode
        # these come from the train+val model (in-sample); in 'holdout' mode
        # they come from the train-only model (out-of-sample). Stored so
        # optimize_thresholds can pick them up consistently.
        self._val_cont_refit = None

        if self.forecast_only:
            # forecast_only path: single model on all data (unchanged)
            model.fit(train_df, freq=self.freq)
            predictions = self._iterative_forecast(model, train_df, test_df)

        elif self.val_mode == 'refit':
            # 'refit': ONE fit on train+validation. The same model supplies the
            # validation predictions used to tune thresholds (in-sample to this
            # model) and the test forecast. Tuning and application therefore
            # share the same forecaster, and the model uses all pre-test data.
            train_val_df = pd.concat([train_df, val_df], ignore_index=True)
            model.fit(train_val_df, freq=self.freq)

            # validation predictions: forecast the val block from the train tail
            preds_val = self._iterative_forecast(model, train_df, val_df)
            # test predictions: forecast the test block from the train+val tail
            preds_test = self._iterative_forecast(model, train_val_df, test_df)

            self._val_cont_refit = np.asarray(preds_val['continuous'])
            predictions = {
                'continuous': np.concatenate([preds_val['continuous'], preds_test['continuous']]),
                'rounded':    np.concatenate([preds_val['rounded'],    preds_test['rounded']]),
            }

        else:
            # 'holdout' (default): fit on training only. The model supplies the
            # validation predictions (out-of-sample to it) for threshold tuning
            # and the test forecast. The validation block is never seen by the
            # model.
            model.fit(train_df, freq=self.freq)
            val_test_df = pd.concat([val_df, test_df], ignore_index=True)
            predictions = self._iterative_forecast(model, train_df, val_test_df)

        # Package results
        if self.forecast_only:
            full_df = self.prophet_data.copy()
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'y': np.nan,
                'yhat_continuous': predictions['continuous'],
                'yhat': predictions['rounded']
            })
            full_df['yhat_continuous'] = np.nan
            full_df['yhat'] = np.nan
            full_df = pd.concat([full_df, forecast_df], ignore_index=True)
        else:
            full_df = self.prophet_data.copy()
            full_df['yhat_continuous'] = np.nan
            full_df['yhat'] = np.nan

            # Fill in predictions for validation + test periods
            start_idx = self.train_end_idx
            full_df.iloc[start_idx:, full_df.columns.get_loc('yhat_continuous')] = predictions['continuous']
            full_df.iloc[start_idx:, full_df.columns.get_loc('yhat')] = predictions['rounded']

        return full_df, model

    def _iterative_forecast(self, model, train_data, test_data):
        """Iterative forecasting loop with optional intermediate rounding."""
        predictions_continuous = []

        history = train_data.copy()

        for i in range(0, len(test_data), self.n_forecasts):
            end_idx = min(i + self.n_forecasts, len(test_data))
            chunk_size = end_idx - i

            # Get the dates we need to forecast
            future_chunk = test_data.iloc[i:end_idx].copy()
            if 'y' in future_chunk.columns:
                future_chunk['y'] = np.nan

            # NeuralProphet requires at least n_forecasts rows in the future portion
            # If we have fewer (at the end of test_data), pad with dummy dates
            if len(future_chunk) < model.n_forecasts:
                last_date = future_chunk['ds'].iloc[-1]
                extra_dates = pd.date_range(
                    start=last_date + pd.Timedelta(1, unit=self.freq),
                    periods=model.n_forecasts - len(future_chunk),
                    freq=self.freq
                )
                padding_dict = {'ds': extra_dates, 'y': np.nan}
                # Pad every covariate column (lagged and future regressor
                # columns alike) with its last known value, so NeuralProphet
                # receives a complete future frame.
                for col in future_chunk.columns:
                    if col not in ('ds', 'y'):
                        padding_dict[col] = future_chunk[col].iloc[-1]
                padding = pd.DataFrame(padding_dict)
                future_chunk = pd.concat([future_chunk, padding], ignore_index=True)

            # Combine history tail with future chunk for prediction
            future_df = pd.concat([history.tail(model.n_lags), future_chunk]).reset_index(drop=True)

            # Predict
            forecast = model.predict(future_df)
            res = forecast.iloc[model.n_lags:].reset_index(drop=True)

            # Extract predictions for each step in chunk (only the ones we actually need)
            chunk_continuous = []

            for j in range(chunk_size):
                step = j + 1
                col = f'yhat{step}'
                pred = res[col].iloc[j] if col in res.columns else res['yhat1'].iloc[j]
                chunk_continuous.append(pred)

            predictions_continuous.extend(chunk_continuous)

            # Feed predictions back into history for next iteration
            next_hist = test_data.iloc[i:end_idx].copy()
            if self.round_intermediate:
                feedback = [int(np.clip(np.round(p), 1, self.num_categories)) for p in chunk_continuous]
            else:
                feedback = chunk_continuous
            next_hist['y'] = feedback
            history = pd.concat([history, next_hist], ignore_index=True).tail(1000)

        # Final rounding
        predictions_continuous = np.array(predictions_continuous)
        predictions_rounded = np.clip(np.round(predictions_continuous), 1, self.num_categories).astype(int)

        return {
            'continuous': predictions_continuous,
            'rounded': predictions_rounded
        }

    # =========================================================
    # THRESHOLD OPTIMIZATION METHODS (v2 - Direct, Softplus, or Scaled Sigmoid)
    # =========================================================

    # --- Softplus functions ---
    @staticmethod
    def _softplus(x):
        """Softplus function: log(1 + exp(x)) ∈ (0, ∞)."""
        x = np.asarray(x)
        return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -20, 20))))

    @staticmethod
    def _inverse_softplus(y):
        """Inverse softplus: get x such that softplus(x) = y."""
        y = np.asarray(y)
        y = np.clip(y, 1e-6, None)
        return np.where(y > 20, y, np.log(np.exp(y) - 1))

    # --- Scaled sigmoid functions ---
    @staticmethod
    def _sigmoid(x):
        """Sigmoid function: 1 / (1 + exp(-x)) ∈ (0, 1)."""
        x_clipped = np.clip(x, -20, 20)
        return 1 / (1 + np.exp(-x_clipped))

    @staticmethod
    def _inverse_sigmoid(y):
        """Inverse sigmoid (logit): get x such that sigmoid(x) = y."""
        y_clipped = np.clip(y, 1e-6, 1 - 1e-6)
        return np.log(y_clipped / (1 - y_clipped))

    def _thresholds_from_params(self, params, gap_method='softplus', max_gap=2.0):
        """
        Convert unconstrained parameters to ordered thresholds.

        Args:
            params: List of parameters
            gap_method: 'direct' (uses sorting), 'softplus' (gaps in (0,∞)),
                        'sigmoid' (gaps in (0,max_gap))
            max_gap: Maximum gap size when using sigmoid method

        Returns:
            List of ordered thresholds
        """
        if gap_method == 'direct':
            # Direct method: just sort the params to enforce ordering
            return sorted(params)

        # Reparameterized methods: t1 = a, t2 = t1 + gap(b), ...
        thresholds = [params[0]]  # t1 = a (free)
        for i in range(1, len(params)):
            if gap_method == 'softplus':
                gap = self._softplus(params[i])  # gap ∈ (0, ∞)
            else:  # sigmoid
                gap = max_gap * self._sigmoid(params[i])  # gap ∈ (0, max_gap)
            thresholds.append(thresholds[-1] + gap)
        return thresholds

    def _params_from_thresholds(self, thresholds, gap_method='softplus', max_gap=2.0):
        """
        Convert thresholds to unconstrained parameters for initialization.
        Inverse of _thresholds_from_params.

        Args:
            thresholds: List of ordered thresholds
            gap_method: 'direct', 'softplus', or 'sigmoid'
            max_gap: Maximum gap size when using sigmoid method

        Returns:
            List of parameters
        """
        if gap_method == 'direct':
            # Direct method: params are just the thresholds directly
            return list(thresholds)

        # Reparameterized methods
        params = [thresholds[0]]  # a = t1
        for i in range(1, len(thresholds)):
            gap = thresholds[i] - thresholds[i-1]
            if gap_method == 'softplus':
                params.append(self._inverse_softplus(gap))
            else:  # sigmoid
                gap_ratio = np.clip(gap / max_gap, 1e-6, 1 - 1e-6)
                params.append(self._inverse_sigmoid(gap_ratio))
        return params

    @staticmethod
    def _apply_thresholds(x, thresholds, num_categories):
        """Map continuous values to ordinal categories based on thresholds."""
        result = np.ones_like(x, dtype=int)
        for i, thresh in enumerate(thresholds):
            result[x > thresh] = i + 2
        return np.clip(result, 1, num_categories)

    def _optimize_thresholds_internal(self, y_true, y_pred_continuous,
                                        n_starts=5, top_k=3,
                                        gap_method='softplus', max_gap=2.0,
                                        kappa_weights='quadratic'):
        """
        Threshold optimization using crude, softplus, or scaled sigmoid parameterization.

        Args:
            y_true: True ordinal labels
            y_pred_continuous: Continuous predictions from NeuralProphet
            n_starts: Number of random starting points
            top_k: Number of top results to average (Ordinal Forest strategy)
            gap_method: 'direct' (sorting), 'softplus' (unbounded gaps), 'sigmoid' (bounded gaps)
            max_gap: Maximum gap between thresholds (only used if gap_method='sigmoid')
            kappa_weights: 'quadratic', 'linear', or None

        Returns:
            (optimal_thresholds, best_kappa)
        """
        num_thresholds = self.num_categories - 1
        pred_min, pred_max = y_pred_continuous.min(), y_pred_continuous.max()

        def objective(params):
            thresholds = self._thresholds_from_params(params, gap_method, max_gap)
            preds = self._apply_thresholds(y_pred_continuous, thresholds, self.num_categories)
            try:
                kappa = cohen_kappa_score(y_true, preds, weights=kappa_weights)
                return -kappa if not np.isnan(kappa) else 0
            except:
                return 0

        results = []

        for i in range(n_starts):
            if i == 0:
                # First start: evenly spaced thresholds (1.5, 2.5, 3.5 for 4 categories)
                init_thresholds = [j + 0.5 for j in range(1, self.num_categories)]
            else:
                # Random starts within prediction range
                init_thresholds = sorted(np.random.uniform(pred_min, pred_max, num_thresholds))

            init_params = self._params_from_thresholds(init_thresholds, gap_method, max_gap)

            try:
                result = minimize(objective, init_params, method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-4})
                final_thresholds = self._thresholds_from_params(result.x, gap_method, max_gap)
                final_score = -result.fun
                results.append((final_score, final_thresholds))
            except:
                continue

        if not results:
            default_thresholds = [j + 0.5 for j in range(1, self.num_categories)]
            return default_thresholds, 0.0

        # Sort by score and average top_k
        results.sort(key=lambda x: x[0], reverse=True)
        top_results = results[:min(top_k, len(results))]

        avg_thresholds = []
        for j in range(num_thresholds):
            avg_thresh = np.mean([r[1][j] for r in top_results])
            avg_thresholds.append(avg_thresh)

        final_preds = self._apply_thresholds(y_pred_continuous, avg_thresholds, self.num_categories)
        final_kappa = cohen_kappa_score(y_true, final_preds, weights=kappa_weights)

        return avg_thresholds, final_kappa

    def optimize_thresholds(self, n_starts=5, kappa_weights='quadratic',
                             gap_method='softplus', max_gap=2.0):
        """
        Run threshold optimization using validation set, evaluate on test set.

        Three parameterization options:

        1. 'direct': Original method using sorted() inside objective
           - Thresholds optimized directly, sorted to enforce ordering
           - Simple but creates discontinuities in optimization landscape

        2. 'softplus': Reparameterized with unbounded gaps
           - t1 = a, t2 = t1 + softplus(b), t3 = t2 + softplus(c), ...
           - Gaps in (0, ∞), smooth optimization landscape

        3. 'sigmoid': Reparameterized with bounded gaps
           - t1 = a, t2 = t1 + max_gap*sigmoid(b), ...
           - Gaps in (0, max_gap), provides regularization

        Args:
            n_starts: Number of random starting points for Nelder-Mead (recommend 200)
            kappa_weights: 'quadratic', 'linear', or None
            gap_method: 'direct', 'softplus', or 'sigmoid'
            max_gap: Maximum gap between adjacent thresholds (only for 'sigmoid')
                     Typical values: 1.0-1.5 (tight), 2.0 (moderate), 3.0+ (flexible)

        Returns:
            dict with optimization results
        """
        if self.forecast_only:
            print("Threshold optimization not available in forecast_only mode.")
            return None

        if gap_method not in ['direct', 'softplus', 'sigmoid']:
            raise ValueError("gap_method must be 'direct', 'softplus', or 'sigmoid'")

        # Get validation and test data
        val_df = self.forecast_df.iloc[self.train_end_idx:self.val_end_idx].copy()

        # For 2b, tune thresholds on the REFIT model's validation predictions
        # (in-sample to that model) rather than the train-only model's. Both
        # 2a and the default use the train-only model's val predictions already
        # stored in forecast_df.
        # In 'refit' mode, tune thresholds on the train+val model's own
        # validation predictions (already stored). 'holdout' uses the
        # train-only model's val predictions already in forecast_df.
        if self.val_mode == 'refit' and self._val_cont_refit is not None:
            n_val_rows = self.val_end_idx - self.train_end_idx
            if len(self._val_cont_refit) == n_val_rows:
                val_df['yhat_continuous'] = self._val_cont_refit
            else:
                print(f"  Warning: refit val prediction length "
                      f"({len(self._val_cont_refit)}) != val rows ({n_val_rows}); "
                      f"falling back to stored val predictions for tuning.")

        val_df = val_df.dropna(subset=['y', 'yhat_continuous'])

        test_df = self.forecast_df.iloc[self.val_end_idx:].copy()
        test_df = test_df.dropna(subset=['y', 'yhat_continuous'])

        y_val_true = val_df['y'].astype(int).values
        y_val_cont = val_df['yhat_continuous'].values

        y_test_true = test_df['y'].astype(int).values
        y_test_cont = test_df['yhat_continuous'].values

        weights_str = kappa_weights if kappa_weights else 'unweighted'
        print(f"Optimizing thresholds...")
        print(f"  Val mode: {self.val_mode}")
        print(f"  Gap method: {gap_method}" + (f" (max_gap={max_gap})" if gap_method == 'sigmoid' else ""))
        print(f"  Validation set: {len(y_val_true)} samples")
        print(f"  Test set: {len(y_test_true)} samples")
        print(f"  Kappa weights: {weights_str}")
        print(f"  Optimization starts: {n_starts}")

        # Optimize on validation set
        optimal_thresholds, val_kappa = self._optimize_thresholds_internal(
            y_val_true, y_val_cont,
            n_starts=n_starts, top_k=3,
            gap_method=gap_method, max_gap=max_gap,
            kappa_weights=kappa_weights
        )

        # Evaluate on test set
        y_test_baseline = np.clip(np.round(y_test_cont), 1, self.num_categories).astype(int)
        y_test_optimized = self._apply_thresholds(y_test_cont, optimal_thresholds, self.num_categories)

        baseline_kappa = cohen_kappa_score(y_test_true, y_test_baseline, weights=kappa_weights)
        optimized_kappa = cohen_kappa_score(y_test_true, y_test_optimized, weights=kappa_weights)

        baseline_accuracy = np.mean(y_test_true == y_test_baseline)
        optimized_accuracy = np.mean(y_test_true == y_test_optimized)

        within_one_baseline = np.mean(np.abs(y_test_true - y_test_baseline) <= 1)
        within_one_optimized = np.mean(np.abs(y_test_true - y_test_optimized) <= 1)

        mae_baseline = np.mean(np.abs(y_test_true - y_test_baseline))
        mae_optimized = np.mean(np.abs(y_test_true - y_test_optimized))

        # Store results
        self.optimal_thresholds = optimal_thresholds
        self.optimization_results = {
            'optimal_thresholds': optimal_thresholds,
            'gap_method': gap_method,
            'max_gap': max_gap if gap_method == 'sigmoid' else None,
            'val_kappa': val_kappa,
            'baseline_kappa': baseline_kappa,
            'optimized_kappa': optimized_kappa,
            'baseline_accuracy': baseline_accuracy,
            'optimized_accuracy': optimized_accuracy,
            'within_one_baseline': within_one_baseline,
            'within_one_optimized': within_one_optimized,
            'mae_baseline': mae_baseline,
            'mae_optimized': mae_optimized,
            'kappa_weights': kappa_weights,
            'y_test_true': y_test_true,
            'y_test_baseline': y_test_baseline,
            'y_test_optimized': y_test_optimized,
            'y_test_continuous': y_test_cont,
            'test_dates': test_df['ds'].values,
            'n_val': len(y_val_true),
            'n_test': len(y_test_true)
        }

        # Add optimized predictions to forecast_df
        all_cont = self.forecast_df.iloc[self.train_end_idx:]['yhat_continuous'].dropna().values
        all_optimized = self._apply_thresholds(all_cont, optimal_thresholds, self.num_categories)
        valid_idx = self.forecast_df.iloc[self.train_end_idx:].dropna(subset=['yhat_continuous']).index
        self.forecast_df.loc[valid_idx, 'yhat_optimized'] = all_optimized

        # Print results
        print(f"\n{'='*60}")
        print(f"THRESHOLD OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        print(f"\nGap Method: {gap_method}" + (f" (max_gap={max_gap})" if gap_method == 'sigmoid' else ""))
        print(f"Optimal Thresholds: {[f'{t:.4f}' for t in optimal_thresholds]}")
        print(f"Validation Kappa: {val_kappa:.4f}")

        gaps = [optimal_thresholds[i+1] - optimal_thresholds[i] for i in range(len(optimal_thresholds)-1)]
        if gaps:
            print(f"Threshold Gaps: {[f'{g:.4f}' for g in gaps]}")
            if gap_method == 'sigmoid':
                gap_utilization = [g / max_gap * 100 for g in gaps]
                print(f"Gap Utilization: {[f'{u:.1f}%' for u in gap_utilization]}")

        print(f"\nTEST SET PERFORMANCE ({len(y_test_true)} samples):")
        print(f"{'Metric':<25} {'Baseline':<12} {'Optimized':<12} {'Change':<12}")
        print("-" * 60)
        print(f"{'Kappa (' + weights_str + ')':<25} {baseline_kappa:<12.4f} {optimized_kappa:<12.4f} {optimized_kappa - baseline_kappa:+.4f}")
        print(f"{'Exact Accuracy':<25} {baseline_accuracy:<12.4f} {optimized_accuracy:<12.4f} {optimized_accuracy - baseline_accuracy:+.4f}")
        print(f"{'Within-1 Accuracy':<25} {within_one_baseline:<12.4f} {within_one_optimized:<12.4f} {within_one_optimized - within_one_baseline:+.4f}")
        print(f"{'MAE':<25} {mae_baseline:<12.4f} {mae_optimized:<12.4f} {mae_optimized - mae_baseline:+.4f}")

        return self.optimization_results

    # =========================================================
    # ASSESSMENT AND VISUALIZATION METHODS
    # =========================================================

    def assessment(self):
        """Basic assessment using simple rounding on TEST set only."""
        if self.forecast_only:
            print("Assessment not available in forecast_only mode.")
            return None

        results = self.forecast_df.iloc[self.val_end_idx:].copy()
        results = results.dropna(subset=['yhat', 'y'])

        y_true = results['y'].values
        y_pred = results['yhat'].values

        accuracy = np.mean(y_true == y_pred)
        within_one = np.mean(np.abs(y_true - y_pred) <= 1)
        mae = np.mean(np.abs(y_true - y_pred))

        print(f"TEST SET ASSESSMENT ({len(y_true)} samples):")
        print(f"Exact Accuracy: {accuracy:.3f}")
        print(f"Within-1 Accuracy: {within_one:.3f}")
        print(f"MAE (ordinal): {mae:.3f}")

        return {'accuracy': accuracy, 'within_one_accuracy': within_one, 'mae': mae, 'n_samples': len(y_true)}

    def plot_forecast(self, include_validation=False):
        """Plot baseline forecast on test set."""
        if self.forecast_only:
            results = self.forecast_df[self.forecast_df['yhat'].notna()].copy()
            title = "Ordinal Forecast"
        else:
            if include_validation:
                results = self.forecast_df.iloc[self.train_end_idx:].copy()
                title = "Ordinal Forecast vs Actuals (Validation + Test)"
            else:
                results = self.forecast_df.iloc[self.val_end_idx:].copy()
                title = "Ordinal Forecast vs Actuals (Test Set)"

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(results['ds'], results['yhat'], label='Predicted',
                color='#0072B2', linewidth=1.5, marker='o', markersize=2)

        if not self.forecast_only:
            ax.plot(results['ds'], results['y'], label='Actual',
                    color='red', alpha=0.7, linewidth=1.5, marker='x', markersize=2)

        ax.set_ylim(0.5, self.num_categories + 0.5)
        ax.set_yticks(range(1, self.num_categories + 1))
        ax.set_ylabel('Ordinal Category')
        ax.set_xlabel('Date')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_comparison(self):
        """Plot baseline vs optimized predictions on TEST set."""
        if self.optimization_results is None:
            print("Run optimize_thresholds() first.")
            return

        res = self.optimization_results
        dates = res['test_dates']
        y_true = res['y_test_true']
        y_baseline = res['y_test_baseline']
        y_optimized = res['y_test_optimized']

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        ax1 = axes[0]
        ax1.step(dates, y_true, label='Actual', color='black', alpha=0.5, where='mid')
        ax1.step(dates, y_baseline, label='Baseline (Rounding)', color='#0072B2',
                 linestyle='--', where='mid')
        ax1.set_ylabel('Ordinal Category')
        ax1.set_title(f"Baseline - Kappa: {res['baseline_kappa']:.4f}, Accuracy: {res['baseline_accuracy']:.4f}")
        ax1.legend(loc='upper right')
        ax1.set_yticks(range(1, self.num_categories + 1))
        ax1.set_ylim(0.5, self.num_categories + 0.5)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.step(dates, y_true, label='Actual', color='black', alpha=0.5, where='mid')
        ax2.step(dates, y_optimized, label='Optimized Thresholds', color='#D55E00',
                 linestyle='--', where='mid')
        ax2.set_ylabel('Ordinal Category')
        ax2.set_xlabel('Date')
        ax2.set_title(f"Optimized - Kappa: {res['optimized_kappa']:.4f}, Accuracy: {res['optimized_accuracy']:.4f}")
        ax2.legend(loc='upper right')
        ax2.set_yticks(range(1, self.num_categories + 1))
        ax2.set_ylim(0.5, self.num_categories + 0.5)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        self._plot_threshold_visualization()

    def _plot_threshold_visualization(self):
        """Visualize threshold mapping."""
        if self.optimization_results is None:
            return

        y_cont = self.optimization_results['y_test_continuous']
        thresholds = self.optimal_thresholds

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.hist(y_cont, bins=50, alpha=0.7, color='steelblue', edgecolor='white')

        # Standard rounding thresholds
        for i in range(1, self.num_categories):
            ax.axvline(x=i + 0.5, color='gray', linestyle=':', linewidth=2,
                      label='Standard (0.5)' if i == 1 else '')

        # Optimized thresholds
        colors = plt.cm.Set1(np.linspace(0, 1, len(thresholds)))
        for i, (thresh, color) in enumerate(zip(thresholds, colors)):
            ax.axvline(x=thresh, color=color, linestyle='-', linewidth=2,
                      label=f'θ{i+1} = {thresh:.3f}')

        ax.set_xlabel('Continuous Prediction')
        ax.set_ylabel('Frequency')
        ax.set_title('Threshold Comparison: Standard vs Optimized')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def get_classification_report(self):
        """Print classification reports for baseline and optimized."""
        if self.optimization_results is None:
            print("Run optimize_thresholds() first.")
            return

        res = self.optimization_results

        print("\n" + "="*60)
        print("CLASSIFICATION REPORT - BASELINE")
        print("="*60)
        print(classification_report(res['y_test_true'], res['y_test_baseline'], zero_division=0))

        print("\n" + "="*60)
        print("CLASSIFICATION REPORT - OPTIMIZED")
        print("="*60)
        print(classification_report(res['y_test_true'], res['y_test_optimized'], zero_division=0))
