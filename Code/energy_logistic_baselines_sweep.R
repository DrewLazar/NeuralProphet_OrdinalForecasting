# =====================================================================
# Two multinomial-logistic baselines swept across horizons, for ENERGY.
#
#   (A) AR baseline: category ~ ylag + covariates + calendar  (blind multi-step)
#   (B) Covariate-only: category ~ covariates + hour + dow    (NO AR term)
#
# Energy covariates: temp + occupancy binaries (is_not_weekend, in_session,
# is_not_holiday, is_not_summerbreak). Calendar features hour + dow capture the
# daily/weekly cycle. FIRST_ORIGIN=2000; horizons match the NP energy sweep.
# Origins are subsampled to MAX_ORIGINS per horizon (the series is long).
#
# ---------------------------------------------------------------------
# AR_SPEC controls the AR baseline's training frame:
#
#   "legacy" -- reproduces the original script. Trains on covariates at t-1
#               paired with y_t, while the blind loop predicts using the
#               covariates at t, so training and prediction disagree by one
#               step. Kept only to verify this harness reproduces the
#               previously reported numbers.
#
#   "fixed"  -- (default) trains on covariates at t, matching what the blind
#               loop supplies at prediction time.
#
# Only make_ar_frame changes. Formulas, the blind-recursion loop, the
# covariate-only baseline, the origins, and the pooling are untouched. The two
# formulas are already properly nested: ar_formula is cov_formula + ylag, so
# the difference between the two lines isolates the autoregressive term.
#
# Runs locally in seconds/fit (multinom); no cluster needed.
# =====================================================================

suppressMessages(library(nnet))

# setwd("C:/Users/dmlazar/Desktop/ordinalcluster/energy")

DATA        <- Sys.getenv("ENERGY_CSV", "energy_data.csv")
FIRST_ORIGIN<- as.integer(Sys.getenv("FIRST_ORIGIN", "2000"))
MAX_ORIGINS <- as.integer(Sys.getenv("MAX_ORIGINS", "150"))
COVARS      <- c("temp", "is_not_weekend", "in_session", "is_not_holiday", "is_not_summerbreak")
NCAT        <- 4L
OUTDIR      <- Sys.getenv("OUTDIR", "energy_out")
AR_SPEC     <- Sys.getenv("AR_SPEC", "fixed")          # "fixed" or "legacy"
stopifnot(AR_SPEC %in% c("fixed", "legacy"))
dir.create(OUTDIR, showWarnings = FALSE)

HORIZONS <- as.integer(strsplit(Sys.getenv("HORIZONS", "12,24,48,72,120,168,240,336"), ",")[[1]])

df <- read.csv(DATA, stringsAsFactors = FALSE)
df$datetime <- as.POSIXct(df$datetime, tz = "UTC",
                          tryFormats = c("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"))
df$hour <- as.integer(format(df$datetime, "%H"))
df$dow  <- as.integer(format(df$datetime, "%u"))
n <- nrow(df)
cat(sprintf("loaded %s: %d rows | AR_SPEC = %s\n", DATA, n, AR_SPEC))

quad_kappa <- function(true, pred, K = NCAT) {
  true <- factor(true, levels = 1:K); pred <- factor(pred, levels = 1:K)
  O <- table(true, pred); O <- O / sum(O)
  w <- outer(1:K, 1:K, function(i, j) (i - j)^2) / (K - 1)^2
  r <- rowSums(O); c <- colSums(O); E <- outer(r, c)
  1 - sum(w * O) / sum(w * E)
}

# ---------------------------------------------------------------------
# AR frame. Row t always uses load_category_{t-1} as the lag; AR_SPEC decides
# whether the covariates paired with y_t are those at t-1 (legacy) or at t
# (fixed, matching the prediction call at line ~"nd <- data.frame(...)").
# ---------------------------------------------------------------------
make_ar_frame <- function(d) {
  L <- nrow(d)
  cov_rows <- if (AR_SPEC == "legacy") 1:(L - 1) else 2:L
  data.frame(
    y     = factor(d$load_category[2:L], levels = 1:NCAT),
    ylag  = factor(d$load_category[1:(L - 1)], levels = 1:NCAT),
    d[cov_rows, COVARS, drop = FALSE],
    hour  = factor(d$hour[2:L], levels = 0:23),
    dow   = factor(d$dow[2:L],  levels = 1:7)
  )
}

make_cov_frame <- function(d) {
  data.frame(
    y     = factor(d$load_category, levels = 1:NCAT),
    d[, COVARS, drop = FALSE],
    hour  = factor(d$hour, levels = 0:23),
    dow   = factor(d$dow,  levels = 1:7)
  )
}

ar_formula  <- as.formula(paste("y ~ ylag +", paste(COVARS, collapse = " + "), "+ hour + dow"))
cov_formula <- as.formula(paste("y ~", paste(COVARS, collapse = " + "), "+ hour + dow"))
cat("AR formula:  ", deparse(ar_formula), "\n")
cat("COV formula: ", deparse(cov_formula), "\n\n")

rows <- list()
for (H in HORIZONS) {
  all_origins <- seq(FIRST_ORIGIN, n - H, by = H)
  if (length(all_origins) > MAX_ORIGINS) {
    idx <- round(seq(1, length(all_origins), length.out = MAX_ORIGINS))
    ORIGINS <- all_origins[idx]
  } else ORIGINS <- all_origins

  ar_true <- integer(0); ar_pred <- integer(0)
  cv_true <- integer(0); cv_pred <- integer(0)
  ar_ndistinct <- integer(0)

  for (origin in ORIGINS) {
    train <- df[1:origin, ]
    test  <- df[(origin + 1):(origin + H), ]

    # (A) AR baseline, blind multi-step
    tr_ar <- make_ar_frame(train)
    fit_ar <- tryCatch(multinom(ar_formula, data = tr_ar, trace = FALSE, maxit = 300, reltol = 1.0e-4),
                       error = function(e) NULL)
    if (!is.null(fit_ar)) {
      pred_blind <- integer(H)
      last_cat <- train$load_category[origin]
      for (h in 1:H) {
        nd <- data.frame(
          ylag = factor(last_cat, levels = 1:NCAT),
          test[h, COVARS, drop = FALSE],
          hour = factor(test$hour[h], levels = 0:23),
          dow  = factor(test$dow[h],  levels = 1:7))
        ph <- as.integer(as.character(predict(fit_ar, newdata = nd)))
        pred_blind[h] <- ph; last_cat <- ph
      }
      ar_true <- c(ar_true, as.integer(test$load_category))
      ar_pred <- c(ar_pred, pred_blind)
      ar_ndistinct <- c(ar_ndistinct, length(unique(pred_blind)))
    }

    # (B) covariate-only
    tr_cv <- make_cov_frame(train)
    fit_cv <- tryCatch(multinom(cov_formula, data = tr_cv, trace = FALSE, maxit = 300, reltol = 1.0e-4),
                       error = function(e) NULL)
    if (!is.null(fit_cv)) {
      nd <- data.frame(
        test[, COVARS, drop = FALSE],
        hour = factor(test$hour, levels = 0:23),
        dow  = factor(test$dow,  levels = 1:7))
      pred_cv <- as.integer(as.character(predict(fit_cv, newdata = nd)))
      cv_true <- c(cv_true, as.integer(test$load_category))
      cv_pred <- c(cv_pred, pred_cv)
    }
  }

  ar_k <- quad_kappa(ar_true, ar_pred); ar_a <- mean(ar_true == ar_pred)
  cv_k <- quad_kappa(cv_true, cv_pred); cv_a <- mean(cv_true == cv_pred)
  ar_m <- mean(abs(ar_true - ar_pred))
  ar_lock <- mean(ar_ndistinct == 1)

  rows[[length(rows) + 1]] <- data.frame(
    H = H, n_origins = length(ORIGINS), ar_spec = AR_SPEC,
    ar_kappa = ar_k, ar_acc = ar_a, ar_mae = ar_m, ar_frac_locked = ar_lock,
    cov_kappa = cv_k, cov_acc = cv_a)
  cat(sprintf("H=%3d | folds %3d | AR kappa %.3f acc %.3f (locked %2.0f%%) | COV kappa %.3f\n",
              H, length(ORIGINS), ar_k, ar_a, 100 * ar_lock, cv_k))
}

out <- do.call(rbind, rows)
fn <- file.path(OUTDIR, sprintf("energy_logistic_baselines_sweep_%s.csv", AR_SPEC))
write.csv(out, fn, row.names = FALSE)
cat(sprintf("\nwrote %s\n", fn))
cat("\n--- TikZ coordinates ---\n")
cat("AR (blind):      ", paste(sprintf("(%d,%.3f)", out$H, out$ar_kappa),  collapse = " "), "\n")
cat("Covariate-only:  ", paste(sprintf("(%d,%.3f)", out$H, out$cov_kappa), collapse = " "), "\n")
