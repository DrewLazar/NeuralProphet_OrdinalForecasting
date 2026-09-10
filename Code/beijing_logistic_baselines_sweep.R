# =====================================================================
# Two multinomial-logistic baselines swept across horizons, one run.
#
#   (A) AR baseline (Chen & Chiu style): category_t ~ ylag + covariates + month
#       Evaluated BLIND multi-step -- iterated one-step, feeding each predicted
#       category back as the lag. Collapses (tends to lock onto one class).
#
#   (B) Covariate-only: category_t ~ covariates + month + hour   (NO AR term)
#       Each hour predicted independently from its own observed covariates;
#       no recursion, so roughly flat across horizons.
#
# Writes one CSV with a row per horizon (both methods' kappa/acc side by side)
# and prints ready-to-paste TikZ coordinates for each line. Runs locally in
# seconds per fit -- no cluster.
# =====================================================================

suppressMessages(library(nnet))

if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable())
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

DATA        <- Sys.getenv("BEIJING_CSV", "BeijingAQI_data2.csv")
FIRST_ORIGIN<- as.integer(Sys.getenv("FIRST_ORIGIN", "2100"))
COVARS      <- c("TEMP", "DEWP", "PRES", "Iws")
NCAT        <- 4L
OUTDIR      <- Sys.getenv("OUTDIR", "beijing_out")
dir.create(OUTDIR, showWarnings = FALSE)

HORIZONS <- as.integer(strsplit(Sys.getenv("HORIZONS", "12,24,48,72,120,168,216,240"), ",")[[1]])

df <- read.csv(DATA, stringsAsFactors = FALSE)
df$datetime <- as.POSIXct(df$datetime, tz = "UTC")
df$month <- as.integer(format(df$datetime, "%m"))
df$hour  <- as.integer(format(df$datetime, "%H"))
n <- nrow(df)
cat(sprintf("loaded %s: %d rows\n", DATA, n))

quad_kappa <- function(true, pred, K = NCAT) {
  true <- factor(true, levels = 1:K); pred <- factor(pred, levels = 1:K)
  O <- table(true, pred); O <- O / sum(O)
  w <- outer(1:K, 1:K, function(i, j) (i - j)^2) / (K - 1)^2
  r <- rowSums(O); c <- colSums(O); E <- outer(r, c)
  1 - sum(w * O) / sum(w * E)
}

# frame for the AR model: row t uses category_{t-1} and covariates_{t-1}
make_ar_frame <- function(d) {
  L <- nrow(d)
  data.frame(
    y     = factor(d$aqi_category[2:L], levels = 1:NCAT),
    ylag  = factor(d$aqi_category[1:(L-1)], levels = 1:NCAT),
    d[1:(L-1), COVARS, drop = FALSE],
    month = factor(d$month[2:L], levels = 1:12)
  )
}
# frame for the covariate-only model: same-time covariates + calendar
make_cov_frame <- function(d) {
  data.frame(
    y     = factor(d$aqi_category, levels = 1:NCAT),
    d[, COVARS, drop = FALSE],
    month = factor(d$month, levels = 1:12),
    hour  = factor(d$hour,  levels = 0:23)
  )
}

rows <- list()
for (H in HORIZONS) {
  STEP <- H
  ORIGINS <- seq(FIRST_ORIGIN, n - H, by = STEP)

  ar_true <- integer(0); ar_pred <- integer(0)
  cv_true <- integer(0); cv_pred <- integer(0)
  ar_ndistinct <- integer(0)

  for (origin in ORIGINS) {
    train <- df[1:origin, ]
    test  <- df[(origin + 1):(origin + H), ]

    # ---------- (A) AR baseline, blind multi-step ----------
    tr_ar <- make_ar_frame(train)
    fit_ar <- tryCatch(
      multinom(y ~ ylag + TEMP + DEWP + PRES + Iws + month,
               data = tr_ar, trace = FALSE, maxit = 200),
      error = function(e) tryCatch(
        multinom(y ~ ylag + TEMP + DEWP + PRES + Iws,
                 data = tr_ar, trace = FALSE, maxit = 200),
        error = function(e2) NULL))
    if (!is.null(fit_ar)) {
      has_month <- "month" %in% all.vars(formula(fit_ar))
      pred_blind <- integer(H)
      last_cat <- train$aqi_category[origin]
      for (h in 1:H) {
        nd <- data.frame(
          ylag  = factor(last_cat, levels = 1:NCAT),
          test[h, COVARS, drop = FALSE],
          month = factor(test$month[h], levels = 1:12))
        if (!has_month) nd$month <- NULL
        ph <- as.integer(as.character(predict(fit_ar, newdata = nd)))
        pred_blind[h] <- ph
        last_cat <- ph
      }
      ar_true <- c(ar_true, as.integer(test$aqi_category))
      ar_pred <- c(ar_pred, pred_blind)
      ar_ndistinct <- c(ar_ndistinct, length(unique(pred_blind)))
    }

    # ---------- (B) covariate-only ----------
    tr_cv <- make_cov_frame(train)
    fit_cv <- tryCatch(
      multinom(y ~ TEMP + DEWP + PRES + Iws + month + hour,
               data = tr_cv, trace = FALSE, maxit = 200),
      error = function(e) tryCatch(
        multinom(y ~ TEMP + DEWP + PRES + Iws + month,
                 data = tr_cv, trace = FALSE, maxit = 200),
        error = function(e2) NULL))
    if (!is.null(fit_cv)) {
      vars <- all.vars(formula(fit_cv))
      nd <- data.frame(
        test[, COVARS, drop = FALSE],
        month = factor(test$month, levels = 1:12),
        hour  = factor(test$hour,  levels = 0:23))
      if (!("hour"  %in% vars)) nd$hour  <- NULL
      if (!("month" %in% vars)) nd$month <- NULL
      pred_cv <- as.integer(as.character(predict(fit_cv, newdata = nd)))
      cv_true <- c(cv_true, as.integer(test$aqi_category))
      cv_pred <- c(cv_pred, pred_cv)
    }
  }

  ar_k <- quad_kappa(ar_true, ar_pred); ar_a <- mean(ar_true == ar_pred)
  cv_k <- quad_kappa(cv_true, cv_pred); cv_a <- mean(cv_true == cv_pred)
  ar_lock <- mean(ar_ndistinct == 1)   # fraction of origins that locked to one class

  rows[[length(rows) + 1]] <- data.frame(
    H = H, n_origins = length(ORIGINS),
    ar_kappa = ar_k, ar_acc = ar_a, ar_frac_locked = ar_lock,
    cov_kappa = cv_k, cov_acc = cv_a)
  cat(sprintf("H=%3d | folds %3d | AR kappa %.3f (locked %.0f%%) | COV kappa %.3f\n",
              H, length(ORIGINS), ar_k, 100 * ar_lock, cv_k))
}

out <- do.call(rbind, rows)
write.csv(out, file.path(OUTDIR, "logistic_baselines_sweep.csv"), row.names = FALSE)
cat(sprintf("\nwrote %s/logistic_baselines_sweep.csv\n", OUTDIR))

cat("\n--- TikZ coordinates ---\n")
cat("AR (blind):      ", paste(sprintf("(%d,%.3f)", out$H, out$ar_kappa),  collapse = " "), "\n")
cat("Covariate-only:  ", paste(sprintf("(%d,%.3f)", out$H, out$cov_kappa), collapse = " "), "\n")
