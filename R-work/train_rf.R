#!/usr/bin/env Rscript
#
# Random forest model for the `satisfied` outcome.
#
# `satisfied` is the dependent variable, every other column is a predictor.
# The script fits a random forest and writes predictions back out for the
# same rows it was trained on.
#
# Usage:
#   Rscript train_rf.R [input.csv] [output.csv]
#
# Defaults: testdata.csv -> predictions.csv (relative to this script's dir).
#
# Engine: ranger. It is the same algorithm family as party::cforest but is
# far faster, handles larger data, and gives permutation importance out of
# the box. randomForest is used as a fallback when ranger is not installed.

suppressPackageStartupMessages({
  ok_ranger <- requireNamespace("ranger", quietly = TRUE)
  ok_rf     <- requireNamespace("randomForest", quietly = TRUE)
})

if (!ok_ranger && !ok_rf) {
  stop("Need either the 'ranger' or the 'randomForest' package. ",
       "Install with: install.packages('ranger')")
}
engine <- if (ok_ranger) "ranger" else "randomForest"

args     <- commandArgs(trailingOnly = TRUE)
in_file  <- if (length(args) >= 1) args[1] else "testdata.csv"
out_file <- if (length(args) >= 2) args[2] else "predictions.csv"
target   <- "satisfied"
SEED     <- 42
NTREE    <- 1000

set.seed(SEED)

cat("== Random forest on", target, "==\n")
cat("engine     :", engine, "\n")
cat("input      :", in_file, "\n\n")

# ---------------------------------------------------------------- load data
if (!file.exists(in_file)) stop("Input file not found: ", in_file)

raw <- read.csv(in_file, stringsAsFactors = FALSE, check.names = TRUE,
                na.strings = c("NA", "", "NULL", "?", "."))

if (!target %in% names(raw)) {
  stop("Column '", target, "' not found. Columns are: ",
       paste(names(raw), collapse = ", "))
}

cat("rows       :", nrow(raw), "\n")
cat("columns    :", ncol(raw), "\n\n")

dat <- raw

# Drop columns that are pure row identifiers -- they carry no signal and a
# random forest will happily memorise them.
id_like <- vapply(names(dat), function(nm) {
  nm != target &&
    (grepl("^(x|id|index|rowid|row|no|序号|编号)$", tolower(nm)) ||
       length(unique(dat[[nm]])) == nrow(dat) && !is.numeric(dat[[nm]]))
}, logical(1))
if (any(id_like)) {
  cat("dropping id-like columns:", paste(names(dat)[id_like], collapse = ", "), "\n\n")
  dat <- dat[, !id_like, drop = FALSE]
}

# ------------------------------------------------------- decide the problem
y_raw  <- dat[[target]]
n_lvl  <- length(unique(y_raw[!is.na(y_raw)]))
# Treat a non-numeric outcome, or a numeric one with few distinct values, as
# classification; anything else is regression.
is_classification <- !is.numeric(y_raw) || n_lvl <= 10

if (is_classification) {
  dat[[target]] <- factor(y_raw)
  cat("task       : classification (", n_lvl, "classes )\n")
  print(table(dat[[target]], useNA = "ifany"))
} else {
  dat[[target]] <- as.numeric(y_raw)
  cat("task       : regression\n")
  print(summary(dat[[target]]))
}
cat("\n")

# Rows with a missing outcome cannot be trained on. Keep them aside so the
# output still lines up with the input row-for-row.
train_rows <- !is.na(dat[[target]])
if (any(!train_rows)) {
  cat("note       :", sum(!train_rows),
      "row(s) have a missing outcome and are predicted but not trained on\n\n")
}

# ----------------------------------------------------------- prepare X
predictors <- setdiff(names(dat), target)

for (nm in predictors) {
  col <- dat[[nm]]
  if (is.character(col) || is.logical(col)) {
    dat[[nm]] <- factor(col)
  } else if (is.numeric(col) && length(unique(col[!is.na(col)])) <= 10 &&
             all(col[!is.na(col)] == round(col[!is.na(col)]))) {
    # Small integer codes are almost always survey categories, not counts.
    dat[[nm]] <- factor(col)
  }
}

# Constant predictors contribute nothing and upset some engines.
const <- vapply(predictors, function(nm)
  length(unique(dat[[nm]][!is.na(dat[[nm]])])) < 2, logical(1))
if (any(const)) {
  cat("dropping constant columns:", paste(predictors[const], collapse = ", "), "\n\n")
  dat <- dat[, !(names(dat) %in% predictors[const]), drop = FALSE]
  predictors <- setdiff(names(dat), target)
}

# Simple imputation: median for numerics, most common level for factors.
# Random forests need complete cases, and dropping rows would break the
# row-for-row output contract.
n_imputed <- 0L
for (nm in predictors) {
  col <- dat[[nm]]
  miss <- is.na(col)
  if (!any(miss)) next
  n_imputed <- n_imputed + sum(miss)
  if (is.factor(col)) {
    fill <- names(sort(table(col), decreasing = TRUE))[1]
    col <- addNA(col, ifany = FALSE)
    col[miss] <- fill
    dat[[nm]] <- droplevels(factor(col, levels = levels(dat[[nm]])))
  } else {
    col[miss] <- median(col, na.rm = TRUE)
    dat[[nm]] <- col
  }
}
if (n_imputed > 0) cat("imputed    :", n_imputed, "missing predictor value(s)\n\n")

cat("predictors :", length(predictors), "->",
    paste(predictors, collapse = ", "), "\n\n")

train <- dat[train_rows, , drop = FALSE]

# mtry: the usual defaults -- sqrt(p) for classification, p/3 for regression.
p    <- length(predictors)
mtry <- max(1L, if (is_classification) floor(sqrt(p)) else floor(p / 3))

# ------------------------------------------------------------------- fit
fml <- as.formula(paste(target, "~ ."))

if (engine == "ranger") {
  fit <- ranger::ranger(
    formula      = fml,
    data         = train,
    num.trees    = NTREE,
    mtry         = mtry,
    importance   = "permutation",
    probability  = FALSE,
    seed         = SEED,
    keep.inbag   = TRUE
  )
  # A second probability forest gives calibrated class probabilities.
  prob_fit <- if (is_classification) ranger::ranger(
    formula     = fml,
    data        = train,
    num.trees   = NTREE,
    mtry        = mtry,
    probability = TRUE,
    seed        = SEED
  ) else NULL

  oob_err <- fit$prediction.error
  imp     <- sort(ranger::importance(fit), decreasing = TRUE)

  # In-sample (all trees vote, including the trees each row grew in).
  pred_in <- predict(fit, data = dat)$predictions
  # Out-of-bag: each row is scored only by trees that never saw it. This is
  # the honest estimate of how the model does on unseen rows.
  pred_oob <- fit$predictions

  prob_in <- if (is_classification) predict(prob_fit, data = dat)$predictions else NULL
  prob_oob <- if (is_classification) prob_fit$predictions else NULL

} else {
  fit <- randomForest::randomForest(
    formula    = fml,
    data       = train,
    ntree      = NTREE,
    mtry       = mtry,
    importance = TRUE
  )
  oob_err <- if (is_classification)
    mean(fit$predicted != train[[target]], na.rm = TRUE) else
    fit$mse[NTREE]
  imp <- sort(randomForest::importance(fit)[, 1], decreasing = TRUE)

  pred_in  <- predict(fit, newdata = dat)
  pred_oob <- fit$predicted
  prob_in  <- if (is_classification) predict(fit, newdata = dat, type = "prob") else NULL
  prob_oob <- if (is_classification) fit$votes else NULL
}

# ---------------------------------------------------------------- report
cat("---- model ----\n")
cat("trees      :", NTREE, "\n")
cat("mtry       :", mtry, "\n")
if (is_classification) {
  cat("OOB error  :", sprintf("%.4f", oob_err),
      sprintf("(OOB accuracy %.2f%%)\n", 100 * (1 - oob_err)))
} else {
  cat("OOB MSE    :", sprintf("%.4f", oob_err),
      sprintf("(RMSE %.4f)\n", sqrt(oob_err)))
}
cat("\n---- variable importance ----\n")
print(round(imp, 5))

# Line the training-row predictions back up with the full data frame.
expand <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.matrix(v)) {
    out <- matrix(NA_real_, nrow = nrow(dat), ncol = ncol(v),
                  dimnames = list(NULL, colnames(v)))
    out[train_rows, ] <- v
    out
  } else {
    out <- rep(NA, nrow(dat))
    out[train_rows] <- as.character(v)
    out
  }
}
pred_oob_full <- expand(pred_oob)
prob_oob_full <- expand(prob_oob)

if (is_classification) {
  actual <- as.character(dat[[target]])
  cat("\n---- in-sample confusion matrix (predicted vs actual) ----\n")
  cm_in <- table(predicted = as.character(pred_in)[train_rows],
                 actual    = actual[train_rows])
  print(cm_in)
  cat(sprintf("in-sample accuracy : %.2f%%\n",
              100 * sum(diag(cm_in)) / sum(cm_in)))

  cat("\n---- out-of-bag confusion matrix (predicted vs actual) ----\n")
  cm_oob <- table(predicted = pred_oob_full[train_rows],
                  actual    = actual[train_rows])
  print(cm_oob)
  cat(sprintf("out-of-bag accuracy: %.2f%%\n",
              100 * sum(diag(cm_oob)) / sum(cm_oob)))
  cat("\nIn-sample accuracy is optimistic -- every row helped grow most of\n",
      "the trees that score it. Judge the model by the out-of-bag numbers.\n", sep = "")
} else {
  res_in  <- dat[[target]][train_rows] - as.numeric(pred_in)[train_rows]
  res_oob <- dat[[target]][train_rows] - as.numeric(pred_oob_full)[train_rows]
  cat(sprintf("\nin-sample  RMSE: %.4f   R2: %.4f\n",
              sqrt(mean(res_in^2)),
              1 - sum(res_in^2) / sum((dat[[target]][train_rows] -
                                         mean(dat[[target]][train_rows]))^2)))
  cat(sprintf("out-of-bag RMSE: %.4f   R2: %.4f\n",
              sqrt(mean(res_oob^2)),
              1 - sum(res_oob^2) / sum((dat[[target]][train_rows] -
                                          mean(dat[[target]][train_rows]))^2)))
}

# ---------------------------------------------------------------- output
out <- raw
out$pred_in_sample <- if (is_classification) as.character(pred_in) else as.numeric(pred_in)
out$pred_oob       <- if (is_classification) pred_oob_full else as.numeric(pred_oob_full)

if (is_classification) {
  if (!is.null(prob_in)) {
    colnames(prob_in) <- paste0("prob_in_", colnames(prob_in))
    out <- cbind(out, round(as.data.frame(prob_in), 4))
  }
  if (!is.null(prob_oob_full)) {
    colnames(prob_oob_full) <- paste0("prob_oob_", colnames(prob_oob_full))
    out <- cbind(out, round(as.data.frame(prob_oob_full), 4))
  }
  out$correct_oob <- out$pred_oob == as.character(raw[[target]])
}

write.csv(out, out_file, row.names = FALSE)
cat("\npredictions written to:", out_file, "\n")
cat("rows:", nrow(out), " columns:", ncol(out), "\n")

saveRDS(fit, file.path(dirname(out_file), "rf_model.rds"))
cat("model saved to:", file.path(dirname(out_file), "rf_model.rds"), "\n")
