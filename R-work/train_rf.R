#!/usr/bin/env Rscript
#
# Train a random forest with `satisfied` as the dependent variable and every
# other column as an independent variable.
#
# Usage:
#   Rscript train_rf.R [input.csv] [predictions.csv] [model.rds]
#
# Defaults: testdata.csv -> predictions.csv, model saved to rf_model.rds.
#
# Engine: ranger -- same algorithm as randomForest but much faster, with
# permutation importance and out-of-bag predictions. Falls back to
# randomForest when ranger is not installed.

source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE),
                                                  value = TRUE)[1])), "rf_common.R"))

args       <- commandArgs(trailingOnly = TRUE)
in_file    <- if (length(args) >= 1) args[1] else "testdata.csv"
out_file   <- if (length(args) >= 2) args[2] else "predictions.csv"
model_file <- if (length(args) >= 3) args[3] else "rf_model.rds"
target     <- "satisfied"
SEED       <- 42
NTREE      <- 1000

set.seed(SEED)
engine <- rf_engine()

cat("== Random forest on", target, "==\n")
cat("engine     :", engine, "\n")
cat("input      :", in_file, "\n\n")

raw <- rf_read_csv(in_file)
cat("rows       :", nrow(raw), "\n")
cat("columns    :", ncol(raw), "\n\n")

recipe <- rf_build_recipe(raw, target)

if (length(recipe$dropped_id))
  cat("dropping id-like columns:",
      paste(recipe$dropped_id, collapse = ", "), "\n")
if (length(recipe$dropped_constant))
  cat("dropping constant columns:",
      paste(recipe$dropped_constant, collapse = ", "), "\n")
if (length(c(recipe$dropped_id, recipe$dropped_constant))) cat("\n")

if (!length(recipe$predictors))
  stop("No usable predictor columns left after dropping id and constant columns.")

is_classification <- recipe$task == "classification"
y <- rf_prepare_target(raw[[target]], recipe)

if (is_classification) {
  cat("task       : classification (", length(recipe$y_levels), "classes )\n")
  print(table(y, useNA = "ifany"))
} else {
  cat("task       : regression\n")
  print(summary(y))
}
cat("\n")

prep <- rf_apply_recipe(raw, recipe)
if (prep$imputed > 0)
  cat("imputed    :", prep$imputed, "missing predictor value(s)\n")

n_fac <- sum(unlist(recipe$types) == "factor")
cat("predictors :", length(recipe$predictors),
    sprintf("(%d categorical, %d numeric)\n", n_fac,
            length(recipe$predictors) - n_fac))
cat("            ", paste(recipe$predictors, collapse = ", "), "\n\n")

model_data <- prep$data
model_data[[target]] <- y

# Rows with no outcome cannot be trained on, but are still predicted so the
# output lines up with the input row for row.
train_rows <- !is.na(y)
if (any(!train_rows))
  cat("note       :", sum(!train_rows),
      "row(s) have a missing outcome: predicted, not trained on\n\n")
train <- model_data[train_rows, , drop = FALSE]

p    <- length(recipe$predictors)
mtry <- max(1L, if (is_classification) floor(sqrt(p)) else floor(p / 3))
fml  <- as.formula(paste(target, "~ ."))

prob_fit <- NULL
if (engine == "ranger") {
  fit <- ranger::ranger(fml, data = train, num.trees = NTREE, mtry = mtry,
                        importance = "permutation", seed = SEED)
  # A separate probability forest gives calibrated class probabilities.
  if (is_classification)
    prob_fit <- ranger::ranger(fml, data = train, num.trees = NTREE,
                               mtry = mtry, probability = TRUE, seed = SEED)

  oob_err  <- fit$prediction.error
  imp      <- sort(ranger::importance(fit), decreasing = TRUE)
  pred_in  <- predict(fit, data = model_data)$predictions
  pred_oob <- fit$predictions
  prob_in  <- if (is_classification) predict(prob_fit, data = model_data)$predictions
  prob_oob <- if (is_classification) prob_fit$predictions
} else {
  fit <- randomForest::randomForest(fml, data = train, ntree = NTREE,
                                    mtry = mtry, importance = TRUE)
  oob_err  <- if (is_classification)
    mean(fit$predicted != train[[target]], na.rm = TRUE) else fit$mse[NTREE]
  imp      <- sort(randomForest::importance(fit)[, 1], decreasing = TRUE)
  pred_in  <- predict(fit, newdata = model_data)
  pred_oob <- fit$predicted
  prob_in  <- if (is_classification) predict(fit, newdata = model_data, type = "prob")
  prob_oob <- if (is_classification) fit$votes
}

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

# OOB results only exist for trained rows; pad back out to the full frame.
expand <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.matrix(v)) {
    o <- matrix(NA_real_, nrow(model_data), ncol(v),
                dimnames = list(NULL, colnames(v)))
    o[train_rows, ] <- v
    o
  } else {
    o <- rep(NA, nrow(model_data)); o[train_rows] <- as.character(v); o
  }
}
pred_oob_full <- expand(pred_oob)
prob_oob_full <- expand(prob_oob)

if (is_classification) {
  actual <- as.character(y)
  cm_in  <- table(predicted = as.character(pred_in)[train_rows],
                  actual    = actual[train_rows])
  cat("\n---- in-sample confusion matrix (predicted vs actual) ----\n")
  print(cm_in)
  cat(sprintf("in-sample accuracy : %.2f%%\n", 100 * sum(diag(cm_in)) / sum(cm_in)))

  cm_oob <- table(predicted = pred_oob_full[train_rows], actual = actual[train_rows])
  cat("\n---- out-of-bag confusion matrix (predicted vs actual) ----\n")
  print(cm_oob)
  cat(sprintf("out-of-bag accuracy: %.2f%%\n", 100 * sum(diag(cm_oob)) / sum(cm_oob)))
  cat("\nIn-sample accuracy is optimistic -- every row helped grow most of\n",
      "the trees that score it. Judge the model by the out-of-bag numbers.\n", sep = "")
} else {
  yt      <- y[train_rows]
  res_in  <- yt - as.numeric(pred_in)[train_rows]
  res_oob <- yt - as.numeric(pred_oob_full)[train_rows]
  sst     <- sum((yt - mean(yt))^2)
  cat(sprintf("\nin-sample  RMSE: %.4f   R2: %.4f\n",
              sqrt(mean(res_in^2)), 1 - sum(res_in^2) / sst))
  cat(sprintf("out-of-bag RMSE: %.4f   R2: %.4f\n",
              sqrt(mean(res_oob^2)), 1 - sum(res_oob^2) / sst))
}

out <- raw
out$pred_in_sample <- if (is_classification) as.character(pred_in) else as.numeric(pred_in)
out$pred_oob       <- if (is_classification) pred_oob_full else as.numeric(pred_oob_full)

if (is_classification) {
  colnames(prob_in) <- paste0("prob_in_", colnames(prob_in))
  out <- cbind(out, round(as.data.frame(prob_in), 4))
  colnames(prob_oob_full) <- paste0("prob_oob_", colnames(prob_oob_full))
  out <- cbind(out, round(as.data.frame(prob_oob_full), 4))
  out$correct_oob <- out$pred_oob == as.character(raw[[target]])
}

write.csv(out, out_file, row.names = FALSE)
cat("\npredictions written to:", out_file,
    sprintf("(%d rows, %d columns)\n", nrow(out), ncol(out)))

# Save the forest together with the recipe, so predict_rf.R can prepare new
# data identically. The fit alone is not enough.
saveRDS(list(fit = fit, prob_fit = prob_fit, recipe = recipe, engine = engine,
             ntree = NTREE, mtry = mtry, seed = SEED,
             trained_on = basename(in_file), n_train = sum(train_rows),
             r_version = R.version.string),
        model_file)
cat("model saved to        :", model_file, "\n")
cat("\nScore a new dataset with:\n  Rscript predict_rf.R <newdata.csv> <out.csv>",
    model_file, "\n")
