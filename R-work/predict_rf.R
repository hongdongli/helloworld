#!/usr/bin/env Rscript
#
# Score a new dataset with a forest saved by train_rf.R.
#
# Usage:
#   Rscript predict_rf.R <newdata.csv> [predictions.csv] [model.rds]
#
# The new file needs the same predictor columns as the training file. Extra
# columns are ignored. The outcome column (`satisfied`) is optional: include
# it and you get an accuracy report against a genuinely held-out set, leave
# it out and you just get predictions.

source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE),
                                                  value = TRUE)[1])), "rf_common.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1)
  stop("Usage: Rscript predict_rf.R <newdata.csv> [predictions.csv] [model.rds]")

in_file    <- args[1]
out_file   <- if (length(args) >= 2) args[2] else "new_predictions.csv"
model_file <- if (length(args) >= 3) args[3] else "rf_model.rds"

if (!file.exists(model_file))
  stop("Model file not found: ", model_file, "\nRun train_rf.R first.")

bundle <- readRDS(model_file)

# Loading the engine's namespace is what registers its predict() method.
if (!requireNamespace(bundle$engine, quietly = TRUE))
  stop("This model was fitted with '", bundle$engine, "', which is not ",
       "installed here. Install it with: install.packages('", bundle$engine, "')")

recipe <- bundle$recipe
target <- recipe$target
is_classification <- recipe$task == "classification"

cat("== Scoring new data ==\n")
cat("model      :", model_file, sprintf("(%s, %d trees, trained on %s / %d rows)\n",
                                        bundle$engine, bundle$ntree,
                                        bundle$trained_on, bundle$n_train))
cat("input      :", in_file, "\n")

raw <- rf_read_csv(in_file)
cat("rows       :", nrow(raw), "\n")

extra <- setdiff(names(raw), c(recipe$predictors, target))
if (length(extra))
  cat("ignoring   :", paste(extra, collapse = ", "), "\n")

prep <- rf_apply_recipe(raw, recipe)
newx <- prep$data

if (prep$imputed > 0)
  cat("imputed    :", prep$imputed,
      "missing predictor value(s) using training-set fill values\n")

# Worth calling out: a category the forest never saw during training has no
# split to route it, so it gets imputed. A lot of these means the new file
# does not look like the training file.
if (length(prep$unseen)) {
  cat("\nWARNING: values not present in the training data (imputed):\n")
  for (nm in names(prep$unseen))
    cat("  ", nm, ": ", paste(utils::head(prep$unseen[[nm]], 8), collapse = ", "),
        if (length(prep$unseen[[nm]]) > 8) ", ..." else "", "\n", sep = "")
}
cat("\n")

if (bundle$engine == "ranger") {
  pred <- predict(bundle$fit, data = newx)$predictions
  prob <- if (is_classification) predict(bundle$prob_fit, data = newx)$predictions
} else {
  pred <- predict(bundle$fit, newdata = newx)
  prob <- if (is_classification) predict(bundle$fit, newdata = newx, type = "prob")
}

out <- raw
out$prediction <- if (is_classification) as.character(pred) else as.numeric(pred)
if (is_classification) {
  colnames(prob) <- paste0("prob_", colnames(prob))
  out <- cbind(out, round(as.data.frame(prob), 4))
}

# If the new file carries the true outcome, this is a real held-out score --
# more trustworthy than anything measured on the training data.
if (target %in% names(raw)) {
  actual <- raw[[target]]
  known  <- !is.na(actual)
  if (any(known)) {
    if (is_classification) {
      cm <- table(predicted = out$prediction[known],
                  actual    = as.character(actual)[known])
      cat("---- confusion matrix on held-out data (predicted vs actual) ----\n")
      print(cm)
      acc <- sum(diag(cm)) / sum(cm)
      cat(sprintf("accuracy: %.2f%% (%d of %d rows)\n",
                  100 * acc, sum(diag(cm)), sum(cm)))
      out$correct <- out$prediction == as.character(actual)
    } else {
      a   <- as.numeric(actual)[known]
      res <- a - out$prediction[known]
      cat(sprintf("held-out RMSE: %.4f   MAE: %.4f   R2: %.4f\n",
                  sqrt(mean(res^2)), mean(abs(res)),
                  1 - sum(res^2) / sum((a - mean(a))^2)))
      out$residual <- as.numeric(actual) - out$prediction
    }
    cat("\n")
  }
} else {
  cat("note: '", target, "' is not in this file, so there is nothing to ",
      "score against -- predictions only.\n\n", sep = "")
}

write.csv(out, out_file, row.names = FALSE)
cat("predictions written to:", out_file,
    sprintf("(%d rows, %d columns)\n", nrow(out), ncol(out)))
