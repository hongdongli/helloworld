# Random forest for `satisfied`

Fits a random forest with `satisfied` as the dependent variable and every other
column as an independent variable, then scores data with it.

| file | what it does |
| --- | --- |
| `train_rf.R` | trains the forest, predicts the training rows, saves the model |
| `predict_rf.R` | scores a **new** dataset with the saved model |
| `rf_common.R` | shared data-preparation recipe used by both |
| `example_data.csv` | synthetic survey data (400 rows) so the scripts run as-is |
| `make_example_data.R` | generates that example file |

## Train

```sh
Rscript train_rf.R testdata.csv predictions.csv rf_model.rds
```

All three arguments are optional (`testdata.csv`, `predictions.csv`,
`rf_model.rds`). This prints out-of-bag error, variable importance and
confusion matrices, writes predictions for the training rows, and saves the
model.

## Predict on new data

```sh
Rscript predict_rf.R newdata.csv new_predictions.csv rf_model.rds
```

The new file needs the same predictor columns as the training file — matched
**by name**, so column order does not matter and extra columns are ignored.
Include `satisfied` in it and you get a held-out accuracy report; leave it out
and you get predictions only.

## Requirements

```r
install.packages("ranger")        # or: install.packages("randomForest")
```

On Debian/Ubuntu without CRAN access: `apt-get install r-cran-ranger`.
Set `RF_ENGINE=randomForest` to force the other engine.

## Why ranger instead of party

`party::cforest` builds conditional inference forests, worth using when you
need unbiased variable importance across predictors with very different
numbers of categories. For straight prediction it is slow — often one to two
orders of magnitude slower than `ranger` — and it does not expose out-of-bag
predictions as directly.

`ranger` is a modern C++ implementation of Breiman's random forest: same
algorithm as `randomForest`, much faster, with permutation importance and a
separate probability forest for calibrated class probabilities. The scripts
fall back to `randomForest` automatically when `ranger` is missing.

## How the data is prepared

Training learns a *recipe* and saves it inside `rf_model.rds` next to the
forest. Prediction replays that same recipe, which is what makes scores on new
data comparable:

- Row-identifier columns (`X`, `id`, `index`, …) are dropped so the forest
  cannot memorise them.
- Task is chosen automatically: classification when `satisfied` is
  non-numeric or has 10 or fewer distinct values, regression otherwise.
- Text columns and small whole-number codes (10 or fewer distinct values —
  survey scales) become factors, with the levels recorded.
- Constant columns are dropped; missing predictors are imputed with the
  **training** median (numeric) or most frequent level (categorical), so no
  row is ever dropped from the output.
- At prediction time a category the forest never saw is imputed and reported
  as a warning — many of those means the new file does not match the training
  file.
- Rows with a missing `satisfied` value are predicted but not trained on.
- 1000 trees, `mtry = sqrt(p)` for classification and `p/3` for regression.

## Output

Both scripts write the input file back out — same rows, same order — with
prediction columns appended.

`train_rf.R` adds `pred_in_sample`, `pred_oob`, `prob_in_*`, `prob_oob_*` and
`correct_oob`. `predict_rf.R` adds `prediction`, `prob_*` and (when the true
outcome is present) `correct` or `residual`.

**Read `pred_oob`, not `pred_in_sample`.** Predicting the training data with a
random forest gives near-perfect accuracy — every row helped grow almost every
tree that then votes on it. Out-of-bag scores each row using only the trees
that never saw it, which is the honest number. On the example data the split is
100% in-sample versus 82% out-of-bag, and a genuine held-out set scores 79% —
right in line with out-of-bag, and nowhere near the in-sample figure.

## Example

```sh
Rscript make_example_data.R                                  # writes example_data.csv
Rscript train_rf.R example_data.csv example_predictions.csv  # -> rf_model.rds
Rscript predict_rf.R example_data.csv rescored.csv           # scoring path
```

`example_data.csv` is **synthetic**, included only so the scripts are runnable.
Replace it with your own `testdata.csv`.
