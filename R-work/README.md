# Random forest for `Satisfied`

Fits a random forest with `Satisfied` as the dependent variable and every other
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
Rscript train_rf.R testdata.tsv predictions.csv rf_model.rds
```

Reads `.tsv`/`.txt`/`.tab` as tab-separated and anything else as CSV. The
outcome column is matched case-insensitively (`Satisfied` or `satisfied`); set
`RF_TARGET` to model a different column.

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
Include `Satisfied` in it and you get a held-out accuracy report; leave it out
and you get predictions only.

## Requirements

```r
install.packages("ranger")        # or: install.packages("randomForest")
```

On Debian/Ubuntu without CRAN access: `apt-get install r-cran-ranger`.
Set `RF_ENGINE=randomForest` to force the other engine.

## Engines

Default is **`party::cforest`** — conditional inference forests, via
`cforest_unbiased()`. That matters for this dataset: every predictor is
categorical with a different number of levels (2 to 6), and a standard
random forest's Gini importance is biased toward the many-level columns.
cforest's permutation importance is not.

Set `RF_ENGINE=ranger` or `RF_ENGINE=randomForest` to switch. `ranger` is much
faster and worth it on large or mostly-numeric data; the scripts fall back
automatically to whichever package is installed.

`RF_VARIMP=conditional` computes *conditional* permutation importance, which
adjusts for correlation between predictors. It is slow (~100 s on 279 rows)
and off by default.

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
- Constant columns are dropped, as is either half of a pair of columns that
  encode the same variable twice (a one-to-one mapping between their levels).
  A duplicated predictor splits importance between the copies and makes each
  look half as important as the underlying variable is.
- Missing **categorical** values are kept as their own `(missing)` level
  rather than imputed — imputing a column that is 11% missing to its mode
  invents data, and "not recorded" is often informative. Set `RF_NA_LEVEL=0`
  to impute to the most frequent level instead. Numeric predictors are still
  imputed with the training median. Either way no row is dropped.
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
random forest is optimistic — every row helped grow most of the trees that then
vote on it. Out-of-bag scores each row using only the trees that never saw it,
which is the honest number. On `testdata.tsv` that is 78.85% in-sample versus
72.04% out-of-bag; on the synthetic example the gap is far wider (100% vs 82%,
with a genuine held-out set scoring 79% — in line with out-of-bag, nowhere near
in-sample).

Always compare against the majority-class baseline, which `train_rf.R` prints.

## Example

```sh
Rscript make_example_data.R                                  # writes example_data.csv
Rscript train_rf.R example_data.csv example_predictions.csv  # -> rf_model.rds
Rscript predict_rf.R example_data.csv rescored.csv           # scoring path
```

`example_data.csv` is **synthetic**, included only so the scripts are runnable
on numeric data. The real dataset is `testdata.tsv`.

## Results on `testdata.tsv`

279 rows, 7 usable predictors (`colPath.1` dropped as a duplicate of
`colPath`), 1000 trees, `mtry = 2`.

| | accuracy |
| --- | --- |
| majority-class baseline | 55.20% |
| out-of-bag | **72.04%** |
| in-sample | 78.85% |

Variable importance, most to least (conditional permutation in brackets):

1. `colMention` — 0.0454 (0.0314)
2. `colRegister` — 0.0453 (0.0263)
3. `colBackground` — 0.0254 (0.0134)
4. `colPath` — 0.0185 (0.0149)
5. `colFaminly` — 0.0177 (0.0051)
6. `colFDC` — 0.0085 (0.0078)
7. `colAge` — −0.0004 (0.0008)

`colAge` sits at zero either way — it carries no information about `Satisfied`
once the others are known.
