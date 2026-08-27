# Shared data preparation for train_rf.R and predict_rf.R.
#
# The point of putting this in one file: a model is only usable on new data if
# the new data is prepared *exactly* the way the training data was -- same
# columns, same factor levels in the same order, same imputation values. So
# training records a "recipe" alongside the fitted forest, and prediction
# replays it.

MAX_CLASS_LEVELS <- 10

rf_read_csv <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
  read.csv(path, stringsAsFactors = FALSE, check.names = TRUE,
           na.strings = c("NA", "", "NULL", "?", "."))
}

# Row identifiers carry no signal and a forest will happily memorise them.
rf_id_like <- function(dat, target) {
  vapply(names(dat), function(nm) {
    if (nm == target) return(FALSE)
    if (grepl("^(x|id|index|rowid|row|no|serial)$", tolower(nm))) return(TRUE)
    # A non-numeric column with a distinct value on every row is a key.
    !is.numeric(dat[[nm]]) && length(unique(dat[[nm]])) == nrow(dat)
  }, logical(1))
}

# Should this column be treated as categorical?
rf_is_categorical <- function(col) {
  obs <- col[!is.na(col)]
  if (is.character(col) || is.logical(col) || is.factor(col)) return(TRUE)
  # Small whole-number codes are survey scales, not counts.
  is.numeric(col) && length(unique(obs)) <= MAX_CLASS_LEVELS &&
    all(obs == round(obs))
}

# Learn the preparation rules from the training data.
rf_build_recipe <- function(raw, target) {
  if (!target %in% names(raw))
    stop("Column '", target, "' not found. Columns are: ",
         paste(names(raw), collapse = ", "))

  drop_cols <- names(raw)[rf_id_like(raw, target)]
  candidates <- setdiff(names(raw), c(target, drop_cols))

  types <- list(); levs <- list(); fills <- list(); const <- character()

  for (nm in candidates) {
    col <- raw[[nm]]
    obs <- col[!is.na(col)]
    if (length(unique(obs)) < 2) { const <- c(const, nm); next }
    if (rf_is_categorical(col)) {
      f <- factor(col)
      types[[nm]] <- "factor"
      levs[[nm]]  <- levels(f)
      fills[[nm]] <- names(sort(table(f), decreasing = TRUE))[1]
    } else {
      types[[nm]] <- "numeric"
      fills[[nm]] <- median(as.numeric(obs))
    }
  }

  y     <- raw[[target]]
  n_lvl <- length(unique(y[!is.na(y)]))
  task  <- if (!is.numeric(y) || n_lvl <= MAX_CLASS_LEVELS)
    "classification" else "regression"

  list(
    target     = target,
    task       = task,
    y_levels   = if (task == "classification") levels(factor(y)) else NULL,
    predictors = setdiff(candidates, const),
    types      = types,
    levels     = levs,
    fills      = fills,
    dropped_id       = drop_cols,
    dropped_constant = const
  )
}

# Replay the recipe on any data frame -- training or new.
#
# Returns the model frame plus what had to be patched up, so the caller can
# report it rather than silently papering over a mismatched file.
rf_apply_recipe <- function(raw, recipe) {
  missing_cols <- setdiff(recipe$predictors, names(raw))
  if (length(missing_cols))
    stop("Data is missing predictor column(s) the model needs: ",
         paste(missing_cols, collapse = ", "))

  out     <- data.frame(row.names = seq_len(nrow(raw)))
  unseen  <- list()
  imputed <- 0L

  for (nm in recipe$predictors) {
    col <- raw[[nm]]
    if (identical(recipe$types[[nm]], "factor")) {
      chr <- as.character(col)
      # A level the forest never saw has no split to follow. Treat it as
      # missing and impute, rather than erroring out on one stray row.
      bad <- !is.na(chr) & !(chr %in% recipe$levels[[nm]])
      if (any(bad)) {
        unseen[[nm]] <- sort(unique(chr[bad]))
        chr[bad] <- NA
      }
      miss <- is.na(chr)
      imputed <- imputed + sum(miss)
      chr[miss] <- recipe$fills[[nm]]
      out[[nm]] <- factor(chr, levels = recipe$levels[[nm]])
    } else {
      num <- suppressWarnings(as.numeric(col))
      miss <- is.na(num)
      imputed <- imputed + sum(miss)
      num[miss] <- recipe$fills[[nm]]
      out[[nm]] <- num
    }
  }

  list(data = out, unseen = unseen, imputed = imputed)
}

# Coerce the outcome column the way the recipe says.
rf_prepare_target <- function(y, recipe) {
  if (recipe$task == "classification")
    factor(as.character(y), levels = recipe$y_levels) else as.numeric(y)
}

rf_engine <- function() {
  # Set RF_ENGINE=randomForest to override the default preference.
  forced <- Sys.getenv("RF_ENGINE", "")
  if (nzchar(forced)) {
    if (!requireNamespace(forced, quietly = TRUE))
      stop("RF_ENGINE is set to '", forced, "', which is not installed.")
    return(forced)
  }
  if (requireNamespace("ranger", quietly = TRUE)) return("ranger")
  if (requireNamespace("randomForest", quietly = TRUE)) return("randomForest")
  stop("Need either the 'ranger' or the 'randomForest' package. ",
       "Install with: install.packages('ranger')")
}
