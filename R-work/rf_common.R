# Shared data preparation for train_rf.R and predict_rf.R.
#
# A model is only usable on new data if the new data is prepared *exactly* the
# way the training data was -- same columns, same factor levels in the same
# order, same treatment of missing values. So training records a "recipe"
# alongside the fitted forest, and prediction replays it.

MAX_CLASS_LEVELS <- 10
NA_LABEL         <- "(missing)"

rf_read_table <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
  # Tab-separated when the extension says so, comma otherwise.
  reader <- if (grepl("\\.(tsv|txt|tab)$", path, ignore.case = TRUE))
    read.delim else read.csv
  reader(path, stringsAsFactors = FALSE, check.names = TRUE,
         na.strings = c("NA", "", "NULL", "?", "."))
}

# Column names survive read.*() with check.names, which renames duplicates
# (a repeated `colPath` becomes colPath and colPath.1). Match the outcome
# case-insensitively so `Satisfied` and `satisfied` both work.
rf_resolve_target <- function(nms, target) {
  if (target %in% nms) return(target)
  hit <- which(tolower(nms) == tolower(target))
  if (length(hit) == 1) return(nms[hit])
  stop("Column '", target, "' not found. Columns are: ",
       paste(nms, collapse = ", "))
}

# Row identifiers carry no signal and a forest will happily memorise them.
rf_id_like <- function(dat, target) {
  vapply(names(dat), function(nm) {
    if (nm == target) return(FALSE)
    if (grepl("^(x|id|index|rowid|row|no|serial)$", tolower(nm))) return(TRUE)
    !is.numeric(dat[[nm]]) && length(unique(dat[[nm]])) == nrow(dat)
  }, logical(1))
}

rf_is_categorical <- function(col) {
  obs <- col[!is.na(col)]
  if (is.character(col) || is.logical(col) || is.factor(col)) return(TRUE)
  # Small whole-number codes are survey scales, not counts.
  is.numeric(col) && length(unique(obs)) <= MAX_CLASS_LEVELS &&
    all(obs == round(obs))
}

# Two categorical columns are redundant when their levels map one-to-one --
# the same variable encoded twice (DCDD/DNDD vs C/N, say). A forest splits
# importance between such copies, making each look half as important as the
# single underlying variable really is, so keep only the first.
rf_redundant_pairs <- function(dat, cols) {
  drop <- character(); keep_for <- character()
  for (i in seq_along(cols)) {
    if (cols[i] %in% drop) next
    for (j in seq_along(cols)) {
      if (j <= i || cols[j] %in% drop) next
      a <- as.character(dat[[cols[i]]]); b <- as.character(dat[[cols[j]]])
      ok <- !is.na(a) & !is.na(b)
      if (!any(ok)) next
      tab <- table(a[ok], b[ok])
      # A bijection: exactly one non-zero cell in every row and every column.
      if (nrow(tab) == ncol(tab) &&
          all(rowSums(tab > 0) == 1) && all(colSums(tab > 0) == 1)) {
        drop <- c(drop, cols[j]); keep_for <- c(keep_for, cols[i])
      }
    }
  }
  list(drop = drop, kept = keep_for)
}

# Learn the preparation rules from the training data.
#
# na_as_level: keep missing categorical values as their own "(missing)" level
# instead of imputing them to the most common one. Better when a column is
# missing often -- imputing 11% of a column to its mode invents data, and
# "not recorded" is frequently informative in survey data.
rf_build_recipe <- function(raw, target, na_as_level = TRUE) {
  target <- rf_resolve_target(names(raw), target)

  drop_id <- names(raw)[rf_id_like(raw, target)]
  cand    <- setdiff(names(raw), c(target, drop_id))

  cat_cols <- cand[vapply(cand, function(nm) rf_is_categorical(raw[[nm]]), logical(1))]
  red      <- rf_redundant_pairs(raw, cat_cols)
  cand     <- setdiff(cand, red$drop)

  types <- list(); levs <- list(); fills <- list(); const <- character()

  for (nm in cand) {
    col <- raw[[nm]]
    obs <- col[!is.na(col)]
    if (length(unique(obs)) < 2) { const <- c(const, nm); next }
    if (rf_is_categorical(col)) {
      chr <- as.character(col)
      if (na_as_level && any(is.na(chr))) chr[is.na(chr)] <- NA_LABEL
      f <- factor(chr)
      types[[nm]] <- "factor"
      levs[[nm]]  <- levels(f)
      # Fill value is only used when na_as_level is off, or for a value the
      # training data never contained.
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
    target      = target,
    task        = task,
    y_levels    = if (task == "classification") levels(factor(y)) else NULL,
    predictors  = setdiff(cand, const),
    types       = types,
    levels      = levs,
    fills       = fills,
    na_as_level = na_as_level,
    dropped_id        = drop_id,
    dropped_constant  = const,
    dropped_redundant = red$drop,
    redundant_with    = red$kept
  )
}

# Replay the recipe on any data frame -- training or new.
rf_apply_recipe <- function(raw, recipe) {
  missing_cols <- setdiff(recipe$predictors, names(raw))
  if (length(missing_cols))
    stop("Data is missing predictor column(s) the model needs: ",
         paste(missing_cols, collapse = ", "))

  out <- data.frame(row.names = seq_len(nrow(raw)))
  unseen <- list(); imputed <- 0L; na_kept <- 0L

  for (nm in recipe$predictors) {
    col <- raw[[nm]]
    if (identical(recipe$types[[nm]], "factor")) {
      chr <- as.character(col)
      if (recipe$na_as_level && NA_LABEL %in% recipe$levels[[nm]]) {
        na_kept <- na_kept + sum(is.na(chr))
        chr[is.na(chr)] <- NA_LABEL
      }
      # A level the forest never saw has no split to follow; treat it as
      # missing rather than erroring out on one stray row.
      bad <- !is.na(chr) & !(chr %in% recipe$levels[[nm]])
      if (any(bad)) { unseen[[nm]] <- sort(unique(chr[bad])); chr[bad] <- NA }
      miss <- is.na(chr)
      imputed <- imputed + sum(miss)
      chr[miss] <- recipe$fills[[nm]]
      out[[nm]] <- factor(chr, levels = recipe$levels[[nm]])
    } else {
      num  <- suppressWarnings(as.numeric(col))
      miss <- is.na(num)
      imputed <- imputed + sum(miss)
      num[miss] <- recipe$fills[[nm]]
      out[[nm]] <- num
    }
  }

  list(data = out, unseen = unseen, imputed = imputed, na_kept = na_kept)
}

rf_prepare_target <- function(y, recipe) {
  if (recipe$task == "classification")
    factor(as.character(y), levels = recipe$y_levels) else as.numeric(y)
}

# cforest first: with categorical predictors that have different numbers of
# levels, its variable importance is the one that is not biased toward the
# many-level columns. RF_ENGINE overrides.
rf_engine <- function() {
  forced <- Sys.getenv("RF_ENGINE", "")
  if (nzchar(forced)) {
    pkg <- if (forced == "cforest") "party" else forced
    if (!requireNamespace(pkg, quietly = TRUE))
      stop("RF_ENGINE is set to '", forced, "', but package '", pkg,
           "' is not installed.")
    return(forced)
  }
  if (requireNamespace("party", quietly = TRUE))        return("cforest")
  if (requireNamespace("ranger", quietly = TRUE))       return("ranger")
  if (requireNamespace("randomForest", quietly = TRUE)) return("randomForest")
  stop("Need one of the 'party', 'ranger' or 'randomForest' packages. ",
       "Install with: install.packages('party')")
}

rf_engine_pkg <- function(engine)
  if (engine == "cforest") "party" else engine

# party returns per-row probabilities as a list of 1-row matrices.
rf_bind_probs <- function(lst, lvls) {
  m <- do.call(rbind, lapply(lst, as.vector))
  colnames(m) <- lvls
  m
}
