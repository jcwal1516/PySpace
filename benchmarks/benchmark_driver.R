#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7) {
  stop("usage: benchmark_driver.R R_REPO MODE INPUT RESULT METADATA REPETITIONS PARALLEL")
}

r_repo <- normalizePath(args[[1]], mustWork = TRUE)
mode <- args[[2]]
input_path <- args[[3]]
result_path <- args[[4]]
metadata_path <- args[[5]]
repetitions <- as.integer(args[[6]])
parallel_mode <- identical(args[[7]], "true")
batch_size <- as.integer(Sys.getenv("PYSPACE_BENCH_BATCH", "1"))
commit <- "94f0a0f9311e2ee2b406ebc45c84a4e93f2f38f8"

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("benchmark driver requires the R package jsonlite")
}
if (is.na(batch_size) || batch_size < 1) {
  stop("PYSPACE_BENCH_BATCH must be a positive integer")
}
actual_commit <- system2("git", c("-C", r_repo, "rev-parse", "HEAD"), stdout = TRUE)
dirty <- system2("git", c("-C", r_repo, "status", "--porcelain"), stdout = TRUE)
if (!identical(actual_commit[[1]], commit) || length(dirty) > 0) {
  stop("R benchmark source must be a pristine checkout of the pinned commit")
}

read_at_commit <- function(path) {
  sub("\r$", "", system2("git", c("-C", r_repo, "show", paste0(commit, ":", path)), stdout = TRUE))
}
eval(parse(text = read_at_commit("R/build_distributions.R")))
eval(parse(text = read_at_commit("R/utils.R")))

if (mode == "distribution") {
  frame <- read.csv(input_path, check.names = FALSE)
  ensemble <- strsplit(Sys.getenv("PYSPACE_BENCH_ENSEMBLE"), ",", fixed = TRUE)[[1]]
  operation <- function() build_dist(frame, ensemble, "all")
  result <- operation()
  samples <- replicate(
    repetitions,
    system.time(replicate(batch_size, operation(), simplify = FALSE))[["elapsed"]] / batch_size
  )
  write.csv(result, result_path, row.names = FALSE)
} else if (mode == "volume") {
  radii <- lapply(jsonlite::fromJSON(input_path, simplifyVector = FALSE), unlist)
  dimensions <- c(100, 100, 20)
  cluster <- NULL
  if (parallel_mode) {
    cluster <- parallel::makeCluster(2)
    parallel::clusterExport(cluster, c("calc_vol", "dimensions"), envir = environment())
    operation <- function() parallel::parLapply(cluster, radii, calc_vol, dims = dimensions)
  } else {
    operation <- function() calc_vols(radii, dimensions)
  }
  result <- operation()
  samples <- replicate(repetitions, system.time(operation())[["elapsed"]])
  if (!is.null(cluster)) {
    parallel::stopCluster(cluster)
  }
  jsonlite::write_json(unlist(result), result_path, auto_unbox = TRUE, digits = NA)
} else {
  stop(paste("unsupported mode", mode))
}

jsonlite::write_json(
  list(samples_seconds = as.numeric(samples), r_version = R.version.string),
  metadata_path,
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = NA
)
