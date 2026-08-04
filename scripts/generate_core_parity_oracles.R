#!/usr/bin/env Rscript

upstream_commit <- "94f0a0f9311e2ee2b406ebc45c84a4e93f2f38f8"
cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[[1]]) else "scripts/generate_core_parity_oracles.R"
repo_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)
trailing_args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default = NULL) {
  idx <- which(trailing_args == flag)
  if (length(idx) > 0 && idx[[1]] < length(trailing_args)) {
    return(trailing_args[[idx[[1]] + 1]])
  }
  inline <- grep(paste0("^", flag, "="), trailing_args, value = TRUE)
  if (length(inline) > 0) {
    return(sub(paste0("^", flag, "="), "", inline[[1]]))
  }
  default
}
write_fixtures <- "--write-fixtures" %in% trailing_args
fixture_out_dir <- file.path(repo_root, "src", "pyspace", "data", "parity", "oracles")
default_out_dir <- if (write_fixtures) fixture_out_dir else tempfile("pyspace-r-oracle-")
out_dir <- arg_value("--out-dir", default_out_dir)
r_repo <- arg_value("--r-repo", Sys.getenv("SPACE_R_REPO", ""))

required_packages <- c(
  "abind", "doParallel", "dplyr", "FNN", "foreach", "igraph", "jsonlite",
  "memuse", "plyr", "purrr", "stringr", "tidyr"
)
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  message("Cannot generate live R oracle; missing packages: ", paste(missing_packages, collapse = ", "))
  quit(status = 1)
}

if (!nzchar(r_repo) || !dir.exists(r_repo)) {
  message("Cannot generate live R oracle; pass --r-repo with a pristine checkout of the pinned commit.")
  quit(status = 1)
}
actual_commit <- system2("git", c("-C", r_repo, "rev-parse", "HEAD"), stdout = TRUE)
dirty_paths <- system2("git", c("-C", r_repo, "status", "--porcelain"), stdout = TRUE)
if (!identical(actual_commit[[1]], upstream_commit)) {
  message("Cannot generate live R oracle; checkout is at ", actual_commit[[1]], " not ", upstream_commit)
  quit(status = 1)
}
if (length(dirty_paths) > 0) {
  message("Cannot generate live R oracle from a dirty checkout: ", paste(dirty_paths, collapse = ", "))
  quit(status = 1)
}

read_at_commit <- function(path) {
  sub("\r$", "", system2("git", c("-C", r_repo, "show", paste0(upstream_commit, ":", path)), stdout = TRUE))
}

eval(parse(text = read_at_commit("R/census_image_table.R")))
eval(parse(text = read_at_commit("R/build_distributions.R")))
eval(parse(text = read_at_commit("R/measure_mutual_info.R")))
eval(parse(text = read_at_commit("R/utils.R")))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
random_plans <- jsonlite::fromJSON(
  file.path(repo_root, "src", "pyspace", "data", "parity", "random_plans.json"),
  simplifyVector = FALSE
)

fixture_names <- c(
  "utils.json",
  "round_column.csv",
  "build_dist_2d.csv",
  "smooth_dist_1d.csv",
  "smooth_dist_2d.csv",
  "smooth_dist_2d_freq.csv",
  "summarize_patches.csv",
  "random_census.csv",
  "census_table_r_style.csv",
  "cismi_1.1.csv",
  "transmi_1.1.csv"
)

writeLines(
  jsonlite::toJSON(
    list(
      upstream_repo = "eschrom/SPACE",
      upstream_commit = upstream_commit,
      numeric_tolerance = 1e-10,
      fixtures = fixture_names
    ),
    auto_unbox = TRUE,
    pretty = TRUE
  ),
  file.path(out_dir, "metadata.json")
)

utils_out <- list(
  calc_vol_ellipsoid = calc_vol(c(2, 2, 1), c(20, 20, 5)),
  calc_vols = calc_vols(list(c(1, 1, 1), c(2, 2, 1)), c(20, 20, 5)),
  total_comp_bins_custom = total_comp_bins(
    dimension = 2,
    bins_per_var = 3,
    min_per_var = c(80, 0, 0),
    max_per_var = c(100, 50, 100)
  ),
  entropy_2d = entropy(matrix(c(1, 1, 1, 2, 2, 1, 2, 2, 2, 2), ncol = 2, byrow = TRUE), 2)
)
jsonlite::write_json(utils_out, file.path(out_dir, "utils.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)

round_values <- c(0, 1, 2.9, 4.1, 5)
write.csv(
  data.frame(
    bin_id = round_column(round_values, 0, 5, 4, FALSE),
    bin_value = round_column(round_values, 0, 5, 4, TRUE)
  ),
  file.path(out_dir, "round_column.csv"),
  row.names = FALSE
)

dist_census <- data.frame(O1.1 = c(0, 0, 1, 1, 1), S1.1 = c(10, 20, 10, 20, 20), check.names = FALSE)
write.csv(build_dist(dist_census, c("O1.1", "S1.1"), "all"), file.path(out_dir, "build_dist_2d.csv"), row.names = FALSE)

joint_dist <- data.frame(O1.1 = c(0, 1), freq = c(1, 2), check.names = FALSE)
min_max <- data.frame(O1.1 = c(0, 3), check.names = FALSE)
write.csv(smooth_dist(joint_dist, 4, min_max, full_dist = TRUE),
          file.path(out_dir, "smooth_dist_1d.csv"), row.names = FALSE)

joint_2d <- data.frame(O1.1 = c(0, 1), S1.1 = c(10, 20), freq = c(1, 2), check.names = FALSE)
min_max_2d <- data.frame(O1.1 = c(0, 1), S1.1 = c(10, 30), check.names = FALSE)
write.csv(smooth_dist(joint_2d, c(2, 3), min_max_2d, full_dist = TRUE),
          file.path(out_dir, "smooth_dist_2d.csv"), row.names = FALSE)
write.csv(data.frame(freq = smooth_dist(joint_2d, c(2, 3), min_max_2d, full_dist = FALSE)),
          file.path(out_dir, "smooth_dist_2d_freq.csv"), row.names = FALSE)

patches <- list(
  O1 = data.frame(
    Area = c(2, 1, 0, 3, 1),
    O1 = c(1, 2, 0, 1, 2),
    S1.1 = c(10, 9, 0, 12, 1),
    Nbhd = c(1, 1, 2, 3, 3),
    check.names = FALSE
  )
)
osp <- list(O1 = matrix(c(1, 0), nrow = 2, ncol = 1, dimnames = list(c("O1.1", "O1.2"), "S1.1")))
write.csv(summarize_patches(patches, osp), file.path(out_dir, "summarize_patches.csv"), row.names = FALSE)

stable_random_patches <- list(
  O1 = data.frame(
    Area = c(1, 2, 3, 1, 2, 3),
    O1 = c(1, 1, 1, 1, 1, 1),
    S1.1 = c(2, 4, 6, 2, 4, 6),
    Nbhd = c(1, 1, 1, 2, 2, 2),
    check.names = FALSE
  )
)
set.seed(1)
write.csv(random_census(stable_random_patches, osp), file.path(out_dir, "random_census.csv"), row.names = FALSE)

object_table <- data.frame(
  X = c(0, 1, 4, 100, 110, 120, 130, 140, 150, 160, 170),
  Y = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
  Z = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
  Object = c(9, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11)
)
set.seed(1)
census_out <- census_table(object_table, seed_points = 9, radii = c(1.1), sample_size = c(1))
write.csv(census_out[[1]], file.path(out_dir, "census_table_r_style.csv"), row.names = FALSE)

make_cismi_inputs <- function() {
  rows <- vector("list", 40)
  patch_rows_1 <- vector("list", 40)
  patch_rows_2 <- vector("list", 40)
  for (nbhd in 1:40) {
    rows[[nbhd]] <- data.frame(O1.1 = 100, O2.1 = 100, X = 0, Y = 0, Z = 0, Radius = 1.1, check.names = FALSE)
    patch_rows_1[[nbhd]] <- data.frame(Area = 1, O1 = 1, Nbhd = nbhd)
    patch_rows_2[[nbhd]] <- data.frame(Area = 1, O2 = 1, Nbhd = nbhd)
  }
  list(
    census = dplyr::bind_rows(rows),
    patch_list = list("1.1" = list(O1 = dplyr::bind_rows(patch_rows_1), O2 = dplyr::bind_rows(patch_rows_2)))
  )
}

set.seed(1)
cismi_inputs <- make_cismi_inputs()
cismi <- measure_cisMI(
  cismi_inputs$census,
  cismi_inputs$patch_list,
  depth = 2,
  radii = c(1.1),
  bootstraps = 2,
  max_bins = 5,
  cores = 1
)
write.csv(cismi[["1.1"]], file.path(out_dir, "cismi_1.1.csv"), row.names = FALSE)

make_transmi_census <- function(pattern) {
  if (pattern == 1) {
    o11 <- c(rep(20, 20), rep(80, 20))
    s11 <- c(rep(1, 20), rep(4, 20))
  } else if (pattern == 2) {
    o11 <- c(rep(20, 10), rep(80, 30))
    s11 <- c(rep(1, 30), rep(4, 10))
  } else if (pattern == 3) {
    o11 <- rep(c(20, 80), 20)
    s11 <- rep(c(1, 4), 20)
  } else {
    o11 <- rep(c(20, 20, 80, 80), 10)
    s11 <- rep(c(1, 4, 1, 4), 10)
  }
  data.frame(
    O1.1 = o11,
    O1.2 = 100 - o11,
    S1.1 = s11,
    O1.1_S1.1 = o11 * s11 / 100,
    X = 0,
    Y = 0,
    Z = 0,
    Radius = 1.1,
    check.names = FALSE
  )
}

permutation_steps <- random_plans$transmi_pair_permutation_steps
permutation_call <- 0L
sample <- function(x, size, replace = FALSE, prob = NULL) {
  if (length(x) == 1L && x == 6L && missing(size) && !replace && is.null(prob)) {
    permutation_call <<- permutation_call + 1L
    step <- permutation_steps[[((permutation_call - 1L) %% length(permutation_steps)) + 1L]]
    return(unlist(step) + 1L)
  }
  if (missing(size)) {
    return(base::sample(x, replace = replace, prob = prob))
  }
  base::sample(x, size = size, replace = replace, prob = prob)
}
transmi <- measure_transMI(
  censuses = list(make_transmi_census(1), make_transmi_census(2), make_transmi_census(3), make_transmi_census(4)),
  groups = data.frame(Status = c("A", "A", "A", "A")),
  depth = 2,
  radii = c(1.1),
  bootstraps = 2,
  max_bins = 5,
  cores = 1
)
rm(sample)
write.csv(transmi[["1.1"]], file.path(out_dir, "transmi_1.1.csv"), row.names = FALSE)

message("Wrote R oracle fixtures for SPACE commit ", upstream_commit)
