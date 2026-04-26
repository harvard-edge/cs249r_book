#!/usr/bin/env Rscript
# Library path and requireNamespace only. Do not source install_packages.R (re-runs install + tinytex).

# Docker/CI: R_LIBS_USER is ignored if the path does not exist, so we mirror install_packages.R
lib <- Sys.getenv("R_LIBS_USER", unset = NA_character_)
if (!is.na(lib) && nzchar(lib)) {
  dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(lib, .libPaths()))
}

a <- commandArgs()
fa <- a[grepl("^--file=", a)]
script_dir <- if (length(fa)) {
  normalizePath(dirname(sub("^--file=", "", fa[1])), mustWork = FALSE)
} else {
  "."
}
# Docker: /tmp; local: repo locations (verify lives under book/docker/linux/)
candidates <- c(
  if (file.exists("/tmp/required_r_packages.R")) "/tmp/required_r_packages.R" else NULL,
  file.path(script_dir, "required_r_packages.R"),
  normalizePath(file.path(script_dir, "..", "tools", "dependencies", "required_r_packages.R"), mustWork = FALSE)
)
candidates <- candidates[file.exists(candidates)]
rpf <- if (length(candidates)) candidates[1] else {
  stop("required_r_packages.R not found (checked /tmp, script dir, book/tools/dependencies).", call. = FALSE)
}
source(rpf, local = FALSE)

missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if(length(missing_packages) > 0) {
  cat('❌ Missing packages:', paste(missing_packages, collapse = ', '), '\n')
  quit(status = 1)
} else {
  cat('✅ All required R packages installed successfully\n')
}
