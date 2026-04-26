# ==============================================================
# R Package Installation Script for Quarto GitHub Actions
#
# This script installs required R packages for rendering Quarto (.qmd) files.
# If you need to add a new package, follow the instructions below.
# ==============================================================

# Docker/CI: R_LIBS_USER is ignored at startup if the path does not exist, so we ensure
# a single target library and pin CRAN (HTTPS) before any install.
lib <- Sys.getenv("R_LIBS_USER", unset = NA_character_)
if (!is.na(lib) && nzchar(lib)) {
  dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(lib, .libPaths()))
}
options(repos = c(CRAN = "https://cloud.r-project.org"))

# required_packages (single list in required_r_packages.R; Docker: /tmp/required_r_packages.R)
a <- commandArgs()
fa <- a[grepl("^--file=", a)]
script_dir <- if (length(fa)) {
  normalizePath(dirname(sub("^--file=", "", fa[1])), mustWork = FALSE)
} else {
  "."
}
rpf <- if (file.exists("/tmp/required_r_packages.R")) {
  "/tmp/required_r_packages.R"
} else {
  file.path(script_dir, "required_r_packages.R")
}
if (!file.exists(rpf)) {
  stop("required_r_packages.R not found: ", rpf, call. = FALSE)
}
source(rpf, local = FALSE) # defines required_packages

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    # Ncpus=1: avoid parallel compiles OOM'ing in Docker/Actions runners
    install.packages(pkg, Ncpus = 1L)
  }
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Failed to install or load: ", pkg, call. = FALSE)
  }
}

invisible(sapply(required_packages, install_if_missing))

if (!requireNamespace("tinytex", quietly = TRUE)) {
  install.packages("tinytex", Ncpus = 1L)
  tinytex::install_tinytex()
}

cat("\n✅ Installed R Packages:\n")
installed_pkgs <- installed.packages()[, "Package"]
print(installed_pkgs)
