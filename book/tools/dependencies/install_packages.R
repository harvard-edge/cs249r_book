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

required_packages <- c(
  "downlit",    # Required for code linking in Quarto
  "ggplot2",    # Visualization package
  "ggrepel",    # Visualization package (pinned below for R < 4.5)
  "knitr",      # Needed for Quarto rendering
  "png",        # PNG support
  "rmarkdown",  # Markdown rendering in R
  "tidyverse",  #
  "reshape2",   #
  "reticulate", #
  "rsvg",       #
  "viridis",    #
  "xml2",       # Required for XML/HTML processing
  "dplyr",      # Data manipulation (used in sustainable_ai.qmd)
  "grid"        # Grid graphics (used in hw_acceleration.qmd)
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    # Ncpus=1: avoid parallel compiles OOM'ing in Docker/Actions runners
    install.packages(pkg, Ncpus = 1L)
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
