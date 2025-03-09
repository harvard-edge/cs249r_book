# ==============================================================
# R Package Installation Script for Quarto GitHub Actions
#
# This script installs required R packages for rendering Quarto (.qmd) files.
# If you need to add a new package, follow the instructions below.
# ==============================================================

required_packages <- c(
  "downlit",    # Required for code linking in Quarto
  "ggplot2",    # Visualization package
  "ggrepel",    # Visualization package
  "knitr",      # Needed for Quarto rendering
  "rmarkdown",  # Markdown rendering in R
  "tidyverse",  # 
  "reshape2",   #
  "rsvg",       #
  "viridis",    #
  "xml2"        # Required for XML/HTML processing
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "http://cran.rstudio.com")
  }
}

invisible(sapply(required_packages, install_if_missing))

if (!requireNamespace("tinytex", quietly = TRUE)) {
  install.packages("tinytex", repos = "http://cran.rstudio.com")
  tinytex::install_tinytex()
}

cat("\n✅ Installed R Packages:\n")
installed_pkgs <- installed.packages()[, "Package"]
print(installed_pkgs)
