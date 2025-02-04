# List of required R packages
required_packages <- c(
  "downlit"  # Required for code linking in Quarto
  "ggplot2", 
  "knitr", 
  "rmarkdown", 
  "xml2", 
)

# Function to install missing packages
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "http://cran.rstudio.com")
  }
}

# Install each package if missing
invisible(sapply(required_packages, install_if_missing))

# Install TinyTeX separately (if needed for PDF output)
if (!requireNamespace("tinytex", quietly = TRUE)) {
  install.packages("tinytex", repos = "http://cran.rstudio.com")
  tinytex::install_tinytex()
}

# Print installed packages for debugging
cat("Installed R Packages:\n")
installed_pkgs <- installed.packages()[, "Package"]
print(installed_pkgs)
