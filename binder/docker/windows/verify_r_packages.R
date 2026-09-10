#!/usr/bin/env Rscript

# Verify R package installation
source('C:/temp/install_packages.R')

missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if(length(missing_packages) > 0) {
  cat('❌ Missing packages:', paste(missing_packages, collapse = ', '), '\n')
  quit(status = 1)
} else {
  cat('✅ All required R packages installed successfully\n')
}
