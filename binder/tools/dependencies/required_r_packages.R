# Shared list for install_packages.R and verify_r_packages.R (single source of truth).
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
