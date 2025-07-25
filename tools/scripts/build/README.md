# Build Scripts

Scripts for building, cleaning, and development workflows.

## Scripts

- **`clean.sh`** - Comprehensive cleanup script (build artifacts, caches, temp files)
- **`standardize_sources.sh`** - Standardize source file formatting  
- **`generate_stats.py`** - Generate statistics about the Quarto project

## Quick Usage

```bash
# Clean all build artifacts
./clean.sh

# Deep clean including caches and virtual environments  
./clean.sh --deep

# Preview what would be cleaned
./clean.sh --dry-run

# Generate project statistics
python generate_stats.py
``` 