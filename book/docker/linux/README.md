# Quarto Build Container

This directory contains the Docker container configuration for the MLSysBook build system.

## Purpose

The container pre-installs all dependencies to eliminate the 30-45 minute setup time for Linux builds, reducing build times from 45 minutes to 5-10 minutes.

## Structure

```
docker/quarto-build/
├── Dockerfile          # Container definition
├── README.md          # This file
└── .dockerignore      # Files to exclude from build
```

## Container Contents

- **Base**: Ubuntu 22.04
- **TeX Live**: Full distribution (texlive-full)
- **R**: R-base with all required packages
- **Python**: Python 3.13 with all requirements
- **Quarto**: Version 1.7.31
- **Tools**: Inkscape, Ghostscript, fonts
- **Dependencies**: All from `tools/dependencies/`

## Build Process

The container is built and tested via GitHub Actions:

```bash
# Trigger container build
gh workflow run build-container.yml
```

## Usage

The container is used in the containerized build workflow:

```yaml
container:
  image: ghcr.io/harvard-edge/cs249r_book/quarto-build:latest
  options: --user root
```

## Testing

The container build includes 17 comprehensive tests:

1. Quarto functionality
2. Python packages (all from requirements.txt)
3. R packages (all from install_packages.R)
4. TeX Live and LaTeX engines
5. Inkscape SVG to PDF conversion
6. Ghostscript PDF compression
7. Fonts and graphics libraries
8. Quarto render test
9. TikZ compilation test
10. System resources check
11. Network connectivity
12. Book structure compatibility
13. Quarto configuration files
14. Dependencies files accessibility
15. Quarto check (same as workflow)
16. Actual build process simulation
17. Memory and disk space verification

## Registry

- **Registry**: GitHub Container Registry (ghcr.io)
- **Image**: `ghcr.io/harvard-edge/cs249r_book/quarto-build`
- **Tags**: `latest`, `main`, `dev`, branch-specific tags
- **Size**: ~2-3GB (includes TeX Live, R, Python packages)

## Performance

The container reduces build times significantly:
- **Traditional Linux build**: 45 minutes (including dependency installation)
- **Containerized build**: 5-10 minutes (dependencies pre-installed)
- **Container size**: ~2-3GB (optimized with multi-layer cleanup)
- **Build phases**: 11 optimized phases with progress tracking

## Recent Improvements (2025)

- Fixed dependency path issues after repository restructuring
- Improved error handling and progress tracking
- Optimized TeX Live package installation
- Enhanced cleanup procedures for smaller image size
- Added comprehensive testing (17 test scenarios)
- Fixed PATH environment variables for all tools

## Build Phases

1. **System Dependencies** - Core Ubuntu packages and libraries
2. **Inkscape Installation** - SVG to PDF conversion capability
3. **Quarto Installation** - Latest Quarto CLI (v1.7.31)
4. **TeX Live Installation** - Complete LaTeX distribution
5. **Ghostscript Installation** - PDF processing capabilities
6. **R Installation** - R base and development packages
7. **Python Installation** - Python 3 with pip
8. **Python Packages** - All production requirements
9. **R Packages** - All required R libraries
10. **R Package Verification** - Validation of successful installation
11. **Comprehensive Cleanup** - Size optimization and cache clearing

- **Traditional build**: 45 minutes
- **Containerized build**: 5-10 minutes
- **Improvement**: 80-90% time reduction
