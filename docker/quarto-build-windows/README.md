# Windows Quarto Build Container

This directory contains the Windows Docker container configuration for the MLSysBook build system.

## Purpose

The Windows container pre-installs all dependencies to eliminate the 30-45 minute setup time for Windows builds, reducing build times from 45 minutes to 5-10 minutes.

## Structure

```
docker/quarto-build-windows/
├── Dockerfile              # Container definition
├── README.md               # This file
├── .dockerignore          # Files to exclude from build
└── verify_r_packages.R    # R package verification script
```

## Container Contents

- **Base**: Windows Server Core LTSC 2022
- **TeX Live**: MikTeX distribution
- **R**: R-base with all required packages
- **Python**: Python 3.x with all requirements
- **Quarto**: Version 1.7.31
- **Tools**: Inkscape, Ghostscript, Git
- **Package Manager**: Chocolatey
- **Dependencies**: All from `tools/dependencies/`

## Build Process

The container is built and tested via GitHub Actions:

```bash
# Trigger Windows container build
gh workflow run build-windows-container.yml
```

## Usage

The container is used in the Windows containerized build workflow:

```yaml
runs-on: windows-latest
container:
  image: ghcr.io/harvard-edge/cs249r_book/quarto-build-windows:latest
```

## Key Differences from Linux Container

1. **Base Image**: Windows Server Core instead of Ubuntu
2. **Package Manager**: Chocolatey instead of apt-get
3. **Path Separators**: Backslashes instead of forward slashes
4. **TeX Distribution**: MikTeX instead of TeX Live
5. **Shell Commands**: PowerShell instead of bash

## Testing

The container includes verification steps for:
- Quarto functionality
- Python packages installation
- R packages installation
- TeX distribution
- Tool availability

## Registry

- **Registry**: GitHub Container Registry (ghcr.io)
- **Image**: `ghcr.io/harvard-edge/cs249r_book/quarto-build-windows`
- **Tags**: `latest`, `main`, `dev`, branch-specific tags
- **Size**: ~4-5GB (larger than Linux due to Windows base)

## Performance

- **Traditional Windows build**: 45+ minutes
- **Containerized Windows build**: 10-15 minutes
- **Improvement**: 70-80% time reduction

## Known Limitations

- Larger image size compared to Linux container
- Windows containers require Windows host for optimal performance
- Some tools may have different behavior compared to Linux equivalents