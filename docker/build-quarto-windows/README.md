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

The Windows container provides significant performance improvements:
- **Traditional Windows build**: 45 minutes (including dependency installation)
- **Containerized build**: 10-15 minutes (dependencies pre-installed)
- **Container size**: ~4-5GB (larger than Linux due to Windows base)
- **Build phases**: 12 optimized phases with progress tracking

## Recent Improvements (2025)

- **CRITICAL FIX**: Replaced hanging Ghostscript direct download with reliable chocolatey installation (most stable for containers)
- Fixed PowerShell command syntax and error handling
- Updated to use PowerShell 7 consistently
- Improved dependency path handling after repository restructuring
- Enhanced testing with Windows-specific commands
- Fixed container testing to use PowerShell instead of bash
- Added comprehensive error checking and progress indicators
- Optimized cleanup procedures

## Build Phases

1. **PowerShell 7 Installation** - Modern PowerShell for better scripting
2. **Chocolatey Installation** - Package manager for Windows
3. **Quarto Installation** - Latest Quarto CLI (v1.7.31)
4. **Python 3.13 Installation** - Latest Python with full package support
5. **Python Package Installation** - All production requirements
6. **Ghostscript Installation** - PDF processing capabilities
7. **Inkscape Installation** - SVG to PDF conversion capability
8. **TeX Live Installation** - Complete LaTeX distribution for Windows
9. **R Installation** - R base with development packages
10. **R Package Installation** - All required R libraries
11. **R Package Verification** - Validation of successful installation
12. **Cleanup** - Temporary file removal and optimization

## Windows-Specific Considerations

- Uses Windows Server Core LTSC 2022 as base image
- PowerShell 7 for modern scripting capabilities
- Chocolatey for reliable package management
- Windows-style path handling (C:/ instead of /)
- Container testing uses PowerShell commands instead of bash
- Larger image size due to Windows base requirements

- **Traditional Windows build**: 45+ minutes
- **Containerized Windows build**: 10-15 minutes
- **Improvement**: 70-80% time reduction

## Known Limitations

- Larger image size compared to Linux container
- Windows containers require Windows host for optimal performance
- Some tools may have different behavior compared to Linux equivalents