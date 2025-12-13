# Containerized Build System

## Overview

This document describes the containerized build system for MLSysBook that significantly reduces build times from 45 minutes to 5-10 minutes for Linux builds.

## Architecture

### Container Strategy
- **Linux builds**: Use pre-built container with all dependencies
- **Windows builds**: Keep traditional approach (unchanged)
- **Container registry**: GitHub Container Registry (ghcr.io)

### Performance Benefits
```
Current Linux Build (45 minutes):
├── Install system packages (5-10 min)
├── Install TeX Live (15-20 min)
├── Install R packages (5-10 min)
├── Install Python packages (2-5 min)
├── Install Quarto (1-2 min)
└── Build content (5-10 min)

Containerized Linux Build (5-10 minutes):
├── Pull container (30 seconds)
├── Checkout code (30 seconds)
└── Build content (5-10 min)
```

## Files

### Core Files
- `docker/linux/Dockerfile` - A single Dockerfile for Linux builds.
- `docker/linux/README.md` - Linux container documentation
- `docker/linux/.dockerignore` - Build exclusions
- `docker/windows/Dockerfile` - A single Dockerfile for Windows builds.
- `docker/windows/README.md` - Windows container documentation
- `docker/windows/.dockerignore` - Build exclusions

### Container Lifecycle
1. **Build**: Weekly automatic rebuilds + manual triggers
   - Linux container: Sunday 12am
   - Windows container: Sunday 2am
2. **Storage**: GitHub Container Registry (ghcr.io)
3. **Usage**: Pulled fresh for each build job
4. **Cleanup**: GitHub manages old images automatically

## Usage

### Registry Paths
- **Linux Registry**: `ghcr.io/harvard-edge/cs249r_book/quarto-linux`
- **Windows Registry**: `ghcr.io/harvard-edge/cs249r_book/quarto-windows`

### Manual Builds
You can build the containers locally using these commands:
- **Linux**:
  ```bash
  docker build -f docker/linux/Dockerfile -t mlsysbook-linux .
  ```
- **Windows**:
  ```powershell
  docker build -f docker/windows/Dockerfile -t mlsysbook-windows .
  ```

### Manual Build Test
```bash
# Test containerized build
gh workflow run quarto-build-container.yml --field os=ubuntu-latest --field format=html
```

### Container Information
- **Linux Registry**: `ghcr.io/harvard-edge/cs249r_book/quarto-linux`
- **Windows Registry**: `ghcr.io/harvard-edge/cs249r_book/quarto-windows`
- **Tags**: `latest`, `main`, `dev`, branch-specific tags
- **Linux Size**: ~2-3GB (includes TeX Live, R, Python packages)
- **Windows Size**: ~4-5GB (includes Windows Server Core + dependencies)

## Workflow Integration

### Current Workflows
- `quarto-build-baremetal.yml` - Original workflow (brute force approach, legacy)
- `quarto-build-container.yml` - Containerized version (fast path, recommended)
- `build-linux-container.yml` - Linux container management
- `build-windows-container.yml` - Windows container management

### Migration Status
1. **✅ Phase 1**: Containerized builds tested and validated
2. **✅ Phase 2**: Performance significantly improved (45min → 5-10min)
3. **✅ Phase 3**: Container workflow is now the primary build method

## Container Contents

### Pre-installed Dependencies

#### Linux Container
- **System**: Ubuntu 22.04 with all required libraries
- **TeX Live**: Full distribution (texlive-full)
- **R**: R-base with all required packages
- **Python**: Python 3.13 with all requirements
- **Quarto**: Version 1.7.31
- **Tools**: Inkscape, Ghostscript, fonts

#### Windows Container
- **System**: Windows Server Core 2022
- **TeX Live**: MiKTeX distribution
- **R**: R-base with all required packages
- **Python**: Python 3.x with all requirements
- **Quarto**: Version 1.7.31
- **Tools**: Inkscape, Ghostscript, Chocolatey package manager

### Environment Variables
```bash
R_LIBS_USER=/usr/local/lib/R/library
QUARTO_LOG_LEVEL=INFO
PYTHONIOENCODING=utf-8
LANG=en_US.UTF-8
LC_ALL=en_US.UTF-8
```

## Troubleshooting

### Container Build Issues
1. Check container build logs in Actions
2. Verify dependency files are up to date
3. Test locally with `docker build -t test .`

### Build Issues
1. Check if container exists: `ghcr.io/harvard-edge/cs249r_book/quarto-linux:latest`
2. Verify container has all dependencies
3. Compare with traditional build logs

### Performance Issues
1. Monitor container pull times
2. Check disk space in container
3. Verify memory allocation

## Future Enhancements

### Potential Improvements
1. **Multi-stage builds** for smaller images
2. **Windows containers** for Windows builds
3. **Layer optimization** for faster pulls
4. **Parallel builds** for multiple formats

### Monitoring
- Container build frequency
- Build time improvements
- Error rates vs traditional builds
- Resource usage optimization

## Rollback Plan

If issues arise:
1. Keep original `quarto-build-baremetal.yml` as backup
2. Switch back to traditional builds immediately
3. Debug container issues separately
4. Re-enable when resolved

## Security

### Container Security
- Uses official Ubuntu base image
- Minimal attack surface
- Regular base image updates
- GitHub security scanning enabled

### Access Control
- Container registry access via GitHub Actions
- No external dependencies
- All builds run in isolated containers

### Building the Containers

To build the containers, use the standard `docker build` command:

```bash
# For Linux
docker build -f docker/linux/Dockerfile -t mlsysbook-linux .

# For Windows
docker build -f docker/windows/Dockerfile -t mlsysbook-windows .
```
