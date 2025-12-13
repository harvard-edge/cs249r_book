# Windows Quarto Build Container

This directory contains the Windows Server 2022 container configuration for building the MLSysBook with Quarto.

## 🐳 Container Features

- **Base Image**: Windows Server 2022 LTSC
- **PowerShell**: 7.4.1 (ZIP install, container-safe)
- **Quarto**: 1.7.31 (ZIP install)
- **Python**: 3.13.1 + production dependencies
- **TeX Live**: 2025 snapshot with required packages
- **R**: 4.3.2 + R Markdown packages
- **Graphics**: Ghostscript + Inkscape (via Chocolatey)

## 🔧 Key Fixes Applied

### 1. PowerShell 7 Path Issues
- **Problem**: Using `pwsh` shorthand can fail in containers
- **Fix**: Use full path `C:\Program Files\PowerShell\7\pwsh.exe`

### 2. TeX Live Installation
- **Problem**: `Start-Process` without `-NoNewWindow` can hang
- **Fix**: Added `-NoNewWindow` flag for container compatibility
- **Problem**: Comments in `tl_packages` file
- **Fix**: Filter out comment lines when installing packages

### 3. TikZ Test Document
- **Problem**: Complex here-string with backticks
- **Fix**: Simplified to standard multi-line string

### 4. Package Installation
- **Problem**: Silent failures in package installation
- **Fix**: Added verbose output and better error handling

## 🚀 Building the Container

### Prerequisites
- Windows Docker Desktop or Windows Server with Docker
- At least 8GB RAM available for Docker
- 20GB+ free disk space

## Local Build
To build the Windows container locally, run the following command from the repository root:
```powershell
docker build -f docker/windows/Dockerfile -t mlsysbook-windows .
```

### Testing
To test the Dockerfile before building, you can use the provided PowerShell script:
```powershell
./docker/windows/test_dockerfile.ps1
```

## Workflow
The container is built and pushed to the GitHub Container Registry via the [`.github/workflows/build-windows-container.yml`](../../.github/workflows/build-windows-container.yml) workflow.
This workflow is triggered manually or on a weekly schedule.

## Notes
- Building the Windows container can take a significant amount of time (often over 2 hours).
- The image is large due to the comprehensive set of pre-installed dependencies.

## 📋 Build Phases

1. **Base Setup**: Directories, environment variables
2. **PowerShell 7**: ZIP installation (container-safe)
3. **Chocolatey**: Package manager installation
4. **Dependencies**: Copy required files
5. **Quarto**: ZIP installation with PATH setup
6. **Python**: 3.13.1 + production requirements
7. **Graphics**: Ghostscript + Inkscape
8. **TeX Live**: 2025 snapshot + packages
9. **R**: 4.3.2 + R Markdown packages
10. **Cleanup**: Remove temporary files

## 🔍 Verification Steps

The container includes comprehensive verification:

- **PowerShell 7**: Version check
- **Quarto**: Version and command availability
- **Python**: Version and pip functionality
- **TeX Live**: Package verification with `kpsewhich`
- **Fonts**: Helvetica font files verification
- **TikZ**: Smoke test with PDF generation
- **R**: Package installation verification

## ⚠️ Common Issues & Solutions

### 1. Build Timeouts
- **Cause**: Large downloads (TeX Live, Python packages)
- **Solution**: Increased timeout values in Dockerfile

### 2. PATH Issues
- **Cause**: Windows PATH not properly updated
- **Solution**: Explicit PATH manipulation with regex escaping

### 3. Package Installation Failures
- **Cause**: Network issues or missing dependencies
- **Solution**: Added verbose output and error checking

### 4. Memory Issues
- **Cause**: TeX Live installation requires significant memory
- **Solution**: Use `scheme-infraonly` for minimal installation

## 🧪 Testing

### Run Container
```powershell
docker run -it mlsysbook-windows pwsh
```

### Test Quarto
```powershell
quarto --version
quarto check
```

### Test Python
```powershell
python --version
python -c "import nltk; print('NLTK available')"
```

### Test R
```powershell
R --version
Rscript -e "library(rmarkdown); print('R Markdown available')"
```

### Test TeX Live
```powershell
lualatex --version
kpsewhich pgf.sty
```

## 📊 Performance Notes

- **Build Time**: ~45-60 minutes on typical hardware
- **Image Size**: ~8-12GB (includes TeX Live, R, Python)
- **Memory Usage**: 4-6GB during build, 2-3GB runtime
- **Disk Space**: 15-20GB for build cache

## 🔧 Troubleshooting

### Build Fails on TeX Live
```powershell
# Check available memory
docker system df
docker system prune -f
```

### PowerShell Issues
```powershell
# Verify PowerShell 7 installation
docker run mlsysbook-windows pwsh -Command "Get-Host"
```

### Package Installation Issues
```powershell
# Check Chocolatey installation
docker run mlsysbook-windows choco --version
```

## 📝 Maintenance

### Updating Dependencies
1. Update version numbers in Dockerfile
2. Test with validation script
3. Rebuild and verify all components

### Adding New Packages
1. Add to appropriate phase in Dockerfile
2. Update verification steps
3. Test thoroughly

### Security Updates
- Regularly update base image
- Monitor for CVE reports
- Update package versions as needed
