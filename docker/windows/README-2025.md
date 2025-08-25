# Windows Server 2025 Support

This directory now supports building containers with both Windows Server 2022 (default) and Windows Server 2025 (experimental).

## Files

- `dockerfile` - Windows Server 2022 (LTSC 2022) - **Default**
- `dockerfile-2025` - Windows Server 2025 (LTSC 2025) - **Experimental**

## Usage

### GitHub Actions Workflow

The Windows container build workflow now includes a `windows_version` input parameter:

```yaml
windows_version:
  description: 'Windows Server version (2022 or 2025)'
  required: false
  default: '2022'
  type: choice
  options:
    - '2022'
    - '2025'
```

### Manual Workflow Trigger

1. Go to Actions → "🐳 Build Windows Container"
2. Click "Run workflow"
3. Select Windows version:
   - **2022** (default) - Uses stable Windows Server 2022 LTSC
   - **2025** (experimental) - Uses Windows Server 2025 LTSC

### Container Tags

- Windows Server 2022: `ghcr.io/harvard-edge/cs249r_book/quarto-windows:latest`
- Windows Server 2025: `ghcr.io/harvard-edge/cs249r_book/quarto-windows:latest-2025`

### Local Building

```powershell
# Windows Server 2022 (default)
docker build -f docker/windows/dockerfile -t mlsysbook-windows-2022 .

# Windows Server 2025 (experimental)
docker build -f docker/windows/dockerfile-2025 -t mlsysbook-windows-2025 .
```

## Differences

The only difference between the dockerfiles is the base image:

- **2022**: `FROM mcr.microsoft.com/windows/server:ltsc2022`
- **2025**: `FROM mcr.microsoft.com/windows/server:ltsc2025`

All other components (PowerShell, Chocolatey, Scoop, Quarto, Python, R, TeX Live, etc.) remain identical.

## Testing Status

- ✅ **Windows Server 2022**: Production ready, fully tested
- ⚠️ **Windows Server 2025**: Experimental, requires validation

## Migration Path

1. **Test Phase**: Use 2025 option to validate compatibility
2. **Validation**: Ensure all tools and dependencies work correctly
3. **Performance**: Compare build times and resource usage
4. **Switch**: Update default when 2025 is proven stable

## Known Considerations

Windows Server 2025 may have:
- Different PowerShell default versions
- Updated security policies
- Changed package manager behaviors
- New .NET Framework requirements
- Different performance characteristics

Test thoroughly before using in production builds.
