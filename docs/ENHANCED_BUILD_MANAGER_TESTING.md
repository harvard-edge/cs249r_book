# Enhanced Build Manager Testing Plan

## Overview

The enhanced build manager provides intelligent container-based builds with fallback to traditional builds. This document outlines the testing strategy for the feature branch.

## Architecture

### Smart Container Management
1. **Container Health Check**: Verifies if containers exist and are up-to-date
2. **Conditional Building**: Only rebuilds containers when needed
3. **Intelligent Routing**: Uses fast containers when available, traditional builds otherwise
4. **Comprehensive Reporting**: Clear visibility into which strategy was used

### Performance Benefits
- **Fast path**: 5-10 minutes (with containers)
- **Traditional path**: 45 minutes (without containers)
- **Graceful degradation**: Always works even if containers fail

## Testing Strategy

### Phase 1: Container Build Testing
Test that containers build correctly from feature branch:

```bash
# Test Linux container build
gh workflow run build-linux-container.yml --ref feature/enhanced-build-manager

# Test Windows container build  
gh workflow run build-windows-container.yml --ref feature/enhanced-build-manager
```

### Phase 2: Enhanced Manager Testing
Test the enhanced manager with different scenarios:

```bash
# Test 1: Full enhanced manager with container building
gh workflow run build-manager-enhanced.yml \
  --ref feature/enhanced-build-manager \
  --field force_container_rebuild=true \
  --field build_format=html

# Test 2: Enhanced manager using existing containers
gh workflow run build-manager-enhanced.yml \
  --ref feature/enhanced-build-manager \
  --field force_container_rebuild=false \
  --field build_format=html

# Test 3: Test with specific branch
gh workflow run build-manager-enhanced.yml \
  --ref feature/enhanced-build-manager \
  --field test_branch=feature/enhanced-build-manager \
  --field build_format=html
```

### Phase 3: Individual Workflow Testing
Test individual workflows still work:

```bash
# Test container-based build directly
gh workflow run quarto-build-container.yml \
  --ref feature/enhanced-build-manager \
  --field os=ubuntu-latest \
  --field format=html

# Test traditional build directly
gh workflow run quarto-build.yml \
  --ref feature/enhanced-build-manager \
  --field os=ubuntu-latest \
  --field format=html
```

## Expected Outcomes

### Successful Container Path
1. Container health check finds containers available
2. Skips container building (unless forced)
3. Uses `quarto-build-container.yml` for fast builds
4. Completes in 5-10 minutes

### Successful Traditional Path
1. Container health check finds containers unavailable
2. Skips container building 
3. Uses `quarto-build.yml` for traditional builds
4. Completes in ~45 minutes

### Container Building Path
1. Container health check determines rebuild needed
2. Builds containers (may take 20-30 minutes first time)
3. Uses newly built containers for fast builds
4. Future runs are fast (5-10 minutes)

## Container Naming Convention

The enhanced manager uses project-based naming:

- **Linux Container**: `ghcr.io/harvard-edge/cs249r_book/mlsysbook-build-linux:latest`
- **Windows Container**: `ghcr.io/harvard-edge/cs249r_book/mlsysbook-build-windows:latest`

This clearly identifies containers as belonging to the ML Systems book project and scales well for future projects.

## Safety Features

### Branch Isolation
- Feature branch won't trigger automatic builds on main/dev
- Manual testing only via `workflow_dispatch`
- No impact on production workflows

### Fallback Protection
- Always falls back to working traditional builds
- Never breaks existing functionality
- Comprehensive error reporting

### Consistency Enforcement
- Single source of truth for container names
- Standardized container references across workflows
- Prevents the naming mismatches we just fixed

## Success Criteria

### Must Have
- [ ] Containers build successfully from feature branch
- [ ] Enhanced manager completes without errors
- [ ] Traditional builds still work as fallback
- [ ] Clear reporting of which strategy was used

### Nice to Have
- [ ] Performance improvement visible in build times
- [ ] Container reuse works (second run much faster)
- [ ] Windows containers also work (when implemented)

## Migration Plan

Once testing is successful:

1. **Validate**: All tests pass on feature branch
2. **Review**: Code review and documentation update
3. **Merge**: Merge to dev branch for broader testing
4. **Monitor**: Watch dev branch builds for any issues
5. **Deploy**: Enable for main branch after dev validation

## Rollback Plan

If issues arise:
1. **Immediate**: Use manual `workflow_dispatch` with traditional builds
2. **Short-term**: Revert to original `build-manager.yml`
3. **Long-term**: Fix issues on feature branch and re-test

## Key Benefits

### For Development
- **Faster iteration**: 5-10 min builds instead of 45 min
- **Better reliability**: Fallback ensures builds always work
- **Clear feedback**: Know immediately which strategy was used

### For Production
- **Consistency**: Single manager orchestrates all builds
- **Performance**: Dramatic build time reduction
- **Maintenance**: Centralized container management
