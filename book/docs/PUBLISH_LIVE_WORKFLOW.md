# 🚀 Publish Live Workflow

## Overview

The publish-live workflow is designed to handle the publication of your Machine Learning Systems textbook with proper PDF management. The key innovation is that **the PDF is available for download but NOT committed to git**.

## 🔄 Workflow Steps

### 1. Manual Trigger
- Go to GitHub Actions → "🚀 Publish Live"
- Fill in the required fields:
  - **Description**: What you're publishing
  - **Release Type**: patch/minor/major
  - **Dev Commit**: Specific commit to publish
  - **Confirm**: Type "PUBLISH" to confirm

### 2. Validation & Version Update
- ✅ Validates the dev commit exists and is from dev branch
- ✅ Calculates next version number (based on release type)
- 📝 **Updates version in `quarto/index.qmd`** (automatic)
- ✅ Commits version update to dev branch
- ✅ Merges dev → main branch (includes version update)
- ✅ Creates release tag
- ✅ Pushes to main (triggers production build)

### 3. Production Build
- 🔄 Triggers the "🎮 Controller" workflow
- 📚 Builds HTML and PDF versions
- 📄 Compresses PDF with Ghostscript
- 📦 Uploads build artifacts (including PDF)

### 4. PDF Handling (NEW)
- 📄 Downloads PDF from build artifacts
- 📦 Creates GitHub Release (DRAFT)
- 📄 Uploads PDF to Release Assets
- ✅ PDF is available for download but NOT in git

### 5. Manual Release Notes (NEW)
- 📝 Creates draft release for manual editing
- 🔧 Provides release notes generator script
- ✏️ Allows custom release notes on GitHub website

## 🔢 Version Number Automation

### How It Works
The version number displayed on the website is **automatically updated** during the publish workflow:

1. **Workflow calculates version**: Based on release type (patch/minor/major)
   - Patch: v0.4.1 → v0.4.2 (bug fixes)
   - Minor: v0.4.1 → v0.5.0 (new features)
   - Major: v0.4.1 → v1.0.0 (breaking changes)

2. **Updates `quarto/index.qmd`**:
   - Modifies the `doi:` field with the new version
   - Commits to dev branch: "chore: update version to vX.X.X"

3. **Merges to main**: Version update included in the merge

4. **Displays on website**:
   - Shows as "Version" in the title metadata
   - Links to GitHub releases page (via JavaScript)
   - Automatically stays in sync with GitHub releases

### Files Involved
- **`quarto/index.qmd`**: Contains the version number (doi field)
- **`quarto/assets/scripts/version-link.js`**: Makes version link to releases page
- **`.github/workflows/publish-live.yml`**: Updates version automatically

### Manual Override
If you need to manually update the version (not recommended):
```yaml
# In quarto/index.qmd
doi: "v0.4.1"  # Change this value
```

**Note**: The next publish will overwrite any manual changes.

## 📄 PDF Management Strategy

### What Changed
- **Before**: PDF was committed to git and pushed to gh-pages
- **After**: PDF is uploaded to GitHub Release assets only

### Benefits
- ✅ **No Large Files in Git**: PDF is not tracked in repository
- ✅ **Faster Clones**: Repository stays small
- ✅ **Version Control**: Each release has its own PDF
- ✅ **Download Access**: PDF still available for students
- ✅ **Clean History**: Git history doesn't bloat with PDF changes

### PDF Access Points
1. **Direct Download**: `https://mlsysbook.ai/assets/downloads/Machine-Learning-Systems.pdf`
2. **Release Assets**: `https://github.com/harvard-edge/cs249r_book/releases/download/vX.Y.Z/Machine-Learning-Systems.pdf`
3. **GitHub Pages**: Available in the deployed site

## 🔧 Technical Implementation

### Modified Files
- `.github/workflows/publish-live.yml`: Added PDF download and upload steps
- `.github/workflows/quarto-build-container.yml`: Modified deployment to exclude PDF from git
- `.gitignore`: Added PDF exclusion rules
- `tools/scripts/quarto_publish/publish.sh`: Removed PDF commit
- `binder`: Removed PDF commit

### Key Changes
1. **PDF Download Step**: Downloads PDF from build artifacts
2. **Release Asset Upload**: Uploads PDF to GitHub Release
3. **Git Exclusion**: PDF is copied to assets but not committed
4. **Artifact Management**: PDF is preserved in build artifacts

## 🧪 Testing

Use the test script to verify PDF handling:

```bash
python tools/scripts/test_publish_live.py
```

This will check:
- ✅ PDF exists in assets/
- ✅ PDF is NOT tracked by git
- ✅ PDF is in .gitignore
- ✅ Git status is clean

## 📝 Release Notes Management

The workflow creates **draft releases** on GitHub that you can edit manually. This gives you full control over release notes.

### Writing Release Notes

1. **Run publish-live workflow** → Creates draft release
2. **Edit on GitHub** → Go to the draft release and write your notes
3. **Publish when ready** → Click "Publish release" when satisfied

Release notes should summarize:
- 📖 Content updates and improvements
- 🔧 Technical changes and infrastructure updates
- 🐛 Bug fixes and corrections

## 📋 Usage Instructions

### For Regular Publishing
1. **Develop on dev branch**: Make your changes
2. **Test thoroughly**: Ensure everything works
3. **Trigger publish-live**: Use GitHub Actions UI
4. **Monitor progress**: Watch the workflow run
5. **Verify deployment**: Check the live site

### For Emergency Fixes
1. **Create hotfix branch**: `git checkout -b hotfix/description`
2. **Make minimal changes**: Fix the critical issue
3. **Merge to dev**: `git checkout dev && git merge hotfix/description`
4. **Publish immediately**: Use publish-live workflow
5. **Clean up**: Delete hotfix branch

## 🔍 Troubleshooting

### Common Issues

**PDF Not Found in Artifacts**
- Check that the build completed successfully
- Verify PDF was built in the quarto-build workflow
- Check artifact retention settings

**Release Creation Fails**
- Verify GitHub token permissions
- Check release tag doesn't already exist
- Ensure proper JSON formatting in release notes

**PDF Upload Fails**
- Check file size limits (GitHub has 2GB limit)
- Verify PDF file exists and is valid
- Check network connectivity

### Debug Commands
```bash
# Check PDF status
python tools/scripts/test_publish_live.py

# Check git status
git status

# Check if PDF is tracked
git ls-files assets/downloads/Machine-Learning-Systems.pdf

# Check .gitignore
grep -n "Machine-Learning-Systems.pdf" .gitignore
```

## 📊 Monitoring

### Workflow Status
- **GitHub Actions**: Monitor workflow progress
- **Release Page**: Check release creation
- **Live Site**: Verify deployment
- **PDF Access**: Test download links

### Success Indicators
- ✅ Workflow completes without errors
- ✅ Release is created with PDF asset
- ✅ Live site is updated
- ✅ PDF is accessible for download
- ✅ Git repository remains clean (no PDF commits)

## 🎯 Best Practices

1. **Always test on dev first**: Never publish directly from main
2. **Use descriptive commit messages**: Helps with release notes
3. **Monitor build times**: PDF generation can take 15-30 minutes
4. **Check artifact retention**: Ensure PDFs are preserved
5. **Verify deployment**: Always test the live site after publishing

## 🔗 Related Documentation

- [Development Guide](DEVELOPMENT.md)
- [Build Process](BUILD.md)
- [Container Builds](CONTAINER_BUILDS.md)
