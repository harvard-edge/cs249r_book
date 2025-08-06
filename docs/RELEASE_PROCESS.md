# Release Process for MLSysBook

## 🎯 Release Strategy

We follow a milestone-based release approach suitable for an academic textbook project.

## 📋 Release Types

### Major Releases (x.0.0)
- Complete textbook releases
- Major structural changes
- Significant content overhauls
- **Frequency:** Semester/annual

### Minor Releases (x.y.0)
- New chapters added
- New lab exercises
- Significant content additions
- **Frequency:** Monthly or per major feature

### Patch Releases (x.y.z)
- Bug fixes and typos
- Minor content updates
- Formatting improvements
- **Frequency:** As needed

## 🔄 Daily Workflow

### For Regular Development (Website Updates)
```bash
# 1. Make changes on feature branches
git checkout -b feature/new-chapter

# 2. Commit and push changes
git add .
git commit -m "feat(chapter): add new optimization techniques chapter"
git push origin feature/new-chapter

# 3. Create PR and merge to main
# (GitHub PR process)

# 4. Deploy to website (no formal release)
./binder publish     # Quick website deployment
```

### For Formal Releases (Milestones)
```bash
# 1. Ensure main is up to date
git checkout main
git pull origin main

# 2. Create formal release with versioning
./binder release

# 3. Follow interactive prompts for:
#    - Semantic version type (patch/minor/major)  
#    - Release description
#    - Git tag creation
#    - GitHub release with PDF attachment
```

## 🏷️ Versioning Guidelines

### Version Number Format: `vMAJOR.MINOR.PATCH`

**Examples:**
- `v1.0.0` - First complete textbook release
- `v1.1.0` - Added new "Federated Learning" chapter
- `v1.1.1` - Fixed typos and updated references
- `v2.0.0` - Major restructuring with new part organization

### When to Increment:

**MAJOR** (x.0.0):
- Complete textbook restructuring
- Breaking changes to existing content organization
- Major pedagogical approach changes

**MINOR** (x.y.0):
- New chapters or major sections
- New lab exercises or projects
- Significant content additions (>10% new material)

**PATCH** (x.y.z):
- Bug fixes, typos, formatting
- Minor content updates (<5% changes)
- Reference updates, link fixes

## 📝 Release Notes

### Automated Generation
- Use `./binder release` for AI-generated release notes
- Always review and edit before publishing
- Include:
  - Overview of changes
  - New content highlights
  - Bug fixes and improvements
  - Any breaking changes

### Manual Release Notes Template
```markdown
# Release v1.1.0: New Optimization Techniques

## 🆕 New Content
- Added Chapter 12: Advanced Optimization Techniques
- New lab exercise on hyperparameter tuning
- Extended bibliography with 50+ new references

## 🐛 Bug Fixes
- Fixed equation formatting in Chapter 8
- Corrected code examples in PyTorch section

## 🔧 Improvements
- Updated all Python code examples to Python 3.11
- Improved figure quality and accessibility
- Enhanced cross-references throughout

## 📊 Statistics
- Total pages: 450 (+25 from previous version)
- New exercises: 12
- Updated figures: 8
```

## 🚀 Deployment Strategy

### Two-Tier Publishing System

#### Website Publishing (`./binder publish`)
- **Purpose:** Quick content updates for daily development
- **Process:** Builds HTML + PDF, deploys to GitHub Pages
- **Requirements:** Minimal git validation (allows uncommitted changes)
- **Result:** Updates https://mlsysbook.ai immediately
- **No Versioning:** No git tags or formal releases created

#### Formal Releases (`./binder release`)  
- **Purpose:** Academic milestones and citation-ready releases
- **Process:** Semantic versioning + GitHub release creation
- **Requirements:** Clean git state, intentional versioning decisions
- **Result:** Tagged releases with attached PDFs for citations
- **Versioning:** Git tags, release notes, academic distribution

### Deployment Locations
- **Live Website:** https://mlsysbook.ai (updated by `publish`)
- **PDF Download:** https://mlsysbook.ai/assets/Machine-Learning-Systems.pdf
- **Tagged Releases:** https://github.com/harvard-edge/cs249r_book/releases
- **Versioned PDFs:** Attached to each GitHub release for citations

## 🛡️ Best Practices

### Before Creating a Release
1. **Content Review:**
   - Proofread new content
   - Verify all links and references
   - Test all code examples
   - Check figure quality and captions

2. **Technical Checks:**
   - Run `./binder build` to ensure clean build
   - Verify PDF generation works
   - Check website deployment
   - Run pre-commit hooks

3. **Documentation:**
   - Update CHANGELOG.md
   - Review and update README if needed
   - Ensure release notes are comprehensive

### Release Timing
- **Avoid:** Releases during exam periods or holidays
- **Prefer:** Beginning of week for better visibility
- **Coordinate:** With course schedules if used in classes

### Communication
- Announce major releases via:
  - GitHub release notifications
  - Course announcements (if applicable)
  - Social media/academic networks
  - Email to collaborators

## 🔄 Maintenance Releases

For critical fixes between planned releases:

```bash
# Create hotfix branch
git checkout -b hotfix/critical-bug-fix

# Make minimal fix
git commit -m "fix: critical typo in equation 8.3"

# Create patch release
./binder release  # Will suggest next patch version
```

## 📋 Release Checklist

### Pre-Release
- [ ] All content reviewed and proofread
- [ ] All code examples tested
- [ ] Links and references verified
- [ ] Clean build successful (`./binder build`)
- [ ] PDF generation working
- [ ] No linting errors
- [ ] CHANGELOG.md updated

### Release Process
- [ ] Created release branch if needed
- [ ] Generated and reviewed release notes
- [ ] Tagged with semantic version
- [ ] GitHub release published
- [ ] PDF attached to release
- [ ] Website deployed and verified

### Post-Release
- [ ] Release announcement sent
- [ ] Social media updates (if applicable)
- [ ] Course materials updated (if applicable)
- [ ] Next release planning initiated

---

## 📞 Questions?

For questions about the release process, see:
- `./binder help` for tool-specific guidance
- GitHub Issues for process improvements
- CONTRIBUTING.md for development workflow