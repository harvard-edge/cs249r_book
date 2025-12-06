# TinyTorch Release Process

## Overview

This document describes the complete release process for TinyTorch, combining automated CI/CD checks with manual agent-driven reviews.

## Release Types

### Patch Release (0.1.X)
- Bug fixes
- Documentation updates
- Minor improvements
- **Timeline:** 1-2 days

### Minor Release (0.X.0)
- New module additions
- Feature enhancements
- Significant improvements
- **Timeline:** 1-2 weeks

### Major Release (X.0.0)
- Complete module sets
- Breaking API changes
- Architectural updates
- **Timeline:** 1-3 months

## Two-Track Quality Assurance

### Track 1: Automated CI/CD (Continuous)

**GitHub Actions** runs on every commit and PR:

```
Every Push/PR:
├── Educational Validation (Module structure, objectives)
├── Implementation Validation (Time, difficulty, tests)
├── Test Validation (All tests, coverage)
├── Package Validation (Builds, installs)
├── Documentation Validation (ABOUT.md, checkpoints)
└── Systems Analysis (Memory, performance, production)
```

**Trigger:** Automatic on push/PR

**Duration:** 15-20 minutes

**Pass Criteria:** All 6 quality gates green

---

### Track 2: Agent-Driven Review (Pre-Release)

**Specialized AI agents** provide deep review before releases:

```
TPM Coordinates:
├── Education Reviewer
│   ├── Pedagogical effectiveness
│   ├── Learning objective alignment
│   ├── Cognitive load assessment
│   └── Assessment quality
│
├── Module Developer
│   ├── Implementation standards
│   ├── Code quality patterns
│   ├── Testing completeness
│   └── PyTorch API alignment
│
├── Quality Assurance
│   ├── Comprehensive test validation
│   ├── Edge case coverage
│   ├── Performance testing
│   └── Integration stability
│
└── Package Manager
    ├── Module integration
    ├── Dependency resolution
    ├── Export/import validation
    └── Build verification
```

**Trigger:** Manual (via TPM)

**Duration:** 2-4 hours

**Pass Criteria:** All agents approve

---

## Complete Release Workflow

### Phase 1: Development (Ongoing)

1. **Feature Development**
   - Implement modules following DEFINITIVE_MODULE_PLAN.md
   - Write tests immediately after each function
   - Ensure NBGrader compatibility
   - Add checkpoint markers to long modules

2. **Local Validation**
   ```bash
   # Run validators locally
   python .github/scripts/validate_time_estimates.py
   python .github/scripts/validate_difficulty_ratings.py
   python .github/scripts/validate_testing_patterns.py
   python .github/scripts/check_checkpoints.py

   # Run tests
   pytest tests/ -v
   ```

3. **Commit & Push**
   ```bash
   git add .
   git commit -m "feat: Add [feature] to [module]"
   git push origin feature-branch
   ```

---

### Phase 2: Pre-Release Review (1-2 days)

1. **Create Release Branch**
   ```bash
   git checkout -b release/v0.X.Y
   git push origin release/v0.X.Y
   ```

2. **Automated CI/CD Check**
   - GitHub Actions runs automatically
   - Review workflow results
   - Fix any failures

3. **Agent-Driven Comprehensive Review**

   **Invoke TPM for multi-agent review:**

   ```
   Request to TPM:
   "I need a comprehensive quality review of all 20 TinyTorch modules
   for release v0.X.Y. Please coordinate:

   1. Education Reviewer - pedagogical validation
   2. Module Developer - implementation standards
   3. Quality Assurance - testing validation
   4. Package Manager - integration health

   Run these in parallel and provide:
   - Consolidated findings report
   - Prioritized action items
   - Estimated effort for fixes
   - Timeline for completion

   Release Type: [patch/minor/major]
   Target Date: [YYYY-MM-DD]"
   ```

4. **Review Agent Reports**
   - Education Reviewer report
   - Module Developer report
   - Quality Assurance report
   - Package Manager report

5. **Address Findings**
   - Fix HIGH priority issues immediately
   - Schedule MEDIUM priority for next sprint
   - Document LOW priority as future improvements

---

### Phase 3: Release Candidate (1 day)

1. **Create Release Candidate**
   ```bash
   git tag -a v0.X.Y-rc1 -m "Release candidate 1 for v0.X.Y"
   git push origin v0.X.Y-rc1
   ```

2. **Final Validation**
   - Run full test suite
   - Build documentation
   - Test package installation
   - Manual smoke testing

3. **Stakeholder Review** (if applicable)
   - Share RC with instructors
   - Collect feedback
   - Make final adjustments

---

### Phase 4: Release (1 day)

1. **Manual Release Check Trigger**

   Via GitHub UI:
   - Go to Actions → TinyTorch Release Check
   - Click "Run workflow"
   - Select:
     - Branch: `release/v0.X.Y`
     - Release Type: `[patch/minor/major]`
     - Check Level: `comprehensive`

2. **Review Release Report**
   - All quality gates pass
   - Download release report artifact
   - Verify all validations green

3. **Merge to Main**
   ```bash
   git checkout main
   git merge --no-ff release/v0.X.Y
   git push origin main
   ```

4. **Create Official Release**
   ```bash
   git tag -a v0.X.Y -m "Release v0.X.Y: [Description]"
   git push origin v0.X.Y
   ```

5. **GitHub Release**
   - Go to Releases → Draft a new release
   - Select tag: `v0.X.Y`
   - Title: `TinyTorch v0.X.Y`
   - Description: Include release report summary
   - Attach artifacts (wheels, documentation)
   - Publish release

6. **Package Distribution**
   ```bash
   # Build distribution packages
   python -m build

   # Upload to PyPI (if applicable)
   python -m twine upload dist/*
   ```

---

### Phase 5: Post-Release (Ongoing)

1. **Documentation Updates**
   - Update README.md with new version
   - Update CHANGELOG.md
   - Rebuild Jupyter Book
   - Deploy to mlsysbook.github.io

2. **Communication**
   - Announce on GitHub
   - Update course materials
   - Notify instructors
   - Social media (if applicable)

3. **Monitoring**
   - Watch for issues
   - Respond to feedback
   - Plan next release

---

## Quality Gates Reference

### Must Pass for ALL Releases

✅ All automated CI/CD checks pass
✅ Test coverage ≥80%
✅ All agent reviews approved
✅ Documentation complete
✅ No HIGH priority issues

### Additional for Major Releases

✅ All 20 modules validated
✅ Complete integration testing
✅ Performance benchmarks meet targets
✅ Comprehensive stakeholder review

---

## Checklist Templates

### Patch Release Checklist

```markdown
## Pre-Release
- [ ] Local validation passes
- [ ] Automated CI/CD passes
- [ ] Bug fix validated
- [ ] Tests updated

## Release
- [ ] Release branch created
- [ ] RC tested
- [ ] Merged to main
- [ ] Tag created
- [ ] GitHub release published

## Post-Release
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Issue closed
```

### Minor Release Checklist

```markdown
## Pre-Release
- [ ] All local validations pass
- [ ] Automated CI/CD passes
- [ ] Agent reviews complete (all 4)
- [ ] High priority issues fixed
- [ ] New modules validated
- [ ] Integration tests pass

## Release
- [ ] Release branch created
- [ ] RC tested
- [ ] Stakeholder review (if needed)
- [ ] Merged to main
- [ ] Tag created
- [ ] GitHub release published
- [ ] Package uploaded (if applicable)

## Post-Release
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Jupyter Book rebuilt
- [ ] Announcement sent
```

### Major Release Checklist

```markdown
## Pre-Release (1-2 weeks)
- [ ] All local validations pass
- [ ] Automated CI/CD passes
- [ ] Comprehensive agent review (TPM-coordinated)
  - [ ] Education Reviewer approved
  - [ ] Module Developer approved
  - [ ] Quality Assurance approved
  - [ ] Package Manager approved
- [ ] ALL modules validated (20/20)
- [ ] Complete integration testing
- [ ] Performance benchmarks met
- [ ] Documentation complete
- [ ] All HIGH/MEDIUM issues resolved

## Release Candidate (3-5 days)
- [ ] RC1 created and tested
- [ ] Stakeholder feedback collected
- [ ] Final adjustments made
- [ ] RC2 validated (if needed)

## Release
- [ ] Release branch created
- [ ] Comprehensive check run
- [ ] All quality gates green
- [ ] Merged to main
- [ ] Tag created
- [ ] GitHub release published
- [ ] Package uploaded to PyPI
- [ ] Backup created

## Post-Release (1 week)
- [ ] Documentation updated everywhere
- [ ] CHANGELOG complete
- [ ] Jupyter Book rebuilt and deployed
- [ ] All stakeholders notified
- [ ] Social media announcement
- [ ] Course materials updated
- [ ] Monitor for issues
```

---

## Emergency Hotfix Process

For critical bugs in production:

1. **Create hotfix branch from main**
   ```bash
   git checkout main
   git checkout -b hotfix/v0.X.Y+1
   ```

2. **Fix the issue**
   - Minimal changes only
   - Focus on critical bug
   - Add regression test

3. **Fast-track validation**
   ```bash
   # Quick validation
   python .github/scripts/validate_time_estimates.py
   pytest tests/ -v -k "test_affected_module"
   ```

4. **Release immediately**
   ```bash
   git checkout main
   git merge --no-ff hotfix/v0.X.Y+1
   git tag -a v0.X.Y+1 -m "Hotfix: [Description]"
   git push origin main --tags
   ```

5. **Backport to release branches if needed**

---

## Tools & Resources

### GitHub Actions
- Workflow: `.github/workflows/release-check.yml`
- Scripts: `.github/scripts/*.py`
- Documentation: `.github/workflows/README.md`

### Agent Coordination
- TPM: `.claude/agents/technical-program-manager.md`
- Agents: `.claude/agents/`
- Workflow: `DEFINITIVE_MODULE_PLAN.md`

### Validation
- Time: `validate_time_estimates.py`
- Difficulty: `validate_difficulty_ratings.py`
- Tests: `validate_testing_patterns.py`
- Checkpoints: `check_checkpoints.py`

---

## Version Numbering

TinyTorch follows [Semantic Versioning](https://semver.org/):

**Format:** `MAJOR.MINOR.PATCH`

- **MAJOR:** Breaking changes, complete module sets
- **MINOR:** New features, module additions
- **PATCH:** Bug fixes, documentation

**Examples:**
- `0.1.0` → `0.1.1`: Bug fix (patch)
- `0.1.1` → `0.2.0`: New module (minor)
- `0.9.0` → `1.0.0`: All 20 modules complete (major)

---

## Contact & Support

**Questions about releases?**
- Check this document first
- Review workflow README: `.github/workflows/README.md`
- Consult TPM agent for complex scenarios
- File issue on GitHub for workflow improvements

---

**Last Updated:** 2024-11-24
**Version:** 1.0.0
**Maintainer:** TinyTorch Team
