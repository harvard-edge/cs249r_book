# MLSysBook Issue Branch Workflow

## Overview
This document establishes our standardized workflow for handling GitHub issues using feature branches with automatic issue linking and closure.

## Branch Naming Convention

### Standard Format
```
issue-{issue_number}-{short-description}
```

### Examples
- `issue-960-duplicate-efficiency-headings`
- `issue-959-framework-components-duplicate`
- `issue-1001-broken-cross-references`
- `issue-1002-missing-code-examples`

### Rules
- **Always include issue number** for automatic GitHub linking
- **Use kebab-case** (lowercase with hyphens)
- **Keep description short** (2-4 words max)
- **Be descriptive** but concise about the main problem

## Workflow Steps

### 1. Create Issue Branch
```bash
# From dev branch (always branch from dev, not main)
git checkout dev
git pull origin dev

# Create and checkout issue branch
git checkout -b issue-{number}-{description}

# Example:
git checkout -b issue-960-duplicate-efficiency-headings
```

### 2. Work on Issue
- Follow our [Feedback Processing System](feedback_processing_system.md)
- Make focused commits that address specific aspects
- Use conventional commit format with issue references

### 3. Commit with Issue References

#### Commit Message Format
```
type(scope): description

- Detailed change 1
- Detailed change 2
- Addresses #issue_number

Closes #issue_number
```

#### Examples
```bash
# For ongoing work (doesn't close issue yet)
git commit -m "fix(efficient_ai): remove duplicate heading

- Renamed subheading to 'Resource-Constrained Trade-offs'
- Eliminates redundancy with main section heading
- Addresses #960"

# For final commit (closes issue)
git commit -m "fix(efficient_ai): eliminate all repetitive content

- Consolidated redundant explanations of efficiency interdependencies
- Streamlined trade-offs introductions
- Improved content flow and readability
- Fully addresses feedback about repetitive structure

Closes #960"
```

### 4. Push and Create Pull Request
```bash
# Push branch to origin
git push origin issue-{number}-{description}

# Create PR via GitHub CLI (recommended)
gh pr create --title "Fix #{number}: {Issue Title}" \
             --body "Resolves #{number}

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [x] Chapter builds successfully
- [x] No linting errors
- [x] All feedback points addressed

## Related Issues
Closes #{number}" \
             --base dev

# Example:
gh pr create --title "Fix #960: Duplicate Efficiency Trade-offs Heading" \
             --body "Resolves #960

## Changes Made
- Removed duplicate 'Efficiency Trade-offs' subheading
- Consolidated repetitive content about efficiency interdependencies
- Improved content flow and eliminated redundancy

## Testing
- [x] Chapter builds successfully
- [x] No linting errors
- [x] All feedback points addressed

## Related Issues
Closes #960" \
             --base dev
```

### 5. Merge and Cleanup
```bash
# After PR approval, merge with --no-ff to preserve history
git checkout dev
git pull origin dev
git merge --no-ff issue-{number}-{description}
git push origin dev

# Clean up local branch
git branch -d issue-{number}-{description}

# Clean up remote branch (if not auto-deleted)
git push origin --delete issue-{number}-{description}
```

## GitHub Automation Keywords

### Issue Linking (in commits and PRs)
- `Addresses #123` - Links to issue without closing
- `Relates to #123` - Links to issue without closing
- `References #123` - Links to issue without closing

### Issue Closing (in commits and PRs)
- `Closes #123` - Closes issue when merged to default branch
- `Fixes #123` - Closes issue when merged to default branch
- `Resolves #123` - Closes issue when merged to default branch

### Multiple Issues
```
Addresses #960, #959
Closes #960
Relates to #958
```

## Branch Management Rules

### When to Create Issue Branches
✅ **Always create for:**
- Bug fixes from GitHub issues
- Feature requests from GitHub issues
- Content improvements from feedback
- Structural changes affecting multiple files
- Any work that addresses a specific GitHub issue

❌ **Don't create for:**
- Typo fixes (commit directly to dev)
- Single-line changes (commit directly to dev)
- Emergency hotfixes (use hotfix branches)

### Branch Lifecycle
1. **Create** from latest dev
2. **Work** with focused commits
3. **Push** regularly for backup
4. **PR** when ready for review
5. **Merge** with --no-ff to dev
6. **Delete** after successful merge

## Commit Message Standards

### Format
```
type(scope): short description

Longer description if needed explaining:
- What was changed
- Why it was changed
- How it addresses the issue

Addresses #issue_number
[Closes #issue_number] # Only on final commit
```

### Types
- `fix`: Bug fixes, content corrections
- `feat`: New features, new content sections
- `docs`: Documentation updates
- `refactor`: Code/content restructuring
- `style`: Formatting, style guide compliance
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Scopes (Chapter/Section Names)
- `efficient_ai`: Chapter 9
- `frameworks`: Chapter 7
- `training`: Training chapter
- `ops`: Operations chapter
- `global`: Cross-chapter changes

## Automation Benefits

### GitHub Will Automatically:
1. **Link commits to issues** when branch name includes issue number
2. **Close issues** when PR with "Closes #123" merges to dev/main
3. **Reference issues** in commit history for traceability
4. **Update issue status** based on commit keywords
5. **Create timeline** showing all related commits and PRs

### Project Benefits:
1. **Full traceability** from issue to resolution
2. **Automatic documentation** of what changed and why
3. **Clear history** of how each issue was resolved
4. **Reduced manual work** in issue management
5. **Better collaboration** through linked discussions

## Example Complete Workflow

```bash
# 1. New issue #1005 reported: "Missing code examples in Chapter 8"
git checkout dev
git pull origin dev
git checkout -b issue-1005-missing-code-examples

# 2. Work on the issue
# ... make changes ...
git add .
git commit -m "feat(training): add basic training loop example

- Added complete PyTorch training example
- Includes data loading and model definition
- Addresses #1005"

# 3. Continue work
# ... more changes ...
git commit -m "feat(training): add distributed training example

- Added multi-GPU training setup
- Includes proper error handling
- Further addresses #1005"

# 4. Final commit
git commit -m "feat(training): complete code examples section

- Added validation loop and metrics tracking
- Included best practices and common pitfalls
- All examples tested and verified working
- Comprehensive documentation added

Closes #1005"

# 5. Push and create PR
git push origin issue-1005-missing-code-examples
gh pr create --title "Fix #1005: Missing Code Examples in Chapter 8" \
             --body "Resolves #1005

## Changes Made
- Added complete PyTorch training loop example
- Added distributed training setup example
- Added validation and metrics tracking
- Included best practices documentation

## Testing
- [x] All code examples tested and working
- [x] Chapter builds successfully
- [x] No linting errors

Closes #1005" \
             --base dev

# 6. After PR approval and merge, GitHub automatically closes issue #1005
```

## Quality Checklist

### Before Creating Branch
- [ ] Issue is clearly understood
- [ ] Branch name follows convention
- [ ] Branching from latest dev

### During Development
- [ ] Commits reference issue number
- [ ] Commit messages are descriptive
- [ ] Changes are focused and related
- [ ] Regular pushes for backup

### Before Final Commit
- [ ] All issue requirements addressed
- [ ] Testing completed successfully
- [ ] Documentation updated if needed
- [ ] Final commit includes "Closes #issue"

### PR Creation
- [ ] Title includes issue number
- [ ] Description links to issue
- [ ] Changes are documented
- [ ] Testing checklist completed

---

*This workflow ensures complete traceability from issue identification through resolution, with automatic GitHub integration for seamless project management.*
