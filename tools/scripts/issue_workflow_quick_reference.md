# Issue Workflow Quick Reference

## 🚀 Quick Start Commands

### 1. Start Working on Issue
```bash
git checkout dev && git pull origin dev
git checkout -b issue-{number}-{short-description}
```

### 2. Commit with Issue Reference
```bash
git commit -m "type(scope): description

- Change details
- Addresses #{number}"
```

### 3. Final Commit (Closes Issue)
```bash
git commit -m "type(scope): final description

- Final changes
- Closes #{number}"
```

### 4. Create PR
```bash
git push origin issue-{number}-{description}
gh pr create --title "Fix #{number}: {Title}" --body "Closes #{number}" --base dev
```

## 📋 Branch Naming Examples
- `issue-960-duplicate-efficiency-headings`
- `issue-959-framework-components-duplicate`
- `issue-1001-broken-cross-references`

## 🔗 GitHub Keywords
- **Link only**: `Addresses #123`, `References #123`, `Relates to #123`
- **Close issue**: `Closes #123`, `Fixes #123`, `Resolves #123`

## ✅ Commit Types
- `fix`: Bug fixes, corrections
- `feat`: New features, content
- `docs`: Documentation
- `refactor`: Restructuring
- `style`: Formatting

## 🎯 Remember
- Always branch from `dev`
- Use `--no-ff` when merging
- Include issue number in branch name
- Final commit should close the issue
- Test before creating PR
