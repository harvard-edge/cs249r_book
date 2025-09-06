# Agent Commit Policy

## IMPORTANT: No Auto-Staging or Committing

All agents (reviewer, editor, footnote, stylist) should follow these rules:

### 1. Branch Creation
- Create appropriate branch for the task
- Use proper naming convention (e.g., `footnote/`, `editor/`, `reviewer/`, `stylist/`)

### 2. Make Changes
- Perform all necessary edits
- Follow agent-specific guidelines
- Ensure quality and consistency

### 3. DO NOT Stage or Commit
- **Never run `git add`**
- **Never run `git commit`**
- Leave all changes unstaged
- User will review and decide what to include

### 4. Report Completion
- Summarize what was done
- Note which files were modified
- Mention any important decisions made

## Rationale

This policy ensures:
- User maintains full control over commits
- User can review all changes before staging
- User can selectively stage specific changes
- User can modify agent work before committing
- Commit messages reflect user's intent

## Example Agent Workflow

```bash
# Create branch
git checkout -b agent-type/task-name

# Make changes (edit files as needed)
# ...

# Check what was changed
git status
git diff

# DO NOT run git add or git commit
# Simply report: "Changes complete, ready for your review"
```

## User Workflow After Agent

```bash
# User reviews changes
git diff

# User selectively stages what they want
git add -p  # or git add specific files

# User commits with their own message
git commit -m "their message"
```