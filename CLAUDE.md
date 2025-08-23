# Claude Code Configuration for MLSysBook

## How I Work

When you give me a command, I:
1. Check if it matches one of the patterns below
2. Read the methodology in `.claude/private/` (if it exists)
3. Follow those instructions exactly

## Command Patterns I Recognize

### Issue Resolution
**Pattern:** `resolve issue <number>` or `fix issue <number>`
**Action:** I will:
<<<<<<< HEAD
1. Create a feature branch
2. Fetch the issue using `gh issue view <number>`
3. Apply methodology from `.claude/private/ISSUE_RESOLVER.md`
4. Make necessary changes
5. Show you the changes for review
6. Create a DRAFT PR for private review

**Pattern:** `review changes`
**Action:** Show all changes in current branch with explanations

**Pattern:** `create draft pr`
**Action:** Create a draft PR (not visible to issue author)

**Pattern:** `finalize pr <number>`
**Action:** Convert draft PR to ready and tag the issue author with thanks
=======
1. Fetch the issue using `gh issue view <number>`
2. Apply methodology from `.claude/private/ISSUE_RESOLVER.md`
3. Make necessary changes
4. Create a PR
>>>>>>> origin/fix/issue-947-dendrite-synapse-correction

**Pattern:** `analyze issue <number>`
**Action:** I will analyze but NOT make changes

**Pattern:** `check issues`
**Action:** I will list all open improvement issues

### Content Quality
- `check chapter <name>` - Review chapter for quality issues
- `improve flow <section>` - Enhance pedagogical progression
- `check terminology` - Ensure consistent terminology

### Quick Fixes
- `fix typo <description>` - Quick typo fixes
- `update reference <old> <new>` - Update citations
- `fix links` - Check and fix broken links

## Workflow

When resolving issues:
1. I fetch and analyze the issue from GitHub
2. I classify it (content, technical, structural)
3. I apply the appropriate methodology
4. I make changes maintaining quality standards
5. I create a PR with clear description

## Quality Standards

Every change must:
- Maintain technical accuracy
- Preserve pedagogical flow
- Keep consistent voice
- Match chapter difficulty level
- Include practical relevance

## File Organization

- Book content: `quarto/contents/core/`
- Labs: `quarto/contents/labs/`
- Build configs: `quarto/config/`
- Scripts: `tools/scripts/`

## Building & Testing

```bash
# Quick build test
cd quarto && quarto render --to html

# Full build
./binder build html

# Check specific chapter
./binder preview <chapter_name>
```

## Git Workflow

- Main branch: `dev`
- PR branches: `fix/issue-<number>-<description>`
- Commit style: Conventional commits

## Notes

- The `.claude/private/` directory contains proprietary methodologies
- Always verify builds before creating PRs
- Tag PRs with appropriate labels