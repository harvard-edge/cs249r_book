# Git Workflow and Merge Rules

## Merge Strategy: No Fast-Forward (--no-ff)

Always use `--no-ff` when merging branches to maintain clear merge history:

```bash
git merge --no-ff branch-name
```

This ensures:
- Clear merge commits in history
- Easier to revert entire features
- Better visibility of feature boundaries
- Preserved branch context

## Branch Naming Conventions

### General Branches
- `cleanup/` - For removing/organizing files
- `improve/` - For enhancing existing features
- `fix/` - For bug fixes
- `feat/` - For new features

### Agent Branch Policy
**Agents should NOT create branches automatically**. Only when explicitly requested:
- `reviewer/` - Created by reviewer agent
- `editor/` - Created by editor agent
- `footnote/` - Created by footnote agent
- `stylist/` - Created by stylist agent

Examples:
- `reviewer/introduction-2025-01-06`
- `editor/fix-forward-refs-ch3`
- `footnote/add-chapter-refs`
- `stylist/academic-tone-cleanup`

## Workflow

1. Create feature branch from dev
2. Make changes
3. Stage selectively with `git add`
4. **DO NOT COMMIT** - leave changes staged for user review
5. User will commit when ready
6. Merge with --no-ff back to dev
7. **Delete feature branch after merge** (mandatory cleanup)

## Important for Agents

- Agents should stage changes but NEVER commit
- Use `git add` to stage modified files
- Leave changes in staging area for user review
- User maintains control over commit timing

## Important Notes

- Never commit directly to main or dev
- Always review diff before committing
- Keep commits atomic and focused
- Write clear commit messages