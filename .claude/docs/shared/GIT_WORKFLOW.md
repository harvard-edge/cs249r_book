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

### Agent-Created Branches
When agents create branches, use the agent name as prefix:
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
3. Stage selectively (not everything)
4. Review changes before committing
5. Merge with --no-ff back to dev
6. Delete feature branch after merge

## Important Notes

- Never commit directly to main or dev
- Always review diff before committing
- Keep commits atomic and focused
- Write clear commit messages