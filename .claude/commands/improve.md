Run multi-perspective textbook review and create small, focused PRs for improvements.

This command:
1. Chunks large .qmd files intelligently to avoid context overflow
2. Reviews each chunk from 5 student perspectives (Junior CS, Senior EE, Masters, PhD, Industry)
3. Consolidates feedback and prioritizes issues
4. Creates small PRs (≤5 changes each) for easy review
5. Generates descriptive commit messages and PR descriptions

Usage:
- Basic: `/improve introduction.qmd`
- With options: `/improve frameworks.qmd --max-prs 3`

The command will:
1. Analyze the file and split into semantic chunks (~400 lines each)
2. Run multi-perspective reviews using Task subagents
3. Identify consensus issues (reported by multiple reviewers)
4. Create separate Git branches for Critical, High, and Medium priority fixes
5. Generate PRs with clear descriptions of what was fixed

Example workflow:
```python
# 1. Chunk the file
chunks = smart_chunk_file(chapter_file)

# 2. Review each chunk from multiple perspectives
for chunk in chunks:
    reviews = run_multi_perspective_review(chunk)
    
# 3. Consolidate and prioritize
issues = consolidate_feedback(reviews)
prioritized = prioritize_by_consensus(issues)

# 4. Create small PR batches
create_pr_for_critical_issues(prioritized[:5])
create_pr_for_high_priority(prioritized[5:10])

# 5. Output summary with PR links
```

The review system files are located in `/review/` directory.