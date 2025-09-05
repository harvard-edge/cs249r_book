Run multi-perspective ML Systems textbook review using parallel Task subagents.

This command reviews your chapter from 5 different engineering perspectives simultaneously, finds consensus issues, and applies improvements directly to the file. You then review the changes in GitKraken and keep what you want.

Usage: `/improve introduction.qmd`

## What Happens:

1. **Creates a branch**: `improve-[chapter]-[timestamp]` for safe experimentation
2. **Chunks the file**: Splits into ~400 line semantic chunks 
3. **Parallel review**: 5 Task subagents review each chunk simultaneously
4. **Finds consensus**: Identifies issues multiple reviewers agree on
5. **Applies improvements**: Makes direct changes to the file (no markdown comments)
6. **Generates report**: Creates `.improve-report-[chapter].md` with all findings

## The 5 Reviewers:

- **Systems Engineer**: Production deployment, scaling, operations
- **ML Practitioner**: Notebook-to-production gap, deployment challenges  
- **Embedded Engineer**: Hardware constraints, power, edge deployment
- **Platform Engineer**: MLOps, infrastructure, team collaboration
- **Data Engineer**: Pipelines, feature stores, data quality

## Consensus Application:

- **4-5 reviewers agree**: Applied automatically
- **3 reviewers agree**: Applied (high priority)
- **2 reviewers agree**: Noted in report for manual review
- **1 reviewer**: Documented only

## After Running:

1. Open GitKraken to see all changes as clean diffs
2. Review the `.improve-report-[chapter].md` for context
3. Stage changes you want to keep
4. Discard changes you don't like
5. Commit when satisfied

The improvements focus on the book's mission: teaching practical ML systems engineering, not just algorithms.