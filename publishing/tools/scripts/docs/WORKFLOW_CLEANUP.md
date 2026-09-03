# GitHub Workflow Runs Cleanup

This script helps clean up old GitHub workflow runs while keeping a configurable number of recent runs per workflow. It's particularly useful when you have many failed runs from debugging.

## Quick Start

1. **Get a GitHub Personal Access Token**
   - Go to https://github.com/settings/tokens
   - Create a token with `actions:write` and `repo` scopes
   - Save the token securely

2. **Set up the token**
   ```bash
   # Option 1: Set environment variable (recommended)
   export GITHUB_TOKEN="your_token_here"

   # Option 2: Use --token flag each time
   ```

3. **Preview what would be cleaned up**
   ```bash
   python3 tools/scripts/cleanup_workflow_runs.py --summary
   python3 tools/scripts/cleanup_workflow_runs.py --dry-run --keep 5
   ```

4. **Actually clean up workflow runs**
   ```bash
   # Keep 5 most recent runs per workflow
   python3 tools/scripts/cleanup_workflow_runs.py --keep 5

   # Keep 10 most recent runs per workflow
   python3 tools/scripts/cleanup_workflow_runs.py --keep 10
   ```

## Usage Examples

### Show Current State
```bash
# See summary of all workflows and run counts
python3 tools/scripts/cleanup_workflow_runs.py --summary
```

### Dry Run (Preview)
```bash
# See what would be deleted (keeping 5 runs per workflow)
python3 tools/scripts/cleanup_workflow_runs.py --dry-run --keep 5

# Preview cleanup for specific workflow
python3 tools/scripts/cleanup_workflow_runs.py --dry-run --workflow "quarto-build-container.yml" --keep 3
```

### Actual Cleanup
```bash
# Clean all workflows, keep 5 most recent runs each
python3 tools/scripts/cleanup_workflow_runs.py --keep 5

# Clean specific workflow only, keep 3 runs
python3 tools/scripts/cleanup_workflow_runs.py --workflow "quarto-build-container.yml" --keep 3

# Keep more runs (10 per workflow)
python3 tools/scripts/cleanup_workflow_runs.py --keep 10
```

## Advanced Usage

### Environment Variables
```bash
# Set token once
export GITHUB_TOKEN="ghp_your_token_here"

# Override repository detection
export GITHUB_REPOSITORY="owner/repo"
```

### Workflow-Specific Cleanup
You can target specific workflows by name or filename:
```bash
# By workflow filename
python3 tools/scripts/cleanup_workflow_runs.py --workflow "quarto-build-container.yml" --keep 3

# By workflow name
python3 tools/scripts/cleanup_workflow_runs.py --workflow "Quarto Build" --keep 3
```

## Safety Features

- **Dry Run Mode**: Always preview before deleting
- **Rate Limiting**: Automatically handles GitHub API rate limits
- **Error Handling**: Graceful handling of API errors and permissions
- **Auto-Detection**: Automatically detects repository from git remote
- **Confirmation**: Clear output showing what was deleted

## Troubleshooting

### Permission Errors
```
❌ Permission denied. Check your token has 'actions:write' scope.
```
- Ensure your GitHub token has `actions:write` and `repo` scopes
- Re-generate token if needed

### Rate Limiting
```
⚠️  Rate limit exceeded. Waiting 60 seconds...
```
- The script automatically waits for rate limit reset
- GitHub allows 5,000 API requests per hour for authenticated users

### Repository Not Found
```
❌ Repository not specified and could not auto-detect from git
```
- Use `--repo owner/repo` flag
- Or ensure you're running from a git repository with GitHub remote

## Script Options

| Option | Default | Description |
|--------|---------|-------------|
| `--keep` | 5 | Number of recent runs to keep per workflow |
| `--workflow` | (all) | Target specific workflow by name or filename |
| `--dry-run` | false | Preview mode - don't actually delete |
| `--summary` | false | Show workflow summary and exit |
| `--token` | env var | GitHub personal access token |
| `--repo` | auto-detect | Repository in `owner/repo` format |

## Best Practices

1. **Always dry run first**: Use `--dry-run` to preview changes
2. **Keep enough runs**: Don't go below 3-5 runs to maintain debugging context
3. **Target specific workflows**: Use `--workflow` for problematic workflows
4. **Regular cleanup**: Run weekly/monthly to prevent accumulation
5. **Monitor rate limits**: Be aware of GitHub API limits for large cleanups
