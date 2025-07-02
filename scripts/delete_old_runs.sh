#!/bin/bash

# Deletes all GitHub Actions workflow runs except the most recent one
# Requires: gh CLI, jq

echo "🔍 Fetching latest 1000 runs..."
RUNS_JSON=$(gh run list --limit 1000 --json databaseId,displayTitle,createdAt)

TOTAL=$(echo "$RUNS_JSON" | jq length)
if [ "$TOTAL" -le 1 ]; then
  echo "✅ Nothing to delete — only one run found."
  exit 0
fi

echo "🧹 Deleting $((TOTAL - 1)) old runs (keeping the most recent one)..."

echo "$RUNS_JSON" | jq -r '.[1:][] | "\(.databaseId) \(.displayTitle) \(.createdAt)"' | while read -r ID TITLE CREATED; do
  echo "🗑️ Deleting run: [$CREATED] $TITLE (ID: $ID)"
  gh run delete "$ID" --confirm
done

echo "✅ Done. All but the most recent run have been deleted."
