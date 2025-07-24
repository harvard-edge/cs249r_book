#!/bin/bash

# Define the search directory
SEARCH_DIR="$1"

# Temp file to hold name + count pairs
TMPFILE=$(mktemp)

# Collect counts
find "$SEARCH_DIR" -name '*.qmd' | while read -r file; do
    count=$(awk '{n += gsub(/\[\^fn-[^]]+\]:/, "")} END {print n}' "$file")
    filename=$(basename "$file")
    echo "$filename $count" >> "$TMPFILE"
done

# Find max length of filename for formatting
max_len=$(awk '{print length($1)}' "$TMPFILE" | sort -nr | head -n1)

# Sort by count descending for readability
sort -k2 -nr "$TMPFILE" | while read -r name count; do
    bar=$(printf '%*s' "$count" '' | tr ' ' '*')
    printf "%-${max_len}s | %3d %s\n" "$name" "$count" "$bar"
done

# Clean up
rm "$TMPFILE"
