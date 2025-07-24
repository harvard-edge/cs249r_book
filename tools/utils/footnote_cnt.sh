#!/bin/bash

# footnote_counter.sh
# --------------------
# Recursively counts footnote definitions of the form [^fn-...]: in .qmd files within a directory.
# Displays a sorted histogram of footnote counts per file.
#
# Usage:
#   ./footnote_counter.sh <search_directory>
#
#   Options:
#     -h, --help      Show this help message and exit.
#

# Exit immediately on error, undefined variable, or pipeline failure
set -euo pipefail

# Function to print usage/help information
show_help() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

# Function to print an error message and exit
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Show help if user asks for it
if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
    show_help
fi

# Check if exactly one positional argument is given
if [[ $# -ne 1 ]]; then
    error_exit "Usage: $0 <search_directory>"
fi

SEARCH_DIR="$1"

# Validate that the directory exists
if [[ ! -d "$SEARCH_DIR" ]]; then
    error_exit "'$SEARCH_DIR' is not a valid directory"
fi

# Create a temporary file to store filename and footnote count pairs
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT

# Find all .qmd files and count footnote definitions in each
while IFS= read -r -d '' file; do
    if [[ -r "$file" ]]; then
        # Count instances of [^fn-...]: footnotes using awk and gsub
        count=$(awk '{n += gsub(/\[\^fn-[^]]+\]:/, "")} END {print n}' "$file")
        filename=$(basename "$file")
        echo "$filename $count" >> "$TMPFILE"
    else
        echo "Warning: Cannot read file '$file'" >&2
    fi
done < <(find "$SEARCH_DIR" -name '*.qmd' -print0)

# Exit if no results were collected
if [[ ! -s "$TMPFILE" ]]; then
    error_exit "No readable .qmd files found in '$SEARCH_DIR'"
fi

# Determine longest filename for alignment
max_len=$(awk '{print length($1)}' "$TMPFILE" | sort -nr | head -n1)

# Display sorted results as a simple bar chart
sort -k1,1 "$TMPFILE" | while read -r name count; do
    bar=$(printf '%*s' "$count" '' | tr ' ' '*')
    printf "%-${max_len}s | %3d %s\n" "$name" "$count" "$bar"
done
