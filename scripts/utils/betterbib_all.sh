#!/bin/bash -x

# Find all .bib files in the contents directory and run betterbib update on each
find $1 -type f -name "*.bib" | while read -r bibfile; do
    echo "Updating: $bibfile"
    betterbib update -i "$bibfile"
done

echo "All .bib files have been updated."
