#!/bin/bash

# Set input/output filenames
INPUT_BIB="$1"
OUTPUT_BIB="cleaned_$INPUT_BIB"
RULES_FILE="bibtool_rules.rsc"

# Check if an input file was provided
if [ -z "$INPUT_BIB" ]; then
  echo "Usage: $0 input.bib"
  exit 1
fi

# 📌 STEP 1: Create a BibTool Rules File for Formatting
cat <<EOT > $RULES_FILE
preserve.key.case = on
preserve.keys = on
preserve.values = on
remove.double.braces = off  # Keeps values wrapped in {{...}}
expand.macros = off         # Prevents breaking macros
modify.field = { 
    title journal booktitle author editor
    replace " \_" "_"
    replace "\\_" "_"
    wrap "{" "}"
}
EOT

echo "✅ BibTool rules file created."

# 📌 STEP 2: Remove Duplicates & Clean Formatting with BibTool
echo "🔍 Cleaning BibTeX file with BibTool..."
bibtool -s -r $RULES_FILE -i "$INPUT_BIB" -o "$OUTPUT_BIB"

# 📌 STEP 3: Ensure Proper Capitalization (Prevent Lowercasing)
echo "🔍 Fixing Capitalization..."
sed -i '' -E 's/(title|journal|booktitle|author|editor) = {([^}]+)}/\1 = {{\2}}/g' "$OUTPUT_BIB"

# 📌 STEP 4: Ensure URLs are Correctly Formatted
echo "🔍 Checking URL formatting..."
sed -i '' -E 's/(url|doi) = "([^"]+)"/\1 = {{\2}}/g' "$OUTPUT_BIB"

# 📌 STEP 5: Sort Entries Alphabetically
echo "🔍 Sorting BibTeX entries..."
bibtool -s -- 'sort.format="{key}\n"' -i "$OUTPUT_BIB" -o "$OUTPUT_BIB.sorted"
mv "$OUTPUT_BIB.sorted" "$OUTPUT_BIB"

# 📌 STEP 6: Fetch Missing Metadata with BetterBib (Optional)
if command -v betterbib &> /dev/null; then
  echo "🔍 Running BetterBib to update metadata..."
  betterbib update -i "$OUTPUT_BIB"
else
  echo "⚠️ BetterBib not installed. Skipping metadata updates."
fi

# 📌 STEP 7: Show Summary of Changes
echo "✅ Cleaning Complete! Output saved to: $OUTPUT_BIB"

