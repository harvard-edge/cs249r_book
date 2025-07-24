#!/bin/bash

# Comprehensive Source Citation Standardization Script
# This script standardizes all source citations in QMD files

echo "🔧 Starting source citation standardization..."

# 1. Convert asterisk-wrapped sources with academic citations
echo "Converting *Source: @citation* to Source: [@citation]."
find contents -name "*.qmd" -exec sed -i '' 's/\*Source: @\([^*]*\)\*/Source: [@\1]./g' {} \;

# 2. Convert asterisk-wrapped sources with links  
echo "Converting *Source: [text](url)* to Source: [text](url)."
find contents -name "*.qmd" -exec sed -i '' 's/\*Source: \(\[[^]]*\]([^)]*)\)\*/Source: \1./g' {} \;

# 3. Convert asterisk-wrapped sources with plain text
echo "Converting *Source: text* to Source: text."
find contents -name "*.qmd" -exec sed -i '' 's/\*Source: \([^*]*\)\*/Source: \1./g' {} \;

# 4. Standardize academic citations without brackets to include brackets
echo "Converting Source: @citation to Source: [@citation]."
find contents -name "*.qmd" -exec sed -i '' 's/Source: @\([a-zA-Z0-9][^.]*\)\./Source: [@\1]./g' {} \;

# 5. Add periods to sources that are missing them (company names, etc.)
echo "Adding periods to sources missing punctuation..."
find contents -name "*.qmd" -exec sed -i '' 's/Source: \([^.@\[]*[^.]\)$/Source: \1./g' {} \;

# 6. Clean up table sources in curly braces
echo "Standardizing table source citations..."
find contents -name "*.qmd" -exec sed -i '' 's/{Source: \([^}]*\)};/Source: \1./g' {} \;

# 7. Clean up any double periods
echo "Cleaning up double periods..."
find contents -name "*.qmd" -exec sed -i '' 's/Source: \([^.]*\)\.\./Source: \1./g' {} \;

# 8. Fix any remaining formatting issues
echo "Final cleanup..."
find contents -name "*.qmd" -exec sed -i '' 's/Source: \[\[@/Source: [@/g' {} \;

echo "✅ Source citation standardization complete!"
echo ""
echo "📊 Summary of standard formats applied:"
echo "  • Academic citations: Source: [@citation]."
echo "  • Company sources: Source: Company Name."
echo "  • Link sources: Source: [Text](URL)."
echo ""
echo "🔍 To verify results, run:"
echo "  grep -r 'Source:' contents --include='*.qmd' | head -20" 