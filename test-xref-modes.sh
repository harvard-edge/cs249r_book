#!/bin/bash

# Test different cross-reference placement strategies

echo "🧪 Testing Cross-Reference Placement Strategies"
echo "=============================================="

# Function to test a mode
test_mode() {
    local mode=$1
    local description=$2
    
    echo ""
    echo "Testing: $description (Mode: $mode)"
    echo "----------------------------------------"
    
    # Set environment variable for the mode
    export XREF_MODE=$mode
    
    # Temporarily swap filters
    mv quarto/filters/inject-xrefs.lua quarto/filters/inject-xrefs-original.lua
    cp quarto/filters/inject-xrefs-experimental.lua quarto/filters/inject-xrefs.lua
    
    # Run the build
    cd quarto
    quarto render contents/core/introduction/introduction.qmd --to pdf --pdf-engine lualatex \
        --output ../test-output-$mode.pdf 2>&1 | grep -E "Cross-Ref|Experiment"
    cd ..
    
    # Restore original filter
    mv quarto/filters/inject-xrefs-original.lua quarto/filters/inject-xrefs.lua
    
    echo "✅ Output saved to: test-output-$mode.pdf"
}

# Test different modes
test_mode "chapter_only" "Chapter-level connections only"
test_mode "section_only" "Section-level connections only (original behavior)"
test_mode "hybrid" "Hybrid approach (chapter + selective sections)"
test_mode "priority_based" "Priority-based filtering"

echo ""
echo "🎯 Experiment Complete!"
echo "Compare the PDFs to see which approach works best:"
echo "  - test-output-chapter_only.pdf"
echo "  - test-output-section_only.pdf" 
echo "  - test-output-hybrid.pdf"
echo "  - test-output-priority_based.pdf"