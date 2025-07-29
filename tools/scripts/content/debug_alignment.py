#!/usr/bin/env python3
"""
Debug script to understand alignment detection issues
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from format_grid_tables import GridTableAnalyzer

def debug_alignment_detection():
    """Debug the alignment detection logic."""
    
    analyzer = GridTableAnalyzer()
    
    # Test the bit-width column values
    bit_width_values = ["32-bit", "16-bit", "8-bit"]
    
    print("Testing bit-width column:")
    print(f"Values: {bit_width_values}")
    
    analysis = analyzer.analyze_column_content(bit_width_values)
    print(f"Analysis result: {analysis}")
    
    print("\nTesting pattern matching:")
    for value in bit_width_values:
        print(f"\nValue: '{value}'")
        
        # Test numeric patterns
        for pattern, subtype in analyzer.numeric_patterns:
            import re
            if re.match(pattern, value, re.IGNORECASE):
                print(f"  ✅ Matches NUMERIC pattern: {pattern} ({subtype})")
            else:
                print(f"  ❌ No match for NUMERIC pattern: {pattern}")
        
        # Test categorical patterns  
        for pattern, subtype in analyzer.categorical_patterns:
            import re
            if re.match(pattern, value, re.IGNORECASE):
                print(f"  ✅ Matches CATEGORICAL pattern: {pattern} ({subtype})")
            else:
                print(f"  ❌ No match for CATEGORICAL pattern: {pattern}")

if __name__ == "__main__":
    debug_alignment_detection() 