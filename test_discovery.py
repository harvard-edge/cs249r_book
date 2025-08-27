#!/usr/bin/env python3
"""
Quick test script for the ChapterDiscovery class.
"""

import sys
from pathlib import Path

# Add cli to path
sys.path.insert(0, str(Path(__file__).parent / "cli"))

from core.discovery import ChapterDiscovery

def test_chapter_discovery():
    """Test basic ChapterDiscovery functionality."""
    print("🧪 Testing ChapterDiscovery...")
    
    # Initialize discovery
    book_dir = Path.cwd() / "quarto"
    discovery = ChapterDiscovery(book_dir)
    
    # Test finding specific chapters
    test_chapters = ["introduction", "ml_systems", "training", "ops"]
    
    for chapter_name in test_chapters:
        chapter_file = discovery.find_chapter_file(chapter_name)
        if chapter_file:
            print(f"✅ Found {chapter_name}: {chapter_file.relative_to(book_dir)}")
        else:
            print(f"❌ Not found: {chapter_name}")
    
    # Test getting all chapters
    all_chapters = discovery.get_all_chapters()
    print(f"\n📚 Total chapters found: {len(all_chapters)}")
    
    # Show first few chapters
    print("\n📋 First 5 chapters:")
    for chapter in all_chapters[:5]:
        print(f"  - {chapter['name']} ({chapter['directory']})")
    
    # Test chapter validation
    print("\n🔍 Testing chapter validation...")
    try:
        valid_chapters = discovery.validate_chapters(["introduction", "ml_systems"])
        print(f"✅ Validated {len(valid_chapters)} chapters")
    except FileNotFoundError as e:
        print(f"❌ Validation error: {e}")
    
    print("\n✅ ChapterDiscovery tests completed!")

if __name__ == "__main__":
    test_chapter_discovery()
