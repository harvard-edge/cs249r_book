#!/usr/bin/env python3
"""
Quick test script for the BuildCommand class.
"""

import sys
from pathlib import Path

# Add cli to path
sys.path.insert(0, str(Path(__file__).parent / "cli"))

from core.config import ConfigManager
from core.discovery import ChapterDiscovery
from commands.build import BuildCommand

def test_build_command():
    """Test basic BuildCommand functionality."""
    print("🧪 Testing BuildCommand...")
    
    # Initialize components
    root_dir = Path.cwd()
    config_manager = ConfigManager(root_dir)
    chapter_discovery = ChapterDiscovery(config_manager.book_dir)
    build_command = BuildCommand(config_manager, chapter_discovery)
    
    print("✅ BuildCommand initialized successfully")
    
    # Test chapter validation (without actually building)
    test_chapters = ["introduction"]
    
    try:
        chapter_files = chapter_discovery.validate_chapters(test_chapters)
        print(f"✅ Validated chapters: {[f.name for f in chapter_files]}")
    except Exception as e:
        print(f"❌ Chapter validation error: {e}")
        return
    
    # Test configuration setup
    for format_type in ["html", "pdf", "epub"]:
        try:
            config_manager.setup_symlink(format_type)
            output_dir = config_manager.get_output_dir(format_type)
            print(f"✅ {format_type.upper()} config setup: {output_dir}")
        except Exception as e:
            print(f"❌ {format_type.upper()} config error: {e}")
    
    print("\n✅ BuildCommand tests completed!")
    print("💡 Note: Actual builds not tested to avoid long execution times")

if __name__ == "__main__":
    test_build_command()
