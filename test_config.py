#!/usr/bin/env python3
"""
Quick test script for the new ConfigManager.
"""

import sys
from pathlib import Path

# Add cli to path
sys.path.insert(0, str(Path(__file__).parent / "cli"))

from core.config import ConfigManager

def test_config_manager():
    """Test basic ConfigManager functionality."""
    print("🧪 Testing ConfigManager...")
    
    # Initialize config manager
    config_manager = ConfigManager(Path.cwd())
    
    # Test config file paths
    print(f"✅ HTML config: {config_manager.html_config}")
    print(f"✅ PDF config: {config_manager.pdf_config}")
    print(f"✅ EPUB config: {config_manager.epub_config}")
    
    # Test config file existence
    for format_type in ["html", "pdf", "epub"]:
        config_file = config_manager.get_config_file(format_type)
        exists = config_file.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {format_type.upper()} config exists: {exists}")
    
    # Test output directory detection
    for format_type in ["html", "pdf", "epub"]:
        try:
            output_dir = config_manager.get_output_dir(format_type)
            print(f"✅ {format_type.upper()} output dir: {output_dir}")
        except Exception as e:
            print(f"❌ {format_type.upper()} output dir error: {e}")
    
    # Test symlink status
    print("\n🔗 Current symlink status:")
    config_manager.show_symlink_status()
    
    print("\n✅ ConfigManager tests completed!")

if __name__ == "__main__":
    test_config_manager()
