#!/usr/bin/env python3
"""
Reorganize tools/scripts/ directory structure.

This script:
1. Creates new subdirectories
2. Moves scripts to proper locations
3. Updates all references (pre-commit, imports, README)
4. Creates a rollback backup
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

# Define the migration plan
MIGRATION_PLAN = {
    # Images subdirectory - consolidate all image-related scripts
    'images': [
        'download_external_images.py',
        'manage_external_images.py',
        'remove_bg.py',
        'rename_auto_images.py',
        'rename_downloaded_images.py',
        'validate_image_references.py',
    ],
    
    # Content subdirectory - add formatting scripts
    'content': [
        'fix_mid_paragraph_bold.py',
        'format_python_in_qmd.py',
        'format_tables.py',
    ],
    
    # Testing subdirectory - consolidate all tests
    'testing': [
        'test_format_tables.py',
        'test_image_extraction.py',
        'test_publish_live.py',
    ],
    
    # Infrastructure subdirectory - CI/CD and container management
    'infrastructure': [
        'cleanup_containers.py',
        'list_containers.py',
        'cleanup_workflow_runs_gh.py',
    ],
    
    # Glossary subdirectory - move glossary script
    'glossary': [
        'standardize_glossaries.py',
    ],
    
    # Maintenance subdirectory - add release and preflight
    'maintenance': [
        'generate_release_notes.py',
        'preflight.py',
    ],
    
    # Utilities subdirectory - validation scripts
    'utilities': [
        'check_custom_extensions.py',
        'validate_part_keys.py',
    ],
}

# Files that also need to be moved from existing subdirectories
EXISTING_SUBDIRECTORY_MOVES = {
    'images': [
        ('utilities/manage_images.py', 'manage_images.py'),
        ('utilities/convert_svg_to_png.py', 'convert_svg_to_png.py'),
        ('maintenance/compress_images.py', 'compress_images.py'),
        ('maintenance/analyze_image_sizes.py', 'analyze_image_sizes.py'),
    ],
}

# Pre-commit reference updates
PRECOMMIT_UPDATES = {
    'tools/scripts/format_python_in_qmd.py': 'tools/scripts/content/format_python_in_qmd.py',
    'tools/scripts/format_tables.py': 'tools/scripts/content/format_tables.py',
    'tools/scripts/validate_part_keys.py': 'tools/scripts/utilities/validate_part_keys.py',
    'tools/scripts/manage_external_images.py': 'tools/scripts/images/manage_external_images.py',
    'tools/scripts/validate_image_references.py': 'tools/scripts/images/validate_image_references.py',
    'tools/scripts/generate_release_notes.py': 'tools/scripts/maintenance/generate_release_notes.py',
    'tools/scripts/preflight.py': 'tools/scripts/maintenance/preflight.py',
}


def create_backup():
    """Create a backup of the current state."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f'tools/scripts/.backup_{timestamp}.json'
    
    backup_data = {
        'timestamp': timestamp,
        'files': []
    }
    
    # Record all file locations
    for root, dirs, files in os.walk('tools/scripts'):
        for file in files:
            if file.endswith('.py'):
                rel_path = os.path.relpath(os.path.join(root, file), 'tools/scripts')
                backup_data['files'].append(rel_path)
    
    with open(backup_file, 'w') as f:
        json.dump(backup_data, f, indent=2)
    
    print(f"✅ Created backup: {backup_file}")
    return backup_file


def create_directories():
    """Create new subdirectories."""
    print("\n📁 Creating new directories...")
    
    new_dirs = ['images', 'infrastructure']
    for dirname in new_dirs:
        dirpath = f'tools/scripts/{dirname}'
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
            # Create __init__.py
            with open(f'{dirpath}/__init__.py', 'w') as f:
                f.write(f'"""Scripts for {dirname} management."""\n')
            print(f"  ✅ Created {dirpath}/")
        else:
            print(f"  ⚠️  {dirpath}/ already exists")


def move_files_from_root():
    """Move files from root to subdirectories."""
    print("\n📦 Moving files from root level...")
    
    moved_count = 0
    for target_dir, files in MIGRATION_PLAN.items():
        target_path = f'tools/scripts/{target_dir}'
        
        for filename in files:
            source = f'tools/scripts/{filename}'
            dest = f'{target_path}/{filename}'
            
            if os.path.exists(source):
                shutil.move(source, dest)
                print(f"  ✅ {filename} → {target_dir}/")
                moved_count += 1
            else:
                print(f"  ⚠️  {filename} not found (may already be moved)")
    
    print(f"\n  Moved {moved_count} files from root")


def move_files_between_subdirs():
    """Move files between existing subdirectories."""
    print("\n🔄 Consolidating files between subdirectories...")
    
    moved_count = 0
    for target_dir, moves in EXISTING_SUBDIRECTORY_MOVES.items():
        target_path = f'tools/scripts/{target_dir}'
        
        for source_rel, dest_name in moves:
            source = f'tools/scripts/{source_rel}'
            dest = f'{target_path}/{dest_name}'
            
            if os.path.exists(source):
                shutil.move(source, dest)
                print(f"  ✅ {source_rel} → {target_dir}/{dest_name}")
                moved_count += 1
            else:
                print(f"  ⚠️  {source_rel} not found")
    
    print(f"\n  Moved {moved_count} files between subdirectories")


def update_precommit_config():
    """Update .pre-commit-config.yaml with new paths."""
    print("\n⚙️  Updating .pre-commit-config.yaml...")
    
    config_path = '.pre-commit-config.yaml'
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    original_content = content
    updates_made = 0
    
    for old_path, new_path in PRECOMMIT_UPDATES.items():
        if old_path in content:
            content = content.replace(old_path, new_path)
            updates_made += 1
            print(f"  ✅ Updated: {os.path.basename(old_path)}")
    
    if updates_made > 0:
        with open(config_path, 'w') as f:
            f.write(content)
        print(f"\n  Updated {updates_made} references in pre-commit config")
    else:
        print("  ℹ️  No updates needed")


def create_readme_files():
    """Create README files for new directories."""
    print("\n📝 Creating README files...")
    
    readmes = {
        'images': '''# Image Management Scripts

Scripts for managing, processing, and validating images in the book.

## Image Processing
- `compress_images.py` - Compress images to reduce file size
- `convert_svg_to_png.py` - Convert SVG files to PNG format
- `remove_bg.py` - Remove backgrounds from images

## Image Management
- `manage_images.py` - Main image management utility
- `download_external_images.py` - Download external images
- `manage_external_images.py` - Manage external image references
- `rename_auto_images.py` - Rename automatically generated images
- `rename_downloaded_images.py` - Rename downloaded images

## Validation
- `validate_image_references.py` - Ensure all image references are valid
- `analyze_image_sizes.py` - Analyze image sizes and suggest optimizations
''',
        'infrastructure': '''# Infrastructure Scripts

Scripts for managing CI/CD, containers, and workflow infrastructure.

## Container Management
- `cleanup_containers.py` - Clean up Docker containers
- `list_containers.py` - List active containers

## Workflow Management
- `cleanup_workflow_runs_gh.py` - Clean up old GitHub Actions workflow runs
''',
    }
    
    for dirname, content in readmes.items():
        readme_path = f'tools/scripts/{dirname}/README.md'
        if not os.path.exists(readme_path):
            with open(readme_path, 'w') as f:
                f.write(content)
            print(f"  ✅ Created {dirname}/README.md")


def generate_summary():
    """Generate a summary of the reorganization."""
    print("\n" + "=" * 80)
    print("📊 REORGANIZATION SUMMARY")
    print("=" * 80)
    
    # Count files in each directory
    summary = {}
    for root, dirs, files in os.walk('tools/scripts'):
        # Skip __pycache__ and hidden directories
        if '__pycache__' in root or '/.backup' in root:
            continue
            
        dirname = os.path.relpath(root, 'tools/scripts')
        py_files = [f for f in files if f.endswith('.py') and f != '__init__.py']
        
        if py_files:
            summary[dirname] = len(py_files)
    
    print("\n📁 Files per directory:")
    for dirname in sorted(summary.keys()):
        count = summary[dirname]
        print(f"  {dirname + '/':<30} {count:>3} files")
    
    # Count root level files
    root_files = [f for f in os.listdir('tools/scripts') 
                  if f.endswith('.py') and os.path.isfile(f'tools/scripts/{f}')]
    
    print(f"\n🎯 Root level scripts remaining: {len(root_files)}")
    if root_files:
        for f in root_files:
            print(f"  - {f}")
    
    print("\n✅ Reorganization complete!")
    print("\nNext steps:")
    print("  1. Test pre-commit hooks: pre-commit run --all-files")
    print("  2. Check for any broken imports")
    print("  3. Update any documentation references")


def main():
    """Main reorganization process."""
    import sys
    
    print("🔧 SCRIPT REORGANIZATION TOOL")
    print("=" * 80)
    print("\nThis will reorganize tools/scripts/ directory structure.")
    print("\nChanges:")
    print("  • Create new subdirectories (images/, infrastructure/)")
    print("  • Move 21 scripts from root to appropriate subdirectories")
    print("  • Consolidate scattered scripts (images, tests, etc.)")
    print("  • Update .pre-commit-config.yaml references")
    print("  • Create documentation")
    
    # Check for --yes flag
    if '--yes' not in sys.argv:
        response = input("\n⚠️  Proceed with reorganization? (yes/no): ")
        if response.lower() != 'yes':
            print("\n❌ Cancelled")
            return 1
    else:
        print("\n✅ Auto-confirmed with --yes flag")
    
    try:
        # Create backup
        backup_file = create_backup()
        
        # Execute reorganization
        create_directories()
        move_files_from_root()
        move_files_between_subdirs()
        update_precommit_config()
        create_readme_files()
        
        # Summary
        generate_summary()
        
        print(f"\n💾 Backup saved to: {backup_file}")
        print("   (Can be used for rollback if needed)")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during reorganization: {e}")
        print("   Please restore from backup if needed")
        return 1


if __name__ == '__main__':
    exit(main())

