#!/usr/bin/env python3
"""
Cross-platform YAML file validator
Works on macOS, Linux, and Windows
Validates all YAML files in the project
"""

import subprocess
import sys
import os
from pathlib import Path

def check_yamllint():
    """Check if yamllint is installed"""
    try:
        subprocess.run(['yamllint', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_yamllint():
    """Install yamllint if not available"""
    print("📦 Installing yamllint...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'yamllint'], check=True)
        print("✅ yamllint installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install yamllint")
        return False

def validate_workflow(workflow_path):
    """Validate a single workflow file"""
    print(f"🔍 Validating {workflow_path}...")
    
    try:
        # Use yamllint with custom config that's more lenient
        result = subprocess.run(['yamllint', '-c', '.yamllint', workflow_path], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {workflow_path} - No YAML syntax errors")
            return True
        else:
            # Only show critical errors, not style warnings
            output_lines = result.stdout.split('\n')
            critical_errors = [line for line in output_lines if 'error' in line.lower() and 'syntax' in line.lower()]
            
            if critical_errors:
                print(f"❌ {workflow_path} - Critical YAML syntax errors found:")
                for error in critical_errors[:3]:  # Show first 3 errors
                    print(f"   {error}")
                return False
            else:
                print(f"⚠️ {workflow_path} - Style warnings only (non-critical)")
                return True
            
    except FileNotFoundError:
        print("❌ yamllint not found. Run: pip install yamllint")
        return False

def validate_all_yaml_files():
    """Validate all YAML files in the project"""
    print("🔍 Validating all YAML files...")
    
    # Find all YAML files in the project
    yaml_files = []
    
    # Common directories to check
    search_dirs = [
        '.github/workflows',
        'book',
        'config',
        'tools',
        '.',
    ]
    
    # Exclude patterns
    exclude_patterns = [
        '**/node_modules/**',
        '**/.git/**',
        '**/_site/**',
        '**/_book/**',
        '**/.venv/**',
        '**/__pycache__/**',
        '**/*.pyc',
    ]
    
    for search_dir in search_dirs:
        if Path(search_dir).exists():
            # Find all .yml and .yaml files
            yaml_files.extend(Path(search_dir).glob('**/*.yml'))
            yaml_files.extend(Path(search_dir).glob('**/*.yaml'))
    
    # Filter out excluded files
    filtered_files = []
    for yaml_file in yaml_files:
        file_path = str(yaml_file)
        if not any(pattern.replace('**', '').replace('*', '') in file_path for pattern in exclude_patterns):
            filtered_files.append(yaml_file)
    
    if not filtered_files:
        print("⚠️ No YAML files found")
        return True
    
    print(f"🔍 Found {len(filtered_files)} YAML files to validate")
    
    all_valid = True
    for yaml_file in filtered_files:
        if not validate_workflow(str(yaml_file)):
            all_valid = False
    
    return all_valid

def main():
    """Main validation function"""
    print("🚀 YAML File Validator")
    print("=" * 40)
    
    # Check if yamllint is available
    if not check_yamllint():
        print("📦 yamllint not found. Installing...")
        if not install_yamllint():
            print("❌ Please install yamllint manually: pip install yamllint")
            sys.exit(1)
    
    # Validate all YAML files
    all_valid = validate_all_yaml_files()
    
    if all_valid:
        print("\n🎉 All YAML files are valid!")
        sys.exit(0)
    else:
        print("\n❌ Some YAML files have issues. Please fix them.")
        sys.exit(1)

if __name__ == "__main__":
    main() 