#!/usr/bin/env python3
"""
Cross-platform GitHub Actions workflow validator
Works on macOS, Linux, and Windows
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

def validate_all_workflows():
    """Validate all workflow files"""
    workflows_dir = Path('.github/workflows')
    
    if not workflows_dir.exists():
        print("❌ .github/workflows directory not found")
        return False
    
    workflow_files = list(workflows_dir.glob('*.yml')) + list(workflows_dir.glob('*.yaml'))
    
    if not workflow_files:
        print("❌ No workflow files found in .github/workflows/")
        return False
    
    print(f"🔍 Found {len(workflow_files)} workflow files")
    
    all_valid = True
    for workflow_file in workflow_files:
        if not validate_workflow(str(workflow_file)):
            all_valid = False
    
    return all_valid

def validate_quarto_files():
    """Validate Quarto YAML files"""
    print("\n📚 Validating Quarto YAML files...")
    
    # Find all Quarto YAML files
    quarto_files = []
    
    # Check for _quarto.yml files
    quarto_files.extend(Path('.').glob('**/_quarto.yml'))
    quarto_files.extend(Path('.').glob('**/_quarto.yaml'))
    
    # Check for config files
    config_dir = Path('book/config')
    if config_dir.exists():
        quarto_files.extend(config_dir.glob('*.yml'))
        quarto_files.extend(config_dir.glob('*.yaml'))
    
    if not quarto_files:
        print("⚠️ No Quarto YAML files found")
        return True
    
    print(f"🔍 Found {len(quarto_files)} Quarto YAML files")
    
    all_valid = True
    for quarto_file in quarto_files:
        if not validate_workflow(str(quarto_file)):
            all_valid = False
    
    return all_valid

def main():
    """Main validation function"""
    print("🚀 GitHub Actions Workflow Validator")
    print("=" * 40)
    
    # Check if yamllint is available
    if not check_yamllint():
        print("📦 yamllint not found. Installing...")
        if not install_yamllint():
            print("❌ Please install yamllint manually: pip install yamllint")
            sys.exit(1)
    
    # Validate workflows
    workflows_valid = validate_all_workflows()
    
    # Validate Quarto files
    quarto_valid = validate_quarto_files()
    
    if workflows_valid and quarto_valid:
        print("\n🎉 All YAML files are valid!")
        sys.exit(0)
    else:
        print("\n❌ Some YAML files have issues. Please fix them.")
        sys.exit(1)

if __name__ == "__main__":
    main() 