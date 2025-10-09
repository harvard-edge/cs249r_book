#!/usr/bin/env python3
"""
Generate comprehensive release notes for the Machine Learning Systems textbook.

This script helps create detailed release notes by analyzing:
- Git commits since last release
- Changed files and directories
- Content updates and improvements
- Technical changes and infrastructure updates
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

def get_git_commits_since_tag(tag):
    """Get commits since the specified tag"""
    try:
        result = subprocess.run(
            ['git', 'log', f'{tag}..HEAD', '--oneline', '--no-merges'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        return []

def get_changed_files_since_tag(tag):
    """Get list of changed files since the specified tag"""
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', f'{tag}..HEAD'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        return []

def categorize_changes(files):
    """Categorize changed files by type"""
    categories = {
        'content': [],
        'infrastructure': [],
        'documentation': [],
        'labs': [],
        'scripts': [],
        'workflows': [],
        'other': []
    }
    
    for file in files:
        if not file:
            continue
            
        if file.startswith('contents/core/') or file.startswith('contents/frontmatter/'):
            categories['content'].append(file)
        elif file.startswith('contents/labs/'):
            categories['labs'].append(file)
        elif file.startswith('.github/workflows/') or file.startswith('tools/scripts/'):
            categories['workflows'].append(file)
        elif file.startswith('docs/') or file.endswith('.md'):
            categories['documentation'].append(file)
        elif file.startswith('binder') or file.startswith('netlify.toml') or file.startswith('.gitignore'):
            categories['infrastructure'].append(file)
        elif file.startswith('tools/'):
            categories['scripts'].append(file)
        else:
            categories['other'].append(file)
    
    return categories

def analyze_commit_messages(commits):
    """Analyze commit messages for key themes"""
    themes = {
        'content': [],
        'infrastructure': [],
        'bugfixes': [],
        'features': [],
        'documentation': [],
        'other': []
    }
    
    for commit in commits:
        if not commit:
            continue
            
        msg = commit.split(' ', 1)[1] if ' ' in commit else commit
        
        if any(keyword in msg.lower() for keyword in ['fix', 'bug', 'error', 'issue']):
            themes['bugfixes'].append(commit)
        elif any(keyword in msg.lower() for keyword in ['feat', 'add', 'new', 'implement']):
            themes['features'].append(commit)
        elif any(keyword in msg.lower() for keyword in ['content', 'chapter', 'section']):
            themes['content'].append(commit)
        elif any(keyword in msg.lower() for keyword in ['workflow', 'ci', 'deploy', 'build']):
            themes['infrastructure'].append(commit)
        elif any(keyword in msg.lower() for keyword in ['doc', 'readme', 'guide']):
            themes['documentation'].append(commit)
        else:
            themes['other'].append(commit)
    
    return themes

def generate_release_notes(version, description, previous_version):
    """Generate comprehensive release notes"""
    
    print(f"🔍 Analyzing changes since {previous_version}...")
    
    # Get commits and files
    commits = get_git_commits_since_tag(previous_version)
    files = get_changed_files_since_tag(previous_version)
    
    # Analyze changes
    file_categories = categorize_changes(files)
    commit_themes = analyze_commit_messages(commits)
    
    # Generate release notes
    notes = f"""## 📚 Release {version}

**{description}**

### 📋 Release Information
- **Type**: Release
- **Previous Version**: {previous_version}
- **Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Changes**: {len(commits)} commits, {len(files)} files modified

### 📝 Summary of Changes

"""
    
    # Content changes
    if file_categories['content']:
        notes += "#### 📖 Content Updates\n"
        notes += f"- **Modified chapters**: {len([f for f in file_categories['content'] if f.endswith('.qmd')])} files\n"
        notes += f"- **Bibliography updates**: {len([f for f in file_categories['content'] if f.endswith('.bib')])} files\n"
        notes += "\n"
    
    # Lab changes
    if file_categories['labs']:
        notes += "#### 🧪 Lab Updates\n"
        notes += f"- **Lab materials**: {len(file_categories['labs'])} files modified\n"
        notes += "\n"
    
    # Infrastructure changes
    if file_categories['infrastructure'] or file_categories['workflows']:
        notes += "#### 🔧 Infrastructure & Workflow Updates\n"
        if file_categories['workflows']:
            notes += f"- **CI/CD workflows**: {len(file_categories['workflows'])} files updated\n"
        if file_categories['infrastructure']:
            notes += f"- **Build system**: {len(file_categories['infrastructure'])} files modified\n"
        notes += "\n"
    
    # Documentation changes
    if file_categories['documentation']:
        notes += "#### 📚 Documentation Updates\n"
        notes += f"- **Documentation**: {len(file_categories['documentation'])} files updated\n"
        notes += "\n"
    
    # Script changes
    if file_categories['scripts']:
        notes += "#### 🛠️ Tool & Script Updates\n"
        notes += f"- **Tools and scripts**: {len(file_categories['scripts'])} files modified\n"
        notes += "\n"
    
    # Key commits
    if commits:
        notes += "#### 🔑 Key Changes\n"
        for i, commit in enumerate(commits[:10], 1):  # Show first 10 commits
            notes += f"{i}. {commit}\n"
        if len(commits) > 10:
            notes += f"... and {len(commits) - 10} more commits\n"
        notes += "\n"
    
    # Quick links
    notes += """### 🔗 Quick Links
- 🌐 [Read Online](https://mlsysbook.ai)
- 📄 [Download PDF](https://mlsysbook.ai/pdf)
- 🧪 [Labs & Exercises](https://mlsysbook.ai/labs)
- 📚 [GitHub Repository](https://github.com/harvard-edge/cs249r_book)

### 📊 Technical Details
- **Build System**: Quarto with custom extensions
- **Deployment**: GitHub Pages + Netlify
- **PDF Generation**: LaTeX with compression
- **Content**: Markdown with interactive elements

---
*Generated automatically by the release notes generator*
"""
    
    return notes

def main():
    """Main function"""
    if len(sys.argv) < 4:
        print("Usage: python generate_release_notes.py <version> <description> <previous_version>")
        print("Example: python generate_release_notes.py v1.2.0 'Add new chapter on TinyML' v1.1.0")
        sys.exit(1)
    
    version = sys.argv[1]
    description = sys.argv[2]
    previous_version = sys.argv[3]
    
    print(f"📝 Generating release notes for {version}...")
    print(f"📋 Description: {description}")
    print(f"🔄 Previous version: {previous_version}")
    print()
    
    notes = generate_release_notes(version, description, previous_version)
    
    # Save to file
    output_file = f"release_notes_{version}.md"
    with open(output_file, 'w') as f:
        f.write(notes)
    
    print(f"✅ Release notes saved to: {output_file}")
    print()
    print("📄 Preview:")
    print("=" * 50)
    print(notes)
    print("=" * 50)
    print()
    print("💡 Next steps:")
    print("1. Review and edit the release notes")
    print("2. Copy the content to your GitHub release")
    print("3. Publish the release")

if __name__ == "__main__":
    main() 