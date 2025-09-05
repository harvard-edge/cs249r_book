#!/usr/bin/env python3
"""
Claude Code /improve command implementation
Handles large files intelligently with chunking and creates small, reviewable PRs

Usage in Claude Code:
    /improve introduction.qmd
    /improve frameworks.qmd --max-prs 3
    /improve efficient_ai.qmd --chunk-lines 300
"""

import json
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import asyncio

class SmartChunker:
    """Intelligently chunk files based on structure"""
    
    def __init__(self, file_path: str, max_chunk_lines: int = 400):
        self.file_path = Path(file_path)
        self.max_chunk_lines = max_chunk_lines
        self.content = self.file_path.read_text()
        self.lines = self.content.split('\n')
        
    def get_semantic_chunks(self) -> List[Dict]:
        """Split file into semantically meaningful chunks"""
        
        chunks = []
        current_chunk = {
            'lines': [],
            'start_line': 1,
            'section': None,
            'subsections': []
        }
        
        for i, line in enumerate(self.lines, 1):
            # Detect section headers
            if line.startswith('## '):
                # Save current chunk if it has content
                if current_chunk['lines']:
                    current_chunk['end_line'] = i - 1
                    current_chunk['content'] = '\n'.join(current_chunk['lines'])
                    chunks.append(current_chunk)
                
                # Start new chunk
                current_chunk = {
                    'lines': [line],
                    'start_line': i,
                    'section': line.strip('# ').strip(),
                    'subsections': []
                }
            elif line.startswith('### '):
                current_chunk['subsections'].append(line.strip('# ').strip())
                current_chunk['lines'].append(line)
            else:
                current_chunk['lines'].append(line)
            
            # Split if chunk gets too large
            if len(current_chunk['lines']) >= self.max_chunk_lines:
                # Find good breaking point (end of paragraph)
                break_point = self._find_break_point(current_chunk['lines'])
                
                if break_point:
                    # Save first part
                    chunk_to_save = {
                        **current_chunk,
                        'lines': current_chunk['lines'][:break_point],
                        'end_line': current_chunk['start_line'] + break_point - 1,
                        'content': '\n'.join(current_chunk['lines'][:break_point])
                    }
                    chunks.append(chunk_to_save)
                    
                    # Continue with rest
                    current_chunk = {
                        'lines': current_chunk['lines'][break_point:],
                        'start_line': current_chunk['start_line'] + break_point,
                        'section': current_chunk['section'] + ' (continued)',
                        'subsections': []
                    }
        
        # Don't forget last chunk
        if current_chunk['lines']:
            current_chunk['end_line'] = len(self.lines)
            current_chunk['content'] = '\n'.join(current_chunk['lines'])
            chunks.append(current_chunk)
        
        return chunks
    
    def _find_break_point(self, lines: List[str], target: int = None) -> Optional[int]:
        """Find a good breaking point (paragraph end) near target"""
        
        if target is None:
            target = self.max_chunk_lines - 50
        
        # Look for paragraph breaks near target
        for i in range(target, min(len(lines), target + 50)):
            if i < len(lines) - 1 and lines[i] == '' and lines[i+1] != '':
                return i + 1
        
        # If no good break, just use target
        return target


class PRGenerator:
    """Generate small, focused PRs"""
    
    def __init__(self, chapter_name: str, base_branch: str = 'main'):
        self.chapter_name = chapter_name
        self.base_branch = base_branch
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
    def create_pr_batches(self, all_improvements: List[Dict], max_changes_per_pr: int = 5) -> List[Dict]:
        """Group improvements into PR batches"""
        
        # Group by priority and location
        batches = []
        
        # First PR: Critical issues
        critical = [i for i in all_improvements if i.get('priority') == 'CRITICAL']
        if critical:
            batches.append({
                'type': 'critical',
                'improvements': critical[:max_changes_per_pr],
                'description': 'Fix critical comprehension barriers'
            })
        
        # Second PR: High priority by section
        high_priority = [i for i in all_improvements if i.get('priority') == 'HIGH']
        sections = {}
        for improvement in high_priority:
            section = improvement.get('section', 'General')
            if section not in sections:
                sections[section] = []
            sections[section].append(improvement)
        
        for section, improvements in sections.items():
            if improvements:
                batches.append({
                    'type': 'high_priority',
                    'improvements': improvements[:max_changes_per_pr],
                    'description': f'Improve {section} section clarity'
                })
        
        # Additional PRs: Medium priority
        medium = [i for i in all_improvements if i.get('priority') == 'MEDIUM']
        for i in range(0, len(medium), max_changes_per_pr):
            batch = medium[i:i+max_changes_per_pr]
            if batch:
                batches.append({
                    'type': 'medium_priority',
                    'improvements': batch,
                    'description': 'Minor clarity improvements'
                })
        
        return batches
    
    def create_pr_branch(self, batch_num: int, batch_type: str) -> str:
        """Create a branch for a PR"""
        
        branch_name = f"improve-{self.chapter_name}-{batch_type}-{batch_num}-{self.timestamp}"
        
        # Create and checkout branch
        subprocess.run(['git', 'checkout', self.base_branch], capture_output=True)
        subprocess.run(['git', 'checkout', '-b', branch_name], capture_output=True)
        
        return branch_name
    
    def generate_pr_description(self, batch: Dict) -> str:
        """Generate PR description"""
        
        lines = [
            f"## 📚 Textbook Improvement: {self.chapter_name}",
            "",
            f"**Type:** {batch['description']}",
            f"**Changes:** {len(batch['improvements'])}",
            "",
            "### Issues Addressed:",
            ""
        ]
        
        for imp in batch['improvements']:
            lines.append(f"- **Line {imp.get('line_start', '?')}-{imp.get('line_end', '?')}**: {imp.get('issue', 'Issue')}")
            if imp.get('consensus_level', 0) > 1:
                lines.append(f"  - Reported by {imp['consensus_level']} reviewers")
        
        lines.extend([
            "",
            "### Review Checklist:",
            "- [ ] Content accuracy preserved",
            "- [ ] Academic tone maintained",
            "- [ ] Improvements enhance clarity",
            "- [ ] No new issues introduced",
            "",
            "---",
            "*Generated by /improve command with multi-perspective review*"
        ])
        
        return '\n'.join(lines)


async def improve_chapter_async(chapter_path: str, config: Dict = None):
    """
    Main async function for Claude Code to run improvements
    This is what gets called when you type /improve
    """
    
    config = config or {}
    chapter_file = Path(chapter_path)
    chapter_name = chapter_file.stem
    
    print(f"\n🚀 Starting /improve command for: {chapter_name}.qmd")
    print("="*60)
    
    # Step 1: Smart chunking
    print("📄 Analyzing file structure...")
    chunker = SmartChunker(chapter_file, max_chunk_lines=config.get('chunk_lines', 400))
    chunks = chunker.get_semantic_chunks()
    print(f"   Split into {len(chunks)} semantic chunks")
    
    # Step 2: Review chunks with Task subagents
    print("\n🔍 Running multi-perspective review...")
    all_improvements = []
    
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}/{len(chunks)}: {chunk.get('section', 'Unknown')} (lines {chunk['start_line']}-{chunk['end_line']})")
        
        # Here we would call Task subagents
        # For now, showing the structure:
        improvements = await review_chunk_with_agents(chunk, chapter_file)
        all_improvements.extend(improvements)
        
        # Show progress
        critical_count = sum(1 for imp in improvements if imp.get('priority') == 'CRITICAL')
        if critical_count > 0:
            print(f"     ⚠️  Found {critical_count} critical issues")
    
    print(f"\n📊 Total improvements identified: {len(all_improvements)}")
    
    # Step 3: Generate PR batches
    print("\n🔀 Creating pull request batches...")
    pr_generator = PRGenerator(chapter_name)
    pr_batches = pr_generator.create_pr_batches(
        all_improvements, 
        max_changes_per_pr=config.get('max_changes_per_pr', 5)
    )
    print(f"   Will create {len(pr_batches)} PRs")
    
    # Step 4: Create PRs
    created_prs = []
    for i, batch in enumerate(pr_batches, 1):
        print(f"\n📝 Creating PR {i}/{len(pr_batches)}: {batch['description']}")
        
        # Create branch
        branch_name = pr_generator.create_pr_branch(i, batch['type'])
        
        # Apply improvements
        apply_improvements_to_file(chapter_file, batch['improvements'])
        
        # Commit
        commit_message = f"feat({chapter_name}): {batch['description']}\n\n- {len(batch['improvements'])} improvements\n- Auto-generated by /improve"
        subprocess.run(['git', 'add', str(chapter_file)], capture_output=True)
        subprocess.run(['git', 'commit', '-m', commit_message], capture_output=True)
        
        # Generate PR description
        pr_description = pr_generator.generate_pr_description(batch)
        
        created_prs.append({
            'branch': branch_name,
            'description': pr_description,
            'improvements': len(batch['improvements'])
        })
        
        print(f"   ✅ Branch created: {branch_name}")
    
    # Step 5: Summary
    print("\n" + "="*60)
    print("✅ /improve COMMAND COMPLETE")
    print("="*60)
    print(f"📚 Chapter: {chapter_name}")
    print(f"📊 Improvements: {len(all_improvements)}")
    print(f"🔀 PRs created: {len(created_prs)}")
    print("\n📋 Next steps:")
    for i, pr in enumerate(created_prs, 1):
        print(f"   {i}. Review branch: {pr['branch']} ({pr['improvements']} changes)")
    print("\nPush branches with: git push origin <branch-name>")
    print("Then create PRs on GitHub for review")
    
    return {
        'success': True,
        'chapter': chapter_name,
        'improvements': len(all_improvements),
        'prs': created_prs
    }


async def review_chunk_with_agents(chunk: Dict, chapter_file: Path) -> List[Dict]:
    """
    Review a chunk using Claude Task subagents
    This is where the actual Task subagent calls would go
    """
    
    # Structure for Task subagent call:
    # result = await Task(
    #     subagent_type="general-purpose",
    #     description=f"Review textbook chunk as multiple students",
    #     prompt=create_review_prompt(chunk)
    # )
    
    # For now, return mock improvements
    return [
        {
            'line_start': chunk['start_line'],
            'line_end': chunk['start_line'] + 10,
            'priority': 'HIGH',
            'issue': 'Undefined technical term',
            'suggestion': 'Add definition',
            'consensus_level': 2,
            'section': chunk.get('section', 'Unknown')
        }
    ]


def apply_improvements_to_file(file_path: Path, improvements: List[Dict]):
    """
    Apply improvements to the file
    In real implementation, this would make actual edits
    """
    # This would use Edit or MultiEdit tools to apply fixes
    pass


# Entry point for Claude Code
def improve_command_handler(args: str) -> Dict:
    """
    Handler for /improve command in Claude Code
    
    Args:
        args: Command arguments (e.g., "introduction.qmd --max-prs 3")
    
    Returns:
        Result dictionary
    """
    
    # Parse arguments
    parts = args.split()
    if not parts:
        return {
            'error': 'No file specified',
            'usage': '/improve <chapter.qmd> [--max-prs N] [--chunk-lines N]'
        }
    
    chapter_file = parts[0]
    
    # Parse optional arguments
    config = {}
    i = 1
    while i < len(parts):
        if parts[i] == '--max-prs' and i + 1 < len(parts):
            config['max_changes_per_pr'] = int(parts[i + 1])
            i += 2
        elif parts[i] == '--chunk-lines' and i + 1 < len(parts):
            config['chunk_lines'] = int(parts[i + 1])
            i += 2
        else:
            i += 1
    
    # Run improvement
    try:
        result = asyncio.run(improve_chapter_async(chapter_file, config))
        return result
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }


if __name__ == "__main__":
    # Test the command
    import sys
    if len(sys.argv) > 1:
        result = improve_command_handler(' '.join(sys.argv[1:]))
        print(json.dumps(result, indent=2))