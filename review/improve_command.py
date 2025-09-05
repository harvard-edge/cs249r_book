#!/usr/bin/env python3
"""
/improve command for Claude Code
Automated multi-perspective review and improvement system

Usage:
    /improve introduction.qmd
    /improve frameworks.qmd --chunk-size 500 --max-issues 5
    /improve efficient_ai.qmd --perspectives junior_cs,phd
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import hashlib

class ChunkProcessor:
    """Process large files in manageable chunks"""
    
    def __init__(self, file_path: str, chunk_size: int = 500):
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.content = self.file_path.read_text()
        self.lines = self.content.split('\n')
        
    def get_chunks(self) -> List[Dict]:
        """Split file into overlapping chunks for context"""
        chunks = []
        i = 0
        
        while i < len(self.lines):
            # Get chunk with context
            start = max(0, i - 50)  # 50 lines before for context
            end = min(len(self.lines), i + self.chunk_size)
            
            chunk = {
                'chunk_id': f"chunk_{i//self.chunk_size}",
                'start_line': i + 1,  # Human-readable line numbers
                'end_line': min(i + self.chunk_size, len(self.lines)),
                'context_start': start + 1,
                'content': '\n'.join(self.lines[i:end]),
                'context': '\n'.join(self.lines[start:i]) if start < i else "",
                'section': self._find_section(i)
            }
            chunks.append(chunk)
            
            i += self.chunk_size
            
        return chunks
    
    def _find_section(self, line_idx: int) -> str:
        """Find the current section heading"""
        for i in range(line_idx, -1, -1):
            if self.lines[i].startswith('#'):
                return self.lines[i].strip()
        return "Beginning"

class ImprovementOrchestrator:
    """Main orchestrator for the /improve command"""
    
    def __init__(self, chapter_file: str, config: Dict = None):
        self.chapter_file = Path(chapter_file)
        self.chapter_name = self.chapter_file.stem
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configuration
        self.config = config or {}
        self.chunk_size = self.config.get('chunk_size', 500)
        self.max_issues_per_pr = self.config.get('max_issues', 5)
        self.perspectives = self.config.get('perspectives', ['junior_cs', 'senior_ee', 'phd'])
        
        # Session tracking
        self.session_dir = Path('review/sessions') / f"{self.chapter_name}_{self.timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
    def run_improvement_cycle(self) -> Dict:
        """Main improvement cycle"""
        
        print(f"\n🚀 Starting /improve for: {self.chapter_name}.qmd")
        print(f"📊 Configuration:")
        print(f"   - Chunk size: {self.chunk_size} lines")
        print(f"   - Max issues per PR: {self.max_issues_per_pr}")
        print(f"   - Perspectives: {', '.join(self.perspectives)}")
        print("-" * 60)
        
        # Step 1: Process file in chunks
        processor = ChunkProcessor(self.chapter_file, self.chunk_size)
        chunks = processor.get_chunks()
        print(f"📄 Split into {len(chunks)} chunks for processing")
        
        # Step 2: Review each chunk
        all_issues = []
        for i, chunk in enumerate(chunks):
            print(f"\n🔍 Processing chunk {i+1}/{len(chunks)} (lines {chunk['start_line']}-{chunk['end_line']})")
            chunk_issues = self.review_chunk(chunk)
            all_issues.extend(chunk_issues)
            
            # Save chunk analysis
            chunk_file = self.session_dir / f"chunk_{i}_analysis.json"
            chunk_file.write_text(json.dumps({
                'chunk_info': chunk,
                'issues': chunk_issues
            }, indent=2))
        
        print(f"\n📋 Total issues found: {len(all_issues)}")
        
        # Step 3: Group and prioritize issues
        grouped_issues = self.group_issues(all_issues)
        
        # Step 4: Create PRs
        prs = self.create_prs(grouped_issues)
        
        # Step 5: Save session summary
        self.save_summary(chunks, all_issues, prs)
        
        return {
            'chapter': self.chapter_name,
            'chunks_processed': len(chunks),
            'total_issues': len(all_issues),
            'prs_created': len(prs),
            'session_dir': str(self.session_dir)
        }
    
    def review_chunk(self, chunk: Dict) -> List[Dict]:
        """Review a single chunk with multiple perspectives"""
        
        issues = []
        
        # Simulate multi-perspective review (would call Task subagents)
        for perspective in self.perspectives:
            # In real implementation, this calls Task subagent
            perspective_issues = self.simulate_review(chunk, perspective)
            issues.extend(perspective_issues)
        
        # Deduplicate and consolidate similar issues
        return self.consolidate_issues(issues)
    
    def simulate_review(self, chunk: Dict, perspective: str) -> List[Dict]:
        """Simulate review (replace with actual Task subagent call)"""
        
        # This would be replaced with actual Task subagent call
        return [{
            'perspective': perspective,
            'chunk_id': chunk['chunk_id'],
            'line_range': f"{chunk['start_line']}-{chunk['end_line']}",
            'type': 'undefined_term',
            'severity': 'HIGH',
            'description': f"Mock issue from {perspective}",
            'suggestion': "Add definition"
        }]
    
    def consolidate_issues(self, issues: List[Dict]) -> List[Dict]:
        """Consolidate similar issues from different perspectives"""
        
        consolidated = {}
        
        for issue in issues:
            # Create key for grouping similar issues
            key = f"{issue['line_range']}_{issue['type']}"
            
            if key not in consolidated:
                consolidated[key] = {
                    **issue,
                    'reported_by': [issue['perspective']],
                    'consensus_level': 1
                }
            else:
                consolidated[key]['reported_by'].append(issue['perspective'])
                consolidated[key]['consensus_level'] += 1
        
        return list(consolidated.values())
    
    def group_issues(self, all_issues: List[Dict]) -> List[List[Dict]]:
        """Group issues into PR-sized batches"""
        
        # Sort by consensus level and severity
        sorted_issues = sorted(all_issues, 
                              key=lambda x: (x.get('consensus_level', 1), 
                                           x.get('severity') == 'HIGH'),
                              reverse=True)
        
        # Group into PR batches
        groups = []
        current_group = []
        
        for issue in sorted_issues:
            current_group.append(issue)
            
            if len(current_group) >= self.max_issues_per_pr:
                groups.append(current_group)
                current_group = []
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def create_prs(self, grouped_issues: List[List[Dict]]) -> List[Dict]:
        """Create separate PRs for each group of issues"""
        
        prs = []
        
        for i, issue_group in enumerate(grouped_issues):
            pr_branch = f"improve-{self.chapter_name}-batch-{i+1}-{self.timestamp}"
            
            print(f"\n🔀 Creating PR {i+1}/{len(grouped_issues)}")
            print(f"   Branch: {pr_branch}")
            print(f"   Issues: {len(issue_group)}")
            
            # Create branch
            subprocess.run(['git', 'checkout', '-b', pr_branch], 
                         capture_output=True, text=True)
            
            # Apply fixes for this batch
            fixes_applied = self.apply_fixes(issue_group)
            
            # Stage and commit
            if fixes_applied:
                subprocess.run(['git', 'add', str(self.chapter_file)],
                             capture_output=True, text=True)
                
                commit_msg = self.generate_commit_message(issue_group, i+1)
                subprocess.run(['git', 'commit', '-m', commit_msg],
                             capture_output=True, text=True)
                
                prs.append({
                    'pr_number': i + 1,
                    'branch': pr_branch,
                    'issues_fixed': len(issue_group),
                    'commit_message': commit_msg
                })
                
                # Return to main branch for next PR
                subprocess.run(['git', 'checkout', 'main'],
                             capture_output=True, text=True)
        
        return prs
    
    def apply_fixes(self, issues: List[Dict]) -> bool:
        """Apply fixes for a group of issues"""
        
        # In real implementation, this would generate and apply actual fixes
        # For now, return True to simulate fixes applied
        return True
    
    def generate_commit_message(self, issues: List[Dict], batch_num: int) -> str:
        """Generate descriptive commit message"""
        
        # Group by issue type
        issue_types = {}
        for issue in issues:
            issue_type = issue.get('type', 'general')
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        # Build commit message
        msg_lines = [
            f"feat(content): improve {self.chapter_name} - batch {batch_num}",
            "",
            "Issues addressed:"
        ]
        
        for issue_type, count in issue_types.items():
            msg_lines.append(f"- {issue_type}: {count} fixes")
        
        msg_lines.extend([
            "",
            f"Consensus level: {max(i.get('consensus_level', 1) for i in issues)}",
            f"Perspectives: {', '.join(set(p for i in issues for p in i.get('reported_by', [])))}",
            "",
            "Generated by /improve command"
        ])
        
        return '\n'.join(msg_lines)
    
    def save_summary(self, chunks: List[Dict], issues: List[Dict], prs: List[Dict]):
        """Save session summary"""
        
        summary = {
            'chapter': self.chapter_name,
            'timestamp': self.timestamp,
            'statistics': {
                'total_lines': sum(c['end_line'] - c['start_line'] + 1 for c in chunks),
                'chunks_processed': len(chunks),
                'total_issues': len(issues),
                'prs_created': len(prs)
            },
            'configuration': {
                'chunk_size': self.chunk_size,
                'max_issues_per_pr': self.max_issues_per_pr,
                'perspectives': self.perspectives
            },
            'prs': prs,
            'high_priority_issues': [i for i in issues if i.get('severity') == 'HIGH'][:10]
        }
        
        summary_file = self.session_dir / 'session_summary.json'
        summary_file.write_text(json.dumps(summary, indent=2))
        
        # Also create a markdown report
        report = self.generate_markdown_report(summary)
        report_file = self.session_dir / 'improvement_report.md'
        report_file.write_text(report)
        
        print(f"\n📊 Summary saved to: {summary_file}")
        print(f"📄 Report saved to: {report_file}")
    
    def generate_markdown_report(self, summary: Dict) -> str:
        """Generate human-readable markdown report"""
        
        lines = [
            f"# Improvement Report: {summary['chapter']}",
            f"**Generated:** {summary['timestamp']}",
            "",
            "## Statistics",
            f"- Total lines processed: {summary['statistics']['total_lines']}",
            f"- Chunks: {summary['statistics']['chunks_processed']}",
            f"- Issues found: {summary['statistics']['total_issues']}",
            f"- PRs created: {summary['statistics']['prs_created']}",
            "",
            "## Pull Requests Created"
        ]
        
        for pr in summary['prs']:
            lines.extend([
                f"### PR #{pr['pr_number']}: {pr['branch']}",
                f"- Issues fixed: {pr['issues_fixed']}",
                "- Commit message:",
                "```",
                pr['commit_message'],
                "```",
                ""
            ])
        
        if summary['high_priority_issues']:
            lines.extend([
                "## High Priority Issues",
                ""
            ])
            
            for issue in summary['high_priority_issues']:
                lines.append(f"- **{issue.get('type', 'unknown')}** at {issue.get('line_range', 'unknown')}")
        
        return '\n'.join(lines)


def improve_command(chapter_file: str, **kwargs) -> Dict:
    """
    Main entry point for /improve command
    
    Args:
        chapter_file: Path to the .qmd file to improve
        **kwargs: Optional configuration parameters
        
    Returns:
        Dictionary with improvement results
    """
    
    orchestrator = ImprovementOrchestrator(chapter_file, kwargs)
    return orchestrator.run_improvement_cycle()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: /improve <chapter.qmd> [options]")
        sys.exit(1)
    
    # Parse arguments
    chapter = sys.argv[1]
    
    # Run improvement
    results = improve_command(chapter)
    
    print("\n" + "="*60)
    print("✅ IMPROVEMENT COMPLETE")
    print("="*60)
    print(f"Chapter: {results['chapter']}")
    print(f"Issues found: {results['total_issues']}")
    print(f"PRs created: {results['prs_created']}")
    print(f"Session data: {results['session_dir']}")