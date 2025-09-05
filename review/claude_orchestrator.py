#!/usr/bin/env python3
"""
Main orchestrator for running multi-perspective reviews using Claude Task subagents
This file should be run from Claude Code to access Task subagents
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from agents.student_reviewers import get_all_agents

class ClaudeReviewOrchestrator:
    """Orchestrates reviews using Claude's Task subagents"""
    
    def __init__(self, chapter_file: str):
        self.chapter_path = Path(chapter_file)
        self.chapter_name = self.chapter_path.stem
        self.content = self.chapter_path.read_text()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Review session directory
        self.session_dir = Path("review/temp") / f"{self.chapter_name}_{self.timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📚 Initialized review for: {self.chapter_name}")
        print(f"📁 Session directory: {self.session_dir}")
    
    async def run_student_review(self, agent_name: str, prompt: str) -> Dict:
        """
        Run a single student reviewer using Claude Task subagent
        This method would be called from Claude Code with access to Task
        """
        
        full_prompt = f"""
{prompt}

IMPORTANT: Return your findings as a JSON object with this structure:
{{
  "agent": "{agent_name}",
  "issues": [
    {{
      "location": "line numbers or section name",
      "type": "undefined_term|missing_context|dense_section|unclear_reference|abrupt_transition",
      "severity": "HIGH|MEDIUM|LOW",
      "description": "Specific confusion description",
      "suggestion": "How to fix this issue"
    }}
  ],
  "summary": {{
    "total_issues": number,
    "high_priority": number,
    "medium_priority": number,
    "low_priority": number,
    "main_concerns": "Brief summary of main confusion points"
  }}
}}

Chapter content to review:
{self.content[:5000]}  # Limiting for initial test
"""
        
        print(f"  🎓 Running {agent_name} review via Claude Task subagent...")
        
        # This is where Claude Code would call the Task subagent
        # Example structure for Claude Code implementation:
        """
        result = await Task(
            subagent_type="general-purpose",
            description=f"Review chapter as {agent_name} student",
            prompt=full_prompt
        )
        """
        
        # For testing without Task access, return mock structure
        return {
            "agent": agent_name,
            "issues": [],
            "summary": {
                "total_issues": 0,
                "high_priority": 0,
                "medium_priority": 0,
                "low_priority": 0,
                "main_concerns": "Mock response - implement with Task subagent"
            }
        }
    
    async def run_full_review(self, selected_agents: List[str] = None) -> Dict:
        """Run complete multi-perspective review"""
        
        agents = get_all_agents()
        if selected_agents:
            agents = {k: v for k, v in agents.items() if k in selected_agents}
        
        print(f"\n🔍 Running {len(agents)} perspective reviews...")
        print("-" * 50)
        
        all_feedback = {}
        
        # Run each agent review
        for agent_name, prompt in agents.items():
            feedback = await self.run_student_review(agent_name, prompt)
            all_feedback[agent_name] = feedback
            
            # Save individual feedback
            feedback_file = self.session_dir / f"{agent_name}_feedback.json"
            feedback_file.write_text(json.dumps(feedback, indent=2))
            
            print(f"    ✓ {agent_name} complete: {feedback['summary']['total_issues']} issues found")
        
        # Consolidate and prioritize
        print("\n📊 Analyzing feedback...")
        analysis = self.analyze_feedback(all_feedback)
        
        # Generate improvements
        print("✍️  Generating improvements...")
        improvements = await self.generate_improvements(analysis)
        
        # Complete results
        results = {
            "chapter": self.chapter_name,
            "timestamp": self.timestamp,
            "session_dir": str(self.session_dir),
            "agents_used": list(agents.keys()),
            "raw_feedback": all_feedback,
            "analysis": analysis,
            "improvements": improvements
        }
        
        # Save results
        results_file = self.session_dir / "complete_review.json"
        results_file.write_text(json.dumps(results, indent=2))
        
        print(f"\n✅ Review complete: {results_file}")
        
        return results
    
    def analyze_feedback(self, all_feedback: Dict) -> Dict:
        """Analyze and consolidate feedback from all agents"""
        
        issues_by_location = {}
        consensus_issues = []
        
        # Group issues by location
        for agent_name, feedback in all_feedback.items():
            for issue in feedback.get("issues", []):
                loc = issue.get("location", "unknown")
                if loc not in issues_by_location:
                    issues_by_location[loc] = []
                issues_by_location[loc].append({
                    "agent": agent_name,
                    **issue
                })
        
        # Find consensus issues (reported by 3+ agents)
        for location, issues in issues_by_location.items():
            if len(issues) >= 3:
                consensus_issues.append({
                    "location": location,
                    "reporter_count": len(issues),
                    "reporters": [i["agent"] for i in issues],
                    "issues": issues
                })
        
        return {
            "issues_by_location": issues_by_location,
            "consensus_issues": consensus_issues,
            "total_unique_locations": len(issues_by_location),
            "high_priority_count": sum(
                1 for agent_feedback in all_feedback.values()
                for issue in agent_feedback.get("issues", [])
                if issue.get("severity") == "HIGH"
            )
        }
    
    async def generate_improvements(self, analysis: Dict) -> List[Dict]:
        """
        Generate specific text improvements using Claude Task subagent
        Focus on consensus issues first
        """
        
        improvements = []
        
        # Focus on top consensus issues
        for consensus_issue in analysis["consensus_issues"][:5]:
            improvement_prompt = f"""
Given these confusion points from multiple student perspectives about the same location:

Location: {consensus_issue["location"]}
Reporters: {", ".join(consensus_issue["reporters"])}

Issues reported:
{json.dumps(consensus_issue["issues"], indent=2)}

Generate a specific text improvement that addresses ALL these concerns.
Provide the exact old text and new improved text.

Return as JSON:
{{
  "location": "{consensus_issue["location"]}",
  "old_text": "exact text to replace",
  "new_text": "improved text",
  "addresses": ["list of specific concerns addressed"]
}}
"""
            
            # This would call Task subagent for improvement generation
            print(f"  📝 Generating improvement for {consensus_issue['location']}...")
            
            # Mock improvement for testing
            improvements.append({
                "location": consensus_issue["location"],
                "old_text": "...",
                "new_text": "...",
                "addresses": [f"Confusion from {r}" for r in consensus_issue["reporters"]]
            })
        
        return improvements


# Function to be called from Claude Code
def review_chapter_with_claude(chapter_file: str, agents: List[str] = None):
    """
    Main entry point for Claude Code to run multi-perspective review
    
    Usage in Claude Code:
    ```python
    from textbook_review_system.claude_orchestrator import review_chapter_with_claude
    results = await review_chapter_with_claude("introduction.qmd")
    ```
    """
    import asyncio
    
    orchestrator = ClaudeReviewOrchestrator(chapter_file)
    results = asyncio.run(orchestrator.run_full_review(agents))
    
    # Print summary
    print("\n" + "="*60)
    print("📋 REVIEW SUMMARY")
    print("="*60)
    print(f"Chapter: {results['chapter']}")
    print(f"Agents Used: {', '.join(results['agents_used'])}")
    print(f"Consensus Issues Found: {len(results['analysis']['consensus_issues'])}")
    print(f"High Priority Issues: {results['analysis']['high_priority_count']}")
    print(f"\nTop Consensus Issues:")
    for issue in results['analysis']['consensus_issues'][:3]:
        print(f"  - {issue['location']}: {issue['reporter_count']} agents reported")
    
    return results


if __name__ == "__main__":
    # For testing structure without Claude Task access
    import sys
    if len(sys.argv) > 1:
        chapter = sys.argv[1]
        print(f"Testing structure with: {chapter}")
        print("Note: This needs to be run from Claude Code to access Task subagents")
        
        orchestrator = ClaudeReviewOrchestrator(chapter)
        # Would need async implementation here