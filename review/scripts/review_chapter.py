#!/usr/bin/env python3
"""
Multi-perspective chapter review orchestrator
Usage: python review_chapter.py <chapter_file.qmd>
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.student_reviewers import get_all_agents, get_agent

class ChapterReviewer:
    """Orchestrates multi-perspective review of textbook chapters"""
    
    def __init__(self, chapter_path: str):
        self.chapter_path = Path(chapter_path)
        if not self.chapter_path.exists():
            raise FileNotFoundError(f"Chapter not found: {chapter_path}")
        
        self.chapter_name = self.chapter_path.stem
        self.content = self.chapter_path.read_text()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create temp directory for this review session
        self.session_dir = Path(__file__).parent.parent / "temp" / f"{self.chapter_name}_{self.timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
    def review_with_agent(self, agent_name: str, prompt: str) -> Dict:
        """
        Run review with a specific student agent
        In actual implementation, this calls Claude Task subagent
        """
        print(f"  🎓 Running {agent_name} perspective...")
        
        # This is where we'd call the Claude Task subagent
        # For now, returning structure for testing
        return {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "issues": [],
            "summary": {
                "total_issues": 0,
                "high_priority": 0,
                "medium_priority": 0,
                "low_priority": 0
            }
        }
    
    def run_multi_perspective_review(self, agents: Optional[List[str]] = None) -> Dict:
        """Run review with multiple student perspectives"""
        
        print(f"\n📚 Reviewing: {self.chapter_name}.qmd")
        print(f"📁 Session: {self.session_dir}")
        print("-" * 50)
        
        # Use all agents if none specified
        all_agents = get_all_agents()
        if agents:
            selected_agents = {k: v for k, v in all_agents.items() if k in agents}
        else:
            selected_agents = all_agents
        
        print(f"🔍 Running {len(selected_agents)} reviewer perspectives:")
        
        # Collect feedback from each agent
        all_feedback = {}
        for agent_name, prompt in selected_agents.items():
            feedback = self.review_with_agent(agent_name, prompt)
            all_feedback[agent_name] = feedback
            
            # Save individual agent feedback
            agent_file = self.session_dir / f"{agent_name}_feedback.json"
            agent_file.write_text(json.dumps(feedback, indent=2))
        
        # Consolidate feedback
        print("\n📊 Consolidating feedback...")
        consolidated = self.consolidate_feedback(all_feedback)
        
        # Prioritize issues  
        print("🎯 Prioritizing issues...")
        prioritized = self.prioritize_issues(consolidated)
        
        # Generate summary report
        summary = self.generate_summary(all_feedback, consolidated, prioritized)
        
        # Save complete results
        results = {
            "chapter": self.chapter_name,
            "timestamp": self.timestamp,
            "agents_used": list(selected_agents.keys()),
            "raw_feedback": all_feedback,
            "consolidated": consolidated,
            "prioritized": prioritized,
            "summary": summary
        }
        
        results_file = self.session_dir / "review_results.json"
        results_file.write_text(json.dumps(results, indent=2))
        
        print(f"\n✅ Review complete!")
        print(f"📄 Results: {results_file}")
        
        return results
    
    def consolidate_feedback(self, all_feedback: Dict) -> List[Dict]:
        """Merge feedback from multiple agents, identifying common issues"""
        
        consolidated = {}
        
        for agent_name, feedback in all_feedback.items():
            for issue in feedback.get("issues", []):
                # Create unique key for grouping similar issues
                key = f"{issue.get('location', 'unknown')}_{issue.get('type', 'unknown')}"
                
                if key not in consolidated:
                    consolidated[key] = {
                        "location": issue.get("location"),
                        "type": issue.get("type"),
                        "reported_by": [],
                        "descriptions": [],
                        "severity_votes": []
                    }
                
                consolidated[key]["reported_by"].append(agent_name)
                consolidated[key]["descriptions"].append({
                    "agent": agent_name,
                    "description": issue.get("description", "")
                })
                consolidated[key]["severity_votes"].append(issue.get("severity", "MEDIUM"))
        
        return list(consolidated.values())
    
    def prioritize_issues(self, consolidated: List[Dict]) -> List[Dict]:
        """Prioritize issues based on multiple factors"""
        
        for issue in consolidated:
            # Base score on number of agents reporting
            score = len(issue["reported_by"]) * 10
            
            # Boost for high severity votes
            high_severity_count = issue["severity_votes"].count("HIGH")
            score += high_severity_count * 5
            
            # Boost for certain agent combinations
            reporters = set(issue["reported_by"])
            
            # Both systems students confused = very important
            if {"junior_cs", "senior_ee"}.issubset(reporters):
                score += 15
            
            # All graduate students confused = important
            if {"masters", "phd"}.issubset(reporters):
                score += 10
                
            # Industry + academic confusion = practical importance
            if "industry" in reporters and "phd" in reporters:
                score += 8
            
            issue["priority_score"] = score
            issue["priority_level"] = self._score_to_level(score)
        
        # Sort by priority score
        return sorted(consolidated, key=lambda x: x["priority_score"], reverse=True)
    
    def _score_to_level(self, score: int) -> str:
        """Convert numeric score to priority level"""
        if score >= 40:
            return "CRITICAL"
        elif score >= 25:
            return "HIGH"
        elif score >= 15:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_summary(self, raw: Dict, consolidated: List, prioritized: List) -> Dict:
        """Generate summary statistics"""
        
        total_issues = len(consolidated)
        
        priority_counts = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0
        }
        
        for issue in prioritized:
            level = issue.get("priority_level", "LOW")
            priority_counts[level] += 1
        
        # Find most problematic sections
        location_counts = {}
        for issue in consolidated:
            loc = issue.get("location", "unknown")
            location_counts[loc] = location_counts.get(loc, 0) + 1
        
        top_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_issues": total_issues,
            "priority_breakdown": priority_counts,
            "consensus_issues": len([i for i in consolidated if len(i["reported_by"]) >= 3]),
            "top_problematic_sections": top_locations,
            "agents_participated": len(raw)
        }


def main():
    """Command-line interface"""
    if len(sys.argv) < 2:
        print("Usage: python review_chapter.py <chapter_file.qmd> [--agents agent1,agent2]")
        sys.exit(1)
    
    chapter_file = sys.argv[1]
    
    # Parse optional agents argument
    agents = None
    if len(sys.argv) > 2 and sys.argv[2].startswith("--agents"):
        agents = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    
    try:
        reviewer = ChapterReviewer(chapter_file)
        results = reviewer.run_multi_perspective_review(agents)
        
        # Print summary
        summary = results["summary"]
        print("\n" + "="*50)
        print("📊 REVIEW SUMMARY")
        print("="*50)
        print(f"Total Issues Found: {summary['total_issues']}")
        print(f"Consensus Issues (3+ agents): {summary['consensus_issues']}")
        print("\nPriority Breakdown:")
        for level, count in summary['priority_breakdown'].items():
            if count > 0:
                print(f"  {level}: {count}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()