# Multi-Perspective Textbook Review System

An automated system for reviewing ML Systems textbook chapters from multiple student perspectives using Claude's Task subagents.

## 🎯 Purpose

Simulates classroom feedback by having AI agents review chapters from different student backgrounds:
- **Junior CS**: Systems perspective (OS, Architecture background)
- **Senior EE**: Hardware perspective (Circuits, Embedded systems)
- **Masters**: Basic ML knowledge, lacks production experience
- **PhD**: Strong theory, lacks systems engineering
- **Industry**: Practical experience, lacks formal ML education

## 📁 Structure

```
textbook_review_system/
├── agents/              # Student reviewer prompts
│   └── student_reviewers.py
├── scripts/             # Orchestration scripts
│   └── review_chapter.py
├── configs/             # Configuration files
├── docs/                # Documentation
├── temp/                # Temporary review sessions
└── claude_orchestrator.py  # Main Claude Code interface
```

## 🚀 Quick Start

### Single Command Review

```bash
# Review introduction chapter with all agents
./textbook_review_system/review.sh introduction.qmd

# Review with specific agents
./textbook_review_system/review.sh frameworks.qmd --agents junior_cs,masters,phd

# Custom branch prefix
./textbook_review_system/review.sh efficient_ai.qmd --branch feature/improve
```

### From Claude Code

```python
# Using Task subagents (run in Claude Code)
from textbook_review_system.claude_orchestrator import review_chapter_with_claude

# Review with all perspectives
results = await review_chapter_with_claude("introduction.qmd")

# Review with specific agents
results = await review_chapter_with_claude("frameworks.qmd", agents=["junior_cs", "phd"])
```

### Python Interface

```python
from textbook_review_system.scripts.review_chapter import ChapterReviewer

# Create reviewer
reviewer = ChapterReviewer("quarto/contents/core/introduction/introduction.qmd")

# Run multi-perspective review
results = reviewer.run_multi_perspective_review()

# Get prioritized issues
for issue in results["prioritized"][:5]:
    print(f"Priority: {issue['priority_level']} - {issue['type']}")
    print(f"Reported by: {', '.join(issue['reported_by'])}")
```

## 📊 Output Format

The system generates comprehensive JSON reports with:

```json
{
  "chapter": "introduction",
  "timestamp": "20240104_143022",
  "agents_used": ["junior_cs", "senior_ee", "masters", "phd", "industry"],
  "analysis": {
    "consensus_issues": [...],  // Issues reported by 3+ agents
    "high_priority_count": 15,
    "issues_by_location": {...}
  },
  "improvements": [
    {
      "location": "line 25-30",
      "old_text": "original text",
      "new_text": "improved text",
      "addresses": ["confusion from junior_cs", "confusion from masters"]
    }
  ]
}
```

## 🔄 Workflow

1. **Review Phase**: Multiple agents read chapter independently
2. **Consolidation**: Identify consensus issues (reported by multiple agents)
3. **Prioritization**: Score issues based on:
   - Number of agents reporting
   - Severity ratings
   - Agent combination patterns
4. **Improvement Generation**: Create specific text fixes
5. **Git Integration**: Automatic branch creation and commit preparation

## 🎯 Priority Scoring

Issues are scored based on:
- **Base Score**: 10 points per reporting agent
- **Severity Bonus**: +5 points per HIGH severity vote
- **Combination Bonuses**:
  - +15 if both junior_cs and senior_ee report (systems confusion)
  - +10 if both masters and phd report (graduate confusion)
  - +8 if industry and phd report (theory-practice gap)

Priority Levels:
- **CRITICAL**: Score ≥ 40 (major comprehension barrier)
- **HIGH**: Score ≥ 25 (significant confusion)
- **MEDIUM**: Score ≥ 15 (moderate issue)
- **LOW**: Score < 15 (minor improvement)

## 🚦 Git Integration

The system automatically:
1. Creates feature branches: `review-[chapter]-[timestamp]`
2. Organizes reviews in timestamped directories
3. Prepares commit messages with improvement summaries

## 📈 Benefits

- **Comprehensive Coverage**: Catches issues single perspective would miss
- **Validated Priorities**: Consensus issues are truly problematic
- **Targeted Fixes**: Address specific audience segments
- **Scalable**: Easy to add new reviewer perspectives
- **Traceable**: All feedback preserved in session directories

## 🔧 Customization

### Add New Student Perspective

Edit `agents/student_reviewers.py`:

```python
SOPHOMORE_CS_PROMPT = """
You are a sophomore CS student...
Review the chapter and identify...
"""
```

### Adjust Priority Scoring

Edit `scripts/review_chapter.py`:

```python
def prioritize_issues(self, consolidated):
    # Customize scoring logic
    ...
```

## 📝 Example Usage Flow

```bash
# 1. Start review
./textbook_review_system/review.sh introduction.qmd

# 2. Check results
cat textbook_review_system/temp/introduction_*/review_results.json | jq '.summary'

# 3. Apply improvements (if satisfied)
python apply_improvements.py introduction.qmd

# 4. Commit changes
git add introduction.qmd
git commit -m "feat: multi-perspective improvements for introduction"

# 5. Push branch
git push origin review-introduction-20240104
```

## 🤝 Contributing

To improve the review system:
1. Add new student perspectives in `agents/`
2. Enhance consolidation logic in `scripts/`
3. Improve prompt templates based on results

## 📄 License

Part of ML Systems Textbook project