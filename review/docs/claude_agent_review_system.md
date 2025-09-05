# Claude Agent-Based Textbook Review System

## Overview
Using Claude Code's built-in Task subagents to create an automated textbook review and improvement pipeline that simulates classroom feedback.

## System Architecture

### Three-Agent Pipeline
1. **Student-Reviewer Agent**: Reads as confused student, identifies issues
2. **Improvement-Writer Agent**: Fixes issues based on feedback
3. **Orchestrator** (main Claude Code): Coordinates and applies changes

## Implementation

### Step 1: Student-Reviewer Agent Prompt

```python
STUDENT_REVIEWER_PROMPT = """
You are a first-time student reading this ML Systems textbook chapter. 
Your task is to identify ALL points of confusion a real student would encounter.

Read the provided chapter content and identify:

1. **Undefined Terms**: Technical jargon used before explanation
2. **Missing Prerequisites**: Concepts that assume prior knowledge not yet covered
3. **Unclear Transitions**: Abrupt topic switches without connection
4. **Dense Sections**: Paragraphs with too many concepts at once
5. **Missing Context**: Examples or code without proper setup
6. **Ambiguous References**: Unclear what "this", "that", "it" refers to

For each issue found, provide:
- Line number range
- Issue type (from above categories)
- Severity: HIGH (blocks understanding), MEDIUM (causes confusion), LOW (minor clarity)
- Specific confusion point
- Suggested fix approach

Output as structured JSON:
{
  "issues": [
    {
      "lines": "45-52",
      "type": "undefined_term",
      "severity": "HIGH",
      "confusion": "Term 'gradient accumulation' used without explanation",
      "suggestion": "Define gradient accumulation before first use or add inline explanation"
    }
  ],
  "summary": {
    "total_issues": N,
    "high_priority": X,
    "medium_priority": Y,
    "low_priority": Z
  }
}

Be thorough - identify EVERY potential confusion point.
"""

IMPROVEMENT_WRITER_PROMPT = """
You are an expert textbook editor. Given a list of student confusion points,
you will generate specific text improvements that:

1. Fix the issues while maintaining academic tone
2. Add missing definitions and context
3. Smooth transitions between topics
4. Break down dense sections
5. Clarify ambiguous references

For each issue, provide the EXACT text replacement:
{
  "improvements": [
    {
      "issue_id": "issue_1",
      "line_range": "45-52",
      "old_text": "exact text to replace",
      "new_text": "improved replacement text",
      "rationale": "why this change helps"
    }
  ]
}

Ensure improvements:
- Maintain technical accuracy
- Use consistent terminology
- Build knowledge progressively
- Keep appropriate section lengths
"""
```

### Step 2: Orchestrator Implementation

```python
# review_chapter.py

def review_and_improve_chapter(chapter_path):
    """
    Full review cycle using Claude subagents
    """
    
    # Phase 1: Read chapter content
    with open(chapter_path, 'r') as f:
        content = f.read()
    
    # Phase 2: Student review via subagent
    print("🎓 Running student review agent...")
    review_feedback = run_student_reviewer(content)
    
    # Phase 3: Generate improvements via subagent  
    if review_feedback['summary']['total_issues'] > 0:
        print(f"📝 Found {review_feedback['summary']['total_issues']} issues")
        print("✍️  Running improvement writer agent...")
        improvements = run_improvement_writer(content, review_feedback)
        
        # Phase 4: Apply improvements
        print("🔧 Applying improvements...")
        improved_content = apply_improvements(content, improvements)
        
        # Phase 5: Save and commit
        save_improved_chapter(chapter_path, improved_content)
        
    return review_feedback, improvements

def run_student_reviewer(content):
    """Run the student reviewer subagent"""
    # This will be called by Claude Code using Task tool
    pass

def run_improvement_writer(content, feedback):
    """Run the improvement writer subagent"""
    # This will be called by Claude Code using Task tool
    pass
```

### Step 3: Batch Processing Script

```bash
#!/bin/bash
# review_all_chapters.sh

CHAPTERS=(
    "quarto/contents/core/introduction/introduction.qmd"
    "quarto/contents/core/frameworks/frameworks.qmd"
    "quarto/contents/core/efficient_ai/efficient_ai.qmd"
    "quarto/contents/core/training/training.qmd"
    "quarto/contents/core/optimizations/optimizations.qmd"
)

for chapter in "${CHAPTERS[@]}"; do
    echo "📚 Processing: $(basename $chapter)"
    
    # Run review cycle
    python review_chapter.py "$chapter"
    
    # Check results
    if [ $? -eq 0 ]; then
        echo "✅ Successfully improved: $chapter"
    else
        echo "❌ Failed to process: $chapter"
    fi
    
    # Add delay to avoid overwhelming
    sleep 2
done
```

## Usage Examples

### Single Chapter Review
```python
# In Claude Code, you would run:
# 1. First, get student feedback
task_result = Task(
    subagent_type="general-purpose",
    description="Review chapter as student",
    prompt=STUDENT_REVIEWER_PROMPT + "\n\nChapter content:\n" + chapter_content
)

# 2. Then generate improvements
improvements = Task(
    subagent_type="general-purpose", 
    description="Generate improvements",
    prompt=IMPROVEMENT_WRITER_PROMPT + "\n\nFeedback:\n" + task_result
)

# 3. Apply the improvements
apply_improvements_to_file(chapter_path, improvements)
```

### Automated Pipeline
```python
def automated_review_pipeline():
    """
    Fully automated review of all chapters
    """
    chapters = glob.glob("quarto/contents/core/*/*.qmd")
    
    for chapter in chapters:
        print(f"\n{'='*60}")
        print(f"Processing: {chapter}")
        print('='*60)
        
        # Run full review cycle
        feedback, improvements = review_and_improve_chapter(chapter)
        
        # Log results
        log_results(chapter, feedback, improvements)
        
        # Commit if successful
        if improvements:
            commit_improvements(chapter)
```

## Real Implementation in Claude Code

Here's how we'll actually use the Task subagents:

```python
# Step 1: Review as confused student
student_feedback = await Task(
    subagent_type="general-purpose",
    description="Review ML textbook chapter",
    prompt=f"""
    Read this textbook chapter as a first-time ML student.
    Identify every point where a student might get confused.
    
    Chapter: {chapter_content}
    
    Find:
    - Undefined technical terms
    - Missing prerequisites
    - Dense paragraphs
    - Unclear examples
    
    Return structured JSON with all issues found.
    """
)

# Step 2: Generate improvements
improvements = await Task(
    subagent_type="general-purpose",
    description="Generate chapter improvements",
    prompt=f"""
    Based on these student confusion points, generate specific text improvements.
    
    Confusion points: {student_feedback}
    Original text: {chapter_content}
    
    Provide exact text replacements that fix each issue.
    Maintain academic tone and technical accuracy.
    """
)

# Step 3: Apply improvements
for improvement in improvements['improvements']:
    edit_file(
        file_path=chapter_path,
        old_string=improvement['old_text'],
        new_string=improvement['new_text']
    )
```

## Advantages of Using Claude Subagents

1. **Parallel Processing**: Can run multiple reviews simultaneously
2. **Perspective Isolation**: Each agent maintains its role consistently
3. **Scalability**: Easy to add more reviewer perspectives
4. **Memory Efficiency**: Subagents handle large chapters independently
5. **Reproducibility**: Same prompts give consistent reviews

## Next Steps

1. Test on introduction chapter first
2. Refine prompts based on results
3. Add more student perspectives (CS, Stats, Engineering backgrounds)
4. Create validation agent to verify improvements
5. Set up automated nightly reviews

Ready to implement this system on your chapters!