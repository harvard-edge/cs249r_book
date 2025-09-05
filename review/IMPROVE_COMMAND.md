# /improve Command for Claude Code

## Overview
The `/improve` command is an intelligent textbook improvement system that:
1. **Chunks large files** intelligently to avoid context overflow
2. **Reviews from multiple perspectives** (CS, EE, PhD students)
3. **Creates small, reviewable PRs** instead of massive changes
4. **Organizes improvements** by priority and section

## Usage

### Basic Command
```bash
/improve introduction.qmd
```

### With Options
```bash
/improve frameworks.qmd --max-prs 3
/improve efficient_ai.qmd --chunk-lines 300
/improve training.qmd --max-prs 5 --chunk-lines 400
```

## Options
- `--max-prs N`: Maximum number of PRs to create (default: 5)
- `--chunk-lines N`: Maximum lines per chunk (default: 400)

## How It Works

### 1. Smart Chunking
The system intelligently splits large files:
- **Semantic boundaries**: Splits at section headers when possible
- **Context preservation**: Maintains context between chunks
- **Size limits**: Respects maximum chunk size to avoid context overflow

Example chunking for a 2000-line file:
```
Chunk 1: Lines 1-400 (## Introduction section)
Chunk 2: Lines 401-850 (## Core Concepts section)
Chunk 3: Lines 851-1300 (## Implementation section)
Chunk 4: Lines 1301-1700 (## Examples section)
Chunk 5: Lines 1701-2000 (## Conclusion section)
```

### 2. Multi-Perspective Review
Each chunk is reviewed by 5 student agents:
- **Junior CS**: Systems perspective
- **Senior EE**: Hardware perspective
- **Masters**: Basic ML knowledge
- **PhD**: Theory perspective
- **Industry**: Practical perspective

### 3. PR Organization
Improvements are grouped into small PRs:

**PR #1: Critical Issues** (≤5 changes)
- Issues that block comprehension
- Reported by 3+ reviewers

**PR #2: High Priority - Section A** (≤5 changes)
- Major clarity issues in specific section
- Reported by 2+ reviewers

**PR #3: High Priority - Section B** (≤5 changes)
- Major issues in another section

**PR #4: Medium Priority** (≤5 changes)
- General improvements
- Single reviewer issues

## Example Workflow

```bash
# 1. Run improve command
/improve introduction.qmd --max-prs 3

# Output:
🚀 Starting /improve command for: introduction.qmd
📄 Analyzing file structure...
   Split into 5 semantic chunks
🔍 Running multi-perspective review...
   Chunk 1/5: Purpose (lines 1-150)
     ⚠️ Found 2 critical issues
   Chunk 2/5: AI Pervasiveness (lines 151-400)
   Chunk 3/5: AI and ML Basics (lines 401-750)
     ⚠️ Found 1 critical issue
   ...
📊 Total improvements identified: 23
🔀 Creating pull request batches...
   Will create 3 PRs

📝 Creating PR 1/3: Fix critical comprehension barriers
   ✅ Branch created: improve-introduction-critical-1-20240104_1430

📝 Creating PR 2/3: Improve Purpose section clarity
   ✅ Branch created: improve-introduction-high_priority-2-20240104_1430

📝 Creating PR 3/3: Minor clarity improvements
   ✅ Branch created: improve-introduction-medium_priority-3-20240104_1430

✅ /improve COMMAND COMPLETE
📚 Chapter: introduction
📊 Improvements: 23
🔀 PRs created: 3

📋 Next steps:
   1. Review branch: improve-introduction-critical-1-20240104_1430 (5 changes)
   2. Review branch: improve-introduction-high_priority-2-20240104_1430 (5 changes)
   3. Review branch: improve-introduction-medium_priority-3-20240104_1430 (5 changes)
```

## PR Structure

Each PR contains:
- **Small, focused changes**: Maximum 5 improvements per PR
- **Clear commit messages**: Describe what's being fixed
- **Detailed PR description**: Lists each issue addressed
- **Review checklist**: Ensures quality

### Example PR Description
```markdown
## 📚 Textbook Improvement: introduction

**Type:** Fix critical comprehension barriers
**Changes:** 5

### Issues Addressed:

- **Line 21-25**: Undefined term "Machine Learning Systems Engineering"
  - Reported by 4 reviewers
- **Line 43-45**: Missing explanation for "mathematical optimization"
  - Reported by 3 reviewers
- **Line 67-70**: Dense paragraph with multiple concepts
  - Reported by 3 reviewers

### Review Checklist:
- [ ] Content accuracy preserved
- [ ] Academic tone maintained
- [ ] Improvements enhance clarity
- [ ] No new issues introduced
```

## Benefits

### 1. Handles Large Files
- Chunks prevent context overflow
- Maintains coherence across chunks
- Processes files of any size

### 2. Easy Review Process
- Small PRs are quick to review
- Each PR has focused purpose
- Can accept/reject individually

### 3. Quality Improvements
- Multi-perspective ensures comprehensive review
- Consensus-based prioritization
- Systematic approach

### 4. Git-Friendly
- Clean branch structure
- Atomic commits
- Professional PR descriptions

## Advanced Features

### Custom Perspectives
Add specific reviewer types for your domain:
```python
# In agents/student_reviewers.py
BIOLOGY_STUDENT_PROMPT = """
You are a biology student reading ML systems content...
"""
```

### Section-Specific Reviews
Focus on particular sections:
```bash
/improve introduction.qmd --sections "Purpose,AI Evolution"
```

### Batch Processing
Process multiple chapters:
```bash
for chapter in introduction frameworks training; do
    /improve ${chapter}.qmd --max-prs 2
done
```

## Troubleshooting

### Context Overflow
If you see context errors, reduce chunk size:
```bash
/improve large_chapter.qmd --chunk-lines 200
```

### Too Many PRs
Increase changes per PR:
```bash
/improve chapter.qmd --max-prs 2
```

### Merge Conflicts
Process chapters sequentially:
1. Improve chapter A
2. Merge PRs
3. Improve chapter B

## Integration with GitHub

After running `/improve`:

```bash
# 1. Push all branches
git push origin improve-introduction-*

# 2. Create PRs on GitHub
gh pr create --base main --head improve-introduction-critical-1-20240104_1430

# 3. Review and merge
# Use GitHub's web interface to review each PR
```

## Summary

The `/improve` command transforms textbook improvement from a manual, overwhelming task into an automated, manageable process with:
- ✅ Intelligent file chunking
- ✅ Multi-perspective review
- ✅ Small, focused PRs
- ✅ Clear improvement tracking
- ✅ Git-integrated workflow

Just type `/improve <chapter.qmd>` and let the system handle the complexity!