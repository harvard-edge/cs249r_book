# 🔐 Private Issue Resolution Methodology
*This is YOUR secret sauce - never committed to GitHub*

## VJ's Content Issue Resolution Workflow

**Goal**: Take any content issue/suggestion and deliver a ready-to-review draft with minimal back-and-forth.

## Command Recognition

When I see these patterns in user messages, I follow the corresponding workflow:

### Pattern: "resolve issue [NUMBER]" or "fix issue [NUMBER]" or "handle issue [NUMBER]"
Example: "resolve issue 947", "handle issue 123", "work on issue 456"
I will: Execute the complete Content Issue Resolution Workflow below

### Pattern: "analyze issue [NUMBER]"  
Example: "analyze issue 947"
I will: Deep analysis only, no changes, provide detailed plan

### Pattern: "check issues"
Example: "check issues", "what issues are open"
I will: List and classify open issues by priority and type

## Issue Classification System

### Type 1: CONTENT_ACCURACY
**Examples:** Wrong explanations, technical errors, outdated information
**Approach:** Deep technical review + pedagogical consistency

### Type 2: CONTENT_CLARITY  
**Examples:** Confusing explanations, missing examples, poor flow
**Approach:** Rewrite for clarity while maintaining technical depth

### Type 3: STRUCTURAL
**Examples:** Missing sections, need exercises, add glossary
**Approach:** Generate new content with consistent format

### Type 4: TECHNICAL
**Examples:** Broken links, build errors, formatting issues
**Approach:** Systematic fix + validation

## VJ's Complete Content Issue Resolution Workflow

### Phase 1: Issue Intelligence Gathering (2-3 minutes)
1. **Fetch & Parse Issue**
   - Get issue details from GitHub API
   - Extract: title, description, reporter, labels, comments
   - Identify issue type (accuracy, clarity, structural, technical)

2. **Context Discovery**
   - Find all relevant files/sections mentioned
   - Locate related content across chapters
   - Understand the pedagogical context and chapter position
   - Check for similar issues or patterns

3. **Impact Assessment**
   - Determine scope: single section vs multi-chapter
   - Identify cross-references that might be affected
   - Assess complexity: simple fix vs major rewrite needed

### Phase 2: Branch Setup & Planning (1 minute)
1. **Create Feature Branch (Linked to Issue)**
   ```bash
   git checkout dev && git pull origin dev
   git checkout -b fix/issue-<number>-<short-description>
   # GitHub automatically links branch to issue via naming convention
   ```

2. **Generate Resolution Plan**
   - Specific files to modify
   - Exact sections/paragraphs to change
   - Cross-references to update
   - Tools to run (section IDs, captions, etc.)

### Phase 3: Solution Development (5-10 minutes)
1. **Apply VJ's Quality Standards**
   - **Technical Accuracy**: Verify against authoritative sources
   - **Pedagogical Flow**: Maintains learning progression
   - **Consistent Voice**: Matches book's academic tone
   - **Progressive Difficulty**: Appropriate for chapter level
   - **Practical Relevance**: Connects to real ML systems

2. **Content Modification Strategy**
   - For ACCURACY: Correct facts, add citations, verify consistency
   - For CLARITY: Rewrite explanations, add examples, improve flow
   - For STRUCTURE: Add sections, reorganize, create exercises
   - For TECHNICAL: Fix links, update references, correct formatting

3. **Cross-Reference Management**
   - Update any affected section IDs
   - Fix broken internal links
   - Ensure figure/table references remain valid

### Phase 4: Clarification & Validation (2-3 minutes)
1. **Identify Ambiguities**
   - Flag any unclear aspects of the issue
   - Note areas where multiple solutions are possible
   - Identify missing context or information

2. **Ask Targeted Questions** (if needed)
   - "Should I also update the related section in Chapter X?"
   - "Do you prefer approach A (simpler) or B (more comprehensive)?"
   - "Should I add a new example or modify the existing one?"

3. **Validate Approach**
   - Confirm the solution addresses the root cause
   - Ensure changes align with book's overall structure
   - Verify no unintended side effects

### Phase 5: Implementation & Draft Creation (5-8 minutes)
1. **Execute Changes**
   - Make all content modifications
   - Run content management tools (section IDs, captions)
   - Update cross-references and links
   - Ensure build compatibility

2. **Quality Assurance**
   - Run pre-commit hooks
   - Verify book builds successfully
   - Check all links and references work
   - Validate formatting consistency

3. **Prepare Branch for Review**
   - Commit with conventional format including issue reference
   - Push branch to remote (auto-links to issue via branch name)
   - **DO NOT create PR yet** (VJ will decide when)
   - Prepare PR description template for later use
   
   **Commit Message Format:**
   ```
   fix: [description]
   
   Thanks to @[reporter] for reporting this issue.
   
   - [specific change 1]
   - [specific change 2]
   
   Addresses #[issue_number]
   ```

### Phase 6: Delivery & Handoff (1 minute)
1. **Present Complete Package**
   - Summary of changes made
   - Rationale for each modification
   - Any remaining questions or considerations
   - **Branch name and location for review**
   - **Prepared PR description** (for when you're ready)

2. **Ready for VJ's Review**
   - All changes implemented and tested
   - **Branch pushed and ready for local review**
   - **You decide when to create PR**
   - Clear next steps provided

## Academic Writing Standards

**CRITICAL**: Before making ANY content changes, read and apply the standards in `.ai/ACADEMIC_WRITING_STANDARDS.md`.

All content modifications must maintain formal academic tone, technical accuracy, and pedagogical consistency as defined in that file.

### Step 3: Content Improvement Patterns

#### For Technical Corrections (like #947):
1. Find the incorrect statement
2. Read 2-3 paragraphs before/after for context
3. Check if correction affects other explanations
4. Update consistently across all mentions
5. Add clarifying example if concept is complex

#### For Clarity Issues:
1. Identify what makes it unclear
2. Check if prerequisites are missing
3. Add concrete examples
4. Use analogies for complex concepts
5. Add visual descriptions where helpful

#### For Missing Content:
1. Check similar sections for format
2. Maintain consistent structure
3. Match depth to chapter level
4. Include practical applications

### Step 4: Verification Process
Before creating PR:
1. Build locally to ensure no breaks
2. Check cross-references still work
3. Verify consistency across chapters
4. Ensure changes maintain quality

## Response Templates

### When Starting Resolution:
"I'll resolve issue #X using our content quality methodology. Let me first analyze the issue deeply..."

### When Making Changes:
"Applying our quality criteria:
- Technical accuracy: [verification]
- Pedagogical flow: [how it fits]
- Consistency: [what I checked]"

### When Creating PR:
"Created PR with:
- Root cause addressed
- All related instances updated
- Quality criteria verified"

## PR and Commit Style (VJ's Voice)

### PR Description Template:
```
Thank you for this excellent feedback! You're absolutely right about [specific issue].

[Explanation of what was wrong and why the suggestion is better]

This PR:
- [Specific change 1]
- [Specific change 2]

Really appreciate you taking the time to help improve the book's technical accuracy.

Fixes #[issue_number]
```

### Commit Message:
```
fix: [clear description]

Thanks to @[username] for catching this and providing the correction.

- [detailed change 1]
- [detailed change 2]

Resolves #[issue_number]
```

### Key Points:
- Always appreciative and warm
- Acknowledge the contributor by GitHub username
- Explain why their feedback improves the book
- Professional but friendly tone

## Special Instructions

### For High-Value Issues:
Issues about core concepts (neural networks, training, optimization):
- Extra careful with technical accuracy
- Add more examples
- Verify with multiple sources

### For Reader-Reported Issues:
- Address exactly what confused them
- Add preventive clarifications
- Thank them in PR description

## VJ's Streamlined Workflow Summary

**Total Time**: 15-20 minutes from issue to draft PR

**Input**: "resolve issue #945" or "handle issue #123"

**Output**: 
- Feature branch created
- Content fixes implemented
- Draft PR ready for your review
- Clear summary of all changes

**Key Principles**:
1. **Minimal back-and-forth** - Ask only essential clarification questions
2. **Complete solutions** - Address root cause, not just symptoms  
3. **Quality first** - Apply all academic and technical standards
4. **Ready to review** - Deliver polished draft, not rough work
5. **Appreciative tone** - Thank contributors in PR descriptions

**When to ask clarification**:
- Multiple valid approaches exist
- Scope is unclear (single section vs multi-chapter)
- Technical accuracy requires domain expertise
- Pedagogical approach needs your input

**When NOT to ask clarification**:
- Obvious typos or formatting issues
- Clear technical corrections
- Standard cross-reference updates
- Routine content improvements

## Quick Wins Pattern

When I need to show value quickly:
1. Start with issues that have clear, specific problems (like #947)
2. Make focused fixes that improve accuracy
3. Create clean PRs with clear descriptions
4. Move fast on technical corrections

## Metrics to Track (Privately)
- Time to resolution: Target < 15 minutes
- Changes per issue: Usually 2-5 files
- Quality score: Must meet all criteria