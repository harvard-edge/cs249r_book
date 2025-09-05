Multi-perspective textbook assessment generating comprehensive scorecards.

Analyzes chapters from student and expert perspectives WITHOUT making changes. Generates detailed report cards showing strengths, weaknesses, and recommendations across all dimensions.

Usage: `/review introduction.qmd`

**Output**: Detailed scorecard only - NO file changes

## Review Architecture (Pure Claude)

### Phase 1: Progressive Student Validation
Run students SEQUENTIALLY, passing learned knowledge forward:
```
CS Junior (OS/Architecture background) → validates ML concepts introduced clearly
     ↓ (pass ML knowledge learned)
CS Senior (+ Some ML exposure) → validates systems integration makes sense
     ↓ (pass cumulative knowledge)  
Masters Student → checks production readiness
```

### Phase 2: Parallel Expert Review
Run all experts SIMULTANEOUSLY for efficiency:
```
[Systems Expert] [Data Expert] [Security Expert] [Platform Expert]
        ↓              ↓              ↓                ↓
    All reviewing the same content in parallel
        ↓              ↓              ↓                ↓
    [Consolidated Expert Feedback]
```

### Phase 3: Pedagogical Review
Single comprehensive teaching effectiveness review

### Phase 4: Consensus Analysis & Application
Main Claude orchestrator finds patterns and applies changes

## How It Works (Claude Native)

When you run `/review chapter.qmd`, Claude will:

1. **Read and chunk** the chapter using built-in file operations
2. **Launch progressive student reviews** using Task subagents:
   - Each student Task gets previous student's "learned knowledge"
   - Validates content builds appropriately
3. **Launch parallel expert reviews** using Task subagents:
   - All experts review simultaneously
   - Each focuses on their domain
4. **Analyze consensus** using Claude's reasoning
5. **Generate scorecard** with actionable feedback

### 🎯 Built-in Constraints (Automatic)
All review agents automatically:
- Skip TikZ code blocks (never analyze)
- Skip all tables (markdown and LaTeX)
- Respect Purpose section single-paragraph rule
- Preserve mathematical equations
- Focus on inline improvements
- Generate clean, actionable feedback

## Reviewer Personas (Optimized for ML Systems Engineering)

### Learning Path Validators (Sequential - Knowledge Building)

**Systems-Focused CS Student (First Reader)**
```
Background: Strong systems (OS, distributed systems, databases, networking)
ML Knowledge: None to minimal - maybe saw ML in one lecture
Focus: Can I leverage my systems knowledge to understand ML systems?
Flags: Unexplained ML concepts, missing systems analogies, unclear mappings
Why Critical: Most readers come from systems background, not ML theory
```

**ML Algorithm Student (Second Reader)**
```
Background: Took ML/DL courses, understands backprop, loss functions, architectures
ML Knowledge: Strong theory but only ran models in Jupyter notebooks
Focus: How do I take my notebook models to production systems?
Flags: Missing deployment steps, unclear scaling strategies, gaps in lifecycle
Why Critical: Represents the notebook-to-production journey most face
```

**Early Career Engineer (Third Reader)**
```
Background: 1-2 years industry experience, deployed some models
ML Knowledge: Practical ML deployment, faced real production issues
Focus: Are the best practices and pitfalls accurately represented?
Flags: Oversimplified problems, missing real-world complexity, dated practices
Why Critical: Validates content against current industry reality
```

### Domain Expert Reviewers (Parallel - Comprehensive Coverage)

**Platform/Infrastructure Architect**
```
Expertise: Cloud platforms, Kubernetes, GPU clusters, cost optimization
Focus: Infrastructure design, resource management, multi-tenancy, scaling
Validates: Chapter coverage of deployment platforms, orchestration, efficiency
Critical Because: ML systems consume massive infrastructure resources
```

**MLOps/DevOps Engineer**
```
Expertise: CI/CD for ML, model versioning, A/B testing, monitoring, rollbacks
Focus: Operational excellence, automation, reproducibility, observability
Validates: Lifecycle management, production practices, debugging strategies
Critical Because: Most ML projects fail at operations, not algorithms
```

**Edge/Embedded Systems Engineer**
```
Expertise: Resource-constrained deployment, quantization, hardware acceleration
Focus: On-device inference, power efficiency, model compression, latency
Validates: Coverage of edge deployment, optimization techniques, hardware
Critical Because: Growing importance of edge AI and efficient inference
```

**Data Platform Engineer**
```
Expertise: Feature stores, streaming systems, data lakes, ETL at scale
Focus: Data pipelines, quality, freshness, lineage, governance
Validates: Data engineering aspects, pipeline design, feature management
Critical Because: Data is the foundation - bad data breaks everything
```

**Professor/Educator (Teaching Perspective)**
```
Expertise: Curriculum design, pedagogical methods, student assessment
Focus: Teachability, exercise quality, concept progression, lecture mapping
Validates: 
  - Clear learning objectives per section
  - Appropriate exercises and labs
  - Slides/lecture alignment potential
  - Prerequisites clearly stated
  - Concepts build semester-long course
Critical Because: Primary adopters who need to teach from this material
Questions: Can I build a syllabus from this? Are there enough exercises?
         How do I assess understanding? What labs can I assign?
```

### Why This Cohort Works Better

1. **Covers the Full Journey**: From systems student → ML learner → practitioner
2. **Balances Perspectives**: Systems-first vs ML-first backgrounds  
3. **Addresses Key Gaps**: Platform, operations, edge, data, and teaching
4. **Industry-Relevant**: Focuses on actual deployment challenges
5. **Academia-Ready**: Ensures teachability and curriculum alignment
6. **Progressive Complexity**: Each reader builds on previous understanding

Each reviewer receives chapter content and these instructions are embedded in their Task subagent prompt.

## How Claude Uses These Personas

When you run `/review`, Claude:

1. **For Students**: Launches Task subagents sequentially
   - Junior Task: "You are a CS Junior with OS/architecture background but NEW to ML..."
   - Senior Task: "You are a CS Senior with basic ML knowledge. The Junior found [X]..."
   - Masters Task: "You are a Masters student with strong foundation. Previous students found [Y]..."

2. **For Experts**: Launches Task subagents in parallel
   - Each gets the same content but different domain focus
   - All return structured feedback simultaneously

The actual prompts are constructed at runtime using these persona definitions.

## Consensus Rules

Claude analyzes all feedback and applies:
- **Critical**: Any expert flags as critical OR 3+ students confused
- **High**: 2+ experts recommend OR 2+ students struggle
- **Medium**: Single expert OR student suggestion
- **Low**: Nice to have improvements

## Pure Claude Implementation

NO Python files needed! Everything happens through:
- **Task subagents**: All review perspectives
- **Edit tool**: Apply improvements
- **Bash tool**: Git operations
- **Claude's reasoning**: Consensus analysis

## Advantages of Claude-Native Approach

1. **No code maintenance**: System evolves with Claude
2. **Natural language config**: Just tell Claude what you want
3. **Adaptive reasoning**: Claude adjusts approach based on content
4. **Integrated workflow**: Everything in Claude Code interface
5. **Self-documenting**: Claude explains what it's doing

## Student Prerequisites (Important!)

Our students HAVE:
- Operating systems knowledge
- Computer architecture understanding  
- Basic compilers/systems programming
- Data structures & algorithms

They DON'T necessarily have:
- Deep learning theory (we teach this)
- ML deployment experience
- Production systems knowledge
- Cloud infrastructure expertise

Claude will adjust the review accordingly!

## Output: Comprehensive Scorecard

Generates `.review-scorecard-[chapter].md` with:

### 📊 Dimension Scores (0-10)
- **Learning Progression**: How well concepts build
- **Technical Accuracy**: Correctness and best practices
- **Production Readiness**: Real-world applicability
- **Pedagogical Effectiveness**: Teaching quality
- **Accessibility**: Diverse learner support

### 📋 Detailed Feedback
- **Strengths**: What works well
- **Critical Issues**: Must-fix problems
- **Recommendations**: Suggested improvements
- **Student Confusion Points**: Where learners struggle
- **Expert Concerns**: Technical gaps

### 🎯 Priority Actions
Ranked list of what to fix first based on consensus.

NO changes are made to files - use `/improve` to apply changes.