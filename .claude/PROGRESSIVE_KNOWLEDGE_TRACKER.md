# Progressive Knowledge Tracker System

## How It Works

When reviewing any chapter, the system:
1. Builds a "knowledge dictionary" of all terms from previous chapters
2. Identifies "forbidden terms" from future chapters
3. Validates every improvement against these dictionaries

## Dynamic Knowledge Dictionary Structure

```python
# Conceptual structure (implemented through Claude's reasoning)
knowledge_state = {
    "chapter_1": {
        "terms_introduced": [
            "machine learning systems",
            "production deployment",
            "model drift",  # mentioned but not detailed
            "containerization",
            "API design",
            "data pipeline",
            "model versioning"
        ],
        "concepts_explained": [
            "notebook to production gap",
            "infrastructure orchestration"
        ],
        "can_reference": []  # No previous chapters
    },
    "chapter_2": {
        "terms_introduced": [
            "Cloud ML",
            "Edge ML",
            "Mobile ML", 
            "TinyML",
            "deployment spectrum",
            "resource constraints",
            "latency",
            "throughput"
        ],
        "concepts_explained": [
            "deployment trade-offs",
            "memory limitations by tier"
        ],
        "can_reference": ["chapter_1"]  # Can use Ch1 terms
    },
    # ... continues for each chapter
}
```

## Implementation in Review Process

### Step 1: Build Available Knowledge
When reviewing Chapter N:
```
available_knowledge = []
for chapter in 1 to N-1:
    available_knowledge.extend(chapter.terms_introduced)
    available_knowledge.extend(chapter.concepts_explained)
```

### Step 2: Build Forbidden Terms
```
forbidden_terms = []
for chapter in N+1 to end:
    forbidden_terms.extend(chapter.terms_introduced)
```

### Step 3: Review with Constraints
Each Task subagent receives:
```
"You are reviewing Chapter {N}: {title}

AVAILABLE KNOWLEDGE (you CAN use these):
{available_knowledge}

FORBIDDEN TERMS (you CANNOT use these):
{forbidden_terms}

If you need to reference a concept from {forbidden_terms}, 
use general language like 'techniques we'll explore later' 
or 'methods for improving efficiency' instead of the specific term."
```

## Practical Example: Reviewing Chapter 3 (DL Primer)

### Available Knowledge Dictionary:
```
From Chapter 1:
- machine learning systems
- production deployment  
- model versioning
- data pipelines
- infrastructure orchestration

From Chapter 2:
- Cloud ML, Edge ML, Mobile ML, TinyML
- deployment spectrum
- resource constraints
- latency, throughput
- memory limitations
```

### Forbidden Terms Dictionary:
```
From Chapter 4+:
- CNN, RNN, Transformer (not yet introduced)
- attention mechanism
- convolutional filters
- pooling

From Chapter 10:
- quantization
- pruning  
- knowledge distillation
- model compression

From Chapter 11:
- GPU, TPU, NPU
- FPGA
- hardware acceleration (specific terms)
```

### Applied in Review:
```
❌ WRONG: "Neural networks can be optimized with quantization for edge deployment"
✅ RIGHT: "Neural networks can be optimized for edge deployment using techniques that reduce their size and computational requirements"

❌ WRONG: "CNNs are particularly effective for image processing"  
✅ RIGHT: "Certain neural network structures are particularly effective for image processing, as we'll explore in the next chapter"

❌ WRONG: "GPUs accelerate training through parallel processing"
✅ RIGHT: "Specialized hardware can accelerate training through parallel processing"
```

## Auto-Detection Rules

The system automatically flags violations:

1. **Term Check**: Is term in forbidden_terms list? → Flag for rewrite
2. **Concept Check**: Is concept explained yet? → Use general language
3. **Forward Reference**: Does it mention "Chapter X" where X > current? → Remove
4. **Acronym Check**: Is acronym defined yet? → Use full description

## Progressive Substitution Guide

| Forbidden Term | Progressive Alternative |
|---------------|------------------------|
| Quantization | "reducing precision" or "optimization techniques" |
| Pruning | "removing unnecessary components" |
| CNN | "specialized neural network structures" |
| GPU/TPU | "specialized hardware" or "accelerated computing" |
| Attention | "dynamic focus mechanisms" |
| Transformer | "advanced architectures" |
| Knowledge distillation | "learning from other models" |
| FLOPs | "computational operations" |
| NPU | "specialized processors" |

## Benefits of This System

1. **Automatic enforcement** - System catches forward references
2. **Clear boundaries** - Each chapter has defined vocabulary
3. **Progressive learning** - Students never see undefined terms
4. **Flexible language** - Can describe concepts without specific terms
5. **Maintainable** - Easy to update as chapters evolve

## Usage in Commands

When `/progressive-improve` is called:

1. Claude reads PROGRESSIVE_KNOWLEDGE_TRACKER.md
2. Identifies current chapter position
3. Builds available and forbidden dictionaries
4. Launches Task subagents with these constraints
5. Validates all improvements against dictionaries
6. Substitutes forbidden terms with progressive alternatives

This ensures true progressive knowledge building throughout the textbook!