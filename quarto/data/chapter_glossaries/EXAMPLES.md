# Example: How Individual Glossaries Reveal Issues

This document shows real inconsistencies that the individual → master glossary workflow would catch.

## 1. Definition Evolution Problem

**introduction_glossary.yml:**
```yaml
ai:
  definition: "The goal of creating machines that can match human intelligence"
```

**ml_systems_glossary.yml:**
```yaml
ai:
  definition: "The systematic pursuit of understanding intelligent behavior"
```

**ISSUE**: Definition changed between chapters!  
**SOLUTION**: Need consistent definition throughout

## 2. Terminology Drift

**efficient_ai_glossary.yml:**
```yaml
on_device_inference:
  definition: "Running models directly on edge devices"
```

**ondevice_learning_glossary.yml:**
```yaml
edge_inference:
  definition: "Running models directly on edge devices"
```

**ISSUE**: Same concept, different terms!  
**SOLUTION**: Standardize to one term

## 3. Granularity Mismatch

**optimizations_glossary.yml:**
```yaml
quantization:
  definition: "Reducing precision from FP32 to INT8 or INT4"
  variants: ["PTQ", "QAT", "mixed-precision"]
```

**hw_acceleration_glossary.yml:**
```yaml
quantization:
  definition: "Converting floating-point to integer representation"
  # No mention of variants!
```

**ISSUE**: Different levels of detail  
**SOLUTION**: Ensure consistent depth

## 4. Context Conflicts

**training_glossary.yml:**
```yaml
batch_size:
  definition: "Number of samples processed before updating weights"
  context: "Affects memory usage and convergence"
```

**data_engineering_glossary.yml:**
```yaml
batch_size:
  definition: "Number of records processed together"
  context: "Affects throughput and latency"
```

**ISSUE**: Same term, different contexts!  
**SOLUTION**: Create unified definition covering both

## 5. Missing Relationships

**dl_primer_glossary.yml:**
```yaml
backpropagation:
  definition: "Algorithm for computing gradients"
  # No related terms
```

**training_glossary.yml:**
```yaml
backpropagation:
  definition: "Algorithm for computing gradients using chain rule"
  related: ["gradient_descent", "automatic_differentiation"]
```

**ISSUE**: Relationships discovered later  
**SOLUTION**: Backfill relationships to earlier chapters

## Why This Matters

1. **Student Confusion**: Inconsistent definitions confuse learners
2. **Authority**: Textbook loses credibility with conflicts
3. **Exam Problems**: Which definition is "correct" for testing?
4. **Future Maintenance**: Harder to update inconsistent content

## The Individual → Master Approach Solves This

1. Build individual glossaries (reveals all issues)
2. Run consistency analysis (identifies conflicts)
3. Editorial reconciliation (make decisions)
4. Generate clean master (single source of truth)
5. Validate consistency (ensure quality)

This is professional textbook development!