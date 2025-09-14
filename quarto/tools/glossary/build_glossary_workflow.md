# Glossary Building Workflow

## Phase 1: Individual Chapter Glossaries

### Step 1: Generate Per-Chapter Glossaries
```bash
# For each chapter, agent creates:
data/chapter_glossaries/introduction_glossary.yml
data/chapter_glossaries/ml_systems_glossary.yml
data/chapter_glossaries/dl_primer_glossary.yml
...
```

### Format:
```yaml
chapter: optimizations
terms:
  quantization:
    definition: "The process of reducing numerical precision..."
    context: "Used in this chapter for model compression"
    usage_count: 15
    variants_found: ["quantized", "quantizing", "quantization-aware"]
    
  pruning:
    definition: "Removing unnecessary parameters from neural networks..."
    context: "Discussed alongside quantization"
    usage_count: 12
    related_terms: ["sparsity", "compression"]
```

## Phase 2: Consistency Analysis

### Step 2: Run Consistency Checker
```python
# Script analyzes all chapter glossaries for:
1. Same term, different definitions
2. Different terms, same concept  
3. Terminology drift across chapters
4. Missing cross-references
5. Definition quality variations
```

### Output: Consistency Report
```yaml
inconsistencies:
  definition_conflicts:
    - term: "gradient_descent"
      chapters: ["dl_primer", "training"]
      definitions:
        dl_primer: "An optimization algorithm..."
        training: "An iterative method..."
      recommendation: "Merge into single comprehensive definition"
      
  terminology_variants:
    - concept: "reducing model size"
      terms_used:
        efficient_ai: "model_compression"
        optimizations: "model_optimization"
        ondevice: "model_reduction"
      recommendation: "Standardize to 'model_compression'"
      
  missing_relationships:
    - term: "quantization"
      should_reference: ["int8", "int4", "mixed_precision"]
      currently_references: []
```

## Phase 3: Reconciliation

### Step 3: Editorial Review & Reconciliation
The glossary-builder agent (or human editor) reviews conflicts and:

1. **Merges compatible definitions**
   ```yaml
   # From: Two partial definitions
   # To: One comprehensive definition
   gradient_descent:
     definition: "An iterative optimization algorithm that minimizes loss by updating parameters proportional to the negative gradient, moving in the direction of steepest descent"
   ```

2. **Standardizes terminology**
   ```yaml
   # Decision: Use "model_compression" everywhere
   # Update all chapters to use consistent term
   ```

3. **Adds missing relationships**
   ```yaml
   quantization:
     see_also: ["int8", "int4", "mixed_precision", "ptq", "qat"]
   ```

## Phase 4: Master Glossary Generation

### Step 4: Build Master Glossary
```python
# Merge reconciled chapter glossaries into master
# Preserve chapter usage metadata
# Generate statistics
```

### Final Structure:
```yaml
# data/master_glossary.yml
glossary:
  quantization:
    definition: "The process of mapping continuous or high-precision numerical values..."
    category: "optimization"
    chapters_used: ["efficient_ai", "optimizations", "hw_acceleration", "ondevice"]
    first_introduced: "efficient_ai"
    usage_frequency: 47
    variants: ["quantized", "quantizing", "quantization-aware"]
    see_also: ["int8", "pruning", "compression"]
```

## Phase 5: Quality Assurance

### Step 5: Validation Checks
- All technical terms defined
- No orphaned cross-references
- Consistent definition style
- Appropriate complexity level
- Complete category coverage

## Benefits of This Workflow

### 1. **Catches Inconsistencies Early**
- Different authors/chapters may use different terminology
- Definitions may drift across chapters
- Relationships might be missed

### 2. **Manageable Processing**
- One chapter at a time
- Parallel processing possible
- Incremental updates

### 3. **Editorial Control**
- Review before merging
- Explicit reconciliation decisions
- Audit trail of changes

### 4. **Quality Improvement**
- Identifies terminology issues
- Enforces consistency
- Improves cross-references

## Automation Tools Needed

1. **glossary_extractor.py** - Extract terms from chapters
2. **consistency_checker.py** - Find conflicts
3. **glossary_reconciler.py** - Merge and reconcile
4. **master_builder.py** - Generate final glossary
5. **quality_validator.py** - Run QA checks

## Example Commands

```bash
# Extract glossary from single chapter
python glossary_extractor.py content/core/introduction/introduction.qmd

# Check consistency across all chapters
python consistency_checker.py data/chapter_glossaries/*.yml

# Build master from reconciled chapters  
python master_builder.py data/chapter_glossaries/*.yml -o data/master_glossary.yml

# Validate final glossary
python quality_validator.py data/master_glossary.yml
```