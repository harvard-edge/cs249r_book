# Source Citation Standardization Plan

## 📊 Current State Analysis

### Issues Identified:
1. **150+ inconsistent source citations** across 15+ QMD files
2. **7 different formats** currently in use
3. **Mixed academic/commercial** citation styles
4. **Inconsistent punctuation** and capitalization

### Current Patterns Found:

| Pattern | Count | Examples | Status |
|---------|-------|----------|---------|
| `Source: [@citation].` | ~30 | `Source: [@kiela2021dynabench].` | ✅ **GOOD** |
| `*Source: text*` | ~50+ | `*Source: Google*` | ❌ **NEEDS FIX** |
| `*Source: [link]*` | ~20 | `*Source: [Tesla](url)*` | ❌ **NEEDS FIX** |
| `*Source: @citation*` | ~10 | `*Source: @chen2023*` | ❌ **NEEDS FIX** |
| `Source: Company` | ~15 | `Source: NVIDIA` | ⚠️ **NEEDS PERIOD** |
| `Source: @citation` | ~5 | `Source: @wu2022sustainable` | ⚠️ **NEEDS BRACKETS** |
| `{Source: text}` | ~1 | `{Source: ABI Research}` | ❌ **NEEDS FIX** |

## 🎯 Recommended Standard

### **Academic Citations (Quarto):**
```markdown
Source: [@citation].
```
**Benefits:** 
- Automatic bibliography integration
- Clean (Author, Year) formatting  
- Standard academic practice

### **Company/Organization Sources:**
```markdown
Source: Company Name.
```
**Benefits:**
- Clear attribution
- Professional appearance
- Consistent punctuation

### **Linked Sources:**
```markdown
Source: [Display Text](URL).
```
**Benefits:**
- Clickable attribution
- Descriptive text
- Standard markdown format

## 🚀 Implementation Options

### **Option 1: Automated Script (Recommended)**

**Pros:**
- ✅ **Fast** - processes all files in ~30 seconds
- ✅ **Comprehensive** - handles all 7 pattern types
- ✅ **Consistent** - applies rules uniformly
- ✅ **Reversible** - git tracks all changes

**Cons:**
- ⚠️ Requires review of changes
- ⚠️ May need minor manual adjustments

**Usage:**
```bash
./standardize_sources.sh
```

### **Option 2: Manual Step-by-Step**

**Pros:**
- ✅ **Full control** over each change
- ✅ **Learning** - understand patterns
- ✅ **Selective** - fix specific files

**Cons:**
- ❌ **Time-consuming** - 2-3 hours for all files
- ❌ **Error-prone** - may miss patterns
- ❌ **Tedious** - 150+ individual fixes

**Process:**
1. Use search/replace in editor
2. Process one pattern type at a time
3. Verify each file individually

## 📋 Quality Assurance Plan

### **Before Running:**
```bash
# Count current inconsistencies
grep -r "\*Source:" contents --include="*.qmd" | wc -l
```

### **After Running:**
```bash
# Verify standardization
grep -r "Source:" contents --include="*.qmd" | head -20

# Check for any remaining asterisk patterns
grep -r "\*Source:" contents --include="*.qmd"

# Verify academic citations
grep -r "Source: \[@" contents --include="*.qmd" | head -10
```

### **Expected Results:**
- **0** asterisk-wrapped sources (`*Source:*`)
- **~40** academic citations (`Source: [@citation].`)
- **~60** company sources (`Source: Company.`)
- **~30** linked sources (`Source: [Text](URL).`)

## 🔧 Rollback Plan

If issues occur:
```bash
# Restore original state
git checkout -- contents/

# Or restore specific files
git checkout -- contents/core/benchmarking/benchmarking.qmd
```

## 📈 Benefits of Standardization

### **Academic Standards:**
- Follows Quarto documentation guidelines
- Matches university style guides
- Professional textbook appearance

### **Technical Benefits:**
- Consistent bibliography generation
- Reliable cross-referencing
- Future automation compatibility

### **Maintenance Benefits:**
- Easier future updates
- Clearer content editing
- Reduced formatting errors

## 💡 Recommendation

**Use the automated script** for the following reasons:

1. **Scale** - 150+ citations need fixing
2. **Consistency** - uniform application of rules  
3. **Speed** - 30 seconds vs. 3 hours
4. **Safety** - git provides complete rollback
5. **Quality** - systematic pattern matching

**Next Steps:**
1. Run `./standardize_sources.sh`
2. Review git diff to verify changes
3. Test render a few files to confirm
4. Commit standardized format
5. Add to style guide for future content 