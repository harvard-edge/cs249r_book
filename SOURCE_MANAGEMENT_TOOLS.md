# Source Citation Management Tools

This document describes the comprehensive set of tools for managing and maintaining source citations in the ML Systems textbook.

## 🛠️ **Available Tools**

### 1. **`check_sources.py`** - Primary Analysis & Cleanup Tool ⭐

**Purpose**: Comprehensive Python tool for analyzing, validating, and cleaning source citations

**Key Features**:
- ✅ **Smart pattern detection** - Finds academic, company, and link sources
- ✅ **Detailed problem identification** - Line-level error reporting
- ✅ **Automatic cleanup** - Fixes common formatting issues
- ✅ **JSON reporting** - Machine-readable analysis results
- ✅ **Consistent with existing tools** - Same Python codebase as `improve_figure_captions.py`

**Usage**:
```bash
# Quick analysis (all files)
python3 check_sources.py --analyze

# Analyze specific directory
python3 check_sources.py --analyze -d contents/core/ml_systems

# Analyze specific file
python3 check_sources.py --analyze --files contents/core/workflow/workflow.qmd

# Clean up specific directories
python3 check_sources.py --clean -d contents/core contents/labs

# Full analysis with detailed report
python3 check_sources.py --full
```

### 2. **`standardize_sources.sh`** - Batch Standardization Script

**Purpose**: One-time comprehensive standardization of all source citation formats

**Usage**:
```bash
./standardize_sources.sh
```

**What it fixes**:
- `*Source: @citation*` → `Source: [@citation].`
- `*Source: [text](url)*` → `Source: [text](url).`
- `*Source: text*` → `Source: text.`
- Missing periods and bracket formatting

### 3. **`check_sources.sh`** - Shell Analysis Tool

**Purpose**: Quick shell-based analysis and cleanup (legacy tool)

**Usage**:
```bash
./check_sources.sh --analyze
./check_sources.sh --clean
```

## 📊 **Citation Format Standards**

### **Academic Citations**
```markdown
Source: [@citation].
```
- ✅ Quarto bibliography integration
- ✅ Automatic (Author, Year) formatting
- ✅ Clickable in digital versions

### **Company/Organization Sources**
```markdown
Source: Company Name.
```
- ✅ Clear attribution
- ✅ Consistent punctuation
- ✅ Professional appearance

### **Link Sources**
```markdown
Source: [Display Text](URL).
```
- ✅ Clickable attribution
- ✅ Descriptive text
- ✅ Standard markdown format

## 🎯 **Targeting Options**

### **All Files** (Default)
```bash
python3 check_sources.py --analyze
```
- Processes all QMD files in `contents/` directory
- Comprehensive textbook-wide analysis

### **Specific Directory**
```bash
python3 check_sources.py --analyze -d contents/core/ml_systems
python3 check_sources.py --clean -d contents/core contents/labs
```
- Process specific directories and subdirectories
- Ideal for working on individual chapters
- Supports multiple directories at once

### **Specific Files**
```bash
python3 check_sources.py --analyze --files workflow.qmd introduction.qmd
python3 check_sources.py --clean --files contents/core/ops/ops.qmd
```
- Process individual QMD files
- Perfect for targeted fixes after editing
- Supports multiple files at once

### **Scope Validation**
- ✅ Cannot combine `--files` and `-d` options
- ✅ Provides clear feedback about processing scope
- ✅ Shows file counts and directory information

## 🔍 **Analysis Capabilities**

### **Pattern Detection**
The tools automatically detect and count:
- Academic citations: `Source: [@citation]`
- Company sources: `Source: Company Name`
- Link sources: `Source: [Text](URL)`
- Problematic asterisk patterns: `*Source:*`
- Missing punctuation
- Double periods
- Malformed citations

### **Problem Identification**
Provides detailed reports showing:
- File path and line number
- Exact problematic text
- Recommended fixes
- Overall statistics

### **Quality Metrics**
Tracks across entire textbook:
- Total sources by type
- Files with source issues
- Success rates after cleanup
- Compliance with standards

## 🚀 **Recommended Workflow**

### **Daily Development**
```bash
# Check status quickly (all files)
python3 check_sources.py --analyze

# Check specific chapter you're working on
python3 check_sources.py --analyze -d contents/core/chapter_name

# Fix issues in specific area
python3 check_sources.py --clean -d contents/core/chapter_name

# Check specific file after editing
python3 check_sources.py --analyze --files path/to/file.qmd
```

### **Quality Assurance**
```bash
# Complete analysis with report
python3 check_sources.py --full

# Review generated report
cat source_analysis_report.json
```

### **Before Publishing**
```bash
# Ensure all sources are standardized
python3 check_sources.py --analyze

# Should show: 0 problematic sources
```

## 📁 **File Organization**

```
├── check_sources.py              # 🌟 Primary Python tool
├── standardize_sources.sh         # One-time standardization
├── check_sources.sh              # Shell alternative
├── source_analysis_report.json   # Generated analysis report
├── SOURCE_STANDARDIZATION_PLAN.md # Original analysis
└── SOURCE_MANAGEMENT_TOOLS.md     # This documentation
```

## 🎯 **Current Status**

As of latest analysis:
- ✅ **149 total sources** properly formatted
- ✅ **45 academic citations** with Quarto integration
- ✅ **47 company sources** with consistent punctuation
- ✅ **57 link sources** in standard markdown format
- ❌ **0 problematic asterisk sources** (cleaned up!)
- ⚠️ **7 double periods** (minor formatting issue)

## 💡 **Best Practices**

### **For Content Authors**
1. **Use the standard formats** from the beginning
2. **Academic sources**: Always use `Source: [@citation].`
3. **Company sources**: Include period: `Source: Company.`
4. **Links**: Use descriptive text: `Source: [Description](URL).`

### **For Maintainers**
1. **Run analysis regularly** during development
2. **Use `--clean` for automatic fixes** before commits
3. **Generate reports** for quality tracking
4. **Update standards** as needed in the tools

### **For Quality Control**
1. **Check before major releases** using `--full` analysis
2. **Monitor trends** in source usage over time
3. **Validate** after bulk content changes
4. **Maintain consistency** across all chapters

## 🔮 **Future Enhancements**

Potential improvements to consider:
- **Integration with `improve_figure_captions.py`** for unified caption management
- **Automatic source validation** against bibliography files
- **Style guide enforcement** for source text formatting
- **Integration with CI/CD** for automated quality checks
- **Visual reports** showing source distribution across chapters

## 🤝 **Integration with Existing Tools**

These source management tools complement:
- **`improve_figure_captions.py`** - For figure caption quality
- **Quarto bibliography system** - For automatic citation formatting
- **Git workflow** - For tracking changes and quality over time
- **Style validation** - For maintaining professional standards

---

**📧 For questions or improvements, consult the branch management guide in `BRANCH_WORKFLOW.md`** 