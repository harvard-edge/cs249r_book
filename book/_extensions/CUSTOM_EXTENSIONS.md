# 🛡️ CUSTOM EXTENSIONS - PROTECTED BY MLSYSBOOK NAMESPACE

This directory contains extensions that have been **heavily customized** for the MLSysBook project. 

## ✅ **PROTECTION STRATEGY: MLSYSBOOK NAMESPACE**

Our custom extensions are **protected by organizing them under the `mlsysbook/` directory** - a namespace that cannot be accidentally overwritten by `quarto add` commands.

## 📋 **Custom Extensions Inventory**

### ✅ **Safe to Reinstall** (Standard Extensions)
- `pandoc-ext/diagram` - Standard Pandoc extension
- `quarto-ext/lightbox` - Standard Quarto extension  

### 🛡️ **PROTECTED** (Custom Extensions in MLSysBook Namespace)

#### `mlsysbook/custom-numbered-blocks` 
- **Status**: Heavily modified for MLSysBook
- **Protection**: `mlsysbook/` namespace prevents any accidental overwrites
- **Customizations**: 
  - Modified CSS styling
  - Custom block types
  - Enhanced PDF rendering
  - MLSysBook-specific configurations
- **Original**: `ute/custom-numbered-blocks` (https://github.com/ute/quarto-numbered-blocks)
- **Last Updated**: Reorganized into mlsysbook namespace

#### `mlsysbook/titlepage`
- **Status**: Customized for MLSysBook  
- **Protection**: `mlsysbook/` namespace prevents any accidental overwrites
- **Customizations**:
  - Custom fonts and styling
  - Modified template partials
  - MLSysBook branding integration
- **Original**: `nmfs-opensci/titlepage` (https://github.com/nmfs-opensci/quarto_titlepages)
- **Last Updated**: Reorganized into mlsysbook namespace

#### `mlsysbook/margin-video` 
- **Status**: 100% Custom Extension
- **Protection**: `mlsysbook/` namespace (no external equivalent exists)
- **Purpose**: YouTube video embedding as margin notes with auto-numbering
- **Features**:
  - HTML: Responsive iframe in margin with CSS auto-numbering
  - PDF: QR codes with LaTeX margin notes
  - YouTube URL validation and error handling
- **Original**: N/A (Created specifically for this project)

## 🔒 **How Protection Works**

**Namespace Strategy**: 
All our custom extensions live in `mlsysbook/` directory.

When someone runs `quarto add ute/custom-numbered-blocks`, it will:
- ✅ **NOT** affect our `mlsysbook/custom-numbered-blocks` directory
- ✅ Create a new `ute/custom-numbered-blocks` directory 
- ✅ Leave our customizations completely intact

**No conflicts possible** - the `mlsysbook/` namespace is exclusive to this project!

## 🚀 **Safe Update Procedures**

### For Standard Extensions:
```bash
# Safe to update these
quarto add pandoc-ext/diagram --update
quarto add quarto-ext/lightbox --update
```

### For Custom Extensions:
```bash
# All these are SAFE - they create separate directories:
quarto add ute/custom-numbered-blocks     # Creates ute/ (separate from mlsysbook/)
quarto add nmfs-opensci/titlepage         # Creates nmfs-opensci/ (separate from mlsysbook/)

# Our customizations remain in mlsysbook/ untouched
```

## 📝 **Adding New Extensions**

When adding new extensions:
1. **Standard extensions**: Install normally (`quarto add`)
2. **Custom extensions**: Place in `mlsysbook/` directory
3. Document here whether they're standard or custom
4. Update this inventory
5. Commit changes to git

## 🆘 **Recovery Procedures**

If an extension is accidentally overwritten:
1. `git checkout HEAD -- book/_extensions/mlsysbook/EXTENSION_NAME/`
2. Review git history for any recent custom changes
3. Test thoroughly before committing

---
**Last Updated**: 2024-12-08
**Maintainer**: MLSysBook Team