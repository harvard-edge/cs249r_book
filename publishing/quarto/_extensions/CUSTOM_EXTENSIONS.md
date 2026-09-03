# 🛡️ CUSTOM EXTENSIONS - PROTECTED BY MLSYSBOOK-EXT NAMESPACE

This directory contains extensions that have been **heavily customized** for the MLSysBook project.

## ✅ **PROTECTION STRATEGY: MLSYSBOOK-EXT NAMESPACE**

Our custom extensions are **protected by organizing them under the `mlsysbook-ext/` directory** - a namespace that cannot be accidentally overwritten by `quarto add` commands.

## 📋 **Custom Extensions Inventory**

### ✅ **Safe to Reinstall** (Standard Extensions)
- `pandoc-ext/diagram` - Standard Pandoc extension
- `quarto-ext/lightbox` - Standard Quarto extension

### 🛡️ **PROTECTED** (Custom Extensions in MLSysBook-Ext Namespace)

#### `mlsysbook-ext/custom-numbered-blocks`
- **Status**: Heavily modified for MLSysBook
- **Protection**: `mlsysbook-ext/` namespace prevents any accidental overwrites
- **Customizations**:
  - Modified CSS styling
  - Custom block types
  - Enhanced PDF rendering
  - MLSysBook-specific configurations
- **Original**: `ute/custom-numbered-blocks` (https://github.com/ute/custom-numbered-blocks)
- **Last Updated**: Reorganized into mlsysbook-ext namespace

#### `mlsysbook-ext/titlepage`
- **Status**: Customized for MLSysBook
- **Protection**: `mlsysbook-ext/` namespace prevents any accidental overwrites
- **Customizations**:
  - Custom fonts and styling
  - Modified template partials
  - MLSysBook branding integration
- **Original**: `nmfs-opensci/titlepage` (https://github.com/nmfs-opensci/quarto_titlepages)
- **Last Updated**: Reorganized into mlsysbook-ext namespace

#### `mlsysbook-ext/margin-video`
- **Status**: 100% Custom Extension
- **Protection**: `mlsysbook-ext/` namespace (no external equivalent exists)
- **Purpose**: YouTube video embedding as margin notes with auto-numbering
- **Features**:
  - HTML: Responsive iframe in margin with CSS auto-numbering
  - PDF: QR codes with LaTeX margin notes
  - YouTube URL validation and error handling
- **Original**: N/A (Created specifically for this project)

#### `mlsysbook-ext/pseudocode`
- **Status**: Customized for MLSysBook
- **Protection**: `mlsysbook-ext/` namespace prevents any accidental overwrites
- **Customizations** (all in `pseudocode.lua`):
  - MIT Press cross-reference casing: `@algo-` renders lowercase "algorithm N" mid-sentence, `@Algo-` renders "Algorithm N" at sentence start
  - Right-aligned italic `▷` comment styling (CLRS column convention) via `pdf-comment-delimiter` / `pdf-italic-comment` / `pdf-right-comment` defaults
  - Static EPUB renderer (upstream supports HTML via pseudocode.js + PDF via algpseudocodex only; EPUB previously dumped raw `\begin{algorithm}`)
- **Original**: `leovan/pseudocode` (https://github.com/leovan/quarto-pseudocode), v1.5.0
- **Last Updated**: Reorganized into mlsysbook-ext namespace and patched for 3-format support (algorithm blocks pass, 2026-06)

## 🔒 **How Protection Works**

**Namespace Strategy**:
All our custom extensions live in `mlsysbook-ext/` directory.

When someone runs `quarto add ute/custom-numbered-blocks`, it will:
- ✅ **NOT** affect our `mlsysbook-ext/custom-numbered-blocks` directory
- ✅ Create a new `ute/custom-numbered-blocks` directory
- ✅ Leave our customizations completely intact

**No conflicts possible** - the `mlsysbook-ext/` namespace is exclusive to this project!

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
quarto add ute/custom-numbered-blocks     # Creates ute/ (separate from mlsysbook-ext/)
quarto add nmfs-opensci/titlepage         # Creates nmfs-opensci/ (separate from mlsysbook-ext/)

# Our customizations remain in mlsysbook-ext/ untouched
```

## 📝 **Adding New Extensions**

When adding new extensions:
1. **Standard extensions**: Install normally (`quarto add`)
2. **Custom extensions**: Place in `mlsysbook-ext/` directory
3. Document here whether they're standard or custom
4. Update this inventory
5. Commit changes to git

## 🆘 **Recovery Procedures**

If an extension is accidentally overwritten:
1. `git checkout HEAD -- book/_extensions/mlsysbook-ext/EXTENSION_NAME/`
2. Review git history for any recent custom changes
3. Test thoroughly before committing

---
**Last Updated**: 2024-12-08
**Maintainer**: MLSysBook Team
