# MLSysBook Extensions

This directory contains Quarto extensions used by the MLSysBook project.

## ⚠️ CRITICAL WARNING - READ BEFORE MAKING CHANGES

**Some extensions in this directory are HEAVILY CUSTOMIZED and should NEVER be reinstalled using `quarto add`.**

### 🔍 Quick Safety Check

Before installing or updating any extension, **ALWAYS** check for:
1. `.CUSTOM_LOCK` files in extension directories
2. Version numbers ending in `-mlsysbook-custom`
3. Warnings in extension `_extension.yml` files

### 📋 Extension Inventory

| Extension | Status | Safe to Update? |
|-----------|--------|-----------------|
| `margin-video` | 🚫 **100% Custom** | Never (MLSysBook only) |
| `ute/custom-numbered-blocks` | 🚫 **Heavy Customization** | Never |
| `nmfs-opensci/titlepage` | 🚫 **Customized** | Never |
| `pandoc-ext/diagram` | ✅ Standard | Yes |
| `quarto-ext/lightbox` | ✅ Standard | Yes |

### 🚨 Emergency Recovery

If you accidentally overwrote a custom extension:
```bash
# Check what changed
git status

# Restore from git if needed
git checkout HEAD -- book/_extensions/EXTENSION_NAME/

# Run protection check
python tools/scripts/check_custom_extensions.py
```

### 📚 Full Documentation

For complete details, see: [`CUSTOM_EXTENSIONS.md`](./CUSTOM_EXTENSIONS.md)

---
**When in doubt, DON'T reinstall - ask the team first!**
