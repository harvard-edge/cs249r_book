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

<table width="100%">
  <thead>
<tr>
<th width="40%"><b>Extension</b></th>
<th width="35%">Status</th>
<th width="25%">Safe to Update?</th>
</tr>
</thead>
<tbody>
<tr><td><b>`margin-video`</b></td><td>🚫 **100% Custom**</td><td>Never (MLSysBook only)</td></tr>
<tr><td><b>`ute/custom-numbered-blocks`</b></td><td>🚫 **Heavy Customization**</td><td>Never</td></tr>
<tr><td><b>`nmfs-opensci/titlepage`</b></td><td>🚫 **Customized**</td><td>Never</td></tr>
<tr><td><b>`pandoc-ext/diagram`</b></td><td>✅ Standard</td><td>Yes</td></tr>
<tr><td><b>`quarto-ext/lightbox`</b></td><td>✅ Standard</td><td>Yes</td></tr>
</tbody>
</table>

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
