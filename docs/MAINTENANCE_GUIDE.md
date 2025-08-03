# 🛠️ MLSysBook Maintenance & Daily Workflow Guide

This guide explains your daily development workflow, maintenance tasks, and how to leverage all the automation we've built to keep the project running smoothly.

## 🌅 Daily Development Workflow

### **Morning Routine (Project Sync)**
```bash
# 1. Sync with latest changes
git pull origin main

# 2. Clean workspace and check health
make clean
make check

# 3. Preview what's currently in the project
make status
```

**What this does:**
- Pulls latest changes from the team
- Cleans any leftover build artifacts
- Shows project health (file counts, dependencies, git status)
- Gives you a quick overview of current state

### **Active Development Session**
```bash
# 1. Start development server (runs in background)
make preview &

# 2. Open browser to preview URL (usually http://localhost:3000)
# 3. Edit content in your favorite editor
# 4. Changes automatically reload in browser

# When ready to test changes:
make test           # Run validation
make lint           # Check for issues
```

**Your workflow:**
1. **Edit content** in `book/contents/` directories
2. **See changes instantly** in browser (auto-reload)
3. **Run quick checks** with `make lint` 
4. **Validate** with `make test` before committing

### **End of Session (Commit)**
```bash
# 1. Final quality check
make clean test

# 2. Stage and commit (pre-commit hooks run automatically)
git add .
git commit -m "Your descriptive commit message"

# 3. Push when ready
git push
```

**What happens automatically:**
- ✅ **Pre-commit hook** cleans artifacts and checks for issues
- ✅ **Build artifacts** are automatically ignored
- ✅ **Large files** and potential secrets are flagged
- ✅ **Only clean commits** are allowed

---

## 🗓️ Weekly Maintenance Tasks

### **Monday: Project Health Check**
```bash
# Comprehensive project review
make check              # Overall health
make clean-dry          # See what needs cleaning
make lint               # Content quality check

# Review any issues
find . -name "*.log" -newer $(date -d '7 days ago' +%Y%m%d) | head -10
git log --oneline -10   # Recent changes
```

### **Wednesday: Content Quality**
```bash
# Run content-specific tools
python tools/scripts/content/find_unreferenced_labels.py
python tools/scripts/content/find_duplicate_labels.py
python tools/scripts/utilities/check_ascii.py
python tools/scripts/utilities/check_images.py
```

### **Friday: Maintenance & Cleanup**
```bash
# Deep clean and update
make clean-deep         # Full cleanup
make build-all          # Test all formats
python tools/scripts/maintenance/generate_release_content.py
```

---

## 🔧 Monthly Maintenance Tasks

### **First Monday of Month: Dependencies**
```bash
# Update dependencies
python tools/scripts/maintenance/update_texlive_packages.py
pip list --outdated    # Check Python packages
make install           # Reinstall if needed

# Test everything still works
make clean build test
```

### **Mid-Month: Content Optimization**
```bash
# Improve content quality
python tools/scripts/content/improve_figure_captions.py
python tools/scripts/content/clean_callout_titles.py
python tools/scripts/content/collapse_blank_lines.py
python tools/scripts/content/sync_bibliographies.py
```

### **End of Month: Performance Review**
```bash
# Generate project statistics
python tools/scripts/build/generate_stats.py

# Clean up old runs and logs
python tools/scripts/maintenance/cleanup_old_runs.sh

# Review project structure
make status
du -sh build/           # Check output size
```

---

## 🎯 Scenario-Based Workflows

### **📝 Working on Content**

**Adding New Chapter:**
```bash
# 1. Create content structure
mkdir -p book/contents/core/new_chapter/{images/png,images/svg}
touch book/contents/core/new_chapter/new_chapter.{qmd,bib}
touch book/contents/core/new_chapter/new_chapter_quizzes.json

# 2. Add to book configuration
# Edit book/_quarto-html.yml to add chapter to chapters list
# Edit book/_quarto-pdf.yml to add chapter to chapters list
# Edit book/_quarto-html.yml to add .bib to bibliography list
# Edit book/_quarto-pdf.yml to add .bib to bibliography list

# 3. Test the build
make clean build
```

**Editing Existing Content:**
```bash
# 1. Start preview server
make preview

# 2. Edit files in book/contents/
# 3. Check for issues as you go
make lint               # Quick content check

# 4. Validate when done
make test
```

**Content Quality Pass:**
```bash
# Run all content quality tools
python tools/scripts/content/find_unreferenced_labels.py
python tools/scripts/content/find_duplicate_labels.py
python tools/scripts/utilities/check_sources.py
python tools/scripts/utilities/check_ascii.py
```

### **🔧 Working on Build System**

**Modifying Scripts:**
```bash
# 1. Find the right script category
ls tools/scripts/       # See categories
cat tools/scripts/README.md  # Get overview

# 2. Edit scripts in appropriate category
# 3. Test the change
./tools/scripts/category/your_script.py --dry-run

# 4. Update documentation if needed
# Edit tools/scripts/category/README.md
```

**Adding New Automation:**
```bash
# 1. Choose appropriate category
# build/ - for build/development tools
# content/ - for content management
# utilities/ - for general utilities
# maintenance/ - for system maintenance

# 2. Follow naming conventions
# verb_noun.py or verb_noun.sh
# Full words, no abbreviations

# 3. Add to category README
# Update tools/scripts/category/README.md

# 4. Test integration with Makefile if needed
```

### **🚨 Troubleshooting Common Issues**

**Build Fails:**
```bash
# 1. Clean everything
make clean-deep

# 2. Check dependencies
make check

# 3. Try minimal build
cd book && quarto render index.qmd --to html

# 4. Check logs
ls -la *.log
```

**Git Hook Blocks Commit:**
```bash
# 1. See what's being blocked
git status

# 2. Clean artifacts
make clean

# 3. Check for large files
find . -size +1M -type f | grep -v .git | grep -v .venv

# 4. Review staged changes
git diff --cached
```

**Content Issues:**
```bash
# 1. Run comprehensive checks
python tools/scripts/utilities/check_sources.py

# 2. Find specific issues
python tools/scripts/content/find_duplicate_labels.py
python tools/scripts/utilities/check_ascii.py

# 3. Fix bibliography issues
python tools/scripts/content/fix_bibliography.py
```

---

## 📊 Understanding Your Tools

### **🔨 Build Tools (`tools/scripts/build/`)**
- **`clean.sh`** - Your daily cleanup tool (use often!)
- **`generate_stats.py`** - Project insights and metrics
- **`standardize_sources.sh`** - Format consistency

### **📝 Content Tools (`tools/scripts/content/`)**
- **`manage_section_ids.py`** - Cross-reference management
- **`improve_figure_captions.py`** - AI-powered caption enhancement
- **`find_unreferenced_labels.py`** - Cleanup unused references
- **`sync_bibliographies.py`** - Keep citations in sync

### **🛠️ Utility Tools (`tools/scripts/utilities/`)**
- **`check_sources.py`** - Comprehensive content validation
- **`check_ascii.py`** - Encoding issue detection
- **`check_images.py`** - Image validation and optimization

### **🔧 Maintenance Tools (`tools/scripts/maintenance/`)**
- **`generate_release_content.py`** - Automated changelog and release notes generation
- **`cleanup_old_runs.sh`** - Remove old build artifacts

---

## ⚡ Power User Tips

### **🚀 Efficiency Shortcuts**
```bash
# Quick health check
alias health="make clean && make check && make test"

# Fast content preview
alias preview="make clean build && make preview"

# Quality check before commit
alias precommit="make clean test lint"

# Full project rebuild
alias rebuild="make clean-deep && make install && make build-all"
```

### **📋 Monitoring Commands**
```bash
# Watch file changes during development
watch -n 2 "ls -la book/contents/core/your_chapter/"

# Monitor build logs
tail -f *.log

# Check project size
du -sh build/ book/ tools/
```

### **🔍 Quick Diagnostics**
```bash
# What's changed recently?
git log --oneline -5
git status --porcelain

# What needs attention?
make lint | head -20
find . -name "TODO" -o -name "FIXME"

# How big is everything?
make status
```

---

## 🎯 Success Indicators

### **✅ Daily Success Checklist**
- [ ] `make check` shows green status
- [ ] `make test` passes without errors
- [ ] `make preview` loads correctly
- [ ] Git commits go through without hook blocks
- [ ] No large files or artifacts in git status

### **✅ Weekly Success Checklist**
- [ ] All content quality tools run clean
- [ ] No broken links or references
- [ ] Build time remains reasonable
- [ ] Documentation stays up to date

### **✅ Monthly Success Checklist**
- [ ] Dependencies are up to date
- [ ] Project statistics show healthy growth
- [ ] No accumulation of old logs or artifacts
- [ ] All automation still working correctly

---

## 🆘 When Things Go Wrong

### **Emergency Recovery**
```bash
# Nuclear option - start fresh
git stash                    # Save current work
make clean-deep             # Clean everything
git reset --hard HEAD      # Reset to last commit
make install               # Reinstall dependencies
make setup-hooks           # Reconfigure hooks
git stash pop              # Restore your work
```

### **Getting Help**
1. **Check logs**: Look in `*.log` files for error details
2. **Run diagnostics**: `make check` for overall health
3. **Review documentation**: `DEVELOPMENT.md` for detailed guides
4. **Community support**: GitHub Discussions for help

### **Escalation Path**
1. **Try automated fixes**: Run relevant scripts in `tools/scripts/`
2. **Check recent changes**: `git log` to see what might have broken
3. **Isolate the issue**: Test with minimal configuration
4. **Document and report**: Create detailed issue with reproduction steps

---

## 🎓 Learning and Growth

### **Mastering the Tools**
- **Week 1-2**: Focus on daily workflow (`make clean`, `make preview`, `make test`)
- **Week 3-4**: Learn content tools (find duplicates, improve captions)
- **Month 2**: Master maintenance tools (updates, cleanup, statistics)
- **Month 3+**: Customize and extend automation for your needs

### **Becoming More Efficient**
- **Learn the scripts**: Explore `tools/scripts/` to understand capabilities
- **Customize workflows**: Add your own aliases and shortcuts
- **Contribute improvements**: Enhance tools based on your experience
- **Share knowledge**: Document new patterns and workflows

---

## 📈 Long-term Project Health

### **Sustainability Practices**
- **Regular cleanup**: Use `make clean` daily, `make clean-deep` weekly
- **Quality monitoring**: Run content tools weekly
- **Dependency updates**: Monthly maintenance cycles
- **Documentation currency**: Keep guides updated with changes

### **Growth Management**
- **Monitor build times**: Watch for performance degradation
- **Track project size**: Ensure efficient asset management
- **Review automation**: Update scripts as project evolves
- **Community health**: Engage with contributors and maintain standards

Your maintenance journey is now largely automated! Focus on content creation while the tools handle quality, consistency, and project health. 🚀 