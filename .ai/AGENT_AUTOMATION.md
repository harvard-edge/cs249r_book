# AI Agent Automation Rules for MLSysBook

This file defines rules and guidelines for AI agents (Claude, Gemini, GPT, etc.) working on the MLSysBook project. These agents can autonomously handle GitHub issues, content improvements, and maintenance tasks while following strict safety and quality protocols.

**Tool Agnostic**: These rules apply to any AI assistant or automation tool used with this project.

## 🤖 Agent Types & Capabilities

### Issue Resolution Agent
**Trigger Patterns:**
- `resolve issue #XXX` - Full automated resolution
- `analyze issue #XXX` - Analysis only, no changes
- `draft issue #XXX` - Create draft solution for review

**Capabilities:**
- Fetch and analyze GitHub issues
- Classify issue types (CONTENT_ACCURACY, CONTENT_CLARITY, STRUCTURAL, TECHNICAL)
- Create appropriate feature branches
- Apply content management tools
- Generate commit messages and PR descriptions
- Handle cross-references and section IDs

**Limitations:**
- Cannot merge PRs (human approval required)
- Cannot publish to live site
- Cannot modify core build configurations
- Must create draft PRs for review

### Content Maintenance Agent
**Trigger Patterns:**
- `fix broken links` - Automated link validation and fixing
- `update cross-references` - Cross-reference maintenance
- `clean formatting` - Standardize formatting across chapters

**Capabilities:**
- Run existing content management tools
- Fix obvious formatting issues
- Update cross-references when sections change
- Validate and fix broken internal links

**Limitations:**
- Cannot change content meaning or technical accuracy
- Cannot add new content without human review
- Cannot modify chapter structure

## 🔒 Safety Protocols

### Mandatory Safety Checks
1. **Always create feature branches** - Never work directly on dev/main
2. **Pre-commit validation** - All changes must pass pre-commit hooks
3. **Build verification** - Must verify book builds successfully
4. **Backup creation** - Create backups before destructive operations
5. **Human escalation** - Escalate complex or ambiguous issues

### Prohibited Actions
- **Never force push** to any branch
- **Never delete files** without explicit permission
- **Never modify build configurations** without human review
- **Never publish to live site** automatically
- **Never merge PRs** without human approval
- **Never commit sensitive data** (API keys, credentials)

### Error Handling
- **Graceful degradation** - If automation fails, provide clear error messages
- **Rollback capability** - Always maintain ability to undo changes
- **Logging** - Comprehensive logging of all actions taken
- **Escalation triggers** - Automatic escalation for undefined scenarios

## 📋 Issue Classification & Handling

### CONTENT_ACCURACY Issues
**Automated Actions:**
- Verify technical details against authoritative sources
- Check consistency with other chapters
- Update cross-references if needed
- Run section ID validation

**Human Escalation Required:**
- Technical corrections requiring domain expertise
- Changes affecting multiple chapters
- Controversial or subjective content

### CONTENT_CLARITY Issues
**Automated Actions:**
- Improve figure captions using AI tools
- Fix obvious formatting inconsistencies
- Update cross-references for clarity
- Standardize terminology usage

**Human Escalation Required:**
- Rewriting explanations
- Adding new examples or analogies
- Restructuring content flow

### STRUCTURAL Issues
**Automated Actions:**
- Update table of contents if sections added
- Fix cross-reference formatting
- Standardize section ID formats
- Update navigation elements

**Human Escalation Required:**
- Adding new sections or chapters
- Reorganizing content structure
- Creating new exercises or examples

### TECHNICAL Issues
**Automated Actions:**
- Fix broken internal links
- Update cross-references
- Correct formatting errors
- Validate build process

**Human Escalation Required:**
- Build configuration changes
- External link validation
- Complex formatting issues

## 🔄 Automated Workflow Process

### 1. Issue Analysis Phase
```
1. Fetch issue from GitHub API
2. Parse issue content and classify type
3. Identify relevant files and sections
4. Assess complexity and automation feasibility
5. Determine if human escalation needed
```

### 2. Branch Creation Phase
```
1. Ensure dev branch is up to date
2. Create feature branch: fix/issue-XXX-description
3. Verify branch creation successful
4. Set up tracking for changes
```

### 3. Implementation Phase
```
1. Apply appropriate content management tools
2. Make targeted changes based on issue type
3. Validate changes don't break build
4. Run pre-commit hooks
5. Verify all tests pass
```

### 4. Review & Submission Phase
```
1. Generate comprehensive change summary
2. Create commit with conventional format
3. Push branch to remote
4. Create DRAFT pull request
5. Tag human reviewers for approval
```

## 🎯 Quality Standards for Agents

### Technical Accuracy
- **Verify all technical claims** against authoritative sources
- **Maintain consistency** with existing content
- **Preserve pedagogical progression** across chapters
- **Ensure examples remain functional**

### Content Quality
- **Maintain academic writing standards**
- **Preserve consistent terminology**
- **Keep appropriate difficulty level** for chapter position
- **Ensure cross-references remain valid**

### Code Quality
- **Follow PEP 8** for all Python code
- **Include type hints** for new functions
- **Add docstrings** for new classes/methods
- **Ensure backward compatibility**

## 🚨 Escalation Triggers

### Immediate Human Escalation Required:
- **Ambiguous issue description** - Cannot determine clear resolution path
- **Multiple valid solutions** - Requires human judgment
- **Cross-chapter impact** - Changes affecting multiple chapters
- **Technical controversy** - Conflicting authoritative sources
- **Build failures** - Cannot resolve build issues automatically
- **Security concerns** - Any potential security implications
- **Policy questions** - Issues requiring policy decisions

### Escalation Process:
1. **Stop all automated actions** immediately
2. **Document current state** and actions taken
3. **Create detailed escalation report**
4. **Tag appropriate human reviewers**
5. **Provide recommended next steps**

## 📊 Monitoring & Reporting

### Success Metrics
- **Issue resolution time** - Track time from trigger to PR creation
- **Success rate** - Percentage of issues successfully automated
- **Build stability** - Ensure automated changes don't break builds
- **Human satisfaction** - Quality of automated solutions

### Failure Analysis
- **Common failure patterns** - Identify recurring issues
- **Improvement opportunities** - Areas where automation can be enhanced
- **Safety incidents** - Any violations of safety protocols
- **Performance bottlenecks** - Slow or inefficient processes

### Reporting Format
```
## Automated Issue Resolution Report
- **Issue**: #XXX - [Title]
- **Classification**: [Type]
- **Actions Taken**: [List of automated actions]
- **Files Modified**: [List of changed files]
- **Tools Applied**: [Content management tools used]
- **Build Status**: [Success/Failure]
- **Human Review Required**: [Yes/No - Reason]
- **Estimated Time Saved**: [Hours]
```

## 🔧 Integration with Existing Tools

### Content Management Tools
- **`manage_section_ids.py`** - Automatic section ID management
- **`improve_figure_captions.py`** - AI-powered caption enhancement
- **`find_unreferenced_labels.py`** - Label validation
- **`fix_bibliography.py`** - Bibliography formatting
- **`cross_refs/manage_cross_references.py`** - Cross-reference updates

### Quality Assurance Tools
- **Pre-commit hooks** - Mandatory for all changes
- **Build validation** - Quarto render verification
- **Link checking** - Internal and external link validation
- **Test suite** - Automated testing where applicable

### Version Control Integration
- **Branch management** - Automated branch creation and cleanup
- **Commit formatting** - Conventional commit message generation
- **PR creation** - Automated draft PR generation
- **Merge prevention** - No automated merging allowed

## 🎓 Learning & Improvement

### Feedback Loops
- **Human feedback** on automated solutions
- **Success/failure pattern analysis**
- **Continuous improvement** of automation rules
- **Regular review** of escalation triggers

### Adaptation Mechanisms
- **Rule refinement** based on experience
- **New tool integration** as they become available
- **Process optimization** for common scenarios
- **Safety protocol updates** as needed

## 📝 Configuration Management

### Environment Variables
- **GITHUB_TOKEN** - For API access (read-only recommended)
- **OPENAI_API_KEY** - For AI-powered tools (if used)
- **BUILD_TIMEOUT** - Maximum time for build verification
- **MAX_FILES_CHANGED** - Safety limit on file modifications

### Feature Flags
- **ENABLE_AUTO_COMMIT** - Allow automatic commits (default: false)
- **ENABLE_AUTO_PR** - Allow automatic PR creation (default: true)
- **ENABLE_BUILD_CHECK** - Require build verification (default: true)
- **ENABLE_HUMAN_REVIEW** - Require human review (default: true)

### Rate Limits
- **API_CALLS_PER_HOUR** - GitHub API rate limiting
- **MAX_CONCURRENT_ISSUES** - Parallel processing limit
- **PROCESSING_TIMEOUT** - Maximum time per issue
- **RETRY_ATTEMPTS** - Number of retries for failed operations

## 🔄 Version Control for Agent Rules

### Change Management
- **All rule changes** must be reviewed by humans
- **Version tracking** for rule modifications
- **Rollback capability** for problematic rule changes
- **Testing environment** for rule validation

### Documentation Requirements
- **Change rationale** for all rule modifications
- **Impact assessment** on existing automation
- **Migration guide** for breaking changes
- **Performance implications** of new rules

---

**Last Updated**: [Current Date]
**Version**: 1.0
**Review Cycle**: Monthly
**Next Review**: [Date]
