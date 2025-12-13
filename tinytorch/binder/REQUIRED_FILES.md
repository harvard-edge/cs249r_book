# Required Files Based on paper.tex

## Exact File References in Paper

### Line 988: Repository Instructor Resources

The paper explicitly states:
> "The repository includes instructor resources: \texttt{CONTRIBUTING.md} (guidelines for bug reports and curriculum improvements), \texttt{INSTRUCTOR.md} (30-minute setup guide, grading rubrics, common student errors), and \texttt{MAINTENANCE.md} (support commitment through 2027, succession planning for community governance)."

**Required Files**:
1. ✅ `CONTRIBUTING.md` - Guidelines for bug reports and curriculum improvements
2. ✅ `INSTRUCTOR.md` - 30-minute setup guide, grading rubrics, common student errors
3. ❌ `MAINTENANCE.md` - **Removed per user request** (paper mentions it but user doesn't want it)

### Line 999: TA Guide

The paper explicitly states:
> "The repository provides \texttt{TA\_GUIDE.md} documenting frequent student errors (gradient shape mismatches, disconnected computational graphs, broadcasting failures) and debugging strategies."

**Required File**:
4. ✅ `TA_GUIDE.md` - Frequent student errors and debugging strategies

### Line 1003: Sample Solutions

The paper states:
> "Sample solutions and grading rubrics in \texttt{INSTRUCTOR.md} calibrate evaluation standards."

**Required Content** (must be in INSTRUCTOR.md):
- Sample solutions (for grading calibration)
- Grading rubrics

## Summary: Required Files

| File | Purpose | Status |
|------|---------|--------|
| `CONTRIBUTING.md` | Bug reports, curriculum improvements | ✅ Exists |
| `INSTRUCTOR.md` | Setup guide, grading rubrics, common errors, sample solutions | ✅ Created |
| `TA_GUIDE.md` | Common errors, debugging strategies | ✅ Created |

## Content Verification

### CONTRIBUTING.md ✅
- Guidelines for bug reports ✅
- Guidelines for curriculum improvements ✅

### INSTRUCTOR.md ✅
- 30-minute setup guide ✅ (Section: "Instructor Setup")
- Grading rubrics ✅ (Section: "Grading Rubric for ML Systems Questions")
- Common student errors ✅ (Section: "Troubleshooting" → "Common Student Issues")
- Sample solutions ⚠️ (Mentioned but need to verify if included)

### TA_GUIDE.md ✅
- Gradient shape mismatches ✅
- Disconnected computational graphs ✅
- Broadcasting failures ✅
- Debugging strategies ✅

## Files NOT Required by Paper

These files exist but are NOT explicitly mentioned in the paper:
- `TEAM_ONBOARDING.md` - Not mentioned (but Model 3 is described in text)
- `MAINTENANCE.md` - Mentioned but removed per user request
- `docs/STUDENT_QUICKSTART.md` - Not explicitly mentioned
- `site/` documentation - Not explicitly mentioned (but needed for website)

## Action Items

1. ✅ Remove MAINTENANCE.md (done)
2. ✅ Verify CONTRIBUTING.md matches paper description
3. ⚠️ Verify INSTRUCTOR.md has sample solutions (need to check/add if missing)
4. ✅ Verify TA_GUIDE.md has all required errors

## Note on MAINTENANCE.md

The paper mentions `MAINTENANCE.md` but the user doesn't want it. The maintenance commitment information (support through 2027, etc.) is described in the paper text but doesn't need to be in a separate file if the user prefers not to have it.
