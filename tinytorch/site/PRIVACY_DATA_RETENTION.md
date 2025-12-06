# Privacy & Data Retention Policy

## Data Collection

TinyTorch collects **optional** information to build a community map and support learning:

- **Country** (optional) - For global visualization
- **Institution** (optional) - For cohort identification
- **Course Type** (optional) - For community insights
- **Experience Level** (optional) - For learning support

**We do NOT collect:**
- Personal names
- Email addresses (unless user provides)
- IP addresses
- Any personally identifiable information

## Anonymous Identification

All users are assigned an **anonymous UUID** when joining the community. This UUID:
- Cannot be linked to personal identity
- Is randomly generated
- Is stored locally in your project

## Data Storage

**Location**: `.tinytorch/` directory (project-local, not home directory)

**Files**:
- `.tinytorch/community/profile.json` - Your community profile
- `.tinytorch/config.json` - Configuration settings
- `.tito/benchmarks/` - Benchmark results
- `.tito/submissions/` - Submission files

**Privacy**: All data is stored locally in your project. You control what is shared.

## Data Retention

**Local Storage**: Data persists until you:
- Run `tito community leave` (removes profile)
- Delete `.tinytorch/` directory
- Remove specific files manually

**Website Sync** (when enabled):
- Data synced to website is retained according to website privacy policy
- You can request deletion via `tito community leave`
- Local data is always removed immediately

## User Rights

**Right to Access**: View your data with `tito community profile`

**Right to Update**: Update your data with `tito community update`

**Right to Deletion**: Remove your data with `tito community leave`

**Right to Opt-Out**: All data collection is optional. You can:
- Skip fields during `tito community join`
- Leave community anytime with `tito community leave`
- Never join community (all features work without joining)

## Consent

**Explicit Consent**: When joining, you'll see:
- What data is collected
- Why it's collected
- How it's stored
- Consent prompt before collection

**Withdrawal**: You can withdraw consent anytime by leaving the community.

## Website Integration

**Current**: Website integration is **disabled by default**. All data stays local.

**Future**: When website integration is enabled:
- You'll be notified before syncing
- You can opt-out of website sync
- Local data remains your primary copy

## Security

**Local Storage**: Files are stored as plain JSON in your project directory.

**Recommendations**:
- Don't commit `.tinytorch/` to public repositories if you include institution info
- Use `.gitignore` to exclude community data if desired
- Keep your project directory secure

## Compliance

**GDPR**: Our design aligns with GDPR principles:
- ✅ Data minimization (only optional fields)
- ✅ Purpose limitation (community map only)
- ✅ User consent (explicit opt-in)
- ✅ Right to deletion (`tito community leave`)
- ✅ Data portability (JSON files)

**FERPA**: For educational institutions:
- No student names collected
- Anonymous identifiers only
- Institution-level aggregation (not individual)

## Questions?

For privacy questions or concerns:
- Review your data: `tito community profile`
- Remove your data: `tito community leave`
- Check configuration: `.tinytorch/config.json`

