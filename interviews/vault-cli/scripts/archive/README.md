# Archived scripts

These scripts are preserved for forensic reference but are no longer part
of the active vault workflow. Do not run them against the current vault.

## `split_corpus.py`

The pre-v1.0 script that split `corpus.json` into per-question YAML files
under the old `questions/<track>/<level>/<zone>/<id>.yaml` hierarchy.

**This script was the source of the v0.1 migration defects** that the
schema v1.0 migration (2026-04-21) corrected:

- The filesystem hierarchy could not represent the paper's full 11-zone
  × 6-level taxonomy, so 943 L6+ and 1,594 questions in "missing" zones
  were silently collapsed into `l1/` and `recall/` respectively.
- 86 published questions were dropped entirely when their target
  directory did not exist.
- The singular `chain:` YAML field truncated multi-chain membership for
  101 questions.

Replaced by `interviews/vault/scripts/migrate_to_v1_0.py`.
