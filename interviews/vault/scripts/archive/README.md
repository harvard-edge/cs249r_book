# Archived scripts

Preserved for forensic reference. Do not run against the current vault.

## `fill_zone_gaps.py`, `expand_tracks.py`

Workarounds for the pre-v1.0 filesystem hierarchy, which had systematically
empty cells because `questions/<track>/<level>/<zone>/` could not represent
the paper's full 11-zone × 6-level taxonomy. These scripts generated or
moved questions to paper over the gaps.

Obsolete after schema v1.0: classification lives in the YAML, not in the
path, so "gap-filling" is a content question, not a directory question.
Use `vault lint` and LLM-assisted review for remaining content-quality
issues.

## `final_balance.sh`, `fill_gaps.sh`

Shell wrappers around the above. Retired alongside them.
