# Vault Audit Workflow

This directory contains release audit reports and archived experiments for the
StaffML YAML question corpus.

Active release workflow:

1. Run the canonical formatter in check mode:
   `python3 interviews/vault/scripts/format_yaml_questions.py`
2. Run the deterministic corpus audit:
   `python3 interviews/vault/scripts/audit_yaml_corpus.py`
3. Apply conservative hygiene fixes when needed:
   `python3 interviews/vault/scripts/fix_yaml_hygiene.py`
4. Build semantic review batches:
   `python3 interviews/vault/scripts/prepare_semantic_review_queue.py`
5. Fix all deterministic findings before release.
6. Use semantic review for published questions to validate question quality,
   answer correctness, napkin math, physical plausibility, and level fit.

Recommended semantic review model:

- `gpt-5.4-mini` for the full corpus pass. It is the default in
  `semantic_audit_questions.py` and balances audit quality, latency, and cost.
- `gpt-5.5` for selective second opinions on disputed or high-severity
  findings.

Run a small smoke test before launching the full semantic review:

```bash
python3 interviews/vault/scripts/semantic_audit_questions.py \
  --limit 2 \
  --workers 1 \
  --out interviews/vault/audit/semantic-review-results/smoke_semantic_findings.jsonl
```

Run all published questions in parallel by track:

```bash
python3 interviews/vault/scripts/run_semantic_audit_tracks.py --workers-per-track 3 --batch-size 10 --request-timeout 120
```

Summarize semantic results:

```bash
python3 interviews/vault/scripts/summarize_semantic_audit.py
```

Current active deterministic reports are written to:

- `fresh-yaml-audit/summary.md`
- `fresh-yaml-audit/issues.jsonl`
- `fresh-yaml-audit/stats.jsonl`

Semantic review inputs are written to:

- `semantic-review-queue/published_semantic_queue.jsonl`
- `semantic-review-queue/<track>_published_semantic_queue.jsonl`
- `semantic-review-queue/batches/<track>/*.jsonl`
- `semantic-review-queue/semantic_review_prompt.md`

Semantic review outputs are written to:

- `semantic-review-results/<track>_semantic_findings.jsonl`
- `semantic-review-results/summary.md`

Historical experiments belong under `archive/` and should not be used as the
release source of truth.
