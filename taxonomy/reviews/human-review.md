# Independent review and user evaluation protocol

Status: prepared protocol; no participants recruited and no human results collected.

## Freeze before collecting judgments

Use `archive_taxonomy.py` to freeze a candidate and record its manifest hash. A coordinator who did not construct the candidate selects fresh, licensed teaching and engineering passages across the published subject outline. Record each passage's source, exact locator, content hash, selection reason, and whether a vocabulary editor has already inspected it. Inspected material is development material, not held out. Keep participant identities private; public records use consented pseudonyms.

Record the intended users, recruitment criteria, planned sample size and rationale, tasks, timing method, exclusion rules, and acceptance criteria before the first session. These are project decisions, not thresholds mandated by ANSI/NISO, ISO, or W3C. Do not retrofit criteria to the results. Include instructors, practitioners, and learners; disclose their prior involvement with the taxonomy.

## Expert review

Give each domain reviewer the frozen entries, neighboring concepts, source passages, and incident relationships. Withhold the editor's preferred answer until the reviewer submits an initial judgment. For each entry, ask whether it represents one coherent meaning, whether its label and definition agree, whether its home supports discovery, and whether entry terms are equivalent, narrower, broader, implementations, or unrelated. For each edge, read the source, predicate, target, and condition as one assertion. Record retain, revise, remove, or insufficient evidence, with a reason and locator. Do not treat silence as acceptance.

Review the bank-conflict split, risk-criteria boundary, graph-tracing scope, withdrawn safety claims, and sparse concepts explicitly. Use `inventory.json` to distinguish entries with targeted changes from the rest of the vocabulary. A domain review must state its actual scope; it cannot certify untouched subjects.

## Indexing trial

Give each participant the same source passage and tagging rules, without other participants' tags or suggested labels. Ask for a primary subject, the concepts centrally taught, an optional missing-concept description, confidence, and a reason. Allow subject-only and out-of-scope judgments. Collect initial judgments independently, then retain both initial and adjudicated decisions. Adjudication is a separate record identifying the adjudicator and rationale.

Use a UTF-8 JSONL file with one record per item and participant:

```json
{"candidate_sha256":"...","participant":"P01","item":"item-001","subject":"...","concepts":[],"missing_concept":null,"confidence":"low|medium|high","rationale":"...","elapsed_seconds":null}
```

A separate item manifest supplies source text or a licensed retrievable source, content hash, locator, selection history, and eligibility. Do not derive reference tags from model scores, candidate ordering, or the taxonomy's own suggestions. A reference answer, if needed, comes from separate domain adjudication. Preserve disagreement and missing labels rather than replacing them with consensus.

## Navigation and retrieval trial

Use tasks grounded in participant work: locate the topic for a teaching passage; find the difference between coalescing and bank conflicts; find a concept from a narrower or implementation search term; follow an engineering relationship and explain its condition; retrieve a stable identifier and a machine-readable download. Randomize task order using a recorded seed. Do not reveal the desired label in a task that tests discovery.

Record task text, candidate hash, participant, initial query, clicked path, selected concept, completion or abandonment, elapsed time if actually measured, requests for help, and verbatim consented feedback. Mark moderator assistance. Record device and viewport for usability failures. Compare outcomes against independently specified task success criteria; do not count every click as success.

## Analysis and release decision

Before collection, choose agreement and task-success measures appropriate to the labels and sample. Publish calculation code with the raw deidentified inputs, missingness and exclusions, all planned outcomes including failures, and uncertainty appropriate to the design. Distinguish descriptive pilot findings from general claims. An independent analyst should rerun the frozen inputs and code; retain the rerun receipt.

The editor resolves substantive disagreements, changes the candidate through recorded reviews, and repeats affected evaluation on fresh material. Approval records must identify the exact candidate and unresolved limitations. A model-generated review or an empty template never substitutes for human participation. Reviewed-release approval and public deployment remain distinct actions.
