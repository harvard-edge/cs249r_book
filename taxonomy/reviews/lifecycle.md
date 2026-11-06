# Public maintenance workflow

The supported interface is `python3 binder/tools/audit/review_taxonomy.py REVIEW.json OUTPUT_DIR [--apply]`. Omit `--apply` to inspect the candidate first. Each change requires its exact before record, sources with locators and support, rationale, reviewer identity and kind, and all baseline incident relationship IDs. Application validates the candidate before replacing files and preserves originals for recovery. It does not commit or publish.

| Operation | Representation and required decision |
|---|---|
| Correct | `operation: update` (the default), with changed fields only. Labels, definitions, alias removals, home, family, scope, and relationship wording, conditions, type, or endpoints are supported. Identity cannot be rewritten. |
| Add | `operation: add`, absent `id`, `before: null`, and the complete new record in `fields`. New aliases require per-term decisions. Evidence must establish a distinct meaning and a useful classification boundary. |
| Move | Update `home`, with an impact record. Check subject-tagged consumers; an unchanged identifier does not prove that existing subject assignments are correct. |
| Split | Keep the historical identifier on its principal retained meaning, add the separated concept, and explicitly update affected edges. Record the changed scope and all affected assignments; never automatically duplicate historical tags onto both successors. The coalescing review is the worked example. |
| Merge or retire | `operation: retire` on a concept, with `fields: {"replaced_by": "mlsys.active-id"}`. Explicitly rewire or remove every incident edge in the same review. The retired record and former term provenance remain in history. A retirement cannot point to itself or another retired concept. |
| Remove an edge | `operation: remove`, exact before record, empty `fields`, and a source-backed reason. Withdrawal of an unsupported claim does not assert that the opposite claim is true. |
| Draft version | An optional top-level `release` object records `from`, `to`, and `rationale`; the version must advance and retain `-draft`. First register the version in `FrameworkVersionEnum` and the schema's top-level version, then validate the complete candidate. |
| Archive | After build/check, run `python3 binder/tools/audit/archive_taxonomy.py OUTPUT_DIR`. The new directory pins canonical source, provenance, methodology, schema, and generated downloads by hash. Existing snapshots are never intentionally overwritten. |

Lifecycle changes additionally require `impact` with nonempty `classification`, `assignments`, and `relationships` explanations. Relationships still use controlled relation/kind pairs and valid endpoints. The full schema and graph validator remains the authority for candidate structure. Sparse edges are review diagnostics, not a reason to invent relationships.

Run the taxonomy build/check, coverage/agreement recomputation, regression tests, StaffML strict validation, and rendering checks after a substantive change. Review the generated assignment inventory for moves and scope changes. The tool does not silently rewrite books, questions, or historical evaluation judgments. Historical agreement data remains historical; create new evaluation records after meaning changes.

The release identity is the version plus the canonical source SHA-256. Repeated draft revisions therefore remain distinguishable even before a version increment. Archive availability, hosting redirects, DOI registration, and deployment require separate verification on the actual publication service.

This interface deliberately cannot declare a normative or independently reviewed release. Human review and evaluation must actually occur, and the existing reviewed-release governance gate remains in force. The public draft process has no mandatory model-family quorum. Historical campaigns remain replayable under their original rules; their study-specific conditions are not standards requirements.

For an added or reclassified entry term, include `terms: {"term": {"kind": "narrower", "rationale": "..."}}` on the concept change, with source evidence. Allowed kinds are synonym, acronym, narrower, broader, implementation, and related. Every added alias requires a decision; unchanged terms can be classified with empty `fields`. Only explicit synonym/acronym decisions emit SKOS alternative labels. Other terms remain retrieval entries.
