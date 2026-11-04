# How the ML Systems Taxonomy is built

Status: **editorial draft**. This document specifies a repeatable construction and maintenance method. It does not claim complete field coverage, full standards conformance, human validation, or community endorsement.

The procedure is repeatable because the scope, sources, starting records, decisions, and checks are recorded. Another editor can inspect and replay a change, challenge its evidence, and apply the same questions to another subject. Different editors may still make different judgments. Agreement and usefulness must be evaluated separately.

## How the vocabulary is organized

The browsing outline groups subjects into areas and broad groups. Each concept has a permanent identifier and one home subject. Its preferred label, aliases, and definition describe its meaning; typed relationships connect it to other concepts. Browsing membership is separate from semantic hierarchy: being filed under a subject does not make that subject a broader concept.

The public name is “taxonomy.” Its representation combines a browsing classification with a controlled concept vocabulary and associative relationships. SKOS collections represent browsing groups, while reviewed specialization links supply broader/narrower relations. This is not a claim that the entire graph is a strict hierarchy or a formal domain ontology.

## Construction and maintenance protocol

The ML systems taxonomy is constructed through a documented evidence-and-review process and maintained using established controlled-vocabulary practices. ANSI/NISO Z39.19 informs construction, testing, and maintenance; ISO 25964 informs concept and term management and interoperability; W3C SKOS supplies the exchange representation. This statement describes the method adopted. The standards assessment below records the limits of verification; no certification or complete conformance is implied.

The existing vocabulary is the starting inventory. Its groups, subjects, concepts, identifiers, source records, and exports are retained unless a documented review supports a change. Existing membership is not sufficient evidence for acceptance, and an incomplete evidence record is not by itself grounds for deleting a useful concept. Such records remain explicitly pending review.

### Standards baseline and project choices

| Authority | Application in this project | Boundary of the claim |
|---|---|---|
| [ANSI/NISO Z39.19-2005 (R2010)](https://www.niso.org/publications/ansiniso-z3919-2005-r2010) | Term selection, scope, equivalence, relationships, testing, and maintenance | Selected warrant, term-equivalence, and construction passages inspected; other clause attributions remain unverified |
| [ISO 25964-1:2011](https://www.iso.org/standard/53657.html) | Concept-centered vocabulary management, preferred and alternative terms, scope, and change records | Public ISO and NISO descriptions support this adoption; a full-text conformance audit remains open |
| [ISO 25964-2:2013](https://www.iso.org/standard/53658.html) | Establishing and maintaining mappings to neighboring vocabularies | Mapping work is planned; no external equivalence is inferred from similar labels |
| [W3C SKOS Reference](https://www.w3.org/TR/skos-reference/) | Concepts, collections, labels, semantic relationships, and mapping properties | Structural validation does not establish domain correctness or completeness |

These editions are the declared baseline. ISO's catalog, checked on November 4, 2026, lists ISO 25964-1:2011 as current with a replacement in development. A later edition requires a recorded impact review before adoption.

The source-count thresholds, source-lineage rules, model-review configuration, one-home browsing policy, minimum relationship count, and evaluation thresholds are project policies. They are not ANSI, ISO, or W3C mandates. Model agreement is review evidence, not proof that a technical claim is true. An editor remains responsible for acceptance and for resolving disagreements against sources. Changes to these policies must be recorded separately from changes to the vocabulary; the simplified correction path is an explicit policy change. Historical records keep their original validation rules; release transitions still have separate gates.

### Repeatable construction procedure

| Stage | Editorial action | Required record and completion condition |
|---|---|---|
| Scope | State the intended users, tasks, boundaries, and level of specificity; examine overlapping vocabularies before adding new structure | A scope statement, neighboring-vocabulary inventory, and reasons for reuse, mapping, or a distinct concept |
| Evidence selection | Select sources using recorded inclusion and exclusion rules across the existing source strata; distinguish construction material from evaluation material | A source inventory with versions, locators, provenance, and exposure status; related sources are not counted as independent without justification |
| Candidate extraction | Extract an engineering idea and its supporting passage before choosing its preferred label or home; record whether it is new or already present | A candidate record with evidence, proposed meaning, and nearest existing concepts; summaries and model suggestions are labeled as such |
| Term reconciliation | Decide whether candidate terms denote the same concept, different concepts, a broader or narrower concept, or an implementation | A preferred label, justified alternative labels, a definition, boundary notes where needed, and an accept, revise, defer, or reject decision with reasons |
| Organization | Place the concept in the browsing outline and assess each semantic relationship independently | A justified home and reviewed relationships; browsing membership is not exported as a broader relation; a minimum edge count never justifies inventing an edge |
| Review | Check scope, sources, duplicates, definitions, aliases, and literal relationship meaning; revisit incident relationships whenever meaning changes | Reviewer judgments, disagreements, their resolution, and the responsible editor; uncertain candidates remain pending |
| Evaluation | Freeze the candidate vocabulary before evaluation; apply it to material that did not shape the changes and observe actual tagging and retrieval tasks | Inputs, judgments, failures, and reproducible calculations; model evaluation and human evaluation are identified separately, and inspected evaluation material is no longer described as unseen |
| Release | Validate identities, schemas, exports, links, and downstream consumers; inspect the rendered site and review the conformance statement | A versioned candidate, changelog, validation records, known limitations, and editorial release decision; public draft and reviewed release remain distinct statuses |

Repeatability means that another editor can inspect the same inputs, apply the same decision rules, and explain differences in judgment. It does not promise identical structures or prove completeness of the field. Retrospective reconstruction of an existing decision must be labeled retrospective; it must not be presented as a protocol followed before the original decision.

### Decision record

For each substantive change, the review record must identify the affected concept or subject, the proposed operation, the source passages and locators, the nearest competing interpretation, the reason for the decision, the reviewer and editor, the date, and the release or content revision. Definition changes also record their effect on existing relationships and tags. Deferrals record the missing evidence or unresolved distinction. These are review requirements mapped onto the existing campaign and provenance records, not new fields to insert into the canonical YAML without a schema change.

For example, a proposal to make a named software system an alias of a general mechanism must establish equivalence within the declared scope. Evidence that the system merely uses the mechanism is insufficient. The appropriate outcome may be an illustrative reference, a related concept, or a deferred proposal. This is an illustrative decision rule, not a report of a completed review.

### Maintenance cycle

Contributions enter through the feedback process and receive a recorded disposition. Accepted changes follow the construction stages relevant to their impact: a spelling correction needs lexical and collision checks, while a changed definition, split, merge, or relationship requires semantic and downstream review. Review depth may vary; provenance and editorial accountability remain required.

Identifiers remain stable and are never reused. Retirements preserve replacement information and history; splits identify their successors and flag existing assignments for review. Each published change records its reason and version. Release artifacts should remain retrievable by version, with license and citation information. Archival publication and live identifier resolution remain release work until verified.

Periodic review examines unresolved proposals, ambiguous or rarely useful terms, new source material, broken links, mapping drift, and feedback from instructors and practitioners. Mappings follow ISO 25964-2 guidance and use SKOS mapping properties only after meaning and scope have been compared. Similar wording alone does not establish an exact match.

### Applying the protocol to the existing inventory

Start with the current canonical vocabulary and its existing evidence records. Review a bounded subject through the stages above, preserving its identifiers and documenting which evidence is historical, newly checked, or absent. Use the resulting decision records to expose ambiguities in the procedure before extending the review across other subjects. Prioritize already recorded alias exceptions, disputed hierarchy edges, and unsupported concepts. A review may retain an existing entry unchanged; edits must resolve an identified problem.

The existing empirical study and refresh machinery can support this work, but their claims require their own receipts. This protocol is a prospective operating specification; adding it does not establish that past runs satisfied it or establish that the historical evidence meets prospective release requirements.

## Executable checks and refresh requirements

The construction protocol above is the editorial recipe. The following commands are available from the repository root:

```bash
./binder/binder concepts check
./binder/binder concepts build
./binder/binder concepts coverage --check
./binder/binder concepts agreement --check
python3 taxonomy/reviews/structured-generation/verify.py
```

The compiler checks the schema, identifiers, graph invariants, provenance consistency, and recorded coverage and agreement inputs. The build generates the public formats and downstream lookup data. Coverage and agreement commands recompute recorded judgments; they do not obtain new judgments or establish that a sample was independent of construction. The worked-example verifier checks its pinned baseline and review inventory, not the technical truth of its recommendations.

Vocabulary changes must preserve the provenance chain. For corrections to existing draft records, use the public `binder/tools/audit/review_taxonomy.py` command described in [Maintaining the taxonomy](README.md#maintaining-the-taxonomy). Each accepted review records exact before/after content, sources and their support, rationale, reviewer identity and kind, and incident relationships. The command previews and validates before application, retains original files, and restores originals after ordinary write or validation failures. It is not a crash-atomic multi-file transaction.

This correction path replaces mandatory model-family agreement with accountable source-backed editorial decisions. Model assistance is optional; model agreement does not establish correctness. Historical campaigns keep their historical validation rules and evidence limitations. The same interface supports additions, moves, explicit splits and retirements, typed entry-term decisions, and draft version transitions; see the [maintenance operations](reviews/lifecycle.md). A separate command freezes versioned draft archives. Earlier private refresh scripts are not a required public maintenance interface.

Reviewer counts and admission thresholds are project policies, not requirements imposed by ANSI/NISO, ISO, or W3C. Human domain review and indexing trials remain necessary evidence for a reviewed release.

## Historical study and evidence recovery

The records in `taxonomy/method-study/` preserve the earlier method study and its reported results. Public end-to-end replication of that study remains unavailable: raw judgments and portions of the workflow are maintainer-local, and the published summaries do not establish the claimed isolation, preregistration, or execution history. The historical metrics are not used here as verified evidence of publication readiness or procedure repeatability.

The current schema-4 exposure ledger is a conservative recovery of the archived schema-3 ledger. `binder/tools/audit/migrate_taxonomy_ledger.py` reconstructs it deterministically from `legacy-burned-clusters-v3.json.gz` and `evaluation-reconstruction.json`. The recovery preserves recorded history and identifies missing evidence. Historical coauthor classifications remain unknown and are excluded from independent-source eligibility. The ledger contains no reconstructed evaluation-unseal events: the recovery does not manufacture timestamps, registrations, or receipts.

`binder/tools/audit/reconstruct_taxonomy_evaluation.py` can recover evaluated item identities and source-file hashes from the retained raw outputs when those files are available. It does not recompute study scores or prove that the evaluated items were sealed beforehand. The recovery inventory and migration tests support these limited claims only.

The structured-generation review in `taxonomy/reviews/structured-generation/` is a public worked example of reviewing an existing subject. It records a baseline, source-linked findings, proposed dispositions, and unresolved questions. Its accepted boundary corrections were applied through the public correction command; the review also retains deferred proposals. Further reviews must record their scope and distinguish targeted editorial screening from exhaustive verification, independent human review, and cross-family review.

## Standards assessment

The declared baseline is ANSI/NISO Z39.19-2005 (R2010), ISO 25964-1:2011, ISO 25964-2:2013, and the W3C SKOS Recommendation. Their roles differ: vocabulary construction and maintenance, thesaurus management, mappings between vocabularies, and web representation. Edition changes require an impact review rather than automatic adoption.

| Aspect | Implemented practice | Verification limit |
|---|---|---|
| Warrant and term decisions | Source-linked definitions, alias decisions, and correction rationales | Targeted source review; the whole inventory has not received assertion-level validation |
| Identity and change history | Stable identifiers, recorded retirements, exact correction replay, and content hashes | A valid chain establishes recorded consistency, not the truth of an editorial judgment |
| Concept and label separation | Concepts have identifiers; labels and entry terms are lexical fields | Legacy retrieval terms do not assert equivalence; explicit term decisions distinguish synonyms, narrower meanings, and implementations |
| Hierarchy and association | Browsing collections are separate from specialization and other relations | Graph checks cannot prove that each specialization is semantically sound |
| SKOS interchange | Labels, collections, hierarchy, and associations are exported; conditional engineering relations retain wording and conditions in statement records | Successful parsing and integrity tests establish only the properties they check; this is not a complete conformance certification |
| External mappings | ISO 25964-2 informs the proposed crosswalk procedure | No external equivalence is asserted merely because two labels look alike; crosswalk work remains outstanding |
| User testing | Coverage and agreement commands recompute recorded judgments | Historical model judgments are not human indexing or usability evidence |

[The standards source check](reviews/standards-check.md) records what was actually inspected. Selected passages of the official NISO text were read. Public ISO catalog descriptions were checked; the complete normative ISO texts were not inspected. Earlier clause attributions in historical provenance remain historical claims until verified. SKOS supplies a representation, not a guarantee of subject correctness.

For a future external mapping, record both vocabulary versions, source and target identifiers, mapping type, evidence, reviewer, and disposition. Check definitions and scope before choosing exact, close, broader, narrower, or related mapping. Review mappings again when either vocabulary changes. This is the project's proposed implementation of interoperability guidance, not a completed ISO audit.

The [editorial dispositions](reviews/dispositions.md) record compound-entry decisions, typed search terms, withdrawn unsupported relationships, and remaining evidence limits. The [inventory](reviews/inventory.json) distinguishes targeted review from mechanical checks across the vocabulary. The [human-review protocol](reviews/human-review.md) is prepared; it does not report participant results.

## Publication and maintenance decisions

An editorial draft may invite public review while clearly stating its limitations. Before proposing publication, an editor must inspect the final diff and rendered site, check downloadable formats and downstream identities, record unresolved decisions, and confirm that the public description matches the evidence. Publication is a separate approval; generating files does not authorize deployment.

A reviewed release requires additional evidence: domain-expert review of contested meanings and relationships, human indexing and navigation trials with recorded methods, a checked standards claim, and a versioned archival record. These activities are not reported as completed. Earlier board sizes, model quorums, sampling thresholds, and source-independence formulas were project study designs; none is imposed by ANSI/NISO, ISO, or W3C.

The public maintenance path records exact changes, source evidence, impact decisions, and provenance for additions, corrections, moves, splits, merges, retirements, and entry-term decisions. Draft version transitions and immutable local archives are supported. Declaring a reviewed release still requires actual human evaluation and the reviewed-release governance gate; an editorial command cannot manufacture that evidence.

Feedback enters through the [taxonomy issue form](https://github.com/harvard-edge/cs249r_book/issues/new?template=taxonomy_feedback.yml). Each accepted change records its reason and preserves identities. When evidence is insufficient, defer the decision explicitly rather than adding unsupported detail. Review depth should follow the impact of the change; a wording correction and a concept split do not need the same investigation.

## Competency design probes

A subject label identifies knowledge; a competency requires observable engineering work. Four draft probes illustrate the difference:

| Task | Connected knowledge | Evidence to assess |
|---|---|---|
| Decide whether quantization improves a constrained inference workload | Profiling, workload, bandwidth, arithmetic intensity, quantization, metrics, SLOs | Reproducible baseline, measured bottleneck, quality comparison, decision with stated limits |
| Recover distributed training under an explicit failure model | Parallelism, training state, checkpoints, consistent cuts, artifact identity, collectives | Checkpoint manifest, injected-failure/restart logs, restored-state checks, guarantee boundaries |
| Configure a retrieval-assisted service under quality and capacity constraints | Retrieval, ANN, embeddings, benchmark design, hallucination, admission, tail latency, goodput, cost | Retrieval and answer evaluation, workload traces, failure accounting, transfer-case reassessment |
| Evaluate timing and fallback behavior of a learned physical component | Operating domain, observation age, feedback delay, execution time, WCET, runtime assurance | Contained experiment, timing evidence, fallback behavior, limits of the tested operating domain |

These are proposed assessment designs, not claims that existing chapters assess those tasks. A later AI engineering profile could select relevant knowledge and tasks with a declared role, autonomy, constraints, and evidence rubric. Reading a chapter, visiting a map, or recognizing concepts does not demonstrate competence. The [CS2023 competency introduction](https://csed.acm.org/wp-content/uploads/2024/04/1.4-Introduction-to-Competency-Framework.pdf) provides a useful task-and-context distinction; [SFIA responsibility levels](https://sfia-online.org/en/sfia-9/responsibilities) describe professional responsibility and must not be converted into a student mastery score.
