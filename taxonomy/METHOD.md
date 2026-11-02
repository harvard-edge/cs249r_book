# How the ML Systems Taxonomy is built

This document states the published standards the taxonomy adopts, how each recommendation is met, and where it is not yet met. Status terms are used consistently: **met** means the record or a check shows it, **partial** means some of it is done and the rest is named, and **open** means it has not been done. Numbers describe version `0.4-draft` and are computed from the release files named beside them.

## The standards

| Layer | Standard adopted | Role |
|---|---|---|
| Construction | [ANSI/NISO Z39.19-2005 (R2010)](https://groups.niso.org/higherlogic/ws/public/download/6484/Guidelines%20for%20the%20Construction,%20Format,%20and%20Management%20of%20Monolingual%20Controlled%20Vocabularies.pdf), *Guidelines for the Construction, Format, and Management of Monolingual Controlled Vocabularies* | How terms are chosen, formed, related, tested, and maintained |
| Structural model | [ISO 25964-1:2011](https://niso.org/schemas/iso25964) (*Thesauri for information retrieval*), via its public NISO/W3C data-model alignment | Explicit separation of concept identity (`skos:Concept` URI) from lexical labels (`prefLabel`, `altLabel`) |
| Representation | [W3C SKOS Reference](https://www.w3.org/TR/skos-reference/) (Recommendation, 2009), with a declared OWL extension for typed relations | The published data model |
| Metadata and publication | [DCTERMS](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/), [W3C DCAT 3](https://www.w3.org/TR/vocab-dcat-3/), [W3C Best Practice Recipes for Publishing RDF Vocabularies](https://www.w3.org/TR/swbp-vocab-pub/), and [Ten simple rules for making a vocabulary FAIR](https://pmc.ncbi.nlm.nih.gov/articles/PMC8238180) | Identifiers, licensing, versions, and reuse |
| Validation | Inter-indexer consistency as used in library science: Hooper's measure, [Rolling's measure](https://informationr.net/ir/18-4/paper601.html), and Krippendorff's $\alpha_{\text{MASI}}$, reported beside Cohen's $\kappa$ and anti-gaming specificity bounds ($p_{\max} \le 15\%$) | Whether people and tools apply the vocabulary the same way |

**ISO 25964 scope.** ISO 25964-1 and ISO 25964-2 are the international thesaurus and interoperability standards, and [NISO describes their scope](https://niso.org/schemas/iso25964). Because their full text is sold rather than openly published, clause-level conformance is itemized against ANSI/NISO Z39.19-2005 (R2010) and W3C SKOS, while adopting ISO 25964-1's concept-centric structural distinction (`ThesaurusConcept` vs. `PreferredTerm`/`SimpleNonPreferredTerm`) in `taxonomy/schema/mlsys-taxonomy.linkml.yml`.

**What Z39.19 says this is.** Under section 5.4 a *taxonomy* connects all preferred terms in a hierarchy, and a *thesaurus* displays several relationship types with standardized indicators. This vocabulary keeps the familiar name but is structurally a thesaurus: most links are associative, and subjects are browsing groups (`skos:Collection`) rather than broader terms.

**Ontology methodologies.** This project adopts controlled-vocabulary guidance and does not claim to implement an ontology-development methodology such as NeOn or METHONTOLOGY. Relations are declared in OWL only so that the SKOS extension is machine-readable.

## How a version is built

Z39.19 section 11.1.3 recommends a committee approach, an empirical approach, or a combination. This project uses empirical construction under a top-down outline, anchored on the published vocabulary and verified by three independent model families (`claude-opus-5-5`, `gpt-6-astra` / `gpt-6.1-sol`, `gemini-4-argon-high`). Full operational rules and mathematical definitions are recorded in `taxonomy/method-study/synthesized-method.md` and executed via `.claude/workflows/taxonomy-refresh.md`:

1. **Scope, fixed structure, and cluster-disjoint sampling frame** (Z39.19 §5.2, §11.1.2). The five groups, 20 areas, and 101 subjects are fixed at the start of a refresh and change only by recorded decision. Completeness and candidate extraction draw from an enumerated five-stratum sampling frame (`taxonomy/method-study/sampling-frame.md`): graduate syllabi (`S1`), systems and ML conference programs (`S2`), engineering and safety/governance standards (`S3`), practitioner role profiles and untagged StaffML interview questions (`S4`), and MLSysBook chapter contracts (`S5`). Every source unit is tagged with a `provenance_family_id` and partitioned by cluster before construction into `60%` `dev` (visible during construction), `25%` `held_out_test` (sealed until post-freeze evaluation), and `15%` `reserve`.
2. **Cross-area lookup against the existing vocabulary** (Z39.19 §11.1.1, §11.1.5). Before proposing any concept for a subject row, builders search all 20 areas and 101 subjects (`taxonomy/downloads/mlsys-taxonomy-concepts.csv`) by preferred label, aliases, and definition tokens. When a concept homed in another area already owns the mechanism (such as `mlsys.memory-planning` in `S2` or `mlsys.selective-prediction` in `P3`), the workflow adds an `alias` or a `secondary_concepts` link on the subject row rather than minting a duplicate concept.
3. **Candidate extraction and provenance-family warrant** (Z39.19 §5.3.5.1–5.3.5.3, §11.1.3.2). Candidates are extracted inductively from `dev` units across three independent model families. Admission requires literary warrant from at least two sources belonging to distinct `provenance_family_id` clusters, at least one of which comes from external strata `S1`–`S4`. **Heading-only warrant ban:** at least one warranting source must contain explanatory prose defining the mechanism; a bare section heading or checklist topic title (`Topic: ...`) may serve as the corroborating second `provenance_family_id` when it names the same concept, but bare headings alone (with zero prose sources) cannot mint a concept or justify extrapolated mechanisms. Single-lineage mechanisms are recorded as declared `gaps`.
4. **Product, vendor, benchmark, and SKU exclusion** (Z39.19 §6.3.3). Named software frameworks, serving engines, hardware SKUs, and benchmark suites (`vLLM`, `TensorRT-LLM`, `SGLang`, `ZeRO`, `CUDA Graphs`, `MLPerf`) are never admitted as concepts; they are recorded as `aliases` (`alt_label`) on the engineering mechanism they implement.
5. **Term form, compound factoring, and `metric × workload` gate** (Z39.19 §6.1–6.7, §7.1–7.6). Each concept receives one singular noun-phrase preferred label (`label`), entry terms (`aliases`), and a one-sentence plain genus-differentia definition (`definition`). Compound candidates (`A and B`, `metric × workload`) are split unless institutionalized across $\ge 2$ provenance families. **Compound metric gate:** generic `metric × workload` combinations (`inference throughput`, `inference energy cost`) are rejected unless they define a stage-specific structural measure (`TTFT`, `TPOT`, `goodput`).
6. **4-part Concept-vs-Alias test and coupled-phase anti-splitting** (Z39.19 §6.1, §8.2). A warranted candidate becomes a distinct `Concept` rather than an `alias` of an existing concept $P$ only if all four hold: (a) it alters a system-level invariant, resource envelope, or failure boundary; (b) it has an independent failure mode or trade-off not identical to $P$; (c) it either fails the `all-and-some` test against $P$ or requires a causal/associative edge not inherited from $P$; and (d) it has a recorded **Differential Indexing Witness** (`witness_justification`) naming two indexing scenarios where one requires $C$ and not $P$ (or requires distinguishing $C$ from a sibling $C'$ under $P$). **Coupled-phase & sub-step rule vs. distinct methodologies:** tightly coupled execution phases (`prefill phase` and `decode phase` under `mlsys.prefill-decode`), acronyms (`ODD`, `FMEA`, `STPA`, `FTA`, `SOTIF`, `ALARP`), and non-independently-indexed sub-steps or artifacts of an admitted algorithm (`draft-verification` under `mlsys.speculative-decoding`, `copy-on-write KV sharing` under `mlsys.paged-kv`, `unsafe control action` / `UCA` under `mlsys.stpa`) are recorded as `aliases` on the parent concept. By contrast, distinct analytical methodologies or mechanisms with different causal models and inputs (such as component-level `Failure modes and effects analysis`, boolean `Fault tree analysis`, and control-loop `System-theoretic process analysis` under `Hazard analysis`) are admitted as separate concepts (`specializes` children of `mlsys.hazard-analysis`).
7. **Home subject and typed relationships** (Z39.19 §8.1, §8.3, §8.4; SKOS S27). Each concept receives exactly one `home` subject under the deterministic 4-step Home-Rule decision tree. Every concept participates in at least two relationships from the 17 controlled verbs. Generic `specializes` (`skos:broader`) requires the 3-step `all-and-some` test (`Z39.19 §8.3.1`); `part-of` stays associative under `skos:related` (`Z39.19 §8.3.3.2`) because system components participate in multiple assemblies; `SKOS S27` forbids mixing `specializes` and associative edges between the same nodes; and every relationship must read true in both forward and inverse phrasing under its explicit `conditions`.
8. **Cross-check, deterministic reconciliation, and independent audit** (Z39.19 §11.1.5, §11.2.2). All three model families cross-check the merged patch item by item (`accept`, `fix`, `reject`). `.claude/workflows/taxonomy-refresh/reconcile.py` exits non-zero if any `fix` or `reject` verdict is unreflected in `final.yaml` without a written override in `overrides.yaml`. After compilation (`./binder/binder concepts build` and `check`), independent auditors verify necessity, publishability, quote grounding, and anti-splitting.
9. **Blind held-out coverage and inter-indexer agreement** (Z39.19 §11.2.2). On the sealed `held_out_test` split, two independent model evaluators score coverage using a 3-level ordinal expressibility scale (`1.0` single-concept exact match, `0.5` post-coordination of 2–3 concepts, `0.0` uncovered) and index held-out items to compute subject $\kappa$, Hooper's consistency, Rolling's consistency, and Krippendorff's $\alpha_{\text{MASI}}$, enforcing an anti-gaming specificity cap ($p_{\max} \le 15\%$ max single-concept assignment share).
10. **Record and release** (Z39.19 §11.3). Changelog, permanent identity checks (`retired` with `replaced_by`), generated downloads (`CSV`, `JSON`, `LinkML`, `JSON Schema`, `SKOS Turtle`, `OWL`, `DCAT`, `GraphML`, `BibTeX`, `CITATION.cff`), and RDF/SKOS integrity validation (`pytest binder/tests/test_taxonomy*.py`).

**Human and machine roles.** Z39.19 section 11.1.3.4 assumes construction calls for intellectual decisions and describes machines as assistants for tasks such as identifying candidate terms. This project goes further: language models proposed terms, cross-checked them, and audited the result, and one human Editor holds final authority. That is a substantive departure from the standard's assumption, and it is the main reason the status stays *editorial draft* until the external editorial board (`taxonomy/method-study/candidate-experts.md`) completes its review.

## Empirical method study (November 2026)

In November 2026, the project ran a six-phase blind empirical study (`taxonomy/method-study/`, receipts in `.claude/_reviews/taxonomy-method/20261102/`) to test this construction method against a blank-slate rebuild on two areas: `P4` (`serving` — Deployment & serving, 9 subjects) and `Q4` (`safety-assurance` — Safety & assurance, 4 subjects):

1. **Blind method competition and citation audit (Phases 1–2).** Three model families (`gpt-6-astra` as `Method-A`, `claude-opus-5-5` as `Method-B`, `gemini-4-argon-high` as `Method-C`) designed end-to-end taxonomy methods in isolated workspaces with zero repository access (`0` leaks across 14 leak patterns; `phase1-manifest.json`). Three blind judges scored the anonymized methods on an 8-dimension rubric (`taxonomy/method-study/rubric.yaml`), and a primary-text citation auditor verified every standard clause claim against `ANSI/NISO Z39.19-2005 (R2010)` (`phase2-results.json`):
   - Cross-family (non-self) mean scores out of `40`: `Method-B` = `37.25` (`4.656/5`), `Method-A` = `34.00` (`4.250/5`), `Method-C` = `25.75` (`3.219/5`).
   - Primary-standard citation audit: `Method-B` `108` upheld / `4` downgraded / `0` vacated; `Method-A` `31` upheld / `5` downgraded / `0` vacated; `Method-C` `45` upheld / `33` downgraded / `1` vacated (`skos:broaderGeneric`/`skos:broaderPartitive` attributed to core W3C SKOS Reference rather than the ISO 25964 extension).
2. **Pre-registered two-area rebuild and anti-reward-hacking audit (Phases 3–5).** A corpus of `382` source units across `S1`–`S5` for `P4` and `Q4` was partitioned by `provenance_family_id` (`seed = 20261103`) into `200` `dev`, `126` `held_out_test`, and `56` `reserve` (`taxonomy/method-study/preregistration.yaml`). Three isolated builders extracted `P4` and `Q4` candidates from `dev` only (`98`, `78`, and `77` raw concepts) and were deterministically reconciled (`reconcile_phase4.py`) into `111` admitted concepts (`68` `P4`, `43` `Q4`; `63` multi-builder, `48` single-builder multi-stratum), `178` relationships, and `27` rejected candidates (`phase4-reconciled-p4-q4.json`). The Phase 5 audit (`phase5-audit-report.json`) verified deterministic rerun hashing (`9468a3dfff0a3c19b433e9c38fa034c994d5fbb64da28bda2f9dd11b8be646ed`), `0` product/SKU leaks, `0` remaining `metric × workload` split quantities (`inference-throughput` and `inference-energy-cost` rejected by the §7 gate), `137/137` verified quote tokens, and `26/35` (`claude`) and `27/35` (`codex`) concepts upheld on a 35-concept stratified semantic audit (`proposal-admission-gate` and `assurance-evidence-record` vacated for extrapolating from bare topic headings).
3. **Blind held-out comparison (Phase 6).** Two blind evaluator families (`claude-opus-5-5`, `gpt-6-astra`) scored `Vocab-Y` (`published_0_4_draft`, `125` concepts in `P4`+`Q4`: `95` `P4`, `30` `Q4`) and `Vocab-X` (`rebuilt_synthesized_method`, `111` concepts: `68` `P4`, `43` `Q4`) on `79` held-out coverage units and `43` held-out indexing scenarios (`phase6-comparison-report.json`):

   | Metric (`P4` + `Q4` held-out slice) | `Vocab-Y` (`0.4-draft`) | `Vocab-X` (single-pass rebuild) |
   |---|---|---|
   | Any-judge central-mechanism coverage (`79` units) | `72.2%` | `36.7%` |
   | Two-judge consensus coverage (`79` units) | `36.7%` | `10.1%` |
   | Mean ordinal expressibility score | `42.4%` | `17.4%` |
   | Held-out subject $\kappa$ (`43` items) | `0.971` | `0.917` |
   | Hooper inter-indexer consistency | `81.8%` | `72.1%` |
   | Rolling inter-indexer consistency | `87.6%` | `81.2%` |
   | Krippendorff's $\alpha_{\text{MASI}}$ | `0.724` | `0.539` |
   | Max single-concept share $p_{\max}$ (cap $\le 15.0\%$) | `9.84%` | `6.94%` |

   Because `0.4-draft` incorporates multi-pass systems-venue mechanisms (`mlsys.attention-ffn-disaggregation`, `mlsys.expert-offloading`, `mlsys.masked-diffusion-decoding`, `mlsys.intra-device-overlap`, `mlsys.compound-workflow-pipelining`, `mlsys.fidelity-adaptive-serving`) that a single-pass rebuild from `dev` excerpts misses, the synthesized method operates as an **incremental refresh and validation pipeline anchored on `0.4-draft`**, never as a blank-slate replacement. Furthermore, `28` practitioner aliases across `15` concepts and `1` new cross-area `secondary_concepts` link (`mlsys.memory-planning` on `P4.9`) surfaced by the blind rebuild were folded back into `0.4-draft`.
4. **Blind repeatability verification on `Q4.1` (Phase 7).** Two fresh isolated builders (`claude-opus-5-5` and `gpt-6-astra`) were given only the written method documents (`README.md`, `METHOD.md`, `taxonomy-refresh.md`), the `Q4.1` (`hazards-operating-domain`) subject scope, and the 23 `Q4` `dev` source units (`phase7-repeatability-test.json`). Round 1 (`phase7-repeatability-r1.json`: `60.0%` Hooper, `75.0%` Rolling between builders; `5 of 7` and `3 of 7` canonical concepts matched) exposed two ambiguities in the written instructions: whether a syllabus/checklist topic heading counts as a corroborating second provenance family when paired with a prose source (Gate 2), and how distinct analytical methodologies (`FMEA`, `FTA`, `STPA`) are separated from internal sub-steps (`UCA`) and recorded in `declared_gaps` when the `dev` slice contains a single family (Gate 5). After clarifying both clauses in this document and `taxonomy-refresh.md`, both blind builders in the verification run (`phase7-repeatability-test.json`) independently recovered `7 of 7` (`100.0%` recall, `100.0%` precision, `0` extraneous concepts) of the published `Q4.1` concepts (`3` active multi-family `dev` concepts + `4` single-family `declared_gaps`, with `7 of 7` inter-builder canonical concept agreement).

## Conformance to Z39.19

Appendix A of the standard is informative and summarizes its recommendations. The table follows the clauses that apply to this vocabulary.

| Clause | Requirement | Status | How it is met, or what is missing |
|---|---|---|---|
| 5.3.1 | Each term has one meaning | partial | The compiler rejects a label or alias shared by two concepts. That proves unique strings, not unambiguous meaning, and human ambiguity review is open |
| 5.3.2 | Each concept has one preferred term; synonyms point to it | met | `label` and `aliases`, exported as `skos:prefLabel` and `skos:altLabel` |
| 5.3.3 | Equivalence, hierarchical, and associative relationships | met | Aliases, `specializes`, and the other 16 relations |
| 5.3.5.1 | Literary warrant | partial | 699 of 878 concepts occur in book chapters and 471 are cited by covered paper, course, or standards topics. An enumerated 5-stratum sampling frame (`S1`–`S5`) with `provenance_family_id` cluster splits is defined in `taxonomy/method-study/sampling-frame.md` and applied in pre-registered slice studies (`P4`, `Q4`), with full-taxonomy backfill ongoing |
| 5.3.4 | Facets | met by design | The standard leaves facet choice to the designer; groups and row facets are stated in the framework |
| 5.3.5.2 | Organizational warrant | partial | 425 concepts are tied to covered book topics or section links. That is coverage; the clause asks for the term forms the organization prefers, and those decisions are not documented per term |
| 5.3.5.3 | User warrant | partial | The standard means user requests and searches, which are not recorded. `user_warrant_proxy` counts StaffML question tags (626 concepts), a term-assignment frequency and not user behavior |
| 6.1 | Term choice by domain, warrant, and specificity | partial | The admission test and 4-part Concept-vs-Alias Differential Indexing Witness test (`§8.2`) are written and applied; 27 concepts have no recorded warrant of any kind and are queued for editorial board review |
| 6.2.2 | Scope notes | partial | Subjects carry includes and excludes, exported as `skos:scopeNote`. Concepts carry definitions, not separate scope notes, and reciprocal notes are not checked |
| 6.2.3, 11.3.2.2 | History notes | partial | Each concept records `added_in`, exported as `skos:historyNote`. That is a release, not an entry date, and later changes are in the changelog, not per concept |
| 6.3–6.7 | Term form and proper-noun exclusion | partial | Label rules and the §6.3.3 product/benchmark exclusion gate are enforced in the refresh workflow; grammatical form across all 878 labels is reviewed rather than compiled |
| 7 | Compound terms | partial | The §7 compound factoring rule and `metric × workload` split gate are enforced in `.claude/workflows/taxonomy-refresh/`; legacy multiword labels (`93` labels joining ideas with "and", "versus", or "vs.") await editorial board review |
| 8.1 | Reciprocal relationships | met | Each relation has forward and inverse phrasing |
| 8.3, 8.3.1 | Generic hierarchy passes the all-and-some test | partial | Two independent audits read all 127 `specializes` edges. They agreed on 111 and both failed 27: 21 were retyped, 6 removed, and 2 of those re-added in reverse direction, leaving 100. These are editorial corrections pending technical review, and the 16 where the audits split await expert review |
| 8.3.3 | Whole-part links are inherent and exclusive | met by exclusion | Section 8.3.3.2 makes parts of multiple wholes associative. `part-of` is therefore exported under `skos:related`, not as hierarchy |
| 8.4 | Associative relationships | partial | Sixteen typed relations with conditions, read literally by reviewers; not checked against the clause item by item |
| 9.5 | Documentation | partial | `README.md`, this document, `taxonomy/method-study/synthesized-method.md`, and published tagging rules, with 878 preferred terms and 1,282 entry terms. Filing rules and an item-by-item check are open |
| 10 | Interoperability | open | Crosswalk design (`skos:exactMatch`, `skos:broadMatch`, `skos:closeMatch`, `skos:relatedMatch`) is specified in `synthesized-method.md`; published crosswalk files to ACM CCS, CS2023, SFIA, and ESCO remain open |
| 11.1.1 | Avoid duplicating existing vocabularies | partial | Neighboring taxonomies are named in the README, and internal cross-area lookup is enforced; external crosswalk files are open |
| 11.1.3 | Construction by subject experts and vocabulary specialists | partial | Candidate 7-seat external editorial board (`F`, `S`, `P`, `Q`, `T` + 2 vocabulary/SKOS specialists) and scoped review packet are defined in `taxonomy/method-study/candidate-experts.md`; board review has not yet been convened |
| 11.1.4 | A term record for every term | partial | Every concept has a record. The optional elements are incomplete: sources exist for 224 of 878 concepts, and there are no dated history notes |
| 11.1.5 | Verification before admission | partial | Candidates are checked by reviewers in three model families with deterministic reconciliation (`reconcile.py`); consulting technical references is recorded per family (`reference_role: reading-basis-not-entailment-proof`) and per study unit in `P4`/`Q4` |
| 11.1.6 | Candidates marked | met | Declared `gaps` and `status: draft` |
| 11.2.1 | Testing methods | open | The named methods are expert heuristic evaluation, card sorting, and usability testing. None has been run with people |
| 11.2.2 | Evaluation criteria, including indexing experiments | partial | Two-judge blind coverage and indexing experiments (`Hooper`, `Rolling`, $\alpha_{\text{MASI}}$, subject $\kappa$, specificity cap $p_{\max} \le 15\%$) have been run on the 300-question StaffML sample and the pre-registered `P4`/`Q4` held-out slice; a trial with human indexers is open |
| 11.3 | Date and responsible person for each change | partial | Git history records both per commit. The term records do not |
| 11.3.1 | Periodic review policy | met | The refresh procedure (`.claude/workflows/taxonomy-refresh.md`), run before each citable release |
| 11.3.1.1, 11.3.1.2 | Nominating and changing terms; review by an editor and board | partial | A public issue form takes nominations, the author is the editor, and the candidate board roster is prepared in `taxonomy/method-study/candidate-experts.md` |
| 11.3.1.3 | Over-used and rarely assigned terms reviewed | partial | The refresh evaluation enforces an anti-gaming specificity cap ($p_{\max} \le 15\%$ max single-concept share) and tracks zero-assignment terms; routine production review across all 878 concepts is open |
| 11.3.2 | Deprecated terms and their history | partial | `retired` with `replaced_by`, exported with `owl:deprecated` and `dct:isReplacedBy`. The former label, date, and predecessor note are not exported |
| 11.3.3 | Hyperlink maintenance | partial | Internal identifiers are checked by the compiler; external links are not routinely checked |

**Review queue from warrant.** 27 concepts have no recorded warrant in any corpus. This project reviews them under section 6.1; it is separate from the indexing-frequency review in 11.3.1.3. No warrant recorded does not mean no warrant exists, because the corpora are limited. Filter the concepts CSV for zero in all four warrant columns to list them. They are not removed until reviewed.

## Conformance to SKOS

| Item | Status | How it is met |
|---|---|---|
| Concepts, scheme, and collections | met | `skos:Concept`, `skos:ConceptScheme`, and `skos:Collection`, with groups, areas, and subjects as collections |
| S9, S37: class disjointness | met | Checked on the release for concepts, collections, and the scheme |
| S13, S14: label rules | met on concepts | Checked for `skos:prefLabel` and `skos:altLabel`; no hidden labels are exported |
| S27: hierarchy and association disjoint | met | Enforced by the compiler and checked on the release for `specializes` and the relation properties |
| Typed relations | met | `mlsys-taxonomy-relations.ttl` declares 17 OWL properties. `specializes` refines `skos:broader`, and 16 refine `skos:related`. `skos:broader` is also written out for tools without inference |
| Mapping properties | open | Awaits the crosswalks |
| Conditional meaning | partial | RDF carries the relation, while the condition under which it holds is in the JSON and GraphML releases only. A consumer of the Turtle alone sees an unconditional link |

The tests are in `binder/tests/test_taxonomy_skos.py`. They are a restricted check of this export, not a full SKOS validator, and passing them shows consistency, not truth.

## Publication and reuse

| Item | Status | How it is met |
|---|---|---|
| Stable identifier per concept | met | `https://mlsysbook.ai/taxonomy/concept/<id>`; identities are permanent |
| Identifiers resolve | partial | The build generates a redirect page for each URI, linking the RDF. Live resolution is unverified until deployment, and the host cannot serve content negotiation |
| License and creator | met | CC BY 4.0, recorded with DCTERMS |
| Catalog metadata | met | `mlsys-taxonomy.dcat.ttl` and `CITATION.cff` in the downloads |
| Per-version stable download | open | The site serves the current release only |
| Persistent citation | open | No DOI has been minted |

## Consistency

Agreement between two independent model taggers on the 300-question StaffML samples (`binder concepts agreement`) and on the pre-registered `P4`+`Q4` held-out slice (`taxonomy/method-study/`):

| Sample | Subject κ | Hooper | Rolling | $\alpha_{\text{MASI}}$ |
|---|---|---|---|---|
| 300 StaffML questions, first rules | 0.781 | 65.7% | 74.6% | — |
| Same 300, rules revised on them | 0.832 | 74.7% | 81.0% | — |
| Held-out 300 StaffML questions | 0.802 | 67.3% | 74.9% | — |
| Held-out `P4`+`Q4` slice (`43` items, `0.4-draft`) | 0.971 | 81.8% | 87.6% | 0.724 |

Hooper's measure is $c / (A + B - c)$ and Rolling's is $2c / (A + B)$, where $c$ is the number of concepts both taggers assigned and $A$ and $B$ are the numbers each assigned. Krippendorff's $\alpha_{\text{MASI}}$ weights set agreement by monotonic set-intersection distance. On the 300-question StaffML samples, scores are per-item means over items where both taggers assigned at least one concept (`280`, `286`, and `282` of `300` respectively; items where only one tagger assigned a concept — `12`, `12`, and `11` — are excluded, and subject $\kappa$ uses all `300`). On the held-out `P4`+`Q4` slice, all `43` items were indexed by both judges with max single-concept share $p_{\max} = 9.84\% \le 15.0\%$. These numbers measure agreement between models on these samples and do not stand in for a trial with human indexers.

## Governance

- **Editor.** Vijay Janapa Reddi holds final authority over scope, admission, and release.
- **Editorial board.** Candidate 7-seat roster (`F`, `S`, `P`, `Q`, `T` + 2 controlled-vocabulary/SKOS specialists) and scoped review packet prepared in `taxonomy/method-study/candidate-experts.md` (not yet contacted).
- **Change requests.** Anyone can propose an addition or correction through the [taxonomy issue form](https://github.com/harvard-edge/cs249r_book/issues/new?template=taxonomy_feedback.yml).
- **Review.** Each refresh decides every declared gap and every proposal, with reasons recorded. A public comment window before a citable release is open.
- **Identity.** Identifiers are permanent. A withdrawn concept is retired with its replacement, never deleted or reused.

## What stands between this and a citable release

1. Convene the external editorial board (`taxonomy/method-study/candidate-experts.md`) to review each group, with the 16 split hierarchy edges and the 27-concept zero-warrant queue as its first work.
2. Extend the pre-registered five-stratum sampling frame (`taxonomy/method-study/sampling-frame.md`) from the `P4` and `Q4` study slices to full-taxonomy provenance-family tagging across all 20 areas.
3. Run the human inter-indexer trial (`Hooper`, `Rolling`, $\alpha_{\text{MASI}}$, and $\kappa$) on a fresh stratified sample using the board calibration protocol.
4. Publish external crosswalks (`ACM CCS`, `CS2023`, `SFIA 9`, `ESCO`) with SKOS mapping properties.
5. Complete the review of the 93 legacy multiword labels and per-term sources against sections 7 and 11.1.4.
6. Issue a DOI and archived per-version downloads.

Until then the correct description is this: an editorial draft that adopts Z39.19 as its construction standard and publishes in SKOS, with partial conformance itemized above. Z39.19 is a set of recommendations, and this document does not claim full conformance.
