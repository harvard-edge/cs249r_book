# How the ML Systems Taxonomy is built

This document states the published standards the taxonomy adopts, how each recommendation is met, and where it is not yet met. Status terms are used strictly: **met** means the record or a check shows it, **partial** means some of it is done and the rest is named, and **open** means it has not been done. Numbers describe version `0.4-draft` and are computed from the release files named beside them.

## The standards

| Layer | Standard adopted | Role |
|---|---|---|
| Construction | [ANSI/NISO Z39.19-2005 (R2010)](https://groups.niso.org/higherlogic/ws/public/download/6484/Guidelines%20for%20the%20Construction,%20Format,%20and%20Management%20of%20Monolingual%20Controlled%20Vocabularies.pdf), *Guidelines for the Construction, Format, and Management of Monolingual Controlled Vocabularies* | How terms are chosen, formed, related, tested, and maintained |
| Representation | [W3C SKOS Reference](https://www.w3.org/TR/skos-reference/) (Recommendation, 2009), with a declared OWL extension for typed relations | The published data model |
| Metadata and publication | [DCTERMS](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/), [W3C DCAT 3](https://www.w3.org/TR/vocab-dcat-3/), [W3C Best Practice Recipes for Publishing RDF Vocabularies](https://www.w3.org/TR/swbp-vocab-pub/), and [Ten simple rules for making a vocabulary FAIR](https://pmc.ncbi.nlm.nih.gov/articles/PMC8238180) | Identifiers, licensing, versions, and reuse |
| Validation | Inter-indexer consistency as used in library science: Hooper's measure and [Rolling's measure](https://informationr.net/ir/18-4/paper601.html), reported beside Cohen's kappa | Whether people and tools apply the vocabulary the same way |

**ISO 25964 is a reference, not a conformance claim.** ISO 25964-1 and ISO 25964-2 are the international thesaurus and interoperability standards, and [NISO describes their scope](https://niso.org/schemas/iso25964). Their text is sold rather than published, and this project has not read it, so no clause-level conformance is claimed. The mapping properties in SKOS are the planned route for its interoperability guidance.

**What Z39.19 says this is.** Under section 5.4 a *taxonomy* connects all preferred terms in a hierarchy, and a *thesaurus* displays several relationship types with standardized indicators. This vocabulary keeps the familiar name but is structurally a thesaurus: most links are associative, and subjects are browsing groups rather than broader terms.

**Ontology methodologies.** This project adopts controlled-vocabulary guidance and does not claim to implement an ontology-development methodology such as NeOn or METHONTOLOGY. Relations are declared in OWL only so that the SKOS extension is machine-readable.

## How a version is built

Z39.19 section 11.1.3 recommends a committee approach, an empirical approach, or a combination. This project uses empirical construction under a top-down outline, with model-assisted review. It has no expert committee, which is a departure. The refresh procedure follows this order:

1. **Scope and structure first** (11.1.2 asks that structure and display be decided before terms are collected). The five groups, 20 areas, and subjects are fixed at the start of a refresh and change only by recorded decision.
2. **Check existing vocabularies** (11.1.1, 11.1.5). Candidates are searched against the vocabulary and named external sources.
3. **Collect candidates** (11.1.3.2, 11.1.3.4). Candidates come inductively from the content being indexed: book chapters, StaffML questions that could not be tagged, conference programs, and courses.
4. **Establish warrant** (5.3.5). Each candidate must show recorded evidence, and the project's two-source rule applies.
5. **Control the term** (6, 7). One preferred label, entry terms as aliases, one definition, and one home subject.
6. **Relate the term** (8). Generic hierarchy must pass the all-and-some test. Everything else is associative.
7. **Verify independently** (11.1.5, 11.2.2). Reviewers from other model families read every proposal, and a reconciliation script fails if a flagged item is unchanged and has no written override. It detects unaddressed verdicts; whether a correction is adequate needs review.
8. **Test** (11.2). Coverage against recorded checklists, a held-out test compiled without access to the vocabulary, and tagging consistency.
9. **Record and release** (11.3). Changelog, identity rules, generated downloads, and validation of the RDF.

The step-by-step procedure is maintained with the project's editorial tooling. `README.md` describes the maintenance loop, and `CHANGELOG.md` records what each round did and which reviewers took part.

**Human and machine roles.** Z39.19 section 11.1.3.4 assumes construction calls for intellectual decisions and describes machines as assistants for tasks such as identifying candidate terms. This project goes further: language models proposed terms, cross-checked them, and audited the result, and one author holds final authority. That is a real departure from the standard's assumption, and it is the main reason the status below stays *editorial draft*.

## Conformance to Z39.19

Appendix A of the standard is informative and summarizes its recommendations. The table follows the clauses that apply to this vocabulary.

| Clause | Requirement | Status | How it is met, or what is missing |
|---|---|---|---|
| 5.3.1 | Each term has one meaning | partial | The compiler rejects a label or alias shared by two concepts. That proves unique strings, not unambiguous meaning, and no ambiguity review is recorded |
| 5.3.2 | Each concept has one preferred term; synonyms point to it | met | `label` and `aliases`, exported as `skos:prefLabel` and `skos:altLabel` |
| 5.3.3 | Equivalence, hierarchical, and associative relationships | met | Aliases, `specializes`, and the other 16 relations |
| 5.3.5.1 | Literary warrant | partial | 699 of 878 concepts occur in book chapters and 471 are cited by covered paper, course, or standards topics. Counts are computed, but the corpus is not yet an enumerated sample of the field |
| 5.3.4 | Facets | met by design | The standard leaves facet choice to the designer; groups and row facets are stated in the framework |
| 5.3.5.2 | Organizational warrant | partial | 425 concepts are tied to covered book topics or section links. That is coverage; the clause asks for the term forms the organization prefers, and those decisions are not documented |
| 5.3.5.3 | User warrant | partial | The standard means user requests and searches, which are not recorded. `user_warrant_proxy` counts StaffML question tags (626 concepts), a term-assignment frequency and not user behavior |
| 6.1 | Term choice by domain, warrant, and specificity | partial | The admission test is written and applied; 27 concepts have no recorded warrant of any kind and are listed below for review |
| 6.2.2 | Scope notes | partial | Subjects carry includes and excludes, exported as `skos:scopeNote`. Concepts carry definitions, not separate scope notes, and reciprocal notes are not checked |
| 6.2.3, 11.3.2.2 | History notes | partial | Each concept records `added_in`, exported as `skos:historyNote`. That is a release, not an entry date, and later changes are in the changelog, not per concept |
| 6.3–6.7 | Term form | partial | Label rules are written; conformance has been reviewed, not checked by a compiled rule |
| 7 | Compound terms | partial | Multiword labels have not been reviewed against the criteria in 7.5 and 7.6. 93 labels join ideas with "and", "versus", or "vs.", such as "Stop conditions and output-length limits", and come first |
| 8.1 | Reciprocal relationships | met | Each relation has forward and inverse phrasing |
| 8.3, 8.3.1 | Generic hierarchy passes the all-and-some test | partial | Two independent audits read all 127 `specializes` edges. They agreed on 111 and both failed 27: 21 were retyped, 6 removed, and 2 of those re-added in reverse direction, leaving 100. These are editorial corrections pending technical review, and the 16 where the audits split await expert review |
| 8.3.3 | Whole-part links are inherent and exclusive | met by exclusion | Section 8.3.3.2 makes parts of multiple wholes associative. `part-of` is therefore exported under `skos:related`, not as hierarchy |
| 8.4 | Associative relationships | partial | Sixteen typed relations with conditions, read literally by reviewers; not checked against the clause item by item |
| 9.5 | Documentation | partial | `README.md`, this document, and published tagging rules, with 878 preferred terms and 1,254 entry terms. Filing rules and an item-by-item check are open |
| 10 | Interoperability | open | No crosswalk to ACM CCS, CS2023, SFIA, or ESCO exists |
| 11.1.1 | Avoid duplicating existing vocabularies | partial | Neighboring taxonomies are named in the README, but the documented comparison is open |
| 11.1.3 | Construction by subject experts and vocabulary specialists | open | No such committee has taken part |
| 11.1.4 | A term record for every term | partial | Every concept has a record. The optional elements are incomplete: sources exist for 224 of 878 concepts, and there are no dated history notes |
| 11.1.5 | Verification before admission | partial | Candidates are checked by reviewers in other model families; consulting technical references is not recorded per term |
| 11.1.6 | Candidates marked | met | Declared `gaps` and `status: draft` |
| 11.2.1 | Testing methods | open | The named methods are expert heuristic evaluation, card sorting, and usability testing. None has been run with people |
| 11.2.2 | Evaluation criteria, including indexing experiments | partial | Model taggers have indexed the same items; an indexing experiment with people is open |
| 11.3 | Date and responsible person for each change | partial | Git history records both per commit. The term records do not |
| 11.3.1 | Periodic review policy | met | The refresh procedure, run before each citable release |
| 11.3.1.1, 11.3.1.2 | Nominating and changing terms; review by an editor and board | partial | A public issue form takes nominations, and the author is the editor. There is no board |
| 11.3.1.3 | Over-used and rarely assigned terms reviewed | open | The clause concerns indexing frequency. No thresholds or review exist yet; StaffML tag counts are available |
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

Agreement between two independent model taggers, recomputed by `binder concepts agreement`:

| Sample | Subject κ | Hooper | Rolling |
|---|---|---|---|
| 300 questions, first rules | 0.781 | 65.7% | 74.6% |
| Same 300, rules revised on them | 0.832 | 74.7% | 81.0% |
| Held-out 300 | 0.802 | 67.3% | 74.9% |

Hooper's measure is c / (A + B − c) and Rolling's is 2c / (A + B), where c is the number of concepts both taggers assigned and A and B are the numbers each assigned. Scores are per-item means over items where both taggers assigned at least one concept: 280, 286, and 282 of 300 respectively. Items where only one tagger assigned a concept (12, 12, and 11) are excluded, and subject kappa uses all 300. These numbers measure agreement between models on these samples and do not stand in for a trial with human indexers.

## Governance

- **Editor.** Vijay Janapa Reddi holds final authority over scope, admission, and release.
- **Editorial board.** Open. Section 11.3.1.1 prefers review by a board; this project plans one named expert for each of the five groups.
- **Change requests.** Anyone can propose an addition or correction through the [taxonomy issue form](https://github.com/harvard-edge/cs249r_book/issues/new?template=taxonomy_feedback.yml).
- **Review.** Each refresh decides every declared gap and every proposal, with reasons recorded. A public comment window before a citable release is open.
- **Identity.** Identifiers are permanent. A withdrawn concept is retired with its replacement, never deleted or reused.

## What stands between this and a citable release

1. A named editorial board reviews each group, with the 16 split hierarchy edges and the review queue as its first work.
2. A written sampling frame defines the courses, venues, standards, and roles that completeness is measured against.
3. An indexing trial with human indexers reports Hooper, Rolling, and kappa on a fresh sample.
4. Crosswalks to neighboring vocabularies are published with SKOS mapping properties.
5. Compound terms and per-term sources are reviewed against sections 7 and 11.1.4.
6. A DOI and per-version downloads are issued.

Until then the correct description is this: an editorial draft that adopts Z39.19 as its construction standard and publishes in SKOS, with partial conformance itemized above. Z39.19 is a set of recommendations, and this document does not claim full conformance.
