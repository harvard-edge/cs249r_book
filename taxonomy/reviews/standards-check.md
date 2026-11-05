# Standards source check

Date: November 4, 2026. Status: limited primary-source inspection, not a conformance audit.

The [official NISO PDF](https://groups.niso.org/higherlogic/ws/public/download/12591/z39-19-2005r2010.pdf), reached through NISO's publication page, was inspected at printed pages 16, 45–47, and 91–92. Warrant includes literature, organizational preferences, and user terminology. Near-synonyms and generic posting require attention to domain and specificity. Construction can combine committee and empirical methods; machine assistance leaves intellectual decisions with the vocabulary builders. The earlier download URL was replaced with this working official link.

The [ISO 25964-1 catalog](https://www.iso.org/standard/53657.html) confirms its scope covers thesaurus development, maintenance, and data interchange. The [ISO 25964-2 catalog](https://www.iso.org/standard/53658.html), also checked on this date, describes establishing and maintaining mappings between vocabularies. Neither full normative ISO text was inspected. The [W3C SKOS Reference](https://www.w3.org/TR/skos-reference/) defines the representation model; it does not establish the correctness of this vocabulary's subject inventory.

Follow-up: verify the remaining clause attributions against primary text and review actual term decisions separately from structural export checks. No historical study receipt was changed by this source check.

## Implemented checks and claim boundary

| Guidance area | Implementation or evidence | Assessment |
|---|---|---|
| NISO warrant and construction | Accepted review records identify sources, locators, interpretation, reviewer kind, and exact before/after content; METHOD supplies the repeatable recipe | Informed by inspected passages; not a clause-by-clause conformance audit |
| NISO term specificity | Explicit term decisions distinguish synonyms/acronyms from narrower terms and implementations; legacy retrieval terms do not automatically become alternative labels | Corrects the known representation problem; unassessed terms remain visible |
| ISO 25964-1 management and interchange | Stable identities, explicit retirement redirects, change history, draft snapshots, and machine-readable exports | Consistent with the catalog-described scope; normative requirements not independently verified |
| ISO 25964-2 interoperability | Mapping procedure records versions, concepts, scope evidence, mapping type, and review | Procedure specified; no completed external crosswalk claimed |
| SKOS lexical integrity | One preferred label per concept; tested preferred/alternative label disjointness; only explicit equivalent-term decisions yield alternative labels | Automated checks cover the implemented rules; source semantics still need editorial review |
| SKOS concepts and collections | Browsing groups/areas/subjects use collections; concepts, collections, and schemes are checked for disjointness | Browsing membership is not falsely asserted as generic hierarchy |
| SKOS semantic relations | Broader/related disjointness and hierarchy checked; conditional engineering claims retained as statement records | Custom properties are documented extensions, not additional W3C-defined relations |
| Human usefulness | Prepared independent-review, indexing, and navigation protocol | No participant results available |

On November 4, 2026, ISO's catalog still identifies the 2011 and 2013 editions as current and signals revision work, including an expected replacement for Part 1. Recheck editions at the next release. This assessment does not purchase or reproduce ISO texts, validate historical clause claims, or certify legal or standards compliance. The supported public wording is “informed by ANSI/NISO and ISO vocabulary practices, with a SKOS representation,” accompanied by these limits.
