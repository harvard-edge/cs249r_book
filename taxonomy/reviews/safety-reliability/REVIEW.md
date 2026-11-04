# Review: safety properties and failure boundaries

Status: terminology and network-dependency corrections applied through `accepted-terminology.json`. The safety-method relationships and combined risk entry remain deferred. Sources inspected November 4, 2026. The packet preserves original records and every incident edge. This is terminology screening, not safety certification or a normative standards audit.

**Functional insufficiency.** The [ISO 21448 abstract](https://www.iso.org/standard/77490.html) distinguishes insufficiencies that contribute to hazards from safety of the intended functionality (SOTIF). Remove SOTIF from the preferred label and aliases, and define the deficiency rather than the resulting hazardous behavior. Keep the identifier. The full normative standard was not inspected.

All incident edges need claim-specific review before acceptance. In particular, distribution shift may expose an existing insufficiency rather than create it; perception limitations can themselves be insufficiencies; sensor degradation may involve faults outside the intended scope. STPA and indicator edges should describe what evidence they can identify under stated conditions, without promising detection. The abstract alone cannot validate these method-specific relationships.

**Failure domain.** [Azure's fault-domain documentation](https://learn.microsoft.com/en-us/azure/virtual-machines/availability-set-overview#fault-domains) distinguishes a group sharing dependencies from correlated failure events. Remove `correlated failure` as an equivalent alias. Network topology contributes network dependencies; it does not determine power and software failure boundaries. The proposed edge wording makes that scope explicit. Replication/failover and blast-radius edges remain provisional under their existing shared-cause and isolation conditions.

**Tolerable risk and ALARP.** Defer the combined entry and its incident edges. The current definition makes a particular risk-reduction approach universal, while the label deliberately combines that approach with general acceptance criteria. A revision must decide whether to separate the concepts or state the contextual scope explicitly. The safety-integrity-level edge also requires a direct source for its domain-specific risk-classification claim. No replacement definition or new concept is approved in this review.

Verification command:

```bash
python3 taxonomy/reviews/mechanism-boundaries/verify.py taxonomy/reviews/safety-reliability/decisions.json
```

This verifies the frozen input and inventory, not the technical conclusions. The proposals need recorded review and provenance-preserving application. Deferred matters require additional evidence and domain review.
