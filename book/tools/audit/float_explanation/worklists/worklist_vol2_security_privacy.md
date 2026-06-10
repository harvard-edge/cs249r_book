# Float-explanation worklist — security_privacy.qmd (vol2)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 25 | 24 | 1 | 0 |
| table | 13 | 12 | 1 | 0 |
| listing | 0 | — | — | — |
| algorithm | 0 | — | — | — |
| equation | 0 | — | — | — |
| **total** | **38** | **36** | **2** | **0** |

Note: one dangling ref flagged by the scanner (`@fig-ondevice-gboard` at L1511 inside a lighthouse callout — the figure definition does not exist in this chapter). Not counted in the float table above as there is no def to judge, but should be resolved separately.

---

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

### ⚠️ `fig-side-channel-curves` — def L1302  (Thin)
- **Caption:** **Power Traces**: Cryptographic computations reveal subtle, data-dependent variations in power consumption that reflect internal states during specific operations.
- **Ref(s):** L1300 `@Fig-side-channel-curves`: "illustrates how cryptographic computations produce data-dependent power consumption signatures that reveal algorithmic state. These variations, while subtle, are measurable and reflect the internal state of the algorithm at specific points in time."
- **Context checked:** ref ✓ (general) · prev ¶ ✓ (introduces SCAAML framework) · next ¶ ✓ (L1373, explains ML decoding) · caption (partial — describes left panel only) · payoff ✓ (explains ML key recovery) · **figure content gap**: the figure's key visual element — the right-panel zoom showing three traces labeled 0000, 1111, and 0101 — is not called out anywhere in the neighborhood. The reader sees binary labels on amplitude-separated curves but is never told that these labels indicate the secret data values whose corresponding power levels differ, which is the entire point of why the traces are distinguishable.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Fig-side-channel-curves illustrates how cryptographic computations produce data-dependent
  - power consumption signatures that reveal algorithmic state. These variations, while subtle,
  - are measurable and reflect the internal state of the algorithm at specific points in time.
  + @Fig-side-channel-curves captures the core leakage mechanism: the left panel overlays three
  + power traces from AES computations on different plaintexts, and the right panel zooms into a
  + single clock region to reveal that the traces separate by amplitude — with the binary labels
  + 0000, 1111, and 0101 indicating which intermediate data value each trace represents. A
  + classifier trained on these shape differences can infer the secret data value directly from
  + the power waveform, without any logical access to the cryptographic computation.
  ```

---

### ⚠️ `tbl-defense-mapping` — def L408  (Thin)
- **Caption:** **Defense Mapping by Attack Surface**: Each layer of the ML system attack surface requires specific defensive mechanisms and detection methods. Effective security integrates protections across all layers while maintaining detection capabilities that can identify attacks that bypass preventive controls.
- **Ref(s):** L399 `@Tbl-defense-mapping`: "@Tbl-defense-mapping provides a concrete mapping from each attack surface layer to the defensive mechanisms and detection methods that protect it."
- **Context checked:** ref ✗ (pure pointer) · prev ¶ ✓ (fail-safe defaults) · next ¶ ✓ (L410, threat modeling framing) · caption ✓ (explains purpose) · payoff ✓ (L410, explains role in chapter arc) · **gap**: the ref sentence is a bare pointer with no indication of what the reader should take away — which layer is the hardest to defend, which column reveals the detection gap, or why this four-layer structure matters. The caption carries the substance but the ref sentence offers no guidance on what to look for.
- **Suggested rewrite (flag-only):**
  ```diff
  - @Tbl-defense-mapping provides a concrete mapping from each attack surface layer to the
  - defensive mechanisms and detection methods that protect it.
  + The architecture organizes defenses by where they sit: @Tbl-defense-mapping maps each
  + attack surface layer — data, model, interface, and infrastructure — to its primary threats,
  + the controls that prevent them, and the detection methods that catch what prevention misses.
  + Reading across the detection column reveals that no layer has a simple "detect everything"
  + mechanism; each depends on behavioral signals specific to that layer's threat surface.
  ```
