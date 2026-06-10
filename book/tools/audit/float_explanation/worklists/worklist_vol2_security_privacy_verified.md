# Verified findings — security_privacy.qmd (vol2)
Prior findings: 2 | Survived: 0 | Refuted: 2

---

## SURVIVING findings

(none)

---

## REFUTED findings

- `fig-side-channel-curves` — REFUTED: explanation across ref sentence (L1300–1301) and payoff paragraph (L1373). The ref sentence states that the traces "produce data-dependent power consumption signatures that reveal algorithmic state" and that the variations "reflect the internal state of the algorithm at specific points in time" — giving the reader the core leakage mechanism. The payoff paragraph (L1373) closes the loop: "a neural network can learn to associate the shape of these signals with the specific data values being processed," which explains why the amplitude-separated traces labeled 0000, 1111, and 0101 are classifiable. The prior audit's specific concern — that the binary labels in the right panel are never explained in visible prose — is addressed in the fig-alt attribute: "Right zooms into highlighted region, revealing amplitude differences between traces labeled with binary values 0000, 1111, and 0101." Ref + payoff + fig-alt together tell the reader what the figure shows and why the separability matters for key recovery.

- `tbl-defense-mapping` — REFUTED: explanation in caption (L408) and payoff paragraph (L410). The caption reads "Effective security integrates protections across all layers while maintaining detection capabilities that can identify attacks that bypass preventive controls" — explaining why the four-layer structure matters, not just what it contains. The three preceding paragraphs (L391–397) explain the defense principles each layer row instantiates, and the payoff paragraph (L410) frames the table's chapter-arc role: "The threat modeling framework provides the analytical foundation for the specific attack vectors and defensive techniques examined throughout the remainder of the chapter." The ref sentence (L399) is a bare pointer, but the bar requires that ANY neighborhood element carry the explanation. Caption + payoff collectively clear that bar.
