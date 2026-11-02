# When does a generated token enter the KV cache?

Source-art prototype for `vol2:fig-inference-lifecycle`. Static source and caption stay unchanged.

Declared token identities: prompt p0,p1; sampled outputs y0,y1. One request crosses CPU/GPU. KV entries are created only for tokens actually consumed as model inputs.

Invariant: Sampler output and consumed model input are distinct; widths are not measured latency.

Run `python3 books/_prototypes/vol1-web-animations/original-v2-inference-lifecycle/check.py` for actual-chapter fixtures and browser checks. Review screenshots are written only under `/tmp`.

## Frozen review assets

`original-v2-inference-lifecycle.js`: `f9c45477b31875a7fa292b2cff161166556e31396012b0e76a2c7a361d83c4cb`
`original-v2-inference-lifecycle.css`: `e1ac750a94d6441203c09e06035b33f4000eeaca515a8ff1b0fea7500a4900ff`
`entry.json`: `6ef272b7a477b7d906fe2b899955609c0cfba1579647388d14b4b968ad6c14ec`
