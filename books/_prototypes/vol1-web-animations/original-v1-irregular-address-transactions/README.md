# Why do equal useful bytes create different memory work?

Show returned-sector surplus separately from the dependency that prevents an unknown payload address from issuing.

Scope: Four-byte words, sixteen-byte aligned sectors; cold cache/no reuse/within-sector merging. Independent address-dependency example.

1. Merge adjacent requests into one fetched sector.
2. Return four scattered sectors for the same useful payload.
3. Read index, retain its result, then issue and receive a dependent payload.

Guard: Not physical DRAM transactions or latency. Address and data values remain distinct; spatial and dependency examples are not combined.

Run `python3 books/_prototypes/vol1-web-animations/original-v1-irregular-address-transactions/check.py`. The fixture uses the full owning chapter from the canonical art index. It tests exact endpoints, repeatable menu jumps, keyboard, real-motion freeze/resume in Expand, reduced motion, overflow, and print player hiding in Chromium/WebKit, desktop/phone, light/dark. Production ordinary visible playback and complete static fallback require the root integration build. Source PNGs: `/tmp/v1-irregular-address-transactions.png`. QA PNGs: `/tmp/original-v1-irregular-address-transactions-review`. No real student study is claimed.
