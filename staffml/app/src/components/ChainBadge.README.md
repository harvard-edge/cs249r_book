# ChainBadge — Phase-5 single intervention

Pre-reveal chain indicator. Mount above question title on any chained question.

## Wiring

```tsx
import ChainBadge from "@/components/ChainBadge";

{question.chain && (
  <ChainBadge
    chainId={question.chain.id}
    chainName={chainMeta?.name}         // optional; look up from chains.yaml
    position={question.chain.position}
    total={chainMeta?.total ?? 1}
    onClick={() => openChainStrip()}    // scroll to existing ChainStrip or open modal
  />
)}
```

## Events emitted (fire-and-forget)

- `chain_badge_shown`  — on mount (debounced per question per session).
- `chain_badge_clicked` — on click.

Both feed the Phase-5 engagement gates in `vault stats` / Grafana.

## Gate thresholds for promoting to multi-intervention

Per ARCHITECTURE.md §8:

1. CTR on the badge > 15% (clicked/shown).
2. Within-chain completion rate on chained questions > 1.5× completion on non-chained at matched level/zone.

Only after two weeks of data AND both thresholds hit:

- Add sidebar "Chained questions only" filter.
- Add first-time tooltip.
- Add `/chains` browse landing page.
- Add chain completion tracking in user dashboard.

Otherwise the badge is enough; do not ship additional surface area on speculation.

## No SSR gotchas

The component is a client component (`"use client"`). Renders stable text
between server and client because it doesn't read `window` during render —
the `queueMicrotask` guard defers the `trackEvent` side effect to after
hydration.
