# StaffML Question Vault — UX/UI Redesign Plan

## Problem Summary

The current Vault page has four core issues:
1. **Insufficient text contrast** — near-black background with thin text
2. **Flat visual hierarchy** — everything competes for attention equally
3. **Generic aesthetics** — no memorable identity or visual language
4. **Information density** — 735 topics across 13 areas is overwhelming

## Color System

### Revised Dark Theme (warm charcoal, not pure black)

```css
:root {
  --background:       #101014;   /* slightly lifted, faint warm tint */
  --surface:          #18181f;   /* more separation from bg */
  --surface-hover:    #222230;   /* visible hover state */
  --surface-elevated: #262634;   /* NEW — cards, panels, modals */

  --border:           #2a2a38;   /* warmer, more visible */
  --border-subtle:    #222230;   /* inner separators */
  --border-highlight: #3e3e50;   /* hover state borders */

  --text-primary:     #ececf0;   /* slightly dimmed to reduce harshness */
  --text-secondary:   #a8a8b8;   /* clear step from primary */
  --text-tertiary:    #6c6c80;   /* clear step from secondary */
  --text-muted:       #4a4a5c;   /* truly de-emphasized (timestamps, meta) */

  --accent-blue:      #4d94ff;
  --accent-red:       #f06060;
  --accent-amber:     #f0b040;
  --accent-green:     #34d399;
  --accent-purple:    #a78bfa;
}
```

### Area Colors (with pre-computed bg/border)

Each area color ships `primary`, `bg` (8% opacity), and `border` (19% opacity):
- memory: `#60a5fa`, compute: `#fbbf24`, deployment: `#4ade80`
- architecture: `#c084fc`, latency: `#f87171`, cross-cutting: `#22d3ee`
- data: `#2dd4bf`, networking: `#fb923c`, power: `#e8b83d`
- optimization: `#a78bfa`, precision: `#f472b6`, reliability: `#818cf8`
- parallelism: `#34d399`

## Typography Scale

**Critical rule: nothing below 11px. Body text weight 500 on dark backgrounds.**

| Token | Size | Weight | Use |
|-------|------|--------|-----|
| display | 32px | 800 | Page title |
| heading | 22px | 700 | Section headers, area names |
| title | 17px | 700 | Card titles when larger |
| body | 15px | 400 (500 on dark) | Default body text |
| body-sm | 14px | 400 | Secondary body, card titles |
| caption | 12px | 500 | Metadata, counts |
| label | 11px | 600 | Section labels (uppercase) |
| mono-sm | 12px | 500 | Numbers, code (JetBrains Mono) |

## Card Design

- 2px colored gradient bar at top (area color fading to transparent)
- Hover: `surface-elevated` bg + subtle box-shadow + border highlight
- Replace tiny vertical level bars with horizontal stacked bar (full width)
- Padding: `p-4 pt-5` (accounting for accent bar)

## Detail Panel

- Header: subtle area-colored gradient wash background
- Section headers: centered with horizontal rules on each side (Linear pattern)
- CTA button: area-colored background (not white)
- Level rows: larger text (14px bold mono for level, 12px for label)
- Mobile: bottom sheet instead of side panel

## Area Identity

Each area gets a lucide icon + colored icon badge (40x40px):
- memory: HardDrive, compute: Cpu, deployment: Rocket
- architecture: Layers, latency: Timer, cross-cutting: Shuffle
- data: Database, networking: Network, power: BatteryCharging
- optimization: Gauge, precision: Binary, reliability: Shield
- parallelism: GitBranch

## Information Architecture

**Tier 1 (default)**: Compact area list — 13 horizontal cards, fits one screen
**Tier 2 (click area)**: Full topic grid for selected area (3-col, all topics shown)
**Tier 3 (click topic)**: Detail panel with progressive drill-down

## Implementation Phases

1. Foundation: CSS vars, Tailwind config, font scale
2. Data: Area colors as triple, area icons
3. Components: AreaSection, TopicCard, TopicDetail, FilterPills
4. Mobile: Bottom sheet, filter sheet, sticky search
5. Accessibility: aria labels, keyboard nav, focus rings, reduced motion
6. Polish: Cross-browser testing, performance audit
