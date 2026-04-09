# Gemini Call 1 — StaffML Holistic Audit

**Date:** 2026-04-07
**Model:** `gemini-3.1-pro-preview` (via Gemini CLI 0.35.3)
**Prompt size:** 7,461 lines / 331 KB / ~85K tokens (full bundle: 4 plan docs +
all 11 page routes + key components + globals.css + next.config + package.json)
**Response:** Substantive structured critique in 6 sections.

## Note on the call

The Gemini CLI's agent loop auto-invoked file-read tools on path-like
substrings inside our pasted source code (e.g. `react-sigma/core`,
`/lib/taxonomy`), producing 1,512 lines of `ENAMETOOLONG` error noise before
the actual critique began. The critique itself (extracted below, lines
1513-1639 of the raw response) was unaffected by the noise. Subsequent calls
should use `--approval-mode plan` to suppress tool invocation.

## Verified vs hallucinated claims

| Claim | Verdict |
|---|---|
| `package.json` line 18 leading space `" @react-sigma/core"` | **HALLUCINATION** — line 17 is clean `"@react-sigma/core"` |
| `deep_dive_url` XSS via `javascript:` protocol | **CONFIRMED** at `practice/page.tsx:676,826` and `vault/TopicDetail.tsx:148` |
| CSP `'unsafe-inline'` defeats script protection | **CONFIRMED** at `layout.tsx:46` |
| Missing `frame-ancestors` | **CONFIRMED** |
| Unbounded `localStorage` in contribute | **CONFIRMED** at `contribute/page.tsx:47-53` |

---

## Critique (verbatim)

### Section 1 — Plan Doc Triage

**1. `V1_REDESIGN_SPEC.md`**
- **Status:** PARTIAL
- **Unbuilt:** The "Aha moment" inline placement quiz (3 questions, 90 seconds) is completely missing from the landing page. Shareable challenge links (`staffml.com/challenge/...`) do not exist. The `LevelExplainer` component was built but is not integrated into the onboarding flow or landing page.
- **Overlap:** The placement quiz is reiterated as a top priority in `FEEDBACK_SYNTHESIS.md`.

**2. `STAFFML_UX_PLAN.md`**
- **Status:** PARTIAL
- **Unbuilt:** The Vault page tab navigation (`Browse | Stats | Map`) was abandoned; Stats became a standalone dashboard route. The Taxonomy Concept Search (showing prerequisite chains) is missing. The Sigma.js graph map was never reintegrated.

**3. `VAULT_DESIGN_PLAN.md`**
- **Status:** OBSOLETE
- **Why delete:** This document covers the exact same "Three Views" concept as `STAFFML_UX_PLAN.md` but is stuck in the ideation/URL-structure phase. The app's architecture has already diverged from it (routing to `/dashboard` instead of `/vault/stats`). It adds noise and should be removed.

**4. `FEEDBACK_SYNTHESIS.md`**
- **Status:** PARTIAL
- **Unbuilt:** The mobile sidebar remains a stacked block rather than a "collapsible drawer." Guided study plans derived from user heat map gaps are not implemented (plans are statically hardcoded). Interviewer curation mode does not exist.

---

### Section 2 — Mobile Responsiveness Audit (Top 5 Risks)

**Route-by-Route Single Biggest Risk:**
- `/` (Home): `h-[calc(100vh-3.5rem)]` causes the bottom sheet (`TopicDetail`) to be cut off behind the iOS Safari address bar.
- `/practice`: The `<aside>` sidebar stacks vertically above the question. A mobile user has to scroll past ~800px of filters just to see the first line of the question scenario.
- `/gauntlet`: The answer textarea uses `min-h-[200px]`. When the virtual keyboard opens, the scenario above it is pushed entirely out of view, forcing the user to constantly scroll up/down while typing.
- `/progress`: The heat map `<table>` spans 6 columns. While `overflow-x-auto` exists, tapping tiny cells (40x48px) inside a scrolling table on a phone will cause high misclick rates.
- `/framework`: The `ElementDetail` overlay modal has no `max-h` or `overflow-y-auto` on the modal body. On short landscape screens, the top and bottom of the modal will bleed off-screen, rendering it uncloseable.
- `/dashboard`: SVG `CompetencyRadar` is hardcoded to `320px` width. On an iPhone SE (375px) with `px-6` (48px) padding, it forces a horizontal layout break.
- `/contribute` & `/roofline`: Inputs use `text-sm` (14px).

**Top 5 Ranked Mobile Risks:**
1. **The Practice Sidebar UX (`/practice`):** Stacking 6 complex filter groups above the main content completely breaks the core loop on mobile. It must be converted into a collapsible bottom sheet or hidden behind a "Filters" FAB.
2. **`100vh` Safari Cutoff (`/` and `/gauntlet`):** Relying on `100vh` without `dvh` support will obscure crucial action buttons (like "Reveal Answer") behind the mobile browser's bottom navigation bar.
3. **Input Zooming on iOS:** Inputs and textareas across the app (`/practice`, `/contribute`, `/gauntlet`, `/roofline`) use `text-sm` (14px) and `text-[13px]`. iOS Safari forces a disorienting page zoom when focusing inputs smaller than 16px.
4. **Modal Overflow (`/framework`):** The Element overlay lacks inner vertical scrolling, making it a fatal trap on small/landscape viewports.
5. **Fixed SVG Widths (`/dashboard`):** Hardcoded `320px` widths on SVGs will cause horizontal scrolling or clipping on the narrowest supported devices (iPhone SE) due to parent padding.

---

### Section 3 — Security Audit (Top 5 Risks)

1. **Unbounded `localStorage` Growth (DoS Vector)**
   - **Citation:** `src/app/contribute/page.tsx` line 44 (`all.push(c); window.localStorage.setItem(...)`)
   - **Risk:** There is no size limit. A user (or malicious script) spamming the "Save & Continue" button will rapidly fill the 5MB `localStorage` quota. Once the quota is hit, synchronous writes across the entire app (`saveAttempt`, theme switching) will throw a fatal `QuotaExceededError`, bricking the app.
   - **Fix:** Enforce a hard array length cap: `if (all.length > 50) all.shift();` before pushing.

2. **Unsanitized Deep-Dive URLs (Reflected XSS)**
   - **Citation:** `src/app/practice/page.tsx` line 490 (`href={current.details.deep_dive_url.replace(...) ? ...}`) and `TopicDetail.tsx` line 147.
   - **Risk:** These URLs are read directly from the untrusted JSON corpus. If a malicious PR introduces a payload like `javascript:alert(localStorage.getItem('...'))` into `deep_dive_url`, React will happily render it into the `href` attribute, resulting in 1-click XSS.
   - **Fix:** Validate the URL protocol before rendering: `href={url.startsWith('http') ? url : '#'}`.

3. **Defeated Content Security Policy (CSP)**
   - **Citation:** `src/app/layout.tsx` line 41 (`script-src 'self' 'unsafe-inline'`)
   - **Risk:** `unsafe-inline` entirely nullifies the primary benefit of CSP (stopping injected scripts). Since this is a static export, you cannot use request-time nonces.
   - **Fix:** Next.js static exports can generate CSP hashes for inline scripts. Configure `next.config.mjs` to generate a strict CSP with SHA hashes for the theme script, and remove `'unsafe-inline'`.

4. **Invalid Package Name breaking CI**
   - **Citation:** `package.json` line 18 (`" @react-sigma/core": "^5.0.6"`)
   - **Risk:** There is a leading space in the package key. This is an invalid npm package name. Depending on the environment, this will cause `npm install` to fail or create phantom resolution bugs.
   - **Fix:** Remove the leading space. (And since the Sigma graph isn't implemented, uninstall the package entirely).

5. **Missing Clickjacking Protection**
   - **Citation:** `next.config.mjs` and `layout.tsx`
   - **Risk:** Without `frame-ancestors 'none'` in the CSP or an `X-Frame-Options` header, the app can be embedded in a malicious iframe to trick users into submitting fake contributions or revealing assessment data.
   - **Fix:** Add `frame-ancestors 'none';` to the CSP meta tag in `layout.tsx`.

---

### Section 4 — Feature Completeness Audit

**Claimed but entirely missing:**
- **Inline Placement Quiz:** The "90-second aha moment" described as a vital acquisition loop in `V1_REDESIGN_SPEC.md` does not exist on the landing page or anywhere else.
- **Challenge Links:** The viral loop (`staffml.com/challenge/abc123`) does not exist. The current "Share Score" button is just a clipboard copy.
- **Taxonomy Map View:** The Sigma.js interactive concept graph specified in `STAFFML_UX_PLAN.md` is absent.

**Half-built or disjointed:**
- **Guided Study Plans:** The UI exists (`/plans`), but it fails the requirement from `FEEDBACK_SYNTHESIS.md` to generate plans based on *the user's heat map gaps*. They are currently just static arrays.
- **Spaced Repetition:** The review queue exists inside `/practice`, but its UX is jarring. Toggling "Review Mode" hijacks the current view without a clear explanation of the algorithm or how many cards remain beyond a small counter.
- **Mobile Sidebar:** Acknowledged as broken in feedback, but still implemented as a massive vertical stack that pushes questions off-screen.

---

### Section 5 — Forward Plan

**1. Build the Mobile-First Practice Filter Drawer**
- **What:** Move all left-sidebar filters in `/practice` into a collapsible bottom sheet triggered by a sticky "Filters" FAB on mobile.
- **Why:** Solves the most severe usability blocker. Mobile users cannot currently practice without tedious scrolling.
- **Effort:** M | **Risk:** High (CSS stacking context bugs with modals/textareas).
- **Checklist:** Test on iPhone SE. Verify virtual keyboard does not push the FAB over the text input.

**2. Ship the Onboarding Placement Quiz**
- **What:** Build a 3-question inline component for `page.tsx` that evaluates the user and writes a recommended starting level to `localStorage`.
- **Why:** Delivers the "Start Here" flow demanded by every persona in the feedback synthesis.
- **Effort:** M | **Risk:** Medium (State complexity in the already heavy `HomePage` component).
- **Checklist:** Ensure quiz UI requires zero vertical scrolling. Validate that text rendered from the corpus inside the quiz is strictly escaped.

**3. Implement `localStorage` Defenses & Input Zoom Fixes**
- **What:** Add LRU capping to `contribute` arrays. Up-size all `<input>` and `<textarea>` fonts to `16px` (`text-base` in Tailwind) on mobile viewports. Validate `href` protocols.
- **Why:** Closes the DoS crash vector, prevents XSS via textbook URLs, and stops the disorienting iOS Safari auto-zoom.
- **Effort:** S | **Risk:** Low.
- **Checklist:** Verify input font sizes in DevTools. Run a script to inject 10MB of data into `saveContribution` and ensure it self-trims.

**4. Viewport Height (`100dvh`) Patch**
- **What:** Replace `h-screen` and `100vh` variants in layout containers with `100dvh` (falling back to `100vh` for legacy browsers).
- **Why:** Fixes the bottom-bar cutoff on iOS Safari that currently obscures the "Next Question" and "Reveal" action bars.
- **Effort:** S | **Risk:** Low.
- **Checklist:** Deploy to staging and verify on physical iOS Safari.

**5. Vault Consolidation (Fix the Navigation)**
- **What:** Fold the `/dashboard` route into a "Stats" tab on the main `/` Vault page.
- **Why:** Fulfills the cohesive "Three Views" intent from the UX plan and reduces top-level navigation clutter.
- **Effort:** M | **Risk:** Low.
- **Checklist:** Ensure tab navigation is horizontally scrollable on mobile to avoid layout breaking.

---

### Section 6 — Strongest Objection

**The architectural constraint (Static/No Backend) fundamentally contradicts your required viral growth mechanics.**

You have mandated a 100% client-side, statically exported app with no accounts. Simultaneously, the product strategy heavily relies on "Shareable challenge links" (`staffml.com/challenge/abc123`) to create a viral acquisition loop, and you want to show users "the challenger's score vs. their own" after they finish.

**You cannot do this securely or cleanly without a backend.** If you encode the challenger's score and question IDs into a Base64 hash in the URL, the URL will become massively bloated, and any engineer can decode it to see the answers/scores before taking the test, defeating the "challenge" aspect. If you cryptographically sign it, you must expose the secret key to the client, ruining the cryptography.

You must make a choice: either abandon the viral "challenge link" mechanic and accept slower organic growth, or relax the "no backend" constraint and introduce a lightweight serverless key-value store (like Cloudflare KV or Vercel KV) strictly for storing ephemeral, anonymous challenge states. You are currently trying to ship a multiplayer mechanic on a single-player architecture.
