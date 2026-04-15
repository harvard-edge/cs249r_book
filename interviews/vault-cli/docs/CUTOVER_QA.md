# Cutover-Day QA Checklist

> **When to use**: Phase 4 cutover (static `corpus.json` → Worker API + D1).
> **Who runs**: release operator, sequentially, alone — not in parallel with other site work.
> **Expands**: ARCHITECTURE.md §19.4.
> **Rehearsal**: this checklist runs end-to-end on **staging** as a dry run before production cutover.

---

## 0. Pre-cutover gate checks (must all be GREEN before starting)

- [ ] `vault verify <release>` on the release to be deployed → exit 0.
- [ ] `vault smoke-test --env staging --samples 50` → 0 divergences.
- [ ] All E2E Playwright tests green on staging against staging D1.
- [ ] Lighthouse CI gates green on staging:
  - [ ] practice/page.js transferred ≤ 300 KB gz.
  - [ ] gauntlet/page.js ≤ 250 KB gz.
  - [ ] landing/page.js ≤ 200 KB gz.
  - [ ] FCP (95th pct, 4G) ≤ 1.2s.
  - [ ] TTI (95th pct, 4G) ≤ 2.5s.
  - [ ] Repeat-visit TTI ≤ 800ms.
  - [ ] API round-trip p99 ≤ 250ms (question detail).
- [ ] FTS5 load-test artifacts from Phase 3 still valid (re-run if >30 days old).
- [ ] R2 pre-deploy snapshot of current production D1 exists and is restore-tested.
- [ ] Rollback drill executed on staging within last 7 days (see §4).
- [ ] Go/no-go reviewed with user. **GO** recorded in an operator log.

If ANY item is red, **do not proceed**. Fix the underlying issue, re-run the gate.

---

## 1. Ship the release

- [ ] `vault ship <release> --env staging --canary-percent 10` → green through all canary stages.
- [ ] `vault smoke-test --env staging --samples 50` post-ship → 0 divergences.
- [ ] Wait 15 min, observe staging dashboards (Cloudflare Analytics + Grafana).
- [ ] All transport SLIs green (5xx <1%, p99 <500ms).
- [ ] All data-plane SLIs green (row-count parity, content-hash sampling, FTS5 parity, schema_fingerprint).
- [ ] `vault ship <release> --env production --canary-percent 10`.
  - [ ] `.ship-journal.json` written; tail the journal.
  - [ ] D1 deploy leg: complete.
  - [ ] Next.js deploy leg: complete.
  - [ ] Soak 15 min OR ≥100 sessions at 10%.
  - [ ] Advance to 50%. Soak 15 min OR ≥100 sessions.
  - [ ] Advance to 100%. Soak 15 min OR ≥100 sessions.
  - [ ] Paper-tag push leg: complete (last).
  - [ ] `point_of_no_return: true` in journal.
- [ ] `vault smoke-test --env production --samples 100` → 0 divergences.

---

## 2. User-facing flows (manual QA on production)

Operator runs each flow in a clean browser window (no extensions, no prior localStorage). Check the box if the flow completes without error AND the expected outcome is visible.

### 2.1 Home / landing

- [ ] `https://staffml.mlsysbook.ai/` loads.
- [ ] Total question count matches `vault stats --release <release>` exact integer.
- [ ] No request in Network tab for `corpus.json` (the 19 MB static file must not be fetched).
- [ ] `practice/page.js` transferred size ≤ 300 KB gzipped (verify in DevTools → Network).
- [ ] FCP ≤ 1.2s (check via Lighthouse).
- [ ] `X-Vault-Release` header present on `/manifest` response; value = current release.

### 2.2 Practice

- [ ] Navigate to `/practice`.
- [ ] Filter by track → results update.
- [ ] Filter by level → results update.
- [ ] Filter by zone → results update.
- [ ] Combination filter (track + level + zone) returns expected subset.
- [ ] Reveal answer on a question → solution renders (Markdown + KaTeX if applicable).
- [ ] Navigate a chained question → "Part N of M" badge visible BEFORE reveal.
- [ ] Click chain-badge link → chain sibling list opens.
- [ ] AskInterviewer tutor → ask a question → response arrives within 10s, no errors.
- [ ] Reveal → AskInterviewer switches to study mode; tutor knows canonical answer.

### 2.3 Gauntlet

- [ ] Start a gauntlet session with filter → session launches.
- [ ] Complete N questions (at least 3, mix of right and wrong) → scores tracked.
- [ ] View post-mortem → per-question feedback shown.
- [ ] Navigate back to landing → session marked complete in localStorage.

### 2.4 Progress

- [ ] `/progress` page loads.
- [ ] Attempts from §2.3 persist.
- [ ] Due-count correct against the test-interval logic.
- [ ] No console errors.

### 2.5 About

- [ ] `/about` loads.
- [ ] "Read the paper" call-out visible **above the fold** (no scrolling required on a 1920×1080 viewport).
- [ ] BibTeX snippet renders.
- [ ] DOI (if registered) clickable.
- [ ] Release ID + release_hash visible in footer for reproducibility.
- [ ] Contributor list renders authors from current release's `authors:` fields.

### 2.6 Command palette / search

- [ ] `⌘K` (Mac) / `Ctrl+K` (Windows/Linux) opens modal from any page.
- [ ] Input placeholder: "Search N questions by title, scenario, or solution."
- [ ] Type a term → 200ms debounce → results appear with snippet highlights.
- [ ] Up/Down arrow navigates results.
- [ ] Enter opens question; `⌘Enter` opens in new tab.
- [ ] Escape closes modal.
- [ ] Empty query state: helpful message + browse-by-topic link.
- [ ] No-results state: "no results for '...'" message + clear-filters CTA.
- [ ] Mobile (iPhone 15 viewport, 393×852): full-screen modal, no iOS zoom on input focus, touch targets ≥ 44px.

### 2.7 Chain UX

- [ ] On a chained question (e.g., part 2 of 4), pre-reveal chain badge is visible.
- [ ] Badge text: "Part 2 of 4 — <chain name>".
- [ ] Badge click → sibling list drawer; shows all chain members with their status (attempted / unattempted).
- [ ] Analytics events fired: `chain_badge_shown`, `chain_badge_clicked` (check Cloudflare Analytics real-time).

### 2.8 Offline resilience

- [ ] With the site loaded and at least 5 questions visited:
  - [ ] Open DevTools → Application → Service Workers → verify `sw.js` registered, controlling.
  - [ ] Network → check "Offline" → reload page.
  - [ ] Site shell renders.
  - [ ] Previously-visited question detail pages load from SW cache.
  - [ ] "Serving from cache" indicator visible.
- [ ] Toggle back online → SW revalidates manifest → indicator disappears.

---

## 3. Network + bundle verification

- [ ] **No `corpus.json` fetch** anywhere in the user journey (Network tab filter `corpus`).
- [ ] **Request to `/manifest` returns < 5 KB.**
- [ ] **Request to `/questions/<id>` returns < 10 KB and has correct `ETag` format** `"<release>:<resource>:<content_hash>"`.
- [ ] **304 behavior**: hard-refresh a just-visited question → browser sends `If-None-Match` → Worker returns 304.
- [ ] **Cache API hit on warm**: refresh → Network tab shows `from disk cache` or `from service worker` for manifest/taxonomy.
- [ ] **No console errors** across all flows above.
- [ ] **No CSP violations** (DevTools → Console filter `Content-Security-Policy`).

---

## 4. Rollback drill (executed on staging before production cutover)

Rehearsal, not optional. Log steps + timings in the operator log.

- [ ] Staging site warm with an active service worker (user has visited ≥10 questions).
- [ ] Set `NEXT_PUBLIC_VAULT_FALLBACK=static` in the site environment.
- [ ] Redeploy site (one command).
- [ ] **Timer start.**
- [ ] User reloads tab.
- [ ] Service worker evicts stale release-keyed entries.
- [ ] Site loads from static inlined corpus + manifest.
- [ ] Question detail pages render.
- [ ] No console errors.
- [ ] AskInterviewer: if worker is still up, tutor works; if down, graceful "tutor temporarily unavailable" indicator.
- [ ] **Timer stop.** Target: rollback complete + user-visible within 10 minutes. Record actual.
- [ ] Restore `NEXT_PUBLIC_VAULT_FALLBACK` unset; redeploy; verify Worker-backed state resumes.

If ANY step is red, do NOT proceed to production cutover. File an issue and fix the rollback path first.

---

## 5. Post-cutover watch (first 48 hours on production)

- [ ] Dashboard watch scheduled: 30 min, 2h, 6h, 12h, 24h, 48h checkpoints.
- [ ] At each checkpoint:
  - [ ] Transport SLIs green.
  - [ ] All data-plane SLIs green (row-count, content-hash sample, FTS5, schema_fingerprint, release-id propagation).
  - [ ] Search latency p99 within budget.
  - [ ] Error-tracker: no new Sentry clusters.
  - [ ] Cost ledger: D1 row-reads tracking within 2× forecast.
- [ ] At 48h: post-cutover review with user; decide on Phase 5 kickoff.

---

## 6. Rollback trigger — when to abort

If any of the following occur within the first 48h, trigger rollback via `NEXT_PUBLIC_VAULT_FALLBACK=static`:

- 5xx rate > 5% sustained for > 2 min.
- p99 latency > 1 s sustained for > 5 min.
- Any data-plane SLI red for > 10 min without explanation.
- Schema-fingerprint mismatch that persists past a single POP cold-start cycle.
- User-visible content corruption (question renders differently from staging).
- Cost forecast exceeded by > 3× over any 1-hour window.

Rollback does NOT require another user approval — this checklist pre-authorizes the operator to roll back on trigger conditions. Forward-fix decisions (vs rollback) are user-approval-gated.

---

## 7. Post-cutover sign-off

After 48h clean watch:

- [ ] Final `vault smoke-test --env production --samples 100` green.
- [ ] Operator log committed to `interviews/vault/releases/<version>/cutover-log.md`.
- [ ] Retention policy noted: keep `corpus.json` in site bundle until first schema-major bump OR 2 releases post-cutover, whichever is later (ARCHITECTURE.md §7.1).
- [ ] Phase 4 marked complete in the project tracker.
- [ ] Post-mortem session scheduled if anything from §6 triggered during watch window.

---

**End of cutover checklist.** File at `interviews/vault-cli/docs/CUTOVER_QA.md` — keep in sync with ARCHITECTURE.md and TESTING.md.
