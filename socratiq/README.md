# SocratiQ

SocratiQ is the book's in-page tutor and study workspace. The shared reader is
included in the HTML configuration for all four volumes. The larger study bundle
loads only when a reader opens **Study tools**.

## Reading and studying

The reader supports selected text → **Explain this**, follow-up questions,
**Simplify**, **Show an example**, and **Check my understanding**. On touch
screens, **Choose a paragraph** previews a passage before a request is sent.
**Recent passages** retains up to five discussions and drafts on the current
page. **Start over** clears only the current discussion. **Copy answer** includes
source provenance; the source link returns to the passage in the same tab.

**Study tools** opens the existing workspace for quizzes, flashcards, saved
conversations, progress, the knowledge graph, settings, and exports. Returning to
the reader preserves its current discussion. The workspace mounts once per page;
opening it does not enable a persistent activation cookie.

The first deliberate reader opening offers a skippable tour; **Replay tour**
reopens it. **Learning preferences** saves the explanation level and optional
background/goal in this browser, shared with study settings. Saving preferences
sends nothing; subsequent AI requests include them. **Clear preferences** resets
the intermediate level and clears both text fields. No account or device sync is
provided.

AI-generated explanations and questions can be wrong. A source link identifies
the supplied passage; it does not establish that every generated claim is
supported. Check-understanding feedback is not a graded assessment or evidence
of mastery. The graph's extracted relationships are not authored prerequisite
relationships.

## Data and service boundaries

- Reader discussions live in page memory and disappear on reload or navigation.
- Study records use browser-local IndexedDB and local storage. They are not
  synchronized to an account or another device. Browser data deletion also
  removes these records; use the workspace's exports to retain study material.
- Progress summarizes recorded results and cumulative-quiz outcomes. An unknown
  number of available chapter quizzes is not replaced with an invented target.
- Quiz identities include their source page so matching titles in different
  volumes do not share scores. Old records without an unambiguous source remain
  unassigned rather than being attributed to an arbitrary volume.
- Schema repairs preserve existing stores and records. Saved chat HTML is
  sanitized on restoration; imports are validated before writes.
- Selecting or previewing text does not send it. Asking for AI help sends the
  selected passage, local section context, question, and bounded conversation
  history to the configured proxy. Study AI actions likewise send their requested
  source content. There is no reader usage or learning analytics pipeline.
- Cloud quiz submission and collaborative synchronization are unavailable. Local
  storage does not imply that those services have been restored.

The production chat gateway is `https://mlsysbook.ai/api/socratiq`.
Its source and operating contract are documented in
[cloudflare/](cloudflare/README.md). Reader and study Groq defaults use the shared
configuration in `src_shared/provider-config.js`; provider credentials stay on
the Worker. The gateway supports the explicitly allowed Groq models and rejects
unsupported providers without choosing a fallback. The study workspace's custom
API option is an explicit user override, separate from the public gateway.

The Worker is activated with a dedicated encrypted Groq credential, after
verification of Groq and Cloudflare Free plans. Activation and local transport
tests do not establish live model quality or available quota. Local tests use
constructed upstream responses. The deployment health check establishes dispatch
only; live inference controls and browser checks establish different parts of the
request path. See the gateway README for limits, disable/rollback procedures,
provider privacy, and the manual deployment workflow. No API key belongs in a
book script or build.

## Build and test

```bash
cd socratiq
npm ci
npx playwright install chromium webkit
npm run build:reader
npm run build:vite
npm run test:local
npm run test:gateway
npm run test:reader:browser
cd ..
python -m pytest -q binder/tests/test_socratiq_html_integration.py --confcutdir=binder/tests
```

The reader build writes `books/shared/assets/scripts/socratiq/reader.js`. The
study build writes `books/shared/assets/scripts/socratiq/study.js` and the legacy
artifact at `binder/tools/scripts/socratiQ/bundle.js`. Commit generated artifacts
with their source changes. Both builds embed WOFF2 math fonts without duplicate
TTF/WOFF copies. Quarto stages the shared directory in every volume;
the study loader resolves its URL beside the deployed reader script, preserving
the volume prefix. The `socratiq-validate.yml` workflow checks rebuild drift and
runs local, browser, and Quarto integration tests.

Tests using constructed responses exercise behavior, not model quality. Storage
tests use isolated browser data. Chromium and WebKit viewport tests cover phone
and tablet layouts; they do not replace physical-device checks of native
selection handles, software keyboards, or assistive technology.

Optional live checks send public or author-constructed passages to the configured
service and save raw evidence. They are excluded from CI:

```bash
node tests/reader-live-smoke.mjs /tmp/socratiq-live-UNIQUE-RUN
node tests/reader-quality-probes.mjs /tmp/socratiq-probes-UNIQUE-RUN
```

Review the saved answers against the source and the declared criteria. Do not
interpret these small, unscored probes as learning-effectiveness measurements.

## Code map

| Location | Responsibility |
| --- | --- |
| `src_reader/index.js` | Book entry point and configuration |
| `src_reader/context.js` | Visible passage capture, math source, bounded context, source return |
| `src_reader/picker.js` | Explicit paragraph picking |
| `src_reader/ui.js` | Reader interaction and request lifecycle |
| `src_reader/passage-sessions.js` | Bounded discussion snapshots and restoration |
| `src_reader/environment.js` | Theme, viewport geometry, and listener cleanup |
| `src_reader/study.js` | Lazy study loading and recovery |
| `src_reader/transport.mjs` | Request validation, streaming, cancellation, and service errors |
| `src_reader/render.js`, `styles.js` | Sanitized Markdown/math and responsive presentation |
| `src_shadow/js/index.js` | Study bootstrap and embedded open/close bridge |
| `src_shadow/js/components/` | Quizzes, flashcards, progress, history, graph, settings |
| `src_shadow/js/libs/` | Provider adapters, storage, and shared utilities |
| `src_shadow/configs/` | Study provider settings and prompts |

For standalone study development, `npm run dev` starts the Vite fixture with
legacy activation enabled. The book integration instead uses an explicit
embedded bridge and keeps the reader as its single entry point.

## Release and rollback

Publish through the existing book release process after review. Neither a local
merge nor a successful build updates the public website. Confirm live inference
and operational ownership before describing a release as restored. Removing the
reader script includes from `books/config/_quarto-html-vol{1,2,3,4}.yml` disables
the companion without changing book content or deleting stored study records.
