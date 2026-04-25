# Socratiq — AI Learning Widget

Socratiq is an AI-powered learning widget that injects itself into any static HTML page (Quarto books, textbooks, documentation sites) via a single `<script>` tag. It runs entirely in a Shadow DOM so it never conflicts with the host page's styles or scripts.

## What it does

- **AI Chat** — Ask questions about the page content. The widget reads the surrounding text as context and streams answers using configurable LLM providers (Cloudflare AI Gateway, Gemini, Groq).
- **Quiz generation** — Generates contextual quizzes from selected text or entire sections. Supports multiple-choice, short-answer, and spaced repetition (flashcard) modes.
- **Highlight & ask** — Select any text on the page to instantly ask a follow-up question or generate a quiz about that passage.
- **Knowledge graph** — Visualizes relationships between concepts on the current page.
- **Progress tracking** — Tracks quiz scores and chapter visits in a local IndexedDB database, with a spaced repetition scheduler for revisiting weak areas.
- **Markdown rendering** — Streams and renders full markdown (including KaTeX math, Mermaid diagrams, syntax-highlighted code) in the chat panel.
- **Chat history** — Saves and restores conversations per page, with export and share functionality.
- **Onboarding & settings** — First-run onboarding flow, theme support (light/dark auto-detection), and a settings panel for choosing LLM model and difficulty level.

The widget is gated behind a `socratiq=true` cookie or URL parameter so it can be embedded in production sites and toggled per-user without affecting other visitors.

## 🚀 Quick Start

### Prerequisites
- Node.js (v18+ recommended)
- npm

### Installation

```bash
git clone https://github.com/harvard-edge/cs249r_book.git
cd cs249r_book/socratiq
npm install
```

### Development

```bash
npm run dev
```

This starts the Vite dev server on `http://localhost:4175` (auto-increments if the port is occupied), opens the encryption textbook test page at `/test_website/encryption_textbook/`, injects the widget with HMR enabled, and sets the `socratiq=true` cookie automatically. Changes in `src_shadow/` are reflected instantly in the browser.

> Uses `vite.config.dev.mjs`, which intercepts requests to test pages, strips any production script tags, and replaces them with the local dev entry point (`/js/index.js`) plus the Vite HMR client.

## 🛠️ Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | **Primary dev command.** HMR dev server on port `4175` with widget injected into the test page. |
| `npm run dev:vite` | Alias for `dev`. |
| `npm run build:vite` | Production build → `dist_vite/bundle.js`. Also copies output to Quarto book destinations. |
| `npm run build:quarto` | Alias for `build:vite`. |
| `npm run preview:vite` | Serves the production build locally for final verification. |

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+/` / `⌘+/` | Toggle the widget panel open/closed |
| `Ctrl+Enter` / `⌘+Enter` | Save a flashcard in the spaced repetition modal |

The widget automatically detects the user's OS and shows the correct modifier key (`⌘` on Mac, `Ctrl` elsewhere) in all UI hints.

## 📂 Project Structure

- **`src_shadow/`** — All widget source code.
  - `js/index.js` — Entry point. Bootstraps the widget, attaches Shadow DOM, wires all components.
  - `js/components/` — UI components: quiz, chat, highlight menu, settings, spaced repetition, knowledge graph, onboarding, etc.
  - `js/libs/` — Agents (LLM streaming), utilities, IndexedDB helpers, fuzzy matching.
  - `configs/` — LLM provider configs, prompt templates, system prompts.
  - `indexHtml.js` / `cssStyles.js` — Widget HTML and CSS injected into the Shadow DOM at runtime.

- **`test_website/`** — Static HTML pages used for local development and testing.
  - `encryption_textbook/` — Primary test page (cryptography textbook).
  - `mlsys_book_removed_most/` — Secondary test page (ML systems book excerpt).

- **`dist_vite/`** — Production build output (`bundle.js` + assets).

- **`cloudflare/`** — Backend infrastructure.
  - `proxy-worker/` — Cloudflare Worker that proxies LLM API requests (keeps API keys server-side).
  - `sync-server/` — Collaborative sync and session logic.

- **`vite.config.dev.mjs`** — Dev server config with injection middleware and HMR.
- **`vite.config.prod.mjs`** — Production build config (single-file bundle, copies to Quarto destinations).

## Embedding in a page

Add to the bottom of any HTML page:

```html
<script type="module" src="path/to/bundle.js"></script>
```

Then navigate to the page with `?socratiq=true` to activate the widget (sets a persistent cookie). Subsequent visits load the widget automatically.

## ⚙️ Configuration

All configuration is centralised in two files — **you should never need to touch anything else** to deploy the widget on a new site.

### `src_shadow/configs/env_configs.js` — Backend & Runtime

This is the **primary config file**. Change these before deploying:

```js
// Switch between local wrangler dev workers and production Cloudflare Workers
export const USE_LOCAL_WORKERS = false;   // true → localhost:8787/8788

// The topic the widget scopes its answers to (used in the system prompt)
export const MAIN_TOPIC = 'MLSysBook.AI: Principles and Practices of Machine Learning Systems Engineering';

// AI provider model selection — change any model here, takes effect everywhere
export const PROVIDER_MODELS = {
  GROQ:        { model: 'llama-3.1-8b-instant',  stream: true  },
  GEMINI:      { model: 'gemini-2.5-flash',       stream: true  },
  // ... see file for full list
};

// Max characters of page text sent per LLM call
export const SIZE_LIMIT_LLM_CALL = 6000;
```

| Variable | What it controls |
|---|---|
| `USE_LOCAL_WORKERS` | `true` → hit `localhost:8787/8788` (wrangler dev); `false` → production workers |
| `MAIN_TOPIC` | Injected into every system prompt to keep answers on-topic |
| `PROVIDER_MODELS` | Model name + streaming flag per provider, tried in order on fallback |
| `SIZE_LIMIT_LLM_CALL` | Text truncation limit before sending to AI (tokens ~= chars/4) |
| `WORKER_URL_AI` / `WORKER_URL_AI_STREAM` | Resolved automatically from `USE_LOCAL_WORKERS` — don't set manually |

### `src_shadow/configs/client.config.js` — Prompts & UX

Contains all LLM **prompt templates** and UI copy. Edit these to customise what the AI says:

| Export | What it controls |
|---|---|
| `quiz_prompt` | Section quiz prompt template (3 questions) |
| `QUIZ_SUMMATIVE_PROMPT` | Cumulative quiz prompt template (10 questions) |
| `SYSTEM_PROMPT_ORIG` | Base system prompt (uses `MAIN_TOPIC` from env_configs) |
| `DIFFICULTY_LEVELS` | Array of 4 difficulty prompts (Beginner → Bloom's Taxonomy) |
| `PROGRESS_REPORT_PROMPT` | Progress report analysis prompt |
| `TEXT_EXTRACTION_CONFIG` | `MAX_TOKENS`, DOM selectors for content vs nav extraction |
| `getConfigs(type)` | Factory — returns a config object for `'quiz'`, `'explain'`, `'query'`, `'summative'`, `'progress_report'` |

### Local worker development

```bash
# 1. Start the Cloudflare Worker locally (in cloudflare/proxy-worker/)
wrangler dev

# 2. Set USE_LOCAL_WORKERS = true in env_configs.js

# 3. Start the widget dev server
npm run dev
```

Revert `USE_LOCAL_WORKERS = false` before building for production.

## 🎨 Theme Detection

The widget auto-detects light/dark mode using the following priority order:

1. `window.SocratiqWidgetTheme = 'dark' | 'light'` — explicit host-page override (highest priority)
2. Quarto color scheme toggle button (`.quarto-color-scheme-toggle`)
3. `document.body.classList` contains `dark-mode`
4. **Page background luminance** — the actual rendered background color (takes priority over system preference)
5. System `prefers-color-scheme: dark` — only used when background luminance is indeterminate
6. Body text luminance as a final heuristic

This means the widget follows the **page's visual theme**, not the OS setting — a light-background page always gets a light widget even if the user's OS is in dark mode.

## 🔌 API Architecture

All LLM calls are routed through a Cloudflare Worker proxy (`cloudflare/proxy-worker/`) which holds the API keys. The widget itself never has access to API keys. Providers are tried in order: **Groq → Gemini → Cerebras → SambaNova → Mistral → OpenRouter → HuggingFace → Awan**.

For quiz generation the proxy uses `llama-3.1-8b-instant` (Groq) by default, falling back to `gemini-2.5-flash`. Chat/query uses streaming via `proxy-worker-streaming.mlsysbook.workers.dev`.

## Notes

- COOP/COEP headers (`Cross-Origin-Embedder-Policy: require-corp`, `Cross-Origin-Opener-Policy: same-origin`) are required for OPFS-backed IndexedDB and SharedArrayBuffer features. The dev server sets these automatically.
- The production build uses `vite-plugin-singlefile` to inline all assets into a single `bundle.js` with no external dependencies.
- To force a specific theme on a page, set `window.SocratiqWidgetTheme = 'light'` (or `'dark'`) before the widget script loads.
