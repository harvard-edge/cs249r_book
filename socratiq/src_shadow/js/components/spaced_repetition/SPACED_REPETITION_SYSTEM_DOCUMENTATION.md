# Spaced Repetition System – Technical Documentation (2025-09)

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Storage Architecture](#storage-architecture)
4. [Data Model](#data-model)
5. [Initialization Flow](#initialization-flow)
6. [Primary Handlers](#primary-handlers)
7. [Data & Event Flow](#data--event-flow)
8. [AI Integration](#ai-integration)
9. [Visualization & Analytics](#visualization--analytics)
10. [Known Limitations & Follow-ups](#known-limitations--follow-ups)
11. [File Map](#file-map)

---

## System Overview

The spaced repetition module lives inside the SocratiQ widget (`src_shadow/js/components/spaced_repetition/`). It supplies a modal for reviewing and managing flashcards, supports AI-assisted card creation, and persists user data in the browser.

### Current Capabilities
- **Deck management**: Users create and curate decks through `DeckCreationHandler` and the modal UI. Decks are grouped by chapter IDs sourced from the host page where available.
- **Storage**: The live build uses IndexedDB via `SpacedRepetitionIndexDBHandler`, backed by the shared `SocratiqDB` instance defined in `src_shadow/js/libs/utils/indexDb.js`. The base storage class still reads/writes to `localStorage` keys for compatibility.
- **AI-generated cards**: Selected text can be sent to DuckAI through the Cloudflare proxy to create flashcards.
- **Analytics**: The widget renders basic stats (Chart.js) and exploratory visualizations (D3) from the in-memory deck data.

### Explicitly Out of Scope Right Now
- **SQLite persistence**: `SpacedRepetitionSQLiteHandler` exists but is *not* instantiated by the modal. Any SQLite worker assets are presently unused.
- **Automated chapter deck creation**: Breadcrumb-based detection utilities remain, but automatic deck creation is disabled during initialization (`ensureCurrentChapterExists` is commented out in `storage-handler.js`).
- **Real-time collaboration / WebRTC**: Sharing is limited to mailto-based export; no active WebRTC logic ships with this build.

---

## Core Architecture

- **`spaced-repetition-modal-handler.js`** instantiates the sub-handlers once the modal DOM is mounted. It wires the modal to global events, manages the Ink MDE editors, and serves as the single source of truth for currently selected chapter/deck state.
- **`spaced-repetition.js`** defines the `FlashCard` class along with the scheduling logic that updates intervals, ease factor, and review history when the user rates a card.
- **UI handlers** (`ui-handler.js`, `floating-controls-handler.js`, `interaction-button-handler.js`, etc.) encapsulate DOM updates, interaction surfaces, and auxiliary tooling (import/export, tooltips).

The modal sets `this.storageHandler = new SpacedRepetitionIndexDBHandler(this)` and then calls `initializeDatabase()` followed by `loadFromLocalStorage(true)` (IndexedDB handler reuses method names from the base class for API compatibility).

---

## Storage Architecture

### Base Class – `storage-handler.js`
- **Responsibilities**: maintain `inMemoryData`, offer debounced save queues, and provide shared helpers (`getChapterTitle`, `getAllChapters`, `getCardsByTag`, etc.).
- **LocalStorage footprint**: legacy keys `chapter_card_sets`, `chapter_progress_data`, `current_chapter`, and `sr-app-first-visit` remain in use for fallback and UI bootstrapping.
- **Debounced saves**: `queueSave()` schedules writes through `processQueue()`, emitting `sr-save-started` and `sr-save-card-completed` events for UI feedback.

### Active Handler – `spaced-repetition-indexdb.js`
- **Database**: Acquires a shared IndexedDB connection from `getDBInstance()` (the SocratiqDB wrapper). Required stores are defined in `configs/db_configs_one.js` (chapters, cards, card_tags, review_history, current_chapter).
- **Initialization**: `initializeDatabase()` logs “Initializing IndexedDB database...” and sets `useFallback = true` only on failure. Most public methods mirror the base class API but operate on IndexedDB transactions.
- **Persistence logic**:
  - `saveToLocalStorage(fullChapterSets)` writes chapters/cards/tags directly to IndexedDB, clearing stale records and updating `modal.chapterSets`.
  - `switchActiveChapter()` marks the active chapter in the `chapters` store and hydrates `inMemoryData.currentDeck`.
  - Tag queries use the `card_tags` index, while review history persists in its own store for analytics.
- **Fallback behaviour**: When IndexedDB operations fail *and* `useFallback` is set, methods delegate to the base localStorage implementations. No automatic SQLite promotion occurs.

### Experimental Handler – `storage-handler-sqlite.js`
- Relies on `SQLocal` to open `spaced_repetition.sqlite3` and set up relational tables. The class mirrors the IndexedDB API but expects successful `initializeDatabase()` calls before use.
- Because the modal never instantiates this class, SQLite schema creation and migrations do not run in the current build. Any observed SQLite console errors typically stem from unused worker bundles being loaded in the page.

---

## Data Model

- **FlashCard (`spaced-repetition.js`)**: tracks `question`, `answer`, `repetitions`, `easeFactor`, `interval`, `nextReviewDate`, `reviewHistory`, `id`, and `lastReviewQuality`.
- **Chapters (`modal.chapterSets`)**: stored as `Map<number, FlashCard[]>` in memory. IndexedDB maintains equivalent records with chapter metadata and card associations.
- **Tags**: persisted per card in the `card_tags` store and aggregated into `inMemoryData.allTags` for filtering.
- **Review history**: aggregated as `{ [date: string]: count }` within IndexedDB for heatmap visualizations.

---

## Initialization Flow

1. `SpacedRepetitionModal.initialize(shadowRoot)` creates the singleton modal instance and registers UI triggers.
2. Constructor builds the modal DOM (`createModal()`), then instantiates handlers inside the resolved promise chain.
3. `SpacedRepetitionIndexDBHandler.initializeDatabase()` obtains the SocratiqDB connection and verifies persistence. Failures set `useFallback = true`.
4. `loadFromLocalStorage(true)` (inherited signature) loads chapter data. In the IndexedDB handler this step hydrates memory using `initializeFromStorage()` and `getAllChapters()`.
5. `SpacedRepetitionInitializationHandler` optionally seeds content for first-time users (logic currently gated by `this.fallbackStorage` checks).
6. UI handler renders decks, tags, stats, and attaches floating controls.

---

## Primary Handlers

- **`SpacedRepetitionEventHandler`**: wires modal buttons, keyboard shortcuts, and chapter selectors. Methods call back into modal functions such as `setCurrentChapter()` and `saveNewCard()`.
- **`SpacedRepetitionInteractionHandler`**: manages import/export/copy/share actions. Exports rely on `storageHandler.exportToJSON()`; shares use a `mailto:` link—no WebRTC channel exists.
- **`DeckCreationHandler`**: renders the deck creation modal inside the main shadow root, validates names against `modal.chapterSets`, and notifies the UI.
- **`FloatingControlsHandler`** and **`SpacedRepetitionUIHandler`**: keep sidebar counts, progress bars, and auxiliary buttons in sync with state changes.

---

## Data & Event Flow

- **Card lifecycle**
  1. `saveCard` button (or keyboard shortcut) gathers editor content.
  2. Modal calls `storageHandler.updateCard()` (base implementation pushes into `inMemoryData.currentDeck` and triggers a save).
  3. IndexedDB handler begins a `cards` transaction, upserts the record, and stores tags.
  4. UI handlers refresh deck lists and progress indicators.

- **Chapter switching**
  1. Sidebar click dispatches to `SpacedRepetitionModal.setCurrentChapter()`.
  2. Modal invokes `storageHandler.switchActiveChapter(chapterNumber)`.
  3. IndexedDB handler updates `chapters.is_current` and reloads associated cards into memory.
  4. Modal updates `this.currentChapter`, `this.flashcards`, and re-renders.

- **Persistence events**
  - `sr-save-started` and `sr-save-card-completed` fire around IndexedDB writes, enabling notification banners.
  - `sr-data-updated` is emitted after imports to instruct the UI to re-query data.

---

## AI Integration

- Triggered via highlight menu (`src_shadow/js/components/highlight_menu/send_text_highlight.js`).
- `modal.storageHandler.addFlashcardsFromText(selectedText)` sends the text to `DuckAI` through `libs/agents/duck-ai-cloudflare.js` and expects a JSON array of card candidates.
- Imported cards pass through `jsonrepair` for resilience and are inserted into the current deck using the same IndexedDB persistence flow.
- Provider fallback logic (GROQ → Gemini → Cerebras) lives inside the DuckAI agent; the spaced repetition layer simply retries on error and surfaces notifications.

---

## Visualization & Analytics

- **`spaced-repetition-visualization.js`**: Builds either a network graph (chapters to cards) or a calendar-style heatmap from review history using D3. Graph interactions dispatch `show-card` events that bubble through the shadow root.
- **`spaced-repetition-stats.js`**: Uses Chart.js to plot “cards mastered over time” and “review quality distribution.” Charts rebuild on demand when the stats panel opens.
- Both modules consume the in-memory datasets populated by the storage handler; no direct database queries occur from visualization code.

---

## Known Limitations & Follow-ups

- **SQLite reintegration**: To activate `SpacedRepetitionSQLiteHandler`, the modal must branch on capability detection (see commented `checkStorageSupport()` logic). Additional work is required to migrate existing IndexedDB/localStorage data.
- **Fallback clarity**: `this.fallbackStorage` is set to `true` even when IndexedDB initializes correctly. Audit whether UI messaging should change before re-enabling SQLite selection.
- **Empty chapter bootstrap**: Automatic deck creation is currently disabled. If we re-enable breadcrumb-driven initialization, ensure it coexists with manual deck creation.
- **Storage duplication**: Some operations still mirror data into localStorage (via base class methods). Long term we should consolidate persistence to a single backend or add migration scripts.

---

## File Map

```
src_shadow/js/components/spaced_repetition/
├── spaced-repetition-modal-handler.js     # Modal orchestrator
├── spaced-repetition.js                   # FlashCard model & scheduling
├── spaced-repetition-visualization.js    # D3 visualizations
├── spaced-repetition-stats.js            # Chart.js analytics
├── spaced-repetition-inner-search-modal.js
├── handlers/
│   ├── storage-handler.js                # Base storage abstraction
│   ├── spaced-repetition-indexdb.js      # Active IndexedDB handler
│   ├── storage-handler-sqlite.js         # Experimental SQLite handler (unused)
│   ├── interaction-button-handler.js     # Import/export/share controls
│   ├── event-handler.js                  # Modal event wiring
│   ├── ui-handler.js                     # Sidebar & progress rendering
│   ├── deck-creation-handler.js          # Deck creation modal
│   ├── floating-controls-handler.js
│   ├── initialization-handler.js
│   └── ui-notification-handler.js
├── utils/
│   ├── anki-converter.js                 # Import/export helpers
│   ├── sqlite-checker.js                 # Capability probe (currently unused)
│   └── chaptermap-utils.js               # Legacy chapter helpers
└── SPACED_REPETITION_SYSTEM_DOCUMENTATION.md
```

---

This document reflects the code base as of 2025-09 and should be updated alongside future storage or feature changes (e.g., enabling SQLite, adding collaboration, or reintroducing automatic deck creation).

---

## System Architecture & Current Issues

```mermaid
graph TB
    subgraph "Browser Environment"
        subgraph "Spaced Repetition System"
            Modal[SpacedRepetitionModal]
            EventHandler[SpacedRepetitionEventHandler]
            UIHandler[SpacedRepetitionUIHandler]
            InteractionHandler[SpacedRepetitionInteractionHandler]

            subgraph "Storage Layer"
                IndexDBHandler[SpacedRepetitionIndexDBHandler<br/>✅ ACTIVE]
                SQLiteHandler[SpacedRepetitionSQLiteHandler<br/>❌ DISABLED]
                BaseHandler[SpacedRepetitionStorageHandler<br/>Base Class]
            end

            subgraph "Data Flow"
                SocratiqDB[(SocratiqDB<br/>IndexedDB)]
                LocalStorage[(localStorage<br/>Fallback)]
            end
        end

        subgraph "External Dependencies"
            SQLiteWorker[sqlite3-worker1-bundler-friendly<br/>❌ FAILING]
            WASM[WebAssembly Module<br/>❌ FAILING]
        end
    end

    subgraph "Current Issues"
        Issue1[❌ SQLite Worker WASM Loading Fails<br/>'both async and sync fetching failed']
        Issue2[❌ Notification Handler Undefined<br/>'Cannot read properties of undefined']
        Issue3[❌ SQLite Module Exception<br/>'Aborted(both async and sync fetching failed)']
    end

    %% Current Flow (Working)
    Modal --> IndexDBHandler
    IndexDBHandler --> SocratiqDB
    IndexDBHandler --> BaseHandler

    %% Disabled Flow (Broken)
    Modal -.-> SQLiteHandler
    SQLiteHandler -.-> SQLiteWorker
    SQLiteWorker -.-> WASM

    %% Error Connections
    SQLiteWorker --> Issue1
    SQLiteWorker --> Issue3
    InteractionHandler --> Issue2

    %% Styling
    classDef active fill:#90EE90,stroke:#333,stroke-width:2px
    classDef disabled fill:#FFB6C1,stroke:#333,stroke-width:2px
    classDef error fill:#FF6B6B,stroke:#333,stroke-width:2px
    classDef storage fill:#87CEEB,stroke:#333,stroke-width:2px

    class IndexDBHandler,SocratiqDB active
    class SQLiteHandler,SQLiteWorker,WASM disabled
    class Issue1,Issue2,Issue3 error
    class LocalStorage,BaseHandler storage
```

### Current System Status

**✅ Working Components:**
- SpacedRepetitionModal initialization
- SpacedRepetitionIndexDBHandler (active storage)
- SocratiqDB IndexedDB connection
- Base storage handler functionality

**❌ Broken Components:**
- SQLite Worker WASM loading (`sqlite3-worker1-bundler-friendly-CbDNa4by.js`)
- SQLite module initialization
- Notification handler in interaction flow

**🔧 Root Causes:**
1. **WASM Loading Failure**: The SQLite worker's WebAssembly module fails to load due to HTTP status issues
2. **Missing Notification Handler**: The interaction handler expects a notification handler that isn't properly initialized
3. **SQLite Disabled**: The system is hardcoded to use IndexedDB, bypassing SQLite entirely

**📋 Recommended Fixes:**
1. Fix WASM loading issues in SQLite worker
2. Ensure notification handler is properly initialized before interaction handler
3. Implement proper error handling and fallback mechanisms
4. Consider enabling SQLite once WASM issues are resolved
