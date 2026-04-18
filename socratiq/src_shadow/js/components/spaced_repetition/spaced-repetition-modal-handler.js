import { SpacedRepetitionEventHandler } from "./handlers/event-handler";
import { SpacedRepetitionUIHandler } from "./handlers/ui-handler";
import { FlashcardVisualization } from "./spaced-repetition-visualization";
import { FlashCard } from "./spaced-repetition";
import { ink } from "ink-mde";
import { DuckAI } from "../../libs/agents/duck-ai-cloudflare";
import { FlashcardStats } from "./spaced-repetition-stats.js";
import { showPopover } from "../../libs/utils/utils.js";
import { SpacedRepetitionInnerSearchModal } from "./spaced-repetition-inner-search-modal.js";
import { SpacedRepetitionDragHandler } from "./handlers/drag-handler";
import { SpacedRepetitionInteractionHandler } from "./handlers/interaction-button-handler";
import { enableTooltip } from "../../components/tooltip/tooltip.js";
// import { SROnboardingHandler } from "../onboarding/sr-onboarding-handler.js";
import { FloatingControlsHandler } from "./handlers/floating-controls-handler";
import { SpacedRepetitionInitializationHandler } from "./handlers/initialization-handler.js";
import { DeckCreationHandler } from "./handlers/deck-creation-handler.js";
// import { checkStorageSupport } from "./utils/sqlite-checker.js";
import { getThemeManager } from "../../libs/utils/theme-manager.js";

export class SpacedRepetitionModal {
  static instance = null;

  static initialize(shadowRoot) {
    if (!SpacedRepetitionModal.instance) {
      SpacedRepetitionModal.instance = new SpacedRepetitionModal(shadowRoot);
      SpacedRepetitionModal.setupTrigger(shadowRoot);
      // Store modal instance on shadowRoot for easy access
      shadowRoot.modalInstance = SpacedRepetitionModal.instance;
    }
    return SpacedRepetitionModal.instance;
  }

  static setupTrigger(shadowRoot) {
    // Setup click handler in the shadow root for the spaced repetition button
    shadowRoot.addEventListener("click", (e) => {
      const spacedRepBtn = e.target.closest("#spaced-repetition-btn");
      const spacedRepBtn2 = e.target.closest("#spaced-repetition-btn2");
      const flashcardsBtn = e.target.closest("#flashcards-button");

      if (spacedRepBtn || spacedRepBtn2 || flashcardsBtn) {
        SpacedRepetitionModal.instance?.show();
      }
    });
  }

  async setupInitialization() {


    if (this.fallbackStorage) {
      return;
    }

    const initHandler = new SpacedRepetitionInitializationHandler(this);
    await initHandler.checkAndInitializeFirstTimeUser();

  }

  constructor(shadowRoot) {
    this.shadowRoot = shadowRoot;
    this.flashcards = [];
    this.currentChapter = null;
    this.chapterSets = new Map();
    this.stats = null;
    this.questionEditor = null;
    this.answerEditor = null;
    this.isFirstLoad = true;
    this.fallbackStorage = false;
    this.themeManager = getThemeManager(shadowRoot);
    // Create modal first
this.createModal().then(async () => {
  this.eventHandler = new SpacedRepetitionEventHandler(this);

  try {
    console.log('[DEBUG] 1. Attempting to initialize storage...');
    await this.initializeStorage();
    
    console.log('[DEBUG] 2. Storage handler assigned. Type:', this.storageHandler?.constructor.name);
    await this.storageHandler.initializeDatabase();
    
    console.log('[DEBUG] 3. Database initialized. Setting up interaction handler...');
    this.interactionHandler = new SpacedRepetitionInteractionHandler(
      this,
      this.storageHandler,
      this.shadowRoot
    );

    console.log('[DEBUG] 4. Core handlers initialized. Setting up UI handlers...');
    this.uiHandler = new SpacedRepetitionUIHandler(this);
    this.searchHandler = new SpacedRepetitionInnerSearchModal(this);
    this.deckCreationHandler = new DeckCreationHandler(this);

    console.log('[DEBUG] 5. UI handlers set up. Initializing editors...');
    this.initializeEditors();

    console.log('[DEBUG] 6. Loading data from storage...');
    let loadedData = await this.storageHandler.loadFromLocalStorage(true);

    console.log('[DEBUG] 7. Initial data loaded. Setting up first-time user...');
    await this.setupInitialization();

    console.log('[DEBUG] 8. Re-loading data after setup...');
    loadedData = await this.storageHandler.loadFromLocalStorage(true);

    this.currentChapter = loadedData.currentChapter;
    this.chapterSets = loadedData.chapterSets;
    this.flashcards = loadedData.flashcards;

    console.log('[DEBUG] 9. Data loaded into component state. Rendering UI...');
    this.renderChapters(true);
    this.showAllCards();
    this.uiHandler.updateProgress();
    this.renderTags();
    this.floatingControlsHandler = new FloatingControlsHandler(this);
    this.setupEventListeners();
    console.log('[DEBUG] 10. Initialization complete.');

  } catch (error) {
    console.error("❌ FATAL INITIALIZATION FAILURE:", error);
    // Optionally display an error message in the UI here
    return;
  }
});

    // Rest of constructor code...
    this.debouncedUpdate = this.debounce(() => {
      requestAnimationFrame(() => {
        this.updateUI();
      });
    }, 250);

    this.dragHandler = new SpacedRepetitionDragHandler(this);
    
    this.setupTooltip();
    this.setupVisualizationEvents();
    // this.onboardingHandler = new SROnboardingHandler(this, shadowRoot);
    // this.setupOnboarding();
  }

  async initializeStorage() {
    try {
      console.log("Using IndexedDB storage (SQLite disabled).");
      
      const { SpacedRepetitionIndexDBHandler } = await import("./handlers/spaced-repetition-indexdb.js");
      this.storageHandler = new SpacedRepetitionIndexDBHandler(this);
      this.fallbackStorage = true;
      
      // Initialize IndexedDB database
      await this.storageHandler.initializeDatabase();
      
      // Show notification
      this.storageHandler.notificationHandler?.updateNotification(
        null,
        "Using IndexedDB storage",
        "info"
      );

      // Listen for storage status events
      window.addEventListener('sr-storage-status', (event) => {
        const { type, message } = event.detail;
        
        if (type === 'fallback') {
          this.storageHandler.notificationHandler?.updateNotification(
            null,
            "Switched to alternative storage method",
            "info"
          );
        } else if (type === 'error') {
          console.error('Storage error:', message);
          this.storageHandler.notificationHandler?.updateNotification(
            null,
            "Storage error occurred. Some features may be limited.",
            "error"
          );
        }
      });

      console.log(`📦 Storage initialized: indexeddb (fallback: ${this.fallbackStorage})`);

    } catch (error) {
      console.error("❌ Failed to initialize storage:", error);
      throw new Error("All storage methods failed to initialize");
    }
  }



  async   createModal() {
    const container = document.createElement("div");
    
    // Get current theme
    const hostElement = this.shadowRoot.host;
    const currentTheme = hostElement?.getAttribute('data-socratiq-theme') || 'light';
    const isDark = currentTheme === 'dark';
    
    console.log('[MODAL DEBUG 01] Creating modal with theme:', {
      hostElement: hostElement,
      currentTheme: currentTheme,
      isDark: isDark,
      hostAttribute: hostElement?.getAttribute('data-socratiq-theme'),
      inputBg: isDark ? 'bg-zinc-700/50' : 'bg-white',
      inputBorder: isDark ? 'border-zinc-600' : 'border-gray-300'
    });
    
    // Theme-aware classes
    const themeClasses = {
      modalBg: isDark ? 'bg-zinc-800' : 'bg-white',
      textPrimary: isDark ? 'text-gray-100' : 'text-gray-900',
      textSecondary: isDark ? 'text-gray-300' : 'text-gray-600',
      textMuted: isDark ? 'text-gray-400' : 'text-gray-500',
      border: isDark ? 'border-zinc-700' : 'border-gray-200',
      bgSecondary: isDark ? 'bg-zinc-700/50' : 'bg-gray-50',
      bgHover: isDark ? 'hover:bg-zinc-600' : 'hover:bg-gray-200',
      bgButton: isDark ? 'bg-zinc-700' : 'bg-gray-100',
      bgButtonHover: isDark ? 'hover:bg-zinc-600' : 'hover:bg-gray-200',
      bgAccent: isDark ? 'bg-blue-900/30' : 'bg-blue-100',
      textAccent: isDark ? 'text-blue-400' : 'text-blue-700',
      bgAccentHover: isDark ? 'hover:bg-blue-800/40' : 'hover:bg-blue-200',
      inputBg: isDark ? 'bg-zinc-700/50' : 'bg-white',
      inputBorder: isDark ? 'border-zinc-600' : 'border-gray-300',
      tagCountBg: isDark ? 'bg-zinc-700' : 'bg-gray-100'
    };
    
    console.log('[MODAL DEBUG] Theme classes generated:', themeClasses);
    
    container.innerHTML = `
        <div id="spacedRepetitionModal" class="hidden fixed inset-0 bg-gray-600 bg-opacity-50 h-full w-full z-[9999]">
           
           <div  id="sr-modal-content"
           >

            <div 
                id="sr-modal-content_inner"
                class="relative top-10 mx-auto mb-6 p-5 w-[90%] max-w-4xl shadow-lg rounded-md ${themeClasses.modalBg}"
               style="max-height: calc(100vh - 80px); background-color: ${isDark ? '#0d1117' : '#ffffff'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important; display: flex; flex-direction: column;"
                >
                    <!-- Close button -->
      <button id="closeModal_sr" class="absolute top-4 right-4 flex items-center z-10" style="color: #9ca3af !important;">             
    <kbd class="mt-1 px-1 text-xs border">
        ESC
    </kbd>
</button>
                    <div class="grid grid-cols-4 mt-8 gap-4" style="flex: 1; min-height: 0; overflow: hidden;">
                        <!-- Sidebar -->
                        <div id="sr-sidebar" class="col-span-1 border-r overflow-y-auto ${themeClasses.border} pr-4" style="max-height: 100%; overflow-y: auto;">
                        <!-- Search input -->
                        <div class="relative mb-4">
                            <input type="text" 
                                id="sidebarSearch"
                                class="w-full pl-10 pr-4 py-2 text-sm border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500" 
                                style="background-color: ${isDark ? '#21262d' : '#ffffff'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important; border-color: ${isDark ? '#30363d' : '#d0d7de'} !important;" 
                                placeholder="Search all...">
                            <svg class="absolute left-3 top-2.5 h-4 w-4 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2">
                                <path stroke-linecap="round" stroke-linejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" stroke="#4169E1"></path>
                            </svg>

                        </div>

                        <!-- Chapters Dropdown -->
                        <div class="mb-6">
                            <button class="w-full flex items-center justify-between px-3 py-2 text-sm text-gray-500 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-zinc-800/50 rounded-md">
                                <div class="flex items-center gap-2">
                                    <svg class="w-4 h-4 transform transition-transform" data-dropdown="chapters" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                                    </svg>
                                    <span>Decks</span>
                                </div>
                                <span id="chapterCount" class="text-xs">${
                                  this.chapterSets.size
                                }</span>
                            </button>
                            <div id="chapterList" class="mt-1 space-y-1" style="max-height: 300px; overflow-y: auto;">
                                <!-- Chapters will be rendered here -->
                            </div>
                        </div>
                        
                        <!-- Tags Dropdown -->
                        <div>
                            <button class="w-full flex items-center justify-between px-3 py-2 text-sm text-gray-500 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-zinc-800/50 rounded-md">
                                <div class="flex items-center gap-2">
                                    <svg class="w-4 h-4 transform transition-transform" data-dropdown="tags" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="transform: rotate(180deg);">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                                    </svg>
                                    <span>Tags</span>
                                </div>
                                <span id="totalTagCount" class="text-xs">${this.getInitialTagCount()}</span>
                            </button>
                            <div id="tagList" class="mt-1 space-y-1" style="max-height: 200px; overflow-y: auto;">
                                <!-- Tags will be rendered here -->
                            </div>
                        </div>
                    </div>

                    
                        <!-- Updated Main Content Area -->
                        <div
                        id="sr-modal-content-container"
                        class="col-span-3 pl-4 overflow-y-auto"
                        style="max-height: 100%;">
                        <!-- Progress Bar -->
                        <div class="mb-4">
                            <div class="flex justify-between items-center mb-2">
                                <span id=progress-label class="text-sm font-medium">Progress</span>
                                <span id="progressText" class="text-sm ${themeClasses.textSecondary}" style="color: ${isDark ? '#e6edf3' : '#1f2328'} !important;">0/0 Cards Learned</span>
                            </div>
                            <div class="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                                <div id="progressBar" class="bg-blue-600 h-2.5 rounded-full" style="width: 0%"></div>
                            </div>
                        </div>

                        <!-- Control Buttons -->
                        <div class="flex justify-between mb-4"
                        id="sr-modal-content-container-controls"
                        >
                            <div id="sr-controls-container" class="space-x-2">
                                <button id="showStats" class="px-3 py-2 text-sm ${themeClasses.bgButton} ${themeClasses.bgButtonHover} rounded-md" style="color: ${isDark ? '#e6edf3' : '#1f2328'} !important;">
                                    Show Stats
                                </button>
                                <button id="showViz" class="px-3 py-2 text-sm ${themeClasses.bgButton} ${themeClasses.bgButtonHover} rounded-md" style="color: ${isDark ? '#e6edf3' : '#1f2328'} !important;">
                                    Show Interactive UI
                                </button>
                                <button id="startReview" class="px-3 py-2 text-sm ${themeClasses.bgAccent} ${themeClasses.bgAccentHover} ${themeClasses.textAccent} rounded-md" style="color: ${isDark ? '#58a6ff' : '#0969da'} !important;">
                                    Review Cards
                                </button>
                            </div>
                            <button id="showAddCard" class="px-4 py-2 text-sm bg-blue-500 text-white rounded-md hover:bg-blue-600">
                                + New Card
                            </button>
                        </div>

                        <!-- Stats View -->
                        <div id="statsView" class="hidden mb-6 ${themeClasses.bgSecondary} rounded-lg p-4 shadow-sm">
                            <!-- Stats will be rendered here -->
                        </div>

                        <!-- Visualization View -->
                        <div id="flashcardViz" class="hidden mb-6 ${themeClasses.bgSecondary} rounded-lg p-4 shadow-sm">
                            <!-- D3 visualization will be rendered here -->
                        </div>

                                 <!-- Dashboard search -->
     <div class="flex items-center justify-between w-full mb-6">
    <div class="relative flex-grow max-w-2xl">
        <input type="text" 
            id="dashboardSearch"
            class="w-full pl-10 pr-4 py-2 text-sm border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500" 
            style="background-color: ${isDark ? '#21262d' : '#ffffff'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important; border-color: ${isDark ? '#30363d' : '#d0d7de'} !important;" 
            placeholder="Filter cards...">
        <svg class="absolute left-3 top-2.5 h-4 w-4 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
    </div>
    <div class="flex gap-2 ml-4">
        <!-- Import -->
        <button 
        id="sr-import-btn"
        class="p-2 text-blue-500 hover:text-blue-600 transition-colors" title="Import">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
            </svg>
        </button>
        <!-- Download -->
        <button id="sr-download-btn" class="p-2 text-blue-500 hover:text-blue-600 transition-colors" title="Download">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
            </svg>
        </button>
        <!-- Copy -->
        <button id="copyButton" class="p-2 text-blue-500 hover:text-blue-600 transition-colors" title="Copy">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 copy-icon">
                <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 0 1-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 0 1 1.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 0 0-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 0 1-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 0 0-3.375-3.375h-1.5a1.125 1.125 0 0 1-1.125-1.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H9.75" />
            </svg>
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 check-icon hidden">
                <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5" />
            </svg>
        </button>
        <!-- Share -->
        <button id="sr-share-btn" class="p-2 text-blue-500 hover:text-blue-600 transition-colors" title="Share">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M7.217 10.907a2.25 2.25 0 1 0 0 2.186m0-2.186c.18.324.283.696.283 1.093s-.103.77-.283 1.093m0-2.186 9.566-5.314m-9.566 7.5 9.566 5.314m0 0a2.25 2.25 0 1 0 3.935 2.186 2.25 2.25 0 0 0-3.935-2.186Zm0-12.814a2.25 2.25 0 1 0 3.933-2.185 2.25 2.25 0 0 0-3.933 2.185Z" />
            </svg>
        </button>
    </div>
</div>

                        <!-- Add Card Form -->
                        <div id="addCardForm" class="hidden space-y-4 mb-6 ${themeClasses.bgSecondary} p-4 rounded-lg">
                            <div>
                                <label id="sr-add-card-question-label" class="block text-sm font-medium ${themeClasses.textPrimary} mb-1">Question</label>
                                <div id="questionEditor" class="min-h-[120px] border rounded-md ${themeClasses.inputBorder}"></div>
                            </div>
                            <div>
                                <label class="block text-sm font-medium ${themeClasses.textPrimary} mb-1">Answer</label>
                                <div class="relative">
                                    <div id="answerEditor" class="min-h-[200px] border rounded-md ${themeClasses.inputBorder}"></div>
                                    <button id="generateAnswer" class="absolute top-2 right-2 p-1.5 text-gray-500 hover:text-blue-800 rounded-md hover:bg-gray-100 dark:hover:bg-zinc-600">
                                        <svg data-slot="icon" fill="none" stroke-width="1.5" stroke="#4169E1" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" class="w-5 h-5">
                                            <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z"></path>
                                        </svg>
                                    </button>
                                </div>
                            </div>
                            
                            <div class="flex justify-end space-x-2">
                                <button id="cancelAdd" class="px-4 py-2 text-sm ${themeClasses.textSecondary} ${themeClasses.bgButtonHover} rounded-md">
                                    Cancel
                                </button>
                                <button id="saveCard" class="px-4 py-2 text-sm bg-blue-500 text-white rounded-md hover:bg-blue-600">
                                   Save Card
        <span class="inline-flex items-center gap-1 ml-2 text-xs" id="save-card-shortcut-hint">
            Ctrl
            ↵
        </span>
                                </button>
                            </div>
                        </div>

                        <!-- Card Navigation -->
                        <div id="cardNavigation" class="hidden relative mb-4">
                            <button id="prevCard" class="absolute left-0 top-1/2 -translate-y-1/2 p-2 ${themeClasses.textMuted} hover:${themeClasses.textSecondary}">
                                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path>
                                </svg>
                            </button>
                            <button id="nextCard" class="absolute right-0 top-1/2 -translate-y-1/2 p-2 ${themeClasses.textMuted} hover:${themeClasses.textSecondary}">
                                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                                </svg>
                            </button>
                        </div>

                        <!-- Flashcard List -->
                        <div id="flashcardList" class="space-y-4">
                            <!-- Flashcards will be rendered here -->
                        </div>
                    </div>
                </div>
            </div>
                </div>
            </div>
        `;

    await this.shadowRoot.appendChild(container.firstElementChild);
    await this.setupEditors();
    this.setupModalControls();
    this.setupTagInput();
    
    // Update theme for dynamically created elements
    this.updateThemeElements();

    // Add click handlers for main dropdowns
    // NOTE: These handlers are already set up in SpacedRepetitionUIHandler (ui-handler.js)
    // Duplicate listeners were causing the dropdowns to toggle twice (open->close->open)
    // effectively making them stuck in one state.
    /*
    const decksDropdown = this.shadowRoot.querySelector(
      '[data-dropdown="chapters"]'
    );
    const tagsDropdown = this.shadowRoot.querySelector(
      '[data-dropdown="tags"]'
    );
    const chapterList = this.shadowRoot.querySelector("#chapterList");
    const tagList = this.shadowRoot.querySelector("#tagList");

    decksDropdown?.closest("button").addEventListener("click", () => {
      chapterList.classList.toggle("hidden");
      decksDropdown.style.transform = chapterList.classList.contains("hidden")
        ? "rotate(0deg)"
        : "rotate(180deg)";
    });

    tagsDropdown?.closest("button").addEventListener("click", () => {
      tagList.classList.toggle("hidden");
      tagsDropdown.style.transform = tagList.classList.contains("hidden")
        ? "rotate(0deg)"
        : "rotate(180deg)";
    });
    */

    // Inside your createModal method, add this style block right after the existing styles
    const style = document.createElement("style");
    style.textContent = `
        /* Customize scrollbars for all scrollable elements */
        #sr-sidebar, #chapterList, #tagList, #cardList {
            /* Thin scrollbar that's almost invisible */
            scrollbar-width: thin;
            scrollbar-color: transparent transparent;
        }

        /* For Webkit browsers (Chrome, Safari, etc.) */
        #sr-sidebar::-webkit-scrollbar,
        #chapterList::-webkit-scrollbar,
        #tagList::-webkit-scrollbar,
        #cardList::-webkit-scrollbar {
            width: 4px;
        }

        #sr-sidebar::-webkit-scrollbar-track,
        #chapterList::-webkit-scrollbar-track,
        #tagList::-webkit-scrollbar-track,
        #cardList::-webkit-scrollbar-track {
            background: transparent;
        }

        #sr-sidebar::-webkit-scrollbar-thumb,
        #chapterList::-webkit-scrollbar-thumb,
        #tagList::-webkit-scrollbar-thumb,
        #cardList::-webkit-scrollbar-thumb {
            background-color: transparent;
            border-radius: 20px;
        }

        /* Show scrollbar on hover */
        #sr-sidebar:hover::-webkit-scrollbar-thumb,
        #chapterList:hover::-webkit-scrollbar-thumb,
        #tagList:hover::-webkit-scrollbar-thumb,
        #cardList:hover::-webkit-scrollbar-thumb {
            background-color: rgba(156, 163, 175, 0.3);
        }

        /* Dark mode support */
        .dark #sr-sidebar:hover::-webkit-scrollbar-thumb,
        .dark #chapterList:hover::-webkit-scrollbar-thumb,
        .dark #tagList:hover::-webkit-scrollbar-thumb,
        .dark #cardList:hover::-webkit-scrollbar-thumb {
            background-color: rgba(255, 255, 255, 0.2);
        }
    `;
    this.shadowRoot.appendChild(style);
  }

  updateThemeElements() {
    if (!this.themeManager) return;
    
    // Update KBD elements
    this.shadowRoot.querySelectorAll('[data-theme-kbd], kbd').forEach(kbd => {
      const themeClasses = this.themeManager.getThemeClasses();
      kbd.className = `px-2 py-1 text-xs transition-colors duration-150 ${themeClasses.kbdBg} ${themeClasses.kbdBorder} ${themeClasses.kbdText} border`;
    });
    
    // Update dynamically created deck buttons
    this.shadowRoot.querySelectorAll('[data-chapter]').forEach(button => {
      this.updateDeckButtonTheme(button);
    });
    
    // Set up theme change listener
    this.themeManager.onThemeChange((theme) => {
      this.updateThemeElements();
      // Update UI handler theme
      if (this.uiHandler) {
        this.uiHandler.updateTheme();
      }
    });
  }

  updateDeckButtonTheme(button) {
    if (!this.themeManager) return;
    
    const themeClasses = this.themeManager.getThemeClasses();
    const isActive = button.classList.contains('bg-blue-50') || button.classList.contains('dark:bg-blue-900/30');
    
    // Remove old theme classes
    button.classList.remove(
      'bg-gray-100', 'dark:bg-zinc-700',
      'bg-blue-50', 'dark:bg-blue-900/30',
      'hover:bg-gray-100', 'dark:hover:bg-zinc-700'
    );
    
    // Add appropriate theme classes
    if (isActive) {
      button.classList.add(themeClasses.buttonActive);
    } else {
      button.classList.add(themeClasses.buttonHover);
    }
  }

  setupOnboarding() {
    // this.onboardingHandler.initializeOnboarding();

    // Listen for modal show events
    const srModal = this.shadowRoot.querySelector("#sr-modal");
    if (srModal) {
      srModal.addEventListener("shown.bs.modal", () => {
        if (!this.hasShownFirstTime) {
          this.hasShownFirstTime = true;
          // this.onboardingHandler.startOnboarding();
        }
      });
    }
  }

  setupTooltip() {
    const tooltipElements = [
      {
        element: this.shadowRoot.querySelector("#showStats"),
        text: "View statistics about your learning progress",
      },
      {
        element: this.shadowRoot.querySelector("#showViz"),
        text: "View interactive visualization of your flashcards",
      },
      {
        element: this.shadowRoot.querySelector("#startReview"),
        text: "Start reviewing cards due for practice",
      },
      {
        element: this.shadowRoot.querySelector("#showAddCard"),
        text: "Create a new flashcard",
      },
      {
        element: this.shadowRoot.querySelector("#generateAnswer"),
        text: "Generate an AI-powered answer",
      },
      {
        element: this.shadowRoot.querySelector("#sr-download-btn"),
        text: "Download your flashcards",
      },
      {
        element: this.shadowRoot.querySelector("#sr-import-btn"),
        text: "Import flashcards from a file",
      },
      {
        element: this.shadowRoot.querySelector("#sr-share-btn"),
        text: "Share your flashcards",
      },
      {
        element: this.shadowRoot.querySelector("#copyButton"),
        text: "Copy flashcards to clipboard",
      },
    ];

    // Enable tooltips for each element
    tooltipElements.forEach(({ element, text }) => {
      if (element) {
        enableTooltip(element, text, this.shadowRoot);
      }
    });
  }

  async setupEditors() {
    // Initialize ink-mde editors with modified tab behavior
    const editorOptions = {
      interface: {
        appearance: "dark",
        toolbar: false,
      },
      // Add placeholders for both editors
      placeholder: "Front of flashcard...", // Default placeholder for question editor
      // Allow tab to move between editors
      trapTab: false,
      // Disable default tab behavior
      keybindings: {
        tab: false,
        shiftTab: false,
      },
      // ... rest of editorOptions ...
    };

    // Add custom CSS for tag highlighting and ink-mde code styling
    const style = document.createElement("style");
    style.textContent = `
        .tag-highlight {
            display: inline-block;
            padding: 0 4px;
            margin: 0 2px;
            border-radius: 4px;
            background-color: rgba(59, 130, 246, 0.1);
            color: #3b82f6;
            font-weight: 500;
        }
        .dark .tag-highlight {
            background-color: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
        }

        /* Style both inline and multiline code blocks */
        .ink-mde {
            --ink-code-background-color: rgba(176, 54, 54, 0.1) !important;
            --ink-code-color: #b03636 !important;
            --ink-block-background-color: rgba(176, 54, 54, 0.1) !important;
        }

        .dark .ink-mde {
            --ink-code-background-color: rgba(176, 54, 54, 0.2) !important;
            --ink-code-color: #d45959 !important;
            --ink-block-background-color: rgba(176, 54, 54, 0.2) !important;
        }

        /* Additional styling for inline code */
        .ink-mde .cm-content .cm-inline-code {
            border-radius: 4px;
            padding: 2px 4px;
            font-family: 'Monaco', Courier, monospace;
        }

        /* Style code blocks */
        .ink-mde .cm-content pre {
            background-color: var(--ink-code-background-color) !important;
            color: var(--ink-code-color) !important;
            border-radius: 4px;
            padding: 8px !important;
        }

        .cm-placeholder {
            color: #6b7280 !important;
        }
        .dark .cm-placeholder {
            color: #9ca3af !important;
        }
    `;

    this.shadowRoot.appendChild(style);

    // Initialize editors
    const questionElement = this.shadowRoot.querySelector("#questionEditor");
    const answerElement = this.shadowRoot.querySelector("#answerEditor");
    const generateBtn = this.shadowRoot.querySelector("#generateAnswer");

    generateBtn?.addEventListener("click", async () => {
      try {
        // Get question content using getDoc()
        const questionContent = await this.questionEditor.getDoc();

        if (!questionContent.trim()) {
          console.warn("Please enter a question first");
          return;
        }

        // Disable button and show loading state
        generateBtn.disabled = true;
        generateBtn.innerHTML = `<svg class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>`;

        // Get AI instance
        const ai = await DuckAI.getInstance();

        // Clear existing answer
        await this.answerEditor.update("");
        let fullAnswer = "";

        // Generate and stream response
        const stream = ai.generateAnswer(questionContent);
        for await (const chunk of stream) {
          fullAnswer += chunk;
          await this.answerEditor.update(fullAnswer);
        }
      } catch (error) {
        console.error("Failed to generate answer:", error);
      } finally {
        // Reset button
        generateBtn.disabled = false;
        generateBtn.innerHTML = `<svg data-slot="icon" fill="none" stroke-width="1.5" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" class="w-5 h-5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09Z"></path>
          </svg>`;
      }
    });

    // Initialize editors with different placeholders
    if (questionElement && answerElement) {
      this.questionEditor = ink(questionElement, {
        ...editorOptions,
        placeholder: "Front of flashcard...",
      });
      this.answerEditor = ink(answerElement, {
        ...editorOptions,
        placeholder:
          "Back of flashcard... (Optional: generate with AI)",
      });
    }

    const saveBtn = this.shadowRoot.querySelector("#saveCard");

    // Patch modifier key label to show ⌘ on Mac
    const shortcutHint = this.shadowRoot.querySelector("#save-card-shortcut-hint");
    if (shortcutHint) {
      const isMac = /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent);
      shortcutHint.childNodes[0].textContent = isMac ? '⌘' : 'Ctrl';
    }

    window.addEventListener("sr-save-card-completed", async (e) => {
      const isMac = /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent);
      saveBtn.innerHTML = `
       Save Card
       <span class="inline-flex items-center gap-1 ml-2 text-xs">
           ${isMac ? '⌘' : 'Ctrl'}
           ↵
       </span>
       `;
    });

    if (saveBtn) {
      saveBtn.addEventListener("click", this.handleSave.bind(this));
      document.addEventListener("keydown", async (e) => {
        if (
          (e.ctrlKey || e.metaKey) &&
          e.key === "Enter" &&
          !this.shadowRoot
            .querySelector("#addCardForm")
            ?.classList.contains("hidden")
        ) {
          e.preventDefault();
          saveBtn.click(); // This will trigger the click handler with proper state management
        }
      });
    }

    // Setup cancel button handler with escape symbol
    const cancelBtn = this.shadowRoot.querySelector("#cancelAdd");
    if (cancelBtn) {
      cancelBtn.innerHTML = `
        Cancel
        <kbd class="px-2 py-1 ml-2 text-xs transition-colors duration-150" data-theme-kbd>
            ESC
        </kbd>
      `;
      cancelBtn.addEventListener("click", (e) => {
        this.handleModalClose(e);
      });

      // Add escape key handler for cancel
      document.addEventListener("keydown", (e) => {
        if (
          e.key === "Escape" &&
          !this.shadowRoot
            .querySelector("#addCardForm")
            ?.classList.contains("hidden")
        ) {
          e.preventDefault();
          this.questionEditor.update("");
          this.answerEditor.update("");
          this.shadowRoot
            .querySelector("#addCardForm")
            ?.classList.add("hidden");
        }
      });
    }
  }

  async handleSave(e) {
    if (this.isSaving) return; // Prevent multiple saves

    e.preventDefault();
    this.isSaving = true;
    const saveBtn = this.shadowRoot.querySelector("#saveCard");

    // try
    saveBtn.disabled = true;
    saveBtn.innerHTML = `
            <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Saving...
        `;

    this.saveNewCard().then(() => {
      this.isSaving = false;
      if (saveBtn) {
        saveBtn.disabled = false;
        saveBtn.innerHTML = `
                Save Card
                <span class="inline-flex items-center gap-1 ml-2 text-xs">
                    ${/Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent) ? '⌘' : 'Ctrl'}
                    ↵
                </span>
            `;
      }
    });
    // } finally {
    //     this.isSaving = false;
    //     if (saveBtn) {
    //         saveBtn.disabled = false;
    //         saveBtn.innerHTML = `
    //             Save Card
    //             <span class="inline-flex items-center gap-1 ml-2 text-xs">
    //                 Ctrl
    //                 ↵
    //             </span>
    //         `;
    //     }
    // }
  }

  updateTagsList(newTags) {
    const tagList = this.shadowRoot.querySelector("#tagList");
    if (!tagList) return;
    
    // Ensure tagList has proper scrolling
    tagList.style.cssText = `
        max-height: 200px;
        overflow-y: auto;
        overflow-x: hidden;
    `;

    // Create tags section header if it doesn't exist
    if (!this.shadowRoot.querySelector("#tagsHeader")) {
      const tagsHeader = document.createElement("div");
      tagsHeader.id = "tagsHeader";
      tagsHeader.className =
        "flex items-center justify-between px-3 py-2 text-sm text-gray-500 dark:text-gray-400";
      tagsHeader.innerHTML = `
                <div class="flex items-center gap-2">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                    </svg>
                    <span>TAGS</span>
                </div>
            `;
      tagList.parentNode.insertBefore(tagsHeader, tagList);
    }

    newTags.forEach((tag) => {
      if (!tagList.querySelector(`[data-tag="${tag}"]`)) {
        const tagButton = document.createElement("button");
        tagButton.className =
          "w-full text-left px-3 py-2 text-sm text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-zinc-700 flex items-center gap-2";
        tagButton.setAttribute("data-tag", tag);
        tagButton.innerHTML = `
                    <span class="text-gray-400">#</span>
                    <span>${tag}</span>
                `;
        tagList.appendChild(tagButton);
      }
    });
  }

  setupTagInput() {
    const tagInput = this.shadowRoot.querySelector("#tagInput");
    const tagContainer = this.shadowRoot.querySelector("#tagInputContainer");

    tagInput?.addEventListener("input", (e) => {
      const value = e.target.value;
      if (value.includes("#")) {
        const tags = value.split("#").filter((tag) => tag.trim());
        tagContainer.innerHTML = tags
          .map(
            (tag) => `
                    <span class="tag inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
                        #${tag.trim()}
                        <button class="ml-1 text-blue-500 hover:text-blue-700" data-tag="${tag.trim()}">×</button>
                    </span>
                `
          )
          .join("");
        e.target.value = "";
      }
    });

    // Remove tag when clicking the  button
    tagContainer?.addEventListener("click", (e) => {
      if (e.target.matches("button[data-tag]")) {
        e.target.parentElement.remove();
      }
    });
  }

  // Kai
  setupModalControls() {
    // Setup event listeners for stats and viz buttons
    const showStatsBtn = this.shadowRoot.querySelector("#showStats");
    const showVizBtn = this.shadowRoot.querySelector("#showViz");
    const statsView = this.shadowRoot.querySelector("#statsView");
    const vizView = this.shadowRoot.querySelector("#flashcardViz");
    const newCardBtn = this.shadowRoot.querySelector("#showAddCard");
    const addCardForm = this.shadowRoot.querySelector("#addCardForm");
    const closeBtn = this.shadowRoot.querySelector("#closeModal_sr");
    const modal = this.shadowRoot.querySelector("#spacedRepetitionModal");
    // Add this to your setupModalControls() method
    const copyButton = this.shadowRoot.querySelector("#copyButton");
    if (copyButton) {
      copyButton.addEventListener("click", () => {
        const copyIcon = copyButton.querySelector(".copy-icon");
        const checkIcon = copyButton.querySelector(".check-icon");

        // Hide copy icon, show check icon
        copyIcon.classList.add("hidden");
        checkIcon.classList.remove("hidden");

        // After 1 second, revert back
        setTimeout(() => {
          copyIcon.classList.remove("hidden");
          checkIcon.classList.add("hidden");
        }, 1000);

        // Add your copy functionality here
      });
    }

    // Setup new card button
    if (newCardBtn && addCardForm) {
      newCardBtn.addEventListener("click", () => {
        addCardForm.classList.remove("hidden");
      });
    }

    if (showStatsBtn) {
      showStatsBtn.innerHTML = `
            <div class="flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z" />
                </svg>
                <span>Show Stats</span>
            </div>`;

      showStatsBtn?.addEventListener("click", () => {
        statsView?.classList.toggle("hidden");
        vizView?.classList.add("hidden");

        // Update button text based on visibility
        const span = showStatsBtn.querySelector("span");
        if (span) {
          span.textContent = statsView?.classList.contains("hidden")
            ? "Show Stats"
            : "Hide Stats";
        }

        if (!statsView?.classList.contains("hidden")) {
          if (!this.stats) {
            this.stats = new FlashcardStats(this.shadowRoot);
          }
          this.stats.showStats(this.flashcards);
        }
      });
    }

    // Update viz button with icon and toggle text
    if (showVizBtn) {
      showVizBtn.innerHTML = `
            <div class="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
  <path stroke-linecap="round" stroke-linejoin="round" d="M15.042 21.672 13.684 16.6m0 0-2.51 2.225.569-9.47 5.227 7.917-3.286-.672Zm-7.518-.267A8.25 8.25 0 1 1 20.25 10.5M8.288 14.212A5.25 5.25 0 1 1 17.25 10.5" />
</svg>

                <span>Show Interactive UI</span>
            </div>`;

      showVizBtn?.addEventListener("click", () => {
        vizView?.classList.toggle("hidden");
        statsView?.classList.add("hidden");

        // Update button text based on visibility
        const span = showVizBtn.querySelector("span");
        if (span) {
          span.textContent = vizView?.classList.contains("hidden")
            ? "Show Interactive UI"
            : "Hide Interactive UI";
        }

        if (!vizView?.classList.contains("hidden")) {
          this.initializeVisualization();
        }
      });
    }

    if (closeBtn) {
      if (closeBtn) {
        closeBtn.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.handleModalClose(e);
        });
      }

      // Setup global escape handler
      this.setupModalEscapeHandler();
    }

    // Setup click outside to close
    if (modal) {
      modal.addEventListener("click", (e) => {
        if (e.target === modal) {
          this.hide();
        }
      });
    }

    // Handle tag removal
    const tagContainer = this.shadowRoot.querySelector("#tagInputContainer");
    tagContainer?.addEventListener("click", (e) => {
      const button = e.target.closest("button[data-tag]");
      if (button) {
        button.closest(".tag").remove();
        this.updateTagsList();
      }
    });

    // Setup card navigation
    const prevBtn = this.shadowRoot.querySelector("#prevCard");
    const nextBtn = this.shadowRoot.querySelector("#nextCard");

    prevBtn?.addEventListener("click", () => {
      if (this.currentCardIndex > 0) {
        this.currentCardIndex--;
        this.renderCurrentCard();
      }
    });

    nextBtn?.addEventListener("click", () => {
      if (
        this.currentChapterCards &&
        this.currentCardIndex < this.currentChapterCards.length - 1
      ) {
        this.currentCardIndex++;
        this.renderCurrentCard();
      }
    });

    // Setup review button
    // ... existing code ...

    // In the HTML template or where the button is defined
    const startReviewBtn = this.shadowRoot.querySelector("#startReview");
    if (startReviewBtn) {
      startReviewBtn.innerHTML = `
        <div class="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
                <path stroke-linecap="round" stroke-linejoin="round" d="M6.75 2.994v2.25m10.5-2.25v2.25m-14.252 13.5V7.491a2.25 2.25 0 0 1 2.25-2.25h13.5a2.25 2.25 0 0 1 2.25 2.25v11.251m-18 0a2.25 2.25 0 0 0 2.25 2.25h13.5a2.25 2.25 0 0 0 2.25-2.25m-18 0v-7.5a2.25 2.25 0 0 1 2.25-2.25h13.5a2.25 2.25 0 0 1 2.25 2.25v7.5m-6.75-6h2.25m-9 2.25h4.5m.002-2.25h.005v.006H12v-.006Zm-.001 4.5h.006v.006h-.006v-.005Zm-2.25.001h.005v.006H9.75v-.006Zm-2.25 0h.005v.005h-.006v-.005Zm6.75-2.247h.005v.005h-.005v-.005Zm0 2.247h.006v.006h-.006v-.006Zm2.25-2.248h.006V15H16.5v-.005Z" />
            </svg>
            <span>Review Cards</span>
        </div>`;

      startReviewBtn.addEventListener("click", () => {
        const currentChapterNum = this.currentChapter?.chapter;
        const cards = this.chapterSets.get(parseInt(currentChapterNum));

        if (cards && cards.length > 0) {
          this.showCard(currentChapterNum, 0);
        } else {
          showPopover(this.shadowRoot, "No cards available to review", "info");
        }
      });
    }

    // Add Escape key handler
    // document.addEventListener('keydown', (e) => {
    //     if (e.key === 'Escape') {
    //         this.hide();
    //     }
    // });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        this.handleModalClose();
      }
    });
  }

  async initializeVisualization() {
    const vizView = this.shadowRoot.querySelector("#flashcardViz");
    if (!vizView) return;

    // Only set up the container and buttons if they don't exist
    if (!this.shadowRoot.querySelector("#visualization-container")) {
      vizView.innerHTML = `
            <div class="flex gap-2 mb-4">
                <button id="viewNetwork" class="px-3 py-1 text-sm rounded-md bg-blue-500 text-white hover:bg-blue-600">
                    Network View
                </button>
                <button id="viewHeatmap" class="px-3 py-1 text-sm rounded-md bg-blue-500 text-white hover:bg-blue-600">
                    Activity View
                </button>
            </div>
            <div id="visualization-container" class="w-full h-[500px] relative bg-white dark:bg-zinc-800 rounded-lg shadow-lg">
            </div>
        `;

      // Add view toggle handlers
      const viewNetworkBtn = this.shadowRoot.querySelector("#viewNetwork");
      const viewHeatmapBtn = this.shadowRoot.querySelector("#viewHeatmap");

      viewNetworkBtn?.addEventListener("click", () => {
        // Pass chapter data again when switching to network view
        this.visualization?.update(this.chapterSets);
        this.visualization?.showNetworkView();
      });

      viewHeatmapBtn?.addEventListener("click", async () => {
        const reviewData = await this.storageHandler.getReviewActivity();
        this.visualization?.updateHeatmapData(reviewData);
        this.visualization?.showHeatmapView();
      });
    }

    // Initialize visualization
    if (!this.visualization) {
      this.visualization = new FlashcardVisualization(this.shadowRoot);
    }

    // Show initial view with network data
    this.visualization.update(this.chapterSets);
    this.visualization.showNetworkView();
  }

  // Core modal methods
  show() {
    const modal = this.shadowRoot.querySelector("#spacedRepetitionModal");
    if (modal) {
      modal.classList.remove("hidden");

      if(this.storageHandler) {
      // Add this line to track modal opens
      this.storageHandler.addReviewActivity()
          .then(async result => {
              // Debug the review history
              // await this.storageHandler.debugReviewHistory();
              
              if (this.visualization && this.visualization.currentView === 'heatmap') {
                  this.visualization.updateHeatmapData(result);
              }
          })
          .catch(err => console.error("Failed to track review activity:", err));
      }

      // Re-initialize floating controls if needed
      if (!this.floatingControlsHandler) {
        try {
          this.floatingControlsHandler = new FloatingControlsHandler(this);
          this.setupEventListeners();
        } catch (error) {
          console.error(
            "Failed to initialize floating controls on show:",
            error
          );
        }
      } else {
      }
      
      // Ensure chapters are rendered when modal is shown
      if (this.storageHandler) {
        this.renderChapters();
      }
    }

    // Ensure modal is fully initialized before potentially starting onboarding
    requestAnimationFrame(() => {
      if (!this.hasShownFirstTime) {
        this.hasShownFirstTime = true;
        // this.onboardingHandler.startOnboarding();
      }
    });
  }


  hide() {
    const modal = this.shadowRoot.querySelector("#spacedRepetitionModal");
    if (modal) {
      modal.classList.add("hidden");
    }
  }

  // KAI
  setCurrentChapter(chapter) {
    this.currentChapter = chapter;
    this.flashcards = this.chapterSets.get(chapter.chapter) || [];

    // Sync with storage handler to ensure cards are saved to the correct deck
    this.storageHandler.inMemoryData.currentChapter = chapter;
    this.storageHandler.inMemoryData.currentDeck = this.chapterSets.get(chapter.chapter) || [];

    // Show first card of the chapter
    if (this.flashcards.length > 0) {
      this.showCard(chapter.chapter, 0);
    }

    this.storageHandler.saveToLocalStorage();
  }

  getChapterTitle(chapter) {
    return this.currentChapter?.chapter === chapter
      ? this.currentChapter.title
      : `Chapter ${chapter}`;
  }

  async saveNewCard(questionInput="", answerInput="") {
    try {
        let loadingId = null;
        // Show loading state immediately
        if(!questionInput && !answerInput) {
            loadingId = this.storageHandler.notificationHandler.showLoadingState("Saving card...");
        }
        
        // Get content from editors using getDoc()
        const question = questionInput || await this.questionEditor.getDoc();
        const answer = answerInput || await this.answerEditor.getDoc();

        if (!question || !answer) {
            console.error("Question and answer are required");
            alert("Question and answer are required");
            window.dispatchEvent(new CustomEvent("sr-save-card-completed", { detail: { success: false } }));
            return;
        }

        // Create new flashcard
        const newCard = new FlashCard(question.trim(), answer.trim());

        // Extract tags
        const questionTags = this.extractTags(question);
        const answerTags = this.extractTags(answer);
        newCard.tags = [...new Set([...questionTags, ...answerTags])];

        // Ensure we have a current chapter
        if (!this.currentChapter) {
            console.error("No chapter selected");
            return;
        }

        // Initialize chapter array if it doesn't exist
        if (!this.chapterSets.has(this.currentChapter.chapter)) {
            this.chapterSets.set(this.currentChapter.chapter, []);
        }

        const chapterCards = this.chapterSets.get(this.currentChapter.chapter);
        if (!Array.isArray(chapterCards)) {
            this.chapterSets.set(this.currentChapter.chapter, []);
        }

        
        // Now safely push the new card
        this.chapterSets.get(this.currentChapter.chapter).push(newCard);
        this.flashcards = this.chapterSets.get(this.currentChapter.chapter);

        // Ensure storageHandler's inMemoryData is initialized
        if (!this.storageHandler.inMemoryData.currentDeck) {
            this.storageHandler.inMemoryData.currentDeck = [];
        }
        
        this.storageHandler.inMemoryData.currentDeck.push(newCard);

        // Save to storage
        await this.storageHandler.saveToLocalStorage();

        // Clear form
        await this.questionEditor.update("");
        await this.answerEditor.update("");
        const addCardForm = this.shadowRoot.querySelector("#addCardForm");
        addCardForm?.classList.add("hidden");

        // Update UI
        this.uiHandler.updateProgress();

        if(questionInput || answerInput) {

            this.immediateUpdate(true);
        } else {
            this.immediateUpdate();
        }

        // Update visualization if visible
        if (this.visualization) {
            this.visualization.update(this.flashcards);
        }

        if(questionInput || answerInput) {
            return
        }
        // Show success notification
        this.storageHandler.notificationHandler.updateNotification(
            loadingId,
            "Card saved successfully!",
            "success"
        );
    } catch (error) {
        console.error("Failed to save new card:", error);
        this.storageHandler.notificationHandler.updateNotification(
            "save-error",
            "Failed to save card",
            "error"
        );
    }
  }

  renderCards() {
    const cardList = this.shadowRoot.querySelector("#cardList");
    if (!cardList) return;

    // Get current theme for flashcard rendering
    const hostElement = this.shadowRoot.host;
    const currentTheme = hostElement?.getAttribute('data-socratiq-theme') || 'light';
    const isDark = currentTheme === 'dark';
    
    console.log('[RENDER CARDS DEBUG] Rendering cards with theme:', {
      currentTheme: currentTheme,
      isDark: isDark,
      cardsCount: this.flashcards.length
    });

    cardList.innerHTML = this.flashcards
      .map(
        (card) => `
            <div class="rounded-lg p-4 shadow-sm" style="box-shadow: 0 10px 20px rgba(0, 0, 255, 0.5); background-color: ${isDark ? '#0d1117' : '#ffffff'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important;">
                <div class="flex justify-between items-start mb-2">
                    <h3 class="text-gray-900 dark:text-white">${
                      card.question
                    }</h3>
                    <div class="flex space-x-2">
                        <button class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" data-card-id="${
                          card.id
                        }" data-action="edit">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/>
                            </svg>
                        </button>
                        <button class="text-gray-400 hover:text-red-600" data-card-id="${
                          card.id
                        }" data-action="delete">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                            </svg>
                        </button>
                    </div>
                </div>
                <p class="text-gray-600 dark:text-gray-300">${card.answer}</p>
                ${
                  card.tags
                    ? `
                    <div class="mt-2 flex flex-wrap gap-1">
                        ${card.tags
                          .map(
                            (tag) => `
                            <span class="inline-flex items-center px-2 py-1 rounded text-xs bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
                                #${tag}
                            </span>
                        `
                          )
                          .join("")}
                    </div>
                `
                    : ""
                }
            </div>
        `
      )
      .join("");

    // Add event listeners for edit and delete buttons
    cardList.querySelectorAll("[data-action]").forEach((button) => {
      button.addEventListener("click", (e) => {
        const cardId = button.dataset.cardId;
        const action = button.dataset.action;

        if (action === "edit") {
          this.editCard(cardId);
        } else if (action === "delete") {
          this.deleteCard(cardId);
        }
      });
    });

    // Add after cards are rendered
    // this.dragHandler.initialize();
  }

  editCard(cardId) {
    const card = this.flashcards.find((c) => c.id === cardId);
    if (!card) return;

    // Show form with existing values
    const addCardForm = this.shadowRoot.querySelector("#addCardForm");
    const questionInput = this.shadowRoot.querySelector("#questionInput");
    const answerInput = this.shadowRoot.querySelector("#answerInput");
    const tagContainer = this.shadowRoot.querySelector("#tagInputContainer");

    if (addCardForm && questionInput && answerInput) {
      questionInput.value = card.question;
      answerInput.value = card.answer;
      addCardForm.classList.remove("hidden");

      // Add tags
      if (card.tags) {
        tagContainer.innerHTML = card.tags
          .map(
            (tag) => `
                    <span class="tag inline-flex items-center px-2 py-1 rounded bg-blue-100 text-blue-700 text-sm">
                        #${tag}
                        <button class="ml-1 text-blue-500 hover:text-blue-700">×</button>
                    </span>
                `
          )
          .join("");
      }

      // Update save button to handle edit
      const saveBtn = this.shadowRoot.querySelector("#saveCard");
      if (saveBtn) {
        saveBtn.dataset.editId = cardId;
      }
    }
  }

  deleteCard(cardId) {
    if (!cardId) {
      console.error("deleteCard called with undefined/null cardId");
      return;
    }
    
    try {
      console.log("🗑️ deleteCard called with cardId:", cardId);
      
      // Check if we're in review mode and this is the current card
      // SAFE: Use optional chaining and check for existence
      let isCurrentCard = false;
      let currentReviewIndex = -1;
      
      if (this.currentReview && this.currentReview.cards && Array.isArray(this.currentReview.cards)) {
        currentReviewIndex = this.currentReview.currentIndex || 0;
        const currentCard = this.currentReview.cards[currentReviewIndex];
        if (currentCard && currentCard.id) {
          isCurrentCard = currentCard.id === cardId;
        }
      }
    
    // PROBLEMATIC CODE FIXED: Filter undefined/null BEFORE accessing .id property
    // This prevents "Cannot read properties of undefined (reading 'id')" errors
    const safeFlashcards = (this.flashcards || []).filter(c => c != null && c.id != null);
    const safeCurrentDeck = (this.storageHandler?.inMemoryData?.currentDeck || []).filter(c => c != null && c.id != null);
    
    // Find the original index in currentDeck (before filtering) for proper index adjustment
    let originalIndexInCurrentDeck = -1;
    if (this.storageHandler?.inMemoryData?.currentDeck) {
      // SAFE: Check c != null FIRST before accessing c.id
      originalIndexInCurrentDeck = this.storageHandler.inMemoryData.currentDeck.findIndex(c => {
        if (c == null) return false; // Skip null/undefined
        if (c.id == null) return false; // Skip elements without id
        return c.id === cardId; // Match the cardId
      });
    }
    
    // Search in flashcards first, then currentDeck
    let foundIn = null;
    let index = safeFlashcards.findIndex((c) => c.id === cardId);
    if (index !== -1) {
      foundIn = 'flashcards';
    } else if (originalIndexInCurrentDeck !== -1) {
      foundIn = 'currentDeck';
      index = originalIndexInCurrentDeck;
    }
    
    if (index === -1) {
      console.warn(`Card with id ${cardId} not found in any array`);
      // Try to delete from storage anyway
      this.storageHandler?.purgeCardFromMemory(cardId);
      this.storageHandler?.saveToLocalStorage();
      return;
    }

    // Update the original arrays by filtering - safely check for null BEFORE accessing id
    if (foundIn === 'flashcards' && this.flashcards) {
      // FIXED: Check c != null FIRST, then safely access c.id
      this.flashcards = this.flashcards.filter(c => {
        if (c == null) return false; // Remove undefined/null elements
        if (c.id == null) return false; // Remove elements without id
        return c.id !== cardId; // Remove the card we're deleting
      });
    }
    
    // currentDeck and currentReview.cards are the same reference, so update both
    if ((foundIn === 'currentDeck' || foundIn === 'currentReview') && this.storageHandler?.inMemoryData?.currentDeck) {
      // FIXED: Check c != null FIRST, then safely access c.id
      const filteredDeck = this.storageHandler.inMemoryData.currentDeck.filter(c => {
        if (c == null) return false; // Remove undefined/null elements
        if (c.id == null) return false; // Remove elements without id
        return c.id !== cardId; // Remove the card we're deleting
      });
      
      // CRITICAL: Reassign to update the reference
      this.storageHandler.inMemoryData.currentDeck = filteredDeck;
      
      // CRITICAL: Update currentReview.cards to point to the new filtered array
      // (The old reference is broken after reassignment, so we need to update it)
      if (this.currentReview) {
        this.currentReview.cards = filteredDeck;
      }
      
      // Adjust the currentIndex if we deleted the current card or one before it
      if (this.currentReview && this.currentReview.cards) {
        // originalIndexInCurrentDeck is the index before deletion
        if (isCurrentCard || originalIndexInCurrentDeck < currentReviewIndex) {
          // If we deleted the current card, go to previous card or stay at 0
          if (isCurrentCard) {
            if (this.currentReview.currentIndex > 0) {
              this.currentReview.currentIndex--;
            } else if (this.currentReview.cards.length > 0) {
              // Stay at index 0, the card array already shifted
              this.currentReview.currentIndex = 0;
            } else {
              // No cards left, close modal
              const modal = this.shadowRoot.querySelector("#cardReviewModal");
              if (modal) modal.classList.add("hidden");
              this.currentReview = null;
            }
          } else {
            // Deleted a card before current, adjust index down by 1
            this.currentReview.currentIndex--;
          }
        }
      }
    }
    
    this.storageHandler.purgeCardFromMemory(cardId);
    this.storageHandler.saveToLocalStorage();
    this.uiHandler.updateProgress();
    this.renderCards();

    // Update visualization if visible
    const vizContainer = this.shadowRoot.querySelector("#flashcardViz");
    if (vizContainer && !vizContainer.classList.contains("hidden")) {
      const safeCards = (this.flashcards || []).filter(c => c != null && c.id != null);
      this.visualization.update(safeCards);
    }
    
    // If we're in review mode and deleted a card, update the review modal
    if (this.currentReview && isCurrentCard) {
      const modal = this.shadowRoot.querySelector("#cardReviewModal");
      if (modal && this.currentReview.cards && this.currentReview.cards.length > 0) {
        // Ensure currentIndex is valid
        if (this.currentReview.currentIndex >= this.currentReview.cards.length) {
          this.currentReview.currentIndex = Math.max(0, this.currentReview.cards.length - 1);
        }
        
        // Update the displayed card - safely check for undefined
        const card = this.currentReview.cards[this.currentReview.currentIndex];
        if (card && card.id && this.reviewEditor) {
          this.reviewEditor.update(card.question);
          this.updateNavigationButtons?.();
        } else if (this.currentReview.cards.length === 0) {
          // No cards left, close modal
          modal.classList.add("hidden");
          this.currentReview = null;
        }
      } else if (modal && (!this.currentReview.cards || this.currentReview.cards.length === 0)) {
        // No cards left, close modal
        modal.classList.add("hidden");
        this.currentReview = null;
      }
    }
    } catch (error) {
      console.error("❌ Error in deleteCard:", error);
      console.error("Error details:", {
        cardId,
        hasCurrentReview: !!this.currentReview,
        currentReviewCards: this.currentReview?.cards,
        currentReviewIndex: this.currentReview?.currentIndex,
        flashcardsLength: this.flashcards?.length,
        currentDeckLength: this.storageHandler?.inMemoryData?.currentDeck?.length
      });
      throw error; // Re-throw to see the full stack trace
    }
  }

  clearInputs() {
    const questionInput = this.shadowRoot.querySelector("#questionInput");
    const answerInput = this.shadowRoot.querySelector("#answerInput");
    if (questionInput) questionInput.value = "";
    if (answerInput) answerInput.value = "";
  }

  renderTags() {
    const tagList = this.shadowRoot.querySelector("#tagList");
    const totalTagCountElement =
      this.shadowRoot.querySelector("#totalTagCount");
    if (!tagList) return;
    
    // Ensure tagList has proper scrolling
    tagList.style.cssText = `
        max-height: 200px;
        overflow-y: auto;
        overflow-x: hidden;
    `;
    
    // Get theme classes for proper styling
    const themeClasses = this.themeManager?.getThemeClasses() || {};
    const tagCountBg = themeClasses.tagCountBg || 'bg-gray-100 dark:bg-zinc-700';

    // Get unique tags and their counts
    const tagCounts = new Map();
    this.chapterSets.forEach((cards) => {
      cards.forEach((card) => {
        if (card.tags) {
          card.tags.forEach((tag) => {
            tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
          });
        }
      });
    });

    // Update total tag count in the header
    if (totalTagCountElement) {
      totalTagCountElement.textContent = tagCounts.size;
    }

    // Render tag list with counts
    tagList.innerHTML = Array.from(tagCounts.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(
        ([tag, count]) => `
            <button class="w-full text-left px-3 py-2 text-sm text-gray-500 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-zinc-800/50 rounded-md flex items-center justify-between"
                    data-tag="${tag}">
                <span class="flex items-center">
                    <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A2 2 0 013 12V7a4 4 0 014-4z"></path>
                    </svg>
                    ${tag}
                </span>
                <span class="text-xs ${tagCountBg} px-2 py-1 rounded-full">
                    ${count}
                </span>
            </button>
        `
      )
      .join("");

    // Add click handlers for tags
    tagList.querySelectorAll("button[data-tag]").forEach((button) => {
      button.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const tag = button.dataset.tag;
        console.log(`[Modal] Tag clicked: "${tag}"`);
        this.switchTag(tag);
      });
    });
  }

  cleanupChapterSets() {
    // Process each chapter's cards
    for (const [chapterNum, cards] of this.chapterSets.entries()) {
      if (!cards) continue;

      const seenContent = new Set();
      const uniqueCards = [];

      cards.forEach((card) => {
        // Create content signature
        const contentSignature = `${card.question}|||${card.answer}`
          .toLowerCase()
          .trim();

        if (!seenContent.has(contentSignature)) {
          seenContent.add(contentSignature);
          uniqueCards.push(card);
        } else {
          console.warn("Removing duplicate card from chapter", chapterNum, {
            question: card.question,
            answer: card.answer,
          });
        }
      });

      // Update chapter with unique cards only
      if (cards.length !== uniqueCards.length) {
        this.chapterSets.set(chapterNum, uniqueCards);

        // Sync with storage handler
        if (
          this.storageHandler &&
          parseInt(this.currentChapter?.chapter) === chapterNum
        ) {
          this.storageHandler.inMemoryData.currentDeck = uniqueCards;
          this.storageHandler
            .saveToLocalStorage()
            .catch((err) => console.error("Failed to save cleaned deck:", err));
        }
      }
    }
  }

  // /KAI
  renderChapters(isNoMessage=false) {
    const chapterList = this.shadowRoot.querySelector("#chapterList");
    if (!chapterList) return;
    
    // Check if storage handler is ready
    if (!this.storageHandler) {
      console.log('[RENDER DEBUG] Storage handler not ready, skipping renderChapters');
      return;
    }

    // Add these style properties to enable independent scrolling
    chapterList.style.cssText = `
        max-height: 300px;
        overflow-y: auto;
        overflow-x: hidden;
    `;

    // Add click handler for chapter selection
    chapterList?.addEventListener("click", (e) => {
      const chapterBtn = e.target.closest("button[data-chapter]");
      if (chapterBtn) {
        const chapterNum = parseInt(chapterBtn.dataset.chapter);

        // Update current chapter
        this.currentChapter = {
          chapter: chapterNum,
          title: this.storageHandler.getChapterTitle(chapterNum),
        };

        // Sync with storage handler to ensure cards are saved to the correct deck
        this.storageHandler.inMemoryData.currentChapter = this.currentChapter;
        this.storageHandler.inMemoryData.currentDeck = this.chapterSets.get(chapterNum) || [];

        // Load the chapter's cards
        const cards = this.storageHandler.loadChapter(chapterNum);

        // Update chapterSets if needed
        if (!this.chapterSets.has(chapterNum)) {
          this.chapterSets.set(chapterNum, cards);
        }
      }
    });

    const chapterCountElement = this.shadowRoot.querySelector("#chapterCount");

    if (!chapterList) {
      console.warn("Chapter list element not found");
      return;
    }

    // Get all chapters and count - MINIMAL CHANGE: Add current chapter if missing
    this.storageHandler
      .getAllChapters()
      .then(async (chapters) => {

        // Add current chapter if it exists and isn't in the list
        if (
          this.currentChapter &&
          !chapters.find((c) => c.chapter === this.currentChapter.chapter)
        ) {
          chapters.push({
            chapter: this.currentChapter.chapter,
            title: this.currentChapter.title,
          });
        }
        
        // Create initial deck if no chapters exist
        if (!chapters || chapters.length === 0) {
          try {
            const initialChapter = {
              chapter: 0,
              title: "Introduction",
              cardCount: 0
            };
            await this.storageHandler.createChapter(initialChapter);
            chapters = [initialChapter];
          } catch (error) {
            console.error("Failed to create initial deck:", error);
          }
        }

        const chapterCount = chapters.length;
        if (chapterCountElement) {
          chapterCountElement.textContent = chapterCount;
        }

        // Generate HTML for all chapters
        let chaptersContent = "";
        chapters.forEach((chapter) => {
          const isCurrentChapter =
            chapter.chapter === this.currentChapter?.chapter;
          const cards = this.chapterSets.get(parseInt(chapter.chapter)) || [];

          const themeClasses = this.themeManager?.getThemeClasses() || {};
          const activeClasses = isCurrentChapter ? themeClasses.buttonActive || "bg-blue-50 dark:bg-blue-900/30" : "";
          const hoverClasses = themeClasses.buttonHover || "hover:bg-gray-50 dark:hover:bg-zinc-800/50";
          
          chaptersContent += `
            <div class="chapter-container group">
                <button class="w-full flex justify-between px-3 py-2 text-sm text-gray-600 dark:text-gray-300 ${hoverClasses} rounded-md ${activeClasses} min-h-[2.5rem]" data-chapter="${chapter.chapter}">
                    <div class="flex gap-2">
                        <svg class="w-4 h-4 transform transition-transform duration-200" 
                             data-dropdown="chapter-${chapter.chapter}" 
                             fill="none" 
                             stroke="currentColor" 
                             viewBox="0 0 24 24">
                            <path stroke-linecap="round" 
                                  stroke-linejoin="round" 
                                  stroke-width="2" 
                                  d="M19 9l-7 7-7-7">
                            </path>
                        </svg>
                        <span class="flex items-start text-left">
                            <span class="text-left">${chapter.title || `Deck ${chapter.chapter}`}</span>
                        </span>
                    </div>
                    <div class="flex flex-col items-center justify-center h-full">
                        <span class="text-xs text-gray-400">${cards.length}</span>
                        <span class="opacity-0 group-hover:opacity-100 transition-opacity duration-200 cursor-pointer text-red-500 dark:text-red-400" 
                              data-delete-chapter="${chapter.chapter}" 
                              title="Delete deck">
                            <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                            </svg>
                        </span>
                    </div>
                </button>
                <div class="chapter-cards ml-6 mt-1 space-y-0 ${
                  isCurrentChapter ? "" : "hidden"
                }">
                    ${cards
                      .map(
                        (card, index) => `
                        <button class="w-full text-left px-3 py-2 text-sm text-gray-500 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-zinc-800/50 rounded-md flex items-start"
                                data-chapter="${chapter.chapter}" 
                                data-card-index="${index}">
                            <span class="mr-2">
                                <svg data-slot="icon" fill="none" stroke-width="1.5" stroke="currentColor" viewBox="0 0 24 24" class="w-4 h-4">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"></path>
                                </svg>
                            </span>
                            <span>${card.question}</span>
                        </button>
                    `
                      )
                      .join("")}
                </div>
            </div>
        `;
        });

        // Add "New Deck" button as the last element
        chaptersContent += `
            <div class="mt-2">
                <button id="newDeckBtn" class="w-full flex items-center justify-center px-3 py-2 text-sm text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-md transition-colors">
                    + New Deck
                </button>
            </div>
        `;

        chapterList.innerHTML = chaptersContent;

        // Add click handlers for all chapters
        chapters.forEach((chapter) => {
          const chapterBtn = chapterList.querySelector(
            `[data-chapter="${chapter.chapter}"]`
          );
          const cardsDiv = chapterBtn?.nextElementSibling;
          const dropdownIcon = chapterList.querySelector(
            `[data-dropdown="chapter-${chapter.chapter}"]`
          );

          if (chapterBtn && cardsDiv && dropdownIcon) {
            chapterBtn.addEventListener("click", (e) => {
              e.preventDefault();
              cardsDiv.classList.toggle("hidden");
              dropdownIcon.style.transform = cardsDiv.classList.contains(
                "hidden"
              )
                ? "rotate(0deg)"
                : "rotate(180deg)";

              // Remove highlight from all chapters
              chapterList.querySelectorAll("[data-chapter]").forEach((btn) => {
                const themeClasses = this.themeManager?.getThemeClasses() || {};
                const activeClasses = themeClasses.buttonActive || "bg-blue-50 dark:bg-blue-900/30";
                btn.classList.remove(...activeClasses.split(' '));
              });

              // Add highlight to clicked chapter
              const themeClasses = this.themeManager?.getThemeClasses() || {};
              const activeClasses = themeClasses.buttonActive || "bg-blue-50 dark:bg-blue-900/30";
              chapterBtn.classList.add(...activeClasses.split(' '));

              if (chapter.chapter !== this.currentChapter?.chapter) {
                this.currentChapter = chapter;
                // Sync with storage handler to ensure cards are saved to the correct deck
                this.storageHandler.inMemoryData.currentChapter = chapter;
                this.storageHandler.inMemoryData.currentDeck = this.chapterSets.get(chapter.chapter) || [];
                this.showAllCards(isNoMessage);
              }
            });
          }
        });

        // Add click handlers for delete buttons
        const deleteButtons = chapterList.querySelectorAll("[data-delete-chapter]");
        console.log(`Found ${deleteButtons.length} delete buttons:`, deleteButtons);
        deleteButtons.forEach((button) => {
          const chapterNum = parseInt(button.dataset.deleteChapter);
          console.log(`Adding delete listener for chapter ${chapterNum}`);
          button.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log(`Delete button clicked for chapter ${chapterNum}`);
            this.showDeleteConfirmation(chapterNum);
          });
        });

        // Add click handlers for all cards
        const cardButtons = chapterList.querySelectorAll("[data-card-index]");
        cardButtons.forEach((button) => {
          button.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const chapterNum = parseInt(button.dataset.chapter);
            const index = parseInt(button.dataset.cardIndex);
            this.showCard(chapterNum, index);
          });
        });

        // Add click handler for main Decks dropdown
        const decksButton = this.shadowRoot.querySelector(
          '[data-dropdown="chapters"]'
        );
        if (decksButton && chapterList) {
          decksButton.closest("button").addEventListener("click", () => {
            chapterList.classList.toggle("hidden");
            decksButton.style.transform = chapterList.classList.contains(
              "hidden"
            )
              ? "rotate(0deg)"
              : "rotate(180deg)";
          });
        }

        if (this.currentChapter) {
          const currentChapterBtn = chapterList.querySelector(
            `[data-chapter="${this.currentChapter.chapter}"]`
          );
          const currentCardsDiv = currentChapterBtn?.nextElementSibling;
          const currentDropdownIcon = chapterList.querySelector(
            `[data-dropdown="chapter-${this.currentChapter.chapter}"]`
          );

          if (currentCardsDiv && currentDropdownIcon) {
            // Show cards for current chapter
            currentCardsDiv.classList.remove("hidden");
            currentDropdownIcon.style.transform = "rotate(180deg)";
            this.showAllCards(); // Update the main flashcard display
          }
        }

        // Add click handler for "New Deck" button
        const newDeckBtn = chapterList.querySelector("#newDeckBtn");
        if (newDeckBtn) {
          newDeckBtn.addEventListener("click", (e) => {
            e.preventDefault();
            console.log("New Deck button clicked!");
            this.deckCreationHandler.showModal();
          });
        }
      })
      .catch((error) => {
        console.error("Failed to load chapters:", error);
      });
  }

  showAllCards(isNoMessage=false) {
    // Get current theme for flashcard rendering
    const hostElement = this.shadowRoot.host;
    const currentTheme = hostElement?.getAttribute('data-socratiq-theme') || 'light';
    const isDark = currentTheme === 'dark';
    
    console.log('[FLASHCARD DEBUG] Rendering cards with theme:', {
      currentTheme: currentTheme,
      isDark: isDark
    });

    // Add custom styles for ink-mde to the shadow DOM
    if (!this.shadowRoot.querySelector("#ink-mde-styles")) {
      const styles = document.createElement("style");
      styles.id = "ink-mde-styles";
      styles.textContent = `

      #cardList, flashcardList# {
    height: calc(100vh - 180px);
    overflow-x: hidden;
}


                /* Target the main ink-mde container */
                .ink.ink-mde {
                    border: none !important;
                }
                
                /* Target the inner container */
                .ink-mde {
                    border: none !important;
                }
                
                /* Additional specific selectors to ensure no borders */
                .ink.ink-mde .ink-mde-editor {
                    border: none !important;
                }
                
                .ink.ink-mde .cm-editor {
                    border: none !important;
                }

                /* Rest of your existing styles */
                .cm-editor {
                    background: transparent !important;
                    border: none !important;
                    padding: 0 !important;
                }
                
                .cm-editor .cm-scroller {
                    font-family: inherit !important;
                    line-height: 1.5 !important;
                }
                
                .cm-editor .cm-content {
                    padding: 0 !important;
                }
                
                .cm-editor .cm-line {
                    padding: 0 !important;
                }
                
                .editor-separator {
                    border: none;
                    border-top: 1px dashed rgba(100, 116, 139, 0.3);
                    margin: 0.5rem 0;
                }
                
                .dark .editor-separator {
                    border-top-color: rgba(148, 163, 184, 0.2);
                }

                .ink .ink-mde{
                    border: none;
                }

                    .cm-cursor {
        border-left: 1.2px solid currentColor !important;
        border-right: none !important;
        width: 0 !important;
        margin-left: -0.6px;
    }

    .cm-focused .cm-cursor {
        visibility: visible !important;
        animation: blink 1.2s steps(1) infinite !important;
    }

    @keyframes blink {
        50% { visibility: hidden; }
    }

    /* Ensure cursor container is properly positioned */
    .cm-cursorLayer {
        visibility: visible !important;
        position: absolute;
        pointer-events: none;
    }

    /* Ensure editor has proper focus styles */
    .cm-editor.cm-focused {
        outline: none !important;
    }

      .ink-mde:focus-within .cm-placeholder {
            display: none !important;
        }

         .cm-editor.cm-focused .cm-placeholder {
            display: none !important;
        }
           
        
           /* Generate button positioning */
    #generateAnswer {
        position: absolute !important;
        top: 8px !important;
        right: 8px !important;
        z-index: 10 !important;
        color: royalblue !important;
        background: transparent !important;
    }
            .dark #generateAnswer:hover {
        background: rgba(113, 113, 122, 0.2) !important;
    }
        `;
      this.shadowRoot.appendChild(styles);
    }

    const getLastReviewQuality = (card) => {
      // Add null check for card
      if (!card) return 0;
      // Safely return 0 (Reset) if lastReviewQuality is not defined
      return card.lastReviewQuality !== undefined ? card.lastReviewQuality : 0;
    };

    const buttonClasses = (quality, card) => {
      // Add null check for card
      if (!card) {
        return `px-3 py-2 text-xs rounded flex flex-col items-start xs:flex-1 min-w-0" style="background-color: ${isDark ? '#21262d' : '#f9fafb'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important;"`;
      }
      
      const baseClasses = `px-3 py-2 text-xs rounded flex flex-col items-start xs:flex-1 min-w-0" style="background-color: ${isDark ? '#21262d' : '#f9fafb'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important;"`;
      const isLastUsed = getLastReviewQuality(card) === quality;

      return isLastUsed
        ? `${baseClasses} ring-1 ring-blue-500/20" style="background-color: ${isDark ? '#1f2937' : '#dbeafe'} !important; color: ${isDark ? '#58a6ff' : '#0969da'} !important;"`
        : baseClasses;
    };

    const flashcardList = this.shadowRoot.querySelector("#flashcardList");
    if (!flashcardList || !this.currentChapter) return;

    const cards =
      this.chapterSets.get(parseInt(this.currentChapter.chapter)) || [];

      console.log("🔄 Modal: Cards:", cards);
      flashcardList.innerHTML = `
      <div class="${cards.length <= 2 
        ? `grid grid-cols-1 md:grid-cols-${cards.length}` 
        : 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3'} gap-4">
          ${cards
            .map(
              (card, index) => `
                <div class="rounded-lg transition-all duration-200 relative"
                     style="box-shadow: 4.2px 8.3px 8.3px hsla(0, 0%, 50%, 0.37);
                            transition: box-shadow 0.2s ease-in-out;
                            background-color: ${isDark ? '#0d1117' : '#ffffff'} !important;
                            color: ${isDark ? '#e6edf3' : '#1f2328'} !important;"
                     onmouseover="this.style.boxShadow='4.2px 8.3px 12px hsla(0, 0%, 50%, 0.45)'"
                     onmouseout="this.style.boxShadow='4.2px 8.3px 8.3px hsla(0, 0%, 50%, 0.37)'"
                     data-card-index="${index}">
                    <!-- Keep existing delete button and content -->
                    <button class="delete-card-btn absolute top-2 right-2 p-1.5 text-gray-400 hover:text-red-500 rounded-md hover:bg-gray-100 dark:hover:bg-zinc-600 z-10"
                            data-delete-index="${index}">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5">
                            <path stroke-linecap="round" stroke-linejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
                        </svg>
                    </button>
                    
                    <div class="p-4 space-y-2">
                        <div id="question-${index}" class="min-h-[120px]"></div>
                        <hr class="editor-separator" />
                        <div id="answer-${index}" class="min-h-[200px]"></div>
                        
                        <div class="grid grid-cols-2 xs:flex gap-2 mt-4">
                            <button class="${buttonClasses(0, card)}"
                                    onclick="event.stopPropagation(); this.closest('[data-card-index]').dispatchEvent(new CustomEvent('reviewCard', {detail: {index: ${index}, quality: 0}}))">
                                <div class="flex items-center gap-2 w-full">
                                    <svg class="w-4 h-4 flex-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                    </svg>
                                    <span class="font-medium">Reset</span>
                                </div>
                            </button>
                            
                            <button class="${buttonClasses(2, card)}"
                                    onclick="event.stopPropagation(); this.closest('[data-card-index]').dispatchEvent(new CustomEvent('reviewCard', {detail: {index: ${index}, quality: 2}}))">
                                <div class="flex items-center gap-2 w-full">
                                    <svg class="w-4 h-4 flex-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                    </svg>
                                    <span class="font-medium">Hard</span>
                                </div>
                            </button>
                            
                            <button class="${buttonClasses(3, card)}"
                                    onclick="event.stopPropagation(); this.closest('[data-card-index]').dispatchEvent(new CustomEvent('reviewCard', {detail: {index: ${index}, quality: 3}}))">
                                <div class="flex items-center gap-2 w-full">
                                    <svg class="w-4 h-4 flex-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                                    </svg>
                                    <span class="font-medium">Good</span>
                                </div>
                            </button>
                            
                            <button class="${buttonClasses(5, card)}"
                                    onclick="event.stopPropagation(); this.closest('[data-card-index]').dispatchEvent(new CustomEvent('reviewCard', {detail: {index: ${index}, quality: 5}}))">
                                <div class="flex items-center gap-2 w-full">
                                    <svg class="w-4 h-4 flex-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    <span class="font-medium">Learned</span>
                                </div>
                            </button>
                        </div>
                        
                        ${
                          card.tags?.length
                            ? `
                            <div class="flex flex-wrap gap-1 mt-3">
                                ${card.tags
                                  .map(
                                    (tag) => `
                                    <span class="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
                                        #${tag}
                                    </span>
                                `
                                  )
                                  .join("")}
                            </div>
                        `
                            : ""
                        }
                    </div>
                </div>
            `
              )
              .join("")}
        </div>
    `;

    // Initialize ink-mde editors with custom styles
    cards.forEach((card, index) => {
      const questionEditor = ink(
        this.shadowRoot.querySelector(`#question-${index}`),
        {
          doc: card.question,
          interface: {
            toolbar: false,
            attribution: false,
            readonly: false,
          },
          hooks: {
            afterUpdate: (doc) => {
              card.question = doc;
              this.storageHandler.saveToLocalStorage(isNoMessage);
            },
          },
        }
      );

      const _ = ink(this.shadowRoot.querySelector(`#answer-${index}`), {
        doc: card.answer,
        interface: {
          toolbar: false,
          attribution: false,
          readonly: false,
        },
        hooks: {
          afterUpdate: (doc) => {
            card.answer = doc;
            this.storageHandler.saveToLocalStorage(isNoMessage);
          },
        },
      });
    });

    // Event listeners remain the same
    const cardElements = flashcardList.querySelectorAll("[data-card-index]");
    cardElements.forEach((card) => {
      card.addEventListener("reviewCard", (e) => {
        this.reviewCard(e.detail.index, e.detail.quality);
      });
    });

    // Add event listeners for delete buttons
    const deleteButtons = flashcardList.querySelectorAll(".delete-card-btn");
    deleteButtons.forEach((button) => {
      button.addEventListener("click", (e) => {
        e.stopPropagation(); // Prevent card click event
        const index = parseInt(button.dataset.deleteIndex);
        if (confirm("Are you sure you want to delete this card?")) {
          this.deleteCard(index);
        }
      });
    });

    // Initialize drag handler after cards are rendered
    this.dragHandler.initialize();
  }

  editCard(index) {
    const card = this.chapterSets.get(parseInt(this.currentChapter.chapter))[
      index
    ];
    const modal = this.shadowRoot.querySelector("#editCardModal");

    // Ensure editors are initialized
    if (!this.questionEditor || !this.answerEditor) {
      this.initializeEditors();
    }

    // Set content using the correct ink-mde methods
    if (this.questionEditor && this.answerEditor) {
      try {
        // Use setValue instead of value
        this.questionEditor.setValue(card.question || "");
        this.answerEditor.setValue(card.answer || "");
      } catch (error) {
        console.error("Error setting editor values:", error);
      }
    } else {
      console.error("Editors not properly initialized");
    }

    modal.classList.remove("hidden");

    // Focus the question editor
    setTimeout(() => {
      try {
        this.questionEditor?.focus();
      } catch (error) {
        console.error("Error focusing editor:", error);
      }
    }, 100);
  }

  async reviewCard(index, quality) {
    const cardElement = this.shadowRoot.querySelector(
      `[data-card-index="${index}"]`
    );
    const clickedButton = cardElement?.querySelector(
      `button[onclick*="quality: ${quality}"]`
    );
    const originalButtonContent = clickedButton?.innerHTML;


    // Immediately update the button classes for visual feedback
    if (cardElement) {
      const allButtons = cardElement.querySelectorAll(
        'button[onclick*="quality:"]'
      );
      allButtons.forEach((btn) => {
        btn.classList.remove(
          "bg-blue-50",
          "dark:bg-blue-900/20",
          "ring-1",
          "ring-blue-500/20"
        );
        btn.classList.add("bg-gray-50", "dark:bg-zinc-700/50");
      });

      clickedButton?.classList.remove("bg-gray-50", "dark:bg-zinc-700/50");
      clickedButton?.classList.add(
        "bg-blue-50",
        "dark:bg-blue-900/20",
        "ring-1",
        "ring-blue-500/20"
      );
    }

    if (clickedButton) {
      clickedButton.innerHTML = `
            <svg class="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
        `;
      clickedButton.disabled = true;
    }

    try {
      const cards = this.chapterSets.get(parseInt(this.currentChapter.chapter));
      if (!cards || !cards[index]) {
        throw new Error("Card not found");
      }

      // Update the card's lastReviewQuality immediately in memory
      cards[index].lastReviewQuality = quality;

      // Convert plain object to FlashCard instance if needed
      let card = cards[index];
      if (!(card instanceof FlashCard)) {
        card = FlashCard.fromData(card);
        cards[index] = card;
      }

      // Apply review
      card.review(quality);

      if (this.storageHandler.inMemoryData.currentDeck) {
        // Find and update the card in currentDeck
        const currentDeckIndex =
          this.storageHandler.inMemoryData.currentDeck.findIndex(
            (c) => c.id === card.id
          );
        if (currentDeckIndex !== -1) {
          this.storageHandler.inMemoryData.currentDeck[currentDeckIndex] = card;
        }
      }

      this.immediateUpdate();

      // Save silently
      this.storageHandler.saveToLocalStorage();

      // Update UI immediately
      this.uiHandler.updateProgress();

      // Show popover based on quality
      let message, type;
      switch (quality) {
        case 0:
          message = "Card reset - You'll see this card again soon";
          type = "info";
          break;
        case 2:
          message = "Marked as hard - We'll review this more frequently";
          type = "info";
          break;
        case 3:
          message = "Good progress! Keep going!";
          type = "success";
          break;
        case 5:
          message = "Perfect! Well memorized!";
          type = "success";
          break;
      }
      showPopover(this.shadowRoot, message, type);

      // Update visualization if needed
      if (this.visualization) {
        this.visualization.update(this.flashcards);
      }

      // Force an immediate progress update
      const stats = this.storageHandler.getCurrentChapterStats();
      const progressBar = this.shadowRoot.querySelector("#progressBar");
      if (progressBar) {
        progressBar.style.width = `${stats.percentage}%`;
        progressBar.setAttribute("aria-valuenow", stats.percentage);
      }

      // Update progress text
      const progressText = this.shadowRoot.querySelector("#progressText");
      if (progressText) {
        progressText.textContent = `${stats.learned}/${stats.total} cards learned (${stats.percentage}%)`;
      }
    } catch (error) {
      console.error("Error during card review:", error);
      showPopover(this.shadowRoot, "Error updating card progress", "error");
    } finally {
      // Restore button content
      if (clickedButton && originalButtonContent) {
        clickedButton.innerHTML = originalButtonContent;
        clickedButton.disabled = false;
      }
    }
  }



  deleteCard(index) {
    try {
      // Get current chapter's cards
      const cards = this.chapterSets.get(parseInt(this.currentChapter.chapter));
      if (!cards) return;

      const cardId = cards[index].id;

      // Remove the card
      cards.splice(index, 1);
      this.storageHandler.purgeCardFromMemory(cardId);
      // Save to storage
      this.storageHandler.saveToLocalStorage();

      // Show success message
      showPopover(this.shadowRoot, "Card deleted successfully", "success");

      // Update UI
      this.immediateUpdate();
    } catch (error) {
      console.error("Error deleting card:", error);
      showPopover(this.shadowRoot, "Error deleting card", "error");
    }
  }

  // Call this after loading data
  // async initialize() {
  //   // await this.storageHandler.initializeDatabase();
  // //  await  this.storageHandler.loadFromLocalStorage(true);
  //   // this.renderChapters();
  //   this.showAllCards(); // Show cards immediately

  // }

  initializeEditors() {
    // Initialize ink-mde editors
    const questionEditorElement = this.shadowRoot.querySelector(
      "#editQuestionEditor"
    );
    const answerEditorElement =
      this.shadowRoot.querySelector("#editAnswerEditor");

    if (questionEditorElement && !this.questionEditor) {
      this.questionEditor = ink(questionEditorElement, {
        placeholder: "Enter question...",
        shortcuts: true,
        toolbar: true,
        value: "", // Initial empty value
      });
    }

    if (answerEditorElement && !this.answerEditor) {
      this.answerEditor = ink(answerEditorElement, {
        placeholder: "Enter answer...",
        shortcuts: true,
        toolbar: true,
        value: "", // Initial empty value
      });
    }
  }

  async saveCard(question="") {
    const loadingId =
      this.storageHandler.notificationHandler.showLoadingState(
        "Saving card..."
      );

    try {
      const question = await this.questionEditor.getDoc();
      const answer = await this.answerEditor.getDoc();

      if (!question || !answer) {
        this.storageHandler.notificationHandler.updateNotification(
          loadingId,
          "Question and answer are required",
          "error"
        );
        return;
      }

      const questionTags = this.extractTags(question);
      const answerTags = this.extractTags(answer);
      const allTags = [...new Set([...questionTags, ...answerTags])]; // Combine and deduplicate tags

      const card = new FlashCard(question, answer);
      let chapterNum = parseInt(this.currentChapter.chapter);
      if(!chapterNum) {
        chapterNum = 0;
      }

      // Ensure storage handler knows the current chapter before saving
      if (!this.storageHandler.inMemoryData.currentChapter || 
          this.storageHandler.inMemoryData.currentChapter.chapter !== this.currentChapter.chapter) {
        this.storageHandler.inMemoryData.currentChapter = this.currentChapter;
        this.storageHandler.inMemoryData.currentDeck = this.chapterSets.get(this.currentChapter.chapter) || [];
      }

      // Update chapterSets
      if (!this.chapterSets.has(chapterNum)) {
        this.chapterSets.set(chapterNum, []);
      }
      this.chapterSets.get(chapterNum).push(card);

      // Update storageHandler's inMemoryData
      if (!this.storageHandler.inMemoryData.currentDeck) {
        this.storageHandler.inMemoryData.currentDeck = [];
      }
      this.storageHandler.inMemoryData.currentDeck.push(card);

      await this.storageHandler.createNewCard(question, answer, allTags);

      // Clear the editors
      await this.questionEditor.update("");
      await this.answerEditor.update("");

      // Show success notification
      this.storageHandler.notificationHandler.updateNotification(
        loadingId,
        "Card saved successfully!",
        "success"
      );

      // Immediate UI updates
      await this.immediateUpdate();

      // Close the modal if it exists
      const modal = this.shadowRoot.querySelector("#newCardModal");
      if (modal) {
        modal.classList.add("hidden");
      }
      return true; // Return success
    } catch (error) {
      console.error("Error saving card:", error);
      this.storageHandler.notificationHandler.updateNotification(
        loadingId,
        "Error saving card",
        "error"
      );
      return false; // Return failure
    }
  }

  extractTags(content) {
    // Match hashtags that:
    // - Start with #
    // - Followed by letters, numbers, underscores or dashes
    // - Not preceded by other word characters (to avoid matching within URLs etc)
    const tagRegex = /(?:^|\s)#([a-zA-Z0-9_-]+)/g;
    const matches = [...content.matchAll(tagRegex)];
    return [...new Set(matches.map((match) => match[1]))]; // Remove duplicates
  }

  showCard(chapterNum, cardIndex) {
    // Use the full deck from storage handler
    const cards = this.storageHandler.inMemoryData.currentDeck;
    if (!cards || cardIndex >= cards.length) return;

    // Get current theme for modal
    const hostElement = this.shadowRoot.host;
    const currentTheme = hostElement?.getAttribute('data-socratiq-theme') || 'light';
    const isDark = currentTheme === 'dark';
    
    console.log('[CARD REVIEW MODAL DEBUG] Creating modal with theme:', {
      currentTheme: currentTheme,
      isDark: isDark
    });

    let modal = this.shadowRoot.querySelector("#cardReviewModal");

    // Store current review state with reference to storage
    this.currentReview = {
      chapterNum: parseInt(chapterNum),
      currentIndex: cardIndex,
      cards: this.storageHandler.inMemoryData.currentDeck, // Direct reference to storage
      showingQuestion: true,
    };

    // Rest of your existing modal creation and event handling code...

    // Update the button styles based on current card's state
    const updateButtonStates = () => {
      const currentCard =
        this.currentReview.cards[this.currentReview.currentIndex];
      const buttons = modal.querySelectorAll('button[onclick*="reviewCard"]');
      buttons.forEach((btn) => {
        const quality = parseInt(
          btn.getAttribute("onclick").match(/quality: (\d+)/)[1]
        );
        
        // Reset to default state
        const isSelected = currentCard.lastReviewQuality === quality;
        const bgColor = isSelected 
          ? (isDark ? 'rgba(30, 58, 138, 0.2)' : '#dbeafe')
          : (isDark ? 'rgba(55, 65, 81, 0.5)' : '#f9fafb');
        const textColor = isDark ? '#f9fafb' : '#111827';
        const borderStyle = isSelected ? '1px solid rgba(59, 130, 246, 0.2)' : 'none';
        
        btn.style.backgroundColor = bgColor + ' !important';
        btn.style.color = textColor + ' !important';
        btn.style.border = borderStyle + ' !important';
      });
    };

    // Call updateButtonStates after modal is created and DOM is updated
    requestAnimationFrame(() => {
      updateButtonStates();
    });

    // Add to existing navigation handlers
    this.shadowRoot
      .querySelector("#prevCardBtn")
      ?.addEventListener("click", () => {
        if (this.currentReview && this.currentReview.currentIndex > 0) {
          this.currentReview.currentIndex--;
          this.currentReview.showingQuestion = true;
          const card =
            this.currentReview.cards[this.currentReview.currentIndex];
          this.reviewEditor?.update(card.question);
          this.updateNavigationButtons();
          updateButtonStates(); // Update button states for new card
        }
      });

    this.shadowRoot
      .querySelector("#nextCardBtn")
      ?.addEventListener("click", () => {
        if (
          this.currentReview &&
          this.currentReview.currentIndex < this.currentReview.cards.length - 1
        ) {
          this.currentReview.currentIndex++;
          this.currentReview.showingQuestion = true;
          const card =
            this.currentReview.cards[this.currentReview.currentIndex];
          this.reviewEditor?.update(card.question);
          this.updateNavigationButtons();
          updateButtonStates(); // Update button states for new card
        }
      });

    // Rest of your existing code...

    // Create the modal if it doesn't exist
    if (!modal) {
      const modalHTML = `
            <div id="cardReviewModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center" style="backdrop-filter: blur(100px) !important;">
                <div class="rounded-lg p-6 w-full max-w-2xl relative flex flex-col max-h-[90vh]" 
                     style="background-color: ${isDark ? '#1f2937' : '#ffffff'} !important; color: ${isDark ? '#f9fafb' : '#111827'} !important;">
             

                    <button id="closeReviewModal" class="absolute top-4 right-4 flex items-center rounded-md hover:bg-gray-100 dark:hover:bg-zinc-700" 
                            style="color: ${isDark ? '#9ca3af' : '#6b7280'} !important;">             
    <kbd class="mt-1 px-1 text-xs transition-colors duration-150 rounded"
         style="background-color: ${isDark ? '#374151' : '#f3f4f6'} !important; color: ${isDark ? '#d1d5db' : '#6b7280'} !important; border: 1px solid ${isDark ? '#4b5563' : '#d1d5db'} !important; box-shadow: none !important;">
        ESC
    </kbd>
</button>

                    <!-- Delete button -->
                    <div class="absolute top-5 right-16 z-10">
                        <button id="deleteCardBtn" class="flex items-center rounded-md hover:bg-gray-100 dark:hover:bg-zinc-700"
                                style="color: ${isDark ? '#9ca3af' : '#6b7280'} !important;">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5">
                                <path stroke-linecap="round" stroke-linejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
                            </svg>
                        </button>
                    </div>
                    

                    <!-- Content Area -->
                    <div class="flex-1 overflow-y-auto min-h-0">
                        <div id="reviewContent" class="min-h-[200px] mb-4"></div>
                    </div>

                    <!-- Review Buttons -->
                    <div class="flex gap-2 mt-4">
                        <button class="px-3 py-2 text-xs rounded flex flex-col items-start flex-1 min-w-0 hover:bg-gray-100 dark:hover:bg-zinc-600/50"
                                style="background-color: ${cards[cardIndex]?.lastReviewQuality === 0 ? (isDark ? 'rgba(30, 58, 138, 0.2)' : '#dbeafe') : (isDark ? 'rgba(55, 65, 81, 0.5)' : '#f9fafb')} !important; color: ${isDark ? '#f9fafb' : '#111827'} !important; ${cards[cardIndex]?.lastReviewQuality === 0 ? 'border: 1px solid rgba(59, 130, 246, 0.2) !important;' : ''}"
                                onclick="event.stopPropagation(); this.closest('#cardReviewModal').dispatchEvent(new CustomEvent('reviewCard', {detail: {index: ${cardIndex}, quality: 0}}))">
                            <div class="flex items-center gap-2">
                                <svg class="w-4 h-4 flex-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                </svg>
                                <span class="font-medium">Reset</span>
                            </div>
                        </button>

                        <button class="px-3 py-2 text-xs rounded flex flex-col items-start flex-1 min-w-0 hover:bg-gray-100 dark:hover:bg-zinc-600/50"
                                style="background-color: ${cards[cardIndex]?.lastReviewQuality === 2 ? (isDark ? 'rgba(30, 58, 138, 0.2)' : '#dbeafe') : (isDark ? 'rgba(55, 65, 81, 0.5)' : '#f9fafb')} !important; color: ${isDark ? '#f9fafb' : '#111827'} !important; ${cards[cardIndex]?.lastReviewQuality === 2 ? 'border: 1px solid rgba(59, 130, 246, 0.2) !important;' : ''}"
                                onclick="event.stopPropagation(); this.closest('#cardReviewModal').dispatchEvent(new CustomEvent('reviewCard', {detail: {index: ${cardIndex}, quality: 2}}))">
                            <div class="flex items-center gap-2">
                                <svg class="w-4 h-4 flex-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                </svg>
                                <span class="font-medium">Hard</span>
                            </div>
                        </button>

                        <button class="px-3 py-2 text-xs rounded flex flex-col items-start flex-1 min-w-0 hover:bg-gray-100 dark:hover:bg-zinc-600/50"
                                style="background-color: ${cards[cardIndex]?.lastReviewQuality === 3 ? (isDark ? 'rgba(30, 58, 138, 0.2)' : '#dbeafe') : (isDark ? 'rgba(55, 65, 81, 0.5)' : '#f9fafb')} !important; color: ${isDark ? '#f9fafb' : '#111827'} !important; ${cards[cardIndex]?.lastReviewQuality === 3 ? 'border: 1px solid rgba(59, 130, 246, 0.2) !important;' : ''}"
                                onclick="event.stopPropagation(); this.closest('#cardReviewModal').dispatchEvent(new CustomEvent('reviewCard', {detail: {index: ${cardIndex}, quality: 3}}))">
                            <div class="flex items-center gap-2">
                                <svg class="w-4 h-4 flex-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                                </svg>
                                <span class="font-medium">Good</span>
                            </div>
                        </button>

                        <button class="px-3 py-2 text-xs rounded flex flex-col items-start flex-1 min-w-0 hover:bg-gray-100 dark:hover:bg-zinc-600/50"
                                style="background-color: ${cards[cardIndex]?.lastReviewQuality === 5 ? (isDark ? 'rgba(30, 58, 138, 0.2)' : '#dbeafe') : (isDark ? 'rgba(55, 65, 81, 0.5)' : '#f9fafb')} !important; color: ${isDark ? '#f9fafb' : '#111827'} !important; ${cards[cardIndex]?.lastReviewQuality === 5 ? 'border: 1px solid rgba(59, 130, 246, 0.2) !important;' : ''}"
                                onclick="event.stopPropagation(); this.closest('#cardReviewModal').dispatchEvent(new CustomEvent('reviewCard', {detail: {index: ${cardIndex}, quality: 5}}))">
                            <div class="flex items-center gap-2">
                                <svg class="w-4 h-4 flex-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <span class="font-medium">Memorized</span>
                            </div>
                        </button>
                    </div>

                    <!-- Navigation Controls -->
                    <div class="flex justify-between mt-4">
                        <button id="prevCardBtn" class="px-4 py-2 text-sm rounded-md hover:bg-gray-200 dark:hover:bg-zinc-600"
                                style="background-color: ${isDark ? '#374151' : '#f3f4f6'} !important; color: ${isDark ? '#e5e7eb' : '#374151'} !important;">
                            ← Previous
                        </button>
                        <button id="flipCardBtn" class="px-4 py-2 text-sm rounded-md hover:bg-blue-200 dark:hover:bg-blue-800/40 flex items-center gap-2"
                                style="background-color: ${isDark ? 'rgba(30, 58, 138, 0.3)' : '#dbeafe'} !important; color: ${isDark ? '#93c5fd' : '#1d4ed8'} !important;">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M7.5 21 3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
                            </svg>
                            <span>Show Answer</span>
                        </button>
                        <button id="nextCardBtn" class="px-4 py-2 text-sm rounded-md hover:bg-gray-200 dark:hover:bg-zinc-600"
                                style="background-color: ${isDark ? '#374151' : '#f3f4f6'} !important; color: ${isDark ? '#e5e7eb' : '#374151'} !important;">
                            Next →
                        </button>
                    </div>
                </div>
            </div>
        `;
      const fragment = document
        .createRange()
        .createContextualFragment(modalHTML);
      this.shadowRoot.appendChild(fragment);
      modal = this.shadowRoot.querySelector("#cardReviewModal");

      // Add event listeners after creating the modal
      modal.addEventListener("reviewCard", (event) => {
        const { index, quality } = event.detail;

        // Only update buttons for the current card
        const allButtons = modal.querySelectorAll(
          `button[onclick*="index: ${index}"]`
        );
        allButtons.forEach((btn) => {
          // Reset to default state
          const bgColor = isDark ? 'rgba(55, 65, 81, 0.5)' : '#f9fafb';
          const textColor = isDark ? '#f9fafb' : '#111827';
          btn.style.backgroundColor = bgColor + ' !important';
          btn.style.color = textColor + ' !important';
          btn.style.border = 'none !important';
        });

        const clickedButton = modal.querySelector(
          `button[onclick*="index: ${index}"][onclick*="quality: ${quality}"]`
        );
        if (clickedButton) {
          // Set selected state
          const bgColor = isDark ? 'rgba(30, 58, 138, 0.2)' : '#dbeafe';
          const textColor = isDark ? '#f9fafb' : '#111827';
          const borderStyle = '1px solid rgba(59, 130, 246, 0.2)';
          clickedButton.style.backgroundColor = bgColor + ' !important';
          clickedButton.style.color = textColor + ' !important';
          clickedButton.style.border = borderStyle + ' !important';
        }

        // Process the review
        this.reviewCard(index, quality);
      });

      // Rest of your existing event listeners...
    }

    // ... rest of existing showCard code ...

    // Store current review state
    this.currentReview = {
      chapterNum: parseInt(chapterNum),
      currentIndex: cardIndex,
      cards: this.storageHandler.inMemoryData.currentDeck,
      showingQuestion: true, // Track which side we're showing
    };

    const card = this.currentReview.cards[cardIndex];

    // Initialize or update editor
    if (!this.reviewEditor) {
      this.reviewEditor = ink(this.shadowRoot.querySelector("#reviewContent"), {
        doc: card.question,
        interface: {
          toolbar: false,
          attribution: false,
        },
        readOnly: true,
      });
    } else {
      this.reviewEditor.update(card.question);
    }
    const flipBtn = this.shadowRoot.querySelector("#flipCardBtn");

    if (flipBtn) {
      // Set initial state
      this.currentReview = {
        chapterNum: parseInt(chapterNum),
        currentIndex: cardIndex,
        cards: this.storageHandler.inMemoryData.currentDeck,
        showingQuestion: true,
      };

      // Set initial button content
      flipBtn.innerHTML = `
            <div class="flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M7.5 21 3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
                </svg>
                <span>Show Answer</span>
            </div>
        `;

      // Remove any existing listeners to prevent duplicates
      const newFlipBtn = flipBtn.cloneNode(true);
      flipBtn.parentNode.replaceChild(newFlipBtn, flipBtn);

      // Add click handler
      newFlipBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();

        if (!this.currentReview || !this.reviewEditor) {
          console.error("Missing required state:", {
            currentReview: !!this.currentReview,
            reviewEditor: !!this.reviewEditor,
          });
          return;
        }

        // Toggle state
        this.currentReview.showingQuestion =
          !this.currentReview.showingQuestion;

        const currentCard =
          this.currentReview.cards[this.currentReview.currentIndex];
        const content = this.currentReview.showingQuestion
          ? currentCard.question
          : currentCard.answer;
        const buttonText = this.currentReview.showingQuestion
          ? "Show Answer"
          : "Show Question";

        // Update content
        this.reviewEditor.update(content);

        // Update button
        newFlipBtn.innerHTML = `
                <div class="flex items-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M7.5 21 3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
                    </svg>
                    <span>${buttonText}</span>
                </div>
            `;
      });
    }

    // Add review button event listeners
    const reviewButtons = {
      resetBtn: { quality: 0, text: "Reset" },
      hardBtn: { quality: 2, text: "Hard" },
      goodBtn: { quality: 3, text: "Good" },
      learnedBtn: { quality: 5, text: "Learned" },
    };

    Object.entries(reviewButtons).forEach(([btnId, { quality, text }]) => {
      this.shadowRoot
        .querySelector(`#${btnId}`)
        ?.addEventListener("click", () => {
          this.reviewCard(this.currentReview.currentIndex, quality);
          showPopover(
            this.shadowRoot,
            `Card marked as ${text}`,
            quality === 0 ? "error" : "success"
          );
          this.immediateUpdate();
        });
    });

    // Inside showCard method, after creating the modal
    modal.addEventListener("reviewCard", (event) => {
      const { index, quality } = event.detail;
      this.reviewCard(index, quality);

      // Show feedback based on quality
      let message = "";
      switch (quality) {
        case 0:
          message = "Card reset";
          break;
        case 2:
          message = "Marked as hard";
          break;
        case 3:
          message = "Marked as good";
          break;
        case 5:
          message = "Marked as memorized";
          break;
      }

      showPopover(
        this.shadowRoot,
        message,
        quality === 0 ? "warning" : "success"
      );

      // Move to next card if not the last one
      const cards = this.chapterSets.get(
        parseInt(this.currentReview.chapterNum)
      );
      if (this.currentReview.currentIndex < cards.length - 1) {
      } else {
        showPopover(this.shadowRoot, "Review completed!", "success");
      }
    });

    // Update navigation handlers
    this.shadowRoot
      .querySelector("#prevCardBtn")
      ?.addEventListener("click", () => {
        if (this.currentReview && this.currentReview.currentIndex > 0) {
          this.currentReview.currentIndex--;
          this.currentReview.showingQuestion = true;
          const card =
            this.currentReview.cards[this.currentReview.currentIndex];
          this.reviewEditor?.update(card.question);
          this.updateNavigationButtons();
        }
      });

    this.shadowRoot
      .querySelector("#nextCardBtn")
      ?.addEventListener("click", () => {
        if (
          this.currentReview &&
          this.currentReview.currentIndex < this.currentReview.cards.length - 1
        ) {
          this.currentReview.currentIndex++;
          this.currentReview.showingQuestion = true;
          const card =
            this.currentReview.cards[this.currentReview.currentIndex];
          this.reviewEditor?.update(card.question);
          this.updateNavigationButtons();
        }
      });

    // Add updateNavigationButtons method
    this.updateNavigationButtons = () => {
      const prevBtn = this.shadowRoot.querySelector("#prevCardBtn");
      const nextBtn = this.shadowRoot.querySelector("#nextCardBtn");

      if (prevBtn) {
        prevBtn.disabled = this.currentReview.currentIndex === 0;
        prevBtn.classList.toggle(
          "opacity-50",
          this.currentReview.currentIndex === 0
        );
      }

      if (nextBtn) {
        const isLastCard =
          this.currentReview.currentIndex ===
          this.currentReview.cards.length - 1;
        nextBtn.disabled = isLastCard;
        nextBtn.classList.toggle("opacity-50", isLastCard);
      }
    };

    // Initial button state update
    this.updateNavigationButtons();

    // Update close button handler
    const closeBtn = this.shadowRoot.querySelector("#closeReviewModal");
    if (closeBtn) {
      closeBtn.addEventListener("click", (e) => {
        this.handleModalClose(e);
      });
    }

    // Add delete button handler
    const deleteBtn = this.shadowRoot.querySelector("#deleteCardBtn");
    if (deleteBtn) {
      deleteBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        try {
          const currentCard = this.currentReview?.cards?.[this.currentReview?.currentIndex];
          console.log("Delete button clicked. Current card:", currentCard);
          console.log("Current review state:", {
            hasReview: !!this.currentReview,
            currentIndex: this.currentReview?.currentIndex,
            cardsLength: this.currentReview?.cards?.length,
            cardId: currentCard?.id
          });
          
          if (currentCard && currentCard.id) {
            const questionText = typeof currentCard.question === 'string' 
              ? currentCard.question.substring(0, 50) 
              : 'this card';
            if (confirm(`Are you sure you want to delete this card?\\n\\nQuestion: ${questionText}...`)) {
              const index = this.currentReview?.currentIndex;
              console.log("Deleting card at index:", index);
              // Use the working index-based deleteCard method
              this.deleteCard(index);
              
              // Handle modal state after deletion
              if (this.currentReview && this.currentReview.cards) {
                if (this.currentReview.cards.length === 0) {
                  // No cards left, close modal
                  const modal = this.shadowRoot.querySelector("#cardReviewModal");
                  if (modal) modal.classList.add("hidden");
                  this.currentReview = null;
                } else {
                  // Adjust index if needed (card was deleted, so if we were at the end, move back)
                  if (this.currentReview.currentIndex >= this.currentReview.cards.length) {
                    this.currentReview.currentIndex = Math.max(0, this.currentReview.cards.length - 1);
                  }
                  // Update displayed card
                  const card = this.currentReview.cards[this.currentReview.currentIndex];
                  if (card && this.reviewEditor) {
                    this.reviewEditor.update(card.question);
                    this.updateNavigationButtons?.();
                  }
                }
              }
            }
          } else {
            console.error("Cannot delete card: card not found or missing ID", {
              currentCard,
              hasId: !!currentCard?.id,
              currentIndex: this.currentReview?.currentIndex,
              cardsLength: this.currentReview?.cards?.length
            });
            alert("Cannot delete card: card information not available.");
          }
        } catch (error) {
          console.error("Error deleting card:", error);
          console.error("Error stack:", error.stack);
          alert("Error deleting card. Please try again.");
        }
      });
    }
  }

  // Add this new method for switching tags
  async switchTag(tag) {
    console.log(`[Modal] switchTag called with: "${tag}"`);
    try {
      if (!this.storageHandler) {
          console.error("[Modal] Storage handler is missing in switchTag");
          return;
      }
      console.log(`🏷️ Switching to tag: "${tag}"`);
      const cards = await this.storageHandler.getCardsByTag(tag);
      console.log(`📚 Found ${cards ? cards.length : 0} cards for tag "${tag}"`);
      
      // Use a special chapter ID for tag view that won't conflict with normal chapters
      const TAG_VIEW_CHAPTER = 999999;
      
      this.chapterSets.set(TAG_VIEW_CHAPTER, cards);
      
      const chapter = {
        chapter: TAG_VIEW_CHAPTER,
        title: `Tag: #${tag}`
      };
      
      this.currentChapter = chapter;
      this.flashcards = cards;
      
      // Sync with storage handler
      this.storageHandler.inMemoryData.currentChapter = chapter;
      this.storageHandler.inMemoryData.currentDeck = cards;
      
      this.showAllCards();
      console.log('✅ Tag view updated');
      
      // Update UI highlighting
      const tagList = this.shadowRoot.querySelector('#tagList');
      if (tagList) {
        const themeClasses = this.themeManager?.getThemeClasses() || {};
        const activeClasses = themeClasses.buttonActive || "bg-blue-50 dark:bg-blue-900/30";
        
        tagList.querySelectorAll('button').forEach(btn => {
          if (btn.dataset.tag === tag) {
            activeClasses.split(' ').forEach(cls => btn.classList.add(cls));
          } else {
            activeClasses.split(' ').forEach(cls => btn.classList.remove(cls));
          }
        });
      }
      
      // Clear chapter highlighting
      const chapterList = this.shadowRoot.querySelector('#chapterList');
      if (chapterList) {
        const themeClasses = this.themeManager?.getThemeClasses() || {};
        const activeClasses = themeClasses.buttonActive || "bg-blue-50 dark:bg-blue-900/30";
        
        chapterList.querySelectorAll('button').forEach(btn => {
          activeClasses.split(' ').forEach(cls => btn.classList.remove(cls));
        });
      }
      
    } catch (error) {
      console.error("Error switching to tag:", error);
    }
  }

  // Add this new method for immediate updates
  async immediateUpdate(isNoMessage=false) {
    if (!this.storageHandler) {
      console.error("Storage handler not initialized");
      return;
    }

    this.cleanupChapterSets();
    
    // Storage handler will handle appropriate storage method
    const data = await this.storageHandler.loadFromLocalStorage(isNoMessage);

    if (data) {
        this.currentChapter = data.currentChapter;
        this.chapterSets = data.chapterSets;
        this.flashcards = data.flashcards;
    }

    // Update left panel
    this.renderChapters();

    // Update main card view
    this.showAllCards();

    // Render tags
    this.renderTags();

    // Force progress update
    if (this.uiHandler) {
      requestAnimationFrame(() => {
        this.uiHandler.updateProgress();
      });
    }
  }

  getInitialTagCount() {
    const tagSet = new Set();
    this.chapterSets.forEach((cards) => {
      cards.forEach((card) => {
        if (card.tags) {
          card.tags.forEach((tag) => tagSet.add(tag));
        }
      });
    });
    return tagSet.size;
  }

  // Add smooth scrolling for stats and visualizations
  showStats() {
    const statsSection = this.shadowRoot.querySelector("#statsSection");
    if (statsSection) {
      statsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  showVisualizations() {
    const visualizationsSection = this.shadowRoot.querySelector(
      "#visualizationsSection"
    );
    if (visualizationsSection) {
      visualizationsSection.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }

  // Debounce helper
  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  // Separate UI update logic
  async updateUI() {
    const updates = [
      this.renderChapters(),
      this.renderTags(),
      this.uiHandler.updateProgress(),
    ];

    // Update in batches
    for (let i = 0; i < updates.length; i += 2) {
      const batch = updates.slice(i, i + 2);
      await Promise.all(
        batch.map((update) => {
          try {
            return Promise.resolve(update);
          } catch (error) {
            console.error("Error in UI update:", error);
            return Promise.resolve();
          }
        })
      );
      // Small delay between batches
      await new Promise((resolve) => setTimeout(resolve, 16)); // ~1 frame
    }
  }

  // Replace immediateUpdate with this
  scheduleUpdate() {
    this.debouncedUpdate();
  }



  // Add this helper method to the class
  getButtonClasses(quality, cardIndex) {
    const baseClasses =
      "px-3 py-2 text-xs rounded flex flex-col items-start flex-1 min-w-0";
    const currentCard = this.currentReview?.cards[cardIndex];
    const isLastUsed = currentCard?.lastReviewQuality === quality;

    return `${baseClasses} ${
      isLastUsed
        ? "bg-blue-50 dark:bg-blue-900/20 ring-1 ring-blue-500/20"
        : "bg-gray-50 hover:bg-gray-100 dark:bg-zinc-700/50 dark:hover:bg-zinc-600/50"
    }`;
  }

  // Show delete confirmation
  async showDeleteConfirmation(chapterNum) {
    const chapter = this.chapterSets.get(chapterNum);
    if (!chapter) return;

    const chapterTitle = chapter[0]?.title || `Chapter ${chapterNum}`;
    const cardCount = chapter.length;

    const message = `Are you sure you want to delete "${chapterTitle}"?\n\nThis will permanently delete the deck and all ${cardCount} cards in it.\n\nThis action cannot be undone.`;
    
    if (confirm(message)) {
      try {
        await this.deleteChapter(chapterNum);
        this.renderChapters(); // Refresh the chapter list
      } catch (error) {
        console.error('Error deleting chapter:', error);
        alert('Failed to delete deck. Please try again.');
      }
    }
  }

  // Delete chapter and all its cards
  async deleteChapter(chapterNum) {
    try {
      if (!this.storageHandler) {
        throw new Error('Storage handler not available');
      }

      // Delete from SQLite
      const { sql } = this.storageHandler.db;
      
      // Delete cards and tags first (foreign key constraints)
      await sql`DELETE FROM card_tags WHERE card_id IN (SELECT id FROM cards WHERE chapter = ${chapterNum})`;
      await sql`DELETE FROM cards WHERE chapter = ${chapterNum}`;
      await sql`DELETE FROM chapters WHERE chapter = ${chapterNum}`;

      // Update in-memory data
      this.chapterSets.delete(chapterNum);
      
      // If this was the current chapter, switch to another one
      if (this.currentChapter?.chapter === chapterNum) {
        const remainingChapters = Array.from(this.chapterSets.keys());
        if (remainingChapters.length > 0) {
          const newChapter = remainingChapters[0];
          this.currentChapter = { chapter: newChapter, title: `Chapter ${newChapter}` };
          await this.showAllCards();
        } else {
          this.currentChapter = null;
          this.flashcards = [];
        }
      }

      console.log(`✅ Deleted chapter ${chapterNum} and all its cards`);
    } catch (error) {
      console.error('Error deleting chapter:', error);
      throw error;
    }
  }

  // Add this helper method to handle smooth UI updates
  async updateCardContent(content) {
    if (!this.reviewEditor) return;

    // Use Promise to handle the update
    return new Promise((resolve) => {
      requestAnimationFrame(() => {
        this.reviewEditor.update(content);
        resolve();
      });
    });
  }

  setupModalEscapeHandler() {
    const handleEscapeKey = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        this.handleModalClose(event);
      }
    };

    // Store the handler reference for cleanup
    this._escapeHandler = handleEscapeKey;
    document.addEventListener("keydown", handleEscapeKey);
  }

  handleModalClose(event) {
    // Get all modals in priority order
    const cardReviewModal = this.shadowRoot.querySelector("#cardReviewModal");
    const addCardForm = this.shadowRoot.querySelector("#addCardForm");
    const innerModals = this.shadowRoot.querySelectorAll(".inner-modal");
    const mainModal = this.shadowRoot.querySelector("#spacedRepetitionModal");

    // If this is a click event on the main modal close button, close everything
    const isMainCloseClick = event?.target?.closest("#closeModal_sr");
    if (isMainCloseClick) {
      mainModal.classList.add("hidden");
      if (this._escapeHandler) {
        document.removeEventListener("keydown", this._escapeHandler);
        this._escapeHandler = null;
      }
      return;
    }

    // Handle inner modals in priority order
    if (cardReviewModal) {
      event?.preventDefault();
      event?.stopPropagation();
      cardReviewModal.remove();
      this.reviewEditor = null;
      return;
    }

    if (addCardForm && !addCardForm.classList.contains("hidden")) {
      event?.preventDefault();
      event?.stopPropagation();
      this.questionEditor?.update("");
      this.answerEditor?.update("");
      addCardForm.classList.add("hidden");
      return;
    }

    // Handle other inner modals
    for (const modal of innerModals) {
      if (!modal.classList.contains("hidden")) {
        event?.preventDefault();
        event?.stopPropagation();
        modal.classList.add("hidden");
        return;
      }
    }

    // Only close main modal if no inner modals are open and it's not from an inner modal action
    if (
      mainModal &&
      !cardReviewModal &&
      !addCardForm?.classList.contains("hidden")
    ) {
      mainModal.classList.add("hidden");
      if (this._escapeHandler) {
        document.removeEventListener("keydown", this._escapeHandler);
        this._escapeHandler = null;
      }
    }
  }

  addCustomCursor = async (editor) => {
    try {
      // Await the editor if needed
      if (editor.then) {
        editor = await editor;
      }

      // Wait for editor to be ready
      await new Promise((resolve) => setTimeout(resolve, 100));

      // Get the CodeMirror content area directly
      const cmContent = this.shadowRoot.querySelector(".cm-content");
      if (!cmContent) {
        return;
      }

      // Check if cursor already exists
      const existingCursor = this.shadowRoot.querySelector(".custom-cursor");
      if (existingCursor) {
        return;
      }

      // Create and add cursor styles if they don't exist
      if (!this.shadowRoot.querySelector("#cursor-styles")) {
        const style = document.createElement("style");
        style.id = "cursor-styles";
        style.textContent = `
                .cm-content {
                    position: relative !important;
                }
                .custom-cursor {
                    width: 2px !important;
                    height: 20px !important;
                    background-color: #ff0000 !important;
                    position: fixed !important;
                    left: 0 !important;
                    top: 0 !important;
                    z-index: 99999 !important;
                    pointer-events: none !important;
                    mix-blend-mode: difference !important;
                    will-change: transform !important;
                    transform: translate(var(--cursor-x, 0px), var(--cursor-y, 0px)) !important;
                }
            `;
        this.shadowRoot.appendChild(style);
      }

      // Create cursor element
      const cursor = document.createElement("div");
      cursor.className = "custom-cursor";

      // Add cursor to content area
      document.body.appendChild(cursor);

      // Track cursor position
      const updateCursorPosition = (e) => {
        const rect = cmContent.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        cursor.style.setProperty("--cursor-x", `${x}px`);
        cursor.style.setProperty("--cursor-y", `${y}px`);
      };

      // Add mouse move listener
      cmContent.addEventListener("mousemove", updateCursorPosition);

      // Store reference for cleanup
      this._cursorElement = cursor;
      this._updateCursorPosition = updateCursorPosition;
    } catch (error) {
      console.error("Error adding custom cursor:", error);
    }
  };

  // Add cleanup method
  cleanupCursor() {
    if (this._cursorElement) {
      this._cursorElement.remove();
      this._cursorElement = null;
    }
    if (this._updateCursorPosition) {
      const cmContent = this.shadowRoot.querySelector(".cm-content");
      if (cmContent) {
        cmContent.removeEventListener("mousemove", this._updateCursorPosition);
      }
      this._updateCursorPosition = null;
    }
  }
  cleanupDuplicateCardsInDOM() {
    // Track duplicates across all containers
    const seenContent = new Map(); // Changed to Map to track first occurrence location

    // First, scan all containers to identify original locations
    const scanForOriginals = (container, containerType) => {
      if (!container) return;
      const cards = Array.from(container.children);
      cards.forEach((cardElement) => {
        let question, answer;

        if (containerType === "chapter") {
          question =
            cardElement.querySelector("span:last-child")?.textContent?.trim() ||
            "";
          answer = ""; // Chapter list only shows questions
        } else {
          question = cardElement.querySelector("h3")?.textContent?.trim() || "";
          answer = cardElement.querySelector("p")?.textContent?.trim() || "";
        }

        const contentSignature = `${question}|||${answer}`.toLowerCase();

        // Only store the first occurrence
        if (!seenContent.has(contentSignature)) {
          seenContent.set(contentSignature, containerType);
        }
      });
    };

    // First scan chapter list (left panel) as it's the source of truth
    const chapterLists = this.shadowRoot.querySelectorAll(".chapter-cards");
    chapterLists.forEach((list) => scanForOriginals(list, "chapter"));

    // Then scan flashcard list
    const flashcardList = this.shadowRoot.querySelector("#flashcardList");
    scanForOriginals(flashcardList, "flashcard");

    // Then scan main card list
    const cardList = this.shadowRoot.querySelector("#cardList");
    scanForOriginals(cardList, "card");

    // Now remove duplicates based on where they should appear
    const removeDuplicates = (container, containerType) => {
      if (!container) return;
      const cards = Array.from(container.children);
      cards.forEach((cardElement) => {
        let question, answer;

        if (containerType === "chapter") {
          question =
            cardElement.querySelector("span:last-child")?.textContent?.trim() ||
            "";
          answer = ""; // Chapter list only shows questions
        } else {
          question = cardElement.querySelector("h3")?.textContent?.trim() || "";
          answer = cardElement.querySelector("p")?.textContent?.trim() || "";
        }

        const contentSignature = `${question}|||${answer}`.toLowerCase();
        const originalLocation = seenContent.get(contentSignature);

        // Remove if this isn't where the card should appear
        if (originalLocation && originalLocation !== containerType) {
          console.warn(`Removing duplicate card from ${containerType}:`, {
            question,
            answer,
            originalLocation,
          });
          cardElement.remove();
        }
      });
    };

    // Remove duplicates in reverse order
    removeDuplicates(cardList, "card");
    removeDuplicates(flashcardList, "flashcard");
    chapterLists.forEach((list) => removeDuplicates(list, "chapter"));
  }

  // Add this to your constructor or initialization method.. KAI
  setupVisualizationEvents() {
    this.shadowRoot.addEventListener("show-card", (e) => {
      const { chapterNum, cardIndex, card } = e.detail;
      this.showCard(chapterNum, cardIndex);
    });
  }

  // Add this method
  setupEventListeners() {
    if (this.floatingControlsHandler) {
      this.floatingControlsHandler.setupEventListeners();
    }
  }
}
