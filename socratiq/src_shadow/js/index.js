// index.js
import { htmlContent } from "../indexHtml.js";
import { styles } from "../cssStyles.js";

// import { updateMarkdownPreview } from "./components/markdown/markdown";
import { updateMarkdownPreview } from "./components/markdown/streamdown_markdown";

import { highlight, showQuiz } from "./components/highlight/highlight";
import {
  alert,
  scrollToBottom,
  scrollToTopSmooth,
  assignUniqueIdsToElementAndChildren,
  add_copy_paste_share_buttons,
  make_links_load_new_page,
  // decodeMessageFromURL,
  assignAndTrackUniqueIds,
  updateCloneAttributes,
  insertDiagramElements,
  removeSkeletonLoaders,
  insertAtProportionalPoints,
  injectOfflineWarning,
  extractQuestion,
} from "./libs/utils/utils.js";
import {
  SYSTEM_PROMPT_ORIG,
  QUERYAGENTPROCESS,
  PROGRESSREPORTAGENTPROCESS,
  GENERALAGENTPROCESS,
  QUIZAGENTPROCESS,
  RESEARCHAGENTPROCESS,
  getConfigs,
  NO_QUIZZES_MESSAGE,
  NO_NEW_QUIZZES_MESSAGE,
  OFFLINE_PROCESS,
} from "../configs/client.config.js";
import {
  query_agent,
  // query_agent_json,
} from "./libs/agents/chat_agent_yield.js";

import { query_agent_gemini_serverless, query_agent_groq_serverless } from "./libs/agents/chat_agent_json_serverless.js";

// import {
//   initiateMarkdown,
//   // convertMarkdownToHTML,
//   reinitializeEditableInputs,
// } from "./components/markdown/markdown";
import {
  initiateMarkdown,
  // convertMarkdownToHTML,
  reinitializeEditableInputs,
} from "./components/markdown/streamdown_markdown";
import { ini_quiz, reinitializeQuizButtons } from "./components/quiz/load_quiz.js";
import { SettingsManager, setupModal } from "./components/settings/settings.js";
// import {
//   injectLoad_chats,
//   setupModal_loadchats,
// } from "./components/settings/previous_conversations.js";
import {
  loadChat,
  retrieveRecentIDFromLocal,
  determineAndSaveChat,
  saveRecentIDToLocal,
  clearMessageContainer,
} from "./libs/utils/save_chats.js";
// import { inject_small_highlight_menu } from "./components/highlight_menu/index_highlight_menu.js";

// import {menu_slide, menu_slide_action} from './components/menu/open_close_menu.js'
import {
  get_message_element,
  get_reference_buttons,
} from "./libs/messaging/messages.js";
import { initiateResearch } from "./libs/agents/research.js";
// import {
//   extractPhrasesInBrackets,
//   extractWords,
// } from "./libs/agents/key_terms_summary.js";

import {
  injectProgress,
  addProgress,
  removeElementById,
  showProgressItem,
  toggleProgress,
} from "./components/progress/progress.js";
// import { check_menu_open } from "./libs/utils/check_menu_open.js";
import { createQuiz } from "./components/quiz/backupQuiz.js";
// import { collapsible_buttons } from "./components/collapsible_buttons/collapsable_buttons.js";
import {
  injectFeedback,
  openFeedback,
} from "./components/feedback/openFeedback.js";
import { showHelpModal } from "./components/help_modal/showHelpModal.js";
// import {
//   saveChatHistoryVector,
//   initiateMemory,
// } from "./libs/memory/initiate_mem.js";
// import { searchChatHistoryVector } from "./libs/memory/initiate_mem.js";
import { replaceElementsWithErrorNotice } from "./libs/utils/cleanup.js";
import { trackColorScheme } from "./libs/utils/trackColorScheme.js";
import { createThemeManager } from "./libs/utils/theme-manager.js";
// import {makeGenerativeAIPage} from './components/generative_page/genAI/initiateGenerativeAIPage.js'
// import {comingSoon, comingSoon2} from './components/generative_page/genAI/comingSoon/comingSoon.js'
// import {makeGenerativeAIPagev2} from './components/genAI_v2/initiate.js'
import { injectSvgButtons } from "./components/quizBtns/injectQuizBtn.js";
// import {initiateStudyBtn} from './components/quiz/studyPastQuizzes.js'
import { initStatsDisplay } from "./components/quiz/showQuizStats.js";
import {
  parseNavigation,
  updateLastVisitedChapter,
} from "./components/quiz/navParser.js";
import { highlight_click } from "./components/highlight_menu/send_text_highlight.js";
import { insertContextButton } from "./components/highlight_menu/context-button.js";
// import {initiate_clear_chats} from './components/settings/clear_chats.js'
import {
  initializeMenuSlide,
  shortcutKeys,
} from "./components/menu/open_close_menu.js";
import { get_text_ref } from "./components/quizBtns/create_text_ref.js";
import { setupAtMentions } from "./components/quizBtns/smartInput.js";
import { initializeAllMessageButtons } from "./components/settings/copy_download.js";
import { menu_slide_on } from "./components/menu/open_close_menu.js";
import { precomputeParagraphFingerprints, findSimilarParagraphsNonBlocking } from "./libs/agents/fuzzy_match.js";
import {
  injectLoad_chats,
  setupModal_loadchats,
} from "./components/settings/previous_conversations.js";
import { MessageObserver } from "./libs/utils/observers/message_observer.js";
import { initKnowledgeGraph } from "./components/visualizations/KnowledgeGraph.js";
import { fillPromptTemplate } from "./libs/utils/prompt_templater.js";
import { tryMultipleProvidersStream, callProviderSingle, callCloudflareAgent } from "./libs/agents/cloudflareAgent.js";
import { renderMermaidDiagram } from "./libs/diagram/mermaid.js";
import { renderMermaidFlow } from "./libs/diagram/mermaid_flow.js";
import {
  initializeSocratiqDB,
  forceInitializeSocratiqDB,
  setDBInstance,
  verifyStores,
} from "./libs/utils/indexDb.js";
import { getDBInstance } from "./libs/utils/indexDb.js";
// Add these imports at the top with your other imports
import { SpacedRepetitionModal } from "./components/spaced_repetition/spaced-repetition-modal-handler.js";

import { initializeImageZoom } from "./components/img_modals/imageModa.js";
import { initializeOnboarding } from "./components/onboarding/socratiq-onboarding.js";
import { enableTooltip } from "./components/tooltip/tooltip.js";
import { initializePersistentTooltips } from "./components/tooltip/persistentTooltip.js";
import { reinitializeButtonListeners } from './components/quiz/create_quiz_button_grp.js';
import { extractTOCAsync, extractTOCWithDebug, saveTOCToChapterMap, getTOCFromChapterMap, getAllChapterMapEntries, shouldExtractTOC, testTOCDatabaseConnection } from "./libs/utils/tocExtractor.js";

const loaderMarkdown = `::: loader
:::
`;
let token;
let getConfig_explain = getConfigs("explain");
let getConfig_quiz = getConfigs("quiz");
let getConfig_query = getConfigs("query");
let getConfig_progress_report = getConfigs("progress_report");
let getConfig_summative = getConfigs("summative");
let getConfig_mermaid = getConfigs("mermaid_diagram");
let accumulatedResponse = "";
let shadowRoot;
let chatId, topicOfConversation;
let main_topic_of_page = "";
let SYSTEM_PROMPT = SYSTEM_PROMPT_ORIG;
let llm_model;
let chatCount = 0;
let currentUrl;
let tempQuizTitle = "";

// let firstLoad = true;

// Add this at the top level of the file
let mermaidCache = new Map();
const diagramElementMap = new Map();

// Add this at the top level of the file
let dbInstance = null;

// Add these constants at the top of the file
const DB_VERSION_KEY = "socratiqDB_version";
const FIRST_VISIT_KEY = "socratiqDB_first_visit";
const REQUIRED_VERSION = "1.1"; // Update this when schema changes

async function initializeDatabase() {
  try {
    const currentVersion = localStorage.getItem(DB_VERSION_KEY);
    const firstVisit = localStorage.getItem(FIRST_VISIT_KEY);

    // Check if this is the first visit
    if (!firstVisit) {
      // Set first visit date
      localStorage.setItem(FIRST_VISIT_KEY, new Date().toISOString());
      localStorage.setItem(DB_VERSION_KEY, REQUIRED_VERSION);

      // Force fresh start for first-time visitors
      dbInstance = await forceInitializeSocratiqDB();
    }
    // Check if version needs updating
    else if (currentVersion !== REQUIRED_VERSION) {
      // Before forcing update, check if we have existing data
      const existingDb = await getDBInstance();
      const existingChats =
        (await existingDb?.getAll("tinyMLDB_chats").catch(() => [])) || [];

      // Force update
      dbInstance = await forceInitializeSocratiqDB();

      // Restore existing chats if any
      if (existingChats.length > 0) {
        const store = dbInstance.db
          .transaction(["tinyMLDB_chats"], "readwrite")
          .objectStore("tinyMLDB_chats");

        for (const chat of existingChats) {
          await store.put(chat);
        }
      }

      // Update version in localStorage
      localStorage.setItem(DB_VERSION_KEY, REQUIRED_VERSION);
    } else {
      dbInstance = await initializeSocratiqDB();
    }

    // Store the instance globally
    setDBInstance(dbInstance);

    // Verify stores were created
    const stores = await verifyStores();

    return dbInstance;
  } catch (error) {
    console.error("Failed to initialize database:", error);
    throw error;
  }
}

// Guard to prevent multiple main() executions
let mainExecuted = false;

function showSocratiqLoader() {
  const loader = document.createElement("div");
  loader.id = "socratiq-init-loader";
  loader.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.8);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: 999999;
    font-family: Inter, system-ui, -apple-system, sans-serif;
  `;

  const spinner = document.createElement("div");
  spinner.style.cssText = `
    width: 50px;
    height: 50px;
    border: 5px solid #f3f3f3;
    border-top: 5px solid #A51C30;
    border-radius: 50%;
    animation: socratiq-spin 1s linear infinite;
    margin-bottom: 20px;
  `;

  const style = document.createElement("style");
  style.textContent = `
    @keyframes socratiq-spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  `;
  document.head.appendChild(style);

  const text = document.createElement("div");
  text.textContent = "Initializing SocratiQ...";
  text.style.cssText = `
    font-size: 18px;
    font-weight: 600;
    color: #1f2328;
  `;

  loader.appendChild(spinner);
  loader.appendChild(text);
  document.body.appendChild(loader);
}

function hideSocratiqLoader() {
  const loader = document.querySelector("#socratiq-init-loader");
  if (loader) {
    loader.remove();
  }
}

async function main(isReEntry = false) {
  if (mainExecuted && !isReEntry) {
    return;
  }
  mainExecuted = true;
  
  try {
    if (isReEntry) {
      showSocratiqLoader();
    }

    // Initialize database as the first operation
    if (!isReEntry) {
      await initializeDatabase();
    }
    
    // Verify database initialization
    const db = await getDBInstance();
    console.log('🔍 Database initialization check:', {
      hasDB: !!db,
      hasDBInstance: !!db?.db,
      storeNames: db?.db ? Array.from(db.db.objectStoreNames) : 'N/A',
      chapterMapExists: db?.db ? db.db.objectStoreNames.contains('chapterMap') : false
    });

    // Load Collaborative Widget as a background process
    // try {
    //   await loadCollaborativeWidget();
    // } catch (error) {
    //   console.warn(`⚠️ Collaborative Widget failed to load and will be unavailable:`, error.message);
    //   // Don't re-throw; allow the main application to continue.
    // }
    
    // Test database write capability
    if (db && db.db && db.db.objectStoreNames.contains('chapterMap')) {
      try {
        const testEntry = {
          url: 'test-url',
          title: 'Test Title',
          tocData: [{ id: 'test', text: 'Test', level: 1 }],
          lastUpdated: new Date().toISOString(),
          headingCount: 1,
          pageType: 'test',
          domain: 'test.com'
        };
        await db.put('chapterMap', testEntry);
        
        // Clean up test entry
        await db.delete('chapterMap', 'test-url');
      } catch (error) {
        console.error('❌ Database write test failed:', error);
      }
    } else {
      console.error('❌ Database not ready for testing - chapterMap store missing');
    }
    
    // Test TOC-specific database connection
    const tocDbTest = await testTOCDatabaseConnection();

    // Always initialize the toggle button listener if present
    if (!isReEntry) {
      initializeSocratiqToggle();
    }

    if (!checkWidgetAccess()) {
      // Clean up URL if we are disabling/not loading the widget
      const params = new URLSearchParams(window.location.search);
      // If we have socratiq=false or widget_access=false (checked by checkWidgetAccess), 
      // we should clean the URL similar to how normalizeUrlAfterScroll does it.
      if (params.has('socratiq') || params.has('widget_access')) {
        const cleanUrl = window.location.pathname;
        window.history.replaceState({}, '', cleanUrl);
      }
      if (isReEntry) hideSocratiqLoader();
      return;
    }
    currentUrl = window.location.href;
    
    // Extract TOC from the page asynchronously (non-blocking)
    // Use smart extraction that checks if TOC is needed
    const mainCallId = Math.random().toString(36).substr(2, 9);
    
    // Check if we should extract TOC for this URL
    // Wait longer to ensure database is fully initialized
    setTimeout(async () => {
      try {
        // Ensure database is ready before TOC operations
        const db = await getDBInstance();
        if (!db) {
          console.error(`❌ [${mainCallId}] Database not ready for TOC operations`);
          return;
        }
        
        
        const currentUrl = window.location.href;
        const shouldExtract = await shouldExtractTOC(currentUrl);
        
        if (shouldExtract) {
          await extractTOCWithDebug();
        } else {
          // Load existing TOC data from chapterMap
          const existingTOC = await getTOCFromChapterMap(currentUrl);
          if (existingTOC) {
            window.pageTOC = existingTOC.tocData;
          }
        }
      } catch (error) {
        console.error(`❌ [${mainCallId}] Error in smart TOC extraction:`, error);
        console.error(`❌ [${mainCallId}] Error details:`, error.message, error.stack);
        // Fallback to regular extraction
        try {
          await extractTOCWithDebug();
        } catch (fallbackError) {
          console.error(`❌ [${mainCallId}] Fallback TOC extraction also failed:`, fallbackError);
        }
      }
    }, 500); // Increased delay from 100ms to 500ms
    // Store scroll parameters for later use after full initialization
    const params = new URLSearchParams(window.location.search);
    const x = params.get("x");
    const y = params.get("y");
    const scrollTo = params.get("scroll-to");
    
    // Store scroll parameters globally for use after initialization
    window.scrollParams = { x, y, scrollTo };

    // Check if the document is already loaded
    if (document.readyState === "loading") {
      // Loading hasn't finished yet
      document.addEventListener("DOMContentLoaded", async () => {
        await inject();
        if (isReEntry) hideSocratiqLoader();
      });
    } else {
      // `DOMContentLoaded` has already fired
      await inject();
      if (isReEntry) hideSocratiqLoader();
    }
  } catch (error) {
    console.error("Failed to initialize application:", error);
    if (isReEntry) hideSocratiqLoader();
  }
}

main();

function initializeSocratiqToggle() {
  // Function to set up the button when it's found
  const setupButton = () => {
    const toggleButton = document.querySelector("#socratiq-toggle");

    if (!toggleButton) {
      return false; // Button not found yet
    }

    // Sync button state with widget enabled state
    if (checkWidgetAccess()) {
      toggleButton.classList.add("on");
    } else {
      toggleButton.classList.remove("on");
    }

    // Instead of replacing the button, add our listener to the existing one
    toggleButton.addEventListener("click", () => {
      // Wait a brief moment for any other click handlers to complete
      setTimeout(() => {
        // Get button state after click (assuming the button class or text changes)
        // We check for 'on' class or 'ON' text to be safe
        const isEnabled = toggleButton.classList.contains("on") || toggleButton.textContent.toLowerCase().includes("on");

        if (isEnabled) {
          // Set cookie and initialize widget
          document.cookie =
            "socratiq=true; path=/; max-age=" + 60 * 60 * 24 * 365;

          // Set initial chapter in localStorage if not exists
          if (!localStorage.getItem('current_chapter')) {
              localStorage.setItem('current_chapter', JSON.stringify({
                  chapter: 0,
                  title: "Introduction",
                  is_current: 1
              }));
          }

          if (!document.querySelector("#widget-chat-container")) {
            main(true);
          }
        } else {
          // Remove cookie and widget
          document.cookie = "socratiq=true; path=/; max-age=0";
          const widgetContainer = document.querySelector(
            "#widget-chat-container"
          );
          if (widgetContainer) {
            widgetContainer.remove();
          }
        }
      }, 100);
    });

    return true; // Button setup complete
  };

  // Try to set up immediately first
  if (!setupButton()) {
    // If button not found, wait for DOM content to be loaded
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", setupButton);
    } else {
      // If DOMContentLoaded already fired, try a few more times
      let attempts = 0;
      const checkInterval = setInterval(() => {
        attempts++;
        if (setupButton() || attempts >= 5) {
          clearInterval(checkInterval);
        }
      }, 1000);
    }
  }
}

function checkWidgetAccess() {
  const params = new URLSearchParams(window.location.search);

  // Check for both parameters in the URL
  const widgetAccess = params.get("widget_access");
  const socratiqAccess = params.get("socratiq");

  // Prioritize URL parameters
  if (widgetAccess === "true" || socratiqAccess === "true") {
    // Set cookie for 1 year
    document.cookie = "socratiq=true; path=/; max-age=" + 60 * 60 * 24 * 365;
    return true;
  } else if (widgetAccess === "false" || socratiqAccess === "false") {
    // Remove cookie by setting max-age to 0
    document.cookie = "socratiq=true; path=/; max-age=0";
    return false;
  }

  // Check if the socratiq cookie exists if no URL param overrides it
  const cookies = document.cookie.split("; ");
  return cookies.some((cookie) => cookie.trim().startsWith("socratiq=true"));
}

async function inject() {
  try {
    console.log('[BUILD LOG] Current build: 2024-12-19-v2 - Theme fixes and New Deck button');
    const hostElement = document.createElement("div");
    hostElement.id = "widget-chat-container";
    // Add styles directly to the hostElement to ensure it has the highest z-index and is fixed
    hostElement.style.position = "fixed";
    hostElement.style.zIndex = "9999";
    document.body.appendChild(hostElement);
    shadowRoot = hostElement.attachShadow({ mode: "open" });

    const styleElement = document.createElement("style");

    styleElement.textContent = `
      ${styles}
    `;

    shadowRoot.appendChild(styleElement);

    shadowRoot.innerHTML += htmlContent;
    precomputeParagraphFingerprints();
    new MessageObserver(shadowRoot);
    setListeners();
    highlight(shadowRoot);
    initiateMarkdown(shadowRoot);
    // reinitializeEditableInputs(shadowRoot);
    setupModal(shadowRoot);
    initializeMenuSlide(shadowRoot);
    parseNavigation();

    injectFeedback(shadowRoot);
    openFeedback(shadowRoot);
    const settingsManager = new SettingsManager(shadowRoot); // Assuming `shadowRoot` is available
    settingsManager.init();
    SYSTEM_PROMPT = settingsManager.generateSystemPrompt();
    chatId = await loadContentAndRetrieveDetails(shadowRoot); // storeName='tinyMLDB_chats', elementId='message-container')
    showHelpModal(shadowRoot);
    trackColorScheme(shadowRoot);
    
    // Initialize theme manager for dynamic theme handling
    const themeManager = createThemeManager(shadowRoot);
    
    // Store theme manager on shadowRoot for easy access
    shadowRoot.themeManager = themeManager;
    
    // Fallback: If theme detection failed, force dark theme for dark backgrounds
    setTimeout(() => {
        const currentTheme = hostElement.getAttribute('data-socratiq-theme');
        
        if (currentTheme === 'light') {
            // Check if we're on a dark background by analyzing the parent page
            const bodyStyle = window.getComputedStyle(document.body);
            const bgColor = bodyStyle.getPropertyValue('background-color');
            const textColor = bodyStyle.getPropertyValue('color');
            
            // If text is light colored, assume dark background and switch to dark theme
            if (textColor.includes('rgb(230, 237, 243)') || textColor.includes('#e6edf3')) {
                hostElement.setAttribute("data-socratiq-theme", "dark");
                const widgetRoot = shadowRoot.querySelector('.socratiq-widget-root');
                if (widgetRoot) {
                    widgetRoot.setAttribute("data-socratiq-theme", "dark");
                }
                // Update theme manager
                themeManager.currentTheme = 'dark';
                themeManager.applyTheme();
            }
        }
    }, 100);

    injectSvgButtons(shadowRoot);
    get_text_ref();
    // initiateStudyBtn(shadowRoot)
    initStatsDisplay(shadowRoot);
    highlight_click(shadowRoot);
    insertContextButton(shadowRoot);
    // initiate_clear_chats(shadowRoot)
    shortcutKeys(shadowRoot);
    setupAtMentions(shadowRoot);
    ensureButtonsInitialized();
    // initializeAllMessageButtons(shadowRoot);
    // await handleSharedMessages();
    injectLoad_chats(shadowRoot);
    setupModal_loadchats(shadowRoot);
    initKnowledgeGraph(shadowRoot);
    // PHASE 1: Disabled automatic chapter detection from breadcrumbs
    // const breadcrumbs = document.querySelector(".quarto-page-breadcrumbs");
    // if (breadcrumbs) {
    //   updateLastVisitedChapter(breadcrumbs);
    // }

    SpacedRepetitionModal.initialize(shadowRoot);
    initializeImageZoom(shadowRoot);
    initializeOnboarding(shadowRoot);

    addToolTips();
    
    // Handle scroll-to-element functionality after full initialization
    // Try immediate scrolling without delay
    handleScrollToElement();
    
  } catch (error) {
    console.error("Failed to initialize application:", error);
  }
}

function addToolTips() {
  const spacedRepBtn = shadowRoot.querySelector('#spaced-repetition-btn');
  const newChatBtn = shadowRoot.querySelector('#new-chat-btn');
  enableTooltip(spacedRepBtn, "Create flashcards", shadowRoot);
  enableTooltip(newChatBtn, "Start a new chat", shadowRoot);

}

// Function to normalize URL by removing query parameters
function normalizeUrlAfterScroll() {
  setTimeout(() => {
    const currentUrl = window.location.href;
    const cleanUrl = window.location.pathname;
    
    // Only normalize if there are actually query parameters to remove
    if (currentUrl.includes('?')) {
      window.history.replaceState({}, '', cleanUrl);
    } else {
    }
  }, 10); // Wait 1 second to ensure scroll animation completes
}

// Function to handle scroll-to-element functionality with retry mechanism
function handleScrollToElement() {
  if (!window.scrollParams) {
    return;
  }

  const { x, y, scrollTo } = window.scrollParams;
  
  if (x && y) {
    // Scrolling to the captured position
    window.scrollTo(parseInt(x, 10), parseInt(y, 10));
    
    // Clean up URL by removing query parameters after successful scroll
    normalizeUrlAfterScroll();
    
    return;
  }

  if (scrollTo) {
    
    // Check how many elements with data-fuzzy-id exist
    const allFuzzyElements = document.querySelectorAll('[data-fuzzy-id]');
    
    // Function to attempt scrolling with retry mechanism
    const attemptScroll = (attempts = 0, maxAttempts = 15) => {
      const startTime = performance.now();
      const targetElement = document.querySelector(`[data-fuzzy-id="${scrollTo}"]`);
      const searchTime = performance.now() - startTime;
      
      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: 'smooth',
          block: 'center'
        });
        
        // Clean up URL by removing query parameters after successful scroll
        normalizeUrlAfterScroll();
        
        return;
      }
      
      if (attempts < maxAttempts) {
        setTimeout(() => attemptScroll(attempts + 1, maxAttempts), 10);
      } else {
        console.warn(`🔍 Could not find element with ID: ${scrollTo} after ${maxAttempts} attempts`);
        // Try alternative selectors as fallback
        const fallbackSelectors = [
          `#${scrollTo}`,
          `[id="${scrollTo}"]`,
          `[name="${scrollTo}"]`,
          `.${scrollTo}`
        ];
        
        for (const selector of fallbackSelectors) {
          const fallbackElement = document.querySelector(selector);
          if (fallbackElement) {
            fallbackElement.scrollIntoView({
              behavior: 'smooth',
              block: 'center'
            });
            
            // Clean up URL by removing query parameters after successful scroll
            normalizeUrlAfterScroll();
            
            return;
          }
        }
        
        console.warn(`🔍 No element found with any selector for: ${scrollTo}`);
      }
    };
    
    // Start the scroll attempt
    attemptScroll();
  }
}

function ensureButtonsInitialized() {
  // Initial check
  initializeAllMessageButtons(shadowRoot);

  // Secondary check after 1 second
  setTimeout(() => {
    const aiMessages = shadowRoot.querySelectorAll(".ai-message-chat");
    initializeAllMessageButtons(shadowRoot);
  }, 1000);
}

async function loadContentAndRetrieveDetails(shadowEle) {
  let lastSavedID;
  try {
    // First, load any saved chat content
    lastSavedID = retrieveRecentIDFromLocal();

    if (lastSavedID !== null) {
      await loadChat(shadowEle, lastSavedID)
        .then(() => {
          reinitializeQuizButtons(shadowRoot);
          initializeAllMessageButtons(shadowRoot);
          reinitializeEditableInputs(shadowRoot);
          reinitializeButtonListeners(shadowRoot);
        })
        .catch((error) => console.error("Error loading chat:", error));
    }
  } catch (error) {
    console.error("Failed to load content:", error);
  }

  // Add a longer delay to ensure DOM updates are complete and content is loaded
  await new Promise((resolve) => setTimeout(resolve, 1000));

  // Now handle any shared messages
  const params = new URLSearchParams(window.location.search);
  const sharedContent = params.get("shared_messages");

  if (sharedContent) {
    try {
      // Clear URL parameters after getting them
      window.history.replaceState({}, "", window.location.pathname);

      // Decode the shared content
      const decodedContent = JSON.parse(decodeURIComponent(sharedContent));

      // Create new message using the markdown content
      appendNewMessage(decodedContent.markdown);

      // Add another small delay to ensure the shared content is appended
      await new Promise((resolve) => setTimeout(resolve, 500));

      // Open the menu and scroll to the bottom
      menu_slide_on(shadowRoot, true);
      scrollToBottom(shadowRoot);
    } catch (error) {
      console.error("Error handling shared message:", error);
      alert(shadowRoot, "Failed to load shared message", "error");
    }
  }

  const lastSavedIDChatId = lastSavedID || 1;
  return lastSavedID;
}

//  TODO: add user input
async function handleGeneralAction(
  text,
  links = ["https://harvard-edge.github.io/cs249r_book/"],
  ele = "",
  tempDifficultyLevel,
  isDiagram = false,
  understanding = 3,
  power_up = 3
) {
  getConfig_explain.set_field("quote", "");
  getConfig_explain.set_field("understanding", understanding);
  getConfig_explain.set_field("power_up", power_up);
  let prompt = text; //getConfig_explain.get_field("prompt");
  // prompt = text;

  if (tempDifficultyLevel) {
    prompt = `${tempDifficultyLevel} \n\n ${prompt}`;

    getConfig_explain.set_field(
      "prompt",
      prompt +
        " For important keywords or phrases with wikipedia pages, you can surround them with double slashes. For example '\\machine learning\\'. At the end of your response, place '%%%' 3 percent signs, and then write 2 to 3 followup questions that the user might ask to expand their knowledge."
    );
  } else {
    getConfig_explain.set_field(
      "prompt",
      SYSTEM_PROMPT +
        " " +
        prompt +
        " For important keywords or phrases with wikipedia pages, you can surround them with double slashes. For example '\\machine learning\\'. At the end of your response, place '%%%' 3 percent signs, and then write 2 to 3 followup questions that the user might ask to expand their knowledge."
    );
  }

  const params = getConfig_explain.return_all_fields();
  params.model = llm_model;

  accumulatedResponse = "";
  const tempAcummulatedMarkdown = accumulatedResponse + loaderMarkdown;

  const clone = get_message_element(shadowRoot, "ai");
  updateCloneAttributes(clone, text, "general");

  // showProgress
  const progress = injectProgress(clone);
  addProgress(GENERALAGENTPROCESS, progress);
  let progressCount = 0;
  showProgressItem(progress, progressCount);

  updateMarkdownPreview(tempAcummulatedMarkdown, clone);

  progressCount += 1;
  showProgressItem(progress, progressCount);

  let count = 0;

  progressCount += 1;
  showProgressItem(progress, progressCount);

  let new_ans = `<details class="user-input-resubmit" style="
background-color: rgba(13, 110, 253, 0.05);
border: 1px solid rgba(13, 110, 253, 0.2);
border-radius: 4px;
padding: 0.5rem;
margin-bottom: 1.5rem;
">
<summary style="
  cursor: pointer;
  color: #0d6efd;
  font-weight: 500;
  padding: 0.25rem;
">User's Input</summary>
<div class="editable-container" style="padding: 0.5rem;">
  <div class="editable-text" contenteditable="true" style="
    min-height: 1.5em;
    padding: 0.25rem;
    border-radius: 3px;
    transition: background-color 0.2s;
  ">${extractQuestion(text)}</div>
  <div style="
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.5rem;
    font-size: 0.875rem;
    color: #6c757d;
  ">Edit, then press${" "}
    <button class="enter-button" style="
      padding: 0.25rem 0.5rem;
      background-color: #f8f9fa;
      border: 1px solid #dee2e6;
      border-radius: 3px;
      font-size: 0.75rem;
      color: #6c757d;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.25rem;
      transition: background-color 0.2s;
    " onmouseover="this.style.backgroundColor='#e9ecef'" 
       onmouseout="this.style.backgroundColor='#f8f9fa'"
       title="Press Enter to submit">
      <span style="margin-right: 2px">➪</span> Enter
    </button>
    <span>to resubmit</span>
  </div>
</div>
</details>

\n`;

  accumulatedResponse += new_ans;

  try {
    // Iterate over each chunk yielded by the generator
    for await (let chunk of query_agent(params, token)) {
      count += 1;
      accumulatedResponse += chunk;
      updateMarkdownPreview(accumulatedResponse, clone);
    }

    // Once all chunks have been processed, use the complete response
    // updateMarkdownPreview(accumulatedResponse);
  } catch (error) {
    console.error("Error processing streamed response:", error);
    alert(shadowRoot, error + ". Please try again.", "error");
  }
  // saveChatHistoryVector(chatId || 1, "assistant", accumulatedResponse, token);
  accumulatedResponse = accumulatedResponse + "\n\n---\n---\n---\n\n";

  progressCount += 1;
  showProgressItem(progress, progressCount);
  const ref_buttons_container = get_reference_buttons(shadowRoot, clone, []);
  add_copy_paste_share_buttons(
    shadowRoot,
    clone,
    ref_buttons_container,
    accumulatedResponse
  );

  removeElementById(clone, "progress");

  assignUniqueIdsToElementAndChildren(clone);
  if (!ele) scrollToTopSmooth(shadowRoot);
  replaceElementsWithErrorNotice("skeleton-loader", shadowRoot);

  await saveChatHistory();
}

function appendNewMessage(text) {
  const clone = get_message_element(shadowRoot, "ai");

  // Format the shared content with header
  const formattedContent = formatSharedContent(text);

  // Update markdown preview with the shared content
  updateMarkdownPreview(formattedContent, clone);

  // Add reference buttons and copy/paste/share buttons
  const ref_buttons_container = get_reference_buttons(shadowRoot, clone, []);
  add_copy_paste_share_buttons(
    shadowRoot,
    clone,
    ref_buttons_container,
    formattedContent
  );

  // Cleanup and finalize
  removeElementById(clone, "progress");
  assignUniqueIdsToElementAndChildren(clone);
  make_links_load_new_page(clone);
  menu_slide_on(shadowRoot, true);
  scrollToBottom(shadowRoot);
  replaceElementsWithErrorNotice("skeleton-loader", shadowRoot);

  // Save to chat history
  saveChatHistory();
}

function formatSharedContent(text) {
  // Sanitize the content first

  // Remove any unwanted style definitions
  const cleanedText = text.replace(
    /ul,\s*ol\s*{\s*list-style-position:\s*outside\s*!important;\s*}\s*ul\s*{\s*list-style-type:\s*disc;\s*}\s*p\s*{\s*margin-bottom:\s*1\.5em;\s*}/g,
    ""
  );

  // Get current date
  const currentDate = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  // Format with header
  return `
::: info Shared Content
*Shared on ${currentDate}*
:::

${cleanedText}

---
`;
}

function generateUniqueString() {
  const randomString = Math.random().toString(36).substring(2, 15);
  return randomString;
}

function paperSearchAddition(input) {
  const text = input.value;
  input.value = "";
  handleResearchAction(
    text,
    ["https://harvard-edge.github.io/cs249r_book/"],
    3,
    3,
    true
  );
  scrollToBottom(shadowRoot);
}

async function handleResearchAction(
  sumOfPage,
  links = ["https://harvard-edge.github.io/cs249r_book/"],

  understanding = 3,
  power_up = 3,
  getNew = false
) {
  const newUrl = window.location.href;
  if (currentUrl !== newUrl) {
    currentUrl = newUrl;
    getNew = true;
  }
  // showProgress
  const progress = injectProgress(clone);
  addProgress(RESEARCHAGENTPROCESS, progress);
  let progressCount = 0;
  showProgressItem(progress, progressCount);

  updateMarkdownPreview(loaderMarkdown, clone);

  progressCount += 1;
  showProgressItem(progress, progressCount);

  getConfig_explain.set_field("quote", "");
  getConfig_explain.set_field("understanding", understanding);
  getConfig_explain.set_field("power_up", power_up);
  let prompt = getConfig_explain.get_field("prompt");

  prompt = `Output a single search term or phrase for this text that I can use to search for the most related papers on Arxiv. Do not add any commentary. Text: ${sumOfPage}`;
  getConfig_explain.set_field("prompt", prompt);

  const params = getConfig_explain.return_all_fields();
  params.model = llm_model;

  try {
    let topics;
    if (!main_topic_of_page || getNew) {
      let research_topics_from_ai = "";

      // Iterate over each chunk yielded by the generator
      for await (let chunk of query_agent(params, token)) {
        research_topics_from_ai += chunk;
      }
      topics = research_topics_from_ai; //extractPhrasesInBrackets(research_topics_from_ai); //.substring(9))

      main_topic_of_page = topics; // set this globally to cache it...
    }

    progressCount += 1;
    showProgressItem(progress, progressCount);

    const randomID = generateUniqueString();

    const researchMarkdown = await initiateResearch(
      sumOfPage,
      token,
      randomID,
      main_topic_of_page,
      10
    );

    updateMarkdownPreview(researchMarkdown, clone, 1);

    const morePapersButton = shadowRoot.querySelector(
      `#button-more-papers-${randomID}`
    );
    const papersContainer = shadowRoot.querySelector(
      `#more-papers-${randomID}`
    );
    const morePaperSearcInput = shadowRoot.querySelector(
      `#more-papers-search-${randomID}`
    );
    const morePaperSearcBtn = shadowRoot.querySelector(
      `#more-papers-search-button-${randomID}`
    );

    morePapersButton.addEventListener("click", function () {
      if (papersContainer) {
        papersContainer.style.display = "block";
        this.style.display = "none";
      }
    });

    morePaperSearcInput.addEventListener("keydown", function (event) {
      if (event.key === "Enter") {
        if (papersContainer) {
          paperSearchAddition(this);
        }
      }
    });

    morePaperSearcBtn.addEventListener("click", function () {
      if (papersContainer) {
        paperSearchAddition(morePaperSearcInput);
      }
    });

    progressCount += 1;
    showProgressItem(progress, progressCount);
  } catch (error) {
    console.error("Error processing streamed response:", error);
    alert(shadowRoot, error + ". Please try again.", "error");
  }

  progressCount += 1;
  showProgressItem(progress, progressCount);

  const ref_buttons_container = get_reference_buttons(shadowRoot, clone, []);
  add_copy_paste_share_buttons(shadowRoot, clone, ref_buttons_container);

  removeElementById(clone, "progress");

  assignUniqueIdsToElementAndChildren(clone);
  make_links_load_new_page(clone);
  scrollToBottom(shadowRoot);

  replaceElementsWithErrorNotice("skeleton-loader", shadowRoot);

  await saveChatHistory();
}

async function handleQueryActionStream(
  text,
  fromRightClickMenu = false,
  ele = "",
  tempDifficultyLevel,
  diagramId = ""
) {

  accumulatedResponse = "";
  let clone = get_message_element(shadowRoot, "ai", ele);

  // If we have an existing element, ensure it has the correct structure
  if (ele) {

    // Get the template from the shadow DOM
    const templateMessage = shadowRoot.querySelector(
      "#bag-of-stuff #ai-message"
    );

    if (!templateMessage) {
      console.error("5. Could not find AI message template in shadow DOM");
      return;
    }

    // Clone the inner content
    const templateContent = templateMessage.cloneNode(true);

    // Clear the existing content and append the template structure
    clone.innerHTML = "";
    while (templateContent.firstChild) {
      clone.appendChild(templateContent.firstChild);
    }
  }

  updateCloneAttributes(clone, text, "query");



  const progress = injectProgress(clone);

  addProgress(QUERYAGENTPROCESS, progress);
  let progressCount = 0;

  showProgressItem(progress, progressCount);

  const tempAcummulatedMarkdown = loaderMarkdown;

  updateMarkdownPreview(tempAcummulatedMarkdown, clone);
  if (!ele) scrollToTopSmooth(shadowRoot);

  let relatedConversations = [];

  progressCount += 1;
  showProgressItem(progress, progressCount);


  let new_ans = `<details class="user-input-resubmit" style="
  background-color: rgba(13, 110, 253, 0.05);
  border: 1px solid rgba(13, 110, 253, 0.2);
  border-radius: 4px;
  padding: 0.5rem;
  margin-bottom: 1.5rem;
">
  <summary style="
    cursor: pointer;
    color: #0d6efd;
    font-weight: 500;
    padding: 0.25rem;
  ">User's Input</summary>
  <div class="editable-container" style="padding: 0.5rem;">
    <div class="editable-text" contenteditable="true" style="
      min-height: 1.5em;
      padding: 0.25rem;
      border-radius: 3px;
      transition: background-color 0.2s;
    ">${text}</div>
    <div style="
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-top: 0.5rem;
      font-size: 0.875rem;
      color: #6c757d;
    ">Edit, then press${" "}
      <button class="enter-button" style="
        padding: 0.25rem 0.5rem;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 3px;
        font-size: 0.75rem;
        color: #6c757d;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 0.25rem;
        transition: background-color 0.2s;
      " onmouseover="this.style.backgroundColor='#e9ecef'" 
         onmouseout="this.style.backgroundColor='#f8f9fa'"
         title="Press Enter to submit">
        <span style="margin-right: 2px">➪</span> Enter
      </button>
      <span>to resubmit</span>
    </div>
  </div>
</details>

\n`;

  let prompt = getConfig_query.get_field("prompt");
  let messages = relatedConversations;
  let newprompt;

  // Find relevant paragraphs using fuzzy matching
  const relevantParagraphs = await findSimilarParagraphsNonBlocking(text, 3);
  
  let backgroundContext = "";
  if (relevantParagraphs.length > 0) {
    backgroundContext = relevantParagraphs.map((match, index) => {
      const refId = `ref-${index + 1}`;
      // Extract the actual paragraph ID from the selector
      const paragraphId = match.selector.match(/data-fuzzy-id="([^"]+)"/)?.[1] || 'unknown-id';
      return `[${refId}] (id=${paragraphId}:url=${currentUrl}) ${match.text}`;
    }).join('\n\n');
  } else {
    backgroundContext = "No relevant content found on current page.";
  }

  // Replace the {{background_context}} placeholder in the prompt
  const enhancedPrompt = prompt.replace('{{background_context}}', backgroundContext);

  if (tempDifficultyLevel) {
    newprompt = `${tempDifficultyLevel} \n\n BACKGROUND KNOWLEDGE: ${"this is a textbook about Machine Learning Systems and tinyML called mlsysbook.ai "} \n\n QUESTION: ${text} \n\n ${enhancedPrompt}`;
  } else {
    newprompt = `${SYSTEM_PROMPT} \n\n QUESTION: ${text} \n\n ${enhancedPrompt}`;
  }

  const params = { prompt: newprompt, messages };
  params.model = llm_model;

  // show progress
  progressCount += 1;
  showProgressItem(progress, progressCount);

  let count = 0;

  // show progress
  progressCount += 1;
  showProgressItem(progress, progressCount);

  try {
    for await (let chunk of query_agent(params, token, true, true)) {
      if (count === 0) {
        progressCount += 1;
        showProgressItem(progress, progressCount);
      }
      count += 1;
      new_ans += chunk;
      updateMarkdownPreview(new_ans, clone);
    }

    const ref_buttons_container = get_reference_buttons(shadowRoot, clone, []);

    // add_copy_paste_share_buttons(shadowRoot, clone='', ref_buttons_container);
    add_copy_paste_share_buttons(
      shadowRoot,
      clone,
      ref_buttons_container,
      accumulatedResponse
    );

    await removeElementById(clone, "progress");

    // assignUniqueIdsToElementAndChildren(clone);

    const idsTracked = assignAndTrackUniqueIds(clone, ["markdown-preview"]);

    // After generating the response, check if we need to update a diagram
    if (diagramId) {
      const diagramData = mermaidCache.get(diagramId);
      if (diagramData) {
        // Update the cached diagram data with the new explanation
        const updatedContent = `${new_ans}\n\n### Diagram:\n\n`;

        accumulatedResponse = updatedContent;

        updateMarkdownPreview(accumulatedResponse, clone, 0, idsTracked[0]);

        const messageContent = `
          ${diagramData.result}
        `;

        insertDiagramElements(messageContent, clone);

        mermaidCache.delete(diagramId);
      } else {
        console.warn(`No diagram found in cache for ID: ${diagramId}`);

        // accumulatedResponse = new_ans + "\n\n---\n---\n---\n";

        // If not in mermaidCache, store in diagramElementMap for later processing
        diagramElementMap.set(diagramId, {
          clone: clone,
          previewId: idsTracked[0],
          // explanation: accumulatedResponse,
          text: accumulatedResponse,
          timestamp: Date.now(),
        });

        // Add diagram generation loader message
        //           const loadingMessage = `
        // ::: loader
        // :::`;

        // Combine the explanation with the loading message
        const tempAccumulatedResponse =
          new_ans +
          "\n\n ### Diagram: \n" +
          loaderMarkdown +
          "\n\n---\n---\n---\n";

        updateMarkdownPreview(tempAccumulatedResponse, clone, 0, idsTracked[0]);
      }
    }

    // else {
    accumulatedResponse = new_ans + "\n\n---\n---\n---\n";
    // }

    if (!ele) scrollToTopSmooth(shadowRoot);

    if (!diagramId) {
      replaceElementsWithErrorNotice("skeleton-loader", shadowRoot);
    }
    await saveChatHistory();
  } catch (error) {
    console.error("Error processing streamed response:", error);
    alert(shadowRoot, error + ". Network error.", "error");

    await handleOfflineResponse(text, clone);
    // alert(shadowRoot, error + ". Please try again.", "error");
  }
}

// KAI NEW FEATURES

async function handleProgressReport(text, xyChart, quadrantChart, ele = "") {
  try {
    // Get the centralized database instance
    const dbManager = await getDBInstance();
    if (!dbManager || !dbManager.db) {
      throw new Error("Database not properly initialized");
    }

    accumulatedResponse = "";
    let clone = get_message_element(shadowRoot, "ai", ele);

    updateCloneAttributes(clone, text, "query");

    // Get the latest report using the centralized database
    let lastReport;
    try {
      lastReport = await dbManager.getLatest("progressReports");
    } catch (error) {
      console.error("Error getting latest report:", error);
      lastReport = null;
    }

    // showProgress
    const progress = injectProgress(clone);
    addProgress(PROGRESSREPORTAGENTPROCESS, progress);

    let progressCount = 0;
    showProgressItem(progress, progressCount);

    // Add more debug logs
    const tempAcummulatedMarkdown = loaderMarkdown;

    updateMarkdownPreview(tempAcummulatedMarkdown, clone);
    if (!ele) scrollToTopSmooth(shadowRoot);

    progressCount += 1;
    showProgressItem(progress, progressCount);

    let new_ans = "";

    if (text.includes("No new quiz attempts since last report")) {
      updateMarkdownPreview(NO_NEW_QUIZZES_MESSAGE, clone);
      new_ans = NO_NEW_QUIZZES_MESSAGE;
    } else if (
      text.includes("It looks like you haven't taken any quizzes yet!")
    ) {
      updateMarkdownPreview(NO_QUIZZES_MESSAGE, clone);
      new_ans = NO_QUIZZES_MESSAGE;
    } else {
      // Modified this section to handle null lastReport
      let prompt = getConfig_progress_report.get_field("prompt");
      prompt = fillPromptTemplate(prompt, {
        progress_report: text,
        previous_evaluation:
          lastReport?.content || "This is your first progress report.",
      });

      let messages = [];
      let newprompt = `${SYSTEM_PROMPT} \n\n ${prompt}`;

      const params = { prompt: newprompt, messages };
      params.model = llm_model;

      progressCount += 1;
      showProgressItem(progress, progressCount);

      let count = 0;

      progressCount += 1;
      showProgressItem(progress, progressCount);

      try {
        for await (let chunk of tryMultipleProvidersStream(
          params,
          token,
          true
        )) {
          if (count === 0) {
            progressCount += 1;
            showProgressItem(progress, progressCount);
          }
          count += 1;
          new_ans += chunk;

          updateMarkdownPreview(new_ans, clone);
        }

        // Save the new progress report using the centralized database
        const reportData = {
          id: Date.now(),
          date: new Date().toISOString(),
          content: new_ans,
          originalText: text,
          metadata: {
            model: llm_model,
            timestamp: Date.now(),
          },
        };

        await dbManager.add("progressReports", reportData);
      } catch (error) {
        console.error("Error processing streamed response:", error);
        alert(shadowRoot, error + ". Please try again.", "error");
      }
    }

    // After processing the text report and before adding buttons
    const markdownContainer = clone.querySelector(
      ".markdown-preview-container"
    );

    // Create and insert XY Chart
    if (markdownContainer) {
      // Create and insert XY Chart
      const xyChartResult = await createXYChart(xyChart);
      const quadrantChartResult = await createQuadrantChart(quadrantChart);

      if (xyChartResult.success && quadrantChartResult.success) {
        insertAtProportionalPoints(
          markdownContainer,
          xyChartResult.element,
          quadrantChartResult.element
        );
      }
    }

    accumulatedResponse = new_ans + "\n\n---\n---\n---\n";
    const ref_buttons_container = get_reference_buttons(shadowRoot, clone, []);

    add_copy_paste_share_buttons(
      shadowRoot,
      clone,
      ref_buttons_container,
      accumulatedResponse
    );

    await removeElementById(clone, "progress");

    assignUniqueIdsToElementAndChildren(clone);
    if (!ele) scrollToTopSmooth(shadowRoot);
    replaceElementsWithErrorNotice("skeleton-loader", shadowRoot);
    await saveChatHistory();
  } catch (error) {
    console.error("Error in handleProgressReport:", error);
    alert(
      shadowRoot,
      "An error occurred while processing the progress report",
      "error"
    );
  }
}

async function handleSummativeAction(
  text,
  title,
  ele = "",
  tempDifficultyLevel = "",
  contentWithSources = null,
  retryCount = 0
) {
  const maxRetries = 2;
  
  // Get the current prompt
  const currentPrompt = getConfig_summative.get_field("prompt");

  if (tempDifficultyLevel) {
    getConfig_summative.set_field(
      "prompt",
      `${tempDifficultyLevel}\n\n${currentPrompt}`
    );
  } else {
    getConfig_summative.set_field(
      "prompt",
      `${SYSTEM_PROMPT}\n\n${currentPrompt}`
    );
  }

  getConfig_summative.set_field("quote", text);
  const params = getConfig_summative.return_all_fields();
  // Model will be set by the Cloudflare agent (llama-3.3-70b-versatile)
  let clone = get_message_element(shadowRoot, "ai", ele);

  updateCloneAttributes(clone, text, "quiz", title);

  let quizRespnse = "";
  let apiUsed = "";

  const progress = injectProgress(clone);
  addProgress(QUIZAGENTPROCESS, progress);
  let progressCount = 0;
  showProgressItem(progress, progressCount);

  // toggleMarkdownActivate();
  const tempAcummulatedMarkdown = loaderMarkdown;
  updateMarkdownPreview(tempAcummulatedMarkdown, clone);

  progressCount += 1;
  showProgressItem(progress, progressCount);
  
  try {
    // Try GROQ (Llama) first
    try {
      quizRespnse = await query_agent_groq_serverless(params);
      apiUsed = "groq";
    } catch (groqError) {
      console.warn("GROQ endpoint failed, trying Gemini:", groqError);
      
      // Try Gemini as fallback (both Cloudflare endpoints)
      try {
        quizRespnse = await query_agent_gemini_serverless(params);
        apiUsed = "gemini";
      } catch (geminiError) {
        console.error("Both Cloudflare endpoints failed:", geminiError);
        throw new Error("All Cloudflare endpoints failed. Please try again later.");
      }
    }
    
    // Validate the response before proceeding
    
    if (!quizRespnse || (!Array.isArray(quizRespnse) && typeof quizRespnse !== 'object')) {
      throw new Error('Invalid response format from AI');
    }
    
    // If it's an object but not an array, check if it has questions
    if (typeof quizRespnse === 'object' && !Array.isArray(quizRespnse)) {
      const keys = Object.keys(quizRespnse);
      if (keys.length === 0 || !quizRespnse[keys[0]] || !Array.isArray(quizRespnse[keys[0]])) {
        throw new Error('Invalid quiz data structure from AI');
      }
    }

    const _ = showQuiz(clone);

    progressCount += 1;
    showProgressItem(progress, progressCount);

    if (Array.isArray(quizRespnse)) {
  
      await ini_quiz(clone, quizRespnse, title, '', '', contentWithSources);
    } else if (quizRespnse !== null && typeof quizRespnse === "object") {
    
      
      const keys = Object.keys(quizRespnse);
      if (keys.length > 0 && quizRespnse[keys[0]].length > 0) {
        const firstKey = keys[0];
        const firstKeyValue = quizRespnse[firstKey];
        await ini_quiz(clone, firstKeyValue, title, '', '', contentWithSources);
      } else {
     
        isOffline = true;
        quizRespnse = "0";
        await ini_quiz(clone, quizRespnse, title, '', '', contentWithSources);
        if (isOffline) {
          injectOfflineWarning(clone);
        }
        alert(
          shadowRoot,
          "AI returned an unexpected response. Building quiz locally.",
          "Error"
        );
      }
    } else {
 
      isOffline = true;
      quizRespnse = createQuiz(text);
      await ini_quiz(clone, quizRespnse, title, '', '', contentWithSources);
      if (isOffline) {
        injectOfflineWarning(clone);
      }
      alert(
        clone,
        "AI returned an unexpected response. Building quiz locally.",
        "Error"
      );
    }
  } catch (error) {
    console.error("Error processing quiz response:", {
      error: error.message,
      stack: error.stack,
      response: quizRespnse,
      retryCount: retryCount
    });
    
    // Check if we should retry
    if (retryCount < maxRetries) {
      
      // Update progress to show retry
      showProgressItem(progress, progressCount);
      
      // Wait a bit before retry
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Retry the function
      return await handleSummativeAction(
        text, 
        title, 
        ele, 
        tempDifficultyLevel, 
        contentWithSources, 
        retryCount + 1
      );
    }
    
    // If all retries failed, fall back to offline quiz
    let isOffline = true;
    const fallbackQuizRespnse = createQuiz(text);
    const _ = showQuiz(clone);
    await ini_quiz(clone, fallbackQuizRespnse, title, '', '', contentWithSources);
    if (isOffline) {
      injectOfflineWarning(clone);
    }
    alert(
      clone,
      "AI returned an unexpected response after multiple retries. Building quiz locally.",
      "Error"
    );
  }

  progressCount += 1;
  showProgressItem(progress, progressCount);

  progressCount += 1;
  showProgressItem(progress, progressCount);

  removeElementById(clone, "progress");
  if (!ele) scrollToTopSmooth(shadowRoot);

  replaceElementsWithErrorNotice("skeleton-loader", shadowRoot);
  await removeElementById(clone, "progress");
  await saveChatHistory();
}

async function createXYChart(data) {
  try {
    const result = await renderMermaidDiagram(data);

    if (result.success) {
      const container = document.createElement("div");
      container.className = "chart-container xy-chart";
      container.style.cssText = `
        margin: 1.5rem 0;
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        width: 100%;
        max-width: 600px;
        min-height: 400px;
        display: flex;
        flex-direction: column;
        align-items: center;
      `;

      // Add title
      const title = document.createElement("h3");
      title.textContent = "Quiz Performance Trends";
      title.style.cssText = `
        margin-bottom: 1rem;
        font-weight: 600;
        color: #1f2937;
        width: 100%;
        text-align: center;
      `;
      container.appendChild(title);

      // Wrap the chart in a container to control its size
      const chartWrapper = document.createElement("div");

      chartWrapper.style.cssText = `
        width: 100%;
        height: 350px;
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
      `;

      // Ensure the chart SVG fits within the wrapper
      const chart = result.element;
      chart.classList.add("zoomable-image"); // Add this class
      chart.classList.add("chart-svg"); // Add this class to distinguish from icons
      chart.style.cssText = `
        max-width: 100%;
        max-height: 100%;
        width: auto;
        height: auto;
      `;

      chartWrapper.appendChild(chart);
      container.appendChild(chartWrapper);

      return {
        success: true,
        element: container,
      };
    }

    throw new Error(result.error || "Failed to render XY chart");
  } catch (error) {
    console.error("Error in createXYChart:", error);
    return {
      success: false,
      error: error.message,
    };
  }
}

async function createQuadrantChart(data) {
  try {
    const result = await renderMermaidDiagram(data);

    if (result.success) {
      const container = document.createElement("div");
      container.className = "chart-container quadrant-chart";
      container.style.cssText = `
        margin: 1.5rem 0;
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        width: 100%;
        max-width: 600px;
        min-height: 400px;
        display: flex;
        flex-direction: column;
        align-items: center;
      `;

      // Wrap the chart in a container to control its size
      const chartWrapper = document.createElement("div");
      chartWrapper.style.cssText = `
        width: 100%;
        height: 440px;
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
      `;

      // Ensure the chart SVG fits within the wrapper
      const chart = result.element;
      chart.classList.add("zoomable-image"); // Add this class for easier targetin
      chart.classList.add("chart-svg"); // Add this class to distinguish from icons
      chart.style.cssText = `
        max-width: 100%;
        max-height: 100%;
        width: auto;
        height: auto;
      `;

      chartWrapper.appendChild(chart);
      container.appendChild(chartWrapper);

      return {
        success: true,
        element: container,
      };
    }

    throw new Error(result.error || "Failed to render quadrant chart");
  } catch (error) {
    console.error("Error in createQuadrantChart:", error);
    return {
      success: false,
      error: error.message,
    };
  }
}

async function handleMermaidDiagramAction(text, id) {
  try {
    const prompt = getConfig_mermaid.get_field("prompt") + "\n" + text;
    let response = "";

    const preferredOrder = ["GROQ", "HUGGINGFACE", "OPEN", "MISTRAL", "GEMINI"];
    let lastError = null;

    for (const provider of preferredOrder) {
      try {
        // Get complete response from provider
        for await (let chunk of callProviderSingle(prompt, provider, null)) {
          response += chunk;
        }

        // Try to render the diagram with the complete response
        const result = await renderMermaidFlow(response);

        // If we get a successful result, use it immediately
        if (result.success) {
          const messageContent = `${result.element.outerHTML}`;
          const storedData = diagramElementMap.get(id);

          if (storedData) {
            // Remove loaders before inserting diagram
            removeSkeletonLoaders(storedData.clone);
            insertDiagramElements(messageContent, storedData.clone);
            await saveChatHistory();
            diagramElementMap.delete(id);
          } else {
            mermaidCache.set(id, {
              code: result.code,
              content: messageContent,
              caption: result.caption,
              timestamp: Date.now(),
              result: result.element.outerHTML,
            });
          }

          return { success: true };
        }

        // If we get here, the result wasn't successful
        throw new Error(result.error || "Failed to render diagram");
      } catch (error) {
        lastError = error;
        continue;
      }
    }

    throw new Error(`All providers failed. Last error: ${lastError.message}`);
  } catch (error) {
    console.error("Error in handleMermaidDiagramAction:", error);
    return {
      success: false,
      error: error.message,
    };
  }
}

// Helper function to retrieve cached diagram
export function getMermaidDiagram(id) {
  const cached = mermaidCache.get(id);
  if (!cached) {
    return null;
  }

  // Optionally, implement cache expiration
  const CACHE_TIMEOUT = 30 * 60 * 1000; // 30 minutes
  if (Date.now() - cached.timestamp > CACHE_TIMEOUT) {
    mermaidCache.delete(id);
    return null;
  }

  return cached;
}

// Helper function to clear cache
export function clearMermaidCache() {
  mermaidCache.clear();
}

async function handleQuizAction(
  text,
  title,
  ele = "",
  tempDifficultyLevel = "",
  context = null
) {
  // Get the current prompt
  const currentPrompt = getConfig_quiz.get_field("prompt");

  if (tempDifficultyLevel) {
    getConfig_quiz.set_field(
      "prompt",
      `${tempDifficultyLevel}\n\n${currentPrompt}`
    );
  } else {
    getConfig_quiz.set_field("prompt", `${SYSTEM_PROMPT}\n\n${currentPrompt}`);
  }

  // Get chapter info from window global
  const chapterInfo = window.currentChapterInfo || {};
  
  getConfig_quiz.set_field("quote", text);
  const params = getConfig_quiz.return_all_fields();
  
  // Add context to params if available
  if (context) {
    params.context = context;
  }
  // Model will be set by the Cloudflare agent (llama-3.3-70b-versatile)
  const chapterTitle = `Chapter ${chapterInfo.chapter || '1'}: ${chapterInfo.title || 'UNKNOWN'}`;
  params.chapterId = chapterTitle;
  params.sectionId = title || chapterInfo.title || 'UNKNOWN';  // Use provided title or fall back to chapter title

  let clone = get_message_element(shadowRoot, "ai", ele);
  updateCloneAttributes(clone, text, "quiz", title);

  let quizRespnse = "";
  let isOffline = false;
  let apiUsed = "";

  const progress = injectProgress(clone);
  addProgress(QUIZAGENTPROCESS, progress);
  let progressCount = 0;
  showProgressItem(progress, progressCount);

  // toggleMarkdownActivate();
  const tempAcummulatedMarkdown = loaderMarkdown;
  updateMarkdownPreview(tempAcummulatedMarkdown, clone);

  progressCount += 1;
  showProgressItem(progress, progressCount);

  try {
    // Try GROQ (Llama) first with retry logic
    let retryCount = 0;
    const maxRetries = 2;
    
    while (retryCount <= maxRetries) {
      try {
        quizRespnse = await query_agent_groq_serverless(params, token);
        apiUsed = "groq";
        
        // Validate the response
        if (Array.isArray(quizRespnse) && quizRespnse.length > 0) {
          break; // Success, exit retry loop
        } else if (typeof quizRespnse === 'string' && quizRespnse.includes('chatcmpl-')) {
          console.warn(`⚠️ GROQ returned raw response instead of parsed data, retrying... (attempt ${retryCount + 1})`);
          retryCount++;
          if (retryCount <= maxRetries) {
            await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second before retry
            continue;
          }
        } else {
          console.warn(`⚠️ GROQ returned unexpected response type: ${typeof quizRespnse}`, quizRespnse);
          retryCount++;
          if (retryCount <= maxRetries) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            continue;
          }
        }
      } catch (groqError) {
        console.warn(`GROQ attempt ${retryCount + 1} failed:`, groqError);
        retryCount++;
        if (retryCount <= maxRetries) {
          await new Promise(resolve => setTimeout(resolve, 1000));
          continue;
        }
      }
      
      // If we get here, all GROQ retries failed, try Gemini
      console.warn("All GROQ attempts failed, trying Gemini as fallback");
      try {
        quizRespnse = await query_agent_gemini_serverless(params, token);
        apiUsed = "gemini";
        
        // Validate Gemini response
        if (Array.isArray(quizRespnse) && quizRespnse.length > 0) {
          break; // Success
        } else {
          console.warn(`⚠️ Gemini returned unexpected response: ${typeof quizRespnse}`, quizRespnse);
        }
      } catch (geminiError) {
        console.error("Gemini fallback also failed:", geminiError);
      }
      
      break; // Exit retry loop
    }
    
    // Final validation
    if (!Array.isArray(quizRespnse) || quizRespnse.length === 0) {
      throw new Error(`All providers failed to return valid quiz data. Last response type: ${typeof quizRespnse}`);
    }
  

    const _ = showQuiz(clone);

    progressCount += 1;
    showProgressItem(progress, progressCount);

    progressCount += 1;
    showProgressItem(progress, progressCount);

    progressCount += 1;
    showProgressItem(progress, progressCount);

    removeElementById(clone, "progress");
    if (!ele) scrollToTopSmooth(shadowRoot);

    if (Array.isArray(quizRespnse)) {
  
      await ini_quiz(clone, quizRespnse, title, chapterTitle, '', context);
    } else if (quizRespnse !== null && typeof quizRespnse === "object") {
      const keys = Object.keys(quizRespnse);
      if (keys.length > 0 && quizRespnse[keys[0]].length > 0) {
        const firstKey = keys[0];
        const firstKeyValue = quizRespnse[firstKey];
        await ini_quiz(clone, firstKeyValue, title, chapterTitle, '', context);
      } else {
  
        isOffline = true;
        quizRespnse = "0";
        await ini_quiz(clone, quizRespnse, title, chapterTitle, '', context);
        if (isOffline) {
          injectOfflineWarning(clone);
        }
        alert(
          shadowRoot,
          "AI returned an unexpected response. Building quiz locally.",
          "Error"
        );
      }
    } else {
  
      isOffline = true;
      quizRespnse = createQuiz(text);
      await ini_quiz(clone, quizRespnse, title, chapterTitle);
      if (isOffline) {
        injectOfflineWarning(clone);
      }
      alert(
        clone,
        "AI returned an unexpected response. Building quiz locally.",
        "Error"
      );
    }
  } catch (error) {
    console.error("Error processing quiz response:", {
      error: error.message,
      stack: error.stack,
    });
    isOffline = true;
    const quizRespnse = createQuiz(text);
    const _ = showQuiz(clone);
    await ini_quiz(clone, quizRespnse, title, chapterTitle);
    if (isOffline) {
      injectOfflineWarning(clone);
    }
    alert(
      clone,
      "AI returned an unexpected response. Building quiz locally.",
      "Error"
    );
  }



  replaceElementsWithErrorNotice("skeleton-loader", shadowRoot);
  await removeElementById(clone, "progress");
  await saveChatHistory();
}

export function setListeners() {
  window.addEventListener("resetAIChat", (e) => {
    // Assuming 'e.detail.containerId' holds the ID of the message container
    // You need to ensure that the event dispatching this carries such a detail or adjust accordingly
    if (e.detail && e.detail.containerId) {
      chatId = clearMessageContainer(shadowRoot, e.detail.containerId);
    } else {
      console.error("No container ID provided in event details");
    }
  });

  window.addEventListener("quizSubmitted", async () => {
    await saveChatHistory();
  });

  window.addEventListener("aiActionCompleted", async (e) => {
    const actionType = e.detail.type;
    const text = e.detail.text;
    const links = e.detail.links;
    const settings = e.detail.settings;
    const ele = e.detail.ele;
    const tempDifficultyLevel = e.detail.tempDifficultyLevel;
    const xyChart = e.detail.xyChart;
    const quadrantChart = e.detail.quadrantChart;
    // const selectedNodeDataForSummative = e.detail.selectedNodesData
    const diagramId = e.detail.diagramId;

    if (!ele) {
      scrollToBottom(shadowRoot);
    }

    // Store quiz title in temp variable if present, otherwise use the stored temp title
    const quizTitle = e.detail.title || tempQuizTitle;
    if (e.detail.title) {
      tempQuizTitle = e.detail.title; // Update temp storage only when new title is provided
    }

    // Reset temp title if action type is not quiz-related
    if (
      actionType !== "quiz" &&
      actionType !== "quiz-study" &&
      actionType !== "summative"
    ) {
      tempQuizTitle = "";
    }

    // let getNew = e.detail.getNew || false;
    const fromRightClickMenu = e.detail.fromRightClickMenu || false;

    switch (actionType) {
      //   break;
      case "query":
        await handleQueryActionStream(
          text,
          fromRightClickMenu,
          ele,
          tempDifficultyLevel,
          diagramId
        );

        chatCount += 1;

        break;
      case "quiz":
        await handleQuizAction(text, quizTitle, ele, tempDifficultyLevel, e.detail.context);
        chatCount += 1;

        break;

      case "summative":
        await handleSummativeAction(text, quizTitle, ele, tempDifficultyLevel, e.detail.contentWithSources);
        chatCount += 1;

        break;

      case "progress_report":
        await handleProgressReport(text, xyChart, quadrantChart);
        // saveChatHistoryVector(chatId || 1, "user", text, token);
        chatCount += 1;

        break;

      case "mermaid_diagram":
        await handleMermaidDiagramAction(text, diagramId);

        break;
      case "general":
        await handleGeneralAction(
          text,
          links,
          ele,
          tempDifficultyLevel,
          diagramId
        );
        // saveChatHistoryVector(chatId || 1, "user", text, token);
        chatCount += 1;

        break;

      case "system_prompt":
        SYSTEM_PROMPT = SYSTEM_PROMPT_ORIG + text;
        break;
      case "settings":
        toggleProgress(settings.show_progress);
        llm_model = settings.llm_model;
        
        // Store custom API settings globally for access by API adapters
        if (settings.customAPI) {
          window.customAPISettings = settings.customAPI;
          console.log('[CUSTOM_API] Custom API settings updated:', settings.customAPI);
        } else {
          window.customAPISettings = null;
          console.log('[CUSTOM_API] Custom API settings cleared');
        }
        break;

      default:
        console.error("Unhandled action type:", actionType);
        alert(shadowRoot, "Unhandled action type: " + actionType, "Error");
    }
  });

  // Add this near your other event listeners
  window.addEventListener("editedTextSubmitted", (event) => {
    const newText = event.detail.text;
    handleQueryActionStream(
      newText,
      false,
      "",
      tempDifficultyLevel,
      "",
      handleExplainAction,
      understanding,
      power_up
    );
  });

  // Handle navigation to source with position
  window.addEventListener("navigateToSource", (event) => {
    const { url, position, sourceData } = event.detail;
    
    // Check if we're already on the target page
    if (window.location.href === url) {
      // Same page - scroll to position
      scrollToPosition(position);
    } else {
      // Different page - navigate and scroll
      navigateToSourceWithPosition(url, position);
    }
  });

  // Helper function to scroll to a specific position
  function scrollToPosition(position) {
    
    if (position && position > 0) {
      window.scrollTo({
        top: position,
        behavior: 'smooth'
      });
      
      // Verify scroll after a delay
      setTimeout(() => {
      }, 1000);
      
    } else {
    }
  }

  // Helper function to navigate to a different page with position
  function navigateToSourceWithPosition(url, position) {
    // Add position as query parameter
    if (position && position > 0) {
      const urlObj = new URL(url);
      urlObj.searchParams.set('scrollTo', position.toString());
      url = urlObj.toString();
    }
    
    // Store the position in sessionStorage for after page load (backup)
    if (position && position > 0) {
      sessionStorage.setItem('scrollToPosition', position.toString());
    }
    
    // Navigate to the URL
    window.location.href = url;
  }

  // Check for stored scroll position on page load
  window.addEventListener('load', () => {
    
  // Note: Persistent tooltips are now initialized after quiz content is loaded
  // in the quiz loading function to ensure proper timing
    
    // Check URL query parameter first
    const urlParams = new URLSearchParams(window.location.search);
    const scrollToParam = urlParams.get('scrollTo');
    
    
    if (scrollToParam) {
      const position = parseFloat(scrollToParam);
      
      if (position > 0) {
        // Wait for everything to be fully rendered
        waitForPageReady(position, () => {
          scrollToPosition(position);
          // Clean up URL by removing the scrollTo parameter
          const newUrl = new URL(window.location);
          newUrl.searchParams.delete('scrollTo');
          window.history.replaceState({}, '', newUrl.toString());
        });
      } else {
      }
    } else {
      // Fallback to sessionStorage
      const storedPosition = sessionStorage.getItem('scrollToPosition');
      
      if (storedPosition) {
        const position = parseFloat(storedPosition);
        
        if (position > 0) {
          // Wait for everything to be fully rendered
          waitForPageReady(position, () => {
            scrollToPosition(position);
            sessionStorage.removeItem('scrollToPosition');
          });
        } else {
        }
      } else {
      }
    }
  });

  // Function to wait for page to be fully ready
  function waitForPageReady(position, callback) {
    let attempts = 0;
    const maxAttempts = 20; // 10 seconds max wait
    const checkInterval = 500; // Check every 500ms
    
    function checkReady() {
      attempts++;
      
      const documentHeight = document.documentElement.scrollHeight;
      const bodyHeight = document.body.scrollHeight;
      const windowHeight = window.innerHeight;
      const hasContent = documentHeight > windowHeight || bodyHeight > windowHeight;
      
      console.log('🔍 Page metrics:', {
        documentHeight,
        bodyHeight,
        windowHeight,
        hasContent,
        position,
        canScroll: position < documentHeight
      });
      
      // Check if page has enough content and position is valid
      if (hasContent && position < documentHeight && position > 0) {
        console.log('✅ Page is ready for scrolling');
        callback();
        return;
      }
      
      // If we've tried enough times, scroll anyway
      if (attempts >= maxAttempts) {
        console.log('⚠️ Max attempts reached, scrolling anyway');
        callback();
        return;
      }
      
      // Wait and try again
      setTimeout(checkReady, checkInterval);
    }
    
    // Start checking after a small initial delay
    setTimeout(checkReady, 1000);
  }

  // Quiz button functionality
  const quizButton = shadowRoot.querySelector('#quiz-button');
  const quizModal = shadowRoot.querySelector('#quizModal');
  const closeQuizModal = shadowRoot.querySelector('#close-quiz-modal');
  const sectionQuizBtn = shadowRoot.querySelector('#section-quiz-btn');
  const cumulativeQuizBtn = shadowRoot.querySelector('#cumulative-quiz-btn');

  if (quizButton && quizModal) {
    quizButton.addEventListener('click', () => {
      quizModal.style.display = 'flex';
    });
  } else if (quizButton) {
    // No modal in template - fire quiz directly from page content
    quizButton.addEventListener('click', async () => {
      try {
        const { ViewportContextCapture } = await import('./utils/viewportContextCapture.js');
        const context = ViewportContextCapture.captureCurrentContext();
        const validation = ViewportContextCapture.validateContext(context);
        const contextSummary = validation.isValid
          ? ViewportContextCapture.createContextSummary(context)
          : document.body.innerText.slice(0, 3000);
        window.dispatchEvent(new CustomEvent('aiActionCompleted', {
          detail: {
            type: 'quiz',
            text: `Generate a quiz based on the following page content:\n\n${contextSummary}`,
            title: document.title || 'Page Quiz',
            sectionId: 'current-section',
            context: validation.isValid ? context : null,
          }
        }));
      } catch (error) {
        console.error('Quiz button error:', error);
        window.dispatchEvent(new CustomEvent('aiActionCompleted', {
          detail: {
            type: 'quiz',
            text: document.body.innerText.slice(0, 3000),
            title: document.title || 'Page Quiz',
            sectionId: 'current-section',
          }
        }));
      }
    });
  }

  if (closeQuizModal && quizModal) {
    closeQuizModal.addEventListener('click', () => {
      quizModal.style.display = 'none';
    });
  }

  // Close modal when clicking outside
  if (quizModal) {
    quizModal.addEventListener('click', (e) => {
      if (e.target === quizModal) {
        quizModal.style.display = 'none';
      }
    });
  }

  // Quiz option button handlers
  if (sectionQuizBtn) {
    sectionQuizBtn.addEventListener('click', async () => {
      if (quizModal) quizModal.style.display = 'none';
      
      try {
        // Import the context capture utility
        const { ViewportContextCapture } = await import('./utils/viewportContextCapture.js');
        
        // Capture current viewport context
        const context = ViewportContextCapture.captureCurrentContext();
        
        // Validate context
        const validation = ViewportContextCapture.validateContext(context);
        
        if (!validation.isValid) {
          // Show error message if context is insufficient
          const errorEvent = new CustomEvent('aiActionCompleted', {
            detail: {
              type: 'error',
              text: `Cannot generate quiz: ${validation.message}`,
              title: 'Section Quiz Error',
              sectionId: 'current-section'
            }
          });
          window.dispatchEvent(errorEvent);
          return;
        }
        
        // Create context summary for the AI
        const contextSummary = ViewportContextCapture.createContextSummary(context);
        
        // Trigger section quiz functionality with captured context
        const aiActionEvent = new CustomEvent('aiActionCompleted', {
          detail: {
            type: 'quiz',
            text: `Generate a section quiz based on the current page content. Here's the context:\n\n${contextSummary}`,
            title: 'Section Quiz',
            sectionId: 'current-section',
            context: context,
            metadata: {
              url: context.url,
              title: context.title,
              contentLength: context.content.totalLength,
              sourceCount: context.content.sourceCount,
              scrollPosition: context.position.scrollPercentage.y
            }
          }
        });
        window.dispatchEvent(aiActionEvent);
        
      } catch (error) {
        console.error('Error capturing viewport context:', error);
        
        // Fallback to basic quiz generation
        const aiActionEvent = new CustomEvent('aiActionCompleted', {
          detail: {
            type: 'quiz',
            text: 'Generate a section quiz based on the current page content',
            title: 'Section Quiz',
            sectionId: 'current-section'
          }
        });
        window.dispatchEvent(aiActionEvent);
      }
    });
  }


  if (cumulativeQuizBtn) {
    cumulativeQuizBtn.addEventListener('click', async () => {
      if (quizModal) quizModal.style.display = 'none';
      
      // Import text extraction functionality
      const { extractPageTextWithSourcesForQuiz, logTextExtraction, getCurrentPageUrl, getCurrentPageTitle } = await import('./utils/textExtractor.js');
      
      // Extract and log page text content with source mapping
      console.log('🎯 Cumulative Quiz clicked - extracting page content with source mapping...');
      const extractionResult = extractPageTextWithSourcesForQuiz();
      logTextExtraction(extractionResult);
      
      // Log additional context
      console.log('📄 Page Context:', {
        url: getCurrentPageUrl(),
        title: getCurrentPageTitle(),
        extractionTimestamp: new Date().toISOString()
      });
      
      // Create content with source mapping for reference tooltips (similar to KnowledgeGraph)
      const contentWithSources = extractionResult.sources.map((source, index) => ({
        sourceId: source.sourceId,
        label: source.label,
        content: source.content,
        pageUrl: source.pageUrl,
        domain: source.domain,
        level: source.level,
        position: source.position,
        elementId: source.elementId,
        elementClass: source.elementClass,
        elementTag: source.elementTag
      }));
      
      // Create title from page context
      const title = `Cumulative Quiz - ${getCurrentPageTitle()}`;
      
      // Trigger cumulative quiz functionality with summative type for 10 questions
      const aiActionEvent = new CustomEvent('aiActionCompleted', {
        detail: {
          type: 'summative', // Changed from 'quiz' to 'summative' for 10 questions
          text: extractionResult.text, // Use the structured text with source sections
          title: title,
          sectionId: 'cumulative-content',
          extractedText: extractionResult.text,
          tokenCount: extractionResult.tokens,
          wasSampled: extractionResult.sampled,
          contentWithSources: contentWithSources, // Add source mapping for tooltips
          sources: extractionResult.sources // Keep original sources data
        }
      });
      window.dispatchEvent(aiActionEvent);
    });
  }

  // More Options dropdown functionality
  const moreOptionsButton = shadowRoot.querySelector('#more-options-button');
  const dropdownTemplate  = shadowRoot.querySelector('#more-options-dropdown');

  // Move dropdown to document.body so position:fixed is true-viewport-relative
  // (Shadow DOM stacking contexts break fixed positioning)
  let moreOptionsDropdown = null;
  if (dropdownTemplate) {
    // Inject dropdown styles into document head once
    if (!document.getElementById('socratiq-dropdown-style')) {
      const ds = document.createElement('style');
      ds.id = 'socratiq-dropdown-style';
      ds.textContent = `
        #socratiq-more-dropdown {
          position: fixed;
          background: #ffffff;
          border: 1px solid #e5e7eb;
          border-radius: 10px;
          box-shadow: 0 8px 28px rgba(0,0,0,0.18);
          min-width: 190px;
          z-index: 2147483640;
          overflow: hidden;
          padding: 4px;
          font-family: system-ui, sans-serif;
        }
        #socratiq-more-dropdown.hidden { display: none !important; }
        #socratiq-more-dropdown .dropdown-item {
          display: flex;
          align-items: center;
          gap: 8px;
          width: 100%;
          padding: 9px 12px;
          border-radius: 7px;
          font-size: 0.82rem;
          color: #374151;
          background: none;
          border: none;
          cursor: pointer;
          text-align: left;
          transition: background 0.12s;
          box-sizing: border-box;
        }
        #socratiq-more-dropdown .dropdown-item:hover {
          background: rgba(99,102,241,0.09);
          color: #6366f1;
        }
        #socratiq-more-dropdown .dropdown-item.active {
          background: rgba(99,102,241,0.13);
          color: #6366f1;
          font-weight: 600;
        }
        @keyframes sqDropRise {
          from { opacity:0; transform:translateY(6px); }
          to   { opacity:1; transform:translateY(0); }
        }
        #socratiq-more-dropdown:not(.hidden) { animation: sqDropRise 0.14s ease-out; }
      `;
      document.head.appendChild(ds);
    }

    // Clone inner content into a fresh div on document.body
    moreOptionsDropdown = document.createElement('div');
    moreOptionsDropdown.id = 'socratiq-more-dropdown';
    moreOptionsDropdown.classList.add('hidden');
    moreOptionsDropdown.innerHTML = dropdownTemplate.querySelector('.dropdown-content').innerHTML;
    document.body.appendChild(moreOptionsDropdown);
    dropdownTemplate.remove(); // remove original from shadow DOM
  }

  const _hideDropdown = () => moreOptionsDropdown?.classList.add('hidden');
  const _showDropdown = () => {
    if (!moreOptionsDropdown) return;
    const rect = moreOptionsButton.getBoundingClientRect();
    moreOptionsDropdown.style.left = `${Math.max(4, rect.right - 190)}px`;
    moreOptionsDropdown.style.top  = `${rect.top - 8}px`;
    moreOptionsDropdown.style.transform = 'translateY(-100%)';
    moreOptionsDropdown.classList.remove('hidden');
  };

  if (moreOptionsButton) {
    moreOptionsButton.addEventListener('click', (e) => {
      e.stopPropagation();
      moreOptionsDropdown?.classList.contains('hidden') ? _showDropdown() : _hideDropdown();
    });
  }

  // Close when clicking anywhere outside
  document.addEventListener('click', (e) => {
    if (moreOptionsDropdown && !moreOptionsDropdown.contains(e.target) && !moreOptionsButton?.contains(e.target)) {
      _hideDropdown();
    }
  });

  // Meditation Timer
  const meditationBtn = moreOptionsDropdown?.querySelector('#meditation-btn');
  if (meditationBtn) {
    meditationBtn.addEventListener('click', async () => {
      _hideDropdown();
      const { openMeditationTimer } = await import('./components/meditation/meditationTimer.js');
      openMeditationTimer(shadowRoot);
    });
  }

  // Draw-to-Select
  const drawSelectBtn = moreOptionsDropdown?.querySelector('#draw-select-btn');
  if (drawSelectBtn) {
    drawSelectBtn.addEventListener('click', async () => {
      _hideDropdown();

      const { startDrawSelect, stopDrawSelect, isDrawSelectActive } = await import('./components/draw-select/drawSelect.js');

      if (isDrawSelectActive()) {
        stopDrawSelect();
        drawSelectBtn.classList.remove('active');
        moreOptionsButton.classList.remove('draw-active');
        return;
      }

      drawSelectBtn.classList.add('active');
      moreOptionsButton.classList.add('draw-active');

      startDrawSelect((capturedText) => {
        drawSelectBtn.classList.remove('active');
        moreOptionsButton.classList.remove('draw-active');

        if (!capturedText) return;

        // Open the chat panel
        import('./components/menu/open_close_menu.js').then(({ menu_slide_on }) => {
          menu_slide_on(shadowRoot, true);
        });

        // Send captured text as a query
        window.dispatchEvent(new CustomEvent('aiActionCompleted', {
          detail: {
            type: 'query',
            text: capturedText,
            title: 'Draw to Select',
            sectionId: 'draw-select',
          }
        }));
      });
    });
  }
}

let prevId = chatId;

export async function saveChatHistory() {
  // Introduce a delay with a Promise that resolves after a set time
  await new Promise((resolve) => setTimeout(resolve, 2000)); // Waits for 2 seconds

  chatId = await determineAndSaveChat(shadowRoot, chatId, topicOfConversation);
  if (chatId != null) {
    saveRecentIDToLocal(chatId);
  } else {
    console.warn("No valid chat ID to save");
  }
  if (chatId != prevId) {
    chatCount = 0;
  }
}

async function handleOfflineResponse(text, clone) {
  try {
    const progress = injectProgress(clone);
    addProgress(OFFLINE_PROCESS, progress);
    let progressCount = 0;

    // Show initial progress
    showProgressItem(progress, progressCount);

    // Start with network warning
    let offlineResponse = `::: network-warning
We are currently unable to connect to our servers. Showing relevant content from the page instead.
:::

`;

    progressCount += 1;
    showProgressItem(progress, progressCount);

    // Add the user's input section
    offlineResponse += `<details class="user-input-resubmit" style="
      background-color: rgba(13, 110, 253, 0.05);
      border: 1px solid rgba(13, 110, 253, 0.2);
      border-radius: 4px;
      padding: 0.5rem;
      margin-bottom: 1.5rem;
    ">
      <summary style="cursor: pointer; color: #0d6efd; font-weight: 500; padding: 0.25rem;">
        User's Input
      </summary>
      <div class="editable-container" style="padding: 0.5rem;">
        <div class="editable-text" contenteditable="true" style="
          min-height: 1.5em;
          padding: 0.25rem;
          border-radius: 3px;
          transition: background-color 0.2s;
        ">${text}</div>
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem; font-size: 0.875rem; color: #6c757d;">
          <button class="enter-button" style="
            padding: 0.25rem 0.5rem;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            font-size: 0.75rem;
            color: #6c757d;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.25rem;
          " title="Press Enter to submit">
            <span style="margin-right: 2px">➪</span> Enter
          </button>
          <span>to resubmit when online</span>
        </div>
      </div>
    </details>

### Related Content Found:

`;

    progressCount += 1;
    showProgressItem(progress, progressCount);

    // kkj Show initial loading state
    updateMarkdownPreview(offlineResponse + loaderMarkdown, clone);

    // Use fuzzy matching to find similar content
    const similarParagraphs = await findSimilarParagraphsNonBlocking(text, 3);

    progressCount += 1;
    showProgressItem(progress, progressCount);

    // Add each similar paragraph to the response
    similarParagraphs.forEach((match, index) => {
      offlineResponse += `#### Match ${index + 1} (${Math.round(
        match.similarity * 100
      )}% similar):\n\n${match.text}\n\n---\n\n`;
    });

    progressCount += 1;
    showProgressItem(progress, progressCount);

    // Add final separator
    offlineResponse += "\n\n---\n---\n---\n";

    // Update the preview with the complete offline response
    updateMarkdownPreview(offlineResponse, clone);

    // Add reference buttons and copy/paste/share buttons
    const ref_buttons_container = get_reference_buttons(shadowRoot, clone, []);
    add_copy_paste_share_buttons(
      shadowRoot,
      clone,
      ref_buttons_container,
      offlineResponse
    );

    // Remove progress indicator
    await removeElementById(clone, "progress");

    // Assign unique IDs and save chat history
    const idsTracked = assignAndTrackUniqueIds(clone, ["markdown-preview"]);
    await saveChatHistory();

    // Scroll to bottom if needed
    scrollToTopSmooth(shadowRoot);
  } catch (error) {
    console.error("Error in offline response:", error);
    alert(shadowRoot, "Failed to generate offline response", "error");
  }
}

// Add initialization code for chapter info
setTimeout(() => {
  try {
    const chapterInfo = JSON.parse(localStorage.getItem('current_chapter') || '{}');
    window.currentChapterInfo = chapterInfo;
  } catch (error) {
    console.error('Error loading chapter info:', error);
    window.currentChapterInfo = {};
  }
}, 100);
console.log('🔥 HMR TEST - 01  - Updated at', new Date().toLocaleTimeString());
console.log('✅ Your edit was detected! Comment change at line 2357:', 'kkj Show initial loading state');
