/**
 * Nuclear Hotkey Prevention
 * The most aggressive approach to prevent parent site hotkeys
 * Uses multiple strategies to ensure hotkeys are blocked
 */

let shadowRoot = null;
let isActive = false;
let cleanupFunctions = [];
let inputElement = null;

/**
 * Initialize nuclear hotkey prevention
 * @param {ShadowRoot} root - The shadow root to protect
 */
export function initializeNuclearHotkeyPrevention(root) {
  shadowRoot = root;
  isActive = true;
  
  // Clean up any existing listeners
  cleanupNuclearHotkeyPrevention();

  // Strategy 1: Override the global event listeners
  overrideGlobalEventListeners();

  // Strategy 2: Block events at the element level
  blockElementLevelEvents();

  // Strategy 3: Use a more aggressive document listener
  setupAggressiveDocumentListener();

  // Strategy 4: Override common hotkey functions
  overrideCommonHotkeyFunctions();

  // Strategy 5: Use a mutation observer to block search overlays
  setupSearchOverlayBlocker();

  return () => cleanupNuclearHotkeyPrevention();
}

/**
 * Strategy 1: Override global event listeners
 */
function overrideGlobalEventListeners() {
  // Store original addEventListener
  const originalAddEventListener = EventTarget.prototype.addEventListener;
  
  // Override addEventListener to intercept hotkey listeners
  EventTarget.prototype.addEventListener = function(type, listener, options) {
    if (type === 'keydown' && this === document) {
      // Wrap the listener to check if it should be blocked
      const wrappedListener = function(event) {
        if (shouldBlockEvent(event)) {
          event.preventDefault();
          event.stopPropagation();
          event.stopImmediatePropagation();
          return false;
        }
        return listener.call(this, event);
      };
      return originalAddEventListener.call(this, type, wrappedListener, options);
    }
    return originalAddEventListener.call(this, type, listener, options);
  };

  // Store cleanup function
  cleanupFunctions.push(() => {
    EventTarget.prototype.addEventListener = originalAddEventListener;
  });
}

/**
 * Strategy 2: Block events at the element level
 */
function blockElementLevelEvents() {
  const blockEvent = (event) => {
    if (shouldBlockEvent(event)) {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      return false;
    }
  };

  // Add listeners to all possible elements
  const elements = [document, window, document.body, document.documentElement];
  
  elements.forEach(element => {
    const keydownListener = () => {
      element.addEventListener('keydown', blockEvent, { capture: true, passive: false });
      return () => element.removeEventListener('keydown', blockEvent, { capture: true });
    };
    
    const keyupListener = () => {
      element.addEventListener('keyup', blockEvent, { capture: true, passive: false });
      return () => element.removeEventListener('keyup', blockEvent, { capture: true });
    };

    cleanupFunctions.push(keydownListener());
    cleanupFunctions.push(keyupListener());
  });
}

/**
 * Strategy 3: Setup aggressive document listener
 */
function setupAggressiveDocumentListener() {
  const handleKeyDown = (event) => {
    if (shouldBlockEvent(event)) {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      
      // Also try to prevent the default action
      if (event.defaultPrevented === false) {
        event.defaultPrevented = true;
      }
      
      return false;
    }
  };

  // Add multiple listeners with different priorities
  const listeners = [
    () => {
      document.addEventListener('keydown', handleKeyDown, { capture: true, passive: false });
      return () => document.removeEventListener('keydown', handleKeyDown, { capture: true });
    },
    () => {
      window.addEventListener('keydown', handleKeyDown, { capture: true, passive: false });
      return () => window.removeEventListener('keydown', handleKeyDown, { capture: true });
    },
    () => {
      document.addEventListener('keydown', handleKeyDown, { capture: false, passive: false });
      return () => document.removeEventListener('keydown', handleKeyDown, { capture: false });
    }
  ];

  listeners.forEach(listener => {
    cleanupFunctions.push(listener());
  });
}

/**
 * Strategy 4: Override common hotkey functions
 */
function overrideCommonHotkeyFunctions() {
  // Override common hotkey functions that might be used by search libraries
  const originalExecCommand = document.execCommand;
  document.execCommand = function(command, showUI, value) {
    if (isActive && shadowRoot && inputElement && shadowRoot.contains(inputElement)) {
      // Block certain commands when typing in our widget
      const blockedCommands = ['search', 'find', 'replace'];
      if (blockedCommands.includes(command)) {
        return false;
      }
    }
    return originalExecCommand.call(this, command, showUI, value);
  };

  // Override focus method to prevent focus stealing
  const originalFocus = HTMLElement.prototype.focus;
  HTMLElement.prototype.focus = function(options) {
    if (isActive && shadowRoot && inputElement && shadowRoot.contains(inputElement)) {
      // If trying to focus on a search overlay, focus our input instead
      if (isSearchOverlay(this)) {
        if (inputElement && inputElement.focus) {
          return inputElement.focus(options);
        }
        return;
      }
    }
    return originalFocus.call(this, options);
  };

  // Store cleanup functions
  cleanupFunctions.push(() => {
    document.execCommand = originalExecCommand;
    HTMLElement.prototype.focus = originalFocus;
  });
}

/**
 * Strategy 5: Use mutation observer to block search overlays
 */
function setupSearchOverlayBlocker() {
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === Node.ELEMENT_NODE) {
          // Check if this is a search overlay
          if (isSearchOverlay(node)) {
            console.log('Search overlay detected, removing it');
            node.remove();
            return;
          }
          
          // Check for search overlays within the added node
          const searchOverlays = node.querySelectorAll ? 
            node.querySelectorAll('.aa-DetachedOverlay, .aa-DetachedContainer, [class*="search"], [class*="autocomplete"]') : [];
          
          searchOverlays.forEach(overlay => {
            console.log('Search overlay found within added node, removing it');
            overlay.remove();
          });
        }
      });
    });
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });

  cleanupFunctions.push(() => {
    observer.disconnect();
  });
}

/**
 * Check if an event should be blocked
 * @param {KeyboardEvent} event - The keyboard event
 * @returns {boolean} - True if the event should be blocked
 */
function shouldBlockEvent(event) {
  if (!isActive || !shadowRoot) return false;

  // Check if the event is from our shadow DOM
  const isFromShadowDOM = shadowRoot.contains(event.target) || 
                         event.target.getRootNode() === shadowRoot;

  if (!isFromShadowDOM) return false;

  // Check if we're focused on an input or textarea
  const activeElement = document.activeElement;
  const isInputFocused = activeElement && 
                        (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA') &&
                        shadowRoot.contains(activeElement);

  if (!isInputFocused) return false;

  // Block specific problematic keys
  const problematicKeys = ['f', 'F', 's', 'S', 'b', 'B', 'c', 'C', 'v', 'V', 'x', 'X', 'z', 'Z'];
  const hasModifiers = event.ctrlKey || event.metaKey || event.altKey || event.shiftKey;

  // Block problematic keys with or without modifiers
  if (problematicKeys.includes(event.key)) {
    return true;
  }

  return false;
}

/**
 * Check if an element is a search overlay
 * @param {Element} element - The element to check
 * @returns {boolean} - True if it's a search overlay
 */
function isSearchOverlay(element) {
  if (!element) return false;
  
  // Check for Algolia search overlay classes
  const algoliaClasses = ['aa-DetachedOverlay', 'aa-DetachedContainer', 'aa-Input', 'aa-Form'];
  if (algoliaClasses.some(className => element.classList.contains(className))) {
    return true;
  }
  
  // Check for common search overlay patterns
  const searchPatterns = [
    'search-overlay',
    'search-modal',
    'search-dialog',
    'autocomplete',
    'search-panel',
    'aa-',
    'quarto-search'
  ];
  
  if (searchPatterns.some(pattern => 
    element.classList.contains(pattern) || 
    element.id.includes(pattern) ||
    element.className.includes(pattern)
  )) {
    return true;
  }
  
  return false;
}

/**
 * Set the input element to protect
 * @param {HTMLElement} element - The input element
 */
export function setInputElement(element) {
  inputElement = element;
}

/**
 * Clean up nuclear hotkey prevention
 */
export function cleanupNuclearHotkeyPrevention() {
  cleanupFunctions.forEach(cleanup => {
    if (typeof cleanup === 'function') {
      cleanup();
    }
  });
  cleanupFunctions = [];
  isActive = false;
  inputElement = null;
}

/**
 * Disable nuclear hotkey prevention
 */
export function disableNuclearHotkeyPrevention() {
  isActive = false;
}

/**
 * Enable nuclear hotkey prevention
 */
export function enableNuclearHotkeyPrevention() {
  isActive = true;
}