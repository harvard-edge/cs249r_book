/**
 * Aggressive Focus Protection
 * Prevents focus from being stolen by parent site elements and maintains focus in shadow DOM
 */

let shadowRoot = null;
let isActive = false;
let lastFocusedElement = null;
let focusProtectionInterval = null;
let cleanupFunctions = [];

/**
 * Initialize aggressive focus protection
 * @param {ShadowRoot} root - The shadow root to protect
 */
export function initializeAggressiveFocusProtection(root) {
  shadowRoot = root;
  isActive = true;
  
  // Clean up any existing listeners
  cleanupAggressiveFocusProtection();

  // Track focus changes within shadow DOM
  const trackFocus = (event) => {
    if (!shadowRoot) return;
    
    const target = event.target;
    if (shadowRoot.contains(target) && 
        (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) {
      lastFocusedElement = target;
      console.log('Focus tracked in shadow DOM:', target);
    }
  };

  // Listen for focus events
  const focusListener = () => {
    shadowRoot.addEventListener('focusin', trackFocus, true);
    return () => shadowRoot.removeEventListener('focusin', trackFocus, true);
  };

  cleanupFunctions.push(focusListener());

  // Prevent focus from being stolen
  const preventFocusStealing = (event) => {
    if (!isActive || !shadowRoot || !lastFocusedElement) return;

    const target = event.target;
    if (!target) return;

    // If focus is being moved outside our shadow DOM
    if (!shadowRoot.contains(target)) {
      // Check if this is a search overlay or parent site element
      if (isSearchOverlay(target) || isParentSiteElement(target)) {
        console.log('Preventing focus steal to:', target);
        
        // Prevent the focus change
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        
        // Restore focus to our input immediately
        if (lastFocusedElement && lastFocusedElement.focus) {
          lastFocusedElement.focus();
        }
        
        return false;
      }
    }
  };

  // Add focus prevention listeners at multiple levels
  const documentFocusListener = () => {
    document.addEventListener('focusin', preventFocusStealing, { 
      capture: true, 
      passive: false 
    });
    return () => document.removeEventListener('focusin', preventFocusStealing, { capture: true });
  };

  const windowFocusListener = () => {
    window.addEventListener('focusin', preventFocusStealing, { 
      capture: true, 
      passive: false 
    });
    return () => window.removeEventListener('focusin', preventFocusStealing, { capture: true });
  };

  cleanupFunctions.push(documentFocusListener());
  cleanupFunctions.push(windowFocusListener());

  // Start aggressive focus protection interval
  startFocusProtectionInterval();

  return () => cleanupAggressiveFocusProtection();
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
    'aa-', // Algolia prefix
    'quarto-search'
  ];
  
  if (searchPatterns.some(pattern => 
    element.classList.contains(pattern) || 
    element.id.includes(pattern) ||
    element.className.includes(pattern) ||
    element.tagName.toLowerCase().includes('search')
  )) {
    return true;
  }
  
  return false;
}

/**
 * Check if an element is from the parent site
 * @param {Element} element - The element to check
 * @returns {boolean} - True if it's from parent site
 */
function isParentSiteElement(element) {
  if (!element) return false;
  
  // Check if element is not within our shadow DOM
  if (shadowRoot && shadowRoot.contains(element)) {
    return false;
  }
  
  // Check if element is in the main document (not shadow DOM)
  if (element.getRootNode && element.getRootNode() === document) {
    return true;
  }
  
  return false;
}

/**
 * Start aggressive focus protection interval
 */
function startFocusProtectionInterval() {
  if (focusProtectionInterval) {
    clearInterval(focusProtectionInterval);
  }
  
  focusProtectionInterval = setInterval(() => {
    if (!isActive || !shadowRoot || !lastFocusedElement) return;
    
    const activeElement = document.activeElement;
    
    // If focus is not in our shadow DOM
    if (activeElement && !shadowRoot.contains(activeElement)) {
      // Check if the active element is a search overlay or parent site element
      if (isSearchOverlay(activeElement) || isParentSiteElement(activeElement)) {
        console.log('Focus protection: restoring focus to shadow DOM input');
        
        // Restore focus to our input
        if (lastFocusedElement && lastFocusedElement.focus) {
          lastFocusedElement.focus();
        }
      }
    }
  }, 10); // Check every 10ms for aggressive protection
}

/**
 * Stop focus protection interval
 */
function stopFocusProtectionInterval() {
  if (focusProtectionInterval) {
    clearInterval(focusProtectionInterval);
    focusProtectionInterval = null;
  }
}

/**
 * Clean up aggressive focus protection
 */
export function cleanupAggressiveFocusProtection() {
  cleanupFunctions.forEach(cleanup => {
    if (typeof cleanup === 'function') {
      cleanup();
    }
  });
  cleanupFunctions = [];
  stopFocusProtectionInterval();
  isActive = false;
  lastFocusedElement = null;
}

/**
 * Force focus to a specific element
 * @param {HTMLElement} element - The element to focus
 */
export function forceFocus(element) {
  if (!element || !element.focus) return;
  
  lastFocusedElement = element;
  element.focus();
  
  // Also trigger focus event
  const focusEvent = new Event('focus', { bubbles: true });
  element.dispatchEvent(focusEvent);
}

/**
 * Get the last focused element
 * @returns {HTMLElement|null} - The last focused element
 */
export function getLastFocusedElement() {
  return lastFocusedElement;
}

/**
 * Disable focus protection
 */
export function disableFocusProtection() {
  isActive = false;
}

/**
 * Enable focus protection
 */
export function enableFocusProtection() {
  isActive = true;
}