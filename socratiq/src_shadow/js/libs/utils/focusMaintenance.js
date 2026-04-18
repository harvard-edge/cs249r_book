/**
 * Focus Maintenance Utility
 * Prevents focus from being stolen by parent site elements (like search overlays)
 * and maintains focus in shadow DOM inputs
 */

let shadowRoot = null;
let isActive = false;
let lastFocusedElement = null;
let focusMaintenanceInterval = null;
let cleanupFunctions = [];

/**
 * Initialize focus maintenance
 * @param {ShadowRoot} root - The shadow root to protect
 */
export function initializeFocusMaintenance(root) {
  shadowRoot = root;
  isActive = true;
  
  // Clean up any existing listeners
  cleanupFocusMaintenance();

  // Track the last focused element in our shadow DOM
  const trackFocus = (event) => {
    if (!shadowRoot) return;
    
    const target = event.target;
    if (shadowRoot.contains(target) && 
        (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) {
      lastFocusedElement = target;
    }
  };

  // Listen for focus events within shadow DOM
  const focusListener = () => {
    shadowRoot.addEventListener('focusin', trackFocus, true);
    return () => shadowRoot.removeEventListener('focusin', trackFocus, true);
  };

  cleanupFunctions.push(focusListener());

  // Prevent focus from being stolen by parent elements
  const preventFocusStealing = (event) => {
    if (!isActive || !shadowRoot) return;

    // Check if focus is being moved outside our shadow DOM
    const target = event.target;
    if (!target) return;

    // If the target is not within our shadow DOM, and we had focus in shadow DOM
    if (!shadowRoot.contains(target) && lastFocusedElement) {
      // Check if this is a search overlay or similar element
      if (isSearchOverlay(target) || isParentSiteElement(target)) {
        // Prevent the focus change
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        
        // Restore focus to our input
        if (lastFocusedElement && lastFocusedElement.focus) {
          setTimeout(() => {
            lastFocusedElement.focus();
          }, 0);
        }
        
        return false;
      }
    }
  };

  // Add focus prevention listeners
  const focusPreventionListener = () => {
    document.addEventListener('focusin', preventFocusStealing, { 
      capture: true, 
      passive: false 
    });
    return () => document.removeEventListener('focusin', preventFocusStealing, { capture: true });
  };

  cleanupFunctions.push(focusPreventionListener());

  // Start focus maintenance interval
  startFocusMaintenanceInterval();

  return () => cleanupFocusMaintenance();
}

/**
 * Check if an element is a search overlay
 * @param {Element} element - The element to check
 * @returns {boolean} - True if it's a search overlay
 */
function isSearchOverlay(element) {
  if (!element) return false;
  
  // Check for Algolia search overlay classes
  const algoliaClasses = ['aa-DetachedOverlay', 'aa-DetachedContainer', 'aa-Input'];
  if (algoliaClasses.some(className => element.classList.contains(className))) {
    return true;
  }
  
  // Check for common search overlay patterns
  const searchPatterns = [
    'search-overlay',
    'search-modal',
    'search-dialog',
    'autocomplete',
    'search-panel'
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
 * Start focus maintenance interval
 */
function startFocusMaintenanceInterval() {
  if (focusMaintenanceInterval) {
    clearInterval(focusMaintenanceInterval);
  }
  
  focusMaintenanceInterval = setInterval(() => {
    if (!isActive || !shadowRoot) return;
    
    const activeElement = document.activeElement;
    
    // If focus is not in our shadow DOM, but we have a last focused element
    if (activeElement && !shadowRoot.contains(activeElement) && lastFocusedElement) {
      // Check if the active element is a search overlay
      if (isSearchOverlay(activeElement) || isParentSiteElement(activeElement)) {
        // Restore focus to our input
        if (lastFocusedElement && lastFocusedElement.focus) {
          lastFocusedElement.focus();
        }
      }
    }
  }, 50); // Check every 50ms
}

/**
 * Stop focus maintenance interval
 */
function stopFocusMaintenanceInterval() {
  if (focusMaintenanceInterval) {
    clearInterval(focusMaintenanceInterval);
    focusMaintenanceInterval = null;
  }
}

/**
 * Clean up focus maintenance
 */
export function cleanupFocusMaintenance() {
  cleanupFunctions.forEach(cleanup => {
    if (typeof cleanup === 'function') {
      cleanup();
    }
  });
  cleanupFunctions = [];
  stopFocusMaintenanceInterval();
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
}

/**
 * Get the last focused element
 * @returns {HTMLElement|null} - The last focused element
 */
export function getLastFocusedElement() {
  return lastFocusedElement;
}

/**
 * Disable focus maintenance
 */
export function disableFocusMaintenance() {
  isActive = false;
}

/**
 * Enable focus maintenance
 */
export function enableFocusMaintenance() {
  isActive = true;
}