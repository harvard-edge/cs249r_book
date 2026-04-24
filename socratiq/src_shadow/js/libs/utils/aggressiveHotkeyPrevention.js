/**
 * Aggressive Hotkey Prevention
 * Intercepts events at the window level to prevent Algolia search and other parent site hotkeys
 * from triggering when typing in the shadow DOM widget
 */

let shadowRoot = null;
let isActive = false;
let cleanupFunctions = [];

/**
 * Initialize aggressive hotkey prevention
 * @param {ShadowRoot} root - The shadow root to protect
 */
export function initializeAggressiveHotkeyPrevention(root) {
  shadowRoot = root;
  isActive = true;
  
  // Clean up any existing listeners
  cleanupAggressiveHotkeyPrevention();

  // List of keys that commonly trigger parent site functionality
  const problematicKeys = {
    'f': true, 'F': true,  // Search hotkey (Algolia)
    's': true, 'S': true,  // Save/search hotkey
    'b': true, 'B': true,  // Bold hotkey
    'c': true, 'C': true,  // Copy hotkey
    'v': true, 'V': true,  // Paste hotkey
    'x': true, 'X': true,  // Cut hotkey
    'z': true, 'Z': true,  // Undo hotkey
    'a': true, 'A': true,  // Select all hotkey
    'g': true, 'G': true,  // Find next hotkey
    'h': true, 'H': true,  // Find and replace hotkey
    'j': true, 'J': true,  // Common hotkey
    'k': true, 'K': true,  // Common hotkey
    'l': true, 'L': true,  // Common hotkey
    'n': true, 'N': true,  // New document hotkey
    'm': true, 'M': true,  // Common hotkey
    'p': true, 'P': true,  // Print hotkey
    'r': true, 'R': true,  // Refresh hotkey
    't': true, 'T': true,  // New tab hotkey
    'u': true, 'U': true,  // Underline hotkey
    'w': true, 'W': true,  // Close tab hotkey
    'q': true, 'Q': true,  // Quit hotkey
    'e': true, 'E': true,  // Common hotkey
    'd': true, 'D': true,  // Common hotkey
    'i': true, 'I': true,  // Italic hotkey
    'o': true, 'O': true,  // Open hotkey
    'y': true, 'Y': true   // Redo hotkey
  };

  const handleWindowKeyDown = (event) => {
    if (!isActive || !shadowRoot) return;

    // Check if the event is coming from within our shadow DOM
    const isFromShadowDOM = isEventFromShadowDOM(event);
    if (!isFromShadowDOM) return;

    // Check if we're focused on an input or textarea
    const isInputFocused = isInputElementFocused();
    if (!isInputFocused) return;

    const key = event.key;
    const hasModifiers = event.ctrlKey || event.metaKey || event.altKey || event.shiftKey;

    // Block problematic keys with modifiers
    if (problematicKeys[key] && hasModifiers) {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      return false;
    }

    // Block standalone problematic keys (especially f and s for search)
    if (problematicKeys[key] && !hasModifiers) {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      return false;
    }
  };

  // Add multiple listeners at different levels for maximum coverage
  const windowListener = () => {
    window.addEventListener('keydown', handleWindowKeyDown, { 
      capture: true, 
      passive: false 
    });
    return () => window.removeEventListener('keydown', handleWindowKeyDown, { capture: true });
  };

  const documentListener = () => {
    document.addEventListener('keydown', handleWindowKeyDown, { 
      capture: true, 
      passive: false 
    });
    return () => document.removeEventListener('keydown', handleWindowKeyDown, { capture: true });
  };

  // Store cleanup functions
  cleanupFunctions.push(windowListener());
  cleanupFunctions.push(documentListener());

  // Also add a keyup listener to prevent any remaining issues
  const handleWindowKeyUp = (event) => {
    if (!isActive || !shadowRoot) return;

    const isFromShadowDOM = isEventFromShadowDOM(event);
    if (!isFromShadowDOM) return;

    const isInputFocused = isInputElementFocused();
    if (!isInputFocused) return;

    const key = event.key;
    const hasModifiers = event.ctrlKey || event.metaKey || event.altKey || event.shiftKey;

    if (problematicKeys[key] && (hasModifiers || !hasModifiers)) {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      return false;
    }
  };

  const keyupListener = () => {
    window.addEventListener('keyup', handleWindowKeyUp, { 
      capture: true, 
      passive: false 
    });
    return () => window.removeEventListener('keyup', handleWindowKeyUp, { capture: true });
  };

  cleanupFunctions.push(keyupListener());

  return () => cleanupAggressiveHotkeyPrevention();
}

/**
 * Check if the event is coming from within our shadow DOM
 * @param {Event} event - The keyboard event
 * @returns {boolean} - True if event is from shadow DOM
 */
function isEventFromShadowDOM(event) {
  if (!shadowRoot) return false;
  
  // Check if the event target is within our shadow DOM
  const target = event.target;
  if (!target) return false;
  
  // Check if target is within shadow DOM
  if (shadowRoot.contains(target)) return true;
  
  // Check if target's root is our shadow DOM
  if (target.getRootNode && target.getRootNode() === shadowRoot) return true;
  
  // Check if target is the shadow root itself
  if (target === shadowRoot) return true;
  
  return false;
}

/**
 * Check if focus is on an input or textarea element
 * @returns {boolean} - True if focused on input/textarea
 */
function isInputElementFocused() {
  const activeElement = document.activeElement;
  if (!activeElement) return false;
  
  // Check if active element is within shadow DOM
  if (!shadowRoot.contains(activeElement)) return false;
  
  // Check if it's an input or textarea
  return activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA';
}

/**
 * Clean up all aggressive hotkey prevention listeners
 */
export function cleanupAggressiveHotkeyPrevention() {
  cleanupFunctions.forEach(cleanup => {
    if (typeof cleanup === 'function') {
      cleanup();
    }
  });
  cleanupFunctions = [];
  isActive = false;
}

/**
 * Disable aggressive hotkey prevention
 */
export function disableAggressiveHotkeyPrevention() {
  isActive = false;
}

/**
 * Enable aggressive hotkey prevention
 */
export function enableAggressiveHotkeyPrevention() {
  isActive = true;
}

/**
 * Check if aggressive hotkey prevention is active
 * @returns {boolean} - True if active
 */
export function isAggressiveHotkeyPreventionActive() {
  return isActive;
}