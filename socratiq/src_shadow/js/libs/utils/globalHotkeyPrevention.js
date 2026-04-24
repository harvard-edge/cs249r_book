/**
 * Global Hotkey Prevention Utility
 * Prevents parent site hotkeys by intercepting events at the document level
 * when focus is within the shadow DOM
 */

let shadowRoot = null;
let isActive = false;

/**
 * Initialize global hotkey prevention for a shadow root
 * @param {ShadowRoot} root - The shadow root to protect
 */
export function initializeGlobalHotkeyPrevention(root) {
  shadowRoot = root;
  isActive = true;
  
  // List of keys that commonly trigger parent site hotkeys
  const problematicKeys = [
    'f', 'F', 'b', 'B', 'c', 'C', 'v', 'V', 'x', 'X', 'z', 'Z', 
    'a', 'A', 's', 'S', 'g', 'G', 'h', 'H', 'j', 'J', 'k', 'K', 
    'l', 'L', 'n', 'N', 'm', 'M', 'p', 'P', 'r', 'R', 't', 'T', 
    'u', 'U', 'w', 'W', 'q', 'Q', 'e', 'E', 'd', 'D', 'i', 'I', 
    'o', 'O', 'y', 'Y'
  ];

  const handleGlobalKeyDown = (event) => {
    if (!isActive || !shadowRoot) return;

    // Check if the event target is within our shadow DOM
    const isWithinShadowDOM = shadowRoot.contains(event.target) || 
                             event.target.getRootNode() === shadowRoot;

    if (!isWithinShadowDOM) return;

    // Check if this is a problematic key combination
    const isProblematicKey = problematicKeys.includes(event.key);
    const hasModifiers = event.ctrlKey || event.metaKey || event.altKey || event.shiftKey;
    
    // Block specific problematic combinations
    if (isProblematicKey && hasModifiers) {
      event.stopPropagation();
      event.stopImmediatePropagation();
      return;
    }

    // Block standalone 'f' key (common search hotkey)
    if (event.key === 'f' || event.key === 'F') {
      // Only block if it's not part of normal typing (no modifiers)
      if (!hasModifiers) {
        event.stopPropagation();
        event.stopImmediatePropagation();
        return;
      }
    }

    // Block other common standalone hotkeys
    const standaloneHotkeys = ['b', 'B', 'c', 'C', 'v', 'V', 'x', 'X', 'z', 'Z'];
    if (standaloneHotkeys.includes(event.key) && !hasModifiers) {
      event.stopPropagation();
      event.stopImmediatePropagation();
      return;
    }
  };

  // Add listener with highest priority (capture phase)
  document.addEventListener('keydown', handleGlobalKeyDown, { 
    capture: true, 
    passive: false 
  });

  // Return cleanup function
  return () => {
    document.removeEventListener('keydown', handleGlobalKeyDown, { 
      capture: true 
    });
    isActive = false;
  };
}

/**
 * Check if focus is within the shadow DOM
 * @param {Event} event - The keyboard event
 * @returns {boolean} - True if focus is within shadow DOM
 */
function isFocusInShadowDOM(event) {
  if (!shadowRoot) return false;
  
  const activeElement = document.activeElement;
  if (!activeElement) return false;
  
  // Check if active element is within shadow DOM
  return shadowRoot.contains(activeElement) || 
         activeElement.getRootNode() === shadowRoot;
}

/**
 * More aggressive approach - blocks all keyboard events when focus is in shadow DOM
 * Use this if the selective approach doesn't work
 */
export function initializeAggressiveGlobalPrevention(root) {
  shadowRoot = root;
  isActive = true;

  const handleGlobalKeyDown = (event) => {
    if (!isActive || !shadowRoot) return;

    // Check if focus is within our shadow DOM
    if (!isFocusInShadowDOM(event)) return;

    // Block all keyboard events except essential ones
    const essentialKeys = [
      'Tab', 'Escape', 'Enter', 'Backspace', 'Delete',
      'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
      'Home', 'End', 'PageUp', 'PageDown', 'Insert'
    ];

    if (!essentialKeys.includes(event.key)) {
      event.stopPropagation();
      event.stopImmediatePropagation();
    }
  };

  document.addEventListener('keydown', handleGlobalKeyDown, { 
    capture: true, 
    passive: false 
  });

  return () => {
    document.removeEventListener('keydown', handleGlobalKeyDown, { 
      capture: true 
    });
    isActive = false;
  };
}

/**
 * Disable global hotkey prevention
 */
export function disableGlobalHotkeyPrevention() {
  isActive = false;
}

/**
 * Enable global hotkey prevention
 */
export function enableGlobalHotkeyPrevention() {
  isActive = true;
}