/**
 * Focused Hotkey Prevention
 * Specifically targets the 'f' key and other common hotkeys when typing in inputs
 */

let shadowRoot = null;
let cleanupFunction = null;

/**
 * Initialize focused hotkey prevention
 * @param {ShadowRoot} root - The shadow root to protect
 */
export function initializeFocusedHotkeyPrevention(root) {
  shadowRoot = root;
  
  // Clean up any existing listeners
  if (cleanupFunction) {
    cleanupFunction();
  }

  const handleDocumentKeyDown = (event) => {
    if (!shadowRoot) return;

    // Check if the event is coming from within our shadow DOM
    const isFromShadowDOM = shadowRoot.contains(event.target) || 
                           event.target.getRootNode() === shadowRoot;

    if (!isFromShadowDOM) return;

    // Check if we're focused on an input or textarea
    const activeElement = document.activeElement;
    const isInputFocused = activeElement && 
                          (activeElement.tagName === 'INPUT' || 
                           activeElement.tagName === 'TEXTAREA') &&
                          shadowRoot.contains(activeElement);

    if (!isInputFocused) return;

    // Block specific problematic keys
    const problematicKeys = {
      'f': true, 'F': true,  // Search hotkey
      'b': true, 'B': true,  // Bold hotkey
      'c': true, 'C': true,  // Copy hotkey
      'v': true, 'V': true,  // Paste hotkey
      'x': true, 'X': true,  // Cut hotkey
      'z': true, 'Z': true,  // Undo hotkey
      'a': true, 'A': true,  // Select all hotkey
      's': true, 'S': true,  // Save hotkey
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

    const key = event.key;
    const hasModifiers = event.ctrlKey || event.metaKey || event.altKey || event.shiftKey;

    // Block problematic keys with modifiers
    if (problematicKeys[key] && hasModifiers) {
      event.stopPropagation();
      event.stopImmediatePropagation();
      return;
    }

    // Block standalone 'f' key (most common issue)
    if ((key === 'f' || key === 'F') && !hasModifiers) {
      event.stopPropagation();
      event.stopImmediatePropagation();
      return;
    }

    // Block other standalone problematic keys
    const standaloneBlockedKeys = ['b', 'B', 'c', 'C', 'v', 'V', 'x', 'X', 'z', 'Z'];
    if (standaloneBlockedKeys.includes(key) && !hasModifiers) {
      event.stopPropagation();
      event.stopImmediatePropagation();
      return;
    }
  };

  // Add listener with highest priority
  document.addEventListener('keydown', handleDocumentKeyDown, { 
    capture: true, 
    passive: false 
  });

  // Store cleanup function
  cleanupFunction = () => {
    document.removeEventListener('keydown', handleDocumentKeyDown, { 
      capture: true 
    });
  };

  return cleanupFunction;
}

/**
 * Clean up focused hotkey prevention
 */
export function cleanupFocusedHotkeyPrevention() {
  if (cleanupFunction) {
    cleanupFunction();
    cleanupFunction = null;
  }
}