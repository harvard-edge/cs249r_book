/**
 * Hotkey Prevention Utility
 * Prevents parent site hotkeys from triggering when typing in shadow DOM inputs
 */

/**
 * Prevents hotkey conflicts by capturing and stopping propagation of keyboard events
 * @param {HTMLElement} element - The input/textarea element to protect
 * @param {Object} options - Configuration options
 * @param {boolean} options.preventAll - Whether to prevent all keyboard events (default: false)
 * @param {Array<string>} options.allowedKeys - Keys that should still trigger parent hotkeys (default: [])
 * @param {Array<string>} options.blockedKeys - Specific keys to always block (default: [])
 * @param {boolean} options.allowModifiers - Whether to allow modifier key combinations (default: false)
 */
export function preventHotkeyConflicts(element, options = {}) {
  if (!element) {
    console.warn('preventHotkeyConflicts: No element provided');
    return;
  }

  const {
    preventAll = false,
    allowedKeys = [],
    blockedKeys = [],
    allowModifiers = false
  } = options;

  // List of common hotkey combinations that should be blocked
  const commonHotkeys = [
    'b', 'B', 'c', 'C', 'v', 'V', 'x', 'X', 'z', 'Z', 'y', 'Y',
    'a', 'A', 's', 'S', 'f', 'F', 'g', 'G', 'h', 'H', 'j', 'J',
    'k', 'K', 'l', 'L', 'n', 'N', 'm', 'M', 'p', 'P', 'r', 'R',
    't', 'T', 'u', 'U', 'w', 'W', 'q', 'Q', 'e', 'E', 'd', 'D',
    'i', 'I', 'o', 'O', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    'Enter', 'Escape', 'Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
    'Home', 'End', 'PageUp', 'PageDown', 'Delete', 'Backspace', 'Insert'
  ];

  const shouldBlockEvent = (event) => {
    // Always allow certain essential keys for basic functionality
    const essentialKeys = ['Tab', 'Escape', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
    if (essentialKeys.includes(event.key)) {
      return false;
    }

    // If preventAll is true, block everything except allowed keys
    if (preventAll) {
      return !allowedKeys.includes(event.key);
    }

    // Check if the key is in the blocked list
    if (blockedKeys.includes(event.key)) {
      return true;
    }

    // Block common hotkey combinations
    if (commonHotkeys.includes(event.key)) {
      // If modifiers are not allowed, block any key with modifiers
      if (!allowModifiers && (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey)) {
        return true;
      }
      
      // Block specific problematic combinations
      if (event.shiftKey && ['B', 'C', 'V', 'X', 'Z', 'A', 'S', 'F', 'G', 'H', 'J', 'K', 'L', 'N', 'M', 'P', 'R', 'T', 'U', 'W', 'Q', 'E', 'D', 'I', 'O'].includes(event.key)) {
        return true;
      }
      
      if ((event.ctrlKey || event.metaKey) && commonHotkeys.includes(event.key)) {
        return true;
      }
    }

    return false;
  };

  const handleKeyDown = (event) => {
    if (shouldBlockEvent(event)) {
      event.stopPropagation();
      event.stopImmediatePropagation();
      // Don't preventDefault for typing keys, only stop propagation
      if (!['a', 'A', 'c', 'C', 'v', 'V', 'x', 'X', 'z', 'Z'].includes(event.key)) {
        event.preventDefault();
      }
    }
  };

  const handleKeyUp = (event) => {
    if (shouldBlockEvent(event)) {
      event.stopPropagation();
      event.stopImmediatePropagation();
    }
  };

  const handleKeyPress = (event) => {
    if (shouldBlockEvent(event)) {
      event.stopPropagation();
      event.stopImmediatePropagation();
    }
  };

  // Add event listeners with capture=true to intercept events before they bubble up
  element.addEventListener('keydown', handleKeyDown, { capture: true });
  element.addEventListener('keyup', handleKeyUp, { capture: true });
  element.addEventListener('keypress', handleKeyPress, { capture: true });

  // Return cleanup function
  return () => {
    element.removeEventListener('keydown', handleKeyDown, { capture: true });
    element.removeEventListener('keyup', handleKeyUp, { capture: true });
    element.removeEventListener('keypress', handleKeyPress, { capture: true });
  };
}

/**
 * Apply hotkey prevention to all input and textarea elements in a shadow root
 * @param {ShadowRoot} shadowRoot - The shadow root containing the inputs
 * @param {Object} options - Configuration options (same as preventHotkeyConflicts)
 */
export function preventHotkeyConflictsInShadowRoot(shadowRoot, options = {}) {
  if (!shadowRoot) {
    console.warn('preventHotkeyConflictsInShadowRoot: No shadow root provided');
    return [];
  }

  const cleanupFunctions = [];
  
  // Find all input and textarea elements
  const inputs = shadowRoot.querySelectorAll('input, textarea');
  
  inputs.forEach(input => {
    const cleanup = preventHotkeyConflicts(input, options);
    if (cleanup) {
      cleanupFunctions.push(cleanup);
    }
  });

  // Also watch for dynamically added inputs
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === Node.ELEMENT_NODE) {
          // Check if the added node is an input/textarea
          if (node.tagName === 'INPUT' || node.tagName === 'TEXTAREA') {
            const cleanup = preventHotkeyConflicts(node, options);
            if (cleanup) {
              cleanupFunctions.push(cleanup);
            }
          }
          
          // Check for inputs/textarea within the added node
          const childInputs = node.querySelectorAll ? node.querySelectorAll('input, textarea') : [];
          childInputs.forEach(input => {
            const cleanup = preventHotkeyConflicts(input, options);
            if (cleanup) {
              cleanupFunctions.push(cleanup);
            }
          });
        }
      });
    });
  });

  observer.observe(shadowRoot, {
    childList: true,
    subtree: true
  });

  // Return cleanup function that removes all listeners and stops observation
  return () => {
    cleanupFunctions.forEach(cleanup => cleanup());
    observer.disconnect();
  };
}

/**
 * Create a more aggressive hotkey prevention that blocks all keyboard events
 * except for essential typing functionality
 * @param {HTMLElement} element - The input/textarea element to protect
 */
export function createAggressiveHotkeyPrevention(element) {
  return preventHotkeyConflicts(element, {
    preventAll: true,
    allowedKeys: [
      'Backspace', 'Delete', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown',
      'Home', 'End', 'Tab', 'Escape'
    ],
    allowModifiers: false
  });
}

/**
 * Create a selective hotkey prevention that only blocks specific problematic keys
 * @param {HTMLElement} element - The input/textarea element to protect
 */
export function createSelectiveHotkeyPrevention(element) {
  return preventHotkeyConflicts(element, {
    preventAll: false,
    blockedKeys: ['b', 'B', 'c', 'C', 'v', 'V', 'x', 'X', 'z', 'Z', 'a', 'A', 's', 'S'],
    allowModifiers: false
  });
}