/**
 * Efficient Hotkey Prevention Utility
 * Lightweight solution that only blocks specific hotkey combinations
 * without interfering with normal typing performance
 */

/**
 * Prevents only specific problematic hotkey combinations from propagating
 * @param {HTMLElement} element - The input/textarea element to protect
 */
export function preventSpecificHotkeys(element) {
  if (!element) return;

  // Only block these specific problematic combinations
  const blockedCombinations = [
    { key: 'b', shift: true },      // Shift+B
    { key: 'B', shift: true },      // Shift+B (uppercase)
    { key: 'c', ctrl: true },       // Ctrl+C
    { key: 'C', ctrl: true },       // Ctrl+C (uppercase)
    { key: 'v', ctrl: true },       // Ctrl+V
    { key: 'V', ctrl: true },       // Ctrl+V (uppercase)
    { key: 'x', ctrl: true },       // Ctrl+X
    { key: 'X', ctrl: true },       // Ctrl+X (uppercase)
    { key: 'z', ctrl: true },       // Ctrl+Z
    { key: 'Z', ctrl: true },       // Ctrl+Z (uppercase)
    { key: 'a', ctrl: true },       // Ctrl+A
    { key: 'A', ctrl: true },       // Ctrl+A (uppercase)
    { key: 's', ctrl: true },       // Ctrl+S
    { key: 'S', ctrl: true },       // Ctrl+S (uppercase)
    { key: 'f', ctrl: true },       // Ctrl+F
    { key: 'F', ctrl: true },       // Ctrl+F (uppercase)
    { key: 'g', ctrl: true },       // Ctrl+G
    { key: 'G', ctrl: true },       // Ctrl+G (uppercase)
    { key: 'h', ctrl: true },       // Ctrl+H
    { key: 'H', ctrl: true },       // Ctrl+H (uppercase)
    { key: 'j', ctrl: true },       // Ctrl+J
    { key: 'J', ctrl: true },       // Ctrl+J (uppercase)
    { key: 'k', ctrl: true },       // Ctrl+K
    { key: 'K', ctrl: true },       // Ctrl+K (uppercase)
    { key: 'l', ctrl: true },       // Ctrl+L
    { key: 'L', ctrl: true },       // Ctrl+L (uppercase)
    { key: 'n', ctrl: true },       // Ctrl+N
    { key: 'N', ctrl: true },       // Ctrl+N (uppercase)
    { key: 'm', ctrl: true },       // Ctrl+M
    { key: 'M', ctrl: true },       // Ctrl+M (uppercase)
    { key: 'p', ctrl: true },       // Ctrl+P
    { key: 'P', ctrl: true },       // Ctrl+P (uppercase)
    { key: 'r', ctrl: true },       // Ctrl+R
    { key: 'R', ctrl: true },       // Ctrl+R (uppercase)
    { key: 't', ctrl: true },       // Ctrl+T
    { key: 'T', ctrl: true },       // Ctrl+T (uppercase)
    { key: 'u', ctrl: true },       // Ctrl+U
    { key: 'U', ctrl: true },       // Ctrl+U (uppercase)
    { key: 'w', ctrl: true },       // Ctrl+W
    { key: 'W', ctrl: true },       // Ctrl+W (uppercase)
    { key: 'q', ctrl: true },       // Ctrl+Q
    { key: 'Q', ctrl: true },       // Ctrl+Q (uppercase)
    { key: 'e', ctrl: true },       // Ctrl+E
    { key: 'E', ctrl: true },       // Ctrl+E (uppercase)
    { key: 'd', ctrl: true },       // Ctrl+D
    { key: 'D', ctrl: true },       // Ctrl+D (uppercase)
    { key: 'i', ctrl: true },       // Ctrl+I
    { key: 'I', ctrl: true },       // Ctrl+I (uppercase)
    { key: 'o', ctrl: true },       // Ctrl+O
    { key: 'O', ctrl: true },       // Ctrl+O (uppercase)
  ];

  const shouldBlockEvent = (event) => {
    // Only check for specific blocked combinations
    return blockedCombinations.some(combo => 
      event.key === combo.key && 
      event.ctrlKey === !!combo.ctrl && 
      event.shiftKey === !!combo.shift &&
      !event.altKey && // Don't block Alt combinations
      !event.metaKey    // Don't block Cmd combinations
    );
  };

  const handleKeyDown = (event) => {
    // Only process if it's a blocked combination
    if (shouldBlockEvent(event)) {
      event.stopPropagation();
      event.stopImmediatePropagation();
      // Don't preventDefault to allow normal typing
    }
    // For all other keys, do nothing - let them work normally
  };

  // Only add keydown listener, no keyup or keypress needed
  element.addEventListener('keydown', handleKeyDown, { capture: true });

  // Return cleanup function
  return () => {
    element.removeEventListener('keydown', handleKeyDown, { capture: true });
  };
}

/**
 * Apply hotkey prevention to all input and textarea elements in a shadow root
 * @param {ShadowRoot} shadowRoot - The shadow root containing the inputs
 */
export function preventHotkeysInShadowRoot(shadowRoot) {
  if (!shadowRoot) return [];

  const cleanupFunctions = [];
  
  // Find all input and textarea elements
  const inputs = shadowRoot.querySelectorAll('input, textarea');
  
  inputs.forEach(input => {
    const cleanup = preventSpecificHotkeys(input);
    if (cleanup) {
      cleanupFunctions.push(cleanup);
    }
  });

  // Watch for dynamically added inputs
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === Node.ELEMENT_NODE) {
          // Check if the added node is an input/textarea
          if (node.tagName === 'INPUT' || node.tagName === 'TEXTAREA') {
            const cleanup = preventSpecificHotkeys(node);
            if (cleanup) {
              cleanupFunctions.push(cleanup);
            }
          }
          
          // Check for inputs/textarea within the added node
          const childInputs = node.querySelectorAll ? node.querySelectorAll('input, textarea') : [];
          childInputs.forEach(input => {
            const cleanup = preventSpecificHotkeys(input);
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

  // Return cleanup function
  return () => {
    cleanupFunctions.forEach(cleanup => cleanup());
    observer.disconnect();
  };
}

/**
 * Simple function to apply to a single input element
 * @param {HTMLElement} element - The input element to protect
 */
export function protectInput(element) {
  return preventSpecificHotkeys(element);
}