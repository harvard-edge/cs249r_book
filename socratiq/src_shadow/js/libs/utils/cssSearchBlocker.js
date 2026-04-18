/**
 * CSS Search Blocker
 * Uses CSS to hide search overlays and prevent them from appearing
 */

let shadowRoot = null;
let isActive = false;
let styleElement = null;

/**
 * Initialize CSS search blocker
 * @param {ShadowRoot} root - The shadow root to protect
 */
export function initializeCSSSearchBlocker(root) {
  shadowRoot = root;
  isActive = true;
  
  // Clean up any existing styles
  cleanupCSSSearchBlocker();

  // Create style element to inject CSS
  styleElement = document.createElement('style');
  styleElement.id = 'search-blocker-styles';
  
  // CSS to hide search overlays
  const css = `
    /* Hide Algolia search overlays */
    .aa-DetachedOverlay,
    .aa-DetachedContainer,
    .aa-DetachedFormContainer,
    .aa-Panel,
    .aa-Source,
    .aa-Input,
    .aa-Form,
    [class*="aa-"],
    [class*="search-overlay"],
    [class*="search-modal"],
    [class*="search-dialog"],
    [class*="autocomplete"],
    [class*="quarto-search"] {
      display: none !important;
      visibility: hidden !important;
      opacity: 0 !important;
      pointer-events: none !important;
      position: absolute !important;
      left: -9999px !important;
      top: -9999px !important;
      width: 0 !important;
      height: 0 !important;
      overflow: hidden !important;
      z-index: -1 !important;
    }
    
    /* Prevent search overlays from being created */
    body > div[class*="aa-"],
    body > div[class*="search"],
    body > div[class*="autocomplete"] {
      display: none !important;
    }
    
    /* Hide any search-related elements */
    [id*="search"],
    [id*="autocomplete"],
    [id*="aa-"] {
      display: none !important;
    }
  `;
  
  styleElement.textContent = css;
  
  // Inject CSS into document head
  document.head.appendChild(styleElement);
  
  // Also inject into shadow DOM if possible
  if (shadowRoot) {
    const shadowStyle = document.createElement('style');
    shadowStyle.textContent = css;
    shadowRoot.appendChild(shadowStyle);
  }
  
  // Use mutation observer to continuously hide search overlays
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === Node.ELEMENT_NODE) {
          hideSearchOverlay(node);
          
          // Also check for search overlays within the added node
          if (node.querySelectorAll) {
            const searchOverlays = node.querySelectorAll('[class*="aa-"], [class*="search"], [class*="autocomplete"]');
            searchOverlays.forEach(overlay => {
              hideSearchOverlay(overlay);
            });
          }
        }
      });
    });
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
  
  // Store observer for cleanup
  styleElement._observer = observer;
  
  return () => cleanupCSSSearchBlocker();
}

/**
 * Hide a search overlay element
 * @param {Element} element - The element to hide
 */
function hideSearchOverlay(element) {
  if (!element) return;
  
  // Check if this is a search overlay
  if (isSearchOverlay(element)) {
    console.log('Hiding search overlay:', element);
    
    // Apply hiding styles
    element.style.display = 'none';
    element.style.visibility = 'hidden';
    element.style.opacity = '0';
    element.style.pointerEvents = 'none';
    element.style.position = 'absolute';
    element.style.left = '-9999px';
    element.style.top = '-9999px';
    element.style.width = '0';
    element.style.height = '0';
    element.style.overflow = 'hidden';
    element.style.zIndex = '-1';
    
    // Remove the element entirely
    if (element.parentNode) {
      element.parentNode.removeChild(element);
    }
  }
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
 * Clean up CSS search blocker
 */
export function cleanupCSSSearchBlocker() {
  if (styleElement) {
    if (styleElement._observer) {
      styleElement._observer.disconnect();
    }
    if (styleElement.parentNode) {
      styleElement.parentNode.removeChild(styleElement);
    }
    styleElement = null;
  }
  isActive = false;
}

/**
 * Disable CSS search blocker
 */
export function disableCSSSearchBlocker() {
  isActive = false;
}

/**
 * Enable CSS search blocker
 */
export function enableCSSSearchBlocker() {
  isActive = true;
}