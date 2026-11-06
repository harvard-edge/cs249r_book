/**
 * Persistent Tooltip System
 * 
 * Provides a global tooltip system that automatically initializes tooltips
 * for reference numbers across page reloads and navigation.
 */

import { ReferenceTooltip } from './referenceTooltip.js';
import {safeSourceUrl, safeSourcePosition} from './source-url.js';

class PersistentTooltipManager {
  constructor() {
    this.tooltip = null;
    this.initialized = false;
    this.init();
  }

  /**
   * Initialize the persistent tooltip system
   */
  init() {
    if (this.initialized) {
      return;
    }
    
    // Create a single tooltip instance for all reference numbers
    this.tooltip = new ReferenceTooltip();
    
    // Initialize tooltips on page load
    this.initializeTooltips();
    
    // Listen for dynamic content changes
    this.observeContentChanges();
    
    // Listen for navigation events
    this.setupNavigationListeners();
    
    this.initialized = true;
  }

  /**
   * Initialize tooltips for all reference numbers on the page
   */
  initializeTooltips() {
    // Find ALL reference numbers in main document
    const mainDocReferenceNumbers = document.querySelectorAll('.reference-number');
    
    // Find ALL reference numbers in shadow DOMs
    const shadowReferenceNumbers = this.findAllReferenceNumbersInShadowDOMs();
    
    // Combine all reference numbers
    const allReferenceNumbers = [...mainDocReferenceNumbers, ...shadowReferenceNumbers];
    
    // Attach tooltips to ALL reference numbers that have source data
    allReferenceNumbers.forEach((element, index) => {
      const hasSourceData = element.getAttribute('data-source-reference') || 
                           element.getAttribute('data-source-label') || 
                           element.getAttribute('data-source-url');
      
      if (hasSourceData) {
        this.attachTooltipToElement(element, index);
      }
    });
  }

  /**
   * Find all reference numbers in shadow DOMs
   * @returns {Array} Array of reference number elements
   */
  findAllReferenceNumbersInShadowDOMs() {
    const shadowReferenceNumbers = [];
    
    // Find all elements that have shadow roots
    const allElements = document.querySelectorAll('*');
    
    allElements.forEach(element => {
      if (element.shadowRoot) {
        // Search for reference numbers in this shadow root
        const referenceNumbers = element.shadowRoot.querySelectorAll('.reference-number');
        shadowReferenceNumbers.push(...referenceNumbers);
      }
    });
    
    return shadowReferenceNumbers;
  }

  /**
   * Attach tooltip functionality to a reference number element
   * @param {HTMLElement} element - The reference number element
   * @param {number} index - Index for logging
   */
  attachTooltipToElement(element, index) {
    // Check if element already has listeners
    if (element._tooltipMouseEnter) {
      return;
    }
    
    // Extract data from attributes
    const sourceUrl = safeSourceUrl(element.getAttribute('data-source-url'));
    const sourceLabel = element.getAttribute('data-source-label') || 
      (sourceUrl ? new URL(sourceUrl).origin : window.location.origin);
    
    const sourceData = {
      sourceReference: element.getAttribute('data-source-reference') || 'No source reference available',
      sourceLabel: sourceLabel,
      sourceUrl: sourceUrl,
      sourcePosition: safeSourcePosition(element.getAttribute('data-source-position'))
    };
    
    element.setAttribute('role', 'button');
    element.tabIndex = 0;
    element.setAttribute('aria-label', `Show source reference ${element.textContent.trim()}`);
    element.setAttribute('aria-haspopup', 'dialog');
    element.setAttribute('aria-expanded', 'false');
    const mouseEnterHandler = () => {
      if (this.tooltip.isVisible && this.tooltip.trigger === element) return;
      this.tooltip.show(element, sourceData);
    };
    const activate = () => {
      if (this.tooltip.isVisible && this.tooltip.trigger === element && this.tooltip.restoreTriggerFocus) {
        this.tooltip.hide();
      } else {
        this.tooltip.show(element, sourceData, {focus: true});
      }
    };
    const keydownHandler = event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        activate();
      }
    };
    
    const mouseLeaveHandler = () => {
      // Small delay to allow moving to tooltip
      setTimeout(() => {
        if (!this.tooltip.isVisible) {
          this.tooltip.hide();
        }
      }, 100);
    };
    
    // Store handlers on the element for cleanup
    element._tooltipMouseEnter = mouseEnterHandler;
    element._tooltipMouseLeave = mouseLeaveHandler;
    
    // Remove existing event listeners to prevent duplicates
    if (element._tooltipMouseEnter) {
      element.removeEventListener('mouseenter', element._tooltipMouseEnter);
    }
    if (element._tooltipMouseLeave) {
      element.removeEventListener('mouseleave', element._tooltipMouseLeave);
    }
    
    // Add new event listeners
    element.addEventListener('mouseenter', mouseEnterHandler);
    element.addEventListener('mouseleave', mouseLeaveHandler);
    element.addEventListener('click', activate);
    element.addEventListener('keydown', keydownHandler);
  }

  /**
   * Observe content changes to initialize tooltips for dynamically added content
   */
  observeContentChanges() {
    // Use MutationObserver to watch for new reference numbers
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            // Check if the added node is a reference number
            if (node.classList && node.classList.contains('reference-number')) {
              const hasSourceData = node.getAttribute('data-source-reference') || 
                                   node.getAttribute('data-source-label') || 
                                   node.getAttribute('data-source-url');
              if (hasSourceData) {
                this.attachTooltipToElement(node, 0); // Index doesn't matter for new elements
              }
            }
            
            // Check for reference numbers within the added node
            const referenceNumbers = node.querySelectorAll && 
              node.querySelectorAll('.reference-number');
            if (referenceNumbers && referenceNumbers.length > 0) {
              referenceNumbers.forEach((element, index) => {
                const hasSourceData = element.getAttribute('data-source-reference') || 
                                     element.getAttribute('data-source-label') || 
                                     element.getAttribute('data-source-url');
                if (hasSourceData) {
                  this.attachTooltipToElement(element, index);
                }
              });
            }
            
            // Check if this node has a shadow root (new shadow DOM created)
            if (node.shadowRoot) {
              const shadowReferenceNumbers = node.shadowRoot.querySelectorAll('.reference-number');
              if (shadowReferenceNumbers.length > 0) {
                shadowReferenceNumbers.forEach((element, index) => {
                  const hasSourceData = element.getAttribute('data-source-reference') || 
                                       element.getAttribute('data-source-label') || 
                                       element.getAttribute('data-source-url');
                  if (hasSourceData) {
                    this.attachTooltipToElement(element, index);
                  }
                });
              }
            }
          }
        });
      });
    });
    
    // Start observing
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }

  /**
   * Setup navigation event listeners
   */
  setupNavigationListeners() {
    // Listen for navigation to source events
    window.addEventListener('navigateToSource', (event) => {
      const detail = event.detail;
      if (!detail || typeof detail !== 'object' || Array.isArray(detail)) return;
      const url = safeSourceUrl(detail.url);
      if (!url) return;
      const position = safeSourcePosition(detail.position);
      
      // Check if we're already on the target page
      if (window.location.href === url) {
        // Same page - scroll to position
        this.scrollToPosition(position);
      } else {
        // Different page - navigate and scroll
        this.navigateToSourceWithPosition(url, position);
      }
    });
  }

  /**
   * Scroll to a specific position on the current page
   * @param {number} position - Scroll position
   */
  scrollToPosition(position) {
    position = safeSourcePosition(position);
    if (position && position > 0) {
      // Ensure position doesn't exceed document height
      const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
      const targetPosition = Math.min(position, maxScroll);
      
      window.scrollTo({
        top: targetPosition,
        behavior: 'smooth'
      });
    }
  }

  /**
   * Navigate to a different page with position
   * @param {string} url - Target URL
   * @param {number} position - Scroll position
   */
  navigateToSourceWithPosition(url, position) {
    url = safeSourceUrl(url);
    if (!url) return;
    position = safeSourcePosition(position);
    // Add position as query parameter
    if (position && position > 0) {
      const urlObj = new URL(url);
      urlObj.searchParams.set('scrollTo', position.toString());
      url = urlObj.toString();
    }
    
    // Store the position in sessionStorage for after page load (backup)
    if (position && position > 0) {
      try {sessionStorage.setItem('scrollToPosition', position.toString());} catch { /* Navigation also carries the position in its URL. */ }
    }
    
    // Navigate to the URL
    window.location.href = url;
  }

  /**
   * Reinitialize tooltips (useful for page reloads)
   */
  reinitialize() {
    this.initializeTooltips();
  }
}

// Create global instance
let persistentTooltipManager = null;

/**
 * Initialize the persistent tooltip system
 */
export function initializePersistentTooltips() {
  if (!persistentTooltipManager) {
    persistentTooltipManager = new PersistentTooltipManager();
  } else {
    persistentTooltipManager.reinitialize();
  }
}

/**
 * Get the persistent tooltip manager instance
 */
export function getPersistentTooltipManager() {
  return persistentTooltipManager;
}

/**
 * Test function to manually trigger a tooltip
 */
export function testTooltip() {
  // Find the first reference number
  const referenceNumber = document.querySelector('.reference-number');
  if (!referenceNumber) {
    console.error('No reference number found for testing');
    return;
  }
  
  // Get the persistent tooltip manager
  const manager = getPersistentTooltipManager();
  if (!manager) {
    console.error('No persistent tooltip manager found');
    return;
  }
  
  // Create test source data
  const testSourceData = {
    sourceReference: 'This is a test reference text',
    sourceLabel: 'Test Source',
    sourceUrl: 'http://example.com',
    sourcePosition: 100
  };
  
  // Try to show tooltip
  try {
    manager.tooltip.show(referenceNumber, testSourceData);
    console.log('Tooltip test completed');
  } catch (error) {
    console.error('Tooltip test failed:', error);
  }
}

/**
 * Check the current state of the page
 */
export function checkPageState() {
  // Check for quiz elements
  const quizForms = document.querySelectorAll('.socratiq-quiz');
  const aiMessages = document.querySelectorAll('.ai-message-chat');
  const referenceNumbers = document.querySelectorAll('.reference-number');
  
  // Check if there are any questions
  const questions = document.querySelectorAll('h4');
  
  // Check for any elements that might contain quiz content
  const potentialQuizContent = document.querySelectorAll('[class*="quiz"], [class*="question"], [class*="answer"]');
  
  return {
    quizForms: quizForms.length,
    aiMessages: aiMessages.length,
    referenceNumbers: referenceNumbers.length,
    questions: questions.length,
    potentialQuizContent: potentialQuizContent.length
  };
}

/**
 * Manually reinitialize tooltips for all existing reference numbers
 */
export function reinitializeTooltips() {
  const manager = getPersistentTooltipManager();
  if (!manager) {
    initializePersistentTooltips();
    return;
  }
  
  // Force reinitialize
  manager.reinitialize();
}

// Make test functions available globally
if (typeof window !== 'undefined') {
  window.testTooltip = testTooltip;
  window.checkPageState = checkPageState;
  window.reinitializeTooltips = reinitializeTooltips;
  
  // Note: Tooltip system is now initialized after quiz content is loaded
  // to ensure proper timing and avoid race conditions
}