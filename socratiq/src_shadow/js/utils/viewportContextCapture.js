/**
 * Viewport Context Capture Utility
 * 
 * Captures the current visible content, position, and URL for Section Quiz generation
 * Based on the reference tooltip pattern for consistent context capture
 */

export class ViewportContextCapture {
  /**
   * Capture the current viewport context for quiz generation
   * @returns {Object} Context object with content, position, and URL
   */
  static captureCurrentContext() {
    console.log('🔍 Capturing current viewport context for Section Quiz');
    
    const context = {
      // Page information
      url: window.location.href,
      title: document.title || 'Untitled Page',
      domain: window.location.hostname,
      timestamp: new Date().toISOString(),
      
      // Viewport information
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
        scrollX: window.scrollX,
        scrollY: window.scrollY
      },
      
      // Content information
      content: this.getVisibleContent(),
      
      // Position information
      position: this.getCurrentPosition()
    };
    
    console.log('🔍 Captured context:', context);
    return context;
  }

  /**
   * Get visible content from the current viewport
   * @returns {Object} Visible content with text and sources
   */
  static getVisibleContent() {
    const visibleElements = this.getVisibleElements();
    const contentSections = [];
    const sources = [];
    
    visibleElements.forEach((element, index) => {
      const textContent = this.extractTextFromElement(element);
      
      if (textContent.length > 50) { // Only include substantial content
        const elementId = element.id || `element-${index}`;
        const elementClass = element.className || '';
        const elementTag = element.tagName.toLowerCase();
        
        // Create source mapping
        const source = {
          sourceId: `viewport-source-${index}`,
          label: elementId || `${elementTag}-${index}`,
          content: textContent,
          pageUrl: window.location.href,
          domain: window.location.hostname,
          level: 'viewport',
          position: index,
          elementId: elementId,
          elementClass: elementClass,
          elementTag: elementTag,
          boundingRect: element.getBoundingClientRect()
        };
        
        sources.push(source);
        contentSections.push(`## ${source.label}\n\n${source.content}`);
      }
    });
    
    // Combine all content sections
    const combinedText = contentSections.join('\n\n---\n\n');
    
    return {
      text: combinedText,
      sources: sources,
      totalLength: combinedText.length,
      sourceCount: sources.length
    };
  }

  /**
   * Get elements that are currently visible in the viewport
   * @returns {Array} Array of visible DOM elements
   */
  static getVisibleElements() {
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;
    const scrollY = window.scrollY;
    const scrollX = window.scrollX;
    
    // Select elements that are likely to contain meaningful content
    const contentSelectors = [
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6', // Headings
      'p', 'div', 'section', 'article', 'main', // Content blocks
      'li', 'td', 'th', // List items and table cells
      'blockquote', 'pre', 'code' // Special content
    ];
    
    const visibleElements = [];
    
    contentSelectors.forEach(selector => {
      const elements = document.querySelectorAll(selector);
      elements.forEach(element => {
        if (this.isElementVisible(element, viewportHeight, viewportWidth, scrollY, scrollX)) {
          visibleElements.push(element);
        }
      });
    });
    
    // Sort by position in document
    visibleElements.sort((a, b) => {
      const aRect = a.getBoundingClientRect();
      const bRect = b.getBoundingClientRect();
      return (aRect.top + scrollY) - (bRect.top + scrollY);
    });
    
    console.log(`🔍 Found ${visibleElements.length} visible elements`);
    return visibleElements;
  }

  /**
   * Check if an element is visible in the current viewport
   * @param {HTMLElement} element - Element to check
   * @param {number} viewportHeight - Viewport height
   * @param {number} viewportWidth - Viewport width
   * @param {number} scrollY - Current scroll Y position
   * @param {number} scrollX - Current scroll X position
   * @returns {boolean} True if element is visible
   */
  static isElementVisible(element, viewportHeight, viewportWidth, scrollY, scrollX) {
    const rect = element.getBoundingClientRect();
    
    // Check if element is within viewport bounds
    const isInViewport = (
      rect.top < viewportHeight &&
      rect.bottom > 0 &&
      rect.left < viewportWidth &&
      rect.right > 0
    );
    
    // Check if element has meaningful content
    const hasContent = element.textContent && element.textContent.trim().length > 10;
    
    // Check if element is not hidden
    const isNotHidden = (
      element.offsetWidth > 0 &&
      element.offsetHeight > 0 &&
      window.getComputedStyle(element).display !== 'none' &&
      window.getComputedStyle(element).visibility !== 'hidden'
    );
    
    return isInViewport && hasContent && isNotHidden;
  }

  /**
   * Extract text content from an element
   * @param {HTMLElement} element - Element to extract text from
   * @returns {string} Extracted text content
   */
  static extractTextFromElement(element) {
    // Clone the element to avoid modifying the original
    const clone = element.cloneNode(true);
    
    // Remove script and style elements
    const scripts = clone.querySelectorAll('script, style');
    scripts.forEach(script => script.remove());
    
    // Get text content and clean it up
    let text = clone.textContent || '';
    
    // Clean up whitespace
    text = text.replace(/\s+/g, ' ').trim();
    
    // Remove common UI elements that aren't content
    text = text.replace(/\b(Home|About|Contact|Login|Sign up|Menu|Navigation|Footer|Header)\b/gi, '');
    
    return text;
  }

  /**
   * Get current scroll position and viewport information
   * @returns {Object} Position information
   */
  static getCurrentPosition() {
    return {
      scrollX: window.scrollX,
      scrollY: window.scrollY,
      scrollMaxX: document.documentElement.scrollWidth - window.innerWidth,
      scrollMaxY: document.documentElement.scrollHeight - window.innerHeight,
      scrollPercentage: {
        x: (window.scrollX / (document.documentElement.scrollWidth - window.innerWidth)) * 100,
        y: (window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100
      }
    };
  }

  /**
   * Create a context summary for quiz generation
   * @param {Object} context - Full context object
   * @returns {string} Context summary
   */
  static createContextSummary(context) {
    const { url, title, content, position } = context;
    
    return `Current Page Context:
- URL: ${url}
- Title: ${title}
- Content Length: ${content.totalLength} characters
- Visible Sources: ${content.sourceCount}
- Scroll Position: ${Math.round(position.scrollPercentage.y)}% down the page
- Viewport: ${context.viewport.width}x${context.viewport.height}

Visible Content:
${content.text}`;
  }

  /**
   * Validate that we have sufficient content for quiz generation
   * @param {Object} context - Context object
   * @returns {Object} Validation result
   */
  static validateContext(context) {
    const { content } = context;
    const minContentLength = 200; // Minimum characters needed for a meaningful quiz
    const minSources = 2; // Minimum sources needed
    
    const isValid = (
      content.totalLength >= minContentLength &&
      content.sourceCount >= minSources &&
      content.text.trim().length > 0
    );
    
    return {
      isValid,
      contentLength: content.totalLength,
      sourceCount: content.sourceCount,
      hasContent: content.text.trim().length > 0,
      minContentLength,
      minSources,
      message: isValid 
        ? 'Context is sufficient for quiz generation'
        : `Context insufficient: need ${minContentLength} chars and ${minSources} sources, got ${content.totalLength} chars and ${content.sourceCount} sources`
    };
  }
}