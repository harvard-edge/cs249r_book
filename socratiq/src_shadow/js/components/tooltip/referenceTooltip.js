/**
 * Reference Tooltip Component
 * 
 * Provides a modular tooltip system for quiz question references
 * with "Go to Source" functionality and position-based navigation.
 */

export class ReferenceTooltip {
  constructor() {
    this.tooltip = null;
    this.isVisible = false;
    this.currentSourceData = null;
  }

  /**
   * Show tooltip with source reference information
   * @param {HTMLElement} element - The element that triggered the tooltip
   * @param {Object} sourceData - Source data containing reference text, URL, and position
   */
  show(element, sourceData) {
    // Ensure element is valid
    if (!element || !element.getBoundingClientRect) {
      console.error('Invalid element provided to tooltip.show');
      return;
    }
    
    this.hide(); // Remove any existing tooltip
    
    this.currentSourceData = sourceData;
    this.tooltip = this.createTooltip(sourceData);
    
    // Position tooltip before adding to DOM
    this.positionTooltip(element);
    this.addEventListeners();
    
    document.body.appendChild(this.tooltip);
    this.isVisible = true;
    
    // Immediate visibility - no animation delays
    this.tooltip.classList.add('tooltip-visible');
    this.tooltip.style.opacity = '1';
    this.tooltip.style.transform = 'translateY(0)';
  }

  /**
   * Hide the tooltip
   */
  hide() {
    if (this.tooltip) {
      this.tooltip.classList.remove('tooltip-visible');
      this.tooltip.style.opacity = '0';
      this.tooltip.style.transform = 'translateY(-10px)';
      
      setTimeout(() => {
        if (this.tooltip && this.tooltip.parentNode) {
          this.tooltip.parentNode.removeChild(this.tooltip);
        }
        this.tooltip = null;
        this.isVisible = false;
        this.currentSourceData = null;
      }, 100); // Faster cleanup
    }
  }

  /**
   * Create the tooltip DOM element
   * @param {Object} sourceData - Source data object
   * @returns {HTMLElement} Tooltip element
   */
  createTooltip(sourceData) {
    const tooltip = document.createElement('div');
    tooltip.id = 'reference-tooltip';
    tooltip.className = 'reference-tooltip';
    
    const { sourceReference, sourceLabel, sourceUrl, sourcePosition } = sourceData;
    
    tooltip.innerHTML = `
      <div class="tooltip-content">
        <div class="tooltip-header">
          <span class="tooltip-title">Source Reference</span>
          <button class="tooltip-close" aria-label="Close tooltip">×</button>
        </div>
        <div class="tooltip-section">
          <div class="tooltip-label">From: ${sourceLabel}</div>
          <div class="tooltip-text">${this.truncateText(sourceReference, 200)}</div>
        </div>
        <div class="tooltip-actions">
          <button class="tooltip-action-btn go-to-source" data-url="${sourceUrl}" data-position="${sourcePosition}">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
              <polyline points="15,3 21,3 21,9"></polyline>
              <line x1="10" y1="14" x2="21" y2="3"></line>
            </svg>
            Go to Source
          </button>
        </div>
      </div>
    `;
    
    return tooltip;
  }

  /**
   * Position the tooltip relative to the trigger element
   * @param {HTMLElement} element - The trigger element
   */
  positionTooltip(element) {
    const rect = element.getBoundingClientRect();
    const tooltipRect = this.tooltip.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    
    // Calculate available space on both sides
    const spaceLeft = rect.left;
    const spaceRight = viewportWidth - rect.right;
    
    let left, top = rect.top - tooltipRect.height - 5; // Closer positioning
    
    // SIMPLE: Always position tooltip to the left of the reference pill
    left = rect.left - tooltipRect.width - 5; // 5px gap from the pill
    
    // If tooltip would go off the left edge, just keep it at the edge
    if (left < 0) {
      left = 0; // Keep at left edge
    }
    
    // Adjust vertical position if tooltip would go off-screen
    if (top < 10) {
      // If tooltip would go above viewport, position it below the element
      top = rect.bottom + 5;
      
      // If it would go off bottom, position it above
      if (top + tooltipRect.height > viewportHeight - 10) {
        top = rect.top - tooltipRect.height - 5;
      }
    }
    
    // Final check to ensure tooltip doesn't go off-screen
    if (top + tooltipRect.height > viewportHeight - 10) {
      top = viewportHeight - tooltipRect.height - 10;
    }
    
    this.tooltip.style.position = 'fixed';
    this.tooltip.style.left = `${left}px`;
    this.tooltip.style.top = `${top}px`;
    this.tooltip.style.zIndex = '9999';
  }

  /**
   * Add event listeners to the tooltip
   */
  addEventListeners() {
    // Close button
    const closeBtn = this.tooltip.querySelector('.tooltip-close');
    closeBtn.addEventListener('click', () => this.hide());
    
    // Go to source button
    const goToSourceBtn = this.tooltip.querySelector('.go-to-source');
    goToSourceBtn.addEventListener('click', (e) => {
      const url = e.target.closest('.go-to-source').dataset.url;
      const position = e.target.closest('.go-to-source').dataset.position;
      this.goToSource(url, position);
    });
    
    // Close on escape key
    const handleKeydown = (e) => {
      if (e.key === 'Escape') {
        this.hide();
        document.removeEventListener('keydown', handleKeydown);
      }
    };
    document.addEventListener('keydown', handleKeydown);
    
    // Close on click outside
    const handleClickOutside = (e) => {
      if (this.tooltip && !this.tooltip.contains(e.target) && !e.target.closest('.reference-number')) {
        this.hide();
        document.removeEventListener('click', handleClickOutside);
      }
    };
    setTimeout(() => {
      document.addEventListener('click', handleClickOutside);
    }, 100);

    // Close when mouse leaves the tooltip
    this.tooltip.addEventListener('mouseleave', () => {
      this.hide();
    });
  }

  /**
   * Navigate to source URL with position
   * @param {string} url - Source URL
   * @param {number} position - Scroll position
   */
  goToSource(url, position) {
    this.hide();
    
    // Dispatch custom event for navigation
    const event = new CustomEvent('navigateToSource', {
      detail: {
        url: url,
        position: parseFloat(position) || 0,
        sourceData: this.currentSourceData
      },
      bubbles: true,
      composed: true
    });
    
    document.dispatchEvent(event);
  }

  /**
   * Truncate text to specified length
   * @param {string} text - Text to truncate
   * @param {number} maxLength - Maximum length
   * @returns {string} Truncated text
   */
  truncateText(text, maxLength = 200) {
    if (!text) return 'No source reference available';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength).trim() + '...';
  }

  /**
   * Add CSS styles to the document
   */
  static addStyles() {
    if (document.getElementById('reference-tooltip-styles')) {
      return; // Styles already added
    }
    
    const style = document.createElement('style');
    style.id = 'reference-tooltip-styles';
    style.textContent = `
      .reference-tooltip {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        max-width: 300px;
        min-width: 250px;
        width: max-content;
        opacity: 0;
        transform: translateY(-10px);
        transition: opacity 0.2s ease, transform 0.2s ease;
        pointer-events: auto;
        display: block !important;
        visibility: visible !important;
        word-wrap: break-word;
        overflow-wrap: break-word;
      }
      
      .reference-tooltip.tooltip-visible {
        opacity: 1;
        transform: translateY(0);
      }
      
      .tooltip-content {
        padding: 16px;
      }
      
      .tooltip-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #f3f4f6;
      }
      
      .tooltip-title {
        font-weight: 600;
        color: #374151;
        font-size: 14px;
      }
      
      .tooltip-close {
        background: none;
        border: none;
        color: #9ca3af;
        cursor: pointer;
        font-size: 18px;
        line-height: 1;
        padding: 0;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 4px;
        transition: background-color 0.2s;
      }
      
      .tooltip-close:hover {
        background-color: #f3f4f6;
        color: #374151;
      }
      
      .tooltip-section {
        margin-bottom: 12px;
      }
      
      .tooltip-label {
        font-size: 12px;
        color: #6b7280;
        font-weight: 500;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }
      
      .tooltip-text {
        color: #374151;
        font-size: 13px;
        line-height: 1.5;
        margin: 0;
      }
      
      .tooltip-actions {
        display: flex;
        justify-content: flex-end;
        margin-top: 12px;
        padding-top: 8px;
        border-top: 1px solid #f3f4f6;
      }
      
      .tooltip-action-btn {
        background: #3b82f6;
        border: none;
        border-radius: 6px;
        color: white;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 500;
        padding: 6px 12px;
        transition: background-color 0.2s;
      }
      
      .tooltip-action-btn:hover {
        background: #2563eb;
      }
      
      .tooltip-action-btn svg {
        width: 14px;
        height: 14px;
      }
      
      .reference-number {
        background: #f3f4f6;
        border: none;
        border-radius: 12px;
        color: #374151;
        cursor: help;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        height: 20px;
        min-width: 20px;
        padding: 0 6px;
        margin-left: 6px;
        opacity: 0.8;
        transition: all 0.2s ease;
        vertical-align: middle;
      }
      
      .reference-number:hover {
        background: #6b7280;
        color: white;
        opacity: 1;
        transform: scale(1.05);
      }
    `;
    
    document.head.appendChild(style);
  }
}

// Initialize styles when module is imported
ReferenceTooltip.addStyles();

// Also initialize styles on page load to ensure they're available on new pages
if (typeof window !== 'undefined') {
  window.addEventListener('load', () => {
    ReferenceTooltip.addStyles();
  });
}