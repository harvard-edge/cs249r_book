import {safeSourceUrl, safeSourcePosition} from './source-url.js';

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
    this.trigger = null;
    this.restoreTriggerFocus = false;
    this.cleanupListeners = null;
  }

  /**
   * Show tooltip with source reference information
   * @param {HTMLElement} element - The element that triggered the tooltip
   * @param {Object} sourceData - Source data containing reference text, URL, and position
   */
  show(element, sourceData, {focus = false} = {}) {
    // Ensure element is valid
    if (!element || !element.getBoundingClientRect) {
      console.error('Invalid element provided to tooltip.show');
      return;
    }
    
    this.hide({restoreFocus: false}); // Remove only the previous tooltip instance
    
    this.trigger = element;
    this.restoreTriggerFocus = focus;
    this.currentSourceData = sourceData;
    this.tooltip = this.createTooltip(sourceData);
    
    // Measure the mounted popover so its actions remain inside the viewport.
    document.body.appendChild(this.tooltip);
    this.positionTooltip(element);
    this.addEventListeners();
    this.isVisible = true;
    
    // Immediate visibility - no animation delays
    this.tooltip.classList.add('tooltip-visible');
    this.tooltip.style.opacity = '1';
    this.tooltip.style.transform = 'translateY(0)';
    element.setAttribute('aria-expanded', 'true');
    if (focus) this.tooltip.querySelector('.tooltip-close').focus({preventScroll: true});
  }

  /**
   * Hide the tooltip
   */
  hide({restoreFocus = true} = {}) {
    const tooltip = this.tooltip;
    if (!tooltip) return;
    const trigger = this.trigger;
    const shouldRestore = restoreFocus && (this.restoreTriggerFocus || tooltip.contains(document.activeElement));
    this.cleanupListeners?.();
    this.cleanupListeners = null;
    this.tooltip = null;
    this.isVisible = false;
    this.currentSourceData = null;
    this.trigger = null;
    this.restoreTriggerFocus = false;
    trigger?.setAttribute('aria-expanded', 'false');
    tooltip.classList.remove('tooltip-visible');
    tooltip.removeAttribute('id');
    tooltip.setAttribute('aria-hidden', 'true');
    tooltip.inert = true;
    tooltip.style.pointerEvents = 'none';
    tooltip.style.opacity = '0';
    tooltip.style.transform = 'translateY(-10px)';
    // A delayed cleanup must never remove a newer tooltip.
    setTimeout(() => tooltip.remove(), 100);
    if (shouldRestore && trigger?.isConnected) trigger.focus({preventScroll: true});
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
    tooltip.setAttribute('role', 'dialog');
    tooltip.setAttribute('aria-label', 'Source reference');

    const { sourceReference, sourceLabel, sourceUrl, sourcePosition } = sourceData;

    const content = document.createElement('div');
    content.className = 'tooltip-content';

    const header = document.createElement('div');
    header.className = 'tooltip-header';

    const titleSpan = document.createElement('span');
    titleSpan.className = 'tooltip-title';
    titleSpan.textContent = 'Source Reference';

    const closeBtn = document.createElement('button');
    closeBtn.className = 'tooltip-close';
    closeBtn.type = 'button';
    closeBtn.setAttribute('aria-label', 'Close tooltip');
    closeBtn.textContent = '×';

    header.append(titleSpan, closeBtn);

    const section = document.createElement('div');
    section.className = 'tooltip-section';

    const labelEl = document.createElement('div');
    labelEl.className = 'tooltip-label';
    labelEl.textContent = `From: ${sourceLabel ?? ''}`;

    const textEl = document.createElement('div');
    textEl.className = 'tooltip-text';
    textEl.textContent = this.truncateText(sourceReference, 200);

    section.append(labelEl, textEl);

    const actions = document.createElement('div');
    actions.className = 'tooltip-actions';

    const goBtn = document.createElement('button');
    goBtn.className = 'tooltip-action-btn go-to-source';
    goBtn.type = 'button';
    const destination = safeSourceUrl(sourceUrl);
    goBtn.disabled = !destination;
    if (destination) goBtn.dataset.url = destination;
    goBtn.dataset.position = String(safeSourcePosition(sourcePosition));
    goBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
              <polyline points="15,3 21,3 21,9"></polyline>
              <line x1="10" y1="14" x2="21" y2="3"></line>
            </svg>
            Go to Source`;

    actions.appendChild(goBtn);

    content.append(header, section, actions);
    tooltip.appendChild(content);

    return tooltip;
  }

  /**
   * Position the tooltip relative to the trigger element
   * @param {HTMLElement} element - The trigger element
   */
  positionTooltip(element) {
    const rect = element.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    Object.assign(this.tooltip.style, {
      position: 'fixed', boxSizing: 'border-box',
      maxWidth: `${Math.max(0, Math.min(300, viewportWidth - 20))}px`,
      minWidth: `${Math.max(0, Math.min(250, viewportWidth - 20))}px`,
      maxHeight: `${Math.max(0, viewportHeight - 20)}px`, overflowY: 'auto',
    });
    const tooltipRect = this.tooltip.getBoundingClientRect();
    const left = Math.min(Math.max(10, rect.left - tooltipRect.width - 5), Math.max(10, viewportWidth - tooltipRect.width - 10));
    let top = rect.top - tooltipRect.height - 5;
    if (top < 10) top = rect.bottom + 5;
    top = Math.max(10, Math.min(top, viewportHeight - tooltipRect.height - 10));
    this.tooltip.style.left = `${left}px`;
    this.tooltip.style.top = `${top}px`;
    this.tooltip.style.zIndex = '9999';
  }

  /**
   * Add event listeners to the tooltip
   */
  addEventListeners() {
    const tooltip = this.tooltip;
    const trigger = this.trigger;
    tooltip.querySelector('.tooltip-close').addEventListener('click', () => this.hide());
    tooltip.querySelector('.go-to-source').addEventListener('click', event => {
      const action = event.target.closest('.go-to-source');
      this.goToSource(action.dataset.url, action.dataset.position);
    });
    const handleKeydown = event => {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopImmediatePropagation();
        this.hide();
      }
    };
    const handleClickOutside = event => {
      const path = event.composedPath();
      if (!path.includes(tooltip) && !path.includes(trigger)) this.hide({restoreFocus: false});
    };
    document.addEventListener('keydown', handleKeydown, true);
    const outsideTimer = setTimeout(() => document.addEventListener('click', handleClickOutside), 100);
    this.cleanupListeners = () => {
      clearTimeout(outsideTimer);
      document.removeEventListener('keydown', handleKeydown, true);
      document.removeEventListener('click', handleClickOutside);
    };
    tooltip.addEventListener('mouseleave', () => {
      if (!this.restoreTriggerFocus && !tooltip.contains(document.activeElement)) this.hide();
    });
  }

  /**
   * Navigate to source URL with position
   * @param {string} url - Source URL
   * @param {number} position - Scroll position
   */
  goToSource(url, position) {
    url = safeSourceUrl(url);
    if (!url) return;
    position = safeSourcePosition(position);
    const sourceData = this.currentSourceData;
    this.hide();
    
    // Dispatch custom event for navigation
    const event = new CustomEvent('navigateToSource', {
      detail: {
        url: url,
        position: position,
        sourceData: sourceData
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
      
      .tooltip-action-btn:not(:disabled):hover {
        background: #2563eb;
      }
      .tooltip-action-btn:disabled {
        background: #e5e7eb;
        color: #4b5563;
        cursor: not-allowed;
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