import React from 'react';
import { createRoot } from 'react-dom/client';
import { CardReviewModal } from './CardReviewModal.jsx';

/**
 * CardReviewModalBridge - A bridge to integrate React CardReviewModal with vanilla JS
 * This allows the existing SpacedRepetitionModal to use the React component
 */
export class CardReviewModalBridge {
  constructor(shadowRoot) {
    this.shadowRoot = shadowRoot;
    this.container = null;
    this.root = null;
    this.currentProps = {};
  }

  /**
   * Create the React component container
   */
  createContainer() {
    if (!this.container) {
      this.container = document.createElement('div');
      this.container.id = 'cardReviewModalContainer';
      this.shadowRoot.appendChild(this.container);
      
      this.root = createRoot(this.container);
    }
  }

  /**
   * Show the modal with the given props
   */
  show(props) {
    this.createContainer();
    this.currentProps = {
      isOpen: true,
      ...props
    };
    this.render();
  }

  /**
   * Hide the modal
   */
  hide() {
    if (this.currentProps) {
      this.currentProps.isOpen = false;
      this.render();
    }
  }

  /**
   * Update the modal props
   */
  update(props) {
    this.currentProps = {
      ...this.currentProps,
      ...props
    };
    this.render();
  }

  /**
   * Render the React component
   */
  render() {
    if (this.root && this.currentProps) {
      this.root.render(React.createElement(CardReviewModal, this.currentProps));
    }
  }

  /**
   * Clean up the component
   */
  destroy() {
    if (this.root) {
      this.root.unmount();
      this.root = null;
    }
    if (this.container && this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
      this.container = null;
    }
  }

  /**
   * Check if the modal is currently open
   */
  isOpen() {
    return this.currentProps?.isOpen || false;
  }

  /**
   * Get the current modal element (for compatibility with existing code)
   */
  getModalElement() {
    return this.container?.querySelector('[data-card-review-modal]') || this.container;
  }
}

export default CardReviewModalBridge;