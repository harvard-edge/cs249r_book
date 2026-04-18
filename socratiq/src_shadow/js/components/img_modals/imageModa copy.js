export function renderModal(shadowRoot, imageElement) {
    // Remove existing modal if present
    const existingModal = shadowRoot.querySelector('.image-modal-overlay');
    if (existingModal) existingModal.remove();
  
    // Create modal elements
    const overlay = document.createElement('div');
    overlay.className = 'image-modal-overlay';
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.8);
      display: flex;
      justify-content: center;
      align-items: center;
      z-index: 9999;
      opacity: 0;
      transition: opacity 0.3s;
    `;
  
    // Clone the image/svg
    const modalImage = imageElement.cloneNode(true);
    modalImage.style.cssText = `
      max-width: 90%;
      max-height: 90vh;
      object-fit: contain;
      transform: scale(0.9);
      transition: transform 0.3s;
    `;
  
    // Add close button
    const closeButton = document.createElement('button');
    closeButton.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
      </svg>
    `;
    closeButton.style.cssText = `
      position: absolute;
      top: 20px;
      right: 20px;
      background: white;
      border: none;
      border-radius: 50%;
      padding: 8px;
      cursor: pointer;
      color: black;
    `;
  
    // Add elements to overlay
    overlay.appendChild(modalImage);
    overlay.appendChild(closeButton);
    shadowRoot.appendChild(overlay);
  
    // Animate in
    requestAnimationFrame(() => {
      overlay.style.opacity = '1';
      modalImage.style.transform = 'scale(1)';
    });
  
    // Add event listeners
    const closeModal = () => {
      overlay.style.opacity = '0';
      modalImage.style.transform = 'scale(0.9)';
      setTimeout(() => overlay.remove(), 300);
    };
  
    closeButton.addEventListener('click', closeModal);
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) closeModal();
    });
    
    // Close on escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeModal();
    }, { once: true });
  }

// Add near other initialization code in the inject() function
export function initializeImageZoom(shadowRoot) {
  shadowRoot.addEventListener('click', (e) => {
    const target = e.target;
    
    // Skip if target is within an ink-mde editor
    if (target.closest('.ink-mde')) return;

    // First check if clicked element is the zoom icon
    const isZoomIcon = target.closest('.zoom-icon');
    if (isZoomIcon) {
      // Find the closest parent with mermaid-diagram class
      const diagramContainer = isZoomIcon.closest('.mermaid-diagram, .mermaid-figure');
      if (diagramContainer) {
        renderModal(shadowRoot, diagramContainer);
      }
      return;
    }

    // Then check for direct clicks on diagrams
    const diagram = target.closest('.mermaid-diagram, .mermaid-figure');
    if (diagram) {
      renderModal(shadowRoot, diagram);
    }
  });

  const addZoomIconToElement = (element) => {
    // Skip if element already has a zoom icon
    if (element.querySelector('.zoom-icon')) return;
    
    // Skip if element is inside a zoom icon or modal
    if (element.closest('.zoom-icon, .image-modal-overlay')) return;

    // Skip if element is nested inside another zoomable element
    // if (element.closest('.zoomable-image, .chart-svg, .diagram-svg') !== element) return;

    // Skip small SVGs (likely icons)
    if (element.tagName.toLowerCase() === 'svg') {
      const rect = element.getBoundingClientRect();
      if (rect.width < 50 || rect.height < 50) return;
    }

    const zoomIcon = document.createElement('button');
    zoomIcon.className = 'zoom-icon';
    zoomIcon.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
        <path d="M5 8a1 1 0 011-1h1V6a1 1 0 012 0v1h1a1 1 0 110 2H9v1a1 1 0 11-2 0V9H6a1 1 0 01-1-1z"/>
        <path fill-rule="evenodd" d="M2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8zm6-4a4 4 0 100 8 4 4 0 000-8z" clip-rule="evenodd"/>
      </svg>
    `;
    zoomIcon.style.cssText = `
      position: absolute;
      top: 8px;
      right: 8px;
      padding: 4px;
      background: rgba(255, 255, 255, 0.9);
      border: none;
      border-radius: 4px;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.2s;
      z-index: 10;
    `;

    // Ensure parent has relative positioning
    const wrapper = element.parentElement;
    if (wrapper && !wrapper.style.position) {
      wrapper.style.position = 'relative';
    }

    // Insert zoom icon after the element
    wrapper.insertBefore(zoomIcon, element.nextSibling);

    // Show/hide zoom icon on hover of the wrapper
    wrapper.addEventListener('mouseenter', () => zoomIcon.style.opacity = '1');
    wrapper.addEventListener('mouseleave', () => zoomIcon.style.opacity = '0');
  };

  const processZoomableElements = () => {
    // More specific selector for Mermaid diagrams
    const diagrams = shadowRoot.querySelectorAll('.mermaid-diagram, .mermaid-figure');
    diagrams.forEach(diagram => {
      if (!diagram.querySelector('.zoom-icon')) {
        addZoomIconToElement(diagram);
      }
    });
  };

  // Initial processing
  processZoomableElements();

  // Create observer to watch for new elements
  const observer = new MutationObserver((mutations) => {
    mutations.forEach(mutation => {
      // Check for new nodes that might be zoomable
      mutation.addedNodes.forEach(node => {
        if (node.nodeType === 1) { // Element node
          // Check if the added node itself is zoomable
          if (node.matches('.zoomable-image, img, svg')) {
            addZoomIconToElement(node);
          }
          // Check for zoomable elements within the added node
          const zoomableElements = node.querySelectorAll('.zoomable-image, img, svg');
          zoomableElements.forEach(element => {
            addZoomIconToElement(element);
          });
        }
      });
    });
  });

  observer.observe(shadowRoot, { 
    childList: true, 
    subtree: true 
  });

  // Return function to manually trigger processing
  return {
    refresh: processZoomableElements
  };
}

// // Add to your inject() function
// async function inject() {
//   try {
//     // ... existing code ...
    
//     // Add this line after other initializations
//     initializeImageZoom(shadowRoot);
    
//     // ... rest of existing code ...
//   } catch (error) {
//     console.error("Failed to initialize application:", error);
//   }
// }