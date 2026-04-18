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
      overflow: hidden; /* Prevent body scroll */
    `;
  
    // Create a scrollable container for the content
    const scrollContainer = document.createElement('div');
    scrollContainer.style.cssText = `
      max-width: 90%;
      max-height: 90vh;
      overflow: auto;
      position: relative;
      background: white;
      border-radius: 4px;
      padding: 16px;
    `;
  
    // Clone the image/svg
    const modalImage = imageElement.cloneNode(true);
    modalImage.style.cssText = `
      width: 100%;
      height: auto;
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
  
    // Append elements
    scrollContainer.appendChild(modalImage);
    overlay.appendChild(scrollContainer);
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
      const figure = isZoomIcon.closest('figure');
      if (figure) {
        renderModal(shadowRoot, figure);
      }
      return;
    }

    // Then check for direct clicks on figures
    const figure = target.closest('figure');
    if (figure) {
      renderModal(shadowRoot, figure);
    }
  });

  const addZoomIconToElement = (element) => {
    // Skip if element already has a zoom icon
    if (element.querySelector('.zoom-icon')) return;
    
    // Skip if element is inside a modal
    if (element.closest('.image-modal-overlay')) return;

    const zoomIcon = document.createElement('button');
    zoomIcon.className = 'zoom-icon';
    zoomIcon.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-6 h-6">
        <path fill-rule="evenodd" d="M10.5 3.75a6.75 6.75 0 100 13.5 6.75 6.75 0 000-13.5zM2.25 10.5a8.25 8.25 0 1114.59 5.28l4.69 4.69a.75.75 0 11-1.06 1.06l-4.69-4.69A8.25 8.25 0 012.25 10.5z" clip-rule="evenodd" />
        <path d="M12.75 9a.75.75 0 00-1.5 0v1.5h-1.5a.75.75 0 000 1.5h1.5v1.5a.75.75 0 001.5 0v-1.5h1.5a.75.75 0 000-1.5h-1.5V9z" />
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
      transition: opacity 0.2s ease-in-out;
      z-index: 10;
      width: 24px;
      height: 24px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    `;

    // Style the SVG icon
    const svgIcon = zoomIcon.querySelector('svg');
    svgIcon.style.cssText = `
      width: 16px;
      height: 16px;
      color: #4B5563;
    `;

    // Ensure figure has relative positioning
    element.style.position = 'relative';
    element.appendChild(zoomIcon);
  };

  const processZoomableElements = () => {
    // Specifically target figures within text-selection-menu
    const textSelectionMenu = shadowRoot.querySelector('#text-selection-menu');
    if (textSelectionMenu) {
      const figures = textSelectionMenu.querySelectorAll('figure');
      figures.forEach(figure => {
        if (!figure.querySelector('.zoom-icon')) {
          addZoomIconToElement(figure);
          
          // Add hover events to the figure
          figure.addEventListener('mouseenter', () => {
            const zoomIcon = figure.querySelector('.zoom-icon');
            if (zoomIcon) {
              zoomIcon.style.opacity = '1';
              zoomIcon.style.transform = 'scale(1)';
            }
          });

          figure.addEventListener('mouseleave', () => {
            const zoomIcon = figure.querySelector('.zoom-icon');
            if (zoomIcon && !zoomIcon.matches(':hover')) {
              zoomIcon.style.opacity = '0';
            }
          });
        }
      });
    }
  };

  // Initial processing
  processZoomableElements();

  // Create observer to watch for new elements
  const observer = new MutationObserver((mutations) => {
    mutations.forEach(mutation => {
      if (mutation.target.id === 'text-selection-menu' || mutation.target.closest('#text-selection-menu')) {
        processZoomableElements();
      }
    });
  });

  // Observe the text-selection-menu or its parent if it exists
  const textSelectionMenu = shadowRoot.querySelector('#text-selection-menu');
  if (textSelectionMenu) {
    observer.observe(textSelectionMenu, { 
      childList: true, 
      subtree: true 
    });
  }

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