import { menu_slide_on } from '../menu/open_close_menu.js';
import { SpacedRepetitionModal } from '../../components/spaced_repetition/spaced-repetition-modal-handler.js';
import { showPopover } from '../../libs/utils/utils.js';

// Declare selectedText in shared scope
let selectedText = '';

function injectFloatingButton(shadowRoot) {
    const styles = document.createElement('style');
    styles.textContent = `
    #floating-ai-btn {
     padding: 2px !important;
    }
      .menu-item {
        display: flex !important;
        align-items: center !important;
        padding: 8px 12px !important;
        gap: 8px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        position: relative !important;
      }
      
      .arrow-icon {
        margin-left: auto !important;
      }

       
      #copy-container {
        display: grid !important;
        grid-template-columns: repeat(3, 1fr) !important;
        gap: 8px !important;
        padding: 8px !important;
        margin-bottom: 12px !important;
        width: 100% !important;
      }
  
      .copy-item {
        background: white !important;
        border-left: 4px solid #22c55e !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        min-height: 40px !important;
        display: flex !important;
        align-items: center !important;
        position: relative !important;
      }
  
      .copy-item p {
        margin: 0 !important;
        font-size: 13px !important;
        line-height: 1.4 !important;
        color: #334155 !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        width: 100% !important;
      }
  
      .copy-item:hover {
        box-shadow: 0 2px 4px rgba(0,0,0,0.15) !important;
        background: #f8fafc !important;
      }
  
      .copy-item.expanded {
        grid-column: span 3 !important;
      }
  
      .copy-item.expanded p {
        white-space: normal !important;
        overflow: visible !important;
      }
    `;
  
    const button = document.createElement('div');
    button.id = 'floating-ai-btn';
    button.style.cssText = `
      display: none;
      position: fixed;
      padding: 2px !important;
      background: rgba(75, 85, 99, 1) !important;
    `;
    button.innerHTML = `
      <div class="menu-item">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" style="width: 16px !important; height: 16px !important; stroke: white !important;">
          <path stroke-linecap="round" stroke-linejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 0 1 .865-.501 48.172 48.172 0 0 0 3.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018Z" />
        </svg>
        <span style="font-size: 12px !important;">Send to Chat</span>
        <svg class="arrow-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" style="width: 14px !important; height: 14px !important; stroke: white !important; opacity: 0;">
          <path stroke-linecap="round" stroke-linejoin="round" d="m4.5 19.5 15-15m0 0H8.25m11.25 0v11.25" />
        </svg>
      </div>
      <div class="menu-divider"></div>
      <div class="menu-item">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" style="width: 16px !important; height: 16px !important; stroke: white !important;">
          <path stroke-linecap="round" stroke-linejoin="round" d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0 1 11.186 0Z" />
        </svg>
        <span style="font-size: 12px !important;">Create Flashcard</span>
        <svg class="arrow-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" style="width: 14px !important; height: 14px !important; stroke: white !important; opacity: 0;">
          <path stroke-linecap="round" stroke-linejoin="round" d="m4.5 19.5 15-15m0 0H8.25m11.25 0v11.25" />
        </svg>
      </div>
    `;

    // Add click handlers for menu items
    const menuItems = button.querySelectorAll('.menu-item');
    menuItems.forEach((item, index) => {
      item.addEventListener('mouseenter', () => {
        item.style.backgroundColor = 'rgb(55 65 79)';
        const arrow = item.querySelector('.arrow-icon');
        if (arrow) {
          arrow.style.opacity = '1';
        }
      });
      
      item.addEventListener('mouseleave', () => {
        item.style.backgroundColor = 'transparent';
        const arrow = item.querySelector('.arrow-icon');
        if (arrow) {
          arrow.style.opacity = '0';
        }
      });

      // Add click handler
      item.addEventListener('click', async (e) => {
        if (!selectedText) return; // Guard against empty selection
        
        if (index === 0) {
          // First menu item - Send to Chat
          button.style.display = 'none';
          addToCopyContainer(selectedText, shadowRoot);
          menu_slide_on(shadowRoot, true);
    
          const userInput = shadowRoot.getElementById('user-input');
          if (userInput) {
            userInput.focus();
          }
        } else if (index === 1) {
          // Second menu item - Create Flashcard
          button.style.display = 'none';
          try {
            // Get the modal instance
            const modal = SpacedRepetitionModal.instance;
            if (!modal) {
              throw new Error('Spaced repetition modal not initialized');
            }

            // Use the storage handler to create flashcards
            const result = await modal.storageHandler.addFlashcardsFromText(selectedText);

            if (result.success) {
              showPopover(shadowRoot, `Created ${result.flashcards.length} flashcards!`, "success", 3000);
            } else {
              showPopover(shadowRoot, "Failed to create flashcards", "error", 3000);
            }
          } catch (error) {
            console.error('Failed to create flashcards:', error);
            showPopover(shadowRoot, "Error creating flashcards", "error", 3000);
          }
        }
      });

      // Add touch events for mobile
      item.addEventListener('touchstart', () => {
        item.style.backgroundColor = 'rgb(55 65 79)';
        const arrow = item.querySelector('.arrow-icon');
        if (arrow) {
          arrow.style.opacity = '1';
        }
      });
      
      item.addEventListener('touchend', () => {
        item.style.backgroundColor = 'transparent';
        const arrow = item.querySelector('.arrow-icon');
        if (arrow) {
          arrow.style.opacity = '0';
        }
      });
    });

    const floatingContainer = document.createElement('div');
    floatingContainer.id = 'floating-ai-container';
    
    floatingContainer.appendChild(button);
    document.body.appendChild(floatingContainer);
    shadowRoot.appendChild(styles);
  
    return { button };
}

export function addToCopyContainer(text, shadowRoot) {
  const copyContainer = shadowRoot.getElementById('copy-container');
  if (!copyContainer) return;

  copyContainer.classList.remove('hidden');

  const copyItem = document.createElement('div');
  copyItem.className = 'copy-item pulse-effect';
  
  // Create wrapper for text and delete button
  const contentWrapper = document.createElement('div');
  contentWrapper.style.cssText = `
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
  `;
  
  // Add the text content
  const p = document.createElement('p');
  p.textContent = text;
  
  // Add delete button
  const deleteBtn = document.createElement('button');
  deleteBtn.innerHTML = '×';
  deleteBtn.style.cssText = `
    margin-left: 4px;
    padding: 0 4px;
    color: #1e40af;
    font-size: 16px;
    cursor: pointer;
    border: none;
    background: none;
  `;
  
  deleteBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    copyItem.remove();
    if (copyContainer.children.length === 0) {
      copyContainer.classList.add('hidden');
    }
  });
  
  contentWrapper.appendChild(p);
  contentWrapper.appendChild(deleteBtn);
  copyItem.appendChild(contentWrapper);
  
  copyItem.style.cssText = `
    max-width: 100px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: flex;
    align-items: center;
    transition: max-width 0.2s ease;
    background-color: #dbeafe !important;
    color: #1e40af !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 0.25rem !important;
    padding: 2px 6px !important;
    font-size: 14px !important;
    line-height: 1 !important;
  `;

  // Toggle expansion on click
  copyItem.addEventListener('click', () => {
    copyContainer.querySelectorAll('.copy-item').forEach(item => {
      if (item !== copyItem) item.classList.remove('expanded');
    });
    copyItem.classList.toggle('expanded');
    
    if (copyItem.classList.contains('expanded')) {
      copyItem.style.maxWidth = '100%';
      copyItem.style.whiteSpace = 'normal';
    } else {
      copyItem.style.maxWidth = '100px';
      copyItem.style.whiteSpace = 'nowrap';
    }
  });

  copyContainer.insertBefore(copyItem, copyContainer.firstChild);
}
  
  
  
export function clearCopyContainer(shadowRoot) {
  const copyContainer = shadowRoot.getElementById('copy-container');
  // const label = shadowRoot.getElementById('ai-prompt-label');

  if (copyContainer) {
    // Remove all child elements (copy-items)
    while (copyContainer.firstChild) {
      copyContainer.removeChild(copyContainer.firstChild);
    }

    // Hide the copy-container
    copyContainer.classList.add('hidden');

    // Hide the label if it exists
    // if (label) {
    //   label.style.display = 'none';
    // }
  }
}

  function updateButtonPosition(button, rect) {
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;

    
    const offsetX = 20;
    const offsetY = 10;
    
    let xR = rect.right + offsetX;
    let yR = rect.top + offsetY;



    let x = rect.left + offsetX;
    let y = rect.bottom + offsetY;

 
    if (x + 90 > viewportWidth) {
      x = rect.left - 90 - offsetX;
    }
  
    if (y < 0) {
      y = rect.bottom + offsetY;
    }
    if (y > viewportHeight) {
      y = viewportHeight - 30;
    }
  
    requestAnimationFrame(() => {
      button.style.cssText = `
        position: fixed !important;
        left: ${x}px !important;
        top: ${y}px !important;
        margin: auto !important;
        display: flex !important;
        background: rgba(75, 85, 99, 1) !important;
        color: white !important;
        flex-direction: column !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        border-radius: 4px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        min-width: 140px !important;
      `;
    });
  }
  

  export function getAllCopyContainerText(shadowRoot) {
    const copyContainer = shadowRoot.getElementById('copy-container');
    if (!copyContainer) return '';
  
    // Retrieve text from each copy-item, except the label
    const texts = Array.from(copyContainer.querySelectorAll('.copy-item p'))
      .map(item => item.textContent.trim())
      .join('\n'); // Join each text with a newline separator
  
    return texts;
  }
  
  export function isCopyContainerHidden(shadowRoot) {
    const copyContainer = shadowRoot.getElementById('copy-container');
    return copyContainer ? copyContainer.classList.contains('hidden') : true;
  }
  

  function initializeTextSelector(shadowRoot) {
    if (!shadowRoot) {
      console.error('Shadow root is not available');
      return;
    }
  
    const { button } = injectFloatingButton(shadowRoot);
  
    document.addEventListener('mouseup', (e) => {
      setTimeout(() => {
        const selection = window.getSelection();
        selectedText = selection.toString().trim(); // Update the shared selectedText variable
  
        if (selectedText) {
          const range = selection.getRangeAt(0);
          const rect = range.getBoundingClientRect();
          updateButtonPosition(button, rect);
          button.style.display = 'flex'; // Show the button
        } else if (!e.target.closest('.copy-item')) {
          button.style.display = 'none';
        }
      }, 10);
    });
  }
export function highlight_click(shadowRoot) {
  if (!shadowRoot) {
    console.error('Shadow root is required');
    return;
  }
  
  requestAnimationFrame(() => {
    initializeTextSelector(shadowRoot);
  });
}