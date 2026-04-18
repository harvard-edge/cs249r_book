import { menu_slide_on } from '../menu/open_close_menu.js';


function injectFloatingButton(shadowRoot) {
    const styles = document.createElement('style');
    styles.textContent = `
   
  
      #floating-ai-btn:hover {
        background: rgba(75, 85, 99, 1) !important;
        box-shadow: 0 3px 6px rgba(0,0,0,0.15) !important;
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
    button.style.cssText = 'display: none; position: fixed;';
    button.innerHTML = `
      <div class="icon-wrapper" style="padding: 0 4px !important;">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" style="width: 16px !important; height: 16px !important; stroke: white !important;">
          <path stroke-linecap="round" stroke-linejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" />
        </svg>
        <span style="font-size: 10px !important;">Send to AI</span>
      </div>
    `;
  
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

    console.log('Button position:', rect);
    
    const offsetX = 20;
    const offsetY = 10;
    
    let xR = rect.right + offsetX;
    let yR = rect.top + offsetY;

    // console.log("xR:", xR);
    // console.log("yR:", yR);

    

    let x = rect.left + offsetX;
    let y = rect.bottom + offsetY;

    console.log('x:', x);
    console.log('y:', y);
  
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
        padding-bottom: 3px !important;
        padding-left: 2px !important;
        padding-right: 2px !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        border-radius: 4px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
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
    let selectedText = '';
  
    document.addEventListener('mouseup', (e) => {
      setTimeout(() => {
        const selection = window.getSelection();
        const text = selection.toString().trim();
  
        if (text) {
          selectedText = text;
          const range = selection.getRangeAt(0);
          const rect = range.getBoundingClientRect();
          updateButtonPosition(button, rect);
        } else if (!e.target.closest('.copy-item')) {
          button.style.display = 'none';
        }
      }, 10);
    });
  
    button.addEventListener('click', () => {
      if (selectedText) {
        button.style.display = 'none';
        addToCopyContainer(selectedText, shadowRoot);
        menu_slide_on(shadowRoot, true);
  
        const userInput = shadowRoot.getElementById('user-input');
        if (userInput) {
          userInput.focus();
        }
      }
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