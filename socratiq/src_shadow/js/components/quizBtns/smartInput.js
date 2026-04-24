// import { generateUniqueId } from '../../libs/utils/utils.js';
import {addToCopyContainer} from '../highlight_menu/send_text_highlight.js';
let currentFilter = null;

export function setupAtMentions(shadowRoot) {
  const userInput = shadowRoot.getElementById('user-input');
  const copyContainer = shadowRoot.getElementById('copy-container');
  const dropupContainer = document.createElement('div');
  dropupContainer.className = 'at-mentions-dropup hidden';
  
  dropupContainer.style.cssText = `
    position: absolute;
    bottom: 100%;
    left: 0;
    width: 100%;
    max-height: 250px;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    box-shadow: 0 -4px 6px -1px rgba(0, 0, 0, 0.1);
    z-index: 50;
    display: flex;
    flex-direction: column;
  `;
  
  userInput.parentElement.appendChild(dropupContainer);

  let atIndex = -1;
  let searchTerm = '';
  let currentMentionId = null;
  let selectedIndex = -1;
  let visibleItems = [];
  let isFolderFocused = false;
  let selectedFolderIndex = -1;

  function hideDropup() {
    dropupContainer.style.display = 'none';
    dropupContainer.classList.add('hidden');
    currentFilter = null;
    selectedIndex = -1;
    selectedFolderIndex = -1;
    isFolderFocused = false;
    visibleItems = [];
  }

  // function addHiddenContent(mentionId, content) {
  //   const hiddenDiv = document.createElement('div');
  //   hiddenDiv.id = `mention-${mentionId}`;
  //   hiddenDiv.className = 'hidden-mention';
  //   hiddenDiv.textContent = content;
  //   copyContainer.appendChild(hiddenDiv);
  // }

  function removeHiddenContent(mentionId) {
    const hiddenDiv = copyContainer.querySelector(`#mention-${mentionId}`);
    if (hiddenDiv) {
      hiddenDiv.remove();
    }
  }

  function getIconForType(type) {
    switch (type) {
      case 'subsection':
        return '📑';
      case 'keyword':
        return '🔑';
      case 'section':
        return '📚';
      default:
        return '📎';
    }
  }

  function searchAllSources(search) {
    const results = [];
    
    // Search subsections
    if (window.subsections) {
      for (const [key, value] of Object.entries(window.subsections)) {
        if (key.toLowerCase().includes(search)) {
          results.push({ type: 'subsection', key, value });
        }
      }
    }

    // Search keywords
    if (window.keywords) {
      for (const [key, value] of Object.entries(window.keywords)) {
        if (key.toLowerCase().includes(search)) {
          results.push({ type: 'keyword', key, value });
        }
      }
    }

    // Search section data
    if (window.sectionData) {
      for (const [key, value] of Object.entries(window.sectionData)) {
        if (value.title.toLowerCase().includes(search)) {
          results.push({ type: 'section', key, value });
        }
      }
    }

    return results;
  }

  function getContentForMention(type, key) {
    let textToAdd = '';
    let prefix = '';
    
    switch (type) {
      case 'subsection':
        if (window.subsections && window.subsections[key]) {
          prefix = `[Subsection: ${key}]\n`;
          textToAdd = window.subsections[key].text;
        }
        break;
      case 'keyword':
        if (window.keywords && window.keywords[key]) {
          prefix = `[Keyword: ${key}]\n`;
          textToAdd = window.keywords[key].text;
        }
        break;
      case 'section':
        if (window.sectionData && window.sectionData[key]) {
          prefix = `[Section: ${window.sectionData[key].title}]\n`;
          textToAdd = window.sectionData[key].content;
        }
        break;
    }
    
    return prefix + textToAdd.trim();
  }

  function insertMention(type, key) {
    const currentContent = userInput.value;
    const beforeAt = currentContent.substring(0, atIndex);
    const afterCursor = currentContent.substring(userInput.selectionStart);
    
    // Add the mention text with padding spaces
    const mention = `@[${key}]   `;
    userInput.value = beforeAt + mention + afterCursor;
    
    // Set cursor position after mention and spaces
    const newCursorPosition = beforeAt.length + mention.length;
    userInput.setSelectionRange(newCursorPosition, newCursorPosition);
    
    // Add to copy container with type prefix
    const content = getContentForMention(type, key);
    const prefix = `[${type.charAt(0).toUpperCase() + type.slice(1)}: ${key}]\n`;
    addToCopyContainer(content, shadowRoot);
    
    hideDropup();
    userInput.focus();
  }

  function updateSelection() {
    // Remove previous selection from items
    dropupContainer.querySelectorAll('.at-mention-item').forEach((item, index) => {
      if (index === selectedIndex) {
        item.classList.add('bg-blue-100', 'shadow-sm'); // More pronounced highlight
      } else {
        item.classList.remove('bg-blue-100', 'shadow-sm');
      }
    });

    // Update folder selection
    dropupContainer.querySelectorAll('.folder-filter').forEach((folder, index) => {
      if (isFolderFocused && index === selectedFolderIndex) {
        folder.classList.add('bg-blue-200', 'shadow-sm'); // More pronounced highlight for folders
      } else {
        folder.classList.remove('bg-blue-200', 'shadow-sm');
      }
    });

    // Scroll selected item into view if needed
    if (!isFolderFocused && selectedIndex >= 0) {
      const selectedItem = dropupContainer.querySelector(`.at-mention-item:nth-child(${selectedIndex + 1})`);
      if (selectedItem) {
        const container = dropupContainer.querySelector('.results-container');
        const itemTop = selectedItem.offsetTop;
        const itemBottom = itemTop + selectedItem.offsetHeight;
        const containerTop = container.scrollTop;
        const containerBottom = containerTop + container.offsetHeight;

        if (itemTop < containerTop) {
          container.scrollTop = itemTop;
        } else if (itemBottom > containerBottom) {
          container.scrollTop = itemBottom - container.offsetHeight;
        }
      }
    }
  }

  function handleFolderNavigation(direction) {
    const filters = ['all', 'section', 'subsection', 'keyword'];
    const currentIndex = selectedFolderIndex;
    
    if (direction === 'right') {
      selectedFolderIndex = (currentIndex + 1) % filters.length;
    } else {
      selectedFolderIndex = (currentIndex - 1 + filters.length) % filters.length;
    }
    
    currentFilter = filters[selectedFolderIndex] === 'all' ? null : filters[selectedFolderIndex];
    showDropup(searchTerm);
  }

  function showDropup(search) {
    dropupContainer.style.display = 'flex';
    const results = searchAllSources(search);
    if (results.length === 0) {
      hideDropup();
      return;
    }

    const folderHtml = `
      <div class="folder-filters flex space-x-2 p-2 border-b border-gray-200">
        <div class="folder-filter p-1.5 rounded-md cursor-pointer flex items-center hover:bg-gray-100 ${!currentFilter ? 'bg-blue-100' : ''}" 
             data-filter="all">
          <span class="mr-1">📁</span>
          <span>All</span>
        </div>
        <div class="folder-filter p-1.5 rounded-md cursor-pointer flex items-center hover:bg-gray-100 ${currentFilter === 'section' ? 'bg-blue-100' : ''}" 
             data-filter="section">
          <span class="mr-1">📚</span>
          <span>Sections</span>
        </div>
        <div class="folder-filter p-1.5 rounded-md cursor-pointer flex items-center hover:bg-gray-100 ${currentFilter === 'subsection' ? 'bg-blue-100' : ''}" 
             data-filter="subsection">
          <span class="mr-1">📑</span>
          <span>Subsections</span>
        </div>
        <div class="folder-filter p-1.5 rounded-md cursor-pointer flex items-center hover:bg-gray-100 ${currentFilter === 'keyword' ? 'bg-blue-100' : ''}" 
             data-filter="keyword">
          <span class="mr-1">🔑</span>
          <span>Keywords</span>
        </div>
      </div>
    `;

    const filteredResults = currentFilter ? 
      results.filter(result => result.type === currentFilter) : 
      results;
    
    dropupContainer.innerHTML = folderHtml + `
      <div class="results-container max-h-[160px] overflow-y-auto">
        ${filteredResults.map(result => `
          <div class="at-mention-item p-2 hover:bg-gray-100 cursor-pointer flex items-center" 
               data-type="${result.type}" 
               data-key="${result.key}">
            <span class="mr-2">${getIconForType(result.type)}</span>
            <span>${result.key}</span>
          </div>
        `).join('')}
      </div>
    `;
    
    dropupContainer.classList.remove('hidden');

    // Add click handlers for folder filters
    dropupContainer.querySelectorAll('.folder-filter').forEach(filter => {
      filter.addEventListener('click', (e) => {
        e.stopPropagation();
        const filterType = filter.dataset.filter;
        currentFilter = filterType === 'all' ? null : filterType;
        showDropup(search);
      });
    });

    // Add click handlers for items
    dropupContainer.querySelectorAll('.at-mention-item').forEach(item => {
      item.addEventListener('click', () => {
        insertMention(item.dataset.type, item.dataset.key);
      });
    });

    // Reset selection state
    selectedIndex = -1;
    if (selectedFolderIndex === -1) selectedFolderIndex = 0;
    visibleItems = filteredResults;
    
    updateSelection();
  }

  // Event Listeners
  userInput.addEventListener('input', (e) => {
    const value = e.target.value;
    const cursorPosition = e.target.selectionStart;
    
    if (currentMentionId && !value.includes(`@[${currentMentionId}]`)) {
      removeHiddenContent(currentMentionId);
      currentMentionId = null;
    }
    
    atIndex = value.lastIndexOf('@', cursorPosition - 1);
    
    if (atIndex !== -1 && atIndex < cursorPosition) {
      searchTerm = value.slice(atIndex + 1, cursorPosition).toLowerCase();
      showDropup(searchTerm);
    } else {
      hideDropup();
    }
  });

  userInput.addEventListener('keydown', (e) => {
    if (!dropupContainer.classList.contains('hidden')) {
      switch (e.key) {
        case 'Enter':
          e.preventDefault(); // Prevent enter from triggering send
          if (isFolderFocused && selectedFolderIndex >= 0) {
            const filters = ['all', 'section', 'subsection', 'keyword'];
            currentFilter = filters[selectedFolderIndex] === 'all' ? null : filters[selectedFolderIndex];
            showDropup(searchTerm);
          } else if (selectedIndex >= 0 && selectedIndex < visibleItems.length) {
            const selectedItem = visibleItems[selectedIndex];
            insertMention(selectedItem.type, selectedItem.key);
          }
          return;
        case 'ArrowDown':
          e.preventDefault();
          if (isFolderFocused) {
            isFolderFocused = false;
            selectedIndex = 0;
          } else {
            selectedIndex = Math.min(selectedIndex + 1, visibleItems.length - 1);
            if (selectedIndex === -1 && visibleItems.length > 0) {
              selectedIndex = 0;
            }
          }
          updateSelection();
          return;
        case 'ArrowUp':
          e.preventDefault();
          if (selectedIndex === 0 || selectedIndex === -1) {
            isFolderFocused = true;
            selectedIndex = -1;
            if (selectedFolderIndex === -1) selectedFolderIndex = 0;
          } else {
            selectedIndex = Math.max(selectedIndex - 1, 0);
          }
          updateSelection();
          return;
        case 'ArrowRight':
          if (isFolderFocused) {
            e.preventDefault();
            handleFolderNavigation('right');
          }
          return;
        case 'ArrowLeft':
          if (isFolderFocused) {
            e.preventDefault();
            handleFolderNavigation('left');
          }
          return;
        case 'Escape':
          hideDropup();
          if (currentMentionId) {
            removeHiddenContent(currentMentionId);
            currentMentionId = null;
          }
          return;
        case 'Tab':
          e.preventDefault();
          isFolderFocused = true;
          selectedIndex = -1;
          if (e.shiftKey) {
            selectedFolderIndex = (selectedFolderIndex - 1 + 4) % 4;
          } else {
            selectedFolderIndex = (selectedFolderIndex + 1) % 4;
          }
          handleFolderNavigation(e.shiftKey ? 'left' : 'right');
          return;
      }
    }
  });

  shadowRoot.addEventListener('click', (e) => {
    if (!dropupContainer.contains(e.target) && e.target !== userInput) {
      hideDropup();
    }
  });
}