import {addToCopyContainer} from './send_text_highlight.js';

export function insertContextButton(shadowRoot) {
  const contextButton = shadowRoot.querySelector('#context-button');
  if (!contextButton) {
    console.error('Context button not found.');
    return;
  }

  const inputWrapper = shadowRoot.querySelector('#input-container');
  if (!inputWrapper) {
    console.error('Input wrapper not found.');
    return;
  }

  inputWrapper.style.position = 'relative';

  const dropUpMenu = document.createElement('div');
  dropUpMenu.id = 'context-menu';
  dropUpMenu.className = 'at-mentions-dropup';
  dropUpMenu.style.cssText = `
    position: absolute;
    bottom: calc(100% + 5px);
    right: 0;
    width: 100%;
    max-height: 250px;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    box-shadow: 0 -4px 6px -1px rgba(0, 0, 0, 0.1);
    z-index: 50;
    display: none;
    flex-direction: column;
  `;

  // Create folder filters section
  const folderFilters = document.createElement('div');
  folderFilters.className = 'folder-filters flex space-x-2 p-2 border-b border-gray-200';
  folderFilters.innerHTML = `
    <div class="folder-filter p-1.5 rounded-md cursor-pointer flex items-center hover:bg-gray-100 bg-blue-100" data-filter="all">
      <span class="mr-1">📁</span><span>All</span>
    </div>
    <div class="folder-filter p-1.5 rounded-md cursor-pointer flex items-center hover:bg-gray-100" data-filter="section">
      <span class="mr-1">📚</span><span>Sections</span>
    </div>
    <div class="folder-filter p-1.5 rounded-md cursor-pointer flex items-center hover:bg-gray-100" data-filter="subsection">
      <span class="mr-1">📑</span><span>Subsections</span>
    </div>
    <div class="folder-filter p-1.5 rounded-md cursor-pointer flex items-center hover:bg-gray-100" data-filter="keyword">
      <span class="mr-1">🔑</span><span>Keywords</span>
    </div>
  `;

  // Add click handlers for folder filters
  folderFilters.querySelectorAll('.folder-filter').forEach((filter, index) => {
    filter.addEventListener('click', (event) => {
      event.stopPropagation();
      const filterType = filter.dataset.filter === 'all' ? null : filter.dataset.filter;
      
      // Update visual state
      folderFilters.querySelectorAll('.folder-filter').forEach(f => f.classList.remove('bg-blue-100'));
      filter.classList.add('bg-blue-100');
      
      // Update selection state
      selectedFolderIndex = index;
      isFolderFocused = true;
      selectedItemIndex = -1;
      
      // Update results and selection
      populateResults(filterType);
      updateSelection();
    });
  });

  const resultsContainer = document.createElement('div');
  resultsContainer.className = 'results-container max-h-[160px] overflow-y-auto';

  dropUpMenu.appendChild(folderFilters);
  dropUpMenu.appendChild(resultsContainer);
  inputWrapper.appendChild(dropUpMenu);

  function populateResults(filter = null) {
    resultsContainer.innerHTML = '';
    let visibleItems = [];
    
    // Search subsections
    if (window.subsections && (!filter || filter === 'subsection')) {
      for (const [key, value] of Object.entries(window.subsections)) {
        visibleItems.push({ 
          type: 'subsection', 
          key, 
          value: {
            title: key,
            content: value.text
          }
        });
      }
    }

    // Search keywords
    if (window.keywords && (!filter || filter === 'keyword')) {
      for (const [key, value] of Object.entries(window.keywords)) {
        visibleItems.push({ 
          type: 'keyword', 
          key, 
          value: {
            title: key,
            content: value.text
          }
        });
      }
    }

    // Search section data
    if (window.sectionData && (!filter || filter === 'section')) {
      for (const [key, value] of Object.entries(window.sectionData)) {
        visibleItems.push({ 
          type: 'section', 
          key, 
          value: {
            title: value.title,
            content: value.content
          }
        });
      }
    }

    // Populate the results
    visibleItems.forEach(item => {
      const icon = item.type === 'section' ? '📚' : 
                  item.type === 'subsection' ? '📑' : 
                  item.type === 'keyword' ? '🔑' : '📎';
      
      const resultItem = createResultItem(
        icon, 
        item.value.title || item.key, 
        () => {
          const content = `${item.value.title || item.key}\n\n${item.value.content}`;
          addToCopyContainer(content, shadowRoot);
          dropUpMenu.style.display = 'none';
        }
      );
      resultsContainer.appendChild(resultItem);
    });

    // Focus the first item if exists
    const firstItem = resultsContainer.querySelector('.at-mention-item');
    if (firstItem) firstItem.focus();
  }

  // Helper function to create result items
  function createResultItem(icon, text, clickHandler) {
    const item = document.createElement('div');
    item.className = 'at-mention-item p-2 hover:bg-gray-100 cursor-pointer flex items-center';
    item.setAttribute('tabindex', '0');
    item.innerHTML = `<span class="mr-2">${icon}</span><span>${text}</span>`;
    item.addEventListener('click', (event) => {
      event.stopPropagation();
      clickHandler();
    });
    return item;
  }

  let selectedFolderIndex = 0;
  let isFolderFocused = true;

  function updateSelection() {
    // Update folder selection
    folderFilters.querySelectorAll('.folder-filter').forEach((folder, index) => {
      if (isFolderFocused && index === selectedFolderIndex) {
        folder.classList.add('bg-blue-200', 'shadow-sm');
      } else {
        folder.classList.remove('bg-blue-200', 'shadow-sm');
      }
    });

    // Update items selection
    if (!isFolderFocused) {
      const items = Array.from(resultsContainer.querySelectorAll('.at-mention-item'));
      items.forEach((item, index) => {
        if (index === selectedItemIndex) {
          item.classList.add('bg-blue-100', 'shadow-sm');
          item.focus();
        } else {
          item.classList.remove('bg-blue-100', 'shadow-sm');
        }
      });
    }
  }

  let selectedItemIndex = -1;

  dropUpMenu.addEventListener('keydown', (event) => {
    const items = Array.from(resultsContainer.querySelectorAll('.at-mention-item'));
    const filters = Array.from(folderFilters.querySelectorAll('.folder-filter'));

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        if (isFolderFocused) {
          isFolderFocused = false;
          selectedItemIndex = 0;
        } else {
          selectedItemIndex = Math.min(selectedItemIndex + 1, items.length - 1);
        }
        updateSelection();
        break;

      case 'ArrowUp':
        event.preventDefault();
        if (selectedItemIndex === 0 || selectedItemIndex === -1) {
          isFolderFocused = true;
          selectedItemIndex = -1;
        } else {
          selectedItemIndex = Math.max(selectedItemIndex - 1, 0);
        }
        updateSelection();
        break;

      case 'ArrowLeft':
      case 'ArrowRight':
        if (isFolderFocused) {
          event.preventDefault();
          selectedFolderIndex = event.key === 'ArrowRight' ?
            (selectedFolderIndex + 1) % filters.length :
            (selectedFolderIndex - 1 + filters.length) % filters.length;
          
          const selectedFilter = filters[selectedFolderIndex];
          const filterType = selectedFilter.dataset.filter === 'all' ? null : selectedFilter.dataset.filter;
          
          filters.forEach(f => f.classList.remove('bg-blue-100'));
          selectedFilter.classList.add('bg-blue-100');
          
          populateResults(filterType);
          updateSelection();
        }
        break;

      case 'Enter':
        event.preventDefault();
        if (isFolderFocused) {
          const selectedFilter = filters[selectedFolderIndex];
          const filterType = selectedFilter.dataset.filter === 'all' ? null : selectedFilter.dataset.filter;
          filters.forEach(f => f.classList.remove('bg-blue-100'));
          selectedFilter.classList.add('bg-blue-100');
          populateResults(filterType);
        } else if (selectedItemIndex >= 0 && items[selectedItemIndex]) {
          items[selectedItemIndex].click();
        }
        break;

      case 'Escape':
        event.preventDefault();
        dropUpMenu.style.display = 'none';
        contextButton.focus();
        break;
    }
  });

  contextButton.addEventListener('click', (event) => {
    event.stopPropagation();
    if (dropUpMenu.style.display === 'none' || !dropUpMenu.style.display) {
      dropUpMenu.style.display = 'flex';
      isFolderFocused = true;
      selectedFolderIndex = 0;
      selectedItemIndex = -1;
      populateResults();
      updateSelection();
    } else {
      dropUpMenu.style.display = 'none';
    }
  });

  document.addEventListener('click', (event) => {
    if (!dropUpMenu.contains(event.target) && !contextButton.contains(event.target)) {
      dropUpMenu.style.display = 'none';
    }
  });
}
