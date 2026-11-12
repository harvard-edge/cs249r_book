import {addToCopyContainer} from './send_text_highlight.js';
import {searchTaxonomyItems} from '../../../../src_shared/taxonomy-map.js';

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
    background: var(--socratiq-study-surface-elevated, #ffffff);
    color: var(--socratiq-study-text, #0f172a);
    border: 1px solid var(--socratiq-study-border, #e2e8f0);
    border-radius: 0.5rem;
    box-shadow: 0 -4px 6px -1px rgba(0, 0, 0, 0.1);
    z-index: 50;
    display: none;
    flex-direction: column;
  `;

  // Create folder filters section
  const folderFilters = document.createElement('div');
  folderFilters.className = 'folder-filters flex flex-wrap gap-1 p-2 border-b border-gray-200';
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
    <div class="folder-filter p-1.5 rounded-md cursor-pointer flex items-center hover:bg-gray-100" data-filter="taxonomy">
      <span class="mr-1">🧭</span><span>Taxonomy</span>
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

  function populateResults(filter = null, filterText = '') {
    resultsContainer.replaceChildren();
    let visibleItems = [];
    const normalizedFilter = String(filterText || '').trim().toLowerCase();

    // Bridge active Focused Reader passage if available
    const activeSource = typeof window !== 'undefined' ? window.__SOCRATIQ_ACTIVE_SOURCE__ : null;
    const activeQuote = typeof activeSource?.quote === 'string' ? activeSource.quote.trim() : '';
    if (activeQuote && (!filter || filter === 'section')) {
      const activeSection = typeof activeSource.section === 'string' ? activeSource.section.trim() : '';
      const activeLabel = `Current reader passage${activeSection ? ' (' + activeSection + ')' : ''}`;
      const matchesFilter = !normalizedFilter
        || 'current reader passage'.includes(normalizedFilter)
        || activeLabel.toLowerCase().includes(normalizedFilter)
        || (activeSection && activeSection.toLowerCase().includes(normalizedFilter))
        || activeQuote.toLowerCase().includes(normalizedFilter);
      if (matchesFilter) {
        const resultItem = createResultItem(
          '📖',
          activeLabel,
          () => {
            const content = `Current reader passage${activeSection ? ' (' + activeSection + ')' : ''}\n\n${activeQuote}`;
            addToCopyContainer(content, shadowRoot);
            dropUpMenu.style.display = 'none';
          }
        );
        resultsContainer.appendChild(resultItem);
      }
    }
    
    // Search subsections
    if (window.subsections && (!filter || filter === 'subsection')) {
      for (const [key, value] of Object.entries(window.subsections)) {
        if (!normalizedFilter || key.toLowerCase().includes(normalizedFilter)) {
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
    }

    // Search keywords
    if (window.keywords && (!filter || filter === 'keyword')) {
      for (const [key, value] of Object.entries(window.keywords)) {
        if (!normalizedFilter || key.toLowerCase().includes(normalizedFilter)) {
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
    }

    // Search section data
    if (window.sectionData && (!filter || filter === 'section')) {
      for (const [key, value] of Object.entries(window.sectionData)) {
        const sectionTitle = value.title || key;
        if (!normalizedFilter || sectionTitle.toLowerCase().includes(normalizedFilter)) {
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
    }

    // Search ML Systems Taxonomy items (placed after sections/subsections/keywords in 'all' so index-0 tests stay unchanged)
    if (!filter || filter === 'taxonomy') {
      const currentUrl = typeof window !== 'undefined' ? (window.location?.href || '') : '';
      const taxonomyMatches = searchTaxonomyItems(filterText, currentUrl);
      const itemsToInclude = filter === 'taxonomy' || normalizedFilter
        ? taxonomyMatches
        : taxonomyMatches.slice(0, 6);
      for (const taxItem of itemsToInclude) {
        visibleItems.push({
          type: 'taxonomy',
          key: taxItem.key,
          value: {
            title: taxItem.title,
            content: taxItem.content
          }
        });
      }
    }

    // Populate the results
    visibleItems.forEach(item => {
      const icon = item.type === 'section' ? '📚' : 
                  item.type === 'subsection' ? '📑' : 
                  item.type === 'keyword' ? '🔑' :
                  item.type === 'taxonomy' ? '🧭' : '📎';
      
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

  function renderItems(filterText = '') {
    populateResults(null, filterText);
  }

  // Helper function to create result items safely without innerHTML
  function createResultItem(icon, text, clickHandler) {
    const item = document.createElement('div');
    item.className = 'at-mention-item p-2 hover:bg-gray-100 cursor-pointer flex items-center';
    item.setAttribute('tabindex', '0');
    const iconSpan = document.createElement('span');
    iconSpan.className = 'mr-2';
    iconSpan.textContent = icon;
    const textSpan = document.createElement('span');
    textSpan.textContent = text;
    item.append(iconSpan, textSpan);
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
    if (dropUpMenu.style.display === 'none' || !dropUpMenu.style.display || dropUpMenu.classList.contains('hidden')) {
      dropUpMenu.classList.remove('hidden');
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

  function attachActiveReaderPassage() {
    const activeSource = typeof window !== 'undefined' ? window.__SOCRATIQ_ACTIVE_SOURCE__ : null;
    const activeQuote = typeof activeSource?.quote === 'string' ? activeSource.quote.trim() : '';
    if (!activeQuote) {
      const copyContainer = shadowRoot.getElementById('copy-container');
      if (copyContainer) {
        copyContainer.querySelectorAll('.copy-item[data-active-reader-passage="true"]').forEach(el => el.remove());
        if (copyContainer.children.length === 0) {
          copyContainer.classList.add('hidden');
        }
      }
      return;
    }
    const activeSection = typeof activeSource.section === 'string' ? activeSource.section.trim() : '';
    const content = `Current reader passage${activeSection ? ' (' + activeSection + ')' : ''}\n\n${activeQuote}`;
    addToCopyContainer(content, shadowRoot, { activeReaderPassage: true });
  }

  attachActiveReaderPassage();
  if (typeof window !== 'undefined') {
    window.addEventListener('socratiq:open', attachActiveReaderPassage);
    window.addEventListener('socratiq:study-open', attachActiveReaderPassage);
    window.addEventListener('socratiq:active-source-updated', attachActiveReaderPassage);
  }
}
