import uFuzzy from '@leeoniya/ufuzzy';

export class SpacedRepetitionInnerSearchModal {
  constructor(modalHandler) {
    this.modalHandler = modalHandler;
    this.uf = new uFuzzy();
    
    // Wait for modal handler to finish loading data before setting up search
    this.modalHandler.storageHandler.loadFromLocalStorage().then(() => {
      this.setupSearchModal();
      this.setupDashboardSearch();
    });
  }

  setupSearchModal() {
    // Get current theme for search modal
    const hostElement = this.modalHandler.shadowRoot.host;
    const currentTheme = hostElement?.getAttribute('data-socratiq-theme') || 'light';
    const isDark = currentTheme === 'dark';
    
    console.log('[SEARCH MODAL DEBUG] Creating search modal with theme:', {
      currentTheme: currentTheme,
      isDark: isDark
    });

    const modalHTML = `
      <div id="searchModal" class="hidden fixed inset-0 bg-black bg-opacity-50 z-[99999]">
        <div class="relative top-20 mx-auto p-5 border w-[90%] max-w-md shadow-lg rounded-md"
             style="background-color: ${isDark ? '#0d1117' : '#ffffff'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important; border-color: ${isDark ? '#30363d' : '#d0d7de'} !important;">
          <!-- Close button -->
          <button id="closeSearchModal" class="absolute top-3 right-3 flex items-center z-10"
                  style="color: ${isDark ? '#9ca3af' : '#6b7280'} !important;">
            
            <kbd style="padding-top: 1px; margin-bottom: 5px; background-color: ${isDark ? '#21262d' : '#f3f4f6'} !important; color: ${isDark ? '#9ca3af' : '#6b7280'} !important; border-color: ${isDark ? '#30363d' : '#d1d5db'} !important;" 
                 class="mt-1 px-1 text-xs border">
              ESC
            </kbd>
          </button>

          <!-- Search input with adjusted padding -->
          <div class="relative mt-6">
            <input type="text" 
              id="cardSearchInput"
              class="w-full pl-10 pr-4 py-2 text-sm border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500" 
              style="background-color: ${isDark ? '#21262d' : '#ffffff'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important; border-color: ${isDark ? '#30363d' : '#d0d7de'} !important;"
              placeholder="Search cards...">
            <svg class="absolute left-3 top-2.5 h-4 w-4 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
            </svg>
          </div>

          <!-- Results list -->
          <div id="searchResults" class="mt-4 max-h-[400px] overflow-y-auto">
            <!-- Results will be populated here -->
          </div>
        </div>
      </div>
    `;

    this.modalHandler.shadowRoot.appendChild(
      document.createRange().createContextualFragment(modalHTML)
    );

    this.setupEventListeners();
  }

  setupEventListeners() {
    const searchInput = this.modalHandler.shadowRoot.querySelector('#cardSearchInput');
    const searchModal = this.modalHandler.shadowRoot.querySelector('#searchModal');
    const resultsContainer = this.modalHandler.shadowRoot.querySelector('#searchResults');
    
    // Add close button handler
    const closeButton = this.modalHandler.shadowRoot.querySelector('#closeSearchModal');
    closeButton?.addEventListener('click', () => {
        this.closeModal();
    });

    // Add this section to handle the sidebar search input
    const sidebarSearch = this.modalHandler.shadowRoot.querySelector('#sidebarSearch');
    sidebarSearch?.addEventListener('click', (e) => {
        e.preventDefault();
        this.show();
    });
    sidebarSearch?.addEventListener('focus', (e) => {
        e.preventDefault();
        this.show();
    });

    searchInput?.addEventListener('input', (e) => {
      const needle = e.target.value;
      if (!needle.trim()) {
        resultsContainer.innerHTML = '';
        return;
      }

      const haystack = this.modalHandler.flashcards.map(card => card.question);
      const [idxs, info, order] = this.uf.search(haystack, needle);

      if (!idxs?.length) {
        resultsContainer.innerHTML = `
          <div class="text-sm text-gray-500 dark:text-gray-400 p-2">
            No results found
          </div>
        `;
        return;
      }

      resultsContainer.innerHTML = order.map(i => {
        const cardIdx = info.idx[i];
        const card = this.modalHandler.flashcards[cardIdx];
        const highlightedQuestion = uFuzzy.highlight(
          card.question,
          info.ranges[i],
          (part, matched) => matched ? `<mark class="bg-yellow-100 dark:bg-yellow-900/30">${part}</mark>` : part
        );

        return `
          <button class="w-full text-left p-2 hover:bg-gray-100 dark:hover:bg-zinc-700/50 rounded-md" 
            data-card-index="${cardIdx}">
            <svg class="inline-block w-4 h-4 mr-2 text-gray-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
            </svg>
            ${highlightedQuestion}
          </button>
        `;
      }).join('');

      // Add click handlers for results
      resultsContainer.querySelectorAll('button[data-card-index]').forEach(btn => {
        btn.addEventListener('click', () => {
          const cardIndex = parseInt(btn.dataset.cardIndex);
          searchModal.classList.add('hidden');
          this.modalHandler.showCard(this.modalHandler.currentChapter, cardIndex);
        });
      });
    });

    // Update escape key handler to use closeModal method
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !searchModal.classList.contains('hidden')) {
            this.closeModal();
        }
    });

    // Update click outside handler to use closeModal method
    searchModal?.addEventListener('click', (e) => {
        if (e.target === searchModal) {
            this.closeModal();
        }
    });
  }

  show() {
    const searchModal = this.modalHandler.shadowRoot.querySelector('#searchModal');
    const searchInput = this.modalHandler.shadowRoot.querySelector('#cardSearchInput');
    searchModal.classList.remove('hidden');
    searchInput.value = '';
    searchInput.focus();
  }

  // Add closeModal method
  closeModal() {
    const searchModal = this.modalHandler.shadowRoot.querySelector('#searchModal');
    searchModal.classList.add('hidden');
  }

  // setupDashboardSearch() {
  //   console.log('Setting up dashboard search...');
  //   const dashboardSearch = this.modalHandler.shadowRoot.querySelector('#dashboardSearch');
  //   if (!dashboardSearch) {
  //     console.warn('Dashboard search input not found!');
  //     return;
  //   }
  //   console.log('Found dashboard search input:', dashboardSearch);

  //   let searchTimeout;

  //   dashboardSearch.addEventListener('input', (e) => {
  //     console.log('Search input event fired:', e.target.value);
  //     // Clear previous timeout
  //     if (searchTimeout) {
  //       console.log('Clearing previous search timeout');
  //       clearTimeout(searchTimeout);
  //     }

  //     // Set new timeout for performance
  //     searchTimeout = setTimeout(() => {
  //       const needle = e.target.value;
  //       console.log('Processing search for:', needle);
        
  //       if (this.modalHandler.chapterSets && this.modalHandler.currentChapter) {
  //         console.log('Current chapter:', this.modalHandler.currentChapter);
  //         console.log('Chapter sets available:', this.modalHandler.chapterSets);
  //         this.filterDashboardCards(needle);
  //       } else {
  //         console.warn('Missing required data:', {
  //           chapterSets: !!this.modalHandler.chapterSets,
  //           currentChapter: !!this.modalHandler.currentChapter
  //         });
  //       }
  //     }, 150);
  //   });

  //   console.log('Dashboard search setup complete');
  // }

  setupDashboardSearch() {
    const dashboardSearch = this.modalHandler.shadowRoot.querySelector('#dashboardSearch');
    if (!dashboardSearch) {
      console.warn('Dashboard search input not found!');
      return;
    }
    
    let searchTimeout;

    dashboardSearch.addEventListener('input', (e) => {
      const searchTerm = e.target.value; // Capture the value immediately
      
      if (searchTimeout) {
        clearTimeout(searchTimeout);
      }

      searchTimeout = setTimeout(() => {
        
        if (this.modalHandler.chapterSets && this.modalHandler.currentChapter) {
          this.filterDashboardCards(searchTerm);
        } else {
          console.warn('Missing required data:', {
            chapterSets: !!this.modalHandler.chapterSets,
            currentChapter: !!this.modalHandler.currentChapter
          });
        }
      }, 150);
    });

}

  async filterDashboardCards(needle) {
    
    try {
      // Make sure we have the current chapter and its cards
      if (!this.modalHandler.currentChapter) {
        console.warn('No current chapter selected');
        return;
      }

      const chapterNumber = parseInt(this.modalHandler.currentChapter.chapter);
      const cards = this.modalHandler.chapterSets.get(chapterNumber);
      
      if (!cards || !cards.length) {
        console.warn('No cards found in current chapter');
        return;
      }

      // Get all card elements
      const cardElements = this.modalHandler.shadowRoot.querySelectorAll('[data-card-index]');

      // If no search term, show all cards
      if (!needle || !needle.trim()) {
        cardElements.forEach(card => {
          card.classList.remove('hidden');
          card.style.display = ''; // Reset display property
        });
        return;
      }

      // Get all card content (both question and answer)
      const cardContents = await Promise.all(cards.map(async (card, index) => {
        try {
          const questionText = await this.getCardText(card.question);
          const answerText = await this.getCardText(card.answer);
          return {
            text: `${questionText} ${answerText}`,
            index
          };
        } catch (error) {
          console.error(`Error processing card ${index}:`, error);
          return {
            text: '',
            index
          };
        }
      }));

      // Search using uFuzzy
      const haystack = cardContents.map(c => c.text);
      const [idxs, info, order] = this.uf.search(haystack, needle);

      // Create a Set of matching indices for O(1) lookup
      const matchingIndices = new Set(order.map(i => cardContents[info.idx[i]].index));

      // Show/hide cards based on search results
      cardElements.forEach((cardElement) => {
        const cardIndex = parseInt(cardElement.dataset.cardIndex);
        if (matchingIndices.has(cardIndex)) {
          cardElement.classList.remove('hidden');
          cardElement.style.display = ''; // Reset display property
        } else {
          cardElement.classList.add('hidden');
          cardElement.style.display = 'none';
        }
      });

      // Show "no results" message if needed
      const flashcardList = this.modalHandler.shadowRoot.querySelector("#flashcardList");
      const noResultsMessage = flashcardList.querySelector('.no-results-message');
      
      if (!idxs?.length) {
        if (!noResultsMessage) {
          const message = document.createElement('div');
          message.className = 'no-results-message text-center py-8 text-gray-500 dark:text-gray-400';
          message.innerHTML = '<p>No cards match your search</p>';
          flashcardList.appendChild(message);
        }
      } else if (noResultsMessage) {
        noResultsMessage.remove();
      }

    } catch (error) {
      console.error('Error in filterDashboardCards:', error);
      // Show all cards on error
      const cardElements = this.modalHandler.shadowRoot.querySelectorAll('[data-card-index]');
      cardElements.forEach(card => {
        card.classList.remove('hidden');
        card.style.display = '';
      });
    }
  }

  async getCardText(content) {
    
    // Return empty string if content is undefined or null
    if (!content) {
      console.warn('Empty content received');
      return '';
    }

    try {
      // If content is an ink-mde editor instance
      if (content && typeof content.getValue === 'function') {
        return content.getValue().trim();
      }
      
      // If content is already plain text
      if (typeof content === 'string') {
        return content.trim();
      }

      // If content is an object (might be ink-mde content)
      if (content && typeof content === 'object') {
        return content.toString().replace(/[#*`]/g, '').trim();
      }
    } catch (error) {
      console.error('Error extracting card text:', error);
    }

    console.warn('Falling back to empty string');
    return '';
  }

  showFilteredCards(matchedCards, needle, searchInfo) {
    const flashcardList = this.modalHandler.shadowRoot.querySelector("#flashcardList");
    if (!flashcardList) return;

    // Get current theme for flashcard rendering
    const hostElement = this.shadowRoot.host;
    const currentTheme = hostElement?.getAttribute('data-socratiq-theme') || 'light';
    const isDark = currentTheme === 'dark';
    
    console.log('[SEARCH MODAL DEBUG] Rendering search results with theme:', {
      currentTheme: currentTheme,
      isDark: isDark,
      matchedCardsCount: matchedCards.length
    });

    // Use the existing card rendering logic but only for matched cards
    flashcardList.innerHTML = `
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        ${matchedCards.map((card, index) => {
          // Get the existing button classes from the modal handler
          const buttonClasses = this.modalHandler.getButtonClasses.bind(this.modalHandler);
          
          return `
            <div class="rounded-lg transition-all duration-200 relative"
                 style="box-shadow: 4.2px 8.3px 8.3px hsla(0, 0%, 50%, 0.37);
                        background-color: ${isDark ? '#0d1117' : '#ffffff'} !important;
                        color: ${isDark ? '#e6edf3' : '#1f2328'} !important;"
                 data-card-index="${index}">
              <div class="p-4 space-y-2">
                <div id="question-${index}" class="min-h-[60px]"></div>
                <hr class="editor-separator" />
                <div id="answer-${index}" class="min-h-[60px]"></div>
                
                <!-- Review buttons -->
                <div class="grid grid-cols-2 xs:flex gap-2 mt-4">
                  ${[0, 2, 3, 5].map(quality => `
                    <button class="${buttonClasses(quality, card)}"
                            onclick="event.stopPropagation(); this.closest('[data-card-index]').dispatchEvent(new CustomEvent('reviewCard', {detail: {index: ${index}, quality: ${quality}}}))">
                      <!-- Button content same as original -->
                    </button>
                  `).join('')}
                </div>

                ${card.tags?.length ? `
                  <div class="flex flex-wrap gap-1 mt-3">
                    ${card.tags.map(tag => `
                      <span class="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
                        #${tag}
                      </span>
                    `).join('')}
                  </div>
                ` : ''}
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;

    // Initialize ink-mde editors for the filtered cards
    matchedCards.forEach((card, index) => {
      const questionEditor = ink(
        this.modalHandler.shadowRoot.querySelector(`#question-${index}`),
        {
          doc: card.question,
          interface: {
            toolbar: false,
            attribution: false,
            readonly: false,
          },
          hooks: {
            afterUpdate: (doc) => {
              card.question = doc;
              this.modalHandler.storageHandler.saveToLocalStorage();
            },
          },
        }
      );

      const answerEditor = ink(
        this.modalHandler.shadowRoot.querySelector(`#answer-${index}`),
        {
          doc: card.answer,
          interface: {
            toolbar: false,
            attribution: false,
            readonly: false,
          },
          hooks: {
            afterUpdate: (doc) => {
              card.answer = doc;
              this.modalHandler.storageHandler.saveToLocalStorage();
            },
          },
        }
      );
    });

    // Reattach event listeners
    const cardElements = flashcardList.querySelectorAll("[data-card-index]");
    cardElements.forEach((cardElement) => {
      cardElement.addEventListener("reviewCard", (e) => {
        this.modalHandler.reviewCard(e.detail.index, e.detail.quality);
      });
    });
  }
}
