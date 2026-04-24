export class DeckCreationHandler {
    constructor(modal) {
        this.modal = modal;
        this.shadowRoot = modal.shadowRoot;
        this.storageHandler = modal.storageHandler;
        this.isVisible = false;
    }

    showModal() {
        if (this.isVisible) return;
        
        this.isVisible = true;
        this.createModalHTML();
        this.setupEventListeners();
        this.focusInput();
    }

    hideModal() {
        if (!this.isVisible) return;
        
        this.isVisible = false;
        this.cleanupModal();
    }

    createModalHTML() {
        // Get current theme for deck creation modal
        const hostElement = this.shadowRoot.host;
        const currentTheme = hostElement?.getAttribute('data-socratiq-theme') || 'light';
        const isDark = currentTheme === 'dark';
        
        console.log('[DECK CREATION MODAL DEBUG] Creating deck creation modal with theme:', {
            currentTheme: currentTheme,
            isDark: isDark
        });

        // Create modal element
        const modalElement = document.createElement('div');
        modalElement.id = 'deckCreationModal';
        modalElement.className = 'fixed inset-0 bg-gray-600 bg-opacity-50 z-[10000]';
        
        modalElement.innerHTML = `
            <div class="relative top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 rounded-lg p-6 w-96 shadow-xl"
                 style="background-color: ${isDark ? '#0d1117' : '#ffffff'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important;">
                <h3 class="text-lg font-semibold mb-4" style="color: ${isDark ? '#e6edf3' : '#1f2328'} !important;">Create New Deck</h3>
                
                <div class="mb-4">
                    <label class="block text-sm font-medium mb-2" style="color: ${isDark ? '#9ca3af' : '#6b7280'} !important;">
                        Deck Name
                    </label>
                    <input 
                        type="text" 
                        id="deckNameInput" 
                        placeholder="Enter deck name..." 
                        class="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        style="background-color: ${isDark ? '#21262d' : '#ffffff'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important; border-color: ${isDark ? '#30363d' : '#d0d7de'} !important;"
                        maxlength="50"
                        autocomplete="off"
                    >
                    <div id="deckNameError" class="text-red-500 text-sm mt-1 hidden"></div>
                </div>
                
                <div class="flex justify-end space-x-2">
                    <button 
                        id="cancelDeckCreation" 
                        class="px-4 py-2 text-sm rounded-md transition-colors"
                        style="background-color: ${isDark ? '#21262d' : '#f3f4f6'} !important; color: ${isDark ? '#e6edf3' : '#1f2328'} !important;"
                    >
                        Cancel
                    </button>
                    <button 
                        id="createDeck" 
                        class="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Create Deck
                    </button>
                </div>
            </div>
        `;

        // Append to the modal's shadowRoot
        this.modal.shadowRoot.appendChild(modalElement);
    }

    setupEventListeners() {
        const modal = this.modal.shadowRoot.querySelector('#deckCreationModal');
        const input = this.modal.shadowRoot.querySelector('#deckNameInput');
        const cancelBtn = this.modal.shadowRoot.querySelector('#cancelDeckCreation');
        const createBtn = this.modal.shadowRoot.querySelector('#createDeck');

        // Input validation on typing
        input.addEventListener('input', () => {
            this.validateInput();
        });

        // Create button click
        createBtn.addEventListener('click', () => {
            this.handleCreate();
        });

        // Cancel button click
        cancelBtn.addEventListener('click', () => {
            this.handleCancel();
        });

        // Click outside modal to close
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.handleCancel();
            }
        });

        // Keyboard shortcuts
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.handleCreate();
            } else if (e.key === 'Escape') {
                e.preventDefault();
                this.handleCancel();
            }
        });

        // Focus trap
        modal.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                this.handleTabKey(e);
            }
        });
    }

    focusInput() {
        const input = this.modal.shadowRoot.querySelector('#deckNameInput');
        if (input) {
            input.focus();
        }
    }

    validateInput() {
        const input = this.modal.shadowRoot.querySelector('#deckNameInput');
        const errorDiv = this.modal.shadowRoot.querySelector('#deckNameError');
        const createBtn = this.modal.shadowRoot.querySelector('#createDeck');
        
        const value = input.value.trim();
        let errorMessage = '';

        if (value.length === 0) {
            errorMessage = 'Please enter a deck name';
        } else if (value.length > 50) {
            errorMessage = 'Deck name must be 50 characters or less';
        } else if (this.isDuplicateName(value)) {
            errorMessage = 'A deck with this name already exists';
        }

        if (errorMessage) {
            errorDiv.textContent = errorMessage;
            errorDiv.classList.remove('hidden');
            createBtn.disabled = true;
        } else {
            errorDiv.classList.add('hidden');
            createBtn.disabled = false;
        }
    }

    isDuplicateName(name) {
        // Check if a deck with this name already exists
        try {
            if (this.modal.storageHandler.useFallback) {
                // Use fallback storage check - get chapters from the same storage as base handler
                const existingSets = JSON.parse(localStorage.getItem('chapter_card_sets') || '[]');
                const progressData = JSON.parse(localStorage.getItem('chapter_progress_data') || '[]');
                
                // Check if any chapter has this name
                return existingSets.some(chapterSet => {
                    const progressInfo = progressData.find(c => c.chapter === chapterSet.chapter);
                    const chapterTitle = progressInfo?.title || `Chapter ${chapterSet.chapter}`;
                    return chapterTitle.toLowerCase() === name.toLowerCase();
                });
            } else {
                // Use SQLite storage check
                const chapters = this.modal.chapterSets;
                for (const [chapterNum, cards] of chapters) {
                    const chapterTitle = this.storageHandler.getChapterTitle(chapterNum);
                    if (chapterTitle && chapterTitle.toLowerCase() === name.toLowerCase()) {
                        return true;
                    }
                }
                return false;
            }
        } catch (error) {
            console.error('Error checking duplicate names:', error);
            return false;
        }
    }

    async handleCreate() {
        const input = this.modal.shadowRoot.querySelector('#deckNameInput');
        const createBtn = this.modal.shadowRoot.querySelector('#createDeck');
        
        const deckName = input.value.trim();
        
        if (!deckName) {
            this.validateInput();
            return;
        }

        // Disable button and show loading state
        createBtn.disabled = true;
        createBtn.textContent = 'Creating...';

        try {
            console.log(`Creating deck: "${deckName}"`);
            
            // Check if storage handler is in fallback mode
            if (this.modal.storageHandler.useFallback) {
                console.log('Storage handler is in fallback mode, using fallback storage');
                // Use the fallback storage method
                const chapterNumber = Date.now(); // Use timestamp as unique chapter number
                const newChapter = {
                    chapter: chapterNumber,
                    title: deckName,
                    is_current: false
                };
                
                // Add to fallback storage (localStorage)
                await this.modal.storageHandler.addChapterFallback(newChapter);
            } else {
                // Create the deck using the SQLite storage handler
                const chapterNumber = Date.now(); // Use timestamp as unique chapter number
                const newChapter = {
                    chapter: chapterNumber,
                    title: deckName,
                    is_current: false
                };
                
                // Add the new chapter to storage
                await this.modal.storageHandler.addChapter(newChapter);
            }
            
            // Don't show alert - deck appears in panel automatically
            // alert(`Deck "${deckName}" created successfully!`);
            
            this.hideModal();
            
            // Refresh the UI to show the new deck
            if (this.modal) {
                this.modal.renderChapters();
            }
            
        } catch (error) {
            console.error('Error creating deck:', error);
            alert('Error creating deck. Please try again.');
        } finally {
            createBtn.disabled = false;
            createBtn.textContent = 'Create Deck';
        }
    }

    handleCancel() {
        this.hideModal();
    }

    handleTabKey(e) {
        const modal = this.modal.shadowRoot.querySelector('#deckCreationModal');
        const focusableElements = modal.querySelectorAll(
            'button, input, textarea, select, a[href], [tabindex]:not([tabindex="-1"])'
        );
        
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        if (e.shiftKey) {
            if (document.activeElement === firstElement) {
                e.preventDefault();
                lastElement.focus();
            }
        } else {
            if (document.activeElement === lastElement) {
                e.preventDefault();
                firstElement.focus();
            }
        }
    }

    cleanupModal() {
        const modal = this.modal.shadowRoot.querySelector('#deckCreationModal');
        if (modal) {
            modal.remove();
        }
    }
}