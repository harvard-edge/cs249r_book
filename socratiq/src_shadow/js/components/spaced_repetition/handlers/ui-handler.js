export class SpacedRepetitionUIHandler {
    constructor(modal) {
        this.modal = modal;
        this.shadowRoot = modal.shadowRoot;
        this.storageHandler = modal.storageHandler;
        this.themeManager = modal.themeManager;

        // Verify required elements exist
        const requiredElements = [
            '#chapterList',
            '#tagList', 
            '#progressBar', 
            '#progressText',
            '#showStats',
            '#showViz',
            '#startReview',
            '#showAddCard'
        ];

        const missingElements = requiredElements.filter(selector => 
            !this.shadowRoot.querySelector(selector)
        );

        if (missingElements.length > 0) {
            console.error("Missing required elements:", missingElements);
            throw new Error("Required UI elements not found");
        }

        // Add these style properties to enable independent scrolling
        const sidebar = this.shadowRoot.querySelector('#sr-sidebar');
        if (sidebar) {
            sidebar.style.cssText = `
                overflow-y: auto;
                overflow-x: hidden;
                position: sticky;
                top: 1rem;
            `;
        }

        // Ensure chapter list has proper overflow handling
        const chapterList = this.shadowRoot.querySelector('#chapterList');
        if (chapterList) {
            chapterList.style.cssText = `
                max-height: calc(60vh - 4rem); /* Adjust as needed */
                overflow-y: auto;
                overflow-x: hidden;
            `;
        }

        // Ensure tag list has proper overflow handling
        const tagList = this.shadowRoot.querySelector('#tagList');
        if (tagList) {
            tagList.style.cssText = `
                max-height: calc(30vh - 4rem); /* Adjust as needed */
                overflow-y: auto;
                overflow-x: hidden;
            `;
        }

        this.setupEventListeners();
        this.initializeWithRetry();
    }

    async initializeWithRetry(attempts = 0) {
        const maxAttempts = 5;
        const delay = 200; // 200ms between attempts

        try {
            const chapters = await this.storageHandler.getAllChapters();
            console.log("Retrieved chapters:", chapters);
            
            if (!chapters || !Array.isArray(chapters)) {
                if (attempts < maxAttempts) {
                    setTimeout(() => this.initializeWithRetry(attempts + 1), delay);
                    return;
                }
            }
            // Don't call this.render() as it overrides the modal's renderChapters
            // The modal will handle rendering
        } catch (error) {
            if (attempts < maxAttempts) {
                setTimeout(() => this.initializeWithRetry(attempts + 1), delay);
                return;
            }
            console.error("Failed to initialize after retries:", error);
        }
    }

    setupEventListeners() {
        // Setup dropdown toggles
        const decksDropdown = this.shadowRoot.querySelector('[data-dropdown="chapters"]');
        const tagsDropdown = this.shadowRoot.querySelector('[data-dropdown="tags"]');
        const chapterList = this.shadowRoot.querySelector("#chapterList");
        const tagList = this.shadowRoot.querySelector("#tagList");

        // Decks dropdown
        decksDropdown?.closest("button").addEventListener("click", () => {
            chapterList.classList.toggle("hidden");
            decksDropdown.style.transform = chapterList.classList.contains("hidden")
                ? "rotate(0deg)"
                : "rotate(180deg)";
        });

        // Tags dropdown
        tagsDropdown?.closest("button").addEventListener("click", () => {
            tagList.classList.toggle("hidden");
            tagsDropdown.style.transform = tagList.classList.contains("hidden")
                ? "rotate(0deg)"
                : "rotate(180deg)";
        });

        // Action buttons
        this.shadowRoot.querySelector('#showStats')?.addEventListener('click', () => {
            this.modal.showStats();
        });

        this.shadowRoot.querySelector('#showViz')?.addEventListener('click', () => {
            this.modal.showVisualizations();
        });

        this.shadowRoot.querySelector('#startReview')?.addEventListener('click', () => {
            const currentChapter = this.modal.currentChapter?.chapter;
            if (currentChapter) {
                this.modal.showCard(currentChapter, 0);
            }
        });

        // New card button
        this.shadowRoot.querySelector('#showAddCard')?.addEventListener('click', () => {
            const addCardForm = this.shadowRoot.querySelector('#addCardForm');
            if (addCardForm) {
                addCardForm.classList.remove('hidden');
            }
        });
    }

    render() {
        this.renderChapters();
        this.renderTags();
        this.updateProgress();
    }

    updateProgress() {
        const progressBar = this.shadowRoot.querySelector('#progressBar');
        const progressText = this.shadowRoot.querySelector('#progressText');
        
        if (!progressBar || !progressText) {
            console.error("Progress elements not found");
            return;
        }

        const stats = this.storageHandler.getCurrentChapterStats();
        
        // Update progress bar
        const percentage = stats.percentage;
        progressBar.style.width = `${percentage}%`;
        
        // Update text (removed the -1)
        progressText.textContent = `${stats.learned}/${stats.total} Cards Learned`;

        // Update progress bar color based on percentage
        if (percentage >= 80) {
            progressBar.classList.add('bg-green-500');
            progressBar.classList.remove('bg-blue-500', 'bg-yellow-500');
        } else if (percentage >= 50) {
            progressBar.classList.add('bg-blue-500');
            progressBar.classList.remove('bg-green-500', 'bg-yellow-500');
        } else {
            progressBar.classList.add('bg-yellow-500');
            progressBar.classList.remove('bg-green-500', 'bg-blue-500');
        }
    }

    async renderChapters() {
        const chapterList = this.shadowRoot.querySelector('#chapterList');
        if (!chapterList) {
            console.error("Chapter list element not found");
            return;
        }

        try {
            const chapters = await this.storageHandler.getAllChapters();
            console.log("Retrieved chapters:", chapters);
            
            if (!Array.isArray(chapters)) {
                console.warn("Chapters is not an array, initializing empty array");
                return;
            }

            const currentChapter = this.storageHandler.inMemoryData?.currentChapter;
            const themeClasses = this.themeManager?.getThemeClasses() || {
                buttonActive: 'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
                buttonHover: 'hover:bg-gray-100 dark:hover:bg-zinc-700'
            };
            
            chapterList.innerHTML = chapters.map(chapter => `
                <button class="w-full text-left px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-2
                    ${currentChapter?.chapter === chapter.chapter ? 
                    themeClasses.buttonActive : 
                    themeClasses.buttonHover}"
                    data-chapter="${chapter.chapter}">
                    <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                            d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                    <div class="flex-1 flex justify-between items-center">
                        <span>${chapter.title}</span>
                        <span class="text-xs text-gray-500 dark:text-gray-400">${chapter.cardCount} cards</span>
                    </div>
                </button>
            `).join('');
            
            // Add "New Deck" button as the last element
            chapterList.innerHTML += `
                <div class="mt-2">
                    <button id="newDeckBtn" class="w-full flex items-center justify-center px-3 py-2 text-sm text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-md transition-colors">
                        + New Deck
                    </button>
                </div>
            `;
        } catch (error) {
            console.error("Error getting chapters:", error);
        }

        // Add click handlers for chapter buttons
        chapterList.querySelectorAll('button[data-chapter]').forEach(button => {
            button.addEventListener('click', async () => {
                const chapterNum = parseInt(button.dataset.chapter);
                const result = await this.modal.storageHandler.switchActiveChapter(chapterNum);
                
                // Sync modal's currentChapter with storage handler's currentChapter
                if (result && result.currentChapter) {
                    this.modal.setCurrentChapter(result.currentChapter);
                } else {
                    // Fallback: get chapter info and sync
                    const chapters = await this.modal.storageHandler.getAllChapters();
                    const selectedChapter = chapters.find(c => c.chapter === chapterNum);
                    if (selectedChapter) {
                        this.modal.setCurrentChapter({
                            chapter: selectedChapter.chapter,
                            title: selectedChapter.title || `Chapter ${selectedChapter.chapter}`
                        });
                    }
                }
                // Don't call this.render() as it overrides the modal's renderChapters
                // The modal will handle re-rendering
            });
        });
        
        // Add click handler for "New Deck" button
        const newDeckBtn = chapterList.querySelector("#newDeckBtn");
        if (newDeckBtn) {
            newDeckBtn.addEventListener("click", (e) => {
                e.preventDefault();
                console.log("New Deck button clicked!");
                this.modal.deckCreationHandler.showModal();
            });
        }
    }

    async renderTags() {
        const tagList = this.shadowRoot.querySelector('#tagList');
        if (!tagList) {
            console.error("Tag list element not found");
            return;
        }

        // Add defensive checks for tags
        let tags;
        try {
            tags = await this.storageHandler.getAllTags();
            console.log("Retrieved tags:", tags); // Debug log
            
            if (!Array.isArray(tags)) {
                console.warn("Tags is not an array, initializing empty array");
                tags = [];
            }
        } catch (error) {
            console.error("Error getting tags:", error);
            tags = [];
        }

        const currentTag = this.storageHandler.inMemoryData?.currentTag;
        const themeClasses = this.themeManager?.getThemeClasses() || {
            buttonActive: 'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
            buttonHover: 'hover:bg-gray-100 dark:hover:bg-zinc-700'
        };

        const tagButtons = await Promise.all(tags.map(async tag => {
            const count = await this.getTagCount(tag);
            return `
            <button class="w-full text-left px-3 py-2 text-sm rounded-md transition-colors flex items-center gap-2
                ${currentTag?.tag === tag.tag ? 
                themeClasses.buttonActive : 
                themeClasses.buttonHover}"
                data-tag="${tag}">
                <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                        d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                <div class="flex-1 flex justify-between items-center">
                    <span>${tag}</span>
                    <span class="text-xs text-gray-500 dark:text-gray-400">${count} cards</span>
                </div>
            </button>
        `}));

        tagList.innerHTML = tagButtons.join('');

        // Add click handlers
        tagList.querySelectorAll('button').forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const tag = button.dataset.tag;
                console.log(`[UI Handler] Tag clicked: "${tag}"`);
                if (this.modal) {
                    console.log(`[UI Handler] Calling modal.switchTag("${tag}")`);
                    this.modal.switchTag(tag);
                } else {
                    console.error("[UI Handler] Modal reference is missing");
                }
            });
        });
    }

    // Add helper method to get card count for a tag
    async getTagCount(tag) {
        try {
            const cards = await this.storageHandler.getCardsByTag(tag);
            return Array.isArray(cards) ? cards.length : 0;
        } catch (error) {
            console.error("Error getting tag count:", error);
            return 0;
        }
    }

    // Update theme for all UI elements
    updateTheme() {
        if (!this.themeManager) return;
        
        // Re-render chapters and tags with new theme
        this.renderChapters();
        this.renderTags();
    }
} 