export class FloatingControlsHandler {
    constructor(modal) {
        this.modal = modal;
        this.shadowRoot = modal.shadowRoot;

        // Check if required elements exist before proceeding
        if (!this.checkRequiredElements()) {
            throw new Error("Required elements not found in DOM");
        }

        this.setupFloatingControls();
    }

    checkRequiredElements() {
        const requiredSelectors = [
            '#sr-modal-content',
            '#sr-sidebar',
            '#sr-modal-content-container',
            '#sr-controls-container'
        ];

        const missingElements = requiredSelectors.filter(selector => {
            const element = this.shadowRoot.querySelector(selector);
            if (!element) {
                console.error(`Missing required element: ${selector}`);
                return true;
            }
            return false;
        });

        return missingElements.length === 0;
    }

    setupFloatingControls() {
        // Create floating controls element
        const floatingControls = document.createElement('div');
        floatingControls.id = 'sr-floating-controls';
        floatingControls.classList.add('pointer-events-none'); // Initially disable interactions
        floatingControls.innerHTML = `
            <div class="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-50 opacity-0 transition-all duration-300 translate-y-full backdrop-blur-md bg-gray-300/90 dark:bg-gray-700/90 rounded-full shadow-lg">
                <div class="flex items-center gap-2 p-2 rounded-full">
                    <button id="floating-stats" class="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-zinc-700 rounded-full" title="Show Stats">
                        <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                    </button>
                    <button id="floating-viz" class="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-zinc-700 rounded-full" title="Show Interactive UI">
                        <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path stroke-linecap="round" stroke-linejoin="round" d="M15.042 21.672 13.684 16.6m0 0-2.51 2.225.569-9.47 5.227 7.917-3.286-.672Zm-7.518-.267A8.25 8.25 0 1 1 20.25 10.5M8.288 14.212A5.25 5.25 0 1 1 17.25 10.5"></path>
                        </svg>
                    </button>
                    <button id="floating-review" class="px-4 py-2 text-sm bg-blue-100 hover:bg-blue-200 dark:bg-blue-900/30 dark:hover:bg-blue-800/40 text-blue-700 dark:text-blue-400 rounded-full">
                        Review Cards
                    </button>
                    <div class="w-px h-6 bg-gray-200 dark:bg-zinc-700"></div>
                    <button id="floating-new-card" class="px-4 py-2 text-sm bg-blue-500 hover:bg-blue-600 text-white rounded-full">
                        + New Card
                    </button>
                </div>
            </div>
        `;


        // Fix the selector - it was missing the # for ID
        const modalContent = this.shadowRoot.querySelector('#sr-modal-content');
        if (modalContent) {
            modalContent.style.position = 'relative';
            modalContent.appendChild(floatingControls);
            
            // Add custom CSS for enhanced glass effect
            this.addGlassEffectStyles();
            
            // Initialize observers immediately after appending
            this.setupModalListener();
            this.setupIntersectionObserver();
        } else {
            console.error('Modal content container not found');
        }
    }

    addGlassEffectStyles() {
        // Check if styles already exist
        if (this.shadowRoot.querySelector('#floating-controls-glass-styles')) {
            return;
        }

        const style = document.createElement('style');
        style.id = 'floating-controls-glass-styles';
        style.textContent = `
            #sr-floating-controls > div:first-child {
                /* Glass effect with proper backdrop blur on outer container */
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
            }
        `;
        
        this.shadowRoot.appendChild(style);
    }

    setupModalListener() {
        const modal = this.shadowRoot.querySelector('#spacedRepetitionModal');
        // const floatingControls = this.shadowRoot.querySelector('#sr-floating-controls');
        
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    const isModalHidden = modal.classList.contains('hidden');
                    if (isModalHidden) {
                        this.hideFloatingControls();
                    }
                }
            });
        });

        observer.observe(modal, {
            attributes: true,
            attributeFilter: ['class']
        });
    }

    hideFloatingControls() {
        const floatingControls = this.shadowRoot.querySelector('#sr-floating-controls');
        if (!floatingControls) {
            console.error('❌ Floating controls element not found when trying to hide');
            return;
        }
        const controlsInner = floatingControls.querySelector('div');
        console.log('🔄 Hiding floating controls');
        floatingControls.classList.remove('visible');
        floatingControls.classList.add('pointer-events-none');
        controlsInner.style.opacity = '0';
        controlsInner.style.transform = 'translateX(-50%) translateY(100%)';
    }

    showFloatingControls() {
        const floatingControls = this.shadowRoot.querySelector('#sr-floating-controls');
        if (!floatingControls) {
            console.error('❌ Floating controls element not found when trying to show');
            return;
        }
        const controlsInner = floatingControls.querySelector('div');
        console.log('🔄 Showing floating controls');
        floatingControls.classList.add('visible');
        floatingControls.classList.remove('pointer-events-none');
        controlsInner.style.opacity = '1';
        controlsInner.style.transform = 'translateX(-50%) translateY(0)';
    }

    setupIntersectionObserver() {
        const options = {
            root: this.shadowRoot.querySelector('#sr-modal-content'),
            threshold: 0,
            rootMargin: '-80px 0px 0px 0px'
        };

        console.log('🔄 Setting up IntersectionObserver with options:', options);

        const observer = new IntersectionObserver((entries) => {
            const controlsEntry = entries[0];
            console.log('🔄 IntersectionObserver triggered:', {
                isIntersecting: controlsEntry?.isIntersecting,
                target: controlsEntry?.target?.id
            });
            if (controlsEntry) {  // Add null check
                if (!controlsEntry.isIntersecting) {
                    console.log('🔄 Showing floating controls');
                    this.showFloatingControls();
                } else {
                    console.log('🔄 Hiding floating controls');
                    this.hideFloatingControls();
                }
            }
        }, options);

        // Start observing the original controls
        const controlsContainer = this.shadowRoot.querySelector('#sr-controls-container');
        if (controlsContainer) {
            console.log('🔄 Observing controls container:', controlsContainer);
            observer.observe(controlsContainer);
        } else {
            console.error('❌ Controls container not found for IntersectionObserver');  // Error log
        }
    }

    setupEventListeners() {
        console.log("🔄 Floating controls setupEventListeners");
        const floatingControls = this.shadowRoot.querySelector('#sr-floating-controls');
        
        // Stats button with smooth scroll
        floatingControls.querySelector('#floating-stats').addEventListener('click', () => {
            this.shadowRoot.querySelector('#showStats')?.click();
            const statsSection = this.shadowRoot.querySelector('#statsSection');
            if (statsSection) {
                statsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });

        // Viz button with smooth scroll
        floatingControls.querySelector('#floating-viz').addEventListener('click', () => {
            this.shadowRoot.querySelector('#showViz')?.click();
            const vizSection = this.shadowRoot.querySelector('#flashcardViz');
            if (vizSection) {
                vizSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });

        // Review button
        floatingControls.querySelector('#floating-review').addEventListener('click', () => {
            this.shadowRoot.querySelector('#startReview')?.click();
        });

        // New card button with smooth scroll
        floatingControls.querySelector('#floating-new-card').addEventListener('click', () => {
            // const addCardForm = this.shadowRoot.querySelector('#addCardForm');
            console.log("🔄 Floating new card button clicked");
            this.shadowRoot.querySelector('#showAddCard')?.click();
            const progressBar = this.shadowRoot.querySelector('#progressBar');
            
            // Wait for the form to be visible
            setTimeout(() => {
                progressBar?.scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'start'
                });
            }, 100);
        });
    }
} 