export class SpacedRepetitionEventHandler {
    constructor(modal) {
        this.modal = modal;
        this.shadowRoot = modal.shadowRoot;
    }

    setupEventListeners() {
        try {
            // Wait for DOM to be ready
            requestAnimationFrame(() => {
                this.setupModalControls();
                this.setupFormControls();
                this.setupKeyboardShortcuts();
                this.setupChapterControls();
            });
        } catch (error) {
            console.error("Failed to setup event listeners:", error);
        }
    }

    setupModalControls() {
        const closeBtn = this.shadowRoot.querySelector('#closeSpacedRepModal');
        const toggleStatsBtn = this.shadowRoot.querySelector('#toggleStats');
        const toggleVizBtn = this.shadowRoot.querySelector('#toggleViz');

        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.modal.hide());
        }

        if (toggleStatsBtn) {
            toggleStatsBtn.addEventListener('click', () => {
                const statsSection = this.shadowRoot.querySelector('#statsSection');
                const showText = toggleStatsBtn.querySelector('.show-stats-text');
                const hideText = toggleStatsBtn.querySelector('.hide-stats-text');
                
                statsSection?.classList.toggle('hidden');
                showText?.classList.toggle('hidden');
                hideText?.classList.toggle('hidden');
            });
        }

        if (toggleVizBtn) {
            toggleVizBtn.addEventListener('click', () => {
                const vizSection = this.shadowRoot.querySelector('#flashcardViz');
                const showText = toggleVizBtn.querySelector('.show-viz-text');
                const hideText = toggleVizBtn.querySelector('.hide-viz-text');
                
                vizSection?.classList.toggle('hidden');
                showText?.classList.toggle('hidden');
                hideText?.classList.toggle('hidden');
            });
        }
    }

    setupFormControls() {
        const showAddBtn = this.shadowRoot.querySelector('#showAddCard');
        const cancelAddBtn = this.shadowRoot.querySelector('#cancelAdd');
        // const saveCardBtn = this.shadowRoot.querySelector('#saveCard');
        const addCardForm = this.shadowRoot.querySelector('#addCardForm');

        if (showAddBtn && addCardForm) {
            showAddBtn.addEventListener('click', () => {
                addCardForm.classList.remove('hidden');
                this.shadowRoot.querySelector('#questionInput')?.focus();
            });
        }

        if (cancelAddBtn) {
            cancelAddBtn.addEventListener('click', () => {
                this.modal.clearInputs();
                addCardForm?.classList.add('hidden');
            });
        }

        // if (saveCardBtn) {
        //     saveCardBtn.addEventListener('click', () => this.modal.saveNewCard());
        // }
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                const addCardForm = this.shadowRoot.querySelector('#addCardForm');
                if (addCardForm && !addCardForm.classList.contains('hidden')) {
                    e.preventDefault();
                    this.modal.saveNewCard();
                }
            }
            if (e.key === 'Escape') {
                const addCardForm = this.shadowRoot.querySelector('#addCardForm');
                if (addCardForm && !addCardForm.classList.contains('hidden')) {
                    addCardForm.classList.add('hidden');
                    this.modal.clearInputs();
                }
            }
        });
    }

    setupChapterControls() {
        const chapterList = this.shadowRoot.querySelector('#chapterList');
        if (chapterList) {
            chapterList.addEventListener('click', (e) => {
                const chapterBtn = e.target.closest('[data-chapter]');
                if (chapterBtn) {
                    const chapter = parseInt(chapterBtn.dataset.chapter);
                    this.modal.setCurrentChapter({
                        chapter,
                        title: this.modal.getChapterTitle(chapter)
                    });
                }
            });
        }
    }
} 