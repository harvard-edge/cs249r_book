export class SpacedRepetitionDragHandler {
    constructor(modalHandler) {
        this.modalHandler = modalHandler;
        this.draggedCard = null;
    }

    initialize() {
        // Find all card elements in the flashcardList container
        const cardContainer = this.modalHandler.shadowRoot.querySelector("#flashcardList");
        if (!cardContainer) {
            console.warn("Card container not found");
            return;
        }

        const cards = cardContainer.querySelectorAll('[data-card-index]');

        cards.forEach((card, index) => {
            
            // Explicitly set draggable attribute
            card.setAttribute('draggable', true);
            
            // Remove any existing listeners first
            card.removeEventListener('dragstart', this._dragStart);
            card.removeEventListener('dragend', this._dragEnd);
            card.removeEventListener('dragover', this._dragOver);
            card.removeEventListener('dragleave', this._dragLeave);
            card.removeEventListener('drop', this._drop);
            
            // Add visual cue that card is draggable
            card.style.cursor = 'grab';
            
            // Add new listeners
            this.addDragListeners(card);
        });
    }

    // Store event listener references
    _dragStart = (e) => {
        
        this.draggedCard = e.target;
        e.target.classList.add('opacity-50');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', e.target.dataset.cardIndex);
        e.target.style.cursor = 'grabbing';
    };

    _dragEnd = (e) => {
        if (this.draggedCard) {
            this.draggedCard.classList.remove('opacity-50');
            this.draggedCard.style.cursor = 'grab';
            this.draggedCard = null;
        }
    };

    _dragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.dataTransfer.dropEffect = 'move';
        const card = e.target.closest('[data-card-index]');
        if (card && card !== this.draggedCard) {
            card.classList.add('bg-blue-50', 'dark:bg-blue-900/20');
        }
    };

    _dragLeave = (e) => {
        const card = e.target.closest('[data-card-index]');
        if (card) {
            card.classList.remove('bg-blue-50', 'dark:bg-blue-900/20');
        }
    };

    _drop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        const card = e.target.closest('[data-card-index]');
        if (!card) return;

   
        card.classList.remove('bg-blue-50', 'dark:bg-blue-900/20');

        if (!this.draggedCard || this.draggedCard === card) {
            return;
        }

        const draggedIdx = parseInt(this.draggedCard.dataset.cardIndex);
        const dropIdx = parseInt(card.dataset.cardIndex);


        this.swapCards(draggedIdx, dropIdx);
    };

    addDragListeners(card) {
        card.addEventListener('dragstart', this._dragStart);
        card.addEventListener('dragend', this._dragEnd);
        card.addEventListener('dragover', this._dragOver);
        card.addEventListener('dragleave', this._dragLeave);
        card.addEventListener('drop', this._drop);
    }

    swapCards(fromIndex, toIndex) {
        
        // Get reference to current chapter's cards
        const cards = this.modalHandler.chapterSets.get(
            parseInt(this.modalHandler.currentChapter.chapter)
        );
        
        if (!cards) {
            console.error("No cards found for current chapter");
            return;
        }

    
        // Swap the cards
        [cards[fromIndex], cards[toIndex]] = [cards[toIndex], cards[fromIndex]];


        // Save the new order and re-render
        try {
            this.modalHandler.storageHandler.saveToLocalStorage();
            
            this.modalHandler.showAllCards(); // Use showAllCards instead of renderCards
        } catch (error) {
            console.error("Error during save/render:", error);
        }
    }
}
