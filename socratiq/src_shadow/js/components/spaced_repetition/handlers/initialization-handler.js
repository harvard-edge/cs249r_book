
export class SpacedRepetitionInitializationHandler {
    constructor(modal) {
        this.modal = modal;
        this.storageHandler = modal.storageHandler;
    }

    async checkAndInitializeFirstTimeUser() {
        try {
            // Check if this is the first visit to the SR app specifically
            const srFirstVisit = localStorage.getItem('sr-app-first-visit');
            
            if (!srFirstVisit) {
                // Check if we already have data (e.g. created by SQLiteHandler)
                const chapters = await this.storageHandler.getAllChapters();
                const hasIntroDeck = chapters && chapters.some(c => c.title === "Introduction" || c.chapter === 0);

                if (hasIntroDeck) {
                    console.log("✅ Introduction deck already exists, skipping creation");
                    localStorage.setItem('sr-app-first-visit', 'true');
                    return;
                }

                // PHASE 1: Disabled automatic initial deck creation
                await this.storageHandler.initializeCurrentChapter();
    
                // Create initial deck for first-time users
                const success = await this.createInitialDeck();
                
                if (success) {
                    console.log("✅ Initial deck created successfully, marking as visited");
                    // Mark SR app as visited ONLY if successful
                    localStorage.setItem('sr-app-first-visit', 'true');
                } else {
                    console.warn("⚠️ Initial deck creation failed, will retry next visit");
                }
            }
        } catch (error) {
            console.error("❌ Error checking first-time user status:", error);
        }
    }

    async createInitialDeck() {
        try {
            // console.log("🔍 BEFORE CREATING INITIAL DECK");
            // await this.storageHandler.debugTablesAsConsoleTable();

            const question = "👋 How to Use Spaced Repetition App";
            const answer = `# How to Use the Spaced Repetition App

This app helps you learn material by creating flashcards and tracking your progress.

## Key Features

* **Decks:** Each chapter of your textbook can be a separate deck of cards.
* **Cards:**  Each card contains a question or concept to be learned.
* **Progress Tracking**: See total cards, mastered cards and cards needing review.
* **Learning Progress Graph**: See a graph of your learned cards.
* **Review Cards**:  Practice your cards based on a spaced repetition algorithm.
* **Add New Cards:** Create new flashcards to help with memorization.
* **Full Markdown Editor**: Use a full markdown editor to create your cards.
* **Tagging**: Tag your cards to help with organization and filtering by using # : \`#tags\`

## Basic Steps

1. **Create Decks:** The application will create a deck for each chapter, which should happen automatically when you visit. 
2. **Add Cards:** As you study, add a new card for each important fact, concept, or question.
3. **Review Cards:** Periodically review the cards in each deck, making sure to keep up with new cards.
4. **Track Progress**: Look at the learning progress to see what cards are mastered and which are not.
5. **Track Quality**: Look at quality graphs to see how well you are doing in rating your reviews.

#introduction`;

            // Create initial chapter if not exists
            if (!this.modal.currentChapter) {
                this.modal.currentChapter = {
                    chapter: 0,
                    title: "Introduction"
                };
            }

            return await this.modal.saveNewCard(question, answer);
        } catch (error) {
            console.error("Error creating initial deck:", error);
            return false;
        }
    }
} 