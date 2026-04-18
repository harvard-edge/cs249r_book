import { jsonrepair } from 'jsonrepair';
import { UINotificationHandler } from "./ui-notification-handler";
import { DuckAI } from "../../../libs/agents/duck-ai-cloudflare.js";

export class SpacedRepetitionStorageHandler extends EventTarget {
    constructor(modal) {
        super();
        this.modal = modal;
        this.saveQueue = [];
        this.processingQueue = false;
        this.inMemoryData = {
            currentChapter: null,
            currentDeck: [], // Only store current deck in memory
            allTags: new Set() // Keep track of all tags
        };
        this.notificationHandler = new UINotificationHandler(modal);
        this.debouncedProcessQueue = this.debounce(this.processQueue.bind(this), 1000);
        this.isFirstLoad = true;
        this.currentSaveNotificationId = null; // Track current save notification
    }

    // Get chapter title from progress data
    getChapterTitle(chapterNum) {
        try {
            const progressData = localStorage.getItem('chapter_progress_data');
            if (progressData) {
                const chapters = JSON.parse(progressData);
                const chapter = chapters.find(c => c.chapter === chapterNum);
                return chapter?.title || `Chapter ${chapterNum}`;
            }
        } catch (error) {
            console.error("Error getting chapter title:", error);
        }
        return `Chapter ${chapterNum}`;
    }

    // Load specific chapter into memory
    loadChapter(chapterNum) {
        try {
            const { currentDeck } = this.switchActiveChapter(chapterNum);
            return currentDeck;
        } catch (error) {
            console.error("Failed to load chapter:", error);
            return [];
        }
    }

    // Get all chapter metadata (without loading cards)
    async getAllChapters() {
        try {
            const savedSets = localStorage.getItem('chapter_card_sets');
            const progressData = localStorage.getItem('chapter_progress_data');

            let allChapters = [];
            if (savedSets) {
                try {
                    allChapters = JSON.parse(savedSets);
                    if (!Array.isArray(allChapters)) {
                        console.warn('Saved chapters is not an array, resetting');
                        allChapters = [];
                    }
                } catch (e) {
                    console.error('Error parsing saved chapters:', e);
                    allChapters = [];
                }
            }

            const chapterProgress = progressData ? JSON.parse(progressData) : [];

            // Map all chapters, not just the current one
            const mappedChapters = allChapters.map(set => {
                const progressInfo = chapterProgress.find(c => c.chapter === set.chapter);
                return {
                    chapter: set.chapter,
                    title: progressInfo?.title || `Chapter ${set.chapter}`,
                    cardCount: set.cards?.length || 0,
                    cards: set.cards // Include the full cards array
                };
            }).sort((a, b) => a.chapter - b.chapter);

            return mappedChapters;
        } catch (error) {
            console.error("Failed to get chapters:", error);
            return [];
        }
    }

    // Save current deck back to storage
    async saveToLocalStorage(fullChapterSets = null) {
        // Dispatch an immediate event to indicate save started
        window.dispatchEvent(new CustomEvent('sr-save-started'));
        
        // Add small delay to allow UI updates to process
        await new Promise(resolve => setTimeout(resolve, 0));

        if (!this.inMemoryData.currentChapter) {
            console.error('Cannot save: No current chapter selected');
            throw new Error('No current chapter selected');
        }

        try {

            this.checkAndPruneDuplicates();

     
            const savedSets = localStorage.getItem('chapter_card_sets');
            
            let allChapters = savedSets ? JSON.parse(savedSets) : [];

            if (fullChapterSets) {
                allChapters = Array.from(fullChapterSets).map(([chapter, cards]) => ({
                    chapter: parseInt(chapter),
                    cards: cards
                }));
            } else {
                const currentChapterIndex = allChapters.findIndex(
                    set => set.chapter === this.inMemoryData.currentChapter.chapter
                );

                if (currentChapterIndex >= 0) {
                    allChapters[currentChapterIndex].cards = [...this.inMemoryData.currentDeck];
                } else {
                    allChapters.push({
                        chapter: this.inMemoryData.currentChapter.chapter,
                        cards: [...this.inMemoryData.currentDeck]
                    });
                }
            }

            // Sort chapters
            allChapters.sort((a, b) => a.chapter - b.chapter);

            // Save to localStorage
            const saveData = JSON.stringify(allChapters);
            localStorage.setItem('chapter_card_sets', saveData);

            // Dispatch save complete event
            window.dispatchEvent(new CustomEvent('sr-save-card-completed', {
                detail: { success: true }
            }));

            return true;
        } catch (error) {
            console.error("Failed to save to localStorage:", error);
            window.dispatchEvent(new CustomEvent('sr-save-card-completed', {
                detail: { success: false, error }
            }));
            throw error;
        }
    }

    // Update card in current deck with immediate feedback
    updateCard(newCard) {
        
        if (!this.inMemoryData.currentChapter) {
            console.error('No current chapter selected');
            throw new Error('No current chapter selected');
        }

        // Initialize current deck if it doesn't exist
        if (!this.inMemoryData.currentDeck) {
            this.inMemoryData.currentDeck = [];
        }

        
        // Add the new card to in-memory deck
        this.inMemoryData.currentDeck.push(newCard);

        // Update tags set
        if (newCard.tags) {
            newCard.tags.forEach(tag => this.inMemoryData.allTags.add(tag));
        }

        // Force immediate save
        this.saveToLocalStorage()
            .catch(err => console.error('Save failed:', err));

        // Return updated state for immediate UI update
        const updatedState = {
            currentDeck: this.inMemoryData.currentDeck,
            allTags: Array.from(this.inMemoryData.allTags),
            stats: this.getCurrentChapterStats(),
            currentChapter: this.inMemoryData.currentChapter
        };
        return updatedState;
    }

    // Get current chapter data from memory
    getCurrentChapterData() {
        return {
            chapter: this.inMemoryData.currentChapter,
            deck: this.inMemoryData.currentDeck,
            stats: this.getCurrentChapterStats(),
            tags: Array.from(this.inMemoryData.allTags)
        };
    }

    // Queue a save operation
    queueSave() {
        // Don't create a new notification if one is already showing
        if (!this.currentSaveNotificationId) {
            this.currentSaveNotificationId = this.notificationHandler.showLoadingState('Saving...');
        }
        
        this.saveQueue.push({
            timestamp: Date.now(),
            notificationId: this.currentSaveNotificationId
        });
        
        this.debouncedProcessQueue();
    }

    // Process the save queue in background
    async processQueue() {
        if (this.processingQueue || this.saveQueue.length === 0) return;

        this.processingQueue = true;
        const latestSave = this.saveQueue[this.saveQueue.length - 1];
        this.saveQueue = [];

        try {
            // Show only one saving notification
            if (!this.currentSaveNotificationId) {
                this.currentSaveNotificationId = this.notificationHandler.showLoadingState('Saving...');
            }

            await this.saveToLocalStorage();
            
            // Update the notification on success
            this.notificationHandler.updateNotification(
                this.currentSaveNotificationId,
                'Saved successfully!',
                'success'
            );
        } catch (error) {
            console.error("Failed to process save queue:", error);
            this.notificationHandler.updateNotification(
                this.currentSaveNotificationId,
                'Failed to save',
                'error'
            );
        } finally {
            // Clear the notification ID
            this.currentSaveNotificationId = null;
            this.processingQueue = false;
            
            // Process any remaining items in the queue
            if (this.saveQueue.length > 0) {
                this.debouncedProcessQueue();
            }
        }
    }

    // Utility debounce function
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Get all tags across all chapters
    getAllTags() {
        try {
            const savedSets = localStorage.getItem('chapter_card_sets');
            const allChapters = savedSets ? JSON.parse(savedSets) : [];
            const tags = new Set();
            
            allChapters.forEach(chapter => {
                chapter.cards.forEach(card => {
                    if (card.tags) {
                        card.tags.forEach(tag => tags.add(tag));
                    }
                });
            });
            
            return Array.from(tags);
        } catch (error) {
            console.error("Failed to get tags:", error);
            return [];
        }
    }

    // Get stats for current chapter
    getCurrentChapterStats() {
        try {
            if (!this.inMemoryData.currentDeck) {
                return { total: 0, learned: 0, percentage: 0 };
            }

            const total = this.inMemoryData.currentDeck.length;
            let totalProgress = 0;

            this.inMemoryData.currentDeck.forEach(card => {
                // Calculate progress based on multiple factors
                let cardProgress = 0;

                // Factor 1: Last Review Quality (50% weight)
                const lastQualityScore = card.lastReviewQuality >= 3 ? 
                    (card.lastReviewQuality - 2) / 3 : 0; // Normalize to 0-1 range
                
                // Factor 2: Review History (30% weight)
                let historyScore = 0;
                if (card.reviewHistory && card.reviewHistory.length > 0) {
                    // Get last 5 reviews (or less if fewer exist)
                    const recentReviews = card.reviewHistory.slice(-5);
                    const avgQuality = recentReviews.reduce((sum, review) => 
                        sum + (review.quality >= 3 ? (review.quality - 2) / 3 : 0), 0
                    ) / recentReviews.length;
                    historyScore = avgQuality;
                }

                // Factor 3: Repetitions and Interval (20% weight)
                const repetitionScore = Math.min(card.repetitions / 5, 1); // Cap at 5 repetitions
                const intervalScore = card.interval > 0 ? 
                    Math.min(Math.log(card.interval) / Math.log(30), 1) : 0; // Log scale, cap at 30 days
                
                // Combine scores with weights
                cardProgress = (lastQualityScore * 0.5) + 
                             (historyScore * 0.3) + 
                             ((repetitionScore + intervalScore) / 2 * 0.2);

                totalProgress += cardProgress;
            });

            // Calculate learned cards (cards with progress > 0.5)
            const learned = Math.round(totalProgress);
            const percentage = total > 0 ? (totalProgress / total) * 100 : 0;

            return {
                total,
                learned,
                percentage: Math.round(percentage * 10) / 10 // Round to 1 decimal
            };
        } catch (error) {
            console.error("Error calculating chapter stats:", error);
            return { total: 0, learned: 0, percentage: 0 };
        }
    }

    // Get stats for all chapters
    getAllChapterStats() {
        try {
            const savedSets = localStorage.getItem('chapter_card_sets');
            const allChapters = savedSets ? JSON.parse(savedSets) : [];
            
            return allChapters.map(chapter => ({
                chapter: chapter.chapter,
                total: chapter.cards.length,
                learned: chapter.cards.filter(card => card.interval > 0).length
            }));
        } catch (error) {
            console.error("Failed to get chapter stats:", error);
            return [];
        }
    }

    // Get cards by tag
    getCardsByTag(tag) {
        try {
            const savedSets = localStorage.getItem('chapter_card_sets');
            const allChapters = savedSets ? JSON.parse(savedSets) : [];
            const cards = [];
            
            allChapters.forEach(chapter => {
                chapter.cards.forEach(card => {
                    if (card.tags && card.tags.includes(tag)) {
                        cards.push({
                            ...card,
                            chapter: chapter.chapter
                        });
                    }
                });
            });
            
            return cards;
        } catch (error) {
            console.error("Failed to get cards by tag:", error);
            return [];
        }
    }

    // Initialize from localStorage
    initializeFromStorage() {
        try {
            // PHASE 1: Disabled automatic chapter creation during initialization
            // this.ensureCurrentChapterExists();
            
            const currentChapter = localStorage.getItem('current_chapter');
            const savedSets = localStorage.getItem('chapter_card_sets');
            const progressData = localStorage.getItem('chapter_progress_data');
            
            const allChapters = savedSets ? JSON.parse(savedSets) : [];
            const chapterProgress = progressData ? JSON.parse(progressData) : [];
            
            this.inMemoryData.currentChapter = currentChapter ? JSON.parse(currentChapter) : null;

            if (this.inMemoryData.currentChapter) {
                const chapter = allChapters.find(set => 
                    set.chapter === this.inMemoryData.currentChapter.chapter
                );
                this.inMemoryData.currentDeck = chapter?.cards || [];
            }

            return {
                currentChapter: this.inMemoryData.currentChapter,
                chapterSets: this.getAllChapters(),
                flashcards: this.inMemoryData.currentDeck,
                tags: Array.from(this.inMemoryData.allTags),
                stats: this.getCurrentChapterStats()
            };
        } catch (error) {
            console.error("Failed to initialize from storage:", error);
            return {
                currentChapter: null,
                chapterSets: [],
                flashcards: [],
                tags: [],
                stats: { total: 0, learned: 0, percentage: 0 }
            };
        }
    }

    // Add this new method
    ensureCurrentChapterExists() {
        try {
            // Get current chapter from localStorage
            const currentChapterData = localStorage.getItem('current_chapter');
            if (!currentChapterData) return;

            const currentChapter = JSON.parse(currentChapterData);
            
            // Check if chapter exists in chapter_card_sets
            const savedSets = localStorage.getItem('chapter_card_sets');
            let allChapters = savedSets ? JSON.parse(savedSets) : [];
            
            // Check if this chapter already exists
            const chapterExists = allChapters.some(set => set.chapter === currentChapter.chapter);
            
            if (!chapterExists) {
                // PHASE 1: Disabled automatic chapter creation
                // Initialize new chapter with empty cards array
                // allChapters.push({
                //     chapter: currentChapter.chapter,
                //     cards: []
                // });
                
                // Sort chapters
                // allChapters.sort((a, b) => a.chapter - b.chapter);
                
                // Save back to localStorage
                // localStorage.setItem('chapter_card_sets', JSON.stringify(allChapters));
                
                // Update in-memory data
                this.inMemoryData.currentChapter = currentChapter;
                this.inMemoryData.currentDeck = [];
            }
            
            return true;
        } catch (error) {
            console.error("Failed to ensure chapter exists:", error);
            return false;
        }
    }

    // Add this new method
    async loadFromLocalStorage() {
        let loadingId;
        if (!this.isFirstLoad) {
            loadingId = this.notificationHandler.showLoadingState();
        }
        let chapterMap = new Map();
        
        try {
            // Add artificial delay to ensure UI is responsive
            await new Promise(resolve => setTimeout(resolve, 100));

            // Load current chapter
            const savedChapter = localStorage.getItem('current_chapter');
            
            if (savedChapter) {
                this.inMemoryData.currentChapter = JSON.parse(savedChapter);
            }

            // Load chapter card sets
            const savedChapterSets = localStorage.getItem('chapter_card_sets');
            
            if (savedChapterSets) {
                const chapterSets = JSON.parse(savedChapterSets);
                
                // Ensure we maintain all chapters when loading
                chapterSets.forEach(set => {
                    const processedCards = set.cards.map(card => ({
                        ...card,
                        nextReviewDate: card.nextReviewDate ? new Date(card.nextReviewDate) : null,
                        reviewHistory: (card.reviewHistory || []).map(history => ({
                            ...history,
                            date: new Date(history.date)
                        }))
                    }));
                    
                    // Always add to chapterMap regardless of current chapter
                    chapterMap.set(set.chapter, processedCards);
                    
                    // Update current deck if this is the current chapter
                    if (this.inMemoryData.currentChapter && 
                        set.chapter === this.inMemoryData.currentChapter.chapter) {
                        this.inMemoryData.currentDeck = processedCards;
                    }

                    // Collect tags
                    processedCards.forEach(card => {
                        if (card.tags) {
                            card.tags.forEach(tag => this.inMemoryData.allTags.add(tag));
                        }
                    });
                });

            }

            // Load review history
            const reviewHistory = localStorage.getItem('review_activity');
            if (reviewHistory) {
                this.inMemoryData.reviewHistory = JSON.parse(reviewHistory);
            } else {
                this.inMemoryData.reviewHistory = {};
            }

            // Update notification handling
            if (!this.isFirstLoad) {
                this.notificationHandler.removeLoadingState(loadingId);
            }
            
            // Set first load to false after successful load
            this.isFirstLoad = false;

            // Return the loaded data
            return {
                currentChapter: this.inMemoryData.currentChapter,
                chapterSets: chapterMap,
                flashcards: this.inMemoryData.currentDeck,
                allTags: Array.from(this.inMemoryData.allTags),
                reviewHistory: this.inMemoryData.reviewHistory
            };

        } catch (error) {
            console.error("Failed to load from localStorage:", error);
            
            // Only show error notification if not first load
            if (!this.isFirstLoad) {
                this.notificationHandler.removeLoadingState(loadingId);
                this.notificationHandler.updateNotification(
                    loadingId,
                    'Failed to load flashcards',
                    'error'
                );
            }
            
            // Set first load to false even after error
            this.isFirstLoad = false;

            // Reset to default state on error
            this.inMemoryData = {
                currentChapter: null,
                currentDeck: [],
                allTags: new Set(),
                reviewHistory: {}
            };
            return {
                currentChapter: null,
                chapterSets: new Map(),
                flashcards: [],
                allTags: [],
                reviewHistory: {}
            };
        }
    }

    // Add this method to SpacedRepetitionStorageHandler
    addReviewToHistory() {
        try {
            const loadingId = this.notificationHandler.showLoadingState();
            const today = new Date().toISOString().split('T')[0]; // Format: YYYY-MM-DD
            
            // Get existing review history or initialize new one
            let reviewHistory = this.inMemoryData.reviewHistory || {};
            
            // Initialize or increment today's count
            reviewHistory[today] = (reviewHistory[today] || 0) + 1;
            
            // Update in-memory data
            this.inMemoryData.reviewHistory = reviewHistory;
            
            // Save to localStorage
            localStorage.setItem('review_activity', JSON.stringify(reviewHistory));
            
            // Show success notification
            this.notificationHandler.removeLoadingState(loadingId);
            this.notificationHandler.updateNotification(
                loadingId,
                'Progress saved',
                'success'
            );
            
            return reviewHistory;
        } catch (error) {
            console.error("Failed to update review history:", error);
            this.notificationHandler.updateNotification(
                'review-error',
                'Failed to save progress',
                'error'
            );
            return null;
        }
    }

    // // Add this method to the SpacedRepetitionStorageHandler class
    // async mergeImportedData(importedData) {
    //     try {
    //         // Validate imported data structure
    //         if (!importedData.chapter_card_sets) {
    //             throw new Error('Invalid import data structure');
    //         }

    //         // Get existing data
    //         const existingSets = JSON.parse(localStorage.getItem('chapter_card_sets') || '[]');
    //         const existingMap = new Map(existingSets.map(set => [set.chapter, set]));

    //         // Merge imported card sets
    //         importedData.chapter_card_sets.forEach(importedSet => {
    //             if (existingMap.has(importedSet.chapter)) {
    //                 // Merge cards for existing chapters
    //                 const existingSet = existingMap.get(importedSet.chapter);
    //                 const existingCards = new Map(existingSet.cards.map(card => [card.id, card]));
                    
    //                 importedSet.cards.forEach(card => {
    //                     if (!existingCards.has(card.id)) {
    //                         existingSet.cards.push(card);
    //                     }
    //                 });
    //             } else {
    //                 // Add new chapters directly
    //                 existingMap.set(importedSet.chapter, importedSet);
    //             }
    //         });

    //         // Convert back to array and save
    //         const mergedSets = Array.from(existingMap.values());
    //         localStorage.setItem('chapter_card_sets', JSON.stringify(mergedSets));

    //         // Reload current chapter if needed
    //         if (this.inMemoryData.currentChapter) {
    //             this.loadChapter(this.inMemoryData.currentChapter.chapter);
    //         }

    //         return true;
    //     } catch (error) {
    //         console.error('Error merging imported data:', error);
    //         throw error;
    //     }
    // }

    async mergeImportedData(importedData) {
        try {
            // Validate imported data structure
            if (!importedData.chapter_card_sets) {
                throw new Error('Invalid import data structure');
            }

            // Get existing data
            const existingSets = JSON.parse(localStorage.getItem('chapter_card_sets') || '[]');
            const existingMap = new Map(existingSets.map(set => [set.chapter, set]));

            // Merge imported card sets
            importedData.chapter_card_sets.forEach(importedSet => {
                if (existingMap.has(importedSet.chapter)) {
                    // Merge cards for existing chapters
                    const existingSet = existingMap.get(importedSet.chapter);
                    const existingCards = new Map(existingSet.cards.map(card => [card.id, card]));
                    
                    importedSet.cards.forEach(card => {
                        if (!existingCards.has(card.id)) {
                            existingSet.cards.push(card);
                        }
                    });
                } else {
                    // Add new chapters directly
                    existingMap.set(importedSet.chapter, importedSet);
                }
            });

            // Convert back to array and save
            const mergedSets = Array.from(existingMap.values());
            localStorage.setItem('chapter_card_sets', JSON.stringify(mergedSets));

            // Update in-memory data if needed
            if (this.inMemoryData.currentChapter) {
                this.loadChapter(this.inMemoryData.currentChapter.chapter);
                // Check for and remove any duplicates
                this.checkAndPruneDuplicates();
            }

            return true;
        } catch (error) {
            console.error('Error merging imported data:', error);
            throw error;
        }
    }

    async addFlashcardsFromText2(text) {
        const loadingId = this.notificationHandler.showLoadingState('Processing flashcards...');
        
        try {
            // Clean the text before processing
            const cleanedText = text.replace(/<svg[\s\S]*?<\/svg>/gi, '') // Remove SVG content
                .replace(/<style[\s\S]*?<\/style>/gi, '') // Remove CSS style tags
                .replace(/data:image\/[^;]+;base64[^"']+/gi, '') // Remove base64 images
                .replace(/<img[^>]+>/gi, '') // Remove img tags
                .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, '') // Remove punctuation
                .replace(/\s{2,}/g, ' ') // Replace multiple spaces with single space
                .trim(); // Remove leading/trailing whitespace

            
            // Get current chapter
            const currentChapter = localStorage.getItem('current_chapter');
            
            if (!currentChapter) {
                throw new Error('No chapter selected');
            }

            const chapterInfo = JSON.parse(currentChapter);

            // Load the current chapter into memory first
            this.loadChapter(chapterInfo.chapter);
            
            // Generate flashcards using DuckAI
            const duckAI = await DuckAI.getInstance();
            const flashcardsGenerator = duckAI.generateFlashcards(cleanedText);
            
            let flashcardsContent = '';
            for await (const chunk of flashcardsGenerator) {
                flashcardsContent += chunk;
            }

            let flashcards;
            try {
                // First try to parse the entire content as JSON
                flashcards = JSON.parse(flashcardsContent);
            } catch {
                // If that fails, try to extract JSON using various patterns
                
                const jsonPattern = /\[[\s\S]*?\](?=\s*$)/;
                const match = flashcardsContent.match(jsonPattern);
                
                if (match) {
                    try {
                        // Use jsonrepair to fix any JSON syntax issues
                        const repairedJson = jsonrepair(match[0]);
                        flashcards = JSON.parse(repairedJson);
                    } catch (e) {
                        console.error('Failed to repair/parse JSON:', e);
                        this.notificationHandler.updateNotification(loadingId, 'Failed to process JSON', 'error');
                        throw new Error('Failed to repair JSON array');
                    }
                } else {
                    this.notificationHandler.updateNotification(loadingId, 'No valid JSON found', 'error');
                    throw new Error('No valid JSON array found in content');
                }
            }
            
            if (!Array.isArray(flashcards)) {
                throw new Error('Parsed content is not an array');
            }


            // Process each flashcard
            const processedCards = flashcards.map(card => ({
                ...card,
                id: crypto.randomUUID(),
                tags: ['AI-Generated'],
                created: new Date().toISOString(),
                repetitions: 0,
                easeFactor: 2.5,
                interval: 0,
                nextReviewDate: null,
                reviewHistory: []
            }));


            // Update in-memory data
            this.inMemoryData.currentDeck = [
                ...this.inMemoryData.currentDeck,
                ...processedCards
            ];

            // Check for and remove any duplicates
            this.checkAndPruneDuplicates();

            // Force immediate save
            await this.saveToLocalStorage();

            // Update modal's data structures
            if (this.modal) {
                
                // Update modal's flashcards array
                if (!this.modal.flashcards) {
                    this.modal.flashcards = [];
                }
                this.modal.flashcards = [...this.modal.flashcards, ...processedCards];

                // Update modal's chapter sets
                const currentChapter = this.inMemoryData.currentChapter.chapter;

                if (!this.modal.chapterSets.has(currentChapter)) {
                    this.modal.chapterSets.set(currentChapter, []);
                }
                const chapterCards = this.modal.chapterSets.get(currentChapter);
                this.modal.chapterSets.set(currentChapter, [...chapterCards, ...processedCards]);

                // Pre-render UI components
                requestAnimationFrame(() => {
                    this.modal.renderChapters();
                    this.modal.renderTags();
                    if (this.modal.uiHandler) {
                        this.modal.uiHandler.updateProgress();
                    }
                });
            } else {
                console.warn('Modal not available for update');
            }

            this.notificationHandler.updateNotification(loadingId, 'Cards added successfully!', 'success');

            return {
                success: true,
                flashcards: processedCards
            };

        } catch (error) {
            console.error("Failed to add flashcards from text:", error);
            this.notificationHandler.updateNotification(loadingId, 'Failed to add flashcards', 'error');
            throw error;
        }
    }

    // Add this new method to switch active chapter
    switchActiveChapter(chapterNum) {
        try {
            // Load the chapter data
            const savedSets = localStorage.getItem('chapter_card_sets');
            
            const allChapters = savedSets ? JSON.parse(savedSets) : [];
            const chapter = allChapters.find(set => set.chapter === chapterNum);
            
            // Update in-memory data
            this.inMemoryData.currentChapter = {
                chapter: chapterNum,
                title: this.getChapterTitle(chapterNum)
            };
            this.inMemoryData.currentDeck = chapter ? chapter.cards : [];
            
            
            // Save current chapter to localStorage
            localStorage.setItem('current_chapter', JSON.stringify(this.inMemoryData.currentChapter));
            
            // PHASE 1: Disabled automatic chapter creation in switchActiveChapter
            // Ensure chapter exists in chapter_card_sets
            // if (!chapter) {
            //     allChapters.push({
            //         chapter: chapterNum,
            //         cards: []
            //     });
            //     allChapters.sort((a, b) => a.chapter - b.chapter);
            //     localStorage.setItem('chapter_card_sets', JSON.stringify(allChapters));
            // }
            
            return {
                currentChapter: this.inMemoryData.currentChapter,
                currentDeck: this.inMemoryData.currentDeck
            };
        } catch (error) {
            console.error("Failed to switch chapter:", error);
            throw error;
        }
    }

    // Add new method for creating cards
    async createNewCard(question, answer, tags = []) {


        if (!this.inMemoryData.currentChapter) {
            console.error('No current chapter selected');
            throw new Error('No current chapter selected');
        }

        // Create new card object
        const newCard = {
            question,
            answer,
            id: crypto.randomUUID(),
            tags: tags,
            created: new Date().toISOString(),
            repetitions: 0,
            easeFactor: 2.5,
            interval: 0,
            nextReviewDate: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
            reviewHistory: [],
            lastReviewQuality: 0
        };


        // Ensure currentDeck is initialized
        if (!this.inMemoryData.currentDeck) {
            this.inMemoryData.currentDeck = [];
        }

        // Add to current deck BEFORE saving
        this.inMemoryData.currentDeck = [...this.inMemoryData.currentDeck, newCard];
        
        // Check for and remove any duplicates
        // this.checkAndPruneDuplicates();

        // Save immediately
        try {
            await this.saveToLocalStorage();
            
            // Notify listeners
            this.dispatchEvent(new CustomEvent('data-updated', {
                detail: {
                    type: 'card-added',
                    card: newCard
                }
            }));

            return newCard;
        } catch (error) {
            console.error('Failed to save new card:', error);
            // Remove card from deck if save failed
            this.inMemoryData.currentDeck = this.inMemoryData.currentDeck.filter(card => card.id !== newCard.id);
            throw error;
        }
    }

    // Add this method to SpacedRepetitionStorageHandler
    checkAndPruneDuplicates() {
        if (!this.inMemoryData.currentDeck) {
            return;
        }

        const seenIds = new Set();
        const seenContent = new Set();
        const uniqueCards = [];
        const duplicates = {
            byId: [],
            byContent: []
        };



        this.inMemoryData.currentDeck.forEach(card => {
            // Create a content signature by combining question and answer
            const contentSignature = `${card.question}|||${card.answer}`.toLowerCase().trim();
            
            // Check for duplicate IDs
            if (seenIds.has(card.id)) {
                console.warn('Duplicate card ID found:', card.id);
                duplicates.byId.push(card);
                return; // Skip this card
            }

            // Check for duplicate content
            if (seenContent.has(contentSignature)) {
                console.warn('Duplicate card content found:', {
                    question: card.question,
                    answer: card.answer
                });
                duplicates.byContent.push(card);
                return; // Skip this card
            }

            // If unique, add to tracking sets and keep the card
            seenIds.add(card.id);
            seenContent.add(contentSignature);
            uniqueCards.push(card);
        });

        // If we found any duplicates, update the deck and save
        if (duplicates.byId.length > 0 || duplicates.byContent.length > 0) {
            console.warn('Found and removed duplicates:', {
                duplicateIds: duplicates.byId.length,
                duplicateContent: duplicates.byContent.length,
                originalCount: this.inMemoryData.currentDeck.length,
                newCount: uniqueCards.length
            });

            // Update the deck with unique cards only
            this.inMemoryData.currentDeck = uniqueCards;
            
            // Save changes to localStorage
            this.saveToLocalStorage()
                .catch(err => console.error('Failed to save after removing duplicates:', err));

            // Notify listeners about the change
            this.dispatchEvent(new CustomEvent('data-updated', {
                detail: {
                    type: 'duplicates-removed',
                    removedCount: {
                        byId: duplicates.byId.length,
                        byContent: duplicates.byContent.length
                    }
                }
            }));
        }

        return {
            uniqueCards,
            duplicates,
            totalRemoved: duplicates.byId.length + duplicates.byContent.length
        };
    }

    addReviewActivity() {
        try {
            const today = new Date().toISOString().split('T')[0]; // YYYY-MM-DD format
            const reviewData = JSON.parse(localStorage.getItem('sr_review_activity') || '{}');
            
            // Increment today's count
            reviewData[today] = (reviewData[today] || 0) + 1;
            
            // Save back to localStorage
            localStorage.setItem('sr_review_activity', JSON.stringify(reviewData));
            
            return reviewData;
        } catch (error) {
            console.error("Error updating review activity:", error);
            return {};
        }
    }

    getReviewActivity() {
        try {
            const reviewData = JSON.parse(localStorage.getItem('sr_review_activity') || '{}');
            console.log("Retrieved review activity data:", reviewData);
            return reviewData;
        } catch (error) {
            console.error("Error getting review activity:", error);
            return {};
        }
    }

    // Add fallback methods for missing implementations
    async createChapter(chapterData) {
        console.warn("createChapter not implemented in this storage handler");
        throw new Error("createChapter method not implemented");
    }

    async addChapter(chapterData) {
        return this.createChapter(chapterData);
    }

    async saveInitialDeckAndCards(chapter, cards) {
        console.warn("saveInitialDeckAndCards not implemented in this storage handler");
        throw new Error("saveInitialDeckAndCards method not implemented");
    }
}