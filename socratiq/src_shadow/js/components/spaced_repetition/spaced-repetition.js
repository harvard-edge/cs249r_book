export class FlashCard {
    static QUALITY = {
        BLACKOUT: 0,      // Complete blackout
        INCORRECT: 1,     // Incorrect, but remembered when shown
        INCORRECT_EASY: 2, // Incorrect, but seemed easy
        DIFFICULT: 3,     // Correct, but difficult
        HESITANT: 4,      // Correct, with hesitation
        PERFECT: 5        // Perfect recall
    };

    constructor(question, answer) {
        this.question = question;
        this.answer = answer;
        this.repetitions = 0;
        this.easeFactor = 2.5;
        this.interval = 0;
        this.nextReviewDate = new Date();
        this.reviewHistory = [];
        this.id = crypto.randomUUID(); // Add unique ID for editing/deleting
        this.lastReviewQuality = 0; // Default to 'Reset' (0)
    }

    static fromData(data) {
        const card = new FlashCard(data.question, data.answer);
        Object.assign(card, data);
        card.nextReviewDate = new Date(data.nextReviewDate);
        return card;
    }

    edit(question, answer) {
        this.question = question;
        this.answer = answer;
        // Optionally reset stats when card is edited
        // this.repetitions = 0;
        // this.interval = 0;
        return this;
    }

    review(quality) {
        // Record review in history
        this.reviewHistory.push({
            date: new Date(),
            quality: quality,
            interval: this.interval,
            easeFactor: this.easeFactor
        });

        // Update stats based on quality
        if (quality < 3) {
            this.repetitions = 0;
            // Reduce ease factor more significantly for very poor performance
            if (quality === 0) {
                this.easeFactor = Math.max(1.3, this.easeFactor - 0.3);
            }
        } else {
            this.repetitions += 1;
        }

        // Calculate new ease factor with more granular adjustments
        this.easeFactor = Math.max(1.3, 
            this.easeFactor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)));

        // Calculate new interval with more sophisticated spacing
        if (this.repetitions === 0) {
            this.interval = 1; // Review tomorrow
        } else if (this.repetitions === 1) {
            this.interval = quality < 3 ? 1 : 3; // 1 or 3 days based on performance
        } else if (this.repetitions === 2) {
            this.interval = quality < 3 ? 3 : 7; // 3 or 7 days based on performance
        } else {
            // Adjust interval based on quality
            const intervalMultiplier = quality < 3 ? 0.75 : (quality === 5 ? 1.2 : 1.0);
            this.interval = Math.round(this.interval * this.easeFactor * intervalMultiplier);
        }

        // Set next review date
        this.nextReviewDate = new Date();
        this.nextReviewDate.setDate(this.nextReviewDate.getDate() + this.interval);

        this.lastReviewQuality = quality; // Store the last review quality
    }

    getReviewStats() {
        return {
            totalReviews: this.reviewHistory.length,
            averageQuality: this.reviewHistory.reduce((sum, review) => sum + review.quality, 0) / this.reviewHistory.length || 0,
            lastReview: this.reviewHistory[this.reviewHistory.length - 1]?.date || null
        };
    }
}