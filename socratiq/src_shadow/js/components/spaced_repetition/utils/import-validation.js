// Imports are validated in full before either storage backend starts writing.
export const MAX_IMPORT_BYTES = 10 * 1024 * 1024;
const MAX_CARDS = 10000;
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const fail = () => { throw new Error('Invalid or oversized flashcard import'); };
const chapterNumber = value => Number.isSafeInteger(value) && value >= 0;
const text = (value, limit = 100000) => typeof value === 'string' && value.length <= limit;

export function validateFlashcardImport(data) {
    // Bound unknown compatibility fields too, and reject prototype-related keys.
    let nodes = 0;
    let size = 0;
    const visit = (value, depth = 0) => {
        if (++nodes > 500000 || depth > 16) fail();
        if (value === null || typeof value === 'boolean') return;
        if (typeof value === 'number') { if (!Number.isFinite(value)) fail(); return; }
        if (typeof value === 'string') {
            size += value.length;
            if (value.length > 100000 || size > MAX_IMPORT_BYTES) fail();
            return;
        }
        if (typeof value !== 'object' || (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype)) fail();
        for (const key of Object.keys(value)) {
            if (['__proto__', 'constructor', 'prototype'].includes(key)) fail();
            size += key.length;
            visit(value[key], depth + 1);
        }
    };
    visit(data);
    if (!record(data) || !Array.isArray(data.chapter_card_sets) || data.chapter_card_sets.length > 1000) fail();
    const chapters = new Set();
    const ids = new Set();
    let count = 0;
    const chapter_card_sets = data.chapter_card_sets.map(set => {
        if (!record(set) || !chapterNumber(set.chapter) || chapters.has(set.chapter) || !Array.isArray(set.cards)) fail();
        chapters.add(set.chapter);
        if (set.title !== undefined && !text(set.title, 1024)) fail();
        count += set.cards.length;
        if (count > MAX_CARDS) fail();
        const cards = set.cards.map(card => {
            if (!record(card) || !text(card.question) || !text(card.answer)) fail();
            if (card.chapter !== undefined && card.chapter !== set.chapter) fail();
            const id = card.id === undefined ? crypto.randomUUID() : card.id;
            if (!(text(id, 256) && id.length > 0) && !(Number.isSafeInteger(id) && id >= 0)) fail();
            if (ids.has(id)) fail();
            ids.add(id);
            const tags = card.tags === undefined ? [] : card.tags;
            if (!Array.isArray(tags) || tags.length > 100 || tags.some(tag => !text(tag, 256))) fail();
            const reviewHistory = card.reviewHistory === undefined ? [] : card.reviewHistory;
            if (!Array.isArray(reviewHistory) || reviewHistory.length > 10000) fail();
            for (const review of reviewHistory) {
                if (!record(review) || !text(review.date, 64) || !Number.isFinite(Date.parse(review.date)) || !Number.isInteger(review.quality) || review.quality < 0 || review.quality > 5) fail();
            }
            for (const key of ['created', 'nextReviewDate']) {
                if (card[key] != null && (!text(card[key], 64) || !Number.isFinite(Date.parse(card[key])))) fail();
            }
            for (const key of ['repetitions', 'interval', 'easeFactor', 'lastReviewQuality']) {
                if (card[key] !== undefined && (typeof card[key] !== 'number' || !Number.isFinite(card[key]) || card[key] < 0)) fail();
            }
            if (card.repetitions !== undefined && (!Number.isSafeInteger(card.repetitions) || card.repetitions > 1000000)) fail();
            if (card.interval !== undefined && card.interval > 365000) fail();
            if (card.easeFactor !== undefined && card.easeFactor > 100) fail();
            if (card.lastReviewQuality !== undefined && (!Number.isInteger(card.lastReviewQuality) || card.lastReviewQuality > 5)) fail();
            return { repetitions: 0, easeFactor: 2.5, interval: 0, nextReviewDate: null, ...card, id, chapter: set.chapter, tags: [...tags], reviewHistory: reviewHistory.map(review => ({ ...review })) };
        });
        return { ...set, cards };
    });
    if (data.current_chapter != null && (!record(data.current_chapter) || !chapterNumber(data.current_chapter.chapter) || !text(data.current_chapter.title, 1024))) fail();
    return { ...data, chapter_card_sets };
}
