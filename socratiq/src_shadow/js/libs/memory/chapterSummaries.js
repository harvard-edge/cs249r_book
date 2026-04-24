import compromise from 'compromise';
import { getDBInstance } from '../utils/indexDb.js';

export class ChapterSummaryStorage {
    constructor() {
        this.STORE_NAME = 'chapterSummaries';
        this.recentSaves = new Map();
        this.DEBOUNCE_DELAY = 2000;
    }

    generateSectionSummary(sectionText) {
        const doc = compromise(sectionText);
        const words = doc.terms()
            .filter(t => !t.isStopWord)
            .out('array')
            .map(word => word.toLowerCase().trim())
            .filter(word => word.length > 0);

        const cleanWords = words.map(word => 
            word.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, "")
        ).filter(word => word.length > 0);

        const chunks = [];
        const CHUNK_SIZE = 10;
        for (let i = 0; i < cleanWords.length; i += CHUNK_SIZE) {
            chunks.push(cleanWords.slice(i, i + CHUNK_SIZE).join(' '));
        }

        const selectedChunks = this.frontWeightedSelection(chunks, 4);
        return selectedChunks.join(' ');
    }

    frontWeightedSelection(chunks, numSelections) {
        const selected = new Set();
        const selections = [];

        while (selections.length < numSelections && selections.length < chunks.length) {
            const randomValue = Math.random() * Math.random();
            const index = Math.floor(randomValue * chunks.length);

            if (!selected.has(index)) {
                selected.add(index);
                selections.push(chunks[index]);
            }
        }

        return Array.from(selected)
            .sort((a, b) => a - b)
            .map(index => chunks[index]);
    }

    async saveSectionSummary(chapterId, sectionId, sectionData) {
        try {
            const saveKey = `${chapterId}-${sectionId}`;
            
            const lastSaveTime = this.recentSaves.get(saveKey);
            const now = Date.now();
            if (lastSaveTime && (now - lastSaveTime) < this.DEBOUNCE_DELAY) {
                return;
            }
            
            this.recentSaves.set(saveKey, now);
            
            setTimeout(() => {
                this.recentSaves.delete(saveKey);
            }, this.DEBOUNCE_DELAY);

            const dbManager = await getDBInstance();
            if (!dbManager) {
                throw new Error('Database not initialized');
            }

            const chapterIdStr = String(chapterId);
            let existingData = await dbManager.getByKey(this.STORE_NAME, chapterIdStr);
            
            existingData = existingData || {
                chapterId: chapterIdStr,
                sections: {},
                lastUpdated: new Date().toISOString(),
                url: window.location.href
            };

            // Check if section already exists and content hasn't changed
            const existingSection = existingData.sections[sectionId];
            if (existingSection && existingSection.content === sectionData.content) {
                // Skip if content hasn't changed
                return;
            }

            existingData.sections[sectionId] = {
                ...sectionData,
                summary: this.generateSectionSummary(sectionData.content),
                lastUpdated: new Date().toISOString()
            };

            await dbManager.update(this.STORE_NAME, existingData);
            
        } catch (error) {
            console.error('Error in saveSectionSummary:', error);
            throw error;
        }
    }

    async getChapterSummaries(chapterId) {
        try {
            const dbManager = await getDBInstance();
            if (!dbManager) {
                throw new Error('Database not initialized');
            }
            
            return await dbManager.getByKey(this.STORE_NAME, chapterId);
        } catch (error) {
            console.error('Error getting chapter summaries:', error);
            throw error;
        }
    }
}
