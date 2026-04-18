import { ChapterSummaryStorage } from './chapterSummaries.js'

export class ChapterDataCollector {
    constructor() {
        this.storage = new ChapterSummaryStorage();
    }

    getChapterNumber() {
        const h1ChapterNumber = document.querySelector('h1 .chapter-number');
        return h1ChapterNumber ? h1ChapterNumber.textContent.trim() : null;
    }

    async collectSectionData(section) {
        const h2 = section.querySelector('h2.anchored');
        if (!h2 || h2.textContent.trim().includes('Resources')) {
            return null;
        }

        const sectionId = section.id;
        const dataNumber = h2.getAttribute('data-number');
        const titleText = h2.textContent.trim();
        const content = section.textContent.trim();

        return {
            id: sectionId,
            dataNumber,
            title: titleText,
            content
        };
    }

    async processCurrentPage() {
        const chapterNumber = this.getChapterNumber();
        if (!chapterNumber) return;

        const level2Sections = document.querySelectorAll('section.level2');
        
        for (const section of level2Sections) {
            const sectionData = await this.collectSectionData(section);
            if (sectionData) {
                await this.storage.saveSectionSummary(
                    chapterNumber,
                    sectionData.id,
                    sectionData
                );
            }
        }
    }
}
