// src_shadow/js/components/spaced_repetition/utils/chaptermap-utils.js
import { getAllChapterMapEntries } from '../../../libs/utils/tocExtractor.js';

/**
 * Extract chapter number from title
 * @param {string} title - The chapter title (e.g., "8 AI Training – Machine Learning Systems")
 * @returns {number} The chapter number or 0 if not found
 */
export function extractChapterNumber(title) {
    if (!title || typeof title !== 'string') return 0;
    
    const match = title.match(/^(\d+)/);
    return match ? parseInt(match[1]) : 0;
}

/**
 * Normalize chapter title for storage
 * @param {string} title - The chapter title
 * @returns {string} Normalized title
 */
export function normalizeChapterTitle(title) {
    if (!title || typeof title !== 'string') return '';
    return title.trim().replace(/\s+/g, ' ');
}

/**
 * Check if chapter is custom (custom chapters start at 1000)
 * @param {number} chapterNum - The chapter number
 * @returns {boolean} True if custom chapter
 */
export function isCustomChapter(chapterNum) {
    return chapterNum >= 1000;
}

/**
 * Find chapterMap entry by URL
 * @param {string} url - The URL to search for
 * @returns {Promise<Object|null>} The chapterMap entry or null
 */
export async function findChapterMapEntryByUrl(url) {
    try {
        const chapterMapEntries = await getAllChapterMapEntries();
        return chapterMapEntries.find(entry => 
            entry.url === url || entry.originalUrl === url
        ) || null;
    } catch (error) {
        console.error('Error finding chapterMap entry by URL:', error);
        return null;
    }
}

/**
 * Get chapterMap entry by title pattern
 * @param {string} titlePattern - The title pattern to search for
 * @returns {Promise<Object|null>} The chapterMap entry or null
 */
export async function findChapterMapEntryByTitle(titlePattern) {
    try {
        const chapterMapEntries = await getAllChapterMapEntries();
        return chapterMapEntries.find(entry => 
            entry.title && entry.title.includes(titlePattern)
        ) || null;
    } catch (error) {
        console.error('Error finding chapterMap entry by title:', error);
        return null;
    }
}

/**
 * Check if chapterMap is available and has data
 * @returns {Promise<boolean>} True if chapterMap is available
 */
export async function isChapterMapAvailable() {
    try {
        const entries = await getAllChapterMapEntries();
        return entries && entries.length > 0;
    } catch (error) {
        console.warn('ChapterMap not available:', error);
        return false;
    }
}

/**
 * Get current URL normalized for chapterMap lookup
 * @returns {string} Current page URL
 */
export function getCurrentUrl() {
    return window.location.href;
}

/**
 * Sort chapters by number
 * @param {Array} chapters - Array of chapter objects
 * @returns {Array} Sorted chapters
 */
export function sortChaptersByNumber(chapters) {
    return chapters.sort((a, b) => {
        const aNum = extractChapterNumber(a.title);
        const bNum = extractChapterNumber(b.title);
        return aNum - bNum;
    });
}
