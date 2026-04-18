// src_shadow\js\components\quiz\navParser.js
import { STORAGE_KEY_CHAPTERS } from '../../../configs/env_configs.js';
import {getChapterProgress} from './quiz-storage.js'
// Constants
const STORAGE_KEY = "chapter_progress_data";
const LAST_CHAPTER_KEY = STORAGE_KEY_CHAPTERS //"last_visited_chapter"; // New storage key
const CURRENT_CHAPTER_KEY = "current_chapter";

// Function to update last visited chapter
export function updateLastVisitedChapter(breadcrumbsNav) {
    try {
        if (!breadcrumbsNav) return;
        
        // Find the last chapter link in breadcrumbs
        const chapterLink = breadcrumbsNav.querySelector('li:last-child a');
        if (!chapterLink) return;

        const chapterNumber = chapterLink.querySelector('span.chapter-number')?.textContent;
        const chapterTitle = chapterLink.querySelector('span.chapter-title')?.textContent;

        if (chapterNumber && chapterTitle) {
            // Store current chapter separately
            const currentChapter = {
                chapter: parseInt(chapterNumber),
                title: chapterTitle.trim()
            };
            localStorage.setItem(CURRENT_CHAPTER_KEY, JSON.stringify(currentChapter));

            // Get existing chapter data
            let existingData = [];
            try {
                const storedData = localStorage.getItem(LAST_CHAPTER_KEY);
                const parsedData = JSON.parse(storedData);
                if (Array.isArray(parsedData)) {
                    existingData = parsedData;
                }
            } catch (error) {
                console.warn('No existing chapter data found or invalid format');
            }

            const lastChapter = {
                chapter: parseInt(chapterNumber),
                title: chapterTitle.trim()
            };

            // If we have existing data, update or add the chapter
            if (existingData.length > 0) {
                const index = existingData.findIndex(ch => ch.chapter === lastChapter.chapter);
                if (index !== -1) {
                    existingData[index] = lastChapter;
                } else {
                    existingData.push(lastChapter);
                }
                localStorage.setItem(LAST_CHAPTER_KEY, JSON.stringify(existingData));
            } else {
                // If no existing data, start a new array
                localStorage.setItem(LAST_CHAPTER_KEY, JSON.stringify([lastChapter]));
            }
        }
    } catch (error) {
        console.error('Error updating last visited chapter:', error);
    }
}


export async function parseNavigation(navElement = document.querySelector('#quarto-sidebar')) {
  // Show loading state
  const loadingDiv = document.createElement('div');
  loadingDiv.className = 'loading-overlay';
  loadingDiv.innerHTML = `
      <div class="fixed inset-0 bg-white bg-opacity-75 flex items-center justify-center z-50">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
  `;
  document.body.appendChild(loadingDiv);

  try {
      // Get quiz progress data
      const chapterProgress = await getChapterProgress();
      
      // Original navigation parsing code...
      const navHTML = typeof navElement === 'string' ? navElement : (navElement?.outerHTML || '');
      if (!navHTML) {
          console.error('No navigation content found');
          return [];
      }

      const parser = new DOMParser();
      const doc = parser.parseFromString(navHTML, 'text/html');
      const chapterElements = doc.querySelectorAll('span.chapter-number');
      const chapters = [];

      chapterElements.forEach(element => {
          const chapterNum = parseInt(element.textContent);
          const titleSpan = element.closest('.sidebar-item-text')?.querySelector('span.chapter-title');
          
          if (titleSpan && !isNaN(chapterNum)) {
              const title = titleSpan.textContent.trim();
              const progress = chapterProgress.get(chapterNum) || { passedQuizzes: 0, totalAttempted: 0 };
              
              chapters[chapterNum-1] = {
                  chapter: chapterNum,
                  title: title,
                  quizzesTaken: progress.totalAttempted,
                  totalQuizzes: 10,  // Set to 10 as specified
                  passedQuizzes: progress.passedQuizzes
              };
          }
      });

      const filteredChapters = chapters.filter(Boolean);
      
      try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(filteredChapters));
      } catch (error) {
          console.error('Error saving chapter data:', error);
      }

      return filteredChapters;
  } finally {
      // Remove loading overlay
      loadingDiv.remove();
  }
}

// Helper function to get last visited chapter
export function getLastVisitedChapter() {
    try {
        const lastChapter = localStorage.getItem(LAST_CHAPTER_KEY);
        return lastChapter ? JSON.parse(lastChapter) : null;
    } catch (error) {
        console.error('Error getting last visited chapter:', error);
        return null;
    }
}