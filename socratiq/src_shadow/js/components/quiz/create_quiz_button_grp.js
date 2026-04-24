// import {openStatsModal} from './showQuizStats.js';
import { highlightNextMatch } from '../settings/copy_download.js';
import { findSimilarParagraphsNonBlocking } from '../../libs/agents/fuzzy_match.js';
import {showPopover, cleanupPopovers} from '../../libs/utils/utils.js';
import { enableTooltip } from '../tooltip/tooltip.js';
import { SpacedRepetitionStorageHandler } from '../spaced_repetition/handlers/storage-handler.js';
import { SERVERLESSSCORE } from '../../../configs/env_configs';


let shadowEl;
let cachedSaveChatHistory;

async function saveIndexChatHistory() {
    // Clean up any lingering popovers before saving
    if (shadowEl) {
        cleanupPopovers(shadowEl);
    }

    if (!cachedSaveChatHistory) {
        const { saveChatHistory } = await import('../../index.js');
        cachedSaveChatHistory = saveChatHistory;
        await saveChatHistory();
    } else {
        await cachedSaveChatHistory();
    }
}

export function createIconButton(iconPath, title, id) {
    const button = document.createElement('button');
    button.className = 'w-6 h-6 flex items-center justify-center hover:text-blue-700 transition-colors duration-200';
    button.id = id;
    button.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
            <path stroke-linecap="round" stroke-linejoin="round" d="${iconPath}"/>
        </svg>
    `;


    if (shadowEl) {
        enableTooltip(button, title, shadowEl);
    }

    return button;
}

// Helper function to temporarily show checkmark icon
function showCheckmark(button, originalPath) {
    const checkmarkPath = "M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z";
    const originalSvg = button.innerHTML;
    
    button.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4 text-green-600">
            <path stroke-linecap="round" stroke-linejoin="round" d="${checkmarkPath}"/>
        </svg>
    `;
    
    setTimeout(() => {
        button.innerHTML = originalSvg;
    }, 2000);
}

// Add new helper function for the back button
function createBackButton() {
    return createIconButton(
        "M9 15L3 9m0 0l6-6M3 9h12a6 6 0 010 12h-3",
        "Back to original reading location",
        "back-btn-group"
    );
}

export function createButtonGroup(shadowRoot) {
    if(!shadowEl) {
        shadowEl = shadowRoot;
    }
    
    const buttonGroup = document.createElement('div');
    buttonGroup.className = 'flex justify-end space-x-2 mt-4 quiz-btn-group';

    const searchButton = createIconButton(
        "m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z",
        "Search this text for similar content",
        "highlight-search-btn"
    );


    const srButton = createIconButton(
        "M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0 1 11.186 0Z",
        "spaced repetition",
        'sr-btn-group'
    );

    const shareButton = createIconButton(
        "M7.217 10.907a2.25 2.25 0 100 2.186m0-2.186c.18.324.283.696.283 1.093s-.103.77-.283 1.093m0-2.186l9.566-5.314m-9.566 7.5l9.566 5.314m0 0a2.25 2.25 0 103.935 2.186 2.25 2.25 0 00-3.935-2.186zm0-12.814a2.25 2.25 0 103.933-2.185 2.25 2.25 0 00-3.933 2.185z",
        "Share",
        'share-btn-group'
    );

    // const graphButton = createIconButton(
    //     "M7.5 14.25v2.25m3-4.5v4.5m3-6.75v6.75m3-9v9M6 20.25h12A2.25 2.25 0 0020.25 18V6A2.25 2.25 0 0018 3.75H6A2.25 2.25 0 003.75 6v12A2.25 2.25 0 006 20.25z",
    //     "Graph",
    //     "quiz-btn-group"
    // );

    const highlightButton = createIconButton(
        "M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 01-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 011.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 00-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 01-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 00-3.375-3.375h-1.5a1.125 1.125 0 01-1.125-1.125v-1.5a3.375 3.375 0 00-3.375-3.375H9.75",
        "Copy",
        "copy-btn-group"
    );

    // Create thumbs down button with tooltip
    const thumbsDownButton = createIconButton(
        "M7.498 15.25H4.372c-1.026 0-1.945-.694-2.054-1.715a12.137 12.137 0 0 1-.068-1.285c0-2.848.992-5.464 2.649-7.521C5.287 4.247 5.886 4 6.504 4h4.016a4.5 4.5 0 0 1 1.423.23l3.114 1.04a4.5 4.5 0 0 0 1.423.23h1.294M7.498 15.25c.618 0 .991.724.725 1.282A7.471 7.471 0 0 0 7.5 19.75 2.25 2.25 0 0 0 9.75 22a.75.75 0 0 0 .75-.75v-.633c0-.573.11-1.14.322-1.672.304-.76.93-1.33 1.653-1.715a9.04 9.04 0 0 0 2.86-2.4c.498-.634 1.226-1.08 2.032-1.08h.384m-10.253 1.5H9.7m8.075-9.75c.01.05.027.1.05.148.593 1.2.925 2.55.925 3.977 0 1.487-.36 2.89-.999 4.125m.023-8.25c-.076-.365.183-.75.575-.75h.908c.889 0 1.713.518 1.972 1.368.339 1.11.521 2.287.521 3.507 0 1.553-.295 3.036-.831 4.398-.306.774-1.086 1.227-1.918 1.227h-1.053c-.472 0-.745-.556-.5-.96a8.95 8.95 0 0 0 .303-.54",
        "Rate this question negatively",
        "thumbs-down-btn"
    );

    // buttonGroup.setAttribute('data-listeners-initialized', 'true');

    // Check for existing score and apply styling
    const existingScore = thumbsDownButton.getAttribute('data-last-score');
    if (existingScore === '-1') {
        thumbsDownButton.classList.add('text-red-500');
        enableTooltip(thumbsDownButton, "Click again to remove negative rating", shadowEl);
    }

    // Add selected state class
    thumbsDownButton.dataset.selected = 'false';

    // Add click handler for thumbs down
    thumbsDownButton.addEventListener('click', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        const quizForm = thumbsDownButton.closest('#quiz-form');
        if (!quizForm) return;

        // Prevent any form submission
        if (quizForm.contains(e.target)) {
            e.preventDefault();
        }

        const currentScore = thumbsDownButton.getAttribute('data-last-score');
        const isCurrentlyNegative = currentScore === '-1';

        // Immediately update UI
        if (!isCurrentlyNegative) {
            thumbsDownButton.setAttribute('data-last-score', '-1');
            thumbsDownButton.classList.add('text-red-500');
            enableTooltip(thumbsDownButton, "Click again to remove negative rating", shadowEl);
            showPopover(shadowEl, "Thanks for the feedback! We'll improve this question.", "success", 3000);
        } else {
            thumbsDownButton.removeAttribute('data-last-score');
            thumbsDownButton.classList.remove('text-red-500');
            enableTooltip(thumbsDownButton, "Rate this question negatively", shadowEl);
            showPopover(shadowEl, "Negative rating removed", "info", 3000);
        }

        try {
            const quizData = JSON.parse(quizForm.getAttribute('data-quiz') || '[]');
            if (quizData.length === 0) return;
            
            const questionId = quizData[0].id;
            const chapterId = quizForm.getAttribute('chapter-title');
            const sectionId = quizForm.getAttribute('data-quiz-title');
            
            const difficultyElement = quizForm.closest('[data-difficulty]');
            const difficultyMap = {
                '0': 'beginner',
                '1': 'intermediate',
                '2': 'advanced',
                '3': 'blooms-taxonomy'
            };
            const difficulty = difficultyElement ? 
                difficultyMap[difficultyElement.getAttribute('data-difficulty')] || 'intermediate' : 
                'intermediate';

            // Submit score based on current state
            await submitScore(questionId, isCurrentlyNegative, chapterId, sectionId, difficulty);
            
            // Wait for popover to be visible before cleaning up and saving
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Clean up popovers and save chat history after successful score submission
            cleanupPopovers(shadowEl);
            await saveIndexChatHistory();

        } catch (error) {
            console.error('Failed to submit score:', error);
            // Revert UI changes on error
            if (!isCurrentlyNegative) {
                thumbsDownButton.removeAttribute('data-last-score');
                thumbsDownButton.classList.remove('text-red-500');
                enableTooltip(thumbsDownButton, "Rate this question negatively", shadowEl);
            } else {
                thumbsDownButton.setAttribute('data-last-score', '-1');
                thumbsDownButton.classList.add('text-red-500');
                enableTooltip(thumbsDownButton, "Click again to remove negative rating", shadowEl);
            }
            showPopover(shadowEl, "Failed to update rating", "error", 3000);
        }
        
        return false;
    });

    searchButton.addEventListener('click', async (e) => {
        e.preventDefault();
        
        // Store original HTML and show spinner FIRST, before any other operations
        const originalHTML = searchButton.innerHTML;
        searchButton.disabled = true; // Prevent double-clicks
        searchButton.innerHTML = `<svg class="animate-spin icons" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>`;

        // Handle existing matches
        if (searchButton.dataset.matches && searchButton.dataset.matches !== '[]') {
            const currentIndex = parseInt(searchButton.dataset.currentIndex || '0');
            searchButton.innerHTML = originalHTML;
            searchButton.disabled = false;
            highlightNextMatch(searchButton, currentIndex);
            return;
        }

        // Get necessary elements and start search
        const btnGroup = searchButton.closest('.quiz-btn-group');
        const promptElement = searchButton.closest('[data-prompt]');
        await handleSearch(searchButton, btnGroup, promptElement, originalHTML);
        searchButton.disabled = false;
    });

    // Separate async function for search handling
    async function handleSearch(searchButton, btnGroup, promptElement, originalHTML) {
        if (!btnGroup) {
            console.error('Button group not found');
            searchButton.innerHTML = originalHTML;
            return;
        }

        btnGroup.dataset.initialScrollY = window.scrollY;
        
        if (!btnGroup.querySelector('#back-btn-group')) {
            const backButton = createBackButton();
            backButton.addEventListener('click', (e) => {
                e.preventDefault();
                const originalPosition = parseInt(btnGroup.dataset.initialScrollY || '0');
                window.scrollTo({
                    top: originalPosition,
                    behavior: 'smooth'
                });
            });
            btnGroup.appendChild(backButton);
        }

        const prompt = promptElement?.getAttribute('data-prompt');
        
        if (!prompt) {
            searchButton.innerHTML = originalHTML;
            return;
        }

        document.querySelectorAll('.fuzzy-highlight').forEach(el => {
            el.classList.remove('fuzzy-highlight');
            el.style.animation = '';
        });

        try {
            const matches = await findSimilarParagraphsNonBlocking(prompt);
            
            if (matches.length > 0) {
                searchButton.dataset.matches = JSON.stringify(matches);
                searchButton.dataset.currentIndex = '0';
                
                if (!document.getElementById('highlight-styles')) {
                    const styles = document.createElement('style');
                    styles.id = 'highlight-styles';
                    styles.textContent = `
                        @keyframes highlightFade {
                            0% { background-color: rgba(59, 130, 246, 0.2); }
                            50% { background-color: rgba(59, 130, 246, 0.4); }
                            100% { background-color: rgba(59, 130, 246, 0.1); }
                        }
                        .fuzzy-highlight {
                            background-color: rgba(59, 130, 246, 0.1);
                            scroll-margin-top: 100px;
                        }
                        .fuzzy-highlight.animate {
                            animation: highlightFade 2s ease-out;
                        }
                    `;
                    document.head.appendChild(styles);
                }
                
                highlightNextMatch(searchButton, 0);
            } else {
                showPopover(shadowEl, "No similar text found in document", "info", 3000);
                searchButton.innerHTML = originalHTML;
            }
        } catch (error) {
            console.error('Error in fuzzy matching:', error);
            searchButton.innerHTML = originalHTML;
            showPopover(shadowEl, "Error finding similar text", "error", 3000);
        }
    }

    shareButton.addEventListener('click', (e) => {
        e.preventDefault();
        const markdownPreview = e.currentTarget.closest('#markdown-preview');
        if (markdownPreview) {
            const content = markdownPreview.innerText || markdownPreview.textContent;
            const emailSubject = 'Shared Content';
            const emailBody = encodeURIComponent(content);
            const mailtoLink = `mailto:?subject=${encodeURIComponent(emailSubject)}&body=${emailBody}`;
            
            // Open mailto link in new window/tab
            window.open(mailtoLink, '_blank');
        } else {
            console.warn('Share button clicked - No markdown preview found');
        }
    });

        srButton.addEventListener('click', async (e) => {
            e.preventDefault();
            
            // Get the original icon and create spinner
            const originalHTML = srButton.innerHTML;
            requestAnimationFrame(() => {
                srButton.innerHTML = `<svg class="animate-spin icons" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>`;
            });

            try {

        const markdownPreview = e.currentTarget.closest('#markdown-preview');

                // Get content from the message
                // const content = markdownPreview.getAttribute('data-markdown') || 
                // markdownPreview.querySelector('.text-sm.text-zinc-800')?.textContent || 
                // markdownPreview.textContent;
            const content = markdownPreview.innerText || markdownPreview.textContent;


                // Initialize storage handler
                const storageHandler = new SpacedRepetitionStorageHandler();
                
                // Process and save flashcards
                const result = await storageHandler.addFlashcardsFromText2(content);

                if (result.success) {
                    showPopover(shadowEl, `Created ${result.flashcards.length} flashcards!`, "success", 3000);
                } else {
                    showPopover(shadowEl, "Failed to create flashcards", "error", 3000);
                }

            } catch (error) {
                console.error('Failed to create flashcards:', error);
                showPopover(shadowEl, "Error creating flashcards", "error", 3000);
            } finally {
                // Restore original icon
                setTimeout(() => {
                    srButton.innerHTML = originalHTML;
                }, 2000);
            }
        });


    // graphButton.addEventListener('click', (e) => {
    //     e.preventDefault();


    //     openStatsModal()
    // });

    highlightButton.addEventListener('click', async (e) => {
        e.preventDefault();

        const markdownPreview = e.currentTarget.closest('#markdown-preview');
        if (markdownPreview) {
            try {
                const content = markdownPreview.innerText || markdownPreview.textContent;
                await navigator.clipboard.writeText(content);
                
                // Show checkmark animation
                const originalPath = highlightButton.querySelector('path').getAttribute('d');
                showCheckmark(highlightButton, originalPath);
                
            } catch (err) {
                console.error('Failed to copy content:', err);
            }
        } else {
        }
    });

    // Add buttons to group in correct order
    buttonGroup.appendChild(thumbsDownButton); // Add thumbs down first
    buttonGroup.appendChild(searchButton);
    buttonGroup.appendChild(srButton);
    buttonGroup.appendChild(shareButton);
    // buttonGroup.appendChild(graphButton);
    buttonGroup.appendChild(highlightButton);


    return buttonGroup;
}

async function submitScore(questionId, isPositive, chapterId, sectionId, difficulty) {
    try {
        const response = await fetch(SERVERLESSSCORE, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                chapterId,
                sectionId,
                questionId,
                difficulty,
                scoreChange: isPositive ? 1 : -1
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error submitting score:', error);
        throw error;
    }
}

export function reinitializeButtonListeners(shadowRoot) {
    // Get all quiz button groups, regardless of initialization state
    const buttonGroups = shadowRoot.querySelectorAll('.quiz-btn-group');
    buttonGroups.forEach(group => {
        const newGroup = createButtonGroup(shadowRoot);
        group.parentNode.replaceChild(newGroup, group);
    });
}