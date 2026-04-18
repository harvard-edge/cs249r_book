import {openStatsModal} from './showQuizStats.js';
import { highlightNextMatch } from '../settings/copy_download.js';
import { findSimilarParagraphsNonBlocking } from '../../libs/agents/fuzzy_match.js';
import {showPopover} from '../../libs/utils/utils.js';
import { enableTooltip } from '../tooltip/tooltip.js';
import { SpacedRepetitionStorageHandler } from '../spaced_repetition/handlers/storage-handler.js';
import { SERVERLESSSCORE } from '../../../configs/env_configs';


let shadowEl;
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

// Add new function to handle score submission
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

export function createButtonGroup(shadowRoot) {
    if(!shadowEl) {
        shadowEl = shadowRoot;
    }
    
    const buttonGroup = document.createElement('div');
    buttonGroup.className = 'flex justify-end space-x-2 mt-4 quiz-btn-group';

    // Create score button without tooltip
    const scoreButton = document.createElement('button');
    scoreButton.className = 'w-6 h-6 flex items-center justify-center hover:text-blue-700 transition-colors duration-200 relative';
    scoreButton.id = 'score-btn-group';
    scoreButton.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
            <path stroke-linecap="round" stroke-linejoin="round" d="M6.633 10.25c.806 0 1.533-.446 2.031-1.08a9.041 9.041 0 0 1 2.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 0 0 .322-1.672V2.75a.75.75 0 0 1 .75-.75 2.25 2.25 0 0 1 2.25 2.25c0 1.152-.26 2.243-.723 3.218-.266.558.107 1.282.725 1.282m0 0h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 0 1-2.649 7.521c-.388.482-.987.729-1.605.729H13.48c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 0 0-1.423-.23H5.904m10.598-9.75H14.25M5.904 18.5c.083.205.173.405.27.602.197.4-.078.898-.523.898h-.908c-.889 0-1.713-.518-1.972-1.368a12 12 0 0 1-.521-3.507c0-1.553.295-3.036.831-4.398C3.387 9.953 4.167 9.5 5 9.5h1.053c.472 0 .745.556.5.96a8.958 8.958 0 0 0-1.302 4.665c0 1.194.232 2.333.654 3.375Z"/>
        </svg>
    `;

    // Add drop-up menu
    const dropUpMenu = document.createElement('div');
    dropUpMenu.className = 'score-options hidden absolute bottom-full left-0 -translate-x-[1rem] mb-2 bg-white dark:bg-zinc-800 rounded-lg shadow-lg p-2 z-50';
    dropUpMenu.style.transform = 'translateX(-25%)';
    dropUpMenu.innerHTML = `
        <div class="flex flex-col space-y-1">
            <button class="thumbs-up-btn p-2 hover:bg-gray-100 dark:hover:bg-zinc-700 rounded-lg">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M6.633 10.25c.806 0 1.533-.446 2.031-1.08a9.041 9.041 0 0 1 2.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 0 0 .322-1.672V2.75a.75.75 0 0 1 .75-.75 2.25 2.25 0 0 1 2.25 2.25c0 1.152-.26 2.243-.723 3.218-.266.558.107 1.282.725 1.282m0 0h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 0 1-2.649 7.521c-.388.482-.987.729-1.605.729H13.48c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 0 0-1.423-.23H5.904m10.598-9.75H14.25M5.904 18.5c.083.205.173.405.27.602.197.4-.078.898-.523.898h-.908c-.889 0-1.713-.518-1.972-1.368a12 12 0 0 1-.521-3.507c0-1.553.295-3.036.831-4.398C3.387 9.953 4.167 9.5 5 9.5h1.053c.472 0 .745.556.5.96a8.958 8.958 0 0 0-1.302 4.665c0 1.194.232 2.333.654 3.375Z"/>
                </svg>
            </button>
            <button class="thumbs-down-btn p-2 hover:bg-gray-100 dark:hover:bg-zinc-700 rounded-lg">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M7.498 15.25H4.372c-1.026 0-1.945-.694-2.054-1.715a12.137 12.137 0 0 1-.068-1.285c0-2.848.992-5.464 2.649-7.521C5.287 4.247 5.886 4 6.504 4h4.016a4.5 4.5 0 0 1 1.423.23l3.114 1.04a4.5 4.5 0 0 0 1.423.23h1.294M7.498 15.25c.618 0 .991.724.725 1.282A7.471 7.471 0 0 0 7.5 19.75 2.25 2.25 0 0 0 9.75 22a.75.75 0 0 0 .75-.75v-.633c0-.573.11-1.14.322-1.672.304-.76.93-1.33 1.653-1.715a9.04 9.04 0 0 0 2.86-2.4c.498-.634 1.226-1.08 2.032-1.08h.384m-10.253 1.5H9.7m8.075-9.75c.01.05.027.1.05.148.593 1.2.925 2.55.925 3.977 0 1.487-.36 2.89-.999 4.125m.023-8.25c-.076-.365.183-.75.575-.75h.908c.889 0 1.713.518 1.972 1.368.339 1.11.521 2.287.521 3.507 0 1.553-.295 3.036-.831 4.398-.306.774-1.086 1.227-1.918 1.227h-1.053c-.472 0-.745-.556-.5-.96a8.95 8.95 0 0 0 .303-.54"/>
                </svg>
            </button>
        </div>
    `;

    scoreButton.appendChild(dropUpMenu);

    // Function to show up arrow
    function showUpArrow() {
        const mainIcon = scoreButton.querySelector('svg:first-child');
        mainIcon.innerHTML = `
            <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 10.5 12 3m0 0 7.5 7.5M12 3v18"/>
        `;
    }

    // Function to restore thumbs up
    function restoreThumbsUp() {
        const mainIcon = scoreButton.querySelector('svg:first-child');
        mainIcon.innerHTML = `
            <path stroke-linecap="round" stroke-linejoin="round" d="${scoreButton.getAttribute('data-original-path')}"/>
        `;
    }

    // Store original path
    scoreButton.setAttribute('data-original-path', scoreButton.querySelector('path').getAttribute('d'));

    let closeTimeout;

    // Click handler for main button
    scoreButton.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const options = scoreButton.querySelector('.score-options');
        options.classList.toggle('hidden');
        
        if (!options.classList.contains('hidden')) {
            showUpArrow();
        } else {
            restoreThumbsUp();
        }
    });

    // Mouseleave handler with delay
    scoreButton.addEventListener('mouseleave', () => {
        closeTimeout = setTimeout(() => {
            const options = scoreButton.querySelector('.score-options');
            options.classList.add('hidden');
            restoreThumbsUp();
        }, 500);
    });

    // Cancel close if mouse enters menu
    dropUpMenu.addEventListener('mouseenter', () => {
        clearTimeout(closeTimeout);
    });

    // Close menu when mouse leaves menu
    dropUpMenu.addEventListener('mouseleave', () => {
        const options = scoreButton.querySelector('.score-options');
        options.classList.add('hidden');
        restoreThumbsUp();
    });

    // Click handlers for rating buttons
    const thumbsUpBtn = dropUpMenu.querySelector('.thumbs-up-btn');
    const thumbsDownBtn = dropUpMenu.querySelector('.thumbs-down-btn');

    thumbsUpBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const quizForm = scoreButton.closest('[data-quiz-id]');
        if (!quizForm) return;
        
        try {
            await submitScore(
                quizForm.getAttribute('data-quiz-id'),
                true,
                quizForm.getAttribute('data-chapter-id'),
                quizForm.getAttribute('data-section-id'),
                quizForm.getAttribute('data-difficulty') || 'intermediate'
            );
            
            // Close drop-up and restore icon
            dropUpMenu.classList.add('hidden');
            restoreThumbsUp();
            
            // Show success popover
            showPopover(shadowEl, "Thanks for the positive feedback!", "success", 3000);
        } catch (error) {
            console.error('Failed to submit positive score:', error);
            showPopover(shadowEl, "Failed to submit rating", "error", 3000);
        }
    });

    thumbsDownBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const quizForm = scoreButton.closest('[data-quiz-id]');
        if (!quizForm) return;
        
        try {
            await submitScore(
                quizForm.getAttribute('data-quiz-id'),
                false,
                quizForm.getAttribute('data-chapter-id'),
                quizForm.getAttribute('data-section-id'),
                quizForm.getAttribute('data-difficulty') || 'intermediate'
            );
            
            // Close drop-up and restore icon
            dropUpMenu.classList.add('hidden');
            restoreThumbsUp();
            
            // Show success popover
            showPopover(shadowEl, "Thanks for the feedback! We'll improve this question.", "success", 3000);
        } catch (error) {
            console.error('Failed to submit negative score:', error);
            showPopover(shadowEl, "Failed to submit rating", "error", 3000);
        }
    });

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

    const graphButton = createIconButton(
        "M7.5 14.25v2.25m3-4.5v4.5m3-6.75v6.75m3-9v9M6 20.25h12A2.25 2.25 0 0020.25 18V6A2.25 2.25 0 0018 3.75H6A2.25 2.25 0 003.75 6v12A2.25 2.25 0 006 20.25z",
        "Graph",
        "quiz-btn-group"
    );

    const highlightButton = createIconButton(
        "M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 01-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 011.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 00-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 01-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 00-3.375-3.375h-1.5a1.125 1.125 0 01-1.125-1.125v-1.5a3.375 3.375 0 00-3.375-3.375H9.75",
        "Copy",
        "copy-btn-group"
    );

    // Add buttons to group in correct order (score button first)
    buttonGroup.appendChild(scoreButton);
    buttonGroup.appendChild(searchButton);
    buttonGroup.appendChild(srButton);
    buttonGroup.appendChild(shareButton);
    buttonGroup.appendChild(graphButton);
    buttonGroup.appendChild(highlightButton);

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


    graphButton.addEventListener('click', (e) => {
        e.preventDefault();


        openStatsModal()
    });

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

    // Add event listeners
    scoreButton.querySelector('.thumbs-down-btn').addEventListener('click', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        const quizForm = scoreButton.closest('#quiz-form');
        if (!quizForm) return;

        const chapterId = quizForm.getAttribute('chapter-title');
        const sectionId = quizForm.getAttribute('data-quiz-title');
        const difficultyElement = quizForm.closest('[data-difficulty]');
        const difficulty = difficultyElement ? 
            ['beginner', 'intermediate', 'advanced', 'blooms-taxonomy'][parseInt(difficultyElement.getAttribute('data-difficulty'))] : 
            'intermediate';
        
        // Get the current question ID from the form's data-quiz attribute
        const quizData = JSON.parse(quizForm.getAttribute('data-quiz') || '[]');
        if (quizData.length === 0) return;
        
        const questionId = quizData[0].id; // Assuming we're scoring the first question

        try {
            await submitScore(questionId, false, chapterId, sectionId, difficulty);
            scoreButton.querySelector('.score-options').classList.add('hidden');
        } catch (error) {
            console.error('Failed to submit score:', error);
        }
    });

    // Add thumbs up click handler
    scoreButton.addEventListener('click', async (e) => {
        if (e.target.closest('.score-options')) return; // Ignore clicks on the options menu
        
        const quizForm = scoreButton.closest('#quiz-form');
        if (!quizForm) return;

        const chapterId = quizForm.getAttribute('chapter-title');
        const sectionId = quizForm.getAttribute('data-quiz-title');
        const difficultyElement = quizForm.closest('[data-difficulty]');
        const difficulty = difficultyElement ? 
            ['beginner', 'intermediate', 'advanced', 'blooms-taxonomy'][parseInt(difficultyElement.getAttribute('data-difficulty'))] : 
            'intermediate';
        
        const quizData = JSON.parse(quizForm.getAttribute('data-quiz') || '[]');
        if (quizData.length === 0) return;
        
        const questionId = quizData[0].id;

        try {
            await submitScore(questionId, true, chapterId, sectionId, difficulty);
        } catch (error) {
            console.error('Failed to submit score:', error);
        }
    });

    // initializeTooltips(shadowRoot);

    return buttonGroup;
}