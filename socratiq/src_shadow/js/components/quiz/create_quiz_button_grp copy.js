import {openStatsModal} from './showQuizStats.js';
import { highlightNextMatch } from '../settings/copy_download.js';
import { findSimilarParagraphsNonBlocking } from '../../libs/agents/fuzzy_match.js';
import {showPopover} from '../../libs/utils/utils.js';
import { enableTooltip } from '../tooltip/tooltip.js';


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

    searchButton.addEventListener('click', (e) => {
        e.preventDefault();
        
        // IMMEDIATE spinner display
        const originalHTML = searchButton.innerHTML;
        requestAnimationFrame(() => {
            searchButton.innerHTML = `<svg class="animate-spin icons" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>`;
        });

        // Handle existing matches synchronously
        if (searchButton.dataset.matches && searchButton.dataset.matches !== '[]') {
            const currentIndex = parseInt(searchButton.dataset.currentIndex || '0');
            highlightNextMatch(searchButton, currentIndex);
            searchButton.innerHTML = originalHTML;
            return;
        }

        // Get necessary elements before async operation
        const btnGroup = searchButton.closest('.quiz-btn-group');
        const promptElement = searchButton.closest('[data-prompt]');

        // Start async operations in a separate function
        setTimeout(() => {
            handleSearch(searchButton, btnGroup, promptElement, originalHTML);
        }, 0);
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

    buttonGroup.appendChild(searchButton);
    buttonGroup.appendChild(shareButton);
    buttonGroup.appendChild(graphButton);
    buttonGroup.appendChild(highlightButton);


    // initializeTooltips(shadowRoot);

    return buttonGroup;
}