import { showPopover} from '../../libs/utils/utils.js'
import { findSimilarParagraphsNonBlocking } from '../../libs/agents/fuzzy_match.js';
import { enableTooltip } from '../../components/tooltip/tooltip.js';
// import { SpacedRepetitionStorageHandler } from '../../components/spaced_repetition/handlers/storage-handler.js';
// import { SpacedRepetitionModal } from '../../components/spaced_repetition/modal.js';
import { SpacedRepetitionModal } from '../../components/spaced_repetition/spaced-repetition-modal-handler.js';


// Add the back button creation function (same as in create_quiz_button_grp.js)
function createBackButton(shadowEle) {
    const button = document.createElement('button');
    button.className = 'w-6 h-6 flex items-center justify-center hover:text-blue-700 transition-colors duration-200';
    button.id = 'back-btn-group';
    button.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 15L3 9m0 0l6-6M3 9h12a6 6 0 010 12h-3"/>
        </svg>
    `;
    
    enableTooltip(button, "Back to original reading location", shadowEle);
    return button;
}

export function copy_download(shadowEle, clone) {
    const copyButton = clone.querySelector('[title="Copy"]');
    const downloadButton = clone.querySelector('[title="Download"]');
    const shareButton = clone.querySelector('[title="Share"]');
    const highlightButton = clone.querySelector('[title="Highlight"]');
    const srButton = clone.querySelector('.st-send-btn');

    enableTooltip(copyButton, "copy", shadowEle);
    enableTooltip(downloadButton, "download", shadowEle);
    enableTooltip(shareButton, "share", shadowEle);
    enableTooltip(highlightButton, "Search Relevant Text", shadowEle);
    if (srButton) {
        enableTooltip(srButton, "Send to Flashcards", shadowEle);
    }
    
    // Get markdown content if available, fallback to text content if not
    const getMarkdownContent = () => {
        const markdown = clone.getAttribute('data-markdown');
        if (markdown) {
            // Clean up the markdown by removing HTML style tags and attributes
            return markdown.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '') // Remove style tags and their content
                          .replace(/<[^>]+style="[^"]*"[^>]*>/g, '') // Remove style attributes
                          .replace(/id="[^"]*"/g, '') // Remove id attributes
                          .replace(/class="[^"]*"/g, '') // Remove class attributes
                          .replace(/<div[^>]*>([\s\S]*?)<\/div>/g, '$1') // Replace div tags with their content
                          .replace(/\s+/g, ' ') // Normalize whitespace
                          .trim(); // Remove leading/trailing whitespace
        }
        
        const textSource = clone.querySelector('.text-sm.text-zinc-800');
        return textSource ? textSource.textContent.trim() : '';
    };

    if (copyButton) {
        copyButton.addEventListener('click', function() {
            const originalHTML = copyButton.innerHTML;
            copyButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="icons">
                <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5" />
            </svg>`;

            const markdownContent = getMarkdownContent();
            navigator.clipboard.writeText(markdownContent).then(() => {
                setTimeout(() => {
                    copyButton.innerHTML = originalHTML;
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy: ', err);
                copyButton.innerHTML = originalHTML;
            });
        });
    }

    if (downloadButton) {
        downloadButton.addEventListener('click', function() {
            const markdownContent = getMarkdownContent();
            const blob = new Blob([markdownContent], {type: 'text/markdown'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'content.md';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    }

    if (shareButton) {
        shareButton.addEventListener('click', (e) => {
            e.preventDefault();
            const aiMessage = e.currentTarget.closest('.p-3.relative.max-w-xs.bg-white, .p-3.max-w-xs.bg-white');
            
            if (!aiMessage) {
                showPopover(shadowEle, "Could not find message to share", "error");
                return;
            }

            const originalHTML = shareButton.innerHTML;
            shareButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="icons">
                <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5" />
            </svg>`;
            
            try {
                const markdownContent = getMarkdownContent();
                
                // Create a structured object for sharing
                const shareContent = {
                    markdown: markdownContent,
                    type: 'ai-message'
                };
                
                // Encode the content properly
                const encodedContent = encodeURIComponent(JSON.stringify(shareContent));
                
                const params = new URLSearchParams(window.location.search);
                params.set('shared_messages', encodedContent);
                params.set('widget_access', 'true');
                
                const shareURL = `${window.location.origin}${window.location.pathname}?${params.toString()}`;
                
                navigator.clipboard.writeText(shareURL)
                    .then(() => {
                        showPopover(shadowEle, "Share URL copied to clipboard! Share this link to show this conversation to others.");
                    })
                    .catch(err => {
                        showPopover(shadowEle, "Failed to copy share URL", "error");
                        console.error('Failed to copy:', err);
                    })
                    .finally(() => {
                        setTimeout(() => {
                            shareButton.innerHTML = originalHTML;
                        }, 2000);
                    });
            } catch (error) {
                console.error('Error sharing message:', error);
                showPopover(shadowEle, "Failed to create share URL", "error");
                shareButton.innerHTML = originalHTML;
            }
        });
    }

    if (highlightButton) {
        highlightButton.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Handle existing matches synchronously
            if (highlightButton.dataset.matches && highlightButton.dataset.matches !== '[]') {
                const currentIndex = parseInt(highlightButton.dataset.currentIndex || '0');
                highlightNextMatch(highlightButton, currentIndex);
                return;
            }

            // IMMEDIATE spinner display for new searches only
            const originalHTML = highlightButton.innerHTML;
            requestAnimationFrame(() => {
                highlightButton.innerHTML = `<svg class="animate-spin icons" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>`;
            });

            // Get necessary elements before async operation
            const buttonContainer = highlightButton.closest('.flex.space-x-2');
            const aiMessage = highlightButton.closest('.p-3.relative.max-w-xs.bg-white, .p-3.max-w-xs.bg-white');

            // Start async operations in a separate function
            setTimeout(() => {
                handleSearch(highlightButton, buttonContainer, aiMessage, originalHTML, shadowEle);
            }, 0);
        });
    }

    // Add SR button handler
    if (srButton) {
        srButton.addEventListener('click', async (e) => {
            e.preventDefault();
            
            // Prevent multiple clicks
            if (srButton.disabled) return;
            srButton.disabled = true;
            
            const originalHTML = srButton.innerHTML;
            requestAnimationFrame(() => {
                srButton.innerHTML = `<svg class="animate-spin icons" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>`;
            });

            try {
                // Get the content only once and store it
                const content = clone.getAttribute('data-markdown') || 
                              clone.querySelector('.text-sm.text-zinc-800')?.textContent || 
                              clone.textContent;

                // Get the modal instance that has the connected storage handler
                const modal = SpacedRepetitionModal.instance;
                if (!modal) {
                    throw new Error('Spaced repetition modal not initialized');
                }

                // Use the storage handler from the modal instance - only process once
                const result = await modal.storageHandler.addFlashcardsFromText(content);

                if (result.success) {
                    showPopover(shadowEle, `Created ${result.flashcards.length} flashcards!`, "success", 3000);
                } else {
                    showPopover(shadowEle, "Failed to create flashcards", "error", 3000);
                }

            } catch (error) {
                console.error('Failed to create flashcards:', error);
                showPopover(shadowEle, "Error creating flashcards", "error", 3000);
            } finally {
                // Re-enable button and restore original HTML
                setTimeout(() => {
                    srButton.disabled = false;
                    srButton.innerHTML = originalHTML;
                }, 2000);
            }
        });
    }
}

// Separate async function for search handling
async function handleSearch(highlightButton, buttonContainer, aiMessage, originalHTML, shadowEle) {
    if (!buttonContainer || !aiMessage) {
        console.error('Required elements not found');
        highlightButton.innerHTML = originalHTML;
        return;
    }

    // Store initial scroll position
    buttonContainer.dataset.initialScrollY = window.scrollY;
    
    // Add back button if needed
    if (!buttonContainer.querySelector('#back-btn-group')) {
        const backButton = createBackButton(shadowEle);
        backButton.addEventListener('click', (e) => {
            e.preventDefault();
            const originalPosition = parseInt(buttonContainer.dataset.initialScrollY || '0');
            window.scrollTo({
                top: originalPosition,
                behavior: 'smooth'
            });
        });
        buttonContainer.appendChild(backButton);
    }

    // Clear existing highlights
    document.querySelectorAll('.fuzzy-highlight').forEach(el => {
        el.classList.remove('fuzzy-highlight');
        el.style.animation = '';
    });

    const content = aiMessage.getAttribute('data-markdown') || 
                   aiMessage.querySelector('.text-sm.text-zinc-800')?.textContent || 
                   aiMessage.textContent;

    try {
        const matches = await findSimilarParagraphsNonBlocking(content);
        
        if (matches.length > 0) {
            highlightButton.dataset.matches = JSON.stringify(matches);
            highlightButton.dataset.currentIndex = '0';
            
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
            
            highlightNextMatch(highlightButton, 0);
        } else {
            showPopover(shadowEle, "No similar text found in document", "info", 3000);
            highlightButton.innerHTML = originalHTML;
        }
    } catch (error) {
        console.error('Error in fuzzy matching:', error);
        highlightButton.innerHTML = originalHTML;
        showPopover(shadowEle, "Error finding similar text", "error", 3000);
    }
}

// Add this helper function to generate a unique selector for elements
export function generateSelector(element) {
    // Generate a unique selector based on tag, classes, or other attributes
    return element.tagName.toLowerCase() + 
           (element.id ? `#${element.id}` : '') +
           (Array.from(element.classList).map(c => `.${c}`).join(''));
}

// Modify highlightNextMatch to find the element using the stored selector
export function highlightNextMatch(button, currentIndex) {
    const matches = JSON.parse(button.dataset.matches || '[]');
    if (!matches.length) return;
    
    // Remove previous highlights and animations
    document.querySelectorAll('.fuzzy-highlight').forEach(el => {
        el.classList.remove('fuzzy-highlight', 'animate');
    });
    
    // Get next match index
    const nextIndex = currentIndex % matches.length;
    const match = matches[nextIndex];
    
    // Find element using the stored selector
    const element = document.querySelector(match.selector);
    
    if (element) {
        element.classList.add('fuzzy-highlight');
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
        
        // Add animation class after a brief delay
        setTimeout(() => {
            element.classList.add('animate');
        }, 100);
        
        // Update button content with counter and cycle icon
        button.innerHTML = `
            <span class="text-xs mr-2">Searches: ${nextIndex + 1}/${matches.length}&nbsp;</span>
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="icons">
                <path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
            </svg>`;
        
        // Show similarity score in popover
        const similarity = Math.round(match.similarity * 100);
        const shadowRoot = button.closest('.shadow-root') || document.body;
        showPopover(
            shadowRoot,
            `Match ${nextIndex + 1} of ${matches.length} (${similarity}% similar)`,
            "info",
            2000
        );
    }
    
    // Update for next click
    button.dataset.currentIndex = (nextIndex + 1).toString();
}

// Update the initial match display
export function updateInitialMatchDisplay(button, matches) {
    const counter = button.querySelector('span');
    if (counter) {
        if (matches.length > 0) {
            counter.textContent = `${1}/${matches.length}`;
            counter.style.display = 'inline-block';
        } else {
            counter.style.display = 'none';
        }
    }
}

export function initializeAllMessageButtons(shadowEle) {
    // Find all AI messages
    const aiMessages = shadowEle.querySelectorAll('.ai-message-chat');

    
    // Initialize buttons for each message
    aiMessages.forEach(clone => {
        copy_download(shadowEle, clone);
    });
}