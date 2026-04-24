import { Boarding } from "boarding.js";
import "boarding.js/styles/main.css";
import "boarding.js/styles/themes/basic.css";

const ONBOARDING_KEY = 'socratiq_onboarding_completed';

// Add a flag to track if onboarding is already running
let isOnboardingActive = false;

export function initializeOnboarding(shadowRoot) {
    
    if (isOnboardingActive) {
        return;
    }

    // Check if user has already seen the onboarding
    const hasCompletedOnboarding = localStorage.getItem(ONBOARDING_KEY);
    
    if (hasCompletedOnboarding) {
        return;
    }

    // Wait for menu to be opened before starting onboarding
    const waitForMenuOpen = () => {
        const menu = shadowRoot.querySelector('#text-selection-menu');
        
        if (!menu) {
            setTimeout(waitForMenuOpen, 500);
            return;
        }

        // Check if menu is visible (not translated)
        const isMenuOpen = !menu.classList.contains('translate-x-full');
        
        if (!isMenuOpen) {
            // Listen for transform changes on the menu
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                        const isNowOpen = !menu.classList.contains('translate-x-full');
                        if (isNowOpen) {
                            observer.disconnect();
                            startOnboarding(shadowRoot);
                        }
                    }
                });
            });

            observer.observe(menu, {
                attributes: true,
                attributeFilter: ['class']
            });
            return;
        }

        startOnboarding(shadowRoot);
    };

    // Start checking for menu open state
    waitForMenuOpen();
}

function startOnboarding(shadowRoot) {
    if (isOnboardingActive) {
        return;
    }

    // Set the flag immediately to prevent future shows
    localStorage.setItem(ONBOARDING_KEY, 'true');
    isOnboardingActive = true;
    

    // Add viewport adjustment helper
    const adjustViewport = (element) => {
        if (!element) return;
        const rect = element.getBoundingClientRect();
        const isInViewport = (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= window.innerHeight &&
            rect.right <= window.innerWidth
        );

        if (!isInViewport) {
            element.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        }
    };

    const boarding = new Boarding({
        animate: true,
        opacity: 0.8,
        padding: 10,
        allowClose: true,
        keyboardControl: true,
        showButtons: true,
        className: 'socratiq-onboarding',
        container: shadowRoot,
        root: shadowRoot,
        // Reduce animation duration to minimize flickering
        animationDuration: 400,
        // Add scroll handling
        scrollIntoViewOptions: {
            behavior: 'smooth',
            block: 'center'
        },
        styles: {
            popover: {
                'z-index': '99999',
                'position': 'fixed',
                'max-width': '300px',
                'transform-origin': 'center center',
                'max-height': '80vh',
                'overflow-y': 'auto'
            }
        },
        onPopoverRender: (popover) => {
            // Remove existing popovers
            const existingPopovers = shadowRoot.querySelectorAll('.socratiq-onboarding');
            existingPopovers.forEach(p => {
                if (p !== popover.popoverWrapper) {
                    p.remove();
                }
            });

            const popoverElement = popover.popoverWrapper;
            if (popoverElement) {
                popoverElement.style.zIndex = '99999';
                popoverElement.style.position = 'fixed';
                popoverElement.style.backgroundColor = 'var(--socratiq-bg, #ffffff)';
                popoverElement.style.color = 'var(--socratiq-text, #1f2328)';
                popoverElement.style.padding = '1rem';
                popoverElement.style.borderRadius = '0.5rem';
                popoverElement.style.boxShadow = '0 0 10px rgba(0,0,0,0.2)';
                popoverElement.style.transition = 'opacity 0.3s ease-in-out';
                popoverElement.style.transformOrigin = 'center center';
                
                // Add viewport boundary check
                const rect = popoverElement.getBoundingClientRect();
                const viewportWidth = window.innerWidth;
                const viewportHeight = window.innerHeight;
                
                // Adjust position if outside viewport
                if (rect.right > viewportWidth) {
                    popoverElement.style.left = `${viewportWidth - rect.width - 20}px`;
                }
                if (rect.bottom > viewportHeight) {
                    popoverElement.style.top = `${viewportHeight - rect.height - 20}px`;
                }
            }
        },
        onBeforeHighlighted: (element) => {
            // Ensure element is in viewport before highlighting
            adjustViewport(element.node);
            return new Promise(resolve => setTimeout(resolve, 100));
        },
        onHighlighted: (element) => {
            // Show new popover smoothly
            if (element.popover && element.popover.popoverWrapper) {
                setTimeout(() => {
                    element.popover.popoverWrapper.style.opacity = '1';
                }, 100);
            }
        },
        onStart: () => {
            localStorage.setItem(ONBOARDING_KEY, 'true');
        },
        onComplete: () => {
            isOnboardingActive = false;
        },
        onClose: () => {
            isOnboardingActive = false;
        },
        onReset: () => {
            isOnboardingActive = false;
        }
    });

    // Helper function to get elements within shadow DOM
    const getElement = (selector) => {
        const element = shadowRoot.querySelector(selector);
        if (element) {
        }
        return element;
    };

    const elements = {
        menu: getElement('#text-selection-menu'),
        newChatBtn: getElement('#new-chat-btn'),
        spacedRepBtn: getElement('#spaced-repetition-btn'),
        knowledgeGraphBtn: getElement('#knowledge-graph-btn'),
        quizBtn: getElement('#chat-quiz-btn'),
        helpBtn: getElement('#help-btn'),
        settingsBtn: getElement('#settings-btn'),
        highlightBtn: getElement('#highlight-btn'),
        srSendBtn: getElement('#sr-send-btn'),
        shareBtn: getElement('#share-btn'),
        copyBtn: getElement('#copy-btn'),
        downloadBtn: getElement('#download-btn'),
        messageContainer: getElement('#message-container'),
        menuToggle: getElement('#toggleButton'),
        enterBtn: getElement('#enter-btn'),
        contextButton: getElement('#context-button')
    };


    const steps = [
        {
            element: elements.menu,
            popover: {
                title: 'Welcome to Socratiq! 👋',
                description: 'This is your chat interface where you can interact with the AI assistant.',
                preferredSide: 'right',
                alignment: 'center',
                offset: {
                    x: Math.min(-200, -(window.innerWidth - 400)),
                    y: Math.min(window.innerHeight / 4, window.innerHeight - 300)
                }
            }
        },
        {
            element: elements.newChatBtn,
            popover: {
                title: 'Start New Chat',
                description: 'Begin a fresh conversation with the AI.',
                preferredSide: 'bottom',
                alignment: 'start'
            }
        },
        {
            element: elements.spacedRepBtn,
            popover: {
                title: 'Spaced Repetition',
                description: 'Access your flashcards and spaced repetition learning.',
                preferredSide: 'bottom',
                alignment: 'start'
            }
        },
        {
            element: elements.knowledgeGraphBtn,
            popover: {
                title: 'Knowledge Graph',
                description: 'Visualize connections between concepts you\'ve learned.',
                preferredSide: 'bottom',
                alignment: 'start'
            }
        },
        {
            element: elements.quizBtn,
            popover: {
                title: 'Quiz Stats',
                description: 'View your quiz performance and earned badges.',
                preferredSide: 'bottom',
                alignment: 'start'
            }
        },
        {
            element: elements.helpBtn,
            popover: {
                title: 'Help',
                description: 'Get assistance with using SocratiQ.',
                preferredSide: 'bottom',
                alignment: 'start'
            }
        },
        {
            element: elements.settingsBtn,
            popover: {
                title: 'Settings',
                description: 'Customize your SocratiQ experience.',
                preferredSide: 'bottom',
                alignment: 'start'
            }
        },
        {
            element: elements.highlightBtn,
            popover: {
                title: 'Search Content',
                description: 'Search the site for related content.',
                preferredSide: 'left',
                alignment: 'end'
            }
        },
        {
            element: elements.srSendBtn,
            popover: {
                title: 'Create Flashcard',
                description: 'Send content to your flashcard deck.',
                preferredSide: 'left',
                alignment: 'end'
            }
        },
        {
            element: elements.shareBtn,
            popover: {
                title: 'Share',
                description: 'Share this conversation with others.',
                preferredSide: 'left',
                alignment: 'end'
            }
        },
        {
            element: elements.copyBtn,
            popover: {
                title: 'Copy',
                description: 'Copy the conversation to clipboard.',
                preferredSide: 'left',
                alignment: 'end'
            }
        },
        {
            element: elements.downloadBtn,
            popover: {
                title: 'Download',
                description: 'Download the conversation.',
                preferredSide: 'left',
                alignment: 'end'
            }
        },
        {
            element: elements.menuToggle,
            popover: {
                title: 'Toggle Menu',
                description: 'Open or close the chat interface. Quick shortcut: Press Cmd/Ctrl + / to toggle.',
                preferredSide: 'right',
                alignment: 'start'
            }
        },
        {
            element: elements.menuToggle,
            popover: {
                title: 'Pro Tip',
                description: 'Highlight any text on the site to quickly send it to the AI or create a flashcard!',
                preferredSide: 'left',
                alignment: 'start'
            },
            overlay:false
        }
    ];

    boarding.defineSteps(steps);

    // Start with a longer delay to ensure everything is ready
    setTimeout(() => {
        try {
            boarding.start();
        } catch (error) {
            console.error("Error starting boarding:", error);
            isOnboardingActive = false;
            // Clean up any remaining popovers
            const popovers = shadowRoot.querySelectorAll('.socratiq-onboarding');
            popovers.forEach(p => p.remove());
        }
    }, 500);
}