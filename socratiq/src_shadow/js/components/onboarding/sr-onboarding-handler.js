import { Boarding } from "boarding.js";
import "boarding.js/styles/main.css";
import "boarding.js/styles/themes/basic.css";

const SR_ONBOARDING_KEY = 'socratiq_sr_onboarding_completed';

export class SROnboardingHandler {
    constructor(srModal, shadowRoot) {
        this.srModal = srModal;
        this.shadowRoot = shadowRoot;
        this.boarding = null;
        this.isOnboardingActive = false;
    }

    hasCompletedOnboarding() {
        return localStorage.getItem(SR_ONBOARDING_KEY) !== null;
    }

    getElement(selector) {
        const element = this.shadowRoot.querySelector(selector);
        if (element) {
        }
        return element;
    }

    initializeOnboarding() {
        if (this.hasCompletedOnboarding() || this.isOnboardingActive) {
            return;
        }

        this.isOnboardingActive = true;

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

        const elements = {
            addCardBtn: this.getElement('#showAddCard'),
            reviewBtn: this.getElement('#startReview'),
            statsBtn: this.getElement('#showStats'),
            vizBtn: this.getElement('#showViz'),
            // chapterList: this.getElement('#sr-sidebar')
        };


        this.boarding = new Boarding({
            animate: true,
            opacity: 0.8,
            padding: 10,
            allowClose: true,
            keyboardControl: true,
            showButtons: true,
            className: 'sr-onboarding',
            container: this.shadowRoot,
            root: this.shadowRoot,
            // Reduce animation duration
            animationDuration: 400,
            // Add scroll handling
            scrollIntoViewOptions: {
                behavior: 'smooth',
                block: 'center'
            },
            onBeforeHighlighted: (element) => {
                adjustViewport(element.node);
                return new Promise(resolve => setTimeout(resolve, 100));
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

                    // Remove any existing popovers
                    const existingPopovers = this.shadowRoot.querySelectorAll('.sr-onboarding');
                    existingPopovers.forEach(p => {
                        if (p !== popoverElement) {
                            p.remove();
                        }
                    });
                }
            },
            onComplete: () => this.completeOnboarding(),
            onClose: () => this.completeOnboarding(),
            onReset: () => this.completeOnboarding()
        });

        const steps = [
            {
                element: this.getElement('#sr-modal-content-container-controls'),
                popover: {
                    title: 'Welcome to Spaced Repetition!',
                    description: 'Spaced repetition is a learning technique that helps you remember information more effectively by reviewing cards at increasing intervals. Cards you find difficult will appear more frequently, while those you know well will appear less often. This scientifically-proven method optimizes your learning and long-term retention.',
                    position: 'left',
                    alignment: 'start',
                },
                overlay: false
            },
            // {
            //     element: this.getElement('#sr-sidebar'),
            //     popover: {
            //         title: 'Organize Your Learning',
            //         description: 'Use decks and tags to organize your flashcards. Group related cards together for focused study sessions.',
            //         position: 'right'
            //     }
            // },
            {
                element: this.getElement('#dashboardSearch'),
                popover: {
                    title: 'Quick Search',
                    description: 'Easily find specific cards by searching through your entire collection.',
                    prefferedSide: 'bottom'
                }
            },
            {
                element: this.getElement('#showAddCard'),
                popover: {
                    title: 'Create New Cards',
                    description: 'Click here to add new flashcards to your collection! 📝',
                    prefferedSide: 'left'
                }
            },
            {
                element: this.getElement('#sr-modal-content-container-controls'),
                popover: {
                    title: 'Use Tags',
                    description: 'Organize your cards with #tags - just type # anywhere in your text! 🏷️',
                    prefferedSide: 'bottom'
                }
            },
            {
                element: this.getElement('#sr-modal-content-container'),
                popover: {
                    title: 'Markdown Support',
                    description: 'Style your text with **bold**, *italic*, and `code` using Markdown! ✨',
                    prefferedSide: 'right'
                }
            },
        
            {
                element: this.getElement('#progressBar').parentElement,
                popover: {
                    title: 'Track Your Progress',
                    description: 'Monitor your learning progress with this bar. It shows how many cards you\'ve mastered.',
                    position: 'bottom'
                }
            },
            {
                element: this.getElement('#showStats'),
                popover: {
                    title: 'Learning Statistics',
                    description: 'View detailed statistics about your learning progress and card performance.',
                    prefferedSide: 'bottom'
                }
            },
            {
                element: this.getElement('#showViz'),
                popover: {
                    title: 'Interactive Visualization',
                    description: 'Explore your flashcards in an interactive visual format.',
                    prefferedSide: 'bottom'
                }
            },
           
        ];

        
        const validSteps = steps.filter(step => step.element !== null);
        if (validSteps.length !== steps.length) {
            console.error("Some elements not found, skipping onboarding");
            this.completeOnboarding();
            return;
        }

        this.boarding.defineSteps(validSteps);
    }

    startOnboarding() {
        if (!this.hasCompletedOnboarding() && this.boarding) {
            
            setTimeout(() => {
                try {
                    localStorage.setItem(SR_ONBOARDING_KEY, 'pending');
                    this.boarding.start();
                } catch (error) {
                    console.error("Error starting boarding:", error);
                    this.completeOnboarding();
                    const popovers = this.shadowRoot.querySelectorAll('.sr-onboarding');
                    popovers.forEach(p => p.remove());
                }
            }, 1000);
        }
    }

    completeOnboarding() {
        localStorage.setItem(SR_ONBOARDING_KEY, new Date().toISOString());
        this.isOnboardingActive = false;
        
        window.dispatchEvent(new CustomEvent('sr-onboarding-completed', {
            detail: {
                timestamp: new Date().toISOString()
            }
        }));
    }

    // For testing/development
    resetOnboarding() {
        localStorage.removeItem(SR_ONBOARDING_KEY);
    }
}
