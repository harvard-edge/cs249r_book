let tooltipTemplate;
let shadowEl;

// Helper to inject tooltip styles into shadow root
function ensureTooltipStyles(shadowRoot) {
    if (!shadowRoot.querySelector('#tooltip-styles-socratiQ')) {
        const style = document.createElement('style');
        style.id = 'tooltip-styles-socratiQ';
        style.textContent = `
            /* Tooltip styles */
            [data-tooltip-text-socratiQ] {
                position: relative;
                cursor: pointer;
            }
            
            .tooltip-container-socratiQ {
                position: fixed;
                z-index: 100000;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.2s;
                background-color: #333;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                white-space: nowrap;
            }

            /* Remove focus outline */
            button {
                outline: none !important;
                box-shadow: none !important;
            }

            button:focus {
                outline: none !important;
                box-shadow: none !important;
            }

            /* Optional: Add a subtle hover effect instead */
            button:hover {
                opacity: 0.8;
            }
        `;
        shadowRoot.appendChild(style);
    }
}

function createTooltipTemplate(shadowRoot) {
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip-container-socratiQ';
    tooltip.style.cssText = `
        position: fixed;
        z-index: 100000;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.2s;
        background-color: #333;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        white-space: nowrap;
        display: none;
    `;
    
    // Append to document.body instead of shadowRoot
    document.body.appendChild(tooltip);
    return tooltip;
}

function positionTooltip(tooltip, element) {
    const elementRect = element.getBoundingClientRect();
    
    // Position above the element
    tooltip.style.left = `${elementRect.left + (elementRect.width / 2)}px`;
    tooltip.style.top = `${elementRect.top - 10}px`;
    tooltip.style.transform = 'translate(-50%, -100%)';
}

function handleMouseEnter(event) {
    const element = event.target;
    const tooltipText = element.getAttribute('data-tooltip-text-socratiQ');
    if (!tooltipText) return;

    let tooltip = document.querySelector('.tooltip-container-socratiQ');
    if (!tooltip) {
        tooltip = createTooltipTemplate(shadowEl);
    }

    tooltip.textContent = tooltipText;
    tooltip.style.display = 'block';
    
    requestAnimationFrame(() => {
        positionTooltip(tooltip, element);
        tooltip.style.opacity = '1';
    });
}

function handleMouseLeave() {
    // Look for tooltip in document.body
    const tooltip = document.querySelector('.tooltip-container-socratiQ');
    if (tooltip) {
        tooltip.style.opacity = '0';
        setTimeout(() => {
            tooltip.style.display = 'none';
        }, 200);
    }
}

export function enableTooltip(element, text, shadowRoot) {

    if(!element) {
        return;
    }

    if (!shadowEl) {
        shadowEl = shadowRoot;
        ensureTooltipStyles(shadowRoot);
    }


    
    // Clean up existing listeners if any
    element.removeEventListener('mouseenter', handleMouseEnter);
    element.removeEventListener('mouseleave', handleMouseLeave);
    
    // Set tooltip text and add listeners
    element.setAttribute('data-tooltip-text-socratiQ', text);
    element.addEventListener('mouseenter', handleMouseEnter);
    element.addEventListener('mouseleave', handleMouseLeave);
    
    element.style.cursor = 'pointer';
}

// Function to initialize tooltips for all elements with data-tooltip-text
export function initializeTooltips(shadowRoot) {
    if (!shadowEl) {
        shadowEl = shadowRoot;
        ensureTooltipStyles(shadowRoot);
    }
    
    // Find all elements with data-tooltip-text attribute
    const tooltipElements = shadowRoot.querySelectorAll('[data-tooltip-text-socratiQ]');
    
    tooltipElements.forEach(element => {
        const text = element.getAttribute('data-tooltip-text-socratiQ');
        enableTooltip(element, text, shadowRoot);
    });
}