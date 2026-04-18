// let tooltipTemplate;
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
    const tooltipRect = tooltip.getBoundingClientRect();
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;
    
    // Default position (above the element)
    let top = elementRect.top - 10;
    let left = elementRect.left + (elementRect.width / 2);
    let transform = 'translate(-50%, -100%)';
    
    // Check if tooltip would go above viewport
    if (top - tooltipRect.height < 0) {
        // Position below element instead
        top = elementRect.bottom + 10;
        transform = 'translate(-50%, 0)';
    }
    
    // Check if tooltip would go beyond right edge
    if (left + (tooltipRect.width / 2) > viewportWidth) {
        left = viewportWidth - 10;
        transform = `translate(-100%, ${top - tooltipRect.height < 0 ? '0' : '-100%'})`;
    }
    
    // Check if tooltip would go beyond left edge
    if (left - (tooltipRect.width / 2) < 0) {
        left = 10;
        transform = `translate(0, ${top - tooltipRect.height < 0 ? '0' : '-100%'})`;
    }
    
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
    tooltip.style.transform = transform;
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
        // console.warn("No element provided to enableTooltip");
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
    
    const tooltipElements = shadowRoot.querySelectorAll('[data-tooltip-text-socratiQ]');
    
    tooltipElements.forEach(element => {
        const text = element.getAttribute('data-tooltip-text-socratiQ');
        enableTooltip(element, text, shadowRoot);
    });
}