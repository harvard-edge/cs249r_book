function parseColorToRgb(color) {
    if (!color) {
        return null;
    }

    const trimmed = color.trim().toLowerCase();

    if (trimmed.startsWith('#')) {
        const hex = trimmed.slice(1);
        const normalized = hex.length === 3
            ? hex.split('').map((ch) => ch + ch).join('')
            : hex;
        if (normalized.length === 6) {
            const r = parseInt(normalized.slice(0, 2), 16);
            const g = parseInt(normalized.slice(2, 4), 16);
            const b = parseInt(normalized.slice(4, 6), 16);
            return { r, g, b };
        }
    }

    const rgbMatch = trimmed.match(/rgba?\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
    if (rgbMatch) {
        return {
            r: parseInt(rgbMatch[1], 10),
            g: parseInt(rgbMatch[2], 10),
            b: parseInt(rgbMatch[3], 10)
        };
    }

    return null;
}

function srgbToLinear(channel) {
    const normalized = channel / 255;
    return normalized <= 0.04045
        ? normalized / 12.92
        : Math.pow((normalized + 0.055) / 1.055, 2.4);
}

function getLuminance(rgb) {
    if (!rgb) {
        return null;
    }

    const r = srgbToLinear(rgb.r);
    const g = srgbToLinear(rgb.g);
    const b = srgbToLinear(rgb.b);

    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function getEffectiveBackgroundColor(element) {
    if (!element) {
        console.log('[THEME DEBUG] getEffectiveBackgroundColor: No element provided');
        return null;
    }

    const style = window.getComputedStyle(element);
    const bgColor = style.getPropertyValue('background-color');
    console.log('[THEME DEBUG] getEffectiveBackgroundColor: Element:', element.tagName, 'Background color:', bgColor);

    if (!bgColor || bgColor === 'transparent' || bgColor === 'rgba(0, 0, 0, 0)') {
        console.log('[THEME DEBUG] getEffectiveBackgroundColor: Background is transparent, checking parent');
        if (element.parentElement) {
            return getEffectiveBackgroundColor(element.parentElement);
        }
        console.log('[THEME DEBUG] getEffectiveBackgroundColor: No parent, defaulting to white');
        return '#ffffff';
    }

    console.log('[THEME DEBUG] getEffectiveBackgroundColor: Returning:', bgColor);
    return bgColor;
}

function updateColorScheme(shadowEle) {
    console.log('[THEME DEBUG] Starting theme detection...');
    let isDarkMode = false;
    let isForced = false;

    // Check for forced theme
    if (typeof window !== 'undefined' && typeof window.SocratiqWidgetTheme === 'string') {
        const preferred = window.SocratiqWidgetTheme.trim().toLowerCase();
        console.log('[THEME DEBUG] Forced theme detected:', preferred);
        if (preferred === 'dark' || preferred === 'light') {
            isDarkMode = preferred === 'dark';
            isForced = true;
        }
    }

    if (!isForced) {
        console.log('[THEME DEBUG] No forced theme, checking automatic detection...');
        
        // Method 1: Check for Quarto toggle button
        const toggleButton = document.querySelector(".quarto-color-scheme-toggle");
        if (toggleButton && toggleButton.classList.contains("alternate")) {
            console.log('[THEME DEBUG] Quarto toggle button found with alternate class');
            isDarkMode = true;
        }

        // Method 2: Check body classes
        if (!isDarkMode) {
            const body = document.body;
            console.log('[THEME DEBUG] Body classes:', body.className);
            if (body && body.classList.contains("dark-mode")) {
                console.log('[THEME DEBUG] Body has dark-mode class');
                isDarkMode = true;
            }
        }

        // Method 3: Analyze background color luminance (takes priority over system preference)
        let luminanceResolved = false;
        if (!isDarkMode) {
            const bgColor = getEffectiveBackgroundColor(document.body);
            console.log('[THEME DEBUG] Computed background color:', bgColor);
            const bgRgb = parseColorToRgb(bgColor);
            console.log('[THEME DEBUG] Parsed RGB:', bgRgb);
            const luminance = getLuminance(bgRgb);
            console.log('[THEME DEBUG] Calculated luminance:', luminance);
            if (typeof luminance === 'number' && luminance < 0.45) {
                console.log('[THEME DEBUG] Low luminance detected, switching to dark mode');
                isDarkMode = true;
                luminanceResolved = true;
            } else if (typeof luminance === 'number' && luminance >= 0.45) {
                console.log('[THEME DEBUG] High luminance detected, light mode');
                isDarkMode = false;
                luminanceResolved = true;
            }
        }

        // Method 4: Check system preference (fallback only when luminance is indeterminate)
        if (!luminanceResolved && !isDarkMode && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
            console.log('[THEME DEBUG] System prefers dark color scheme');
            isDarkMode = true;
        }

        // Method 5: Check text color vs background contrast
        if (!luminanceResolved && !isDarkMode) {
            const bodyStyle = window.getComputedStyle(document.body);
            const textColor = bodyStyle.getPropertyValue('color');
            console.log('[THEME DEBUG] Body text color:', textColor);
            
            // If text is light colored, assume dark background
            const textRgb = parseColorToRgb(textColor);
            if (textRgb) {
                const textLuminance = getLuminance(textRgb);
                console.log('[THEME DEBUG] Text luminance:', textLuminance);
                if (textLuminance > 0.6) { // Light text suggests dark background
                    console.log('[THEME DEBUG] Light text detected, assuming dark background');
                    isDarkMode = true;
                }
            }
        }
    }

    console.log('[THEME DEBUG] Final isDarkMode:', isDarkMode);

    // Set theme attribute on host element
    const themeValue = isDarkMode ? "dark" : "light";
    console.log('[THEME DEBUG] Setting theme value to:', themeValue);
    
    const hostTarget = shadowEle.host || shadowEle;
    if (hostTarget?.setAttribute) {
        hostTarget.setAttribute("data-socratiq-theme", themeValue);
        console.log('[THEME DEBUG] Set host element theme attribute:', hostTarget.getAttribute("data-socratiq-theme"));
        
        // Log computed styles to debug CSS variables
        const computedStyle = window.getComputedStyle(hostTarget);
        console.log('[THEME DEBUG] Host computed styles:', {
            color: computedStyle.color,
            backgroundColor: computedStyle.backgroundColor,
            '--socratiq-text': computedStyle.getPropertyValue('--socratiq-text'),
            '--socratiq-bg': computedStyle.getPropertyValue('--socratiq-bg')
        });
    }

    const widgetRoot = shadowEle.querySelector?.('.socratiq-widget-root');
    if (widgetRoot?.setAttribute) {
        widgetRoot.setAttribute("data-socratiq-theme", themeValue);
        console.log('[THEME DEBUG] Set widget root theme attribute:', widgetRoot.getAttribute("data-socratiq-theme"));
        
        // Log computed styles for widget root
        const computedStyle = window.getComputedStyle(widgetRoot);
        console.log('[THEME DEBUG] Widget root computed styles:', {
            color: computedStyle.color,
            backgroundColor: computedStyle.backgroundColor,
            '--socratiq-text': computedStyle.getPropertyValue('--socratiq-text'),
            '--socratiq-bg': computedStyle.getPropertyValue('--socratiq-bg')
        });
    }
    
    // Check if modal exists and log its styles
    const modal = shadowEle.querySelector?.('#spacedRepetitionModal');
    if (modal) {
        const modalStyle = window.getComputedStyle(modal);
        console.log('[THEME DEBUG] Modal computed styles:', {
            color: modalStyle.color,
            backgroundColor: modalStyle.backgroundColor,
            '--socratiq-text': modalStyle.getPropertyValue('--socratiq-text'),
            '--socratiq-bg': modalStyle.getPropertyValue('--socratiq-bg')
        });
    }
    
    // Helper function to apply or remove dark-mode class
    function applyDarkMode(element) {
        if (isDarkMode) {
            element.classList.add("dark-mode");
        } else {
            element.classList.remove("dark-mode");
        }
    }

    // List of component IDs to be updated
    const componentIds = [
        "text-selection-menu",
        "text-selection-menu-highlight",
        "popover-container",
        "modal1",
        "modal_feedback",
        "cardReviewModal"
    ];

    // Iterate over each component ID and apply dark mode to the container only
    componentIds.forEach(id => {
        const targetElement = shadowEle.getElementById(id);
        if (targetElement) {
            applyDarkMode(targetElement);
        }
    });
}


export function trackColorScheme(shadowEle) {
    console.log('[THEME DEBUG] trackColorScheme function called with shadowEle:', shadowEle);
    // Initial color scheme detection
    updateColorScheme(shadowEle);

    // Observe changes in the class attribute of the body element
    const bodyObserver = new MutationObserver((mutationsList) => {
        for (const mutation of mutationsList) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                updateColorScheme(shadowEle);
            }
        }
    });

    bodyObserver.observe(document.body, {
        attributes: true, // Listen to attribute changes
        attributeFilter: ['class'], // Specifically for class attribute
    });

    // Listen for system color scheme changes
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
        updateColorScheme(shadowEle);
    });
}

