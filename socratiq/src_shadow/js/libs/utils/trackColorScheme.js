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
        return null;
    }

    const style = window.getComputedStyle(element);
    const bgColor = style.getPropertyValue('background-color');

    if (!bgColor || bgColor === 'transparent' || bgColor === 'rgba(0, 0, 0, 0)') {
        if (element.parentElement) {
            return getEffectiveBackgroundColor(element.parentElement);
        }
        return '#ffffff';
    }

    return bgColor;
}

function updateColorScheme(shadowEle) {
    let isDarkMode = false;
    let isForced = false;

    // Check for forced theme
    if (typeof window !== 'undefined' && typeof window.SocratiqWidgetTheme === 'string') {
        const preferred = window.SocratiqWidgetTheme.trim().toLowerCase();
        if (preferred === 'dark' || preferred === 'light') {
            isDarkMode = preferred === 'dark';
            isForced = true;
        }
    }

    if (!isForced) {
        // Method 1: Check for Quarto toggle button
        const toggleButton = document.querySelector(".quarto-color-scheme-toggle");
        if (toggleButton && toggleButton.classList.contains("alternate")) {
            isDarkMode = true;
        }

        // Method 2: Check body classes
        if (!isDarkMode) {
            const body = document.body;
            if (body && body.classList.contains("dark-mode")) {
                isDarkMode = true;
            }
        }

        // Method 3: Analyze background color luminance (takes priority over system preference)
        let luminanceResolved = false;
        if (!isDarkMode) {
            const bgColor = getEffectiveBackgroundColor(document.body);
            const bgRgb = parseColorToRgb(bgColor);
            const luminance = getLuminance(bgRgb);
            if (typeof luminance === 'number' && luminance < 0.45) {
                isDarkMode = true;
                luminanceResolved = true;
            } else if (typeof luminance === 'number' && luminance >= 0.45) {
                isDarkMode = false;
                luminanceResolved = true;
            }
        }

        // Method 4: Check system preference (fallback only when luminance is indeterminate)
        if (!luminanceResolved && !isDarkMode && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
            isDarkMode = true;
        }

        // Method 5: Check text color vs background contrast
        if (!luminanceResolved && !isDarkMode) {
            const bodyStyle = window.getComputedStyle(document.body);
            const textColor = bodyStyle.getPropertyValue('color');
            
            // If text is light colored, assume dark background
            const textRgb = parseColorToRgb(textColor);
            if (textRgb) {
                const textLuminance = getLuminance(textRgb);
                if (textLuminance > 0.6) { // Light text suggests dark background
                    isDarkMode = true;
                }
            }
        }
    }

    // Set theme and volume attributes on host element
    const themeValue = isDarkMode ? "dark" : "light";
    const detectVolume = () => {
        const normalize = (v) => {
            const s = String(v || "").trim().toLowerCase();
            return /^vol[1-4]$/.test(s) ? s : "";
        };
        const explicit = normalize(window.SocratiqWidgetVolume) ||
            normalize(document.documentElement?.dataset?.socratiqVolume) ||
            normalize(document.body?.dataset?.socratiqVolume) ||
            normalize(document.querySelector('script[data-volume]')?.dataset?.volume);
        if (explicit) return explicit;
        const themeColor = document.querySelector('meta[name="theme-color"]')?.content?.trim().toLowerCase();
        if (themeColor === '#1f407a') return 'vol2';
        if (themeColor === '#581c87') return 'vol3';
        if (themeColor === '#065f46') return 'vol4';
        if (themeColor === '#a51c30' || themeColor === '#8e1537') return 'vol1';
        const pathMatch = location.pathname.match(/\/(vol[1-4])(\/|$)/i);
        if (pathMatch) return pathMatch[1].toLowerCase();
        return 'vol1';
    };
    const volumeValue = detectVolume();
    
    const hostTarget = shadowEle.host || shadowEle;
    if (hostTarget?.setAttribute) {
        hostTarget.setAttribute("data-socratiq-theme", themeValue);
        hostTarget.setAttribute("data-socratiq-volume", volumeValue);
    }

    const widgetRoot = shadowEle.querySelector?.('.socratiq-widget-root');
    if (widgetRoot?.setAttribute) {
        widgetRoot.setAttribute("data-socratiq-theme", themeValue);
        widgetRoot.setAttribute("data-socratiq-volume", volumeValue);
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

