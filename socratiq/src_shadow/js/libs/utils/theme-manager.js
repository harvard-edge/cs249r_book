/**
 * Theme Manager - Centralized theme detection and management
 * Provides dynamic theme detection and utilities for theme-aware styling
 */

class ThemeManager {
    constructor(shadowRoot) {
        this.shadowRoot = shadowRoot;
        this.hostElement = shadowRoot.host;
        this.currentTheme = 'light';
        this.themeChangeCallbacks = new Set();
        this.initializeTheme();
        this.setupThemeObserver();
    }

    /**
     * Initialize theme detection
     */
    initializeTheme() {
        this.currentTheme = this.detectTheme();
        this.applyTheme();
    }

    /**
     * Detect current theme using multiple methods
     */
    detectTheme() {
        // Check for forced theme
        if (typeof window !== 'undefined' && typeof window.SocratiqWidgetTheme === 'string') {
            const preferred = window.SocratiqWidgetTheme.trim().toLowerCase();
            if (preferred === 'dark' || preferred === 'light') {
                return preferred;
            }
        }

        // Method 1: Check for Quarto toggle button
        const toggleButton = document.querySelector(".quarto-color-scheme-toggle");
        if (toggleButton && toggleButton.classList.contains("alternate")) {
            return 'dark';
        }

        // Method 2: Check body classes
        const body = document.body;
        if (body && body.classList.contains("dark-mode")) {
            return 'dark';
        }

        // Method 3: Analyze actual page background luminance (takes priority over system preference)
        const bgColor = this.getEffectiveBackgroundColor(document.body);
        const bgRgb = this.parseColorToRgb(bgColor);
        const luminance = this.getLuminance(bgRgb);
        if (typeof luminance === 'number' && luminance < 0.45) {
            return 'dark';
        }
        if (typeof luminance === 'number' && luminance >= 0.45) {
            return 'light';
        }

        // Method 4: Check text color vs background contrast
        const bodyStyle = window.getComputedStyle(document.body);
        const textColor = bodyStyle.getPropertyValue('color');
        const textRgb = this.parseColorToRgb(textColor);
        if (textRgb) {
            const textLuminance = this.getLuminance(textRgb);
            if (textLuminance > 0.6) { // Light text suggests dark background
                return 'dark';
            }
        }

        // Method 5: Fall back to system preference only when page bg is indeterminate
        if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
            return 'dark';
        }

        return 'light';
    }

    /**
     * Apply theme to host element and widget root
     */
    applyTheme() {
        if (this.hostElement?.setAttribute) {
            this.hostElement.setAttribute("data-socratiq-theme", this.currentTheme);
        }

        const widgetRoot = this.shadowRoot.querySelector?.('.socratiq-widget-root');
        if (widgetRoot?.setAttribute) {
            widgetRoot.setAttribute("data-socratiq-theme", this.currentTheme);
        } else {
            // If .socratiq-widget-root doesn't exist, try to find the main container
            const mainContainer = this.shadowRoot.querySelector('#widget-chat-container') || this.shadowRoot.querySelector('#collaborative-widget-host');
            if (mainContainer?.setAttribute) {
                mainContainer.setAttribute("data-socratiq-theme", this.currentTheme);
            }
        }

        // Notify all callbacks
        this.themeChangeCallbacks.forEach(callback => {
            try {
                callback(this.currentTheme);
            } catch (error) {
                console.error('[ThemeManager] Error in theme change callback:', error);
            }
        });
    }

    /**
     * Setup theme change observers
     */
    setupThemeObserver() {
        // Observe changes in the class attribute of the body element
        const bodyObserver = new MutationObserver((mutationsList) => {
            for (const mutation of mutationsList) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    const newTheme = this.detectTheme();
                    if (newTheme !== this.currentTheme) {
                        this.currentTheme = newTheme;
                        this.applyTheme();
                    }
                }
            }
        });

        bodyObserver.observe(document.body, {
            attributes: true,
            attributeFilter: ['class'],
        });

        // Listen for system color scheme changes
        if (window.matchMedia) {
            window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
                const newTheme = this.detectTheme();
                if (newTheme !== this.currentTheme) {
                    this.currentTheme = newTheme;
                    this.applyTheme();
                }
            });
        }
    }

    /**
     * Get current theme
     */
    getCurrentTheme() {
        return this.currentTheme;
    }

    /**
     * Check if current theme is dark
     */
    isDark() {
        return this.currentTheme === 'dark';
    }

    /**
     * Check if current theme is light
     */
    isLight() {
        return this.currentTheme === 'light';
    }

    /**
     * Register a callback for theme changes
     */
    onThemeChange(callback) {
        this.themeChangeCallbacks.add(callback);
        return () => this.themeChangeCallbacks.delete(callback);
    }

    /**
     * Get theme-aware CSS classes
     */
    getThemeClasses() {
        const isDark = this.isDark();
        return {
            // Backgrounds
            bgPrimary: isDark ? 'bg-zinc-800' : 'bg-white',
            bgSecondary: isDark ? 'bg-zinc-700/50' : 'bg-gray-50',
            bgHover: isDark ? 'hover:bg-zinc-600' : 'hover:bg-gray-200',
            bgButton: isDark ? 'bg-zinc-700' : 'bg-gray-100',
            bgButtonHover: isDark ? 'hover:bg-zinc-600' : 'hover:bg-gray-200',
            bgAccent: isDark ? 'bg-blue-900/30' : 'bg-blue-100',
            bgCard: isDark ? 'bg-zinc-700' : 'bg-white',
            
            // Text colors
            textPrimary: isDark ? 'text-gray-100' : 'text-gray-900',
            textSecondary: isDark ? 'text-gray-300' : 'text-gray-600',
            textMuted: isDark ? 'text-gray-400' : 'text-gray-500',
            textAccent: isDark ? 'text-blue-400' : 'text-blue-700',
            
            // Borders
            borderPrimary: isDark ? 'border-zinc-600' : 'border-gray-300',
            borderSecondary: isDark ? 'border-zinc-700' : 'border-gray-200',
            
            // Input fields
            inputBg: isDark ? 'bg-zinc-700/50' : 'bg-white',
            inputBorder: isDark ? 'border-zinc-600' : 'border-gray-300',
            
            // KBD styling
            kbdBg: isDark ? 'bg-gray-600' : 'bg-gray-100',
            kbdBorder: isDark ? 'border-gray-500' : 'border-gray-200',
            kbdText: isDark ? 'text-gray-200' : 'text-gray-700',
            
            // Button states
            buttonActive: isDark ? 'bg-blue-800/50 text-blue-300 border border-blue-600/30' : 'bg-blue-100 text-blue-800 border border-blue-200',
            buttonHover: isDark ? 'hover:bg-zinc-700' : 'hover:bg-gray-100',
            
            // Special elements
            deleteHover: isDark ? 'hover:bg-zinc-600' : 'hover:bg-gray-100',
            deleteText: isDark ? 'hover:text-red-400' : 'hover:text-red-500'
        };
    }

    /**
     * Get theme-aware inline styles
     */
    getThemeStyles() {
        const isDark = this.isDark();
        return {
            bgPrimary: isDark ? '#0d1117' : '#ffffff',
            textPrimary: isDark ? '#e6edf3' : '#1f2328',
            bgSecondary: isDark ? '#21262d' : '#f8f9fa',
            borderPrimary: isDark ? '#30363d' : '#d0d7de'
        };
    }

    /**
     * Apply theme classes to an element
     */
    applyThemeToElement(element, classMap) {
        if (!element) return;
        
        const themeClasses = this.getThemeClasses();
        Object.entries(classMap).forEach(([key, classes]) => {
            if (themeClasses[key]) {
                element.classList.add(themeClasses[key]);
            }
        });
    }

    /**
     * Update dynamically created elements with current theme
     */
    updateDynamicElements() {
        // Update all KBD elements
        this.shadowRoot.querySelectorAll('kbd, [data-theme-kbd]').forEach(kbd => {
            const themeClasses = this.getThemeClasses();
            kbd.className = `px-2 py-1 text-xs transition-colors duration-150 ${themeClasses.kbdBg} ${themeClasses.kbdBorder} ${themeClasses.kbdText} border`;
        });

        // Update all buttons with hardcoded classes
        this.shadowRoot.querySelectorAll('button').forEach(button => {
            this.updateButtonTheme(button);
        });

        // Update all dynamically created deck buttons
        this.shadowRoot.querySelectorAll('[data-chapter]').forEach(button => {
            this.updateDeckButtonTheme(button);
        });
    }

    /**
     * Update button theme classes
     */
    updateButtonTheme(button) {
        const themeClasses = this.getThemeClasses();
        const isActive = button.classList.contains('bg-blue-50') || button.classList.contains('dark:bg-blue-900/30');
        
        // Remove old theme classes
        button.classList.remove(
            'bg-gray-100', 'dark:bg-zinc-700',
            'bg-blue-50', 'dark:bg-blue-900/30',
            'hover:bg-gray-100', 'dark:hover:bg-zinc-700'
        );
        
        // Add appropriate theme classes
        if (isActive) {
            button.classList.add(themeClasses.buttonActive);
        } else {
            button.classList.add(themeClasses.buttonHover);
        }
    }

    /**
     * Update deck button theme classes
     */
    updateDeckButtonTheme(button) {
        const themeClasses = this.getThemeClasses();
        const isActive = button.classList.contains('bg-blue-50') || button.classList.contains('dark:bg-blue-900/30');
        
        // Remove old theme classes
        button.classList.remove(
            'bg-gray-100', 'dark:bg-zinc-700',
            'bg-blue-50', 'dark:bg-blue-900/30',
            'hover:bg-gray-100', 'dark:hover:bg-zinc-700'
        );
        
        // Add appropriate theme classes
        if (isActive) {
            button.classList.add(themeClasses.buttonActive);
        } else {
            button.classList.add(themeClasses.buttonHover);
        }
    }

    // Helper methods for color analysis
    parseColorToRgb(color) {
        if (!color) return null;

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

    srgbToLinear(channel) {
        const normalized = channel / 255;
        return normalized <= 0.04045
            ? normalized / 12.92
            : Math.pow((normalized + 0.055) / 1.055, 2.4);
    }

    getLuminance(rgb) {
        if (!rgb) return null;

        const r = this.srgbToLinear(rgb.r);
        const g = this.srgbToLinear(rgb.g);
        const b = this.srgbToLinear(rgb.b);

        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    getEffectiveBackgroundColor(element) {
        if (!element) return null;

        const style = window.getComputedStyle(element);
        const bgColor = style.getPropertyValue('background-color');

        if (!bgColor || bgColor === 'transparent' || bgColor === 'rgba(0, 0, 0, 0)') {
            if (element.parentElement) {
                return this.getEffectiveBackgroundColor(element.parentElement);
            }
            return '#ffffff';
        }

        return bgColor;
    }
}

// Export singleton instance
let themeManagerInstance = null;

export function getThemeManager(shadowRoot) {
    if (!themeManagerInstance) {
        themeManagerInstance = new ThemeManager(shadowRoot);
    }
    return themeManagerInstance;
}

export function createThemeManager(shadowRoot) {
    return new ThemeManager(shadowRoot);
}

export { ThemeManager };