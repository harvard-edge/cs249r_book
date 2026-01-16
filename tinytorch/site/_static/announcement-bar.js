/**
 * TinyTorch Announcement Bar
 * 
 * Displays a dismissible announcement bar at the top of the page.
 * Configuration is loaded from announcement.json.
 * 
 * Features:
 * - Multiple announcement items supported
 * - Dismissible (remembers via localStorage)
 * - Auto-updated by publish workflow on releases
 */

(function() {
    'use strict';

    const CONFIG_PATH = '_static/announcement.json';
    const STORAGE_PREFIX = 'tinytorch-announcement-dismissed-';

    async function loadConfig() {
        try {
            const response = await fetch(CONFIG_PATH);
            if (!response.ok) return null;
            return await response.json();
        } catch (e) {
            console.warn('Announcement config not found:', e);
            return null;
        }
    }

    function isDismissed(dismissId) {
        return localStorage.getItem(STORAGE_PREFIX + dismissId) === 'true';
    }

    function setDismissed(dismissId) {
        localStorage.setItem(STORAGE_PREFIX + dismissId, 'true');
    }

    function createAnnouncementBar(config) {
        if (!config.enabled || !config.items || config.items.length === 0) return null;
        if (isDismissed(config.dismissId)) return null;

        const bar = document.createElement('div');
        bar.className = 'tinytorch-announcement-bar';
        bar.setAttribute('role', 'banner');
        bar.setAttribute('aria-label', 'Announcements');

        // Create content container
        const content = document.createElement('div');
        content.className = 'announcement-content';

        // Add each announcement item
        config.items.forEach((item, index) => {
            const itemEl = document.createElement('div');
            itemEl.className = 'announcement-item';
            
            const icon = item.icon ? `<span class="announcement-icon">${item.icon}</span>` : '';
            const link = item.link ? `<a href="${item.link}" class="announcement-link">${item.linkText || 'Learn more →'}</a>` : '';
            
            itemEl.innerHTML = `${icon}<span class="announcement-text">${item.text}</span>${link}`;
            content.appendChild(itemEl);
        });

        bar.appendChild(content);

        // Add dismiss button
        const dismissBtn = document.createElement('button');
        dismissBtn.className = 'announcement-dismiss';
        dismissBtn.setAttribute('aria-label', 'Dismiss announcement');
        dismissBtn.innerHTML = '✕';
        dismissBtn.addEventListener('click', () => {
            setDismissed(config.dismissId);
            bar.classList.add('dismissed');
            document.body.classList.remove('has-announcement');
            setTimeout(() => bar.remove(), 300);
        });
        bar.appendChild(dismissBtn);

        return bar;
    }

    function insertBar(bar) {
        if (!bar) return;
        
        // Insert at the very top of the page
        const body = document.body;
        body.insertBefore(bar, body.firstChild);
        
        // Add class to body for CSS adjustments
        body.classList.add('has-announcement');
    }

    // Initialize when DOM is ready
    document.addEventListener('DOMContentLoaded', async () => {
        const config = await loadConfig();
        if (config) {
            const bar = createAnnouncementBar(config);
            insertBar(bar);
        }
    });
})();
