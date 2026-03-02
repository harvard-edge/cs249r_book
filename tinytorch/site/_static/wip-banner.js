/**
 * TinyTorch Top Bar
 * Elegant navigation bar matching MLSysBook style
 */

// ── Release info (auto-updated by CI on publish) ──────────────
const TINYTORCH_VERSION = '0.1.9';
const TINYTORCH_RELEASE_DATE = 'Feb 18, 2026';
// ───────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function() {
    // Only inject if not already present
    if (document.getElementById('tinytorch-bar')) return;

    // Calculate base path for relative URLs based on current page location
    // If we're in a subdirectory (modules/, tiers/, tito/), we need "../" to reach root
    const path = window.location.pathname;
    const tinytorchIndex = path.indexOf('/tinytorch/');
    let siteRoot = '';
    if (tinytorchIndex !== -1) {
        // Get the path after /tinytorch/
        const subPath = path.substring(tinytorchIndex + '/tinytorch/'.length);
        // Count directory levels (exclude the .html file itself)
        const dirs = subPath.split('/').filter(p => p && !p.endsWith('.html'));
        siteRoot = '../'.repeat(dirs.length);
    }

    const barHTML = `
        <div class="tinytorch-bar" id="tinytorch-bar">
            <div class="tinytorch-bar-content">
                <div class="tinytorch-bar-left">
                    <a href="${siteRoot}intro.html" class="tinytorch-bar-brand">
                        Tiny<span class="brand-fire">🔥</span>Torch
                    </a>
                    <a href="https://github.com/harvard-edge/cs249r_book/releases" target="_blank" rel="noopener noreferrer" class="tinytorch-bar-version" title="View releases on GitHub"><span class="version-number">v${TINYTORCH_VERSION}</span><span class="version-date"> · ${TINYTORCH_RELEASE_DATE}</span></a>
                    <span class="tinytorch-bar-badge">Under Construction</span>
                </div>
                <div class="tinytorch-bar-links">
                    <a href="${siteRoot}_static/downloads/TinyTorch-Guide.pdf" class="download-link" title="Download Course Guide (PDF)">
                        <span class="link-icon">↓</span>
                        <span class="link-text">Guide</span>
                    </a>
                    <a href="https://arxiv.org/abs/2601.19107" target="_blank" class="download-link" title="Read Research Paper on arXiv">
                        <span class="link-icon">↗</span>
                        <span class="link-text">Paper</span>
                    </a>
                    <a href="https://mlsysbook.ai" target="_blank" class="link-secondary">
                        <span class="link-icon">📖</span>
                        <span class="link-text">MLSysBook</span>
                    </a>
                    <a href="#" class="subscribe-trigger link-secondary" onclick="event.preventDefault(); if(window.openSubscribeModal) openSubscribeModal();">
                        <span class="link-icon">✉️</span>
                        <span class="link-text">Subscribe</span>
                    </a>
                    <a href="https://github.com/harvard-edge/cs249r_book" target="_blank">
                        <span class="link-icon">⭐</span>
                        <span class="link-text">Star</span>
                    </a>
                    <a href="${siteRoot}community.html" target="_blank" class="link-secondary">
                        <span class="link-icon">🌍</span>
                        <span class="link-text">Community</span>
                    </a>
                </div>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('afterbegin', barHTML);

    // Smart sticky: hide on scroll down, show on scroll up
    const bar = document.getElementById('tinytorch-bar');
    let lastScrollY = window.scrollY;
    let ticking = false;

    function updateBar() {
        const currentScrollY = window.scrollY;

        if (currentScrollY < 50) {
            // Always show at top of page
            bar.classList.remove('hidden');
        } else if (currentScrollY > lastScrollY) {
            // Scrolling down - hide
            bar.classList.add('hidden');
        } else {
            // Scrolling up - show
            bar.classList.remove('hidden');
        }

        lastScrollY = currentScrollY;
        ticking = false;
    }

    window.addEventListener('scroll', function() {
        if (!ticking) {
            requestAnimationFrame(updateBar);
            ticking = true;
        }
    }, { passive: true });
});
