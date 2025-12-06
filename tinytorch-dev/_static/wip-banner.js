/**
 * TinyTorch Top Bar
 * Elegant navigation bar matching MLSysBook style
 */

document.addEventListener('DOMContentLoaded', function() {
    // Only inject if not already present
    if (document.getElementById('tinytorch-bar')) return;
    
    const barHTML = `
        <div class="tinytorch-bar" id="tinytorch-bar">
            <div class="tinytorch-bar-content">
                <div class="tinytorch-bar-left">
                    <a href="intro.html" class="tinytorch-bar-brand">
                        <span class="brand-fire">🔥</span>Tiny<span class="brand-fire">🔥</span>Torch
                    </a>
                    <span class="tinytorch-bar-badge">Under Construction</span>
                </div>
                <div class="tinytorch-bar-links">
                    <a href="https://mlsysbook.ai" target="_blank">
                        <span class="link-icon">📖</span>
                        <span class="link-text">MLSysBook</span>
                    </a>
                    <a href="#" class="subscribe-trigger" onclick="event.preventDefault(); if(window.openSubscribeModal) openSubscribeModal();">
                        <span class="link-icon">✉️</span>
                        <span class="link-text">Subscribe</span>
                    </a>
                    <a href="https://github.com/mlsysbook/TinyTorch" target="_blank">
                        <span class="link-icon">⭐</span>
                        <span class="link-text">Star</span>
                    </a>
                    <a href="https://tinytorch.ai/join" target="_blank">
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
