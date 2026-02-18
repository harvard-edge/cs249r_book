/**
 * TinyTorch Version Badge
 * 
 * Adds a version badge to the footer that links to the GitHub releases page.
 * Version is read from a data attribute or defaults to checking package version.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Try to get version from meta tag first (set during build)
    const versionMeta = document.querySelector('meta[name="tinytorch-version"]');
    const version = versionMeta ? versionMeta.content : '0.1.9';
    
    // Find the footer
    const footer = document.querySelector('.footer');
    if (!footer) return;
    
    // Create version badge container
    const versionContainer = document.createElement('div');
    versionContainer.className = 'tinytorch-version-badge';
    versionContainer.innerHTML = `
        <a href="https://github.com/harvard-edge/cs249r_book/releases" 
           target="_blank" 
           rel="noopener noreferrer"
           title="View all TinyTorch releases">
            <span class="version-label">v${version}</span>
        </a>
    `;
    
    // Add to footer
    footer.appendChild(versionContainer);
});
