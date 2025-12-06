/**
 * Marimo Badge Integration for TinyTorch
 * Adds Marimo "Open in Marimo" badges to notebook pages
 */

document.addEventListener('DOMContentLoaded', function() {
    // Find all notebook pages (they have launch buttons)
    const launchButtons = document.querySelectorAll('.launch-buttons, .jb-launch-buttons');
    
    if (launchButtons.length === 0) return;
    
    // Add informational message about local setup requirement
    const infoMessage = document.createElement('div');
    infoMessage.className = 'notebook-platform-info';
    infoMessage.style.cssText = `
        margin: 1rem 0;
        padding: 1rem;
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 0.25rem;
        font-size: 0.9rem;
        color: #856404;
    `;
    infoMessage.innerHTML = `
        <strong>💡 Note:</strong> These online notebooks are for <strong>viewing and exploration only</strong>. 
        To actually build modules, run milestone validations, and use the full TinyTorch package, 
        you need <a href="../quickstart-guide.html" style="color: #856404; text-decoration: underline; font-weight: 600;">local setup</a>.
    `;
    
    // Get the current page path to construct marimo URL
    const currentPath = window.location.pathname;
    const notebookName = currentPath.split('/').pop().replace('.html', '');
    
    // Find the repository info from the page
    const repoUrl = 'https://github.com/mlsysbook/TinyTorch';
    const repoPath = 'mlsysbook/TinyTorch';
    const branch = 'main';
    
    // Construct marimo molab URL
    // Marimo can open .ipynb files directly from GitHub
    // Format: https://marimo.app/molab?repo=owner/repo&path=path/to/file.ipynb
    // Works for all modules: 01_tensor, 02_activations, etc.
    const marimoUrl = `https://marimo.app/molab?repo=${repoPath}&path=docs/chapters/modules/${notebookName}.ipynb`;
    
    // Create marimo badge
    const marimoBadge = document.createElement('div');
    marimoBadge.className = 'marimo-launch-badge';
    marimoBadge.style.cssText = `
        margin-top: 1rem;
        padding: 0.75rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 0.5rem;
        text-align: center;
    `;
    
    const marimoLink = document.createElement('a');
    marimoLink.href = marimoUrl;
    marimoLink.target = '_blank';
    marimoLink.rel = 'noopener noreferrer';
    marimoLink.style.cssText = `
        color: white;
        text-decoration: none;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    `;
    marimoLink.innerHTML = `
        <span>🍃</span>
        <span>Open in Marimo</span>
        <span style="font-size: 0.85em;">→</span>
    `;
    
    marimoBadge.appendChild(marimoLink);
    
    // Add info message and marimo badge after launch buttons
    launchButtons.forEach(buttonContainer => {
        // Add info message first (if not already present)
        if (!buttonContainer.querySelector('.notebook-platform-info')) {
            buttonContainer.appendChild(infoMessage.cloneNode(true));
        }
        
        // Check if marimo badge already exists
        if (!buttonContainer.querySelector('.marimo-launch-badge')) {
            buttonContainer.appendChild(marimoBadge.cloneNode(true));
        }
    });
    
    // Also add to any existing launch button sections
    const launchSections = document.querySelectorAll('[class*="launch"], [id*="launch"]');
    launchSections.forEach(section => {
        // Add info message if not present
        if (!section.querySelector('.notebook-platform-info')) {
            const infoClone = infoMessage.cloneNode(true);
            infoClone.style.marginTop = '1rem';
            section.appendChild(infoClone);
        }
        
        // Add marimo badge if not present
        if (!section.querySelector('.marimo-launch-badge')) {
            const badgeClone = marimoBadge.cloneNode(true);
            badgeClone.style.marginTop = '1rem';
            section.appendChild(badgeClone);
        }
    });
});

