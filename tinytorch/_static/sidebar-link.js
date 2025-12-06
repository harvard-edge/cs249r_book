// Add permanent textbook link to sidebar on all pages
document.addEventListener('DOMContentLoaded', function() {
    // Find the sidebar header (logo area)
    const sidebarHeader = document.querySelector('.sidebar-header-items.sidebar-primary__section');

    if (sidebarHeader) {
        // Create the link container
        const linkBox = document.createElement('div');
        linkBox.className = 'sidebar-textbook-link';
        linkBox.style.cssText = `
            margin: 0.5rem 1rem;
            padding: 0.6rem 0.8rem;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            text-align: center;
        `;

        // Create the actual link
        const link = document.createElement('a');
        link.href = 'https://mlsysbook.ai';
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = 'Hands-on labs for the ML Systems textbook';
        link.style.cssText = `
            font-size: 0.75rem;
            color: #64748b;
            text-decoration: none;
            line-height: 1.4;
            display: block;
            transition: color 0.2s ease;
        `;

        // Add hover effect
        link.addEventListener('mouseenter', function() {
            this.style.color = '#1e293b';
        });
        link.addEventListener('mouseleave', function() {
            this.style.color = '#64748b';
        });

        // Assemble and insert
        linkBox.appendChild(link);
        sidebarHeader.appendChild(linkBox);
    }
});
