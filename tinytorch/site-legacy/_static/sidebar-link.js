// Add tagline directly under the logo in the sidebar
document.addEventListener('DOMContentLoaded', function() {
    // Find the logo link in the sidebar
    const logoLink = document.querySelector('.navbar-brand.logo');

    if (logoLink) {
        // Create the tagline element
        const tagline = document.createElement('a');
        tagline.href = 'https://mlsysbook.ai';
        tagline.target = '_blank';
        tagline.rel = 'noopener noreferrer';
        tagline.className = 'sidebar-tagline';
        tagline.innerHTML = 'A Build-It-Yourself Companion to the <strong>Machine Learning Systems</strong> textbook';
        tagline.style.cssText = `
            display: block;
            font-size: 0.7rem;
            color: #64748b;
            text-decoration: none;
            line-height: 1.4;
            margin-top: 0.25rem;
            padding: 0 0.5rem;
            text-align: center;
            transition: color 0.2s ease;
        `;

        // Add hover effect
        tagline.addEventListener('mouseenter', function() {
            this.style.color = '#f97316';
        });
        tagline.addEventListener('mouseleave', function() {
            this.style.color = '#64748b';
        });

        // Insert right after the logo
        logoLink.parentNode.insertBefore(tagline, logoLink.nextSibling);
    }
});
