// Make version number (in DOI field) link to releases page
document.addEventListener('DOMContentLoaded', function() {
  // Find the DOI field in the title metadata
  const doiElements = document.querySelectorAll('.quarto-title-meta-contents .doi, .quarto-title-meta .doi, [class*="doi"]');

  doiElements.forEach(function(element) {
    // Check if this element contains version-like text (vX.X.X)
    const text = element.textContent || element.innerText;
    if (text && text.match(/v\d+\.\d+\.\d+/)) {
      // Find any links within this element
      const links = element.querySelectorAll('a');
      links.forEach(function(link) {
        // Override the link to point to releases page
        link.href = 'https://github.com/harvard-edge/cs249r_book/releases';
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
      });

      // If no link exists, wrap the text in a link
      if (links.length === 0 && element.tagName !== 'A') {
        const link = document.createElement('a');
        link.href = 'https://github.com/harvard-edge/cs249r_book/releases';
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = text;
        link.style.color = 'inherit';
        element.textContent = '';
        element.appendChild(link);
      }
    }
  });
});
