export function toggleMarkdownPreviewVisibility() {
    const markdownPreview = document.getElementById('markdown-preview');
    // Check if the markdownPreview has text content
    if (markdownPreview.textContent.trim() === '') {
      // If there is no text, add the hidden class from TailwindCSS
      markdownPreview.classList.add('hidden');
    } else {
      // If there is text, remove the hidden class to make it visible
      markdownPreview.classList.remove('hidden');
    }
  }

export function toggleMarkdownActivate() {

    const markdownPreview = document.getElementById('markdown-preview');

    // Check if the preview is already active to avoid re-triggering the animation
    if (!markdownPreview.classList.contains('markdown-preview-active')) {
        markdownPreview.classList.add('markdown-preview-active');
    }
    markdownPreview.classList.remove('markdown-preview-inactive');

    // markdownPreview.style.opacity = 1;

    markdownPreview.classList.remove('hidden');

}

export function toggleMarkdownDeActivate() {
    const markdownPreview = document.getElementById('markdown-preview');
    markdownPreview.classList.remove('markdown-preview-active');

    if (!markdownPreview.classList.contains('markdown-preview-inactive')) {
        markdownPreview.classList.add('markdown-preview-inactive');

    }
    // markdownPreview.style.opacity = 0;

    markdownPreview.classList.add('hidden');

}

  // Run the function to initially set the correct visibility

  // Example of updating the markdownPreview and toggling visibility
  // This part can be integrated into the part of your code where the markdownPreview gets updated
  function updateMarkdownPreview(content) {
    const markdownPreview = document.getElementById('markdown-preview');
    markdownPreview.textContent = content; // Updating the content
    toggleMarkdownPreviewVisibility(); // Adjust visibility based on new content
  }