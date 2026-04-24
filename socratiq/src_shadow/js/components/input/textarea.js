export function textarea_size_font(shadowEle) {
    const textarea = shadowEle.getElementById('expandingTextarea');
    const initialFontSize = window.getComputedStyle(textarea).fontSize; // Get the default font size
    const minFontSize = 14; // Minimum font size in pixels
    const fixedLineHeight = 1.2; // Fixed line height in em
    const threshold = 20; // Number of characters to start reducing font size
    // Set the placeholder text
    textarea.placeholder = "Ask me anything about this page...";

    function adjustTextAreaStyle() {

        // let currentLength = textarea.value.length;

        // // Determine the scale factor based on the current text length
        // let scaleFactor = currentLength < threshold ? 1 : Math.max(1 - (currentLength - threshold) / threshold, 0);

        // // Calculate new font size, but do not go below the minimum values
        // let newFontSize = Math.max(parseFloat(initialFontSize) * scaleFactor, minFontSize);
        
        // // Set a fixed line height
        // textarea.style.lineHeight = `${fixedLineHeight}em`;
        // textarea.style.fontSize = `${newFontSize}px`;

        let currentLength = textarea.value.length;
        // customPlaceholder.style.display = currentLength === 0 ? 'block' : 'none';
     
            // Existing logic to adjust the font size
            let scaleFactor = currentLength < threshold ? 1 : Math.max(1 - (currentLength - threshold) / threshold, 0);
            let newFontSize = Math.max(parseFloat(initialFontSize) * scaleFactor, minFontSize);
            textarea.style.lineHeight = `${fixedLineHeight}em`;
            textarea.style.fontSize = `${newFontSize}px`;
        
    }

    // Add the input event listener to the textarea
    textarea.addEventListener('input', adjustTextAreaStyle);

    // Set initial styles based on the content of textarea on load
    textarea.value=""
    adjustTextAreaStyle();
}
