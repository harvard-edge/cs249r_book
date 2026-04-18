export function addRedVerticalLine() {
    const selection = window.getSelection();
    if (selection.isCollapsed) return;  // Exit if there's no selection

    // Track the leftmost position and vertical extent of the selection
    let minLeft = Infinity;
    let overallTop = Infinity;
    let overallBottom = 0;

    // Calculate the bounds of the selection
    for (let i = 0; i < selection.rangeCount; i++) {
        const range = selection.getRangeAt(i);
        let startNode = range.startContainer;

        // If the startNode is a text node, use its parent element for positioning
        if (startNode.nodeType === Node.TEXT_NODE) {
            startNode = startNode.parentNode;
        }

        // Get the bounding rectangle of the parent element
        const rect = startNode.getBoundingClientRect();
        minLeft = Math.min(minLeft, rect.left);
        overallTop = Math.min(overallTop, rect.top + window.scrollY);  // Adjust for page scroll
        overallBottom = Math.max(overallBottom, rect.bottom + window.scrollY);
    }

    // Check if a vertical line already exists
    let verticalLine = document.querySelector('.vertical-line-highlight');
    if (!verticalLine) {
        // Create a new vertical line if it doesn't exist
        verticalLine = document.createElement('div');
        verticalLine.classList.add('vertical-line-highlight');
        verticalLine.style.position = 'absolute';
        verticalLine.style.width = '2px';
        verticalLine.style.backgroundColor = 'red';
        document.body.appendChild(verticalLine);
    }

    // Update or set the position and size of the vertical line
    verticalLine.style.left = `${minLeft - 5}px`; // 5px to the left of the leftmost edge
    verticalLine.style.top = `${overallTop}px`;
    verticalLine.style.height = `${overallBottom - overallTop}px`;
}

export function removeRedVerticalLine() {
    // Find all elements with the class 'vertical-line-highlight' and remove them
    document.querySelectorAll('.vertical-line-highlight').forEach(line => line.remove());
}

// export function removeRedVerticalLine() {
//     document.querySelectorAll('.vertical-line-highlight').forEach(line => line.remove());
// }
