export function addRedVerticalLine() {
    const selection = window.getSelection();
    if (selection.isCollapsed) return;

    // Track the leftmost position
    let minLeft = Infinity;
    let overallTop = Infinity;
    let overallBottom = 0;

    for (let i = 0; i < selection.rangeCount; i++) {
        const range = selection.getRangeAt(i);
        let startNode = range.startContainer;

        // If startNode is a text node, use its parent element
        if (startNode.nodeType === Node.TEXT_NODE) {
            startNode = startNode.parentNode;
        }

        // Get the bounding rectangle of the parent element
        const rect = startNode.getBoundingClientRect();

        // Update the coordinates for the vertical line
        minLeft = Math.min(minLeft, rect.left);
        overallTop = Math.min(overallTop, rect.top + window.scrollY); // Adjust for page scroll
        overallBottom = Math.max(overallBottom, rect.bottom + window.scrollY);
    }

    // Create the vertical line element
    const verticalLine = document.createElement('div');
    verticalLine.classList.add('vertical-line-highlight');
    verticalLine.style.position = 'absolute'; // Use 'absolute' for positioning relative to the document
    verticalLine.style.left = `${minLeft - 5}px`; // 5px to the left of the leftmost edge
    verticalLine.style.top = `${overallTop}px`;
    verticalLine.style.height = `${overallBottom - overallTop}px`;
    verticalLine.style.width = '2px';
    verticalLine.style.backgroundColor = 'red';
    document.body.appendChild(verticalLine);
}

export function removeRedVerticalLine() {
    document.querySelectorAll('.vertical-line-highlight').forEach(line => line.remove());
}
