

export function collapsible_buttons(shadowEle) {
    let isOpen = true; // Initial state is open
    const content = shadowEle.getElementById('collapsible-content');
    const carrot = shadowEle.getElementById('carrot');

    // Initially hide the content if it should start collapsed
    // content.style.height = content.scrollHeight + 'px'; 
    content.style.overflow = 'hidden'; // Prevent content from overflowing during transition

    // Function to update display based on isOpen state
    function updateDisplay() {
        if (isOpen) {
            // Expand the content
            content.style.height = content.scrollHeight + 'px';  // Use scrollHeight to get the natural height of the content
            carrot.classList.remove('rotate-180');
        } else {
            // Collapse the content
            content.style.height = '0';
            carrot.classList.add('rotate-180');
        }
    }

    // Toggle function to change isOpen state and update display
    function toggleCollapse() {
        isOpen = !isOpen;
        updateDisplay();
    }

    // Attach event listener to carrot for click event
    carrot.addEventListener('click', toggleCollapse);

    // Expose the toggle function if needed externally
    shadowEle.toggleCollapse = toggleCollapse;

    // Set initial state properly based on whether it should be open or not
    // updateDisplay();
}

// This function allows you to control the collapsible component externally
export function closeCollapsible(shadowEle) {
    // Ensure that toggleCollapse method is available and the component is open before closing
    if (shadowEle.toggleCollapse && isOpen) {
        shadowEle.toggleCollapse();
    }
}
