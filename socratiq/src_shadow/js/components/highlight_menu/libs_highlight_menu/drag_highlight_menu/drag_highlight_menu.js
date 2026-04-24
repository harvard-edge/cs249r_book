
let shadowElement;
export function drag(shadowRoot){

const element = shadowRoot.getElementById('text-selection-menu');
    let posX = 0, posY = 0, mouseX = 0, mouseY = 0;

    const mouseDownHandler = function(e) {
        // Get the current mouse position
        mouseX = e.clientX;
        mouseY = e.clientY;

        // Attach the listeners to `document`
        document.addEventListener('mousemove', mouseMoveHandler);
        document.addEventListener('mouseup', mouseUpHandler);
    };

    const mouseMoveHandler = function(e) {
        // How far the mouse has been moved
        const dx = e.clientX - mouseX;
        const dy = e.clientY - mouseY;

        // Set the position of element
        posX = posX + dx;
        posY = posY + dy;

        element.style.left = posX + 'px';
        element.style.top = posY + 'px';

        // Reassign the position of mouse
        mouseX = e.clientX;
        mouseY = e.clientY;
    };

    const mouseUpHandler = function() {
        // Remove the handlers of `mousemove` and `mouseup`
        document.removeEventListener('mousemove', mouseMoveHandler);
        document.removeEventListener('mouseup', mouseUpHandler);
    };

    // Attach the handler
    element.addEventListener('mousedown', mouseDownHandler);}


    
// export function makeDraggable(chatElement) {
//     shadowElement = chatElement.getElementById('text-selection-menu');
//     let isDragging = false;
//     let dragStartX, dragStartY;
  
//     chatElement.addEventListener('mousedown', (e) => {
//     //   if (isSliderDragging) return; // Prevent chat widget from dragging when slider thumb is dragged

//       isDragging = true;
//       dragStartX = e.clientX - shadowElement.offsetLeft;
//       dragStartY = e.clientY - shadowElement.offsetTop;
//       shadowElement.style.cursor = 'grabbing';
//       e.stopPropagation(); // This prevents event bubbling up which might cause unintended behavior
//     });
  
//     document.addEventListener('mousemove', (e) => {
//       if (isDragging) { //&& !isSliderDragging) { // Check again in case of race conditions

//         const newX = e.clientX - dragStartX;
//         const newY = e.clientY - dragStartY;
//         shadowElement.style.left = `${newX}px`;
//         shadowElement.style.top = `${newY}px`;
//       }
//     });
  
//     document.addEventListener('mouseup', () => {
//       if (isDragging) {
//         isDragging = false;
//         shadowElement.style.cursor = 'grab';
//       }
//     });
//   }

export function makeDraggable(chatElement) {
  shadowElement = chatElement.getElementById('text-selection-menu');
  let isDragging = false;
  let dragStartX, dragStartY;

  chatElement.addEventListener('mousedown', (e) => {
      // Check if the target is the slider or part of the slider (e.g., labels)
      // if (e.target.classList.contains('slider') || e.target.closest('.no-drag')) {
      //     // If so, don't initiate drag
      //     return;
      // }
      if (e.target.id === 'expandingTextarea' || e.target.classList.contains('slider') || e.target.closest('.no-drag')) {
        // If so, allow default behavior (e.g., typing, slider adjustment)
        return;
    }

      isDragging = true;
      dragStartX = e.clientX - shadowElement.offsetLeft;
      dragStartY = e.clientY - shadowElement.offsetTop;
      shadowElement.style.cursor = 'grabbing';
      e.preventDefault(); // Prevents other actions like text selection
  });

  document.addEventListener('mousemove', (e) => {
    if (isDragging) {
      const newX = e.clientX - dragStartX;
      const newY = e.clientY - dragStartY;
      shadowElement.style.left = `${newX}px`;
      shadowElement.style.top = `${newY}px`;
      e.preventDefault(); // Again, prevents text selection during drag
    }
  });

  document.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      shadowElement.style.cursor = 'grab';
    }
  });
}
