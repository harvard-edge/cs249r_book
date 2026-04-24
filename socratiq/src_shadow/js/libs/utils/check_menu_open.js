// const elementId = "text-selection-menu-highlight"

// If menu is closed open it!
export function check_menu_open(shadowEle) {
  // Get the checkbox element
  const menuToggleCheckbox = shadowEle.getElementById("menu-toggle");

  // Check if the element exists to avoid errors
  if (menuToggleCheckbox) {
    // Add an event listener for the 'change' event
    menuToggleCheckbox.addEventListener("change", function () {
      if (this.checked) {
        toggleHiddenClass(shadowEle);
      }
    });
  } else {
  }
}

/**
 * Accesses the shadow root to find an element by its ID and, if found, removes the 'hidden' class from the element.
 *
 * @param {Element} shadowEle - The shadow element to work with
 * @param {string} elementId - The ID of the element to find (default is 'text-selection-menu-highlight')
 */
function toggleHiddenClass(
  shadowEle,
  elementId = "text-selection-menu-highlight"
) {
  // Access the shadow root and find the element by its ID
  const elementToHide = shadowEle.querySelector(`#${elementId}`);

  if (elementToHide) {
    // Remove the 'hidden' class if it exists
    if (!elementToHide.classList.contains("hidden")) {
      elementToHide.classList.add("hidden");
    }
  } else {
    console.error("Element not found in the shadow DOM.");
  }
}

/**
 * Toggles the menu open or closed based on the provided shadow element and isOpen flag.
 *
 * @param {Element} shadowEle - The shadow element to work with.
 * @param {boolean} isOpen - The flag indicating whether the menu should be open or closed. Default is true.
 */
export function menu_open(shadowEle, isOpen = true) {
  // Get the checkbox element
  const menuToggleCheckbox = shadowEle.getElementById("menu-toggle");

  // Check if the element exists to avoid errors
  if (isOpen) {
    // Add an event listener for the 'change' event
    menuToggleCheckbox.checked = true;
  } else {
    menuToggleCheckbox.checked = false;
  }
}

export function isMenuOpen(shadowEle) {
  // Get the checkbox element
  const menuToggleCheckbox = shadowEle.getElementById("text-selection-menu");

  // Check if the element exists to avoid errors
  // if (menuToggleCheckbox.checked) {
  if (menuToggleCheckbox.classList.contains("translate-x-0")) {
    // Add an event listener for the 'change' event
    return true;
  }
  return false;
}
