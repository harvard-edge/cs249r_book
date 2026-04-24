/**
 * Initializes menu slide functionality and toggle button event listeners
 * @param {Element} shadowEle - The shadow root element
 */
export function initializeMenuSlide(shadowEle) {
  if (!shadowEle) {
    console.error('Shadow element is required');
    return;
  }

  const toggleButton = shadowEle.getElementById('toggleButton');
  if (!toggleButton) {
    console.error('Toggle button not found');
    return;
  }

  // Remove any existing listeners to prevent duplicates
  const newToggleButton = toggleButton.cloneNode(true);
  toggleButton.parentNode.replaceChild(newToggleButton, toggleButton);

  // Add click event listener to toggle button
  newToggleButton.addEventListener('click', () => menu_slide(shadowEle));
}

/**
 * Handles sliding the menu in and out
 * @param {Element} shadowEle - The shadow root element
 */
export function menu_slide(shadowEle) {
  if (!shadowEle) {
    console.error('Shadow element is required');
    return;
  }

  const menu = shadowEle.getElementById('text-selection-menu');
  const menuToggleCheckbox = shadowEle.getElementById('menu-toggle');
  const toggleButton = shadowEle.getElementById('toggleButton');

  if (!menu || !menuToggleCheckbox || !toggleButton) {
    console.error('Required elements not found:', {
      menu: !!menu,
      checkbox: !!menuToggleCheckbox,
      button: !!toggleButton
    });
    return;
  }

  // Check current state and toggle accordingly
  const isMenuOpen = menu.classList.contains('translate-x-0');
  
  // Load push content setting
  const settings = JSON.parse(localStorage.getItem('userSettings') || '{}');
  const isPushEnabled = settings.pushContent || false;

  if (isMenuOpen) {
    // Slide in (hide)
    slideMenuClosed(menu, menuToggleCheckbox, toggleButton);
    if (isPushEnabled) {
      document.body.style.marginRight = '0';
      document.body.style.transition = 'margin-right 0.3s ease-in-out';
    }
  } else {
    // Slide out (show)
    slideMenuOpen(menu, menuToggleCheckbox, toggleButton);
    if (isPushEnabled) {
      document.body.style.marginRight = '400px'; // Match menu width
      document.body.style.transition = 'margin-right 0.3s ease-in-out';
    }
  }
}

/**
 * Opens or closes the menu based on the isStayOpen parameter
 * @param {Element} shadowEle - The shadow root element
 * @param {boolean} isStayOpen - Whether to force the menu to stay open
 */
export function menu_slide_on(shadowEle, isStayOpen = false) {
  if (!shadowEle) {
    console.error('Shadow element is required');
    return;
  }

  const menu = shadowEle.getElementById('text-selection-menu');
  const menuToggleCheckbox = shadowEle.getElementById('menu-toggle');
  const toggleButton = shadowEle.getElementById('toggleButton');

  if (!menu || !menuToggleCheckbox || !toggleButton) {
    console.error('Required elements not found');
    return;
  }

  if (isStayOpen) {
    // Always open the menu if isStayOpen is true
    slideMenuOpen(menu, menuToggleCheckbox, toggleButton);
  } else {
    // Toggle based on current state
    const isMenuOpen = menu.classList.contains('translate-x-0');
    if (isMenuOpen) {
      slideMenuClosed(menu, menuToggleCheckbox, toggleButton);
    } else {
      slideMenuOpen(menu, menuToggleCheckbox, toggleButton);
    }
  }
}

/**
 * Helper function to slide the menu open
 * @param {Element} menu - The menu element
 * @param {Element} checkbox - The menu toggle checkbox
 * @param {Element} button - The toggle button
 */
function slideMenuOpen(menu, checkbox, button) {
  // Add transition class if not present
  if (!menu.classList.contains('transition-transform')) {
    menu.classList.add('transition-transform');
  }
  
  checkbox.checked = true;
  menu.classList.remove('translate-x-full');
  menu.classList.add('translate-x-0');
  
  // Dispatch custom event for other components that might need to know
  menu.dispatchEvent(new CustomEvent('menuOpened'));
}

/**
 * Helper function to slide the menu closed
 * @param {Element} menu - The menu element
 * @param {Element} checkbox - The menu toggle checkbox
 * @param {Element} button - The toggle button
 */
function slideMenuClosed(menu, checkbox, button) {
  // Add transition class if not present
  if (!menu.classList.contains('transition-transform')) {
    menu.classList.add('transition-transform');
  }
  
  checkbox.checked = false;
  menu.classList.remove('translate-x-0');
  menu.classList.add('translate-x-full');
  
  // Dispatch custom event for other components that might need to know
  menu.dispatchEvent(new CustomEvent('menuClosed'));
}

export function shortcutKeys(shadowEle){
  document.addEventListener('keydown', (e) => {
    // Check for Ctrl/Cmd + /
    if ((e.ctrlKey || e.metaKey) && e.key === '/') {
      e.preventDefault(); // Prevent default browser behavior
      menu_slide(shadowEle);
      
      // Focus the input after menu opens
      const menu = shadowEle.getElementById('text-selection-menu');
      const userInput = menu.querySelector('#user-input');
      if (userInput) {
        setTimeout(() => {
          userInput.focus();
        }, 100);
      }
    }
  })
}
