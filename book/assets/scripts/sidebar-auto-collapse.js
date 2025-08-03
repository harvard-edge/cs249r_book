// Auto-collapse sections marked with auto-collapse: true
document.addEventListener('DOMContentLoaded', function() {
  // Try multiple times with different delays in case DOM is still loading
  [1000, 2000, 3000].forEach(delay => {
    setTimeout(() => {
      // Find all collapse toggle buttons in sidebar
      const allToggleButtons = document.querySelectorAll('.sidebar-navigation a[data-bs-toggle="collapse"]');
      
      allToggleButtons.forEach(button => {
        const target = button.getAttribute('data-bs-target');
        const menuText = button.querySelector('.menu-text')?.textContent?.trim();
        
        // Check if this is a section we want to auto-collapse
        const shouldCollapse = [
          'Hands-on Labs',
          'Arduino', 
          'Seeed XIAO ESP32S3',
          'Grove Vision',
          'Raspberry Pi',
          'Shared',
          'Resources'
        ].includes(menuText);
        
        if (shouldCollapse) {
          const targetElement = document.querySelector(target);
          // Only click if the section is currently expanded
          if (targetElement && !targetElement.classList.contains('collapse') || 
              targetElement?.classList.contains('show')) {
            button.click();
          }
        }
      });
    }, delay);
  });
});