// Auto-collapse sections marked with auto-collapse: true
document.addEventListener('DOMContentLoaded', function() {
  setTimeout(() => {
    console.log('=== AUTO-COLLAPSING SECTIONS ===');
    
    // Sections with auto-collapse: true in config
    const autoCollapseSectionIds = [
      'labs-overview',
      'arduino', 
      'seeed-xiao',
      'grove-vision',
      'raspberry-pi',
      'shared-labs',
      'resources'
    ];
    
    // Try to find sections that should be auto-collapsed
    autoCollapseSectionIds.forEach(sectionId => {
      // Look for elements with this ID or related to this section
      const possibleElements = [
        document.getElementById(sectionId),
        document.querySelector(`[data-bs-target="#${sectionId}"]`),
        document.querySelector(`[href="#${sectionId}"]`)
      ].filter(Boolean);
      
      possibleElements.forEach(element => {
        // Find the toggle button associated with this section
        let toggleButton = null;
        
        if (element.hasAttribute('data-bs-toggle')) {
          toggleButton = element;
        } else {
          toggleButton = element.querySelector('a[data-bs-toggle="collapse"]') ||
                       element.closest('li').querySelector('a[data-bs-toggle="collapse"]');
        }
        
        if (toggleButton) {
          const sectionName = toggleButton.querySelector('.menu-text')?.textContent?.trim() || sectionId;
          console.log(`Auto-collapsing: ${sectionName}`);
          toggleButton.click();
        }
      });
    });
  }, 1000);
});