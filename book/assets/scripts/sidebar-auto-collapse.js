// Auto-collapse sections and add part dividers
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

        // Add part dividers before major sections
        const partMappings = {
          'Systems Foundations': 'Part I',
          'Design Principles': 'Part II', 
          'Performance Engineering': 'Part III',
          'Robust Deployment': 'Part IV',
          'Trustworthy Systems': 'Part V',
          'Frontiers of ML Systems': 'Part VI',
          'Hands-on Labs': 'Part VII'
        };

        if (partMappings[menuText]) {
          const sidebarItem = button.closest('.sidebar-item');
          if (sidebarItem && !sidebarItem.querySelector('.part-divider')) {
            const partDivider = document.createElement('div');
            partDivider.className = menuText === 'Hands-on Labs' ? 'part-divider part-labs' : 'part-divider';
            partDivider.textContent = partMappings[menuText];
            sidebarItem.insertBefore(partDivider, sidebarItem.firstChild);
          }
        }
      });
    }, delay);
  });

  // Handle navbar active states
  function setNavbarActiveState() {
    const currentUrl = window.location.href;
    const currentPath = window.location.pathname;
    const navbarLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navbarLinks.forEach(link => {
      // Remove existing active classes
      link.classList.remove('active');
      link.removeAttribute('aria-current');
      
      const href = link.getAttribute('href');
      if (href && !href.includes('mlsysbook.ai')) { // Skip external PDF link
        let linkUrl = href;
        
        // If it's a relative path, make it absolute for comparison
        if (!href.startsWith('http')) {
          linkUrl = window.location.origin + '/' + href.replace(/^\//, '');
        }
        
        // Convert .qmd to .html for comparison since that's what gets rendered
        linkUrl = linkUrl.replace(/\.qmd$/, '.html');
        
        // Check for exact match
        if (currentUrl === linkUrl || currentPath === new URL(linkUrl).pathname) {
          link.classList.add('active');
          link.setAttribute('aria-current', 'page');
          console.log('Setting active:', link.textContent.trim(), 'Current:', currentUrl, 'Link:', linkUrl);
        }
      }
    });
  }

  // Set initial active state
  setNavbarActiveState();
  
  // Also check after a brief delay in case page is still loading
  setTimeout(setNavbarActiveState, 500);
  
  // Fix quiz numbering: Remove chapter numbers from quiz callouts (HTML only)
  // Transform "Self-Check: Question 1.3" to "Self-Check: Question 3"
  function fixQuizNumbering() {
    const quizCallouts = document.querySelectorAll('.callout-quiz-question, .callout-quiz-answer');
    
    quizCallouts.forEach(callout => {
      const caption = callout.querySelector('.callout-caption');
      if (caption) {
        const originalText = caption.textContent;
        // Match pattern like "Self-Check: Question 1.3" and extract just the number after the dot
        const match = originalText.match(/^(.*?)\s+(\d+)\.(\d+)(.*)$/);
        if (match) {
          const [, prefix, chapterNum, questionNum, suffix] = match;
          caption.textContent = `${prefix} ${questionNum}${suffix}`;
        }
      }
    });
  }
  
  // Run quiz numbering fix with multiple delays to ensure DOM is ready
  [100, 500, 1000, 2000].forEach(delay => {
    setTimeout(fixQuizNumbering, delay);
  });
  
  // Also run on any dynamic content loads
  const observer = new MutationObserver(() => {
    setTimeout(fixQuizNumbering, 100);
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
});