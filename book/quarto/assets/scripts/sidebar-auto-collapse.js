// Auto-collapse sections and add part dividers
document.addEventListener('DOMContentLoaded', function() {

  // Track if we're intentionally closing sidebar for mobile navigation
  let isMobileNavigating = false;

  // Function to expand sidebar to show current page
  function expandToCurrentPage() {
    const currentPath = window.location.pathname;

    // Extract just the filename for matching
    const currentFile = currentPath.split('/').pop();

    // Find the active sidebar link matching current page
    const sidebarLinks = document.querySelectorAll('#quarto-sidebar .sidebar-link');
    let activeLink = null;

    sidebarLinks.forEach(link => {
      const href = link.getAttribute('href');
      if (href) {
        // Get the target file from the href
        const hrefFile = href.split('/').pop().replace(/\.qmd$/, '.html');

        // Match by filename
        if (currentFile === hrefFile) {
          activeLink = link;
          // Add active class to highlight current page
          link.classList.add('active');
          link.setAttribute('aria-current', 'page');
        }
      }
    });

    // If we found the active link, expand all parent collapse sections
    if (activeLink) {
      const sidebar = document.getElementById('quarto-sidebar');
      let parent = activeLink.parentElement;

      while (parent && parent !== document.body) {
        // Stop if we've reached the sidebar itself - don't expand it
        if (parent === sidebar) {
          break;
        }

        // Look for collapsed sections that need to be expanded (but not the sidebar itself)
        if (parent.classList.contains('collapse') &&
            !parent.classList.contains('show') &&
            !parent.classList.contains('quarto-sidebar-collapse-item')) {
          parent.classList.add('show');

          // Find the toggle button for this section and update its aria-expanded
          const toggleButton = document.querySelector(`[data-bs-target="#${parent.id}"]`);
          if (toggleButton) {
            toggleButton.setAttribute('aria-expanded', 'true');
            toggleButton.classList.remove('collapsed');
          }
        }
        parent = parent.parentElement;
      }

      // Scroll the active link into view with smooth behavior
      // Use a longer delay to ensure expansion animations complete
      setTimeout(() => {
        // Try multiple possible scroll containers
        const sidebar = document.querySelector('#quarto-sidebar') ||
                       document.querySelector('.sidebar-navigation') ||
                       document.querySelector('.sidebar');

        if (sidebar && sidebar.scrollHeight > sidebar.clientHeight) {
          // Get the position of the active link relative to the sidebar
          const linkRect = activeLink.getBoundingClientRect();
          const sidebarRect = sidebar.getBoundingClientRect();

          // Calculate how much to scroll to center the active link in the sidebar
          const offset = linkRect.top - sidebarRect.top - (sidebarRect.height / 2) + (linkRect.height / 2);

          // Smooth scroll the sidebar
          sidebar.scrollBy({
            top: offset,
            behavior: 'smooth'
          });
        } else {
          // Fallback to standard scrollIntoView
          activeLink.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 500);
    }
  }

  // Add part dividers (do this once, early)
  setTimeout(() => {
    const allToggleButtons = document.querySelectorAll('#quarto-sidebar a[data-bs-toggle="collapse"]');

    allToggleButtons.forEach(button => {
      const menuText = button.querySelector('.menu-text')?.textContent?.trim();

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
  }, 500);

  // FIRST: Expand to show current page (but don't force sidebar open in desktop view)
  setTimeout(() => {
    // Only expand sections within sidebar, don't force the sidebar itself to open
    expandToCurrentPage();

    // THEN: Collapse sections that DON'T contain the current page
    setTimeout(() => {
      const allToggleButtons = document.querySelectorAll('#quarto-sidebar a[data-bs-toggle="collapse"]');

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

          // Check if this section contains the active page
          const containsActivePage = targetElement?.querySelector('.sidebar-link.active');

          // Only collapse if it's expanded AND doesn't contain the active page
          if (!containsActivePage && targetElement && targetElement.classList.contains('show')) {
            button.click();
          }
        }
      });
    }, 400);
  }, 1000);

  // Prevent sidebar collapse from hiding the active page
  // Listen for collapse events and re-expand if needed
  document.addEventListener('shown.bs.collapse', function(e) {
    // Don't interfere if we're closing sidebar for mobile navigation
    if (isMobileNavigating) return;

    // Only check if this is a collapse event within the sidebar, not the sidebar itself
    const sidebar = document.getElementById('quarto-sidebar');
    if (!sidebar || !sidebar.contains(e.target)) return;

    // Small delay to let Bootstrap finish, then ensure active page is visible
    setTimeout(() => {
      const activeLink = document.querySelector('.sidebar-navigation .sidebar-link.active');
      if (activeLink) {
        const isVisible = activeLink.offsetParent !== null;
        if (!isVisible) {
          expandToCurrentPage();
        }
      }
    }, 50);
  });

  document.addEventListener('hidden.bs.collapse', function(e) {
    // Don't interfere if we're closing sidebar for mobile navigation
    if (isMobileNavigating) return;

    // Only check if this is a collapse event within the sidebar, not the sidebar itself
    const sidebar = document.getElementById('quarto-sidebar');
    if (!sidebar || !sidebar.contains(e.target)) return;

    // Small delay to let Bootstrap finish, then ensure active page is visible
    setTimeout(() => {
      const activeLink = document.querySelector('.sidebar-navigation .sidebar-link.active');
      if (activeLink) {
        const isVisible = activeLink.offsetParent !== null;
        if (!isVisible) {
          expandToCurrentPage();
        }
      }
    }, 50);
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
      // Try multiple selectors to find the text element
      const textElement = callout.querySelector('summary strong') ||
                         callout.querySelector('summary') ||
                         callout.querySelector('.callout-caption') ||
                         callout.querySelector('.callout-title');

      if (textElement) {
        const originalText = textElement.textContent;
        // Match pattern like "Self-Check: Question 1.3" and extract just the number after the dot
        const match = originalText.match(/^(.*?)\s+(\d+)\.(\d+)(.*)$/);
        if (match) {
          const [, prefix, chapterNum, questionNum, suffix] = match;
          textElement.textContent = `${prefix} ${questionNum}${suffix}`;
          console.log(`Fixed quiz numbering: "${originalText}" → "${textElement.textContent}"`);
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

  // Automatically close sidebar on link click in mobile view
  function setupMobileSidebarToggle() {
    const sidebar = document.getElementById('quarto-sidebar');
    if (!sidebar) return;

    const sidebarToggler = document.querySelector('.quarto-btn-toggle');
    if (!sidebarToggler) return;

    const sidebarLinks = sidebar.querySelectorAll('a.sidebar-link');

    sidebarLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        // Check if sidebar is in mobile/collapsed mode
        const isSidebarOpen = sidebar.classList.contains('show');

        // Check if we're in mobile view (sidebar toggle button is visible)
        const isMobileView = window.getComputedStyle(sidebarToggler).display !== 'none';

        if (isMobileView && isSidebarOpen) {
          // Set flag to prevent collapse handlers from reopening sidebar
          isMobileNavigating = true;

          // Close the sidebar
          sidebarToggler.click();

          // Clear flag after navigation completes
          setTimeout(() => {
            isMobileNavigating = false;
          }, 1000);
        }
      });
    });
  }

  // Run after a short delay to ensure everything is initialized
  setTimeout(setupMobileSidebarToggle, 500);
});
