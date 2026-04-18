import { menu_slide_on} from '../menu/open_close_menu.js';
import { SIZE_LIMIT_LLM_CALL} from '../../../configs/env_configs.js' // Adjust this value as needed
import {reduceTextSize} from '../../libs/utils/textUtils.js'
import {QuizStorage} from '../../libs/memory/quizStorage.js'
import { ChapterDataCollector } from '../../libs/memory/chapterDataCollector.js';


export function injectSvgButtons(shadowRoot) {
  const collector = new ChapterDataCollector(); // Initialize the collector
  const currentUrl = window.location.href;
  
  // Add this check at the beginning - return if there's NO chapter number
  const h1WithChapterNumber = document.querySelector('h1 .chapter-number');
  if (!h1WithChapterNumber) {
    return;
  }

  // Skip quiz injection for specific URLs
  if (
    currentUrl.match(/^[^/]+\/?$/) || // Matches root URL with or without trailing slash
    currentUrl.includes('about.html') ||
    currentUrl.includes('acknowledgements.html') ||
    currentUrl.includes('socratiq.html') ||
    isRootUrl(currentUrl)
  ) {
    return;
  }
  const level2Sections = document.querySelectorAll('section.level2');
  let quizTitles = [];
  const quizStorage = new QuizStorage();

  // Initialize data structure in the window object
  if (!window.sectionData) {
    window.sectionData = {};
  }

  level2Sections.forEach((section, index) => {
    const h2 = section.querySelector('h2.anchored');
    if (!h2 || h2.textContent.trim().includes('Resources')) {
        return;
    }

    const sectionText = section.textContent.trim();
    const sectionId = section.id;
    const dataNumber = h2.getAttribute('data-number');
    const titleText = h2.textContent.trim();
    const title = dataNumber + ' ' + titleText.replace(`${dataNumber} `, '');
    quizTitles.push(title);

    // Save data to the window's data structure
    window.sectionData[sectionId] = {
      title: title,
      content: sectionText
    };

    scheduleBackgroundCollection(collector, {
      section,
      sectionId,
      title,
      sectionText,
      dataNumber
    });

    // Create and append the quiz button directly to the end of the section
    const iconWrapper = createIconWrapper(title, sectionText, sectionId, shadowRoot);
    section.appendChild(iconWrapper);
  });

  // Handle the quiz component for the entire chapter
  const lastLevel2Section = level2Sections[level2Sections.length - 1];
  if (lastLevel2Section) {
    const quizComponent = createQuizComponent(quizTitles, shadowRoot);
    lastLevel2Section.insertAdjacentElement('afterend', quizComponent);
  }

  // Store the quiz titles in IndexedDB in the background
  if (quizTitles.length > 0) {
    // Use requestIdleCallback if available, otherwise use setTimeout
    const scheduleBackgroundTask = window.requestIdleCallback || 
      ((callback) => setTimeout(callback, 0));

    scheduleBackgroundTask(() => {
      quizStorage.saveQuizTitles(currentUrl, quizTitles)
        .catch(error => console.error('Error saving quiz titles:', error));
    });
  }
}

function createIconWrapper(title, sectionText, sectionId, shadowRoot) {
  const wrapperDiv = document.createElement('div');
  wrapperDiv.style.cssText = `
    margin-top: 0.5rem;
    display: flex;
    justify-content: flex-start;
    width: 100%;
  `;

  const iconWrapper = document.createElement('div');
  iconWrapper.className = 'callout callout-style-simple callout-exercise no-icon callout-titled';
  iconWrapper.style.width = '102%';
  
  iconWrapper.innerHTML = `
    <div class="callout-header d-flex align-content-center">
      <div class="callout-icon-container">
        🧩
      </div>
      <div class="callout-title-container flex-fill">
        &nbsp; Section Quiz
      </div>
    </div>
  `;

  // Update hover effect styles
  const style = document.createElement('style');
  style.textContent = `
    .callout-exercise.callout {
      cursor: pointer !important;
      transition: background-color 0.3s ease;
      margin-bottom: 0;
      width: 102%;
      border-left-color: #DC143C !important;
    }
    .callout-exercise.callout:hover {
      background-color: rgb(243, 244, 246) !important;
      cursor: pointer !important;
    }
    .callout-icon-container {
      display: flex;
      align-items: center;
      margin-right: 10px !important;
      font-size: 1.1em !important;
    }
  `;
  iconWrapper.appendChild(style);

  iconWrapper.dataset.sectionId = sectionId;
  iconWrapper.dataset.sectionText = sectionText;

  iconWrapper.addEventListener('click', (event) => {
    event.stopPropagation();
    handleQuizClick(title, sectionText, sectionId, shadowRoot);
  });

  wrapperDiv.appendChild(iconWrapper);
  return wrapperDiv;
}


function createQuizComponent(quizTitles, shadowRoot) {
  const component = document.createElement('div');
  component.className = 'callout callout-style-simple callout-exercise no-icon callout-titled custom-quiz-callout';
  
  // Add custom CSS for the callout
  const style = document.createElement('style');
  style.textContent = `
    .custom-quiz-callout.callout-exercise.callout {
      border-left-color: #DC143C !important;
      width: 102% !important;
    }
    .custom-quiz-callout.callout-exercise.callout > .callout-header::before {
      font-family: sans-serif !important;
      content: "🧩" !important;
      margin-right: 10px !important;
      display: inline-block !important;
      text-decoration: none !important;
      font-size: 1.1em !important;
    }
    .custom-quiz-callout.callout-exercise.callout > .callout-header {
      text-decoration: none !important;
    }
  `;
  component.appendChild(style);
  
  component.innerHTML += `
    <div class="callout-header d-flex align-content-center" data-bs-toggle="collapse" data-bs-target=".callout-3-contents" aria-controls="callout-3" aria-expanded="true" aria-label="Toggle callout">
      <div class="callout-icon-container">
        <i class="callout-icon no-icon"></i>
      </div>
      <div class="callout-title-container flex-fill">
        Sub Section Quick AI Quizzes
      </div>
      <div class="callout-btn-toggle d-inline-block border-0 py-1 ps-1 pe-0 float-end"><i class="callout-toggle"></i></div>
    </div>
    <div id="callout-3" class="callout-3-contents callout-collapse collapse show">
      <div class="callout-body-container callout-body">
        <ul>
          ${quizTitles
            .filter(title => !title.toLowerCase().includes('null resources'))
            .map((title, index) => {
              const match = title.match(/^(\d+\.\d+)\s+\1\s+(.+)$/);
              if (match) {
                return `<li><a href="#" class="quiz-link" data-index="${index}">${match[1]} ${match[2]}</a></li>`;
              }
              return `<li><a href="#" class="quiz-link" data-index="${index}">${title}</a></li>`;
            })
            .join('')}
        </ul>
      </div>
    </div>
  `;

  component.querySelectorAll('.quiz-link').forEach((link) => {
    link.addEventListener('click', (event) => {
      event.preventDefault();
      const index = parseInt(link.getAttribute('data-index'), 10);
      const title = quizTitles[index];
      const sectionId = Object.keys(window.sectionData)[index];
      if (sectionId && window.sectionData[sectionId]) {
        const sectionText = window.sectionData[sectionId].content;
        handleQuizClick(title, sectionText, sectionId, shadowRoot);
      } else {
        console.error('No matching section data found for index:', index);
      }
    });
  });


  return component;
}
 

  function handleQuizClick(title, sectionText, sectionId, shadowRoot) {
    if (shadowRoot) {
      menu_slide_on(shadowRoot, true);
    } else {
      console.error('Shadow root not found');
    }
    

    const reducedText = reduceTextSize(sectionText, SIZE_LIMIT_LLM_CALL); // leep text under limit
    
    
    triggerAIAction(title, reducedText, sectionId)
  }


  
function triggerAIAction(title, sectionText, sectionId) {

  // Dispatch a custom event to trigger the AI action
  const aiActionEvent = new CustomEvent('aiActionCompleted', {
    detail: {
      type: 'quiz', // or 'explain', depending on what you want to trigger
      text: sectionText,
      title: title,
      sectionId: sectionId
    }
  });
  window.dispatchEvent(aiActionEvent);
}

// Helper function to check if it's a root URL
function isRootUrl(url) {
  try {
    const urlObj = new URL(url);
    return urlObj.pathname === '/' || urlObj.pathname === '';
  } catch (e) {
    console.error('Invalid URL:', e);
    return false;
  }
}

function scheduleBackgroundCollection(collector, data) {
  const scheduleTask = window.requestIdleCallback || 
      ((callback) => setTimeout(callback, 0));

  scheduleTask(async () => {
      try {
          const chapterNumber = collector.getChapterNumber();
          if (!chapterNumber) return;

          await collector.storage.saveSectionSummary(
              chapterNumber,
              data.sectionId,
              {
                  id: data.sectionId,
                  dataNumber: data.dataNumber,
                  title: data.title,
                  content: data.sectionText,
                  url: window.location.href
              }
          );
      } catch (error) {
          console.error('Error in background collection:', error);
      }
  });
}
