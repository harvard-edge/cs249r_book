import { extractReadableBookText } from '../../utils/textExtractor.js';

const IGNORED_CONTEXT_SELECTOR = 'nav, aside:not(.column-margin), header, footer, #socratiq-reader, #widget-chat-container, #collaborative-widget-host, .chapter-concept-map, [data-socratiq-ignore]';

// Keep the old function for backward compatibility with other parts that might need random IDs
function generateUniqueId() {
    const timePart = new Date().getTime();
    const randomPart = Math.random().toString(36).substring(2, 15);
    return `${timePart}-${randomPart}`;
}

// Standalone stable ID generation function (same as in fuzzy_match.js)
function generateStableId(textContent, existingIds = new Set()) {
    // Fast hash using only first 100 chars + length + last 50 chars for speed
    const text = textContent.trim();
    const len = text.length;
    
    // For very short text, use the whole thing
    if (len <= 20) {
        const hash = simpleHash(text.toLowerCase());
        return `p-${Math.abs(hash).toString(36)}`;
    }
    
    // For longer text, use strategic sampling for speed
    const start = text.substring(0, Math.min(50, len)).toLowerCase();
    const end = text.substring(Math.max(0, len - 30)).toLowerCase();
    const middle = len > 100 ? text.substring(Math.floor(len/2), Math.floor(len/2) + 20).toLowerCase() : '';
    
    // Combine key parts with length for uniqueness
    const combined = `${start}${middle}${end}${len}`;
    const hash = simpleHash(combined);
    
    let baseId = `p-${Math.abs(hash).toString(36)}`;
    
    // Handle potential conflicts by adding a suffix
    let finalId = baseId;
    let counter = 1;
    while (existingIds.has(finalId)) {
        finalId = `${baseId}-${counter}`;
        counter++;
    }
    
    return finalId;
}

// Ultra-fast hash function - optimized for speed
function simpleHash(str) {
    let hash = 0;
    // Process every 2nd character for even more speed on long strings
    for (let i = 0; i < str.length; i += 2) {
        hash = ((hash << 5) - hash + str.charCodeAt(i)) & 0xffffffff;
    }
    return hash;
}

// New function to generate stable class IDs based on content
function generateStableClassId(textContent) {
    // Use the same stable ID generation as data-fuzzy-id
    return generateStableId(textContent);
}

export function get_text_ref() {
    const extractSubsections = () => {
      const sectionElements = Array.from(document.querySelectorAll("section"));
      sectionElements.forEach((section, sectionIndex) => {
        if (section.closest?.(IGNORED_CONTEXT_SELECTOR)) return;

        const treeWalker = document.createTreeWalker(
          section,
          NodeFilter.SHOW_ELEMENT,
          {
            acceptNode: (node) => {
              if (node.closest?.(IGNORED_CONTEXT_SELECTOR)) {
                return NodeFilter.FILTER_REJECT;
              }
              if (node.tagName === "H3" || node.tagName === "P" || node.tagName === "LI") {
                if (node.tagName === "P" && node.closest?.("li")) {
                  return NodeFilter.FILTER_SKIP;
                }
                return NodeFilter.FILTER_ACCEPT;
              }
              return NodeFilter.FILTER_SKIP;
            },
          }
        );

        let currentSubsection = "";
        let currentText = "";
        let currentHeadingElement = null;
        const subsectionElements = new Map();
        let currentElement;
        let uniqueClass;
        let existingClasses;

        while (treeWalker.nextNode()) {
          currentElement = treeWalker.currentNode;
          if (currentElement.tagName === "H3") {
            if (currentSubsection !== "") {
              subsectionElements.set(currentSubsection, {
                text: currentText.trim(),
                element: currentHeadingElement || currentElement,
                section: sectionIndex + 1,
              });
              currentText = "";
            }
            currentSubsection = extractReadableBookText(currentElement) || currentElement.textContent.trim();
            currentHeadingElement = currentElement;
          } else if (currentElement.tagName === "P" || currentElement.tagName === "LI") {
            const readable = extractReadableBookText(currentElement) || currentElement.textContent.trim();
            if (readable) {
              currentText = currentText ? `${currentText} ${readable}` : readable;
            }
          }

          if (currentElement.tagName === "H3" || currentElement.tagName === "P" || currentElement.tagName === "LI") {
            // Use stable ID based on content instead of random timestamp
            uniqueClass = generateStableClassId(currentElement.textContent);
            existingClasses = (currentElement.getAttribute("class") || "").trim();
            if (!existingClasses.split(/\s+/).includes(uniqueClass)) {
              currentElement.setAttribute("class", existingClasses ? `${existingClasses} ${uniqueClass}` : uniqueClass);
            }
          }
        }

        if (currentSubsection !== "") {
          subsectionElements.set(currentSubsection, {
            text: currentText.trim(),
            element: currentHeadingElement || currentElement,
            section: sectionIndex + 1,
          });
        }

        // Index MLSysBook callouts (foldboxes, callout divs, fallacy-pitfall blocks)
        const callouts = section.querySelectorAll('details[class*="callout-"], div.callout, div.fallacy-pitfall');
        callouts.forEach((callout) => {
          if (
            callout.closest?.(IGNORED_CONTEXT_SELECTOR) ||
            callout.classList?.contains('callout-exercise') ||
            callout.classList?.contains('custom-quiz-callout')
          ) {
            return;
          }
          const titleNode = callout.querySelector('summary, .callout-title-container, .callout-header');
          const calloutKey = (titleNode ? extractReadableBookText(titleNode) || titleNode.textContent : '').trim();
          const calloutText = (
            extractReadableBookText(callout, { includeClosedDetails: true }) ||
            extractReadableBookText(callout) ||
            callout.textContent ||
            ''
          ).trim();
          if (calloutKey && calloutText) {
            subsectionElements.set(calloutKey, {
              text: calloutText,
              element: callout,
              section: sectionIndex + 1,
            });
          }
        });

        window.subsections = { ...window.subsections, ...Object.fromEntries(subsectionElements) };
      });
    };

    const extractKeywords = () => {
      const keywords = {};
      const contentRoot = document.querySelector('main#quarto-document-content, main, article') || document.body;
      const keywordElements = Array.from(contentRoot ? contentRoot.querySelectorAll("strong") : []);
      keywordElements.forEach((element) => {
        if (element.closest?.(IGNORED_CONTEXT_SELECTOR)) {
          return;
        }
        let identifier = element.getAttribute("data-identifier");
        if (!identifier) {
          // Use stable ID based on the keyword text content
          identifier = generateStableClassId(element.textContent);
          element.setAttribute("data-identifier", identifier);
        }
        const rawKeyword = extractReadableBookText(element) || element.textContent || "";
        const keyword = rawKeyword.trim().replace(/[.,!?;:]$/g, '');
        if (!keyword) return;

        const summaryElement = element.closest?.('summary');
        const detailsElement = summaryElement ? summaryElement.closest?.('details') : null;
        let textContent = "";
        if (detailsElement) {
          textContent =
            extractReadableBookText(detailsElement, { includeClosedDetails: true }) ||
            extractReadableBookText(detailsElement);
        } else {
          const container =
            element.closest?.('p, li, dd, blockquote, div[id^="fn"], .callout') ||
            element.parentElement ||
            element.parentNode;
          textContent = container ? extractReadableBookText(container) : "";
          if (!textContent && container?.textContent) {
            textContent = container.textContent.replace(/\s+/g, ' ').trim();
          }
        }
        if (!textContent) {
          textContent = keyword;
        }
        keywords[keyword] = { text: textContent, identifier };
      });

      window.keywords = keywords;
    };

    const main = () => {
      extractSubsections();
      extractKeywords();
    };

    main();

}