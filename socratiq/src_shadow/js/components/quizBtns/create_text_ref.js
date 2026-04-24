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
      sectionElements.forEach((section) => {
        const treeWalker = document.createTreeWalker(
          section,
          NodeFilter.SHOW_ELEMENT,
          {
            acceptNode: (node) => {
              if (node.tagName === "H3" || node.tagName === "P") {
                return NodeFilter.FILTER_ACCEPT;
              }
              return NodeFilter.FILTER_REJECT;
            },
          }
        );

        let currentSubsection = "";
        let currentText = "";
        const subsectionElements = new Map();
        let currentElement;
        let uniqueClass;
        let existingClasses;

        while (treeWalker.nextNode()) {
          currentElement = treeWalker.currentNode;
          if (currentElement.tagName === "H3") {
            if (currentSubsection !== "") {
              subsectionElements.set(currentSubsection, { text: currentText, element: currentElement });
              currentText = "";
            }
            currentSubsection = currentElement.textContent.trim();
          } else if (currentElement.tagName === "P") {
            currentText += "" + currentElement.textContent.trim();
          }

          if (currentElement.tagName === "H3" || currentElement.tagName === "P") {
            // Use stable ID based on content instead of random timestamp
            uniqueClass = generateStableClassId(currentElement.textContent);
            existingClasses = currentElement.className.trim();
            currentElement.setAttribute("class", `${existingClasses} ${uniqueClass}`);
          }
        }

        if (currentSubsection !== "") {
          subsectionElements.set(currentSubsection, { text: currentText, element: currentElement });
        }

        window.subsections = {...window.subsections, ...Object.fromEntries(subsectionElements) };
      });
    };

    const extractKeywords = () => {
      const keywords = {};

      const keywordElements = Array.from(document.querySelectorAll("strong"));
      keywordElements.forEach((element) => {
        let identifier = element.getAttribute("data-identifier");
        if (!identifier) {
          // Use stable ID based on the keyword text content
          identifier = generateStableClassId(element.textContent);
          element.setAttribute("data-identifier", identifier);
        }
        const keyword = element.textContent.trim().replace(/[.,!?;:]$/g, '');
        const parentElement = element.parentNode;
        const textContent = Array.from(parentElement.childNodes).reduce(
          (accumulator, node) => {
            if (node.nodeType === Node.TEXT_NODE) {
              return accumulator + node.textContent.trim();
            }
            return accumulator;
          },
          ""
        );
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