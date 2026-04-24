// reference_renderer.js - Reference rendering system for STREAMDOWN markdown
// Handles footnote references [^ref-1] with tooltips and scroll-to-element functionality

/**
 * Parse markdown text to extract footnote references and definitions
 * @param {string} markdownText - The markdown content to parse
 * @returns {Object} - Object containing references and definitions
 */
function parseReferences(markdownText) {
  if (!markdownText || typeof markdownText !== 'string') {
    return { references: [], definitions: {} };
  }

  const references = [];
  const definitions = {};

  // Extract footnote references [^ref-1]
  const referenceRegex = /\[\^([^\]]+)\]/g;
  let match;
  while ((match = referenceRegex.exec(markdownText)) !== null) {
    const refId = match[1];
    if (!references.includes(refId)) {
      references.push(refId);
    }
  }

  // Extract footnote definitions [^ref-1]: definition text
  // But only if they appear before the %%% marker (to avoid processing the duplicate definitions at the bottom)
  const percentIndex = markdownText.indexOf('%%%');
  const contentToProcess = percentIndex !== -1 ? markdownText.substring(0, percentIndex) : markdownText;
  
  const definitionRegex = /\[\^([^\]]+)\]:\s*([^\n]+(?:\n(?!\[\^)[^\n]*)*)/g;
  while ((match = definitionRegex.exec(contentToProcess)) !== null) {
    const refId = match[1];
    const definition = match[2].trim();
    definitions[refId] = definition;
  }

  return { references, definitions };
}

/**
 * Generate reference links from HTML content
 * @param {string} htmlContent - The HTML content to process
 * @param {Object} definitions - Object containing reference definitions
 * @returns {string} - HTML with reference links
 */
function generateReferenceLinks(htmlContent, definitions) {
  if (!htmlContent || typeof htmlContent !== 'string') {
    return htmlContent;
  }


  // Check if we're in a browser environment
  if (typeof document === 'undefined') {
    // Fallback for non-browser environments - use regex replacement
    const result = htmlContent.replace(/\[\^([^\]]+)\]/g, (match, refId) => {
      const definition = definitions[refId];
      if (definition) {
        const linkHTML = createReferenceLinkHTML(refId, definition);
        return linkHTML;
      }
      return match; // Keep original if no definition found
    });
    return result;
  }


  // Create a temporary container to work with the HTML
  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = htmlContent;

  // Find all text nodes and process them
  const textNodes = [];
  const walker = document.createTreeWalker(
    tempDiv,
    NodeFilter.SHOW_TEXT,
    null,
    false
  );

  let node;
  while (node = walker.nextNode()) {
    if (node.textContent.trim()) {
      textNodes.push(node);
    }
  }

  // Process each text node for references
  for (const textNode of textNodes) {
    const text = textNode.textContent;
    const referenceRegex = /\[\^([^\]]+)\]/g;
    
    if (referenceRegex.test(text)) {
      const processedHTML = text.replace(referenceRegex, (match, refId) => {
        const definition = definitions[refId];
        if (definition) {
          return createReferenceLink(refId, definition);
        }
        return match; // Keep original if no definition found
      });

      if (processedHTML !== text) {
        const span = document.createElement('span');
        span.innerHTML = processedHTML;
        textNode.parentNode.replaceChild(span, textNode);
      }
    }
  }

  return tempDiv.innerHTML;
}

/**
 * Create a reference link HTML (for non-DOM environments)
 * @param {string} refId - The reference ID
 * @param {string} definition - The reference definition
 * @returns {string} - HTML for the reference link
 */
function createReferenceLinkHTML(refId, definition) {
  const linkId = `ref-link-${refId}`;
  
  // Extract paragraph ID and URL from definition
  const paragraphIdMatch = definition.match(/\(id=([^:)]+)(?::url=([^)]+))?\)/);
  const paragraphId = paragraphIdMatch ? paragraphIdMatch[1] : null;
  const sourceUrl = paragraphIdMatch ? paragraphIdMatch[2] : null;
  
  // Extract just the content without the ID part and clean it up
  // Handle both old format (id=...) and new format (id=...:url=...)
  let cleanDefinition = definition.replace(/^\(id=[^)]+(?::url=[^)]+)?\)\s*/, '');
  
  // If the definition appears to be just a URL (starts with http), 
  // it means the parsing failed and we should use a fallback
  if (cleanDefinition.startsWith('http') || cleanDefinition.trim() === '') {
    cleanDefinition = `Reference content from ${sourceUrl || 'external source'}`;
  }
  
  // Truncate very long definitions and remove unwanted content
  cleanDefinition = cleanDefinition
    .replace(/\n\n### Follow-up questions:[\s\S]*$/g, '') // Remove follow-up questions
    .replace(/\n\n### Diagram:[\s\S]*$/g, '') // Remove diagram sections
    .replace(/\n\n---\n---\n---[\s\S]*$/g, '') // Remove separator lines
    .replace(/\n\n::: loader[\s\S]*$/g, '') // Remove loader sections
    .trim();
  
  // Truncate if still too long (more than 200 characters)
  if (cleanDefinition.length > 200) {
    cleanDefinition = cleanDefinition.substring(0, 200) + '...';
  }
  
  // Build data attributes for cross-page navigation
  const dataAttributes = [
    `data-ref-id="${refId}"`,
    `data-definition="${escapeHtml(cleanDefinition)}"`
  ];
  
  if (paragraphId) {
    dataAttributes.push(`data-paragraph-id="${paragraphId}"`);
  }
  
  if (sourceUrl) {
    dataAttributes.push(`data-source-url="${escapeHtml(sourceUrl)}"`);
  }
  
  return `<span class="reference-container" style="position: relative; display: inline; white-space: nowrap;"><a href="javascript:void(0)" class="reference-pill" id="${linkId}" ${dataAttributes.join(' ')} onclick="handleReferenceClick('${refId}', event); return false;" onmouseenter="showReferenceTooltip('${refId}', event, '${escapeHtml(cleanDefinition)}')" onmouseleave="hideReferenceTooltip('${refId}')" style="display: inline; background: #e5e7eb; color: #374151; padding: 0.125rem 0.25rem; border-radius: 0.375rem; font-size: 0.625rem; font-weight: 600; cursor: pointer; transition: all 0.2s ease; text-decoration: none; margin: 0 0.125rem; border: 1px solid transparent; vertical-align: super; line-height: 1; white-space: nowrap;" onmouseover="this.style.background='#d1d5db'; this.style.transform='scale(1.1)'" onmouseout="this.style.background='#e5e7eb'; this.style.transform='scale(1)'">${refId.replace('ref-', '')}</a></span>`;
}

/**
 * Create a reference link element (for DOM environments)
 * @param {string} refId - The reference ID
 * @param {string} definition - The reference definition
 * @returns {string} - HTML for the reference link
 */
function createReferenceLink(refId, definition) {
  // Use the same HTML structure as the non-DOM version
  return createReferenceLinkHTML(refId, definition);
}

/**
 * Escape HTML characters
 * @param {string} text - Text to escape
 * @returns {string} - Escaped text
 */
function escapeHtml(text) {
  if (typeof document !== 'undefined') {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  } else {
    // Fallback for non-DOM environments
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
}

/**
 * Handle reference link clicks
 * @param {string} refId - The reference ID
 */
function handleReferenceClick(refId, event) {
  console.log(`🔍 handleReferenceClick called with refId: ${refId}`);
  
  // Prevent any default link behavior
  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }
  
  // Extract paragraph ID and source URL from definition if available
  const linkElement = document.getElementById(`ref-link-${refId}`);
  console.log(`🔍 Looking for link element with ID: ref-link-${refId}`);
  console.log(`🔍 Link element found:`, linkElement);
  
  if (linkElement) {
    const definition = linkElement.getAttribute('data-definition');
    const paragraphId = linkElement.getAttribute('data-paragraph-id');
    const sourceUrl = linkElement.getAttribute('data-source-url');
    const currentUrl = window.location.href;
    
    console.log(`🔍 Link element found, definition: ${definition}`);
    console.log(`🔍 Paragraph ID: ${paragraphId}`);
    console.log(`🔍 Source URL: ${sourceUrl}`);
    console.log(`🔍 Current URL: ${currentUrl}`);
    
    // Check if this is a cross-page reference
    console.log(`🔍 URL comparison: sourceUrl="${sourceUrl}", currentUrl="${currentUrl}"`);
    
    // Normalize URLs for comparison (remove trailing slashes, query params, etc.)
    const normalizeUrl = (url) => {
      if (!url) return '';
      return url.replace(/\/$/, '').split('?')[0].split('#')[0];
    };
    
    const normalizedSourceUrl = normalizeUrl(sourceUrl);
    const normalizedCurrentUrl = normalizeUrl(currentUrl);
    
    console.log(`🔍 Normalized URLs: source="${normalizedSourceUrl}", current="${normalizedCurrentUrl}"`);
    
    if (sourceUrl && normalizedSourceUrl !== normalizedCurrentUrl) {
      console.log(`🔍 Cross-page reference detected: ${normalizedSourceUrl} !== ${normalizedCurrentUrl}`);
      try {
        const safeUrl = new URL(sourceUrl, window.location.origin);
        const isHttpProtocol = safeUrl.protocol === 'http:' || safeUrl.protocol === 'https:';
        const isSameOrigin = safeUrl.origin === window.location.origin;
        if (!isHttpProtocol || !isSameOrigin) {
          console.warn(`⚠️ Blocked unsafe cross-page navigation URL: ${sourceUrl}`);
          return;
        }
        if (paragraphId) {
          console.log(`🔍 Navigating to ${safeUrl.toString()} with scroll-to=${paragraphId}`);
          safeUrl.searchParams.set('scroll-to', paragraphId);
        } else {
          console.log(`🔍 No paragraph ID available for cross-page navigation`);
        }
        window.location.href = safeUrl.toString();
      } catch (e) {
        console.warn(`⚠️ Invalid cross-page navigation URL: ${sourceUrl}`, e);
      }
      return;
    }
    
    // Same-page reference - use existing logic
    if (definition && definition.length > 10) {
      if (paragraphId) {
        console.log(`🔍 Same-page reference with paragraph ID: ${paragraphId}`);
        scrollToParagraph(paragraphId);
        return;
      }
      
      // If no paragraph ID, try to find paragraph by matching definition text
      console.log(`🔍 No paragraph ID found, searching by definition text...`);
      const targetParagraph = findParagraphByText(definition);
      if (targetParagraph) {
        console.log(`🔍 Found paragraph by definition text: ${targetParagraph.getAttribute('data-fuzzy-id')}`);
        scrollToParagraph(targetParagraph.getAttribute('data-fuzzy-id'));
        return;
      } else {
        console.log(`🔍 No paragraph found by definition text, falling back to refId search`);
      }
    } else {
      console.log(`🔍 No valid definition found, falling back to refId search`);
    }
  } else {
    console.log(`🔍 No link element found, falling back to refId search`);
  }
  
  // If no paragraph ID found in definition, try to scroll to the reference itself
  console.log(`🔍 No paragraph ID found, trying to scroll to reference: ${refId}`);
  scrollToParagraph(refId);
}

/**
 * Find a paragraph by matching text content
 * @param {string} searchText - The text to search for
 * @returns {HTMLElement|null} - The matching paragraph element or null
 */
function findParagraphByText(searchText) {
  if (!searchText || searchText.length < 10) {
    return null;
  }
  
  console.log(`🔍 Searching for paragraph containing text: ${searchText.substring(0, 100)}...`);
  
  // Clean the search text - remove ID markers and extra whitespace
  const cleanSearchText = searchText
    .replace(/\(id=[^)]+\)/g, '') // Remove ID markers
    .replace(/\s+/g, ' ') // Normalize whitespace
    .trim();
  
  if (cleanSearchText.length < 10) {
    return null;
  }
  
  // Get all paragraphs with data-fuzzy-id
  const paragraphs = document.querySelectorAll('[data-fuzzy-id]');
  console.log(`🔍 Searching through ${paragraphs.length} paragraphs...`);
  
  let bestMatch = null;
  let bestScore = 0;
  
  for (const paragraph of paragraphs) {
    const paragraphText = paragraph.textContent || '';
    const cleanParagraphText = paragraphText.replace(/\s+/g, ' ').trim();
    
    // Calculate similarity score
    const score = calculateTextSimilarity(cleanSearchText, cleanParagraphText);
    
    if (score > bestScore && score > 0.3) { // Minimum 30% similarity
      bestScore = score;
      bestMatch = paragraph;
      console.log(`🔍 Found potential match (${Math.round(score * 100)}%): ${paragraph.getAttribute('data-fuzzy-id')}`);
      console.log(`🔍 Paragraph text: ${cleanParagraphText.substring(0, 100)}...`);
    }
  }
  
  if (bestMatch) {
    console.log(`🔍 Best match found with ${Math.round(bestScore * 100)}% similarity`);
  } else {
    console.log(`🔍 No suitable match found`);
  }
  
  return bestMatch;
}

/**
 * Calculate text similarity using simple word overlap
 * @param {string} text1 - First text
 * @param {string} text2 - Second text
 * @returns {number} - Similarity score between 0 and 1
 */
function calculateTextSimilarity(text1, text2) {
  const words1 = new Set(text1.toLowerCase().split(/\s+/).filter(w => w.length > 2));
  const words2 = new Set(text2.toLowerCase().split(/\s+/).filter(w => w.length > 2));
  
  if (words1.size === 0 || words2.size === 0) {
    return 0;
  }
  
  const intersection = new Set([...words1].filter(x => words2.has(x)));
  const union = new Set([...words1, ...words2]);
  
  return intersection.size / union.size;
}

/**
 * Show reference tooltip
 * @param {string} refId - The reference ID
 * @param {Event} event - Mouse event
 * @param {string} definition - The tooltip content
 */
function showReferenceTooltip(refId, event, definition) {
  // Clear any existing hide timeout
  if (window.tooltipHideTimeout) {
    clearTimeout(window.tooltipHideTimeout);
    window.tooltipHideTimeout = null;
  }
  
  // Create or get the global tooltip element
  let tooltip = document.getElementById('global-reference-tooltip');
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.id = 'global-reference-tooltip';
    tooltip.style.cssText = `
      position: fixed;
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      padding: 0.75rem;
      max-width: 300px;
      z-index: 10000;
      font-size: 0.875rem;
      line-height: 1.4;
      display: none;
      pointer-events: auto;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    `;
    document.body.appendChild(tooltip);
  }
  
  // Get reference link element to check for cross-page navigation
  const linkElement = document.getElementById(`ref-link-${refId}`);
  const sourceUrl = linkElement ? linkElement.getAttribute('data-source-url') : null;
  const currentUrl = window.location.href;
  const isCrossPage = sourceUrl && sourceUrl !== currentUrl;
  
  // Update tooltip content
  tooltip.innerHTML = `
    <div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">
      Reference ${refId.replace('ref-', '')}
    </div>
    <div style="color: #6b7280; margin-bottom: 0.75rem;">
      ${definition}
    </div>
    <a href="javascript:void(0)" 
       onclick="handleReferenceClick('${refId}', event); return false;"
       style="
         color: #3b82f6;
         text-decoration: none;
         font-size: 0.75rem;
         font-weight: 500;
         display: inline-flex;
         align-items: center;
         gap: 0.25rem;
       "
       onmouseover="this.style.textDecoration='underline'"
       onmouseout="this.style.textDecoration='none'">
      ${isCrossPage ? 'Go to page →' : 'Scroll to paragraph →'}
    </a>
  `;
  
  // Add hover events to the tooltip to keep it visible
  tooltip.onmouseenter = () => {
    if (window.tooltipHideTimeout) {
      clearTimeout(window.tooltipHideTimeout);
      window.tooltipHideTimeout = null;
    }
  };
  
  tooltip.onmouseleave = () => {
    tooltip.style.display = 'none';
  };
  
  // Position tooltip relative to the mouse cursor
  const mouseX = event.clientX;
  const mouseY = event.clientY;
  
  // Show tooltip first to get dimensions
  tooltip.style.display = 'block';
  tooltip.style.visibility = 'hidden';
  
  // Get tooltip dimensions
  const tooltipRect = tooltip.getBoundingClientRect();
  const tooltipWidth = tooltipRect.width;
  const tooltipHeight = tooltipRect.height;
  
  // Calculate position
  let left = mouseX - (tooltipWidth / 2);
  let top = mouseY - tooltipHeight - 10;
  
  // Adjust if tooltip goes off screen
  if (left < 8) left = 8;
  if (left + tooltipWidth > window.innerWidth - 8) {
    left = window.innerWidth - tooltipWidth - 8;
  }
  if (top < 8) {
    // Show below cursor if no space above
    top = mouseY + 10;
  }
  
  // Set final position
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
  tooltip.style.visibility = 'visible';
}

/**
 * Hide reference tooltip
 * @param {string} refId - The reference ID
 */
function hideReferenceTooltip(refId) {
  // Add a delay before hiding to allow mouse to move to tooltip
  window.tooltipHideTimeout = setTimeout(() => {
    const tooltip = document.getElementById('global-reference-tooltip');
    if (tooltip) {
      tooltip.style.display = 'none';
    }
  }, 300); // 300ms delay
}

/**
 * Scroll to a paragraph by its ID
 * @param {string} refId - The reference ID (e.g., "ref-2")
 */
function scrollToParagraph(refId) {
  console.log(`🔍 ScrollToParagraph called with refId: ${refId}`);
  
  // Extract the reference number from refId (e.g., "ref-2" -> "2")
  const refNumber = refId.replace('ref-', '');
  console.log(`🔍 Looking for reference number: ${refNumber}`);
  
  // Find all paragraphs with data-fuzzy-id that might contain this reference
  const paragraphs = document.querySelectorAll('[data-fuzzy-id]');
  console.log(`🔍 Found ${paragraphs.length} paragraphs with data-fuzzy-id`);
  
  // Also look for paragraphs that might contain the reference number in their text
  const allParagraphs = document.querySelectorAll('p, div, section, article');
  console.log(`🔍 Found ${allParagraphs.length} total paragraphs/containers`);
  
  // Look for a paragraph that contains the reference number in its text or nearby elements
  let targetParagraph = null;
  let bestMatch = null;
  
  // First, try to find a paragraph that contains the reference number in its text
  for (const paragraph of allParagraphs) {
    const paragraphText = paragraph.textContent || '';
    const paragraphId = paragraph.getAttribute('data-fuzzy-id') || 'no-id';
    
    // Check if this paragraph contains the reference number
    if (paragraphText.includes(`[^ref-${refNumber}]`) || paragraphText.includes(`ref-${refNumber}`)) {
      console.log(`🔍 Found paragraph with reference ${refNumber}: ${paragraphId}`);
      bestMatch = paragraph;
      break;
    }
  }
  
  // If no direct match, try to find by paragraph ID if the refId contains an ID
  if (!bestMatch) {
    console.log(`🔍 No direct reference match found, trying to find by paragraph ID...`);
    
    // Check if the refId might be a paragraph ID itself
    const directElement = document.querySelector(`[data-fuzzy-id="${refId}"]`);
    if (directElement) {
      console.log(`🔍 Found direct element with data-fuzzy-id="${refId}"`);
      bestMatch = directElement;
    } else {
      // Try to find by partial ID match (in case there are slight differences)
      console.log(`🔍 No exact match, trying partial ID match...`);
      for (const paragraph of paragraphs) {
        const paragraphId = paragraph.getAttribute('data-fuzzy-id');
        if (paragraphId && paragraphId.includes(refId)) {
          console.log(`🔍 Found partial match: ${paragraphId}`);
          bestMatch = paragraph;
          break;
        }
      }
    }
  }
  
  // If still no match, look for substantial paragraphs with data-fuzzy-id
  if (!bestMatch) {
    console.log(`🔍 No direct reference match found, looking for substantial paragraphs...`);
    
    // Try to find a paragraph that might contain the reference text
    for (const paragraph of paragraphs) {
      const paragraphText = paragraph.textContent || '';
      const paragraphId = paragraph.getAttribute('data-fuzzy-id');
      
      // Check if this paragraph contains content that might be referenced
      if (paragraphText.length > 50) { // Only consider substantial paragraphs
        console.log(`🔍 Checking substantial paragraph ${paragraphId}: ${paragraphText.substring(0, 100)}...`);
        
        // Look for any mention of the reference in the paragraph text
        if (paragraphText.toLowerCase().includes(refId.toLowerCase()) || 
            paragraphText.toLowerCase().includes(`ref-${refNumber}`.toLowerCase())) {
          console.log(`🔍 Found paragraph containing reference text: ${paragraphId}`);
          bestMatch = paragraph;
          break;
        }
        
        // Don't use fallback substantial paragraph - it's usually wrong
        // Only use it if we have no other options and the refId looks like a paragraph ID
        if (!targetParagraph && refId.startsWith('p-')) {
          console.log(`🔍 Using fallback substantial paragraph for paragraph ID: ${paragraphId}`);
          targetParagraph = paragraph;
        }
      }
    }
  }
  
  const finalTarget = bestMatch || targetParagraph;
  
  if (finalTarget) {
    const paragraphId = finalTarget.getAttribute('data-fuzzy-id') || 'no-id';
    const paragraphText = finalTarget.textContent || '';
    console.log(`🔍 Scrolling to paragraph: ${paragraphId}`);
    console.log(`🔍 Target paragraph text preview: ${paragraphText.substring(0, 200)}...`);
    console.log(`🔍 Target element:`, finalTarget);
    
    // Warn if we're using a fallback target that might be wrong
    if (!bestMatch && targetParagraph) {
      console.warn(`⚠️ Using fallback substantial paragraph - this might not be the correct target!`);
      console.warn(`⚠️ Original refId: ${refId}, Target: ${paragraphId}`);
      
      // Check if this looks like a reasonable match
      if (!paragraphText.toLowerCase().includes(refId.toLowerCase()) && 
          !paragraphText.toLowerCase().includes(`ref-${refNumber}`.toLowerCase())) {
        console.error(`❌ This paragraph doesn't contain the reference text - scrolling might be incorrect!`);
        // Don't scroll if it's clearly wrong
        return;
      }
    }
    
    // Clear any existing fuzzy highlights first
    document.querySelectorAll('.fuzzy-highlight').forEach(el => {
      el.classList.remove('fuzzy-highlight', 'animate');
    });
    
    // Add fuzzy-highlight class for consistency with fuzzy match system
    finalTarget.classList.add('fuzzy-highlight');
    
    // Check if element is visible and in DOM
    const rect = finalTarget.getBoundingClientRect();
    console.log(`🔍 Element position:`, {
      top: rect.top,
      left: rect.left,
      width: rect.width,
      height: rect.height,
      visible: rect.width > 0 && rect.height > 0
    });
    
    // Use scrollIntoView with smooth behavior like the fuzzy match system
    try {
      finalTarget.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
      console.log(`🔍 scrollIntoView called successfully`);
      
      // Fallback: if scrollIntoView doesn't work, try manual scrolling
      setTimeout(() => {
        const currentRect = finalTarget.getBoundingClientRect();
        if (currentRect.top < 0 || currentRect.top > window.innerHeight) {
          console.log(`🔍 Element not visible after scrollIntoView, trying manual scroll`);
          const scrollTop = window.pageYOffset + currentRect.top - 100;
          window.scrollTo({
            top: scrollTop,
            behavior: 'smooth'
          });
        }
      }, 100);
      
    } catch (error) {
      console.error(`🔍 Error calling scrollIntoView:`, error);
      // Fallback to manual scrolling
      const rect = finalTarget.getBoundingClientRect();
      const scrollTop = window.pageYOffset + rect.top - 100;
      window.scrollTo({
        top: scrollTop,
        behavior: 'smooth'
      });
    }
    
    // Add animation class after a brief delay
    setTimeout(() => {
      finalTarget.classList.add('animate');
    }, 100);
    
    // Remove highlight after 3 seconds
    setTimeout(() => {
      finalTarget.classList.remove('fuzzy-highlight', 'animate');
      console.log(`🔍 Removed highlight from paragraph: ${paragraphId}`);
    }, 3000);
  } else {
    console.warn(`🔍 No suitable paragraph found for reference ${refId}`);
  }
}

/**
 * Process markdown content to add reference rendering
 * @param {string} markdownText - The markdown content
 * @param {string} htmlContent - The processed HTML content
 * @returns {string} - HTML with reference rendering
 */
export function processReferences(markdownText, htmlContent) {
  try {
    console.log('🔍 processReferences called with:', { markdownText: markdownText?.substring(0, 200), htmlContent: htmlContent?.substring(0, 200) });
    
    // Parse references from markdown
    const { references, definitions } = parseReferences(markdownText);
    console.log('🔍 Parsed references:', { references, definitions });
    
    if (references.length === 0) {
      console.log('🔍 No references found, returning original HTML');
      return htmlContent; // No references to process
    }
    
    if (Object.keys(definitions).length === 0) {
      console.log('🔍 No definitions found, returning original HTML');
      return htmlContent; // No definitions to process
    }
    
    // Generate reference links
    let processedHTML = generateReferenceLinks(htmlContent, definitions);
    console.log('🔍 Generated reference links, processed HTML length:', processedHTML.length);
    
    // Remove any remaining footnote definitions from the HTML to prevent duplicates
    // This removes patterns like [^ref-1]: definition that might still be in the HTML
    processedHTML = processedHTML.replace(/\[\^([^\]]+)\]:\s*[^\n]+/g, '');
    
    // Also remove any remaining footnote definitions that might span multiple lines
    processedHTML = processedHTML.replace(/\[\^([^\]]+)\]:\s*[^\n]+(?:\n[^\n\[\^]*)*/g, '');
    
    // Remove any standalone footnote definitions that might be in paragraphs
    processedHTML = processedHTML.replace(/<p>\s*\[\^([^\]]+)\]:\s*[^<]*<\/p>/g, '');
    
    // Remove any remaining footnote definitions that might be in the HTML as text nodes
    // Look for patterns like "1: (id=...)" that appear as standalone text
    processedHTML = processedHTML.replace(/\b\d+:\s*\(id=[^)]+\)[^<]*/g, '');
    
    // Remove patterns like "2: (id=p-1757052753202-fy5b75n8y)" that appear in the text
    processedHTML = processedHTML.replace(/\d+:\s*\(id=[^)]+\)/g, '');
    
    // Remove any remaining standalone footnote definitions that might be in divs or other containers
    processedHTML = processedHTML.replace(/<[^>]*>\s*\d+:\s*\(id=[^)]+\)[^<]*<\/[^>]*>/g, '');
    
    return processedHTML;
  } catch (error) {
    console.error('Error processing references:', error);
    return htmlContent; // Return original content on error
  }
}

/**
 * Add reference rendering styles to the document
 * @param {HTMLElement} container - Container to add styles to
 */
export function addReferenceStyles(container) {
  const styleId = 'reference-renderer-styles';
  
  // Check if styles already exist
  if (container.querySelector(`#${styleId}`)) {
    return;
  }
  
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = `
    /* Reference pill styles */
    .reference-pill {
      display: inline;
      background: #e5e7eb;
      color: #374151;
      padding: 0.125rem 0.25rem;
      border-radius: 0.375rem;
      font-size: 0.625rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      text-decoration: none;
      margin: 0 0.125rem;
      border: 1px solid transparent;
      vertical-align: super;
      line-height: 1;
    }
    
    .reference-pill:hover {
      background: #d1d5db;
      transform: scale(1.1);
    }
    
    /* Cross-page reference styles */
    .reference-pill[data-source-url]:not([data-source-url=""]) {
      background: #dbeafe;
      color: #1e40af;
      border: 1px solid #93c5fd;
    }
    
    .reference-pill[data-source-url]:not([data-source-url=""]):hover {
      background: #bfdbfe;
      transform: scale(1.1);
    }
    
    /* Cross-page indicator */
    .reference-pill[data-source-url]:not([data-source-url=""])::after {
      content: "↗";
      font-size: 0.5rem;
      margin-left: 0.125rem;
      opacity: 0.7;
    }
    
    .reference-pill:focus {
      outline: 2px solid #3b82f6;
      outline-offset: 2px;
    }
    
    /* Tooltip styles */
    .reference-tooltip {
      position: fixed;
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      padding: 0.75rem;
      max-width: 300px;
      z-index: 10000;
      font-size: 0.875rem;
      line-height: 1.4;
      pointer-events: none;
    }
    
    .reference-tooltip::after {
      content: '';
      position: absolute;
      top: 100%;
      left: 50%;
      transform: translateX(-50%);
      border: 5px solid transparent;
      border-top-color: white;
    }
    
    /* Paragraph highlight styles */
    .paragraph-highlight {
      background: rgba(59, 130, 246, 0.1) !important;
      border-left: 3px solid #3b82f6 !important;
      transition: all 0.3s ease !important;
      animation: highlightPulse 0.5s ease-in-out;
    }
    
    @keyframes highlightPulse {
      0% { background: rgba(59, 130, 246, 0.2); }
      50% { background: rgba(59, 130, 246, 0.15); }
      100% { background: rgba(59, 130, 246, 0.1); }
    }
    
    /* Fuzzy highlight styles - consistent with fuzzy match system */
    @keyframes highlightFade {
      0% { background-color: rgba(59, 130, 246, 0.2); }
      50% { background-color: rgba(59, 130, 246, 0.4); }
      100% { background-color: rgba(59, 130, 246, 0.1); }
    }
    
    .fuzzy-highlight {
      background-color: rgba(59, 130, 246, 0.1);
      scroll-margin-top: 100px;
    }
    
    .fuzzy-highlight.animate {
      animation: highlightFade 2s ease-out;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
      .reference-pill {
        background: #374151;
        color: #e5e7eb;
      }
      
      .reference-pill:hover {
        background: #4b5563;
      }
      
      .reference-tooltip {
        background: #1f2937;
        border-color: #374151;
        color: #e5e7eb;
      }
      
      .reference-tooltip::after {
        border-top-color: #1f2937;
      }
    }
  `;
  
  container.appendChild(style);
}

/**
 * Initialize reference rendering system
 * @param {HTMLElement} shadowElement - The shadow DOM element
 */
export function initializeReferenceRenderer(shadowElement) {
  if (!shadowElement) {
    console.warn('Shadow element not provided for reference renderer');
    return;
  }
  
  // Add reference styles
  addReferenceStyles(shadowElement);
  
  // Make functions globally available for inline event handlers
  if (typeof window !== 'undefined') {
    window.handleReferenceClick = handleReferenceClick;
    window.showReferenceTooltip = showReferenceTooltip;
    window.hideReferenceTooltip = hideReferenceTooltip;
    window.scrollToParagraph = scrollToParagraph;
  }
  
  console.log('Reference renderer initialized');
}

/**
 * Clean up reference rendering system
 */
export function cleanupReferenceRenderer() {
  // Remove global functions
  if (typeof window !== 'undefined') {
    delete window.handleReferenceClick;
    delete window.showReferenceTooltip;
    delete window.hideReferenceTooltip;
    delete window.scrollToParagraph;
  }
  
  // Remove styles
  const styleElement = document.getElementById('reference-renderer-styles');
  if (styleElement) {
    styleElement.remove();
  }
}