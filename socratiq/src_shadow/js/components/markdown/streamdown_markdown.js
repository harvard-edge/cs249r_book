// streamdown_markdown.js - STREAMDOWN-powered markdown with custom features
// This file uses STREAMDOWN's core libraries for powerful streaming markdown

import mermaid from 'mermaid';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import markdownit from 'markdown-it';
import markdownItContainer from 'markdown-it-container';
import { htmlLoader } from './loader';
import { findClosestAIMessage } from '../../libs/utils/utils.js';
import { containsWordReference } from '../../libs/diagram/mermaid';
import { generateUniqueId } from '../../libs/utils/utils.js';
import { initializeReferenceRenderer } from './reference_renderer.js';

let shadowEle;
let mermaidInitialized = false;

// ===== STREAMDOWN CORE FUNCTIONALITY =====

// Initialize STREAMDOWN's core libraries
async function initializeStreamdown() {
  // Initialize Mermaid
  if (!mermaidInitialized) {
    mermaid.initialize({
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'strict',
      fontFamily: 'monospace',
      suppressErrorRendering: true
    });
    mermaidInitialized = true;
  }

}

// STREAMDOWN's incomplete markdown parsing (from their source)
function parseIncompleteMarkdown(text) {
  if (!text || typeof text !== 'string') return text;

  // Handle incomplete code blocks
  const codeBlockMatches = text.match(/```[\s\S]*$/);
  if (codeBlockMatches && !text.includes('```', text.lastIndexOf('```') + 3)) {
    text += '\n```';
  }

  // Handle incomplete bold/italic
  const boldMatches = text.match(/\*\*[^*]*$/);
  if (boldMatches && !text.includes('**', text.lastIndexOf('**') + 2)) {
    text += '**';
  }

  const italicMatches = text.match(/\*[^*]*$/);
  if (italicMatches && !text.includes('*', text.lastIndexOf('*') + 1)) {
    text += '*';
  }

  // Handle incomplete inline code
  const codeMatches = text.match(/`[^`]*$/);
  if (codeMatches && !text.includes('`', text.lastIndexOf('`') + 1)) {
    text += '`';
  }

  // Handle incomplete links
  const linkMatches = text.match(/(!?\[)([^\]]*?)$/);
  if (linkMatches) {
    if (linkMatches[1].startsWith('!')) {
      const lastIndex = text.lastIndexOf(linkMatches[1]);
      text = text.substring(0, lastIndex);
    } else {
      text = `${text}](streamdown:incomplete-link)`;
    }
  }

  // Handle incomplete math
  const mathMatches = text.match(/\$\$[\s\S]*$/);
  if (mathMatches && (text.match(/\$\$/g) || []).length % 2 === 1) {
    text += '$$';
  }

  return text;
}

// STREAMDOWN's markdown parsing with streaming support
function parseMarkdownWithStreaming(text, parseIncomplete = true) {
  if (parseIncomplete) {
    text = parseIncompleteMarkdown(text);
  }

  // Use markdown-it for parsing (already imported for custom containers)
  return md.render(text);
}

// Code highlighting using plain pre/code with language class (compatible with any CSS theme)
function highlightCode(code, language) {
  const escaped = code
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  const langClass = language ? ` class="language-${language}"` : '';
  return `<pre><code${langClass}>${escaped}</code></pre>`;
}

// STREAMDOWN's Mermaid rendering
async function renderMermaid(chart, containerId) {
  if (!mermaidInitialized) {
    await initializeStreamdown();
  }

  try {
    const id = `mermaid-${containerId}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const { svg } = await mermaid.render(id, chart);
    return svg;
  } catch (error) {
    console.error('Mermaid rendering error:', error);
    return `<div class="mermaid-error" style="
      background-color: #f8d7da;
      border: 1px solid #f5c6cb;
      color: #721c24;
      padding: 1rem;
      border-radius: 4px;
      margin: 1rem 0;
    ">
      <strong>Mermaid Error:</strong> ${error.message}
      <details style="margin-top: 0.5rem;">
        <summary style="cursor: pointer; color: #721c24;">Show Code</summary>
        <pre style="margin-top: 0.5rem; background: #f5c6cb; padding: 0.5rem; border-radius: 3px; font-size: 0.875rem;">${chart}</pre>
      </details>
    </div>`;
  }
}

// STREAMDOWN's math rendering
function renderMath(text) {
  try {
    // Handle block math ($$...$$)
    text = text.replace(/\$\$([\s\S]*?)\$\$/g, (match, math) => {
      try {
        return katex.renderToString(math.trim(), { displayMode: true });
      } catch (error) {
        return `<span class="math-error">Math Error: ${error.message}</span>`;
      }
    });

    // Handle inline math ($...$)
    text = text.replace(/\$([^$]+)\$/g, (match, math) => {
      try {
        return katex.renderToString(math.trim(), { displayMode: false });
      } catch (error) {
        return `<span class="math-error">Math Error: ${error.message}</span>`;
      }
    });

    return text;
  } catch (error) {
    console.warn('Math rendering error:', error);
    return text;
  }
}

// ===== CUSTOM CONTAINERS (Our Unique Features) =====

// Create a lightweight markdown-it instance ONLY for custom containers
const md = markdownit({
  html: true,
});

// 1. Spoiler container with editable input
md.use(markdownItContainer, 'spoiler', {
  validate: function(params) {
    return params.trim().match(/^spoiler\s+(.*)$/);
  },
  render: function(tokens, idx, options, env, self) {
    const m = tokens[idx].info.trim().match(/^spoiler\s+(.*)$/);
    
    if (tokens[idx].nesting === 1) {
      let content = '';
      let i = idx + 1;
      while (i < tokens.length && tokens[i].type !== 'container_spoiler_close') {
        if (tokens[i].type === 'inline') {
          content += tokens[i].content;
        }
        i++;
      }

      return `<details style="
        background-color: rgba(13, 110, 253, 0.05);
        border: 1px solid rgba(13, 110, 253, 0.2);
        border-radius: 4px;
        padding: 0.5rem;
        margin-bottom: 1.5rem;
      "><summary style="
        cursor: pointer;
        color: #0d6efd;
        font-weight: 500;
        padding: 0.25rem;
      ">${md.utils.escapeHtml(m[1])}</summary>
      <div class="editable-container" style="padding: 0.5rem;">
        <div class="editable-text" contenteditable="true" style="
          min-height: 1.5em;
          padding: 0.25rem;
          border-radius: 3px;
          transition: background-color 0.2s;
        ">${md.utils.escapeHtml(content.trim())}</div>
        <div style="
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-top: 0.5rem;
          font-size: 0.875rem;
          color: #6c757d;
        ">
          <button class="enter-button" style="
            padding: 0.25rem 0.5rem;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            font-size: 0.75rem;
            color: #6c757d;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.25rem;
          " title="Press Enter to submit">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 12h18m-5-5l5 5-5 5"/>
            </svg>
            <span>Enter to resubmit</span>
          </button>
        </div>
      </div></details>`;
    }
    return '';
  }
});

// 2. Info container
md.use(markdownItContainer, 'info', {
  validate: function(params) {
    return params.trim().match(/^info\s*(.*)$/);
  },
  render: function(tokens, idx) {
    if (tokens[idx].nesting === 1) {
      return `<div class="info-box" style="
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-left: 4px solid #0d6efd;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #1f2937;
      ">
        <div style="
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.25rem;
        ">
          <span style="font-size: 1.25rem;">ℹ</span>
          <strong>Information</strong>
        </div>
        <div style="margin-left: 1.75rem;">`;
    } else {
      return '</div></div>';
    }
  }
});

// 3. Network warning container
md.use(markdownItContainer, 'network-warning', {
  validate: function(params) {
    return params.trim().match(/^network-warning\s*(.*)$/);
  },
  render: function(tokens, idx) {
    if (tokens[idx].nesting === 1) {
      return `<div class="network-warning-box" style="
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-left: 4px solid #ffc107;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #856404;
      ">
        <div style="
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.25rem;
        ">
          <span style="font-size: 1.25rem;">⚠️</span>
          <strong>Network Warning</strong>
        </div>
        <div style="margin-left: 1.75rem;">`;
    } else {
      return '</div></div>';
    }
  }
});

// 4. Loader container
md.use(markdownItContainer, 'loader', {
  render: function (tokens, idx) {
    if (tokens[idx].nesting === 1) {
      return htmlLoader;
    } else {
      return '';
    }
  }
});

// 5. Custom Reference Syntax Plugin
// Handles {{ref:target-id:display-text}} syntax
md.use(function(md) {
  // Add inline rule for custom references
  md.inline.ruler.before('text', 'custom_ref', function(state, silent) {
    const start = state.pos;
    const max = state.posMax;
    
    // Look for {{ref: pattern
    if (state.src.charCodeAt(start) !== 0x7B /* { */) return false;
    if (state.src.charCodeAt(start + 1) !== 0x7B /* { */) return false;
    
    
    // Find the closing }}
    let pos = start + 2;
    let found = false;
    while (pos < max - 1) {
      if (state.src.charCodeAt(pos) === 0x7D /* } */ && 
          state.src.charCodeAt(pos + 1) === 0x7D /* } */) {
        found = true;
        break;
      }
      pos++;
    }
    
    if (!found) return false;
    
    const content = state.src.slice(start + 2, pos);
    
    // Parse the reference: ref:target-id:display-text:source-url or ref:target-id:display-text or ref:target-id
    // Handle both 2, 3, and 4 part formats
    if (!content.startsWith('ref:')) {
      return false;
    }
    
    const refContent = content.substring(4); // Remove 'ref:' prefix
    const parts = refContent.split(':');
    
    let targetId, displayText, sourceUrl;
    
    if (parts.length === 1) {
      // Format: ref:target-id
      targetId = parts[0];
      displayText = targetId;
      sourceUrl = null;
    } else if (parts.length === 2) {
      // Format: ref:target-id:display-text
      targetId = parts[0];
      displayText = parts[1];
      sourceUrl = null;
    } else if (parts.length >= 3) {
      // Format: ref:target-id:display-text:source-url
      targetId = parts[0];
      displayText = parts[1];
      sourceUrl = parts.slice(2).join(':'); // Join remaining parts in case URL contains colons
      
        // Custom reference parsed successfully
    }
    
    
    if (silent) return true;
    
    // Create token
    const token = state.push('custom_ref', '', 0);
    token.content = content;
    token.targetId = targetId;
    token.displayText = displayText;
    token.sourceUrl = sourceUrl;
    token.markup = '{{}}';
    
    state.pos = pos + 2;
    return true;
  });
  
  // Add renderer for custom references
  md.renderer.rules.custom_ref = function(tokens, idx, options, env, self) {
    const token = tokens[idx];
    const targetId = token.targetId;
    const displayText = token.displayText;
    const sourceUrl = token.sourceUrl;
    
    
    // Generate unique ID for this reference
    const refId = `custom-ref-${targetId}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // Try to get the actual paragraph content for the tooltip
    let actualContent = displayText; // Default to displayText
    try {
      // Try to find the target element to get its actual content
      const targetElement = document.querySelector(`[data-fuzzy-id="${targetId}"]`);
      if (targetElement && targetElement.textContent) {
        actualContent = targetElement.textContent.trim();
      } else {
      }
    } catch (error) {
    }
    
    // Build data attributes
    const dataAttributes = [
      `data-target-id="${targetId}"`,
      `data-display-text="${md.utils.escapeHtml(displayText)}"`,
      `data-quoted-text="${md.utils.escapeHtml(actualContent)}"` // Store the actual quoted text as attribute
    ];
    
    if (sourceUrl) {
      dataAttributes.push(`data-source-url="${md.utils.escapeHtml(sourceUrl)}"`);
    }
    
    return `<span class="custom-reference-container" style="position: relative; display: inline-block; max-width: 100%;">
      <a href="javascript:void(0)" 
         class="custom-reference-pill" 
         id="${refId}" 
         ${dataAttributes.join(' ')}
         onclick="handleCustomReferenceClick('${targetId}', event); return false;" 
         onmouseenter="showCustomReferenceTooltip('${targetId}', event, '${md.utils.escapeHtml(displayText)}')" 
         onmouseleave="hideCustomReferenceTooltip('${targetId}')" 
         style="
           display: inline-block; 
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
           vertical-align: middle; 
           line-height: 1.2; 
           word-wrap: break-word;
           word-break: break-word;
           max-width: 100%;
         " 
         onmouseover="this.style.background='#d1d5db'; this.style.transform='scale(1.1)'" 
         onmouseout="this.style.background='#e5e7eb'; this.style.transform='scale(1)'">
        ${md.utils.escapeHtml(displayText)}
      </a>
    </span>`;
  };
});

// ===== CUSTOM REFERENCE HANDLERS =====

/**
 * Handle custom reference clicks
 * @param {string} targetId - The target ID to scroll to
 */
function handleCustomReferenceClick(targetId, event) {
  console.log(`🔍 handleCustomReferenceClick called with targetId: ${targetId}`);
  
  // Prevent any default link behavior
  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }
  
  // Hide any open tooltips first
  const tooltip = shadowEle ? shadowEle.querySelector('#global-custom-reference-tooltip') : document.getElementById('global-custom-reference-tooltip');
  if (tooltip) {
    tooltip.style.display = 'none';
  }
  
  // Find the reference link element to check for cross-page navigation
  // Search in shadow DOM where the reference links are actually located
  const referenceLinks = shadowEle ? shadowEle.querySelectorAll(`[data-target-id="${targetId}"]`) : document.querySelectorAll(`[data-target-id="${targetId}"]`);
  
  if (referenceLinks.length > 0) {
    const linkElement = referenceLinks[0];
    const sourceUrl = linkElement.getAttribute('data-source-url');
    const currentUrl = window.location.href;
    
    const quotedText = linkElement.getAttribute('data-quoted-text');
    console.log(`🔍 Reference click: targetId="${targetId}", sourceUrl="${sourceUrl}"`);
    
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
      console.log(`🔍 Navigating to ${sourceUrl} with scroll-to=${targetId}`);
      // Navigate to the source page with scroll parameter
      window.location.href = `${sourceUrl}?scroll-to=${targetId}`;
      return;
    } else if (!sourceUrl) {
      console.log(`❌ ERROR: No data-source-url found! Reference data may have been lost during save/load.`);
    } else {
      console.log(`🔍 Same-page reference detected: ${normalizedSourceUrl} === ${normalizedCurrentUrl}`);
      console.log(`🔍 Proceeding with same-page scroll to target: ${targetId}`);
    }
  } else {
    console.log(`🔍 No reference links found for targetId: ${targetId}`);
  }
  
  // Same-page reference - try to find the target element by various methods
  let targetElement = null;
  
  // Method 1: Direct ID match
  targetElement = document.querySelector(`[data-fuzzy-id="${targetId}"]`);
  if (targetElement) {
    console.log(`🔍 Found target by direct ID: ${targetId}`);
    scrollToCustomReference(targetElement);
    return;
  }
  
  // Method 2: Partial ID match
  const allElements = document.querySelectorAll('[data-fuzzy-id]');
  for (const element of allElements) {
    const elementId = element.getAttribute('data-fuzzy-id');
    if (elementId && elementId.includes(targetId)) {
      console.log(`🔍 Found target by partial ID match: ${elementId}`);
      targetElement = element;
      break;
    }
  }
  
  if (targetElement) {
    scrollToCustomReference(targetElement);
    return;
  }
  
  // Method 3: Text content search
  const paragraphs = document.querySelectorAll('p, div, section, article');
  for (const paragraph of paragraphs) {
    const text = paragraph.textContent || '';
    if (text.includes(targetId) || text.toLowerCase().includes(targetId.toLowerCase())) {
      console.log(`🔍 Found target by text content: ${targetId}`);
      targetElement = paragraph;
      break;
    }
  }
  
  if (targetElement) {
    scrollToCustomReference(targetElement);
  } else {
    console.warn(`🔍 No target found for custom reference: ${targetId}`);
  }
}

/**
 * Scroll to a custom reference target
 * @param {HTMLElement} targetElement - The element to scroll to
 */
function scrollToCustomReference(targetElement) {
  console.log(`🔍 Scrolling to custom reference target:`, targetElement);
  
  // Clear any existing highlights
  document.querySelectorAll('.custom-reference-highlight').forEach(el => {
    el.classList.remove('custom-reference-highlight', 'animate');
  });
  
  // Add highlight class
  targetElement.classList.add('custom-reference-highlight');
  
  // Scroll to element
  try {
    targetElement.scrollIntoView({
      behavior: 'smooth',
      block: 'center'
    });
    
    // Add animation class
    setTimeout(() => {
      targetElement.classList.add('animate');
    }, 100);
    
    // Remove highlight after 3 seconds
    setTimeout(() => {
      targetElement.classList.remove('custom-reference-highlight', 'animate');
    }, 3000);
    
  } catch (error) {
    console.error('Error scrolling to custom reference:', error);
  }
}

/**
 * Show custom reference tooltip
 * @param {string} targetId - The target ID
 * @param {Event} event - Mouse event
 * @param {string} displayText - The display text
 */
function showCustomReferenceTooltip(targetId, event, displayText) {
  
  // Clear any existing hide timeout
  if (window.customTooltipHideTimeout) {
    clearTimeout(window.customTooltipHideTimeout);
    window.customTooltipHideTimeout = null;
  }
  
  // First, try to get the content from the stored reference data
  let targetContent = '';
  let isCrossPage = false;
  
  // Find the reference link element to get stored data
  const referenceLinks = shadowEle ? shadowEle.querySelectorAll(`[data-target-id="${targetId}"]`) : document.querySelectorAll(`[data-target-id="${targetId}"]`);
  
  if (referenceLinks.length > 0) {
    const linkElement = referenceLinks[0];
    const sourceUrl = linkElement.getAttribute('data-source-url');
    const currentUrl = window.location.href;
    const quotedText = linkElement.getAttribute('data-quoted-text');
    
    // Check if this is a cross-page reference
    isCrossPage = sourceUrl && sourceUrl !== currentUrl;
    
    // Use the stored quoted text if available
    if (quotedText) {
      targetContent = quotedText;
    } else {
    }
  }
  
  // If no stored content, try to find the target element on current page
  if (!targetContent) {
    let targetElement = null;
    
    // Method 1: Direct ID match
    targetElement = document.querySelector(`[data-fuzzy-id="${targetId}"]`);
    if (targetElement) {
      targetContent = targetElement.textContent || '';
    } else {
      // Method 2: Partial ID match
      const allElements = document.querySelectorAll('[data-fuzzy-id]');
      for (const element of allElements) {
        const elementId = element.getAttribute('data-fuzzy-id');
        if (elementId && elementId.includes(targetId)) {
          targetElement = element;
          targetContent = element.textContent || '';
          break;
        }
      }
    }
  }
  
  // Truncate the content if it's too long
  let truncatedContent = targetContent;
  if (targetContent.length > 150) {
    truncatedContent = targetContent.substring(0, 150) + '...';
  }
  
  // Clean up the content - remove extra whitespace and newlines
  truncatedContent = truncatedContent.replace(/\s+/g, ' ').trim();
  
  // Create or get the global tooltip element in shadow DOM
  const container = shadowEle || document;
  let tooltip = container.querySelector('#global-custom-reference-tooltip');
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.id = 'global-custom-reference-tooltip';
    tooltip.style.cssText = `
      position: fixed;
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.5rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      padding: 0.75rem;
      max-width: 350px;
      z-index: 10000;
      font-size: 0.875rem;
      line-height: 1.4;
      display: none;
      pointer-events: auto;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    `;
    container.appendChild(tooltip);
  }
  
  // isCrossPage is already determined above
  
  // Update tooltip content
  const tooltipHTML = `
    <div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">
      ${displayText}
    </div>
    ${truncatedContent ? `
    <div style="color: #4b5563; margin-bottom: 0.75rem; font-style: italic; border-left: 3px solid #e5e7eb; padding-left: 0.5rem;">
      "${truncatedContent}"
    </div>
    ` : ''}
    <a href="#" 
       onclick="event.preventDefault(); handleCustomReferenceClick('${targetId}'); return false;"
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
      ${isCrossPage ? 'Go to page →' : 'Go to target →'}
    </a>
  `;
  
  tooltip.innerHTML = tooltipHTML;
  
  // Add hover events to the tooltip
  tooltip.onmouseenter = () => {
    if (window.customTooltipHideTimeout) {
      clearTimeout(window.customTooltipHideTimeout);
      window.customTooltipHideTimeout = null;
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
  
  // Calculate position - simplified for debugging
  let left = mouseX + 10; // Show to the right of cursor
  let top = mouseY - 10;  // Show above cursor
  
  // Adjust if tooltip goes off screen
  if (left + tooltipWidth > window.innerWidth - 8) {
    left = mouseX - tooltipWidth - 10; // Show to the left of cursor
  }
  if (top < 8) {
    top = mouseY + 10; // Show below cursor
  }
  
  // Set final position
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
  tooltip.style.visibility = 'visible';
  
}

/**
 * Hide custom reference tooltip
 * @param {string} targetId - The target ID
 */
function hideCustomReferenceTooltip(targetId) {
  window.customTooltipHideTimeout = setTimeout(() => {
    const container = shadowEle || document;
    const tooltip = container.querySelector('#global-custom-reference-tooltip');
    if (tooltip) {
      tooltip.style.display = 'none';
    }
  }, 300);
}

// ===== CORE FUNCTIONS =====

export function initiateMarkdown(shadowElement) {
  shadowEle = shadowElement;
  // Initialize STREAMDOWN libraries
  initializeStreamdown();
  // Initialize reference renderer
  initializeReferenceRenderer(shadowElement);
  
  // Make custom reference functions globally available
  if (typeof window !== 'undefined') {
    window.handleCustomReferenceClick = handleCustomReferenceClick;
    window.showCustomReferenceTooltip = showCustomReferenceTooltip;
    window.hideCustomReferenceTooltip = hideCustomReferenceTooltip;
  }
  
}

// Content normalization (preserved from original)
function normalizeMarkdownText(text) {
  const lines = text.split('\n');
  
  // Find questions and %%% marker
  let questionLines = [];
  let percentLineIndex = -1;
  
  const isQuestion = (line) => {
    const cleanLine = line.replace(/^\d+\.\s*/, '')
                         .replace(/\*\*/g, '')
                         .trim();
    return cleanLine.endsWith('?');
  };
  
  // Scan from bottom to top
  let consecutiveQuestions = [];
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    
    if (line.includes('%%%')) {
      percentLineIndex = i;
      continue;
    }
    
    if (isQuestion(line)) {
      consecutiveQuestions.unshift(lines[i]);
    } else if (consecutiveQuestions.length > 0) {
      if (line === '' || line.includes('**')) {
        continue;
      }
      break;
    }
  }
  
  // If we found consecutive questions (2 or more)
  if (consecutiveQuestions.length >= 2) {
    let insertPoint = -1;
    for (let i = lines.length - consecutiveQuestions.length - 1; i >= 0; i--) {
      const line = lines[i].trim();
      if (line !== '' && !line.includes('**')) {
        insertPoint = i;
        break;
      }
    }
    
    if (insertPoint !== -1) {
      const mainContent = lines.slice(0, insertPoint + 1);
      
      while (mainContent[mainContent.length - 1]?.trim() === '') {
        mainContent.pop();
      }
      
      return [
        ...mainContent,
        '',
        '%%%',
        '',
        ...consecutiveQuestions
      ].join('\n');
    }
  }
  
  return text;
}

// Enhanced markdown processing with STREAMDOWN features
async function processMarkdownWithStreamdown(text, parseIncomplete = true) {
  // Normalize content
  const normalizedText = normalizeMarkdownText(text);
  
  // Parse with STREAMDOWN's incomplete markdown support
  let processedText = normalizedText;
  if (parseIncomplete) {
    processedText = parseIncompleteMarkdown(normalizedText);
  }

  // Process custom containers first
  const customContainerHtml = md.render(processedText);
  
  // Create a temporary container to process the HTML
  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = customContainerHtml;

  // Process code blocks with STREAMDOWN's highlighting
  const codeBlocks = tempDiv.querySelectorAll('pre code');
  for (const codeBlock of codeBlocks) {
    const code = codeBlock.textContent;
    const className = codeBlock.className;
    const languageMatch = className.match(/language-(\w+)/);
    const language = languageMatch ? languageMatch[1] : 'text';
    
    if (language === 'mermaid') {
      // Handle Mermaid diagrams
      const mermaidContainer = document.createElement('div');
      mermaidContainer.className = 'mermaid-container';
      mermaidContainer.innerHTML = `
        <div class="mermaid-loading">Loading diagram...</div>
      `;
      
      try {
        const svg = await renderMermaid(code, Date.now());
        mermaidContainer.innerHTML = `
          <div class="mermaid-diagram">
            ${svg}
          </div>
        `;
      } catch (error) {
        mermaidContainer.innerHTML = `
          <div class="mermaid-error">
            <strong>Mermaid Error:</strong> ${error.message}
            <details>
              <summary>Show Code</summary>
              <pre>${code}</pre>
            </details>
          </div>
        `;
      }
      
      codeBlock.parentElement.parentElement.replaceWith(mermaidContainer);
    } else {
      // Handle regular code blocks with STREAMDOWN's highlighting
      try {
        const highlightedHtml = highlightCode(code, language);
        const highlightedDiv = document.createElement('div');
        highlightedDiv.innerHTML = highlightedHtml;
        highlightedDiv.className = 'code-block-container';
        
        // Add copy button
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.innerHTML = '📋';
        copyButton.title = 'Copy code';
        copyButton.onclick = () => {
          navigator.clipboard.writeText(code);
          copyButton.innerHTML = '✅';
          setTimeout(() => copyButton.innerHTML = '📋', 2000);
        };
        
        highlightedDiv.appendChild(copyButton);
        codeBlock.parentElement.parentElement.replaceWith(highlightedDiv);
      } catch (error) {
        console.warn('Code highlighting failed:', error);
      }
    }
  }

  // Process math expressions
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
  
  for (const textNode of textNodes) {
    const processedText = renderMath(textNode.textContent);
    if (processedText !== textNode.textContent) {
      const mathDiv = document.createElement('span');
      mathDiv.innerHTML = processedText;
      textNode.parentNode.replaceChild(mathDiv, textNode);
    }
  }

  // Custom references are already processed by the markdown-it plugin above
  // No need to call processReferences from reference_renderer.js as it handles footnote references
  // and would interfere with our custom reference system

  return tempDiv.innerHTML;
}

// Main markdown update function using STREAMDOWN
export async function updateMarkdownPreview(text, clone, isResearch = 0, markdownPreviewId = '') {
  let preview;
  if (markdownPreviewId) {
    preview = clone.querySelector('#' + markdownPreviewId);
  } else {
    preview = clone.querySelector('#markdown-preview');
  }

  let contentToRender = text;
  let questions = [];

  // Handle %%% questions (preserved functionality)
  const percentIndex = text.indexOf('%%%');
  if (percentIndex !== -1) {
    const questionsText = text.substring(percentIndex + 3);
    questions = questionsText.split('\n')
      .map(sentence => sentence.trim())
      .filter(sentence => sentence.endsWith('?'));
    contentToRender = text.substring(0, percentIndex);
  }

  // Process with STREAMDOWN
  const htmlContent = await processMarkdownWithStreamdown(contentToRender, true);
  
  // Create the final HTML
  const finalHtml = `
    <div class="streamdown-wrapper">
      <style>
        /* STREAMDOWN-inspired styles */
        .streamdown-wrapper {
          line-height: 1.6;
          color: #1f2937;
          word-wrap: break-word;
          word-break: break-word;
          overflow-wrap: break-word;
        }
        .streamdown-wrapper pre, .streamdown-wrapper code {
          white-space: pre-wrap !important;
          word-wrap: break-word;
          word-break: break-word;
          overflow-wrap: break-word;
        }
        .streamdown-wrapper pre[style*="white-space: normal"] {
          white-space: pre-wrap !important;
        }
        .streamdown-wrapper pre.dark-mode[style*="white-space: normal"] {
          white-space: pre-wrap !important;
        }
        .streamdown-wrapper pre.dark-mode {
          white-space: pre-wrap !important;
          word-wrap: break-word;
          word-break: break-word;
          overflow-wrap: break-word;
        }
        .streamdown-wrapper pre[style*="white-space"] {
          white-space: pre-wrap !important;
        }
        .streamdown-wrapper pre code {
          white-space: pre-wrap !important;
          word-wrap: break-word;
          word-break: break-word;
          overflow-wrap: break-word;
        }
        .streamdown-wrapper pre {
          max-width: 100%;
          overflow-x: auto;
        }
        .streamdown-wrapper h3, .streamdown-wrapper h4 {
          word-wrap: break-word;
          word-break: break-word;
          overflow-wrap: break-word;
        }
        .streamdown-wrapper hr {
          margin: 1rem 0;
          border: none;
          border-top: 1px solid #e5e7eb;
        }
        .streamdown-wrapper * {
          max-width: 100%;
          box-sizing: border-box;
        }
        .streamdown-wrapper h1, .streamdown-wrapper h2, .streamdown-wrapper h3,
        .streamdown-wrapper h4, .streamdown-wrapper h5, .streamdown-wrapper h6 {
          margin-top: 1.5rem;
          margin-bottom: 0.5rem;
          font-weight: 600;
        }
        .streamdown-wrapper h1 { font-size: 1.875rem; }
        .streamdown-wrapper h2 { font-size: 1.5rem; }
        .streamdown-wrapper h3 { font-size: 1.25rem; }
        .streamdown-wrapper h4 { font-size: 1.125rem; }
        .streamdown-wrapper h5 { font-size: 1rem; }
        .streamdown-wrapper h6 { font-size: 0.875rem; }
        .streamdown-wrapper p { margin-bottom: 1rem; }
        .streamdown-wrapper ul, .streamdown-wrapper ol {
          margin-bottom: 1rem;
          padding-left: 1.5rem;
        }
        .streamdown-wrapper li { margin-bottom: 0.25rem; }
        .streamdown-wrapper blockquote {
          border-left: 4px solid #e5e7eb;
          padding-left: 1rem;
          margin: 1rem 0;
          color: #6b7280;
          font-style: italic;
        }
        .streamdown-wrapper table {
          border-collapse: collapse;
          width: 100%;
          margin: 1rem 0;
        }
        .streamdown-wrapper th, .streamdown-wrapper td {
          border: 1px solid #e5e7eb;
          padding: 0.5rem;
          text-align: left;
        }
        .streamdown-wrapper th {
          background-color: #f9fafb;
          font-weight: 600;
        }
        .code-block-container {
          position: relative;
          margin: 1rem 0;
        }
        .copy-button {
          position: absolute;
          top: 0.5rem;
          right: 0.5rem;
          background: #f3f4f6;
          border: 1px solid #d1d5db;
          border-radius: 0.25rem;
          padding: 0.25rem;
          cursor: pointer;
          font-size: 0.875rem;
        }
        .mermaid-container {
          margin: 1rem 0;
          text-align: center;
        }
        .mermaid-loading {
          padding: 2rem;
          color: #6b7280;
        }
        .mermaid-diagram {
          background: #f9fafb;
          border-radius: 0.5rem;
          padding: 1rem;
        }
        .mermaid-error {
          background: #fef2f2;
          border: 1px solid #fecaca;
          color: #dc2626;
          padding: 1rem;
          border-radius: 0.5rem;
          margin: 1rem 0;
        }
        .math-error {
          color: #dc2626;
          background: #fef2f2;
          padding: 0.125rem 0.25rem;
          border-radius: 0.25rem;
          font-size: 0.875rem;
        }
        /* Custom Reference Styles */
        .custom-reference-pill {
          display: inline-block;
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
          vertical-align: middle;
          line-height: 1.2;
          word-wrap: break-word;
          word-break: break-word;
          max-width: 100%;
        }
        .custom-reference-pill:hover {
          background: #d1d5db;
          transform: scale(1.1);
        }
        
        /* Cross-page custom reference styles */
        .custom-reference-pill[data-source-url]:not([data-source-url=""]) {
          background: #dbeafe;
          color: #1e40af;
          border: 1px solid #93c5fd;
        }
        
        .custom-reference-pill[data-source-url]:not([data-source-url=""]):hover {
          background: #bfdbfe;
          transform: scale(1.1);
        }
        
        /* Cross-page indicator for custom references */
        .custom-reference-pill[data-source-url]:not([data-source-url=""])::after {
          content: "↗";
          font-size: 0.5rem;
          margin-left: 0.125rem;
          opacity: 0.7;
        }
        
        .custom-reference-highlight {
          background-color: rgba(59, 130, 246, 0.1);
          scroll-margin-top: 100px;
        }
        .custom-reference-highlight.animate {
          animation: customHighlightFade 2s ease-out;
        }
        @keyframes customHighlightFade {
          0% { background-color: rgba(59, 130, 246, 0.2); }
          50% { background-color: rgba(59, 130, 246, 0.4); }
          100% { background-color: rgba(59, 130, 246, 0.1); }
        }
      </style>
      ${htmlContent}
    </div>
  `;

  preview.innerHTML = finalHtml;

  // Initialize editable inputs (preserved functionality)
  initializeSpoilerEditing(preview);

  // Add questions section (preserved functionality)
  if (questions.length > 0) {
    addQuestionsToPreview(preview, questions, contentToRender);
  }
}

// Initialize spoiler editing (preserved functionality)
export function initializeSpoilerEditing(preview) {
  const editableContainers = preview.querySelectorAll('.editable-container');
  
  editableContainers.forEach(container => {
    const editableText = container.querySelector('.editable-text');
    const enterButton = container.querySelector('.enter-button');
    
    if (!editableText || !enterButton) return;

    // Add tooltip-style hint
    editableText.setAttribute('title', 'Click to edit, press Enter to submit');
    
    // Handle enter key press
    editableText.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitEditedText(editableText.textContent, e);
      }
    });
    
    // Handle button click
    enterButton.addEventListener('click', (e) => {
      submitEditedText(editableText.textContent, e);
    });
    
    // Add visual cues
    editableText.addEventListener('mouseover', () => {
      editableText.style.backgroundColor = 'rgba(13, 110, 253, 0.1)';
      editableText.style.cursor = 'text';
    });
    
    editableText.addEventListener('mouseout', () => {
      editableText.style.backgroundColor = '';
    });
    
    editableText.addEventListener('focus', () => {
      editableText.style.outline = '2px solid rgba(13, 110, 253, 0.2)';
      editableText.style.backgroundColor = 'rgba(13, 110, 253, 0.05)';
    });
    
    editableText.addEventListener('blur', () => {
      editableText.style.outline = '';
      editableText.style.backgroundColor = '';
    });
  });
}

// Submit edited text (preserved functionality)
function submitEditedText(text, e) {
  const aiMessageElement = findClosestAIMessage(e.target);
  
  if (!aiMessageElement) {
    console.warn('Could not find parent AI message element');
    return;
  }

  let diagramId = '';
  if (containsWordReference(text)) {
    diagramId = generateUniqueId();
    generateDiagramId("aiActionCompleted", text, "mermaid_diagram", diagramId);
  }

  const event = new CustomEvent('aiActionCompleted', {
    detail: {
      text,
      type: "query",
      ele: aiMessageElement,
      fromRightClickMenu: false,
      source: {
        type: 'editableText',
        containerId: aiMessageElement.id
      }
    }
  });
  window.dispatchEvent(event);
}

// Add questions to preview (preserved functionality)
function addQuestionsToPreview(preview, questions, contentToRender) {
  const relatedDiv = document.createElement('div');
  relatedDiv.innerHTML = `
    <p style="display: flex; align-items: center;margin-top: 10px;margin-bottom: 4px;">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-git-pull-request-create-icon lucide-git-pull-request-create" style="margin-right: 5px; height: 24px;width: 24px;margin-bottom: 1px;"><circle cx="6" cy="6" r="3"/><path d="M6 9v12"/><path d="M13 6h3a2 2 0 0 1 2 2v3"/><path d="M18 15v6"/><path d="M21 18h-6"/></svg>
      <strong>Related</strong>
    </p>`;

  questions.forEach((question, i) => {
    if (i !== 0) {
      const hr = document.createElement('hr');
      relatedDiv.appendChild(hr);
    }
    const button = document.createElement('button');
    button.innerHTML = `+ ${removeLeadingNumberAndAsterisks(question)}`;
    button.style.backgroundColor = 'transparent';
    button.style.border = 'none';
    button.classList.add('followup-button');
    button.style.display = 'block';
    button.style.marginBottom = '5px';
    button.style.color = "royalblue";
    button.style.textAlign = "left";
    button.addEventListener('click', function(event) {
      const prompt = `Use this question: ${question} to expand upon this content ${contentToRender}`;
      general_agent(prompt);
    });
    relatedDiv.appendChild(button);
  });

  preview.appendChild(relatedDiv);
}

// Helper functions (preserved)
function removeLeadingNumberAndAsterisks(text) {
  let cleanedText = text.replace(/^\d+\.\s*/, '');
  cleanedText = cleanedText.replace(/\*/g, '');
  return cleanedText;
}

function general_agent(text, links = [window.location.href]) {
  const event = new CustomEvent("aiActionCompleted", {
    detail: {
      text,
      type: "general",
      links: links,
    },
  });
  window.dispatchEvent(event);
}

function generateDiagramId(type_of_custom_event, text, type_of_action, id) {
  const event = new CustomEvent(type_of_custom_event, {
    detail: {
      text: text,
      type: type_of_action,
      diagramId: id,
    },
  });
  window.dispatchEvent(event);
}

// Reinitialize editable inputs (preserved functionality)
export function reinitializeEditableInputs(shadowRoot) {
  const editableContainers = shadowRoot.querySelectorAll('.user-input-resubmit');
  
  editableContainers.forEach(container => {
    const editableText = container.querySelector('.editable-text');
    const enterButton = container.querySelector('.enter-button');
    
    if (!editableText || !enterButton) return;

    // Remove existing event listeners
    const newEditableText = editableText.cloneNode(true);
    const newEnterButton = enterButton.cloneNode(true);
    editableText.parentNode.replaceChild(newEditableText, editableText);
    enterButton.parentNode.replaceChild(newEnterButton, enterButton);
    
    // Reinitialize with new elements
    initializeSpoilerEditing(container);
  });
}

// ===== STREAMDOWN STREAMING RENDERER =====
// Enhanced streaming renderer using STREAMDOWN's core functionality

export class StreamdownMarkdownRenderer {
  constructor(shadowElement) {
    this.shadowElement = shadowElement;
    this.partialContent = '';
    this.isStreaming = false;
    this.streamingQueue = [];
  }

  // Start streaming with STREAMDOWN
  async startStream(markdownPreviewId = '') {
    this.isStreaming = true;
    this.partialContent = '';
    this.streamingQueue = [];
    
    // Initialize STREAMDOWN libraries
    await initializeStreamdown();
    
    const preview = this.shadowElement.querySelector('#' + markdownPreviewId || '#markdown-preview');
    if (preview) {
      preview.innerHTML = '<div class="streaming-loader">Streaming content...</div>';
    }
  }

  // Add chunk with STREAMDOWN's parseIncompleteMarkdown feature
  async addChunk(chunk, markdownPreviewId = '') {
    if (!this.isStreaming) return;
    
    this.partialContent += chunk;
    this.streamingQueue.push(chunk);
    
    // Use STREAMDOWN's incomplete markdown parsing
    const processedContent = parseIncompleteMarkdown(this.partialContent);
    
    // Render with STREAMDOWN features
    await this.renderWithStreamdown(processedContent, markdownPreviewId);
  }

  // Render with STREAMDOWN features
  async renderWithStreamdown(content, markdownPreviewId = '') {
    const preview = this.shadowElement.querySelector('#' + markdownPreviewId || '#markdown-preview');
    if (!preview) return;

    try {
      // Process with STREAMDOWN
      const htmlContent = await processMarkdownWithStreamdown(content, true);
      
      // Create the final HTML
      const finalHtml = `
        <div class="streamdown-wrapper">
          <style>
            /* STREAMDOWN-inspired styles */
            .streamdown-wrapper {
              line-height: 1.6;
              color: #1f2937;
              word-wrap: break-word;
              word-break: break-word;
              overflow-wrap: break-word;
            }
            .streamdown-wrapper pre, .streamdown-wrapper code {
              white-space: pre-wrap !important;
              word-wrap: break-word;
              word-break: break-word;
              overflow-wrap: break-word;
            }
            .streamdown-wrapper pre[style*="white-space: normal"] {
              white-space: pre-wrap !important;
            }
            .streamdown-wrapper pre.dark-mode[style*="white-space: normal"] {
              white-space: pre-wrap !important;
            }
            .streamdown-wrapper pre.dark-mode {
              white-space: pre-wrap !important;
              word-wrap: break-word;
              word-break: break-word;
              overflow-wrap: break-word;
            }
            .streamdown-wrapper pre[style*="white-space"] {
              white-space: pre-wrap !important;
            }
            .streamdown-wrapper pre code {
              white-space: pre-wrap !important;
              word-wrap: break-word;
              word-break: break-word;
              overflow-wrap: break-word;
            }
            .streamdown-wrapper pre {
              max-width: 100%;
              overflow-x: auto;
            }
            .streamdown-wrapper h3, .streamdown-wrapper h4 {
              word-wrap: break-word;
              word-break: break-word;
              overflow-wrap: break-word;
            }
            .streamdown-wrapper hr {
              margin: 1rem 0;
              border: none;
              border-top: 1px solid #e5e7eb;
            }
            .streamdown-wrapper * {
              max-width: 100%;
              box-sizing: border-box;
            }
            .streamdown-wrapper h1, .streamdown-wrapper h2, .streamdown-wrapper h3,
            .streamdown-wrapper h4, .streamdown-wrapper h5, .streamdown-wrapper h6 {
              margin-top: 1.5rem;
              margin-bottom: 0.5rem;
              font-weight: 600;
            }
            .streamdown-wrapper h1 { font-size: 1.875rem; }
            .streamdown-wrapper h2 { font-size: 1.5rem; }
            .streamdown-wrapper h3 { font-size: 1.25rem; }
            .streamdown-wrapper h4 { font-size: 1.125rem; }
            .streamdown-wrapper h5 { font-size: 1rem; }
            .streamdown-wrapper h6 { font-size: 0.875rem; }
            .streamdown-wrapper p { margin-bottom: 1rem; }
            .streamdown-wrapper ul, .streamdown-wrapper ol {
              margin-bottom: 1rem;
              padding-left: 1.5rem;
            }
            .streamdown-wrapper li { margin-bottom: 0.25rem; }
            .streamdown-wrapper blockquote {
              border-left: 4px solid #e5e7eb;
              padding-left: 1rem;
              margin: 1rem 0;
              color: #6b7280;
              font-style: italic;
            }
            .streamdown-wrapper table {
              border-collapse: collapse;
              width: 100%;
              margin: 1rem 0;
            }
            .streamdown-wrapper th, .streamdown-wrapper td {
              border: 1px solid #e5e7eb;
              padding: 0.5rem;
              text-align: left;
            }
            .streamdown-wrapper th {
              background-color: #f9fafb;
              font-weight: 600;
            }
            .code-block-container {
              position: relative;
              margin: 1rem 0;
            }
            .copy-button {
              position: absolute;
              top: 0.5rem;
              right: 0.5rem;
              background: #f3f4f6;
              border: 1px solid #d1d5db;
              border-radius: 0.25rem;
              padding: 0.25rem;
              cursor: pointer;
              font-size: 0.875rem;
            }
            .mermaid-container {
              margin: 1rem 0;
              text-align: center;
            }
            .mermaid-loading {
              padding: 2rem;
              color: #6b7280;
            }
            .mermaid-diagram {
              background: #f9fafb;
              border-radius: 0.5rem;
              padding: 1rem;
            }
            .mermaid-error {
              background: #fef2f2;
              border: 1px solid #fecaca;
              color: #dc2626;
              padding: 1rem;
              border-radius: 0.5rem;
              margin: 1rem 0;
            }
            .math-error {
              color: #dc2626;
              background: #fef2f2;
              padding: 0.125rem 0.25rem;
              border-radius: 0.25rem;
              font-size: 0.875rem;
            }
            /* Custom Reference Styles */
            .custom-reference-pill {
              display: inline-block;
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
              vertical-align: middle;
              line-height: 1.2;
              word-wrap: break-word;
              word-break: break-word;
              max-width: 100%;
            }
            .custom-reference-pill:hover {
              background: #d1d5db;
              transform: scale(1.1);
            }
            .custom-reference-highlight {
              background-color: rgba(59, 130, 246, 0.1);
              scroll-margin-top: 100px;
            }
            .custom-reference-highlight.animate {
              animation: customHighlightFade 2s ease-out;
            }
            @keyframes customHighlightFade {
              0% { background-color: rgba(59, 130, 246, 0.2); }
              50% { background-color: rgba(59, 130, 246, 0.4); }
              100% { background-color: rgba(59, 130, 246, 0.1); }
            }
          </style>
          ${htmlContent}
        </div>
      `;

      preview.innerHTML = finalHtml;
      
      // Reinitialize custom features
      this.reinitializeCustomFeatures(preview);
      
    } catch (error) {
      console.warn('Error rendering streaming content:', error);
      preview.innerHTML = `<pre>${this.partialContent}</pre>`;
    }
  }

  // Reinitialize custom features
  reinitializeCustomFeatures(preview) {
    // Initialize editable inputs
    initializeSpoilerEditing(preview);
    
    // Handle question processing if %%% marker is complete
    this.processQuestionsIfComplete(preview);
  }

  // Process questions only when %%% marker is complete
  processQuestionsIfComplete(preview) {
    if (this.partialContent.includes('%%%')) {
      const percentIndex = this.partialContent.indexOf('%%%');
      const questionsText = this.partialContent.substring(percentIndex + 3);
      
      if (questionsText.trim() && questionsText.includes('?')) {
        const questions = questionsText.split('\n')
          .map(sentence => sentence.trim())
          .filter(sentence => sentence.endsWith('?'));
        
        if (questions.length > 0) {
          addQuestionsToPreview(preview, questions, this.partialContent.substring(0, percentIndex));
        }
      }
    }
  }

  // Complete streaming
  async completeStream(markdownPreviewId = '') {
    this.isStreaming = false;
    
    // Final render with complete content
    await this.renderWithStreamdown(this.partialContent, markdownPreviewId);
    
    // Clear streaming queue
    this.streamingQueue = [];
  }

  // Handle streaming errors
  handleStreamingError(error, markdownPreviewId = '') {
    this.isStreaming = false;
    
    const preview = this.shadowElement.querySelector('#' + markdownPreviewId || '#markdown-preview');
    if (preview) {
      preview.innerHTML = `
        <div class="streaming-error" style="
          background-color: #f8d7da;
          border: 1px solid #f5c6cb;
          color: #721c24;
          padding: 1rem;
          border-radius: 4px;
          margin: 1rem 0;
        ">
          <strong>Streaming Error:</strong> ${error.message}
          <br><br>
          <strong>Partial Content:</strong>
          <pre style="background: white; padding: 0.5rem; border-radius: 3px; margin-top: 0.5rem;">${this.partialContent}</pre>
        </div>
      `;
    }
  }
}

// StreamdownMarkdownRenderer is already exported as a class declaration above