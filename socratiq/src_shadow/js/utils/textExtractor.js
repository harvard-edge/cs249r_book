// textExtractor.js
import { TEXT_EXTRACTION_CONFIG } from '../../configs/client.config.js';

const READABLE_BOOK_EXCLUDE_SELECTOR = [
  'nav',
  'aside:not(.column-margin)',
  'header',
  'footer',
  'script',
  'style',
  'noscript',
  'form',
  'button',
  'input',
  'select',
  'textarea',
  '#socratiq-reader',
  '#widget-chat-container',
  '#collaborative-widget-host',
  '.chapter-concept-map',
  '.mlsys-figure-tabs',
  '.mlsys-explore-panel',
  'textarea.pseudocode',
  'mjx-assistive-mml',
  '[data-socratiq-ignore]',
  '[hidden]',
  '[inert]',
  '[aria-hidden="true"]',
  ...(TEXT_EXTRACTION_CONFIG.excludeSelectors || TEXT_EXTRACTION_CONFIG.NAV_SELECTORS || [])
].filter((value, index, self) => value && self.indexOf(value) === index).join(', ');

/**
 * Estimates the number of tokens in a text string
 * Rough approximation: 1 token ≈ 4 characters for English text
 * @param {string} text - The text to estimate tokens for
 * @returns {number} - Estimated token count
 */
function estimateTokens(text) {
  if (!text) return 0;
  // Remove extra whitespace and count characters
  const cleanText = text.replace(/\s+/g, ' ').trim();
  return Math.ceil(cleanText.length / 4);
}

/**
 * Checks if an element should be excluded from text extraction
 * @param {Element} element - The DOM element to check
 * @returns {boolean} - True if element should be excluded
 */
function shouldExcludeElement(element) {
  if (!element || element.nodeType !== 1) return false;

  if (typeof element.matches === 'function' && element.matches(READABLE_BOOK_EXCLUDE_SELECTOR)) {
    return true;
  }

  // Check if element is hidden via computed styles when available
  const view = element.ownerDocument?.defaultView || (typeof window !== 'undefined' ? window : null);
  if (view && typeof view.getComputedStyle === 'function') {
    try {
      const style = view.getComputedStyle(element);
      if (
        style?.display === 'none' ||
        style?.visibility === 'hidden' ||
        style?.visibility === 'collapse' ||
        style?.contentVisibility === 'hidden' ||
        style?.opacity === '0'
      ) {
        return true;
      }
    } catch {
      // Ignore computed style errors in non-browser test environments
    }
  }

  // Check for common navigation attributes
  const role = element.getAttribute?.('role');
  if (role && ['navigation', 'banner', 'contentinfo'].includes(role)) {
    return true;
  }

  return false;
}

/**
 * Extracts readable book text from a DOM subtree, preserving MathJax and KaTeX
 * inline TeX ($...$) and filtering out navigation, widgets, and hidden chrome.
 * @param {Element} rootElement - The root element to extract text from
 * @param {Object} [options] - Optional extraction configuration
 * @param {boolean} [options.includeClosedDetails=false] - Whether to traverse closed <details> bodies
 * @returns {string} - Clean readable text
 */
export function extractReadableBookText(rootElement = typeof document !== 'undefined' ? document.body : null, options = {}) {
  if (!rootElement) return '';
  const { includeClosedDetails = false } = options;

  const mathMap = new Map();
  const win = rootElement.ownerDocument?.defaultView || (typeof window !== 'undefined' ? window : null);
  try {
    const mathItems = win?.MathJax?.startup?.document?.getMathItemsWithin?.(rootElement) || [];
    for (const item of mathItems) {
      const root = item?.typesetRoot?.nodeType === 1 ? item.typesetRoot : (item?.element?.nodeType === 1 ? item.element : null);
      const tex = typeof item?.math === 'string' ? item.math.trim() : (typeof item?.tex === 'string' ? item.tex.trim() : '');
      if (root && tex) {
        mathMap.set(root, tex);
      }
    }
  } catch {
    // MathJax may not be initialized yet
  }

  const parts = [];

  const visit = (node) => {
    if (!node) return;
    if (node.nodeType === 3) {
      parts.push(node.nodeValue || node.data || '');
      return;
    }
    if (node.nodeType !== 1) return;
    if (shouldExcludeElement(node)) return;

    if (mathMap.has(node)) {
      const tex = mathMap.get(node);
      if (tex) parts.push(` $${tex}$ `);
      return;
    }

    if (typeof node.matches === 'function' && node.matches('.katex, mjx-container')) {
      const annotation = node.querySelector?.('annotation[encoding="application/x-tex"]');
      const tex = annotation?.textContent?.trim();
      if (tex) {
        parts.push(` $${tex}$ `);
        return;
      }
      const assistive = node.querySelector?.('mjx-assistive-mml')?.textContent?.trim();
      if (assistive) {
        parts.push(` ${assistive} `);
        return;
      }
    }

    if (node.tagName === 'DETAILS' && !node.open && !includeClosedDetails) {
      const summary = Array.from(node.children || node.childNodes || []).find(
        child => child?.tagName === 'SUMMARY'
      ) || node.querySelector?.('summary');
      if (summary) visit(summary);
      return;
    }

    for (const child of node.childNodes || []) {
      visit(child);
    }
  };

  visit(rootElement);
  if (typeof rootElement.querySelectorAll === 'function') {
    const panels = [...rootElement.querySelectorAll('[data-mlsys-explore] .mlsys-explore-panel')];
    const pairs = [];
    for (const panel of panels) {
      if (panel.closest?.('form, fieldset, nav, aside, [hidden], [inert]')) continue;
      for (const ctrl of panel.querySelectorAll('input[type="range"], select')) {
        if (ctrl.closest?.('form, fieldset, [hidden], [inert]')) continue;
        let label = (ctrl.getAttribute('aria-label') || '').trim();
        if (!label && ctrl.id && ctrl.ownerDocument) {
          const explicit = ctrl.ownerDocument.querySelector(`label[for="${ctrl.id.replace(/"/g, '\\"')}"]`);
          if (explicit) label = explicit.textContent.trim();
        }
        if (!label) {
          const parentLabel = ctrl.closest('label');
          if (parentLabel) {
            label = [...parentLabel.childNodes]
              .filter(n => n !== ctrl && !(n.nodeType === 1 && n.contains(ctrl)))
              .map(n => n.textContent || '')
              .join(' ')
              .trim();
          }
        }
        if (!label && ctrl.previousElementSibling && /^(LABEL|SPAN)$/i.test(ctrl.previousElementSibling.tagName)) {
          label = ctrl.previousElementSibling.textContent.trim();
        }
        label = label.replace(/[:\s]+$/, '').replace(/\s+/g, ' ').trim();
        const rawValue = ctrl.tagName === 'SELECT'
          ? (ctrl.selectedOptions?.[0]?.textContent || ctrl.value || '')
          : (ctrl.value || '');
        const value = String(rawValue).replace(/\s+/g, ' ').trim();
        if (label && value) pairs.push(`${label} = ${value}`);
      }
    }
    if (pairs.length) {
      parts.push(` [Interactive figure parameters: ${pairs.join(', ')}] `);
    }
  }
  return parts.join('').replace(/\s+/g, ' ').trim();
}

/**
 * Extracts clean text content from an element using extractReadableBookText
 * @param {Element} element - Element to extract from
 * @param {Object} [options] - Optional extraction configuration
 * @returns {string} - Clean text content
 */
export function extractCleanText(element = typeof document !== 'undefined' ? document.body : null, options = {}) {
  if (!element) return '';
  const readable = extractReadableBookText(element, options);
  if (readable) return readable;
  return '';
}

/**
 * Extracts text content from a DOM element, excluding navigation components
 * @param {Element} rootElement - The root element to extract text from (defaults to document.body)
 * @returns {string} - Extracted text content
 */
export function extractTextContent(rootElement = document.body) {
  if (!rootElement) {
    console.warn('No root element provided for text extraction');
    return '';
  }

  return extractReadableBookText(rootElement);
}

/**
 * Extracts text content with source mapping for reference tracking
 * @param {Element} rootElement - The root element to extract text from (defaults to document.body)
 * @returns {Object} - Object containing text content and source mapping
 */
function extractTextContentWithSources(rootElement = document.body) {
  if (!rootElement) {
    console.warn('No root element provided for text extraction');
    return { text: '', sources: [] };
  }
  
  // Find content sections and extract with source mapping
  const sources = [];
  const contentSections = [];
  
  // Look for main content areas first
  const mainContentSelectors = TEXT_EXTRACTION_CONFIG.CONTENT_SELECTORS;
  let contentElements = [];
  
  for (const selector of mainContentSelectors) {
    const elements = rootElement.querySelectorAll(selector);
    if (elements.length > 0) {
      contentElements = Array.from(elements);
      break;
    }
  }
  
  // If no main content found, use the root element directly
  if (contentElements.length === 0) {
    contentElements = [rootElement];
  }
  
  // Process each content element
  contentElements.forEach((element, index) => {
    const elementId = element.id || `content-section-${index}`;
    const elementClass = element.className || '';
    const elementTag = element.tagName.toLowerCase();
    
    // Get clean readable text content from this element
    const textContent = extractReadableBookText(element);
    
    if (textContent.length > 50) { // Only include substantial content
      // Create source mapping
      const source = {
        sourceId: `source-${index}`,
        label: elementId || `${elementTag}-${index}`,
        content: textContent,
        pageUrl: window.location.href,
        domain: window.location.hostname,
        level: 'page',
        position: index,
        elementId: elementId,
        elementClass: elementClass,
        elementTag: elementTag
      };
      
      sources.push(source);
      contentSections.push(`## ${source.label}\n\n${source.content}`);
    }
  });
  
  // Combine all content sections
  const combinedText = contentSections.join('\n\n---\n\n');
  
  return {
    text: combinedText,
    sources: sources
  };
}

/**
 * Samples text content to fit within token limits using random distributed sampling
 * @param {string} text - The full text content
 * @param {number} maxTokens - Maximum number of tokens allowed
 * @returns {string} - Sampled text content
 */
function sampleTextContent(text, maxTokens = TEXT_EXTRACTION_CONFIG.MAX_TOKENS) {
  const currentTokens = estimateTokens(text);
  
  if (currentTokens <= maxTokens) {
    return text;
  }
  
  
  // Split text into sentences for better sampling
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  
  if (sentences.length === 0) {
    const samplingRatio = maxTokens / currentTokens;
    return text.substring(0, Math.floor(text.length * samplingRatio));
  }
  
  // Calculate how many sentences we can fit
  const avgTokensPerSentence = currentTokens / sentences.length;
  const targetSentenceCount = Math.floor(maxTokens / avgTokensPerSentence);
  
  // Ensure we don't try to sample more sentences than exist
  const actualTargetCount = Math.min(targetSentenceCount, sentences.length);
  
  // Random distributed sampling strategy
  const sampledSentences = randomDistributedSampling(sentences, actualTargetCount, maxTokens);
  
  return sampledSentences.join('. ') + '.';
}

/**
 * Performs random distributed sampling across the entire text
 * @param {Array} sentences - Array of sentences
 * @param {number} targetCount - Target number of sentences to sample
 * @param {number} maxTokens - Maximum tokens allowed
 * @returns {Array} - Array of sampled sentences
 */
function randomDistributedSampling(sentences, targetCount, maxTokens) {
  if (sentences.length <= targetCount) {
    return sentences;
  }
  
  const sampledSentences = [];
  let sampledTokens = 0;
  
  // Create indices for the entire sentence array
  const allIndices = Array.from({ length: sentences.length }, (_, i) => i);
  
  // Shuffle indices randomly
  const shuffledIndices = shuffleArray([...allIndices]);
  
  // Sample from shuffled indices
  for (const index of shuffledIndices) {
    const sentence = sentences[index].trim();
    const sentenceTokens = estimateTokens(sentence);
    
    if (sampledTokens + sentenceTokens <= maxTokens && sampledSentences.length < targetCount) {
      sampledSentences.push(sentence);
      sampledTokens += sentenceTokens;
    }
    
    // Stop if we've reached our target or token limit
    if (sampledSentences.length >= targetCount || sampledTokens >= maxTokens * 0.95) {
      break;
    }
  }
  
  // If we still have room and haven't sampled enough, try to add more
  if (sampledSentences.length < targetCount && sampledTokens < maxTokens * 0.9) {
    const remainingIndices = shuffledIndices.filter(i => !sampledSentences.includes(sentences[i]));
    
    for (const index of remainingIndices) {
      const sentence = sentences[index].trim();
      const sentenceTokens = estimateTokens(sentence);
      
      if (sampledTokens + sentenceTokens <= maxTokens) {
        sampledSentences.push(sentence);
        sampledTokens += sentenceTokens;
      }
      
      if (sampledTokens >= maxTokens * 0.95) {
        break;
      }
    }
  }
  
  return sampledSentences;
}

/**
 * Shuffles an array using Fisher-Yates algorithm
 * @param {Array} array - Array to shuffle
 * @returns {Array} - Shuffled array
 */
function shuffleArray(array) {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

/**
 * Main function to extract page text content for quiz generation
 * @param {Element} rootElement - Optional root element to extract from
 * @returns {Object} - Object containing extracted text and metadata
 */
export function extractPageTextForQuiz(rootElement = document.body) {
  try {
    
    // Extract text content
    const fullText = extractTextContent(rootElement);
    
    if (!fullText) {
      console.warn('No text content found on the page');
      return {
        text: '',
        tokens: 0,
        sampled: false,
        originalTokens: 0
      };
    }
    
    // Estimate tokens
    const originalTokens = estimateTokens(fullText);
    
    // Sample if necessary
    const finalText = sampleTextContent(fullText, TEXT_EXTRACTION_CONFIG.MAX_TOKENS);
    const finalTokens = estimateTokens(finalText);
    const wasSampled = finalTokens < originalTokens;
    
    console.log(`Text extraction complete:`, {
      originalTokens,
      finalTokens,
      sampled: wasSampled,
      textLength: finalText.length
    });
    
    return {
      text: finalText,
      tokens: finalTokens,
      sampled: wasSampled,
      originalTokens: originalTokens
    };
    
  } catch (error) {
    console.error('Error extracting page text:', error);
    return {
      text: '',
      tokens: 0,
      sampled: false,
      originalTokens: 0,
      error: error.message
    };
  }
}

/**
 * Main function to extract page text content with source mapping for cumulative quiz generation
 * @param {Element} rootElement - Optional root element to extract from
 * @returns {Object} - Object containing extracted text, sources, and metadata
 */
export function extractPageTextWithSourcesForQuiz(rootElement = document.body) {
  try {
    
    // Extract text content with source mapping
    const extractionResult = extractTextContentWithSources(rootElement);
    
    if (!extractionResult.text || extractionResult.sources.length === 0) {
      console.warn('No text content or sources found on the page');
      return {
        text: '',
        sources: [],
        tokens: 0,
        sampled: false,
        originalTokens: 0
      };
    }
    
    // Estimate tokens
    const originalTokens = estimateTokens(extractionResult.text);
    
    // Use source-aware sampling for better distribution
    const samplingResult = sampleTextContentWithSources(extractionResult.sources, TEXT_EXTRACTION_CONFIG.MAX_TOKENS);
    const finalText = samplingResult.text;
    const finalTokens = samplingResult.tokens;
    const wasSampled = finalTokens < originalTokens;
    
    console.log(`Text extraction with sources complete:`, {
      originalTokens,
      finalTokens,
      sampled: wasSampled,
      textLength: finalText.length,
      sourceCount: extractionResult.sources.length,
      sampledSourceCount: samplingResult.sampledSources.length
    });
    
    return {
      text: finalText,
      sources: samplingResult.sampledSources,
      tokens: finalTokens,
      sampled: wasSampled,
      originalTokens: originalTokens
    };
    
  } catch (error) {
    console.error('Error extracting page text with sources:', error);
    return {
      text: '',
      sources: [],
      tokens: 0,
      sampled: false,
      originalTokens: 0,
      error: error.message
    };
  }
}

/**
 * Samples text content from multiple sources ensuring distribution across sources
 * @param {Array} sources - Array of source objects with content
 * @param {number} maxTokens - Maximum tokens allowed
 * @returns {Object} - Object containing sampled text and sources
 */
function sampleTextContentWithSources(sources, maxTokens = TEXT_EXTRACTION_CONFIG.MAX_TOKENS) {
  if (!sources || sources.length === 0) {
    return { text: '', tokens: 0, sampledSources: [] };
  }
  
  // Calculate total tokens across all sources
  const totalTokens = sources.reduce((sum, source) => sum + estimateTokens(source.content), 0);
  
  if (totalTokens <= maxTokens) {
    // No sampling needed
    const combinedText = sources.map(source => `## ${source.label}\n\n${source.content}`).join('\n\n---\n\n');
    return {
      text: combinedText,
      tokens: totalTokens,
      sampledSources: sources
    };
  }
  
  
  // Calculate tokens per source and determine sampling strategy
  const sourceTokens = sources.map(source => ({
    ...source,
    tokens: estimateTokens(source.content)
  }));
  
  // Sort sources by token count (largest first) for better distribution
  sourceTokens.sort((a, b) => b.tokens - a.tokens);
  
  const sampledSources = [];
  let sampledTokens = 0;
  
  // Distribute sampling across sources
  const tokensPerSource = Math.floor(maxTokens / sources.length);
  const minTokensPerSource = Math.floor(tokensPerSource * 0.5); // Ensure minimum representation
  
  for (const source of sourceTokens) {
    const targetTokens = Math.min(source.tokens, tokensPerSource);
    
    if (sampledTokens + targetTokens <= maxTokens) {
      // Sample content from this source
      const sampledContent = sampleTextContent(source.content, targetTokens);
      const actualTokens = estimateTokens(sampledContent);
      
      if (sampledTokens + actualTokens <= maxTokens) {
        sampledSources.push({
          ...source,
          content: sampledContent,
          tokens: actualTokens
        });
        sampledTokens += actualTokens;
      }
    }
    
    // Stop if we're close to the limit
    if (sampledTokens >= maxTokens * 0.95) {
      break;
    }
  }
  
  // If we still have room, try to add more content from sources that weren't fully sampled
  if (sampledTokens < maxTokens * 0.8) {
    const remainingSources = sourceTokens.filter(source => 
      !sampledSources.some(sampled => sampled.sourceId === source.sourceId)
    );
    
    for (const source of remainingSources) {
      const remainingTokens = maxTokens - sampledTokens;
      if (remainingTokens > minTokensPerSource) {
        const sampledContent = sampleTextContent(source.content, remainingTokens);
        const actualTokens = estimateTokens(sampledContent);
        
        if (sampledTokens + actualTokens <= maxTokens) {
          sampledSources.push({
            ...source,
            content: sampledContent,
            tokens: actualTokens
          });
          sampledTokens += actualTokens;
        }
      }
    }
  }
  
  // Combine sampled content
  const combinedText = sampledSources.map(source => `## ${source.label}\n\n${source.content}`).join('\n\n---\n\n');
  
  
  return {
    text: combinedText,
    tokens: sampledTokens,
    sampledSources: sampledSources
  };
}

/**
 * Logs text extraction results for debugging
 * @param {Object} extractionResult - Result from extractPageTextForQuiz or extractPageTextWithSourcesForQuiz
 */
export function logTextExtraction(extractionResult) {
  // Log extraction results only if there's an error
  if (extractionResult.error) {
    console.error('Extraction Error:', extractionResult.error);
  }
}

/**
 * Gets the current page URL for reference
 * @returns {string} - Current page URL
 */
export function getCurrentPageUrl() {
  return window.location.href;
}

/**
 * Gets the current page title for reference
 * @returns {string} - Current page title
 */
export function getCurrentPageTitle() {
  return document.title || 'Untitled Page';
}