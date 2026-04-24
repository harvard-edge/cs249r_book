// textExtractor.js
import { TEXT_EXTRACTION_CONFIG } from '../../configs/client.config.js';

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
  // Check if element matches any navigation selectors
  const navSelectors = TEXT_EXTRACTION_CONFIG.NAV_SELECTORS;
  
  for (const selector of navSelectors) {
    if (element.matches(selector)) {
      return true;
    }
  }
  
  // Check if element is hidden
  const style = window.getComputedStyle(element);
  if (style.display === 'none' || style.visibility === 'hidden') {
    return true;
  }
  
  // Check for common navigation attributes
  const role = element.getAttribute('role');
  if (role && ['navigation', 'banner', 'contentinfo'].includes(role)) {
    return true;
  }
  
  return false;
}

/**
 * Extracts text content from a DOM element, excluding navigation components
 * @param {Element} rootElement - The root element to extract text from (defaults to document.body)
 * @returns {string} - Extracted text content
 */
function extractTextContent(rootElement = document.body) {
  if (!rootElement) {
    console.warn('No root element provided for text extraction');
    return '';
  }
  
  // Clone the element to avoid modifying the original DOM
  const clonedElement = rootElement.cloneNode(true);
  
  // Remove navigation elements from the clone
  TEXT_EXTRACTION_CONFIG.NAV_SELECTORS.forEach(selector => {
    const navElements = clonedElement.querySelectorAll(selector);
    navElements.forEach(el => el.remove());
  });
  
  // Remove elements with navigation roles
  const navRoleElements = clonedElement.querySelectorAll('[role="navigation"], [role="banner"], [role="contentinfo"]');
  navRoleElements.forEach(el => el.remove());
  
  // Remove script and style elements
  const scriptElements = clonedElement.querySelectorAll('script, style');
  scriptElements.forEach(el => el.remove());
  
  // Get text content
  let textContent = clonedElement.textContent || '';
  
  // Clean up the text
  textContent = textContent
    .replace(/\s+/g, ' ') // Replace multiple whitespace with single space
    .replace(/\n\s*\n/g, '\n') // Remove empty lines
    .trim();
  
  return textContent;
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
  
  // Clone the element to avoid modifying the original DOM
  const clonedElement = rootElement.cloneNode(true);
  
  // Remove navigation elements from the clone
  TEXT_EXTRACTION_CONFIG.NAV_SELECTORS.forEach(selector => {
    const navElements = clonedElement.querySelectorAll(selector);
    navElements.forEach(el => el.remove());
  });
  
  // Remove elements with navigation roles
  const navRoleElements = clonedElement.querySelectorAll('[role="navigation"], [role="banner"], [role="contentinfo"]');
  navRoleElements.forEach(el => el.remove());
  
  // Remove script and style elements
  const scriptElements = clonedElement.querySelectorAll('script, style');
  scriptElements.forEach(el => el.remove());
  
  // Find content sections and extract with source mapping
  const sources = [];
  const contentSections = [];
  
  // Look for main content areas first
  const mainContentSelectors = TEXT_EXTRACTION_CONFIG.CONTENT_SELECTORS;
  let contentElements = [];
  
  for (const selector of mainContentSelectors) {
    const elements = clonedElement.querySelectorAll(selector);
    if (elements.length > 0) {
      contentElements = Array.from(elements);
      break;
    }
  }
  
  // If no main content found, use the entire cloned element
  if (contentElements.length === 0) {
    contentElements = [clonedElement];
  }
  
  // Process each content element
  contentElements.forEach((element, index) => {
    const elementId = element.id || `content-section-${index}`;
    const elementClass = element.className || '';
    const elementTag = element.tagName.toLowerCase();
    
    // Get text content from this element
    let textContent = element.textContent || '';
    
    // Clean up the text
    textContent = textContent
      .replace(/\s+/g, ' ')
      .replace(/\n\s*\n/g, '\n')
      .trim();
    
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