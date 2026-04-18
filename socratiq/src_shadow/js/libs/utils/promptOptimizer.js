/**
 * Prompt optimization utility
 * Helps optimize prompts to stay within token limits
 */

import { TokenCounter } from './tokenCounter.js';

export class PromptOptimizer {
  /**
   * Optimize a prompt to stay within token limits
   * @param {string} prompt - Original prompt
   * @param {number} maxTokens - Maximum tokens allowed
   * @returns {string} Optimized prompt
   */
  static optimizePrompt(prompt, maxTokens = TokenCounter.getSafeGroqLimit()) {
    if (!prompt || typeof prompt !== 'string') return '';
    
    const estimatedTokens = TokenCounter.estimateTokens(prompt);
    
    if (estimatedTokens <= maxTokens) {
      return prompt; // No optimization needed
    }
    
    console.log(`[PROMPT_OPTIMIZATION] Original prompt: ~${estimatedTokens} tokens, target: ${maxTokens} tokens`);
    
    // Strategy 1: Remove extra whitespace and normalize
    let optimized = prompt.trim().replace(/\s+/g, ' ');
    
    // Strategy 2: Remove common verbose phrases
    optimized = optimized
      .replace(/\b(please|kindly|thank you|thanks|appreciate)\b/gi, '')
      .replace(/\b(could you|could you please|would you|would you please)\b/gi, '')
      .replace(/\b(I would like|I need|I want|I'm looking for)\b/gi, '')
      .replace(/\s+/g, ' '); // Clean up extra spaces
    
    // Strategy 3: If still too long, truncate intelligently
    const optimizedTokens = TokenCounter.estimateTokens(optimized);
    if (optimizedTokens > maxTokens) {
      // Calculate how many characters we can keep
      const maxChars = Math.floor(maxTokens * 4 * 0.9); // 90% of estimated capacity
      
      if (optimized.length > maxChars) {
        // Try to truncate at sentence boundaries
        const sentences = optimized.split(/[.!?]+/);
        let truncated = '';
        
        for (const sentence of sentences) {
          const testTruncated = truncated + sentence + '. ';
          if (TokenCounter.estimateTokens(testTruncated) > maxTokens) {
            break;
          }
          truncated = testTruncated;
        }
        
        if (truncated.trim()) {
          optimized = truncated.trim() + '\n\n[Content truncated due to size limits]';
        } else {
          // Fallback: hard truncate
          optimized = optimized.substring(0, maxChars) + '\n\n[Content truncated due to size limits]';
        }
      }
    }
    
    const finalTokens = TokenCounter.estimateTokens(optimized);
    console.log(`[PROMPT_OPTIMIZATION] Optimized prompt: ~${finalTokens} tokens`);
    
    return optimized;
  }

  /**
   * Create a summary prompt for very long content
   * @param {string} content - Long content to summarize
   * @param {string} task - What task to perform on the content
   * @returns {string} Summary prompt
   */
  static createSummaryPrompt(content, task = 'analyze') {
    const maxContentLength = 15000; // Rough character limit
    
    if (content.length <= maxContentLength) {
      return `${task}: ${content}`;
    }
    
    // Create a summary of the content
    const summary = content.substring(0, maxContentLength) + '\n\n[Content truncated - showing first portion only]';
    
    return `${task} this content (truncated for size): ${summary}`;
  }

  /**
   * Split a large prompt into multiple smaller requests
   * @param {string} prompt - Large prompt
   * @param {number} maxTokensPerChunk - Max tokens per chunk
   * @returns {Array} Array of smaller prompts
   */
  static splitPrompt(prompt, maxTokensPerChunk = TokenCounter.getSafeGroqLimit()) {
    return TokenCounter.splitTextIntoChunks(prompt, maxTokensPerChunk);
  }

  /**
   * Get recommendations for reducing prompt size
   * @param {string} prompt - Prompt to analyze
   * @returns {Array} Array of recommendations
   */
  static getOptimizationRecommendations(prompt) {
    const recommendations = [];
    const tokens = TokenCounter.estimateTokens(prompt);
    
    if (tokens > TokenCounter.getSafeGroqLimit()) {
      recommendations.push(`Prompt is ${tokens} tokens, exceeds safe limit of ${TokenCounter.getSafeGroqLimit()}`);
      
      // Check for common issues
      if (prompt.includes('please') || prompt.includes('kindly')) {
        recommendations.push('Remove polite phrases like "please" and "kindly"');
      }
      
      if (prompt.includes('thank you') || prompt.includes('thanks')) {
        recommendations.push('Remove thank you phrases');
      }
      
      if (prompt.match(/\b(could you|would you)\b/gi)) {
        recommendations.push('Replace "could you" with direct commands');
      }
      
      if (prompt.length > 20000) {
        recommendations.push('Consider splitting into multiple smaller requests');
      }
      
      if (prompt.includes('\n\n\n')) {
        recommendations.push('Remove excessive line breaks');
      }
    }
    
    return recommendations;
  }
}