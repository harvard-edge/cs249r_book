/**
 * Simple token counter utility
 * Provides rough estimation of token count for text
 */

export class TokenCounter {
  /**
   * Rough estimation of token count
   * Rule of thumb: ~4 characters per token for English text
   * @param {string} text - Text to count tokens for
   * @returns {number} Estimated token count
   */
  static estimateTokens(text) {
    if (!text || typeof text !== 'string') return 0;
    
    // Remove extra whitespace and normalize
    const normalizedText = text.trim().replace(/\s+/g, ' ');
    
    // Rough estimation: ~4 characters per token
    // This is a conservative estimate for English text
    return Math.ceil(normalizedText.length / 4);
  }

  /**
   * Count tokens for a messages array (OpenAI format)
   * @param {Array} messages - Array of message objects
   * @returns {number} Total estimated token count
   */
  static countMessagesTokens(messages) {
    if (!Array.isArray(messages)) return 0;
    
    let totalTokens = 0;
    
    for (const message of messages) {
      if (message.content) {
        totalTokens += this.estimateTokens(message.content);
      }
      
      // Add overhead for message structure (~4 tokens per message)
      totalTokens += 4;
    }
    
    // Add overhead for request structure (~10 tokens)
    totalTokens += 10;
    
    return totalTokens;
  }

  /**
   * Check if request would exceed token limit
   * @param {Array} messages - Messages array
   * @param {number} limit - Token limit (default: 10000 for safety)
   * @returns {boolean} True if would exceed limit
   */
  static wouldExceedLimit(messages, limit = 10000) {
    const tokenCount = this.countMessagesTokens(messages);
    return tokenCount > limit;
  }

  /**
   * Split large text into chunks that fit within token limit
   * @param {string} text - Text to split
   * @param {number} maxTokensPerChunk - Maximum tokens per chunk
   * @returns {Array} Array of text chunks
   */
  static splitTextIntoChunks(text, maxTokensPerChunk = 8000) {
    if (!text || typeof text !== 'string') return [];
    
    const chunks = [];
    const sentences = text.split(/[.!?]+/).filter(s => s.trim());
    
    let currentChunk = '';
    
    for (const sentence of sentences) {
      const testChunk = currentChunk + sentence + '. ';
      const testTokens = this.estimateTokens(testChunk);
      
      if (testTokens > maxTokensPerChunk && currentChunk) {
        // Current chunk is full, start a new one
        chunks.push(currentChunk.trim());
        currentChunk = sentence + '. ';
      } else {
        currentChunk = testChunk;
      }
    }
    
    // Add the last chunk if it has content
    if (currentChunk.trim()) {
      chunks.push(currentChunk.trim());
    }
    
    return chunks;
  }

  /**
   * Get safe token limit for Groq on-demand tier
   * @returns {number} Safe token limit (6000 to leave room for response)
   */
  static getSafeGroqLimit() {
    // Groq on-demand limit is 12,000 TPM
    // Use 6,000 to leave room for response tokens and be more conservative
    return 6000;
  }

  /**
   * Log token usage for debugging
   * @param {Array} messages - Messages array
   * @param {string} context - Context for logging
   */
  static logTokenUsage(messages, context = 'Request') {
    const tokenCount = this.countMessagesTokens(messages);
    console.log(`[TOKEN_COUNT] ${context}: ~${tokenCount} tokens`);
    
    if (tokenCount > this.getSafeGroqLimit()) {
      console.warn(`[TOKEN_WARNING] ${context} exceeds safe limit (${tokenCount} > ${this.getSafeGroqLimit()})`);
    }
  }
}