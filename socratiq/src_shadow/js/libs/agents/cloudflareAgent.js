import { alert } from "../utils/utils.js";
import { TokenCounter } from "../utils/tokenCounter.js";
import { CustomAPIAdapter } from "./custom-api-adapter.js";

// ===== CONFIGURATION =====
// Set to true to use localhost workers, false to use production workers
const USE_LOCAL_WORKERS = false;
// =========================

// Configuration for different providers - ordered by speed
const providerConfigs = {
  GROQ: {
    model: "llama-3.1-8b-instant",
    stream: true,
  },
  GEMINI: {
    model: "gemini-2.5-flash",
    stream: true,
  },
  CEREBRAS: {
    model: "gpt-oss-120b",
    stream: true,
  },
  SAMBANOVA: {
    model: "gpt-oss-120b",
    stream: true,
  },
  MISTRAL: {
    model: "mistral-tiny",
    stream: true,
  },
  OPENAI: {
    model: "deepseek/deepseek-chat-v3.1:free",
    stream: true,
  },
  HUGGINGFACE: {
    model: "meta-llama/Llama-3.1-8B-Instruct:featherless-ai",
    stream: false,
  },
  AWAN: {
    model: "Meta-Llama-3.1-70B-Instruct",
    stream: true,
  },
};

// Determine API URL based on configuration
function getApiUrl(isStreaming = false) {
  if (USE_LOCAL_WORKERS) {
    // Local development - use your Cloudflare Worker
    return isStreaming 
      ? "http://localhost:8788/ai/stream"
      : "http://localhost:8787/ai";
  } else {
    // Production - deployed workers
    return isStreaming
      ? "https://proxy-worker-streaming.mlsysbook.workers.dev/ai/stream"
      : "https://proxy-worker.mlsysbook.workers.dev/ai";
  }
}

// Serialize response content for consistent output
function serializeResponse(response, isStreaming = false) {
  if (isStreaming) {
    // For streaming, we expect delta content
    return response.choices?.[0]?.delta?.content || '';
  } else {
    // For single response, we expect full message content
    return response.choices?.[0]?.message?.content || 
           response.text || 
           response.message || 
           JSON.stringify(response);
  }
}

// Call provider with single response
async function callProviderSingle(prompt, provider, signal) {
  const config = providerConfigs[provider];
  
  if (!config) {
    throw new Error(`Unknown provider: ${provider}`);
  }

  const messages = [
    {
      role: "user",
      content: prompt,
    },
  ];

  // Check token count and optimize if needed
  TokenCounter.logTokenUsage(messages, `${provider} Single Request`);
  
  if (TokenCounter.wouldExceedLimit(messages, TokenCounter.getSafeGroqLimit())) {
    console.warn(`[TOKEN_WARNING] Request too large for ${provider}, attempting to optimize...`);
    
    // For large prompts, try to split or truncate
    if (prompt.length > 15000) { // More conservative character limit
      messages[0].content = prompt.substring(0, 15000) + "\n\n[Content truncated due to size limits]";
    }
  }

  try {
    const apiUrl = getApiUrl(false);
    const targetUrl = getTargetUrl(provider);
    
    const response = await fetch(`${apiUrl}?url=${encodeURIComponent(targetUrl)}&provider=${provider.toLowerCase()}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages,
        model: config.model,
        temperature: config.temperature || 0.7,
        top_p: config.top_p || 0.9,
      }),
      signal,
    });

    if (!response.ok) {
      // Handle rate limit errors specifically
      if (response.status === 429) {
        const errorText = await response.text();
        console.error(`[RATE_LIMIT] ${provider} rate limit exceeded:`, errorText);
        throw new Error(`Rate limit exceeded for ${provider}. Please try again in a moment or reduce your request size.`);
      }
      throw new Error(`${provider} failed with status ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    return serializeResponse(result, false);
    
  } catch (error) {
    console.error("Error in callProviderSingle:", error);
    if (error.name === "AbortError") {
      throw new Error("Request was cancelled");
    }
    throw error;
  }
}

// Call provider with streaming response
async function* callProviderStream(prompt, provider, signal) {
  const config = providerConfigs[provider];
  
  if (!config) {
    throw new Error(`Unknown provider: ${provider}`);
  }

  const messages = [
    {
      role: "user",
      content: prompt,
    },
  ];

  // Check token count and optimize if needed
  TokenCounter.logTokenUsage(messages, `${provider} Streaming Request`);
  
  if (TokenCounter.wouldExceedLimit(messages, TokenCounter.getSafeGroqLimit())) {
    console.warn(`[TOKEN_WARNING] Request too large for ${provider}, attempting to optimize...`);
    
    // For large prompts, try to split or truncate
    if (prompt.length > 15000) { // More conservative character limit
      messages[0].content = prompt.substring(0, 15000) + "\n\n[Content truncated due to size limits]";
    }
  }

  try {
    const apiUrl = getApiUrl(true);
    const targetUrl = getTargetUrl(provider);
    
    const response = await fetch(`${apiUrl}?url=${encodeURIComponent(targetUrl)}&provider=${provider.toLowerCase()}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages,
        model: config.model,
        temperature: config.temperature || 0.7,
        top_p: config.top_p || 0.9,
      }),
      signal,
    });

    if (!response.ok) {
      // Handle rate limit errors specifically
      if (response.status === 429) {
        const errorText = await response.text();
        console.error(`[RATE_LIMIT] ${provider} rate limit exceeded:`, errorText);
        throw new Error(`Rate limit exceeded for ${provider}. Please try again in a moment or reduce your request size.`);
      }
      throw new Error(`${provider} failed with status ${response.status}: ${response.statusText}`);
    }

    // Handle streaming responses
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let hasYieldedContent = false;
    let streamCompleted = false;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          streamCompleted = true;
          break;
        }

        // Decode the chunk and add to buffer
        const newText = decoder.decode(value, { stream: true });
        buffer += newText;
        
        // Handle Server-Sent Events format (data: prefix)
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6).trim();
            if (jsonStr === '[DONE]') {
              streamCompleted = true;
              return; // End of stream
            }
            
            try {
              const parsed = JSON.parse(jsonStr);
              const content = serializeResponse(parsed, true);
              if (content) {
                hasYieldedContent = true;
                yield content;
              }
              
              // Check for finish_reason to detect completion
              if (parsed.choices?.[0]?.finish_reason) {
                streamCompleted = true;
                return;
              }
            } catch (e) {
              console.warn('Failed to parse streaming chunk:', e, 'Raw:', jsonStr);
            }
          }
        }
      }
    } catch (error) {
      console.error('Stream reading error:', error);
      throw error;
    } finally {
      // Ensure we always close the reader
      try {
        reader.releaseLock();
      } catch (e) {
        // Reader might already be released
      }
    }

    // Check if we got any content but stream didn't complete properly
    if (hasYieldedContent && !streamCompleted) {
      console.warn('Stream ended without proper completion signal');
    } else if (!hasYieldedContent) {
      throw new Error('No content received from streaming response');
    }

    // Process any remaining data in buffer
    if (buffer.trim()) {
      const lines = buffer.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.slice(6).trim();
          if (jsonStr === '[DONE]') {
            return;
          }
          
          try {
            const parsed = JSON.parse(jsonStr);
            const content = serializeResponse(parsed, true);
            if (content) {
              yield content;
            }
          } catch (e) {
            console.warn('Failed to parse final streaming chunk:', e);
          }
        }
      }
    }
    
  } catch (error) {
    console.error("Error in callProviderStream:", error);
    if (error.name === "AbortError") {
      throw new Error("Request was cancelled");
    }
    throw error;
  }
}

// Get target URL for each provider
function getTargetUrl(provider) {
  const urls = {
    GROQ: "https://api.groq.com/openai/v1/chat/completions",
    GEMINI: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
    CEREBRAS: "https://api.cerebras.ai/v1/chat/completions",
    SAMBANOVA: "https://api.sambanova.ai/v1/chat/completions",
    MISTRAL: "https://api.mistral.ai/v1/chat/completions",
    OPENAI: "https://openrouter.ai/api/v1/chat/completions",
    HUGGINGFACE: "https://router.huggingface.co/v1/chat/completions",
    AWAN: "https://api.awanllm.com/v1/chat/completions",
  };
  
  return urls[provider] || urls.GROQ; // Default to Groq
}

// Try multiple providers with single response
async function tryMultipleProvidersSingle(
  params,
  token,
  json = false,
) {
  // Try custom API first if configured
  const settings = getCurrentSettings();
  if (settings?.customAPI?.endpoint) {
    try {
      console.log(`[CUSTOM_API] Attempting custom API for single response: ${settings.customAPI.provider}`);
      const customResult = CustomAPIAdapter.callCustomAPI(params.prompt, settings, false);
      if (customResult) {
        // Collect all chunks from the generator
        let fullResponse = '';
        for await (const chunk of customResult) {
          fullResponse += chunk;
        }
        console.log(`[CUSTOM_API] Successfully completed single response with custom API: ${settings.customAPI.provider}`);
        return { text: fullResponse };
      }
    } catch (error) {
      console.warn('[CUSTOM_API] Custom API failed for single response, falling back to default providers:', error);
    }
  }

  const preferredOrder = ["GROQ", "GEMINI", "CEREBRAS", "SAMBANOVA", "MISTRAL", "OPENAI", "HUGGINGFACE", "AWAN"];
  const controller = new AbortController();
  const { signal } = controller;

  let lastError = null;

  for (const provider of preferredOrder) {
    try {
      // TRACE: Track which backup provider is being called
      console.trace(`[AI_TRACE] cloudflareAgent.js - Trying backup provider (single): ${provider}`, {
        model: providerConfigs[provider]?.model || 'Unknown model',
        promptPreview: params.prompt?.substring(0, 100) + '...' || 'No prompt'
      });

      const result = await callProviderSingle(params.prompt, provider, signal);
      return result;
    } catch (error) {
      lastError = error;
      console.error(`${provider} failed:`, error);
      // Continue to next provider unless cancelled
      if (error.message === "Request was cancelled") {
        throw error;
      }
    }
  }

  throw new Error(`All providers failed. Last error: ${lastError.message}`);
}

// Try multiple providers with streaming response
async function* tryMultipleProvidersStream(
  params,
  token,
  json = false,
) {
  // Try custom API first if configured
  const settings = getCurrentSettings();
  if (settings?.customAPI?.endpoint) {
    try {
      console.log(`[CUSTOM_API] Attempting custom API for streaming: ${settings.customAPI.provider}`);
      const customResult = CustomAPIAdapter.callCustomAPI(params.prompt, settings, true);
      if (customResult) {
        yield* customResult;
        console.log(`[CUSTOM_API] Successfully completed streaming with custom API: ${settings.customAPI.provider}`);
        return; // Success with custom API
      }
    } catch (error) {
      console.warn('[CUSTOM_API] Custom API failed for streaming, falling back to default providers:', error);
    }
  }

  const preferredOrder = ["GROQ", "GEMINI", "CEREBRAS", "SAMBANOVA", "MISTRAL", "OPENAI", "HUGGINGFACE", "AWAN"];
  const controller = new AbortController();
  const { signal } = controller;

  // Set a timeout to prevent hanging streams
  const timeoutId = setTimeout(() => {
    console.warn('Stream timeout reached, aborting request');
    controller.abort();
  }, 60000); // 60 second timeout

  let lastError = null;

  for (const provider of preferredOrder) {
    try {
      // TRACE: Track which backup provider is being called
      console.trace(`[AI_TRACE] cloudflareAgent.js - Trying backup provider: ${provider}`, {
        model: providerConfigs[provider]?.model || 'Unknown model',
        promptPreview: params.prompt?.substring(0, 100) + '...' || 'No prompt'
      });

      // Get the generator from callProviderStream
      const generator = callProviderStream(params.prompt, provider, signal);

      let hasYieldedContent = false;
      // Yield each chunk from the generator
      for await (const chunk of generator) {
        hasYieldedContent = true;
        yield chunk;
      }

      // If we got content, we successfully completed streaming
      if (hasYieldedContent) {
        clearTimeout(timeoutId);
        return;
      } else {
        throw new Error(`No content received from ${provider}`);
      }
    } catch (error) {
      lastError = error;
      console.error(`${provider} failed:`, error);
      // Continue to next provider unless cancelled
      if (error.message === "Request was cancelled") {
        clearTimeout(timeoutId);
        throw error;
      }
    }
  }

  clearTimeout(timeoutId);
  throw new Error(`All providers failed. Last error: ${lastError.message}`);
}

// Main function that handles both single and streaming
async function* callCloudflareAgent(
  params,
  token,
  stream = true,
  json = false,
) {
  if (stream) {
    // Use streaming version
    yield* tryMultipleProvidersStream(params, token, json);
  } else {
    // Use single response version
    const result = await tryMultipleProvidersSingle(params, token, json);
    yield result;
  }
}

/**
 * Translate legacy quiz format to OpenAI-compatible format
 * @param {Object} legacyParams - The legacy parameters object
 * @param {string} legacyParams.chapterId - Chapter identifier
 * @param {string} legacyParams.sectionId - Section identifier  
 * @param {string} legacyParams.prompt - The quiz prompt
 * @param {string} legacyParams.quote - The chapter section content
 * @param {boolean} legacyParams.saveToDb - Whether to save to database
 * @returns {Object} OpenAI-compatible request format
 */
function translateLegacyQuizFormat(legacyParams) {
  // Optimize the quote content to prevent token limit issues
  let optimizedQuote = legacyParams.quote;
  
  // If quote is too long, truncate it intelligently
  if (optimizedQuote && optimizedQuote.length > 10000) {
    
    // Try to truncate at sentence boundaries
    const sentences = optimizedQuote.split(/[.!?]+/);
    let truncatedQuote = '';
    
    for (const sentence of sentences) {
      const testQuote = truncatedQuote + sentence + '. ';
      if (testQuote.length > 10000) {
        break;
      }
      truncatedQuote = testQuote;
    }
    
    if (truncatedQuote.trim()) {
      optimizedQuote = truncatedQuote.trim() + '\n\n[Chapter section truncated due to size limits]';
    } else {
      // Fallback: hard truncate
      optimizedQuote = optimizedQuote.substring(0, 10000) + '\n\n[Chapter section truncated due to size limits]';
    }
  }
  
  // Combine prompt and optimized quote into a single content string
  const fullPrompt = `${legacyParams.prompt}\n\nCHAPTER SECTION: ${optimizedQuote}`;
  
  return {
    messages: [{ role: 'user', content: fullPrompt }],
    model: 'llama-3.3-70b-versatile', // Default to Groq model
    temperature: 0.7,
    // Preserve original metadata for potential use
    metadata: {
      chapterId: legacyParams.chapterId,
      sectionId: legacyParams.sectionId,
      saveToDb: legacyParams.saveToDb,
      context: legacyParams.context
    }
  };
}

/**
 * Parse quiz response to extract JSON array from AI response
 * @param {Object} response - Full OpenAI response object
 * @param {Object} context - Context data from viewport capture
 * @returns {Array} Parsed questions array
 */
function parseQuizResponse(response, context = null) {
  try {
    
    // Extract content from the response
    const content = response.choices?.[0]?.message?.content;
    if (!content) {
      console.error('No content found in response:', response);
      throw new Error('No content found in response');
    }


    // Look for JSON code block in the content
    const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonMatch) {
      const jsonString = jsonMatch[1].trim();
      const parsed = JSON.parse(jsonString);
      
      // Ensure each question has sourceReference field
      const questions = parsed.questions || parsed;
      if (Array.isArray(questions)) {
        questions.forEach((q, index) => {
          // Always ensure sourceReference exists and is meaningful
          if (!q.sourceReference || q.sourceReference.includes('not provided by AI')) {
            // Generate a more meaningful source reference using context data
            if (context) {
              q.sourceReference = `Based on content from "${context.title}" (Question ${index + 1})`;
              q.sourceLabel = context.title;
              q.sourceUrl = context.url;
              q.sourcePosition = context.position.scrollY;
            } else {
              q.sourceReference = `Based on content from the current page section (Question ${index + 1})`;
              q.sourceLabel = "Current Page Section";
              q.sourceUrl = window.location.href;
              q.sourcePosition = window.scrollY;
            }
          } else {
            // AI provided sourceReference, but ensure we have the metadata
            if (context) {
              q.sourceLabel = context.title;
              q.sourceUrl = context.url;
              q.sourcePosition = context.position.scrollY;
            } else {
              q.sourceLabel = "Current Page Section";
              q.sourceUrl = window.location.href;
              q.sourcePosition = window.scrollY;
            }
          }
          
          // Ensure all answers have explanations
          if (q.answers && Array.isArray(q.answers)) {
            q.answers.forEach((answer, answerIndex) => {
              if (!answer.explanation) {
                answer.explanation = answer.correct ? 
                  "This is the correct answer." : 
                  "This is not the correct answer.";
              }
            });
          }
        });
      }
      
      return questions;
    }

    // If no code block found, try to parse the entire content as JSON
    const trimmedContent = content.trim();
    if (trimmedContent.startsWith('{') || trimmedContent.startsWith('[')) {
      const parsed = JSON.parse(trimmedContent);
      const questions = parsed.questions || parsed;
      
      if (Array.isArray(questions)) {
        questions.forEach((q, index) => {
          // Always ensure sourceReference exists and is meaningful
          if (!q.sourceReference || q.sourceReference.includes('not provided by AI')) {
            // Generate a more meaningful source reference using context data
            if (context) {
              q.sourceReference = `Based on content from "${context.title}" (Question ${index + 1})`;
              q.sourceLabel = context.title;
              q.sourceUrl = context.url;
              q.sourcePosition = context.position.scrollY;
            } else {
              q.sourceReference = `Based on content from the current page section (Question ${index + 1})`;
              q.sourceLabel = "Current Page Section";
              q.sourceUrl = window.location.href;
              q.sourcePosition = window.scrollY;
            }
          } else {
            // AI provided sourceReference, but ensure we have the metadata
            if (context) {
              q.sourceLabel = context.title;
              q.sourceUrl = context.url;
              q.sourcePosition = context.position.scrollY;
            } else {
              q.sourceLabel = "Current Page Section";
              q.sourceUrl = window.location.href;
              q.sourcePosition = window.scrollY;
            }
          }
          
          // Ensure all answers have explanations
          if (q.answers && Array.isArray(q.answers)) {
            q.answers.forEach((answer, answerIndex) => {
              if (!answer.explanation) {
                answer.explanation = answer.correct ? 
                  "This is the correct answer." : 
                  "This is not the correct answer.";
              }
            });
          }
        });
      }
      
      return questions;
    }

    // If all else fails, return the original response
    console.warn('Could not parse quiz response, returning original response');
    console.warn('Content that failed to parse:', content);
    return response;
    
  } catch (error) {
    console.error('Error parsing quiz response:', error);
    console.error('Response that caused error:', response);
    // Return the original response as fallback
    return response;
  }
}

/**
 * Call Cloudflare proxy with legacy quiz format translation
 * @param {Object} legacyParams - Legacy parameters object
 * @param {string} provider - Provider name (groq, gemini, openai, etc.)
 * @param {boolean} isStreaming - Whether to use streaming endpoint
 * @returns {Promise|AsyncGenerator} Response from the proxy
 */
async function callCloudflareWithLegacyFormat(legacyParams, provider = 'groq', isStreaming = false) {
  // Translate legacy format to OpenAI format
  const openaiFormat = translateLegacyQuizFormat(legacyParams);
  
  // Check token count and optimize if needed
  TokenCounter.logTokenUsage(openaiFormat.messages, `${provider} Legacy Quiz Request`);
  
  if (TokenCounter.wouldExceedLimit(openaiFormat.messages, TokenCounter.getSafeGroqLimit())) {
    console.warn(`[TOKEN_WARNING] Legacy quiz request too large for ${provider}, attempting to optimize...`);
    
    // Optimize the prompt content
    const originalContent = openaiFormat.messages[0].content;
    if (originalContent.length > 15000) {
      openaiFormat.messages[0].content = originalContent.substring(0, 15000) + "\n\n[Content truncated due to size limits]";
    }
  }
  
  // Set the appropriate model for the provider
  const providerModels = {
    groq: 'llama-3.1-8b-instant',
    gemini: 'gemini-2.5-flash', 
    cerebras: 'gpt-oss-120b',
    sambanova: 'gpt-oss-120b',
    mistral: 'mistral-tiny',
    openai: 'deepseek/deepseek-chat-v3.1:free',
    huggingface: 'meta-llama/Llama-3.1-8B-Instruct:featherless-ai',
    awan: 'Meta-Llama-3.1-70B-Instruct'
  };
  
  openaiFormat.model = providerModels[provider.toLowerCase()] || providerModels.groq;
  
  // For non-streaming, make direct API call and parse the response
  if (!isStreaming) {
    const config = providerConfigs[provider.toUpperCase()];
    if (!config) {
      throw new Error(`Unknown provider: ${provider}`);
    }

    const apiUrl = getApiUrl(false);
    const targetUrl = getTargetUrl(provider);
    
    const response = await fetch(`${apiUrl}?url=${encodeURIComponent(targetUrl)}&provider=${provider.toLowerCase()}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: openaiFormat.messages,
        model: openaiFormat.model,
        temperature: openaiFormat.temperature,
      }),
    });

    if (!response.ok) {
      // Handle rate limit errors specifically
      if (response.status === 429) {
        const errorText = await response.text();
        console.error(`[RATE_LIMIT] ${provider} rate limit exceeded in legacy format:`, errorText);
        throw new Error(`Rate limit exceeded for ${provider}. Please try again in a moment or reduce your request size.`);
      }
      throw new Error(`${provider} failed with status ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    
    // Parse the response to extract the quiz data
    return parseQuizResponse(result, openaiFormat.metadata?.context);
  } else {
    // For streaming, use the existing function
    return callProviderStream(openaiFormat.messages[0].content, provider, null);
  }
}

// Export functions
// Get current user settings from localStorage
function getCurrentSettings() {
  try {
    const savedSettings = localStorage.getItem("userSettings");
    return savedSettings ? JSON.parse(savedSettings) : {};
  } catch (error) {
    console.error('[CUSTOM_API] Error loading settings:', error);
    return {};
  }
}

export {
  callProviderSingle, 
  callProviderStream, 
  tryMultipleProvidersSingle, 
  tryMultipleProvidersStream,
  callCloudflareAgent,
  translateLegacyQuizFormat,
  callCloudflareWithLegacyFormat
};