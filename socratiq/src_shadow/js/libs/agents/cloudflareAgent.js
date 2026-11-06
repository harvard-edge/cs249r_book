import { TokenCounter } from "../utils/tokenCounter.js";
import { CustomAPIAdapter } from "./custom-api-adapter.js";
import { WORKER_URL_AI, WORKER_URL_AI_STREAM, PROVIDER_MODELS, DEFAULT_PROVIDER, MAX_OUTPUT_TOKENS } from "../../../configs/env_configs.js";
import { SHARED_PROVIDER } from '../../../../src_shared/provider-config.js';
import { readLearnerPreferences } from '../../../../src_reader/preferences.js';
import { withLearnerCalibration } from '../../../../src_reader/calibration.mjs';

// Provider configs are now centrally managed in src_shadow/configs/env_configs.js
const providerConfigs = PROVIDER_MODELS;

// Determine API URL based on configuration — controlled via env_configs.js
function getApiUrl(isStreaming = false) {
  return isStreaming ? WORKER_URL_AI_STREAM : WORKER_URL_AI;
}

// Serialize response content for consistent output
function serializeResponse(response, isStreaming = false) {
  if (isStreaming) {
    // For streaming, we expect delta content
    return response.choices?.[0]?.delta?.content ||
      response.candidates?.[0]?.content?.parts?.map(part => part.text || '').join('') || '';
  } else {
    // For single response, we expect full message content
    return response.choices?.[0]?.message?.content || 
           response.candidates?.[0]?.content?.parts?.map(part => part.text || '').join('') ||
           response.text || 
           (typeof response.message === 'string' ? response.message : '');
  }
}

function requestMessages(params) {
  const messages = params.messages?.length ? params.messages : [{ role: 'user', content: params.prompt || '' }];
  return params.systemPrompt ? [{ role: 'system', content: params.systemPrompt }, ...messages] : messages;
}

// A length/filter/tool finish is not a completed text answer or quiz.
function validateCompletion(result) {
  const reason = result.choices?.[0]?.finish_reason;
  if (reason && reason !== 'stop') {
    throw new Error(`Provider response did not complete (finish_reason: ${reason})`);
  }
}

// Call provider with single response
async function callProviderSingle(prompt, provider = DEFAULT_PROVIDER, signal, suppliedMessages, learnerPreferences) {
  provider = provider.toUpperCase();
  const config = providerConfigs[provider];
  
  if (!config) {
    throw new Error(`Unknown provider: ${provider}`);
  }

  // Preserve caller policy and conversation order. The gateway validates size;
  // truncating arbitrary messages here could remove part of a system policy.
  const messages = withLearnerCalibration(suppliedMessages?.length ? suppliedMessages : [{ role: "user", content: prompt }], learnerPreferences ?? readLearnerPreferences());

  // Check token count and optimize if needed
  TokenCounter.logTokenUsage(messages, `${provider} Single Request`);
  
  if (TokenCounter.wouldExceedLimit(messages, TokenCounter.getSafeGroqLimit())) {
    console.warn(`[TOKEN_WARNING] Request too large for ${provider}, attempting to optimize...`);
    
    // For large prompts, try to split or truncate
    if (!suppliedMessages?.length && prompt.length > 15000) { // Only truncate the legacy plain prompt.
      messages[messages.length - 1].content = prompt.substring(0, 15000) + "\n\n[Content truncated due to size limits]";
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
        max_tokens: MAX_OUTPUT_TOKENS,
        temperature: config.temperature || 0.7,
        top_p: config.top_p || 0.9,
        stream: false,
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
    validateCompletion(result);
    if (result.error) throw new Error(result.error.message || result.error.code || 'Provider returned an error');
    const content = serializeResponse(result, false);
    if (!content) throw new Error('No content received from provider');
    return content;
    
  } catch (error) {
    console.error("Error in callProviderSingle:", error);
    if (error.name === "AbortError") {
      throw new Error("Request was cancelled");
    }
    throw error;
  }
}

// Call provider with streaming response
async function* callProviderStream(prompt, provider = DEFAULT_PROVIDER, signal, suppliedMessages, learnerPreferences) {
  provider = provider.toUpperCase();
  const config = providerConfigs[provider];
  
  if (!config) {
    throw new Error(`Unknown provider: ${provider}`);
  }

  // Preserve caller policy and conversation order. The gateway validates size;
  // truncating arbitrary messages here could remove part of a system policy.
  const messages = withLearnerCalibration(suppliedMessages?.length ? suppliedMessages : [{ role: "user", content: prompt }], learnerPreferences ?? readLearnerPreferences());

  // Check token count and optimize if needed
  TokenCounter.logTokenUsage(messages, `${provider} Streaming Request`);
  
  if (TokenCounter.wouldExceedLimit(messages, TokenCounter.getSafeGroqLimit())) {
    console.warn(`[TOKEN_WARNING] Request too large for ${provider}, attempting to optimize...`);
    
    // For large prompts, try to split or truncate
    if (!suppliedMessages?.length && prompt.length > 15000) { // Only truncate the legacy plain prompt.
      messages[messages.length - 1].content = prompt.substring(0, 15000) + "\n\n[Content truncated due to size limits]";
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
        max_tokens: MAX_OUTPUT_TOKENS,
        temperature: config.temperature || 0.7,
        top_p: config.top_p || 0.9,
        stream: config.stream !== false,
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

    // Providers configured without streaming return a normal JSON response.
    if (response.headers.get('content-type')?.includes('application/json')) {
      const result = await response.json();
      validateCompletion(result);
      if (result.error) throw new Error(result.error.message || result.error.code || 'Provider returned an error');
      const content = serializeResponse(result);
      if (!content) throw new Error('No content received from provider');
      yield content;
      return;
    }
    if (!response.body) throw new Error('Provider returned no response body');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let hasContent = false;
    const parseLine = line => {
      if (!line.startsWith('data:')) return null;
      const data = line.slice(5).trim();
      if (!data) return null;
      if (data === '[DONE]') return { done: true };
      const result = JSON.parse(data);
      if (result.error) throw new Error(result.error.message || result.error.code || 'Provider stream returned an error');
      validateCompletion(result);
      return { content: serializeResponse(result, true), done: result.choices?.[0]?.finish_reason === 'stop' };
    };
    try {
      let done = false;
      while (!done) {
        const chunk = await reader.read();
        done = chunk.done;
        buffer += decoder.decode(chunk.value, { stream: !done });
        const lines = buffer.split('\n');
        buffer = done ? '' : lines.pop() || '';
        for (const line of lines) {
          const event = parseLine(line.replace(/\r$/, ''));
          if (event?.content) {
            hasContent = true;
            yield event.content;
          }
          if (event?.done) {
            if (!hasContent) throw new Error('No content received from streaming response');
            return;
          }
        }
      }
      throw new Error('Provider stream ended before completion');
    } finally {
      await reader.cancel().catch(() => {});
      reader.releaseLock();
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
  provider = provider.toUpperCase();
  const urls = {
    GROQ: SHARED_PROVIDER.target,
    GEMINI: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
    CEREBRAS: "https://api.cerebras.ai/v1/chat/completions",
    SAMBANOVA: "https://api.sambanova.ai/v1/chat/completions",
    MISTRAL: "https://api.mistral.ai/v1/chat/completions",
    OPENAI: "https://openrouter.ai/api/v1/chat/completions",
    HUGGINGFACE: "https://router.huggingface.co/v1/chat/completions",
    AWAN: "https://api.awanllm.com/v1/chat/completions",
  };
  
  if (!urls[provider]) throw new Error(`Unknown provider: ${provider}`);
  return urls[provider];
}

// Try multiple providers with single response
async function tryMultipleProvidersSingle(
  params,
  token,
  json = false,
) {
  // An explicitly selected custom service is a privacy boundary. Never send its
  // failed request to the hosted service without the learner changing settings.
  const settings = getCurrentSettings();
  if (settings?.customAPI?.enabled) {
    if (!CustomAPIAdapter.hasCustomAPIConfig(settings)) throw new Error('Complete your custom API settings, or disable the custom API to use the shared service.');
    try {
      console.log(`[CUSTOM_API] Attempting custom API for single response: ${settings.customAPI.provider}`);
      const customResult = CustomAPIAdapter.callCustomAPI(params.prompt, settings, false, withLearnerCalibration(requestMessages(params), params.learnerPreferences ?? readLearnerPreferences()));
      if (customResult) {
        // Collect all chunks from the generator
        let fullResponse = '';
        for await (const chunk of customResult) {
          fullResponse += chunk;
        }
        console.log(`[CUSTOM_API] Successfully completed single response with custom API: ${settings.customAPI.provider}`);
        if (!fullResponse) throw new Error('Custom API returned no content');
        return fullResponse;
      }
    } catch (error) {
      throw new Error('Your custom API request failed. Check its endpoint, model, and credentials, or disable the custom API to use the shared service.');
    }
    throw new Error('Your custom API returned no answer. Check its settings, or disable the custom API to use the shared service.');
  }

  const preferredOrder = [params.provider || DEFAULT_PROVIDER];
  const controller = new AbortController();
  const { signal } = controller;

  let lastError = null;

  for (const provider of preferredOrder) {
    try {

      const result = await callProviderSingle(params.prompt, provider, signal, requestMessages(params), params.learnerPreferences);
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
  // A custom API failure must not forward the learner's prompt to another service.
  let customHasContent = false;
  const settings = getCurrentSettings();
  if (settings?.customAPI?.enabled) {
    if (!CustomAPIAdapter.hasCustomAPIConfig(settings)) throw new Error('Complete your custom API settings, or disable the custom API to use the shared service.');
    try {
      console.log(`[CUSTOM_API] Attempting custom API for streaming: ${settings.customAPI.provider}`);
      const customResult = CustomAPIAdapter.callCustomAPI(params.prompt, settings, true, withLearnerCalibration(requestMessages(params), params.learnerPreferences ?? readLearnerPreferences()));
      if (customResult) {
        let received = false;
        for await (const chunk of customResult) {
          if (chunk) { customHasContent = true; received = true; yield chunk; }
        }
        if (!received) throw new Error('Custom API returned no content');
        console.log(`[CUSTOM_API] Successfully completed streaming with custom API: ${settings.customAPI.provider}`);
        return; // Success with custom API
      }
    } catch (error) {
      throw new Error('Your custom API request failed. Check its endpoint, model, and credentials, or disable the custom API to use the shared service.');
    }
    throw new Error('Your custom API returned no answer. Check its settings, or disable the custom API to use the shared service.');
  }

  const preferredOrder = [params.provider || DEFAULT_PROVIDER];
  const controller = new AbortController();
  const { signal } = controller;

  // Set a timeout to prevent hanging streams
  const timeoutId = setTimeout(() => {
    console.warn('Stream timeout reached, aborting request');
    controller.abort();
  }, 60000); // 60 second timeout

  let lastError = null;

  try {
  for (const provider of preferredOrder) {
    let hasYieldedContent = false;
    try {

      // Get the generator from callProviderStream
      const generator = callProviderStream(params.prompt, provider, signal, requestMessages(params), params.learnerPreferences);

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
      if (hasYieldedContent || error.message === "Request was cancelled") {
        clearTimeout(timeoutId);
        throw error;
      }
    }
  }

  throw new Error(`All providers failed. Last error: ${lastError?.message || 'No response'}`);
  } finally {
    clearTimeout(timeoutId);
  }
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
    messages: requestMessages({ ...legacyParams, prompt: fullPrompt }),
    model: 'openai/gpt-oss-120b', // Default to Groq model
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
    const content = response.choices?.[0]?.message?.content || response.candidates?.[0]?.content?.parts?.map(part => part.text || '').join('');
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
              q.sourcePosition = context.position?.scrollY || 0;
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
              q.sourcePosition = context.position?.scrollY || 0;
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
              q.sourcePosition = context.position?.scrollY || 0;
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
              q.sourcePosition = context.position?.scrollY || 0;
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
async function callCloudflareWithLegacyFormat(legacyParams, provider = DEFAULT_PROVIDER, isStreaming = false) {
  // Translate legacy format to OpenAI format
  const openaiFormat = translateLegacyQuizFormat(legacyParams);
  
  // Check token count and optimize if needed
  TokenCounter.logTokenUsage(openaiFormat.messages, `${provider} Legacy Quiz Request`);
  
  if (TokenCounter.wouldExceedLimit(openaiFormat.messages, TokenCounter.getSafeGroqLimit())) {
    console.warn(`[TOKEN_WARNING] Legacy quiz request too large for ${provider}, attempting to optimize...`);
    
    // Optimize the prompt content
    const originalContent = openaiFormat.messages[0].content;
    if (!legacyParams.messages?.length && !legacyParams.systemPrompt && originalContent.length > 15000) {
      openaiFormat.messages[0].content = originalContent.substring(0, 15000) + "\n\n[Content truncated due to size limits]";
    }
  }
  
  // Every learner-facing quiz respects the selected custom service just as chat
  // does. Keep explicit low-level provider calls separate for transport tests.
  const params = {...legacyParams, provider, messages:openaiFormat.messages};
  if (isStreaming) return tryMultipleProvidersStream(params);
  const content = await tryMultipleProvidersSingle(params);
  return parseQuizResponse({choices:[{message:{content}}]}, openaiFormat.metadata?.context);
}

// Export functions
// Get current user settings from localStorage
function getCurrentSettings() {
  return CustomAPIAdapter.getRuntimeSettings();
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
