import { TokenCounter } from '../utils/tokenCounter.js';
import { BYOK_SESSION_KEY, redactSecrets } from '../../../../src_shared/byok-provider.js';

/**
 * Custom API Adapter for handling user-provided API endpoints
 * Supports multiple API formats including OpenAI, Ollama, Gemini, etc.
 */
export class CustomAPIAdapter {
  // SettingsManager publishes current custom settings after its event reaches
  // the workspace. Use that session value before the persisted, redacted copy.
  static getRuntimeSettings() {
    let saved = {};
    try {
      const raw = globalThis.localStorage?.getItem('userSettings');
      const parsed = raw && raw.length <= 65536 ? JSON.parse(raw) : null;
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) saved = parsed;
    } catch { /* A blocked or damaged local store must not erase session settings. */ }
    const session = globalThis.window?.customAPISettings;
    if (session && typeof session === 'object' && !Array.isArray(session) && typeof session.enabled === 'boolean' && typeof session.provider === 'string' && typeof session.endpoint === 'string') {
      return {...saved, customAPI: {
        enabled: session.enabled, provider: session.provider, endpoint: session.endpoint,
        model: typeof session.model === 'string' ? session.model : '',
        apiKey: typeof session.apiKey === 'string' ? session.apiKey : '',
        saveLocally: session.saveLocally === true,
      }};
    }
    try {
      const byokSessionRaw = globalThis.sessionStorage?.getItem(BYOK_SESSION_KEY);
      if (byokSessionRaw && byokSessionRaw.length <= 8192) {
        const byok = JSON.parse(byokSessionRaw);
        if (byok && typeof byok === 'object' && byok.enabled === true && typeof byok.provider === 'string') {
          return {...saved, customAPI: {
            enabled: true,
            provider: byok.provider === 'openrouter' ? 'open-router' : byok.provider,
            endpoint: typeof byok.endpoint === 'string' ? byok.endpoint : '',
            model: typeof byok.model === 'string' ? byok.model : '',
            apiKey: typeof byok.apiKey === 'string' ? byok.apiKey : '',
            saveLocally: byok.saveLocally === true,
          }};
        }
      }
    } catch { /* Ignore blocked sessionStorage */ }
    // Honor an opted-out legacy record even if an older version wrote its key.
    if (saved.customAPI && saved.customAPI.saveLocally !== true) {
      saved = {...saved,customAPI:{...saved.customAPI,apiKey:''}};
    }
    return saved;
  }
  
  /**
   * Main entry point for custom API calls
   * @param {string} prompt - The prompt to send
   * @param {Object} settings - User settings including custom API config
   * @param {boolean} isStreaming - Whether to use streaming
   * @returns {AsyncGenerator|null} - Generator for streaming response or null if no custom API
   */
  static async *callCustomAPI(prompt, settings, isStreaming = true, suppliedMessages) {
    // Check if custom API is configured
    if (!this.hasCustomAPIConfig(settings)) {
      return null; // Signal to use default APIs
    }
    
    const config = this.getCustomAPIConfig(settings);
    
    try {

      
      if (isStreaming) {
        yield* this._callCustomAPIStreaming(prompt, config, suppliedMessages);
      } else {
        yield* this._callCustomAPINonStreaming(prompt, config, suppliedMessages);
      }
    } catch (error) {
      console.error('[CUSTOM_API] Custom API request failed');
      throw error; // The caller decides whether it is safe to fall back.
    }
  }
  
  /**
   * Check if custom API is properly configured
   * @param {Object} settings - User settings
   * @returns {boolean} - True if custom API is configured
   */
  static hasCustomAPIConfig(settings) {
    return settings?.customAPI?.enabled &&
           settings?.customAPI?.endpoint && 
           settings?.customAPI?.provider && 
           settings?.customAPI?.endpoint.trim() !== '';
  }
  
  /**
   * Get custom API configuration from settings
   * @param {Object} settings - User settings
   * @returns {Object} - Custom API configuration
   */
  static getCustomAPIConfig(settings) {
    return {
      endpoint: settings.customAPI.endpoint,
      provider: settings.customAPI.provider,
      model: settings.customAPI.model || this._getDefaultModel(settings.customAPI.provider),
      apiKey: settings.customAPI.apiKey || null
    };
  }
  
  /**
   * Get default model for provider
   * @param {string} provider - Provider name
   * @returns {string} - Default model name
   */
  static _getDefaultModel(provider) {
    const defaultModels = {
      'ollama': 'gemma3:270m',
      'openai': 'gpt-4.1-mini',
      'open-router': 'google/gemini-2.5-flash',
      'groq': 'openai/gpt-oss-120b',
      'google-gemini': 'gemini-2.5-flash',
      'anthropic': 'claude-sonnet-4-20250514'
    };
    return defaultModels[provider] || 'default';
  }
  
  /**
   * Make streaming API call to custom endpoint
   * @param {string} prompt - The prompt to send
   * @param {Object} config - API configuration
   * @returns {AsyncGenerator} - Generator for streaming response
   */
  static async *_callCustomAPIStreaming(prompt, config, suppliedMessages) {
    const messages = suppliedMessages?.length ? suppliedMessages : [{ role: 'user', content: prompt }];
    
    // Check token count and optimize if needed
    TokenCounter.logTokenUsage(messages, `Custom API ${config.provider} Streaming`);
    
    if (TokenCounter.wouldExceedLimit(messages, TokenCounter.getSafeGroqLimit())) {
      console.warn(`[CUSTOM_API] Request too large for ${config.provider}, attempting to optimize...`);
      
      if (!suppliedMessages?.length && prompt.length > 15000) {
        console.log(`[CUSTOM_API] Truncating large prompt from ${prompt.length} to 15000 characters`);
        messages[0].content = prompt.substring(0, 15000) + "\n\n[Content truncated due to size limits]";
      }
    }
    
    // Determine API format based on provider
    const requestBody = this._formatRequestForProvider(messages, config, true);
    
    // Handle Gemini's special endpoint format
    const endpoint = this._getEndpointForProvider(config);
    
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this._getHeadersForProvider(config.provider, config.apiKey, config.endpoint)
      },
      body: JSON.stringify(requestBody)
    });
    
    if (!response.ok) {
      // Handle rate limit errors specifically
      if (response.status === 429) {
        const errorText = redactSecrets(await response.text());
        console.error(`[CUSTOM_API] Rate limit exceeded for ${config.provider}:`, errorText);
        throw new Error(`Rate limit exceeded for ${config.provider}. Please try again in a moment or reduce your request size.`);
      }
      throw new Error(`Custom API error: ${response.status} ${response.statusText}`);
    }
    
    // Handle different response formats
    yield* this._parseStreamingResponse(response, config.provider);
  }
  
  /**
   * Make non-streaming API call to custom endpoint
   * @param {string} prompt - The prompt to send
   * @param {Object} config - API configuration
   * @returns {AsyncGenerator} - Generator for response
   */
  static async *_callCustomAPINonStreaming(prompt, config, suppliedMessages) {
    const messages = suppliedMessages?.length ? suppliedMessages : [{ role: 'user', content: prompt }];
    
    // Determine API format based on provider
    const requestBody = this._formatRequestForProvider(messages, config, false);
    
    // Handle Gemini's special endpoint format
    const endpoint = this._getEndpointForProvider(config, false);
    
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this._getHeadersForProvider(config.provider, config.apiKey, config.endpoint)
      },
      body: JSON.stringify(requestBody)
    });
    
    if (!response.ok) {
      // Handle rate limit errors specifically
      if (response.status === 429) {
        const errorText = redactSecrets(await response.text());
        console.error(`[CUSTOM_API] Rate limit exceeded for ${config.provider}:`, errorText);
        throw new Error(`Rate limit exceeded for ${config.provider}. Please try again in a moment or reduce your request size.`);
      }
      throw new Error(`Custom API error: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    
    // Handle different response formats
    const content = this._extractContentFromNonStreamingResponse(data, config.provider);
    
    if (content) {
      // Simulate streaming by yielding word by word
      const words = content.split(' ');
      for (const word of words) {
        yield word + ' ';
        // Small delay to simulate streaming
        await new Promise(resolve => setTimeout(resolve, 30));
      }
    }
  }
  
  /**
   * Format request body for different providers
   * @param {Array} messages - Message array
   * @param {Object} config - API configuration
   * @param {boolean} streaming - Whether streaming is enabled
   * @returns {Object} - Formatted request body
   */
  static _formatRequestForProvider(messages, config, streaming = true) {
    const system = messages.filter(message => message.role === 'system').map(message => message.content).join('\n\n');
    const conversation = messages.filter(message => message.role !== 'system');
    switch (config.provider) {
      case 'ollama':
        return {
          model: config.model,
          prompt: conversation.length === 1 ? conversation[0].content : conversation.map(message => `${message.role}: ${message.content}`).join('\n\n'),
          ...(system ? { system } : {}),
          stream: streaming,
          options: {
            temperature: 0.7
          }
        };
        
      case 'openai':
      case 'open-router':
      case 'groq':
        return {
          messages: messages,
          model: config.model,
          temperature: 0.7,
          stream: streaming
        };
        
      case 'google-gemini':
        if (String(config.endpoint || '').includes('/openai/')) {
          return {
            messages: messages,
            model: config.model,
            temperature: 0.7,
            stream: streaming
          };
        }
        return {
          contents: conversation.map(message => ({
            role: message.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: message.content }],
          })),
          ...(system ? { systemInstruction: { parts: [{ text: system }] } } : {}),
          generationConfig: {
            temperature: 0.7
          }
        };
        
      case 'anthropic':
        return {
          model: config.model,
          max_tokens: 1000,
          messages: conversation,
          ...(system ? { system } : {}),
          stream: streaming
        };
        
      default:
        // Default to OpenAI format
        return {
          messages: messages,
          model: config.model,
          temperature: 0.7,
          stream: streaming
        };
    }
  }
  
  /**
   * Get endpoint URL for different providers
   * @param {Object} config - API configuration
   * @returns {string} - Endpoint URL
   */
  static _getEndpointForProvider(config, streaming = true) {
    let parsed;
    try {
      parsed = new URL(String(config.endpoint || '').trim());
    } catch {
      throw new Error('Custom API endpoint must be a valid URL.');
    }
    const host = parsed.hostname.toLowerCase();
    if (host === 'mlsysbook.ai' || host === 'www.mlsysbook.ai' || host === 'harvard-edge.github.io' || parsed.pathname.includes('/api/socratiq')) {
      throw new Error('Refusing to send a personal API key to the shared SocratiQ gateway.');
    }
    const isLocal = host === 'localhost' || host === '127.0.0.1';
    if (!isLocal && parsed.protocol !== 'https:') {
      throw new Error('Custom API endpoints must use HTTPS.');
    }
    parsed.username = '';
    parsed.password = '';
    parsed.searchParams.delete('key');
    parsed.searchParams.delete('api_key');
    const cleanEndpoint = parsed.toString();

    if (config.provider === 'google-gemini' && !cleanEndpoint.includes('/openai/')) {
      // Native Gemini endpoint: never put API key in the URL query string; use x-goog-api-key header instead.
      const model = config.model || 'gemini-2.5-flash';
      const baseUrl = cleanEndpoint.replace(/\/$/, '');
      return `${baseUrl}/models/${model}:${streaming ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
    }
    return cleanEndpoint;
  }

  /**
   * Get headers for different providers
   * @param {string} provider - Provider name
   * @param {string} apiKey - API key (if any)
   * @param {string} endpoint - Endpoint URL
   * @returns {Object} - Headers object
   */
  static _getHeadersForProvider(provider, apiKey = null, endpoint = '') {
    const headers = {};
    
    if (apiKey) {
      switch (provider) {
        case 'openai':
        case 'open-router':
        case 'groq':
          headers['Authorization'] = `Bearer ${apiKey}`;
          break;
        case 'google-gemini':
          if (String(endpoint || '').includes('/openai/')) {
            headers['Authorization'] = `Bearer ${apiKey}`;
          } else {
            headers['x-goog-api-key'] = apiKey;
          }
          break;
        case 'anthropic':
          headers['x-api-key'] = apiKey;
          headers['anthropic-version'] = '2023-06-01';
          headers['anthropic-dangerous-direct-browser-access'] = 'true';
          break;
        default:
          headers['Authorization'] = `Bearer ${apiKey}`;
      }
    }
    
    return headers;
  }
  
  /**
   * Parse streaming response from different providers
   * @param {Response} response - Fetch response object
   * @param {string} provider - Provider name
   * @returns {AsyncGenerator} - Generator for streaming content
   */
  static async *_parseStreamingResponse(response, provider) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;
          
          try {
            const parsed = JSON.parse(data);
            const content = this._extractContentFromStreamingResponse(parsed, provider);
            if (content) yield content;
          } catch (e) {
            console.error('[CUSTOM_API] Failed to parse streaming JSON:', e);
          }
        } else if (provider === 'ollama' && line.trim()) {
          // Ollama sends JSON objects directly without 'data: ' prefix
          try {
            const parsed = JSON.parse(line);
            const content = this._extractContentFromStreamingResponse(parsed, provider);
            if (content) yield content;
            if (parsed.done) return;
          } catch (e) {
            console.error('[CUSTOM_API] Failed to parse Ollama JSON:', e);
          }
        }
      }
    }
  }
  
  /**
   * Extract content from streaming response based on provider
   * @param {Object} parsed - Parsed JSON response
   * @param {string} provider - Provider name
   * @returns {string|null} - Extracted content
   */
  static _extractContentFromStreamingResponse(parsed, provider) {
    switch (provider) {
      case 'ollama':
        return parsed.response;
        
      case 'openai':
      case 'open-router':
      case 'groq':
        return parsed.choices?.[0]?.delta?.content;
        
      case 'google-gemini':
        return parsed.choices?.[0]?.delta?.content ?? parsed.candidates?.[0]?.content?.parts?.[0]?.text;
        
      case 'anthropic':
        return parsed.delta?.text;
        
      default:
        return parsed.choices?.[0]?.delta?.content || 
               parsed.response || 
               parsed.text ||
               parsed.delta?.text;
    }
  }
  
  /**
   * Extract content from non-streaming response based on provider
   * @param {Object} data - Response data
   * @param {string} provider - Provider name
   * @returns {string|null} - Extracted content
   */
  static _extractContentFromNonStreamingResponse(data, provider) {
    switch (provider) {
      case 'ollama':
        return data.response;
        
      case 'openai':
      case 'open-router':
      case 'groq':
        return data.choices?.[0]?.message?.content;
        
      case 'google-gemini':
        return data.choices?.[0]?.message?.content ?? data.candidates?.[0]?.content?.parts?.[0]?.text;
        
      case 'anthropic':
        return data.content?.[0]?.text;
        
      default:
        return data.choices?.[0]?.message?.content || 
               data.response || 
               data.text ||
               data.content?.[0]?.text;
    }
  }
  
  /**
   * Test custom API connection
   * @param {Object} config - API configuration
   * @returns {Promise<boolean>} - True if connection successful
   */
  static async testConnection(config) {
    try {
      const testPrompt = "Hello";
      const requestBody = this._formatRequestForProvider([{ role: 'user', content: testPrompt }], config, false);
      const endpoint = this._getEndpointForProvider(config, false);
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...this._getHeadersForProvider(config.provider, config.apiKey, config.endpoint)
        },
        body: JSON.stringify(requestBody)
      });
      
      return response.ok;
    } catch (error) {
      console.error('[CUSTOM_API] Connection test failed:', error);
      return false;
    }
  }
}
