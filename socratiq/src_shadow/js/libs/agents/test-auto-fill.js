/**
 * Test script to verify auto-fill functionality
 * Run this in browser console to test provider auto-fill
 */

export class AutoFillTester {
  
  /**
   * Test auto-fill for all providers
   */
  static testAllProviders() {
    console.log('[AUTO_FILL_TEST] Testing auto-fill for all providers...');
    
    const providerDefaults = {
      'google-gemini': {
        endpoint: 'https://generativelanguage.googleapis.com/v1beta',
        model: 'gemini-2.5-flash'
      },
      'ollama': {
        endpoint: 'http://localhost:11434/api/generate',
        model: 'gemma3:270m'
      },
      'openai': {
        endpoint: 'https://api.openai.com/v1/chat/completions',
        model: 'gpt-3.5-turbo'
      },
      'open-router': {
        endpoint: 'https://openrouter.ai/api/v1/chat/completions',
        model: 'meta-llama/llama-4-scout:free'
      },
      'groq': {
        endpoint: 'https://api.groq.com/openai/v1/chat/completions',
        model: 'llama-3.1-8b-instant'
      },
      'anthropic': {
        endpoint: 'https://api.anthropic.com/v1/messages',
        model: 'claude-3-sonnet-20240229'
      }
    };
    
    console.log('[AUTO_FILL_TEST] Provider defaults:');
    Object.entries(providerDefaults).forEach(([provider, config]) => {
      console.log(`[AUTO_FILL_TEST] ${provider}:`, config);
    });
    
    return providerDefaults;
  }
  
  /**
   * Test specific provider auto-fill
   */
  static testProvider(provider) {
    console.log(`[AUTO_FILL_TEST] Testing auto-fill for ${provider}...`);
    
    const providerDefaults = {
      'google-gemini': {
        endpoint: 'https://generativelanguage.googleapis.com/v1beta',
        model: 'gemini-2.5-flash'
      },
      'ollama': {
        endpoint: 'http://localhost:11434/api/generate',
        model: 'gemma3:270m'
      },
      'openai': {
        endpoint: 'https://api.openai.com/v1/chat/completions',
        model: 'gpt-3.5-turbo'
      },
      'open-router': {
        endpoint: 'https://openrouter.ai/api/v1/chat/completions',
        model: 'meta-llama/llama-4-scout:free'
      },
      'groq': {
        endpoint: 'https://api.groq.com/openai/v1/chat/completions',
        model: 'llama-3.1-8b-instant'
      },
      'anthropic': {
        endpoint: 'https://api.anthropic.com/v1/messages',
        model: 'claude-3-sonnet-20240229'
      }
    };
    
    const config = providerDefaults[provider];
    if (config) {
      console.log(`[AUTO_FILL_TEST] ${provider} auto-fill:`, config);
      return { success: true, config };
    } else {
      console.log(`[AUTO_FILL_TEST] Unknown provider: ${provider}`);
      return { success: false, error: 'Unknown provider' };
    }
  }
  
  /**
   * Test OpenRouter specific configuration
   */
  static testOpenRouter() {
    console.log('[AUTO_FILL_TEST] Testing OpenRouter configuration...');
    
    const openRouterConfig = {
      provider: 'open-router',
      endpoint: 'https://openrouter.ai/api/v1/chat/completions',
      model: 'meta-llama/llama-4-scout:free',
      apiKey: 'sk-or-...',
      saveLocally: false
    };
    
    console.log('[AUTO_FILL_TEST] OpenRouter config:', openRouterConfig);
    
    // Test URL construction
    const expectedUrl = `${openRouterConfig.endpoint}`;
    console.log('[AUTO_FILL_TEST] Expected OpenRouter URL:', expectedUrl);
    
    return { success: true, config: openRouterConfig };
  }
}

// Make it available globally for browser console testing
if (typeof window !== 'undefined') {
  window.AutoFillTester = AutoFillTester;
}