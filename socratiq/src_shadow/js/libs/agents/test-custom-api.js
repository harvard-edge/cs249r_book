/**
 * Test script for custom API integration
 * This can be run in the browser console to test the integration
 */

import { CustomAPIAdapter } from './custom-api-adapter.js';

export class CustomAPITester {
  
  /**
   * Test Gemini integration
   */
  static async testGemini() {
    console.log('[TEST] Testing Gemini integration...');
    
    const testSettings = {
      customAPI: {
        provider: 'google-gemini',
        endpoint: 'https://generativelanguage.googleapis.com/v1beta',
        model: 'gemini-2.5-flash',
        apiKey: 'YOUR_GEMINI_API_KEY', // Replace with actual API key
        saveLocally: false
      }
    };
    
    const testPrompt = "What is the capital of France?";
    
    try {
      console.log('[TEST] Making test call to Gemini...');
      const result = CustomAPIAdapter.callCustomAPI(testPrompt, testSettings, true);
      
      if (result) {
        let response = '';
        for await (const chunk of result) {
          response += chunk;
          console.log('[TEST] Received chunk:', chunk);
        }
        console.log('[TEST] Full response:', response);
        return { success: true, response };
      } else {
        console.log('[TEST] No result returned from custom API');
        return { success: false, error: 'No result returned' };
      }
    } catch (error) {
      console.error('[TEST] Gemini test failed:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Test Ollama integration
   */
  static async testOllama() {
    console.log('[TEST] Testing Ollama integration...');
    
    const testSettings = {
      customAPI: {
        provider: 'ollama',
        endpoint: 'http://localhost:11434/api/generate',
        model: 'gemma3:270m',
        apiKey: '',
        saveLocally: false
      }
    };
    
    const testPrompt = "What is the capital of France?";
    
    try {
      console.log('[TEST] Making test call to Ollama...');
      const result = CustomAPIAdapter.callCustomAPI(testPrompt, testSettings, true);
      
      if (result) {
        let response = '';
        for await (const chunk of result) {
          response += chunk;
          console.log('[TEST] Received chunk:', chunk);
        }
        console.log('[TEST] Full response:', response);
        return { success: true, response };
      } else {
        console.log('[TEST] No result returned from custom API');
        return { success: false, error: 'No result returned' };
      }
    } catch (error) {
      console.error('[TEST] Ollama test failed:', error);
      return { success: false, error: error.message };
    }
  }
  
  /**
   * Test connection to custom API
   */
  static async testConnection(provider, endpoint, model = 'default') {
    console.log(`[TEST] Testing connection to ${provider} at ${endpoint}...`);
    
    const config = {
      provider: provider,
      endpoint: endpoint,
      model: model,
      apiKey: ''
    };
    
    try {
      const isConnected = await CustomAPIAdapter.testConnection(config);
      console.log(`[TEST] Connection test result: ${isConnected ? 'SUCCESS' : 'FAILED'}`);
      return { success: isConnected };
    } catch (error) {
      console.error('[TEST] Connection test failed:', error);
      return { success: false, error: error.message };
    }
  }
  
  /**
   * Test settings integration
   */
  static testSettingsIntegration() {
    console.log('[TEST] Testing settings integration...');
    
    // Test saving settings
    const testSettings = {
      customAPI: {
        provider: 'ollama',
        endpoint: 'http://localhost:11434/api/generate',
        model: 'gemma3:270m',
        apiKey: '',
        saveLocally: true
      }
    };
    
    localStorage.setItem('userSettings', JSON.stringify(testSettings));
    console.log('[TEST] Settings saved to localStorage');
    
    // Test loading settings
    const loadedSettings = JSON.parse(localStorage.getItem('userSettings') || '{}');
    console.log('[TEST] Settings loaded from localStorage:', loadedSettings);
    
    // Test custom API detection
    const hasCustomAPI = CustomAPIAdapter.hasCustomAPIConfig(loadedSettings);
    console.log('[TEST] Custom API detected:', hasCustomAPI);
    
    if (hasCustomAPI) {
      const config = CustomAPIAdapter.getCustomAPIConfig(loadedSettings);
      console.log('[TEST] Custom API config:', config);
    }
    
    return { success: true, settings: loadedSettings };
  }
  
  /**
   * Run all tests
   */
  static async runAllTests() {
    console.log('[TEST] Starting custom API integration tests...');
    
    const results = {
      settings: null,
      connection: null,
      ollama: null,
      gemini: null
    };
    
    // Test 1: Settings integration
    try {
      results.settings = this.testSettingsIntegration();
    } catch (error) {
      console.error('[TEST] Settings test failed:', error);
      results.settings = { success: false, error: error.message };
    }
    
    // Test 2: Connection test (only if Ollama is running)
    try {
      results.connection = await this.testConnection('ollama', 'http://localhost:11434/api/generate', 'gemma3:270m');
    } catch (error) {
      console.error('[TEST] Connection test failed:', error);
      results.connection = { success: false, error: error.message };
    }
    
    // Test 3: Ollama integration (only if Ollama is running)
    if (results.connection?.success) {
      try {
        results.ollama = await this.testOllama();
      } catch (error) {
        console.error('[TEST] Ollama test failed:', error);
        results.ollama = { success: false, error: error.message };
      }
    } else {
      results.ollama = { success: false, error: 'Skipped - Ollama not running' };
    }

    // Test 4: Gemini integration (requires API key)
    try {
      results.gemini = await this.testGemini();
    } catch (error) {
      console.error('[TEST] Gemini test failed:', error);
      results.gemini = { success: false, error: error.message };
    }
    
    console.log('[TEST] All tests completed:', results);
    return results;
  }
}

// Make it available globally for browser console testing
if (typeof window !== 'undefined') {
  window.CustomAPITester = CustomAPITester;
}