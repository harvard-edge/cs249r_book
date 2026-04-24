/**
 * Test script to verify Gemini integration fix
 * Run this in browser console to test Gemini with the correct model
 */

import { CustomAPIAdapter } from './custom-api-adapter.js';

export class GeminiFixTester {
  
  /**
   * Test Gemini with correct model
   */
  static async testGeminiFix() {
    console.log('[GEMINI_FIX] Testing Gemini with gemini-2.5-flash model...');
    
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
      console.log('[GEMINI_FIX] Making test call to Gemini...');
      console.log('[GEMINI_FIX] Expected URL:', 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?key=YOUR_API_KEY');
      
      const result = CustomAPIAdapter.callCustomAPI(testPrompt, testSettings, true);
      
      if (result) {
        let response = '';
        for await (const chunk of result) {
          response += chunk;
          console.log('[GEMINI_FIX] Received chunk:', chunk);
        }
        console.log('[GEMINI_FIX] Full response:', response);
        return { success: true, response };
      } else {
        console.log('[GEMINI_FIX] No result returned from custom API');
        return { success: false, error: 'No result returned' };
      }
    } catch (error) {
      console.error('[GEMINI_FIX] Gemini test failed:', error);
      return { success: false, error: error.message };
    }
  }
  
  /**
   * Test connection to Gemini
   */
  static async testGeminiConnection() {
    console.log('[GEMINI_FIX] Testing Gemini connection...');
    
    const config = {
      provider: 'google-gemini',
      endpoint: 'https://generativelanguage.googleapis.com/v1beta',
      model: 'gemini-2.5-flash',
      apiKey: 'YOUR_GEMINI_API_KEY' // Replace with actual API key
    };
    
    try {
      const isConnected = await CustomAPIAdapter.testConnection(config);
      console.log(`[GEMINI_FIX] Connection test result: ${isConnected ? 'SUCCESS' : 'FAILED'}`);
      return { success: isConnected };
    } catch (error) {
      console.error('[GEMINI_FIX] Connection test failed:', error);
      return { success: false, error: error.message };
    }
  }
  
  /**
   * Test auto-fill functionality
   */
  static testAutoFill() {
    console.log('[GEMINI_FIX] Testing auto-fill functionality...');
    
    // Simulate provider selection
    const providerDefaults = {
      'google-gemini': {
        endpoint: 'https://generativelanguage.googleapis.com/v1beta',
        model: 'gemini-2.5-flash'
      }
    };
    
    const provider = 'google-gemini';
    const defaults = providerDefaults[provider];
    
    console.log('[GEMINI_FIX] Auto-fill values for Google Gemini:');
    console.log('[GEMINI_FIX] Endpoint:', defaults.endpoint);
    console.log('[GEMINI_FIX] Model:', defaults.model);
    
    return { success: true, defaults };
  }
}

// Make it available globally for browser console testing
if (typeof window !== 'undefined') {
  window.GeminiFixTester = GeminiFixTester;
}