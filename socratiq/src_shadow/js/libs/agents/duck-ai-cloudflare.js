import { callProviderSingle, callProviderStream } from './cloudflareAgent.js';
import { DEFAULT_PROVIDER, PROVIDER_MODELS } from '../../../configs/env_configs.js';
import { CustomAPIAdapter } from './custom-api-adapter.js';

// Shared gateway defaults are configured in one place.
const PROVIDER_CONFIGS = [{
    name: DEFAULT_PROVIDER.toLowerCase(),
    model: PROVIDER_MODELS[DEFAULT_PROVIDER].model,
    streaming: PROVIDER_MODELS[DEFAULT_PROVIDER].stream
}];

export class DuckAI {
    static instance = null;
    
    static async getInstance() {
        if (!DuckAI.instance) {
            DuckAI.instance = new DuckAI();
        }
        return DuckAI.instance;
    }
    
    /**
     * Generate an answer using the optional custom service or configured provider
     */
    async *generateAnswer(question) {
        const prompt = `Format your response in markdown. Answer this question or define this term concisely in max 2 to 3 sentences without any meta-commentary or introductory phrases: "${question}"`;
        
        // Try custom API first if configured
        const settings = this.getCurrentSettings();
        let customHasContent = false;
        if (CustomAPIAdapter.hasCustomAPIConfig(settings)) {
            try {
                console.log(`[CUSTOM_API] Attempting custom API: ${settings.customAPI.provider}`);
                const customResult = CustomAPIAdapter.callCustomAPI(prompt, settings, true);
                if (customResult) {
                    for await (const chunk of customResult) {
                        if (chunk) { customHasContent = true; yield chunk; }
                    }
                    if (!customHasContent) throw new Error('Custom API returned no content');
                    console.log(`[CUSTOM_API] Successfully completed with custom API: ${settings.customAPI.provider}`);
                    return; // Success with custom API
                }
            } catch (error) {
                if (customHasContent) throw error;
                console.warn('[CUSTOM_API] Custom API failed, falling back to default providers:', error);
            }
        }
        
        // Fallback to default providers
        let lastError = null;
        
        for (const provider of PROVIDER_CONFIGS) {
            try {
                console.log(`Trying provider: ${provider.name}`);
                
                if (provider.streaming) {
                    yield* this._generateAnswerStreaming(prompt, provider);
                } else {
                    yield* this._generateAnswerNonStreaming(prompt, provider);
                }
                
                console.log(`Successfully completed with ${provider.name}`);
                return; // Success, exit the retry loop
                
            } catch (error) {
                lastError = error;
                console.error(`${provider.name} failed:`, error);
                // Continue to next provider
            }
        }
        
        // If all providers failed
        throw new Error(`All providers failed. Last error: ${lastError?.message || 'Unknown error'}`);
    }
    
    /**
     * Generate flashcards using the optional custom service or configured provider
     */
    async *generateFlashcards(text) {
        const prompt = `Create up to 3 flashcards from this text. Format as JSON array with "question" and "answer" fields. Make questions concise but specific. Answers should be clear and informative. Text: "${text}"

Example format:
[
    {"question": "What is X?", "answer": "X is..."},
    {"question": "How does Y work?", "answer": "Y works by..."}
]
 IMPORTANT: OUTPUT ONLY JSON ARRAY, NOTHING ELSE`;

        // Try custom API first if configured
        const settings = this.getCurrentSettings();
        let customHasContent = false;
        if (CustomAPIAdapter.hasCustomAPIConfig(settings)) {
            try {
                console.log(`[CUSTOM_API] Attempting custom API for flashcards: ${settings.customAPI.provider}`);
                const customResult = CustomAPIAdapter.callCustomAPI(prompt, settings, true);
                if (customResult) {
                    for await (const chunk of customResult) {
                        if (chunk) { customHasContent = true; yield chunk; }
                    }
                    if (!customHasContent) throw new Error('Custom API returned no content');
                    console.log(`[CUSTOM_API] Successfully completed flashcards with custom API: ${settings.customAPI.provider}`);
                    return; // Success with custom API
                }
            } catch (error) {
                if (customHasContent) throw error;
                console.warn('[CUSTOM_API] Custom API failed for flashcards, falling back to default providers:', error);
            }
        }

        // Fallback to default providers
        let lastError = null;
        
        for (const provider of PROVIDER_CONFIGS) {
            try {
                console.log(`Trying provider for flashcards: ${provider.name}`);
                
                if (provider.streaming) {
                    const result = yield* this._generateFlashcardsStreaming(prompt, provider);
                    if (result) {
                        console.log(`Successfully completed flashcards with ${provider.name}`);
                        return result;
                    }
                } else {
                    const result = yield* this._generateFlashcardsNonStreaming(prompt, provider);
                    if (result) {
                        console.log(`Successfully completed flashcards with ${provider.name}`);
                        return result;
                    }
                }
                
                console.log(`Successfully completed flashcards with ${provider.name}`);
                return; // Success, exit the retry loop
                
            } catch (error) {
                lastError = error;
                console.error(`${provider.name} failed for flashcards:`, error);
                // Continue to next provider
            }
        }
        
        // If all providers failed
        throw new Error(`All providers failed for flashcards. Last error: ${lastError?.message || 'Unknown error'}`);
    }
    
    /**
     * Streaming answer generation
     */
    async *_generateAnswerStreaming(prompt, provider) {
        yield* callProviderStream(prompt, provider.name);
    }
    
    /**
     * Non-streaming answer generation
     */
    async *_generateAnswerNonStreaming(prompt, provider) {
        yield await callProviderSingle(prompt, provider.name);
    }
    
    /**
     * Streaming flashcards generation
     */
    async *_generateFlashcardsStreaming(prompt, provider) {
        let content = '';
        // Parse only after the shared transport confirms a complete response.
        for await (const chunk of callProviderStream(prompt, provider.name)) {
            content += chunk;
            yield chunk;
        }
        const result = this._parseFlashcardsJSON(content);
        if (!Array.isArray(result) || !result.length) throw new Error('Provider returned no valid flashcards');
        return result;
    }
    
    /**
     * Non-streaming flashcards generation
     */
    async *_generateFlashcardsNonStreaming(prompt, provider) {
        const content = await callProviderSingle(prompt, provider.name);
        const result = this._parseFlashcardsJSON(content);
        if (!Array.isArray(result) || !result.length) throw new Error('Provider returned no valid flashcards');
        yield content;
        return result;
    }
    
    /**
     * Parse flashcards JSON from content
     */
    _parseFlashcardsJSON(content) {
        try {

            
            // First try to parse the entire content as JSON
            if (content.trim().startsWith('[') && content.trim().endsWith(']')) {
                const finalResult = JSON.parse(content.trim());

                return finalResult;
            }
            
            // Extract JSON array from the content using multiple patterns
            const patterns = [
                /\[[\s\S]*?\]/,  // Original pattern
                /\[[\s\S]*\]/,   // Greedy pattern
                /\[.*?\]/s       // Dotall pattern
            ];
            
            for (const pattern of patterns) {
                const jsonMatch = content.match(pattern);
                if (jsonMatch) {
                    const jsonArrayString = jsonMatch[0];

                    try {
                        const finalResult = JSON.parse(jsonArrayString);

                        return finalResult;
                    } catch (parseError) {
                        console.warn("Failed to parse JSON match, trying next pattern:", parseError);
                        continue;
                    }
                }
            }
            
            console.error("No flashcard JSON array found in response");
            return null;
        } catch (e) {
            console.error("Failed to parse flashcards JSON:", e);
            return null;
        }
    }
    
    /**
     * Get current user settings from localStorage
     * @returns {Object} - User settings object
     */
    getCurrentSettings() {
        try {
            const savedSettings = localStorage.getItem("userSettings");
            return savedSettings ? JSON.parse(savedSettings) : {};
        } catch (error) {
            console.error('[CUSTOM_API] Error loading settings:', error);
            return {};
        }
    }
}