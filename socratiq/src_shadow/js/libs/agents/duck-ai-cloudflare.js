import { jsonrepair } from 'jsonrepair';
import { TokenCounter } from '../utils/tokenCounter.js';
import { CustomAPIAdapter } from './custom-api-adapter.js';

// Cloudflare AI Proxy endpoints
const CLOUDFLARE_PROXY_STREAMING = "https://proxy-worker-streaming.mlsysbook.workers.dev/ai/stream";
const CLOUDFLARE_PROXY_REGULAR = "https://proxy-worker.mlsysbook.workers.dev/ai";

// Provider configurations with fallback order - ordered by speed
const PROVIDER_CONFIGS = [
    {
        name: "groq",
        url: "https://api.groq.com/openai/v1/chat/completions",
        model: "llama-3.1-8b-instant",
        streaming: true
    },
    {
        name: "gemini",
        url: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        model: "gemini-2.5-flash",
        streaming: true
    },
    {
        name: "cerebras",
        url: "https://api.cerebras.ai/v1/chat/completions",
        model: "gpt-oss-120b",
        streaming: true
    },
    {
        name: "sambanova",
        url: "https://api.sambanova.ai/v1/chat/completions",
        model: "gpt-oss-120b",
        streaming: true
    },
    {
        name: "mistral",
        url: "https://api.mistral.ai/v1/chat/completions", 
        model: "mistral-tiny",
        streaming: true
    },
    {
        name: "openai", 
        url: "https://openrouter.ai/api/v1/chat/completions",
        model: "deepseek/deepseek-chat-v3.1:free",
        streaming: true
    },
    {
        name: "huggingface",
        url: "https://router.huggingface.co/v1/chat/completions",
        model: "meta-llama/Llama-3.1-8B-Instruct:featherless-ai",
        streaming: false
    },
    {
        name: "awan",
        url: "https://api.awanllm.com/v1/chat/completions",
        model: "Meta-Llama-3.1-70B-Instruct",
        streaming: true
    }
];

export class DuckAI {
    static instance = null;
    
    static async getInstance() {
        if (!DuckAI.instance) {
            DuckAI.instance = new DuckAI();
        }
        return DuckAI.instance;
    }
    
    /**
     * Generate answer with multi-agent retry and fallback
     */
    async *generateAnswer(question) {
        const prompt = `Format your response in markdown. Answer this question or define this term concisely in max 2 to 3 sentnces without any meta-commentary or introductory phrases: "${question}"`;
        
        // Try custom API first if configured
        const settings = this.getCurrentSettings();
        if (settings?.customAPI?.endpoint) {
            try {
                console.log(`[CUSTOM_API] Attempting custom API: ${settings.customAPI.provider}`);
                const customResult = CustomAPIAdapter.callCustomAPI(prompt, settings, true);
                if (customResult) {
                    yield* customResult;
                    console.log(`[CUSTOM_API] Successfully completed with custom API: ${settings.customAPI.provider}`);
                    return; // Success with custom API
                }
            } catch (error) {
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
     * Generate flashcards with multi-agent retry and fallback
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
        if (settings?.customAPI?.endpoint) {
            try {
                console.log(`[CUSTOM_API] Attempting custom API for flashcards: ${settings.customAPI.provider}`);
                const customResult = CustomAPIAdapter.callCustomAPI(prompt, settings, true);
                if (customResult) {
                    yield* customResult;
                    console.log(`[CUSTOM_API] Successfully completed flashcards with custom API: ${settings.customAPI.provider}`);
                    return; // Success with custom API
                }
            } catch (error) {
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
        const messages = [{ role: 'user', content: prompt }];
        
        // Check token count and optimize if needed
        TokenCounter.logTokenUsage(messages, `${provider.name} Answer Streaming`);
        
        if (TokenCounter.wouldExceedLimit(messages, TokenCounter.getSafeGroqLimit())) {
            console.warn(`[TOKEN_WARNING] Request too large for ${provider.name}, attempting to optimize...`);
            
            if (prompt.length > 15000) {
                console.log(`[TOKEN_OPTIMIZATION] Truncating large prompt from ${prompt.length} to 15000 characters`);
                messages[0].content = prompt.substring(0, 15000) + "\n\n[Content truncated due to size limits]";
            }
        }

        const response = await fetch(`${CLOUDFLARE_PROXY_STREAMING}?url=${encodeURIComponent(provider.url)}&provider=${provider.name}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                messages: messages,
                model: provider.model,
                temperature: 0.7
            })
        });

        if (!response.ok) {
            // Handle rate limit errors specifically
            if (response.status === 429) {
                const errorText = await response.text();
                console.error(`[RATE_LIMIT] ${provider.name} rate limit exceeded:`, errorText);
                throw new Error(`Rate limit exceeded for ${provider.name}. Please try again in a moment or reduce your request size.`);
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }

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
                        
                        // Handle different response formats
                        if (parsed.choices?.[0]?.delta?.content) {
                            // Standard OpenAI format
                            yield parsed.choices[0].delta.content;
                        } else if (parsed.text) {
                            // Legacy format
                            yield parsed.text;
                        } else if (parsed.response) {
                            // Direct response format
                            yield parsed.response;
                        }
                    } catch (e) {
                        console.error('Failed to parse streaming JSON:', e);
                    }
                }
            }
        }
    }
    
    /**
     * Non-streaming answer generation (for Gemini)
     */
    async *_generateAnswerNonStreaming(prompt, provider) {
        const response = await fetch(`${CLOUDFLARE_PROXY_REGULAR}?url=${encodeURIComponent(provider.url)}&provider=${provider.name}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                messages: [{ role: 'user', content: prompt }],
                model: provider.model,
                temperature: 0.7
            })
        });

        if (!response.ok) {
            // Handle rate limit errors specifically
            if (response.status === 429) {
                const errorText = await response.text();
                console.error(`[RATE_LIMIT] ${provider.name} rate limit exceeded:`, errorText);
                throw new Error(`Rate limit exceeded for ${provider.name}. Please try again in a moment or reduce your request size.`);
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Handle different response formats
        let content = '';
        if (data.choices?.[0]?.message?.content) {
            content = data.choices[0].message.content;
        } else if (data.text) {
            content = data.text;
        } else if (data.response) {
            content = data.response;
        } else if (data.candidates?.[0]?.content?.parts?.[0]?.text) {
            // Gemini format
            content = data.candidates[0].content.parts[0].text;
        }
        
        // Simulate streaming by yielding word by word
        const words = content.split(' ');
        for (const word of words) {
            yield word + ' ';
            // Small delay to simulate streaming
            await new Promise(resolve => setTimeout(resolve, 30));
        }
    }
    
    /**
     * Streaming flashcards generation
     */
    async *_generateFlashcardsStreaming(prompt, provider) {
        const response = await fetch(`${CLOUDFLARE_PROXY_STREAMING}?url=${encodeURIComponent(provider.url)}&provider=${provider.name}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                messages: [{ role: 'user', content: prompt }],
                model: provider.model,
                temperature: 0.3
            })
        });

        if (!response.ok) {
            // Handle rate limit errors specifically
            if (response.status === 429) {
                const errorText = await response.text();
                console.error(`[RATE_LIMIT] ${provider.name} rate limit exceeded:`, errorText);
                throw new Error(`Rate limit exceeded for ${provider.name}. Please try again in a moment or reduce your request size.`);
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        let jsonContent = '';
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let insideCodeBlock = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') break;

                    try {
                        const parsed = JSON.parse(data);
                        let text = '';
                        
                        if (parsed.choices?.[0]?.delta?.content) {
                            text = parsed.choices[0].delta.content;
                        } else if (parsed.text) {
                            text = parsed.text;
                        } else if (parsed.response) {
                            text = parsed.response;
                        }
                        
                        if (text) {
                            if (text.includes('```json')) {
                                insideCodeBlock = true;
                                continue;
                            }
                            if (text.includes('```') && insideCodeBlock) {
                                insideCodeBlock = false;
                                continue;
                            }
                            if (insideCodeBlock) {
                                jsonContent += text;
                            } else {
                                if (text.includes('[') || text.includes('{')) {
                                    jsonContent += text;
                                }
                            }
                            yield text;
                        }
                    } catch (e) {
                        console.error('Failed to parse streaming JSON:', e);
                    }
                }
            }
        }

        // Try to parse the collected JSON content
        const result = this._parseFlashcardsJSON(jsonContent);
        if (result) {
            return result;
        }
    }
    
    /**
     * Non-streaming flashcards generation (for Gemini)
     */
    async *_generateFlashcardsNonStreaming(prompt, provider) {
        const response = await fetch(`${CLOUDFLARE_PROXY_REGULAR}?url=${encodeURIComponent(provider.url)}&provider=${provider.name}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                messages: [{ role: 'user', content: prompt }],
                model: provider.model,
                temperature: 0.3
            })
        });

        if (!response.ok) {
            // Handle rate limit errors specifically
            if (response.status === 429) {
                const errorText = await response.text();
                console.error(`[RATE_LIMIT] ${provider.name} rate limit exceeded:`, errorText);
                throw new Error(`Rate limit exceeded for ${provider.name}. Please try again in a moment or reduce your request size.`);
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Handle different response formats
        let content = '';
        if (data.choices?.[0]?.message?.content) {
            content = data.choices[0].message.content;
        } else if (data.text) {
            content = data.text;
        } else if (data.response) {
            content = data.response;
        } else if (data.candidates?.[0]?.content?.parts?.[0]?.text) {
            // Gemini format
            content = data.candidates[0].content.parts[0].text;
        }
        
        // Simulate streaming by yielding the content
        yield content;
        
        // Try to parse the JSON content
        const result = this._parseFlashcardsJSON(content);
        if (result) {
            return result;
        }
    }
    
    /**
     * Parse flashcards JSON from content
     */
    _parseFlashcardsJSON(content) {
        try {
            console.log("Parsing JSON content:", content);
            
            // First try to parse the entire content as JSON
            if (content.trim().startsWith('[') && content.trim().endsWith(']')) {
                const finalResult = JSON.parse(content.trim());
                console.log("Successfully parsed flashcards JSON directly:", finalResult);
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
                    console.log("Found JSON match:", jsonArrayString);
                    try {
                        const finalResult = JSON.parse(jsonArrayString);
                        console.log("Successfully parsed flashcards JSON:", finalResult);
                        return finalResult;
                    } catch (parseError) {
                        console.warn("Failed to parse JSON match, trying next pattern:", parseError);
                        continue;
                    }
                }
            }
            
            console.error("No JSON array found in response:", content);
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