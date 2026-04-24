/**
 * Gemini Adapter for Cloudflare AI Proxy
 * Handles Gemini-specific request/response formatting
 */

export class GeminiAdapter {
    /**
     * Convert OpenAI format to Gemini format
     */
    static convertRequestToGeminiFormat(openaiRequest) {
        const { messages, model, temperature, top_p } = openaiRequest;
        
        // Convert messages to Gemini format
        const contents = messages.map(msg => {
            if (msg.role === 'system') {
                // Gemini uses system_instruction instead of system messages
                return null;
            }
            
            return {
                role: msg.role === 'assistant' ? 'model' : 'user',
                parts: this._convertContentToParts(msg.content)
            };
        }).filter(Boolean);
        
        // Extract system instruction if present
        const systemInstruction = messages.find(msg => msg.role === 'system');
        
        const geminiRequest = {
            contents,
            generationConfig: {
                temperature: temperature || 0.7,
                topP: top_p || 0.9,
                stopSequences: []
            }
        };
        
        // Add system instruction if present
        if (systemInstruction) {
            geminiRequest.systemInstruction = {
                parts: this._convertContentToParts(systemInstruction.content)
            };
        }
        
        return geminiRequest;
    }
    
    /**
     * Convert Gemini response to OpenAI format
     */
    static convertResponseFromGeminiFormat(geminiResponse) {
        const { candidates, usageMetadata } = geminiResponse;
        
        if (!candidates || candidates.length === 0) {
            throw new Error('No candidates in Gemini response');
        }
        
        const candidate = candidates[0];
        const content = candidate.content;
        
        if (!content || !content.parts || content.parts.length === 0) {
            throw new Error('No content parts in Gemini response');
        }
        
        // Extract text from parts
        const text = content.parts
            .filter(part => part.text)
            .map(part => part.text)
            .join('');
        
        // Convert to OpenAI format
        const openaiResponse = {
            id: `gemini-${Date.now()}`,
            object: 'chat.completion',
            created: Math.floor(Date.now() / 1000),
            model: 'gemini-2.5-flash',
            choices: [
                {
                    index: 0,
                    message: {
                        role: 'assistant',
                        content: text
                    },
                    finish_reason: candidate.finishReason || 'stop'
                }
            ],
            usage: {
                prompt_tokens: usageMetadata?.promptTokenCount || 0,
                completion_tokens: usageMetadata?.candidatesTokenCount || 0,
                total_tokens: usageMetadata?.totalTokenCount || 0
            }
        };
        
        return openaiResponse;
    }
    
    /**
     * Convert content to Gemini parts format
     */
    static _convertContentToParts(content) {
        if (typeof content === 'string') {
            return [{ text: content }];
        }
        
        if (Array.isArray(content)) {
            return content.map(item => {
                if (item.type === 'text') {
                    return { text: item.text };
                } else if (item.type === 'image_url') {
                    return {
                        inline_data: {
                            mime_type: this._getMimeTypeFromUrl(item.image_url.url),
                            data: this._extractBase64FromUrl(item.image_url.url)
                        }
                    };
                }
                return { text: String(item) };
            });
        }
        
        return [{ text: String(content) }];
    }
    
    /**
     * Extract base64 data from data URL
     */
    static _extractBase64FromUrl(url) {
        if (url.startsWith('data:')) {
            const base64Index = url.indexOf(',');
            if (base64Index !== -1) {
                return url.substring(base64Index + 1);
            }
        }
        return url;
    }
    
    /**
     * Get MIME type from URL
     */
    static _getMimeTypeFromUrl(url) {
        if (url.startsWith('data:')) {
            const mimeIndex = url.indexOf(':');
            const commaIndex = url.indexOf(',');
            if (mimeIndex !== -1 && commaIndex !== -1) {
                return url.substring(mimeIndex + 1, commaIndex);
            }
        }
        
        // Default to JPEG if we can't determine
        return 'image/jpeg';
    }
    
    /**
     * Handle Gemini streaming response format
     */
    static convertStreamingResponseFromGemini(geminiChunk) {
        try {
            const parsed = JSON.parse(geminiChunk);
            
            if (parsed.candidates && parsed.candidates[0] && parsed.candidates[0].content) {
                const content = parsed.candidates[0].content;
                const text = content.parts?.[0]?.text || '';
                
                return {
                    choices: [
                        {
                            index: 0,
                            delta: {
                                content: text
                            },
                            finish_reason: parsed.candidates[0].finishReason || null
                        }
                    ]
                };
            }
            
            return null;
        } catch (e) {
            console.error('Failed to parse Gemini streaming chunk:', e);
            return null;
        }
    }
    
    /**
     * Check if request contains images
     */
    static hasImages(messages) {
        return messages.some(msg => 
            Array.isArray(msg.content) && 
            msg.content.some(item => item.type === 'image_url')
        );
    }
    
    /**
     * Validate image format for Gemini
     */
    static validateImageFormat(imageUrl) {
        const supportedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
        
        if (imageUrl.startsWith('data:')) {
            const mimeType = this._getMimeTypeFromUrl(imageUrl);
            return supportedTypes.includes(mimeType);
        }
        
        // Assume URL images are supported
        return true;
    }
}