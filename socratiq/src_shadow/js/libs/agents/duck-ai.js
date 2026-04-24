import { jsonrepair } from 'jsonrepair';
import {SERVELESSURLDuckAI} from '../../../configs/env_configs'
let tempURL = "http://localhost:8787"

export class DuckAI {
    static instance = null;
    
    static async getInstance() {
        if (!DuckAI.instance) {
            DuckAI.instance = new DuckAI();
        }
        return DuckAI.instance;
    }
    
    async *generateAnswer(question) {
        try {
            const prompt = `Format your response in markdown. Answer this question or define this term concisely in max 2 to 3 sentnces without any meta-commentary or introductory phrases: "${question}"`;
            
            const response = await fetch(SERVELESSURLDuckAI, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    model: 'claude-3-haiku'
                })
            });

            if (!response.ok) {
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
                buffer = lines.pop() || ''; // Keep the last partial line
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6); // Remove 'data: ' prefix
                        if (data === '[DONE]') return;
                        
                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.text && parsed.text.startsWith('data: ')) {
                                // Parse the nested JSON string
                                const innerData = JSON.parse(parsed.text.slice(6));
                                if (innerData.response) {
                                    yield innerData.response;
                                }
                            }
                        } catch (e) {
                            console.error('Failed to parse JSON:', e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error("Error generating answer:", error);
            throw error;
        }
    }

    async *generateFlashcards(text) {
        try {
            
            const prompt = `Create up to 3 flashcards from this text. Format as JSON array with "question" and "answer" fields. Make questions concise but specific. Answers should be clear and informative. Text: "${text}"

Example format:
[
    {"question": "What is X?", "answer": "X is..."},
    {"question": "How does Y work?", "answer": "Y works by..."}
]
    IMPORTANT: OUTPUT ONLY JSON ARRAY, NOTHING ELSE`;

            const response = await fetch(SERVELESSURLDuckAI, { //'https://ddg-ai-worker.duckai.workers.dev', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    model: 'claude-3-haiku'
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            let jsonContent = '';
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let insideCodeBlock = false;

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') {
                            break;
                        }

                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.text) {
                                const text = parsed.text;
                                
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
                                    // If not in a code block, try to collect JSON-like content
                                    if (text.includes('[') || text.includes('{')) {
                                        jsonContent += text;
                                    }
                                }
                                yield text;
                            }
                        } catch (e) {
                            console.error('Failed to parse JSON:', e);
                            console.error('Problematic line:', line);
                        }
                    }
                }
            }

            // Try to repair and parse the collected JSON content
            console.log("Attempting to repair JSON content:", jsonContent);
            let jsonString = '';

            try {
                // Accumulate the fragments into a complete JSON string
                jsonContent.split('\n').forEach(line => {
                    if (line.trim().startsWith('data: ')) {
                        const dataContent = line.substring(6).trim();
                        if (dataContent !== '[DONE]') {
                            try {
                                const parsed = JSON.parse(dataContent);
                                if (parsed.response) {
                                    jsonString += parsed.response;
                                }
                            } catch (e) {
                                console.warn('Failed to parse data chunk:', dataContent);
                            }
                        }
                    }
                });

                // Extract just the JSON array from the text response
                const jsonMatch = jsonString.match(/\[[\s\S]*\]/);
                if (jsonMatch) {
                    const jsonArrayString = jsonMatch[0];
                    const finalResult = JSON.parse(jsonArrayString);
                    console.log("Successfully parsed JSON:", finalResult);
                    return finalResult;
                } else {
                    console.error("No JSON array found in response");
                    return null;
                }
            } catch (e) {
                console.error("Failed to parse final JSON:", jsonString);
                console.error(e);
                return null;
            }
        } catch (error) {
            console.error("Error generating flashcards:", error);
            console.error("Error stack:", error.stack);
            throw error;
        }
    }
}