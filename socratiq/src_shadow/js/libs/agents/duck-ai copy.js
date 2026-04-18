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
            const prompt = `Format your response in markdown. Answer this question or define this term concisely without any meta-commentary or introductory phrases: "${question}"`;
            
            const response = await fetch(SERVELESSURLDuckAI, {//'//https://ddg-ai-worker.duckai.workers.dev', {
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
                            if (parsed.text) {
                                yield parsed.text;
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
            console.log("Starting generateFlashcards with text:", text.substring(0, 100) + "...");
            
            const prompt = `Create up to 3 flashcards from this text. Format as JSON array with "question" and "answer" fields. Make questions concise but specific. Answers should be clear and informative. Text: "${text}"

Example format:
[
    {"question": "What is X?", "answer": "X is..."},
    {"question": "How does Y work?", "answer": "Y works by..."}
]`;

            console.log("Sending request to AI worker...");
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

            console.log("Starting to process response stream...");
            let jsonContent = '';
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let insideCodeBlock = false;

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    console.log("Stream complete");
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') {
                            console.log("Received [DONE] signal");
                            break;
                        }

                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.text) {
                                const text = parsed.text;
                                console.log("Received text chunk:", text);
                                
                                if (text.includes('```json')) {
                                    console.log("Found start of JSON block");
                                    insideCodeBlock = true;
                                    continue;
                                }
                                if (text.includes('```') && insideCodeBlock) {
                                    console.log("Found end of JSON block");
                                    insideCodeBlock = false;
                                    continue;
                                }
                                if (insideCodeBlock) {
                                    console.log("Adding to JSON content:", text);
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
            try {
                const repairedJson = jsonrepair(jsonContent);
                console.log("Repaired JSON:", repairedJson);
                const flashcards = JSON.parse(repairedJson);
                console.log("Successfully parsed flashcards:", flashcards);
                return flashcards;
            } catch (e) {
                console.error('Failed to repair/parse JSON:', e);
                console.error('JSON content that failed to parse:', jsonContent);
                return null;
            }
        } catch (error) {
            console.error("Error generating flashcards:", error);
            console.error("Error stack:", error.stack);
            throw error;
        }
    }
}