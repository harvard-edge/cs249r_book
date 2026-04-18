const API_URL =
  "https://proxy-worker.mlsysbook.workers.dev/ai";
import { alert } from "../utils/utils.js";
// Configuration for different providers
const providerConfigs = {
  GROQ: {
    model: "llama-3.3-70b-versatile",
    stream: true,
  },
  MISTRAL: {
    model: "mistral-large-latest",
    stream: false,
  },
  OPEN: {
    model: "qwen/qwen-2-7b-instruct:free",
    temperature: 0.7,
    top_p: 0.9,
    stream: true,
  },
  HUGGINGFACE: {
    temperature: 0.5,
    top_p: 0.7,
    stream: false,
  },
  GEMINI: {
    responseMimeType: "text/plain",
  },
};

// Get target URL for each provider
function getTargetUrl(provider) {
  const urls = {
    GROQ: "https://api.groq.com/openai/v1/chat/completions",
    MISTRAL: "https://api.mistral.ai/v1/chat/completions",
    OPEN: "https://api.openai.com/v1/chat/completions",
    HUGGINGFACE: "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large",
    GEMINI: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
  };
  
  return urls[provider] || urls.GROQ; // Default to Groq
}

// Modified callProvider function to handle streaming properly
async function* callProvider(prompt, provider, signal) {
  const config = providerConfigs[provider];

  const messages = [
    {
      role: "user",
      content: prompt,
    },
  ];

  // Get the target URL for the provider
  const targetUrl = getTargetUrl(provider);

  try {
    const response = await fetch(`${API_URL}?url=${encodeURIComponent(targetUrl)}&provider=${provider.toLowerCase()}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages,
        model: config.model,
        temperature: config.temperature || 0.7,
        top_p: config.top_p || 0.9,
        stream: config.stream || false,
      }),
      signal,
    });

    if (!response.ok) {
      throw new Error(`${provider} failed with status ${response.status}`);
    }


    // Handle streaming responses
    if (config.stream) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Decode the chunk and add to buffer
        const newText = decoder.decode(value, { stream: true });
        buffer += newText;
        
        // Find complete JSON objects in the buffer
        let startIndex = 0;
        let curlyIndex;
        
        while ((curlyIndex = buffer.indexOf('{', startIndex)) !== -1) {
          try {
            // Find the matching closing brace
            let openBraces = 1;
            let endIndex = curlyIndex + 1;
            
            while (openBraces > 0 && endIndex < buffer.length) {
              if (buffer[endIndex] === '{') openBraces++;
              if (buffer[endIndex] === '}') openBraces--;
              endIndex++;
            }
            
            if (openBraces === 0) {
              // We found a complete JSON object
              const jsonStr = buffer.substring(curlyIndex, endIndex);
              try {
                const parsed = JSON.parse(jsonStr);
                if (parsed.choices?.[0]?.delta?.content) {
                  yield parsed.choices[0].delta.content;
                }
                // Move start index past this object
                startIndex = endIndex;
              } catch (e) {
                // If we can't parse this as JSON, move to next potential object
                startIndex = curlyIndex + 1;
              }
            } else {
              // Incomplete JSON object, break out of inner loop
              break;
            }
          } catch (e) {
            // If anything goes wrong, move to next potential object
            startIndex = curlyIndex + 1;
          }
        }
        
        // Keep only the part of buffer that might contain an incomplete object
        buffer = buffer.substring(startIndex);
      }

      // Process any remaining complete JSON in buffer
      if (buffer.trim()) {
        try {
          const parsed = JSON.parse(buffer);
          if (parsed.choices?.[0]?.delta?.content) {
            yield parsed.choices[0].delta.content;
          }
        } catch (e) {
          console.warn('Failed to parse final buffer:', e);
        }
      }
    } else {
      // For non-streaming responses
      const result = await response.json();



      yield result.text || result.message || JSON.stringify(result);
    }
  } catch (error) {
    console.error("Error in callProvider:", error);
    if (error.name === "AbortError") {
      throw new Error("Request was cancelled");
    }
    throw error;
  }
}

// Modified tryMultipleProviders to handle streaming
async function* tryMultipleProviders(
  params,
  token,
  stream = true,
  json = false,
) {
  const preferredOrder = ["GROQ", "OPEN", "MISTRAL", "HUGGINGFACE", "GEMINI"];
  const controller = new AbortController();
  const { signal } = controller;

  let lastError = null;

  for (const provider of preferredOrder) {
    try {

      // Get the generator from callProvider
      const generator = callProvider(params.prompt, provider, signal);

      // Yield each chunk from the generator
      for await (const chunk of generator) {
        yield chunk;
      }

      // If we get here, we successfully completed streaming
      return;
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
// Export functions if needed
export { callProvider, tryMultipleProviders };
