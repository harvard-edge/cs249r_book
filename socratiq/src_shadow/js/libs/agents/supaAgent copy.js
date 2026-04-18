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

// Modified callProvider function to handle streaming properly
async function* callProvider(prompt, provider, signal) {
  console.log("i am ai!!", prompt);
  const config = providerConfigs[provider];

  const messages = [
    {
      role: "user",
      content: prompt,
    },
  ];

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Provider": provider,
      },
      body: JSON.stringify({
        messages,
        ...config,
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

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Decode the chunk and yield it
        const chunk = decoder.decode(value, { stream: true });
        yield chunk;
      }
    } else {
      // For non-streaming responses, yield the entire response at once
      const result = await response.json();
      yield result.text || result.message || JSON.stringify(result);
    }
  } catch (error) {
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
      console.log(`${provider} failed:`, error);
      // Continue to next provider unless cancelled
      if (error.message === "Request was cancelled") {
        throw error;
      }
    }
  }

  throw new Error(`All providers failed. Last error: ${lastError.message}`);
}

// Usage examples:
async function example() {
  // Single provider
  try {
    const controller = new AbortController();
    const result = await callProvider(
      "What is the meaning of life?",
      "GROQ",
      controller.signal,
    );
    console.log(result);

    // To cancel:
    // controller.abort();
  } catch (error) {
    console.error("Single provider error:", error);
  }

  // Multiple providers with fallback
  try {
    const result = await tryMultipleProviders("What is the meaning of life?");
    console.log(`Success with ${result.provider}:`, result.result);
  } catch (error) {
    console.error("All providers failed:", error);
  }
}

// Export functions if needed
export { callProvider, tryMultipleProviders };
