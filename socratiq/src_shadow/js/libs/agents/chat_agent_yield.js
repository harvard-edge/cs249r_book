import {BASEURL} from '../../../configs/env_configs'
import { tryMultipleProvidersStream } from './cloudflareAgent.js';
// import { showPopover } from '../utils/utils.js';

const url_no_memory_stream_tiny = BASEURL+"query_agent_stream_tiny";
const url_json_tiny = BASEURL + "open_ai_agent_json_non_stream_tiny";
const url_json_powerful = BASEURL + "api/query_agent_stream_powerful";
const url_conversational_memory = BASEURL + "api/query_agent_conversational_memory";


export async function* query_agent(query, token, isMorePowerful=false, conversational = false) {

  // DEACTIVATED: Primary agent is disabled due to "No data received from primary agent" errors
  // Skipping directly to backup system
  console.log("[AI_TRACE] chat_agent_yield.js - PRIMARY AGENT DEACTIVATED - Using backup system directly", {
    isMorePowerful,
    conversational,
    queryPreview: query?.prompt?.substring(0, 100) + '...' || 'No prompt'
  });

  // Skip primary agent entirely and go straight to backup system
  try {
    const promptText = query.prompt || '';

    const formattedPrompt = {
      prompt: promptText,
      messages: [{
        role: "user",
        content: promptText
      }]
    };
    console.log("Calling backup with formatted prompt:", formattedPrompt);

    const generator = tryMultipleProvidersStream(formattedPrompt, null, true);
    
    let hasYieldedContent = false;
    for await (const chunk of generator) {
      hasYieldedContent = true;
      yield chunk;
    }

    if (!hasYieldedContent) {
      throw new Error("No content received from any providers");
    }

  } catch (supaError) {
    console.error("Complete backup system failure:", supaError);
    throw supaError;
  }
}

export async function query_agent_json(query, token) {
  try {
    // Perform the fetch request
    const response = await fetch(url_json_tiny, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(query),
    });

    // Check if the request was successful
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    return data;
  } catch (error) {
    console.error("Failed to fetch:", error);
  }
}


