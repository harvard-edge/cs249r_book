import {BASEURL} from '../../../configs/env_configs'

const url_no_memory_stream_tiny = BASEURL+"query_agent_stream_tiny";
const url_json_tiny = BASEURL + "open_ai_agent_json_non_stream_tiny";
const url_json_powerful = BASEURL + "api/query_agent_stream_powerful";
const url_conversational_memory = BASEURL + "api/query_agent_conversational_memory";


export async function* query_agent(query, token, isMorePowerful=false, conversational = false) {

  console.log("query_agent", query);

  let url = url_no_memory_stream_tiny;
  try {
    if (isMorePowerful){
      url = url_json_powerful
    }
    if(conversational){
      url = url_conversational_memory
    }

    // TRACE: Track which AI endpoint is being called
    console.trace(`[AI_TRACE] chat_agent_yield copy.js - Calling AI endpoint: ${url}`, {
      isMorePowerful,
      conversational,
      queryPreview: query?.prompt?.substring(0, 100) + '...' || 'No prompt'
    });

    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(query),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}, response: ${response}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      let chunk = decoder.decode(value, { stream: true });
      yield chunk; // Yield the chunk so it can be used immediately.
    }
  } catch (error) {
    console.error("Failed to query agent:", error);
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


