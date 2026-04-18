import { BASEURL } from "../../../configs/env_configs";

// import { TextSelection } from 'prosemirror-state';
// import {insertTextAndMoveCursor, appendTextWithNewTextNode, appendTextToNode, getCurrentNode} from '../text_editor/insert_into_editor.js';
// const url = "http://localhost:3000/query-memory-agent"; // Replace with your actual endpoint URL
// const url_no_memory = "http://localhost:3000/query-agent";
// const url_no_memory_stream = "http://localhost:3000/query-agent_stream";
// const url_no_memory_stream_tiny =
//   "http://localhost:3000/query_agent_stream_tiny";
// const url_query_indices = "http://localhost:3000/api/query_index";
// const url_query_indices_pinecone = "http://localhost:3000/api/query-pinecone";
const url_query_indices_pinecone = BASEURL + "/api/query-pinecone";
// const url_query_indices_azure = 'http://localhost:3000' + "/api/query-lance_azure";
const url_query_indices_azure = BASEURL + "api/query-lance_azure";

const url_qeury_indices_azure_index = BASEURL + "api/query_index-lance_azure";






/**
 * Asynchronously retrieves indexed data using the provided text and token.
 *
 * @param {string} text - The text to use for querying indices
 * @param {string} token - The authorization token for the fetch request
 * @return {Promise} The retrieved data as a promise
 */
// export async function retrieve_indexed_data(text, token) {
export async function retrieve_indexed_data(text, token) {
  try {
    // TRACE: Track which AI endpoint is being called
    console.trace(`[AI_TRACE] retrieve_indexed_data.js - Calling Pinecone endpoint: ${url_query_indices_pinecone}`, {
      textPreview: text?.substring(0, 100) + '...' || 'No text'
    });

    // Perform the fetch request
    const response = await fetch(url_query_indices_pinecone, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // Pass the token in the Authorization header
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({text: text}),
    });

    // Check if the request was successful
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Parse the JSON response
    const data = await response.json();

    // Use the response data as needed
    return data;
  } catch (error) {
    console.error("Failed to fetch:", error);
  }
}



export async function retrieve_indexed_data_azure(text, token) {
try {
  // TRACE: Track which AI endpoint is being called
  console.trace(`[AI_TRACE] retrieve_indexed_data.js - Calling Azure endpoint: ${url_query_indices_azure}`, {
    textPreview: text?.substring(0, 100) + '...' || 'No text'
  });

  // Perform the fetch request
  const response = await fetch(url_query_indices_azure, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // Pass the token in the Authorization header
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({text: text}),
  });

  // Check if the request was successful
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  // Parse the JSON response
  const data = await response.json();

  // Use the response data as needed
  return data;
} catch (error) {
  console.error("Failed to fetch:", error);
}
}





export async function retrieve_indexed_data_azure_index(text, token) {
try {
  // Perform the fetch request
  const response = await fetch(url_qeury_indices_azure_index, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // Pass the token in the Authorization header
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({text: text}),
  });

  // Check if the request was successful
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  // Parse the JSON response
  const data = await response.json();

  // Use the response data as needed
  return data;
} catch (error) {
  console.error("Failed to fetch:", error);
}
}



