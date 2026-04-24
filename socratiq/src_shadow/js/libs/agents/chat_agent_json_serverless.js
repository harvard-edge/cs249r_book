import {SERVELESSURLJSON, SERVELESSURLGROQ, SERVERLESSGEMINI} from '../../../configs/env_configs'
import { callCloudflareWithLegacyFormat } from './cloudflareAgent.js'


export async function query_agent_json_serverless(query, tok) {
  try {
    const response = await fetch(SERVELESSURLJSON, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(query),
        credentials: 'include', // Keep this if you need to send cookies
        mode: 'cors' // Explicitly set CORS mode
      });


    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Failed to fetch:", error);
    throw error; // Re-throw the error for the caller to handle
  }
}

export async function query_agent_groq_serverless(params) {
  try {
    // Use the translation function from cloudflareAgent.js
    const response = await callCloudflareWithLegacyFormat(params, 'groq', false);
    return response;
  } catch (error) {
    console.error("Failed to fetch from GROQ:", error);
    throw error;
  }
}

export async function query_agent_gemini_serverless(params) {
  try {
    // Use the translation function from cloudflareAgent.js
    const response = await callCloudflareWithLegacyFormat(params, 'gemini', false);
    return response;
  } catch (error) {
    console.error("Failed to fetch from Gemini:", error);
    throw error;
  }
}
