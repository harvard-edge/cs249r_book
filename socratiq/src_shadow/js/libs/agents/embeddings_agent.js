import {BASEURL} from '../../../configs/env_configs'
// const TEMPURL = 'http://localhost:3000'
const embeddingURL = BASEURL + "api/embeddings-binary";
const embeddingSearchURL = BASEURL + "api/search-embeddings-binary";
const URL_GET_INDICES = BASEURL + 'api/find-similar-texts';

let chat_history;

export function initiate_chat_history(){
    chat_history = {text: [], embeddings: []}
}

export function get_chat_history()
{
    return chat_history;
}

export async function* embedding_binary(text, token) {

  try {
    // TRACE: Track which AI endpoint is being called
    console.trace(`[AI_TRACE] embeddings_agent.js - Calling embeddings endpoint: ${embeddingURL}`, {
      textPreview: text?.substring(0, 100) + '...' || 'No text'
    });

    const response = await fetch(embeddingURL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(text),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    chat_history.text.push(text);
    chat_history.embeddings.push(response)


    return response
  } catch (error) {
    console.error("Failed to query agent:", error);
  }
}

export async function topKIndices(queryText, token, k=3) {

  try {
    // TRACE: Track which AI endpoint is being called
    console.trace(`[AI_TRACE] embeddings_agent.js - Calling embedding search endpoint: ${embeddingSearchURL}`, {
      queryTextPreview: queryText?.substring(0, 100) + '...' || 'No query text',
      k
    });

   const queryEmbedding = await embedding_binary(text, token)
    
    // Perform the fetch request
    const response = await fetch(embeddingSearchURL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // Pass the token in the Authorization header
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(chat_history.embeddings, queryEmbedding, k),
    });
    const ans = new Array(k).fill('');
       // Print the most similar texts based on the indices
    
       response.forEach((index,i) => {
            ans[i] = chat_history.text[index]
    //       console.log(`Text ${index} "${texts[index]}" is one of the most similar texts.`);

      });
    return ans
    // Check if the request was successful
  } catch (error) {
    console.error("Failed to fetch search query of embeddings:", error);
  }
}
// dataArray, queryString, k, textKey = null, includeEmbeddings = false
export async function getIndicesBetQueryTexts(queryText, textsToSearch, token, k = 10, textKey = "summary", includeEmbeddings = false) {
 console.log("queryText", queryText,  "k", k, "textKey", textKey, "includeEmbeddings", includeEmbeddings)
 
  try {
    // TRACE: Track which AI endpoint is being called
    console.trace(`[AI_TRACE] embeddings_agent.js - Calling get indices endpoint: ${URL_GET_INDICES}`, {
      queryTextPreview: queryText?.substring(0, 100) + '...' || 'No query text',
      k, textKey, includeEmbeddings
    });

    const response = await fetch(URL_GET_INDICES, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Pass the token in the Authorization header
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ queryText, textsToSearch, k, textKey, includeEmbeddings }),
    });

    console.log("response", response)

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error:', error);
    throw error; // Re-throw the error after logging it
  }
}


export async function getIndicesBetQueryTextsInChunks(queryText, textsToSearch, token, k = 3, textKey = "summary", includeEmbeddings = false, CHUNK_SIZE = 20) {
  console.log("queryText", queryText, "k", k, "textKey", textKey, "includeEmbeddings", includeEmbeddings);

  const chunks = [];
  for (let i = 0; i < textsToSearch.length; i += CHUNK_SIZE) {
    chunks.push(textsToSearch.slice(i, i + CHUNK_SIZE));
  }

  const results = [];
  for (const chunk of chunks) {
    try {
      // TRACE: Track which AI endpoint is being called
      console.trace(`[AI_TRACE] embeddings_agent.js - Calling get indices endpoint (chunked): ${URL_GET_INDICES}`, {
        queryTextPreview: queryText?.substring(0, 100) + '...' || 'No query text',
        chunkSize: chunk.length, k, textKey, includeEmbeddings
      });

      const response = await fetch(URL_GET_INDICES, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ queryText, textsToSearch: chunk, k, textKey, includeEmbeddings }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      console.log("me predata", response);
      let data = await response.json()
      data = data.similarTexts
      //['similarTexts']
      console.log("me data", data);

      results.push(...data); // Assuming data is an array and can be spread into results
    } catch (error) {
      console.error('Error fetching chunk:', error);
      // Decide how you want to handle errors for individual chunks
      // e.g., skip, throw error, etc.
    }
  }

  // Sort the results by similarity score in descending order
  results.sort((a, b) => b.similarity - a.similarity);

  return results;
}
