import {BASEURL} from '../../../../../configs/env_configs.js'
const tempURL = `http://localhost:3000/`
const embeddings_url =`${BASEURL}api/embeddings-binary`
const find_url = `${BASEURL}api/search-embeddings-binary`


export function fetchEmbeddings(text, token) {

    // TRACE: Track which AI endpoint is being called
    console.trace(`[AI_TRACE] api_call_embeddings.js - Calling embeddings endpoint: ${embeddings_url}`, {
      textPreview: text?.substring(0, 100) + '...' || 'No text'
    });

    return fetch(embeddings_url, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text })
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // console.log("response", response)

      return response.json();

    })
    .catch(error => console.error('Error fetching embeddings:', error));
  }
  

  export function fetchTopKSimilarIndices(targetData, queryEmbedding, token, k = 3) {
    
    // TRACE: Track which AI endpoint is being called
    console.trace(`[AI_TRACE] api_call_embeddings.js - Calling find similar indices endpoint: ${find_url}`, {
      k,
      targetDataPreview: targetData?.substring(0, 100) + '...' || 'No target data'
    });

    return fetch(find_url, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ targetData, queryEmbedding, k })
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .catch(error => console.error('Error fetching top K similar indices:', error));
  }
  