import { getTopics } from "../agents/key_terms_summary.js"; 
import { getDBInstance } from './indexDb.js';  
import { reinitializeButtonListeners } from '../../components/quiz/create_quiz_button_grp.js'; // for previous quizzes
import { initializeAllMessageButtons } from '../../components/settings/copy_download.js'
// import { reinitializeEditableInputs } from '../../components/markdown/markdown.js';
import { reinitializeEditableInputs } from '../../components/markdown/streamdown_markdown.js';

function openDB(dbName = "socratiqDB", storeName = "tinyMLDB_chats") {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(dbName, 1);

    request.onerror = function (event) {
      reject("Database error: " + event.target.errorCode);
    };

    request.onsuccess = function (event) {
      resolve(event.target.result);
    };

    request.onupgradeneeded = function (event) {
      const db = event.target.result;
      db.createObjectStore(storeName, { keyPath: "id" });
    };
  });
}
// Adjusting firstTimeSaveHtmlContent to return an object including the ID
async function firstTimeSaveHtmlContent(
  title,
  htmlContent,
  storeName = "tinyMLDB_chats"
) {
  try {
    const dbManager = await getDBInstance();
    if (!dbManager || !dbManager.db) {
      throw new Error('Database not initialized');
    }

    // Get the store
    const store = dbManager.db
      .transaction(['tinyMLDB_chats'], 'readwrite')
      .objectStore('tinyMLDB_chats');

    // Get the next ID
    const countRequest = store.count();
    const id = await new Promise((resolve, reject) => {
      countRequest.onsuccess = () => resolve(countRequest.result + 1);
      countRequest.onerror = (event) => reject(new Error('Error counting rows: ' + event.target.error));
    });

    // Save the chat
    return new Promise((resolve, reject) => {
      const request = store.put({
        id: id,
        title: title,
        html_content: htmlContent,
        date: new Date().toISOString()  // Add date for tracking
      });

      request.onsuccess = () => resolve({ success: true, id: id });
      request.onerror = (event) => reject(new Error('Error saving content: ' + event.target.error));
    });
  } catch (error) {
    console.error('Database operation failed:', error);
    throw error;
  }
}


function extractVisibleText(element) {
  let text = '';
  // Create a TreeWalker to walk through text nodes only
  const walker = document.createTreeWalker(
    element,
    NodeFilter.SHOW_TEXT, 
    {
      acceptNode: function(node) {
        // Filter to exclude script and style text, and consider trimming whitespace
        if (!node.parentNode.tagName.match(/SCRIPT|STYLE/i) && node.nodeValue.trim() !== '') {
          return NodeFilter.FILTER_ACCEPT;
        }
        return NodeFilter.FILTER_REJECT;
      }
    },
    false
  );

  // Iterate through each text node
  let node;
  while (node = walker.nextNode()) {
    text += node.textContent + ' '; // Collect text and add a space for separation
  }
  return text.trim(); // Trim the final string to remove any excess whitespace
}


/**
 * A function that removes all non-letter characters from the input string.
 *
 * @param {string} inputString - The string to be cleaned.
 * @return {string} The cleaned string with only letters and spaces.
 */
function cleanString(inputString) {
  // This regular expression matches any character that is not a letter (a-z, A-Z) or space
  return inputString.replace(/[^a-zA-Z\s]/g, '');
}


export async function newSaveChats(shadowEle) {
  const element = shadowEle.getElementById("message-container");
  const htmlContent = element.innerHTML;
  let textOfChat = extractVisibleText(element);
  
  
  let topics = getTopics(textOfChat);

  topics = topics.length > 3
  ? [ ...topics ].sort(() => 0.5 - Math.random()).slice(0, 3).join("-")
  : topics.join("-");

  if (topics.length === 0) {
    return topics = new Date().toString();
  }

  try {
    const res = await firstTimeSaveHtmlContent(topics, htmlContent);
    return res.id; // This returns the ID to the caller of newSaveChats
  } catch (error) {
    console.error("Error saving chat:", error);
    return null; // Indicative of an error situation
  }
}

async function getRowByKey(
  dbName = "socratiqDB",
  storeName = "tinyMLDB_chats",
  key
) {
  const db = await openDB(dbName);
  const transaction = db.transaction([storeName], "readonly");
  const objectStore = transaction.objectStore(storeName);
  const request = objectStore.get(key);

  return new Promise((resolve, reject) => {
    request.onsuccess = (event) => {
      resolve(event.target.result); // Return the found entry
    };

    request.onerror = (event) => {
      reject("Error in getting row: " + event.target.errorCode);
    };
  });
}


/**
 * Retrieves all rows from a specified database and store.
 *
 * @param {string} dbName - The name of the database to retrieve rows from.
 * @param {string} storeName - The name of the store within the database to retrieve rows from.
 * @return {Promise<Array>} A promise that resolves to an array of all rows retrieved from the specified store.
 */
export async function listAllRows(dbName, storeName) {
  const db = await openDB(dbName);
  const transaction = db.transaction([storeName], "readonly");
  const objectStore = transaction.objectStore(storeName);
  const request = objectStore.openCursor();
  const entries = [];

  return new Promise((resolve, reject) => {
    request.onsuccess = (event) => {
      const cursor = event.target.result;
      if (cursor) {
        entries.push(cursor.value);
        cursor.continue();
      } else {
        resolve(entries); // Done iterating
      }
    };

    request.onerror = (event) => {
      reject("Error in listing rows: " + event.target.errorCode);
    };
  });
}

// Helper function to validate and convert ID
function validateAndConvertId(id) {
  if (id === null || id === undefined) {
    return null;
  }
  
  if (typeof id === 'object' && id !== null) {
    id = id.id;
  }
  
  const numId = typeof id === 'string' ? parseInt(id, 10) : id;
  
  if (isNaN(numId) || numId <= 0) {
    return null;
  }
  
  return numId;
}

// Update loadChat to use the database instance
export async function loadChat(shadowEle, id) {
  try {
    const validId = validateAndConvertId(id);
    console.log('Attempting to load chat with ID:', validId);
    
    if (validId === null) {
      console.warn("Invalid chat ID:", id);
      return false;
    }

    const element = shadowEle.querySelector("#message-container");
    if (!element) {
      console.warn("Message container not found");
      return false;
    }

    const dbManager = await getDBInstance();
    if (!dbManager || !dbManager.db) {
      console.warn('Database not properly initialized');
      return false;
    }

    // Add debugging
    const store = dbManager.db.transaction(['tinyMLDB_chats'], 'readonly')
      .objectStore('tinyMLDB_chats');
    
    // Log all available chats
    const allChats = await new Promise((resolve) => {
      const request = store.getAll();
      request.onsuccess = () => resolve(request.result);
    });
    console.log('Available chats:', allChats);

    const data = await dbManager.getByKey('tinyMLDB_chats', validId);
    if (!data || !data.html_content) {
      console.warn(`No data found for chat ID: ${validId}`);
      return false;
    }

    element.innerHTML = data.html_content;
    
    // Reinitialize quiz buttons after loading chat
    const { reinitializeQuizButtons } = await import('../../components/quiz/load_quiz.js');
    // reinitializeQuizButtons(shadowEle);
    
    saveRecentIDToLocal(validId);

    initializeAllMessageButtons(shadowEle);
    reinitializeEditableInputs(shadowEle);
    reinitializeButtonListeners(shadowEle);
    reinitializeQuizButtons(shadowEle);

    return true;
  } catch (error) {
    console.error("Error loading chat:", error);
    return false;
  }
}

export async function deleteChat(key) {
  try {
    const dbManager = await getDBInstance();
    if (!dbManager) {
      throw new Error('Database not initialized');
    }
    await dbManager.delete('tinyMLDB_chats', key);
    return true;
  } catch (error) {
    console.error(`Error deleting chat with key ${key}:`, error);
    return false;
  }
}

export async function clearAllRows() {
  try {
    const dbManager = await getDBInstance();
    if (!dbManager) {
      throw new Error('Database not initialized');
    }
    
    // Get all entries and delete them one by one
    const entries = await dbManager.getAll('tinyMLDB_chats');
    for (const entry of entries) {
      await dbManager.delete('tinyMLDB_chats', entry.id);
    }
    return true;
  } catch (error) {
    console.error('Error clearing rows:', error);
    throw error;
  }
}


/**
 * Load the last saved row from the specified IndexedDB object store into a specified element.
 *
 * @param {Object} shadowEle - The shadow DOM element to append the content to.
 * @param {string} [storeName='tinyMLDB_chats'] - The name of the IndexedDB object store to retrieve the data from.
 * @param {string} [elementId='message-container'] - The ID of the element to load the content into.
 * @return {Promise} A promise that resolves with an object containing the ID and topic of the loaded content, or rejects with an error message.
 */
export async function loadLastSavedRowIntoElement(shadowEle) {
  try {
    // First check local storage for recent ID
    const recentId = retrieveRecentIDFromLocal();
    if (recentId !== null) {
      await loadChat(shadowEle, recentId);
      return { id: recentId };
    }

    // If no recent ID, load the last saved chat
    const db = await openDB();
    const transaction = db.transaction([""], "readonly");
    const objectStore = transaction.objectStore("tinyMLDB_chats");
    const request = objectStore.openCursor(null, "prev");

    return new Promise((resolve, reject) => {
      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          const { id, html_content } = cursor.value;
          const element = shadowEle.getElementById("message-container");
          if (element) {
            element.innerHTML = html_content;
            saveRecentIDToLocal(id); // Save this as the current chat
            resolve({ id });
          }
        } else {
          resolve({ id: null }); // No chats found
        }
      };
      request.onerror = () => reject("Error fetching last record");
    });
  } catch (error) {
    console.error("Error loading last saved row:", error);
    return { id: null };
  }
}


/**
 * Determine and save chat based on the provided id, handling new chats accordingly.
 *
 * @param {Element} shadowEle - The shadow element containing the chat messages.
 * @param {string} id - The id of the chat.
 * @return {string} The updated or new chat id.
 */
export async function determineAndSaveChat(shadowEle, id, topic = null) {
  try {
    const element = shadowEle.querySelector("#message-container");
    if (!element) {
      throw new Error("Element '#message-container' not found.");
    }

    const htmlContent = element.innerHTML;
    
    // Check database connection first
    const dbManager = await getDBInstance();
    if (!dbManager || !dbManager.db) {
      throw new Error('Database not properly initialized');
    }
    
    if (typeof id === "undefined" || id === null) {
      let textOfChat = extractVisibleText(element);
      const topics = topic || getTopics(textOfChat);
      const result = await firstTimeSaveHtmlContent(topics, htmlContent);
      return result.id;
    }
    
    const actualId = id?.id || id;
    const result = await saveChatWithId(actualId, htmlContent);
    
    if (!result.success) {
      console.warn(`Failed to update chat with ID: ${actualId}, creating new chat`);
      const newResult = await firstTimeSaveHtmlContent(topic || "Recovered Chat", htmlContent);
      return newResult.id;
    }
    
    return result.id;
  } catch (error) {
    console.error("Error in determineAndSaveChat:", error);
    throw error; // Rethrow to allow proper error handling upstream
  }
}


// Update saveChatWithId to use the new database instance
async function saveChatWithId(id, htmlContent) {
  try {
    const dbManager = await getDBInstance();
    if (!dbManager || !dbManager.db) {
      throw new Error('Database not initialized');
    }

    const store = dbManager.db
      .transaction(['tinyMLDB_chats'], 'readwrite')
      .objectStore('tinyMLDB_chats');

    // First check if the entry exists
    const getRequest = store.get(id);
    const existingData = await new Promise((resolve, reject) => {
      getRequest.onsuccess = () => resolve(getRequest.result);
      getRequest.onerror = () => reject(new Error('Error retrieving entry'));
    });

    if (!existingData) {
      console.warn(`No existing entry found for ID: ${id}`);
      return { success: false, id: id };
    }

    // Update the existing entry
    const updatedData = {
      ...existingData,
      html_content: htmlContent,
      last_updated: new Date().toISOString()
    };

    return new Promise((resolve, reject) => {
      const request = store.put(updatedData);
      request.onsuccess = () => resolve({ success: true, id: id });
      request.onerror = (event) => reject(new Error('Error updating content: ' + event.target.error));
    });
  } catch (error) {
    console.error('Database operation failed:', error);
    throw error;
  }
}



/**
 * Saves the most recent ID to local storage.
 * @param {number} id - The ID to save.
 */
export function saveRecentIDToLocal(id) {
  if (id !== null && id !== undefined) {
    localStorage.setItem("mostRecentID", id.toString());
  }
}

/**
 * Retrieves the most recent ID from local storage.
 * @return {number | null} The most recent ID, or null if not found.
 */
export function retrieveRecentIDFromLocal() {
  const id = localStorage.getItem("mostRecentID");
  return id ? parseInt(id, 10) : null;
}


// Function to clear message container and return null for an ID
export function clearMessageContainer(shadowEle) {
  const messageContainer = shadowEle.querySelector("#message-container");
  if (messageContainer) {
    messageContainer.innerHTML = '';
  } else {
    console.error('Message container not found');
  }
  return undefined;
}


// SEARCH FEATURES /////////////////////
////////////////////////////////////////


export async function listAllRows_search(dbName, storeName) {
  const db = await openDB(dbName);
  const transaction = db.transaction([storeName], "readonly");
  const objectStore = transaction.objectStore(storeName);
  const request = objectStore.openCursor();
  const entries = [];

  return new Promise((resolve, reject) => {
    request.onsuccess = (event) => {
      const cursor = event.target.result;
      if (cursor) {
        entries.push(cursor.value);
        cursor.continue();
      } else {
        resolve(entries); // Done iterating
      }
    };

    request.onerror = (event) => {
      reject("Error in listing rows: " + event.target.errorCode);
    };
  });
}

export async function searchChatsByText(searchText) {
  try {
    const dbManager = await getDBInstance();
    if (!dbManager) {
      throw new Error('Database not initialized');
    }

    const allChats = await dbManager.getAll('tinyMLDB_chats');
    return allChats.filter(chat => 
      chat.html_content.toLowerCase().includes(searchText.toLowerCase())
    );
  } catch (error) {
    console.error("Error searching chats:", error);
    return [];
  }
}

// Usage:
// searchChatsByText('tinyMLDB', 'tinyMLDB_chats', 'search this text')
//   .then(matches => console.log('Found entries:', matches))
//   .catch(error => console.error(error));
