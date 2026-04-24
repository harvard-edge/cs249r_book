import { getDBInstance, resetSocratiqDB } from '../../libs/utils/indexDb.js';
import { loadChat,deleteChat, clearAllRows} from "../../libs/utils/save_chats";
import {alert, scrollToBottomSmooth} from "../../libs/utils/utils.js";
import { enableTooltip } from '../tooltip/tooltip.js';

const prev_chats = `
<div id="popover-container" class="hidden">
  <div class="modal-content bg-transparent p-4 max-w-md w-full rounded-lg">
    <div class="flex justify-between items-center mb-4">
      <h2 class="text-lg font-semibold text-gray-800">Previous Conversations</h2>
      <button id="close-btn2" class="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors duration-200">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6">
          <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>

    <input 
      id="chat-search" 
      class="search-input w-full px-3 py-2 rounded-lg border border-gray-200 focus:outline-none focus:border-blue-400 mb-4 text-gray-600 placeholder-gray-400"
      placeholder="Search conversations..."
    />

    <ul id="new-chat" class="mb-3 list-none">
      <li class="transition-all duration-200">
        <button
          class="flex items-center justify-between w-full px-4 py-3 text-left rounded-lg bg-blue-50 hover:bg-blue-100 transition-all duration-200 group"
        >
          <span class="text-blue-600 group-hover:text-blue-700 font-medium">New Chat</span>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke-width="1.5"
            stroke="currentColor"
            class="w-5 h-5 text-blue-600 group-hover:text-blue-700"
          >
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 6v12m6-6H6" />
          </svg>
        </button>
      </li>
    </ul>

    <ul id="list-container" class="space-y-1 overflow-auto max-h-[350px] pr-1">
      <!-- Chat items will be inserted here -->
    </ul>

    <div class="mt-4 flex justify-between">
      <button 
        id="reset-btn" 
        class="flex items-center px-4 py-2 text-yellow-600 bg-yellow-50 rounded-lg hover:bg-yellow-100 transition-colors duration-200"
      >
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          fill="none" 
          viewBox="0 0 24 24" 
          stroke-width="1.5" 
          stroke="currentColor" 
          class="w-5 h-5 mr-2"
        >
          <path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
        </svg>
        Reset SocratiQ
      </button>
      <button 
        id="clear-btn" 
        class="flex items-center px-4 py-2 text-red-500 bg-red-50 rounded-lg hover:bg-red-100 transition-colors duration-200"
      >
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          fill="none" 
          viewBox="0 0 24 24" 
          stroke-width="1.5" 
          stroke="currentColor" 
          class="w-5 h-5 mr-2"
        >
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
        </svg>
        Delete all Chats
      </button>
    </div>
  </div>
</div>
`;
 
function addItem(shadowEle, entry) {
  const saveList = shadowEle.querySelector("#list-container");
  if (saveList) {
    const item = document.createElement("li");
    item.id = "chat-item";
    item.className = "mb-3 transition-all duration-200 ease-in-out";
    item.innerHTML = `
    <div
      id="chat-${entry.id}"
      class="flex items-center justify-between w-full px-4 py-3 text-left rounded-lg cursor-pointer transition-all duration-200 hover:bg-blue-50 group"
      style="background-color: #fafafa;"
    >
      <span class="text-gray-700 group-hover:text-blue-600 font-medium truncate">${entry.title}</span>
      <div class="flex items-center space-x-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
        <div id="load-chat-${entry.id}" class="p-1.5 rounded-lg hover:bg-blue-100 transition-colors duration-200">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke-width="1.5"
            stroke="currentColor"
            class="w-5 h-5 text-blue-600"
          >
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 6v12m6-6H6" />
          </svg>
        </div>
        <div id="delete-chat-${entry.id}" class="p-1.5 rounded-lg hover:bg-red-100 transition-colors duration-200">
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke-width="1.5" 
            stroke="currentColor" 
            class="w-5 h-5 text-red-500"
          >
            <path stroke-linecap="round" stroke-linejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
          </svg>
        </div>
      </div>
    </div>`;

  const chatItem = item.querySelector(`#chat-${entry.id}`);
  const deleteButton = item.querySelector(`#delete-chat-${entry.id}`);
  // const loadButton = item.querySelector(`#load-chat-${entry.id}`);

  // Allow clicking anywhere on the item to load the chat
  chatItem.addEventListener("click", (e) => {
    // Prevent triggering when clicking the delete button
    if (!e.target.closest(`#delete-chat-${entry.id}`)) {
      rowClicked(shadowEle, entry.id);
    }
  });

  deleteButton.addEventListener("click", async (e) => {
    e.stopPropagation(); // Prevent chat loading when deleting
    await deleteChat(entry.id);
    populateListAndShowPopover(shadowEle);
  });

  saveList.appendChild(item);
  }
}

function triggerResetChat() {
  const event = new CustomEvent("resetAIChat", {
    detail: {
      containerId: "reset", // Pass the ID of the message container
    },
  });
  window.dispatchEvent(event);
}

let prevChats;
// Function to populate a list with conversations and show popover
async function populateListAndShowPopover(shadowEle) {
  try {
    // Get database instance
    const dbManager = await getDBInstance();
    if (!dbManager || !dbManager.db) {
      throw new Error('Database not properly initialized');
    }

    const listContainer = shadowEle.querySelector("#list-container");
    if (!listContainer) {
      throw new Error('List container not found');
    }

    // Get the store
    const store = dbManager.db
      .transaction(['tinyMLDB_chats'], 'readonly')
      .objectStore('tinyMLDB_chats');

    // Get all entries
    const entries = await new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(new Error('Error getting entries'));
    });

    if (prevChats) {
      if (entries.length === prevChats.length) return; // Entries have not changed
    }

    listContainer.innerHTML = "";
    entries.forEach((entry, index) => {
      addItem(shadowEle, entry, index);
    });
    
    prevChats = entries;
  } catch (error) {
    console.error('Failed to populate list:', error);
    throw error;
  }
}

// Function called when a row/item is clicked
async function rowClicked(shadowEle, id) {
  try {
    const result = await loadChat(shadowEle, id, "socratiqDB", "tinyMLDB_chats");
    if (result) {
      alert(shadowEle, "Loading chat " + id, "success");
      toggleModal(shadowEle);
      scrollToBottomSmooth(shadowEle);
    } else {
      alert(shadowEle, "Failed to load chat " + id, "error");
    }
  } catch (error) {
    console.error('Failed to load chat:', error);
    alert(shadowEle, "Error loading chat", "error");
  }
}

const toggleModal = async (shadowEle) => {
  try {
    const modal = shadowEle.querySelector("#popover-container");
    let overlay = shadowEle.querySelector(".overlay-progressive");

    if (!modal || !overlay) {
      throw new Error('Modal or overlay not found');
    }

    if (modal.style.display === "block") {
      modal.style.display = "none";
      overlay.style.display = "none";
    } else {
      await populateListAndShowPopover(shadowEle);
      modal.style.display = "block";
      overlay.style.display = "block"; // Show the overlay when the modal is opened
    }
  } catch (error) {
    console.error('Failed to toggle modal:', error);
    throw error;
  }
};

// Function to setup modal and integrate settings
export function setupModal_loadchats(shadowEle) {
  const modal = shadowEle.querySelector("#popover-container");
  const closeBtn = shadowEle.querySelector("#close-btn2");
  const search_chats = shadowEle.querySelector("#chat-search");
  const clearAllButton = shadowEle.querySelector("#clear-btn");
  
  // Create overlay inside the shadow DOM
  const overlay = document.createElement('div');
  overlay.classList.add("overlay-progressive");
  overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    width: 100vw;
    height: 100vh;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: 99999;
    display: none;
  `;

  // Style the modal container with absolute positioning relative to viewport
  modal.style.cssText = `
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 95%;
    max-width: 32rem;
    background: white;
    z-index: 100000;
    display: none;
  `;

  // Append overlay to shadow root
  shadowEle.appendChild(overlay);

  // Event Listeners
  const newChatsBtn = shadowEle.querySelector("#new-chat");
  const initiateLoadChatsBtn = shadowEle.querySelector("#new-chat-btn");
  enableTooltip(newChatsBtn, "Start a new chat or load a previous chat", shadowEle);

  newChatsBtn.addEventListener("click", () => {
    triggerResetChat();
    toggleModal(shadowEle);
  });

  initiateLoadChatsBtn?.addEventListener("click", async () => {
    toggleModal(shadowEle);
  });

  closeBtn?.addEventListener("click", () => {
    toggleModal(shadowEle);
  });

  // Close on overlay click
  overlay.addEventListener('click', () => {
    toggleModal(shadowEle);
  });

  if (search_chats) {
    search_chats.addEventListener("keyup", async (event) => {
      await searchAndShow(event.target.value, shadowEle);
    });
  }

  // Prevent modal from closing when clicking inside it
  modal.addEventListener("click", (event) => {
    event.stopPropagation();
  });

  // Close on escape key
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && modal.style.display === "block") {
      toggleModal(shadowEle);
    }
  });

  // Add clear button logic
  if (clearAllButton) {
    clearAllButton.addEventListener("click", async () => {
      try {
        await clearAllRows();
        const listContainer = shadowEle.querySelector("#list-container");
        if (listContainer) {
          listContainer.innerHTML = "";
        }
        prevChats = [];
        alert(shadowEle, "All chats have been deleted", "success");
        toggleModal(shadowEle);
      } catch (error) {
        console.error("Error clearing chats:", error);
        alert(shadowEle, "Error clearing chats", "error");
      }
    });
  }

  const resetButton = shadowEle.querySelector("#reset-btn");
  if (resetButton) {
    resetButton.addEventListener("click", async () => {
      const confirmed = confirm("Are you sure you want to reset SocratiQ? This will clear all data and refresh the application.");
      if (confirmed) {
        try {
          await resetSocratiqDB();
          alert(shadowEle, "SocratiQ has been reset. Refreshing...", "success");
          setTimeout(() => {
            window.location.reload();
          }, 1500);
        } catch (error) {
          console.error("Error resetting SocratiQ:", error);
          alert(shadowEle, "Error resetting SocratiQ", "error");
        }
      }
    });
  }

  // Return the cleanup function
  return () => {
    if (overlay && overlay.parentNode) {
      overlay.parentNode.removeChild(overlay);
    }
  };
}

let prevEntries = []
async function searchAndShow(queryText, shadowEle){
  try {
    const dbManager = await getDBInstance();
    if (!dbManager || !dbManager.db) {
      throw new Error('Database not properly initialized');
    }

    const store = dbManager.db
      .transaction(['tinyMLDB_chats'], 'readonly')
      .objectStore('tinyMLDB_chats');

    const entries = await new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => {
        const allEntries = request.result;
        const filteredEntries = allEntries.filter(entry => 
          entry.html_content.toLowerCase().includes(queryText.toLowerCase())
        );
        resolve(filteredEntries);
      };
      request.onerror = () => reject(new Error('Error getting entries'));
    });

    if (!arraysEqual(entries, prevEntries)) {
      const saveList = shadowEle.querySelector("#list-container");
      if (saveList) {
        saveList.innerHTML = '';
        entries.forEach((entry) => {
          addItem(shadowEle, entry);
        });
        prevEntries = entries;
      }
    }
  } catch (error) {
    console.error('Failed to search chats:', error);
    throw error;
  }
}

function arraysEqual(a, b) {
  if (a === b) return true; // If both are the same instance
  if (a == null || b == null) return false; // If one is null (and the other is not)
  if (a.length !== b.length) return false; // If their lengths are different

  for (let i = 0; i < a.length; ++i) {
    if (a[i] !== b[i]) return false; // If any corresponding elements are not equal
  }
  return true;
}


export function injectLoad_chats(shadowEle) {
  const template = document.createElement("template");
  template.innerHTML = prev_chats;
  const load_chats = shadowEle.querySelector("#load_chats");

  if (load_chats) {
    // Clone the node deeply
    // originalprogressContent = load_chats.cloneNode(true);
    load_chats.innerHTML = ""; // Clear existing content if necessary
    load_chats.appendChild(template.content.cloneNode(true));
  } else {
    console.error("Button menu not found");
  }
  return load_chats;
}
