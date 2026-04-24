import {sendToWorker} from "../workers/initiate_worker.js";
let currChatId;

export function initiateMemory(dbName){
    currChatId = parseInt(dbName.slice(-1));
    sendToWorker('initiate', {"dbName": dbName}, '', false)
}
export function saveChatHistoryVector(chatID, type, text, token) {
    

    // console.log("chatID", chatID)
    if (currChatId !== chatID) {
        const newDbName = "vectorDB_" + chatID
        initiateMemory(newDbName)
    }
    // const currentDateTime = new Date();
    // chatId = await determineAndSaveChat(shadowRoot, chatId, topicOfConversation);
    sendToWorker('create', {"chatId": chatID, "role": type, "text": `${text}`}, token, false)
  }

  export async function searchChatHistoryVector(chatID, text, k=5, token) {
    let newDBname = "vectorDB_" + chatID;
    let chats;
  
    if (currChatId !== chatID) {
      newDBname = "vectorDB_" + chatID;
      initiateMemory(newDBname);
    }
    
    // Proceed with search if the database is not empty
    chats = await sendToWorker('search', {"chatId": chatID, "text": text, "k": k}, token, false);

    chats = organizeConversations(chats);

    
    return chats;
  }

  // Function to convert the data object into a conversations array
function organizeConversations(data) {
    const conversations = [];
    let currentRole = null;
    let currentText = "";
  
    data.forEach((item) => {
      const { role, text } = item.object;
  
      if (role !== currentRole) {
        if (currentRole !== null) {
          conversations.push({ role: currentRole, content: currentText });
        }
        currentRole = role;
        currentText = text;
      } else {
        currentText += " " + text;
      }
    });
  
    // Push the last accumulated message
    if (currentRole !== null) {
      conversations.push({ role: currentRole, content: currentText });
    }
  
    return conversations;
  }