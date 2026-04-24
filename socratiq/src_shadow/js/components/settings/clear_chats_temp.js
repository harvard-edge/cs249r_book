import { enableTooltip } from '../tooltip/tooltip.js';


export function initiate_clear_chats(shadowDom){
    const newChat = shadowDom.querySelector("#new-chat-btn")
    enableTooltip(newChat, "Start a new chat or load a previous chat", shadowDom);
    newChat.addEventListener("click", () => {
      const confirmation = confirm("Are you sure you want to start a new chat?");
    if (confirmation) {
        triggerResetChat()
      }
    })
}   

function triggerResetChat() {
    const event = new CustomEvent("resetAIChat", {
      detail: {
        containerId: "reset", // Pass the ID of the message container
      },
    });
    window.dispatchEvent(event);
  }