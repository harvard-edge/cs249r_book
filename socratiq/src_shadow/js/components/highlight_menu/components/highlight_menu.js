import { isMenuOpen } from "../../../libs/utils/check_menu_open";
import { menu_slide } from "../../menu/open_close_menu";
import { query_agent } from "../../../libs/agents/chat_agent_yield";
import { createCitation } from "../../../libs/agents/citation_agent";
import { close_menu } from "./menu_ui";
let menu;
let main_menu;
let isOpen = false;  //if mini menu is open
let preventClose = false; // New flag to prevent immediate closure
let text;
let prev_text = "";
let shadowEle;
let sendButton;
let explainButton;
let quizButton;
let researchButton;
let apa;
let chicago;
let harvard;
let token;
let citationPrompt = `given the partial citation info, create a citation in the style of: `;
let textArea;
let touchTimer;

// need a way to turn off smaller menu when outer menu is open
export function small_highlight_menu(shadowRoot) {


  shadowEle = shadowRoot;
  menu = shadowRoot.getElementById("text-selection-menu-highlight");
  main_menu = shadowRoot.getElementById("text-selection-menu");

  document.addEventListener("touchstart", async function (event) {
    touchTimer = setTimeout(async () => {
      await bringUpMenu(event.touches[0]);
    }, 800); // Long press threshold (800ms)
  });

  document.addEventListener("touchend", function (event) {
    clearTimeout(touchTimer);

    if (!menu.contains(event.target) && !menu.classList.contains === "hidden") {
      // menu.classList.add("hidden");
      close_menu(shadowEle);
    }
  });

  document.addEventListener("mouseup", async (e) => {
    await bringUpMenu(e);
  });

  async function bringUpMenu(e) {
    // const isMenuOpenNow = isMenuOpen(shadowEle);

    // adjustPosition(autoresizingTextarea)
    const sel = await window.getSelection();
    const eventPath = e.composedPath();
    const clickedInsideMenu = eventPath.includes(menu);
    const clickedInsideMenu2 = eventPath.includes(main_menu);

    if (!clickedInsideMenu && !clickedInsideMenu2) {
      // addRedVerticalLine(sel)
      text = sel.toString().trim();

      // console.log("i am text just highlighted!", text)
    }
    // else{
    //   text =''
    // }

    if (text.length > 0 && !isOpen && !clickedInsideMenu2) {
      //} && !isMenuOpenNow) {
      // Check if the menu is not already open
      preventClose = true; // Prevent closing immediately after opening
      setTimeout(() => {
        preventClose = false;
      }, 20); // Reset after a short delay

      elementFollowMouse(menu, e);
      menu.classList.remove("hidden");
      isOpen = true;
    } else if (
      text.length > 0 &&
      isOpen &&
      prev_text !== text &&
      !clickedInsideMenu
    ) {
    }
    prev_text = text; // Save previous selection
  }

  document.addEventListener("click", (event) => {
    const path = event.composedPath();
    const currentSelection = window.getSelection().toString().trim();

    if (!preventClose && isOpen) {
      const clickedInsideMenu = path.includes(menu); // Check if the menu is in the event path
      if (!clickedInsideMenu && currentSelection.length === 0) {
        //} && currentSelection === text) {
        // menu.classList.add("hidden");
        close_menu(shadowEle);

        isOpen = false;
        // removeRedVerticalLine();
      } else if (clickedInsideMenu) {
        // Here, you handle the scenario when clicks occur inside the menu. No action needed if the menu stays open.
      } else if (currentSelection !== text && currentSelection.length > 0 && !clickedInsideMenu) {
        // Handle new text selection
        // console.log("i am text just highlighted!", currentSelection)
        text = currentSelection;
        // addRedVerticalLine(window.getSelection());
        elementFollowMouse(menu, event);
        // explain_agent(text)
      }
    }
  });
}

let difference = 0;
export function adjustPosition(status, extra_buffer = 0) {
  const buffer = 10 + extra_buffer;
  const menuElement = menu; // Assuming 'menu' is already defined
  // let difference = 0
  setTimeout(() => {
    const rect = menuElement.getBoundingClientRect();
    // If the bottom of the menu goes beyond the bottom of the viewport
    if (rect.bottom > window.innerHeight) {
      difference = Math.max(rect.bottom - window.innerHeight, difference);
      difference = difference + buffer;
      menuElement.style.transform = `translateY(-${difference}px)`;
    }
  }, 0);
  if (status === "final") {
    difference = 0;
  } else if (status === "in_process") {
  } else {
    console.error("status needs to be in_process or final. Is is ", status);
  }
  return difference;
}

function elementFollowMouse(menu, e) {
  // const menu = document.getElementById("text-selection-menu");
  const buffer = 25;

  menu.style.position = "fixed";
  // menu.style.top = `${e.clientY}px`;
  menu.style.top = `${e.clientY - 8 * buffer}px`;
  menu.style.left = `${e.clientX}px`;
  //   menu.style.display = "block";
  menu.classList.remove("hidden");

  setTimeout(() => {
    const menuRect = menu.getBoundingClientRect();

    if (menuRect.right > window.innerWidth) {
      // console.log(
      //   "menuRect.right > window.innerWidth",
      //   menuRect.right,
      //   window.innerWidth
      // );
      menu.style.left = `${window.innerWidth - menuRect.width - buffer}px`;
    }
    if (menuRect.bottom > window.innerHeight) {
      // console.log(
      //   "menuRect.bottom > window.innerHeight",
      //   menuRect.bottom,
      //   window.innerHeight
      // );
      menu.style.top = `${window.innerHeight - menuRect.height}px`;
    }
  }, 0);
}

export function ai_query(
  text_combo,
  type_of_custom_event = "aiActionCompleted",
  type_of_action = "query"
) {
  const event = new CustomEvent(type_of_custom_event, {
    detail: {
      text: text_combo, // the original query object used for the request
      type: type_of_action, // the type of AI action
      fromRightClickMenu: true,
    },
  });
  window.dispatchEvent(event);
}

export function explain_agent(text) {
  const event = new CustomEvent("aiActionCompleted", {
    detail: {
      text, // the original query object used for the request
      type: "explain", // the type of AI action
      links: [window.location.href],
    },
  });
  window.dispatchEvent(event);
}

function resarch_agent(text, links = "") {
  const event = new CustomEvent("aiActionCompleted", {
    detail: {
      text, // the original query object used for the request
      type: "research", // the type of AI action
      links: [window.location.href],
      getNew: true,
    },
  });
  window.dispatchEvent(event);
}

export function quiz_agent(text) {
  const event = new CustomEvent("aiActionCompleted", {
    detail: {
      text, // the original query object used for the request
      type: "quiz", // the type of AI action
      links: [window.location.href],
    },
  });
  window.dispatchEvent(event);
}

function general_agent(text, links = "") {
  const event = new CustomEvent("aiActionCompleted", {
    detail: {
      text, // the original query object used for the request
      type: "general", // the type of AI action
      links: [window.location.href],
    },
  });
  window.dispatchEvent(event);
}

export async function suggestion_agent(prompt, links = "") {
  prompt = `TEXT: ${text} \n ${prompt}`;
  const params = { prompt: prompt };
  let resp = "";
  // const response = query_agent

  try {
    // Iterate over each chunk yielded by the generator
    for await (let chunk of query_agent(params, token)) {
      resp += chunk;
    }
  } catch (e) {
    console.error("unable to get suggestions from AI", e);
  }

  return resp;
}

export function setButtons(shadowEle) {
  sendButton = shadowEle.querySelector(`#sendMiniBtn`);
  explainButton = shadowEle.querySelector(`#explainMiniBtn`);
  quizButton = shadowEle.querySelector(`#quizMiniBtn`);
  researchButton = shadowEle.querySelector(`#researchMiniBtn`);
  chicago = shadowEle.querySelector(`#chicago`);
  apa = shadowEle.querySelector(`#apa`);
  harvard = shadowEle.querySelector(`#harvard`);
  textArea = shadowEle.querySelector("#responseTextarea");
  const citation = createCitation();

  citationPrompt = `citation: ${citation} ${citationPrompt}`;

  textArea.addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
      sendQuery(event);
    }
  });

  sendButton.addEventListener("click", (event) => {
    sendQuery(event);
  });

  function sendQuery(event) {
    event.preventDefault(); // Prevent the default Enter key behavior

    if (!isMenuOpen(shadowEle)) {
      openClose();
    }

    const textFromTextArea = textArea.value;

    textArea.value = "";

    const prompt =
      "~+~ " +
      text +
      " ~+~" +
      " and this query: " +
      "~+~" +
      textFromTextArea +
      " ~+~" 
    ai_query(prompt);
  }

  explainButton.addEventListener("click", (event) => {
    // console.log("explain agent", text)
    explain_agent(text);
    // openClose();
    if (!isMenuOpen(shadowEle)) {
      openClose();
    }
  });

  quizButton.addEventListener("click", (event) => {
    quiz_agent(text);
    // openClose();
    if (!isMenuOpen(shadowEle)) {
      openClose();
    }
  });

  researchButton.addEventListener("click", (event) => {
    console.log("research agent", text)
    console.log("prev text", prev_text)


    resarch_agent(text);
    // openClose();
    if (!isMenuOpen(shadowEle)) {
      openClose();
    }
  });
  chicago.addEventListener("click", (event) => {
    // console.log("chicago agent", citationPrompt)
    general_agent(citationPrompt + "chicago");
    // openClose();
    if (!isMenuOpen(shadowEle)) {
      openClose();
    }
  });
  apa.addEventListener("click", (event) => {
    general_agent(citationPrompt + "APA");

    // openClose();
    if (!isMenuOpen(shadowEle)) {
      openClose();
    }
  });
  harvard.addEventListener("click", (event) => {
    general_agent(citationPrompt + "Harvard");

    // openClose();
    if (!isMenuOpen(shadowEle)) {
      openClose();
    }
  });

  function openClose() {
    menu_slide(shadowEle);
    close_menu(shadowEle);
  }

  function getToken() {
    token = localStorage.getItem("token_avaya");
  }
  getToken();
}

