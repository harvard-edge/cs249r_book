// import { toggleMarkdownDeActivate} from '../markdown/markdown_show.js'
import { containsWordReference } from "../../libs/diagram/mermaid.js";
import { generateUniqueId } from "../../libs/utils/utils.js";
const quizHtml = `
  <div id="quiz">
    <form id="quiz-form">
    </form>
    <button id="submit-quiz-btn" class="big-quiz-btn w-full mt-4 mb-4 px-4 py-2 bg-black hover:bg-gray-800 text-white rounded flex items-center justify-center gap-2 transition-colors">
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5">
        <path stroke-linecap="round" stroke-linejoin="round" d="m4.5 19.5 15-15m0 0H8.25m11.25 0v11.25" />
      </svg>
      Submit
    </button>
    <div id="result"></div>
  </div>
`;
import {
  getAllCopyContainerText,
  isCopyContainerHidden,
  clearCopyContainer,
} from "../highlight_menu/send_text_highlight.js";

import { SIZE_LIMIT_LLM_CALL } from "../../../configs/env_configs.js"; // Adjust this value as needed
import { reduceTextSize } from "../../libs/utils/textUtils.js";
let menu;  // DO NOT REMOVE


let shadowEle;



export function highlight(shadowRoot) {
  shadowEle = shadowRoot;

  menu = shadowRoot.getElementById("text-selection-menu");
  const autoresizingTextarea = shadowRoot.getElementById("user-input");
  const enterBtn = shadowRoot.getElementById("enter-btn");

  autoresizingTextarea.addEventListener("keydown", (e) => {
    const text_combo = autoresizingTextarea.value;
    const dropupContainer = shadowRoot.querySelector(".at-mentions-dropup");
   

    // Only proceed with Enter handling if dropup is hidden
    if (
      e.key === "Enter" &&
      !e.shiftKey &&
      !e.ctrlKey &&
      (dropupContainer.classList.contains("hidden") ||
        dropupContainer.style.display === "none")
    ) {

      let diagramId = '';
      if (containsWordReference(text_combo)) {
        diagramId = generateUniqueId();
        console.log("CONTAINS DIAGRAM!!");
        diagramId = generateUniqueId();
        generateDiagramId("aiActionCompleted", text_combo, "mermaid_diagram", diagramId)
  
      }
      
      e.preventDefault();
      ai_query(text_combo, "aiActionCompleted", "query", diagramId);
      autoresizingTextarea.value = "";
    }
  });

  // explainBtn.addEventListener("click", (e) => {
  //   explain_agent(text);
  // });

  enterBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const text_combo =
      // "Quote:" + '' + "\n" + "Question:" + autoresizingTextarea.value;
      autoresizingTextarea.value;
    let diagramId = '';
    if (containsWordReference(text_combo)) {
      console.log("DIAGRAM FOUND");
      diagramId = generateUniqueId();
      generateDiagramId("aiActionCompleted", text_combo, "mermaid_diagram", diagramId)
    }

    ai_query(text_combo, "aiActionCompleted", "query", diagramId);
  });

  function ai_query(text_combo, type_of_custom_event, type_of_action, diagramId = '') {
    autoresizingTextarea.value = "";
    let copyContext = text_combo;

    if (!isCopyContainerHidden(shadowEle)) {
      copyContext =
        "given the following content /n" +
        reduceTextSize(
          getAllCopyContainerText(shadowEle),
          SIZE_LIMIT_LLM_CALL,
        ) +
        "\n" +
        "Question:" +
        text_combo;
      clearCopyContainer(shadowEle);
    }

    const event = new CustomEvent(type_of_custom_event, {
      detail: {
        text: copyContext, // the original query object used for the request
        type: type_of_action, // the type of AI action
        diagramId: diagramId,
      },
    });
    window.dispatchEvent(event);
  }
}

function generateDiagramId(type_of_custom_event, text, type_of_action, id) {

  const event = new CustomEvent(type_of_custom_event, {
    detail: {
      text: text, // the original query object used for the request
      type: type_of_action, // the type of AI action
      diagramId: id,
    },
  });
  window.dispatchEvent(event);
}

export function showQuiz(clone) {
  const markdownPrev = clone.querySelector("#markdown-preview");

  markdownPrev.innerHTML = quizHtml;

  return markdownPrev;
}
