import htmlContent from "./index_highlight_menu.html";

import { small_highlight_menu, setButtons } from "./components/highlight_menu";

import { text_area_suggestions } from "./components/text_area_suggestions.js";

import { initiate_menu } from "./components/menu_ui.js";



export function inject_small_highlight_menu(shadowEle) {
  // const parser = new DOMParser();
  // const doc = parser.parseFromString(htmlContent, "text/html");

  // This extracts the body contents of the parsed HTML document.
  // const node = doc.body.firstChild;

  const template = document.createElement('template');
  template.innerHTML = htmlContent;


  // Append the node to the shadow DOM.
  // shadowEle.appendChild(node);
  shadowEle.appendChild(template.content.cloneNode(true));


  // Additional functions to enhance the shadow DOM's functionality.
  small_highlight_menu(shadowEle);
  initiate_menu(shadowEle);
  setButtons(shadowEle)
  text_area_suggestions(shadowEle);

}
