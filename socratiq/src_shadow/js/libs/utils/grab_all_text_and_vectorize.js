// import {create_embeddings} from "../../frontend_engine/loaders/ini_embeddings.js";

export function processWebsiteText() {
    const textContent = grabWebsiteTextContent();
    // const currentURL = window.location.href; 
    const cleanedText = cleanUpText(textContent); // Assume this is a function you've written to clean the text
    // create_embeddings(cleanedText, currentURL);
    return cleanedText
  }
  

  export function getTextFromParagraphs() {
    // Select all <p> elements on the page
    const paragraphs = document.querySelectorAll('p');
    // Map over each paragraph and return its text content
    const allText = Array.from(paragraphs).map(p => p.textContent.trim());
    // Return the array of text content
    return allText;
}

  // processWebsiteText()
  // Example cleanup function (very basic)
  function cleanUpText(text) {
    return text.replace(/\s+/g, ' ').trim(); // Replace multiple whitespace with a single space and trim
  }
  
  // processWebsiteText();
  

function grabWebsiteTextContent() {
    return document.body.textContent;
  }
  