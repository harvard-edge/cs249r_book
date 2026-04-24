import markdownit from 'markdown-it';
import markdownItContainer from 'markdown-it-container';
import hljs from 'highlight.js'; // Import highlight.js
import 'highlight.js/styles/github.css'; // Import a highlight.js theme
import { htmlLoader } from './loader';
// import { renderMermaidDiagram } from '../../libs/diagram/mermaid';
import { findClosestAIMessage } from '../../libs/utils/utils.js';
import { containsWordReference } from '../../libs/diagram/mermaid';
import { generateUniqueId } from '../../libs/utils/utils.js';
let shadowEle;

// Initialize markdown-it and use highlight.js for code highlighting
const md = markdownit({
  html: true,      // Enable HTML tags in source
  // breaks: true,    // Convert '\n' in paragraphs into <br>
  highlight: function (str, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return hljs.highlight(str, { language: lang }).value;
      } catch (__) {}
    }
    return md.utils.escapeHtml(str); // Use external default escaping
  }
});


const wikiLinkRenderer = (tokens, idx, options, env, self) => {
  const token = tokens[idx];
  let content = token.content;

  // Find phrases surrounded by double backslashes (\\word\\)
  const doubleBackslashRegex = /\\(.*?)\\(?!\*)/g;

  // Replace double backslash pattern (\\word\\)
  content = content.replace(doubleBackslashRegex, (match, phrase) => {
    const trimmedPhrase = phrase.trim();
    const encodedPhrase = encodeURIComponent(trimmedPhrase);
    // Create a single, properly formatted link
    return `<a href="https://en.wikipedia.org/wiki/${encodedPhrase}" target="_blank" class="wiki-link">${trimmedPhrase}</a>`;
  });

  return content;
};

// Add the custom inline rule to the markdown-it renderer
md.renderer.rules.text = wikiLinkRenderer;


// Add custom classes to the heading renderers
md.renderer.rules.heading_open = (tokens, idx) => {
  const level = tokens[idx].tag;
  const classes = {
    h1: 'text-3xl font-bold mt-2',
    h2: 'text-2xl font-bold mt-2',
    h3: 'text-xl font-semibold mt-2',
    h4: 'text-lg font-semibold mt-2',
    h5: 'text-base font-medium mt-2',
    h6: 'text-sm font-medium mt-2',
  };
  return `<${level} class="${classes[level]}">`;
};
md.renderer.rules.heading_close = (tokens, idx) => {
  const level = tokens[idx].tag;
  return `</${level}>`;
};

// Add custom classes to list elements and ensure list item markers are styled correctly
md.renderer.rules.bullet_list_open = () => '<ul class="list-disc list-outside ml-4">';
md.renderer.rules.ordered_list_open = () => '<ol class="list-decimal list-outside ml-4">';
md.renderer.rules.list_item_open = () => '<li class="mb-1">';
md.renderer.rules.list_item_close = () => '</li>';

// Add custom rule for lines with four or more equals signs
md.block.ruler.before('hr', 'double_hr', (state, startLine, endLine, silent) => {
  const pos = state.bMarks[startLine] + state.tShift[startLine];
  const max = state.eMarks[startLine];

  if (pos >= max) return false;

  // Check if the line consists of 4 or more '=' characters
  const line = state.src.slice(pos, max).trim();
  if (!/^={4,}$/.test(line)) return false;

  if (silent) return true;

  state.line = startLine + 1;

  const token = state.push('html_block', '', 0);
  token.content = '<hr class="border-t-2 border-gray-300 my-4"><hr class="border-t-2 border-gray-300 my-4">';
  token.block = true;
  token.map = [startLine, state.line];
  token.markup = line;

  return true;
});

// Custom spoiler container
md.use(markdownItContainer, 'spoiler', {
  validate: function(params) {
    return params.trim().match(/^spoiler\s+(.*)$/);
  },
  render: function(tokens, idx, options, env, self) {
    const m = tokens[idx].info.trim().match(/^spoiler\s+(.*)$/);
    
    if (tokens[idx].nesting === 1) {
      // Find all content between opening and closing tags
      let content = '';
      let i = idx + 1;
      
      // Keep collecting content until we hit the closing tag
      while (i < tokens.length && tokens[i].type !== 'container_spoiler_close') {
        if (tokens[i].type === 'inline') {
          content += tokens[i].content;
        }
        i++;
      }

      return `<details style="
        background-color: rgba(13, 110, 253, 0.05);
        border: 1px solid rgba(13, 110, 253, 0.2);
        border-radius: 4px;
        padding: 0.5rem;
        margin-bottom: 1.5rem;
      "><summary style="
        cursor: pointer;
        color: #0d6efd;
        font-weight: 500;
        padding: 0.25rem;
      ">${md.utils.escapeHtml(m[1])}</summary>
      <div class="editable-container" style="padding: 0.5rem;">
        <div class="editable-text" contenteditable="true" style="
          min-height: 1.5em;
          padding: 0.25rem;
          border-radius: 3px;
          transition: background-color 0.2s;
        ">${md.utils.escapeHtml(content.trim())}</div>
        <div style="
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-top: 0.5rem;
          font-size: 0.875rem;
          color: #6c757d;
        ">
          <button class="enter-button" style="
            padding: 0.25rem 0.5rem;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            font-size: 0.75rem;
            color: #6c757d;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.25rem;
          " title="Press Enter to submit">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 12h18m-5-5l5 5-5 5"/>
            </svg>
            <span>Enter to resubmit</span>
          </button>
        </div>
      </div></details>`;
    }
    return ''; // Return empty string for closing tag
  }
});

// Add this near your other markdown-it container configurations
md.use(markdownItContainer, 'loader', {
  render: function (tokens, idx) {
    if (tokens[idx].nesting === 1) {
      // opening tag
      return htmlLoader;
    } else {
      // closing tag
      return ''; // Close the container without any additional markup
    }
  }
});

// Add this near your other markdown-it container configurations
md.use(markdownItContainer, 'mermaid-figure', {
  validate: function(params) {
    return params.trim().match(/^mermaid-figure\s*(.*)$/);
  },
  render: function(tokens, idx) {
    const m = tokens[idx].info.trim().match(/^mermaid-figure\s*(.*)$/);
    
    if (tokens[idx].nesting === 1) {
      // Opening tag - if there's a caption in the params, use it
      const caption = m[1] ? `<figcaption>${md.utils.escapeHtml(m[1])}</figcaption>` : '';
      return `<figure class="mermaid-figure">`;
    } else {
      // Closing tag
      return '</figure>';
    }
  }
});

// Modify your existing mermaid container to work with the figure container
md.use(markdownItContainer, 'mermaid', {
  validate: function(params) {
    return params.trim() === 'mermaid';
  },
  render: function(tokens, idx) {
    if (tokens[idx].nesting === 1) {
      return '<div class="mermaid">';
    } else {
      return '</div>';
    }
  }
});

// Add custom info container
md.use(markdownItContainer, 'info', {
  validate: function(params) {
    return params.trim().match(/^info\s*(.*)$/);
  },
  render: function(tokens, idx) {
    if (tokens[idx].nesting === 1) {
      // opening tag
      return `<div class="info-box" style="
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-left: 4px solid #0d6efd;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #1f2937;
      ">
        <div style="
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.25rem;
        ">
          <span style="font-size: 1.25rem;">ℹ</span>
          <strong>Information</strong>
        </div>
        <div style="margin-left: 1.75rem;">`;
    } else {
      // closing tag
      return '</div></div>';
    }
  }
});

// Add custom warning container for network issues
md.use(markdownItContainer, 'network-warning', {
  validate: function(params) {
    return params.trim().match(/^network-warning\s*(.*)$/);
  },
  render: function(tokens, idx) {
    if (tokens[idx].nesting === 1) {
      // opening tag
      return `<div class="network-warning-box" style="
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-left: 4px solid #ffc107;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #856404;
      ">
        <div style="
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.25rem;
        ">
          <span style="font-size: 1.25rem;">⚠️</span>
          <strong>Network Warning</strong>
        </div>
        <div style="margin-left: 1.75rem;">`;
    } else {
      // closing tag
      return '</div></div>';
    }
  }
});


export function initiateMarkdown(shadowElement) {
  shadowEle = shadowElement;
}

function wrapCodeBlocks(preview) {
  // Find all <pre> elements
  const preBlocks = preview.querySelectorAll('pre');
  preBlocks.forEach(preBlock => {
    // Ensure <pre> has class 'hljs'
    preBlock.classList.add('hljs');

    // Ensure the preBlock does not exceed parent width and text wraps
    preBlock.style.whiteSpace = 'pre-wrap';
    preBlock.style.lineHeight = '1.2em';
    preBlock.style.wordWrap = 'break-word';
    preBlock.style.color = 'royalblue';
    preBlock.style.padding='10px'
    preBlock.style.backgroundColor = '#f0f0f0'; // Light gray background color
    preBlock.style.position = 'relative'; // Ensure the button can be positioned absolutely

    // Create copy button
    const copyButton = document.createElement('button');
    copyButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-6">
    <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 0 1-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 0 1 1.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 0 0-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 0 1-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 0 0-3.375-3.375h-1.5a1.125 1.125 0 0 1-1.125-1.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H9.75" />
  </svg>
  `; // Font Awesome copy icon
    copyButton.classList.add('copy-btn');
    copyButton.style.position = 'absolute';
    copyButton.style.top = '8px';
    copyButton.style.right = '8px';
    copyButton.style.border = 'none';
    copyButton.style.borderRadius = '5px'
    copyButton.style.background = 'transparent';
    copyButton.style.cursor = 'pointer';
    copyButton.style.color = '#333'; // Icon color
    preBlock.appendChild(copyButton);

    // Add event listener to copy button
 // Add event listener to copy button
copyButton.addEventListener('click', () => {
  const tempCopy = copyButton.innerHTML;
  navigator.clipboard.writeText(preBlock.textContent);
  
  // Change button innerHTML to "copied!"
  copyButton.innerHTML = `copied!`;

  // Set a timeout to revert the innerHTML back to the original icon
  setTimeout(() => {
    copyButton.innerHTML = tempCopy;
  }, 2000);
});
  });
}


function removeLeadingNumberAndAsterisks(text) {
  let cleanedText = text.replace(/^\d+\.\s*/, ''); // Remove leading number and period
  cleanedText = cleanedText.replace(/\*/g, ''); // Remove all asterisks
  return cleanedText;
}

function normalizeMarkdownText(text) {
  const lines = text.split('\n');
  
  // Find questions and %%% marker
  let questionLines = [];
  
  // Helper function to identify a question line
  const isQuestion = (line) => {
    // Remove numbers, asterisks, and other formatting
    const cleanLine = line.replace(/^\d+\.\s*/, '')  // Remove leading numbers
                         .replace(/\*\*/g, '')        // Remove bold markers
                         .trim();
    return cleanLine.endsWith('?');
  };
  
  // Scan from bottom to top
  let consecutiveQuestions = [];
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    
    // Check for %%% marker
    if (line.includes('%%%')) {
      continue;
    }
    
    // If it's a question
    if (isQuestion(line)) {
      consecutiveQuestions.unshift(lines[i]);
    } else if (consecutiveQuestions.length > 0) {
      // If we hit non-question content and already have questions
      if (line === '' || line.includes('**')) {
        // Skip empty lines and headers
        continue;
      }
      // Break if we hit real content
      break;
    }
  }
  
  // If we found consecutive questions (2 or more)
  if (consecutiveQuestions.length >= 2) {
    // Find where to insert %%%
    let insertPoint = -1;
    for (let i = lines.length - consecutiveQuestions.length - 1; i >= 0; i--) {
      const line = lines[i].trim();
      if (line !== '' && !line.includes('**')) {
        insertPoint = i;
        break;
      }
    }
    
    if (insertPoint !== -1) {
      // Rebuild the text
      const mainContent = lines.slice(0, insertPoint + 1);
      
      // Clean trailing empty lines from main content
      while (mainContent[mainContent.length - 1]?.trim() === '') {
        mainContent.pop();
      }
      
      // Build final text
      return [
        ...mainContent,
        '',
        '%%%',
        '',
        ...consecutiveQuestions
      ].join('\n');
    }
  }
  
  return text;
}

export function updateMarkdownPreview(text, clone, isResearch=0, markdownPreviewId = '') {
  text = normalizeMarkdownText(text);

  let preview;
  if(markdownPreviewId) {
    preview = clone.querySelector('#'+markdownPreviewId);
    console.log("I am updating markdown preview", preview, markdownPreviewId, "TEXT", text);
  } else {
    preview = clone.querySelector('#markdown-preview');
  }

  let htmlContent = '';
  let contentToRender = text;
  let questions = [];

  // Check for the presence of '%%%' and handle related questions
  const percentIndex = text.indexOf('%%%');
  if (percentIndex !== -1) {
    const questionsText = text.substring(percentIndex + 3);
    questions = questionsText.split('\n')
      .map(sentence => sentence.trim())
      .filter(sentence => sentence.endsWith('?'));
    contentToRender = text.substring(0, percentIndex);
  }

  if(isResearch === 0) {
    htmlContent = `
      <style>
        ul, ol {
          list-style-position: outside !important;
        }
        ul {
          list-style-type: disc; 
        }
        p {
          margin-bottom: 1.5em; 
        }
        figure.mermaid-figure {
          margin: 1em 0;
          display: flex;
          flex-direction: column;
          align-items: center;
          background: rgb(248, 249, 250);
          padding: 1em;
          border-radius: 4px;
        }
        figure.mermaid-figure figcaption {
          margin-top: 0.5em;
          text-align: center;
          font-style: italic;
          color: rgb(102, 102, 102);
        }
        .figure-description {
          margin-top: 1em;
          color: #666;
        }
      </style>
      ${md.render(contentToRender)}
    `;
  } else {
    htmlContent = `
      <style>
        ul, ol {
          list-style-position: outside !important;
        }
        ul {
          list-style-type: disc;
        }
        p {
          margin-bottom: 0; 
        }
        figure.mermaid-figure {
          margin: 1em 0;
          display: flex;
          flex-direction: column;
          align-items: center;
          background: rgb(248, 249, 250);
          padding: 1em;
          border-radius: 4px;
        }
        figure.mermaid-figure figcaption {
          margin-top: 0.5em;
          text-align: center;
          font-style: italic;
          color: rgb(102, 102, 102);
        }
        .figure-description {
          margin-top: 1em;
          color: #666;
        }
      </style>
      ${md.render(contentToRender)}
    `;
  }

  // Remove trailing '**' from paragraph content
  htmlContent = cleanupAsterisks(htmlContent);

  // Update preview with the cleaned-up HTML content
  preview.innerHTML = htmlContent;

  // Initialize spoiler editing
  initializeSpoilerEditing(preview);

  // Process any figures and their descriptions
  const figures = preview.querySelectorAll('figure');
  figures.forEach(figure => {
    const nextElement = figure.nextElementSibling;
    if (nextElement && nextElement.tagName === 'P' && 
        (nextElement.textContent.trim().startsWith('This flowchart') || 
         nextElement.textContent.trim().startsWith('This diagram'))) {
      nextElement.classList.add('figure-description');
    }
  });

  wrapCodeBlocks(preview);

  if (questions.length > 0) {
    const relatedDiv = document.createElement('div');
    relatedDiv.innerHTML = `
      <p style="display: flex; align-items: left;">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" style="margin-right: 5px; height: 24px;width: 24px;">
          <path stroke-linecap="round" stroke-linejoin="round" d="M4.26 10.147a60.438 60.438 0 0 0-.491 6.347A48.62 48.62 0 0 1 12 20.904a48.62 48.62 0 0 1 8.232-4.41 60.46 60.46 0 0 0-.491-6.347m-15.482 0a50.636 50.636 0 0 0-2.658-.813A59.906 59.906 0 0 1 12 3.493a59.903 59.903 0 0 1 10.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.717 50.717 0 0 1 12 13.489a50.702 50.702 0 0 1 7.74-3.342M6.75 15a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm0 0v-3.675A55.378 55.378 0 0 1 12 8.443m-7.007 11.55A5.981 5.981 0 0 0 6.75 15.75v-1.5" />
        </svg>
        <strong>Related</strong>
      </p>`;

    questions.forEach((question, i) => {
      if (i !== 0) {
        const hr = document.createElement('hr');
        relatedDiv.appendChild(hr);
      }
      const button = document.createElement('button');
      button.innerHTML = `+ ${removeLeadingNumberAndAsterisks(question)}`;
      button.style.backgroundColor = 'transparent';
      button.style.border = 'none';
      button.classList.add('followup-button');
      button.style.display = 'block';
      button.style.marginBottom = '5px';
      button.style.color = "royalblue";
      button.style.textAlign = "left";
      button.addEventListener('click', function(event) {
        const prompt = `Use this question: ${question} to expand upon this content ${contentToRender}`;
        general_agent(prompt);
      });
      relatedDiv.appendChild(button);
    });

    preview.appendChild(relatedDiv);
  }
}

function cleanupAsterisks(htmlContent) {
  // Remove leading ** from paragraphs
  htmlContent = htmlContent.replace(/<p>\s*\*\*\s*([^*](?:(?!\*\*).)*)<\/p>/g, '<p>$1</p>');
  
  // Remove trailing ** from paragraphs
  htmlContent = htmlContent.replace(/<p>((?:(?!\*\*).)*[^*])\s*\*\*\s*<\/p>/g, '<p>$1</p>');
  
  return htmlContent;
}

export function convertMarkdownToHTML(markdownText) {
  // Render the Markdown content initially
  let renderedContent = md.render(markdownText);

  // Create a temporary container to parse the rendered HTML
  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = renderedContent;

  // // Apply bold formatting to <p> elements that end with '**'
  // tempDiv.querySelectorAll('p').forEach((p) => {
  //   const regex = /(\S+)\*\*([\s]*)$/; // Match any word ending with '**' at the end of the paragraph
  //   if (regex.test(p.textContent)) {
  //     p.innerHTML = p.innerHTML.replace(regex, `<strong>$1</strong>$2`);
  //   }
  // });

  // Wrap code blocks if necessary
  wrapCodeBlocks(tempDiv);

  // Return the final HTML content as a string
  return tempDiv.innerHTML;
}



function general_agent(text, links=[window.location.href]) {
  const event = new CustomEvent("aiActionCompleted", {
    detail: {
      text, // the original query object used for the request
      type: "general", // the type of AI action
      links: links,
    },
  });
  window.dispatchEvent(event);
}

// Add the event handlers after markdown rendering
export function initializeSpoilerEditing(preview) {
  const editableContainers = preview.querySelectorAll('.editable-container');
  
  editableContainers.forEach(container => {
    const editableText = container.querySelector('.editable-text');
    const enterButton = container.querySelector('.enter-button');
    
    // Add tooltip-style hint
    editableText.setAttribute('title', 'Click to edit, press Enter to submit');
    
    // Handle enter key press
    editableText.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitEditedText(editableText.textContent, e);
      }
    });
    
    // Handle button click
    enterButton.addEventListener('click', (e) => {
      submitEditedText(editableText.textContent, e);
    });
    
    // Add subtle visual cues
    editableText.addEventListener('mouseover', () => {
      editableText.style.backgroundColor = 'rgba(13, 110, 253, 0.1)';
      editableText.style.cursor = 'text';
    });
    
    editableText.addEventListener('mouseout', () => {
      editableText.style.backgroundColor = '';
    });
    
    // Add focus handling
    editableText.addEventListener('focus', () => {
      editableText.style.outline = '2px solid rgba(13, 110, 253, 0.2)';
      editableText.style.backgroundColor = 'rgba(13, 110, 253, 0.05)';
    });
    
    editableText.addEventListener('blur', () => {
      editableText.style.outline = '';
      editableText.style.backgroundColor = '';
    });
  });
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

// TODO:KAI MARKDOWN EDITOR FIX
function submitEditedText(text, e) {
  // Find the closest AI message container
  const aiMessageElement = findClosestAIMessage(e.target);
  
  if (!aiMessageElement) {
    console.warn('Could not find parent AI message element');
    return;
  }

  let diagramId = '';
  if (containsWordReference(text)) {
    console.log("CONTAINS DIAGRAM!!");
    diagramId = generateUniqueId();
    generateDiagramId("aiActionCompleted", text, "mermaid_diagram", diagramId)

  }

  // Create and dispatch a custom event with additional context
  const event = new CustomEvent('aiActionCompleted', {
    detail: {
      text,
      type: "query",
      ele: aiMessageElement,
      fromRightClickMenu: false,
      // Add source information to help with reconstruction
      source: {
        type: 'editableText',
        containerId: aiMessageElement.id
      }
    }
  });
  window.dispatchEvent(event);
}

// Add this new function to reinitialize all editable inputs
export function reinitializeEditableInputs(shadowRoot) {
  // Find all user input containers
  const editableContainers = shadowRoot.querySelectorAll('.user-input-resubmit');
  
  editableContainers.forEach(container => {
    const editableText = container.querySelector('.editable-text');
    const enterButton = container.querySelector('.enter-button');
    
    if (!editableText || !enterButton) return;

    // Remove existing event listeners (to prevent duplicates)
    const newEditableText = editableText.cloneNode(true);
    const newEnterButton = enterButton.cloneNode(true);
    editableText.parentNode.replaceChild(newEditableText, editableText);
    enterButton.parentNode.replaceChild(newEnterButton, enterButton);
    
    // Add tooltip-style hint
    newEditableText.setAttribute('title', 'Click to edit, press Enter to submit');
    
    // Handle enter key press
    newEditableText.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitEditedText(newEditableText.textContent, e);
      }
    });
    
    // Handle button click
    newEnterButton.addEventListener('click', (e) => {
      submitEditedText(newEditableText.textContent, e);
    });
    
    // Add hover effects
    newEditableText.addEventListener('mouseover', () => {
      newEditableText.style.backgroundColor = 'rgba(13, 110, 253, 0.1)';
      newEditableText.style.cursor = 'text';
    });
    
    newEditableText.addEventListener('mouseout', () => {
      newEditableText.style.backgroundColor = '';
    });
    
    // Add focus handling
    newEditableText.addEventListener('focus', () => {
      newEditableText.style.outline = '2px solid rgba(13, 110, 253, 0.2)';
      newEditableText.style.backgroundColor = 'rgba(13, 110, 253, 0.05)';
    });
    
    newEditableText.addEventListener('blur', () => {
      newEditableText.style.outline = '';
      newEditableText.style.backgroundColor = '';
    });
  });
}