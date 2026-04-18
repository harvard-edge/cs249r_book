import { openDB } from "idb"; // Ensure you have `idb` library installed or included
import { copy_download } from "../../components/settings/copy_download.js";

const copy_buttons = `
          
<div id="utility-btn-container"  class="text-blue-400 flex space-x-2 p-2">
<!-- <button id="reference-btn" class="bg-blue-100 text-zinc-900 px-4 py-0.5 rounded-md">1</button> -->
    <!-- Buttons with icons -->

         <button id='highlight-btn' class="w-4 h-4  flex items-center justify-center hover:text-blue-700"
              title="Highlight">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="icons">
                <path stroke-linecap="round" stroke-linejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
              </svg>
            </button>   
    </button>

                <button id='sr-send-btn' class="w-4 h-4  mr-2  flex items-center justify-center hover:text-blue-700 st-send-btn"
               >
                

                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="icons">
  <path stroke-linecap="round" stroke-linejoin="round" d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0 1 11.186 0Z" />
</svg>

              </button>
<button  id='share-btn' class="w-4 h-4 flex items-center justify-center hover:text-blue-700"  title="Share"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="icons">
<path stroke-linecap="round" stroke-linejoin="round" d="M7.217 10.907a2.25 2.25 0 1 0 0 2.186m0-2.186c.18.324.283.696.283 1.093s-.103.77-.283 1.093m0-2.186 9.566-5.314m-9.566 7.5 9.566 5.314m0 0a2.25 2.25 0 1 0 3.935 2.186 2.25 2.25 0 0 0-3.935-2.186Zm0-12.814a2.25 2.25 0 1 0 3.933-2.185 2.25 2.25 0 0 0-3.933 2.185Z" />
</svg>
</button>
<button id='copy-btn' class="w-4 h-4  mr-2  flex items-center justify-center hover:text-blue-700"  title="Copy"><svg id="copy-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="icons">
<path stroke-linecap="round" stroke-linejoin="round" d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 0 1-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 0 1 1.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 0 0-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 0 1-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 0 0-3.375-3.375h-1.5a1.125 1.125 0 0 1-1.125-1.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H9.75" />
</svg>
</button>
<button id='download-btn'class="w-4 h-4  mr-2  flex items-center justify-center hover:text-blue-700" title="Download"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="icons">
<path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
</svg>

</div>`;

// const button_util_container = `
// <div class="flex justify-between"> </div>
// `

export function debounce(func, wait) {
  let timeout;
  return function (...args) {
    const context = this;
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(context, args), wait);
  };
}

export function extractQuestion(text) {
  // Remove leading/trailing whitespace
  text = text.trim();
  
  // Remove "Use this question:" and variants
  text = text.replace(/^(Use this question:?\s*)/i, '');
  
  // Case 1: Text between * and ?
  if (text.startsWith('*')) {
      const match = text.match(/\*(.*?)\?/);
      if (match) return match[1].trim() + '?';
  }
  
  // Case 2: Text up to first ?
  const questionMatch = text.match(/(.*?)\?/);
  if (questionMatch) return questionMatch[0].trim();
  
  // Case 3: Return full text if no ? found
  return text;
}

export function generateUniqueId() {
  // Get current timestamp in milliseconds
  const timestamp = new Date().getTime();

  // Generate a random number and convert to base 36 (includes letters)
  const randomPart = Math.random().toString(36).substring(2, 10);

  // Get performance timing if available for more precision
  const performance =
    window.performance && window.performance.now
      ? window.performance.now().toString(36).replace(".", "")
      : "";

  // Combine all parts
  return `id-${timestamp}-${performance}-${randomPart}`;
}

// We can also add a simpler version if needed
export function generateSimpleId() {
  return `${new Date().getTime()}-${Math.random()
    .toString(36)
    .substring(2, 15)}`;
}

export function truncateToLastChars(text, maxChars) {
  if (text.length <= maxChars) return text;
  return text.substr(text.length - maxChars);
}
export function alert(shadowElem, text, type) {
  const token = "none";
  try {
    let noticeElement;
    if (type === "success") {
      noticeElement = shadowElem.querySelector("#success-notice");
      if (!noticeElement) {
        // Create the success notice element
        noticeElement = document.createElement("div");
        noticeElement.id = "success-notice";
        noticeElement.style.display = "none";
        noticeElement.style.position = "fixed";
        noticeElement.style.bottom = "5px";
        noticeElement.style.left = "5px";
        noticeElement.style.background = "lightgreen";
        noticeElement.style.padding = "5px";
        noticeElement.style.borderRadius = "5px";
        noticeElement.style.zIndex = "9999";

        shadowElem.appendChild(noticeElement);
      }
      noticeElement.textContent = text;
      noticeElement.style.display = "block";

      // Hide the success notice after 3 seconds
      setTimeout(() => {
        noticeElement.style.display = "none";
      }, 2000);
      return token;
    } else {
      noticeElement = shadowElem.querySelector("#error-notice");
      if (!noticeElement) {
        // Create the error notice element
        noticeElement = document.createElement("div");
        noticeElement.id = "error-notice";
        noticeElement.style.display = "none";
        noticeElement.style.position = "fixed";
        noticeElement.style.bottom = "5px";
        noticeElement.style.left = "5px";
        noticeElement.style.background = "lightcoral";
        noticeElement.style.padding = "5px";
        noticeElement.style.borderRadius = "5px";
        noticeElement.style.zIndex = "9999";
        shadowElem.appendChild(noticeElement);
      }
      noticeElement.textContent = text;
      noticeElement.style.display = "block";

      // Hide the error notice after 3 seconds
      setTimeout(() => {
        noticeElement.style.display = "none";
      }, 2000);
    }
  } catch (error) {
    console.error("Error in alert function:", error);
  }
}

export async function saveToken(token) {
  let db;
  try {
    db = await openDB("tinyML_token", 1, {
      upgrade(db) {
        if (!db.objectStoreNames.contains("tokens")) {
          db.createObjectStore("tokens", { keyPath: "id" });
        }
      },
    });
    const tx = db.transaction("tokens", "readwrite");
    const store = tx.objectStore("tokens");
    await store.put({ id: "token_avaya", value: token });
    await tx.done; // Ensures the transaction completes successfully
  } catch (error) {
    console.error("Failed to save token:", error);
    // Handle the error (e.g., by retrying, logging, or notifying the user)
  } finally {
    if (db) db.close(); // Close the database connection to prevent leaks
  }
}

export async function getToken() {
  const db = await openDB("tinyML_token", 1);
  const tx = db.transaction("tokens", "readonly");
  const store = tx.objectStore("tokens");
  const tokenObject = await store.get("token_avaya");
  await tx.done;
  return tokenObject ? tokenObject.value : null;
}

export function scrollToBottom(shadowRoot) {
  const iframeContainer = shadowRoot.querySelector("#message-container");

  if (iframeContainer) {
    // Wait for the next 'paint' after the browser has had a chance to layout content
    requestAnimationFrame(() => {
      iframeContainer.scrollTop = iframeContainer.scrollHeight;
    });
  }
}

export function scrollToBottomSmooth(shadowRoot) {
  const iframeContainer = shadowRoot.querySelector("#message-container");
  let scrollTarget = shadowRoot.querySelector("#scroll-target");

  if (iframeContainer && scrollTarget) {
    scrollTarget.scrollIntoView({ behavior: "smooth" });
  } else if (iframeContainer && !scrollTarget) {
    // Create a new element with the ID 'scroll-target'
    scrollTarget = document.createElement("div");
    scrollTarget.id = "scroll-target";
    // Append the new element to the iframeContainer
    iframeContainer.appendChild(scrollTarget);
    // Now scroll into view
    scrollTarget.scrollIntoView({ behavior: "smooth" });
  }
}

export function scrollToTopSmooth(shadowRoot) {
  const iframeContainer = shadowRoot.querySelector("#message-container");
  console.log("we gonna SMOOTH scroll down");
  
  // Get ALL difficulty-dropdown elements and select the last one
  const difficultyDropdowns = iframeContainer.querySelectorAll(".difficulty-dropdown");
  const lastDifficultyDropdown = difficultyDropdowns[difficultyDropdowns.length - 1];
  
  // Scroll the last difficultyDropdown into view if it exists
  if (lastDifficultyDropdown) {
    lastDifficultyDropdown.scrollIntoView({ behavior: "smooth", block: 'start' });
    console.log("Scrolling to last difficulty dropdown:", lastDifficultyDropdown);
  } else {
    console.log("No difficulty dropdowns found");
  }
}

export function cloneElementById(
  shadowEle,
  originalElementId,
  newId = "",
  targetContainerId = ""
) {

  const originalElement = shadowEle.querySelector(`#${originalElementId}`);
  if (!originalElement) {
    console.error("Element with ID " + originalElementId + " not found.");
    return null;
  }

  // Clone the original element
  const clone = originalElement.cloneNode(true);

  // Assign a new ID to the clone if provided
  if (newId !== "") {
    clone.id = newId;
  } else {
    clone.removeAttribute("id"); // Remove ID attribute if no new ID provided
  }

  // If a target container ID is provided, append the clone to this container
  if (targetContainerId !== "") {
    const targetContainer = shadowEle.querySelector(`#${targetContainerId}`);

    if (targetContainer) {
      if (targetContainerId === "message-container") {
        const scrollTarget = shadowEle.querySelector("#scroll-target");

        // Insert the new message just before the scroll-target
        targetContainer.insertBefore(clone, scrollTarget);
      } else {
        // if (targetContainer) {
        targetContainer.appendChild(clone);
      }
    } else {
      console.error(
        "Target container with ID " + targetContainerId + " not found."
      );
    }
  }

  // Return the cloned element
  return clone;
}

export function cloneElementById_OLD_DELETE(
  shadowEle,
  originalElementId,
  newId = "",
  targetContainerId = ""
) {
  // Assuming shadowEle is a shadow root or document
  const originalElement = shadowEle.querySelector(`#${originalElementId}`);
  if (!originalElement) {
    console.error("Element with ID " + originalElementId + " not found.");
    return null;
  }

  // Clone the original element
  const clone = originalElement.cloneNode(true);

  // Assign a new ID to the clone if provided
  if (newId !== "") {
    clone.id = newId;
  } else {
    clone.removeAttribute("id"); // Remove ID attribute if no new ID provided
  }

  // If a target container ID is provided, append the clone to this container
  if (targetContainerId !== "") {
    const targetContainer = shadowEle.querySelector(`#${targetContainerId}`);
    if (targetContainer) {
      targetContainer.appendChild(clone);
    } else {
      console.error(
        "Target container with ID " + targetContainerId + " not found."
      );
    }
  }

  // Return the cloned element
  return clone;
}

export function generateHighPrecisionUniqueId() {
  const perfTime = performance.now().toString().replace(".", "");
  const randomPortion = Math.random().toString(36).substring(2, 15);
  return `id-${perfTime}-${randomPortion}`;
}

export function assignUniqueIdsToElementAndChildren(element) {
  // Assign a unique ID to the root element
  element.id = generateHighPrecisionUniqueId();

  // Iterate over all child elements and recursively assign unique IDs
  element.querySelectorAll("*").forEach(assignUniqueIdsToElementAndChildren);
}

export function assignAndTrackUniqueIds(element, idsToTrack = []) {
  // Initialize array to store new IDs in same order as idsToTrack
  const trackedNewIds = new Array(idsToTrack.length).fill(null);
  
  // Assign unique ID to root element
  const rootId = generateHighPrecisionUniqueId();
  element.id = rootId;
  
  // Check if root element's original ID was in tracking list
  const rootIndex = idsToTrack.indexOf(element.id);
  if (rootIndex !== -1) {
    trackedNewIds[rootIndex] = rootId;
  }

  // Process all child elements
  element.querySelectorAll("*").forEach(child => {
    const originalId = child.id;
    const newId = generateHighPrecisionUniqueId();
    child.id = newId;
    
    // If this child's original ID was in our tracking list, store the new ID
    const trackIndex = idsToTrack.indexOf(originalId);
    if (trackIndex !== -1) {
      trackedNewIds[trackIndex] = newId;
    }
  });

  return trackedNewIds;
}

// Add this new function to encode messages for URL
export function encodeMessageForURL(text) {
  try {
    // If text is an HTML element, get its outerHTML
    const htmlContent = typeof text === "string" ? text : text.outerHTML;
    return encodeURIComponent(htmlContent).replace(/%20/g, "+");
  } catch (error) {
    console.error("Error encoding message for URL:", error);
    return "";
  }
}

// Add this new function to decode messages from URL
export function decodeMessageFromURL(text) {
  try {
    // First decode the URI component
    const decodedText = decodeURIComponent(text.replace(/\+/g, " "));

    // Parse the HTML string into actual DOM elements
    const parser = new DOMParser();
    const doc = parser.parseFromString(decodedText, "text/html");

    // Return the first element from the body
    // This will preserve the entire HTML structure
    return doc.body.firstElementChild;
  } catch (error) {
    console.error("Error decoding message from URL:", error);
    return null;
  }
}

// Add new popover function
export function showPopover(shadowEle, message, type = "info", duration = 3000) {
    const popover = document.createElement('div');
    popover.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 20px;
        padding: 12px 24px;
        border-radius: 8px;
        opacity: 0;
        transform: translateY(20px);
        transition: all 0.3s ease;
        z-index: 10000;
        font-size: 14px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    `;
    popover.classList.add('popover-socratiq');

    // Set style based on type
    switch(type) {
        case "info":
            popover.style.backgroundColor = 'rgba(59, 130, 246, 0.9)'; // Blue
            popover.style.color = 'white';
            break;
        case "error":
            popover.style.backgroundColor = 'rgba(239, 68, 68, 0.9)'; // Red
            popover.style.color = 'white';
            break;
        case "success":
            popover.style.backgroundColor = 'rgba(34, 197, 94, 0.9)'; // Green
            popover.style.color = 'white';
            break;
        default:
            popover.style.backgroundColor = 'rgba(59, 130, 246, 0.9)'; // Default blue
            popover.style.color = 'white';
    }

    popover.textContent = message;
    shadowEle.appendChild(popover);

    // Trigger animation
    setTimeout(() => {
        popover.style.opacity = "1";
        popover.style.transform = "translateY(0)";
    }, 10);

    // Remove after duration
    setTimeout(() => {
        popover.style.opacity = "0";
        popover.style.transform = "translateY(-20px)";
        setTimeout(() => popover.remove(), 300);
    }, duration);
}

// Update share button code
export function add_copy_paste_share_buttons(
  shadowEle,
  clone,
  ref_buttons_element,
  accumulatedResponse = ""
) {
  // Store the markdown content as a data attribute
  if (accumulatedResponse) {
    clone.setAttribute("data-markdown", accumulatedResponse);
  }

  const container = document.createElement("div");
  container.setAttribute("class", "flex justify-between");
  container.classList.add("copy-paste-share-btn-container");


  const refButtonsContainer = document.createElement("div");
  refButtonsContainer.setAttribute("class", "flex items-center");

  if (ref_buttons_element instanceof Element) {
    refButtonsContainer.appendChild(ref_buttons_element);
  }

  const copyButtonsContainer = document.createElement("div");
  copyButtonsContainer.innerHTML = copy_buttons;

  container.appendChild(refButtonsContainer);
  container.appendChild(copyButtonsContainer);


  clone.appendChild(container);

  copy_download(shadowEle, clone);

  return clone;
}

/**
 * Sets the target attribute of all <a> tags inside <td> elements to '_blank' and adds rel='noopener noreferrer'.
 *
 * @param {HTMLElement} clone - The element to clone and search for <a> tags
 */
export function make_links_load_new_page(clone) {
  // Query all <a> tags inside <td> elements
  const links = clone.querySelectorAll("a");

  // Set the target attribute of each link to '_blank'
  links.forEach((link) => {
    link.setAttribute("target", "_blank");
    // Optional: Add rel="noopener noreferrer" for security reasons
    link.setAttribute("rel", "noopener noreferrer");
  });
}


export function updateCloneAttributes(clone, prompt, type, title='') {
  // Ensure the clone is a valid DOM element
  if (!(clone instanceof HTMLElement)) {
    throw new Error("Invalid clone element provided.");
  }


  clone.setAttribute('data-prompt', prompt);
  clone.setAttribute('data-type', type);
  clone.setAttribute('data-title', title);
}


export function removeSkeletonLoaders(clone) {
  // Remove all skeleton loader elements
  const skeletonLoaders = clone.querySelectorAll('.skeleton-loader');
  skeletonLoaders.forEach(loader => loader.remove());
  
  // Also remove any elements with loader class
  const loaderElements = clone.querySelectorAll('.loader');
  loaderElements.forEach(loader => loader.remove());

  return clone;
}


export function insertDiagramElements(messageContent, clone) {
  // Find all strong elements
  const strongElements = clone.querySelectorAll('strong');
  
  // Find the one containing "Related" text
  const relatedElement = Array.from(strongElements).find(el => 
    el.textContent.toLowerCase().includes('related')
  );
          
  if (relatedElement) {
    // Create new div for diagram
    const diagramDiv = document.createElement('div');
    diagramDiv.className = 'generated-diagram';
    diagramDiv.style.cssText = `
      margin: 1.5rem 0;
      padding: 1rem;
      border-radius: 8px;
      background-color: #f8f9fa;
      border: 1px solid #e9ecef;
    `;
    
    diagramDiv.innerHTML = `
      <h3 style="margin-bottom: 1rem; font-weight: 600; color: #1f2937;">Generated Diagram</h3>
      ${messageContent}
    `;
    
    // Insert before the parent element containing "related"
    relatedElement.parentElement.parentElement.insertBefore(
      diagramDiv, 
      relatedElement.parentElement
    );
  } else {
    // If no "Related" section found, append to the end of clone
    clone.appendChild(diagramDiv);
  }
}

export function findClosestAIMessage(element) {
  if (!element) {
    console.warn('No element provided to findClosestAIMessage');
    return null;
  }
  
  // Try to find the closest parent with any of these selectors
  const selectors = [
    '.ai-message-chat',
    '.message-container[data-type="ai"]',
    '[id^="id-"][class*="text-sm"][class*="markdown-preview-container"]',
    '.markdown-preview-container'
  ];
  
  for (const selector of selectors) {
    const match = element.closest(selector);
    if (match) {
      // For debugging
      console.log(`Found AI message container with selector: ${selector}`);
      return match;
    }
  }
  
  // If no match is found, log the element for debugging
  console.warn('Could not find parent AI message element. Element:', element);
  return null;
}

export function insertAtProportionalPoints(parentElement, firstComponent, secondComponent) {
    try {
        // Log initial state
        console.log('Inserting charts - Initial container state:', {
            parentElement,
            childCount: parentElement.children.length,
            firstComponent,
            secondComponent
        });

        // Get all child nodes that aren't charts (to avoid counting previously inserted charts)
        const children = Array.from(parentElement.children).filter(child => 
            !child.classList.contains('chart-wrapper')
        );
        const totalChildren = children.length;

        console.log('Filtered children count:', totalChildren);

        // Calculate insertion points (rounded to nearest integer)
        const oneThirdIndex = Math.round(totalChildren / 3);
        const twoThirdsIndex = Math.round((totalChildren * 2) / 3);

        console.log('Calculated insertion points:', {
            oneThirdIndex,
            twoThirdsIndex
        });

        // Create wrapper divs for the charts
        const firstWrapper = document.createElement('div');
        firstWrapper.className = 'chart-wrapper xy-chart-container';
        firstWrapper.style.cssText = `
            margin: 1.5rem 0;
            padding: 1rem;
            border-radius: 8px;
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
        `;
        firstWrapper.appendChild(firstComponent);

        const secondWrapper = document.createElement('div');
        secondWrapper.className = 'chart-wrapper quadrant-chart-container';
        secondWrapper.style.cssText = `
            margin: 1.5rem 0;
            padding: 1rem;
            border-radius: 8px;
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
        `;
        secondWrapper.appendChild(secondComponent);

        // Store references to elements at insertion points
        const firstInsertionPoint = children[oneThirdIndex];
        const secondInsertionPoint = children[twoThirdsIndex];

        console.log('Insertion points found:', {
            firstInsertionPoint,
            secondInsertionPoint
        });

        // Insert components at calculated positions
        if (firstInsertionPoint) {
            firstInsertionPoint.parentNode.insertBefore(firstWrapper, firstInsertionPoint);
            console.log('Inserted first chart before:', firstInsertionPoint);
        } else {
            parentElement.appendChild(firstWrapper);
            console.log('Appended first chart to end (no insertion point found)');
        }

        // Recalculate children after first insertion to maintain proper spacing
        const updatedChildren = Array.from(parentElement.children).filter(child => 
            !child.classList.contains('chart-wrapper')
        );
        const newSecondInsertionPoint = updatedChildren[twoThirdsIndex];

        if (newSecondInsertionPoint) {
            newSecondInsertionPoint.parentNode.insertBefore(secondWrapper, newSecondInsertionPoint);
            console.log('Inserted second chart before:', newSecondInsertionPoint);
        } else {
            parentElement.appendChild(secondWrapper);
            console.log('Appended second chart to end (no insertion point found)');
        }

        console.log('Final container state:', {
            childCount: parentElement.children.length,
            chartWrappers: parentElement.querySelectorAll('.chart-wrapper').length
        });

        return true;
    } catch (error) {
        console.error('Error inserting components at proportional points:', error);
        return false;
    }
}

export function injectOfflineWarning(clone) {
  const markdownContainer = clone.querySelector('.markdown-preview-container');
  if (!markdownContainer) return;

  // Create warning element
  const warningDiv = document.createElement('div');
  warningDiv.className = 'offline-warning';
  warningDiv.style.cssText = `
    background-color: #fff3cd;
    border: 1px solid #ffeeba;
    border-left: 4px solid #ffc107;
    border-radius: 4px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
    color: #856404;
  `;

  // Create warning content
  warningDiv.innerHTML = `
    <div style="
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.25rem;
    ">
      <span style="font-size: 1.25rem;">⚠️</span>
      <strong>Offline Mode</strong>
    </div>
    <div style="margin-left: 1.75rem;">
      This quiz was generated offline. Some features may be limited.
    </div>
  `;

  // Insert at the beginning of the markdown container
  markdownContainer.insertBefore(warningDiv, markdownContainer.firstChild);
}

// Add this new function
export function cleanupPopovers(shadowEle) {
    const popovers = shadowEle.querySelectorAll('.popover-socratiq');
    popovers.forEach(popover => popover.remove());
}