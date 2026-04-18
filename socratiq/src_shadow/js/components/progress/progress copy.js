import progressContent from "./progress.html";

let originalprogressContent;
let turnOffProgress = false;

export function toggleProgress(progressOff){
  turnOffProgress = !progressOff
}

export function injectProgress(clone) {

  console.log("i am progress", clone)

  if(turnOffProgress) return

  const template = document.createElement("template");
  template.innerHTML = progressContent;
  const progress = clone.querySelector("#progress");

  if (progress) {
    // Clone the node deeply
    originalprogressContent = progress.cloneNode(true);
    progress.innerHTML = ""; // Clear existing content if necessary
    progress.appendChild(template.content.cloneNode(true));
  } else {
    console.error("Button menu not found");
  }
  return progress;
}

export async function removeElementById(clone, elementId) {
  // Attempt to find the element by ID within the given DOM structure (clone)
  if(turnOffProgress) return

  const elementToRemove = clone.querySelector(`#${elementId}`);

  if (elementToRemove) {
    // Reveal all hidden SVGs before proceeding with element removal
    revealSVGs(elementToRemove);
    // Delay removal of the element by 1000 milliseconds (1 second)
    setTimeout(() => {
      // If the element is still part of the DOM, remove it
      if (elementToRemove.parentNode) {
        elementToRemove.parentNode.removeChild(elementToRemove);
      }
    }, 500);
  } else {
    // If the element is not found, log an error message
    console.error(`Element with ID '${elementId}' not found.`);
  }
}

/**
 * Reveals all hidden SVG elements within the provided clone.
 *
 * @param {Element} clone - The clone element to search for hidden SVGs.
 * @return {void} No return value.
 */
export function revealSVGs(clone) {

  if(turnOffProgress) return


  // Query all SVG elements within the clone
  const svgElements = clone.querySelectorAll("svg.hidden");

  // Iterate through each SVG element and remove the 'hidden' class
  svgElements.forEach((svg) => {
    svg.classList.remove("hidden");
  });
}

export function injectMenu(shadowEle) {
  
  if(turnOffProgress) return

  const progress = clone.querySelector("#progress");

  if (progress && originalprogressContent) {
    // Clear and append the cloned original content
    progress.innerHTML = "";
    progress.appendChild(originalprogressContent.cloneNode(true));
  } else {
    console.error("Button menu not found or original content not stored");
  }
}

function addItem(shadowEle, text, index) {
  if(turnOffProgress) return


  const checklist = shadowEle.querySelector("#checklist");
  if (checklist) {
    const item = document.createElement("div");
    item.className =
      "flex items-center text-blue-400 dark:text-blue-400 bg-transparent p-1 rounded-lg border-b border-zinc-300 dark:border-zinc-700 animate-fadeInUp";
    item.innerHTML = `
    <div class="flex justify-between w-full items-center bg-white>
      <span class="font-mono">• ${text}</span>
      <svg class="w-6 h-6 hidden" id=${
        "progress" + index
      } fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
        </div>
      `;

    checklist.appendChild(item);
  }
}

export function waitForElementToDisplay(
  selector,
  clone,
  callback,
  checkFrequencyInMs,
  timeoutInMs
) {
  if(turnOffProgress) return

  const startTimeInMs = Date.now();
  const interval = setInterval(function () {
    if (Date.now() - startTimeInMs > timeoutInMs) {
      clearInterval(interval);
      return;
    }
    const element = clone.querySelector(selector);
    if (element) {
      clearInterval(interval);
      callback(element);
    }
  }, checkFrequencyInMs);
}

// Use this function to monitor when the SVG becomes available
export function showProgressItem(clone, index) {

  if(turnOffProgress) return


  const svgId = `progress${index}`;
  waitForElementToDisplay(
    `#${svgId}`,
    clone,
    function (element) {
      element.classList.remove("hidden"); // or element.style.display = 'block';
    },
    100,
    5000
  ); // check every 100ms, timeout after 5000ms
}

export function addProgress(list, shadowEle) {

  if(turnOffProgress) return


  list.forEach((item, index) => {
    // addItem(shadowEle, item, index)
    setTimeout(() => addItem(shadowEle, item, index), 100 * index); // Staggered addition
  });
}

// function removeItem(item) {

//   item.classList.add("animate-fadeOutDown");
//   item.addEventListener("animationend", () => {
//     item.remove(); // Remove the item after the animation ends
//   });
// }

export function removeTopItem(shadowEle) {
  if(turnOffProgress) return


  const checklist = shadowEle.querySelector("#checklist");
  if (checklist.firstChild) {
    const item = checklist.firstChild;
    item.classList.add("animate-fadeOutDown");
    item.addEventListener("animationend", function handler() {
      item.remove(); // Remove the item after the animation ends
      item.removeEventListener("animationend", handler);
    });
  }
}

//   EXAMPLE
// Assuming this call is triggered by some event, like a button click
// document.getElementById('remove-button').addEventListener('click', () => removeTopItem(document.querySelector('#your-shadow-root')));
//
