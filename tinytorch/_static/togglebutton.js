/**
 * Add Toggle Buttons to elements
 */

let toggleChevron = `
<svg xmlns="http://www.w3.org/2000/svg" class="tb-icon toggle-chevron-right" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round">
<path stroke="none" d="M0 0h24v24H0z" fill="none"/>
<polyline points="9 6 15 12 9 18" />
</svg>`;

var initToggleItems = () => {
  var itemsToToggle = document.querySelectorAll(togglebuttonSelector);
  console.log(`[togglebutton]: Adding toggle buttons to ${itemsToToggle.length} items`)
  // Add the button to each admonition and hook up a callback to toggle visibility
  itemsToToggle.forEach((item, index) => {
    if (item.classList.contains("admonition")) {
      // If it's an admonition block, then we'll add a button inside
      // Generate unique IDs for this item
      var toggleID = `toggle-${index}`;
      var buttonID = `button-${toggleID}`;

      item.setAttribute('id', toggleID);
      if (!item.classList.contains("toggle")){
        item.classList.add("toggle");
      }
      // This is the button that will be added to each item to trigger the toggle
      var collapseButton = `
        <button type="button" id="${buttonID}" class="toggle-button" data-target="${toggleID}" data-button="${buttonID}" data-toggle-hint="${toggleHintShow}" aria-label="Toggle hidden content">
            ${toggleChevron}
        </button>`;

      title = item.querySelector(".admonition-title")
      title.insertAdjacentHTML("beforeend", collapseButton);
      thisButton = document.getElementById(buttonID);

      // Add click handlers for the button + admonition title (if admonition)
      admonitionTitle = document.querySelector(`#${toggleID} > .admonition-title`)
      if (admonitionTitle) {
        // If an admonition, then make the whole title block clickable
        admonitionTitle.addEventListener('click', toggleClickHandler);
        admonitionTitle.dataset.target = toggleID
        admonitionTitle.dataset.button = buttonID
      } else {
        // If not an admonition then we'll listen for the button click
        thisButton.addEventListener('click', toggleClickHandler);
      }

      // Now hide the item for this toggle button unless explicitly noted to show
      if (!item.classList.contains("toggle-shown")) {
        toggleHidden(thisButton);
      }
    } else {
      // If not an admonition, wrap the block in a <details> block
      // Define the structure of the details block and insert it as a sibling
      var detailsBlock = `
        <details class="toggle-details">
            <summary class="toggle-details__summary">
              ${toggleChevron}
              <span class="toggle-details__summary-text">${toggleHintShow}</span>
            </summary>
        </details>`;
      item.insertAdjacentHTML("beforebegin", detailsBlock);

      // Now move the toggle-able content inside of the details block
      details = item.previousElementSibling
      details.appendChild(item)
      item.classList.add("toggle-details__container")

      // Set up a click trigger to change the text as needed
      details.addEventListener('click', (click) => {
        let parent = click.target.parentElement;
        if (parent.tagName.toLowerCase() == "details") {
          summary = parent.querySelector("summary");
          details = parent;
        } else {
          summary = parent;
          details = parent.parentElement;
        }
        // Update the inner text for the proper hint
        if (details.open) {
          summary.querySelector("span.toggle-details__summary-text").innerText = toggleHintShow;
        } else {
          summary.querySelector("span.toggle-details__summary-text").innerText = toggleHintHide;
        }
        
      });

      // If we have a toggle-shown class, open details block should be open
      if (item.classList.contains("toggle-shown")) {
        details.click();
      }
    }
  })
};

// This should simply add / remove the collapsed class and change the button text
var toggleHidden = (button) => {
  target = button.dataset['target']
  var itemToToggle = document.getElementById(target);
  if (itemToToggle.classList.contains("toggle-hidden")) {
    itemToToggle.classList.remove("toggle-hidden");
    button.classList.remove("toggle-button-hidden");
  } else {
    itemToToggle.classList.add("toggle-hidden");
    button.classList.add("toggle-button-hidden");
  }
}

var toggleClickHandler = (click) => {
  // Be cause the admonition title is clickable and extends to the whole admonition
  // We only look for a click event on this title to trigger the toggle.

  if (click.target.classList.contains("admonition-title")) {
    button = click.target.querySelector(".toggle-button");
  } else if (click.target.classList.contains("tb-icon")) {
    // We've clicked the icon and need to search up one parent for the button
    button = click.target.parentElement;
  } else if (click.target.tagName == "polyline") {
    // We've clicked the SVG elements inside the button, need to up 2 layers
    button = click.target.parentElement.parentElement;
  } else if (click.target.classList.contains("toggle-button")) {
    // We've clicked the button itself and so don't need to do anything
    button = click.target;
  } else {
    console.log(`[togglebutton]: Couldn't find button for ${click.target}`)
  }
  target = document.getElementById(button.dataset['button']);
  toggleHidden(target);
}

// If we want to blanket-add toggle classes to certain cells
var addToggleToSelector = () => {
  const selector = "";
  if (selector.length > 0) {
    document.querySelectorAll(selector).forEach((item) => {
      item.classList.add("toggle");
    })
  }
}

// Helper function to run when the DOM is finished
const sphinxToggleRunWhenDOMLoaded = cb => {
  if (document.readyState != 'loading') {
    cb()
  } else if (document.addEventListener) {
    document.addEventListener('DOMContentLoaded', cb)
  } else {
    document.attachEvent('onreadystatechange', function() {
      if (document.readyState == 'complete') cb()
    })
  }
}
sphinxToggleRunWhenDOMLoaded(addToggleToSelector)
sphinxToggleRunWhenDOMLoaded(initToggleItems)

/** Toggle details blocks to be open when printing */
if (toggleOpenOnPrint == "true") {
  window.addEventListener("beforeprint", () => {
    // Open the details
    document.querySelectorAll("details.toggle-details").forEach((el) => {
      el.dataset["togglestatus"] = el.open;
      el.open = true;
    });
  
    // Open the admonitions
    document.querySelectorAll(".admonition.toggle.toggle-hidden").forEach((el) => {
      console.log(el);
      el.querySelector("button.toggle-button").click();
      el.dataset["toggle_after_print"] = "true";
    });
  });
  window.addEventListener("afterprint", () => {
    // Re-close the details that were closed
    document.querySelectorAll("details.toggle-details").forEach((el) => {
      el.open = el.dataset["togglestatus"] == "true";
      delete el.dataset["togglestatus"];
    });
  
    // Re-close the admonition toggle buttons
    document.querySelectorAll(".admonition.toggle").forEach((el) => {
      if (el.dataset["toggle_after_print"] == "true") {
        el.querySelector("button.toggle-button").click();
        delete el.dataset["toggle_after_print"];
      }
    });
  });
}
