/**
 * Add attributes to Thebe blocks to initialize thebe properly
 */
var configureThebe = () => {
  // Load thebe config in case we want to update it as some point
  console.log("[sphinx-thebe]: Loading thebe config...");
  thebe_config = document.querySelector("script[type=\"text/x-thebe-config\"]");

  // If we already detect a Thebe cell, don't re-run
  if (document.querySelectorAll("div.thebe-cell").length > 0) {
    return;
  }

  // Update thebe buttons with loading message
  document.querySelectorAll(".thebe-launch-button").forEach((button) => {
    button.innerHTML = `
        <div class="spinner">
            <div class="rect1"></div>
            <div class="rect2"></div>
            <div class="rect3"></div>
            <div class="rect4"></div>
        </div>
        <span class="loading-text"></span>`;
  });

  // Set thebe event hooks
  var thebeStatus;
  thebelab.on("status", function (evt, data) {
    console.log("Status changed:", data.status, data.message);

    const button = document.querySelector(".thebe-launch-button ");
    button.classList.replace(
	    `thebe-status-${thebeStatus}`, 
	    `thebe-status-${data.status}`
    )
    button.querySelector(".loading-text")
      .innerHTML = (
        `<span class='launch_msg'>Launching from mybinder.org: </span>
	 <span class='status'>${data.status}</span>`
      );

    // Now update our thebe status
    thebeStatus = data.status;

    // Find any cells with an initialization tag and ask thebe to run them when ready
    if (data.status === "ready") {
      var thebeInitCells = document.querySelectorAll(
        ".thebe-init, .tag_thebe-init"
      );
      thebeInitCells.forEach((cell) => {
        console.log("Initializing Thebe with cell: " + cell.id);
        cell.querySelector(".thebelab-run-button").click();
      });
    }
  });
};

/**
 * Update the page DOM to use Thebe elements
 */
var modifyDOMForThebe = () => {
  // Find all code cells, replace with Thebe interactive code cells
  const codeCells = document.querySelectorAll(thebe_selector);
  codeCells.forEach((codeCell, index) => {
    const codeCellId = (index) => `codecell${index}`;
    codeCell.id = codeCellId(index);
    codeCellText = codeCell.querySelector(thebe_selector_input);
    codeCellOutput = codeCell.querySelector(thebe_selector_output);

    // Clean up the language to make it work w/ CodeMirror and add it to the cell
    dataLanguage = detectLanguage(kernelName);

    // Re-arrange the cell and add metadata
    if (codeCellText) {
      codeCellText.setAttribute("data-language", dataLanguage);
      codeCellText.setAttribute("data-executable", "true");

      // If we had an output, insert it just after the `pre` cell
      if (codeCellOutput) {
        codeCellOutput.setAttribute("data-output", "");
        codeCellText.insertAdjacentElement('afterend', codeCellOutput);
      }
    }

    // Remove sphinx-copybutton blocks, which are common in Sphinx
    codeCell.querySelectorAll("button.copybtn").forEach((el) => {
      el.remove();
    });
  });
};

var initThebe = () => {
  // Load thebe dynamically if it's not already loaded
  if (typeof thebelab === "undefined") {
    console.log("[sphinx-thebe]: Loading thebe from CDN...");
    document.querySelector(".thebe-launch-button ").innerText = "Loading thebe from CDN...";

    const script = document.createElement("script");
    script.src = `${THEBE_JS_URL}`;
    document.head.appendChild(script);

    // Runs once the script has finished loading
    script.addEventListener("load", () => {
      console.log("[sphinx-thebe]: Finished loading thebe from CDN...");
      configureThebe();
      modifyDOMForThebe();
      thebelab.bootstrap();
    });
  } else {
    console.log(
      "[sphinx-thebe]: thebe already loaded, not loading from CDN..."
    );
    configureThebe();
    modifyDOMForThebe();
    thebelab.bootstrap();
  }
};

// Helper function to munge the language name
var detectLanguage = (language) => {
  if (language.indexOf("python") > -1) {
    language = "python";
  } else if (language === "ir") {
    language = "r";
  }
  return language;
};
