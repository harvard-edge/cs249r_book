window.QuartoLineHighlight = function () {
  function isPrintView() {
    return /print-pdf/gi.test(window.location.search) || /view=print/gi.test(window.location.search);
  }

  const delimiters = {
    step: "|",
    line: ",",
    lineRange: "-",
  };

  const regex = new RegExp(
    "^[\\d" + Object.values(delimiters).join("") + "]+$"
  );

  function handleLinesSelector(deck, attr) {
    // if we are in printview with pdfSeparateFragments: false
    // then we'll also want to supress
    if (regex.test(attr)) {
      if (isPrintView() && deck.getConfig().pdfSeparateFragments !== true) {
        return false;
      } else {
        return true;
      }
    } else {
      return false;
    }
  }

  const kCodeLineNumbersAttr = "data-code-line-numbers";
  const kFragmentIndex = "data-fragment-index";

  function initQuartoLineHighlight(deck) {
    const divSourceCode = deck
      .getRevealElement()
      .querySelectorAll("div.sourceCode");
    // Process each div created by Pandoc highlighting - numbered line are already included.
    divSourceCode.forEach((el) => {
      if (el.hasAttribute(kCodeLineNumbersAttr)) {
        const codeLineAttr = el.getAttribute(kCodeLineNumbersAttr);
        el.removeAttribute(kCodeLineNumbersAttr);
        if (handleLinesSelector(deck, codeLineAttr)) {
          // Only process if attr is a string to select lines to highlights
          // e.g "1|3,6|8-11"
          const codeBlock = el.querySelectorAll("pre code");
          codeBlock.forEach((code) => {
            // move attributes on code block
            code.setAttribute(kCodeLineNumbersAttr, codeLineAttr);

            const scrollState = { currentBlock: code };

            // Check if there are steps and duplicate code block accordingly
            const highlightSteps = splitLineNumbers(codeLineAttr);
            if (highlightSteps.length > 1) {
              // If the original code block has a fragment-index,
              // each clone should follow in an incremental sequence
              let fragmentIndex = parseInt(
                code.getAttribute(kFragmentIndex),
                10
              );
              fragmentIndex =
                typeof fragmentIndex !== "number" || isNaN(fragmentIndex)
                  ? null
                  : fragmentIndex;

              let stepN = 1;
              highlightSteps.slice(1).forEach(
                // Generate fragments for all steps except the original block
                (step) => {
                  var fragmentBlock = code.cloneNode(true);
                  fragmentBlock.setAttribute(
                    "data-code-line-numbers",
                    joinLineNumbers([step])
                  );
                  fragmentBlock.classList.add("fragment");

                  // Pandoc sets id on spans we need to keep unique
                  fragmentBlock
                    .querySelectorAll(":scope > span")
                    .forEach((span) => {
                      if (span.hasAttribute("id")) {
                        span.setAttribute(
                          "id",
                          span.getAttribute("id").concat("-" + stepN)
                        );
                      }
                    });
                  stepN = ++stepN;

                  // Add duplicated <code> element after existing one
                  code.parentNode.appendChild(fragmentBlock);

                  // Each new <code> element is highlighted based on the new attributes value
                  highlightCodeBlock(fragmentBlock);

                  if (typeof fragmentIndex === "number") {
                    fragmentBlock.setAttribute(kFragmentIndex, fragmentIndex);
                    fragmentIndex += 1;
                  } else {
                    fragmentBlock.removeAttribute(kFragmentIndex);
                  }

                  // Scroll highlights into view as we step through them
                  fragmentBlock.addEventListener(
                    "visible",
                    scrollHighlightedLineIntoView.bind(
                      this,
                      fragmentBlock,
                      scrollState
                    )
                  );
                  fragmentBlock.addEventListener(
                    "hidden",
                    scrollHighlightedLineIntoView.bind(
                      this,
                      fragmentBlock.previousSibling,
                      scrollState
                    )
                  );
                }
              );
              code.removeAttribute(kFragmentIndex);
              code.setAttribute(
                kCodeLineNumbersAttr,
                joinLineNumbers([highlightSteps[0]])
              );
            }

            // Scroll the first highlight into view when the slide becomes visible.
            const slide =
              typeof code.closest === "function"
                ? code.closest("section:not(.stack)")
                : null;
            if (slide) {
              const scrollFirstHighlightIntoView = function () {
                scrollHighlightedLineIntoView(code, scrollState, true);
                slide.removeEventListener(
                  "visible",
                  scrollFirstHighlightIntoView
                );
              };
              slide.addEventListener("visible", scrollFirstHighlightIntoView);
            }

            highlightCodeBlock(code);
          });
        }
      }
    });
  }

  function highlightCodeBlock(codeBlock) {
    const highlightSteps = splitLineNumbers(
      codeBlock.getAttribute(kCodeLineNumbersAttr)
    );

    if (highlightSteps.length) {
      // If we have at least one step, we generate fragments
      highlightSteps[0].forEach((highlight) => {
        // Add expected class on <pre> for reveal CSS
        codeBlock.parentNode.classList.add("code-wrapper");

        // Select lines to highlight
        spanToHighlight = [];
        if (typeof highlight.last === "number") {
          spanToHighlight = [].slice.call(
            codeBlock.querySelectorAll(
              ":scope > span:nth-of-type(n+" +
                highlight.first +
                "):nth-of-type(-n+" +
                highlight.last +
                ")"
            )
          );
        } else if (typeof highlight.first === "number") {
          spanToHighlight = [].slice.call(
            codeBlock.querySelectorAll(
              ":scope > span:nth-of-type(" + highlight.first + ")"
            )
          );
        }
        if (spanToHighlight.length) {
          // Add a class on <code> and <span> to select line to highlight
          spanToHighlight.forEach((span) =>
            span.classList.add("highlight-line")
          );
          codeBlock.classList.add("has-line-highlights");
        }
      });
    }
  }

  /**
   * Animates scrolling to the first highlighted line
   * in the given code block.
   */
  function scrollHighlightedLineIntoView(block, scrollState, skipAnimation) {
    window.cancelAnimationFrame(scrollState.animationFrameID);

    // Match the scroll position of the currently visible
    // code block
    if (scrollState.currentBlock) {
      block.scrollTop = scrollState.currentBlock.scrollTop;
    }

    // Remember the current code block so that we can match
    // its scroll position when showing/hiding fragments
    scrollState.currentBlock = block;

    const highlightBounds = getHighlightedLineBounds(block);
    let viewportHeight = block.offsetHeight;

    // Subtract padding from the viewport height
    const blockStyles = window.getComputedStyle(block);
    viewportHeight -=
      parseInt(blockStyles.paddingTop) + parseInt(blockStyles.paddingBottom);

    // Scroll position which centers all highlights
    const startTop = block.scrollTop;
    let targetTop =
      highlightBounds.top +
      (Math.min(highlightBounds.bottom - highlightBounds.top, viewportHeight) -
        viewportHeight) /
        2;

    // Make sure the scroll target is within bounds
    targetTop = Math.max(
      Math.min(targetTop, block.scrollHeight - viewportHeight),
      0
    );

    if (skipAnimation === true || startTop === targetTop) {
      block.scrollTop = targetTop;
    } else {
      // Don't attempt to scroll if there is no overflow
      if (block.scrollHeight <= viewportHeight) return;

      let time = 0;

      const animate = function () {
        time = Math.min(time + 0.02, 1);

        // Update our eased scroll position
        block.scrollTop =
          startTop + (targetTop - startTop) * easeInOutQuart(time);

        // Keep animating unless we've reached the end
        if (time < 1) {
          scrollState.animationFrameID = requestAnimationFrame(animate);
        }
      };

      animate();
    }
  }

  function getHighlightedLineBounds(block) {
    const highlightedLines = block.querySelectorAll(".highlight-line");
    if (highlightedLines.length === 0) {
      return { top: 0, bottom: 0 };
    } else {
      const firstHighlight = highlightedLines[0];
      const lastHighlight = highlightedLines[highlightedLines.length - 1];

      return {
        top: firstHighlight.offsetTop,
        bottom: lastHighlight.offsetTop + lastHighlight.offsetHeight,
      };
    }
  }

  /**
   * The easing function used when scrolling.
   */
  function easeInOutQuart(t) {
    // easeInOutQuart
    return t < 0.5 ? 8 * t * t * t * t : 1 - 8 * --t * t * t * t;
  }

  function splitLineNumbers(lineNumbersAttr) {
    // remove space
    lineNumbersAttr = lineNumbersAttr.replace("/s/g", "");
    // seperate steps (for fragment)
    lineNumbersAttr = lineNumbersAttr.split(delimiters.step);

    // for each step, calculate first and last line, if any
    return lineNumbersAttr.map((highlights) => {
      // detect lines
      const lines = highlights.split(delimiters.line);
      return lines.map((range) => {
        if (/^[\d-]+$/.test(range)) {
          range = range.split(delimiters.lineRange);
          const firstLine = parseInt(range[0], 10);
          const lastLine = range[1] ? parseInt(range[1], 10) : undefined;
          return {
            first: firstLine,
            last: lastLine,
          };
        } else {
          return {};
        }
      });
    });
  }

  function joinLineNumbers(splittedLineNumbers) {
    return splittedLineNumbers
      .map(function (highlights) {
        return highlights
          .map(function (highlight) {
            // Line range
            if (typeof highlight.last === "number") {
              return highlight.first + delimiters.lineRange + highlight.last;
            }
            // Single line
            else if (typeof highlight.first === "number") {
              return highlight.first;
            }
            // All lines
            else {
              return "";
            }
          })
          .join(delimiters.line);
      })
      .join(delimiters.step);
  }

  return {
    id: "quarto-line-highlight",
    init: function (deck) {
      initQuartoLineHighlight(deck);

      // If we're printing to PDF, scroll the code highlights of
      // all blocks in the deck into view at once
      deck.on("pdf-ready", function () {
        [].slice
          .call(
            deck
              .getRevealElement()
              .querySelectorAll(
                "pre code[data-code-line-numbers].current-fragment"
              )
          )
          .forEach(function (block) {
            scrollHighlightedLineIntoView(block, {}, true);
          });
      });
    },
  };
};
