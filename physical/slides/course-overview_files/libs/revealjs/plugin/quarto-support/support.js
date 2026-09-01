// catch all plugin for various quarto features
window.QuartoSupport = function () {
  function isPrintView() {
    return /print-pdf/gi.test(window.location.search) || /view=print/gi.test(window.location.search);
  }

  // helper for theme toggling
  function toggleBackgroundTheme(el, onDarkBackground, onLightBackground) {
    if (onDarkBackground) {
      el.classList.add('has-dark-background')
    } else {
      el.classList.remove('has-dark-background')
    }
    if (onLightBackground) {
      el.classList.add('has-light-background')
    } else {
      el.classList.remove('has-light-background')
    }
  }

  // implement controlsAudo
  function controlsAuto(deck) {
    const config = deck.getConfig();
    if (config.controlsAuto === true) {
      const iframe = window.location !== window.parent.location;
      const localhost =
        window.location.hostname === "localhost" ||
        window.location.hostname === "127.0.0.1";
      deck.configure({
        controls:
          (iframe && !localhost) ||
          (deck.hasVerticalSlides() && config.navigationMode !== "linear"),
      });
    }
  }

  // helper to provide event handlers for all links in a container
  function handleLinkClickEvents(deck, container) {
    Array.from(container.querySelectorAll("a")).forEach((el) => {
      const url = el.getAttribute("href");
      if (/^(http|www)/gi.test(url)) {
        el.addEventListener(
          "click",
          (ev) => {
            const fullscreen = !!window.document.fullscreen;
            const dataPreviewLink = el.getAttribute("data-preview-link");

            // if there is a local specifcation then use that
            if (dataPreviewLink) {
              if (
                dataPreviewLink === "true" ||
                (dataPreviewLink === "auto" && fullscreen)
              ) {
                ev.preventDefault();
                deck.showPreview(url);
                return false;
              }
            } else {
              const previewLinks = !!deck.getConfig().previewLinks;
              const previewLinksAuto =
                deck.getConfig().previewLinksAuto === true;
              if (previewLinks == true || (previewLinksAuto && fullscreen)) {
                ev.preventDefault();
                deck.showPreview(url);
                return false;
              }
            }

            // if the deck is in an iframe we want to open it externally
            // (don't do this when in vscode though as it has its own
            // handler for opening links externally that will be play)
            const iframe = window.location !== window.parent.location;
            if (
              iframe &&
              !window.location.search.includes("quartoPreviewReqId=")
            ) {
              ev.preventDefault();
              ev.stopImmediatePropagation();
              window.open(url, "_blank");
              return false;
            }

            // if the user has set data-preview-link to "auto" we need to handle the event
            // (because reveal will interpret "auto" as true)
            if (dataPreviewLink === "auto") {
              ev.preventDefault();
              ev.stopImmediatePropagation();
              const target =
                el.getAttribute("target") ||
                (ev.ctrlKey || ev.metaKey ? "_blank" : "");
              if (target) {
                window.open(url, target);
              } else {
                window.location.href = url;
              }
              return false;
            }
          },
          false
        );
      }
    });
  }

  // implement previewLinksAuto
  function previewLinksAuto(deck) {
    handleLinkClickEvents(deck, deck.getRevealElement());
  }

  // apply styles
  function applyGlobalStyles(deck) {
    if (deck.getConfig()["smaller"] === true) {
      const revealParent = deck.getRevealElement();
      revealParent.classList.add("smaller");
    }
  }

  // add logo image
  function addLogoImage(deck) {
    const revealParent = deck.getRevealElement();
    const logoImg = document.querySelector(".slide-logo");
    if (logoImg) {
      revealParent.appendChild(logoImg);
      revealParent.classList.add("has-logo");
    }
  }

  // tweak slide-number element
  function tweakSlideNumber(deck) {
    deck.on("slidechanged", function (ev) {
      // No slide number in scroll view
      if (deck.isScrollView()) { return }
      const revealParent = deck.getRevealElement();
      const slideNumberEl = revealParent.querySelector(".slide-number");
      const slideBackground = Reveal.getSlideBackground(ev.currentSlide);
      const onDarkBackground = slideBackground.classList.contains('has-dark-background')
      const onLightBackground = slideBackground.classList.contains('has-light-background')
      toggleBackgroundTheme(slideNumberEl, onDarkBackground, onLightBackground);
    })
  }

  // add footer text
  function addFooter(deck) {
    const revealParent = deck.getRevealElement();
    const defaultFooterDiv = document.querySelector(".footer-default");
    // Set per slide footer if any defined, 
    // or show default unless data-footer="false" for no footer on this slide
    const setSlideFooter = (ev, defaultFooterDiv) => {
      const currentSlideFooter = ev.currentSlide.querySelector(".footer");
      const onDarkBackground = deck.getSlideBackground(ev.currentSlide).classList.contains('has-dark-background')
      const onLightBackground = deck.getSlideBackground(ev.currentSlide).classList.contains('has-light-background')
      if (currentSlideFooter) {
        defaultFooterDiv.style.display = "none";
        const slideFooter = currentSlideFooter.cloneNode(true);
        handleLinkClickEvents(deck, slideFooter);
        deck.getRevealElement().appendChild(slideFooter);
        toggleBackgroundTheme(slideFooter, onDarkBackground, onLightBackground)
      } else if (ev.currentSlide.getAttribute("data-footer") === "false") {
        defaultFooterDiv.style.display = "none";
      } else {
        defaultFooterDiv.style.display = "block";
        toggleBackgroundTheme(defaultFooterDiv, onDarkBackground, onLightBackground)
      }
    }
    if (defaultFooterDiv) {
      // move default footnote to the div.reveal element
      revealParent.appendChild(defaultFooterDiv);
      handleLinkClickEvents(deck, defaultFooterDiv);

      if (!isPrintView()) {
        // Ready even is needed so that footer customization applies on first loaded slide
        deck.on('ready', (ev) => {
          // Set footer (custom, default or none)
          setSlideFooter(ev, defaultFooterDiv)
        });
        // Any new navigated new slide will get the custom footnote check
        deck.on("slidechanged", function (ev) {
          // Remove presentation footer defined by previous slide
          const prevSlideFooter = document.querySelector(
            ".reveal > .footer:not(.footer-default)"
          );
          if (prevSlideFooter) {
            prevSlideFooter.remove();
          }
          // Set new one (custom, default or none)
          setSlideFooter(ev, defaultFooterDiv)
        });
      }
    }
  }

  // add chalkboard buttons
  function addChalkboardButtons(deck) {
    const chalkboard = deck.getPlugin("RevealChalkboard");
    if (chalkboard && !isPrintView()) {
      const revealParent = deck.getRevealElement();
      const chalkboardDiv = document.createElement("div");
      chalkboardDiv.classList.add("slide-chalkboard-buttons");
      if (document.querySelector(".slide-menu-button")) {
        chalkboardDiv.classList.add("slide-menu-offset");
      }
      // add buttons
      const buttons = [
        {
          icon: "easel2",
          title: "Toggle Chalkboard (b)",
          onclick: chalkboard.toggleChalkboard,
        },
        {
          icon: "brush",
          title: "Toggle Notes Canvas (c)",
          onclick: chalkboard.toggleNotesCanvas,
        },
      ];
      buttons.forEach(function (button) {
        const span = document.createElement("span");
        span.title = button.title;
        const icon = document.createElement("i");
        icon.classList.add("fas");
        icon.classList.add("fa-" + button.icon);
        span.appendChild(icon);
        span.onclick = function (event) {
          event.preventDefault();
          button.onclick();
        };
        chalkboardDiv.appendChild(span);
      });
      revealParent.appendChild(chalkboardDiv);
      const config = deck.getConfig();
      if (!config.chalkboard.buttons) {
        chalkboardDiv.classList.add("hidden");
      }

      // show and hide chalkboard buttons on slidechange
      deck.on("slidechanged", function (ev) {
        const config = deck.getConfig();
        let buttons = !!config.chalkboard.buttons;
        const slideButtons = ev.currentSlide.getAttribute(
          "data-chalkboard-buttons"
        );
        if (slideButtons) {
          if (slideButtons === "true" || slideButtons === "1") {
            buttons = true;
          } else if (slideButtons === "false" || slideButtons === "0") {
            buttons = false;
          }
        }
        if (buttons) {
          chalkboardDiv.classList.remove("hidden");
        } else {
          chalkboardDiv.classList.add("hidden");
        }
      });
    }
  }

  function handleTabbyClicks() {
    const tabs = document.querySelectorAll(".panel-tabset-tabby > li > a");
    for (let i = 0; i < tabs.length; i++) {
      const tab = tabs[i];
      tab.onclick = function (ev) {
        ev.preventDefault();
        ev.stopPropagation();
        return false;
      };
    }
  }

  function fixupForPrint(deck) {
    if (isPrintView()) {
      const slides = deck.getSlides();
      slides.forEach(function (slide) {
        slide.removeAttribute("data-auto-animate");
      });
      window.document.querySelectorAll(".hljs").forEach(function (el) {
        el.classList.remove("hljs");
      });
      window.document.querySelectorAll(".hljs-ln-code").forEach(function (el) {
        el.classList.remove("hljs-ln-code");
      });
    }
  }

  // dispatch for htmlwidgets
  // they use slideenter event to trigger resize
  const fireSlideEnter = () => {
    const event = window.document.createEvent("Event");
    event.initEvent("slideenter", true, true);
    window.document.dispatchEvent(event);
  };

  // dispatch for shiny
  // they use BS shown and hidden events to trigger rendering
  const distpatchShinyEvents = (previous, current) => {
    if (window.jQuery) {
      if (previous) {
        window.jQuery(previous).trigger("hidden");
      }
      if (current) {
        window.jQuery(current).trigger("shown");
      }
    }
  };

  function handleSlideChanges(deck) {

    const fireSlideChanged = (previousSlide, currentSlide) => {
      fireSlideEnter();
      distpatchShinyEvents(previousSlide, currentSlide);
    };

    deck.on("slidechanged", function (event) {
      fireSlideChanged(event.previousSlide, event.currentSlide);
    });
  }

  function handleTabbyChanges() {
    const fireTabChanged = (previousTab, currentTab) => {
      fireSlideEnter()
      distpatchShinyEvents(previousTab, currentTab);
    };
    document.addEventListener("tabby", function(event) {
      fireTabChanged(event.detail.previousTab, event.detail.tab);
    }, false);
  }

  function workaroundMermaidDistance(deck) {
    if (window.document.querySelector("pre.mermaid-js")) {
      const slideCount = deck.getTotalSlides();
      deck.configure({
        mobileViewDistance: slideCount,
        viewDistance: slideCount,
      });
    }
  }

  function handleWhiteSpaceInColumns(deck) {
    for (const outerDiv of window.document.querySelectorAll("div.columns")) {
      // remove all whitespace text nodes
      // whitespace nodes cause the columns to be misaligned
      // since they have inline-block layout
      // 
      // Quarto emits no whitespace nodes, but third-party tooling
      // has bugs that can cause whitespace nodes to be emitted.
      // See https://github.com/quarto-dev/quarto-cli/issues/8382
      for (const node of outerDiv.childNodes) {
        if (node.nodeType === 3 && node.nodeValue.trim() === "") {
          outerDiv.removeChild(node);
        }
      }
    }
  }

  function cleanEmptyAutoGeneratedContent(deck) {
    const div = document.querySelector('div.quarto-auto-generated-content')
    if (div && div.textContent.trim() === '') {
      div.remove()
    }
  }

  // FIXME: Possibly remove this wrapper class when upstream trigger is fixed
  // https://github.com/hakimel/reveal.js/issues/3688
  // Currently, scrollActivationWidth needs to be unset for toggle to work
  class ScrollViewToggler {
    constructor(deck) {
      this.deck = deck;
      this.oldScrollActivationWidth = deck.getConfig()['scrollActivationWidth'];
    }
  
    toggleScrollViewWrapper() {
      if (this.deck.isScrollView() === true) {
        this.deck.configure({ scrollActivationWidth: this.oldScrollActivationWidth });
        this.deck.toggleScrollView(false);
      } else if (this.deck.isScrollView() === false) {
        this.deck.configure({ scrollActivationWidth: null });
        this.deck.toggleScrollView(true);
      }
    }
  }

  let scrollViewToggler;

  function installScollViewKeyBindings(deck) {
		var config = deck.getConfig();
		var shortcut = config.scrollViewShortcut || 'R';
		Reveal.addKeyBinding({
			keyCode: shortcut.toUpperCase().charCodeAt( 0 ),
			key: shortcut.toUpperCase(),
			description: 'Scroll View Mode'
		}, () => { scrollViewToggler.toggleScrollViewWrapper() } );
	}

  return {
    id: "quarto-support",
    init: function (deck) {
      scrollViewToggler = new ScrollViewToggler(deck);
      controlsAuto(deck);
      previewLinksAuto(deck);
      fixupForPrint(deck);
      applyGlobalStyles(deck);
      addLogoImage(deck);
      tweakSlideNumber(deck);
      addFooter(deck);
      addChalkboardButtons(deck);
      handleTabbyClicks();
      handleTabbyChanges();
      handleSlideChanges(deck);
      workaroundMermaidDistance(deck);
      handleWhiteSpaceInColumns(deck);
      installScollViewKeyBindings(deck);
      // should stay last
      cleanEmptyAutoGeneratedContent(deck);
    },
    // Export for adding in menu
    toggleScrollView: function() {
      scrollViewToggler.toggleScrollViewWrapper();
    }
  };
};
