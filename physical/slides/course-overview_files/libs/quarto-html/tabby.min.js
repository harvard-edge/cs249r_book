(function (root, factory) {
  if (typeof define === "function" && define.amd) {
    define([], function () {
      return factory(root);
    });
  } else if (typeof exports === "object") {
    module.exports = factory(root);
  } else {
    root.Tabby = factory(root);
  }
})(
  typeof global !== "undefined"
    ? global
    : typeof window !== "undefined"
    ? window
    : this,
  function (window) {
    "use strict";

    //
    // Variables
    //

    var defaults = {
      idPrefix: "tabby-toggle_",
      default: "[data-tabby-default]",
    };

    //
    // Methods
    //

    /**
     * Merge two or more objects together.
     * @param   {Object}   objects  The objects to merge together
     * @returns {Object}            Merged values of defaults and options
     */
    var extend = function () {
      var merged = {};
      Array.prototype.forEach.call(arguments, function (obj) {
        for (var key in obj) {
          if (!obj.hasOwnProperty(key)) return;
          merged[key] = obj[key];
        }
      });
      return merged;
    };

    /**
     * Emit a custom event
     * @param  {String} type    The event type
     * @param  {Node}   tab     The tab to attach the event to
     * @param  {Node}   details Details about the event
     */
    var emitEvent = function (tab, details) {
      // Create a new event
      var event;
      if (typeof window.CustomEvent === "function") {
        event = new CustomEvent("tabby", {
          bubbles: true,
          cancelable: true,
          detail: details,
        });
      } else {
        event = document.createEvent("CustomEvent");
        event.initCustomEvent("tabby", true, true, details);
      }

      // Dispatch the event
      tab.dispatchEvent(event);
    };

    var focusHandler = function (event) {
      toggle(event.target);
    };

    var getKeyboardFocusableElements = function (element) {
      return [
        ...element.querySelectorAll(
          'a[href], button, input, textarea, select, details,[tabindex]:not([tabindex="-1"])'
        ),
      ].filter(
        (el) => !el.hasAttribute("disabled") && !el.getAttribute("aria-hidden")
      );
    };

    /**
     * Remove roles and attributes from a tab and its content
     * @param  {Node}   tab      The tab
     * @param  {Node}   content  The tab content
     * @param  {Object} settings User settings and options
     */
    var destroyTab = function (tab, content, settings) {
      // Remove the generated ID
      if (tab.id.slice(0, settings.idPrefix.length) === settings.idPrefix) {
        tab.id = "";
      }

      // remove event listener
      tab.removeEventListener("focus", focusHandler, true);

      // Remove roles
      tab.removeAttribute("role");
      tab.removeAttribute("aria-controls");
      tab.removeAttribute("aria-selected");
      tab.removeAttribute("tabindex");
      tab.closest("li").removeAttribute("role");
      content.removeAttribute("role");
      content.removeAttribute("aria-labelledby");
      content.removeAttribute("hidden");
    };

    /**
     * Add the required roles and attributes to a tab and its content
     * @param  {Node}   tab      The tab
     * @param  {Node}   content  The tab content
     * @param  {Object} settings User settings and options
     */
    var setupTab = function (tab, content, settings) {
      // Give tab an ID if it doesn't already have one
      if (!tab.id) {
        tab.id = settings.idPrefix + content.id;
      }

      // Add roles
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-controls", content.id);
      tab.closest("li").setAttribute("role", "presentation");
      content.setAttribute("role", "tabpanel");
      content.setAttribute("aria-labelledby", tab.id);

      // Add selected state
      if (tab.matches(settings.default)) {
        tab.setAttribute("aria-selected", "true");
      } else {
        tab.setAttribute("aria-selected", "false");
        content.setAttribute("hidden", "hidden");
      }

      // add focus event listender
      tab.addEventListener("focus", focusHandler);
    };

    /**
     * Hide a tab and its content
     * @param  {Node} newTab The new tab that's replacing it
     */
    var hide = function (newTab) {
      // Variables
      var tabGroup = newTab.closest('[role="tablist"]');
      if (!tabGroup) return {};
      var tab = tabGroup.querySelector('[role="tab"][aria-selected="true"]');
      if (!tab) return {};
      var content = document.querySelector(tab.hash);

      // Hide the tab
      tab.setAttribute("aria-selected", "false");

      // Hide the content
      if (!content) return { previousTab: tab };
      content.setAttribute("hidden", "hidden");

      // Return the hidden tab and content
      return {
        previousTab: tab,
        previousContent: content,
      };
    };

    /**
     * Show a tab and its content
     * @param  {Node} tab      The tab
     * @param  {Node} content  The tab content
     */
    var show = function (tab, content) {
      tab.setAttribute("aria-selected", "true");
      content.removeAttribute("hidden");
      tab.focus();
    };

    /**
     * Toggle a new tab
     * @param  {Node} tab The tab to show
     */
    var toggle = function (tab) {
      // Make sure there's a tab to toggle and it's not already active
      if (!tab || tab.getAttribute("aria-selected") == "true") return;

      // Variables
      var content = document.querySelector(tab.hash);
      if (!content) return;

      // Hide active tab and content
      var details = hide(tab);

      // Show new tab and content
      show(tab, content);

      // Add event details
      details.tab = tab;
      details.content = content;

      // Emit a custom event
      emitEvent(tab, details);
    };

    /**
     * Get all of the tabs in a tablist
     * @param  {Node}   tab  A tab from the list
     * @return {Object}      The tabs and the index of the currently active one
     */
    var getTabsMap = function (tab) {
      var tabGroup = tab.closest('[role="tablist"]');
      var tabs = tabGroup ? tabGroup.querySelectorAll('[role="tab"]') : null;
      if (!tabs) return;
      return {
        tabs: tabs,
        index: Array.prototype.indexOf.call(tabs, tab),
      };
    };

    /**
     * Switch the active tab based on keyboard activity
     * @param  {Node} tab The currently active tab
     * @param  {Key}  key The key that was pressed
     */
    var switchTabs = function (tab, key) {
      // Get a map of tabs
      var map = getTabsMap(tab);
      if (!map) return;
      var length = map.tabs.length - 1;
      var index;

      // Go to previous tab
      if (["ArrowUp", "ArrowLeft", "Up", "Left"].indexOf(key) > -1) {
        index = map.index < 1 ? length : map.index - 1;
      }

      // Go to next tab
      else if (["ArrowDown", "ArrowRight", "Down", "Right"].indexOf(key) > -1) {
        index = map.index === length ? 0 : map.index + 1;
      }

      // Go to home
      else if (key === "Home") {
        index = 0;
      }

      // Go to end
      else if (key === "End") {
        index = length;
      }

      // Toggle the tab
      toggle(map.tabs[index]);
    };

    /**
     * Create the Constructor object
     */
    var Constructor = function (selector, options) {
      //
      // Variables
      //

      var publicAPIs = {};
      var settings, tabWrapper;

      //
      // Methods
      //

      publicAPIs.destroy = function () {
        // Get all tabs
        var tabs = tabWrapper.querySelectorAll("a");

        // Add roles to tabs
        Array.prototype.forEach.call(tabs, function (tab) {
          // Get the tab content
          var content = document.querySelector(tab.hash);
          if (!content) return;

          // Setup the tab
          destroyTab(tab, content, settings);
        });

        // Remove role from wrapper
        tabWrapper.removeAttribute("role");

        // Remove event listeners
        document.documentElement.removeEventListener(
          "click",
          clickHandler,
          true
        );
        tabWrapper.removeEventListener("keydown", keyHandler, true);

        // Reset variables
        settings = null;
        tabWrapper = null;
      };

      /**
       * Setup the DOM with the proper attributes
       */
      publicAPIs.setup = function () {
        // Variables
        tabWrapper = document.querySelector(selector);
        if (!tabWrapper) return;
        var tabs = tabWrapper.querySelectorAll("a");

        // Add role to wrapper
        tabWrapper.setAttribute("role", "tablist");

        // Add roles to tabs. provide dynanmic tab indexes if we are within reveal
        var contentTabindexes =
          window.document.body.classList.contains("reveal-viewport");
        var nextTabindex = 1;
        Array.prototype.forEach.call(tabs, function (tab) {
          if (contentTabindexes) {
            tab.setAttribute("tabindex", "" + nextTabindex++);
          } else {
            tab.setAttribute("tabindex", "0");
          }

          // Get the tab content
          var content = document.querySelector(tab.hash);
          if (!content) return;

          // set tab indexes for content
          if (contentTabindexes) {
            getKeyboardFocusableElements(content).forEach(function (el) {
              el.setAttribute("tabindex", "" + nextTabindex++);
            });
          }

          // Setup the tab
          setupTab(tab, content, settings);
        });
      };

      /**
       * Toggle a tab based on an ID
       * @param  {String|Node} id The tab to toggle
       */
      publicAPIs.toggle = function (id) {
        // Get the tab
        var tab = id;
        if (typeof id === "string") {
          tab = document.querySelector(
            selector + ' [role="tab"][href*="' + id + '"]'
          );
        }

        // Toggle the tab
        toggle(tab);
      };

      /**
       * Handle click events
       */
      var clickHandler = function (event) {
        // Only run on toggles
        var tab = event.target.closest(selector + ' [role="tab"]');
        if (!tab) return;

        // Prevent link behavior
        event.preventDefault();

        // Toggle the tab
        toggle(tab);
      };

      /**
       * Handle keydown events
       */
      var keyHandler = function (event) {
        // Only run if a tab is in focus
        var tab = document.activeElement;
        if (!tab.matches(selector + ' [role="tab"]')) return;

        // Only run for specific keys
        if (["Home", "End"].indexOf(event.key) < 0) return;

        // Switch tabs
        switchTabs(tab, event.key);
      };

      /**
       * Initialize the instance
       */
      var init = function () {
        // Merge user options with defaults
        settings = extend(defaults, options || {});

        // Setup the DOM
        publicAPIs.setup();

        // Add event listeners
        document.documentElement.addEventListener("click", clickHandler, true);
        tabWrapper.addEventListener("keydown", keyHandler, true);
      };

      //
      // Initialize and return the Public APIs
      //

      init();
      return publicAPIs;
    };

    //
    // Return the Constructor
    //

    return Constructor;
  }
);
