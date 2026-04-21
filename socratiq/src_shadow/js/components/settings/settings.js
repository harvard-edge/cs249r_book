import {  showPopover } from "../../libs/utils/utils";
import { DIFFICULTY_LEVELS } from "../../../configs/client.config.js";
import { DropdownHandler } from './dropdown-handler.js';
import { enableTooltip } from '../tooltip/tooltip.js';


export class SettingsManager {
  constructor(shadowEle, isNoSlider=false) {
    this.allShadow = shadowEle;
    this.isNoSlider = isNoSlider;
    this.isInitializing = true;

    // Only initialize full settings if isNoSlider is false
    if (!isNoSlider) {
      this.shadowEle = shadowEle.querySelector("#modal1");
      this.settings = {
        sliderValue: 1,
        selectedDropdownValue: "",
        checkboxes: {},
        modalDisplayed: false,
        pushContent: false,
        customAPI: {
          provider: "",
          endpoint: "",
          model: "",
          apiKey: "",
          saveLocally: false,
          enabled: false
        }
      };
      this.loadSettings();
      this.initializeMenuBehavior();
    }
    this.isInitializing = false;
  }

  // Initialize settings from the UI components
  init() {
    if (!this.isNoSlider) {
      // Set up the slider
      const slider = this.shadowEle.querySelector("#understanding-slider");
      if (slider) {
        slider.addEventListener("input", (event) => {
          this.updateSlider(event.target.value);
        });
        this.updateSlider(slider.value);
      }

      // Set up checkboxes
      const checkboxes = this.shadowEle.querySelectorAll(
        'input[type="checkbox"]'
      );

      checkboxes.forEach((checkbox) => {
        checkbox.addEventListener("change", (event) => {
          this.updateCheckbox(checkbox.id, event.target.checked);
        });
        this.updateCheckbox(checkbox.id, checkbox.checked);
      });

      // Initialize difficulty dropdowns
      this.initializeDifficultyDropdown();

      // Initialize difficulty dropdown toggle
      this.initializeDifficultyDropdownToggle();

      // Initialize custom API settings
      this.initializeCustomAPISettings();

      this.dispatchAndUpdate("init", true);
      this.dispatchAndUpdate_settings("init", true);
    } else {
      // When isNoSlider is true, only initialize dropdowns
      const difficultyDropdowns = this.allShadow.querySelectorAll('.difficulty-dropdown');
      difficultyDropdowns.forEach(dropdown => {
        this.initializeSingleDropdown(dropdown, true);
      });
    }
  }

  // Dispatch highlight-menu event
  dispatchHighlightMenuEvent(isEnabled) {
    const event = new CustomEvent("highlight-menu-starts", {
      detail: { enabled: isEnabled },
    });
    window.dispatchEvent(event);
  }

  // Helper function to dispatch custom events and call alert
  dispatchAndUpdate(key, value) {
    // Dispatch custom event
    const event = new CustomEvent("aiActionCompleted", {
      detail: { type: "system_prompt", text: this.generateSystemPrompt() },
    });
    window.dispatchEvent(event);
    // if (key !== "init")
    //   alert(this.shadowEle, "Option " + key + " with value " + value + " is updated", "success");
  }

  dispatchAndUpdate_settings(key, value) {
    // Dispatch custom event
    const event = new CustomEvent("aiActionCompleted", {
      detail: {
        settings: this.generateSettings(), // the original query object used for the request
        type: "settings", // the type of AI action
      },
    });
    window.dispatchEvent(event);

    if (key !== "init") {
      let displayValue = value;
      if (typeof value === 'object' && value !== null) {
        if (key === "customAPI" && value.provider) {
          displayValue = `${value.provider} API`;
        } else {
          displayValue = JSON.stringify(value);
        }
      }
      showPopover(this.shadowEle, "setting " + key + " with value " + displayValue + " is updated", "success");
    }
  }

  // Update slider value in the settings
  updateSlider(value) {
    this.settings.sliderValue = parseInt(value, 10);
    this.saveSettings();
    
    window.current_difficulty = this.settings.sliderValue;
    
    // Only dispatch if not initializing
    if (!this.isInitializing) {
      const event = new CustomEvent("aiActionCompleted", {
        detail: { 
          type: "system_prompt", 
          text: DIFFICULTY_LEVELS[this.settings.sliderValue]
        },
      });
      window.dispatchEvent(event);
    }

    // Update all difficulty dropdowns
    const difficultyLevels = ['🚲 Beginner', '🚗 Intermediate', '🚁 Advanced', '🛸 Bloom\'s Taxonomy'];
    const currentLevelElements = this.allShadow.querySelectorAll('.current-difficulty-level');
    currentLevelElements.forEach(element => {
      element.textContent = difficultyLevels[value];
    });

    // Update the visual highlighting (existing code)
    const difficultyLevelElements = this.shadowEle.querySelectorAll('.difficulty-level');
    difficultyLevelElements.forEach((level, index) => {
      if (index === this.settings.sliderValue) {
        level.classList.add('bg-blue-50', 'dark:bg-zinc-700', 'p-2', 'rounded');
      } else {
        level.classList.remove('bg-blue-50', 'dark:bg-zinc-700', 'p-2', 'rounded');
      }
    });
  }

  // Update dropdown value in the settings
  updateDropdown(value) {
    this.settings.selectedDropdownValue = value;
    // this.saveSettings(); // Save settings to local storage
    // this.dispatchAndUpdate_settings(
    //   "selectedDropdownValue",
    //   this.settings.selectedDropdownValue
    // );
  }

  // Update checkbox status in the settings
  updateCheckbox(id, isChecked) {
    this.settings.checkboxes[id] = isChecked;
    this.saveSettings(); // Save settings to local storage

    if (id === "show-answers") {
      this.dispatchAndUpdate(id, isChecked);
    } else {
      this.dispatchAndUpdate_settings(id, isChecked);
    }
  }

  generateSystemPrompt() {
    const understandingLevels = [
      "Beginner: Focus on foundational concepts, definitions, and straightforward applications in machine learning systems, suitable for learners with little to no prior knowledge.",
      "Intermediate: Emphasize problem-solving, system design, and practical implementations, targeting learners with a basic understanding of machine learning principles.",
      "Advanced: Challenge learners to analyze, innovate, and optimize complex machine learning systems, requiring deep expertise and a holistic grasp of advanced techniques.",
      "requesting Bloom's Taxonomy: You are an expert ML teacher using Bloom's Taxonomy: Create responses that progress through Bloom's levels: remember, understand, apply, analyze, evaluate, and create." //Bloom's Taxonomy: Bloom's Taxonomy is an educational framework ranking cognitive skills from basic recall to complex evaluation. https://en.wikipedia.org/wiki/Bloom%27s_taxonomy"
    ];

    const understanding = understandingLevels[this.settings.sliderValue] || "unknown level";
    
    const difficultyLevels = this.shadowEle.querySelectorAll('.difficulty-level');
    difficultyLevels.forEach((level, index) => {
      if (index === this.settings.sliderValue) {
        level.classList.add('bg-blue-50', 'dark:bg-zinc-700', 'p-2', 'rounded');
      } else {
        level.classList.remove('bg-blue-50', 'dark:bg-zinc-700', 'p-2', 'rounded');
      }
    });

    const showAnswers = this.settings.checkboxes["show-answers"];
    const useBlooms = this.settings.checkboxes["Apply-blooms-taxonomy"];

    const answersDescription = '';
    showPopover(this.shadowEle, "Question difficulty level: " + understandingLevels[this.settings.sliderValue].split(':')[0], "success");
    
    return `Tailor your response for a: ${understanding}. ${answersDescription} ${useBlooms ? 'Apply Bloom\'s Taxonomy in your response structure.' : ''}`;
  }

  generateSettings() {
    // Prepare the dropdown value description
    const llm_model = this.settings.selectedDropdownValue;

    // Check the checkbox for showing answers
    const show_progress = this.settings.checkboxes["show-chain-of-thought"];

    return { 
      llm_model: llm_model, 
      show_progress: show_progress,
      customAPI: this.settings.customAPI
    };
  }

  // Save settings to local storage
  saveSettings() {
    localStorage.setItem("userSettings", JSON.stringify(this.settings));
  }

  // Load settings from local storage
  loadSettings() {
    const savedSettings = localStorage.getItem("userSettings");
    if (savedSettings) {
      this.settings = JSON.parse(savedSettings);
    }
    
    window.current_difficulty = this.settings.sliderValue;
    
    const slider = this.shadowEle.querySelector("#understanding-slider");
    if (slider) {
      slider.value = this.settings.sliderValue;
      // Remove the dispatch from here since updateSlider will handle it
      
      // Update the visual highlighting for initial load
      const difficultyLevels = this.shadowEle.querySelectorAll('.difficulty-level');
      difficultyLevels.forEach((level, index) => {
        if (index === this.settings.sliderValue) {
          level.classList.add('bg-blue-50', 'dark:bg-zinc-700', 'p-2', 'rounded');
        } else {
          level.classList.remove('bg-blue-50', 'dark:bg-zinc-700', 'p-2', 'rounded');
        }
      });
    }
  }

  // Add these methods to your SettingsManager class
  initializeDifficultyDropdown() {
    const difficultyDropdowns = this.allShadow.querySelectorAll('.difficulty-dropdown');
    difficultyDropdowns.forEach(dropdown => {
    });
    // const dropdownHandler = new DropdownHandler(this.allShadow);
    
    difficultyDropdowns.forEach(dropdown => {
      if (!dropdown.dataset.initialized) {
        // Set initial difficulty level
        const currentLevel = dropdown.querySelector('.current-difficulty-level');
        const difficultyLevels = ['🚲 Beginner', '🚗 Intermediate', '🚁 Advanced', '🛸 Bloom\'s Taxonomy'];
        if (currentLevel) {
          // Remove emoji for display
          currentLevel.textContent = difficultyLevels[this.settings.sliderValue].slice(2).trim();
        }

        // Add click handler for the dropdown button
        const button = dropdown.querySelector('button');
        const options = dropdown.querySelector('.difficulty-options');

        if (button) {
          button.addEventListener('click', (e) => {
            e.stopPropagation();
            options.classList.toggle('hidden'); // Toggle visibility
          });
        }

        // Mark this dropdown as initialized
        dropdown.dataset.initialized = true;

        // Close dropdown when clicking outside
        document.addEventListener('click', () => {
          options.classList.add('hidden');
        });
      }
    });
  }

  initializeSingleDropdown(dropdown, isRedo=false) {
    const dropdownHandler = new DropdownHandler(this.allShadow);
    // enableTooltip(dropdown, "Select a difficulty level to redo the AI response", this.allShadow);

    const currentLevel = dropdown.querySelector('.current-difficulty-level');
    const difficultyLevels = ['🚲 Beginner', '🚗 Intermediate', '🚁 Advanced', '🛸 Bloom\'s Taxonomy'];
    
    // Only set initial text if not in noSlider mode
    if (currentLevel && !isRedo && !this.isNoSlider) {
      currentLevel.textContent = difficultyLevels[this.settings.sliderValue].slice(2).trim();
    }

    const button = dropdown.querySelector('button');
    enableTooltip(button, "Redo this AI response with this a new learner understanding level", this.allShadow);
    const options = dropdown.querySelector('.difficulty-options');

    if (button) {
      button.addEventListener('click', (e) => {
        e.stopPropagation();
        options.classList.toggle('hidden'); // Toggle visibility
      });
    }

    const optionElements = dropdown.querySelectorAll('.difficulty-option');
    optionElements.forEach((option, index) => {
      option.addEventListener('click', () => {
        // this.updateSlider(index);
        currentLevel.textContent = difficultyLevels[index].slice(2).trim();
        options.classList.add('hidden'); // Hide options after selection
        // Get the parent ai-message element
        const aiMessageComponent = dropdown.closest('.ai-message-chat');



        if (!aiMessageComponent) {
          console.error('Could not find parent AI message component');
          return;
        }


        // Create and inject progress element
        const progressElement = document.createElement('div');
        progressElement.id = 'progress';
        
        // Find the content container within aiMessageComponent
        const contentContainer = aiMessageComponent.querySelector('.markdown-preview-container');

        const shareButtonContainer = aiMessageComponent.querySelector('.copy-paste-share-btn-container');
        if (shareButtonContainer) {
          shareButtonContainer.remove();
        }



        if (contentContainer) {
          // Insert progress element before the content container
          contentContainer.parentNode.insertBefore(progressElement, contentContainer);
          
          // Find markdown preview container and add id
         
          contentContainer.id = 'markdown-preview';
        } else {
          console.error('Could not find content container in AI message');
          return;
        }



        dropdownHandler.handleDifficultySelection(dropdown, index, aiMessageComponent);
      });
    });

    dropdown.dataset.initialized = true;

    document.addEventListener('click', (e) => {
      if (!dropdown.contains(e.target)) {
        options.classList.add('hidden'); // Hide options when clicking outside
      }
    });
  }

  // Add this method to handle the push content toggle
  updatePushContent(isPushed) {
    this.settings.pushContent = isPushed;
    this.saveSettings();
    
    // If menu is currently open, update the margin immediately
    const menu = this.allShadow.querySelector('#text-selection-menu');
    const isMenuOpen = menu && menu.classList.contains('translate-x-0');
    
    if (isMenuOpen) {
      document.body.style.marginRight = isPushed ? '400px' : '0';
      document.body.style.transition = 'margin-right 0.3s ease-in-out';
    }
  }

  // Add as a class method
  initializeMenuBehavior() {
    const body = document.querySelector('#mybody');
    const menu = this.allShadow.querySelector('#text-selection-menu');
    const toggle = this.shadowEle.querySelector('#push-content-toggle');
    
    if (!toggle) {
      console.warn('Push content toggle not found');
      return;
    }
    
    // Load initial state from settings
    const isPushEnabled = this.settings.pushContent || false;
    
    // Set initial states
    toggle.checked = isPushEnabled;
    document.body.classList.toggle('push-content-enabled', isPushEnabled);
    document.body.classList.toggle('push-content-disabled', !isPushEnabled);
    
    // Handle toggle changes
    toggle.addEventListener('change', (e) => {
      const isPushed = e.target.checked;
      this.updatePushContent(isPushed);
    });
    
    // Listen for menu display mode changes
    window.addEventListener('menuDisplayModeChanged', (e) => {
      const isPushed = e.detail.pushContent;
      document.body.classList.toggle('push-content-enabled', isPushed);
      document.body.classList.toggle('push-content-disabled', !isPushed);
    });
    
    // Update classes when menu opens/closes
    const menuToggle = this.allShadow.querySelector('#menu-toggle');
    if (menuToggle) {
      menuToggle.addEventListener('change', (e) => {
        document.body.classList.toggle('menu-open', e.target.checked);
      });
    }
  }

  // Initialize difficulty dropdown toggle
  initializeDifficultyDropdownToggle() {
    const dropdownToggle = this.shadowEle.querySelector('#difficulty-dropdown-toggle');
    const difficultyContent = this.shadowEle.querySelector('#difficulty-content');
    
    if (!dropdownToggle || !difficultyContent) {
      console.warn('Difficulty dropdown elements not found');
      return;
    }
    
    // Set initial state - collapsed by default
    difficultyContent.classList.add('difficulty-content-collapsed');
    difficultyContent.classList.remove('difficulty-content-expanded');
    
    // Handle toggle click
    dropdownToggle.addEventListener('click', () => {
      const isCollapsed = difficultyContent.classList.contains('difficulty-content-collapsed');
      
      if (isCollapsed) {
        // Expand
        difficultyContent.classList.remove('difficulty-content-collapsed');
        difficultyContent.classList.add('difficulty-content-expanded');
        dropdownToggle.classList.add('rotated');
      } else {
        // Collapse
        difficultyContent.classList.remove('difficulty-content-expanded');
        difficultyContent.classList.add('difficulty-content-collapsed');
        dropdownToggle.classList.remove('rotated');
      }
    });
  }

  // Initialize custom API settings
  initializeCustomAPISettings() {
    const customApiToggle = this.shadowEle.querySelector("#custom-api-toggle");
    const customApiConfig = this.shadowEle.querySelector("#custom-api-config");
    const defaultApiInfo = this.shadowEle.querySelector("#default-api-info");
    const providerSelect = this.shadowEle.querySelector("#ai-provider");
    const modelInput = this.shadowEle.querySelector("#api-model");
    const endpointInput = this.shadowEle.querySelector("#api-endpoint");
    const apiKeyInput = this.shadowEle.querySelector("#api-key");
    const saveLocallyCheckbox = this.shadowEle.querySelector("#save-locally");
    const resetButton = this.shadowEle.querySelector("#reset-custom-api");
    
    if (!customApiToggle || !customApiConfig || !defaultApiInfo || !providerSelect || !modelInput || !endpointInput || !apiKeyInput || !saveLocallyCheckbox || !resetButton) {
      console.warn('Custom API elements not found in settings modal');
      return;
    }
    
    // Load existing values
    if (this.settings.customAPI) {
      customApiToggle.checked = this.settings.customAPI.enabled || false;
      providerSelect.value = this.settings.customAPI.provider || "";
      modelInput.value = this.settings.customAPI.model || "";
      endpointInput.value = this.settings.customAPI.endpoint || "";
      apiKeyInput.value = this.settings.customAPI.apiKey || "";
      saveLocallyCheckbox.checked = this.settings.customAPI.saveLocally || false;
    }
    
    // Update UI based on toggle state
    this.updateCustomAPIUI(customApiToggle.checked, customApiConfig, defaultApiInfo);
    
    // Add change listeners
    customApiToggle.addEventListener("change", (e) => {
      this.updateCustomAPIUI(e.target.checked, customApiConfig, defaultApiInfo);
      this.updateCustomAPI(providerSelect.value, endpointInput.value, modelInput.value, apiKeyInput.value, saveLocallyCheckbox.checked, e.target.checked);
    });
    
    providerSelect.addEventListener("change", (e) => {
      this.handleProviderChange(e.target.value, modelInput, endpointInput);
      this.updateCustomAPI(e.target.value, endpointInput.value, modelInput.value, apiKeyInput.value, saveLocallyCheckbox.checked, customApiToggle.checked);
    });
    
    modelInput.addEventListener("input", (e) => {
      this.updateCustomAPI(providerSelect.value, endpointInput.value, e.target.value, apiKeyInput.value, saveLocallyCheckbox.checked, customApiToggle.checked);
    });
    
    endpointInput.addEventListener("input", (e) => {
      this.updateCustomAPI(providerSelect.value, e.target.value, modelInput.value, apiKeyInput.value, saveLocallyCheckbox.checked, customApiToggle.checked);
    });
    
    apiKeyInput.addEventListener("input", (e) => {
      this.updateCustomAPI(providerSelect.value, endpointInput.value, modelInput.value, e.target.value, saveLocallyCheckbox.checked, customApiToggle.checked);
    });
    
    saveLocallyCheckbox.addEventListener("change", (e) => {
      this.updateCustomAPI(providerSelect.value, endpointInput.value, modelInput.value, apiKeyInput.value, e.target.checked, customApiToggle.checked);
    });
    
    resetButton.addEventListener("click", (e) => {
      this.resetCustomAPISettings(providerSelect, modelInput, endpointInput, apiKeyInput, saveLocallyCheckbox, customApiToggle);
    });
  }

  // Handle provider change and auto-fill endpoint and model
  handleProviderChange(provider, modelInput, endpointInput) {
    const providerDefaults = {
      'google-gemini': {
        endpoint: 'https://generativelanguage.googleapis.com/v1beta',
        model: 'gemini-2.5-flash'
      },
      'ollama': {
        endpoint: 'http://localhost:11434/api/generate',
        model: 'gemma3:270m'
      },
      'openai': {
        endpoint: 'https://api.openai.com/v1/chat/completions',
        model: 'gpt-3.5-turbo'
      },
      'open-router': {
        endpoint: 'https://openrouter.ai/api/v1/chat/completions',
        model: 'meta-llama/llama-4-scout:free'
      },
      'groq': {
        endpoint: 'https://api.groq.com/openai/v1/chat/completions',
        model: 'llama-3.1-8b-instant'
      },
      'anthropic': {
        endpoint: 'https://api.anthropic.com/v1/messages',
        model: 'claude-3-sonnet-20240229'
      }
    };

    if (provider && providerDefaults[provider]) {
      const defaults = providerDefaults[provider];
      
      // Always auto-fill when provider changes
      endpointInput.value = defaults.endpoint;
      modelInput.value = defaults.model;
      
      console.log(`[AUTO_FILL] Provider changed to ${provider}:`, defaults);
    } else if (provider === '') {
      // Clear fields when no provider is selected
      endpointInput.value = '';
      modelInput.value = '';
    }
  }

  // Update custom API UI based on toggle state
  updateCustomAPIUI(enabled, customApiConfig, defaultApiInfo) {
    if (enabled) {
      customApiConfig.classList.remove('hidden');
      defaultApiInfo.classList.add('hidden');
    } else {
      customApiConfig.classList.add('hidden');
      defaultApiInfo.classList.remove('hidden');
    }
  }

  // Reset custom API settings to defaults
  resetCustomAPISettings(providerSelect, modelInput, endpointInput, apiKeyInput, saveLocallyCheckbox, customApiToggle) {
    // Reset all fields to empty/default values
    providerSelect.value = "";
    modelInput.value = "";
    endpointInput.value = "";
    apiKeyInput.value = "";
    saveLocallyCheckbox.checked = false;
    customApiToggle.checked = false;
    
    // Update settings
    this.settings.customAPI = {
      provider: "",
      endpoint: "",
      model: "",
      apiKey: "",
      saveLocally: false,
      enabled: false
    };
    
    this.saveSettings();
    this.dispatchAndUpdate_settings("customAPI", this.settings.customAPI);
    
    // Update UI
    this.updateCustomAPIUI(false, this.shadowEle.querySelector("#custom-api-config"), this.shadowEle.querySelector("#default-api-info"));
    
    showPopover(this.shadowEle, "Custom API settings reset to defaults", "success");
  }

  // Update custom API configuration
  updateCustomAPI(provider, endpoint, model, apiKey, saveLocally, enabled) {
    this.settings.customAPI = {
      provider: provider || "",
      endpoint: endpoint || "",
      model: model || "",
      apiKey: apiKey || "",
      saveLocally: saveLocally || false,
      enabled: enabled || false
    };
    
    this.saveSettings();
    this.dispatchAndUpdate_settings("customAPI", this.settings.customAPI);
    
    // Show success message
    if (enabled && provider && endpoint) {
      showPopover(this.shadowEle, `Custom API configured: ${provider}`, "success");
    } else if (!enabled) {
      showPopover(this.shadowEle, "Using SocratiQ default AI providers", "success");
    }
  }
}

export function setupModal(shadowEle) {
  const modal = shadowEle.querySelector("#modal1");
  const settingsBtn = shadowEle.querySelector("#settings-btn");
  enableTooltip(settingsBtn, "Open settings", shadowEle);
  const closeBtn = modal.querySelector("#close-btn");

  // Create overlay inside the shadow DOM instead of document body
  const overlay = document.createElement("div");
  overlay.classList.add("settings-modal-overlay");
  overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    width: 100vw;
    height: 100vh;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: 99999; /* Increased z-index to be higher than the menu */
    display: none;
    pointer-events: auto; /* Ensure clicks are captured */
  `;

  // Style the modal to appear above the overlay
  modal.style.cssText = `
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 100000; /* Higher than overlay */
    display: none;
    max-height: 90vh;
    overflow-y: auto;
    border-radius: 0.5rem;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    pointer-events: auto;
  `;

  // Append overlay to shadow root instead of document body
  shadowEle.appendChild(overlay);

  const toggleModal = () => {
    const isDisplayed = modal.style.display === "block";
    modal.style.display = isDisplayed ? "none" : "block";
    overlay.style.display = isDisplayed ? "none" : "block";
    
    if (!isDisplayed) {
      document.body.style.overflow = "hidden";
      // Ensure the overlay is above all shadow DOM content
      overlay.style.position = "fixed";
      overlay.style.zIndex = "99999";
    } else {
      document.body.style.overflow = "";
    }
  };

  const closeModal = () => {
    modal.style.display = "none";
    overlay.style.display = "none";
    document.body.style.overflow = ""; // Restore body scrolling
  };

  // Event listeners
  settingsBtn.addEventListener("click", toggleModal);
  closeBtn.addEventListener("click", closeModal);
  overlay.addEventListener("click", closeModal);
  
  // Prevent modal from closing when clicking inside it
  modal.addEventListener("click", (event) => {
    event.stopPropagation();
  });

  // Close modal with Escape key
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && modal.style.display === "block") {
      closeModal();
    }
  });

  // Clean up function to remove overlay when needed
  return () => {
    if (overlay && overlay.parentNode) {
      overlay.parentNode.removeChild(overlay);
    }
  };
}
