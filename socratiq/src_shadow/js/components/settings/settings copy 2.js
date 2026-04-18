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
      };
      this.loadSettings();
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

    if (key !== "init")
      showPopover(this.shadowEle, "setting " + key + " with value " + value + " is updated", "success");
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

    if (id === "Show answers") {
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

    const showAnswers = this.settings.checkboxes["Show answers"];
    const useBlooms = this.settings.checkboxes["Apply-blooms-taxonomy"];

    const answersDescription = '';
    showPopover(this.shadowEle, "Question difficulty level: " + understandingLevels[this.settings.sliderValue].split(':')[0], "success");
    
    return `Tailor your response for a: ${understanding}. ${answersDescription} ${useBlooms ? 'Apply Bloom\'s Taxonomy in your response structure.' : ''}`;
  }

  generateSettings() {
    // Prepare the dropdown value description
    const llm_model = this.settings.selectedDropdownValue;

    // Check the checkbox for showing answers
    const show_progress = this.settings.checkboxes["Show chain of thought"];

    return { llm_model: llm_model, show_progress: show_progress };
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
    const dropdownHandler = new DropdownHandler(this.allShadow);
    
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
            console.log('Dropdown button clicked');
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
    console.log('Initializing single dropdown', dropdown);
    const dropdownHandler = new DropdownHandler(this.allShadow);
    
    const currentLevel = dropdown.querySelector('.current-difficulty-level');
    const difficultyLevels = ['🚲 Beginner', '🚗 Intermediate', '🚁 Advanced', '🛸 Bloom\'s Taxonomy'];
    
    // Only set initial text if not in noSlider mode
    if (currentLevel && !isRedo && !this.isNoSlider) {
      currentLevel.textContent = difficultyLevels[this.settings.sliderValue].slice(2).trim();
    }

    const button = dropdown.querySelector('button');
    const options = dropdown.querySelector('.difficulty-options');

    console.log("me button in dropdown", button)
    if (button) {
      console.log("added dropdown button")
      button.addEventListener('click', (e) => {
        e.stopPropagation();
        console.log('Dropdown button clicked');
        options.classList.toggle('hidden'); // Toggle visibility
      });
    }

    const optionElements = dropdown.querySelectorAll('.difficulty-option');
    optionElements.forEach((option, index) => {
      option.addEventListener('click', () => {
        console.log(`Option ${index} clicked`);
        // this.updateSlider(index);
        currentLevel.textContent = difficultyLevels[index].slice(2).trim();
        options.classList.add('hidden'); // Hide options after selection
        // Get the parent ai-message element
        const aiMessageComponent = dropdown.closest('.ai-message-chat');



        if (!aiMessageComponent) {
          console.error('Could not find parent AI message component');
          return;
        }

        console.log("PROGRESS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        // Create and inject progress element
        const progressElement = document.createElement('div');
        progressElement.id = 'progress';
        
        console.log("======================= BEGINNING OF initializeDifficultyDropdown==================")
        // Find the content container within aiMessageComponent
        const contentContainer = aiMessageComponent.querySelector('.markdown-preview-container');

        const shareButtonContainer = aiMessageComponent.querySelector('.copy-paste-share-btn-container');
        if (shareButtonContainer) {
          shareButtonContainer.remove();
        }



        console.log("contentContainer progress in initializeDifficultyDropdown", contentContainer)
        if (contentContainer) {
          // Insert progress element before the content container
          contentContainer.parentNode.insertBefore(progressElement, contentContainer);
          
          // Find markdown preview container and add id
         
          contentContainer.id = 'markdown-preview';
        } else {
          console.error('Could not find content container in AI message');
          return;
        }

        console.log('Found AI message component:', aiMessageComponent);


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
}

export function setupModal(shadowEle) {
  const modal = shadowEle.querySelector("#modal1");

  const settingsBtn = shadowEle.querySelector("#settings-btn");
  enableTooltip(settingsBtn, "Open settings", shadowEle);
  const closeBtn = modal.querySelector("#close-btn");

  let overlay = shadowEle.querySelector(".overlay-settings");

  if (!overlay) {
    const menu = shadowEle.querySelector("#text-selection-menu");

    // Create and append the overlay
    overlay = document.createElement("div");
    overlay.classList.add("overlay-settings");

    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background-color: rgba(0, 0, 0, 0.5);
      z-index: 1000;
      display: none;
    `;
    menu.appendChild(overlay);
  }

  // Function to toggle the modal display
  const toggleModal = () => {
    const isDisplayed = modal.style.display === "block";
    modal.style.display = isDisplayed ? "none" : "block";
    overlay.style.display = isDisplayed ? "none" : "block";
  };

  // Close the modal when clicking on the close button
  const closeModal = () => {
    modal.style.display = "none";
    overlay.style.display = "none";
  };

  // Event listeners
  settingsBtn.addEventListener("click", toggleModal);
  closeBtn.addEventListener("click", closeModal);
  overlay.addEventListener("click", closeModal);
  modal.addEventListener("click", (event) => event.stopPropagation());

  // Ensure the modal closes when pressing the Escape key
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && modal.style.display === "block") {
      closeModal();
    }
  });
}
