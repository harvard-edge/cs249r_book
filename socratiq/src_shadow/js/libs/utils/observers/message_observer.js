import { DropdownHandler } from '../../../components/settings/dropdown-handler.js';

export class MessageObserver {
  constructor(shadowRoot) {
    this.shadowRoot = shadowRoot;
    this.dropdownHandler = new DropdownHandler(shadowRoot);
    this.hasInitializedGlobalListener = false;
    this.init();
  }

  init() {
    
    // Create an observer instance
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === 1) {
            const aiMessages = node.classList?.contains('ai-message-chat') ? 
              [node] : 
              Array.from(node.querySelectorAll('.ai-message-chat'));
            
            aiMessages.forEach(msg => this.initializeMessageDropdown(msg));
          }
        });
      });
    });

    observer.observe(this.shadowRoot, {
      childList: true,
      subtree: true
    });

    // Initialize existing messages
    const existingMessages = this.shadowRoot.querySelectorAll('.ai-message-chat');
    existingMessages.forEach(msg => this.initializeMessageDropdown(msg));
  }

  initializeMessageDropdown(aiMessage) {
    
    const dropdown = aiMessage.querySelector('.difficulty-dropdown');
    if (!dropdown) {
      return;
    }

    const difficultyLevels = ['🚲 Beginner', '🚗 Intermediate', '🚁 Advanced', '🛸 Bloom\'s Taxonomy'];
    const currentLevel = dropdown.querySelector('.current-difficulty-level');
    const button = dropdown.querySelector('button');
    const options = dropdown.querySelector('.difficulty-options');


    // Set initial text based on data-difficulty attribute
    if (currentLevel) {
      const storedDifficulty = aiMessage.getAttribute('data-difficulty');
      
      if (storedDifficulty !== null) {
        // Convert to number and ensure it's within bounds
        const difficultyIndex = Math.min(Math.max(parseInt(storedDifficulty, 10), 0), difficultyLevels.length - 1);
        currentLevel.textContent = difficultyLevels[difficultyIndex].slice(2).trim();
      } else {
        currentLevel.textContent = difficultyLevels[0].slice(2).trim();
      }
    }

    // Setup button click
    if (button) {
      const newButton = button.cloneNode(true);
      button.parentNode.replaceChild(newButton, button);
      newButton.addEventListener('click', (e) => {
        e.stopPropagation();
        options.classList.toggle('hidden');
      });
    }

    // Setup options
    const optionElements = dropdown.querySelectorAll('.difficulty-option');
    optionElements.forEach((option, index) => {
      const newOption = option.cloneNode(true);
      option.parentNode.replaceChild(newOption, option);
      
      newOption.addEventListener('click', () => {
        if (currentLevel) {
          currentLevel.textContent = difficultyLevels[index].slice(2).trim();
        }
        options.classList.add('hidden');

        const aiMessageComponent = dropdown.closest('.ai-message-chat');
        if (!aiMessageComponent) return;

        // Store the selected index in the AI message element itself
        aiMessageComponent.dataset.selectedDifficulty = index;
        
        // Create progress element
        const progressElement = document.createElement('div');
        progressElement.id = 'progress';
        
        // Handle content container
        const contentContainer = aiMessageComponent.querySelector('.markdown-preview-container');
        
        // Remove share buttons if they exist
        const shareButtonContainer = aiMessageComponent.querySelector('.copy-paste-share-btn-container');
        if (shareButtonContainer) {
          shareButtonContainer.remove();
        }

        if (contentContainer) {
          contentContainer.parentNode.insertBefore(progressElement, contentContainer);
          contentContainer.id = 'markdown-preview';
        }

        this.dropdownHandler.handleDifficultySelection(dropdown, index, aiMessageComponent);
      });
    });

    // Setup global click listener only once
    if (!this.hasInitializedGlobalListener) {
      document.addEventListener('click', (e) => {
        const dropdowns = this.shadowRoot.querySelectorAll('.difficulty-dropdown');
        dropdowns.forEach(dropdown => {
          if (!dropdown.contains(e.target)) {
            const options = dropdown.querySelector('.difficulty-options');
            if (options) options.classList.add('hidden');
          }
        });
      });
      this.hasInitializedGlobalListener = true;
    }

    dropdown.dataset.initialized = 'true';
  }
}