import { DIFFICULTY_LEVELS } from '../../../configs/client.config.js';
// import { enableTooltip } from '../tooltip/tooltip.js';

export class DropdownHandler {
  constructor(shadowRoot) {
    this.shadowRoot = shadowRoot;
  }

  handleDifficultySelection(dropdownElement, selectedIndex, ele) {
    // Get the parent ai-message element
    const aiMessage = dropdownElement.closest('.ai-message-chat');
    if (!aiMessage) {
      console.error('Could not find parent AI message');
      return;
    }

    // enableTooltip(dropdownElement, "Select a difficulty level to redo the AI response", this.shadowRoot);


    // Get the original prompt and type from the message attributes
    const originalPrompt = aiMessage.getAttribute('data-prompt');
    const messageType = aiMessage.getAttribute('data-type');
    const quizTitle = aiMessage.getAttribute('data-title');


    if (!originalPrompt || !messageType) {
      console.error('Missing required attributes on AI message');
      return;
    }

    // Store the selected difficulty as an attribute
    aiMessage.setAttribute('data-difficulty', selectedIndex);

    // Get the selected difficulty level instruction
    const selectedDifficulty = DIFFICULTY_LEVELS[selectedIndex];

    // Combine the original prompt with the difficulty level
    // const newPrompt = `${selectedDifficulty}\n\n${originalPrompt}`;
    const newPrompt = `$${originalPrompt}`;



    // Dispatch custom event with the new configuration
    const event = new CustomEvent('aiActionCompleted', {
      detail: {
        type: messageType,
        text: newPrompt,
        ele: aiMessage,
        tempDifficultyLevel: selectedDifficulty,
        quizTitle: quizTitle
      }
    });

    // Dispatch the event
    window.dispatchEvent(event);
  }
}