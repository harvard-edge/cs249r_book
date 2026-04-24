import {cloneElementById} from '../utils/utils'
import { SettingsManager } from '../../components/settings/settings'

// Helper function to update dropdown text
function updateDropdownText(element, difficultyIndex) {
    const difficultyLevels = ['🚲 Beginner', '🚗 Intermediate', '🚁 Advanced', '🛸 Bloom\'s Taxonomy'];
    const dropdown = element.querySelector('.difficulty-dropdown');
    if (dropdown) {
          dropdown.setAttribute('title', 'Select a difficulty level to redo the AI response');

        const currentLevel = dropdown.querySelector('.current-difficulty-level');
        if (currentLevel) {
            const index = Math.min(Math.max(parseInt(difficultyIndex, 10), 0), difficultyLevels.length - 1);
            currentLevel.textContent = difficultyLevels[index].slice(2).trim();
        }
    }
}

// /**
//  * Generates a message element based on the type given.
//  *
//  * @param {Element} shadowEle - The shadow element to clone from
//  * @param {string} type - The type of message to generate ('ai' or other)
//  * @return {Element} The cloned message element
//  */
// export function get_message_element(shadowEle, type, ele=''){
//     let clone;
//     if (type === 'ai'){
//         if(!ele){
//             clone = cloneElementById(shadowEle, "ai-message", '', 'message-container');
//             // Use global difficulty or default to 1
//             const currentDifficulty = window.current_difficulty || 1;
//             clone.setAttribute('data-difficulty', currentDifficulty);
//             // Update dropdown text for new message
//             updateDropdownText(clone, currentDifficulty);
//         }
//         else {
//             clone = ele;
//             // Get the stored difficulty from the original message's attribute
//             const storedDifficulty = ele.getAttribute('data-difficulty');
//             if (storedDifficulty !== null) {
//                 clone.setAttribute('data-difficulty', storedDifficulty);
//                 // Update dropdown text for regenerated message
//                 updateDropdownText(clone, storedDifficulty);
//             }
//         }
//     }
//     else {
//         clone = cloneElementById(shadowEle, "human-message", '', 'message-container');
//     }

//     clone.classList.add('new-message');
//     // Remove new-message class after animation completes
//     setTimeout(() => {
//       clone.classList.remove('new-message');
//     }, 8000);
    
//     return clone;
// }
export function get_message_element(shadowEle, type, ele='') {
    let clone;
    
    if (type === 'ai') {
      if (!ele) {
        clone = cloneElementById(shadowEle, "ai-message", '', 'message-container');
        const currentDifficulty = window.current_difficulty || 1;
        clone.setAttribute('data-difficulty', currentDifficulty);
        updateDropdownText(clone, currentDifficulty);
      } else {
        clone = ele;
        const storedDifficulty = ele.getAttribute('data-difficulty');
        if (storedDifficulty !== null) {
          clone.setAttribute('data-difficulty', storedDifficulty);
          updateDropdownText(clone, storedDifficulty);
        }
      }
  
      // Apply initial styles
      Object.assign(clone.style, {
        position: 'relative',
        borderRadius: '15px',
        border: '2px solid rgba(59, 130, 246, 0.2)'
      });
  
      // Create and add keyframe animation
      const keyframes = [
        { border: '2px solid rgba(59, 130, 246, 0.2)' },
        { border: '2px solid rgba(59, 130, 246, 0.8)' },
        { border: '2px solid rgba(59, 130, 246, 0.2)' }
      ];
  
      const animation = clone.animate(keyframes, {
        duration: 2000,
        iterations: 3,
        easing: 'ease-in-out'
      });
  
      // Cleanup after animation ends
      animation.onfinish = () => {
        clone.style.border = 'none';
      };
  
    } else {
      clone = cloneElementById(shadowEle, "human-message", '', 'message-container');
    }
    
    return clone;
  }


//   export function get_message_element(shadowEle, type, ele=''){
//     let clone;
//     if (type === 'ai'){
//         if(!ele){
//             clone = cloneElementById(shadowEle, "ai-message", '', 'message-container');
//             // Use global difficulty or default to 1
//             const currentDifficulty = window.current_difficulty || 1;
//             clone.setAttribute('data-difficulty', currentDifficulty);
//             // Update dropdown text for new message
//             updateDropdownText(clone, currentDifficulty);
//         }
//         else {
//             clone = ele;
//             // Get the stored difficulty from the original message's attribute
//             const storedDifficulty = ele.getAttribute('data-difficulty');
//             if (storedDifficulty !== null) {
//                 clone.setAttribute('data-difficulty', storedDifficulty);
//                 // Update dropdown text for regenerated message
//                 updateDropdownText(clone, storedDifficulty);
//             }
//         }
//     }
//     else {
//         clone = cloneElementById(shadowEle, "human-message", '', 'message-container');
//     }
    
//     return clone;
// }


/**
 * Generates reference buttons using the provided shadow element, clone, and links.
 *
 * @param {Element} shadowEle - The shadow DOM element to work with
 * @param {Element} clone - The clone element to work with
 * @param {Array<string>} links - The array of links to use for creating buttons
 */
export function get_reference_buttons (shadowEle, clone, links){
    // const container = cloneElementById(shadowEle, "reference-btn-container", '',)
// const container =  clone.querySelector('#reference-btn-container')
const container = cloneElementById(shadowEle, "reference-btn-container", '')

 links.forEach((link, i) => {

    const a = cloneElementById(shadowEle, "reference-btn", '')
    a.setAttribute('href', link)
    a.textContent = i + 1
    container.appendChild(a);
  
 })  
//  clone.appendChild(container);

return container

}