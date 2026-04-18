import feedbackHtml from './feedback_modal.html'

export function injectFeedback(shadowEle){
    
    const template = document.createElement("template");
    template.innerHTML = feedbackHtml;
    const modal_feedback = shadowEle.querySelector("#modal_feedback");
  
    if (modal_feedback) {
      // Clone the node deeply
      modal_feedback.appendChild(template.content.cloneNode(true));
    } else {
      console.error("feedbackModel not found");
    }
  }
  

export function openFeedback(shadowRoot) {

const openModalButton = shadowRoot.querySelector('#openModal-feedback');
const closeModalButton = shadowRoot.querySelector('#closeModal');
const modal = shadowRoot.querySelector('#myModal');

openModalButton.addEventListener('click', () => {
  modal.classList.remove('hidden');
});

closeModalButton.addEventListener('click', () => {
  modal.classList.add('hidden');
});
}
