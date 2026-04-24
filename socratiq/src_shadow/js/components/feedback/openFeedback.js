// Remove the HTML import
// import feedbackHtml from './feedback_modal.html'

// Add the feedback modal HTML as a template literal
const feedbackHtml = `
<!-- Modal Structure -->
<div id="myModal" class="hidden fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center p-4">
  <div class="bg-white rounded-lg overflow-hidden shadow-xl transform transition-all sm:max-w-lg sm:w-full" style="background-color: var(--socratiq-bg, #ffffff); color: var(--socratiq-text, #1f2328);">
    <div class="px-4 pt-2 pb-1 sm:p-6 sm:pb-4" style="max-height: 500px;">
      <iframe 
        id="feedback-iframe" 
        src="" 
        width="100%" 
        height="584" 
        frameborder="0" 
        marginheight="0" 
        marginwidth="0">Loading…</iframe>
    </div>
    <div class="px-4 py-1 sm:px-6 sm:flex sm:flex-row-reverse">
      <button 
        type="button" 
        id="closeModal" 
        class="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm">
        Close
      </button>
    </div>
  </div>
</div>
`;

export function injectFeedback(shadowEle) {
    const template = document.createElement("template");
    template.innerHTML = feedbackHtml;
    const modal_feedback = shadowEle.querySelector("#modal_feedback");
  
    if (modal_feedback) {
        modal_feedback.appendChild(template.content.cloneNode(true));
    } else {
        console.error("feedbackModel not found");
    }
}

export function openFeedback(shadowRoot) {
    const openModalButton = shadowRoot.querySelector('#openModal-feedback');
    const closeModalButton = shadowRoot.querySelector('#closeModal');
    const modal = shadowRoot.querySelector('#myModal');
    const iframe = shadowRoot.querySelector('#feedback-iframe');

    // Function to close modal
    const closeModal = () => {
        modal.classList.add('hidden');
    };

    // Open modal
    openModalButton.addEventListener('click', () => {
        if (iframe && !iframe.src) {
            iframe.src = "https://docs.google.com/forms/d/e/1FAIpQLSeK8RXgc6kbT1IbWVLjyUhwowp3x1ySbAjUQQqztdDs5ccmmQ/viewform?embedded=true";
        }
        modal.classList.remove('hidden');
    });

    // Close with close button
    closeModalButton.addEventListener('click', closeModal);

    // Close when clicking outside modal
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });

    // Close with Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
            closeModal();
        }
    });
}
