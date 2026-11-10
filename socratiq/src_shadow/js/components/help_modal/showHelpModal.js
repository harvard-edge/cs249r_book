import { enableTooltip } from '../tooltip/tooltip.js';
function createModal() {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full flex justify-center items-center z-50';
    modal.id = 'helpModal';

    // Create a container for the entire modal content
    const modalContainer = document.createElement('div');
    modalContainer.className = 'bg-white shadow-lg rounded-lg relative max-w-lg mx-auto min-h-[180px] gap-5';
    modalContainer.style.backgroundColor = 'var(--socratiq-bg, #ffffff)';
    modalContainer.style.color = 'var(--socratiq-text, #1f2328)';
    modalContainer.style.maxWidth = '460px'; // Limit the width of the modal
    modalContainer.id="helpModalContainer";
    modalContainer.style.width = 'min(460px, calc(100vw - 24px))'; // Ensures it fills the modal area but keeps 12px margin on mobile
    modalContainer.style.margin = '0 auto';
    modalContainer.onclick = event => event.stopPropagation(); // Prevent clicks from closing modal

    // Create scrollable content area
    const content = document.createElement('div');
    content.className = 'p-4 md:p-6 lg:p-8 overflow-auto';
    content.style.maxHeight = '500px'; // Max height for scrollable content

    // Convert Markdown to HTML and set as content
    const markdownContent = document.createElement('div');
    markdownContent.className = 'markdown-body space-y-4';
    // markdownContent.innerHTML =  `<iframe id="youtubeVideo" width="100%" height="315" src="https://www.youtube.com/embed/mIT9nIsxCe0?enablejsapi=1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>`
    //convertMarkdownToHTML(helpMarkdown);
    const pathname = typeof window !== 'undefined' ? (window.location?.pathname || '') : '';
    const guideHref = /\/(vol[1-4])(\/|$)/.test(pathname)
        ? pathname.replace(/\/(vol[1-4])\/.*$/, '/$1/shared/frontmatter/socratiq/socratiq.html').replace(/\/(vol[1-4])$/, '/$1/shared/frontmatter/socratiq/socratiq.html')
        : '/vol1/shared/frontmatter/socratiq/socratiq.html';
    markdownContent.innerHTML = `<p class="text-sm mb-3">Read the <a href="${guideHref}" target="_blank" rel="noopener noreferrer" class="text-blue-600 dark:text-blue-300 underline font-medium">SocratiQ User Guide</a> or our paper in the <a href="https://doi.org/10.1145/3743646.3750010" target="_blank" rel="noopener noreferrer" class="text-blue-600 dark:text-blue-300 underline font-medium">WCAE '25 proceedings</a> (<a href="https://arxiv.org/abs/2502.00341" target="_blank" rel="noopener noreferrer" class="text-blue-600 dark:text-blue-300 underline font-medium">arXiv:2502.00341</a>).</p><div id="youtubeVideo-container"></div>`;

    content.appendChild(markdownContent);
    modalContainer.appendChild(content);

    // Create a container for the buttons
    const buttonContainer = document.createElement('div');
    buttonContainer.className = 'flex justify-between items-center p-4';

    // Create and add Email button at the bottom of the modal
    const emailButton = document.createElement('button');
    emailButton.innerText = 'Contact Us';
    emailButton.className = 'bg-gray-600 rounded-lg p-2 hover:bg-gray-300 text-white';
    emailButton.id = 'emailButtonHelpModal';
    emailButton.onclick = () => {
        window.location.href = 'mailto:tinyml.aichat@gmail.com?subject=Help with SocratiQ';
    };

    // Create and add Close button at the bottom of the modal
    const closeButton = document.createElement('button');
    closeButton.innerText = 'Close';
    closeButton.className = 'bg-gray-600 rounded-lg p-2 text-white hover:bg-gray-300';
    closeButton.id = 'closeButtonHelpModal';

    buttonContainer.appendChild(emailButton);
    buttonContainer.appendChild(closeButton);

    // Append the buttonContainer to the modalContainer
    modalContainer.appendChild(buttonContainer);

    // Append the modalContainer to the modal
    modal.appendChild(modalContainer);

    

    return modal;
}


function toggleHelpModal(shadowEle) {
    const modal = shadowEle.querySelector('#helpModal');
    const isHidden = modal.classList.contains('hidden');
    const youtubeVid = modal.querySelector('#youtubeVideo-container');

    if (isHidden) {
        modal.classList.remove('hidden');
        youtubeVid.innerHTML = `<iframe id="youtubeVideo" width="100%" height="200" style="width: 100%; aspect-ratio: 16 / 9; height: auto; min-height: 180px; max-height: 240px; border: 0; border-radius: 8px; display: block;" src="https://www.youtube.com/embed/mIT9nIsxCe0?enablejsapi=1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>`;
    } else {
        modal.classList.add('hidden');
        stopYouTubeVideo(modal);
        youtubeVid.innerHTML = '';
    }
}

// Function to initialize and show the modal within a shadow DOM
export function showHelpModal(shadowEle) {
    const modal = createModal();
    const helpModal = shadowEle.querySelector('#helpModal');
    if (!helpModal.contains(modal)) {
        helpModal.appendChild(modal);
    }
    setUpHelpButton(shadowEle);

    // Close modal with Escape key
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            const modal = shadowEle.querySelector('#helpModal');
            if (modal && !modal.classList.contains('hidden')) {
                toggleHelpModal(shadowEle);
            }
        }
    });

    // Add click handler to modal background
    modal.addEventListener('click', (event) => {
        const modalContainer = modal.querySelector('#helpModalContainer');
        if (event.target === modal && !modalContainer.contains(event.target)) {
            toggleHelpModal(shadowEle);
        }
    });

    const closeButton = modal.querySelector('#closeButtonHelpModal');
    closeButton.onclick = () => toggleHelpModal(shadowEle);
}

// Function to set up the trigger button within the shadow DOM
function setUpHelpButton(shadowEle) {
    const helpButton = shadowEle.querySelector('#help-btn');
    enableTooltip(helpButton, "Get help with SocratiQ", shadowEle);

    helpButton.onclick = () => toggleHelpModal(shadowEle);
}

function stopYouTubeVideo(modal) {
    var iframe = modal.querySelector('iframe');
    if (iframe && iframe.src.includes('youtube.com/embed')) {
        iframe.contentWindow.postMessage(JSON.stringify({
            'event': 'command',
            'func': 'pauseVideo'
        }), '*');
    }
}