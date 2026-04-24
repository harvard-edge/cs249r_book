export function initiate_menu(event) {
    const textarea = document.getElementById('autoresizing-textarea');

    textarea.addEventListener('input', (event) => {
    const target = event.target;
    target.style.height = 'auto';
    target.style.height = `${Math.min(target.scrollHeight, 400)}px`;
    });
}