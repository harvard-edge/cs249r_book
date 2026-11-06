// Explicit selection mode: preview a paragraph before sending it to the service.
export function createParagraphPicker({shell, host, captureParagraph, onPick, onCancel}) {
  const banner = document.createElement('div');
  banner.className = 'reader-picker';
  banner.hidden = true;
  banner.setAttribute('role', 'region');
  banner.setAttribute('aria-label', 'Choose a book paragraph');
  const message = document.createElement('p');
  message.setAttribute('role', 'status');
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.textContent = 'Cancel picking';
  banner.append(message, cancel);
  shell.appendChild(banner);

  function stop() { banner.hidden = true; }
  function cancelPicking() {
    if (banner.hidden) return;
    stop();
    onCancel();
  }
  cancel.addEventListener('click', cancelPicking);
  function handleKey(event) {
    if (!banner.hidden && event.key === 'Escape') {
      event.preventDefault();
      event.stopImmediatePropagation();
      cancelPicking();
    }
  }
  function handleClick(event) {
    if (banner.hidden || event.composedPath().includes(host)) return;
    const target = event.target instanceof Element ? event.target : event.target?.parentElement;
    if (!target || target.closest('a, button, input, textarea, select, [role="button"], [contenteditable]')) return;
    let source;
    try { source = captureParagraph(target); }
    catch {
      message.textContent = 'That paragraph is too long. Choose a shorter paragraph, or cancel and highlight a few sentences.';
      return;
    }
    if (!source) {
      message.textContent = 'Choose a paragraph in the main book text. Links and controls still work normally.';
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    stop();
    onPick(source);
  }
  document.addEventListener('keydown', handleKey, true);
  document.addEventListener('click', handleClick, true);

  return {
    get active() { return !banner.hidden; },
    start() {
      message.textContent = 'Tap or click a paragraph in the book to preview it. Nothing is sent yet.';
      banner.hidden = false;
      cancel.focus({preventScroll: true});
    },
    stop,
    destroy() {
      stop();
      document.removeEventListener('keydown', handleKey, true);
      document.removeEventListener('click', handleClick, true);
      banner.remove();
    },
  };
}
