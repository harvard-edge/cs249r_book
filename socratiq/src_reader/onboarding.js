const TOUR_KEY = 'socratiq_reader_tour_v1';

/** A nonmodal spotlight: the reader remains usable and no step sends a request. */
export function createReaderTour({shell, targets}) {
  const spotlight = document.createElement('div');
  spotlight.className = 'reader-tour-spotlight';
  spotlight.setAttribute('aria-hidden', 'true');
  const card = document.createElement('section');
  card.className = 'reader-tour';
  card.setAttribute('role', 'dialog');
  card.setAttribute('aria-label', 'SocratiQ guided tour');
  card.innerHTML = `<p class="reader-tour-count" aria-live="polite"></p><h3 tabindex="-1"></h3><p class="reader-tour-description"></p><div class="reader-tour-actions"><button type="button" data-tour="skip">Skip tour</button><button type="button" data-tour="back">Back</button><button type="button" data-tour="next">Next</button></div>`;
  card.hidden = spotlight.hidden = true;
  shell.append(spotlight, card);
  const heading = card.querySelector('h3');
  const back = card.querySelector('[data-tour="back"]');
  const next = card.querySelector('[data-tour="next"]');
  const steps = [
    {target: targets.preferences, title: 'Make explanations work for you', text: 'Choose Beginner, Intermediate, or Advanced depth, customize selection bar actions, or connect your own Gemini, Claude, or OpenRouter model (free shared model by default; saved locally).'},
    {target: targets.choose, title: 'Start with a passage', text: 'Highlight text for Explain this, Ask…, Check me, or Taxonomy (use ⚙ to customize), or use Choose a paragraph to preview before asking.'},
    {target: targets.study, title: 'Keep learning', text: 'After an explanation, try Simplify, Show an example, Check my understanding, or Map to taxonomy. Study tools opens quizzes, flashcards, and local progress.'},
  ];
  let index = 0;
  let previousFocus = null;
  let completedInMemory = false;
  function hasSeen() {
    try {return completedInMemory || Boolean(localStorage.getItem(TOUR_KEY));} catch {return completedInMemory;}
  }
  function finish({restoreFocus = true, remember = true} = {}) {
    if (card.hidden) return;
    card.hidden = spotlight.hidden = true;
    if (remember) {
      completedInMemory = true;
      try {localStorage.setItem(TOUR_KEY, 'true');} catch { /* Session dismissal still works. */ }
    }
    if (restoreFocus && previousFocus?.isConnected) previousFocus.focus({preventScroll: true});
  }
  function position() {
    if (card.hidden) return;
    const target = steps[index].target();
    const rect = target?.getBoundingClientRect();
    const viewport = window.visualViewport;
    const left = viewport?.offsetLeft || 0;
    const top = viewport?.offsetTop || 0;
    const width = viewport?.width || window.innerWidth;
    const height = viewport?.height || window.innerHeight;
    card.style.maxHeight = `${Math.max(100, height - 24)}px`;
    const cardWidth = Math.min(320, width - 24);
    card.style.width = `${cardWidth}px`;
    const box = card.getBoundingClientRect();
    const visible = rect && rect.width > 0 && rect.height > 0 && rect.bottom > top && rect.top < top + height;
    spotlight.hidden = !visible;
    if (visible) {
      Object.assign(spotlight.style, {left: `${Math.max(left + 4, rect.left - 4)}px`, top: `${Math.max(top + 4, rect.top - 4)}px`, width: `${Math.min(rect.width + 8, width - 8)}px`, height: `${Math.min(rect.height + 8, height - 8)}px`});
    }
    let cardTop = visible ? rect.bottom + 12 : top + (height - box.height) / 2;
    if (cardTop + box.height > top + height - 12) cardTop = visible ? rect.top - box.height - 12 : cardTop;
    Object.assign(card.style, {left: `${Math.min(Math.max(visible ? rect.left : left + (width - cardWidth) / 2, left + 12), left + width - cardWidth - 12)}px`, top: `${Math.max(top + 12, Math.min(cardTop, top + height - box.height - 12))}px`});
  }
  function render() {
    const step = steps[index];
    heading.textContent = step.title;
    card.querySelector('.reader-tour-count').textContent = `Step ${index + 1} of ${steps.length}`;
    card.querySelector('.reader-tour-description').textContent = step.text;
    back.disabled = index === 0;
    next.textContent = index === steps.length - 1 ? 'Done' : 'Next';
    const target = step.target();
    target?.scrollIntoView({block: 'nearest', behavior: 'instant'});
    position();
    heading.focus({preventScroll: true});
  }
  card.addEventListener('click', event => {
    const action = event.target.closest('[data-tour]')?.dataset.tour;
    if (action === 'skip') finish();
    if (action === 'back' && index > 0) {index--; render();}
    if (action === 'next') {if (index === steps.length - 1) finish(); else {index++; render();}}
  });
  function escape(event) {
    if (event.key !== 'Escape' || card.hidden) return;
    event.preventDefault(); event.stopImmediatePropagation(); finish();
  }
  // Clicking a reader control naturally ends the introduction without blocking it.
  function interact(event) {
    if (!card.hidden && !event.composedPath().includes(card)) finish({restoreFocus: false});
  }
  shell.addEventListener('pointerdown', interact);
  document.addEventListener('keydown', escape, true);
  window.addEventListener('resize', position, {passive: true});
  document.addEventListener('scroll', position, {capture: true, passive: true});
  window.visualViewport?.addEventListener('resize', position, {passive: true});
  window.visualViewport?.addEventListener('scroll', position, {passive: true});
  return {
    start({replay = false} = {}) {
      if (!card.hidden || (!replay && hasSeen())) return;
      previousFocus = shell.getRootNode().activeElement;
      index = 0; card.hidden = false; render();
    },
    stop: finish,
    destroy() {
      finish({restoreFocus: false, remember: false});
      shell.removeEventListener('pointerdown', interact);
      document.removeEventListener('keydown', escape, true);
      window.removeEventListener('resize', position);
      document.removeEventListener('scroll', position, true);
      window.visualViewport?.removeEventListener('resize', position);
      window.visualViewport?.removeEventListener('scroll', position);
      card.remove(); spotlight.remove();
    },
  };
}
