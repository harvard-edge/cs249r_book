const MAX_PASSAGE_SESSIONS = 5;

export function passageKey(source) {
  if (!source) return '';
  return [source.url, source.section, source.quote].map(value => String(value || '')).join('\u0000');
}

/** Keep inactive conversations, including their existing DOM event listeners. */
export function createPassageSessions({ thread, body, input, status, errorText, errorBox, retryButton, select, control }) {
  const sessions = new Map();

  return {
    sourceForKey(key) {
      return sessions.get(key)?.source;
    },

    save(source, conversation) {
      if (!source) return;
      sessions.set(passageKey(source), {
        source, conversation, nodes: [...thread.childNodes],
        draft: input.value, draftHeight: input.style.height,
        status: status.textContent, error: errorText.textContent,
        errorHidden: errorBox.hidden, retryHidden: retryButton.hidden,
        scrollTop: body.scrollTop,
      });
    },

    activate(source) {
      const key = passageKey(source);
      const saved = sessions.get(key);
      if (saved) {
        thread.replaceChildren(...saved.nodes);
        input.value = saved.draft;
        input.style.height = saved.draftHeight;
        status.textContent = saved.status;
        errorText.textContent = saved.error;
        errorBox.hidden = saved.errorHidden;
        retryButton.hidden = saved.retryHidden;
        body.scrollTop = saved.scrollTop;
      }
      const activeSource = saved?.source || source;
      // Insertion order tracks the least recently visited inactive passage.
      sessions.delete(key);
      sessions.set(key, { source: activeSource });
      while (sessions.size > MAX_PASSAGE_SESSIONS) sessions.delete(sessions.keys().next().value);
      return { source: activeSource, conversation: saved?.conversation };
    },

    render(activeSource) {
      control.hidden = sessions.size < 2;
      select.replaceChildren();
      for (const [key, session] of sessions) {
        const option = document.createElement('option');
        option.value = key;
        const source = session.source;
        const heading = String(source.section || source.title || 'Textbook passage').replace(/\s+/g, ' ').trim().slice(0, 45);
        const quote = String(source.quote || '').replace(/\s+/g, ' ').trim();
        option.textContent = `${heading} — ${quote.slice(0, 85)}${quote.length > 85 ? '…' : ''}`;
        option.selected = key === passageKey(activeSource);
        select.appendChild(option);
      }
    },
  };
}
