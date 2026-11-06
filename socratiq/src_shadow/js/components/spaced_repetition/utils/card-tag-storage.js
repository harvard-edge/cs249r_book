// Compound IndexedDB keys keep numeric and string card IDs distinct and avoid
// delimiter collisions. Legacy string keys remain readable through the indexes.
export function cardTagKey(cardId, tag) {
  if ((typeof cardId !== 'string' && typeof cardId !== 'number') ||
      (typeof cardId === 'number' && !Number.isFinite(cardId)) || typeof tag !== 'string') {
    throw new TypeError('Card tags require a string or numeric card ID and a string tag');
  }
  return [typeof cardId, cardId, tag];
}

function writeTags(store, cardId, tags, replaceAll, metadata = {}) {
  const distinctTags = [...new Set(tags)];
  for (const tag of distinctTags) cardTagKey(cardId, tag);
  const previous = new Map();
  // Migrate only this card's affected records, inside the caller's transaction.
  // An old key that collided with another card is never deleted by its spelling.
  const request = store.index('card_id').openCursor(IDBKeyRange.only(cardId));
  request.addEventListener('success', () => {
    try {
      const cursor = request.result;
      if (cursor) {
        const record = cursor.value;
        if (replaceAll || distinctTags.includes(record.tag)) {
          previous.set(record.tag, { ...previous.get(record.tag), ...record });
          cursor.delete();
        }
        cursor.continue();
      } else {
        for (const tag of distinctTags) {
          store.put({ ...previous.get(tag), ...metadata,
            id: cardTagKey(cardId, tag), card_id: cardId, tag });
        }
      }
    } catch {
      store.transaction.abort();
    }
  });
  return request;
}

export function putCardTag(store, cardId, tag, metadata = {}) {
  return writeTags(store, cardId, [tag], false, metadata);
}

export function replaceCardTags(store, cardId, tags) {
  return writeTags(store, cardId, tags, true);
}
