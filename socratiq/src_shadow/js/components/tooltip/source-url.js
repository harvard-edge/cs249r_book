// Source metadata can come from saved conversations and is untrusted.
export function safeSourceUrl(value, base = globalThis.location?.href) {
  if (typeof value !== 'string' || !value.trim()) return '';
  try {
    const url = new URL(value.trim(), base);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return '';
    return url.href;
  } catch {return '';}
}

export function safeSourcePosition(value) {
  if (typeof value !== 'number' && typeof value !== 'string') return 0;
  const position = Number(value);
  return Number.isFinite(position) && position >= 0 ? position : 0;
}
