// Sidebar labels are plain text, including labels restored from imported decks.
// Escape both text and quoted attributes before assembling the static controls.
export function escapeSidebarText(value) {
  return String(value ?? '').replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[character]);
}
