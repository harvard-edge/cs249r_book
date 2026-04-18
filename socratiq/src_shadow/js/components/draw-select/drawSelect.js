/**
 * Draw-to-Select: freehand lasso tool that captures DOM text inside the drawn region
 * and sends it to the AI chat panel.
 */

let active = false;
let canvas = null;
let ctx = null;
let path = [];
let onCaptured = null; // callback(capturedText)

// ─── Public API ───────────────────────────────────────────────────────────────

export function startDrawSelect(capturedCallback) {
  if (active) return;
  onCaptured = capturedCallback;
  _mount();
}

export function stopDrawSelect() {
  _unmount();
}

export function isDrawSelectActive() {
  return active;
}

// ─── Canvas lifecycle ─────────────────────────────────────────────────────────

function _mount() {
  active = true;
  canvas = document.createElement('canvas');
  canvas.id = 'socratiq-lasso-canvas';
  canvas.style.cssText = `
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    z-index: 2147483646;
    cursor: crosshair;
    pointer-events: all;
  `;
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  document.body.appendChild(canvas);

  ctx = canvas.getContext('2d');
  path = [];

  canvas.addEventListener('mousedown', _onStart);
  canvas.addEventListener('mousemove', _onMove);
  canvas.addEventListener('mouseup', _onEnd);
  canvas.addEventListener('touchstart', _onStart, { passive: false });
  canvas.addEventListener('touchmove', _onMove, { passive: false });
  canvas.addEventListener('touchend', _onEnd);
  document.addEventListener('keydown', _onKeyEscape);

  _drawHint();
}

function _unmount() {
  if (!canvas) return;
  canvas.removeEventListener('mousedown', _onStart);
  canvas.removeEventListener('mousemove', _onMove);
  canvas.removeEventListener('mouseup', _onEnd);
  canvas.removeEventListener('touchstart', _onStart);
  canvas.removeEventListener('touchmove', _onMove);
  canvas.removeEventListener('touchend', _onEnd);
  document.removeEventListener('keydown', _onKeyEscape);
  canvas.remove();
  canvas = null;
  ctx = null;
  path = [];
  active = false;
}

// ─── Drawing handlers ─────────────────────────────────────────────────────────

let drawing = false;

function _getPos(e) {
  if (e.touches) {
    return { x: e.touches[0].clientX, y: e.touches[0].clientY };
  }
  return { x: e.clientX, y: e.clientY };
}

function _onStart(e) {
  e.preventDefault();
  drawing = true;
  path = [];
  const pos = _getPos(e);
  path.push(pos);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function _onMove(e) {
  e.preventDefault();
  if (!drawing) return;
  const pos = _getPos(e);
  path.push(pos);
  _renderPath();
}

function _onEnd(e) {
  e.preventDefault();
  if (!drawing || path.length < 3) {
    drawing = false;
    return;
  }
  drawing = false;
  _closePath();
  const text = _extractTextInPath(path);
  _unmount();
  if (text && text.trim().length > 0 && onCaptured) {
    onCaptured(text.trim());
  }
}

function _onKeyEscape(e) {
  if (e.key === 'Escape') _unmount();
}

// ─── Rendering ────────────────────────────────────────────────────────────────

function _renderPath() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (path.length < 2) return;

  ctx.beginPath();
  ctx.moveTo(path[0].x, path[0].y);
  for (let i = 1; i < path.length; i++) {
    ctx.lineTo(path[i].x, path[i].y);
  }
  ctx.strokeStyle = 'rgba(99, 102, 241, 0.9)';
  ctx.lineWidth = 2.5;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.setLineDash([]);
  ctx.stroke();

  // Fill with semi-transparent purple
  ctx.fillStyle = 'rgba(99, 102, 241, 0.08)';
  ctx.fill();
}

function _closePath() {
  if (path.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(path[0].x, path[0].y);
  for (let i = 1; i < path.length; i++) ctx.lineTo(path[i].x, path[i].y);
  ctx.closePath();
  ctx.strokeStyle = 'rgba(99, 102, 241, 1)';
  ctx.lineWidth = 2.5;
  ctx.stroke();
  ctx.fillStyle = 'rgba(99, 102, 241, 0.15)';
  ctx.fill();
}

function _drawHint() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const text = 'Draw a circle around anything — press Esc to cancel';
  const px = canvas.width / 2;
  const py = 36;
  ctx.font = '14px system-ui, sans-serif';
  ctx.textAlign = 'center';
  const tw = ctx.measureText(text).width;
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.beginPath();
  ctx.roundRect(px - tw / 2 - 14, py - 18, tw + 28, 30, 8);
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.fillText(text, px, py);
}

// ─── Text extraction ──────────────────────────────────────────────────────────

function _pointInPolygon(px, py, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    const intersect = ((yi > py) !== (yj > py)) &&
      (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function _rectIntersectsPolygon(rect, polygon) {
  // Check the four corners and the centre of the rect
  const points = [
    { x: rect.left,  y: rect.top },
    { x: rect.right, y: rect.top },
    { x: rect.left,  y: rect.bottom },
    { x: rect.right, y: rect.bottom },
    { x: (rect.left + rect.right) / 2, y: (rect.top + rect.bottom) / 2 },
  ];
  return points.some(p => _pointInPolygon(p.x, p.y, polygon));
}

function _extractTextInPath(polygon) {
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        const p = node.parentElement;
        if (!p) return NodeFilter.FILTER_REJECT;
        const tag = p.tagName?.toLowerCase();
        // Skip scripts, styles, hidden elements
        if (['script', 'style', 'noscript', 'svg'].includes(tag)) return NodeFilter.FILTER_REJECT;
        const cs = getComputedStyle(p);
        if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    }
  );

  const captured = [];
  let node;

  while ((node = walker.nextNode())) {
    const text = node.textContent;
    if (!text.trim()) continue;

    // Use a Range to get the bounding rect of this text node
    const range = document.createRange();
    range.selectNode(node);
    const rects = range.getClientRects();

    for (const rect of rects) {
      if (rect.width === 0 || rect.height === 0) continue;
      if (_rectIntersectsPolygon(rect, polygon)) {
        captured.push(text.trim());
        break; // don't double-add same node
      }
    }
  }

  return captured.join(' ');
}
