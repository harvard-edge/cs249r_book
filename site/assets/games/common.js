/* ============================================================
   MLSys Playground — shared runtime
   Best-score persistence, input helpers, aha-card rendering.
   Every game is a function that mounts into a <canvas>.
   ============================================================ */

window.MLSP = window.MLSP || {};
MLSP.games = MLSP.games || {};

MLSP.bestScore = {
  get: function(key) {
    try { return parseInt(localStorage.getItem("mlsp-best-" + key) || "0", 10) || 0; }
    catch (e) { return 0; }
  },
  set: function(key, val) {
    try { localStorage.setItem("mlsp-best-" + key, String(val)); }
    catch (e) { /* ignore */ }
  }
};

MLSP.inViewport = function(el) {
  var r = el.getBoundingClientRect();
  return r.bottom > 0 && r.top < (window.innerHeight || document.documentElement.clientHeight);
};

// Convert a pointer/mouse/touch event to canvas coordinates, handling CSS scaling.
MLSP.canvasPoint = function(canvas, evt) {
  var rect = canvas.getBoundingClientRect();
  var scaleX = canvas.width / rect.width;
  var scaleY = canvas.height / rect.height;
  var cx, cy;
  if (evt.touches && evt.touches.length) {
    cx = evt.touches[0].clientX;
    cy = evt.touches[0].clientY;
  } else {
    cx = evt.clientX;
    cy = evt.clientY;
  }
  return { x: (cx - rect.left) * scaleX, y: (cy - rect.top) * scaleY };
};

// Distance from point (px,py) to line segment (x1,y1)-(x2,y2).
MLSP.distToSegment = function(px, py, x1, y1, x2, y2) {
  var dx = x2 - x1, dy = y2 - y1;
  var len2 = dx * dx + dy * dy;
  if (len2 === 0) return Math.hypot(px - x1, py - y1);
  var t = ((px - x1) * dx + (py - y1) * dy) / len2;
  if (t < 0) t = 0; else if (t > 1) t = 1;
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
};

// Standard Gaussian (Box–Muller). Games use this for weight initialisation.
MLSP.gauss = function() {
  var u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

// Render the "aha" explainer card after a game ends. Appends to the given container.
MLSP.showAhaCard = function(container, label, text) {
  if (!container) return;
  if (container.querySelector(".mlsp-aha-card")) return; // don't double-append
  var card = document.createElement("div");
  card.className = "mlsp-aha-card";
  var lbl = document.createElement("span");
  lbl.className = "mlsp-aha-label";
  lbl.textContent = label;
  var p = document.createElement("p");
  p.textContent = text;
  card.appendChild(lbl);
  card.appendChild(p);
  container.appendChild(card);
};

// Rounded-rect helper on a 2D canvas context.
MLSP.roundRect = function(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
};
