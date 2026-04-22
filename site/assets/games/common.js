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

/* ============================================================
   Juice module — shared visual feedback primitives.
   Pop ring on score events, screen flash on catastrophes,
   ease curves for tween work. Each game wires these in via
   four lines: pop on success, flash on milestone, tickJuice
   in the frame loop, drawJuice at the end of draw().
   ============================================================ */
MLSP.easeOutCubic = function(t) { return 1 - Math.pow(1 - t, 3); };
MLSP.easeOutBack  = function(t) { var c = 1.7; return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2); };

MLSP.pop = function(state, x, y, color, opts) {
  opts = opts || {};
  state.pops = state.pops || [];
  state.pops.push({
    x: x, y: y, color: color,
    age: 0, maxAge: opts.ms || 360,
    radius: opts.r || 16
  });
};

MLSP.flash = function(state, color, ms) {
  state.flash = { color: color, age: 0, maxAge: ms || 220 };
};

MLSP.tickJuice = function(state, dt) {
  if (state.pops) {
    for (var i = 0; i < state.pops.length; i++) state.pops[i].age += dt;
    state.pops = state.pops.filter(function(p){ return p.age < p.maxAge; });
  }
  if (state.flash) {
    state.flash.age += dt;
    if (state.flash.age >= state.flash.maxAge) state.flash = null;
  }
};

MLSP.drawJuice = function(ctx, state, W, H) {
  if (state.pops) {
    for (var i = 0; i < state.pops.length; i++) {
      var p = state.pops[i];
      var t = p.age / p.maxAge;
      var eased = MLSP.easeOutBack(t);
      ctx.globalAlpha = 1 - MLSP.easeOutCubic(t);
      ctx.strokeStyle = p.color;
      ctx.lineWidth = 2.2 * (1 - t);
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius * eased, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  if (state.flash) {
    ctx.fillStyle = state.flash.color;
    ctx.globalAlpha = (1 - state.flash.age / state.flash.maxAge) * 0.40;
    ctx.fillRect(0, 0, W, H);
    ctx.globalAlpha = 1;
  }
};

/* Days since launch — gives shareables a Wordle-style daily number */
MLSP.dayNumber = function() {
  var launch = new Date("2026-04-22T00:00:00Z");
  var now = new Date();
  return Math.max(1, Math.floor((now - launch) / (1000 * 60 * 60 * 24)) + 1);
};
