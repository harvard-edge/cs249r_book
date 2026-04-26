/* ============================================================
   MLSys Playground — Sharp Shot (new, replaces Quantization Cliff)
   ------------------------------------------------------------
   Target shooting. Per-layer precision dials are your sight.
   Lower precision = cheaper but the target blurs, jitters, or
   drifts away from your crosshair depending on which layer you
   quantized.

   Song Han's per-layer mapping:
     - Edge layers (embedding, output) at int4 → target POSITION
       DRIFTS. Systematic bias. Visual misalignment. You aim at
       the bullseye but the true target has moved — classic
       edge-layer quantization collapse (LLM.int8, 2022).
     - Attention layers at low precision → target JITTERS.
       Softmax amplifies noise.
     - FFN layers at low precision → target BLURS (contrast loss).
       Most robust to aggressive quantization.

   10 shots per round. Score = hits. Hit ≥7/10 to ship.
   ============================================================ */

window.MLSP = window.MLSP || {};
MLSP.games = MLSP.games || {};

MLSP.games.quantization = function(canvas, opts) {
  opts = opts || {};
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;

  var PRECISIONS = [
    { name: "fp32", bits: 32, color: "#cfe2f3", stroke: "#4a90c4" },
    { name: "fp16", bits: 16, color: "#d4edda", stroke: "#3d9e5a" },
    { name: "int8", bits: 8,  color: "#fdebd0", stroke: "#c87b2a" },
    { name: "int4", bits: 4,  color: "#f9d6d5", stroke: "#c44" }
  ];
  var NUM_LAYERS = 6;
  var MAX_BUDGET = 96;
  var SHOTS_PER_ROUND = 10;
  var WIN_HITS = 7;
  var TARGET_RADIUS = 30;

  /* Layer roles determine how low precision manifests visually */
  var LAYER_ROLES = ["edge", "attn", "ffn", "attn", "ffn", "edge"];
  var LAYER_NAMES = ["embedding", "attn.1", "ffn.1", "attn.2", "ffn.2", "output"];

  function hashString(s) { var h = 2166136261 >>> 0; for (var i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)>>>0;} return h; }
  function mulberry32(seed) { var a = seed >>> 0; return function(){ a = (a + 0x6D2B79F5) >>> 0; var t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  var today = new Date().toISOString().slice(0, 10);
  var rand = mulberry32(hashString("quant-" + today));

  /* Target area (top portion of canvas) */
  var TARGET_AREA = { x: 260, y: 60, w: W - 280, h: 260 };

  var state = {
    layers: [],
    trueTargetX: 0, trueTargetY: 0,
    targetPhase: 0,
    crosshairX: TARGET_AREA.x + TARGET_AREA.w / 2,
    crosshairY: TARGET_AREA.y + TARGET_AREA.h / 2,
    shotsLeft: SHOTS_PER_ROUND,
    hits: 0, misses: 0,
    /* visual effects derived from layers */
    blur: 0,
    jitter: 0,
    driftX: 0, driftY: 0,
    /* animation */
    shotFlashMs: 0, lastShotHit: false, lastShotX: 0, lastShotY: 0,
    shakeAmt: 0, shakeT: 0,
    over: false, won: false,
    particles: [], floats: [], pops: [], flash: null
  };

  for (var i = 0; i < NUM_LAYERS; i++) state.layers.push({ precisionIdx: 0 });
  state.trueTargetX = TARGET_AREA.x + TARGET_AREA.w / 2;
  state.trueTargetY = TARGET_AREA.y + TARGET_AREA.h / 2;

  var alltimeBest = MLSP.bestScore.get("quantization");

  function bitsUsed() { var sum = 0; for (var i = 0; i < NUM_LAYERS; i++) sum += PRECISIONS[state.layers[i].precisionIdx].bits; return sum; }

  /* Derive blur / jitter / drift from current layer precisions.
     Called whenever a layer is cycled. */
  function updateVisualEffects() {
    state.blur = 0;
    state.jitter = 0;
    state.driftX = 0; state.driftY = 0;
    for (var i = 0; i < NUM_LAYERS; i++) {
      var p = state.layers[i].precisionIdx;
      var bitsReduction = (32 - PRECISIONS[p].bits) / 4;  // 0, 4, 6, 7
      if (bitsReduction === 0) continue;
      var role = LAYER_ROLES[i];
      if (role === "edge") {
        // Drift — systematic bias. Int4 on edge is the cliff.
        var sign = (i === 0) ? 1 : -1;
        if (p >= 3) { state.driftX += sign * 35; state.driftY += sign * 18; }        // int4 cliff
        else if (p >= 2) { state.driftX += sign * 6; state.driftY += sign * 3; }    // int8 small drift
        // fp16 no drift
      } else if (role === "attn") {
        state.jitter += bitsReduction * 1.5;
      } else if (role === "ffn") {
        state.blur += bitsReduction * 0.9;
      }
    }
  }

  function cycleLayer(idx) {
    if (state.over) return;
    var layer = state.layers[idx];
    layer.precisionIdx = (layer.precisionIdx + 1) % PRECISIONS.length;
    updateVisualEffects();
    var r = layerRowRect(idx);
    MLSP.pop(state, r.x + r.w / 2, r.y + r.h / 2, "#a31f34", { r: 10, ms: 200 });
  }

  function fireShot() {
    if (state.over || state.shotsLeft <= 0) return;
    if (bitsUsed() > MAX_BUDGET) {
      addFloat(state.crosshairX, state.crosshairY - 20, "over budget — reduce bits", "#c44");
      shake(6, 200);
      MLSP.flash(state, "#c44", 160);
      return;
    }
    state.shotsLeft--;
    // Hit test against TRUE target (not displayed — drift actually misaligns your aim)
    var dx = state.trueTargetX - state.crosshairX;
    var dy = state.trueTargetY - state.crosshairY;
    var dist = Math.sqrt(dx*dx + dy*dy);
    var hit = dist < TARGET_RADIUS;
    state.lastShotHit = hit;
    state.lastShotX = state.crosshairX;
    state.lastShotY = state.crosshairY;
    state.shotFlashMs = 400;
    if (hit) {
      state.hits++;
      addFloat(state.crosshairX, state.crosshairY - 14, "✓ hit", "#3d9e5a");
      burst(state.crosshairX, state.crosshairY, "#3d9e5a", 10);
      MLSP.pop(state, state.crosshairX, state.crosshairY, "#3d9e5a", { r: 20 });
    } else {
      state.misses++;
      addFloat(state.crosshairX, state.crosshairY - 14, "✗ miss", "#c44");
      burst(state.crosshairX, state.crosshairY, "#c44", 8);
      MLSP.pop(state, state.crosshairX, state.crosshairY, "#c44", { r: 22 });
      shake(4, 180);
    }
    // Move the true target after each shot
    state.trueTargetX = TARGET_AREA.x + 50 + rand() * (TARGET_AREA.w - 100);
    state.trueTargetY = TARGET_AREA.y + 40 + rand() * (TARGET_AREA.h - 80);

    if (state.shotsLeft <= 0) {
      state.over = true;
      state.won = state.hits >= WIN_HITS;
      MLSP.flash(state, state.won ? "#3d9e5a" : "#a31f34", 360);
      endGame();
    }
  }

  function endGame() {
    var finalHits = state.hits;
    if (finalHits > alltimeBest) { alltimeBest = finalHits; MLSP.bestScore.set("quantization", alltimeBest); }
    if (opts.onGameOver) opts.onGameOver({
      hits: state.hits, shots: SHOTS_PER_ROUND, won: state.won,
      bitsUsed: bitsUsed(), alltimeBest: alltimeBest,
      precisions: state.layers.map(function(l){ return l.precisionIdx; })
    });
  }

  function shake(a, ms) { state.shakeAmt = Math.max(state.shakeAmt, a); state.shakeT = Math.max(state.shakeT, ms); }
  function burst(x, y, color, n) {
    for (var i = 0; i < n; i++) {
      var ang = rand() * Math.PI * 2, spd = 1 + rand() * 2;
      state.particles.push({ x: x, y: y, vx: Math.cos(ang)*spd, vy: Math.sin(ang)*spd, age: 0, maxAge: 600, color: color });
    }
  }
  function addFloat(x, y, t, c) { state.floats.push({ x: x, y: y, text: t, color: c, age: 0, maxAge: 1200 }); }

  /* Layout */
  function layerRowRect(idx) {
    var colW = 230;
    var rowH = 38;
    var startY = 70;
    return { x: 20, y: startY + idx * rowH, w: colW, h: rowH - 6 };
  }
  function fireBtnRect() { return { x: 20, y: H - 80, w: 230, h: 48 }; }

  /* Input */
  canvas.addEventListener("mousemove", function(e) {
    if (state.over) return;
    var p = MLSP.canvasPoint(canvas, e);
    if (p.x >= TARGET_AREA.x && p.x <= TARGET_AREA.x + TARGET_AREA.w &&
        p.y >= TARGET_AREA.y && p.y <= TARGET_AREA.y + TARGET_AREA.h) {
      state.crosshairX = p.x;
      state.crosshairY = p.y;
    }
  });
  canvas.addEventListener("pointerdown", function(e) {
    e.preventDefault();
    if (state.over) { if (opts.onRetry) opts.onRetry(); return; }
    var p = MLSP.canvasPoint(canvas, e);
    // Layer click?
    for (var i = 0; i < NUM_LAYERS; i++) {
      var r = layerRowRect(i);
      if (p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h) { cycleLayer(i); return; }
    }
    // Fire button?
    var btn = fireBtnRect();
    if (p.x >= btn.x && p.x <= btn.x + btn.w && p.y >= btn.y && p.y <= btn.y + btn.h) { fireShot(); return; }
    // Shot from crosshair (clicking in target area)
    if (p.x >= TARGET_AREA.x && p.x <= TARGET_AREA.x + TARGET_AREA.w &&
        p.y >= TARGET_AREA.y && p.y <= TARGET_AREA.y + TARGET_AREA.h) {
      state.crosshairX = p.x;
      state.crosshairY = p.y;
      fireShot();
    }
  });
  window.addEventListener("keydown", function(e) {
    if (!MLSP.inViewport(canvas)) return;
    if (e.key === "r" || e.key === "R") { e.preventDefault(); if (opts.onRetry) opts.onRetry(); return; }
    if (state.over) return;
    if (e.key === " " || e.key === "Enter") { e.preventDefault(); fireShot(); }
    if (e.key >= "1" && e.key <= "6") { e.preventDefault(); cycleLayer(parseInt(e.key, 10) - 1); }
    // Aim with arrow keys
    var step = 8;
    if (e.key === "ArrowLeft")  { e.preventDefault(); state.crosshairX = Math.max(TARGET_AREA.x, state.crosshairX - step); }
    if (e.key === "ArrowRight") { e.preventDefault(); state.crosshairX = Math.min(TARGET_AREA.x + TARGET_AREA.w, state.crosshairX + step); }
    if (e.key === "ArrowUp")    { e.preventDefault(); state.crosshairY = Math.max(TARGET_AREA.y, state.crosshairY - step); }
    if (e.key === "ArrowDown")  { e.preventDefault(); state.crosshairY = Math.min(TARGET_AREA.y + TARGET_AREA.h, state.crosshairY + step); }
  });

  updateVisualEffects();

  var lastTime = 0;
  function frame(now) {
    var dt = lastTime ? (now - lastTime) : 16; lastTime = now; if (dt > 100) dt = 100;

    state.targetPhase += dt * 0.003;
    state.shotFlashMs = Math.max(0, state.shotFlashMs - dt);
    state.shakeT = Math.max(0, state.shakeT - dt);
    if (state.shakeT === 0) state.shakeAmt = 0;
    for (var pp of state.particles) { pp.x += pp.vx; pp.y += pp.vy; pp.vy += 0.15; pp.age += dt; }
    state.particles = state.particles.filter(function(x){ return x.age < x.maxAge; });
    for (var ff of state.floats) { ff.age += dt; ff.y -= dt * 0.04; }
    state.floats = state.floats.filter(function(x){ return x.age < x.maxAge; });
    MLSP.tickJuice(state, dt);

    if (opts.onScoreChange && !state.over) opts.onScoreChange({
      bitsUsed: bitsUsed(), budget: MAX_BUDGET,
      shotsLeft: state.shotsLeft, hits: state.hits, alltimeBest: alltimeBest
    });

    draw();
    requestAnimationFrame(frame);
  }

  function draw() {
    var sx = 0, sy = 0;
    if (state.shakeAmt > 0) { sx = (rand()-0.5)*state.shakeAmt; sy = (rand()-0.5)*state.shakeAmt; }
    ctx.save(); ctx.translate(sx, sy);
    ctx.clearRect(-20, -20, W + 40, H + 40);

    /* Header */
    ctx.fillStyle = "#333"; ctx.font = "bold 15px 'Helvetica Neue', Arial, sans-serif"; ctx.textAlign = "center";
    ctx.fillText("Sharp Shot", W/2, 24);
    ctx.font = "10.5px 'Helvetica Neue', Arial, sans-serif"; ctx.fillStyle = "#888";
    ctx.fillText("click a layer to cycle precision · lower bits = cheaper but blurs / jitters / drifts the target · " + WIN_HITS + "/" + SHOTS_PER_ROUND + " to ship", W/2, 42);

    /* Target area border */
    ctx.strokeStyle = "#e0e0e0"; ctx.lineWidth = 1;
    ctx.strokeRect(TARGET_AREA.x, TARGET_AREA.y, TARGET_AREA.w, TARGET_AREA.h);

    /* Displayed target position: true position + drift + per-frame jitter */
    var jitterX = state.jitter > 0 ? (Math.sin(state.targetPhase * 7) + (rand()-0.5)*0.6) * state.jitter : 0;
    var jitterY = state.jitter > 0 ? (Math.cos(state.targetPhase * 5.5) + (rand()-0.5)*0.6) * state.jitter : 0;
    var displayedX = state.trueTargetX + state.driftX + jitterX;
    var displayedY = state.trueTargetY + state.driftY + jitterY;

    /* Draw target with blur filter */
    ctx.save();
    if (state.blur > 0) ctx.filter = "blur(" + state.blur.toFixed(1) + "px)";
    drawTarget(displayedX, displayedY, TARGET_RADIUS);
    ctx.restore();

    /* If the last shot missed, briefly reveal the TRUE target so the player learns */
    if (state.shotFlashMs > 0 && !state.lastShotHit) {
      var a = state.shotFlashMs / 400;
      ctx.globalAlpha = a * 0.8;
      ctx.strokeStyle = "#c44"; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.arc(state.trueTargetX, state.trueTargetY, TARGET_RADIUS, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#c44"; ctx.font = "italic 10px 'Helvetica Neue', Arial, sans-serif"; ctx.textAlign = "center";
      ctx.fillText("true target", state.trueTargetX, state.trueTargetY - TARGET_RADIUS - 6);
      ctx.globalAlpha = 1;
    }

    /* Crosshair */
    drawCrosshair(state.crosshairX, state.crosshairY);

    /* Shot marker */
    if (state.shotFlashMs > 0) {
      var a = state.shotFlashMs / 400;
      ctx.globalAlpha = a;
      ctx.strokeStyle = state.lastShotHit ? "#3d9e5a" : "#c44";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(state.lastShotX, state.lastShotY, 6, 0, Math.PI * 2); ctx.stroke();
      ctx.globalAlpha = 1;
    }

    /* Layer dials (left panel) */
    for (var i = 0; i < NUM_LAYERS; i++) {
      var r = layerRowRect(i);
      var layer = state.layers[i];
      var prec = PRECISIONS[layer.precisionIdx];
      ctx.fillStyle = prec.color; ctx.strokeStyle = prec.stroke; ctx.lineWidth = 1.5;
      MLSP.roundRect(ctx, r.x, r.y, r.w, r.h, 5);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#333"; ctx.font = "bold 11px 'Helvetica Neue', Arial, sans-serif"; ctx.textAlign = "left";
      ctx.fillText(LAYER_NAMES[i], r.x + 10, r.y + r.h/2 + 4);
      ctx.fillStyle = "#777"; ctx.font = "9px 'Helvetica Neue', Arial, sans-serif";
      ctx.fillText(LAYER_ROLES[i], r.x + 10, r.y + r.h - 4);
      ctx.textAlign = "right"; ctx.fillStyle = prec.stroke;
      ctx.font = "bold 11px 'Helvetica Neue', Arial, sans-serif";
      ctx.fillText(prec.name, r.x + r.w - 10, r.y + r.h/2 + 4);
    }

    /* Budget bar (left panel, top) */
    var budgetX = 20, budgetY = 52, budgetW = 230, budgetH = 6;
    ctx.fillStyle = "#eee"; MLSP.roundRect(ctx, budgetX, budgetY, budgetW, budgetH, 3); ctx.fill();
    var used = bitsUsed();
    var frac = Math.min(1, used / MAX_BUDGET);
    var over = used > MAX_BUDGET;
    ctx.fillStyle = over ? "#c44" : (frac > 0.9 ? "#c87b2a" : "#4a90c4");
    MLSP.roundRect(ctx, budgetX, budgetY, budgetW * frac, budgetH, 3); ctx.fill();
    ctx.font = "10px 'Helvetica Neue', Arial, sans-serif"; ctx.fillStyle = "#555"; ctx.textAlign = "left";
    ctx.fillText("budget " + used + " / " + MAX_BUDGET + " bits", budgetX, budgetY - 3);

    /* Fire button */
    var btn = fireBtnRect();
    var canFire = state.shotsLeft > 0 && !state.over && used <= MAX_BUDGET;
    ctx.fillStyle = canFire ? "#a31f34" : "#bbb";
    MLSP.roundRect(ctx, btn.x, btn.y, btn.w, btn.h, 6); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.font = "bold 14px 'Helvetica Neue', Arial, sans-serif"; ctx.textAlign = "center";
    ctx.fillText("FIRE", btn.x + btn.w/2, btn.y + btn.h/2 - 2);
    ctx.font = "9px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("(space or click)", btn.x + btn.w/2, btn.y + btn.h/2 + 14);

    /* Particles + floats */
    for (var pi = 0; pi < state.particles.length; pi++) {
      var pa = state.particles[pi];
      ctx.globalAlpha = Math.max(0, 1 - pa.age / pa.maxAge);
      ctx.fillStyle = pa.color;
      ctx.beginPath(); ctx.arc(pa.x, pa.y, 2, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    for (var fi = 0; fi < state.floats.length; fi++) {
      var ff = state.floats[fi];
      ctx.globalAlpha = Math.max(0, 1 - ff.age / ff.maxAge);
      ctx.fillStyle = ff.color;
      ctx.font = "bold 12px 'Helvetica Neue', Arial, sans-serif"; ctx.textAlign = "center";
      ctx.fillText(ff.text, ff.x, ff.y);
    }
    ctx.globalAlpha = 1;

    /* HUD bottom strip */
    ctx.font = "11px 'Helvetica Neue', Arial, sans-serif"; ctx.fillStyle = "#333";
    ctx.textAlign = "left";
    ctx.fillText("shots left: " + state.shotsLeft, TARGET_AREA.x, H - 28);
    ctx.textAlign = "center";
    ctx.fillText("hits: " + state.hits + " / " + WIN_HITS + " to ship", TARGET_AREA.x + TARGET_AREA.w / 2, H - 28);
    ctx.textAlign = "right";
    ctx.fillText("all-time best " + alltimeBest, TARGET_AREA.x + TARGET_AREA.w, H - 28);
    ctx.font = "9px 'Helvetica Neue', Arial, sans-serif"; ctx.fillStyle = "#999"; ctx.textAlign = "left";
    ctx.fillText("daily " + today + " · day " + MLSP.dayNumber() + " · arrows / mouse aim · 1-6 cycles a layer · R retry", TARGET_AREA.x, H - 10);

    MLSP.drawJuice(ctx, state, W, H);
    if (state.over) drawGameOver();
    ctx.restore();
  }

  function drawTarget(cx, cy, radius) {
    // Concentric bullseye — outer white-ish, then blue, then red center
    ctx.fillStyle = "#fdebd0";
    ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#cfe2f3";
    ctx.beginPath(); ctx.arc(cx, cy, radius * 0.7, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#f9d6d5";
    ctx.beginPath(); ctx.arc(cx, cy, radius * 0.4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#a31f34";
    ctx.beginPath(); ctx.arc(cx, cy, radius * 0.15, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#555"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.stroke();
  }

  function drawCrosshair(x, y) {
    ctx.strokeStyle = "#a31f34"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x - 12, y); ctx.lineTo(x - 4, y);
    ctx.moveTo(x + 4, y); ctx.lineTo(x + 12, y);
    ctx.moveTo(x, y - 12); ctx.lineTo(x, y - 4);
    ctx.moveTo(x, y + 4); ctx.lineTo(x, y + 12);
    ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.stroke();
  }

  function drawGameOver() {
    ctx.fillStyle = "rgba(255,255,255,0.93)"; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = state.won ? "#3d9e5a" : "#a31f34";
    ctx.font = "bold 26px 'Helvetica Neue', Arial, sans-serif"; ctx.textAlign = "center";
    ctx.fillText(state.won ? "🏆 model shipped!" : "accuracy below spec", W/2, H/2 - 18);
    ctx.fillStyle = "#333"; ctx.font = "14px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("hit " + state.hits + " / " + SHOTS_PER_ROUND + " · " + bitsUsed() + "/" + MAX_BUDGET + " bits", W/2, H/2 + 8);
    ctx.fillStyle = "#777"; ctx.font = "italic 11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("tap or press R to retry", W/2, H/2 + 32);
  }

  requestAnimationFrame(frame);

  return {
    id: "quantization",
    name: "Sharp Shot",
    ahaLabel: "You just played at",
    ahaText: "Quantization error. Lower precision means noise, and that noise isn't uniform — it's what you saw. Edge layers (embedding and output) at int4 cause *systematic bias* (the target drifted away from your crosshair); attention layers amplify noise (jitter); FFN layers mostly tolerate it (just soft blur). Real quantization navigates exactly this non-uniform sensitivity. Mixed-precision allocation (HAWQ, Dong 2019) and calibration-based rounding (GPTQ, AWQ) exist to manage it.",
    buildShareText: function(r) {
      var ladder = "";
      var precs = r.precisions || state.layers.map(function(l){ return l.precisionIdx; });
      for (var i = 0; i < precs.length; i++) {
        ladder += precs[i] === 0 ? "🟦" : precs[i] === 1 ? "🟩" : precs[i] === 2 ? "🟧" : "🟥";
      }
      return "MLSysBook Playground · Sharp Shot · Day " + (MLSP.dayNumber ? MLSP.dayNumber() : today) + "\n" +
             (r.won ? "🏆 shipped" : "✗ off-spec") + " · " + r.hits + " / " + r.shots + " hits · " + r.bitsUsed + " bits\n" +
             ladder + "  ← layer precisions\n" +
             "play → mlsysbook.ai/games/quantization/";
    }
  };
};
