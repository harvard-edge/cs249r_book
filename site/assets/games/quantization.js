/* ============================================================
   MLSys Playground — Quantization Cliff
   ------------------------------------------------------------
   Six layers with precision dials (fp32 / fp16 / int8 / int4).
   Total bit budget enforced. Click a layer to cycle its
   precision. Click "deploy" to reveal accuracy.

   Sensitivity is HIDDEN and PER-LAYER. First and last layers
   are most sensitive to low precision; middle layers tolerate
   int4 fine. 3 deploys per run. Best score = highest accuracy
   that fits the budget.
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
  var MAX_BUDGET = 96;            // e.g. half of 6*32 = 96
  var DEPLOYS_PER_RUN = 3;
  var WIN_ACCURACY = 85;

  var rand = Math.random;

  // Sensitivity per layer: how much accuracy loss per bit-reduction step.
  // First and last layers are very sensitive. Pre-seeded each run.
  function makeSensitivity() {
    var s = [];
    for (var i = 0; i < NUM_LAYERS; i++) {
      var edgeDistance = Math.min(i, NUM_LAYERS - 1 - i);
      // Edge layers: sensitivity ~0.9; middle: ~0.2
      var base = 0.9 - 0.7 * Math.min(1, edgeDistance / 2);
      s.push(base + (rand() - 0.5) * 0.15);
    }
    return s;
  }

  var layerNames = ["embedding", "attn.1", "ffn.1", "attn.2", "ffn.2", "output"];

  var state = {
    layers: [],              // array of { precisionIdx, revealedAccDrop, lastDeployedIdx }
    sensitivity: makeSensitivity(),
    deploysLeft: DEPLOYS_PER_RUN,
    lastAccuracy: 100,
    bestAccuracy: 0,
    over: false,
    won: false,
    shakeAmt: 0, shakeT: 0,
    floats: []
  };
  for (var i = 0; i < NUM_LAYERS; i++) {
    state.layers.push({ precisionIdx: 0, accDrop: 0, revealed: false });
  }

  var alltimeBest = MLSP.bestScore.get("quantization");

  function bitsUsed() {
    var sum = 0;
    for (var i = 0; i < NUM_LAYERS; i++) sum += PRECISIONS[state.layers[i].precisionIdx].bits;
    return sum;
  }

  function cycleLayer(idx) {
    if (state.over) return;
    var layer = state.layers[idx];
    var next = (layer.precisionIdx + 1) % PRECISIONS.length;
    layer.precisionIdx = next;
    layer.revealed = false;
  }

  function deploy() {
    if (state.over || state.deploysLeft <= 0) return;
    if (bitsUsed() > MAX_BUDGET) {
      addFloat(W/2, 80, "over budget!", "#c44");
      shake(6, 200);
      return;
    }
    state.deploysLeft--;
    var totalDrop = 0;
    for (var i = 0; i < NUM_LAYERS; i++) {
      var layer = state.layers[i];
      var prec = PRECISIONS[layer.precisionIdx];
      var bitsReduction = (32 - prec.bits) / 4;  // 0, 4, 6, 7 steps
      var drop = bitsReduction * state.sensitivity[i] * (3 + rand() * 2);
      layer.accDrop = drop;
      layer.revealed = true;
      totalDrop += drop;
    }
    state.lastAccuracy = Math.max(0, 100 - totalDrop);
    if (state.lastAccuracy > state.bestAccuracy) state.bestAccuracy = state.lastAccuracy;

    addFloat(W/2, 80, state.lastAccuracy.toFixed(1) + "% accuracy", state.lastAccuracy >= WIN_ACCURACY ? "#3d9e5a" : "#c87b2a");

    if (state.lastAccuracy >= WIN_ACCURACY) {
      state.won = true;
      state.over = true;
      endGame();
    } else if (state.deploysLeft <= 0) {
      state.over = true;
      endGame();
    }
  }

  function endGame() {
    var finalBest = Math.round(state.bestAccuracy);
    if (finalBest > alltimeBest) {
      alltimeBest = finalBest;
      MLSP.bestScore.set("quantization", alltimeBest);
    }
    if (opts.onGameOver) opts.onGameOver({
      bestAccuracy: state.bestAccuracy,
      won: state.won,
      bitsUsed: bitsUsed(),
      alltimeBest: alltimeBest
    });
  }

  function shake(a, ms) { state.shakeAmt = Math.max(state.shakeAmt, a); state.shakeT = Math.max(state.shakeT, ms); }
  function addFloat(x, y, t, c) { state.floats.push({ x: x, y: y, text: t, color: c, age: 0, maxAge: 1500 }); }

  /* Layout — compute layer row positions for hit testing */
  function layerRowRect(idx) {
    var rowH = 32;
    var totalH = NUM_LAYERS * rowH;
    var startY = (H - totalH) / 2 - 10;
    return { x: 80, y: startY + idx * rowH, w: W - 160, h: rowH - 4 };
  }
  function deployBtnRect() {
    return { x: W/2 - 80, y: H - 80, w: 160, h: 36 };
  }

  /* Input */
  canvas.addEventListener("pointerdown", function(e) {
    e.preventDefault();
    if (state.over) { if (opts.onRetry) opts.onRetry(); return; }
    var p = MLSP.canvasPoint(canvas, e);
    // Layer clicks
    for (var i = 0; i < NUM_LAYERS; i++) {
      var r = layerRowRect(i);
      if (p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h) { cycleLayer(i); return; }
    }
    var btn = deployBtnRect();
    if (p.x >= btn.x && p.x <= btn.x + btn.w && p.y >= btn.y && p.y <= btn.y + btn.h) { deploy(); return; }
  });
  window.addEventListener("keydown", function(e) {
    if (!MLSP.inViewport(canvas)) return;
    if (e.key === "r" || e.key === "R") { e.preventDefault(); if (opts.onRetry) opts.onRetry(); return; }
    if (e.key === " " || e.key === "Enter") { e.preventDefault(); deploy(); }
  });

  var lastTime = 0;
  function frame(now) {
    var dt = lastTime ? (now - lastTime) : 16;
    lastTime = now;
    if (dt > 100) dt = 100;
    state.shakeT = Math.max(0, state.shakeT - dt);
    if (state.shakeT === 0) state.shakeAmt = 0;
    for (var ff of state.floats) { ff.age += dt; ff.y -= dt * 0.04; }
    state.floats = state.floats.filter(function(x){ return x.age < x.maxAge; });

    if (opts.onScoreChange && !state.over) opts.onScoreChange({
      bitsUsed: bitsUsed(), budget: MAX_BUDGET, deploysLeft: state.deploysLeft, alltimeBest: alltimeBest
    });

    draw();
    requestAnimationFrame(frame);
  }

  function draw() {
    var sx = 0, sy = 0;
    if (state.shakeAmt > 0) { sx = (rand()-0.5)*state.shakeAmt; sy = (rand()-0.5)*state.shakeAmt; }
    ctx.save();
    ctx.translate(sx, sy);
    ctx.clearRect(-20, -20, W + 40, H + 40);

    // Header
    ctx.fillStyle = "#333";
    ctx.font = "bold 15px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Quantization Cliff", W/2, 24);
    ctx.font = "10.5px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#888";
    ctx.fillText("click a layer to cycle precision · deploy when ready · hit " + WIN_ACCURACY + "% accuracy to win", W/2, 42);

    // Budget bar
    var budgetX = 60, budgetY = 56, budgetW = W - 120, budgetH = 6;
    ctx.fillStyle = "#eee";
    MLSP.roundRect(ctx, budgetX, budgetY, budgetW, budgetH, 3); ctx.fill();
    var used = bitsUsed();
    var frac = Math.min(1, used / MAX_BUDGET);
    var over = used > MAX_BUDGET;
    ctx.fillStyle = over ? "#c44" : (frac > 0.9 ? "#c87b2a" : "#4a90c4");
    MLSP.roundRect(ctx, budgetX, budgetY, budgetW * frac, budgetH, 3); ctx.fill();
    ctx.font = "10px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#555";
    ctx.textAlign = "left";
    ctx.fillText("budget " + used + " / " + MAX_BUDGET + " bits", budgetX, budgetY - 4);
    ctx.textAlign = "right";
    ctx.fillText("deploys left: " + state.deploysLeft + " / " + DEPLOYS_PER_RUN, budgetX + budgetW, budgetY - 4);

    // Layer rows
    for (var i = 0; i < NUM_LAYERS; i++) {
      var r = layerRowRect(i);
      var layer = state.layers[i];
      var prec = PRECISIONS[layer.precisionIdx];
      ctx.fillStyle = prec.color;
      ctx.strokeStyle = prec.stroke;
      ctx.lineWidth = 1.5;
      MLSP.roundRect(ctx, r.x, r.y, r.w, r.h, 5);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#333";
      ctx.font = "bold 12px 'Helvetica Neue', Arial, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(layerNames[i], r.x + 12, r.y + r.h/2 + 4);
      ctx.textAlign = "right";
      ctx.fillStyle = prec.stroke;
      ctx.fillText(prec.name + "  " + prec.bits + "-bit", r.x + r.w - 12, r.y + r.h/2 + 4);
      // If revealed from last deploy: show drop
      if (layer.revealed) {
        ctx.fillStyle = layer.accDrop < 1 ? "#3d9e5a" : layer.accDrop < 4 ? "#c87b2a" : "#c44";
        ctx.font = "10px 'Helvetica Neue', Arial, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText("−" + layer.accDrop.toFixed(1) + "% acc", r.x + r.w - 90, r.y + r.h/2 + 4);
      }
    }

    // Deploy button
    var btn = deployBtnRect();
    var canDeploy = state.deploysLeft > 0 && bitsUsed() <= MAX_BUDGET && !state.over;
    ctx.fillStyle = canDeploy ? "#a31f34" : "#bbb";
    MLSP.roundRect(ctx, btn.x, btn.y, btn.w, btn.h, 6); ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.font = "bold 13px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("deploy", btn.x + btn.w/2, btn.y + btn.h/2 + 5);

    // Last accuracy readout
    if (state.lastAccuracy < 100) {
      ctx.font = "bold 15px 'Helvetica Neue', Arial, sans-serif";
      ctx.fillStyle = state.lastAccuracy >= WIN_ACCURACY ? "#3d9e5a" : "#c87b2a";
      ctx.textAlign = "center";
      ctx.fillText("last deploy: " + state.lastAccuracy.toFixed(1) + "%", W/2, H - 30);
    }

    // Floats
    for (var fi = 0; fi < state.floats.length; fi++) {
      var ff = state.floats[fi];
      ctx.globalAlpha = Math.max(0, 1 - ff.age / ff.maxAge);
      ctx.fillStyle = ff.color;
      ctx.font = "bold 13px 'Helvetica Neue', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(ff.text, ff.x, ff.y);
    }
    ctx.globalAlpha = 1;

    if (state.over) drawGameOver();
    ctx.restore();
  }

  function drawGameOver() {
    ctx.fillStyle = "rgba(255,255,255,0.93)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = state.won ? "#3d9e5a" : "#a31f34";
    ctx.font = "bold 24px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(state.won ? "🏆 model shipped!" : "accuracy below spec", W/2, H/2 - 18);
    ctx.fillStyle = "#333";
    ctx.font = "14px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("best accuracy " + state.bestAccuracy.toFixed(1) + "% · " + bitsUsed() + "/" + MAX_BUDGET + " bits", W/2, H/2 + 8);
    ctx.fillStyle = "#777";
    ctx.font = "italic 11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("tap or press R to retry", W/2, H/2 + 32);
  }

  requestAnimationFrame(frame);

  return {
    id: "quantization",
    name: "Quantization Cliff",
    ahaLabel: "You just played at",
    ahaText: "Mixed-precision quantization. In real ML systems, layers have wildly different sensitivity to low precision — first and last layers (embeddings, output heads) hate int4, while middle layers often tolerate it. Finding the right allocation is what techniques like HAQ and SmoothQuant automate. You just did it by feel.",
    buildShareText: function(r) {
      return "MLSys Playground · Quantization Cliff\n" +
             (r.won ? "🏆 shipped" : "✗ off-spec") + " · " + r.bestAccuracy.toFixed(0) + "% accuracy · " + r.bitsUsed + " bits\n" +
             "play → mlsysbook.ai/games/quantization/";
    }
  };
};
