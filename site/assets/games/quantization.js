/* ============================================================
   MLSys Playground — Quantization Cliff (v2)
   ------------------------------------------------------------
   Six layers, four precisions (fp32/fp16/int8/int4), three
   deploys, 96-bit budget, target 85% accuracy.

   v2: deterministic accuracy calc (was randomised, made the
   game unlearnable). Same configuration always produces the
   same result. Daily seed.
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
  var DEPLOYS_PER_RUN = 3;
  var WIN_ACCURACY = 85;

  /* Daily seed */
  function hashString(s) { var h = 2166136261 >>> 0; for (var i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)>>>0;} return h; }
  function mulberry32(seed) { var a = seed >>> 0; return function(){ a = (a + 0x6D2B79F5) >>> 0; var t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  var today = new Date().toISOString().slice(0, 10);
  var rand = mulberry32(hashString("quant-" + today));

  // Sensitivity per layer is now fixed for the day (deterministic).
  function makeSensitivity() {
    var s = [];
    for (var i = 0; i < NUM_LAYERS; i++) {
      var edgeDistance = Math.min(i, NUM_LAYERS - 1 - i);
      var base = 0.9 - 0.7 * Math.min(1, edgeDistance / 2);
      s.push(base + (rand() - 0.5) * 0.15);   // seeded once per day, never re-rolled
    }
    return s;
  }

  var layerNames = ["embedding", "attn.1", "ffn.1", "attn.2", "ffn.2", "output"];

  var state = {
    layers: [],
    sensitivity: makeSensitivity(),
    deploysLeft: DEPLOYS_PER_RUN,
    lastAccuracy: 100,
    bestAccuracy: 0,
    over: false,
    won: false,
    shakeAmt: 0, shakeT: 0,
    floats: []
  };
  for (var i = 0; i < NUM_LAYERS; i++) state.layers.push({ precisionIdx: 0, accDrop: 0, revealed: false });

  var alltimeBest = MLSP.bestScore.get("quantization");

  function bitsUsed() {
    var sum = 0;
    for (var i = 0; i < NUM_LAYERS; i++) sum += PRECISIONS[state.layers[i].precisionIdx].bits;
    return sum;
  }

  function cycleLayer(idx) {
    if (state.over) return;
    var layer = state.layers[idx];
    layer.precisionIdx = (layer.precisionIdx + 1) % PRECISIONS.length;
    layer.revealed = false;
  }

  /* Deterministic accuracy calculation — same config = same result */
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
      var bitsReduction = (32 - prec.bits) / 4;     // 0, 4, 6, 7
      // DETERMINISTIC: the same precision pick always yields the same accuracy drop.
      var drop = bitsReduction * state.sensitivity[i] * 4;
      layer.accDrop = drop;
      layer.revealed = true;
      totalDrop += drop;
    }
    state.lastAccuracy = Math.max(0, 100 - totalDrop);
    if (state.lastAccuracy > state.bestAccuracy) state.bestAccuracy = state.lastAccuracy;

    addFloat(W/2, 80, state.lastAccuracy.toFixed(1) + "% accuracy", state.lastAccuracy >= WIN_ACCURACY ? "#3d9e5a" : "#c87b2a");

    if (state.lastAccuracy >= WIN_ACCURACY) { state.won = true; state.over = true; endGame(); }
    else if (state.deploysLeft <= 0)        { state.over = true;                   endGame(); }
  }

  function endGame() {
    var finalBest = Math.round(state.bestAccuracy);
    if (finalBest > alltimeBest) {
      alltimeBest = finalBest;
      MLSP.bestScore.set("quantization", alltimeBest);
    }
    if (opts.onGameOver) opts.onGameOver({
      bestAccuracy: state.bestAccuracy, won: state.won,
      bitsUsed: bitsUsed(), alltimeBest: alltimeBest
    });
  }

  function shake(a, ms) { state.shakeAmt = Math.max(state.shakeAmt, a); state.shakeT = Math.max(state.shakeT, ms); }
  function addFloat(x, y, t, c) { state.floats.push({ x: x, y: y, text: t, color: c, age: 0, maxAge: 1500 }); }

  function layerRowRect(idx) {
    var rowH = 32;
    var totalH = NUM_LAYERS * rowH;
    var startY = (H - totalH) / 2 - 10;
    return { x: 80, y: startY + idx * rowH, w: W - 160, h: rowH - 4 };
  }
  function deployBtnRect() { return { x: W/2 - 80, y: H - 80, w: 160, h: 36 }; }

  canvas.addEventListener("pointerdown", function(e) {
    e.preventDefault();
    if (state.over) { if (opts.onRetry) opts.onRetry(); return; }
    var p = MLSP.canvasPoint(canvas, e);
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
    if (opts.onScoreChange && !state.over) opts.onScoreChange({ bitsUsed: bitsUsed(), budget: MAX_BUDGET, deploysLeft: state.deploysLeft, alltimeBest: alltimeBest });
    draw();
    requestAnimationFrame(frame);
  }

  function draw() {
    var sx = 0, sy = 0;
    if (state.shakeAmt > 0) { sx = (rand()-0.5)*state.shakeAmt; sy = (rand()-0.5)*state.shakeAmt; }
    ctx.save(); ctx.translate(sx, sy);
    ctx.clearRect(-20, -20, W + 40, H + 40);

    ctx.fillStyle = "#333";
    ctx.font = "bold 15px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Quantization Cliff", W/2, 24);
    ctx.font = "10.5px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#888";
    ctx.fillText("click a layer to cycle precision · deploy · hit " + WIN_ACCURACY + "% accuracy under " + MAX_BUDGET + " bits", W/2, 42);

    // Budget bar
    var budgetX = 60, budgetY = 56, budgetW = W - 120, budgetH = 6;
    ctx.fillStyle = "#eee";
    MLSP.roundRect(ctx, budgetX, budgetY, budgetW, budgetH, 3); ctx.fill();
    var used = bitsUsed();
    var frac = Math.min(1, used / MAX_BUDGET);
    var over = used > MAX_BUDGET;
    ctx.fillStyle = over ? "#c44" : (frac > 0.9 ? "#c87b2a" : "#4a90c4");
    MLSP.roundRect(ctx, budgetX, budgetY, budgetW * frac, budgetH, 3); ctx.fill();
    ctx.font = "10px 'Helvetica Neue', Arial, sans-serif"; ctx.fillStyle = "#555"; ctx.textAlign = "left";
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

    if (state.lastAccuracy < 100) {
      ctx.font = "bold 15px 'Helvetica Neue', Arial, sans-serif";
      ctx.fillStyle = state.lastAccuracy >= WIN_ACCURACY ? "#3d9e5a" : "#c87b2a";
      ctx.textAlign = "center";
      ctx.fillText("last deploy: " + state.lastAccuracy.toFixed(1) + "%", W/2, H - 30);
    }

    // Daily seed line
    ctx.font = "9px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#999";
    ctx.textAlign = "left";
    ctx.fillText("daily " + today + "  · same network for everyone today  · alltime best " + alltimeBest + "%", 20, H - 10);

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
    ahaText: "Mixed-precision quantization. Layers have wildly different sensitivity to low precision — first and last layers (embeddings, output heads) hate int4, while middle layers often tolerate it. Production techniques like AWQ and GPTQ search for the same per-layer bit allocation you just made by feel.",
    buildShareText: function(r) {
      return "MLSys Playground · Quantization Cliff · " + today + "\n" +
             (r.won ? "🏆 shipped" : "✗ off-spec") + " · " + r.bestAccuracy.toFixed(0) + "% accuracy · " + r.bitsUsed + " bits\n" +
             "play → mlsysbook.ai/games/quantization/";
    }
  };
};
