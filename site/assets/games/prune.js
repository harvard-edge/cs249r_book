/* ============================================================
   MLSys Playground — Prune
   Click weights to remove them. Keep accuracy above threshold.
   Teaches magnitude-based pruning and the lottery-ticket hypothesis.
   ============================================================ */

window.MLSP = window.MLSP || {};
MLSP.games = MLSP.games || {};

MLSP.games.prune = function(canvas, opts) {
  opts = opts || {};
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;

  // Game parameters
  var LAYERS = [5, 8, 3];           // input, hidden, output
  var ACCURACY_THRESHOLD = 60;      // game over below this
  var HOVER_RADIUS_PX = 8;

  // Build neuron positions
  var neurons = [];
  var topMargin = 60, bottomMargin = 70;
  var leftMargin = 70, rightMargin = 70;
  var innerW = W - leftMargin - rightMargin;
  var innerH = H - topMargin - bottomMargin;
  var layerDX = innerW / (LAYERS.length - 1);
  LAYERS.forEach(function(count, li) {
    var cellH = innerH / Math.max(count - 1, 1);
    var y0 = count === 1 ? (topMargin + innerH / 2) : topMargin;
    for (var i = 0; i < count; i++) {
      neurons.push({
        layer: li,
        idx: i,
        x: leftMargin + li * layerDX,
        y: y0 + i * cellH
      });
    }
  });

  // Build weights between adjacent layers
  var weights = [];
  for (var li = 0; li < LAYERS.length - 1; li++) {
    var from = neurons.filter(function(n){ return n.layer === li; });
    var to   = neurons.filter(function(n){ return n.layer === li + 1; });
    from.forEach(function(f) {
      to.forEach(function(t) {
        var mag = Math.abs(MLSP.gauss());
        // Importance correlates with magnitude but has noise — the core puzzle:
        // you see magnitude, but the true importance is slightly different.
        var importance = Math.max(0, mag + MLSP.gauss() * 0.3);
        weights.push({
          from: f, to: t,
          magnitude: mag,
          importance: importance,
          pruned: false,
          fadeAlpha: 1.0
        });
      });
    });
  }

  var totalImportance = weights.reduce(function(s, w){ return s + w.importance; }, 0);
  var removedImportance = 0;

  var state = {
    accuracy: 100,
    sparsity: 0,
    pruned: 0,
    total: weights.length,
    over: false,
    hoverIdx: -1
  };

  var best = MLSP.bestScore.get("prune");

  function updateHud() {
    if (opts.onScoreChange) opts.onScoreChange({
      accuracy: state.accuracy,
      sparsity: state.sparsity,
      pruned: state.pruned,
      total: state.total,
      best: best
    });
  }

  function pruneWeight(w) {
    if (w.pruned || state.over) return;
    w.pruned = true;
    state.pruned++;
    removedImportance += w.importance;
    state.accuracy = 100 * (1 - removedImportance / totalImportance);
    state.sparsity = (state.pruned / state.total) * 100;
    if (state.accuracy < ACCURACY_THRESHOLD) {
      state.over = true;
      var finalSparsity = Math.round(state.sparsity);
      if (finalSparsity > best) {
        best = finalSparsity;
        MLSP.bestScore.set("prune", best);
      }
      if (opts.onGameOver) opts.onGameOver({
        accuracy: state.accuracy,
        sparsity: state.sparsity,
        finalSparsity: finalSparsity,
        best: best
      });
    }
    updateHud();
  }

  function findHover(px, py) {
    var bestIdx = -1;
    var bestDist = HOVER_RADIUS_PX;
    for (var i = 0; i < weights.length; i++) {
      var w = weights[i];
      if (w.pruned) continue;
      var d = MLSP.distToSegment(px, py, w.from.x, w.from.y, w.to.x, w.to.y);
      if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    return bestIdx;
  }

  // Input handling
  function onMove(e) {
    if (state.over) return;
    var p = MLSP.canvasPoint(canvas, e);
    state.hoverIdx = findHover(p.x, p.y);
    canvas.style.cursor = state.hoverIdx >= 0 ? "pointer" : "default";
  }
  function onDown(e) {
    e.preventDefault();
    if (state.over) {
      if (opts.onRetry) opts.onRetry();
      return;
    }
    var p = MLSP.canvasPoint(canvas, e);
    var idx = findHover(p.x, p.y);
    if (idx >= 0) pruneWeight(weights[idx]);
  }
  canvas.addEventListener("mousemove", onMove);
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("touchmove", function(e){ e.preventDefault(); }, { passive: false });

  // Rendering
  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Title + subtitle
    ctx.fillStyle = "#333";
    ctx.font = "bold 15px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Prune the network", W / 2, 26);
    ctx.font = "11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#777";
    ctx.fillText("Click faint weights. Keep accuracy above " + ACCURACY_THRESHOLD + "%.", W / 2, 44);

    // Edges
    for (var i = 0; i < weights.length; i++) {
      var w = weights[i];
      if (w.pruned) {
        if (w.fadeAlpha > 0) w.fadeAlpha = Math.max(0, w.fadeAlpha - 0.06);
        if (w.fadeAlpha === 0) continue;
        ctx.globalAlpha = w.fadeAlpha * 0.25;
        ctx.strokeStyle = "#bbb";
        ctx.lineWidth = 1;
      } else {
        var alpha = Math.max(0.08, Math.min(1, w.magnitude * 0.55));
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = (i === state.hoverIdx) ? "#a31f34" : "#4a90c4";
        ctx.lineWidth = (i === state.hoverIdx) ? 2.5 : 1.4;
      }
      ctx.beginPath();
      ctx.moveTo(w.from.x, w.from.y);
      ctx.lineTo(w.to.x, w.to.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Neurons
    for (var ni = 0; ni < neurons.length; ni++) {
      var n = neurons[ni];
      ctx.fillStyle = "#cfe2f3";
      ctx.strokeStyle = "#4a90c4";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(n.x, n.y, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    // Layer labels
    ctx.fillStyle = "#999";
    ctx.font = "10px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("input",  neurons[0].x, H - 40);
    ctx.fillText("hidden", neurons[LAYERS[0]].x, H - 40);
    ctx.fillText("output", neurons[LAYERS[0] + LAYERS[1]].x, H - 40);

    // Tooltip
    if (state.hoverIdx >= 0 && !state.over) {
      var hw = weights[state.hoverIdx];
      var mx = (hw.from.x + hw.to.x) / 2;
      var my = (hw.from.y + hw.to.y) / 2;
      var label = "|w| = " + hw.magnitude.toFixed(2);
      ctx.font = "bold 11px 'Helvetica Neue', Arial, sans-serif";
      var tw = ctx.measureText(label).width;
      var padX = 6, padY = 3;
      var bx = mx - tw / 2 - padX;
      var by = my - 22 - padY;
      var bw = tw + padX * 2;
      var bh = 14 + padY * 2;
      ctx.fillStyle = "rgba(255,255,255,0.95)";
      ctx.strokeStyle = "#bbb";
      ctx.lineWidth = 1;
      MLSP.roundRect(ctx, bx, by, bw, bh, 3);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#333";
      ctx.textAlign = "center";
      ctx.fillText(label, mx, my - 12);
    }

    // Inline on-canvas HUD (accuracy bar + sparsity)
    drawStatusBar();

    // Game over overlay
    if (state.over) {
      ctx.fillStyle = "rgba(255,255,255,0.92)";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#a31f34";
      ctx.font = "bold 22px 'Helvetica Neue', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Accuracy collapsed", W / 2, H / 2 - 20);
      ctx.fillStyle = "#333";
      ctx.font = "14px 'Helvetica Neue', Arial, sans-serif";
      ctx.fillText(
        "You pruned " + Math.round(state.sparsity) + "% of weights. Best: " + best + "%.",
        W / 2, H / 2 + 6
      );
      ctx.fillStyle = "#777";
      ctx.font = "italic 11px 'Helvetica Neue', Arial, sans-serif";
      ctx.fillText("Tap or press R to retry", W / 2, H / 2 + 30);
    }

    requestAnimationFrame(draw);
  }

  function drawStatusBar() {
    // Accuracy bar across the top
    var barX = 20, barY = H - 18, barW = W - 40, barH = 8;
    ctx.fillStyle = "#eee";
    MLSP.roundRect(ctx, barX, barY, barW, barH, 4);
    ctx.fill();
    var accW = barW * Math.max(0, Math.min(1, state.accuracy / 100));
    var accColor = state.accuracy >= 90 ? "#3d9e5a"
                  : state.accuracy >= 75 ? "#4a90c4"
                  : state.accuracy >= ACCURACY_THRESHOLD ? "#c87b2a"
                  : "#c44";
    ctx.fillStyle = accColor;
    MLSP.roundRect(ctx, barX, barY, accW, barH, 4);
    ctx.fill();

    ctx.font = "10px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#555";
    ctx.textAlign = "left";
    ctx.fillText("accuracy " + state.accuracy.toFixed(1) + "%", barX, barY - 4);
    ctx.textAlign = "right";
    ctx.fillText("sparsity " + state.sparsity.toFixed(0) + "%  ·  best " + best + "%", barX + barW, barY - 4);
  }

  // Keyboard retry
  window.addEventListener("keydown", function(e) {
    if (!state.over) return;
    if (!MLSP.inViewport(canvas)) return;
    if (e.key === "r" || e.key === "R" || e.key === "Enter" || e.code === "Space") {
      e.preventDefault();
      if (opts.onRetry) opts.onRetry();
    }
  });

  updateHud();
  draw();

  return {
    id: "prune",
    name: "Prune",
    ahaLabel: "You just discovered",
    ahaText: "Magnitude-based pruning. Most weights in a trained network are near-zero and contribute little — you can safely remove them. The hard part is the few small-looking weights that actually matter. This is the 'lottery ticket' intuition: a sparse sub-network is often enough to carry the full model's accuracy."
  };
};
