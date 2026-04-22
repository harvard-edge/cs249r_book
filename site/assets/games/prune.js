/* ============================================================
   MLSys Playground — Pulse Prune (v5, simplified)
   ------------------------------------------------------------
   Simple arcade: click dim weights to prune them.
   Race a 45-second timer. Hit 60% sparsity without accuracy
   dropping below 50%. That's it. No simulation.
   ============================================================ */

window.MLSP = window.MLSP || {};
MLSP.games = MLSP.games || {};

MLSP.games.prune = function(canvas, opts) {
  opts = opts || {};
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;

  var TARGET_SPARSITY = 60;
  var ACCURACY_FLOOR = 50;
  var TIME_LIMIT_MS = 45000;
  var LAYERS = [4, 6, 3];

  /* Seeded PRNG for daily mode */
  function hashString(s) {
    var h = 2166136261 >>> 0;
    for (var i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return h;
  }
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function() {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  var today = new Date().toISOString().slice(0, 10);
  var rand = mulberry32(hashString(today));

  /* Build neurons */
  var neurons = [];
  var topM = 70, botM = 90, leftM = 90, rightM = 90;
  var innerW = W - leftM - rightM;
  var innerH = H - topM - botM;
  var layerDX = innerW / (LAYERS.length - 1);
  LAYERS.forEach(function(count, li) {
    var cellH = innerH / Math.max(count - 1, 1);
    for (var i = 0; i < count; i++) {
      neurons.push({
        layer: li, idx: i,
        x: leftM + li * layerDX,
        y: topM + i * cellH,
        pulse: 0
      });
    }
  });

  /* Build weights — magnitude and importance fixed at init */
  var weights = [];
  for (var li = 0; li < LAYERS.length - 1; li++) {
    var from = neurons.filter(function(n){ return n.layer === li; });
    var to   = neurons.filter(function(n){ return n.layer === li + 1; });
    from.forEach(function(f) {
      to.forEach(function(t) {
        var mag = Math.pow(rand(), 1.6); // skewed toward low — many dim weights
        var imp = Math.max(0.01, mag + (rand() - 0.5) * 0.15);
        weights.push({
          from: f, to: t,
          magnitude: mag,
          importance: imp,
          pruned: false,
          activation: 0
        });
      });
    });
  }
  var totalImportance = weights.reduce(function(s,w){ return s + w.importance; }, 0);

  var state = {
    accuracy: 100,
    sparsity: 0,
    pruned: 0,
    total: weights.length,
    removedImp: 0,
    timeLeft: TIME_LIMIT_MS,
    over: false,
    won: false,
    hoverIdx: -1,
    shakeAmt: 0, shakeT: 0,
    particles: [], floats: [],
    inferencePulse: null,
    pulseCooldown: 1200
  };

  var alltimeBest = MLSP.bestScore.get("prune");

  /* Player cut */
  function pruneWeight(w) {
    if (w.pruned || state.over) return;
    w.pruned = true;
    state.pruned++;
    state.sparsity = (state.pruned / state.total) * 100;
    state.removedImp += w.importance;
    state.accuracy = Math.max(0, 100 * (1 - state.removedImp / totalImportance));

    var mx = (w.from.x + w.to.x) / 2;
    var my = (w.from.y + w.to.y) / 2;
    var isBright = w.magnitude > 0.45;
    if (isBright) {
      shake(7, 260);
      addFloat(mx, my - 6, "−" + (w.importance / totalImportance * 100).toFixed(1) + "% critical!", "#c44");
      burst(mx, my, "#c44", 10);
    } else {
      addFloat(mx, my - 6, "+1", "#3d9e5a");
      burst(mx, my, "#3d9e5a", 6);
    }

    // Win condition
    if (state.sparsity >= TARGET_SPARSITY && state.accuracy >= ACCURACY_FLOOR && !state.over) {
      state.over = true;
      state.won = true;
      var outs = neurons.filter(function(n){ return n.layer === LAYERS.length - 1; });
      for (var oi = 0; oi < outs.length; oi++) burst(outs[oi].x, outs[oi].y, "#3d9e5a", 20);
      endGame();
    } else if (state.accuracy < ACCURACY_FLOOR && !state.over) {
      state.over = true;
      state.won = false;
      endGame();
    }
  }

  function endGame() {
    var final = Math.round(state.sparsity);
    if (state.won && final > alltimeBest) {
      alltimeBest = final;
      MLSP.bestScore.set("prune", final);
    }
    if (opts.onGameOver) opts.onGameOver({
      sparsity: state.sparsity,
      accuracy: state.accuracy,
      finalSparsity: final,
      won: state.won,
      alltimeBest: alltimeBest,
      date: today,
      timeUsed: Math.round((TIME_LIMIT_MS - state.timeLeft) / 1000)
    });
  }

  /* Inference pulse (visual only — traces a random unpruned path) */
  function spawnPulse() {
    var inputs = neurons.filter(function(n){ return n.layer === 0; });
    var src = inputs[Math.floor(rand() * inputs.length)];
    var legs = [];
    var cur = src;
    for (var li = 1; li < LAYERS.length; li++) {
      var cands = [];
      for (var i = 0; i < weights.length; i++) {
        if (!weights[i].pruned && weights[i].from === cur && weights[i].to.layer === li) cands.push(weights[i]);
      }
      if (cands.length === 0) break;
      var chosen = cands[Math.floor(rand() * cands.length)];
      legs.push(chosen);
      cur = chosen.to;
    }
    state.inferencePulse = { legs: legs, currentLeg: 0, progress: 0 };
    src.pulse = 1;
  }

  function updatePulse(dt) {
    var p = state.inferencePulse;
    if (!p) return;
    if (p.legs.length === 0) { state.inferencePulse = null; return; }
    p.progress += dt / 700;
    p.legs[p.currentLeg].activation = Math.min(1, p.legs[p.currentLeg].activation + 0.25);
    if (p.progress >= 1) {
      p.legs[p.currentLeg].to.pulse = 1;
      p.currentLeg++;
      p.progress = 0;
      if (p.currentLeg >= p.legs.length) state.inferencePulse = null;
    }
  }

  /* Feedback helpers */
  function shake(a, ms) { state.shakeAmt = Math.max(state.shakeAmt, a); state.shakeT = Math.max(state.shakeT, ms); }
  function burst(x, y, color, n) {
    for (var i = 0; i < n; i++) {
      var ang = rand() * Math.PI * 2;
      var spd = 1 + rand() * 2.2;
      state.particles.push({ x: x, y: y, vx: Math.cos(ang)*spd, vy: Math.sin(ang)*spd - 0.4, age: 0, maxAge: 700, color: color });
    }
  }
  function addFloat(x, y, text, color) {
    state.floats.push({ x: x, y: y, text: text, color: color, age: 0, maxAge: 1200 });
  }

  /* Input */
  function findHover(px, py) {
    var bi = -1, bd = 10;
    for (var i = 0; i < weights.length; i++) {
      if (weights[i].pruned) continue;
      var d = MLSP.distToSegment(px, py, weights[i].from.x, weights[i].from.y, weights[i].to.x, weights[i].to.y);
      if (d < bd) { bd = d; bi = i; }
    }
    return bi;
  }
  canvas.addEventListener("mousemove", function(e) {
    if (state.over) return;
    var p = MLSP.canvasPoint(canvas, e);
    state.hoverIdx = findHover(p.x, p.y);
    canvas.style.cursor = state.hoverIdx >= 0 ? "pointer" : "default";
  });
  canvas.addEventListener("pointerdown", function(e) {
    e.preventDefault();
    if (state.over) { if (opts.onRetry) opts.onRetry(); return; }
    var p = MLSP.canvasPoint(canvas, e);
    var idx = findHover(p.x, p.y);
    if (idx >= 0) pruneWeight(weights[idx]);
  });
  canvas.addEventListener("touchmove", function(e){ e.preventDefault(); }, { passive: false });
  window.addEventListener("keydown", function(e) {
    if (!MLSP.inViewport(canvas)) return;
    if (e.key === "r" || e.key === "R") { e.preventDefault(); if (opts.onRetry) opts.onRetry(); }
  });

  /* Main loop */
  var lastTime = 0;
  function frame(now) {
    var dt = lastTime ? (now - lastTime) : 16;
    lastTime = now;
    if (dt > 100) dt = 100;

    if (!state.over) {
      state.timeLeft -= dt;
      if (state.timeLeft <= 0) {
        state.timeLeft = 0;
        state.over = true;
        state.won = state.sparsity >= TARGET_SPARSITY && state.accuracy >= ACCURACY_FLOOR;
        endGame();
      }
      if (!state.inferencePulse) {
        state.pulseCooldown -= dt;
        if (state.pulseCooldown <= 0) { spawnPulse(); state.pulseCooldown = 1200; }
      } else updatePulse(dt);
    }

    for (var i = 0; i < weights.length; i++) if (!weights[i].pruned) weights[i].activation *= 0.92;
    for (var ni = 0; ni < neurons.length; ni++) neurons[ni].pulse *= 0.93;

    state.shakeT = Math.max(0, state.shakeT - dt);
    if (state.shakeT === 0) state.shakeAmt = 0;
    for (var pp of state.particles) { pp.x += pp.vx; pp.y += pp.vy; pp.vy += 0.16; pp.age += dt; }
    state.particles = state.particles.filter(function(x){ return x.age < x.maxAge; });
    for (var ff of state.floats) { ff.age += dt; ff.y -= dt * 0.035; }
    state.floats = state.floats.filter(function(x){ return x.age < x.maxAge; });

    if (opts.onScoreChange && !state.over) opts.onScoreChange({
      accuracy: state.accuracy,
      sparsity: state.sparsity,
      timeLeft: state.timeLeft,
      alltimeBest: alltimeBest
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
    ctx.fillText("Pulse Prune", W/2, 24);
    ctx.font = "10.5px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#888";
    ctx.fillText("click dim weights · keep bright ones · hit " + TARGET_SPARSITY + "% sparsity before time runs out", W/2, 42);

    // Edges
    for (var i = 0; i < weights.length; i++) {
      var w = weights[i];
      if (w.pruned) continue;
      var baseAlpha = Math.max(0.1, Math.min(1, w.magnitude * 0.7));
      var active = w.activation;
      ctx.globalAlpha = Math.min(1, baseAlpha + active * 0.5);
      ctx.strokeStyle = (i === state.hoverIdx) ? "#a31f34"
                      : (w.magnitude > 0.55 ? "#4a90c4"
                      : (w.magnitude > 0.3 ? "#88b4d8" : "#c0d4e8"));
      ctx.lineWidth = (i === state.hoverIdx) ? 3 : 1.2 + w.magnitude * 1.8;
      ctx.beginPath();
      ctx.moveTo(w.from.x, w.from.y);
      ctx.lineTo(w.to.x, w.to.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Inference pulse dot
    if (state.inferencePulse && state.inferencePulse.legs.length > 0) {
      var p = state.inferencePulse;
      var leg = p.legs[p.currentLeg];
      if (leg) {
        var t = Math.max(0, Math.min(1, p.progress));
        var dx = leg.from.x + (leg.to.x - leg.from.x) * t;
        var dy = leg.from.y + (leg.to.y - leg.from.y) * t;
        var g = ctx.createRadialGradient(dx, dy, 0, dx, dy, 12);
        g.addColorStop(0, "#4a90c4");
        g.addColorStop(1, "rgba(74,144,196,0)");
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(dx, dy, 12, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#4a90c4";
        ctx.beginPath(); ctx.arc(dx, dy, 3, 0, Math.PI * 2); ctx.fill();
      }
    }

    // Neurons
    for (var ni = 0; ni < neurons.length; ni++) {
      var n = neurons[ni];
      ctx.fillStyle = "#cfe2f3";
      ctx.strokeStyle = "#4a90c4";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(n.x, n.y, 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      if (n.pulse > 0.05) {
        ctx.globalAlpha = n.pulse;
        ctx.strokeStyle = "#a31f34"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(n.x, n.y, 9 + n.pulse * 8, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // Particles
    for (var pi = 0; pi < state.particles.length; pi++) {
      var pa = state.particles[pi];
      ctx.globalAlpha = Math.max(0, 1 - pa.age / pa.maxAge);
      ctx.fillStyle = pa.color;
      ctx.beginPath(); ctx.arc(pa.x, pa.y, 2, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Floats
    for (var fi = 0; fi < state.floats.length; fi++) {
      var ff = state.floats[fi];
      ctx.globalAlpha = Math.max(0, 1 - ff.age / ff.maxAge);
      ctx.fillStyle = ff.color;
      ctx.font = "bold 12px 'Helvetica Neue', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(ff.text, ff.x, ff.y);
    }
    ctx.globalAlpha = 1;

    drawHud();
    if (state.over) drawGameOver();
    ctx.restore();
  }

  function drawHud() {
    var barX = 20, barY = H - 28, barW = W - 40, barH = 8;
    ctx.fillStyle = "#eee";
    MLSP.roundRect(ctx, barX, barY, barW, barH, 4); ctx.fill();
    var accFrac = Math.max(0, Math.min(1, state.accuracy / 100));
    var accColor = state.accuracy >= 80 ? "#3d9e5a" : state.accuracy >= ACCURACY_FLOOR ? "#c87b2a" : "#c44";
    ctx.fillStyle = accColor;
    MLSP.roundRect(ctx, barX, barY, barW * accFrac, barH, 4); ctx.fill();
    var floorX = barX + barW * (ACCURACY_FLOOR / 100);
    ctx.strokeStyle = "#c44"; ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(floorX, barY - 2); ctx.lineTo(floorX, barY + barH + 2); ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = "11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#333";
    ctx.textAlign = "left";
    ctx.fillText("accuracy " + state.accuracy.toFixed(1) + "%", barX, barY - 5);
    ctx.textAlign = "center";
    var secs = Math.ceil(state.timeLeft / 1000);
    ctx.fillStyle = secs <= 10 ? "#c44" : "#333";
    ctx.font = "bold 13px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("⏱ " + secs + "s", W/2, barY - 5);
    ctx.fillStyle = "#333";
    ctx.font = "11px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("sparsity " + state.sparsity.toFixed(0) + "% / " + TARGET_SPARSITY + "%", W - 20, barY - 5);

    ctx.font = "9px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#999";
    ctx.textAlign = "left";
    ctx.fillText("alltime best " + alltimeBest + "%  ·  press R to retry", barX, barY + barH + 12);
  }

  function drawGameOver() {
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = state.won ? "#3d9e5a" : "#a31f34";
    ctx.font = "bold 24px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(state.won ? "🏆 network compressed!" : "accuracy collapsed", W/2, H/2 - 20);
    ctx.fillStyle = "#333";
    ctx.font = "14px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("sparsity " + Math.round(state.sparsity) + "%  ·  accuracy " + state.accuracy.toFixed(1) + "%", W/2, H/2 + 8);
    ctx.fillStyle = "#777";
    ctx.font = "italic 11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("tap or press R to retry", W/2, H/2 + 32);
  }

  requestAnimationFrame(frame);

  return {
    id: "prune",
    name: "Pulse Prune",
    ahaLabel: "You just played at",
    ahaText: "Magnitude-based pruning. Real neural networks have many redundant weights — the small ones. Production compression pipelines cut those dim weights to substantially shrink parameter counts (combined with quantization and distillation) while keeping accuracy intact. You just did that by eye.",
    buildShareText: function(result) {
      var tag = result.won ? "🏆 compressed" : "✗ diverged";
      return "MLSys Playground · Pulse Prune · " + today + "\n" +
             tag + " · sparsity " + result.finalSparsity + "% · accuracy " + result.accuracy.toFixed(0) + "%\n" +
             "play → mlsysbook.ai/games/prune/";
    }
  };
};
