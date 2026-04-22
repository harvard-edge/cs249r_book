/* ============================================================
   MLSys Playground — Pulse Prune (v2)
   ------------------------------------------------------------
   Core insight: pruning and training are coupled. The network
   rewires around your cuts. What looks redundant at 30% sparsity
   becomes critical at 60%. Cut aggressively and you outpace
   recovery; cut thoughtfully and the network stays healthy.

   Three simultaneous beats keep the game reactive:
     - Fine-tune tick   (~500ms): weight magnitudes drift, accuracy
                                   ceiling adjusts, idle = staleness.
     - Inference pulse  (~900ms): a sample flows through, ticks the
                                   correct/attempted counter, misclass
                                   causes screenshake + red flash.
     - Player cuts      (any time): click a weight; good cuts combo,
                                     bad cuts shake + reset combo.

   Daily seed: everyone who plays today gets the same network.
   Score: max sparsity reached before accuracy collapses beyond
   recovery. Alltime and daily bests are persisted separately.
   ============================================================ */

window.MLSP = window.MLSP || {};
MLSP.games = MLSP.games || {};

MLSP.games.prune = function(canvas, opts) {
  opts = opts || {};
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;

  /* ----- Tunable parameters ----- */
  var LAYERS = [5, 8, 3];
  var ACCURACY_FLOOR = 60;               // game-over threshold
  var TICK_MS = 500;                     // fine-tune tick period
  var PULSE_MS = 900;                    // inference pulse period
  var PULSE_TRAVERSE_MS = 550;           // how long one pulse takes
  var HOVER_RADIUS = 9;
  var STALENESS_RATE = 0.04;             // per tick with no cuts
  var RECOVERY_RATE = 0.18;              // accuracy → ceiling per tick
  var COMBO_TICK_THRESHOLD = 3;          // low-mag across N ticks = combo

  /* ----- Seeded PRNG (mulberry32) for daily mode ----- */
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
  var seedString = opts.seed || today;
  var rand = mulberry32(hashString(seedString));
  function gauss() {
    var u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ----- Build network topology ----- */
  var neurons = [];
  var topM = 62, botM = 72, leftM = 80, rightM = 80;
  var innerW = W - leftM - rightM;
  var innerH = H - topM - botM;
  var layerDX = innerW / (LAYERS.length - 1);
  LAYERS.forEach(function(count, li) {
    var cellH = innerH / Math.max(count - 1, 1);
    var y0 = count === 1 ? (topM + innerH / 2) : topM;
    for (var i = 0; i < count; i++) {
      neurons.push({
        layer: li, idx: i,
        x: leftM + li * layerDX,
        y: y0 + i * cellH,
        pulse: 0   // transient glow when an inference sample passes
      });
    }
  });

  /* ----- Build weights with hidden importance ----- */
  var weights = [];
  for (var li = 0; li < LAYERS.length - 1; li++) {
    var from = neurons.filter(function(n){ return n.layer === li; });
    var to   = neurons.filter(function(n){ return n.layer === li + 1; });
    from.forEach(function(f) {
      to.forEach(function(t) {
        var baseMag = Math.abs(gauss());
        // True importance correlated with baseMag but with meaningful noise.
        // This is the CRUX of the game: magnitude is a lossy proxy for importance.
        var importance = Math.max(0.01, baseMag + gauss() * 0.35);
        weights.push({
          from: f, to: t,
          magnitude: baseMag,
          targetMag: baseMag,       // drift target, refreshed each tick
          importance: importance,   // hidden, constant
          pruned: false,
          fadeAlpha: 1.0,
          ticksSmall: 0,            // consecutive ticks below small threshold
          activation: 0             // glow during inference pulse traversal
        });
      });
    });
  }
  var totalImportance = weights.reduce(function(s,w){ return s + w.importance; }, 0);
  var lostImportance = 0;

  /* ----- Game state ----- */
  var state = {
    accuracy: 100,
    ceiling:  100,
    sparsity: 0,
    pruned: 0,
    total: weights.length,
    staleness: 0,      // 0..1, grows when no cuts, decays when cuts happen
    combo: 1,
    comboPeak: 1,
    correct: 0,
    attempts: 0,
    ticksSinceCut: 0,
    over: false,
    hoverIdx: -1,
    shakeAmt: 0,
    shakeT: 0,
    particles: [],
    floats: [],        // floating +number texts
    inferencePulse: null
  };

  var alltimeBest = MLSP.bestScore.get("prune");
  var dailyStore = readDailyBest();
  var dailyBest = (dailyStore.date === today) ? dailyStore.best : 0;

  function readDailyBest() {
    try {
      var raw = localStorage.getItem("mlsp-daily-prune");
      if (!raw) return { date: today, best: 0 };
      return JSON.parse(raw);
    } catch(e) { return { date: today, best: 0 }; }
  }
  function writeDailyBest(v) {
    try { localStorage.setItem("mlsp-daily-prune", JSON.stringify({ date: today, best: v })); }
    catch(e) {}
  }

  /* ----- Fine-tune tick: weights drift, ceiling adjusts ----- */
  function fineTuneTick() {
    // Refresh drift targets. Weights whose importance is high tend toward
    // higher magnitudes; redundant weights drift toward zero. Subtle noise
    // adds life to the visible magnitudes.
    for (var i = 0; i < weights.length; i++) {
      var w = weights[i];
      if (w.pruned) continue;

      // Gentle pull toward a magnitude that reflects importance,
      // scaled by current capacity so surviving weights "strengthen"
      // as sparsity grows.
      var capacity = 1 - lostImportance / totalImportance;
      var target = w.importance * (0.7 + 0.5 * (1 - capacity)) + gauss() * 0.05;
      w.targetMag = Math.max(0.02, target);

      w.magnitude += 0.22 * (w.targetMag - w.magnitude) + gauss() * 0.015;
      if (w.magnitude < 0) w.magnitude = 0;

      if (w.magnitude < 0.28) w.ticksSmall++;
      else w.ticksSmall = 0;
    }

    // Ceiling reflects remaining capacity with a concave falloff.
    var cap = 1 - lostImportance / totalImportance;
    var stalenessPenalty = 1 - 0.25 * state.staleness;
    state.ceiling = Math.max(0, 60 + 40 * Math.pow(cap, 0.55) * stalenessPenalty);

    // Accuracy drifts toward ceiling (recovery).
    var delta = state.ceiling - state.accuracy;
    state.accuracy += RECOVERY_RATE * delta;
    state.accuracy = Math.max(0, Math.min(100, state.accuracy));

    // Staleness: if no cuts for ~6 ticks, it grows.
    state.ticksSinceCut++;
    if (state.ticksSinceCut > 6) {
      state.staleness = Math.min(1, state.staleness + STALENESS_RATE);
    }

    // Game-over: accuracy below floor AND ceiling also below floor+3 = unrecoverable
    if (!state.over && state.accuracy < ACCURACY_FLOOR && state.ceiling < ACCURACY_FLOOR + 3) {
      endGame();
    }
  }

  /* ----- Inference pulse: sample flows through the network ----- */
  function spawnPulse() {
    // Choose an input neuron at random to originate the pulse
    var inputs = neurons.filter(function(n){ return n.layer === 0; });
    var outputs = neurons.filter(function(n){ return n.layer === LAYERS.length - 1; });
    var inIdx = Math.floor(rand() * inputs.length);
    // Simulated classification: probability of correctness scales with accuracy
    var correctProb = Math.max(0.02, state.accuracy / 100);
    var correct = rand() < correctProb;
    state.inferencePulse = {
      progress: 0,
      sourceIdx: inIdx,
      targetIdx: Math.floor(rand() * outputs.length),
      correct: correct,
      trailPoints: []
    };
  }

  function updatePulse(dt) {
    var p = state.inferencePulse;
    if (!p) return;
    p.progress += dt / PULSE_TRAVERSE_MS;

    // Activate weights along the pulse's path at current layer
    var nLayers = LAYERS.length;
    var pos = p.progress * (nLayers - 1);
    var currLayer = Math.floor(pos);
    var frac = pos - currLayer;

    // Make weights from currLayer to currLayer+1 glow more as pulse crosses
    if (currLayer < nLayers - 1) {
      for (var i = 0; i < weights.length; i++) {
        var w = weights[i];
        if (w.pruned) continue;
        if (w.from.layer === currLayer) {
          // Edges connected to the pulse "path" activate most
          var onPath = (w.from.idx === p.sourceIdx && currLayer === 0) ||
                       (w.to.layer === nLayers - 1 && w.to.idx === p.targetIdx) ||
                       (currLayer > 0 && currLayer < nLayers - 1);
          var boost = (onPath ? 0.8 : 0.3) * w.magnitude;
          w.activation = Math.min(1, w.activation + boost * 0.25);
        }
      }
    }

    // Pulse the source and target neurons
    var inputs  = neurons.filter(function(n){ return n.layer === 0; });
    var outputs = neurons.filter(function(n){ return n.layer === nLayers - 1; });
    if (p.progress < 0.1) inputs[p.sourceIdx].pulse = 1;
    if (p.progress > 0.88) outputs[p.targetIdx].pulse = 1;

    if (p.progress >= 1) {
      state.attempts++;
      if (p.correct) {
        state.correct++;
      } else {
        // Misclassification — visible consequence: small accuracy nudge + shake
        state.accuracy = Math.max(0, state.accuracy - 0.8);
        shake(3, 180);
        addFloat(outputs[p.targetIdx].x, outputs[p.targetIdx].y - 12, "✗", "#c44");
      }
      if (p.correct) {
        addFloat(outputs[p.targetIdx].x, outputs[p.targetIdx].y - 12, "✓", "#3d9e5a");
      }
      state.inferencePulse = null;
    }
  }

  /* ----- Player action: prune a weight ----- */
  function pruneWeight(w) {
    if (w.pruned || state.over) return;
    w.pruned = true;
    state.pruned++;
    state.sparsity = (state.pruned / state.total) * 100;
    lostImportance += w.importance;

    var severity = (w.importance / totalImportance) * 100; // % of total importance
    state.accuracy = Math.max(0, state.accuracy - severity * 0.85);
    state.ticksSinceCut = 0;
    state.staleness = Math.max(0, state.staleness - 0.35);

    // Combo: was this weight small for enough consecutive ticks?
    var wasSmallStable = w.ticksSmall >= COMBO_TICK_THRESHOLD;
    var mx = (w.from.x + w.to.x) / 2;
    var my = (w.from.y + w.to.y) / 2;
    if (wasSmallStable) {
      state.combo++;
      if (state.combo > state.comboPeak) state.comboPeak = state.combo;
      addFloat(mx, my - 6, "+1  ×" + state.combo, "#3d9e5a");
      burst(mx, my, "#3d9e5a", 6);
    } else {
      if (severity > 3) {
        // Bad cut — critical weight
        shake(7, 260);
        addFloat(mx, my - 6, "−" + severity.toFixed(1) + "%", "#c44");
        burst(mx, my, "#c44", 10);
        state.combo = 1;
      } else {
        // Marginal cut — middling weight, neutral
        addFloat(mx, my - 6, "−" + severity.toFixed(1) + "%", "#c87b2a");
        burst(mx, my, "#c87b2a", 5);
        state.combo = 1;
      }
    }

    // Check immediate collapse (pruned enough critical mass that ceiling can't save it)
    if (state.accuracy < ACCURACY_FLOOR && state.ceiling < ACCURACY_FLOOR + 3) {
      endGame();
    }
  }

  function endGame() {
    state.over = true;
    var final = Math.round(state.sparsity);
    if (final > alltimeBest) {
      alltimeBest = final;
      MLSP.bestScore.set("prune", final);
    }
    if (final > dailyBest) {
      dailyBest = final;
      writeDailyBest(final);
    }
    if (opts.onGameOver) opts.onGameOver({
      sparsity: state.sparsity,
      accuracy: state.accuracy,
      correct: state.correct,
      attempts: state.attempts,
      comboPeak: state.comboPeak,
      finalSparsity: final,
      alltimeBest: alltimeBest,
      dailyBest: dailyBest,
      date: today
    });
  }

  /* ----- Feedback helpers: shake, burst, float text ----- */
  function shake(amp, ms) { state.shakeAmt = Math.max(state.shakeAmt, amp); state.shakeT = Math.max(state.shakeT, ms); }
  function burst(x, y, color, n) {
    for (var i = 0; i < n; i++) {
      var ang = rand() * Math.PI * 2;
      var spd = 1 + rand() * 2.4;
      state.particles.push({
        x: x, y: y,
        vx: Math.cos(ang) * spd,
        vy: Math.sin(ang) * spd - 0.4,
        age: 0, maxAge: 560,
        color: color
      });
    }
  }
  function addFloat(x, y, text, color) {
    state.floats.push({ x: x, y: y, text: text, color: color, age: 0, maxAge: 900 });
  }

  /* ----- Input handling ----- */
  function findHover(px, py) {
    var bestIdx = -1;
    var bestDist = HOVER_RADIUS;
    for (var i = 0; i < weights.length; i++) {
      var w = weights[i];
      if (w.pruned) continue;
      var d = MLSP.distToSegment(px, py, w.from.x, w.from.y, w.to.x, w.to.y);
      if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    return bestIdx;
  }
  canvas.addEventListener("mousemove", function(e) {
    if (state.over) return;
    var p = MLSP.canvasPoint(canvas, e);
    state.hoverIdx = findHover(p.x, p.y);
    canvas.style.cursor = state.hoverIdx >= 0 ? "pointer" : "default";
  });
  canvas.addEventListener("pointerdown", function(e) {
    e.preventDefault();
    if (state.over) {
      if (opts.onRetry) opts.onRetry();
      return;
    }
    var p = MLSP.canvasPoint(canvas, e);
    var idx = findHover(p.x, p.y);
    if (idx >= 0) pruneWeight(weights[idx]);
  });
  canvas.addEventListener("touchmove", function(e){ e.preventDefault(); }, { passive: false });
  window.addEventListener("keydown", function(e) {
    if (!state.over || !MLSP.inViewport(canvas)) return;
    if (e.key === "r" || e.key === "R" || e.key === "Enter" || e.code === "Space") {
      e.preventDefault();
      if (opts.onRetry) opts.onRetry();
    }
  });

  /* ----- Main loop ----- */
  var lastTime = 0;
  var tickAcc = 0;
  var pulseAcc = 0;
  function frame(now) {
    var dt = lastTime ? (now - lastTime) : 16;
    lastTime = now;
    if (dt > 100) dt = 100; // cap after tab was backgrounded

    if (!state.over) {
      tickAcc += dt;
      while (tickAcc >= TICK_MS) {
        fineTuneTick();
        tickAcc -= TICK_MS;
      }
      pulseAcc += dt;
      if (pulseAcc >= PULSE_MS && !state.inferencePulse) {
        spawnPulse();
        pulseAcc = 0;
      }
      updatePulse(dt);

      // Decay weight activations
      for (var i = 0; i < weights.length; i++) {
        if (!weights[i].pruned) weights[i].activation *= 0.82;
      }
      for (var ni = 0; ni < neurons.length; ni++) {
        neurons[ni].pulse *= 0.88;
      }

      // Feedback layer updates
      state.shakeT = Math.max(0, state.shakeT - dt);
      if (state.shakeT === 0) state.shakeAmt = 0;
      for (var p of state.particles) {
        p.x += p.vx; p.y += p.vy;
        p.vy += 0.16;
        p.age += dt;
      }
      state.particles = state.particles.filter(function(pp){ return pp.age < pp.maxAge; });
      for (var f of state.floats) {
        f.age += dt; f.y -= dt * 0.035;
      }
      state.floats = state.floats.filter(function(fl){ return fl.age < fl.maxAge; });

      if (opts.onScoreChange) opts.onScoreChange({
        accuracy: state.accuracy,
        sparsity: state.sparsity,
        correct: state.correct,
        attempts: state.attempts,
        combo: state.combo,
        comboPeak: state.comboPeak,
        alltimeBest: alltimeBest,
        dailyBest: dailyBest
      });
    }

    draw();
    requestAnimationFrame(frame);
  }

  /* ----- Rendering ----- */
  function draw() {
    var sx = 0, sy = 0;
    if (state.shakeAmt > 0) {
      sx = (rand() - 0.5) * state.shakeAmt;
      sy = (rand() - 0.5) * state.shakeAmt;
    }
    ctx.save();
    ctx.translate(sx, sy);
    ctx.clearRect(-20, -20, W + 40, H + 40);

    /* Title + subtitle */
    ctx.fillStyle = "#333";
    ctx.font = "bold 15px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Pulse Prune", W / 2, 22);
    ctx.font = "10.5px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#888";
    ctx.fillText("cut weights that stay dim · watch the network rewire around you", W / 2, 38);

    /* Edges */
    for (var i = 0; i < weights.length; i++) {
      var w = weights[i];
      if (w.pruned) {
        if (w.fadeAlpha > 0) w.fadeAlpha = Math.max(0, w.fadeAlpha - 0.07);
        if (w.fadeAlpha === 0) continue;
        ctx.globalAlpha = w.fadeAlpha * 0.18;
        ctx.strokeStyle = "#bbb";
        ctx.lineWidth = 1;
      } else {
        var baseAlpha = Math.max(0.06, Math.min(1, w.magnitude * 0.55));
        var active = w.activation;
        ctx.globalAlpha = Math.min(1, baseAlpha + active * 0.5);
        ctx.strokeStyle = (i === state.hoverIdx)
          ? "#a31f34"
          : (active > 0.15 ? "#3d9e5a" : "#4a90c4");
        ctx.lineWidth = (i === state.hoverIdx ? 2.6 : 1.2 + active * 1.2);
      }
      ctx.beginPath();
      ctx.moveTo(w.from.x, w.from.y);
      ctx.lineTo(w.to.x, w.to.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    /* Inference pulse sample traveling L→R */
    if (state.inferencePulse) {
      var p = state.inferencePulse;
      var inputs = neurons.filter(function(n){ return n.layer === 0; });
      var outputs = neurons.filter(function(n){ return n.layer === LAYERS.length - 1; });
      var x1 = inputs[p.sourceIdx].x, y1 = inputs[p.sourceIdx].y;
      var x2 = outputs[p.targetIdx].x, y2 = outputs[p.targetIdx].y;
      var pe = p.progress;
      var dotX = x1 + (x2 - x1) * pe;
      var dotY = y1 + (y2 - y1) * pe;
      var glow = ctx.createRadialGradient(dotX, dotY, 0, dotX, dotY, 10);
      glow.addColorStop(0, "rgba(163,31,52,0.8)");
      glow.addColorStop(1, "rgba(163,31,52,0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(dotX, dotY, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#a31f34";
      ctx.beginPath();
      ctx.arc(dotX, dotY, 3.2, 0, Math.PI * 2);
      ctx.fill();
    }

    /* Neurons */
    for (var ni = 0; ni < neurons.length; ni++) {
      var n = neurons[ni];
      ctx.fillStyle = "#cfe2f3";
      ctx.strokeStyle = "#4a90c4";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(n.x, n.y, 9, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      if (n.pulse > 0.05) {
        ctx.globalAlpha = n.pulse;
        ctx.strokeStyle = "#a31f34";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(n.x, n.y, 9 + n.pulse * 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    /* Layer labels */
    ctx.fillStyle = "#999";
    ctx.font = "10px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("input",  neurons[0].x, H - 44);
    ctx.fillText("hidden", neurons[LAYERS[0]].x, H - 44);
    ctx.fillText("output", neurons[LAYERS[0] + LAYERS[1]].x, H - 44);

    /* Hover tooltip */
    if (state.hoverIdx >= 0 && !state.over) {
      var hw = weights[state.hoverIdx];
      var mx = (hw.from.x + hw.to.x) / 2;
      var my = (hw.from.y + hw.to.y) / 2;
      var label = "|w| = " + hw.magnitude.toFixed(2) +
                  (hw.ticksSmall >= COMBO_TICK_THRESHOLD ? "  ✨ stable-small" : "");
      ctx.font = "bold 11px 'Helvetica Neue', Arial, sans-serif";
      var tw = ctx.measureText(label).width;
      var padX = 6, padY = 3;
      var bx = mx - tw / 2 - padX, by = my - 24 - padY;
      var bw = tw + padX * 2, bh = 14 + padY * 2;
      ctx.fillStyle = "rgba(255,255,255,0.96)";
      ctx.strokeStyle = "#bbb";
      ctx.lineWidth = 1;
      MLSP.roundRect(ctx, bx, by, bw, bh, 3);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#333";
      ctx.textAlign = "center";
      ctx.fillText(label, mx, my - 14);
    }

    /* Particles */
    for (var pi = 0; pi < state.particles.length; pi++) {
      var pa = state.particles[pi];
      ctx.globalAlpha = Math.max(0, 1 - pa.age / pa.maxAge);
      ctx.fillStyle = pa.color;
      ctx.beginPath();
      ctx.arc(pa.x, pa.y, 2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    /* Floating texts */
    for (var fi = 0; fi < state.floats.length; fi++) {
      var ff = state.floats[fi];
      ctx.globalAlpha = Math.max(0, 1 - ff.age / ff.maxAge);
      ctx.fillStyle = ff.color;
      ctx.font = "bold 12px 'Helvetica Neue', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(ff.text, ff.x, ff.y);
    }
    ctx.globalAlpha = 1;

    /* HUD — bars + stats */
    drawHud();

    /* Game-over overlay */
    if (state.over) drawGameOver();

    ctx.restore();
  }

  function drawHud() {
    // Accuracy bar
    var barX = 18, barY = H - 24, barW = W * 0.55, barH = 8;
    ctx.fillStyle = "#eee";
    MLSP.roundRect(ctx, barX, barY, barW, barH, 4); ctx.fill();
    var accFrac = Math.max(0, Math.min(1, state.accuracy / 100));
    var accColor = state.accuracy >= 90 ? "#3d9e5a"
                 : state.accuracy >= 75 ? "#4a90c4"
                 : state.accuracy >= ACCURACY_FLOOR ? "#c87b2a"
                 : "#c44";
    ctx.fillStyle = accColor;
    MLSP.roundRect(ctx, barX, barY, barW * accFrac, barH, 4); ctx.fill();
    // Ceiling tick mark
    var ceilX = barX + barW * Math.max(0, Math.min(1, state.ceiling / 100));
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(ceilX, barY - 3); ctx.lineTo(ceilX, barY + barH + 3);
    ctx.stroke();

    ctx.font = "10px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#555";
    ctx.textAlign = "left";
    ctx.fillText("accuracy " + state.accuracy.toFixed(1) + "%  · ceiling " + state.ceiling.toFixed(0) + "%", barX, barY - 4);

    // Right side: sparsity + combo + classification tally
    ctx.textAlign = "right";
    var rightText = "sparsity " + state.sparsity.toFixed(0) + "%  · combo ×" + state.combo +
                    "  · " + state.correct + "/" + state.attempts;
    ctx.fillText(rightText, W - 18, barY - 4);
    // Daily/alltime best line
    ctx.fillStyle = "#999";
    ctx.font = "9px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("today best " + dailyBest + "%  · alltime " + alltimeBest + "%", W - 18, barY + barH + 12);
    ctx.textAlign = "left";
    ctx.fillText("daily seed " + today, 18, barY + barH + 12);
  }

  function drawGameOver() {
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#a31f34";
    ctx.font = "bold 22px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Training diverged", W / 2, H / 2 - 32);
    ctx.fillStyle = "#333";
    ctx.font = "14px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText(
      "sparsity " + Math.round(state.sparsity) + "%  · combo peak ×" + state.comboPeak +
      "  · " + state.correct + "/" + state.attempts + " correct",
      W / 2, H / 2 - 6
    );
    ctx.fillStyle = "#555";
    ctx.font = "12px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText(
      "today best " + dailyBest + "%  · alltime " + alltimeBest + "%",
      W / 2, H / 2 + 16
    );
    ctx.fillStyle = "#777";
    ctx.font = "italic 11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("tap or press R to retry", W / 2, H / 2 + 40);
  }

  requestAnimationFrame(frame);

  return {
    id: "prune",
    name: "Pulse Prune",
    ahaLabel: "You just felt",
    ahaText: "The network rewiring around your cuts. Magnitude-based pruning works because most weights are redundant — but redundancy is dynamic: what looks safe to remove at 30% sparsity is often load-bearing at 60%. That's why production pruning is iterative (prune a little, fine-tune, repeat) rather than one-shot. Your fingers just played through the core intuition behind the lottery-ticket hypothesis (Frankle & Carbin 2018) and gradual magnitude pruning (Zhu & Gupta 2017).",
    buildShareText: function(result) {
      return "MLSys Playground · Pulse Prune · " + today + "\n" +
             "pruned " + result.finalSparsity + "% of weights · combo peak ×" + result.comboPeak + "\n" +
             "can you beat it? → https://mlsysbook.ai/games/prune/";
    }
  };
};
