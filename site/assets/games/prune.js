/* ============================================================
   MLSys Playground — Pulse Prune (v4)
   ------------------------------------------------------------
   Thesis: subtraction under scarcity. Every cut buys you sparsity
   and costs you potential paths through the network.

   v4 additions:
     - VISIBLE TARGET: reach 60% sparsity while accuracy stays up.
       Crosses the target = celebratory banner + bonus phase.
     - REWIRING VIZ: on each prune, sibling weights (sharing an
       endpoint) immediately brighten and thicken — the player
       literally sees the network redistribute capacity.
     - R RETRIES ANY TIME: previously R only worked on game-over.
       Now R starts a fresh run whenever the canvas is in view.

   Three simultaneous beats:
     - Fine-tune tick (~2 Hz): weight magnitudes drift toward
       importance-weighted targets.
     - Inference pulse (one at a time, ~500ms gap): picks a
       concrete path through unpruned weights; dies at broken
       boundaries.
     - Player cuts: patience (weight stayed dim for N ticks) →
       score. Load-bearing cuts → screenshake + flagged as
       mistake in failure skeleton.

   Daily seed: everyone playing today gets the same network.
   ============================================================ */

window.MLSP = window.MLSP || {};
MLSP.games = MLSP.games || {};

MLSP.games.prune = function(canvas, opts) {
  opts = opts || {};
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;

  /* ===== Tunables ===== */
  var LAYERS = [5, 8, 3];
  var ACCURACY_FLOOR = 60;
  var TARGET_SPARSITY = 60;             // win the day at this sparsity
  var TICK_MS = 500;
  var PULSE_LEG_MS = 850;
  var PULSE_GAP_MS = 500;
  var DEMO_MS = 3000;
  var HOVER_RADIUS = 9;
  var MAGNITUDE_JITTER = 0.006;
  var ACTIVATION_DECAY = 0.92;
  var NEURON_PULSE_DECAY = 0.94;
  var STALENESS_RATE = 0.04;
  var RECOVERY_RATE = 0.18;
  var PATIENCE_REWARD_TICKS = 5;

  /* ===== Seeded PRNG (mulberry32) for daily mode ===== */
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

  /* ===== Build neurons ===== */
  var neurons = [];
  var topM = 60, botM = 80, leftM = 90, rightM = 90;
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
        pulse: 0
      });
    }
  });

  /* ===== Build weights ===== */
  var weights = [];
  for (var li = 0; li < LAYERS.length - 1; li++) {
    var from = neurons.filter(function(n){ return n.layer === li; });
    var to   = neurons.filter(function(n){ return n.layer === li + 1; });
    from.forEach(function(f) {
      to.forEach(function(t) {
        var baseMag = Math.abs(gauss());
        var importance = Math.max(0.01, baseMag + gauss() * 0.35);
        weights.push({
          from: f, to: t,
          magnitude: baseMag,
          targetMag: baseMag,
          importance: importance,
          pruned: false,
          fadeAlpha: 1.0,
          ticksSmall: 0,
          activation: 0,
          wasCriticalCut: false
        });
      });
    });
  }
  var totalImportance = weights.reduce(function(s,w){ return s + w.importance; }, 0);
  var lostImportance = 0;

  /* ===== State ===== */
  var state = {
    accuracy: 100,
    ceiling:  100,
    sparsity: 0,
    pruned: 0,
    total: weights.length,
    staleness: 0,
    confidenceScore: 0,
    cuts: 0,
    correct: 0,
    attempts: 0,
    dropped: 0,
    ticksSinceCut: 0,
    over: false,
    targetReached: false,
    targetBannerTime: 0,
    hoverIdx: -1,
    shakeAmt: 0, shakeT: 0,
    particles: [], floats: [],
    inferencePulse: null,
    pulseCooldown: PULSE_GAP_MS + 600,
    demoMode: true,
    demoStart: 0,
    demoTargetIdx: -1,
    ghostCursor: { x: 50, y: 50, alpha: 0 },
    promptVisible: false
  };

  var alltimeBest = MLSP.bestScore.get("prune");
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
  var dailyStore = readDailyBest();
  var dailyBest = (dailyStore.date === today) ? dailyStore.best : 0;

  /* ===== Path picking ===== */
  function pickOutgoingUnpruned(fromNeuron, toLayer) {
    var candidates = [];
    for (var i = 0; i < weights.length; i++) {
      var w = weights[i];
      if (w.pruned) continue;
      if (w.from === fromNeuron && w.to.layer === toLayer) candidates.push(w);
    }
    if (candidates.length === 0) return null;
    var total = 0;
    for (var i = 0; i < candidates.length; i++) total += candidates[i].magnitude + 0.05;
    var r = rand() * total;
    for (var i = 0; i < candidates.length; i++) {
      r -= candidates[i].magnitude + 0.05;
      if (r <= 0) return candidates[i];
    }
    return candidates[candidates.length - 1];
  }

  /* ===== Fine-tune tick ===== */
  function fineTuneTick() {
    var cap = 1 - lostImportance / totalImportance;
    for (var i = 0; i < weights.length; i++) {
      var w = weights[i];
      if (w.pruned) continue;
      var target = w.importance * (0.7 + 0.5 * (1 - cap)) + gauss() * 0.04;
      w.targetMag = Math.max(0.02, target);
      w.magnitude += 0.18 * (w.targetMag - w.magnitude) + gauss() * MAGNITUDE_JITTER;
      if (w.magnitude < 0) w.magnitude = 0;
      if (w.magnitude < 0.28) w.ticksSmall++;
      else w.ticksSmall = 0;
    }
    var stalePenalty = 1 - 0.25 * state.staleness;
    state.ceiling = Math.max(0, 60 + 40 * Math.pow(cap, 0.55) * stalePenalty);
    state.accuracy += RECOVERY_RATE * (state.ceiling - state.accuracy);
    state.accuracy = Math.max(0, Math.min(100, state.accuracy));

    state.ticksSinceCut++;
    if (state.ticksSinceCut > 6) {
      state.staleness = Math.min(1, state.staleness + STALENESS_RATE);
    }
    if (!state.over && state.accuracy < ACCURACY_FLOOR && state.ceiling < ACCURACY_FLOOR + 3) {
      endGame();
    }
  }

  /* ===== Inference pulse ===== */
  function spawnPulse() {
    var inputs = neurons.filter(function(n){ return n.layer === 0; });
    var sourceIdx = Math.floor(rand() * inputs.length);
    var sourceNeuron = inputs[sourceIdx];

    var legs = [];
    var diedAt = null;
    var leg1 = pickOutgoingUnpruned(sourceNeuron, 1);
    if (!leg1) {
      diedAt = sourceNeuron;
    } else {
      legs.push(leg1);
      var leg2 = pickOutgoingUnpruned(leg1.to, 2);
      if (!leg2) diedAt = leg1.to;
      else legs.push(leg2);
    }
    var pathAlive = legs.length === LAYERS.length - 1;
    var correct = pathAlive && rand() < (state.accuracy / 100);

    state.inferencePulse = {
      sourceNeuron: sourceNeuron,
      legs: legs,
      diedAt: diedAt,
      pathAlive: pathAlive,
      correct: correct,
      currentLeg: 0,
      progress: 0,
      done: false,
      deathLingerMs: 0
    };
    sourceNeuron.pulse = 1;
  }

  function updatePulse(dt) {
    var p = state.inferencePulse;
    if (!p) return;
    if (p.done) {
      p.deathLingerMs += dt;
      if (p.deathLingerMs > 350) state.inferencePulse = null;
      return;
    }

    if (p.legs.length === 0) {
      p.deathLingerMs = 0;
      p.done = true;
      state.attempts++; state.dropped++;
      addFloat(p.sourceNeuron.x, p.sourceNeuron.y - 14, "✗ dropped", "#c44");
      shake(2, 140);
      return;
    }

    p.progress += dt / PULSE_LEG_MS;
    var currentWeight = p.legs[p.currentLeg];
    currentWeight.activation = Math.min(1, currentWeight.activation + 0.18);

    if (p.progress >= 1) {
      p.currentLeg++;
      p.progress = 0;
      if (p.currentLeg >= p.legs.length) {
        p.done = true;
        p.deathLingerMs = 0;
        state.attempts++;
        var outNeuron = p.legs[p.legs.length - 1].to;
        outNeuron.pulse = 1;
        if (p.pathAlive && p.correct) {
          state.correct++;
          addFloat(outNeuron.x, outNeuron.y - 14, "✓", "#3d9e5a");
        } else if (p.pathAlive) {
          state.accuracy = Math.max(0, state.accuracy - 0.6);
          shake(2, 140);
          addFloat(outNeuron.x, outNeuron.y - 14, "✗", "#c44");
        }
      } else {
        p.legs[p.currentLeg - 1].to.pulse = 1;
      }
    }

    if (!p.pathAlive && p.currentLeg >= p.legs.length && !p.done) {
      p.done = true;
      p.deathLingerMs = 0;
      state.attempts++; state.dropped++;
      addFloat(p.diedAt.x, p.diedAt.y - 14, "✗ dropped", "#c44");
      shake(2, 140);
    }
  }

  /* ===== Player cut ===== */
  function pruneWeight(w) {
    if (w.pruned || state.over) return;
    w.pruned = true;
    state.pruned++;
    state.cuts++;
    state.sparsity = (state.pruned / state.total) * 100;
    lostImportance += w.importance;

    var severity = (w.importance / totalImportance) * 100;
    state.accuracy = Math.max(0, state.accuracy - severity * 0.85);
    state.ticksSinceCut = 0;
    state.staleness = Math.max(0, state.staleness - 0.35);

    var patience = Math.min(w.ticksSmall, PATIENCE_REWARD_TICKS);
    var mx = (w.from.x + w.to.x) / 2;
    var my = (w.from.y + w.to.y) / 2;

    if (severity > 3) {
      w.wasCriticalCut = true;
      shake(7, 260);
      addFloat(mx, my - 6, "−" + severity.toFixed(1) + "% load-bearing", "#c44");
      burst(mx, my, "#c44", 10);
    } else if (patience >= 3) {
      state.confidenceScore += patience;
      addFloat(mx, my - 6, "+" + patience + " patience", "#3d9e5a");
      burst(mx, my, "#3d9e5a", 6);
    } else {
      addFloat(mx, my - 6, "+0 hasty", "#c87b2a");
      burst(mx, my, "#c87b2a", 5);
    }

    /* ----- REWIRING VIZ: siblings pick up capacity ----- */
    var redistribution = w.importance * 0.22;
    for (var si = 0; si < weights.length; si++) {
      var sibling = weights[si];
      if (sibling === w || sibling.pruned) continue;
      if (sibling.from === w.from || sibling.to === w.to) {
        sibling.activation = Math.min(1, sibling.activation + 0.6);
        sibling.targetMag = Math.min(1.3, sibling.targetMag + redistribution);
      }
    }

    /* ----- TARGET ACHIEVEMENT ----- */
    if (state.sparsity >= TARGET_SPARSITY && !state.targetReached) {
      state.targetReached = true;
      state.targetBannerTime = 0;
      var outs = neurons.filter(function(n){ return n.layer === LAYERS.length - 1; });
      for (var oi = 0; oi < outs.length; oi++) {
        burst(outs[oi].x, outs[oi].y, "#3d9e5a", 18);
      }
      addFloat(W/2, H/2 - 40, "🏆 " + TARGET_SPARSITY + "% sparsity!", "#3d9e5a");
      addFloat(W/2, H/2 - 20, "keep going for bonus", "#3d9e5a");
    }

    if (state.accuracy < ACCURACY_FLOOR && state.ceiling < ACCURACY_FLOOR + 3) {
      endGame();
    }
  }

  /* ===== Demo ===== */
  function setupDemo(now) {
    state.demoStart = now;
    var smallest = -1, smallestMag = Infinity;
    for (var i = 0; i < weights.length; i++) {
      if (weights[i].magnitude < smallestMag) {
        smallestMag = weights[i].magnitude;
        smallest = i;
      }
    }
    state.demoTargetIdx = smallest;
    var w = weights[smallest];
    state.ghostCursor = { x: 30, y: 30, alpha: 0 };
    state._demoTargetX = (w.from.x + w.to.x) / 2;
    state._demoTargetY = (w.from.y + w.to.y) / 2;
  }

  function updateDemo(now, dt) {
    var elapsed = now - state.demoStart;
    var t = Math.max(0, Math.min(1, elapsed / DEMO_MS));
    var targetX = state._demoTargetX;
    var targetY = state._demoTargetY;
    var ease = 1 - Math.pow(1 - t, 3);
    state.ghostCursor.x = 30 + (targetX - 30) * ease;
    state.ghostCursor.y = 30 + (targetY - 30) * ease;
    state.ghostCursor.alpha = Math.min(1, t * 4);
    if (state.demoTargetIdx >= 0 && elapsed > 800) {
      weights[state.demoTargetIdx].activation = Math.min(1, weights[state.demoTargetIdx].activation + 0.04);
    }
    if (elapsed > DEMO_MS * 0.75 && state.demoTargetIdx >= 0) {
      pruneWeight(weights[state.demoTargetIdx]);
      state.demoTargetIdx = -1;
    }
    if (elapsed >= DEMO_MS) {
      state.demoMode = false;
      state.promptVisible = true;
      setTimeout(function(){ state.promptVisible = false; }, 2000);
    }
  }

  /* ===== Game over ===== */
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
    var grid = buildEmojiGrid();
    if (opts.onGameOver) opts.onGameOver({
      sparsity: state.sparsity,
      accuracy: state.accuracy,
      correct: state.correct,
      attempts: state.attempts,
      dropped: state.dropped,
      confidenceScore: state.confidenceScore,
      cuts: state.cuts,
      finalSparsity: final,
      targetReached: state.targetReached,
      alltimeBest: alltimeBest,
      dailyBest: dailyBest,
      date: today,
      emojiGrid: grid
    });
  }

  /* ===== Emoji grid ===== */
  function buildEmojiGrid() {
    var rows = [];
    for (var i = 0; i < LAYERS[0]; i++) {
      var row = "";
      for (var j = 0; j < LAYERS[1]; j++) {
        var w = null;
        for (var k = 0; k < weights.length; k++) {
          if (weights[k].from.layer === 0 && weights[k].from.idx === i &&
              weights[k].to.layer === 1 && weights[k].to.idx === j) {
            w = weights[k];
            break;
          }
        }
        if (!w) row += "⬜";
        else if (w.pruned && w.wasCriticalCut) row += "🟥";
        else if (w.pruned) row += "⬛";
        else if (w.magnitude > 0.6) row += "🟦";
        else row += "🟩";
      }
      rows.push(row);
    }
    return rows.join("\n");
  }

  /* ===== Feedback helpers ===== */
  function shake(amp, ms) { state.shakeAmt = Math.max(state.shakeAmt, amp); state.shakeT = Math.max(state.shakeT, ms); }
  function burst(x, y, color, n) {
    for (var i = 0; i < n; i++) {
      var ang = rand() * Math.PI * 2;
      var spd = 1 + rand() * 2.4;
      state.particles.push({
        x: x, y: y, vx: Math.cos(ang) * spd, vy: Math.sin(ang) * spd - 0.4,
        age: 0, maxAge: 700, color: color
      });
    }
  }
  function addFloat(x, y, text, color) {
    state.floats.push({ x: x, y: y, text: text, color: color, age: 0, maxAge: 1400 });
  }

  /* ===== Input ===== */
  function findHover(px, py) {
    var bestIdx = -1, bestDist = HOVER_RADIUS;
    for (var i = 0; i < weights.length; i++) {
      var w = weights[i];
      if (w.pruned) continue;
      var d = MLSP.distToSegment(px, py, w.from.x, w.from.y, w.to.x, w.to.y);
      if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    return bestIdx;
  }
  canvas.addEventListener("mousemove", function(e) {
    if (state.over || state.demoMode) return;
    var p = MLSP.canvasPoint(canvas, e);
    state.hoverIdx = findHover(p.x, p.y);
    canvas.style.cursor = state.hoverIdx >= 0 ? "pointer" : "default";
  });
  canvas.addEventListener("pointerdown", function(e) {
    e.preventDefault();
    if (state.demoMode) return;
    if (state.over) {
      if (opts.onRetry) opts.onRetry();
      return;
    }
    state.promptVisible = false;
    var p = MLSP.canvasPoint(canvas, e);
    var idx = findHover(p.x, p.y);
    if (idx >= 0) pruneWeight(weights[idx]);
  });
  canvas.addEventListener("touchmove", function(e){ e.preventDefault(); }, { passive: false });

  // R retries any time (not just on game-over) — usability fix
  // Enter/Space only retries on game-over to avoid hijacking page scroll mid-play
  window.addEventListener("keydown", function(e) {
    if (!MLSP.inViewport(canvas)) return;
    if (e.key === "r" || e.key === "R") {
      e.preventDefault();
      if (opts.onRetry) opts.onRetry();
      return;
    }
    if (state.over && (e.key === "Enter" || e.code === "Space")) {
      e.preventDefault();
      if (opts.onRetry) opts.onRetry();
    }
  });

  /* ===== Main loop ===== */
  var lastTime = 0;
  var tickAcc = 0;
  function frame(now) {
    var dt = lastTime ? (now - lastTime) : 16;
    lastTime = now;
    if (dt > 100) dt = 100;
    if (state.demoStart === 0) setupDemo(now);

    if (state.demoMode) {
      updateDemo(now, dt);
    } else if (!state.over) {
      tickAcc += dt;
      while (tickAcc >= TICK_MS) { fineTuneTick(); tickAcc -= TICK_MS; }
      if (!state.inferencePulse) {
        state.pulseCooldown -= dt;
        if (state.pulseCooldown <= 0) {
          spawnPulse();
          state.pulseCooldown = PULSE_GAP_MS;
        }
      } else {
        updatePulse(dt);
      }
    }

    for (var i = 0; i < weights.length; i++) {
      if (!weights[i].pruned) weights[i].activation *= ACTIVATION_DECAY;
    }
    for (var ni = 0; ni < neurons.length; ni++) {
      neurons[ni].pulse *= NEURON_PULSE_DECAY;
    }

    if (state.targetReached && state.targetBannerTime < 3000) {
      state.targetBannerTime += dt;
    }

    if (!state.over) {
      state.shakeT = Math.max(0, state.shakeT - dt);
      if (state.shakeT === 0) state.shakeAmt = 0;
      for (var pp of state.particles) {
        pp.x += pp.vx; pp.y += pp.vy; pp.vy += 0.16; pp.age += dt;
      }
      state.particles = state.particles.filter(function(x){ return x.age < x.maxAge; });
      for (var ff of state.floats) { ff.age += dt; ff.y -= dt * 0.035; }
      state.floats = state.floats.filter(function(x){ return x.age < x.maxAge; });

      if (opts.onScoreChange) opts.onScoreChange({
        accuracy: state.accuracy,
        sparsity: state.sparsity,
        targetReached: state.targetReached,
        correct: state.correct,
        attempts: state.attempts,
        confidenceScore: state.confidenceScore,
        alltimeBest: alltimeBest,
        dailyBest: dailyBest
      });
    }

    draw();
    requestAnimationFrame(frame);
  }

  /* ===== Render ===== */
  function draw() {
    var sx = 0, sy = 0;
    if (state.shakeAmt > 0) {
      sx = (rand() - 0.5) * state.shakeAmt;
      sy = (rand() - 0.5) * state.shakeAmt;
    }
    ctx.save();
    ctx.translate(sx, sy);
    ctx.clearRect(-20, -20, W + 40, H + 40);

    // Title row with GOAL printed explicitly
    ctx.fillStyle = "#333";
    ctx.font = "bold 15px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Pulse Prune", W / 2, 22);
    ctx.font = "10.5px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#888";
    ctx.fillText(
      "goal: reach " + TARGET_SPARSITY + "% sparsity without accuracy collapsing",
      W / 2, 38
    );

    // Edges
    for (var i = 0; i < weights.length; i++) {
      var w = weights[i];
      if (w.pruned) {
        if (state.over) {
          ctx.globalAlpha = w.wasCriticalCut ? 0.65 : 0.35;
          ctx.strokeStyle = w.wasCriticalCut ? "#a31f34" : "#c44";
          ctx.lineWidth = w.wasCriticalCut ? 1.6 : 1.0;
          ctx.setLineDash([3, 3]);
        } else {
          if (w.fadeAlpha > 0) w.fadeAlpha = Math.max(0, w.fadeAlpha - 0.07);
          if (w.fadeAlpha === 0) continue;
          ctx.globalAlpha = w.fadeAlpha * 0.18;
          ctx.strokeStyle = "#bbb";
          ctx.lineWidth = 1;
        }
      } else {
        var baseAlpha = Math.max(0.06, Math.min(1, w.magnitude * 0.55));
        var active = w.activation;
        ctx.globalAlpha = Math.min(1, baseAlpha + active * 0.55);
        ctx.strokeStyle = (i === state.hoverIdx)
          ? "#a31f34"
          : (active > 0.15 ? "#3d9e5a" : "#4a90c4");
        ctx.lineWidth = (i === state.hoverIdx ? 2.6 : 1.2 + active * 1.4);
      }
      ctx.beginPath();
      ctx.moveTo(w.from.x, w.from.y);
      ctx.lineTo(w.to.x, w.to.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.globalAlpha = 1;

    // Inference pulse
    if (state.inferencePulse) {
      var p = state.inferencePulse;
      var dotX, dotY;
      var pulseColor = p.pathAlive ? "#4a90c4" : "#c44";
      if (p.legs.length === 0) {
        dotX = p.sourceNeuron.x;
        dotY = p.sourceNeuron.y;
      } else if (p.done) {
        var terminal = p.legs[p.legs.length - 1].to;
        dotX = p.diedAt ? p.diedAt.x : terminal.x;
        dotY = p.diedAt ? p.diedAt.y : terminal.y;
      } else {
        var leg = p.legs[p.currentLeg];
        var t = Math.max(0, Math.min(1, p.progress));
        dotX = leg.from.x + (leg.to.x - leg.from.x) * t;
        dotY = leg.from.y + (leg.to.y - leg.from.y) * t;
      }
      var glow = ctx.createRadialGradient(dotX, dotY, 0, dotX, dotY, 12);
      glow.addColorStop(0, pulseColor);
      glow.addColorStop(1, pulseColor + "00");
      ctx.fillStyle = glow;
      ctx.globalAlpha = p.done ? Math.max(0, 1 - p.deathLingerMs / 350) : 1;
      ctx.beginPath();
      ctx.arc(dotX, dotY, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = pulseColor;
      ctx.beginPath();
      ctx.arc(dotX, dotY, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Neurons
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

    // Layer labels
    ctx.fillStyle = "#999";
    ctx.font = "10px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("input",  neurons[0].x, H - 50);
    ctx.fillText("hidden", neurons[LAYERS[0]].x, H - 50);
    ctx.fillText("output", neurons[LAYERS[0] + LAYERS[1]].x, H - 50);

    // Hover tooltip
    if (state.hoverIdx >= 0 && !state.over && !state.demoMode) {
      var hw = weights[state.hoverIdx];
      var mx = (hw.from.x + hw.to.x) / 2;
      var my = (hw.from.y + hw.to.y) / 2;
      var label = "|w| = " + hw.magnitude.toFixed(2);
      if (hw.ticksSmall >= 3) label += "  · dim for " + hw.ticksSmall + " ticks";
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

    // Particles
    for (var pi = 0; pi < state.particles.length; pi++) {
      var pa = state.particles[pi];
      ctx.globalAlpha = Math.max(0, 1 - pa.age / pa.maxAge);
      ctx.fillStyle = pa.color;
      ctx.beginPath();
      ctx.arc(pa.x, pa.y, 2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Floating texts
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

    // Demo banner
    if (state.demoMode && state.ghostCursor.alpha > 0) {
      ctx.globalAlpha = state.ghostCursor.alpha;
      ctx.fillStyle = "#a31f34";
      ctx.beginPath();
      ctx.arc(state.ghostCursor.x, state.ghostCursor.y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(state.ghostCursor.x, state.ghostCursor.y, 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      MLSP.roundRect(ctx, W/2 - 130, 50, 260, 22, 4);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 11px 'Helvetica Neue', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("watch · then click dim weights to prune them", W/2, 65);
    } else if (state.promptVisible) {
      ctx.fillStyle = "rgba(163,31,52,0.92)";
      MLSP.roundRect(ctx, W/2 - 100, 50, 200, 22, 4);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 11px 'Helvetica Neue', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("your turn — click a dim weight", W/2, 65);
    }

    // Target-reached celebration banner
    if (state.targetReached && state.targetBannerTime < 3000) {
      var a = state.targetBannerTime < 2200 ? 1 : Math.max(0, 1 - (state.targetBannerTime - 2200) / 800);
      ctx.globalAlpha = a;
      ctx.fillStyle = "#3d9e5a";
      MLSP.roundRect(ctx, W/2 - 150, 50, 300, 24, 4);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 12px 'Helvetica Neue', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("🏆 target reached — keep going for bonus sparsity", W/2, 67);
      ctx.globalAlpha = 1;
    }

    if (state.over) drawGameOver();

    ctx.restore();
  }

  function drawHud() {
    var barX = 20, barY = H - 26, barW = W - 40, barH = 8;
    ctx.fillStyle = "#eee";
    MLSP.roundRect(ctx, barX, barY, barW, barH, 4); ctx.fill();
    var accFrac = Math.max(0, Math.min(1, state.accuracy / 100));
    var accColor = state.accuracy >= 90 ? "#3d9e5a"
                 : state.accuracy >= 75 ? "#4a90c4"
                 : state.accuracy >= ACCURACY_FLOOR ? "#c87b2a"
                 : "#c44";
    ctx.fillStyle = accColor;
    MLSP.roundRect(ctx, barX, barY, barW * accFrac, barH, 4); ctx.fill();
    var ceilX = barX + barW * Math.max(0, Math.min(1, state.ceiling / 100));
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(ceilX, barY - 3); ctx.lineTo(ceilX, barY + barH + 3);
    ctx.stroke();
    var floorX = barX + barW * (ACCURACY_FLOOR / 100);
    ctx.strokeStyle = "#c44";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(floorX, barY - 2); ctx.lineTo(floorX, barY + barH + 2);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = "10px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#555";
    ctx.textAlign = "left";
    ctx.fillText("accuracy " + state.accuracy.toFixed(1) + "%", barX, barY - 4);
    ctx.textAlign = "right";
    var sparsityLabel = "sparsity " + state.sparsity.toFixed(0) + "% / " + TARGET_SPARSITY + "% goal";
    if (state.targetReached) sparsityLabel += "  🏆";
    ctx.fillText(sparsityLabel, W - 20, barY - 4);
    ctx.fillStyle = "#999";
    ctx.font = "9px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("daily " + today + "  · today best " + dailyBest + "%  · alltime " + alltimeBest + "%", barX, barY + barH + 12);
    ctx.textAlign = "right";
    ctx.fillText("press R to retry", W - 20, barY + barH + 12);
  }

  function drawGameOver() {
    ctx.fillStyle = "rgba(255,255,255,0.86)";
    ctx.fillRect(0, 0, W, H);

    var title, titleColor;
    if (state.targetReached) {
      title = "🏆 goal cleared";
      titleColor = "#3d9e5a";
    } else {
      title = "accuracy collapsed";
      titleColor = "#a31f34";
    }
    ctx.fillStyle = titleColor;
    ctx.font = "bold 22px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(title, W / 2, 90);
    ctx.fillStyle = "#555";
    ctx.font = "italic 11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("the dashed red lines are weights you destroyed", W / 2, 108);

    var bandY = H - 100;
    ctx.fillStyle = "rgba(255,255,255,0.97)";
    MLSP.roundRect(ctx, 30, bandY, W - 60, 80, 6);
    ctx.fill();
    ctx.strokeStyle = "#e0e0e0";
    ctx.lineWidth = 1;
    MLSP.roundRect(ctx, 30, bandY, W - 60, 80, 6);
    ctx.stroke();

    ctx.fillStyle = "#333";
    ctx.font = "bold 15px 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    var targetText = state.targetReached ? "target reached ✓" : "target missed ✗";
    ctx.fillText(
      "sparsity " + Math.round(state.sparsity) + "%  ·  " + targetText,
      W / 2, bandY + 24
    );
    ctx.font = "12px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = "#555";
    ctx.fillText(
      "patience " + state.confidenceScore + "  ·  " + state.correct + "/" + state.attempts + " correct  ·  " + state.dropped + " dropped",
      W / 2, bandY + 44
    );
    ctx.fillText("today best " + dailyBest + "%  ·  alltime " + alltimeBest + "%", W / 2, bandY + 60);
    ctx.fillStyle = "#777";
    ctx.font = "italic 11px 'Helvetica Neue', Arial, sans-serif";
    ctx.fillText("tap or press R to retry", W / 2, bandY + 76);
  }

  requestAnimationFrame(frame);

  return {
    id: "prune",
    name: "Pulse Prune",
    ahaLabel: "You just felt",
    ahaText: "The network rewiring around your cuts. Magnitude-based pruning works because most weights are redundant — but redundancy is dynamic. When you cut a weight, its neighbours (the 'sibling' weights sharing an endpoint) pick up the lost capacity and thicken on the next fine-tune tick. That's the visible rewiring. The art is finding weights that have stayed dim long enough that their siblings can absorb the load without damage. This is the lottery-ticket hypothesis (Frankle & Carbin 2018) plus gradual magnitude pruning (Zhu & Gupta 2017) — felt in your fingers.",
    buildShareText: function(result) {
      var targetLine = result.targetReached
        ? "🏆 " + TARGET_SPARSITY + "% target reached"
        : "✗ missed " + TARGET_SPARSITY + "% target";
      return "MLSys Playground · Pulse Prune · " + today + "\n" +
             "sparsity " + result.finalSparsity + "% · " + targetLine + " · patience " + result.confidenceScore + "\n" +
             result.emojiGrid + "\n" +
             "play → mlsysbook.ai/games/prune/";
    }
  };
};
