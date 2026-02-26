/**
 * Subtle animated neural-network background for MLSysBook landing.
 * Light grid of nodes + edges with a gentle opacity wave. No D3 dependency.
 */
(function () {
  var container = document.getElementById('mls-neural-bg');
  if (!container) return;

  var canvas = document.createElement('canvas');
  var ctx = canvas.getContext('2d');
  container.appendChild(canvas);

  var cols = 12, rows = 8;
  var nodes = [];
  var edges = [];
  var time = 0;

  function initNodes() {
    nodes = [];
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        nodes.push({
          x: (j + 0.5) / cols,
          y: (i + 0.5) / rows,
          phase: (i * cols + j) * 0.7
        });
      }
    }
    edges = [];
    for (var i = 0; i < nodes.length; i++) {
      var r = Math.floor(i / cols), c = i % cols;
      if (c < cols - 1) edges.push([i, i + 1]);
      if (r < rows - 1) edges.push([i, i + cols]);
    }
  }

  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = window.innerWidth * dpr;
    canvas.height = window.innerHeight * dpr;
    canvas.style.width = window.innerWidth + 'px';
    canvas.style.height = window.innerHeight + 'px';
    ctx.scale(dpr, dpr);
    draw();
  }

  function draw() {
    var w = window.innerWidth, h = window.innerHeight;
    ctx.clearRect(0, 0, w, h);

    time += 0.012;
    var baseOpacity = 0.08;
    var waveAmplitude = 0.06;

    // Edges (lines)
    ctx.strokeStyle = 'rgba(165, 28, 48, 0.4)';
    edges.forEach(function (e) {
      var a = nodes[e[0]], b = nodes[e[1]];
      var phase = (a.phase + b.phase) * 0.5;
      var opacity = baseOpacity + waveAmplitude * Math.sin(time + phase);
      ctx.globalAlpha = Math.max(0.02, opacity);
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(a.x * w, a.y * h);
      ctx.lineTo(b.x * w, b.y * h);
      ctx.stroke();
    });

    // Nodes (small circles)
    ctx.fillStyle = 'rgb(165, 28, 48)';
    nodes.forEach(function (n) {
      var opacity = baseOpacity + waveAmplitude * Math.sin(time + n.phase);
      ctx.globalAlpha = Math.max(0.02, opacity);
      ctx.beginPath();
      ctx.arc(n.x * w, n.y * h, 1.5, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.globalAlpha = 1;
  }

  function tick() {
    draw();
    requestAnimationFrame(tick);
  }

  initNodes();
  resize();
  tick();
  window.addEventListener('resize', resize);
})();
