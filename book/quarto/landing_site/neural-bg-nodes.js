document.addEventListener('DOMContentLoaded', function(){
  var container = document.getElementById('mls-neural-bg');
  if (!container) return;
  var canvas = document.createElement('canvas');
  var ctx = canvas.getContext('2d');
  container.appendChild(canvas);
  
  var nodes = [];
  var numNodes = 100;
  
  var isDark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
  
  // Observe theme changes
  var observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.attributeName === 'data-bs-theme') {
        isDark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
      }
    });
  });
  observer.observe(document.documentElement, { attributes: true });

  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = window.innerWidth || 1000;
    var h = window.innerHeight || 800;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    initNodes(w, h);
  }

  function initNodes(w, h) {
    nodes = [];
    for (var i = 0; i < numNodes; i++) {
      nodes.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        radius: Math.random() * 2 + 1
      });
    }
  }

  function draw() {
    var w = window.innerWidth || 1000;
    var h = window.innerHeight || 800;
    ctx.clearRect(0, 0, w, h);
    
    // Light mode: Crimson dots and lines. Dark mode: Neon blue dots and lines.
    ctx.fillStyle = isDark ? '#00e5ff' : '#A51C30';
    
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      n.x += n.vx;
      n.y += n.vy;
      
      if (n.x < 0 || n.x > w) n.vx *= -1;
      if (n.y < 0 || n.y > h) n.vy *= -1;
      
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
      ctx.fill();
      
      for (var j = i + 1; j < nodes.length; j++) {
        var n2 = nodes[j];
        var dx = n.x - n2.x;
        var dy = n.y - n2.y;
        var dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 150) {
          ctx.beginPath();
          ctx.moveTo(n.x, n.y);
          ctx.lineTo(n2.x, n2.y);
          
          var alpha = 1 - (dist / 150);
          // Dark mode lines: glowing blue. Light mode lines: crimson.
          if (isDark) {
            ctx.strokeStyle = 'rgba(0, 229, 255, ' + (alpha * 0.2) + ')';
          } else {
            ctx.strokeStyle = 'rgba(165, 28, 48, ' + (alpha * 0.15) + ')';
          }
          ctx.stroke();
        }
      }
    }
  }
  
  var lastTime = 0;
  function tick(timestamp) { 
    requestAnimationFrame(tick); 
    if (timestamp - lastTime < 33) return;
    lastTime = timestamp;
    draw(); 
  }
  resize();
  requestAnimationFrame(tick);
  
  var resizeTimeout;
  window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(resize, 200);
  });
});
