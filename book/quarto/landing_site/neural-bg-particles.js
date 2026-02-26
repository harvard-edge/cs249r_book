document.addEventListener('DOMContentLoaded', function(){
  var container = document.getElementById('mls-neural-bg');
  if (!container) return;
  var canvas = document.createElement('canvas');
  var ctx = canvas.getContext('2d');
  container.appendChild(canvas);
  
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

  var particles = [];
  var numParticles = 150;
  var lastW = window.innerWidth;
  var lastH = window.innerHeight;
  var time = 0;

  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
    var h = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
    
    if (w === 0) w = 1000;
    if (h === 0) h = 800;

    if (Math.abs(w - lastW) > 50 || particles.length === 0) {
      var canvasH = h + 200; 
      
      canvas.width = w * dpr;
      canvas.height = canvasH * dpr;
      canvas.style.width = w + 'px';
      canvas.style.height = canvasH + 'px';
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.scale(dpr, dpr);
      
      lastW = w;
      lastH = h;
      
      initParticles(w, canvasH);
    }
  }

  function initParticles(w, h) {
    particles = [];
    for(var i = 0; i < numParticles; i++) {
      particles.push({
        x: Math.random() * w,
        y: Math.random() * h,
        size: Math.random() * 3 + 1,
        speedX: Math.random() * 1 - 0.5,
        speedY: Math.random() * 1 - 0.5,
        baseAlpha: Math.random() * 0.5 + 0.1,
        pulseSpeed: Math.random() * 0.05 + 0.01
      });
    }
  }

  function draw() {
    var w = canvas.width / Math.min(window.devicePixelRatio || 1, 2);
    var h = canvas.height / Math.min(window.devicePixelRatio || 1, 2);
    
    ctx.clearRect(0, 0, w, h);
    time += 1;
    
    for(var i = 0; i < particles.length; i++) {
      var p = particles[i];
      
      // Move
      p.x += p.speedX;
      p.y += p.speedY;
      
      // Wrap around edges
      if(p.x < 0) p.x = w;
      if(p.x > w) p.x = 0;
      if(p.y < 0) p.y = h;
      if(p.y > h) p.y = 0;
      
      // Pulse alpha
      var currentAlpha = p.baseAlpha + Math.sin(time * p.pulseSpeed) * 0.2;
      if(currentAlpha < 0) currentAlpha = 0;
      if(currentAlpha > 1) currentAlpha = 1;
      
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      
      if (isDark) {
        // Glowing cyan/purple for dark mode
        var r = i % 2 === 0 ? 0 : 179;
        var g = i % 2 === 0 ? 229 : 136;
        var b = 255;
        ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + currentAlpha + ')';
        ctx.shadowBlur = 10;
        ctx.shadowColor = ctx.fillStyle;
      } else {
        // Harvard Crimson / Dark Grey for light mode
        var r = i % 2 === 0 ? 165 : 100;
        var g = i % 2 === 0 ? 28 : 100;
        var b = i % 2 === 0 ? 48 : 100;
        ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + currentAlpha + ')';
        ctx.shadowBlur = 0;
      }
      
      ctx.fill();
    }
  }
  
  var lastTime = 0;
  function tick(timestamp) { 
    requestAnimationFrame(tick); 
    if (timestamp - lastTime < 33) return; // ~30fps
    lastTime = timestamp;
    draw(); 
  }
  resize();
  requestAnimationFrame(tick);
  
  var resizeTimeout;
  window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(function() {
      if (window.innerWidth !== lastW) {
        resize();
      }
    }, 200);
  });
});
