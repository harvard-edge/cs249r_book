document.addEventListener('DOMContentLoaded', function(){
  var container = document.getElementById('mls-neural-bg');
  if (!container) return;
  var canvas = document.createElement('canvas');
  var ctx = canvas.getContext('2d');
  container.appendChild(canvas);
  
  var time = 0;
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
  }

  function draw() {
    var w = window.innerWidth || 1000;
    var h = window.innerHeight || 800;
    ctx.clearRect(0, 0, w, h);
    time += 0.01;
    
    var lines = 10;
    var spacing = h / lines;
    
    ctx.lineWidth = 1.5;
    
    for (var i = 0; i < lines; i++) {
      ctx.beginPath();
      for (var x = 0; x <= w; x += 20) {
        // Complex sine wave
        var y = (i * spacing) + 
                Math.sin(x * 0.005 + time + i) * 50 + 
                Math.cos(x * 0.01 - time * 1.5) * 20;
        
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      
      var alpha = 0.1 + (i / lines) * 0.15;
      if (isDark) {
        ctx.strokeStyle = 'rgba(0, 229, 255, ' + alpha + ')'; // Neon blue
      } else {
        ctx.strokeStyle = 'rgba(165, 28, 48, ' + alpha + ')'; // Crimson
      }
      ctx.stroke();
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
    resizeTimeout = setTimeout(resize, 200);
  });
});
