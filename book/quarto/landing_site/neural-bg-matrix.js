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

  var fontSize = 14;
  var columns = 0;
  var drops = [];
  
  // ML Systems related characters
  var chars = "01∑∫∇∂∆∏µσλθωαβγδεζηικλμνξοπρστυφχψωABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/=<>[]{}()";
  chars = chars.split("");

  var lastW = window.innerWidth;
  var lastH = window.innerHeight;

  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
    var h = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
    
    if (w === 0) w = 1000;
    if (h === 0) h = 800;

    if (Math.abs(w - lastW) > 50 || drops.length === 0) {
      var canvasH = h + 200; 
      
      canvas.width = w * dpr;
      canvas.height = canvasH * dpr;
      canvas.style.width = w + 'px';
      canvas.style.height = canvasH + 'px';
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.scale(dpr, dpr);
      
      lastW = w;
      lastH = h;
      
      columns = Math.floor(w / fontSize);
      drops = [];
      for(var x = 0; x < columns; x++) {
        drops[x] = Math.random() * -100; // Start off screen randomly
      }
    }
  }

  function draw() {
    var w = canvas.width / Math.min(window.devicePixelRatio || 1, 2);
    var h = canvas.height / Math.min(window.devicePixelRatio || 1, 2);
    
    // Translucent background to create trail effect
    if (isDark) {
      ctx.fillStyle = "rgba(10, 10, 12, 0.05)";
    } else {
      ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
    }
    ctx.fillRect(0, 0, w, h);
    
    // Set text color and font
    ctx.font = fontSize + "px monospace";
    
    for(var i = 0; i < drops.length; i++) {
      var text = chars[Math.floor(Math.random() * chars.length)];
      
      // Color based on theme
      if (isDark) {
        // Glowing cyan/blue for dark mode
        ctx.fillStyle = Math.random() > 0.9 ? "#fff" : "#00e5ff";
        ctx.globalAlpha = Math.random() * 0.5 + 0.1;
      } else {
        // Harvard Crimson for light mode
        ctx.fillStyle = Math.random() > 0.9 ? "#111" : "#A51C30";
        ctx.globalAlpha = Math.random() * 0.3 + 0.05;
      }
      
      ctx.fillText(text, i * fontSize, drops[i] * fontSize);
      ctx.globalAlpha = 1;
      
      // Reset drop to top randomly
      if(drops[i] * fontSize > h && Math.random() > 0.975) {
        drops[i] = 0;
      }
      
      drops[i]++;
    }
  }
  
  var lastTime = 0;
  function tick(timestamp) { 
    requestAnimationFrame(tick); 
    // Matrix effect looks better slightly slower (~20-25fps)
    if (timestamp - lastTime < 40) return;
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
