document.addEventListener('DOMContentLoaded', function(){
  var container = document.getElementById('mls-neural-bg');
  if (!container) return;
  var canvas = document.createElement('canvas');
  var ctx = canvas.getContext('2d');
  container.appendChild(canvas);
  
  var pixels = [];
  var time = 0;
  
  var lightColors = [
    '#A51C30', '#4285F4', '#FBBC05', '#34A853', '#8E24AA', 
    '#E0E0E0', '#BDBDBD', '#9E9E9E',
    '#F1F3F4', '#E8F0FE', '#FCE8E6', '#FEF7E0', '#E6F4EA', '#F3E8FD'
  ];
  
  var darkColors = [
    '#ff4d6d', '#00e5ff', '#b388ff', '#00e676', '#ffea00',
    '#333333', '#444444', '#555555',
    '#1a1a1a', '#111111', '#222222', '#151515', '#0a0a0a', '#1c1c1c'
  ];
  
  var isDark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
  
  // Observe theme changes
  var observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.attributeName === 'data-bs-theme') {
        isDark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
        // Re-initialize pixels to pick up new colors immediately
        initPixels(canvas.width / (window.devicePixelRatio || 1), canvas.height / (window.devicePixelRatio || 1));
      }
    });
  });
  observer.observe(document.documentElement, { attributes: true });

  var pixelSize = 10;
  var spacing = 4;
  var cols = 0, rows = 0;
  
  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
    var h = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
    
    if (w === 0) w = 1000;
    if (h === 0) h = 800;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    initPixels(w, h);
  }

  function initPixels(w, h) {
    pixels = [];
    cols = Math.ceil(w / (pixelSize + spacing));
    rows = Math.ceil(h / (pixelSize + spacing));
    
    var colors = isDark ? darkColors : lightColors;
    
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        if (Math.random() > 0.25) {
          pixels.push({
            c: j,
            r: i,
            color: colors[Math.floor(Math.random() * colors.length)],
            phase: Math.random() * Math.PI * 2,
            speed: 0.04 + Math.random() * 0.06
          });
        }
      }
    }
  }

  function draw() {
    var w = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
    var h = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
    ctx.clearRect(0, 0, w, h);
    time += 1.5;
    
    pixels.forEach(function(p) {
      var alpha = 0.1 + 0.3 * Math.sin(time * p.speed + p.phase);
      if (alpha < 0) alpha = 0;
      
      // In dark mode, we might want slightly higher alpha for the neon colors to pop
      if (isDark) {
        alpha = alpha * 1.5;
        if (alpha > 1) alpha = 1;
      }
      
      ctx.globalAlpha = alpha;
      ctx.fillStyle = p.color;
      var px = p.c * (pixelSize + spacing);
      var py = p.r * (pixelSize + spacing);
      ctx.fillRect(px, py, pixelSize, pixelSize);
    });
    ctx.globalAlpha = 1;
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
