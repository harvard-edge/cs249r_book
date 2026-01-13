// --- CANDLE ANIMATION MODULE ---
export function initCandle(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");

    // Coordinate system setup: 
    // Translate to bottom-left corner.
    // New height is 24.
    ctx.translate(0, 24);
    ctx.scale(1, -1);

    // Animation Timing
    const fps = 12; 
    const interval = 1000 / fps;
    let prev = Date.now();

    // Flame Parameters
    const yBase = [2, 1, 0, 0, 0, 0, 1, 2];
    const maxBase = [7, 9, 11, 13, 13, 11, 9, 7];
    const minBase = [4, 7, 8, 10, 10, 8, 7, 4];
    
    // Vertical offset to sit on top of the taller candle (approx y=12)
    const Y_OFFSET = 12; 

    // Interaction State
    let mouseX = -100;
    let mouseY = -100;
    let isDisturbed = false;

    // Spark Particle System
    const particles = [];

    // Track mouse over the canvas
    window.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        
        const normX = (e.clientX - rect.left) / rect.width;
        const normY = (e.clientY - rect.top) / rect.height;

        // Map to internal 16x24 grid
        mouseX = normX * 16;
        
        // Screen Y (0 at top) to Canvas Y (0 at bottom)
        const screenYPixel = normY * 24;
        mouseY = 24 - screenYPixel;
    });

    window.addEventListener('touchmove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const touch = e.touches[0];
        const normX = (touch.clientX - rect.left) / rect.width;
        const normY = (touch.clientY - rect.top) / rect.height;
        mouseX = normX * 16;
        mouseY = 24 - (normY * 24);
    });

    function drawCandle() {
        // --- CANDLE BODY ---
        // Height 10 pixels (0 to 10).
        
        // Main Fill
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(5, 0, 6, 10);

        // --- ENHANCED PIXELATED SHADING (Right side) ---
        
        // 1. Deepest Shadow (Far right x=10) - Solid
        // Using a darker grey to define the edge
        ctx.fillStyle = "#a0aec0"; 
        ctx.fillRect(10, 0, 1, 10); 

        // 2. Mid Shadow (x=9) - Checkerboard Dither
        // Draws on every even Y pixel
        ctx.fillStyle = "#cbd5e0";
        for (let y = 0; y < 10; y++) {
            if (y % 2 === 0) ctx.fillRect(9, y, 1, 1);
        }

        // 3. Light Shadow (x=8) - Sparse Dither
        // Draws on every 4th pixel to fade the shadow out
        for (let y = 0; y < 10; y+=4) {
            ctx.fillRect(8, y, 1, 1);
        }

        // Outline (Pen drawn style - dark grey/black)
        ctx.strokeStyle = "#2d3748"; 
        ctx.lineWidth = 0.5; // Thinner border

        // Left wall
        ctx.beginPath(); ctx.moveTo(5.25, 0); ctx.lineTo(5.25, 10); ctx.stroke();
        // Right wall
        ctx.beginPath(); ctx.moveTo(10.75, 0); ctx.lineTo(10.75, 10); ctx.stroke();
        // Bottom
        ctx.beginPath(); ctx.moveTo(5, 0.5); ctx.lineTo(11, 0.5); ctx.stroke();
        // Top rim
        ctx.beginPath(); ctx.moveTo(5, 9.5); ctx.lineTo(11, 9.5); ctx.stroke();

        // --- WICK ---
        // Starts at top (y=10), goes to y=12
        ctx.strokeStyle = "#4a5568";
        ctx.beginPath();
        // Moved to x=8 to align perfectly with the flame's center
        ctx.moveTo(8, 10);
        ctx.lineTo(8, 12); 
        ctx.stroke();
    }

    function createSpark() {
        // Create sparks that scatter outwards
        const colors = ["#ffeb3b", "#ff9800", "#ff5722", "#ffffff"];
        particles.push({
            // Centered spawning around x=8 (7.0 + 0 to 2 = 7 to 9)
            x: 7.0 + Math.random() * 2, 
            y: 12 + Math.random() * 2,    // Start near wick/flame base
            vy: 0.5 + Math.random() * 1.5, // Fast upward velocity
            vx: (Math.random() - 0.5) * 3, // Wide scatter left/right
            color: colors[Math.floor(Math.random() * colors.length)],
            life: 1.0                   
        });
    }

    function drawSparks() {
        for (let i = particles.length - 1; i >= 0; i--) {
            let p = particles[i];
            
            // Physics
            p.y += p.vy;
            p.x += p.vx;
            p.life -= 0.1; // Fade fast

            // Draw pixel
            ctx.fillStyle = p.color;
            // Use globalAlpha for fading
            ctx.globalAlpha = p.life > 0 ? p.life : 0;
            ctx.fillRect(Math.floor(p.x), Math.floor(p.y), 1, 1);
            ctx.globalAlpha = 1.0;

            // Remove dead particles
            if (p.life <= 0 || p.y > 24 || p.x < 0 || p.x > 16) {
                particles.splice(i, 1);
            }
        }
    }

    function loop() {
        requestAnimationFrame(loop);

        const now = Date.now();
        const dif = now - prev;

        if (dif > interval) {
            prev = now;

            // Clear canvas (0,0 to 16,24)
            ctx.clearRect(0, 0, 16, 24);

            // 1. Draw Static Elements
            drawCandle();

            // 2. Interaction Logic
            // Wick tip is at y=12. Flame extends up to ~20.
            const inFlameZone = (mouseX > 4 && mouseX < 12 && mouseY > 11 && mouseY < 22);
            
            if (inFlameZone) {
                isDisturbed = true;
                // Scatter sparks vigorously
                createSpark();
                createSpark();
                createSpark();
            } else {
                isDisturbed = false;
            }

            // 3. Draw Flame (if not disturbed)
            if (!isDisturbed) {
                ctx.lineWidth = 1; // Restore line width for flame

                // Outer Red/Orange
                ctx.strokeStyle = "#d14234";
                let i = 0;
                for (let x = 4; x < 12; x++) {
                    let localMin = minBase[i];
                    let localMax = maxBase[i];
                    let localY = yBase[i];
                    
                    let startY = localY + Y_OFFSET; 
                    let height = Math.random() * (localMax - localMin + 1) + localMin;
                    let endY = height + Y_OFFSET;

                    ctx.beginPath();
                    ctx.moveTo(x + 0.5, startY);
                    ctx.lineTo(x + 0.5, endY);
                    ctx.stroke();
                    i++;
                }

                // Middle Orange
                ctx.strokeStyle = "#f2a55f";
                let j = 1;
                for (let x = 5; x < 11; x++) {
                    let localMin = minBase[j];
                    let localMax = maxBase[j];
                    let localY = yBase[j];
                    
                    let innerMin = localMin - 5;
                    let innerMax = localMax - 5;
                    
                    let startY = localY + 1 + Y_OFFSET; 
                    let h = Math.random() * (innerMax - innerMin + 1) + innerMin;
                    let endY = h + Y_OFFSET + 3; 

                    ctx.beginPath();
                    ctx.moveTo(x + 0.5, startY);
                    ctx.lineTo(x + 0.5, endY);
                    ctx.stroke();
                    j++;
                }

                // Inner White/Yellow core
                ctx.strokeStyle = "#e8dec5";
                let k = 3;
                for (let x = 7; x < 9; x++) {
                    let localMin = minBase[k];
                    let localMax = maxBase[k];
                    let localY = yBase[k];

                    let innerMin = localMin - 9;
                    let innerMax = localMax - 9;

                    let startY = localY + Y_OFFSET + 2;
                    let h = Math.random() * (innerMax - innerMin + 1) + innerMin;
                    let endY = h + Y_OFFSET + 6;

                    ctx.beginPath();
                    ctx.moveTo(x + 0.5, startY);
                    ctx.lineTo(x + 0.5, endY);
                    ctx.stroke();
                    k++;
                }
            }

            // 4. Draw Sparks
            if (particles.length > 0) {
                drawSparks();
            }
        }
    }

    loop();
}
