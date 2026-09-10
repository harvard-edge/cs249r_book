import { Noise } from './noise.js';

export function initTerrain(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let width, height;
    // Static random seed offset so it doesn't change on every resize/redraw if we don't want it to
    // But random on refresh is fine.
    const time = Math.random() * 100; 

    // Configuration
    const config = {
        scale: 0.003,      // Zoom level of the map
        levels: 6,         // Number of topographic steps
        baseColor: '#f0f4f8',
        darkColor: '#dbe2e8', // Darker shade for pockets
        greenColor: '#e8f5e9', // Subtle green for land
        lineColor: '#cbd5e0'
    };

    function resize() {
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;
        draw();
    }

    function draw() {
        // Clear background
        ctx.fillStyle = config.baseColor;
        ctx.fillRect(0, 0, width, height);

        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;

        // Colors as RGB
        // Base: #f0f4f8 -> 240, 244, 248
        // Dark: #dbe2e8 -> 219, 226, 232
        const baseR = 240, baseG = 244, baseB = 248;
        const darkR = 219, darkG = 226, darkB = 232;
        // Green: #e8f5e9 -> 232, 245, 233 (Subtle green)
        const greenR = 232, greenG = 245, greenB = 233;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const noiseVal = Noise.perlin2((x * config.scale) + time, (y * config.scale) + time);
                const norm = (noiseVal + 1) / 2; // 0 to 1

                let r = baseR, g = baseG, b = baseB;

                // Thresholds for pockets and land
                if (norm < 0.3) {
                    // Deep pocket (Darker)
                    r = darkR; g = darkG; b = darkB;
                } else if (norm < 0.4) {
                    // Transition pocket
                    r = (baseR + darkR)/2;
                    g = (baseG + darkG)/2;
                    b = (baseB + darkB)/2;
                } else if (norm > 0.6) {
                    // Higher ground (Subtle Green)
                    r = greenR; g = greenG; b = greenB;
                }

                // Topographic Contour Lines
                const stepSize = 1 / config.levels;
                const remainder = (norm % stepSize);
                
                // If remainder is very small, it's a line
                if (remainder < 0.015) { 
                    r = 203; g = 213; b = 224; // #cbd5e0 line color
                }

                const index = (y * width + x) * 4;
                data[index] = r;
                data[index + 1] = g;
                data[index + 2] = b;
                data[index + 3] = 255; // Alpha
            }
        }

        ctx.putImageData(imageData, 0, 0);

        // Draw Symbols (X marks the spot)
        drawSymbols();
    }

    function drawSymbols() {
        const symbolCount = 12;
        const colors = ['#81c784', '#64b5f6', '#ffb74d']; // Green, Blue, Orange
        const chars = ['×', '○', '+', '△'];

        ctx.font = '14px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        for(let i=0; i<symbolCount; i++) {
            const rx = Math.random() * width;
            const ry = Math.random() * height;
            
            // Random Color & Char
            ctx.fillStyle = colors[Math.floor(Math.random() * colors.length)];
            const char = chars[Math.floor(Math.random() * chars.length)];

            ctx.fillText(char, rx, ry); 
            
            // Small number
            ctx.font = '10px monospace';
            ctx.fillText((Math.random()*1000).toFixed(0), rx + 15, ry + 5);
            ctx.font = '14px monospace'; // Reset
        }
    }

    window.addEventListener('resize', resize);
    resize();
}