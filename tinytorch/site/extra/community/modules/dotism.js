import { SimplexNoise } from './noise.js';

const DotismScene = (function() {
    const DOTISM_CONFIG = {
        chunkSize: 100,
        chunkWidth: 200,
        numParticles: 40000,
        fogNear: 20,
        fogFar: 140
    };

    let scene, camera, renderer, dotTexture;
    let chunks = [];
    let lastChunkZ = 0;
    let simplex;
    let config;

    function createDotTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 32;
        canvas.height = 32;
        const ctx = canvas.getContext('2d');
        ctx.beginPath();
        ctx.arc(16, 16, 12, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        return new THREE.CanvasTexture(canvas);
    }

    function generateChunk(offsetZ, skyState) {
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        const sizes = [];

        const chunkWidth = config.chunkWidth;
        const chunkSize = config.chunkSize;
        const numParticles = config.numParticles;

        for (let i = 0; i < numParticles; i++) {
            let x = (Math.random() - 0.5) * chunkWidth;
            let zLocal = Math.random() * chunkSize;
            let z = offsetZ - zLocal;

            let biomeVal = simplex.noise2D(x * 0.02, z * 0.02);

            let h1 = simplex.noise2D(x * 0.01, z * 0.01) * 25;
            let h2 = simplex.noise2D(x * 0.03, z * 0.03) * 8;
            let y = h1 + h2 - 10;

            let pColor;
            let pSize;
            let includeParticle = false;

            if (biomeVal > 0.4) {
                if (Math.random() > 0.5) {
                    includeParticle = true;
                    pColor = config.colors[3];
                    pSize = 0.8 + Math.random() * 0.5;
                    y += Math.random() * 3;
                }
            } else if (biomeVal < -0.3) {
                let flow = Math.sin(x * 0.1 + z * 0.05 + simplex.noise2D(x * 0.05, z * 0.05) * 5);
                if (Math.abs(flow) > 0.6) {
                    includeParticle = true;
                    pColor = config.colors[0];
                    pSize = 1.0 + Math.random() * 0.6;
                    x += (Math.random() - 0.5) * 2;
                    y -= 5;
                    y += (Math.random() - 0.5) * 2.0;
                }
            } else {
                includeParticle = true;
                let hNorm = (y + 20) / 40;
                hNorm += (Math.random() - 0.5) * 0.2;

                if (hNorm < 0.3) pColor = config.colors[0];
                else if (hNorm < 0.5) pColor = config.colors[1];
                else if (hNorm < 0.7) pColor = config.colors[4];
                else pColor = config.colors[2];

                pSize = 0.7 + Math.random() * 0.8;
                y += (Math.random() - 0.5) * 4.0;

                if (Math.random() > 0.5) {
                    x += (Math.random() - 0.5) * 3;
                } else {
                    x = Math.floor(x / 2) * 2 + (Math.random() * 0.5);
                }
            }

            if (includeParticle) {
                positions.push(x, y, z);
                colors.push(pColor.r, pColor.g, pColor.b);
                sizes.push(pSize);
            }
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

        const fogColor = scene.fog ? scene.fog.color : new THREE.Color(0xeeeee6);

        const shaderMaterial = new THREE.ShaderMaterial({
            uniforms: {
                pointTexture: { value: dotTexture },
                fogColor: { value: fogColor },
                fogNear: { value: config.fogNear },
                fogFar: { value: config.fogFar }
            },
            vertexShader: `
                attribute float size;
                varying vec3 vColor;
                varying float vFogDepth;
                void main() {
                    vColor = color;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = size * (300.0 / -mvPosition.z);
                    gl_Position = projectionMatrix * mvPosition;
                    vFogDepth = -mvPosition.z;
                }
            `,
            fragmentShader: `
                uniform sampler2D pointTexture;
                uniform vec3 fogColor;
                uniform float fogNear;
                uniform float fogFar;
                varying vec3 vColor;
                varying float vFogDepth;
                void main() {
                    gl_FragColor = vec4(vColor, 1.0);
                    gl_FragColor = gl_FragColor * texture2D(pointTexture, gl_PointCoord);
                    if (gl_FragColor.a < 0.5) discard;
                    float fogFactor = smoothstep(fogNear, fogFar, vFogDepth);
                    gl_FragColor.rgb = mix(gl_FragColor.rgb, fogColor, fogFactor);
                }
            `,
            transparent: true,
            depthTest: true,
            vertexColors: true
        });

        const points = new THREE.Points(geometry, shaderMaterial);
        scene.add(points);

        return { mesh: points, zStart: offsetZ, zEnd: offsetZ - chunkSize };
    }

    function init(options) {
        scene = options.scene;
        camera = options.camera;
        renderer = options.renderer;

        simplex = new SimplexNoise();

        const palettes = {
            'Spring': ['#1a5e3a', '#3cb371', '#9acd32', '#f0e68c', '#8fbc8f'],
            'Autumn': ['#4a2511', '#a64b1e', '#d99029', '#f2dca2', '#cc6633'],
            'Winter': ['#0f2e47', '#2b5975', '#7db3c9', '#e3f4f7', '#538da3'],
            'Lavender': ['#2d1b4e', '#583671', '#9e6eb3', '#e6d4f0', '#7a528a'],
            'Charcoal': ['#1a1a1a', '#404040', '#808080', '#e0e0e0', '#606060']
        };

        const paletteKeys = Object.keys(palettes);
        const randomKey = paletteKeys[Math.floor(Math.random() * paletteKeys.length)];
        const selectedPalette = palettes[randomKey];

        config = {
            ...DOTISM_CONFIG,
            colors: selectedPalette.map(hex => new THREE.Color(hex))
        };

        dotTexture = createDotTexture();

        for (let i = 0; i < 3; i++) {
            let chunk = generateChunk(lastChunkZ);
            chunks.push(chunk);
            lastChunkZ -= config.chunkSize;
        }
    }

    function update(skyState) {
        if (skyState && scene.fog) {
            scene.fog.color.copy(skyState.color);
        }

        const lookAhead = 120;
        const lookBehind = 20;
        const zMax = camera.position.z + lookBehind;
        const zMin = camera.position.z - lookAhead;

        // Determine required chunk indices based on current camera view
        const firstChunkIndex = Math.floor(zMax / config.chunkSize);
        const lastChunkIndex = Math.floor(zMin / config.chunkSize);

        // Generate missing chunks in range
        for (let i = firstChunkIndex; i >= lastChunkIndex; i--) {
            const chunkZ = i * config.chunkSize;
            if (!chunks.find(c => c.zStart === chunkZ)) {
                let chunk = generateChunk(chunkZ, skyState);
                chunks.push(chunk);
            }
        }

        // Remove chunks entirely outside the view range
        chunks = chunks.filter(chunk => {
            if (chunk.zEnd > zMax || chunk.zStart < zMin) {
                scene.remove(chunk.mesh);
                chunk.mesh.geometry.dispose();
                chunk.mesh.material.dispose();
                return false;
            }
            return true;
        });
    }

    function resize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    return {
        init,
        update,
        resize,
        getSimplex: () => simplex
    };
})();

export { DotismScene };
