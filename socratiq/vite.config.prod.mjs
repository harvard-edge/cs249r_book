// vite.config.prod.mjs - Production configuration with all build optimizations
import { defineConfig } from 'vite';
import { resolve } from 'path';
import { viteSingleFile } from 'vite-plugin-singlefile';
import { copyFileSync, mkdirSync, existsSync, readdirSync, statSync } from 'fs';

export default defineConfig({
  root: 'src_shadow',
  
  // Production build configuration
  build: {
    outDir: '../dist_vite',
    emptyOutDir: true,
    target: 'esnext',
    assetsInlineLimit: 100000000,
    chunkSizeWarningLimit: 100000000,
    cssCodeSplit: false,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'src_shadow/js/index.js'),
      },
      output: {
        manualChunks: undefined,
        inlineDynamicImports: true,
        entryFileNames: 'bundle.js',
        assetFileNames: 'assets/[name][extname]',
      },
      external: [],
    },
    write: true,
    minify: true,
    sourcemap: false,
    worker: {
      format: 'es',
      plugins: [],
      rollupOptions: {
        output: {
          inlineDynamicImports: true,
          format: 'iife'
        },
      },
    },
  },
  
  // Preview configuration
  preview: {
    port: 4175,
  },
  
  // Resolve aliases
  resolve: {
    alias: {
      '@': resolve(__dirname, './src_shadow'),
    },
  },
  
  // CSS configuration
  css: {
    modules: {
      generateScopedName: '[name]__[local]___[hash:base64:5]',
    },
    postcss: {
      plugins: [],
    },
  },
  
  // Worker configuration
  worker: {
    format: 'es',
    plugins: () => [],
  },
  
  // Production plugins
  plugins: [
    {
      name: 'multi-dist-copy-and-cleanup',
      writeBundle() {
        const mainDistPath = resolve(__dirname, 'dist_vite');
        
        // --- Destinations ---
        const destinations = [
          { 
            path: resolve(__dirname, '../book/quarto/tools/scripts/socratiQ'), 
            name: "Quarto Source" 
          },
          { 
            path: resolve(__dirname, '../book/quarto/_build/html-vol1/tools/scripts/socratiQ'), 
            name: "Quarto Build Vol1" 
          },
          { 
            path: resolve(__dirname, '../book/quarto/_build/html-vol2/tools/scripts/socratiQ'), 
            name: "Quarto Build Vol2" 
          }
        ];

        const copyRecursive = (src, dest) => {
          if (!existsSync(src)) return;
          const items = readdirSync(src);
          items.forEach(item => {
            const srcPath = resolve(src, item);
            const destPath = resolve(dest, item);
            if (statSync(srcPath).isDirectory()) {
              if (!existsSync(destPath)) {
                mkdirSync(destPath, { recursive: true });
              }
              copyRecursive(srcPath, destPath);
            } else {
              copyFileSync(srcPath, destPath);
            }
          });
        };

        // --- Copy to all destinations ---
        try {
          destinations.forEach(destInfo => {
            // For build directories, only copy if they already exist (to avoid cluttering if not built)
            if (destInfo.name.includes("Build") && !existsSync(resolve(destInfo.path, '../../../../'))) {
               // If the _build directory doesn't even exist, skip
               return;
            }
            
            if (!existsSync(destInfo.path)) {
              mkdirSync(destInfo.path, { recursive: true });
            }
            copyRecursive(mainDistPath, destInfo.path);
            console.log(`✅ Build output copied to ${destInfo.name}: ${destInfo.path}`);
          });
          console.log('✨ All copy operations completed successfully.');
        } catch (error) {
          console.error('❌ Error during file copy operations:', error);
        }
      },
    },
    {
      name: 'configure-response-headers',
      configureServer: (server) => {
        server.middlewares.use((_req, res, next) => {
          // Headers for WASM support
          res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
          res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
          res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
          
          // Additional headers for WASM files
          if (_req.url && _req.url.endsWith('.wasm')) {
            res.setHeader('Content-Type', 'application/wasm');
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
            res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
          }
          
          next();
        });
      },
    },
    // Single file plugin for production builds
    viteSingleFile({
      useRecommendedBuildConfig: true,
      inlinePattern: [
        '**/*.{js,css,html,wasm}',
      ],
      removeViteModuleLoader: true,
      enforceInline: true,
    }),
    {
      name: 'configure-wasm',
      transform(code, id) {
        if (id.endsWith('.wasm')) {
          return {
            code: `export default "${code}";`,
            map: null
          };
        }
      },
      load(id) {
        if (id.endsWith('.wasm')) {
          return `export default "${id}";`;
        }
      }
    }
  ],
});