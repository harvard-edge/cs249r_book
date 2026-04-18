// vite.config.dev.mjs - Development configuration optimized for HMR
import { defineConfig } from 'vite';
import { resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  root: 'src_shadow',
  
  // Optimize dependencies for development
  optimizeDeps: {
    // Restrict scanning to the main JS entry to avoid scanning all HTML demo files
    entries: [resolve(__dirname, 'src_shadow/js/index.js')], // absolute path is fine for entries
  },
  
  // Development server configuration
  server: {
    port: 4175,
    open: '/test_website/encryption_textbook/index.html?socratiq=true',
    host: true,
    watch: {
      usePolling: true,
      interval: 1000,
    },
    // Enable HMR
    hmr: {
      overlay: true,
    },
    // Serve files from the test website directory
    fs: {
      allow: ['..']
    }
  },
  
  // Preview configuration
  preview: {
    port: 4173,
    host: true,
    open: '/test_website/encryption_textbook/index.html?socratiq=true',
    cors: true,
    fs: {
      allow: ['..']
    }
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
  
  // Development-only plugins
  plugins: [
    // Serve static files under /test_website/* from disk so the book loads under Vite dev server
    {
      name: 'serve-test-website-static',
      configureServer(server) {
        server.middlewares.use(async (req, res, next) => {
          try {
            if (!req.url) return next();
            const urlPath = req.url.split('?')[0].split('#')[0];
            if (!urlPath.startsWith('/test_website/')) return next();
            
            // Let the injector handle index files so it can append the dev module
            const isIndexFile = 
              urlPath.endsWith('/index.html') || 
              urlPath.endsWith('/') || 
              urlPath === '/test_website/encryption_textbook' ||
              urlPath === '/test_website/mlsys_book_removed_most';

            if (isIndexFile) return next();

            // console debug for static serving
            if (urlPath !== '/favicon.ico') {
              console.log(`[vite-dev] static serve: ${urlPath}`);
            }
            const fs = await import('fs');
            const path = await import('path');
            const mime = (p) => {
              const ext = path.extname(p).toLowerCase();
              switch (ext) {
                case '.html': return 'text/html';
                case '.css': return 'text/css';
                case '.js': return 'application/javascript';
                case '.mjs': return 'application/javascript';
                case '.json': return 'application/json';
                case '.png': return 'image/png';
                case '.jpg': case '.jpeg': return 'image/jpeg';
                case '.svg': return 'image/svg+xml';
                case '.gif': return 'image/gif';
                case '.wasm': return 'application/wasm';
                case '.ico': return 'image/x-icon';
                default: return 'application/octet-stream';
              }
            };
            let fullPath = path.resolve(process.cwd(), `.${urlPath}`);
            if (fs.existsSync(fullPath) && fs.statSync(fullPath).isDirectory()) {
              fullPath = path.join(fullPath, 'index.html');
            }
            if (!fs.existsSync(fullPath)) return next();
            // Ensure COOP/COEP for OPFS/SharedArrayBuffer
            res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
            res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
            res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
            res.setHeader('Content-Type', mime(fullPath));
            fs.createReadStream(fullPath).pipe(res);
            return;
          } catch {
            return next();
          }
        });
      },
    },
    // Dev-only HTML transformer so the external test page loads Vite entry for HMR
    {
      name: 'inject-dev-entry-into-test-page',
      configureServer(server) {
        server.middlewares.use(async (req, res, next) => {
          try {
            if (!req.url) return next();
            // Normalize URL path without query/hash
            const url = req.url.split('?')[0].split('#')[0];
            
            const isMainBookTest = url === '/test_website/mlsys_book_removed_most/' || 
                                  url === '/test_website/mlsys_book_removed_most' || 
                                  url === '/test_website/mlsys_book_removed_most/index.html';
            
            const isEncryptionTest = url === '/test_website/encryption_textbook/index.html' || 
                                   url === '/test_website/encryption_textbook/' || 
                                   url === '/test_website/encryption_textbook';
            
            const isSmallTest = url === '/test_website/index.html' || 
                               url === '/test_website/' || 
                               url === '/test_website';

            if (isMainBookTest || isSmallTest || isEncryptionTest) {
              let indexPath;
              if (isMainBookTest) {
                indexPath = '/test_website/mlsys_book_removed_most/index.html';
              } else if (isEncryptionTest) {
                indexPath = '/test_website/encryption_textbook/index.html';
              } else {
                indexPath = '/test_website/index.html';
              }
              
              console.log(`[vite-dev] injector handling: ${url} using ${indexPath}`);
              
              // Read original HTML from disk
              const fs = await import('fs');
              const path = await import('path');
              const fullPath = path.resolve(process.cwd(), `.${indexPath}`);
              if (!fs.existsSync(fullPath)) return next();
              let html = fs.readFileSync(fullPath, 'utf-8');

              // Remove ANY script tags pointing to src_shadow/js/index.js or production dist
              // This handles various relative paths and optional attributes like type="module"
              const distScriptRe = /<script\b[^>]*src=["'](?:(?:\.\.\/)*src_shadow\/js\/|\.\/scripts\/ai_menu\/dist\/)[^"']+\.js["'][^>]*><\/script>/gi;
              const before = html;
              html = html.replace(distScriptRe, '');
              const removedCount = (before.match(distScriptRe) || []).length;
              if (removedCount > 0) console.log(`[vite-dev] injector removed ${removedCount} script tag(s)`);

              // Append dev entry which Vite will transform & HMR
              const devScriptTags = [
                '<script>document.cookie="socratiq=true; path=/; max-age=" + (60*60*24*365);</script>',
                '<script type="module" src="/@vite/client"></script>',
                '<script type="module" src="/js/index.js"></script>'
              ].join('\n');
              
              if (html.includes('</head>')) {
                html = html.replace('</head>', `\n${devScriptTags}\n</head>`);
              } else if (html.includes('</body>')) {
                html = html.replace('</body>', `\n${devScriptTags}\n</body>`);
              } else {
                html = `${html}\n${devScriptTags}`;
              }

              console.log('[vite-dev] injected HMR client and /js/index.js into test page');

              // Ensure COOP/COEP for OPFS/SharedArrayBuffer
              res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
              res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
              res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
              res.setHeader('Content-Type', 'text/html');
              return res.end(html);
            }
            return next();
          } catch (e) {
            return next();
          }
        });
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
    },
  ],
});