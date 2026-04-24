// vite.dev.config.mjs - Development configuration for test website
import { defineConfig } from 'vite';
import { resolve } from 'path';
import { fileURLToPath } from 'url';
import { copyFileSync, mkdirSync, existsSync, readdirSync, statSync, watch } from 'fs';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

// Set Node.js options to handle large headers
if (!process.env.NODE_OPTIONS || !process.env.NODE_OPTIONS.includes('max-http-header-size')) {
  process.env.NODE_OPTIONS = (process.env.NODE_OPTIONS || '') + ' --max-http-header-size=65536';
}

export default defineConfig({
  root: '.', // Serve from project root to access test_website
  server: {
    port: 3000,
    host: true,
    open: '/test_website/mlsys_book_removed_most/',
    cors: true,
    watch: {
      usePolling: true,
      interval: 1000,
    },
    fs: {
      allow: ['..', './test_website']
    },
    headers: {
      'Cache-Control': 'no-cache',
      'Connection': 'close'
    }
  },
  plugins: [
    {
      name: 'suppress-sourcemap-warnings',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          // Allow bundle.js to be served normally
          
          // Strip large headers to prevent 431 errors
          Object.keys(req.headers).forEach(key => {
            if (req.headers[key] && typeof req.headers[key] === 'string' && req.headers[key].length > 2048) {
              req.headers[key] = req.headers[key].substring(0, 100) + '...';
            }
          });
          
          // Suppress source map warnings
          if (req.url && req.url.endsWith('.map')) {
            res.statusCode = 404;
            res.end('Source map not found');
            return;
          }
          next();
        });
      },
    },
    {
      name: 'dev-file-watcher',
      configureServer(server) {
        // Watch source files for changes and trigger rebuild
        const srcPath = resolve(__dirname, 'src_shadow');
        console.log('🔍 Development mode: Watching for changes in:', srcPath);
        
        let isBuilding = false;
        let buildTimeout = null;
        
        const triggerBuild = () => {
          if (isBuilding) {
            console.log('⏳ Build already in progress, skipping...');
            return;
          }
          
          isBuilding = true;
          console.log('🔄 Triggering rebuild and copy...');
          
          import('child_process').then(({ exec }) => {
            exec('npm run build:vite', { cwd: process.cwd() }, (error, stdout, stderr) => {
              isBuilding = false;
              if (error) {
                console.error('❌ Build error:', error.message);
                if (stderr) console.error('❌ Build stderr:', stderr);
                return;
              }
              console.log('✅ Build completed, files updated in test website');
              if (stdout) {
                const lines = stdout.split('\n');
                const importantLines = lines.filter(line => 
                  line.includes('✓') || 
                  line.includes('✅') || 
                  line.includes('Copied') ||
                  line.includes('built in')
                );
                if (importantLines.length > 0) {
                  console.log('📦 Build summary:', importantLines.join('\n'));
                }
              }
            });
          }).catch(err => {
            isBuilding = false;
            console.error('❌ Failed to import child_process:', err);
          });
        };
        
        // Initialize file watcher after server starts
        setTimeout(async () => {
          try {
            const chokidar = await import('chokidar').catch(() => null);
            
            if (chokidar) {
              const watcher = chokidar.watch(srcPath, {
                ignored: /(^|[\/\\])\../, 
                persistent: true,
                ignoreInitial: true
              });
              
              watcher.on('change', (path) => {
                console.log(`📝 File changed: ${path}`);
                if (buildTimeout) clearTimeout(buildTimeout);
                buildTimeout = setTimeout(triggerBuild, 500);
              });
              
              console.log('✅ Chokidar file watcher started successfully');
            } else {
              const watcher = watch(srcPath, (eventType, filename) => {
                if (!filename) return;
                
                if (filename.includes('node_modules') || 
                    filename.includes('.git') || 
                    filename.includes('.DS_Store') ||
                    filename.endsWith('.tmp') ||
                    filename.endsWith('.swp')) {
                  return;
                }
                
                console.log(`📝 File changed: ${filename} (${eventType})`);
                if (buildTimeout) clearTimeout(buildTimeout);
                buildTimeout = setTimeout(triggerBuild, 500);
              });
              
              console.log('✅ Basic file watcher started (non-recursive)');
            }
            
          } catch (error) {
            console.log('⚠️  File watching not available on this platform:', error.message);
            console.log('💡 You can manually run "npm run build:vite" after making changes');
          }
        }, 1000);
      },
    },
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, './src_shadow'),
    },
  },
});