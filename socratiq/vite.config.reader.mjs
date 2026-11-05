import {defineConfig} from 'vite';
import {fileURLToPath} from 'node:url';

export default defineConfig({
  plugins: [{
    name: 'reader-woff2-fonts',
    enforce: 'pre',
    transform(code, id) {
      if (!id.includes('/katex/dist/katex.min.css')) return;
      // Supported browsers use WOFF2. Do not embed duplicate TTF/WOFF fonts
      // in the single-file reader along with the WOFF2 copies.
      return code.replace(/src:([^;}]+)([;}])/g, (original, sources, terminator) => {
        const woff2 = sources.split(',').find(source => source.includes('.woff2'));
        return woff2 ? `src:${woff2}${terminator}` : original;
      });
    },
  }],
  build: {
    outDir: '../books/shared/assets/scripts/socratiq',
    emptyOutDir: false,
    lib: {
      entry: fileURLToPath(new URL('./src_reader/index.js', import.meta.url)),
      name: 'SocratiqReader', formats: ['iife'], fileName: () => 'reader.js',
    },
    target: 'es2022',
  },
});
