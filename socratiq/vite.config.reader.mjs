import {defineConfig} from 'vite';
import {fileURLToPath} from 'node:url';
import {woff2MathFonts} from './vite.fonts.mjs';

export default defineConfig({
  plugins: [woff2MathFonts()],
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
