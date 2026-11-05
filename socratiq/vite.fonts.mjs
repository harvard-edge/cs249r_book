/** Supported browsers use WOFF2; omit duplicate math font formats in both bundles. */
export function woff2MathFonts() {
  return {
    name: 'socratiq-woff2-fonts',
    enforce: 'pre',
    transform(code, id) {
      if (!id.includes('/katex/dist/katex.min.css')) return;
      return code.replace(/src:([^;}]+)([;}])/g, (original, sources, terminator) => {
        const woff2 = sources.split(',').find(source => source.includes('.woff2'));
        return woff2 ? `src:${woff2}${terminator}` : original;
      });
    },
  };
}
