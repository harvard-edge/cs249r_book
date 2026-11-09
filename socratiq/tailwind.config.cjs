const path = require('node:path');

const brand = {
  50: 'rgb(var(--socratiq-brand-50, 252 244 246) / <alpha-value>)',
  100: 'rgb(var(--socratiq-brand-100, 247 230 234) / <alpha-value>)',
  200: 'rgb(var(--socratiq-brand-200, 238 208 215) / <alpha-value>)',
  300: 'rgb(var(--socratiq-brand-300, 223 170 184) / <alpha-value>)',
  400: 'rgb(var(--socratiq-brand-400, 199 112 136) / <alpha-value>)',
  500: 'rgb(var(--socratiq-brand-500, 142 21 55) / <alpha-value>)',
  600: 'rgb(var(--socratiq-brand-600, 116 19 47) / <alpha-value>)',
  700: 'rgb(var(--socratiq-brand-700, 104 16 41) / <alpha-value>)',
  800: 'rgb(var(--socratiq-brand-800, 86 13 34) / <alpha-value>)',
  900: 'rgb(var(--socratiq-brand-900, 70 11 28) / <alpha-value>)',
  950: 'rgb(var(--socratiq-brand-950, 43 6 17) / <alpha-value>)',
};

module.exports = {
  content: [path.join(__dirname, 'src_shadow/**/*.{html,js}')],
  // Existing theme tracking marks the workspace containers with dark-mode.
  darkMode: ['selector', '.dark-mode'],
  theme: {
    extend: {
      zIndex: {'60': '60', '9999': '9999'},
      colors: {
        blue: brand,
        indigo: brand,
      },
    },
  },
  // These existing classes are assembled from card counts / PDF status.
  safelist: ['text-green-600', 'text-red-600', {pattern: /grid-cols-[1-4]/, variants: ['md']}],
};
