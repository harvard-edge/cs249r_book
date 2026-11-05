const path = require('node:path');

module.exports = {
  content: [path.join(__dirname, 'src_shadow/**/*.{html,js}')],
  // Existing theme tracking marks the workspace containers with dark-mode.
  darkMode: ['selector', '.dark-mode'],
  theme: {extend: {zIndex: {'60': '60', '9999': '9999'}}},
  // These existing classes are assembled from card counts / PDF status.
  safelist: ['text-green-600', 'text-red-600', {pattern: /grid-cols-[1-4]/, variants: ['md']}],
};
