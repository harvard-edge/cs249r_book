/* eslint-disable no-console */
const { getBaseHighlightStyle } = require('../out/providers/qmdHighlightPalette');

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function assertDistinctTypeRefs(style, preset) {
  const pairs = [
    ['figureRefColor', 'tableRefColor'],
    ['figureRefColor', 'listingRefColor'],
    ['figureRefColor', 'equationRefColor'],
    ['figureRefColor', 'inlinePythonColor'],
    ['tableRefColor', 'listingRefColor'],
    ['tableRefColor', 'equationRefColor'],
    ['tableRefColor', 'inlinePythonColor'],
    ['listingRefColor', 'equationRefColor'],
    ['listingRefColor', 'inlinePythonColor'],
    ['equationRefColor', 'inlinePythonColor'],
  ];
  for (const [left, right] of pairs) {
    assert(
      style[left] !== style[right],
      `${preset}: ${left} unexpectedly equals ${right}`,
    );
  }
}

function run() {
  const presets = ['subtle', 'balanced', 'highContrast'];
  const themes = ['dark', 'light'];

  for (const theme of themes) {
    for (const preset of presets) {
      const label = `${theme}/${preset}`;
      const base = getBaseHighlightStyle(preset, theme);
      assertDistinctTypeRefs(base, label);
    }
  }

  // Verify dark and light palettes are actually different
  for (const preset of presets) {
    const dark = getBaseHighlightStyle(preset, 'dark');
    const light = getBaseHighlightStyle(preset, 'light');
    assert(
      dark.figureRefColor !== light.figureRefColor,
      `${preset}: dark and light figureRefColor should differ`,
    );
    assert(
      dark.sectionRefColor !== light.sectionRefColor,
      `${preset}: dark and light sectionRefColor should differ`,
    );
  }

  console.log('Palette checks passed.');
}

run();
