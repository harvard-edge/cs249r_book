/* eslint-disable no-console */
const { resolveHighlightStyle } = require('../out/providers/qmdHighlightPalette');

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
      const base = resolveHighlightStyle(preset, {}, theme);
      assertDistinctTypeRefs(base, label);

      const withStructuralOverride = resolveHighlightStyle(preset, {
        structuralRefColor: '#ffffff',
      }, theme);
      assert(
        withStructuralOverride.figureRefColor === base.figureRefColor,
        `${label}: figureRefColor should not inherit structuralRefColor`,
      );
      assert(
        withStructuralOverride.tableRefColor === base.tableRefColor,
        `${label}: tableRefColor should not inherit structuralRefColor`,
      );
      assert(
        withStructuralOverride.listingRefColor === base.listingRefColor,
        `${label}: listingRefColor should not inherit structuralRefColor`,
      );
    }
  }

  // Verify dark and light palettes are actually different
  for (const preset of presets) {
    const dark = resolveHighlightStyle(preset, {}, 'dark');
    const light = resolveHighlightStyle(preset, {}, 'light');
    assert(
      dark.figureRefColor !== light.figureRefColor,
      `${preset}: dark and light figureRefColor should differ`,
    );
    assert(
      dark.sectionRefColor !== light.sectionRefColor,
      `${preset}: dark and light sectionRefColor should differ`,
    );
  }

  const labelFallback = resolveHighlightStyle('balanced', {
    tableRefColor: '#00ff00',
  });
  assert(
    labelFallback.tableLabelDefColor === '#00ff00',
    'tableLabelDefColor should fallback to tableRefColor',
  );

  const explicitLabelOverride = resolveHighlightStyle('balanced', {
    tableRefColor: '#00ff00',
    tableLabelDefColor: '#123456',
  });
  assert(
    explicitLabelOverride.tableLabelDefColor === '#123456',
    'tableLabelDefColor override should take precedence',
  );

  console.log('Palette checks passed.');
}

run();
