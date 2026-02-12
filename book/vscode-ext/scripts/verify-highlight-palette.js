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
    ['tableRefColor', 'listingRefColor'],
    ['tableRefColor', 'equationRefColor'],
    ['listingRefColor', 'equationRefColor'],
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
  for (const preset of presets) {
    const base = resolveHighlightStyle(preset, {});
    assertDistinctTypeRefs(base, preset);

    const withStructuralOverride = resolveHighlightStyle(preset, {
      structuralRefColor: '#ffffff',
    });
    assert(
      withStructuralOverride.figureRefColor === base.figureRefColor,
      `${preset}: figureRefColor should not inherit structuralRefColor`,
    );
    assert(
      withStructuralOverride.tableRefColor === base.tableRefColor,
      `${preset}: tableRefColor should not inherit structuralRefColor`,
    );
    assert(
      withStructuralOverride.listingRefColor === base.listingRefColor,
      `${preset}: listingRefColor should not inherit structuralRefColor`,
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
