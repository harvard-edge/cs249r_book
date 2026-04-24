// test_reference_renderer.js - Test file for reference rendering system

import { processReferences } from './reference_renderer.js';

/**
 * Test the reference parsing functionality
 */
function testReferenceParsing() {
  console.log('🧪 Testing reference parsing...');
  
  const testMarkdown = `
This is a test paragraph with a reference[^ref-1] and another reference[^ref-2].

Here's some more content with multiple references[^ref-1] in the same paragraph.

[^ref-1]: (id=p-1757047226191-5rhhu0559) Distribution shifts arise from a variety of underlying mechanisms—both natural and system-driven. Understanding these mechanisms helps practitioners detect, diagnose, and design mitigation strategies.

[^ref-2]: (id=p-1757047226193-lv53eb2n8) Data poisoning can be avoided by cleaning data, which involves identifying and removing or correcting noisy, incomplete, or inconsistent data points.
  `;
  
  const testHTML = `
<p>This is a test paragraph with a reference[^ref-1] and another reference[^ref-2].</p>
<p>Here's some more content with multiple references[^ref-1] in the same paragraph.</p>
  `;
  const result = processReferences(testMarkdown, testHTML);
  
  console.log('Processed HTML:', result);
  
  // Expected: ref-1 appears at least twice, ref-2 appears once
  const ref1Matches = result.match(/ref-link-ref-1/g) || [];
  const ref2Matches = result.match(/ref-link-ref-2/g) || [];
  if (ref1Matches.length >= 2 && ref2Matches.length >= 1) {
    console.log('✅ Reference parsing test passed');
  } else {
    console.log('❌ Reference parsing test failed');
  }
  
  if (result.includes('reference-pill') && result.includes('ref-link-ref-1') && result.includes('ref-link-ref-2')) {
    console.log('✅ Definition parsing test passed');
  } else {
    console.log('❌ Definition parsing test failed');
  }
}

/**
 * Test the reference processing with HTML content
 */
function testReferenceProcessing() {
  console.log('🧪 Testing reference processing...');
  
  const testMarkdown = `
This is a test paragraph with a reference[^ref-1].

[^ref-1]: (id=p-1757047226191-5rhhu0559) Distribution shifts arise from a variety of underlying mechanisms.
  `;
  
  const testHTML = '<p>This is a test paragraph with a reference[^ref-1].</p>';
  
  const result = processReferences(testMarkdown, testHTML);
  
  console.log('Processed HTML:', result);
  
  // Check if reference was converted to a link
  if (result.includes('reference-pill') && result.includes('ref-link-ref-1')) {
    console.log('✅ Reference processing test passed');
  } else {
    console.log('❌ Reference processing test failed');
  }
}

/**
 * Test edge cases
 */
function testEdgeCases() {
  console.log('🧪 Testing edge cases...');
  
  // Test with no references
  const noRefs = processReferences('No references here', '<p>No references here</p>');
  if (noRefs === '<p>No references here</p>') {
    console.log('✅ No references test passed');
  } else {
    console.log('❌ No references test failed');
  }
  
  // Test with malformed references
  const malformed = processReferences('Malformed[^ref-1', '<p>Malformed[^ref-1</p>');
  if (malformed.includes('Malformed[^ref-1')) {
    console.log('✅ Malformed references test passed');
  } else {
    console.log('❌ Malformed references test failed');
  }
  
  // Test with empty input
  const empty = processReferences('', '');
  if (empty === '') {
    console.log('✅ Empty input test passed');
  } else {
    console.log('❌ Empty input test failed');
  }
}

/**
 * Run all tests
 */
export function runReferenceRendererTests() {
  console.log('🚀 Starting reference renderer tests...');
  
  try {
    testReferenceParsing();
    testReferenceProcessing();
    testEdgeCases();
    
    console.log('🎉 All reference renderer tests completed!');
  } catch (error) {
    console.error('❌ Test error:', error);
  }
}

// Auto-run tests if this file is loaded directly
if (typeof window !== 'undefined' && window.location.pathname.includes('test')) {
  runReferenceRendererTests();
}