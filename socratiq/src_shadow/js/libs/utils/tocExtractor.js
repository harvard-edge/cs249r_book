/**
 * TOC (Table of Contents) Extractor Utility
 * 
 * This utility extracts the table of contents from the current page using native DOM methods
 * and logs it to the console. It's designed to run fast without blocking app load.
 * 
 * The extracted TOC will be used later to build a knowledge graph.
 * 
 * Features:
 * - No external dependencies
 * - Works without requiring heading IDs
 * - Efficient native DOM traversal
 * - Handles dynamic content loading
 * - Backward compatible API
 */

import { getDBInstance } from "./indexDb.js";

// Guard to prevent duplicate TOC extractions
let tocExtractionInProgress = false;
let lastTOCExtractionTime = 0;

/**
 * Custom TOC extractor that finds headings without requiring IDs
 * @param {Object} options - Configuration options
 * @returns {Promise<Array>} The extracted TOC data
 */
export async function extractPageTOC(options = {}) {
  // Prevent duplicate extractions within 1 second
  const now = Date.now();
  if (tocExtractionInProgress || (now - lastTOCExtractionTime < 1000)) {
    return window.pageTOC || [];
  }
  
  tocExtractionInProgress = true;
  lastTOCExtractionTime = now;
  
  try {
    // Default configuration
    const defaultOptions = {
      // Where to grab the headings to build the table of contents
      contentSelector: 'body',
      // Which headings to grab inside of the contentSelector element
      headingSelector: 'h1, h2, h3, h4, h5, h6',
      // For headings inside relative or absolute positioned containers within content
      hasInnerContainers: true,
      // Include HTML in the extracted data
      includeHtml: false,
      // Wait time for dynamic content to load
      waitTime: 100,
      // Maximum retries for finding headings
      maxRetries: 3
    };

    // Merge with provided options
    const config = { ...defaultOptions, ...options };

    // Function to extract content snippet from paragraphs following a heading
    const extractContentSnippet = (heading) => {
      try {
        // Get the next sibling elements after the heading
        let nextElement = heading.nextElementSibling;
        let contentText = '';
        let sentencesFound = 0;
        const maxSentences = 2;
        
        // Look through the next few elements to find paragraphs
        while (nextElement && sentencesFound < maxSentences) {
          // Skip other headings (we've reached the next section)
          if (nextElement.tagName && nextElement.tagName.match(/^H[1-6]$/)) {
            break;
          }
          
          // Look for paragraphs, divs, or other text containers
          if (nextElement.tagName === 'P' || 
              nextElement.tagName === 'DIV' || 
              nextElement.tagName === 'SECTION' ||
              nextElement.tagName === 'ARTICLE') {
            
            const elementText = nextElement.textContent.trim();
            if (elementText) {
              // Split into sentences (rough approximation)
              const sentences = elementText.split(/[.!?]+/).filter(s => s.trim().length > 0);
              
              for (const sentence of sentences) {
                if (sentencesFound >= maxSentences) break;
                
                const cleanSentence = sentence.trim();
                if (cleanSentence.length > 10) { // Skip very short fragments
                  if (contentText) contentText += ' ';
                  contentText += cleanSentence;
                  sentencesFound++;
                  
                  // Add punctuation if it was removed by split
                  if (!cleanSentence.match(/[.!?]$/)) {
                    contentText += '.';
                  }
                }
              }
            }
          }
          
          nextElement = nextElement.nextElementSibling;
        }
        
        // Clean up the content
        contentText = contentText.trim();
        
        // Limit total length to avoid overly long snippets
        if (contentText.length > 300) {
          contentText = contentText.substring(0, 300).trim();
          // Try to end at a sentence boundary
          const lastPeriod = contentText.lastIndexOf('.');
          if (lastPeriod > 200) {
            contentText = contentText.substring(0, lastPeriod + 1);
          }
        }
        
        return contentText;
      } catch (error) {
        console.warn('Error extracting content snippet:', error);
        return '';
      }
    };

    // Function to extract headings from DOM
    const extractHeadings = () => {
      const tocData = [];
      
      // Get the content container
      const contentElement = config.contentSelector === 'body' 
        ? document.body 
        : document.querySelector(config.contentSelector);
      
      if (!contentElement) {
        console.warn('Content element not found:', config.contentSelector);
        return tocData;
      }

      // Find all headings
      const headings = contentElement.querySelectorAll(config.headingSelector);
      
      headings.forEach((heading, index) => {
        // Skip hidden headings
        if (heading.offsetParent === null) return;
        
        // Get heading text (strip HTML if includeHtml is false)
        const text = config.includeHtml ? heading.innerHTML : heading.textContent.trim();
        
        // Skip empty headings
        if (!text) return;
        
        // Generate a unique ID if one doesn't exist
        let id = heading.id;
        if (!id) {
          // Create a safe ID from the text
          id = `heading-${index}-${text.toLowerCase()
            .replace(/[^a-z0-9\s-]/g, '')
            .replace(/\s+/g, '-')
            .substring(0, 50)}`;
          heading.id = id;
        }
        
        // Calculate position
        const rect = heading.getBoundingClientRect();
        const position = rect.top + window.scrollY;
        
        // Get heading level
        const level = parseInt(heading.tagName.substring(1));
        
        // Extract content snippet from following paragraphs
        const content = extractContentSnippet(heading);
        
        tocData.push({
          id: id,
          text: text,
          level: level,
          position: position,
          index: index,
          content: content // Add the content snippet
        });
      });
      
      return tocData;
    };

    // Try to extract headings with retries for dynamic content
    let tocData = [];
    let attempts = 0;
    
    while (attempts < config.maxRetries && tocData.length === 0) {
      tocData = extractHeadings();
      
      if (tocData.length === 0 && attempts < config.maxRetries - 1) {
        // Wait a bit longer for dynamic content
        await new Promise(resolve => setTimeout(resolve, config.waitTime * (attempts + 1)));
        attempts++;
      } else {
        break;
      }
    }

    // Log the TOC to console
    const extractionId = Math.random().toString(36).substr(2, 9);
    console.group(`📚 [${extractionId}] Page Table of Contents (TOC)`);
    console.log(`📊 [${extractionId}] Total headings found:`, tocData.length);
    console.log(`📋 [${extractionId}] TOC Structure:`);
    
    if (tocData.length === 0) {
      console.log(`❌ [${extractionId}] No headings found on this page`);
      console.log(`🔍 [${extractionId}] Debug info:`);
      console.log(`- Content selector:`, config.contentSelector);
      console.log(`- Heading selector:`, config.headingSelector);
      console.log(`- Content element found:`, !!document.querySelector(config.contentSelector));
      console.log(`- All headings in document:`, document.querySelectorAll('h1, h2, h3, h4, h5, h6').length);
    } else {
      // Log the hierarchical structure
      tocData.forEach((heading, index) => {
        const indent = '  '.repeat(heading.level - 1);
        console.log(`${indent}${heading.level}. ${heading.text} (ID: ${heading.id})`);
      });
      
      // Log the full data structure
      console.log(`\n📦 [${extractionId}] Full TOC Data:`, tocData);
    }
    
    console.groupEnd();

    return tocData;

  } catch (error) {
    console.error('Error extracting TOC:', error);
    return [];
  } finally {
    // Reset the guard
    tocExtractionInProgress = false;
  }
}

/**
 * Quick TOC extraction that runs immediately without waiting
 * This is designed to be non-blocking and fast
 */
export function extractTOCAsync() {
  // Run the extraction asynchronously without blocking
  setTimeout(async () => {
    try {
      const tocData = await extractPageTOC();
      
      // Store the TOC data globally for later use
      window.pageTOC = tocData;
      
      // Dispatch a custom event to notify other parts of the app
      window.dispatchEvent(new CustomEvent('tocExtracted', {
        detail: { tocData }
      }));
      
    } catch (error) {
      console.error('Async TOC extraction failed:', error);
    }
  }, 0);
}

/**
 * Get the extracted TOC data if it exists
 * @returns {Array} The TOC data or empty array
 */
export function getExtractedTOC() {
  return window.pageTOC || [];
}

/**
 * Check if TOC has been extracted
 * @returns {boolean} True if TOC data exists
 */
export function hasTOCData() {
  return !!(window.pageTOC && window.pageTOC.length > 0);
}

/**
 * Extract TOC with custom heading selector
 * @param {string} headingSelector - CSS selector for headings (e.g., 'h1, h2, h3')
 * @returns {Promise<Array>} The extracted TOC data
 */
export async function extractTOCWithSelector(headingSelector) {
  return await extractPageTOC({
    headingSelector: headingSelector
  });
}

/**
 * Extract TOC from a specific content area
 * @param {string} contentSelector - CSS selector for the content area
 * @returns {Promise<Array>} The extracted TOC data
 */
export async function extractTOCFromContent(contentSelector) {
  return await extractPageTOC({
    contentSelector: contentSelector
  });
}

/**
 * Extract TOC with enhanced debugging for Quarto sites
 * @param {Object} options - Configuration options
 * @returns {Promise<Array>} The extracted TOC data
 */
export async function extractTOCWithDebug(options = {}) {
  const debugOptions = {
    ...options,
    waitTime: 200, // Longer wait for Quarto sites
    maxRetries: 5, // More retries for dynamic content
  };
  
  const callId = Math.random().toString(36).substr(2, 9);
  
  const result = await extractPageTOC(debugOptions);
  
      // Save TOC data to chapterMap table
    if (result.length > 0) {
      const currentUrl = window.location.href;
      const currentTitle = document.title;
      
      console.log(`🔄 [${callId}] About to save TOC data to chapterMap...`);
      console.log(`📊 [${callId}] TOC data:`, {
        url: currentUrl,
        title: currentTitle,
        headingCount: result.length,
        firstHeading: result[0] ? result[0].text : 'N/A'
      });
      
      // Test database connection before attempting to save
      console.log(`🔍 [${callId}] Testing database connection for TOC operations...`);
      const dbTestResult = await testTOCDatabaseConnection();
      if (!dbTestResult) {
        console.error(`❌ [${callId}] Database test failed, skipping TOC save`);
        return result;
      }
      
      // Check if we should extract TOC for this URL first
      const shouldExtract = await shouldExtractTOC(currentUrl);
      console.log(`🔍 [${callId}] Should extract TOC:`, shouldExtract);
      
      if (shouldExtract) {
        try {
          const saveResult = await saveTOCToChapterMap(result, currentUrl, currentTitle);
          if (saveResult) {
            console.log(`💾 [${callId}] TOC data saved to chapterMap successfully`);
            console.log(`🔗 [${callId}] Normalized URL: ${normalizeUrl(currentUrl)}`);
          } else {
            console.error(`❌ [${callId}] TOC data save returned false`);
          }
        } catch (error) {
          console.error(`❌ [${callId}] Failed to save TOC to chapterMap:`, error);
          console.error(`❌ [${callId}] Error stack:`, error.stack);
        }
      } else {
        console.log(`⏭️ [${callId}] TOC extraction skipped - data is fresh`);
        console.log(`🔗 [${callId}] Normalized URL: ${normalizeUrl(currentUrl)}`);
      }
    } else {
      console.log(`📝 [${callId}] No TOC data to save (result.length = ${result.length})`);
    }
  
  return result;
}

/**
 * Wait for headings to appear and then extract TOC
 * Useful for dynamic content loading
 * @param {number} maxWaitTime - Maximum time to wait in milliseconds
 * @param {Object} options - Configuration options
 * @returns {Promise<Array>} The extracted TOC data
 */
export async function extractTOCWhenReady(maxWaitTime = 5000, options = {}) {
  const startTime = Date.now();
  
  while (Date.now() - startTime < maxWaitTime) {
    const tocData = await extractPageTOC(options);
    
    if (tocData.length > 0) {
      console.log(`✅ TOC extracted after ${Date.now() - startTime}ms`);
      return tocData;
    }
    
    // Wait 200ms before trying again
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  
  console.warn(`⚠️ No headings found after waiting ${maxWaitTime}ms`);
  return [];
}

/**
 * Test database connectivity and chapterMap store specifically for TOC operations
 * @returns {Promise<boolean>} True if database is ready for TOC operations
 */
export async function testTOCDatabaseConnection() {
  try {
    const db = await getDBInstance();
    if (!db) {
      console.error('❌ TOC Database test: No database instance');
      return false;
    }

    if (!db.db) {
      console.error('❌ TOC Database test: Database connection not established');
      return false;
    }

    const storeNames = Array.from(db.db.objectStoreNames);
    console.log('🔍 TOC Database test: Available stores:', storeNames);

    if (!db.db.objectStoreNames.contains('chapterMap')) {
      console.error('❌ TOC Database test: chapterMap store missing');
      return false;
    }

    // Test a simple read operation
    try {
      await db.getAll('chapterMap');
      console.log('✅ TOC Database test: Read operation successful');
    } catch (error) {
      console.error('❌ TOC Database test: Read operation failed:', error);
      return false;
    }

    console.log('✅ TOC Database test: All tests passed');
    return true;
  } catch (error) {
    console.error('❌ TOC Database test failed:', error);
    return false;
  }
}

/**
 * Normalize URL by removing query parameters and fragments
 * This prevents duplicate entries for the same page with different scroll positions
 * @param {string} url - The URL to normalize
 * @returns {string} The normalized URL
 */
export function normalizeUrl(url) {
  try {
    const urlObj = new URL(url);
    // Remove query parameters and fragments, keep only protocol, host, and pathname
    return `${urlObj.protocol}//${urlObj.host}${urlObj.pathname}`;
  } catch (error) {
    console.warn('Error normalizing URL:', error);
    return url; // Return original URL if parsing fails
  }
}

/**
 * Save TOC data to the chapterMap table in IndexedDB
 * @param {Array} tocData - The extracted TOC data
 * @param {string} url - The current page URL
 * @param {string} title - The page title
 * @returns {Promise<boolean>} Success status
 */
export async function saveTOCToChapterMap(tocData, url, title) {
  let db = null;
  
  try {
    if (!tocData || tocData.length === 0) {
      console.log('📝 No TOC data to save');
      return false;
    }

    db = await getDBInstance();
    if (!db) {
      console.error('❌ Database not available for saving TOC');
      console.error('❌ Database instance:', db);
      return false;
    }

    // Verify the chapterMap store exists
    if (!db.db.objectStoreNames.contains('chapterMap')) {
      console.error('❌ chapterMap store does not exist in database');
      console.error('❌ Available stores:', Array.from(db.db.objectStoreNames));
      return false;
    }

    // Normalize the URL to prevent duplicates from query parameters
    const normalizedUrl = normalizeUrl(url);
    
    const chapterMapEntry = {
      url: normalizedUrl, // Use normalized URL as the key
      originalUrl: url, // Keep original URL for reference
      title: title || document.title || 'Untitled',
      tocData: tocData,
      lastUpdated: new Date().toISOString(),
      headingCount: tocData.length,
      // Add some metadata
      pageType: url.includes('socratiq.html') ? 'socratiq' : 'external',
      domain: new URL(url).hostname
    };

    console.log(`🔄 Attempting to save TOC to chapterMap:`, {
      originalUrl: url,
      normalizedUrl: normalizedUrl,
      title: title,
      headingCount: tocData.length,
      storeExists: db.db.objectStoreNames.contains('chapterMap'),
      entryData: chapterMapEntry
    });

    // Save to chapterMap table (will replace if normalized URL exists)
    const saveResult = await db.put('chapterMap', chapterMapEntry);
    console.log(`🔄 Save operation result:`, saveResult);
    
    // Verify the save by immediately reading it back using normalized URL
    const verifyResult = await db.get('chapterMap', normalizedUrl);
    console.log(`🔍 Verification read result:`, verifyResult);
    
    if (verifyResult) {
      console.log(`💾 TOC saved to chapterMap for normalized URL: ${normalizedUrl}`);
      console.log(`📊 Saved ${tocData.length} headings for "${title}"`);
      console.log(`🔗 Original URL: ${url}`);
      return true;
    } else {
      console.error(`❌ TOC save verification failed - data not found after save`);
      return false;
    }
    
  } catch (error) {
    console.error('❌ Error saving TOC to chapterMap:', error);
    console.error('❌ Error details:', error.message, error.stack);
    console.error('❌ Database state:', {
      hasDB: !!db,
      hasDBInstance: !!db?.db,
      storeNames: db?.db ? Array.from(db.db.objectStoreNames) : 'N/A'
    });
    return false;
  }
}

/**
 * Get TOC data from chapterMap for a specific URL
 * @param {string} url - The URL to look up
 * @returns {Promise<Object|null>} The chapter map entry or null
 */
export async function getTOCFromChapterMap(url) {
  try {
    const db = await getDBInstance();
    if (!db) {
      console.error('❌ Database not available for retrieving TOC');
      return null;
    }

    // Verify the chapterMap store exists
    if (!db.db.objectStoreNames.contains('chapterMap')) {
      console.error('❌ chapterMap store does not exist in database for retrieval');
      console.error('❌ Available stores:', Array.from(db.db.objectStoreNames));
      return null;
    }

    const entry = await db.get('chapterMap', url);
    return entry || null;
  } catch (error) {
    console.error('❌ Error retrieving TOC from chapterMap:', error);
    console.error('❌ Error details:', error.message, error.stack);
    return null;
  }
}

/**
 * Get all chapter map entries
 * @returns {Promise<Array>} All chapter map entries
 */
export async function getAllChapterMapEntries() {
  try {
    const db = await getDBInstance();
    if (!db) {
      console.error('❌ Database not available for retrieving chapter map');
      return [];
    }

    // Verify the chapterMap store exists
    if (!db.db.objectStoreNames.contains('chapterMap')) {
      console.error('❌ chapterMap store does not exist in database for getAll');
      console.error('❌ Available stores:', Array.from(db.db.objectStoreNames));
      return [];
    }

    const entries = await db.getAll('chapterMap');
    return entries || [];
  } catch (error) {
    console.error('❌ Error retrieving all chapter map entries:', error);
    console.error('❌ Error details:', error.message, error.stack);
    return [];
  }
}

/**
 * Check if TOC should be extracted for the current URL
 * @param {string} url - The URL to check
 * @param {number} maxAgeHours - Maximum age in hours before re-extraction (default: 24)
 * @returns {Promise<boolean>} True if TOC should be extracted
 */
export async function shouldExtractTOC(url, maxAgeHours = 24) {
  try {
    // Use normalized URL to check for existing entries
    const normalizedUrl = normalizeUrl(url);
    const existingEntry = await getTOCFromChapterMap(normalizedUrl);
    
    if (!existingEntry) {
      console.log(`🆕 No existing TOC found for normalized URL: ${normalizedUrl}`);
      return true;
    }
    
    const lastUpdated = new Date(existingEntry.lastUpdated);
    const now = new Date();
    const ageHours = (now - lastUpdated) / (1000 * 60 * 60);
    
    if (ageHours > maxAgeHours) {
      console.log(`⏰ TOC is ${ageHours.toFixed(1)} hours old, re-extracting for normalized URL: ${normalizedUrl}`);
      return true;
    }
    
    console.log(`✅ TOC is fresh (${ageHours.toFixed(1)} hours old), skipping extraction for normalized URL: ${normalizedUrl}`);
    return false;
  } catch (error) {
    console.error('❌ Error checking TOC age:', error);
    return true; // Default to extracting if there's an error
  }
}

/**
 * Test function to debug TOC saving issues
 * This function can be called from the browser console to test the database functionality
 * @returns {Promise<void>}
 */
export async function testTOCSaveFunction() {
  console.log('🧪 Starting TOC save function test...');
  
  try {
    // Test 1: Check database instance
    const db = await getDBInstance();
    console.log('✅ Test 1 - Database instance:', {
      hasDB: !!db,
      hasDBInstance: !!db?.db,
      storeNames: db?.db ? Array.from(db.db.objectStoreNames) : 'N/A',
      chapterMapExists: db?.db ? db.db.objectStoreNames.contains('chapterMap') : false
    });
    
    if (!db || !db.db) {
      console.error('❌ Database not available');
      return;
    }
    
    // Test 2: Check chapterMap store
    if (!db.db.objectStoreNames.contains('chapterMap')) {
      console.error('❌ chapterMap store does not exist');
      return;
    }
    console.log('✅ Test 2 - chapterMap store exists');
    
    // Test 3: Create test TOC data
    const testTOCData = [
      { id: 'test-heading-1', text: 'Test Heading 1', level: 1, position: 100, index: 0 },
      { id: 'test-heading-2', text: 'Test Heading 2', level: 2, position: 200, index: 1 }
    ];
    
    const testUrl = 'https://test.example.com/test-page';
    const testTitle = 'Test Page Title';
    
    console.log('✅ Test 3 - Test data created:', {
      tocDataLength: testTOCData.length,
      url: testUrl,
      title: testTitle
    });
    
    // Test 4: Save test data
    console.log('🔄 Test 4 - Attempting to save test data...');
    const saveResult = await saveTOCToChapterMap(testTOCData, testUrl, testTitle);
    console.log('✅ Test 4 - Save result:', saveResult);
    
    // Test 5: Verify data was saved
    console.log('🔄 Test 5 - Verifying saved data...');
    const retrievedData = await getTOCFromChapterMap(testUrl);
    console.log('✅ Test 5 - Retrieved data:', retrievedData);
    
    // Test 6: Get all entries
    console.log('🔄 Test 6 - Getting all chapter map entries...');
    const allEntries = await getAllChapterMapEntries();
    console.log('✅ Test 6 - All entries:', allEntries);
    
    // Test 7: Clean up test data
    console.log('🔄 Test 7 - Cleaning up test data...');
    await db.delete('chapterMap', testUrl);
    console.log('✅ Test 7 - Test data cleaned up');
    
    console.log('🎉 All tests completed successfully!');
    
  } catch (error) {
    console.error('❌ Test failed:', error);
    console.error('❌ Error details:', error.message, error.stack);
  }
}

/**
 * Clean up duplicate entries in chapterMap by consolidating URLs with query parameters
 * This function removes duplicates and keeps the most recent entry for each normalized URL
 * @returns {Promise<Object>} Cleanup results
 */
export async function cleanupDuplicateChapterMapEntries() {
  console.log('🧹 Starting cleanup of duplicate chapterMap entries...');
  
  try {
    const db = await getDBInstance();
    if (!db) {
      console.error('❌ Database not available for cleanup');
      return { success: false, error: 'Database not available' };
    }

    // Get all entries
    const allEntries = await getAllChapterMapEntries();
    console.log(`📊 Found ${allEntries.length} total entries to analyze`);

    // Group entries by normalized URL
    const normalizedGroups = {};
    const duplicates = [];
    
    allEntries.forEach(entry => {
      const normalizedUrl = normalizeUrl(entry.url);
      
      if (!normalizedGroups[normalizedUrl]) {
        normalizedGroups[normalizedUrl] = [];
      }
      
      normalizedGroups[normalizedUrl].push(entry);
      
      // If we have more than one entry for this normalized URL, it's a duplicate
      if (normalizedGroups[normalizedUrl].length > 1) {
        duplicates.push(normalizedUrl);
      }
    });

    console.log(`🔍 Found ${duplicates.length} normalized URLs with duplicates`);
    
    let cleanedCount = 0;
    let keptCount = 0;
    
    // Process each group with duplicates
    for (const normalizedUrl of duplicates) {
      const entries = normalizedGroups[normalizedUrl];
      console.log(`🔄 Processing ${entries.length} entries for: ${normalizedUrl}`);
      
      // Sort by lastUpdated (most recent first)
      entries.sort((a, b) => new Date(b.lastUpdated) - new Date(a.lastUpdated));
      
      // Keep the most recent entry
      const keepEntry = entries[0];
      const deleteEntries = entries.slice(1);
      
      console.log(`✅ Keeping: ${keepEntry.url} (${keepEntry.lastUpdated})`);
      console.log(`🗑️ Deleting: ${deleteEntries.map(e => e.url).join(', ')}`);
      
      // Delete the duplicate entries
      for (const deleteEntry of deleteEntries) {
        await db.delete('chapterMap', deleteEntry.url);
        cleanedCount++;
      }
      
      // Update the kept entry to use normalized URL if it doesn't already
      if (keepEntry.url !== normalizedUrl) {
        const updatedEntry = {
          ...keepEntry,
          url: normalizedUrl,
          originalUrl: keepEntry.url
        };
        await db.put('chapterMap', updatedEntry);
        console.log(`🔄 Updated entry to use normalized URL: ${normalizedUrl}`);
      }
      
      keptCount++;
    }

    const result = {
      success: true,
      totalEntries: allEntries.length,
      duplicatesFound: duplicates.length,
      entriesCleaned: cleanedCount,
      entriesKept: keptCount,
      duplicatesProcessed: duplicates
    };

    console.log('🎉 Cleanup completed:', result);
    return result;

  } catch (error) {
    console.error('❌ Cleanup failed:', error);
    return { success: false, error: error.message };
  }
}

/**
 * Manually trigger TOC extraction and saving for the current page
 * This function can be called from the browser console to force TOC extraction
 * @returns {Promise<void>}
 */
export async function forceTOCExtractionAndSave() {
  console.log('🔄 Manually triggering TOC extraction and save...');
  
  try {
    const currentUrl = window.location.href;
    const currentTitle = document.title;
    
    console.log('📊 Current page info:', {
      url: currentUrl,
      title: currentTitle
    });
    
    // Extract TOC
    console.log('🔄 Extracting TOC...');
    const tocData = await extractPageTOC();
    console.log('✅ TOC extracted:', {
      headingCount: tocData.length,
      headings: tocData.map(h => ({ text: h.text, level: h.level }))
    });
    
    if (tocData.length > 0) {
      // Save TOC
      console.log('🔄 Saving TOC to chapterMap...');
      const saveResult = await saveTOCToChapterMap(tocData, currentUrl, currentTitle);
      
      if (saveResult) {
        console.log('✅ TOC saved successfully!');
        
        // Verify save
        console.log('🔄 Verifying save...');
        const retrievedData = await getTOCFromChapterMap(currentUrl);
        if (retrievedData) {
          console.log('✅ Verification successful:', {
            url: retrievedData.url,
            title: retrievedData.title,
            headingCount: retrievedData.headingCount,
            lastUpdated: retrievedData.lastUpdated
          });
        } else {
          console.error('❌ Verification failed - data not found');
        }
      } else {
        console.error('❌ TOC save failed');
      }
    } else {
      console.log('📝 No headings found on this page');
    }
    
  } catch (error) {
    console.error('❌ Manual TOC extraction failed:', error);
    console.error('❌ Error details:', error.message, error.stack);
  }
}

// Make test functions available globally for console access
if (typeof window !== 'undefined') {
  window.testTOCSaveFunction = testTOCSaveFunction;
  window.forceTOCExtractionAndSave = forceTOCExtractionAndSave;
  window.cleanupDuplicateChapterMapEntries = cleanupDuplicateChapterMapEntries;
  window.normalizeUrl = normalizeUrl;
}