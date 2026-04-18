import { extractTOCAsync, extractTOCWithDebug } from "../utils/tocExtractor.js";

// Add this helper function to generate valid selectors
// function generateValidSelector(element) {
//     // Generate a unique data attribute for the element
//     const uniqueId = `fuzzy-match-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
//     element.setAttribute('data-fuzzy-id', uniqueId);
//     return `[data-fuzzy-id="${uniqueId}"]`;
// }

// Improved function to compute a fast and accurate n-gram fingerprint for text
function computeTextFingerprint(text, n = 3) {
    // Normalize and clean the text
    const normalized = text.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (normalized.length < n) return 0;
    
    // Extract character n-grams for better accuracy
    const ngrams = new Set();
    for (let i = 0; i <= normalized.length - n; i++) {
        ngrams.add(normalized.substring(i, i + n));
    }
    
    // Fast hash of n-gram set (much faster than average char codes)
    let hash = 0;
    for (const ngram of ngrams) {
        for (let i = 0; i < ngram.length; i++) {
            hash = ((hash << 5) - hash + ngram.charCodeAt(i)) & 0xffffffff;
        }
    }
    return Math.abs(hash);
}

// Add this global structure to store our sorted text map
window.textMap = {
    sortedEntries: [], // [{fingerprint: number, id: string, text: string}]
    initialized: false
};

// Generate stable ID for paragraphs based on content hash - OPTIMIZED FOR SPEED
export function generateStableId(textContent, existingIds = new Set()) {
    // Fast hash using only first 100 chars + length + last 50 chars for speed
    const text = textContent.trim();
    const len = text.length;
    
    // For very short text, use the whole thing
    if (len <= 20) {
        const hash = simpleHash(text.toLowerCase());
        return `p-${Math.abs(hash).toString(36)}`;
    }
    
    // For longer text, use strategic sampling for speed
    const start = text.substring(0, Math.min(50, len)).toLowerCase();
    const end = text.substring(Math.max(0, len - 30)).toLowerCase();
    const middle = len > 100 ? text.substring(Math.floor(len/2), Math.floor(len/2) + 20).toLowerCase() : '';
    
    // Combine key parts with length for uniqueness
    const combined = `${start}${middle}${end}${len}`;
    const hash = simpleHash(combined);
    
    let baseId = `p-${Math.abs(hash).toString(36)}`;
    
    // Handle potential conflicts by adding a suffix
    let finalId = baseId;
    let counter = 1;
    while (existingIds.has(finalId)) {
        finalId = `${baseId}-${counter}`;
        counter++;
    }
    
    return finalId;
}

// Ultra-fast hash function - optimized for speed
function simpleHash(str) {
    let hash = 0;
    // Process every 2nd character for even more speed on long strings
    for (let i = 0; i < str.length; i += 2) {
        hash = ((hash << 5) - hash + str.charCodeAt(i)) & 0xffffffff;
    }
    return hash;
}

// Modified precompute function - OPTIMIZED FOR SPEED
export function precomputeParagraphFingerprints() {
    const paragraphs = document.querySelectorAll('p');
    const newEntries = [];
    const existingIds = new Set();
    
    // Performance optimization: limit processing on very large pages
    const maxParagraphs = 1000;
    const paragraphsToProcess = paragraphs.length > maxParagraphs ? 
        Array.from(paragraphs).slice(0, maxParagraphs) : paragraphs;

    // First pass: collect existing IDs to avoid conflicts (fast)
    paragraphsToProcess.forEach(p => {
        if (p.hasAttribute('data-fuzzy-id')) {
            existingIds.add(p.getAttribute('data-fuzzy-id'));
        }
    });

    // Second pass: only process paragraphs without IDs (much faster)
    paragraphsToProcess.forEach(p => {
        // Skip if already processed - this is the key optimization
        if (!p.hasAttribute('data-fuzzy-id')) {
            const textContent = p.textContent;
            
            // Skip empty or very short paragraphs to avoid noise
            if (textContent.trim().length < 10) {
                return;
            }
            
            const stableId = generateStableId(textContent, existingIds);
            const fingerprint = computeTextFingerprint(textContent);
            
            // Add to existing IDs set to prevent conflicts in this batch
            existingIds.add(stableId);
            
            // Add attributes to paragraph
            p.setAttribute('data-fuzzy-id', stableId);
            p.setAttribute('data-fingerprint', fingerprint.toString());
            
            // Store minimal information
            newEntries.push({
                fingerprint,
                id: stableId,
                text: textContent // Store text to avoid DOM reads later
            });
        }
    });

    if (newEntries.length > 0) {
        if (!window.textMap.initialized) {
            window.textMap.sortedEntries = newEntries.sort((a, b) => a.fingerprint - b.fingerprint);
            window.textMap.initialized = true;
        } else {
            // Merge new entries into existing sorted list
            window.textMap.sortedEntries = mergeSort(
                window.textMap.sortedEntries,
                newEntries.sort((a, b) => a.fingerprint - b.fingerprint)
            );
        }
        // console.log(`TextMap updated: ${window.textMap.sortedEntries.length} total entries`);
    }
    
    // TOC extraction is now handled in main index.js to prevent duplicates
    // extractTOCAsync();
}

// Binary search to find closest matches
function findClosestFingerprints(targetFingerprint, numMatches = 10) {
    const entries = window.textMap.sortedEntries;
    if (!entries.length) return [];

    // Find closest index using binary search
    let left = 0;
    let right = entries.length - 1;
    let closest = 0;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (entries[mid].fingerprint === targetFingerprint) {
            closest = mid;
            break;
        }
        
        if (entries[mid].fingerprint < targetFingerprint) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
        
        if (Math.abs(entries[mid].fingerprint - targetFingerprint) < 
            Math.abs(entries[closest].fingerprint - targetFingerprint)) {
            closest = mid;
        }
    }

    // Gather nearest neighbors
    const results = [];
    let leftPtr = closest;
    let rightPtr = closest + 1;

    while (results.length < numMatches && (leftPtr >= 0 || rightPtr < entries.length)) {
        const leftDiff = leftPtr >= 0 ? 
            Math.abs(entries[leftPtr].fingerprint - targetFingerprint) : Infinity;
        const rightDiff = rightPtr < entries.length ? 
            Math.abs(entries[rightPtr].fingerprint - targetFingerprint) : Infinity;

        if (leftDiff <= rightDiff) {
            results.push(entries[leftPtr]);
            leftPtr--;
        } else {
            results.push(entries[rightPtr]);
            rightPtr++;
        }
    }

    return results;
}

// Fast n-gram similarity calculation (much faster than Levenshtein)
function computeNgramSimilarity(text1, text2, n = 3) {
    const getNgrams = (text) => {
        const normalized = text.toLowerCase().replace(/[^a-z0-9]/g, '');
        const ngrams = new Set();
        for (let i = 0; i <= normalized.length - n; i++) {
            ngrams.add(normalized.substring(i, i + n));
        }
        return ngrams;
    };
    
    const ngrams1 = getNgrams(text1);
    const ngrams2 = getNgrams(text2);
    
    if (ngrams1.size === 0 && ngrams2.size === 0) return 1;
    if (ngrams1.size === 0 || ngrams2.size === 0) return 0;
    
    const intersection = new Set([...ngrams1].filter(x => ngrams2.has(x)));
    const union = new Set([...ngrams1, ...ngrams2]);
    
    return intersection.size / union.size;
}

// Improved findSimilarParagraphsNonBlocking function - much faster and more accurate
export async function findSimilarParagraphsNonBlocking(aiResponse, numMatches = 5) {
    const responseFingerprint = computeTextFingerprint(aiResponse);
    
    console.time('Finding rough matches');
    const roughMatches = findClosestFingerprints(responseFingerprint, 15); // Get more candidates for better results
    console.timeEnd('Finding rough matches');
    
    console.time('N-gram similarity calculation');
    const similarities = [];
    
    for (const match of roughMatches) {
        // Find element by ID when needed
        const element = document.querySelector(`[data-fuzzy-id="${match.id}"]`);
        if (!element) continue;

        // Use fast n-gram similarity instead of expensive Levenshtein distance
        const similarity = computeNgramSimilarity(aiResponse, match.text);
        
        similarities.push({
            selector: `[data-fuzzy-id="${match.id}"]`,
            element,
            text: match.text,
            similarity
        });
    }
    console.timeEnd('N-gram similarity calculation');

    return similarities
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, numMatches);
}

// Helper function to merge sorted arrays (unchanged)
function mergeSort(arr1, arr2) {
    const result = [];
    let i = 0, j = 0;

    while (i < arr1.length && j < arr2.length) {
        if (arr1[i].fingerprint <= arr2[j].fingerprint) {
            result.push(arr1[i]);
            i++;
        } else {
            result.push(arr2[j]);
            j++;
        }
    }

    return result.concat(arr1.slice(i)).concat(arr2.slice(j));
}

