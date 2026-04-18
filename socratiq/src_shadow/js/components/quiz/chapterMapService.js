// chapterMapService.js - Book-agnostic chapter system using chapterMap data
import { getAllChapterMapEntries } from '../../libs/utils/tocExtractor.js';
import { getDBInstance } from '../../libs/utils/indexDb.js';

/**
 * Match a quiz title to a page in chapterMap based on various strategies
 * @param {string} quizTitle - The quiz title to match
 * @param {Array} chapterMapEntries - Array of chapterMap entries
 * @returns {Object|null} - The matching page or null
 */
function matchQuizToPage(quizTitle, chapterMapEntries) {
    if (!quizTitle || !chapterMapEntries || chapterMapEntries.length === 0) {
        return null;
    }

    // Strategy 1: Extract numbers from quiz title (e.g., "11.4 System Benchmarking" -> ["11", "4"])
    const quizNumbers = quizTitle.match(/\d+/g);
    
    // Strategy 2: Find page with matching numbers in title
    if (quizNumbers && quizNumbers.length > 0) {
        const matchingPage = chapterMapEntries.find(page => {
            const pageNumbers = page.title.match(/\d+/g);
            if (!pageNumbers) return false;
            
            // Check if any quiz number appears in page numbers
            return quizNumbers.some(qNum => pageNumbers.includes(qNum));
        });
        
        if (matchingPage) {
            return matchingPage;
        }
    }
    
    // Strategy 3: Fallback to title similarity (fuzzy matching)
    return findBestTitleMatch(quizTitle, chapterMapEntries);
}

/**
 * Find the best title match using fuzzy string matching
 * @param {string} quizTitle - The quiz title to match
 * @param {Array} chapterMapEntries - Array of chapterMap entries
 * @returns {Object|null} - The best matching page or null
 */
function findBestTitleMatch(quizTitle, chapterMapEntries) {
    if (!quizTitle || !chapterMapEntries || chapterMapEntries.length === 0) {
        return null;
    }

    // Extract key words from quiz title (remove numbers and common words)
    const quizWords = quizTitle.toLowerCase()
        .replace(/\d+/g, '')
        .replace(/[^\w\s]/g, '')
        .split(/\s+/)
        .filter(word => word.length > 2 && !['the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for'].includes(word));

    let bestMatch = null;
    let bestScore = 0;

    chapterMapEntries.forEach(page => {
        const pageWords = page.title.toLowerCase()
            .replace(/\d+/g, '')
            .replace(/[^\w\s]/g, '')
            .split(/\s+/)
            .filter(word => word.length > 2);

        // Calculate similarity score
        const commonWords = quizWords.filter(word => pageWords.includes(word));
        const score = commonWords.length / Math.max(quizWords.length, pageWords.length);

        if (score > bestScore && score > 0.3) { // Minimum 30% similarity
            bestScore = score;
            bestMatch = page;
        }
    });

    return bestMatch;
}

/**
 * Calculate progress for a specific page based on related quiz scores and cumulative quiz data
 * @param {Object} page - The chapterMap page entry
 * @param {Array} quizScores - Array of quiz high scores
 * @param {Array} pageCompletions - Array of page completion records
 * @param {Array} cumulativeQuizScores - Array of cumulative quiz scores
 * @returns {Object} - Progress data for the page
 */
function calculatePageProgress(page, quizScores, pageCompletions = [], cumulativeQuizScores = []) {
    if (!page || !quizScores) {
        return {
            pageUrl: page?.url || '',
            pageTitle: page?.title || '',
            passedQuizzes: 0,
            totalAttempted: 0,
            completionPercentage: 0
        };
    }

    // Check if page is completed by cumulative quiz using the enhanced table
    const enhancedRecord = cumulativeQuizScores.find(record => {
        // Normalize both URLs for comparison
        const recordUrl = record.normalizedUrl || record.originalUrl;
        const pageUrl = page.url;
        
        // Try exact match first
        if (recordUrl === pageUrl) return true;
        
        // Try normalized comparison
        try {
            const normalizedPageUrl = normalizeUrlForCumulativeQuiz(pageUrl);
            return recordUrl === normalizedPageUrl;
        } catch (error) {
            return false;
        }
    });

    // If page is completed by cumulative quiz, mark as 100% complete
    if (enhancedRecord && enhancedRecord.completedByCumulativeQuiz) {
        return {
            pageUrl: page.url,
            pageTitle: page.title,
            passedQuizzes: 10, // Full completion
            totalAttempted: 1, // One cumulative quiz
            completionPercentage: 100,
            totalQuizzes: 10,
            lastUpdated: page.lastUpdated,
            completedByCumulativeQuiz: true,
            cumulativeQuizTitle: enhancedRecord.cumulativeQuizTitle,
            cumulativeScore: enhancedRecord.lastQuizScore?.percentage || 0
        };
    }

    // Fallback: Check old pageCompletions system
    const pageCompletion = pageCompletions.find(completion => 
        completion.pageUrl === page.url
    );

    if (pageCompletion && pageCompletion.completedByCumulativeQuiz) {
        return {
            pageUrl: page.url,
            pageTitle: page.title,
            passedQuizzes: 10, // Full completion
            totalAttempted: 1, // One cumulative quiz
            completionPercentage: 100,
            totalQuizzes: 10,
            lastUpdated: page.lastUpdated,
            completedByCumulativeQuiz: true,
            cumulativeQuizTitle: pageCompletion.cumulativeQuizTitle
        };
    }

    // Calculate section quiz progress from enhanced table
    let sectionQuizProgress = { passedQuizzes: 0, totalAttempted: 0 };
    
    if (enhancedRecord && enhancedRecord.quizHistory) {
        const sectionQuizzes = enhancedRecord.quizHistory.filter(quiz => quiz.quizType === 'section');
        sectionQuizProgress.totalAttempted = sectionQuizzes.length;
        sectionQuizProgress.passedQuizzes = sectionQuizzes.filter(quiz => quiz.score.passed).length;
    } else {
        // Fallback to old system for section quizzes
        const relatedQuizzes = quizScores.filter(score => {
            const matchingPage = matchQuizToPage(score.quizTitle, [page]);
            return matchingPage && matchingPage.url === page.url && !score.isCumulative;
        });

        sectionQuizProgress.totalAttempted = relatedQuizzes.length;
        sectionQuizProgress.passedQuizzes = relatedQuizzes.filter(score => 
            parseFloat(score.percentageScore) >= 60
        ).length;
    }

    // Assume 10 quizzes per "chapter" for now (can be made dynamic later)
    const totalQuizzes = 10;
    const completionPercentage = Math.min((sectionQuizProgress.passedQuizzes / totalQuizzes) * 100, 100);

    return {
        pageUrl: page.url,
        pageTitle: page.title,
        passedQuizzes: sectionQuizProgress.passedQuizzes,
        totalAttempted: sectionQuizProgress.totalAttempted,
        completionPercentage: completionPercentage,
        totalQuizzes: totalQuizzes,
        lastUpdated: page.lastUpdated,
        completedByCumulativeQuiz: false,
        hasSectionQuizzes: sectionQuizProgress.totalAttempted > 0,
        hasCumulativeQuiz: enhancedRecord && enhancedRecord.quizHistory && 
            enhancedRecord.quizHistory.some(quiz => quiz.quizType === 'cumulative')
    };
}

/**
 * Normalize URL for cumulative quiz tracking (imported from quiz-storage.js)
 * @param {string} url - The URL to normalize
 * @returns {string} - The normalized URL
 */
function normalizeUrlForCumulativeQuiz(url) {
    try {
        const urlObj = new URL(url);
        // Remove query parameters, fragments, and trailing slashes
        return `${urlObj.protocol}//${urlObj.host}${urlObj.pathname}`.replace(/\/$/, '');
    } catch (error) {
        console.warn('Error normalizing URL for cumulative quiz:', error);
        return url;
    }
}

/**
 * Get all chapters from chapterMap data, formatted for the dashboard
 * @returns {Promise<Array>} - Array of chapter data for dashboard
 */
export async function getChaptersFromChapterMap() {
    try {
        const chapterMapEntries = await getAllChapterMapEntries();
        
        if (!chapterMapEntries || chapterMapEntries.length === 0) {
            console.warn('No chapterMap entries found');
            return [];
        }

        // Filter out non-content pages (like index pages, etc.)
        const contentPages = chapterMapEntries.filter(page => {
            // Skip pages that are likely not content chapters
            const skipPatterns = [
                /index\.html?$/i,
                /table.of.contents/i,
                /toc/i,
                /preface/i,
                /introduction/i
            ];
            
            return !skipPatterns.some(pattern => 
                pattern.test(page.url) || pattern.test(page.title)
            );
        });

        // Sort pages by title (assuming they contain numbers)
        const sortedPages = contentPages.sort((a, b) => {
            const aNumbers = a.title.match(/\d+/g);
            const bNumbers = b.title.match(/\d+/g);
            
            if (aNumbers && bNumbers) {
                return parseInt(aNumbers[0]) - parseInt(bNumbers[0]);
            }
            
            return a.title.localeCompare(b.title);
        });

        // Transform to dashboard format
        const chapters = sortedPages.map((page, index) => ({
            chapter: index + 1, // Sequential numbering for dashboard
            title: page.title,
            url: page.url,
            originalPage: page, // Keep reference to original data
            quizzesTaken: 0, // Will be updated by progress calculation
            totalQuizzes: 10, // Default assumption
            passedQuizzes: 0 // Will be updated by progress calculation
        }));

        console.log(`📚 Generated ${chapters.length} chapters from chapterMap data`);
        return chapters;

    } catch (error) {
        console.error('Error getting chapters from chapterMap:', error);
        return [];
    }
}

/**
 * Get chapter progress data using chapterMap instead of hardcoded chapters
 * @returns {Promise<Map>} - Map of chapter progress data
 */
export async function getChapterProgressFromChapterMap() {
    try {
        const dbManager = await getDBInstance();
        if (!dbManager) {
            throw new Error('Database not available');
        }

        // Get quiz scores, chapterMap data, page completions, and cumulative quiz scores
        const [quizScores, chapterMapEntries, pageCompletions, cumulativeQuizScores] = await Promise.all([
            dbManager.getAll('quizHighScores'),
            getAllChapterMapEntries(),
            dbManager.getAll('pageCompletions').catch(() => []), // Handle case where store doesn't exist yet
            dbManager.getAll('cumulativeQuizScores').catch(() => []) // Handle case where store doesn't exist yet
        ]);

        const chapterProgress = new Map();

        // Process each page in chapterMap
        chapterMapEntries.forEach(page => {
            const progress = calculatePageProgress(page, quizScores, pageCompletions, cumulativeQuizScores);
            
            // Use page URL as key for the progress map
            chapterProgress.set(page.url, {
                passedQuizzes: progress.passedQuizzes,
                totalAttempted: progress.totalAttempted,
                completionPercentage: progress.completionPercentage,
                pageTitle: progress.pageTitle,
                pageUrl: progress.pageUrl,
                completedByCumulativeQuiz: progress.completedByCumulativeQuiz,
                cumulativeQuizTitle: progress.cumulativeQuizTitle,
                cumulativeScore: progress.cumulativeScore
            });
        });

        console.log(`📊 Calculated progress for ${chapterProgress.size} pages from chapterMap`);
        console.log(`🎯 Found ${pageCompletions.length} page completions from old system`);
        console.log(`🎯 Found ${cumulativeQuizScores.length} cumulative quiz records`);
        return chapterProgress;

    } catch (error) {
        console.error('Error getting chapter progress from chapterMap:', error);
        return new Map();
    }
}

/**
 * Get chapter progress using the new chapterMap system with fallback to old system
 * @returns {Promise<Map>} - Map of chapter progress data
 */
export async function getChapterProgress() {
    try {
        // Try new chapterMap system first
        const chapterMapProgress = await getChapterProgressFromChapterMap();
        
        if (chapterMapProgress.size > 0) {
            console.log('✅ Using chapterMap-based progress system');
            return chapterMapProgress;
        }

        // Fallback to old system if chapterMap is empty
        console.warn('⚠️ chapterMap is empty, falling back to old chapter system');
        return await getChapterProgressLegacy();

    } catch (error) {
        console.error('Error in getChapterProgress:', error);
        // Fallback to legacy system
        return await getChapterProgressLegacy();
    }
}

/**
 * Legacy chapter progress function (fallback)
 * @returns {Promise<Map>} - Map of chapter progress data
 */
async function getChapterProgressLegacy() {
    try {
        const dbManager = await getDBInstance();
        const scores = await dbManager.getAll('quizHighScores');
        const chapterProgress = new Map();
        
        // Process each quiz score using old logic
        scores.forEach(score => {
            // Extract chapter number from quiz title (e.g., "11.4 System Benchmarking" -> 11)
            const chapterMatch = score.quizTitle.match(/^(\d+)\./);
            if (chapterMatch) {
                const chapter = parseInt(chapterMatch[1]);
                const passed = parseFloat(score.percentageScore) >= 60;
                
                if (!chapterProgress.has(chapter)) {
                    chapterProgress.set(chapter, {
                        passedQuizzes: 0,
                        totalAttempted: 0
                    });
                }
                
                const progress = chapterProgress.get(chapter);
                progress.totalAttempted++;
                if (passed) progress.passedQuizzes++;
            }
        });
        
        return chapterProgress;
    } catch (error) {
        console.error('Error in legacy chapter progress:', error);
        return new Map();
    }
}

/**
 * Get dashboard-ready chapter data with progress information
 * @returns {Promise<Array>} - Array of chapter data with progress for dashboard
 */
export async function getDashboardChapterData() {
    try {
        const [chapters, progress] = await Promise.all([
            getChaptersFromChapterMap(),
            getChapterProgressFromChapterMap()
        ]);

        // Combine chapter data with progress information
        const dashboardData = chapters.map(chapter => {
            const pageProgress = progress.get(chapter.url);
            
            return {
                ...chapter,
                quizzesTaken: pageProgress?.totalAttempted || 0,
                passedQuizzes: pageProgress?.passedQuizzes || 0,
                completionPercentage: pageProgress?.completionPercentage || 0
            };
        });

        return dashboardData;

    } catch (error) {
        console.error('Error getting dashboard chapter data:', error);
        return [];
    }
}

/**
 * Test function to verify chapterMap integration
 * @returns {Promise<Object>} - Test results
 */
export async function testChapterMapIntegration() {
    console.log('🧪 Testing chapterMap integration...');
    
    try {
        const chapterMapEntries = await getAllChapterMapEntries();
        const chapters = await getChaptersFromChapterMap();
        const progress = await getChapterProgressFromChapterMap();
        const dashboardData = await getDashboardChapterData();

        const results = {
            success: true,
            chapterMapEntries: chapterMapEntries.length,
            chapters: chapters.length,
            progressEntries: progress.size,
            dashboardData: dashboardData.length,
            sampleData: {
                firstChapter: chapters[0],
                firstProgress: progress.values().next().value,
                firstDashboard: dashboardData[0]
            }
        };

        console.log('✅ ChapterMap integration test results:', results);
        return results;

    } catch (error) {
        console.error('❌ ChapterMap integration test failed:', error);
        return {
            success: false,
            error: error.message
        };
    }
}

/**
 * Test function to verify cumulative quiz progress tracking
 * @returns {Promise<Object>} - Test results
 */
export async function testCumulativeQuizProgress() {
    console.log('🧪 Testing cumulative quiz progress tracking...');
    
    try {
        const dbManager = await getDBInstance();
        
        // Test 1: Check if stores exist
        const storeNames = Array.from(dbManager.db.objectStoreNames);
        const hasPageCompletions = storeNames.includes('pageCompletions');
        const hasCumulativeQuizScores = storeNames.includes('cumulativeQuizScores');
        
        // Test 2: Get page completions data (old system)
        const pageCompletions = hasPageCompletions ? 
            await dbManager.getAll('pageCompletions') : [];
        
        // Test 3: Get enhanced quiz scores (new system)
        const enhancedQuizScores = hasCumulativeQuizScores ? 
            await dbManager.getAll('cumulativeQuizScores') : [];
        
        // Test 4: Get quiz stats with cumulative quizzes
        const quizStats = await dbManager.getAll('quizStats');
        const cumulativeQuizzes = quizStats.filter(stat => stat.isCumulative);
        
        // Test 5: Analyze enhanced quiz data
        const enhancedStats = {
            totalPages: enhancedQuizScores.length,
            pagesWithCumulativeQuizzes: enhancedQuizScores.filter(r => r.completedByCumulativeQuiz).length,
            pagesWithSectionQuizzes: enhancedQuizScores.filter(r => 
                r.quizHistory && r.quizHistory.some(q => q.quizType === 'section')
            ).length,
            totalQuizAttempts: enhancedQuizScores.reduce((sum, r) => 
                sum + (r.quizHistory ? r.quizHistory.length : 0), 0
            )
        };
        
        // Test 6: Test progress calculation with enhanced data
        const progress = await getChapterProgressFromChapterMap();
        const completedPages = Array.from(progress.values()).filter(p => p.completedByCumulativeQuiz);
        const pagesWithSectionQuizzes = Array.from(progress.values()).filter(p => p.hasSectionQuizzes);
        
        // Test 7: Test URL normalization
        const testUrls = [
            'http://localhost:3000/page.html?scroll=100',
            'http://localhost:3000/page.html#section1',
            'http://localhost:3000/page/'
        ];
        const normalizedUrls = testUrls.map(url => normalizeUrlForCumulativeQuiz(url));
        
        const results = {
            success: true,
            stores: {
                hasPageCompletionsStore: hasPageCompletions,
                hasCumulativeQuizScoresStore: hasCumulativeQuizScores,
                allStores: storeNames
            },
            data: {
                pageCompletionsCount: pageCompletions.length,
                enhancedQuizScoresCount: enhancedQuizScores.length,
                cumulativeQuizzesCount: cumulativeQuizzes.length,
                completedPagesCount: completedPages.length,
                pagesWithSectionQuizzesCount: pagesWithSectionQuizzes.length
            },
            enhancedStats: enhancedStats,
            urlNormalization: {
                testUrls: testUrls,
                normalizedUrls: normalizedUrls
            },
            samples: {
                pageCompletions: pageCompletions.slice(0, 2),
                enhancedQuizScores: enhancedQuizScores.slice(0, 2),
                cumulativeQuizzes: cumulativeQuizzes.slice(0, 2),
                completedPages: completedPages.slice(0, 2),
                pagesWithSectionQuizzes: pagesWithSectionQuizzes.slice(0, 2)
            }
        };

        console.log('✅ Cumulative quiz progress test results:', results);
        return results;

    } catch (error) {
        console.error('❌ Cumulative quiz progress test failed:', error);
        return {
            success: false,
            error: error.message
        };
    }
}

// Make test functions available globally for console access
if (typeof window !== 'undefined') {
    window.testChapterMapIntegration = testChapterMapIntegration;
    window.testCumulativeQuizProgress = testCumulativeQuizProgress;
    window.normalizeUrlForCumulativeQuiz = normalizeUrlForCumulativeQuiz;
}