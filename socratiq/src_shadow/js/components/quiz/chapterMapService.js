// chapterMapService.js - Book-agnostic chapter system using chapterMap data
import { getAllChapterMapEntries } from '../../libs/utils/tocExtractor.js';
import { getDBInstance } from '../../libs/utils/indexDb.js';
import { formatChapterTaxonomyPrompt, resolveChapterTaxonomy } from '../../../../src_shared/taxonomy-map.js';

/**
 * Build a taxonomy-grounded concept-map generation prompt for a chapter URL/path.
 * @param {string} urlOrPath - Chapter URL or path
 * @returns {string} - Concept-map generation prompt aligned with canonical taxonomy subjects
 */
export function buildChapterConceptMapPrompt(urlOrPath = '') {
    const taxonomyGrounding = formatChapterTaxonomyPrompt(urlOrPath);
    return `Generate a chapter concept map aligned with the canonical ML Systems Taxonomy subjects. ${taxonomyGrounding}`.trim();
}

/**
 * Match a quiz title to a page in chapterMap based on various strategies
 * @param {string} quizTitle - The quiz title to match
 * @param {Array} chapterMapEntries - Array of chapterMap entries
 * @returns {Object|null} - The matching page or null
 */
function matchQuizToPage(score, chapterMapEntries) {
    if (!score || !Array.isArray(chapterMapEntries)) return null;
    const pageUrl = score.pageUrl || score.originalUrl || score.normalizedUrl || score.url;
    if (score.quizId && !pageUrl) return null;
    if (Array.isArray(score.sourcePageUrls) && score.sourcePageUrls.some(url =>
        normalizeUrlForCumulativeQuiz(url) !== normalizeUrlForCumulativeQuiz(pageUrl))) return null;
    if (pageUrl) {
        const canonical = normalizeUrlForCumulativeQuiz(pageUrl);
        return chapterMapEntries.find(page => normalizeUrlForCumulativeQuiz(page.url) === canonical) || null;
    }
    // Older records have only a title. Assign them only when the complete title
    // or section heading identifies one page across all visited volumes.
    const title = normalizeQuizTitle(score.title || score.displayTitle || score.quizTitle);
    if (!title) return null;
    const exact = chapterMapEntries.filter(page => {
        const headings = page.tocData || page.toc || [];
        return normalizeQuizTitle(page.title) === title || (Array.isArray(headings) &&
            headings.some(heading => normalizeQuizTitle(heading.text || heading.title) === title));
    });
    if (exact.length) return exact.length === 1 ? exact[0] : null;

    // Retain matching numbered section titles to their chapter only when both
    // the chapter number and meaningful title words agree, with no ambiguity.
    const words = title.replace(/\d+/g, '').split(/\s+/).filter(word => word.length > 2);
    const chapterNumber = title.match(/^(\d+)\./)?.[1];
    const matches = chapterMapEntries.filter(page => {
        const pageTitle = normalizeQuizTitle(page.title);
        const pageNumber = pageTitle.match(/^(\d+)(?:\.|\s)/)?.[1];
        return chapterNumber && pageNumber === chapterNumber && words.length > 0 &&
            words.every(word => pageTitle.split(/\s+/).includes(word));
    });
    return matches.length === 1 ? matches[0] : null;
}

function normalizeQuizTitle(title) {
    return typeof title === 'string' ? title.toLowerCase()
        .replace(/^cumulative quiz\s*-\s*/i, '').replace(/[^\w.\s]/g, '')
        .replace(/\s+/g, ' ').trim() : '';
}

/**
 * Calculate progress for a specific page based on related quiz scores and cumulative quiz data
 * @param {Object} page - The chapterMap page entry
 * @param {Array} quizScores - Array of quiz high scores
 * @param {Array} pageCompletions - Array of page completion records
 * @param {Array} cumulativeQuizScores - Array of cumulative quiz scores
 * @returns {Object} - Progress data for the page
 */
function calculatePageProgress(page, quizScores, pageCompletions = [], cumulativeQuizScores = [], chapterMapEntries = []) {
    const pageUrl = page?.url || '';
    const enhancedRecord = cumulativeQuizScores.find(record =>
        normalizeUrlForCumulativeQuiz(record.normalizedUrl || record.originalUrl) === normalizeUrlForCumulativeQuiz(pageUrl));
    const pageCompletion = pageCompletions.find(record =>
        normalizeUrlForCumulativeQuiz(record.pageUrl) === normalizeUrlForCumulativeQuiz(pageUrl));
    const history = Array.isArray(enhancedRecord?.quizHistory) ? enhancedRecord.quizHistory : [];
    const sectionHistory = history.filter(quiz => quiz.quizType === 'section');
    const relatedQuizzes = quizScores.filter(score => {
        const matchingPage = matchQuizToPage(score, chapterMapEntries);
        return matchingPage?.url === pageUrl && !score.isCumulative && score.quizType !== 'cumulative';
    });
    const results = sectionHistory.length ? sectionHistory : relatedQuizzes;
    const passedQuizzes = sectionHistory.length
        ? sectionHistory.filter(quiz => quiz.score?.passed).length
        : relatedQuizzes.filter(score => Number.parseFloat(score.percentageScore) >= 60).length;
    const completedByCumulativeQuiz = Boolean(enhancedRecord?.completedByCumulativeQuiz || pageCompletion?.completedByCumulativeQuiz);
    const cumulativeResult = [...history].reverse().find(quiz =>
        quiz.quizType === 'cumulative' && (!completedByCumulativeQuiz || quiz.score?.passed));
    const legacyCompletion = pageCompletion?.completedByCumulativeQuiz ? pageCompletion : null;
    const cumulativeScoreRecord = cumulativeResult ? cumulativeResult.score : legacyCompletion?.score;
    let cumulativeScore = null;
    if (Number.isFinite(cumulativeScoreRecord?.percentage)) {
        cumulativeScore = cumulativeScoreRecord.percentage;
    } else if (Number.isFinite(cumulativeScoreRecord?.correct) &&
        Number.isFinite(cumulativeScoreRecord?.total) && cumulativeScoreRecord.total > 0) {
        cumulativeScore = (cumulativeScoreRecord.correct / cumulativeScoreRecord.total) * 100;
    }
    return {
        pageUrl,
        pageTitle: page?.title || '',
        passedQuizzes,
        totalAttempted: results.length,
        // Section results do not establish how many quizzes a chapter contains.
        totalQuizzes: null,
        completionPercentage: completedByCumulativeQuiz ? 100 : null,
        completionBasis: completedByCumulativeQuiz ? 'cumulative-quiz' : null,
        passRate: results.length ? (passedQuizzes / results.length) * 100 : null,
        lastUpdated: page?.lastUpdated,
        completedByCumulativeQuiz,
        cumulativeQuizTitle: cumulativeResult ? cumulativeResult.quizTitle ?? null : legacyCompletion?.cumulativeQuizTitle ?? null,
        cumulativeScore,
        hasSectionQuizzes: results.length > 0,
        hasCumulativeQuiz: completedByCumulativeQuiz || history.some(quiz => quiz.quizType === 'cumulative'),
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
        const chapters = sortedPages.map((page, index) => {
            const taxonomy = resolveChapterTaxonomy(page.url);
            return {
                chapter: index + 1, // Sequential numbering for dashboard
                title: page.title,
                url: page.url,
                originalPage: page, // Keep reference to original data
                taxonomyGroup: taxonomy ? taxonomy.primaryGroup : (page.taxonomyGroup || null),
                taxonomySubjects: taxonomy ? taxonomy.subjects : (page.taxonomySubjects || []),
                taxonomyPrompt: formatChapterTaxonomyPrompt(page.url),
                quizzesTaken: 0, // Will be updated by progress calculation
                totalQuizzes: null, // The complete quiz inventory is unknown
                passedQuizzes: 0 // Will be updated by progress calculation
            };
        });

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
            const progress = calculatePageProgress(page, quizScores, pageCompletions, cumulativeQuizScores, chapterMapEntries);
            
            // Use page URL as key for the progress map
            chapterProgress.set(page.url, {
                passedQuizzes: progress.passedQuizzes,
                totalAttempted: progress.totalAttempted,
                completionPercentage: progress.completionPercentage,
                completionBasis: progress.completionBasis,
                totalQuizzes: progress.totalQuizzes,
                passRate: progress.passRate,
                hasSectionQuizzes: progress.hasSectionQuizzes,
                hasCumulativeQuiz: progress.hasCumulativeQuiz,
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
                totalQuizzes: pageProgress?.totalQuizzes ?? null,
                completionPercentage: pageProgress?.completionPercentage ?? null,
                completedByCumulativeQuiz: Boolean(pageProgress?.completedByCumulativeQuiz)
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