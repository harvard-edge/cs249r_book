// quiz-storage.js
import { showPopover } from "../../libs/utils/utils.js";
import { getDBInstance } from "../../libs/utils/indexDb.js";

/**
 * Normalize URL for cumulative quiz tracking by removing query parameters and fragments
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

async function updateQuizStreak(score) {
    const isPerfectScore = score.correct === score.total;
    const currentStreak = parseInt(localStorage.getItem('quizStreak') || '0');
    
    if (isPerfectScore) {
        localStorage.setItem('quizStreak', currentStreak + 1);
    } else {
        localStorage.setItem('quizStreak', '0');
    }
}

function incrementQuizCounter() {
    const currentCount = parseInt(localStorage.getItem('totalQuizzes') || '0');
    localStorage.setItem('totalQuizzes', currentCount + 1);
}

async function initDB() {
    try {
        const dbManager = await getDBInstance();
        if (!dbManager || !dbManager.db) {
            throw new Error('Database not properly initialized');
        }
        return dbManager;
    } catch (error) {
        console.error('Error initializing database:', error);
        throw error;
    }
}

async function getHighScore(quizTitle) {
    try {
        const dbManager = await initDB();
        return await dbManager.getByKey('quizHighScores', quizTitle);
    } catch (error) {
        console.error('Error getting high score:', error);
        throw error;
    }
}

async function updateHighScore(quizTitle, newScore) {
    if (!quizTitle) {
        throw new Error('Quiz title is required for updating high score');
    }

    try {
        const dbManager = await initDB();
        const highScoreData = {
            quizTitle: quizTitle,
            score: newScore,
            percentageScore: (newScore.correct / newScore.total * 100).toFixed(2),
            achievedAt: new Date().toISOString()
        };
        
        await dbManager.update('quizHighScores', highScoreData);
    } catch (error) {
        console.error('Error updating high score:', error);
        throw error;
    }
}

async function getAllHighScores() {
    try {
        const dbManager = await initDB();
        return await dbManager.getAll('quizHighScores');
    } catch (error) {
        console.error('Error getting all high scores:', error);
        throw error;
    }
}

async function saveToIndexDB(storeName, data) {
    try {
        const dbManager = await initDB();
        
        // Add required ID fields based on store configuration
        const dataWithId = {
            ...data,
            id: data.id || `${Date.now()}-${Math.random().toString(36).substr(2, 9)}` // Generate unique ID if not provided
        };
        
        // For quizStats store, ensure we have the required fields
        if (storeName === 'quizStats') {
            dataWithId.id = dataWithId.id || `quiz-${Date.now()}`;
            dataWithId.date = dataWithId.date || new Date().toISOString().split('T')[0];
            dataWithId.score = dataWithId.score || 0;
        }
        
        return await dbManager.add(storeName, dataWithId);
    } catch (error) {
        console.error('Error saving to indexDB:', error);
        throw error;
    }
}

async function updateQuizAttempts() {
    try {
        const dbManager = await initDB();
        const today = new Date().toISOString().split('T')[0];
        
        const attempts = await dbManager.getAll('quizAttempts');
        const todayAttempt = attempts.find(attempt => attempt.date === today);
        
        if (todayAttempt) {
            todayAttempt.attempts += 1;
            await dbManager.update('quizAttempts', todayAttempt);
        } else {
            await dbManager.add('quizAttempts', { date: today, attempts: 1 });
        }
    } catch (error) {
        console.error('Error updating quiz attempts:', error);
        throw error;
    }
}

async function updateIncorrectQuestions(newIncorrectQuestions) {
    try {
        const dbManager = await initDB();
        const storeKey = 'ongoingIncorrectQuestions'; // Match the store name from db_configs_one.js
        const incorrectQuestions = await dbManager.getByKey(storeKey, 'current'); // Use consistent ID
        
        if (incorrectQuestions) {
            const questionMap = new Map();
            
            incorrectQuestions.questions.forEach(q => {
                questionMap.set(q.question, q);
            });
            
            newIncorrectQuestions.forEach(q => {
                questionMap.set(q.question, q);
            });
            
            const updatedQuestions = Array.from(questionMap.values());
            
            await dbManager.update(storeKey, {
                id: 'current', // Use consistent ID
                questions: updatedQuestions,
                date: new Date().toISOString(), // Add date for the index
                lastUpdated: new Date().toISOString()
            });
        } else {
            await dbManager.add(storeKey, {
                id: 'current', // Use consistent ID
                questions: newIncorrectQuestions,
                date: new Date().toISOString(), // Add date for the index
                lastUpdated: new Date().toISOString()
            });
        }
    } catch (error) {
        console.error('Error updating incorrect questions:', error);
        throw error;
    }
}

async function saveQuizResult(incorrectQuestions, score, title, quizType = 'section') {
    if (!title) {
        console.error('Quiz title is required');
        return;
    }
    
    const timestamp = new Date().toISOString();
    const date = timestamp.split('T')[0];
    
    // Determine if this is a cumulative quiz
    const isCumulative = quizType === 'cumulative' || title.toLowerCase().includes('cumulative');
    
    try {
        await Promise.all([
            updateIncorrectQuestions(incorrectQuestions),
            updateHighScore(title, score),
            saveToIndexDB('quizStats', {
                id: `quiz-${Date.now()}`, // Add unique ID
                date,
                timestamp,
                title,
                score,
                percentageScore: (score.correct / score.total * 100).toFixed(2),
                quizType: isCumulative ? 'cumulative' : 'section', // Track quiz type
                isCumulative: isCumulative // Boolean flag for easy filtering
            })
        ]);
        
        // Update enhanced table for both quiz types
        const currentPageUrl = window.location.href;
        await updateQuizScoreInEnhancedTable(title, score, currentPageUrl, quizType);
        
        // If it's a cumulative quiz, also update the old page completion system for backward compatibility
        if (isCumulative) {
            await updatePageCompletionFromCumulativeQuiz(title, score);
        }
    } catch (error) {
        console.error('Error in saveQuizResult:', error);
        throw error;
    }
}

export function enhanceQuizSubmit(originalHandleQuizSubmit, quizForm, title) {
    return async function enhancedHandleQuizSubmit(event) {
        event.preventDefault();
        
        let quizTitle = title;
        if (!quizTitle) {
            // Try to get the last quiz title from localStorage
            quizTitle = localStorage.getItem('lastQuizTaken');
            
            if (!quizTitle) {
                console.error('Quiz title is missing');
                showPopover(quizForm, 'Quiz results not saved - missing quiz title', 'error');
                return;
            }
        } else {
            // Save the current quiz title to localStorage
            localStorage.setItem('lastQuizTaken', quizTitle);
        }
        
        const selectedAnswers = quizForm.querySelectorAll('input[type="radio"]:checked');
        const totalQuestions = quizForm.querySelectorAll('div.mb-8').length;
        let correctCount = 0;
        let incorrectQuestions = [];

        // Collect quiz data and answers
        const quizData = Array.from(quizForm.querySelectorAll('div.mb-8')).map(questionDiv => {
            const questionElement = questionDiv.querySelector('h4');
            if (!questionElement) {
                console.warn('Question element not found in div:', questionDiv);
                return null;
            }
            
            return {
                question: questionElement.textContent,
                answers: Array.from(questionDiv.querySelectorAll('input[type="radio"]')).map(input => {
                    const nextElement = input.nextElementSibling;
                    const explanationElement = input.closest('li')?.querySelector('.explanation');
                    
                    if (!nextElement || !explanationElement) {
                        console.warn('Missing elements for input:', {
                            nextElement: !!nextElement,
                            explanationElement: !!explanationElement
                        });
                        return null;
                    }
                    
                    return {
                        text: nextElement.textContent,
                        correct: input.dataset.correct === 'true',
                        explanation: explanationElement.textContent
                    };
                }).filter(answer => answer !== null)
            };
        }).filter(question => question !== null);

        selectedAnswers.forEach((answer) => {
            const correct = answer.dataset.correct === 'true';
            const questionDiv = answer.closest('.mb-8');
            
            if (correct) {
                correctCount++;
            } else {
                const questionElement = questionDiv.querySelector('h4');
                if (questionElement) {
                    incorrectQuestions.push({
                        question: questionElement.textContent,
                        answers: Array.from(questionDiv.querySelectorAll('input[type="radio"]')).map(input => {
                            const nextElement = input.nextElementSibling;
                            const explanationElement = input.closest('li')?.querySelector('.explanation');
                            
                            if (!nextElement || !explanationElement) {
                                console.warn('Missing elements for incorrect answer input:', {
                                    nextElement: !!nextElement,
                                    explanationElement: !!explanationElement
                                });
                                return null;
                            }
                            
                            return {
                                text: nextElement.textContent,
                                correct: input.dataset.correct === 'true',
                                explanation: explanationElement.textContent
                            };
                        }).filter(answer => answer !== null)
                    });
                } else {
                    console.warn('Question element not found for incorrect answer:', questionDiv);
                }
            }
        });

        const score = {
            correct: correctCount,
            total: totalQuestions,
            timestamp: new Date().toISOString()
        };

        // Determine quiz type based on title
        const quizType = quizTitle.toLowerCase().includes('cumulative') ? 'cumulative' : 'section';
        
        try {
            await Promise.all([
                saveQuizResult(incorrectQuestions, score, quizTitle, quizType),
                updateQuizAttempts(),
                saveQuizHistory(quizTitle, quizData, selectedAnswers)
            ]);
            updateQuizStreak(score);
            incrementQuizCounter();
        } catch (error) {
            console.error('Error saving quiz results:', error);
            showPopover(quizForm, 'Error saving quiz results', 'error');
        }

        originalHandleQuizSubmit.call(this, event);
    };
}

export async function getQuizStats() {
    try {
        const dbManager = await initDB();
        const stats = await dbManager.getAll('quizStats');
        const attempts = await dbManager.getAll('quizAttempts');
        
        return {
            attempts: attempts.length,
            totalCorrect: stats.reduce((sum, stat) => sum + stat.score.correct, 0),
            totalQuestions: stats.reduce((sum, stat) => sum + stat.score.total, 0),
            recentScores: stats.slice(-5),
            allStats: stats
        };
    } catch (error) {
        console.error('Error getting quiz stats:', error);
        throw error;
    }
}

export async function getIncorrectQuestions() {
    try {
        const dbManager = await initDB();
        const incorrectQuestions = await dbManager.getByKey('ongoing-incorrect-questions');
        
        return incorrectQuestions ? incorrectQuestions.questions : [];
    } catch (error) {
        console.error('Error getting incorrect questions:', error);
        throw error;
    }
}

export async function getChapterProgress() {
    try {
        // Import the new chapterMap service
        const { getChapterProgress } = await import('./chapterMapService.js');
        return await getChapterProgress();
    } catch (error) {
        console.error('Error getting chapter progress:', error);
        throw error;
    }
}

async function saveQuizHistory(quizTitle, quizData, selectedAnswers) {
    try {
        const dbManager = await initDB();
        
        // Create the history object with a proper ID
        const historyData = {
            id: quizTitle, // Using quizTitle as the unique identifier
            quizTitle: quizTitle,
            attempts: []
        };

        // Try to get existing history first
        const existingHistory = await dbManager.getByKey('quizHistory', quizTitle);
        
        // Create new attempt entry
        const newAttempt = {
            date: new Date().toISOString(),
            answers: Array.from(selectedAnswers).map(answer => {
                const questionDiv = answer.closest('.mb-8');
                if (!questionDiv) {
                    console.warn('Question div not found for answer:', answer);
                    return null;
                }
                
                const questionElement = questionDiv.querySelector('h4');
                const nextElement = answer.nextElementSibling;
                const correctInput = Array.from(questionDiv.querySelectorAll('input[type="radio"]'))
                    .find(input => input.dataset.correct === 'true');
                const explanationElement = answer.closest('li')?.querySelector('.explanation');
                
                if (!questionElement || !nextElement || !correctInput || !explanationElement) {
                    console.warn('Missing DOM elements for answer:', {
                        questionElement: !!questionElement,
                        nextElement: !!nextElement,
                        correctInput: !!correctInput,
                        explanationElement: !!explanationElement
                    });
                    return null;
                }
                
                return {
                    question: questionElement.textContent,
                    selectedAnswerText: nextElement.textContent,
                    selectedAnswerIndex: parseInt(answer.value),
                    wasCorrect: answer.dataset.correct === 'true',
                    correctAnswer: correctInput.nextElementSibling?.textContent || 'Unknown',
                    explanation: explanationElement.textContent
                };
            }).filter(answer => answer !== null),
            score: {
                correct: Array.from(selectedAnswers).filter(a => a.dataset.correct === 'true').length,
                total: quizData.length
            },
            quizData: quizData
        };

        if (existingHistory) {
            // If history exists, append the new attempt
            existingHistory.attempts.push(newAttempt);
            await dbManager.update('quizHistory', existingHistory);
        } else {
            // If no history exists, create new entry with first attempt
            historyData.attempts.push(newAttempt);
            await dbManager.add('quizHistory', historyData);
        }
        
        return newAttempt;
    } catch (error) {
        console.error('Error saving quiz history:', error);
        throw error;
    }
}

async function getQuizHistory(quizTitle) {
    try {
        const dbManager = await initDB();
        const history = await dbManager.getByKey('quizHistory', quizTitle);
        
        return history?.attempts || [];
    } catch (error) {
        console.error('Error getting quiz history:', error);
        throw error;
    }
}

export async function getQuizHistoryStats() {
    try {
        const dbManager = await initDB();
        const allHistory = await dbManager.getAll('quizHistory');
        
        return allHistory.map(history => ({
            quizTitle: history.quizTitle,
            totalAttempts: history.attempts.length,
            averageScore: history.attempts.reduce((sum, attempt) => 
                sum + (attempt.score.correct / attempt.score.total), 0) / history.attempts.length,
            lastAttempt: history.attempts[history.attempts.length - 1]
        }));
    } catch (error) {
        console.error('Error getting quiz history stats:', error);
        throw error;
    }
}

/**
 * Update quiz scores in enhanced table (handles both section and cumulative quizzes)
 * @param {string} quizTitle - The quiz title
 * @param {Object} score - The quiz score
 * @param {string} pageUrl - The current page URL
 * @param {string} quizType - The quiz type ('section' or 'cumulative')
 */
async function updateQuizScoreInEnhancedTable(quizTitle, score, pageUrl, quizType) {
    try {
        const dbManager = await initDB();
        const normalizedUrl = normalizeUrlForCumulativeQuiz(pageUrl);
        const percentage = parseFloat((score.correct / score.total * 100).toFixed(1));
        const passed = percentage >= 60;
        const isCumulative = quizType === 'cumulative';
        
        // Get existing record for this URL
        const existingRecord = await dbManager.getByKey('cumulativeQuizScores', normalizedUrl);
        
        let quizRecord;
        if (existingRecord) {
            // Update existing record
            quizRecord = {
                ...existingRecord,
                // Update with latest quiz data
                lastQuizTitle: quizTitle,
                lastQuizType: quizType,
                lastQuizScore: {
                    correct: score.correct,
                    total: score.total,
                    percentage: percentage,
                    passed: passed
                },
                lastUpdated: new Date().toISOString(),
                // Track quiz history
                quizHistory: existingRecord.quizHistory || [],
                // Update completion status
                completedByCumulativeQuiz: isCumulative && passed,
                cumulativeQuizTitle: isCumulative ? quizTitle : existingRecord.cumulativeQuizTitle
            };
            
            // Add to quiz history
            quizRecord.quizHistory.push({
                quizTitle: quizTitle,
                quizType: quizType,
                score: {
                    correct: score.correct,
                    total: score.total,
                    percentage: percentage,
                    passed: passed
                },
                completedAt: new Date().toISOString()
            });
            
        } else {
            // Create new record
            quizRecord = {
                normalizedUrl: normalizedUrl,
                originalUrl: pageUrl,
                pageTitle: document.title || 'Untitled',
                domain: new URL(pageUrl).hostname,
                // Current quiz data
                lastQuizTitle: quizTitle,
                lastQuizType: quizType,
                lastQuizScore: {
                    correct: score.correct,
                    total: score.total,
                    percentage: percentage,
                    passed: passed
                },
                // Completion tracking
                completedByCumulativeQuiz: isCumulative && passed,
                cumulativeQuizTitle: isCumulative ? quizTitle : null,
                // Quiz history
                quizHistory: [{
                    quizTitle: quizTitle,
                    quizType: quizType,
                    score: {
                        correct: score.correct,
                        total: score.total,
                        percentage: percentage,
                        passed: passed
                    },
                    completedAt: new Date().toISOString()
                }],
                // Metadata
                firstQuizAt: new Date().toISOString(),
                lastUpdated: new Date().toISOString()
            };
        }
        
        await dbManager.update('cumulativeQuizScores', quizRecord);
        console.log(`✅ Quiz score recorded for: ${normalizedUrl} (${quizType}, ${percentage}%)`);
        
    } catch (error) {
        console.error('Error updating quiz score in enhanced table:', error);
    }
}

/**
 * Update page completion status when a cumulative quiz is passed
 * @param {string} quizTitle - The cumulative quiz title
 * @param {Object} score - The quiz score
 */
async function updatePageCompletionFromCumulativeQuiz(quizTitle, score) {
    try {
        const dbManager = await initDB();
        const passed = parseFloat((score.correct / score.total * 100).toFixed(2)) >= 60;
        
        if (!passed) {
            console.log('Cumulative quiz not passed, no page completion update');
            return;
        }
        
        // Extract page title from cumulative quiz title
        // Format: "Cumulative Quiz - Page Title" -> "Page Title"
        const pageTitle = quizTitle.replace(/^Cumulative Quiz - /i, '').trim();
        
        // Get all chapterMap entries to find the matching page
        const { getAllChapterMapEntries } = await import('../../libs/utils/tocExtractor.js');
        const chapterMapEntries = await getAllChapterMapEntries();
        
        // Find the page that matches this cumulative quiz
        const matchingPage = chapterMapEntries.find(page => 
            page.title.toLowerCase().includes(pageTitle.toLowerCase()) ||
            pageTitle.toLowerCase().includes(page.title.toLowerCase())
        );
        
        if (matchingPage) {
            // Create or update page completion record
            const completionData = {
                id: `completion-${matchingPage.url}`,
                pageUrl: matchingPage.url,
                pageTitle: matchingPage.title,
                completedByCumulativeQuiz: true,
                cumulativeQuizTitle: quizTitle,
                completedAt: new Date().toISOString(),
                score: score
            };
            
            await dbManager.update('pageCompletions', completionData);
            console.log(`✅ Page completion recorded for: ${matchingPage.title}`);
        } else {
            console.warn(`⚠️ No matching page found for cumulative quiz: ${quizTitle}`);
        }
        
    } catch (error) {
        console.error('Error updating page completion from cumulative quiz:', error);
    }
}

/**
 * Check if a page is completed by cumulative quiz
 * @param {string} pageUrl - The page URL to check
 * @returns {Promise<boolean>} - True if page is completed by cumulative quiz
 */
export async function isPageCompletedByCumulativeQuiz(pageUrl) {
    try {
        const dbManager = await initDB();
        const normalizedUrl = normalizeUrlForCumulativeQuiz(pageUrl);
        const record = await dbManager.getByKey('cumulativeQuizScores', normalizedUrl);
        return record && record.completedByCumulativeQuiz;
    } catch (error) {
        console.error('Error checking cumulative quiz completion:', error);
        return false;
    }
}

/**
 * Get all pages completed by cumulative quizzes
 * @returns {Promise<Array>} - Array of completed pages
 */
export async function getCompletedPagesByCumulativeQuiz() {
    try {
        const dbManager = await initDB();
        const allRecords = await dbManager.getAll('cumulativeQuizScores');
        return allRecords.filter(record => record.completedByCumulativeQuiz);
    } catch (error) {
        console.error('Error getting completed pages by cumulative quiz:', error);
        return [];
    }
}

/**
 * Get enhanced quiz statistics (both section and cumulative)
 * @returns {Promise<Object>} - Statistics about all quizzes
 */
export async function getEnhancedQuizStats() {
    try {
        const dbManager = await initDB();
        const allRecords = await dbManager.getAll('cumulativeQuizScores');
        
        // Calculate stats from quiz history
        let totalAttempts = 0;
        let cumulativeAttempts = 0;
        let sectionAttempts = 0;
        let cumulativePassed = 0;
        let sectionPassed = 0;
        let totalScore = 0;
        
        allRecords.forEach(record => {
            if (record.quizHistory) {
                record.quizHistory.forEach(quiz => {
                    totalAttempts++;
                    totalScore += quiz.score.percentage;
                    
                    if (quiz.quizType === 'cumulative') {
                        cumulativeAttempts++;
                        if (quiz.score.passed) cumulativePassed++;
                    } else {
                        sectionAttempts++;
                        if (quiz.score.passed) sectionPassed++;
                    }
                });
            }
        });
        
        const stats = {
            totalPages: allRecords.length,
            totalAttempts: totalAttempts,
            cumulativeAttempts: cumulativeAttempts,
            sectionAttempts: sectionAttempts,
            cumulativePassed: cumulativePassed,
            sectionPassed: sectionPassed,
            averageScore: totalAttempts > 0 ? totalScore / totalAttempts : 0,
            completedPagesByCumulative: allRecords.filter(r => r.completedByCumulativeQuiz).length,
            pagesWithSectionQuizzes: allRecords.filter(r => 
                r.quizHistory && r.quizHistory.some(q => q.quizType === 'section')
            ).length
        };
        
        return stats;
    } catch (error) {
        console.error('Error getting enhanced quiz stats:', error);
        return {
            totalPages: 0,
            totalAttempts: 0,
            cumulativeAttempts: 0,
            sectionAttempts: 0,
            cumulativePassed: 0,
            sectionPassed: 0,
            averageScore: 0,
            completedPagesByCumulative: 0,
            pagesWithSectionQuizzes: 0
        };
    }
}

/**
 * Get cumulative quiz record for a specific page
 * @param {string} pageUrl - The page URL
 * @returns {Promise<Object|null>} - The cumulative quiz record or null
 */
export async function getCumulativeQuizRecord(pageUrl) {
    try {
        const dbManager = await initDB();
        const normalizedUrl = normalizeUrlForCumulativeQuiz(pageUrl);
        return await dbManager.getByKey('cumulativeQuizScores', normalizedUrl);
    } catch (error) {
        console.error('Error getting cumulative quiz record:', error);
        return null;
    }
}

export { getHighScore, getAllHighScores, saveQuizHistory, getQuizHistory };