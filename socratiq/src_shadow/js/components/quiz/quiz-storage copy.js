// quiz-storage.js
import { showPopover } from "../../libs/utils/utils.js";
import { getDBInstance } from "../../libs/utils/indexDb.js";

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

async function saveQuizResult(incorrectQuestions, score, title) {
    if (!title) {
        console.error('Quiz title is required');
        return;
    }
    
    const timestamp = new Date().toISOString();
    const date = timestamp.split('T')[0];
    
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
                percentageScore: (score.correct / score.total * 100).toFixed(2)
            })
        ]);
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
        const quizData = Array.from(quizForm.querySelectorAll('div.mb-8')).map(questionDiv => ({
            question: questionDiv.querySelector('h4').textContent,
            answers: Array.from(questionDiv.querySelectorAll('input[type="radio"]')).map(input => ({
                text: input.nextElementSibling.textContent,
                correct: input.dataset.correct === 'true',
                explanation: input.closest('li').querySelector('.explanation').textContent
            }))
        }));

        selectedAnswers.forEach((answer) => {
            const correct = answer.dataset.correct === 'true';
            const questionDiv = answer.closest('.mb-8');
            
            if (correct) {
                correctCount++;
            } else {
                incorrectQuestions.push({
                    question: questionDiv.querySelector('h4').textContent,
                    answers: Array.from(questionDiv.querySelectorAll('input[type="radio"]')).map(input => ({
                        text: input.nextElementSibling.textContent,
                        correct: input.dataset.correct === 'true',
                        explanation: input.closest('li').querySelector('.explanation').textContent
                    }))
                });
            }
        });

        const score = {
            correct: correctCount,
            total: totalQuestions,
            timestamp: new Date().toISOString()
        };

        try {
            await Promise.all([
                saveQuizResult(incorrectQuestions, score, quizTitle),
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
        const dbManager = await initDB();
        const scores = await dbManager.getAll('quizHighScores');
        const chapterProgress = new Map();
        
        // Process each quiz score
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
        console.error('Error getting chapter progress:', error);
        throw error;
    }
}

async function saveQuizHistory(quizTitle, quizData, selectedAnswers) {
    try {
        const dbManager = await initDB();
        const history = await dbManager.getByKey('quizHistory', quizTitle);
        const existingHistory = history?.attempts || [];
        
        // Create new attempt entry with more comprehensive data
        const newAttempt = {
            date: new Date().toISOString(),
            answers: Array.from(selectedAnswers).map(answer => {
                const questionDiv = answer.closest('.mb-8');
                return {
                    question: questionDiv.querySelector('h4').textContent,
                    selectedAnswerText: answer.nextElementSibling.textContent,
                    selectedAnswerIndex: parseInt(answer.value),
                    wasCorrect: answer.dataset.correct === 'true',
                    correctAnswer: Array.from(questionDiv.querySelectorAll('input[type="radio"]'))
                        .find(input => input.dataset.correct === 'true')
                        .nextElementSibling.textContent,
                    explanation: questionDiv.querySelector('.explanation').textContent
                };
            }),
            score: {
                correct: Array.from(selectedAnswers).filter(a => a.dataset.correct === 'true').length,
                total: quizData.length
            },
            quizData: quizData // Store original quiz structure
        };
        
        // Combine existing history with new attempt
        const updatedHistory = {
            quizTitle,
            attempts: [...existingHistory, newAttempt]
        };
        
        // Save updated history
        await dbManager.update('quizHistory', updatedHistory);
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

export { getHighScore, getAllHighScores, saveQuizHistory, getQuizHistory };