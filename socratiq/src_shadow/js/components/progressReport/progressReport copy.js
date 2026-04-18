import { getDBInstance } from '../../libs/utils/indexDb.js';

async function initDB() {
    const dbManager = await getDBInstance();
    if (!dbManager || !dbManager.db) {
        throw new Error('Database not properly initialized');
    }
    return dbManager;
}

async function getLastReportDate() {
    try {
        const dbManager = await initDB();
        const reports = await dbManager.getAll('reportHistory');
        
        if (reports.length === 0) {
            return null;
        }
        
        const lastReport = reports.reduce((latest, current) => 
            latest.date > current.date ? latest : current
        );
        return new Date(lastReport.date);
    } catch (error) {
        console.error('Error getting last report date:', error);
        throw error;
    }
}

async function getNewQuizAttempts(lastReportDate) {
    try {
        const dbManager = await initDB();
        const allQuizzes = await dbManager.getAll('quizHistory');
        const newAttempts = {};

        allQuizzes.forEach(quiz => {
            const filteredAttempts = quiz.attempts.filter(attempt => {
                const attemptDate = new Date(attempt.date);
                return !lastReportDate || attemptDate > lastReportDate;
            });

            if (filteredAttempts.length > 0) {
                newAttempts[quiz.quizTitle] = filteredAttempts;
            }
        });

        return newAttempts;
    } catch (error) {
        console.error('Error getting new quiz attempts:', error);
        throw error;
    }
}

async function saveReportHistory(report) {
    try {
        const dbManager = await initDB();
        await dbManager.add('reportHistory', {
            date: new Date().toISOString(),
            report: report
        });
    } catch (error) {
        console.error('Error saving report history:', error);
        throw error;
    }
}

function formatQuizResults(attempts) {
    let report = '';
    
    Object.entries(attempts).forEach(([quizTitle, quizAttempts]) => {
        report += `\n## ${quizTitle}\n\n`;
        
        quizAttempts.forEach((attempt, index) => {
            const date = new Date(attempt.date).toLocaleString();
            const score = `${attempt.score.correct}/${attempt.score.total}`;
            const percentage = ((attempt.score.correct / attempt.score.total) * 100).toFixed(1);
            
            report += `Attempt ${index + 1} (${date})\n`;
            report += `Score: ${score} (${percentage}%)\n\n`;
            
            attempt.answers.forEach(answer => {
                report += `Q: ${answer.question}\n`;
                report += `A: ${answer.selectedAnswerText}\n`;
                report += answer.wasCorrect ? 
                    `Note: Student answered this correctly with "${answer.selectedAnswerText}"\n` :
                    `Note: Student answered this incorrectly with "${answer.selectedAnswerText}"\n`;
                report += '\n';
            });
            report += '---\n\n';
        });
    });

    return report;
}

async function getAllQuizHistory() {
    try {
        const dbManager = await initDB();
        const allQuizzes = await dbManager.getAll('quizHistory');
        const quizHistory = {};
        
        allQuizzes.forEach(quiz => {
            quizHistory[quiz.quizTitle] = quiz.attempts;
        });
        
        return quizHistory;
    } catch (error) {
        console.error('Error getting quiz history:', error);
        throw error;
    }
}

async function getLastReport() {
    try {
        const dbManager = await initDB();
        const reports = await dbManager.getAll('reportHistory');
        
        if (reports.length === 0) {
            return null;
        }
        
        const lastReport = reports.reduce((latest, current) => 
            latest.date > current.date ? latest : current
        );
        return lastReport;
    } catch (error) {
        console.error('Error getting last report:', error);
        throw error;
    }
}

function formatProgressSummary(chapterData, allQuizHistory) {
    let summary = '\n## Progress Summary\n\n';
    
    // Sort chapters numerically
    const chapters = chapterData.sort((a, b) => 
        parseInt(a.chapter) - parseInt(b.chapter)
    );
    
    // Get all quiz titles from history
    const completedSections = new Set(Object.keys(allQuizHistory));
    
    summary += '### Completed Chapters/Sections:\n';
    let hasCompletedSections = false;
    
    chapters.forEach(chapter => {
        const chapterSections = [];
        
        // Check for completed sections in this chapter
        completedSections.forEach(quizTitle => {
            if (quizTitle.startsWith(`${chapter.chapter}.`)) {
                const sectionMatch = quizTitle.match(/^\d+\.(\d+)/);
                if (sectionMatch) {
                    const sectionNum = sectionMatch[1];
                    const sectionTitle = quizTitle.split(' ').slice(1).join(' ');
                    chapterSections.push(`  - Section ${sectionNum}: ${sectionTitle}`);
                }
            }
        });
        
        // Only show chapters that have completed sections
        if (chapterSections.length > 0) {
            hasCompletedSections = true;
            summary += `\nChapter ${chapter.chapter}: ${chapter.title}\n`;
            chapterSections.sort().forEach(section => {
                summary += `${section}\n`;
            });
        }
    });
    
    if (!hasCompletedSections) {
        summary += '\nNo sections completed yet.\n';
    }

    summary += '\n### Remaining Chapters:\n';
    chapters.forEach(chapter => {
        if (!completedSections.has(`${chapter.chapter}.1`)) {
            summary += `○ Chapter ${chapter.chapter}: ${chapter.title}\n`;
        }
    });

    return summary;
}

// Add new function to process quiz data for charts
async function processQuizDataForCharts(allQuizHistory) {
    console.log("Starting processQuizDataForCharts with:", allQuizHistory);
    
    // Process dates and scores
    const dateAttempts = new Map();
    
    // Process the quiz history object to calculate averages
    Object.entries(allQuizHistory).forEach(([quizTitle, quizData]) => {
        if (Array.isArray(quizData)) {
            quizData.forEach(attempt => {
                const date = new Date(attempt.date).toLocaleDateString('en-US', { 
                    month: 'short',
                    day: 'numeric'
                }).toLowerCase();
                
                if (!dateAttempts.has(date)) {
                    dateAttempts.set(date, { correct: 0, total: 0 });
                }
                
                const stats = dateAttempts.get(date);
                if (attempt.score) {
                    stats.correct += attempt.score.correct || 0;
                    stats.total += attempt.score.total || 0;
                }
            });
        }
    });

    // Format XY Chart data with averages
    const dates = Array.from(dateAttempts.keys());
    const averageScores = dates.map(date => {
        const stats = dateAttempts.get(date);
        return stats.total > 0 ? Math.round((stats.correct / stats.total) * 100) : 0;
    });
    
    // Create mermaid chart block
    let mermaidXYChart = '```mermaid\n';
    mermaidXYChart += 'xychart-beta\n';
    mermaidXYChart += '    title "Quiz Performance Over Time"\n';
    mermaidXYChart += '    x-axis [' + dates.join(', ') + ']\n';
    mermaidXYChart += '    y-axis "Score %" 0 --> 100\n';
    mermaidXYChart += '    bar [' + averageScores.join(', ') + ']\n';
    mermaidXYChart += '```\n';
    mermaidXYChart += '$$This chart shows your average quiz scores over time. Each bar represents the percentage of correct answers for that day.$$\n';

    // Process questions by category
    const categoryStats = {
        theoretical: { correct: 0, total: 0 },
        practical: { correct: 0, total: 0 },
        conceptual: { correct: 0, total: 0 },
        implementation: { correct: 0, total: 0 }
    };

    // Process the quiz history for categories
    Object.entries(allQuizHistory).forEach(([quizTitle, quizData]) => {
        if (Array.isArray(quizData)) {
            quizData.forEach(attempt => {
                if (Array.isArray(attempt.answers)) {
                    attempt.answers.forEach(answer => {
                        const category = categorizeQuestion(answer.question);
                        categoryStats[category].total++;
                        if (answer.wasCorrect) {
                            categoryStats[category].correct++;
                        }
                    });
                }
            });
        }
    });

    console.log("Category stats:", categoryStats);

    // Create Quadrant Chart
    let mermaidQuadrantChart = '```mermaid\n';
    mermaidQuadrantChart += 'quadrantChart\n';
    mermaidQuadrantChart += '    title Learning Progress Matrix\n';
    mermaidQuadrantChart += '    x-axis Low Understanding --> High Understanding\n';
    mermaidQuadrantChart += '    y-axis Low Performance --> High Performance\n';
    mermaidQuadrantChart += '    quadrant-1 "Strong Understanding"\n';
    mermaidQuadrantChart += '    quadrant-2 "Good Performance"\n';
    mermaidQuadrantChart += '    quadrant-3 "Needs Review"\n';
    mermaidQuadrantChart += '    quadrant-4 "Developing Skills"\n';

    // Constants for visualization
    const DOT_RADIUS = 4;  // Single consistent radius for all dots
    const SPREAD_RADIUS = 0.15;  // For dot distribution
    const SPIRAL_FACTOR = 0.005; // For spiral pattern

    // Add points for each category with individual dots for correct/incorrect answers
    Object.entries(categoryStats).forEach(([category, stats]) => {
        // Base position for this category
        let baseX, baseY;
        switch(category) {
            case 'theoretical':
                baseX = 0.3; baseY = 0.7;
                break;
            case 'practical':
                baseX = 0.7; baseY = 0.7;
                break;
            case 'conceptual':
                baseX = 0.3; baseY = 0.3;
                break;
            case 'implementation':
                baseX = 0.7; baseY = 0.3;
                break;
            default:
                baseX = 0.5; baseY = 0.5;
        }

        let correctCount = 0;
        let totalCount = 0;

        // Process quiz history for individual answer dots
        Object.entries(allQuizHistory).forEach(([quizTitle, quizData]) => {
            if (Array.isArray(quizData)) {
                quizData.forEach(attempt => {
                    if (Array.isArray(attempt.answers)) {
                        attempt.answers.forEach(answer => {
                            const answerCategory = categorizeQuestion(answer.question);
                            if (answerCategory === category) {
                                // Calculate spiral position for better distribution
                                const angle = (2 * Math.PI * totalCount) / (stats.total || 1);
                                const spiralRadius = SPREAD_RADIUS * (1 + SPIRAL_FACTOR * totalCount);
                                const offsetX = spiralRadius * Math.cos(angle);
                                const offsetY = spiralRadius * Math.sin(angle);

                                // Ensure dots stay within quadrant bounds
                                const dotX = Math.max(0.1, Math.min(0.9, baseX + offsetX));
                                const dotY = Math.max(0.1, Math.min(0.9, baseY + offsetY));
                                
                                // Use consistent radius for all dots
                                mermaidQuadrantChart += `    ${category}_${totalCount}: [${dotX}, ${dotY}] color: ${answer.wasCorrect ? '#00ff00' : '#ff0000'}, radius: ${DOT_RADIUS}\n`;

                                if (answer.wasCorrect) correctCount++;
                                totalCount++;
                            }
                        });
                    }
                });
            }
        });

        // Category marker also uses same radius
        const score = totalCount > 0 ? correctCount / totalCount : 0;
        const categoryColor = score >= 0.7 ? '#00ff00' : score >= 0.4 ? '#ffff00' : '#ff0000';
        mermaidQuadrantChart += `    ${category}: [${baseX}, ${baseY}] color: ${categoryColor}, radius: ${DOT_RADIUS}\n`;
    });

    mermaidQuadrantChart += '```\n';
    mermaidQuadrantChart += '$$This matrix shows your learning progress across four problem types: theoretical (concept understanding), practical (hands-on application), conceptual (analysis & design), and implementation (coding & deployment). Green dots represent correct answers, red dots represent incorrect answers.$$\n';

    return {
        xyChart: mermaidXYChart,
        quadrantChart: mermaidQuadrantChart
    };
}

// Helper function to categorize questions
function categorizeQuestion(question) {
    const keywords = {
        theoretical: ['what', 'why', 'describe', 'explain', 'define', 'concept'],
        practical: ['how', 'implement', 'build', 'create', 'develop'],
        conceptual: ['compare', 'contrast', 'analyze', 'evaluate', 'design'],
        implementation: ['code', 'debug', 'optimize', 'configure', 'deploy']
    };

    const questionLower = question.toLowerCase();
    for (const [category, words] of Object.entries(keywords)) {
        if (words.some(word => questionLower.includes(word))) {
            return category;
        }
    }
    
    return 'theoretical'; // default category
}

export async function generateProgressReport() {
    try {
        const lastReportDate = await getLastReportDate();
        const allQuizHistory = await getAllQuizHistory();

        console.log('Starting generateProgressReport with quiz history:', allQuizHistory);

        let report = '# Learning Progress Report\n';
        report += `Generated on: ${new Date().toLocaleString()}\n\n`;

        // Check if there's no quiz history
        if (!allQuizHistory || Object.keys(allQuizHistory).length === 0) {
            report += `## Getting Started with Quizzes\n\n`;
            report += `It looks like you haven't taken any quizzes yet! Here's how to get started:\n\n`;
            report += `1. **Find Section Quizzes**: At the end of each section in every chapter, you'll find interactive quizzes.\n\n`;
            report += `2. **Test Your Knowledge**: Take these quizzes to reinforce your learning and identify areas that need more attention.\n\n`;
            report += `3. **Track Your Progress**: Each quiz attempt is recorded, allowing you to:\n`;
            report += `   - See your improvement over time\n`;
            report += `   - Review questions you found challenging\n`;
            report += `   - Build confidence in your understanding\n\n`;
            report += `4. **Learning Strategy**: Consider taking quizzes:\n`;
            report += `   - Right after studying a section while it's fresh in your mind\n`;
            report += `   - Again after a few days to test your retention\n`;
            report += `   - Before moving on to new chapters\n\n`;
            report += `Ready to start? Head to any chapter section and look for the quiz button at the bottom!\n`;

            return {
                success: true,
                report,
                charts: null,
                newAttempts: 0,
                lastReportDate: null,
                isFromCache: false
            };
        }

        const newAttempts = await getNewQuizAttempts(lastReportDate);
        const chapterData = JSON.parse(localStorage.getItem('chapter_progress_data')) || [];
        const lastReport = await getLastReport();

        // If no new attempts and we have a last report, return it
        if (Object.keys(newAttempts).length === 0 && lastReport) {
            console.log('No new attempts, using cached report');
            return {
                success: true,
                report: lastReport.report,
                charts: lastReport.charts || await processQuizDataForCharts(allQuizHistory), // Try to generate charts even for cached report
                newAttempts: 0,
                lastReportDate,
                isFromCache: true
            };
        }

        // Add complete progress summary using all quiz history
        report += formatProgressSummary(chapterData, allQuizHistory);

        // Add only new quiz results if there are any
        if (Object.keys(newAttempts).length > 0) {
            report += '\n## Recent Quiz Results\n';
            report += formatQuizResults(newAttempts);
            await saveReportHistory(report);
        } else {
            report += '\nNo new quiz attempts since last report.\n';
        }

        // Process charts with proper wrapping and captions
        console.log('Processing charts with quiz history:', allQuizHistory);
        const charts = await processQuizDataForCharts(allQuizHistory);
        console.log('Generated charts:', charts);

        // Add chart descriptions to the report text
        // report += '\n## Performance Analysis\n\n';
        // report += 'Your learning progress is visualized through two charts:\n\n';
        // report += '1. **Quiz Performance Trends**: Shows your quiz scores over time, ' +
        //          'helping you track your progress and identify patterns in your learning journey.\n\n';
        // report += '2. **Learning Competency Matrix**: Maps your strengths across different ' +
        //          'learning domains, helping you identify areas for focused improvement.\n\n';

        const result = {
            success: true,
            report,
            charts,
            newAttempts: Object.keys(newAttempts).length,
            lastReportDate,
            isFromCache: false
        };

        console.log('Final report result:', result);
        return result;

    } catch (error) {
        console.error('Error generating progress report:', error);
        return {
            success: false,
            message: "Error generating report",
            error: error.message
        };
    }
}
