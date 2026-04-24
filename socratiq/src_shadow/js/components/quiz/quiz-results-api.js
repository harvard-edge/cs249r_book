import { SERVERLESSRRESULTS } from '../../../configs/env_configs';

// Difficulty level mapping
const DIFFICULTY_MAP = {
    '0': 'beginner',
    '1': 'intermediate',
    '2': 'advanced',
    '3': 'blooms-taxonomy'
};

export async function submitQuizResults(quizForm, selectedAnswers) {
    // API submission disabled to prevent 404 errors
    console.log('Quiz results API submission disabled - results saved to IndexedDB only');
    return { success: true, message: 'API submission disabled' };
    
    // Original code commented out below:
    /*
    const chapterId = quizForm.getAttribute('chapter-title');
    const sectionId = quizForm.getAttribute('data-quiz-title');
    const quizData = JSON.parse(quizForm.getAttribute('data-quiz'));
    
    // Get difficulty from closest parent with data-difficulty attribute
    const difficultyElement = quizForm.closest('[data-difficulty]');
    const difficultyNum = difficultyElement ? difficultyElement.getAttribute('data-difficulty') : '1';
    const difficulty = DIFFICULTY_MAP[difficultyNum] || 'intermediate';
    
    // Create updates array for API
    const updates = [];
    
    selectedAnswers.forEach((answer, index) => {
        const question = quizData[index];
        const isCorrect = answer.dataset.correct === 'true';
        
        updates.push({
            chapterId,
            sectionId,
            questionId: question.id,
            isCorrect,
            difficulty
        });
    });

    try {
        console.log('Submitting quiz results:', {
            updates,
            difficulty,
            difficultyNum
        });

        const response = await fetch(SERVERLESSRRESULTS, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ updates })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Quiz results submitted successfully:', data);
        return data;
    } catch (error) {
        console.error('Error submitting quiz results:', error);
        throw error;
    }
    */
} 