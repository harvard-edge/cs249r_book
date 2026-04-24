import { getIncorrectQuestions } from './quiz-storage.js';

export async function initiateStudyBtn(shadowRoot) {
    const studyBtn = shadowRoot.querySelector('#quiz-study-btn');
    
    if (!studyBtn) {
        console.error('Study button not found in shadow root');
        return;
    }

    studyBtn.addEventListener('click', async () => {
        try {
            // Get incorrect questions from IndexDB
            const incorrectQuestions = await getIncorrectQuestions();
            
            if (!incorrectQuestions || incorrectQuestions.length === 0) {
                console.log('No incorrect questions found to study');
                // Optionally notify the user
                return;
            }

            // Format questions to match the expected structure
            // Note: Our stored format already matches the target format,
            // but we'll ensure it here for safety
            const formattedQuestions = incorrectQuestions.map(q => ({
                question: q.question,
                answers: q.answers.map(a => ({
                    text: a.text,
                    correct: a.correct,
                    explanation: a.explanation
                }))
            }));

            // Trigger the AI action with formatted questions
            triggerAIAction(formattedQuestions);
            
        } catch (error) {
            console.error('Error loading incorrect questions:', error);
        }
    });
}

function triggerAIAction(quizData) {
    const aiActionEvent = new CustomEvent('aiActionCompleted', {
        detail: {
            type: 'quiz-study',
            text: quizData,
            attributes: {
                'final-review': true,
            }
        }
    });
    window.dispatchEvent(aiActionEvent);
}