import {createButtonGroup} from './create_quiz_button_grp'
import { enhanceQuizSubmit } from './quiz-storage.js';
import { submitQuizResults } from './quiz-results-api.js';
import { initializePersistentTooltips } from '../tooltip/persistentTooltip.js';
let cachedSaveChatHistory = null;


 async function saveIndexChatHistory() {
    if (!cachedSaveChatHistory) {
      const { saveChatHistory } = await import('../../index.js');
      cachedSaveChatHistory = saveChatHistory;
      await saveChatHistory();
    }

    else {
       await cachedSaveChatHistory();
    }
  }
  

export async function ini_quiz(shadowRoot, quizData, title, chapterTitle, attributes='', contentWithSources=null, context=null) {
    const quizForm = shadowRoot.querySelector('#quiz-form');
    const submitButton = shadowRoot.querySelector('#submit-quiz-btn');

    // Add unique class if it's a final review quiz
    if (attributes && attributes['final-review']) {
        quizForm.classList.add('final-review-quiz');
    }

    // Define generateUniqueId function at the top level
    const generateUniqueId = () => {
        const timePart = new Date().getTime();
        const randomPart = Math.random().toString(36).substring(2, 15);
        return `${timePart}-${randomPart}`;
    };

    const uniqueButtonId = `submit-quiz-btn-${generateUniqueId()}`;
    submitButton.id = uniqueButtonId;
    submitButton.classList.add('socratiq-quiz-submit');


    // Add styles to the shadow DOM
    const style = document.createElement('style');
    style.textContent = `
        #${uniqueButtonId} {
            margin-top: 1rem;
            width: 100%;
            background-color: black;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            transition: background-color 0.2s;
            margin-bottom: 10px;
        }
        
        #${uniqueButtonId}:hover {
            background-color: #333;
        }

        .final-review-quiz {
            background-color: #EFE4D4;  /* Lighter version of #E0C9A6 */
            padding: 20px;
            border-radius: 8px;
        }

        .reference-number {
            background: #f3f4f6;
            border: none;
            border-radius: 12px;
            color: #374151;
            cursor: help;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 600;
            height: 20px;
            min-width: 20px;
            padding: 0 6px;
            margin-left: 6px;
            opacity: 0.8;
            transition: all 0.2s ease;
            vertical-align: middle;
        }
        
        .reference-number:hover {
            background: #6b7280;
            color: white;
            opacity: 1;
            transform: scale(1.05);
        }
    `;
    shadowRoot.appendChild(style);

    function loadQuiz(quizData) {
        quizForm.innerHTML = ''; 
        
        // Add the socratiq-quiz class
        quizForm.classList.add('socratiq-quiz');
        
        // Store quiz data in the form itself
        quizForm.setAttribute('data-quiz', JSON.stringify(quizData));
        quizForm.setAttribute('chapter-title', chapterTitle);
        quizForm.setAttribute('data-quiz-title', title);
        
        // Store context metadata if available
        if (context) {
            quizForm.setAttribute('data-context-url', context.url);
            quizForm.setAttribute('data-context-title', context.title);
            quizForm.setAttribute('data-context-content-length', context.content.totalLength);
            quizForm.setAttribute('data-context-source-count', context.content.sourceCount);
            quizForm.setAttribute('data-context-scroll-position', context.position.scrollPercentage.y);
            quizForm.setAttribute('data-context-timestamp', context.timestamp);
        }

        // Note: Tooltip initialization moved to after reference numbers are created


        // Check if quizData is valid
        if (!quizData || !Array.isArray(quizData)) {
            console.error('❌ Invalid quiz data received:', quizData);
            console.error('❌ Quiz data type:', typeof quizData);
            console.error('❌ Quiz data value:', quizData);
            
            let errorMessage = 'Error: Invalid quiz data received';
            if (quizData === null || quizData === undefined) {
                errorMessage = 'Error: No quiz data received from AI';
            } else if (typeof quizData === 'string') {
                errorMessage = `Error: AI returned text instead of quiz data: "${quizData.substring(0, 100)}..."`;
            } else if (typeof quizData === 'object' && !Array.isArray(quizData)) {
                errorMessage = `Error: AI returned object instead of quiz array: ${JSON.stringify(quizData).substring(0, 100)}...`;
            }
            
            quizForm.innerHTML = `<div class="error">${errorMessage}</div>`;
            return;
        }

        quizData.forEach((q, qi) => {
            // Validate question object
            if (!q || typeof q !== 'object') {
                console.error('Invalid question object:', q);
                return;
            }

            if (!q.question) {
                console.error('Question missing text:', q);
                return;
            }
            

            const questionDiv = document.createElement('div');
            questionDiv.classList.add('mb-8');

            // Create question container with reference
            const questionContainer = document.createElement('div');
            questionContainer.classList.add('flex', 'items-start', 'gap-2');

            const questionTitle = document.createElement('h4');
            questionTitle.classList.add('text-lg', 'font-semibold', 'mb-2', 'flex-1');
            questionTitle.textContent = q.question;

            // Create reference number if sourceReference exists
            if (q.sourceReference) {
                const referenceSpan = document.createElement('span');
                referenceSpan.classList.add('reference-number');
                referenceSpan.textContent = `${qi + 1}`;
                
                // Create source data for tooltip
                const sourceUrl = contentWithSources && contentWithSources[qi] ? contentWithSources[qi].pageUrl : '';
                const sourceLabel = contentWithSources && contentWithSources[qi] ? contentWithSources[qi].label : 
                    (sourceUrl ? new URL(sourceUrl).origin : window.location.origin);
                
                const sourceData = {
                    sourceReference: q.sourceReference,
                    sourceLabel: sourceLabel,
                    sourceUrl: sourceUrl,
                    sourcePosition: contentWithSources && contentWithSources[qi] ? contentWithSources[qi].position : 0
                };
                
                // Store all tooltip data as DOM attributes for persistence
                referenceSpan.setAttribute('data-source-reference', sourceData.sourceReference);
                referenceSpan.setAttribute('data-source-label', sourceData.sourceLabel);
                referenceSpan.setAttribute('data-source-url', sourceData.sourceUrl);
                referenceSpan.setAttribute('data-source-position', sourceData.sourcePosition.toString());
                referenceSpan.setAttribute('data-tooltip-enabled', 'true');
                
                
                questionTitle.appendChild(referenceSpan);
            } else {
            }

            questionContainer.appendChild(questionTitle);
            questionDiv.appendChild(questionContainer);

            const answersUl = document.createElement('ul');
            answersUl.classList.add('list-none', 'space-y-2');

            q.answers.forEach((a, ai) => {
                const answerLi = document.createElement('li');
                const answerDiv = document.createElement('div');
                answerDiv.classList.add('answer-option', 'flex', 'items-center');

                const input = document.createElement('input');
                input.type = 'radio';
                const ans_id = generateUniqueId();
                input.id = ans_id;
                input.name = `question${qi}`;
                input.value = ai; // index as value
                input.classList.add('mr-2');
                input.dataset.correct = a.correct;

                const label = document.createElement('label');
                label.htmlFor = ans_id;
                label.textContent = a.text;
                label.classList.add('flex-1');

                const explains_wrapper = document.createElement('div');
                explains_wrapper.classList.add('explains_wrapper', 'hidden');

                if(a.correct){
                    explains_wrapper.classList.add("relative","mt-2", "rounded-lg", "border-2", "border-green-500", "p-3");
                    const explains_label = document.createElement('div');
                    explains_label.classList.add("absolute", "-top-3", "left-2", "rounded-lg", "border-2", "border-green-500", "bg-white", "px-2", "text-xs", "text-green-500");
                    explains_label.textContent = "Correct";
                    explains_wrapper.appendChild(explains_label);
                }
                const explain = document.createElement('p');
                explain.textContent = a.explanation;
                explain.classList.add('explanation', 'text-sm', 'text-purple-500');
                explains_wrapper.appendChild(explain);
                answerDiv.appendChild(input);
                answerDiv.appendChild(label);
                answerLi.append(answerDiv);
                answerLi.appendChild(explains_wrapper);
                answersUl.appendChild(answerLi);
            });

            questionDiv.appendChild(answersUl);
            quizForm.appendChild(questionDiv);
        });

        // Initialize persistent tooltip system after all reference numbers are created
        initializePersistentTooltips();
   
    }

    function handleQuizSubmit(event) {
        event.preventDefault();

        const existingButtonGroup = quizForm.querySelector('.quiz-btn-group');
        if (existingButtonGroup) {
            existingButtonGroup.remove();
        }
    

        const selectedAnswers = quizForm.querySelectorAll('input[type="radio"]:checked');
        const explains = quizForm.querySelectorAll('.explains_wrapper');

        let correctCount = 0;

        selectedAnswers.forEach((answer) => {
            const correct = answer.dataset.correct === 'true';
            const parentLi = answer.closest('li');

            if (correct) {
                correctCount++;
                parentLi.classList.add('correct-answer');
            } else {
                parentLi.classList.add('wrong-answer');
            }
        });

        explains.forEach((explain) => {
            explain.classList.remove('hidden');
        });

        const resultText = `You got ${correctCount} out of ${quizData.length} questions correct.`;
        shadowRoot.querySelector('#result').textContent = resultText;

        // Store results in the button for reinitialization
        submitButton.setAttribute('data-quiz-results', quizForm.textContent);
        
        // Submit results to API
        submitQuizResults(quizForm, selectedAnswers)
            .catch(error => console.error('Failed to submit quiz results:', error));

        // Update button state
        submitButton.textContent = "Explain in more detail";
        submitButton.removeEventListener('click', handleQuizSubmit);
        submitButton.addEventListener('click', handleExplanationRequest);

        const buttonGroup = createButtonGroup(shadowRoot);
        quizForm.appendChild(buttonGroup);

        // Dispatch custom event to trigger chat history save
        window.dispatchEvent(new CustomEvent('quizSubmitted'));
    }

    function handleExplanationRequest() {


        // function ai_query(text_combo, type_of_custom_event, type_of_action) {
            const event = new CustomEvent("aiActionCompleted", {
              detail: {
                text: submitButton.dataset.quizResults, // the original query object used for the request
                type: "query", // the type of AI action
              },
            });
            window.dispatchEvent(event);
        //   }
    }

    // submitButton.addEventListener('click', handleQuizSubmit);
    const enhancedHandler = enhanceQuizSubmit(handleQuizSubmit.bind(null), quizForm, title);
    submitButton.addEventListener('click', enhancedHandler);


    loadQuiz(quizData);
    await saveIndexChatHistory();
}

async function handleReinitializedQuizSubmit(event, form, quizTitle) {
    
    event.preventDefault();
    event.stopPropagation();
    
    try {
        const quizData = JSON.parse(form.getAttribute('data-quiz'));
        
        // Create enhanced submit handler with stored quiz data
        const enhancedHandler = enhanceQuizSubmit(
            async (event) => {
                
                const selectedAnswers = form.querySelectorAll('input[type="radio"]:checked');
                const explains = form.querySelectorAll('.explains_wrapper');
                
                
                let correctCount = 0;
                selectedAnswers.forEach((answer, index) => {
                    const correct = answer.dataset.correct === 'true';
                    const parentLi = answer.closest('li');

                    if (correct) {
                        correctCount++;
                        parentLi.classList.add('correct-answer');
                    } else {
                        parentLi.classList.add('wrong-answer');
                    }
                });

                explains.forEach((explain) => {
                    explain.classList.remove('hidden');
                });

                // Update result text
                const resultText = `You got ${correctCount} out of ${quizData.length} questions correct.`;
                const quizContainer = form.closest('#quiz');
                const resultDiv = quizContainer.querySelector('#result');
                resultDiv.textContent = resultText;

                // Find and update submit button
                const submitButton = quizContainer.querySelector('.socratiq-quiz-submit');
                if (!submitButton) {
                    console.error('Submit button not found');
                    return;
                }

                // Store results in the button
                submitButton.setAttribute('data-quiz-results', form.textContent);

                // Submit results to API
                submitQuizResults(form, selectedAnswers)
                    .catch(error => console.error('Failed to submit quiz results:', error));

                // Update button with SVG preserved
                const svgContent = submitButton.querySelector('svg');
                submitButton.textContent = "Explain in more detail";
                if (svgContent) {
                    submitButton.insertBefore(svgContent, submitButton.firstChild);
                }

                // Remove existing button group if present
                const existingButtonGroup = form.querySelector('.quiz-btn-group');
                if (existingButtonGroup) {
                    existingButtonGroup.remove();
                }

                // Add new button group
                const buttonGroup = createButtonGroup(form.getRootNode());
                form.appendChild(buttonGroup);

                // Clone and replace button with new event listener
                const newButton = submitButton.cloneNode(true);
                submitButton.parentNode.replaceChild(newButton, submitButton);
                newButton.addEventListener('click', () => {
                    handleReinitializedExplanation(form);
                }, { once: true });

                // Dispatch quiz submitted event
                window.dispatchEvent(new CustomEvent('quizSubmitted'));
                
            },
            form,
            quizTitle
        );
        
        return enhancedHandler(event);
    } catch (error) {
        console.error('Error in handleReinitializedQuizSubmit:', error, {
            stack: error.stack,
            formData: form ? {
                hasQuizAttr: form.hasAttribute('data-quiz'),
                quizAttrValue: form.getAttribute('data-quiz'),
                title: quizTitle
            } : 'No form'
        });
        throw error;
    }
}

async function handleReinitializedExplanation(form) {
    const quizContent = form.textContent;
    const event = new CustomEvent("aiActionCompleted", {
        detail: {
            text: quizContent,
            type: "query",
        },
    });
    window.dispatchEvent(event);
}


export function reinitializeQuizButtons(shadowRoot) {
    const aiMessages = shadowRoot.querySelectorAll('.ai-message-chat');
    
    aiMessages.forEach((message, index) => {
        const quizForm = message.querySelector('.socratiq-quiz');
        const submitButton = message.querySelector('.socratiq-quiz-submit');
        
        console.log(`Processing message ${index + 1}:`, {
            hasQuizForm: !!quizForm,
            hasSubmitButton: !!submitButton
        });

        if (!quizForm || !submitButton) return;

        try {
            const quizTitle = quizForm.getAttribute('data-quiz-title');
            
            // First, remove any existing event listeners
            const oldButton = submitButton.cloneNode(true);
            submitButton.parentNode.replaceChild(oldButton, submitButton);
            
            // Check if quiz has been submitted by looking for data-quiz-results
            const hasSubmitted = oldButton.hasAttribute('data-quiz-results');
            
            if (hasSubmitted) {
                oldButton.textContent = "Explain in more detail";
                const explanationHandler = () => {
                    const event = new CustomEvent("aiActionCompleted", {
                        detail: {
                            text: oldButton.getAttribute('data-quiz-results'),
                            type: "query",
                        },
                    });
                    window.dispatchEvent(event);
                };
                oldButton.addEventListener('click', explanationHandler, { once: true });
            } else {
                const submitHandler = (e) => {
                    try {
                        return handleReinitializedQuizSubmit(e, quizForm, quizTitle);
                    } catch (error) {
                        console.error('Error in submit handler:', error);
                        throw error;
                    }
                };
                oldButton.addEventListener('click', submitHandler, { once: true });
            }
        } catch (error) {
            console.error('Error reinitializing quiz button:', error, {
                quizForm: quizForm,
                submitButton: submitButton,
                stack: error.stack
            });
        }
    });
    
    // Initialize persistent tooltip system for all existing quizzes
    initializePersistentTooltips();
}