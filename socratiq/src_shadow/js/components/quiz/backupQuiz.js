import nlp from "compromise";

function extractEntitiesAndNouns(text) {
  let doc = nlp(text);
  let sentences = doc.sentences().out("array");
  let data = sentences.map((sentence) => {
    let sentenceDoc = nlp(sentence);
    let entities = sentenceDoc.topics().out("array");
    let nouns = sentenceDoc
      .nouns()
      .out("array")
      .filter((noun) => !entities.includes(noun));
  
    return { sentence, entities, nouns };
  });
  return data;
}

function generateQuestions(data) {
  let questions = [];
  data.forEach(({ sentence, entities, nouns }) => {
    let words = nlp(sentence)
      .terms()
      .out("array")
      .filter((term) => !entities.includes(term) && !nouns.includes(term)); // Extract additional terms for fallback
    let primarySource = entities.length > 0 ? entities : nouns;

    primarySource.forEach((entityOrNoun) => {
      let questionText = sentence.replace(entityOrNoun, "______");
      let correctAnswer = entityOrNoun;

      // Combine entities, nouns, and additional words to form choices for wrong answers
      let choices = primarySource
        .filter((item) => item !== entityOrNoun)
        .concat(
          nouns.filter((noun) => noun !== entityOrNoun),
          words
        );

      // Shuffle and pick up to three unique wrong answers
      shuffleArray(choices);
      let wrongAnswers = choices.slice(0, 3);

      // Ensure there are enough choices, fall back to other words in the sentence if necessary
      while (wrongAnswers.length < 3 && words.length > 0) {
        let additionalChoice = words.pop();
        if (!wrongAnswers.includes(additionalChoice)) {
          wrongAnswers.push(additionalChoice);
        }
      }

      let question = {
        question: questionText,
        answers: [],
      };

      // Add wrong answers with explanations
      wrongAnswers.forEach((answer) => {
        question.answers.push({
          text: answer,
          correct: false,
          explanation: "This is not the correct answer for the blank.",
        });
      });

      // Add the correct answer
      question.answers.push({
        text: correctAnswer,
        correct: true,
        explanation: "This is the correct answer based on the context.",
      });

      // Shuffle answers to mix correct and wrong answers
      shuffleArray(question.answers);

      questions.push(question);
    });
  });

  return questions;
}

export function createQuiz(text) {
  const data = extractEntitiesAndNouns(text);
  const questions = generateQuestions(data);

  const resp = filterAndProcessQuizData(questions);
  return resp["questions"];
}

function removeHTMLTagsAndSpecialPatterns(text) {
  // Remove HTML tags
  text = text.replace(/<\/?[^>]+(>|$)/g, "");

  // Remove patterns like 'H1:', 'H2:', etc., more aggressively including possible white spaces
  text = text.replace(/\b[A-Z]{1,2}\s*:\s*/g, "");

  return text;
}

function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]]; // Swap elements
  }
}

function isValidQuestion(question) {
  const cleanText = removeHTMLTagsAndSpecialPatterns(question.question);
  const wordsCount = cleanText
    .replace(/______/g, "")
    .trim()
    .split(/\s+/).length;
  return wordsCount >= 3;
}

function filterAndProcessQuizData(quizData) {
  let validQuestions = quizData.filter(isValidQuestion);
  shuffleArray(validQuestions); // Shuffle all valid questions

  let filteredQuestions = validQuestions
    .slice(0, 3) // Take the top 3 questions after shuffling
    .map((question) => {
      return {
        ...question,
        question: removeHTMLTagsAndSpecialPatterns(question.question),
      };
    });

  return { questions: filteredQuestions };
}
