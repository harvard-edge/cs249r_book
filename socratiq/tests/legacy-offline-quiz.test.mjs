import test from 'node:test';
import assert from 'node:assert/strict';
import {createQuiz} from '../src_shadow/js/components/quiz/backupQuiz.js';

test('offline quizzes keep answer choices distinct when nouns overlap fallback terms', () => {
  const random = Math.random;
  Math.random = () => 0;
  try {
    const questions = createQuiz('The memory bandwidth limits performance of the accelerator during inference.');
    assert.ok(questions.length > 0);
    for (const question of questions) {
      const answers = question.answers.map(answer => answer.text);
      assert.equal(new Set(answers).size, answers.length);
      assert.equal(question.answers.filter(answer => answer.correct).length, 1);
      assert.ok(question.question.includes('______'));
    }
  } finally {
    Math.random = random;
  }
});

test('offline quizzes finish with sparse passages and do not invent extra choices', () => {
  for (const passage of ['', 'Memory.', 'The memory is fast.']) {
    const questions = createQuiz(passage);
    for (const question of questions) {
      assert.equal(new Set(question.answers.map(answer => answer.text)).size, question.answers.length);
      assert.equal(question.answers.filter(answer => answer.correct).length, 1);
    }
  }
});
