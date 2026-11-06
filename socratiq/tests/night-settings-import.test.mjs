import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const validatorURL = new URL('../src_shadow/js/components/spaced_repetition/utils/import-validation.js', import.meta.url);
const validatorSource = await readFile(validatorURL, 'utf8');
const validatorModule = `data:text/javascript;base64,${Buffer.from(validatorSource).toString('base64')}`;
const { validateFlashcardImport } = await import(validatorModule);
const ankiSource = (await readFile(new URL('../src_shadow/js/components/spaced_repetition/utils/anki-converter.js', import.meta.url), 'utf8')).replace("'./import-validation.js'", JSON.stringify(validatorModule));
const { AnkiConverter } = await import(`data:text/javascript;base64,${Buffer.from(ankiSource).toString('base64')}`);
const backup = () => ({ chapter_card_sets: [{ chapter: 1, cards: [{ id: 'constructed-card', question: 'Constructed question', answer: 'Constructed answer', customField: { source: 'constructed fixture' } }] }] });

test('legacy backup defaults preserve content and unknown compatibility fields without mutating input', () => {
  const input = backup();
  const before = JSON.stringify(input);
  const output = validateFlashcardImport(input);
  assert.equal(JSON.stringify(input), before);
  assert.equal(output.chapter_card_sets[0].cards[0].chapter, 1);
  assert.deepEqual(output.chapter_card_sets[0].cards[0].tags, []);
  assert.deepEqual(output.chapter_card_sets[0].cards[0].customField, { source: 'constructed fixture' });
});

test('import validation rejects an invalid final record before the input can be written', () => {
  for (const badCard of [null, { question: {}, answer: 'A' }, { question: 'Q', answer: 'A', tags: {} }, { question: 'Q', answer: 'A', chapter: 2 }, { question: 'Q', answer: 'A', interval: Infinity }, { question: 'Q', answer: 'A', interval: 1e300 }, { question: 'Q', answer: 'A', repetitions: 0.5 }, { question: 'Q', answer: 'A', reviewHistory: [{ quality: 7, date: '2026-01-01' }] }, { question: 'Q', answer: 'A', answer2: JSON.parse('{"__proto__":{"polluted":true}}') }]) {
    const input = backup();
    input.chapter_card_sets[0].cards.push(badCard);
    assert.throws(() => validateFlashcardImport(input));
  }
  for (const data of [null, [], {}, { chapter_card_sets: {} }, { chapter_card_sets: [{ chapter: -1, cards: [] }] }, { chapter_card_sets: [{ chapter: 0, cards: Array.from({ length: 10001 }, (_, id) => ({ id, question: 'Q', answer: 'A' })) }] }]) assert.throws(() => validateFlashcardImport(data));
});

test('Anki parsing keeps the first headerless card and round trips quoted multiline content', t => {
  t.mock.method(console, 'error', () => {});
  const parsed = AnkiConverter.parseAnkiCSV('\n#separator:tab\n\nFront one\tBack one\n# Constructed ignored comment\nFront two\tBack two');
  assert.equal(parsed.chapter_card_sets[0].cards.length, 2);
  assert.equal(parsed.chapter_card_sets[0].cards[0].question, 'Front one');
  assert.equal(AnkiConverter.parseAnkiCSV('First front\tFirst back\nSecond front\tSecond back').chapter_card_sets[0].cards.length, 2);
  const input = backup();
  input.current_chapter = { chapter: 1, title: 'Constructed: deck' };
  input.chapter_card_sets[0].cards[0].question = 'Line one; "quote"\nLine two';
  input.chapter_card_sets[0].cards[0].answer = '<p>Answer; more</p>';
  input.chapter_card_sets[0].cards[0].tags = ['tag;one'];
  const output = AnkiConverter.parseAnkiCSV(AnkiConverter.convertToAnkiFormat(input));
  assert.equal(output.current_chapter.title, input.current_chapter.title);
  for (const field of ['question', 'answer', 'tags']) assert.deepEqual(output.chapter_card_sets[0].cards[0][field], input.chapter_card_sets[0].cards[0][field]);
  assert.throws(() => AnkiConverter.parseAnkiCSV('#separator:Semicolon\n"Unclosed;answer'));
  assert.throws(() => AnkiConverter.parseAnkiCSV('not a valid row'));
});
