// Extract rubric checkpoints from question answers for structured self-assessment

export interface RubricItem {
  text: string;
  checked: boolean;
}

/**
 * Extract 2-4 rubric checkpoints from a question's solution and common mistake.
 * Uses heuristic parsing: sentence splitting + key concept extraction.
 */
export function extractRubric(
  solution: string,
  commonMistake?: string,
  napkinMath?: string
): RubricItem[] {
  const items: RubricItem[] = [];

  // Split solution into sentences, take the most informative ones
  const sentences = solution
    .split(/(?<=[.!?])\s+/)
    .filter(s => s.length > 20 && s.length < 200)
    .map(s => s.trim());

  // Take up to 3 key points from the solution
  const keyPoints = sentences.slice(0, 3);
  keyPoints.forEach(point => {
    // Shorten to the core concept
    const shortened = shortenToCheckpoint(point);
    if (shortened) {
      items.push({ text: shortened, checked: false });
    }
  });

  // Add "avoided common mistake" checkpoint if available
  if (commonMistake && commonMistake.length > 10) {
    const mistakeCore = shortenToCheckpoint(commonMistake);
    if (mistakeCore) {
      items.push({ text: `Avoided: ${mistakeCore}`, checked: false });
    }
  }

  // Add napkin math checkpoint if present
  if (napkinMath && napkinMath.length > 10) {
    items.push({ text: 'Included quantitative estimate (napkin math)', checked: false });
  }

  // Cap at 4 items
  return items.slice(0, 4);
}

/**
 * Shorten a sentence to a rubric-friendly checkpoint.
 * Removes filler, keeps the technical core.
 */
function shortenToCheckpoint(sentence: string): string | null {
  let s = sentence.trim();

  // Remove leading "You must", "The key is", etc.
  s = s.replace(/^(you\s+(must|should|need\s+to)|the\s+key\s+(is|here)\s+is|this\s+means)\s+/i, '');

  // Remove trailing period
  s = s.replace(/\.$/, '');

  // Capitalize first letter
  s = s.charAt(0).toUpperCase() + s.slice(1);

  // Skip if too short or too generic
  if (s.length < 15 || /^(it|this|that|the)\s/i.test(s)) return null;

  // Truncate if still too long
  if (s.length > 100) {
    const cutoff = s.lastIndexOf(' ', 95);
    s = s.substring(0, cutoff > 50 ? cutoff : 95) + '…';
  }

  return s;
}

/**
 * Compute score from checked rubric items.
 * Returns 0-3 scale matching self-assessment.
 */
export function rubricToScore(items: RubricItem[]): number {
  if (items.length === 0) return 0;
  const checked = items.filter(i => i.checked).length;
  const ratio = checked / items.length;
  if (ratio >= 0.75) return 3; // nailed it
  if (ratio >= 0.5) return 2;  // partial
  if (ratio > 0) return 1;     // wrong-ish
  return 0;                     // skipped
}
