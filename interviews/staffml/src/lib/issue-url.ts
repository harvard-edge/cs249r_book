// Shared utilities for GitHub issue reporting and community contribution

const REPO = 'harvard-edge/cs249r_book';
const LABELS = 'staffml';

export type ReportCategory =
  | 'answer_incorrect'
  | 'math_error'
  | 'unclear_question'
  | 'wrong_difficulty'
  | 'duplicate'
  | 'other';

const CATEGORY_LABELS: Record<ReportCategory, string> = {
  answer_incorrect: 'Answer is incorrect',
  math_error: "Math doesn't check out",
  unclear_question: 'Question is unclear or ambiguous',
  wrong_difficulty: 'Difficulty level seems wrong',
  duplicate: 'Duplicate of another question',
  other: 'Other',
};

export function getCategoryLabel(cat: ReportCategory): string {
  return CATEGORY_LABELS[cat];
}

export function getReportCategories(): { value: ReportCategory; label: string }[] {
  return Object.entries(CATEGORY_LABELS).map(([value, label]) => ({
    value: value as ReportCategory,
    label,
  }));
}

interface QuestionInfo {
  id: string;
  title: string;
  level: string;
  track: string;
  competency_area: string;
  topic?: string;
  zone?: string;
}

/** Build a GitHub issue URL for reporting a problem with a question */
export function buildReportUrl(
  q: QuestionInfo,
  category?: ReportCategory,
): string {
  const categoryLine = category
    ? `**Category:** ${CATEGORY_LABELS[category]}\n`
    : '';

  const body = [
    `**Question ID:** \`${q.id}\``,
    `**Title:** ${q.title}`,
    `**Level:** ${q.level}`,
    `**Track:** ${q.track}`,
    `**Area:** ${q.competency_area}`,
    q.topic ? `**Topic:** ${q.topic}` : '',
    q.zone ? `**Zone:** ${q.zone}` : '',
    categoryLine ? `\n${categoryLine}` : '',
    `\n**What's wrong:**\n\n`,
    `**Expected:**\n\n`,
  ]
    .filter(Boolean)
    .join('\n');

  const title = `[StaffML] Issue with: ${q.title}`;

  return `https://github.com/${REPO}/issues/new?labels=${LABELS}&title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
}

/** Build a GitHub issue URL for suggesting an improvement to a question's answer */
export function buildSuggestUrl(q: QuestionInfo): string {
  const body = [
    `**Question ID:** \`${q.id}\``,
    `**Title:** ${q.title}`,
    `**Level:** ${q.level} | **Track:** ${q.track} | **Area:** ${q.competency_area}`,
    '',
    '**Suggested improvement:**',
    '',
    '',
    '**Why this is better:**',
    '',
    '',
  ].join('\n');

  const title = `[StaffML] Improve: ${q.title}`;

  return `https://github.com/${REPO}/issues/new?labels=${LABELS},improvement&title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
}

/** Build a generic site-level issue URL */
export function buildSiteIssueUrl(): string {
  return `https://github.com/${REPO}/issues/new?template=blank&title=${encodeURIComponent('[StaffML] ')}&labels=${LABELS}`;
}

/** Build a URL for contributing a new question */
export function buildContributeUrl(): string {
  const body = [
    '## New Question Submission',
    '',
    '**Track:** <!-- cloud / edge / mobile / tinyml -->',
    '**Suggested Level:** <!-- L1 (recall) / L2 (understand) / L3 (apply) / L4 (analyze) / L5 (evaluate) / L6+ (create) -->',
    '**Topic:** <!-- Pick from existing topics or suggest a new one -->',
    '**Zone:** <!-- recall / analyze / design / implement / diagnosis / specification / fluency / evaluation / realization / optimization / mastery -->',
    '',
    '### Scenario',
    '<!-- The interview question as the interviewer would ask it -->',
    '',
    '',
    '### Expected Answer',
    '<!-- The model answer with reasoning -->',
    '',
    '',
    '### Common Mistake',
    '<!-- What candidates typically get wrong -->',
    '',
    '',
    '### Napkin Math (if applicable)',
    '<!-- Step-by-step calculation with real hardware specs -->',
    '',
    '',
    '---',
    '*Thank you for contributing! All submissions are reviewed before being added to the vault.*',
  ].join('\n');

  const title = '[StaffML] New question: ';

  return `https://github.com/${REPO}/issues/new?labels=${LABELS},contribution&title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`;
}
