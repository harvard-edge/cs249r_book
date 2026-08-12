// Extract rubric checkpoints from question answers for structured self-assessment

export interface RubricItem {
  text: string;
  checked: boolean;
}

const TECHNICAL_SIGNALS = /\b(bottleneck|trade-?off|latency|throughput|bandwidth|memory|compute|power|energy|scaling|overhead|utilization|efficiency|architecture|pipeline|parallelism|quantization|pruning|distillation|roofline|FLOPS?|TFLOPS?|cache|buffer|batch|inference|training|prefetch|fusion|kernel|tensor|gradient|checkpoint|shard|replica|allreduce|RDMA|NVLink|PCIe|DRAM|SRAM|HBM|SoC|ASIC|FPGA|NPU|DMA|SIMD)\b/i;

const CAUSAL_SIGNALS = /\b(because|therefore|since|causes?|results? in|leads? to|means|requires?|prevents?|reduces?|increases?|enables?|limits?|dominates?|determines?)\b/i;

const FILLER_OPENERS = /^(you\s+(must|should|need\s+to|can|would)|the\s+key\s+(is|here\s+is)|this\s+(means|is\s+because|ensures?|requires?)|in\s+(this|the)\s+case|note\s+that|importantly|essentially|basically|overall|in\s+summary|for\s+example|in\s+other\s+words|it\s+is\s+important\s+to)\s+/i;

/**
 * Extract 2-4 rubric checkpoints from a question's solution and common mistake.
 * Scores sentences by technical density and causal reasoning signals,
 * preferring sentences that name mechanisms and trade-offs over generic openers.
 */
export function extractRubric(
  solution: string,
  commonMistake?: string,
  napkinMath?: string
): RubricItem[] {
  const items: RubricItem[] = [];

  const sentences = solution
    .split(/(?<=[.!?])\s+/)
    .filter(s => s.length > 20 && s.length < 250)
    .map(s => s.trim());

  const scored = sentences
    .map((s) => ({ text: s, score: scoreSentence(s) }))
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score);

  const selected = scored.slice(0, 3);
  selected.sort((a, b) => sentences.indexOf(a.text) - sentences.indexOf(b.text));

  for (const { text } of selected) {
    const shortened = shortenToCheckpoint(text);
    if (shortened) {
      items.push({ text: shortened, checked: false });
    }
  }

  if (commonMistake && commonMistake.length > 10) {
    const mistakeSentences = commonMistake
      .split(/(?<=[.!?])\s+/)
      .filter(s => s.length > 15)
      .map(s => s.trim());
    const best = mistakeSentences
      .map(s => ({ text: s, score: scoreSentence(s) }))
      .sort((a, b) => b.score - a.score)[0];
    const source = best ? best.text : commonMistake;
    const mistakeCore = shortenToCheckpoint(source);
    if (mistakeCore) {
      items.push({ text: `Avoided: ${mistakeCore}`, checked: false });
    }
  }

  if (napkinMath && napkinMath.length > 10) {
    items.push({ text: 'Included quantitative estimate (napkin math)', checked: false });
  }

  return items.slice(0, 4);
}

function scoreSentence(sentence: string): number {
  let score = 0;
  const techMatches = sentence.match(TECHNICAL_SIGNALS);
  if (techMatches) score += techMatches.length * 2;
  if (CAUSAL_SIGNALS.test(sentence)) score += 3;
  const numberCount = (sentence.match(/\d+(\.\d+)?/g) || []).length;
  if (numberCount > 0) score += Math.min(numberCount, 3);
  if (/\b\d+(\.\d+)?[x×]\b/i.test(sentence)) score += 2;
  if (sentence.length < 30) score -= 2;
  if (/^(the|this|it|a|an)\s/i.test(sentence) && !CAUSAL_SIGNALS.test(sentence)) score -= 1;
  return score;
}

function shortenToCheckpoint(sentence: string): string | null {
  let s = sentence.trim();

  s = s.replace(FILLER_OPENERS, '');
  s = s.replace(/\.$/, '');
  s = s.charAt(0).toUpperCase() + s.slice(1);

  if (s.length < 15 || /^(it|this|that)\s/i.test(s)) return null;

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
