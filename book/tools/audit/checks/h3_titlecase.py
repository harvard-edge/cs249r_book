"""Check: H3+ headings in title case (should be sentence case).

Rule: book-prose-merged.md section 10.9
    H1 and H2: headline style (capitalize principal words)
    H3 and below: sentence style (first word + proper nouns only)

Detection: any heading at H3/H4/H5/H6 where more than one word starts with
a capital letter (excluding the first word). Proper-noun detection is NOT
safe to automate — "ResNet Architecture" should become "ResNet architecture"
but "Hardware Balance" should become "Hardware balance". A subagent must
review each one.

Auto-fixable: NO. Every issue is marked needs_subagent=True.

The section header slug (e.g. {#sec-foo-bar-1234}) is not considered part
of the heading text for case-analysis purposes; we strip it before
counting caps.

Protected contexts this check skips:
- Code fences, YAML, display math, HTML comments (default line skip)
- H1 and H2 headings (the rule only applies to H3+)
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import (
    LineWalker,
    default_line_skip,
    heading_level,
)

CATEGORY = "h3-titlecase"
RULE = "book-prose-merged.md section 10.9"
RULE_TEXT = "H3+ headings use sentence case (first word + proper nouns only)"

# Extract the heading text (without the leading ### and without any
# trailing {#sec-...} anchor ID or {.unnumbered} class.
_HEADING_PREFIX_RE = re.compile(r"^\s*#{1,6}\s+")
_HEADING_SUFFIX_RE = re.compile(r"\s*\{[^}]*\}\s*$")


def _heading_text(line: str) -> str:
    """Return the heading text with prefix hashes and trailing attrs stripped."""
    text = _HEADING_PREFIX_RE.sub("", line)
    text = _HEADING_SUFFIX_RE.sub("", text)
    return text.rstrip()


# Words that should stay lowercase in sentence case (articles, conjunctions,
# short prepositions). These are the "minor words" that title case would
# also lowercase, so they are neutral for title-case detection.
_TITLE_CASE_SKIP = {
    "a", "an", "and", "the", "of", "in", "on", "at", "to", "for",
    "or", "but", "nor", "so", "yet", "as", "by", "vs", "vs.",
    "from", "with", "into", "onto", "upon", "over", "under",
    "via", "per",
}

# Word pattern: starts with a letter, may contain letters/digits/hyphens.
# Apostrophes split words (e.g. "Sutton's" is "Sutton" + "'s" — we treat the
# whole thing as one word since the apostrophe is inside).
_WORD_RE = re.compile(r"[A-Za-z][\w'-]*")


# ── Proper-noun / acronym detection (Pass 16 Item B) ────────────────────────
#
# The Pass 15 detector treated every post-first-word capital as a potential
# title-case violation. This over-flagged 75 headings across vol1 and vol2
# where the capitals were legitimately: acronyms (GPU, I/O, SIMD), product
# names (NVIDIA DGX, AMD Instinct MI300X), named principles (Amdahl's Law),
# legislation (EU AI Act), D·A·M/C³ taxonomy axes (Machine), and the first
# word after a colon (CMS 17 §8.158). Pass 15 manually walked all 75 and
# documented them in book-prose-merged.md §10.9.
#
# Pass 16 Item B teaches the detector what §10.9 already knows:
#
#   1. Phrase preprocessor: replace known multi-word named phrases
#      ("Amdahl's Law", "Flash Attention", "EU AI Act", ...) with sentinels
#      BEFORE tokenization, so the individual words never enter the cap
#      counter.
#
#   2. Axis preprocessor: replace parenthetical D·A·M / C³ axis labels
#      ("(Data)", "(Machine)", "(Computation)", ...) with sentinels.
#
#   3. Slash-acronym preprocessor: replace slash-joined single-cap
#      clusters ("I/O", "M/G/c/K") with a sentinel, since `_WORD_RE`
#      would otherwise tokenize each letter separately.
#
#   4. Post-colon first-word skip: the word immediately after the first
#      colon is exempt from cap counting (CMS 17 §8.158 caps it).
#
#   5. Proper-noun classifier: tokens matching all-caps patterns, hardware
#      SKUs (A100, MI300X), single-cap-hyphen-lowercase patterns (S-curve),
#      contiguous CamelCase (PyTorch, SuperPOD), or the hand-curated set
#      are skipped from cap counting.
#
#   6. Concept-term override: if the heading contains any §10.3 lowercase
#      concept term ("iron law", "memory wall", "compute wall", ...), the
#      cap-count threshold drops from 2 to 1. This catches the Pass 15
#      Batch 1 false negative `#### Exercise two: *iron law Analysis*`
#      where "Analysis" is the only mis-cased token.
#
# Hand-curated proper-noun set. Only add single-cap names that the all-caps
# and CamelCase regexes miss. Companies, products, people, and "attached"
# words (Law / Act) that sit grammatically next to a proper noun in named
# phrases.

_PROPER_NOUN_SET = frozenset({
    # Companies / organizations (single-cap, not caught by all-caps regex)
    "Google", "Meta", "Intel", "Apple", "Amazon", "Microsoft", "Anthropic",
    "Nvidia",  # rare lowercase-i spelling; NVIDIA all-caps is caught by regex
    "Qualcomm", "Cerebras", "Graphcore", "Groq", "Habana", "Arm",
    # Products / frameworks (single-cap or unusual casing)
    "Python", "Ray", "Spark", "Dask", "Kafka", "Docker", "Kubernetes",
    "Linux", "Windows", "Android", "Arduino", "Slurm", "Vertex",
    "Instinct",  # AMD Instinct product line
    "Llama", "Mistral", "Gemini", "Claude", "Falcon",  # model families
    "MobileNet", "AlexNet", "ResNet", "EfficientNet",  # model architectures
    "PyTorch", "TensorFlow", "NumPy", "SciPy", "JAX",
    "SuperPOD",  # NVIDIA DGX SuperPOD
    "Teton", "Grand",  # Meta Grand Teton
    "Hopper", "Ampere", "Volta", "Pascal", "Kepler", "Maxwell",  # NVIDIA arches
    # People who own named principles — their possessives are handled by
    # multi-word phrase preprocessing, but their bare surnames also show up
    "Amdahl", "Gustafson", "Patterson", "Hennessy", "Turing", "Feynman",
})

# NOTE: "Law" and "Act" are deliberately NOT in _PROPER_NOUN_SET. Named
# principles (Amdahl's Law, EU AI Act) are handled by the phrase
# preprocessor instead. Listing them as proper nouns would cause false
# negatives for legitimate concept headings like "### Young-Daly Law"
# (which should read "### Young-Daly law" per §10.9 named-principle
# sub-rule for non-canonical principles).

# Multi-word named phrases. These are replaced with a sentinel BEFORE
# tokenization so that the individual words never enter the cap counter.
# Order matters: longer phrases must be replaced first to avoid partial
# matches (e.g. "Flash Attention" before "Flash").
_NAMED_PHRASES = (
    "Amdahl's Law",
    "Gustafson's Law",
    "Moore's Law",
    "Young-Daly law",     # lowercase per §10.9 coined-principle rule
    "Young-Daly Law",     # defensive: catch the miscased form too
    "Young-Daly model",
    "Young-Daly",         # bare surname pair used in cross-refs
    "EU AI Act",
    "CHIPS Act",
    "Flash Attention",
    "Paged Attention",
    "Tensor Cores",
    "Tensor Core",
    "Compute Express Link",
    "Google Vertex AI",
    "Vertex AI",
    "Stable Diffusion",
    "Training GPT-2",  # Running example heading in training.qmd
    # Book-canonical taxonomy names. The middle-dot (U+00B7) and math
    # superscript notation confuse the word tokenizer, so we replace
    # them as whole phrases before tokenization. Both the Unicode "³"
    # form and the LaTeX "$^3$" form appear in the corpus.
    "D·A·M taxonomy",
    "D·A·M",
    "C$^3$",
    "C³",
)

# D·A·M / C³ taxonomy axis labels when they appear in parentheses in a
# heading. Only the parenthesized form is a framework label — bare
# "Machine" or "Data" in body prose would be ambiguous.
_DAM_C3_AXIS_WORDS = (
    "Data", "Algorithm", "Machine",
    "Computation", "Communication", "Coordination",
)

# Sentinels used by the preprocessors. Each sentinel matches `_WORD_RE` as
# a single token and is recognized by `_is_proper_noun` as a proper noun.
_SENTINEL_PHRASE = "XXNAMEDPHRASEXX"
_SENTINEL_AXIS = "XXAXISXX"
_SENTINEL_SLASH = "XXSLASHACRONYMXX"
_SENTINELS = frozenset({_SENTINEL_PHRASE, _SENTINEL_AXIS, _SENTINEL_SLASH})

# Slash-joined single-cap acronyms: I/O, M/G/c/K, A/B, CI/CD (latter is
# also caught by the all-caps regex on each token, but no harm in
# preprocessing it). Must start with a letter and contain at least one
# slash-separated letter cluster.
_SLASH_ACRONYM_RE = re.compile(r"\b[A-Za-z](?:/[A-Za-z])+\b")

# Hardware SKUs: A100, H100, V100, GH200, MI300, MI300X — capital letters
# followed by digits, optionally terminated by a capital suffix.
_HW_SKU_RE = re.compile(r"^[A-Z]+\d+[A-Z]?$")

# Leading-caps acronym: starts with a capital followed by 1+ cap/digit,
# optionally continuing with hyphen-separated cap/digit/lowercase groups.
# Matches NVIDIA, GPU, CI, CD, GPT-2, IO-aware. Does NOT match
# "Memory-Bound" (only one cap before hyphen).
_LEADING_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9]+(?:-[A-Za-z0-9]+)*$")

# Single cap + hyphen + lowercase (S-curve, V-shape, U-shape, T-test).
_CAP_HYPHEN_RE = re.compile(r"^[A-Z]-[a-z]+$")

# Contiguous CamelCase with ≥2 caps and no hyphen between them (PyTorch,
# NumPy, SuperPOD, TensorFlow, RoC). The no-hyphen requirement prevents
# this rule from swallowing legitimate title-case violations like
# "Memory-Bound" or "Iron-Law".
_CAMEL_CASE_RE = re.compile(r"^[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*$")


# Concept terms from book-prose-merged.md §10.3 that must be lowercase in
# body prose. When a heading contains any of these as a substring, the
# cap-count threshold drops from 2 to 1 — because a single stray cap next
# to a concept term is strongly suggestive of title-case drift. The term
# list is kept conservative: we include only the multi-word concept terms
# where the override meaningfully helps, not single-word terms like
# "transformer" that appear in many correct headings.
_CONCEPT_TERMS_LOWER = (
    "iron law",
    "degradation equation",
    "verification gap",
    "data wall",
    "compute wall",
    "memory wall",
    "power wall",
    "energy corollary",
    "bitter lesson",
    "scaling laws",
    "information roofline",
    "long tail",
    "data gravity",
    "napkin math",
    "starving accelerator",
    "latency cliff",
    "four pillars framework",
)


def _preprocess(text: str) -> str:
    """Replace named phrases, axis labels, and slash-acronyms with sentinels.

    Preprocessing runs in a fixed order — longest-first for phrase
    matching, then axis labels, then slash-acronyms. The output is fed
    to `_WORD_RE.findall()` with the sentinels appearing as single
    tokens that `_is_proper_noun` recognizes.
    """
    # 1. Named phrases (longest first to avoid partial matches)
    for phrase in sorted(_NAMED_PHRASES, key=len, reverse=True):
        text = text.replace(phrase, _SENTINEL_PHRASE)
    # 2. D·A·M / C³ axis labels inside parentheses
    for axis in _DAM_C3_AXIS_WORDS:
        text = text.replace(f"({axis})", f"({_SENTINEL_AXIS})")
    # 3. Slash-joined single-cap acronyms
    text = _SLASH_ACRONYM_RE.sub(_SENTINEL_SLASH, text)
    return text


def _is_proper_noun(w: str) -> bool:
    """Return True if `w` should not count as a title-case violation.

    This is the union of all proper-noun rules: sentinels from the
    preprocessor, the hand-curated name set, the base-before-hyphen
    lookup (for Llama-3 → Llama), the leading-caps regex, the hardware
    SKU regex, the single-cap-hyphen-lowercase regex, and the
    contiguous CamelCase regex. Called once per non-minor token; the
    first match short-circuits.
    """
    if w in _SENTINELS:
        return True
    if w in _PROPER_NOUN_SET:
        return True
    # Llama-3, GPT-4, MI300X-variant — check the base before the first hyphen
    base = w.split("-", 1)[0]
    if base and base != w and base in _PROPER_NOUN_SET:
        return True
    if _LEADING_CAPS_RE.match(w):
        return True
    if _HW_SKU_RE.match(w):
        return True
    if _CAP_HYPHEN_RE.match(w):
        return True
    if _CAMEL_CASE_RE.match(w):
        return True
    return False


def _post_colon_first_word_index(words: list[str], text: str) -> int | None:
    """Return the index in `words` of the first word after the first colon.

    The word is exempt from cap counting per CMS 17 §8.158 ("After a
    colon, the first word is capitalized"). Matching is position-based:
    we walk `words` keeping a cursor into `text`, and return the first
    token whose start index in `text` is strictly greater than the
    position of the first colon.
    """
    colon_pos = text.find(":")
    if colon_pos == -1:
        return None
    cursor = 0
    for i, w in enumerate(words):
        idx = text.find(w, cursor)
        if idx == -1:
            continue
        cursor = idx + len(w)
        if idx > colon_pos:
            return i
    return None


def _has_concept_term(text: str) -> bool:
    """Return True if any §10.3 concept term appears in the heading."""
    lower = text.lower()
    return any(term in lower for term in _CONCEPT_TERMS_LOWER)


# Hyphenated compound of the form "Xxxx-Yyyy" where BOTH parts start with
# a capital followed by one or more lowercase letters. Matches:
#   Memory-Bound, Non-Linear, Mixed-Precision, Cost-Performance, On-Chip,
#   Host-Accelerator, Row-Major, Break-Even, Training-Serving, Tail-Tolerant
# Does NOT match (correctly rejected):
#   Multi-GPU (GPU is all caps), Multi-NUMA (NUMA all caps),
#   GPT-2 (digit), Llama-3 (digit), IO-aware (aware is lowercase start),
#   memory-bound (first part lowercase — the correct form).
_COMPOUND_VIOLATION_RE = re.compile(r"\b([A-Z][a-z]+)-([A-Z][a-z]+)\b")


def _has_compound_violation(text: str) -> bool:
    """Return True if the heading contains a title-cased hyphenated compound
    whose second half is neither an acronym nor a known proper noun.

    Per book-prose-merged.md §10.9: "Hyphenated compound second-part
    lowercase. In H3+ headings, the second part of a hyphenated compound
    is lowercase unless it is itself a proper noun or acronym."

    This check runs AFTER phrase preprocessing (so "Compute Express Link"
    and "Amdahl's Law" are already sentinels) but directly on the raw
    heading text for the compound pattern, because the compound pattern
    is word-internal and the tokenizer has no concept of compounds.
    """
    for m in _COMPOUND_VIOLATION_RE.finditer(text):
        first, second = m.group(1), m.group(2)
        # Skip compounds whose first part is a known proper noun — e.g.
        # "Apple-Silicon" (if we ever encounter it) or "Meta-Ranker".
        # Both halves being mixed-case and neither being a proper noun
        # is the signature of a title-case violation.
        if _is_proper_noun(first) or _is_proper_noun(second):
            continue
        return True
    return False


def _looks_titlecase(text: str) -> bool:
    """Return True if the heading text appears to be in title case.

    This is the Pass 16 Item B detector. See the long comment above
    `_PROPER_NOUN_SET` for the design rationale. The decision procedure:

      1. Preprocess the text to replace named phrases, D·A·M/C³ axis
         labels, and slash-joined acronyms with sentinels.
      2. Tokenize with `_WORD_RE`.
      3. Find the post-colon first-word index (exempt by CMS 8.158).
      4. For each token after position 0 and except the post-colon
         first word: if it's a minor word, skip. If it's a proper noun
         (any of the six rules), skip. Otherwise count it as a content
         word, and increment cap_count if it starts with a capital.
      5. If the heading contains a §10.3 concept term, use threshold 1;
         otherwise use threshold 2. Flag if content_count and cap_count
         both meet the threshold.
      6. Independently of (5), flag any heading containing a
         hyphenated-compound title-case violation (§10.9 "Hyphenated
         compound second-part lowercase"): `### Memory-Bound workloads`
         or `### On-Chip Memory`. The compound check runs on the
         preprocessed text so named phrases are already sentinels.
    """
    preprocessed = _preprocess(text)

    # §10.9 hyphenated-compound second-part rule. This is an independent
    # signal from the cap-count threshold and runs first because a
    # compound violation is unambiguous — no threshold needed.
    if _has_compound_violation(preprocessed):
        return True

    words = _WORD_RE.findall(preprocessed)
    if len(words) < 2:
        return False  # single-word headings are always "correct" case-wise

    post_colon_idx = _post_colon_first_word_index(words, preprocessed)

    cap_count = 0
    content_count = 0
    for i, w in enumerate(words):
        if i == 0:
            continue  # first word is always capitalized in sentence case
        if i == post_colon_idx:
            continue  # CMS 17 §8.158: first word after a colon is capitalized
        if w.lower() in _TITLE_CASE_SKIP:
            continue
        if _is_proper_noun(w):
            continue
        content_count += 1
        if w[0].isupper():
            cap_count += 1

    threshold = 1 if _has_concept_term(text) else 2
    return content_count >= threshold and cap_count >= threshold


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for H3+ headings that look like title case.

    Returns (issues, next_counter). Every issue has needs_subagent=True
    because proper-noun detection is not safe to automate.
    """
    issues: list[Issue] = []
    counter = start_counter

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if default_line_skip(line, state):
            continue

        level = heading_level(line)
        if level is None or level < 3:
            continue

        heading_text = _heading_text(line)
        if not _looks_titlecase(heading_text):
            continue

        issues.append(
            Issue(
                id=make_issue_id(scope, CATEGORY, counter),
                category=CATEGORY,
                rule=RULE,
                rule_text=RULE_TEXT,
                file=str(file_path),
                line=line_num,
                col=0,
                before=line,
                suggested_after="",  # Subagent fills this in
                auto_fixable=False,
                needs_subagent=True,
                reason=f"H{level} heading in title case: {heading_text!r}",
            )
        )
        counter += 1

    return issues, counter


# ── Adversarial self-test (Pass 16 Item B) ──────────────────────────────────
#
# Run with:
#     python3 book/tools/audit/checks/h3_titlecase.py
#
# Two corpora:
#
#   POSITIVE_CASES (detector MUST flag) — hand-crafted title-case violations
#   that represent the failure mode this check is supposed to catch. These
#   are the "canary" cases. If any positive case stops being flagged after a
#   detector change, a regression has been introduced.
#
#   NEGATIVE_CASES (detector MUST NOT flag) — representative samples of the
#   Pass 15 accept-list, one per rule category, plus additional "correct"
#   headings pulled from the book. These are the false-positive patterns.
#
# Neither list is exhaustive. The full regression test is re-running the
# audit scanner against the 75 accept-list entries in accepted_fps.json and
# verifying that all 75 still route to `accepted` (or equivalently, that
# `scan --no-accept-list` produces zero open h3-titlecase issues).

_POSITIVE_CASES = [
    # Classic title case — every content word capitalized.
    "### Hardware Acceleration Methods",
    "### Iron Law Of ML Systems",
    "### The Memory Hierarchy Of Modern GPUs",
    "### Training Pipeline Architecture",
    "### Batch Normalization Techniques",
    # Hyphenated-compound violations (must not be swallowed by the
    # CamelCase or leading-caps regex).
    "### Memory-Bound Workloads",
    "### Compute-Bound Training Jobs",
    # The Pass 15 Batch 1 false negative — concept term + one stray cap.
    # Threshold drops to 1 when §10.3 concept term present.
    "#### Exercise two: *iron law Analysis*",
    # A §10.3 concept term with one bad cap elsewhere.
    "### The Memory Wall Problem",
]

_NEGATIVE_CASES = [
    # After-colon CMS 8.158 — first word after colon legitimately capped.
    "### The ICR frontier: When data becomes a tax",
    "### Step 3: Estimate ROI",
    "### Data echoing: Amortizing I/O costs",
    "### The dispatch tax: Python overhead vs. GPU reality",
    "### The technology S-curve: Why we must shift",
    "### Step 2: Simplify for large N",
    # Numbered prefix patterns.
    "#### Step 1: Calculate the failure rate per GPU subsystem",
    "#### Phase 2: Backward pass",
    "### Case 3: The compute wall (Machine)",
    # Named principles and legislation.
    "### Amdahl's Law and Gustafson's Law",
    "#### Strong scaling (Amdahl's Law)",
    "#### The EU AI Act",
    # D·A·M / C³ taxonomy axis labels.
    "### Case 1: The starving accelerator (Data)",
    "#### Computation ∩ Communication",
    # Acronym-dense headings.
    "#### Evolution from SIMD to SIMT architectures",
    "#### Overlapping I/O and compute",
    "##### Skew detection in CI/CD",
    "#### The M/G/c/K queue model for GPU serving",
    "### Performance metrics: TTFT and TPOT",
    # Hardware product names.
    "### AMD Instinct MI300X",
    "#### NVIDIA DGX SuperPOD",
    "#### Transformer architectures (GPT-2/Llama)",
    # Proper-noun company/product names.
    "### Advanced Slurm configuration for ML",
    "### Google Vertex AI",
    "### Large-scale LLM training at Meta",
    # Phrase-preprocessed named techniques.
    "### Flash Attention: IO-aware attention optimization",
    "#### When to use Flash Attention",
    # Mixed-case product names (CamelCase regex territory).
    "### PyTorch vs. TensorFlow",
    "### The NumPy array model",
    # Short headings and edge cases.
    "### TinyML",  # single word
    "### The bitter lesson",  # concept term, no extra caps
    "### Memory hierarchy basics",  # one content cap, below threshold
]


def _self_test() -> int:
    """Run the adversarial test corpora and report failures.

    Returns a non-zero exit code if any case is misclassified, zero
    otherwise. Designed for manual invocation during detector edits.
    """
    pos_fail: list[str] = []
    neg_fail: list[str] = []

    for line in _POSITIVE_CASES:
        text = _heading_text(line)
        if not _looks_titlecase(text):
            pos_fail.append(line)

    for line in _NEGATIVE_CASES:
        text = _heading_text(line)
        if _looks_titlecase(text):
            neg_fail.append(line)

    total = len(_POSITIVE_CASES) + len(_NEGATIVE_CASES)
    failures = len(pos_fail) + len(neg_fail)
    passed = total - failures

    print(f"h3_titlecase self-test: {passed}/{total} passed")
    if pos_fail:
        print(f"\n{len(pos_fail)} POSITIVE case(s) NOT flagged (false negatives):")
        for line in pos_fail:
            print(f"  - {line}")
    if neg_fail:
        print(f"\n{len(neg_fail)} NEGATIVE case(s) flagged (false positives):")
        for line in neg_fail:
            print(f"  - {line}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_self_test())
