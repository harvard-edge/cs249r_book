---
paths:
  - "book/**"
---

# Book Prose V2 — MIT Press Editorial Addendum

This document supplements `book-prose.md` with MIT Press copy editor decisions from the 2026 copyedit round (Pamela Hunt, Job #837-149678). All `book-prose.md` rules remain in effect unless explicitly superseded. Applies to both Volume 1 and Volume 2.

Reference: The Chicago Manual of Style, 17th ed.; Webster's New Collegiate Dictionary, 11th ed.

---

## 1. Numbers

This is the most error-prone area. Follow these rules exactly:

- **Spell out one through nine** in body prose: "three models," "five layers"
- **Digits for 10 and above**: "32 GPUs," "50 percent," "12 layers"
- **Always digits with units**: "3 GB," "7 ms," "8-bit," "64-bit" — NEVER "eight-bit" or "thirty-two-bit"
- **Always digits with "percent"**: "50 percent," "10 percent" — NEVER "fifty percent"
- **Digits for technical specs**: batch sizes ("batch size 32"), dimensions ("$32\times32$"), counts with technical context ("16 filters," "64 elements")
- **Digits in ranges**: "10 to 100," "2--4$\times$" — never mix spelled/digit ("ten to 100")
- **Spell out at sentence start**: "Thirty-two GPUs were used" (or restructure to avoid)
- **Full digits in year/page ranges**: 1992--1993, 175--185 (not 175--85)

**Common mistake to avoid**: An automated number-speller will try to spell out ALL single-digit numbers. It MUST skip: numbers with units, numbers in technical specs, numbers adjacent to $\times$, version strings (GPT-3), decimals (3.14), ranges, dimensions, bit-widths.

## 2. Percent

- **Spell out "percent"** in body prose: `94 percent accuracy`
- Use `%` symbol only inside: tables, equations (`$60\%$`), code blocks
- Inline Python: `` `{python} val_str` percent `` (not `` `{python} val_str`% ``)

## 3. Em Dashes

- Closed em dashes: `word—word` (no spaces before or after)
- Maximum one per paragraph
- Em-dash is not a colon — use a colon to introduce consequences/lists
- Do NOT close em dashes inside table cells (the pipe table prettifier manages table spacing)

## 4. Capitalization

### Concept terms — lowercase in body prose

- iron law (not Iron Law)
- degradation equation (not Degradation Equation)
- verification gap (not Verification Gap)
- bitter lesson (the concept; "The Bitter Lesson" only when citing Sutton's essay title)
- ML node (not ML Node)
- data wall, compute wall, memory wall, power wall
- energy corollary
- transformer (the neural network architecture) — lowercase
- four pillars framework

### "transformer" — special handling

Lowercase "transformer" in body prose when referring to the architecture type generically. BUT keep capitalized in:

- **Proper model names**: "Generative Pre-trained Transformer (GPT)," "Vision Transformer (ViT)"
- **Specific architectural components**: "Transformer Decoder," "Transformer Encoder"
- **"Transformer Engine"** (NVIDIA product name)
- **Sentence start**: "Transformers are defined by..."
- **Bold definitions**: `**Transformer**\index{...}`
- **\index{} entries**, section headers, callout titles

### Product names — always capitalized

D-A-M, TinyML, AllReduce, FlashAttention, PagedAttention, CUDA, cuDNN, PyTorch, TensorFlow, MATLAB, ImageNet, BERT, GPT-4, Stable Diffusion, DALL-E, Tensor Cores (NVIDIA product), TPU, Weights & Biases, ResNet, AlexNet, MobileNet, EfficientNet, ONNX, TensorRT, MLPerf, MLCommons, LAPACK, LINPACK, JAX.

### Exceptions — keep caps even for lowercase terms

Sentence start, bold definitions (`**term**`), triple bold (`***term***`), H1/H2 headers, `\index{}` entries, callout `title=` attributes, bold table headers, bold structural labels in callouts (`**The Iron Law Connection:**`).

## 5. Document Position References

**Never use "above" or "below"** to refer to other parts of the text.

Preferred alternatives (in order):
1. **Quarto cross-references**: `@sec-xxx`, `@fig-xxx`, `@eq-xxx` (always best)
2. **"earlier" / "later"**: "as discussed earlier," "examined later"
3. **"preceding" / "following"**: "the preceding analysis," "the following callout"

**Critical word-order rule**: Put "preceding"/"following" BEFORE the noun phrase:
- CORRECT: "the preceding roofline analysis"
- WRONG: "the roofline preceding analysis"

## 6. Abbreviations

### First use per chapter
Every abbreviation expanded on first use in each chapter. Expansion resets at chapter boundaries. Pattern: `convolutional neural network (CNN)` then `CNN`.

### Special cases
- CUDA, cuDNN: no expansion needed
- i.i.d.: always with periods; expand in every chapter
- vs.: always abbreviated with period; never "versus" or bare "vs"
- Adam: Adaptive Moment Estimation (Adam)

### In running text
- "for example" not e.g. (e.g. OK inside parentheses)
- "that is" not i.e. (i.e. OK inside parentheses)
- "and so on" not etc. (etc. OK inside parentheses)

## 7. Compound Words — Canonical Forms

| Term | Form | Notes |
|------|------|-------|
| data center | two words (noun) | "data-center scale" as compound modifier OK |
| dataset | one word | not "data set" |
| trade-off | hyphenated (noun) | not "tradeoff" |
| speedup | one word (noun) | not "speed-up" |
| nonzero | one word | not "non-zero" |
| healthcare | one word | |
| lifecycle | one word | |
| endpoint | one word | |
| scatterplot | one word | |
| smartphone | one word | |
| open-source | hyphenated (adj) | "is open source" unhyphenated as predicate |
| compute-bound | hyphenated (adj before noun) | "is compute bound" unhyphenated as predicate |
| memory-bound | hyphenated (adj before noun) | "is memory bound" unhyphenated as predicate |
| real-time | hyphenated (adj) | "in real time" as noun |

## 8. References in Prose

- "chapter 12" not "Chapter 12"
- "section 3.2" not "Section 3.2"
- Use Quarto `@sec-`, `@fig-`, `@tbl-`, `@eq-` cross-references

## 9. Heading Style

- H1 and H2: **headline style** (capitalize principal words)
- H3 and below: **sentence style** (first word + proper nouns only)
- Figure captions and table titles: **sentence style**; no colon after figure/table number

## 10. Slashes

No spaces around slashes: `training/inference` not `training / inference`.

## 11. Bibliography

- Every `@inproceedings` must have `publisher`
- Every `@article` must have `journal`
- Include `pages` and `doi` when available
- No em dashes for repeat author names
- Letter-by-letter alphabetical order
- Publisher locations removed for consistency
- Confirm all URLs

## 12. Glossary

- Product names use proper capitalization: ImageNet, PyTorch, TensorFlow, etc.
- Generic terms lowercase unless proper noun
- Per CMS: initial cap only if proper noun

## 13. Punctuation

- Serial (Oxford) comma: always
- Comma after e.g., i.e.
- Double quotation marks; period/comma inside, colon/semicolon outside
- Contractions forbidden in body prose
- "vs." always with period

## 14. Lessons Learned from This Copyedit

These patterns caused the most rework and should be watched for in future writing:

1. **Number spelling is the #1 source of errors.** Automated tools over-apply. Always verify: digits with units, digits for 10+, spelled out for 1-9 only.
2. **"Transformer" capitalization requires judgment.** Generic architecture = lowercase. Proper names (GPT = Generative Pre-trained Transformer) = uppercase. Sentence start = uppercase.
3. **"above"/"below" replacement needs word-order awareness.** "Preceding" goes BEFORE the noun. Agents often place it after.
4. **Pre-commit hooks can absorb or lose agent changes** via stash cycles. Always verify changes were committed by checking `git diff` after commit.
5. **OCR extraction from docx files introduces corruption** (spaces inserted in words). Filter corrupted entries before applying.
6. **Compound modifier hyphenation depends on position**: "compute-bound workload" (before noun) vs. "the system is compute bound" (predicate).
7. **Section headings drive the auto-generated TOC.** Front matter docx tracked changes that "look like TOC edits" actually require fixing the underlying H2/H3/H4 in the chapter QMD. The TOC will regenerate from the corrected headings.
8. **Heading case is level-dependent.** H1/H2 use headline style (capitalize principal words); H3 and below use sentence style (first word + proper nouns only). Pre-CE headings often used title case at H3+ — these need to be lowercased when edited.
9. **"Acknowledgments" not "Acknowledgements"** — American spelling per Webster's 11th. The QMD filename can stay as `acknowledgements.qmd` (path not rendered); only the H1 heading text needs to change.

## 15. Compound Prefix Close-Up Rule (CMS 17 + Webster's 11)

Close up compounds formed with common prefixes — `multi-`, `pre-`, `semi-`, `anti-`, `non-`, `re-` — unless the result is ambiguous, hard to read, or precedes a proper noun/acronym. The copy editor explicitly applied this rule to section headings during round 1.

**Closed-up forms (apply consistently):**

| Hyphenated (wrong) | Closed up (correct) |
|---|---|
| Multi-Layer | Multilayer |
| Multi-Chip | Multichip |
| Multi-Server | Multiserver |
| Multi-Model | Multimodel |
| Multi-Stream | Multistream |
| Multi-Scale | Multiscale |
| Pre-Training | Pretraining |
| Pre-Deployment | Predeployment |
| Pre-Learning | Prelearning |
| Pre-trained | Pretrained |
| Semi-Supervised | Semisupervised |
| Anti-Pattern | Antipattern |
| Non-Zero | Nonzero |

**Keep the hyphen when:**
- Before an acronym or proper noun: `Multi-GPU`, `Pre-CUDA`, `Multi-NUMA`
- Before a number or symbol: `pre-2010`, `multi-$100$`
- When close-up would be unreadable or ambiguous (`re-cover` vs. `recover`, `pre-eminent` if needed for clarity)

**Why this matters:** The pre-CE manuscript was inconsistent — some chapters used `Multi-Chip`, others `Multichip`. Consistency now matches CMS 17 §7.89 and the Webster's 11 entries the copy editor referenced. When introducing new compounds in vol2 or future writing, default to the closed-up form.

## 16. Alt-Text Style Compliance

Alt-text strings live in `fig-alt="..."` attributes and are easy to forget during prose passes — but they ARE body prose for screen readers. The same style rules apply:

- **Concept terms lowercase**: "machine learning" not "Machine Learning"; "memory wall" not "Memory Wall"; "data-centric AI" not "Data-Centric AI"
- **Hyphenation**: same close-up rules as section headings (`general-purpose`, `data-centric`, `pretraining`)
- **vs.** with period
- **Numbers**: spell out 1-9, digits for 10+ — but units always digits
- **Inline Python**: `` `{python} val_str` `` references should be preserved if present in source; if the CE-supplied alt-text omits them, prefer the source version

The pass 7 alt-text update (round 1, June) missed 64 of 212 entries due to OCR corruption in the docx extraction; these were repaired and applied 2026-04-06. The applied versions correctly use lowercase concept terms — when adding NEW figures, match this style from the start.

## 17. OCR Corruption Patterns from docx Extraction

When extracting tracked changes or alt-text from MS Word `.docx` files via `python-docx` or XML parsing, OCR-rendered text frequently has this corruption pattern:

```
"  m achine  l earning  " instead of "machine learning"
"  e dge  AI"             instead of "edge AI"
"( a ctive  l earning )"  instead of "(active learning)"
```

The marker is **double-space + single letter + space + rest of word**. A safer fix than naive regex: require the double-space marker before splitting (single-space "a x" is real English like "a moment").

Detector: `re.compile(r'(  |\( )[a-z] [a-z]{2,}')`

Fixer (single-letter prefix only):
```python
re.sub(r'(  )([a-z]) ([a-z]{2,})', r'\1\2\3', text)
re.sub(r'(\( )([a-z]) ([a-z]{2,})', r'\1\2\3', text)
```

Residual 2-char prefix splits (`Qu ality`, `ef ficiency`, `Pre trained`) need targeted manual patches. Don't trust an automated fix that drops to `\b[a-z] [a-z]{2,}` — it will join real English words ("at the" → "atthe").
