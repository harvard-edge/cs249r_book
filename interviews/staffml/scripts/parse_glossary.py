#!/usr/bin/env python3
"""Parse MLSysBook glossary QMD files and produce a JSON glossary
for the StaffML interview platform's acronym hover tooltip feature.

Source files:
  - vol1: book/quarto/contents/vol1/backmatter/glossary/glossary.qmd
  - vol2: book/quarto/contents/vol2/backmatter/glossary/glossary.qmd

Output:
  interviews/staffml/src/data/glossary.json
"""

import json
import re
import sys
from pathlib import Path

# Resolve paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VOL1_GLOSSARY = REPO_ROOT / "book/quarto/contents/vol1/backmatter/glossary/glossary.qmd"
VOL2_GLOSSARY = REPO_ROOT / "book/quarto/contents/vol2/backmatter/glossary/glossary.qmd"
OUTPUT_PATH = REPO_ROOT / "interviews/staffml/src/data/glossary.json"


def parse_glossary(filepath: Path) -> dict[str, dict]:
    """Parse a QMD glossary file and return {lowercase_term: {display, definition}}."""
    text = filepath.read_text(encoding="utf-8")
    entries: dict[str, dict] = {}

    # Pattern: **term name**\n: definition
    # The term is wrapped in ** ... ** on its own line.
    # The definition starts with ": " on the next line (may span multiple lines
    # until the next blank line or next **term**).
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match a term line: **something**
        term_match = re.match(r"^\*\*(.+?)\*\*\s*$", line)
        if term_match:
            display = term_match.group(1).strip()

            # Collect definition lines starting with ": " on the next line
            definition_parts = []
            j = i + 1
            while j < len(lines):
                defline = lines[j]
                stripped = defline.strip()

                if j == i + 1:
                    # First line after term must start with ": "
                    if stripped.startswith(": "):
                        definition_parts.append(stripped[2:].strip())
                    else:
                        break
                else:
                    # Continuation lines: stop at blank line, next term, or
                    # section header
                    if stripped == "":
                        break
                    if re.match(r"^\*\*(.+?)\*\*\s*$", stripped):
                        break
                    if stripped.startswith("## "):
                        break
                    if stripped.startswith("```"):
                        break
                    # Continuation of the definition
                    definition_parts.append(stripped)
                j += 1

            if definition_parts:
                definition = " ".join(definition_parts)
                term_key = display.lower()
                entries[term_key] = {
                    "display": display,
                    "definition": definition,
                }
            i = j
        else:
            i += 1

    return entries


def detect_acronym(display: str, definition: str) -> str | None:
    """Detect if a term is an acronym and return its expansion.

    Strategies:
    1. The term contains a parenthetical acronym like
       "application-specific integrated circuit (ASIC)".
    2. The term itself is all-caps (or all-caps with digits/hyphens), and the
       definition starts with the expansion as capitalized words matching the
       letters.
    3. The definition contains a parenthetical with the term acronym, e.g.,
       "Graphics Processing Unit (GPU)".
    4. The definition starts with "Expansion, a/an..." or "Expansion that..."
       where the first-letter matching works.
    """
    term_clean = display.strip()

    # Strategy 1: Term has a parenthetical expansion like
    # "application-specific integrated circuit (ASIC)"
    # or "ZeRO (Zero Redundancy Optimizer)"
    paren_in_term = re.match(
        r"^(.+?)\s*\(([A-Za-z][A-Za-z0-9/.-]+)\)\s*$", term_clean
    )
    if paren_in_term:
        part1 = paren_in_term.group(1).strip()
        part2 = paren_in_term.group(2).strip()
        # Determine which part is the acronym and which is the expansion.
        # If part2 is all-caps (the acronym), expansion is part1.
        # If part1 is all-caps (the acronym), expansion is part2.
        alpha2 = re.sub(r"[^a-zA-Z]", "", part2)
        alpha1 = re.sub(r"[^a-zA-Z]", "", part1)
        if len(alpha2) >= 2 and alpha2 == alpha2.upper():
            return part1
        elif len(alpha1) >= 2 and alpha1 == alpha1.upper():
            return part2
        else:
            # e.g., "built-in self-test (bist)" -- lowercase acronym
            if len(part2) <= 6 and len(part1) > len(part2):
                return part1

    # Determine if the term itself is an all-caps acronym
    alpha_chars = re.sub(r"[^a-zA-Z]", "", term_clean)
    is_allcaps = (
        len(alpha_chars) >= 2
        and alpha_chars == alpha_chars.upper()
        and not re.search(r"[a-z]", alpha_chars)
    )

    # Exclude numeric format designators (FP16, FP32, FP8, BF16, INT8, etc.)
    # These are format names, not acronyms with expansions.
    if is_allcaps and re.match(
        r"^(FP|BF|INT|UINT)\d+$", term_clean, re.IGNORECASE
    ):
        is_allcaps = False

    # Exclude model names with version numbers (GPT-2, GPT-4, etc.)
    if re.match(r"^[A-Z]+-\d+$", term_clean):
        is_allcaps = False

    if is_allcaps:
        # Strategy 2: Try to find the expansion at the start of the definition.
        # Pattern: "Word Word Word. rest..." or "Word Word Word, rest..."
        expansion = _match_expansion_from_start(alpha_chars, definition)
        if expansion:
            # Validate: the expansion should be at least 2 words for a 2+ letter
            # acronym, and should contain mostly capitalized first letters
            word_count = len(expansion.split())
            if word_count >= 2 or len(alpha_chars) <= 2:
                return expansion

        # Strategy 3: "Expansion (ACRONYM)" inside definition
        paren_match = re.search(
            r"^(.+?)\s*\(" + re.escape(term_clean) + r"\)",
            definition,
            re.IGNORECASE,
        )
        if paren_match:
            return paren_match.group(1).strip().rstrip(",").strip()

        # Strategy 4: "Full Name, a/an/that/which..." pattern
        # e.g., "Long Short-Term Memory, a type of recurrent..."
        # e.g., "Open Neural Network Exchange, a standardized format..."
        comma_match = re.match(r"^([^,]+),\s+(?:a|an|the|that|which)\b", definition)
        if comma_match:
            candidate = comma_match.group(1).strip()
            # First try exact letter matching
            expansion = _match_expansion_exact(alpha_chars, candidate)
            if expansion:
                return expansion
            # For some terms the comma-delimited phrase IS the expansion
            # even if first-letter matching is imperfect (e.g., SHAP ->
            # "SHapley Additive exPlanations" where letters come from
            # unusual positions). Accept if the candidate looks like a
            # plausible proper-noun expansion (mostly capitalized words,
            # length >= 2 words).
            candidate_words = candidate.split()
            cap_count = sum(1 for w in candidate_words if w[0].isupper())
            if len(candidate_words) >= 2 and cap_count >= len(candidate_words) // 2:
                return candidate

        # Strategy 5: first sentence/clause before period
        sentence_match = re.match(r"^([^.]+)\.", definition)
        if sentence_match:
            first_sentence = sentence_match.group(1).strip()
            expansion = _match_expansion_exact(alpha_chars, first_sentence)
            if expansion:
                return expansion

    # Manual well-known acronyms that have unusual definition patterns
    _manual_acronyms: dict[str, str] = {
        "auc": "Area Under the ROC Curve",
        "jax": None,  # Not an acronym; it's a library name
        "lapack": "Linear Algebra Package",
        "linpack": "Linear Algebra Package (Benchmark)",
        "spec cpu": "Standard Performance Evaluation Corporation CPU",
    }
    manual = _manual_acronyms.get(term_clean.lower())
    if manual is not None:
        return manual if manual else None

    return None


def _match_expansion_from_start(
    acronym_letters: str, definition: str
) -> str | None:
    """Try to match an expansion at the start of the definition.

    For "GPU" with definition "Graphics Processing Unit. A massively...",
    return "Graphics Processing Unit".
    """
    # Split definition into words, trying to match acronym letters
    # to the first letter of each significant word.
    words = definition.split()
    if not words:
        return None

    target = acronym_letters.upper()
    matched_words = []
    target_idx = 0

    for word in words:
        if target_idx >= len(target):
            break
        # Clean word of trailing punctuation for matching
        clean = re.sub(r"[^a-zA-Z]", "", word)
        if not clean:
            matched_words.append(word)
            continue

        if clean[0].upper() == target[target_idx]:
            matched_words.append(word)
            target_idx += 1
        else:
            # If we haven't matched all letters yet and hit a mismatch,
            # check if this is a small word we should skip (articles,
            # prepositions, conjunctions)
            skip_words = {
                "a",
                "an",
                "the",
                "of",
                "for",
                "in",
                "on",
                "to",
                "and",
                "or",
                "with",
                "by",
                "as",
                "at",
                "from",
                "per",
            }
            if clean.lower() in skip_words:
                matched_words.append(word)
                continue
            else:
                break

    if target_idx == len(target) and matched_words:
        # Clean trailing punctuation from the last word
        result = " ".join(matched_words)
        result = re.sub(r"[.,;:!?]+$", "", result)
        return result

    return None


def _match_expansion_exact(acronym_letters: str, text: str) -> str | None:
    """Try exact first-letter matching against text words."""
    words = text.split()
    target = acronym_letters.upper()

    skip_words = {
        "a",
        "an",
        "the",
        "of",
        "for",
        "in",
        "on",
        "to",
        "and",
        "or",
        "with",
        "by",
        "as",
        "at",
        "from",
        "per",
    }

    matched_words = []
    target_idx = 0

    for word in words:
        if target_idx >= len(target):
            break
        clean = re.sub(r"[^a-zA-Z]", "", word)
        if not clean:
            continue
        if clean.lower() in skip_words and target_idx > 0:
            matched_words.append(word)
            continue
        if clean[0].upper() == target[target_idx]:
            matched_words.append(word)
            target_idx += 1
        else:
            break

    if target_idx == len(target) and matched_words:
        result = " ".join(matched_words)
        result = re.sub(r"[.,;:!?]+$", "", result)
        return result

    return None


def main():
    # Parse both volumes
    vol1_entries = parse_glossary(VOL1_GLOSSARY)
    vol2_entries = parse_glossary(VOL2_GLOSSARY)

    print(f"Vol1 terms parsed: {len(vol1_entries)}")
    print(f"Vol2 terms parsed: {len(vol2_entries)}")

    # Merge: vol2 takes precedence for duplicate terms
    merged = {}
    merged.update(vol1_entries)

    dupes = 0
    for key, val in vol2_entries.items():
        if key in merged:
            dupes += 1
        merged[key] = val

    print(f"Duplicates (vol2 preferred): {dupes}")
    print(f"Merged unique terms: {len(merged)}")

    # Build output list with acronym detection
    output = []
    acronym_count = 0

    for term_key in sorted(merged.keys()):
        entry = merged[term_key]
        display = entry["display"]
        definition = entry["definition"]

        acronym = detect_acronym(display, definition)
        if acronym:
            acronym_count += 1

        output.append(
            {
                "term": term_key,
                "display": display,
                "definition": definition,
                "acronym": acronym,
            }
        )

    print(f"Acronyms detected: {acronym_count}")
    print(f"Total output entries: {len(output)}")

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nWritten to: {OUTPUT_PATH}")

    # Print some sample acronyms for verification
    print("\nSample acronyms detected:")
    acronym_entries = [e for e in output if e["acronym"]]
    for entry in acronym_entries[:20]:
        print(f"  {entry['display']:30s} -> {entry['acronym']}")

    # Print terms with parenthetical expansions
    print("\nTerms with parenthetical expansions in display name:")
    paren_terms = [
        e for e in output if "(" in e["display"] and e["acronym"]
    ]
    for entry in paren_terms[:10]:
        print(f"  {entry['display']:50s} -> {entry['acronym']}")


if __name__ == "__main__":
    main()
