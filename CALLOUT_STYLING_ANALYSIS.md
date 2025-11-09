# Custom Callout Styling Architecture Analysis

**Branch**: `refactor/consolidate-callout-styles`  
**Date**: 2025-11-09  
**Purpose**: Document current architecture and propose consolidation for easier maintenance

---

## Current Architecture (As-Built)

### File Structure

**4 Files Handle Callout Styling:**

1. **`quarto/_extensions/mlsysbook-ext/custom-numbered-blocks/style/foldbox.css`** (33 refs)
   - **Purpose**: Core extension styles
   - **Contents**: 
     - Color variables (--border-color, --background-color) for all callout types
     - Base layout and structure
     - Icon positioning and sizing
     - Light mode defaults
     - Dark mode overrides (lines 310-370)
   - **Loaded**: Always (by extension)
   - **Role**: PRIMARY source of truth

2. **`quarto/assets/styles/style.scss`** (52 refs - MOST!)
   - **Purpose**: Defensive overrides to prevent Quarto interference
   - **Contents**:
     - Exclusions from general `.callout` styling (removes box-shadow, borders)
     - Left-alignment rules for content/lists/summaries
     - Empty paragraph handling for definition/colab
   - **Loaded**: Compiled into BOTH light AND dark themes
   - **Role**: DEFENSIVE - neutralizes Quarto's default callout styles

3. **`quarto/assets/styles/dark-mode.scss`** (36 refs)
   - **Purpose**: Dark mode color overrides
   - **Contents**:
     - Text colors for dark backgrounds (#e6e6e6)
     - Border colors for dark mode (#454d55)
     - Summary text colors (#f0f0f0)
   - **Loaded**: Compiled into ONLY dark theme
   - **Role**: DUPLICATION - repeats foldbox.css dark mode section
   - **⚠️ ISSUE**: `callout-colab` is MISSING from dark-mode.scss!

4. **`quarto/assets/styles/epub.css`** (31 refs)
   - **Purpose**: EPUB-specific fallbacks
   - **Contents**: 
     - Styles for plain `<div>` rendering (when extension disabled)
     - Includes `::before` pseudo-elements for titles
     - Fallback for all callout types including colab
   - **Loaded**: Only in EPUB builds (specified in `_quarto-epub.yml`)
   - **Role**: FALLBACK for non-extension rendering

---

## Build System Integration

###HTML Builds

```yaml
# _quarto-html.yml
theme:
  light:
    - default
    - assets/styles/style.scss          # ← Compiled in
  dark:
    - default
    - assets/styles/style.scss          # ← Compiled in
    - assets/styles/dark-mode.scss      # ← Compiled in
```

**Result**: 
- `foldbox.css` loaded directly from extension
- `style.scss` and `dark-mode.scss` compiled into theme CSS
- Dark mode activated via `@media (prefers-color-scheme: dark)`

### EPUB Builds

```yaml
# _quarto-epub.yml
css: "assets/styles/epub.css"
```

**Result**:
- ONLY `epub.css` is used
- `foldbox.css` MAY be included by extension (needs verification)
- No dark mode support (EPUB readers handle themes)

---

## Problems Identified

### 1. **Duplication**
Dark mode styles exist in BOTH:
- `foldbox.css` (lines 310-370)
- `dark-mode.scss` (lines 715-750)

**Example**:
```css
/* foldbox.css */
@media (prefers-color-scheme: dark) {
  details.callout-definition {
    --text-color: #e6e6e6;
    --background-color: rgba(27, 79, 114, 0.12);
  }
}

/* dark-mode.scss */
details.callout-definition {
  --text-color: #e6e6e6 !important;
  border-color: #454d55 !important;
}
```

### 2. **Inconsistency**
`callout-colab` is:
- ✅ In `foldbox.css` dark mode section
- ✅ In `style.scss` exclusion rules
- ✅ In `epub.css` fallbacks
- ❌ MISSING from `dark-mode.scss`

### 3. **Scattered Logic**
- Structural styles → `foldbox.css`
- Exclusion rules → `style.scss`
- Dark mode → BOTH `foldbox.css` AND `dark-mode.scss`
- EPUB fallbacks → `epub.css`

### 4. **Unclear Separation**
Not obvious which file handles what without deep investigation.

---

## Recommended Consolidation (Option A)

### **Extension-First Architecture**

**Principle**: The custom-numbered-blocks extension should be self-contained.

```
foldbox.css          → ALL callout styles (light + dark, structure, colors)
style.scss           → ONLY minimal Quarto interference prevention
epub.css             → ONLY EPUB fallbacks (extension disabled)
dark-mode.scss       → REMOVE callout-specific rules (handled by foldbox.css)
```

### Detailed Changes

#### 1. `foldbox.css` - Keep As-Is (Self-Contained) ✅
- Already contains light mode colors
- Already contains dark mode section (`@media`)
- Already handles all structural styling
- **Action**: KEEP - no changes needed

#### 2. `style.scss` - Minimal Exclusions Only
```scss
/* ONLY exclude custom foldbox callouts from Quarto's default styling */
.callout.callout-quiz-question,
.callout.callout-quiz-answer,
.callout.callout-definition,
.callout.callout-example,
.callout.callout-colab {
  margin: 0 !important;
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}

/* Left-align content (Quarto defaults to center for some elements) */
.callout.callout-quiz-question div,
.callout.callout-quiz-answer div,
.callout.callout-definition div,
.callout.callout-example div,
.callout.callout-colab div {
  text-align: left !important;
}

/* Hide empty paragraphs generated by extension */
.callout-definition > p:empty,
details.callout-definition p:empty,
details.callout-definition > div > p:empty,
.callout-colab > p:empty,
details.callout-colab p:empty,
details.callout-colab > div > p:empty {
  display: none !important;
}
```

**Removed from style.scss**:
- All `details.callout-*` styling (let foldbox.css handle)
- List alignment (let foldbox.css handle)
- Summary alignment (let foldbox.css handle)

#### 3. `dark-mode.scss` - Remove ALL Callout Rules
```scss
/* REMOVE THESE (already in foldbox.css): */
details.callout-definition,
details.callout-example,
details.callout-quiz-question,
... etc ...
```

**Rationale**: `foldbox.css` already has a `@media (prefers-color-scheme: dark)` section that handles all dark mode styling for callouts.

#### 4. `epub.css` - Keep As-Is (Fallback) ✅
- Needed for when extension is disabled
- **Action**: KEEP - no changes needed

---

## Alternative: Add Missing colab to dark-mode.scss (Option B)

**IF** we keep the current architecture, then we must:

```scss
/* dark-mode.scss - ADD MISSING */
details.callout-colab {
  --text-color: #e6e6e6 !important;
  border-color: #454d55 !important;
}

details.callout-colab summary,
details.callout-colab summary strong,
details.callout-colab > summary {
  color: #f0f0f0 !important;
}
```

**But this still leaves duplication problem unsolved.**

---

## Testing Plan

### Before Changes
1. ✅ Build HTML intro: `./binder html intro`
2. ✅ Build EPUB intro: `./binder epub intro`
3. **TODO**: Test light mode callouts (definition, example, quiz, colab)
4. **TODO**: Test dark mode callouts (toggle dark mode in browser)
5. **TODO**: Test EPUB rendering

### After Changes
1. Repeat all above tests
2. Verify NO visual differences
3. Confirm dark mode still works
4. Confirm EPUB still renders correctly

---

## Recommendation

**Proceed with Option A (Extension-First):**

**Pros**:
- Single source of truth for callout styling
- No duplication
- Easier to debug (one file for structure, one for overrides)
- Self-contained extension
- Cleaner separation of concerns

**Cons**:
- Requires careful refactoring
- Must test thoroughly to avoid breaking anything

**Next Steps**:
1. ✅ Create branch: `refactor/consolidate-callout-styles`
2. ✅ Build test outputs (HTML + EPUB)
3. ⏳ Test current dark mode functionality
4. ⏳ Document expected behavior
5. ⏳ Make changes one file at a time
6. ⏳ Test after each change
7. ⏳ Commit when working

---

## Notes

- **DON'T BREAK ANYTHING**: Everything must work exactly as before
- **Test incrementally**: Change one file, test, commit
- **Keep git history clean**: Small, focused commits
- **Document decisions**: Update this file as we go


