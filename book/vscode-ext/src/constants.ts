import { PrecommitHook, ActionDef, PrecommitFileFixer } from './types';

// =============================================================================
// Pre-commit hooks (run via pre-commit CLI)
// =============================================================================

export const PRECOMMIT_CHECK_HOOKS: PrecommitHook[] = [
  { id: 'codespell', label: 'Global: Check Spelling', command: 'pre-commit run codespell --all-files' },
  { id: 'check-json', label: 'Global: Validate JSON Syntax', command: 'pre-commit run check-json --all-files' },
  { id: 'check-yaml', label: 'Global: Validate YAML Syntax', command: 'pre-commit run check-yaml --all-files' },
  { id: 'book-validate-json', label: 'Book: Validate JSON Files', command: 'pre-commit run book-validate-json --all-files' },
  { id: 'book-check-unreferenced-labels', label: 'Book: Check Unreferenced Labels', command: 'pre-commit run book-check-unreferenced-labels --all-files' },
  { id: 'book-check-duplicate-labels', label: 'Book: Check Duplicate Labels', command: 'pre-commit run book-check-duplicate-labels --all-files' },
  { id: 'book-validate-citations', label: 'Book: Validate Citation References', command: 'pre-commit run book-validate-citations --all-files' },
  { id: 'book-check-references', label: 'Book: Check Reference/Citation Issues', command: 'pre-commit run book-check-references --all-files' },
  { id: 'book-validate-footnotes', label: 'Book: Validate Footnote References', command: 'pre-commit run book-validate-footnotes --all-files' },
  { id: 'book-check-forbidden-footnotes', label: 'Book: Check Forbidden Footnotes', command: 'pre-commit run book-check-forbidden-footnotes --all-files' },
  { id: 'book-check-purpose-unnumbered', label: 'Book: Check Purpose Section Numbering', command: 'pre-commit run book-check-purpose-unnumbered --all-files' },
  { id: 'book-check-unclosed-divs', label: 'Book: Check Unclosed Div Fences (:::)', command: 'pre-commit run book-check-unclosed-divs --all-files' },
  { id: 'book-check-four-colon-space', label: 'Book: Check Malformed Nested Div Fences (::::)', command: 'pre-commit run book-check-four-colon-space --all-files' },
  { id: 'book-check-figure-completeness', label: 'Book: Check Figure Completeness', command: 'pre-commit run book-check-figure-completeness --all-files' },
  { id: 'book-check-figure-placement', label: 'Book: Check Figure/Table Placement', command: 'pre-commit run book-check-figure-placement --all-files' },
  { id: 'book-check-table-formatting', label: 'Book: Check Table Formatting', command: 'pre-commit run book-check-table-formatting --all-files' },
  { id: 'book-check-grid-tables', label: 'Book: Check Grid Tables', command: 'pre-commit run book-check-grid-tables --all-files' },
  { id: 'book-check-render-patterns', label: 'Book: Check Render Patterns', command: 'pre-commit run book-check-render-patterns --all-files' },
  { id: 'book-validate-dropcap', label: 'Book: Validate Dropcap Compatibility', command: 'pre-commit run book-validate-dropcap --all-files' },
  { id: 'book-mlsys-validate-inline', label: 'Book: MLSys Inline Validation', command: 'pre-commit run book-mlsys-validate-inline --all-files' },
  { id: 'book-mlsys-test-units', label: 'Book: MLSys Unit Checks', command: 'pre-commit run book-mlsys-test-units --all-files' },
  { id: 'book-check-index-placement', label: 'Book: Check Index Placement', command: 'pre-commit run book-check-index-placement --all-files' },
  { id: 'book-validate-part-keys', label: 'Book: Validate Part Keys', command: 'pre-commit run book-validate-part-keys --all-files' },
  { id: 'book-validate-images', label: 'Book: Validate Image Files', command: 'pre-commit run book-validate-images --all-files' },
  { id: 'book-validate-external-images', label: 'Book: Validate External Images', command: 'pre-commit run book-validate-external-images --all-files' },
  { id: 'book-validate-image-references', label: 'Book: Validate Image References', command: 'pre-commit run book-validate-image-references --all-files' },
  { id: 'book-check-heading-levels', label: 'Book: Check Heading Level Hierarchy', command: 'pre-commit run book-check-heading-levels --all-files' },
  { id: 'book-check-duplicate-words', label: 'Book: Check Duplicate Consecutive Words', command: 'pre-commit run book-check-duplicate-words --all-files' },
];

export const PRECOMMIT_FIXER_HOOKS: PrecommitHook[] = [
  { id: 'trailing-whitespace', label: 'Global: Trim Trailing Whitespace', command: 'pre-commit run trailing-whitespace --all-files' },
  { id: 'end-of-file-fixer', label: 'Global: Fix End-of-File Newlines', command: 'pre-commit run end-of-file-fixer --all-files' },
  { id: 'mdformat', label: 'Book: Format Quarto Markdown', command: 'pre-commit run mdformat --all-files' },
  { id: 'bibtex-tidy', label: 'Book: Tidy BibTeX Files', command: 'pre-commit run bibtex-tidy --all-files' },
  { id: 'book-format-python', label: 'Book: Format Python in QMD', command: 'pre-commit run book-format-python --all-files' },
  { id: 'book-collapse-blank-lines', label: 'Book: Collapse Extra Blank Lines', command: 'pre-commit run book-collapse-blank-lines --all-files' },
  { id: 'book-fix-list-spacing', label: 'Book: Fix List Spacing', command: 'pre-commit run book-fix-list-spacing --all-files' },
  { id: 'book-prettify-pipe-tables', label: 'Book: Prettify Pipe Tables (All Files)', command: 'pre-commit run book-prettify-pipe-tables --all-files' },
  { id: 'book-cleanup-artifacts', label: 'Book: Cleanup Build Artifacts', command: 'pre-commit run book-cleanup-artifacts --all-files' },
];

export const PRECOMMIT_QMD_FILE_FIXERS: PrecommitFileFixer[] = [
  { id: 'mdformat-current-file', label: 'Format Quarto Markdown', hookId: 'mdformat' },
  { id: 'format-python-current-file', label: 'Format Python Code Blocks', hookId: 'book-format-python' },
  { id: 'collapse-blank-lines-current-file', label: 'Collapse Extra Blank Lines', hookId: 'book-collapse-blank-lines' },
  { id: 'fix-list-spacing-current-file', label: 'Fix List Spacing', hookId: 'book-fix-list-spacing' },
  { id: 'prettify-pipe-tables-current-file', label: 'Prettify Pipe Tables', hookId: 'book-prettify-pipe-tables' },
];

// =============================================================================
// Publish actions (MIT Press build pipeline — shell scripts stay standalone)
// =============================================================================

export const PUBLISH_ACTIONS: ActionDef[] = [
  { id: 'mit-press-vol1', label: 'MIT Press Vol1', command: 'bash book/tools/scripts/publish/mit-press-release.sh --vol1', icon: 'rocket' },
  { id: 'mit-press-vol2', label: 'MIT Press Vol2', command: 'bash book/tools/scripts/publish/mit-press-release.sh --vol2', icon: 'rocket' },
  { id: 'mit-press-vol1-copyedit', label: 'MIT Press Vol1 (Copy-edit)', command: 'bash book/tools/scripts/publish/mit-press-release.sh --vol1 --copyedit', icon: 'rocket' },
  { id: 'extract-figures-vol1', label: 'Extract Figures Vol1', command: './book/binder info figures --vol1 --format markdown', icon: 'file-media' },
  { id: 'extract-figures-vol2', label: 'Extract Figures Vol2', command: './book/binder info figures --vol2 --format markdown', icon: 'file-media' },
];

// =============================================================================
// Maintenance actions (all via binder)
// =============================================================================

export const MAINTENANCE_ACTIONS: ActionDef[] = [
  { id: 'clean', label: 'Clean Build Artifacts', command: './book/binder clean', icon: 'trash' },
  { id: 'doctor', label: 'Doctor (Health Check)', command: './book/binder doctor', icon: 'heart' },
  { id: 'glossary', label: 'Build Global Glossary', command: './book/binder fix glossary build', icon: 'book' },
  { id: 'compress-images', label: 'Compress Images (Dry Run)', command: './book/binder fix images compress --all --smart-compression', icon: 'file-media' },
  { id: 'repo-health', label: 'Repo Health Check', command: './book/binder fix repo-health', icon: 'pulse' },
  { id: 'bib-list', label: 'Bib: List All', command: './book/binder bib list', icon: 'list-unordered' },
  { id: 'bib-clean-dry', label: 'Bib: Clean (Dry Run)', command: './book/binder bib clean --dry-run', icon: 'eye' },
  { id: 'bib-sync', label: 'Bib: Sync (Clean + Update)', command: './book/binder bib sync', icon: 'sync' },
];

// =============================================================================
// Binder check actions (fast, focused validation — all via binder)
// =============================================================================

export const CHECK_ACTIONS: ActionDef[] = [
  { id: 'check-all', label: 'Check: All', command: './book/binder check all', icon: 'shield' },
  { id: 'check-refs', label: 'Check: References', command: './book/binder check refs', icon: 'link' },
  { id: 'check-labels', label: 'Check: Labels', command: './book/binder check labels', icon: 'search' },
  { id: 'check-headers', label: 'Check: Headers', command: './book/binder check headers', icon: 'symbol-structure' },
  { id: 'check-footnotes', label: 'Check: Footnotes', command: './book/binder check footnotes', icon: 'note' },
  { id: 'check-figures', label: 'Check: Figures', command: './book/binder check figures', icon: 'file-media' },
  { id: 'check-rendering', label: 'Check: Rendering', command: './book/binder check rendering', icon: 'warning' },
  { id: 'check-images', label: 'Check: Images', command: './book/binder check images', icon: 'device-camera' },
  { id: 'check-json', label: 'Check: JSON', command: './book/binder check json', icon: 'json' },
  { id: 'check-units', label: 'Check: Units', command: './book/binder check units', icon: 'beaker' },
  { id: 'check-spelling', label: 'Check: Spelling', command: './book/binder check spelling', icon: 'whole-word' },
  { id: 'check-epub', label: 'Check: EPUB', command: './book/binder check epub', icon: 'book' },
  { id: 'check-sources', label: 'Check: Sources', command: './book/binder check sources', icon: 'references' },
];

// =============================================================================
// Binder info actions (book statistics and extraction — all via binder)
// =============================================================================

export const INFO_ACTIONS: ActionDef[] = [
  { id: 'info-stats', label: 'Info: Book Stats', command: './book/binder info stats', icon: 'graph' },
  { id: 'info-stats-vol1', label: 'Info: Stats Vol1', command: './book/binder info stats --vol1 --by-chapter', icon: 'graph' },
  { id: 'info-stats-vol2', label: 'Info: Stats Vol2', command: './book/binder info stats --vol2 --by-chapter', icon: 'graph' },
  { id: 'info-figures', label: 'Info: Figure List', command: './book/binder info figures', icon: 'file-media' },
  { id: 'info-concepts', label: 'Info: Key Concepts', command: './book/binder info concepts', icon: 'lightbulb' },
  { id: 'info-headers', label: 'Info: Section Headers', command: './book/binder info headers', icon: 'symbol-structure' },
  { id: 'info-acronyms', label: 'Info: Acronyms', command: './book/binder info acronyms', icon: 'text-size' },
];
