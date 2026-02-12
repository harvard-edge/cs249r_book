import { PrecommitHook, ActionDef, PrecommitFileFixer } from './types';

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
];

export const PRECOMMIT_FIXER_HOOKS: PrecommitHook[] = [
  { id: 'trailing-whitespace', label: 'Global: Trim Trailing Whitespace', command: 'pre-commit run trailing-whitespace --all-files' },
  { id: 'end-of-file-fixer', label: 'Global: Fix End-of-File Newlines', command: 'pre-commit run end-of-file-fixer --all-files' },
  { id: 'mdformat', label: 'Book: Format Quarto Markdown', command: 'pre-commit run mdformat --all-files' },
  { id: 'bibtex-tidy', label: 'Book: Tidy BibTeX Files', command: 'pre-commit run bibtex-tidy --all-files' },
  { id: 'book-format-python', label: 'Book: Format Python in QMD', command: 'pre-commit run book-format-python --all-files' },
  { id: 'book-collapse-blank-lines', label: 'Book: Collapse Extra Blank Lines', command: 'pre-commit run book-collapse-blank-lines --all-files' },
  { id: 'book-check-list-formatting', label: 'Book: Fix List Formatting', command: 'pre-commit run book-check-list-formatting --all-files' },
  { id: 'book-fix-bullet-spacing', label: 'Book: Fix Bullet Spacing', command: 'pre-commit run book-fix-bullet-spacing --all-files' },
  { id: 'book-prettify-pipe-tables', label: 'Book: Prettify Pipe Tables (All Files)', command: 'pre-commit run book-prettify-pipe-tables --all-files' },
  { id: 'book-cleanup-artifacts', label: 'Book: Cleanup Build Artifacts', command: 'pre-commit run book-cleanup-artifacts --all-files' },
];

export const PRECOMMIT_QMD_FILE_FIXERS: PrecommitFileFixer[] = [
  { id: 'mdformat-current-file', label: 'Format Quarto Markdown', hookId: 'mdformat' },
  { id: 'format-python-current-file', label: 'Format Python Code Blocks', hookId: 'book-format-python' },
  { id: 'collapse-blank-lines-current-file', label: 'Collapse Extra Blank Lines', hookId: 'book-collapse-blank-lines' },
  { id: 'fix-list-formatting-current-file', label: 'Fix List Formatting', hookId: 'book-check-list-formatting' },
  { id: 'fix-bullet-spacing-current-file', label: 'Fix Bullet Spacing', hookId: 'book-fix-bullet-spacing' },
  { id: 'prettify-pipe-tables-current-file', label: 'Prettify Pipe Tables', hookId: 'book-prettify-pipe-tables' },
];

export const PUBLISH_ACTIONS: ActionDef[] = [
  { id: 'mit-press-vol1', label: 'MIT Press Vol1', command: 'bash book/tools/scripts/publish/mit-press-release.sh --vol1', icon: 'rocket' },
  { id: 'mit-press-vol2', label: 'MIT Press Vol2', command: 'bash book/tools/scripts/publish/mit-press-release.sh --vol2', icon: 'rocket' },
  { id: 'mit-press-vol1-copyedit', label: 'MIT Press Vol1 (Copy-edit)', command: 'bash book/tools/scripts/publish/mit-press-release.sh --vol1 --copyedit', icon: 'rocket' },
  { id: 'extract-figures-vol1', label: 'Extract Figures Vol1', command: 'python3 book/tools/scripts/publish/extract_figures.py --vol 1', icon: 'file-media' },
];

export const MAINTENANCE_ACTIONS: ActionDef[] = [
  { id: 'clean', label: 'Clean Build Artifacts', command: './book/binder clean', icon: 'trash' },
  { id: 'doctor', label: 'Doctor (Health Check)', command: './book/binder doctor', icon: 'heart' },
  { id: 'glossary', label: 'Build Global Glossary', command: './book/binder maintain glossary build', icon: 'book' },
  { id: 'compress-images', label: 'Compress Images (Dry Run, All)', command: './book/binder maintain images compress --all --smart-compression', icon: 'file-media' },
  { id: 'repo-health', label: 'Repo Health Check', command: './book/binder maintain repo-health', icon: 'pulse' },
];

export const VALIDATE_ACTIONS: ActionDef[] = [
  { id: 'validate-all', label: 'Validate: All (Binder Native)', command: './book/binder validate all', icon: 'shield' },
  { id: 'validate-inline-python', label: 'Validate: Inline Python', command: './book/binder validate inline-python', icon: 'code' },
  { id: 'validate-refs', label: 'Validate: References in Raw/Code', command: './book/binder validate refs', icon: 'link' },
  { id: 'validate-citations', label: 'Validate: Citation Keys', command: './book/binder validate citations', icon: 'book' },
  { id: 'validate-duplicate-labels', label: 'Validate: Duplicate Labels', command: './book/binder validate duplicate-labels', icon: 'warning' },
  { id: 'validate-unreferenced-labels', label: 'Validate: Unreferenced Labels', command: './book/binder validate unreferenced-labels', icon: 'search' },
  { id: 'validate-inline-refs', label: 'Validate: Inline Refs', command: './book/binder validate inline-refs --check-patterns', icon: 'symbol-variable' },
];
