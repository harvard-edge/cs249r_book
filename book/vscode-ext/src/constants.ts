import { PrecommitHook, ActionDef } from './types';

export const PRECOMMIT_HOOKS: PrecommitHook[] = [
  { id: 'book-check-duplicate-labels', label: 'Check Duplicate Labels', command: 'pre-commit run book-check-duplicate-labels --all-files' },
  { id: 'book-check-unreferenced-labels', label: 'Check Unreferenced Labels', command: 'pre-commit run book-check-unreferenced-labels --all-files' },
  { id: 'book-validate-citations', label: 'Check Citations', command: 'pre-commit run book-validate-citations --all-files' },
  { id: 'book-check-references', label: 'Check References', command: 'pre-commit run book-check-references --all-files' },
  { id: 'book-validate-image-references', label: 'Check Image References', command: 'pre-commit run book-validate-image-references --all-files' },
  { id: 'book-validate-footnotes', label: 'Check Footnotes', command: 'pre-commit run book-validate-footnotes --all-files' },
  { id: 'book-check-forbidden-footnotes', label: 'Check Forbidden Footnotes', command: 'pre-commit run book-check-forbidden-footnotes --all-files' },
  { id: 'book-check-table-formatting', label: 'Check Table Formatting', command: 'pre-commit run book-check-table-formatting --all-files' },
  { id: 'book-check-unclosed-divs', label: 'Check Unclosed Divs', command: 'pre-commit run book-check-unclosed-divs --all-files' },
  { id: 'book-check-purpose-unnumbered', label: 'Check Purpose Unnumbered', command: 'pre-commit run book-check-purpose-unnumbered --all-files' },
  { id: 'book-format-python', label: 'Format Python in QMD', command: 'pre-commit run book-format-python --all-files' },
  { id: 'book-collapse-blank-lines', label: 'Collapse Blank Lines', command: 'pre-commit run book-collapse-blank-lines --all-files' },
  { id: 'codespell', label: 'Codespell', command: 'pre-commit run codespell --all-files' },
  { id: 'mdformat', label: 'Format Markdown', command: 'pre-commit run mdformat --all-files' },
  { id: 'bibtex-tidy', label: 'Tidy BibTeX', command: 'pre-commit run bibtex-tidy --all-files' },
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
