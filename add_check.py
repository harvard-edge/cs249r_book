import re

with open('book/cli/commands/validate.py', 'r', encoding='utf-8') as f:
    content = f.read()

dispatch_insert = '            ("mitpress-spaced-slash", "_run_mitpress_spaced_slash"),\n'
if 'mitpress-spaced-slash' not in content:
    content = content.replace('("mitpress-spaced-emdash", "_run_mitpress_spaced_emdash"),\n',
                              '("mitpress-spaced-emdash", "_run_mitpress_spaced_emdash"),\n' + dispatch_insert)

func = '''
    def _run_mitpress_spaced_slash(self, root: Path) -> ValidationRunResult:
        """Flag spaced slashes (word / word) in prose — should be closed (word/word)."""
        start = time.time()
        files = self._qmd_files(root)
        issues: List[ValidationIssue] = []
        spaced_slash = re.compile(r"[a-zA-Z0-9] / [a-zA-Z0-9]")

        for file in files:
            lines = self._read_text(file).splitlines()
            in_code = False
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                if stripped.startswith("#|") or stripped.startswith("# "):
                    continue
                for m in spaced_slash.finditer(line):
                    context = line[max(0, m.start() - 5) : min(len(line), m.end() + 15)].strip()
                    issues.append(
                        ValidationIssue(
                            file=self._relative_file(file),
                            line=idx,
                            code="mitpress_spaced_slash",
                            message="Close up slash: word/word not word / word (§1)",
                            severity="warning",
                            context=context,
                        )
                    )

        return ValidationRunResult(
            name="mitpress-spaced-slash",
            description="No spaced slashes in prose — use word/word (MIT Press §1)",
            files_checked=len(files),
            issues=issues,
            elapsed_ms=int((time.time() - start) * 1000),
        )
'''

if '_run_mitpress_spaced_slash' not in content:
    content = content.replace('    def _run_mitpress_vs_period', func.lstrip('\n') + '\n    def _run_mitpress_vs_period')

with open('book/cli/commands/validate.py', 'w', encoding='utf-8') as f:
    f.write(content)

with open('.pre-commit-config.yaml', 'r', encoding='utf-8') as f:
    yaml_content = f.read()

yaml_hook = '''
      - id: mitpress-spaced-slash
        name: "MIT Press: No spaced slashes (word/word not word / word)"
        entry: ./book/binder check rendering --scope mitpress-spaced-slash
        language: system
        pass_filenames: false
        files: ^book/quarto/contents/.*\.qmd$
'''
if 'mitpress-spaced-slash' not in yaml_content:
    yaml_content = yaml_content.replace('      - id: mitpress-spaced-emdash\n        name: "MIT Press: No spaced em dashes (word—word not word — word)"\n        entry: ./book/binder check rendering --scope mitpress-spaced-emdash\n        language: system\n        pass_filenames: false\n        files: ^book/quarto/contents/.*\.qmd$',
                                        '      - id: mitpress-spaced-emdash\n        name: "MIT Press: No spaced em dashes (word—word not word — word)"\n        entry: ./book/binder check rendering --scope mitpress-spaced-emdash\n        language: system\n        pass_filenames: false\n        files: ^book/quarto/contents/.*\\.qmd$\n' + yaml_hook)

with open('.pre-commit-config.yaml', 'w', encoding='utf-8') as f:
    f.write(yaml_content)

