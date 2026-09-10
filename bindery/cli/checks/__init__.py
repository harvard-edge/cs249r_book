"""
Binder-native check implementations.

Check logic that powers ``./binder check <group> --scope …`` lives here
as ordinary Python modules. ``bindery/cli/commands/validate.py`` imports from
this package and converts results to ``ValidationIssue`` records.

Temporary standalone shims may import from ``cli.checks`` during migration, but
Binder must not depend on scripts under ``bindery/tools/`` for core checks.

See ``bindery/cli/README.md`` → "Check implementation layout".
"""
