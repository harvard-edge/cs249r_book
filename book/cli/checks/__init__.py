"""
Binder-native check implementations.

Check logic that powers ``./book/binder check <group> --scope …`` lives here
as ordinary Python modules. ``book/cli/commands/validate.py`` imports from
this package and converts results to ``ValidationIssue`` records.

Temporary standalone shims may import from ``cli.checks`` during migration, but
Binder must not depend on scripts under ``book/tools/`` for core checks.

See ``book/cli/README.md`` → "Check implementation layout".
"""
