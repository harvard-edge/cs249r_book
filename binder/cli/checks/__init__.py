"""
Binder-native check implementations.

Check logic that powers ``./binder/binder check <group> --scope …`` lives here
as ordinary Python modules. ``binder/cli/commands/validate.py`` imports from
this package and converts results to ``ValidationIssue`` records.

Temporary standalone shims may import from ``cli.checks`` during migration, but
Binder must not depend on scripts under ``binder/tools/`` for core checks.

See ``binder/cli/README.md`` → "Check implementation layout".
"""
