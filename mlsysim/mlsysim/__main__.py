"""Entry point for `python -m mlsysim`."""

try:
    from mlsysim.cli.main import app
except ImportError:
    import sys
    print(
        "Unable to import the mlsysim CLI.\n"
        "Install or repair the package with: pip install mlsysim",
        file=sys.stderr,
    )
    sys.exit(1)

if __name__ == "__main__":
    app()
