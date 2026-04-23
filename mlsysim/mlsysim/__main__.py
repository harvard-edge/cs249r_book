"""Entry point for `python -m mlsysim`."""

try:
    from mlsysim.cli.main import app
except ImportError:
    import sys
    print(
        "mlsysim CLI requires additional dependencies.\n"
        "Install with: pip install 'mlsysim[cli]'\n"
        "\nFor the Python API (no CLI), use: import mlsysim",
        file=sys.stderr,
    )
    sys.exit(1)

if __name__ == "__main__":
    app()
