# MLSysim Workbench (VS Code Extension)

The **MLSysim Workbench** is the official Visual Studio Code extension for the [MLSys·im analytical modeling framework](https://mlsysbook.ai/mlsysim). It provides a deeply integrated, interactive UI for exploring the Silicon Zoo, running evaluations, and searching the design space without leaving your editor.

## Features

- **Zoo Explorer:** Browse the built-in `Hardware` and `Models` registries directly from the sidebar.
- **Quick Evaluation:** Right-click any hardware or model in the sidebar to run an instant `mlsysim eval` directly in the integrated terminal.
- **YAML Scenario Integration:** Right-click any `.yaml` configuration file to evaluate, optimize, or export schemas.
- **Test Runner:** Run the full MLSys·im pytest suite or individual test files with one click.
- **Action History:** Re-run recent evaluations and optimizations from the "Recent Runs" pane.

## Requirements

You must have the `mlsysim` Python package installed in your active environment.

```bash
pip install mlsysim
```

If you are using a virtual environment (`venv`, `conda`, etc.), make sure VS Code has the correct Python interpreter selected, or explicitly configure the path in settings.

## Extension Settings

This extension contributes the following settings:

* `mlsysim.pythonPath`: Path to the Python interpreter (default: `python3`).
* `mlsysim.defaultPrecision`: Default precision used for Quick Evals (e.g., `fp16`).
* `mlsysim.defaultBatchSize`: Default batch size for Quick Evals (default: `1`).
* `mlsysim.defaultEfficiency`: Default MFU efficiency for Quick Evals (default: `0.5`).
* `mlsysim.outputFormat`: The format for CLI output (`text`, `json`, `markdown`).

## Building & Installing Locally

To package and install the extension manually:

1. Install `vsce` globally: `npm install -g @vscode/vsce`
2. Run `npm install` inside the `vscode-ext` directory.
3. Package the extension: `vsce package`
4. Install the generated `.vsix` file in VS Code:
   * View -> Command Palette -> `Extensions: Install from VSIX...`

---
*Part of the Machine Learning Systems textbook ecosystem.*
