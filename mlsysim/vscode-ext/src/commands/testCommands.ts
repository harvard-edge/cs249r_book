import * as vscode from 'vscode';
import { runInTerminal } from '../utils/terminal';

export function registerTestCommands(context: vscode.ExtensionContext, root: string): void {
  const pytest = `cd "${root}" && python3 -m pytest`;

  // Run all tests
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysim.runAllTests', () => {
      runInTerminal(`${pytest} mlsysim/tests/ -v`, root, 'All Tests');
    }),
  );

  // Run a specific test suite (quick pick)
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysim.runTestSuite', async () => {
      const suites = [
        { label: 'Engine Tests', file: 'test_engine.py' },
        { label: 'Hardware Tests', file: 'test_hardware.py' },
        { label: 'Solver Suite', file: 'test_solver_suite.py' },
        { label: 'Empirical Validation', file: 'test_empirical.py' },
        { label: 'SOTA Paradigm', file: 'test_sota.py' },
      ];

      const pick = await vscode.window.showQuickPick(suites, {
        placeHolder: 'Select test suite to run',
      });
      if (!pick) { return; }

      runInTerminal(`${pytest} mlsysim/tests/${pick.file} -v`, root, pick.label);
    }),
  );

  // Validate paper anchors
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysim.validateAnchors', () => {
      runInTerminal(
        `cd "${root}/mlsysim/paper" && python3 validate_anchors.py`,
        root,
        'Validate Paper Anchors',
      );
    }),
  );

  // Health check
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysim.healthCheck', () => {
      runInTerminal(`${pytest} mlsysim/tests/test_engine.py -v --tb=short`, root, 'Health Check');
    }),
  );
}
