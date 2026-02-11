import * as vscode from 'vscode';
import { getRepoRoot } from './utils/workspace';
import { BuildTreeProvider } from './providers/buildTreeProvider';
import { DebugTreeProvider } from './providers/debugTreeProvider';
import { PrecommitTreeProvider } from './providers/precommitTreeProvider';
import { PublishTreeProvider } from './providers/publishTreeProvider';
import { ChapterNavigatorProvider } from './providers/chapterNavigatorProvider';
import { RunHistoryProvider } from './providers/runHistoryProvider';
import { QmdDiagnosticsManager } from './validation/qmdDiagnostics';
import { QmdChunkHighlighter } from './providers/qmdChunkHighlighter';
import { renameLabelAcrossWorkspace } from './utils/labelRename';
import { registerBuildCommands } from './commands/buildCommands';
import { registerDebugCommands } from './commands/debugCommands';
import { registerPrecommitCommands } from './commands/precommitCommands';
import { registerPublishCommands } from './commands/publishCommands';
import { registerContextMenuCommands } from './commands/contextMenuCommands';
import {
  initializeRunManager,
  rerunLastCommand,
  revealRunTerminal,
  showLastFailureDetails,
  setExecutionModeInteractively,
} from './utils/terminal';

type NavigatorPresetId = 'all' | 'writing' | 'reference' | 'structure';

interface NavigatorPreset {
  label: string;
  id: NavigatorPresetId;
  kinds: string[];
}

const NAVIGATOR_PRESETS: NavigatorPreset[] = [
  { label: 'Show All', id: 'all', kinds: ['figures', 'tables', 'listings', 'divs', 'citations'] },
  { label: 'Writing Focus', id: 'writing', kinds: ['figures', 'tables', 'listings'] },
  { label: 'Reference Focus', id: 'reference', kinds: ['figures', 'tables', 'citations'] },
  { label: 'Structure Focus', id: 'structure', kinds: ['divs', 'listings'] },
];

export function activate(context: vscode.ExtensionContext): void {
  const root = getRepoRoot();
  if (!root) {
    vscode.window.showWarningMessage('MLSysBook Workbench: could not find repo root (book/binder not found).');
    return;
  }
  initializeRunManager(context);

  // Tree view providers
  const buildProvider = new BuildTreeProvider(root);
  const debugProvider = new DebugTreeProvider();
  const precommitProvider = new PrecommitTreeProvider();
  const publishProvider = new PublishTreeProvider();
  const navigatorProvider = new ChapterNavigatorProvider();
  const runHistoryProvider = new RunHistoryProvider();
  const diagnosticsManager = new QmdDiagnosticsManager();
  const chunkHighlighter = new QmdChunkHighlighter();
  const config = vscode.workspace.getConfiguration('mlsysbook');
  const defaultPreset = config.get<NavigatorPresetId>('defaultNavigatorPreset', 'all');
  const preset = NAVIGATOR_PRESETS.find(p => p.id === defaultPreset) ?? NAVIGATOR_PRESETS[0];
  void config.update('navigatorVisibleEntryKinds', preset.kinds, vscode.ConfigurationTarget.Workspace);
  navigatorProvider.refreshFromEditor(vscode.window.activeTextEditor);
  diagnosticsManager.start();
  chunkHighlighter.start();

  context.subscriptions.push(
    vscode.window.createTreeView('mlsysbook.build', { treeDataProvider: buildProvider }),
    vscode.window.createTreeView('mlsysbook.debug', { treeDataProvider: debugProvider }),
    vscode.window.createTreeView('mlsysbook.runs', { treeDataProvider: runHistoryProvider }),
    vscode.window.createTreeView('mlsysbook.precommit', { treeDataProvider: precommitProvider }),
    vscode.window.createTreeView('mlsysbook.publish', { treeDataProvider: publishProvider }),
    runHistoryProvider,
    diagnosticsManager,
    chunkHighlighter,
  );
  const navigatorTreeView = vscode.window.createTreeView('mlsysbook.navigator', { treeDataProvider: navigatorProvider });
  context.subscriptions.push(navigatorTreeView);

  // Refresh command for build tree
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.refreshBuildTree', () => buildProvider.refresh()),
    vscode.commands.registerCommand('mlsysbook.refreshNavigator', () => {
      navigatorProvider.refreshFromEditor(vscode.window.activeTextEditor);
    }),
    vscode.commands.registerCommand('mlsysbook.setNavigatorFilterPreset', async () => {
      const currentPreset = vscode.workspace
        .getConfiguration('mlsysbook')
        .get<NavigatorPresetId>('defaultNavigatorPreset', 'all');
      const pick = await vscode.window.showQuickPick(
        NAVIGATOR_PRESETS.map(p => ({
          ...p,
          description: p.id === currentPreset ? 'current default' : '',
        })),
        { placeHolder: 'Select navigator filter preset' },
      );
      if (!pick) { return; }
      const shouldSetDefault = await vscode.window.showQuickPick(
        [
          { label: 'Apply now only', id: 'apply' },
          { label: 'Apply and set as workspace default', id: 'default' },
        ],
        { placeHolder: 'Apply this preset temporarily or pin it as default?' },
      );
      if (!shouldSetDefault) { return; }

      await config.update('navigatorVisibleEntryKinds', pick.kinds, vscode.ConfigurationTarget.Workspace);
      if (shouldSetDefault.id === 'default') {
        await config.update('defaultNavigatorPreset', pick.id, vscode.ConfigurationTarget.Workspace);
      }
      navigatorProvider.refreshView();
    }),
    vscode.commands.registerCommand('mlsysbook.refreshRunHistory', () => {
      runHistoryProvider.refresh();
    }),
    vscode.commands.registerCommand('mlsysbook.refreshQmdDiagnostics', () => {
      diagnosticsManager.refreshActiveEditorDiagnostics();
    }),
    vscode.commands.registerCommand('mlsysbook.openNavigatorLocation', async (uri: vscode.Uri, line: number) => {
      const document = await vscode.workspace.openTextDocument(uri);
      const editor = await vscode.window.showTextDocument(document, { preview: false });
      const position = new vscode.Position(line, 0);
      editor.selection = new vscode.Selection(position, position);
      editor.revealRange(new vscode.Range(position, position), vscode.TextEditorRevealType.InCenter);
    }),
    vscode.commands.registerCommand('mlsysbook.rerunLastCommand', () => rerunLastCommand(false)),
    vscode.commands.registerCommand('mlsysbook.rerunLastCommandRaw', () => rerunLastCommand(true)),
    vscode.commands.registerCommand('mlsysbook.revealTerminal', () => revealRunTerminal(root)),
    vscode.commands.registerCommand('mlsysbook.openLastFailureDetails', () => showLastFailureDetails()),
    vscode.commands.registerCommand('mlsysbook.setExecutionMode', () => void setExecutionModeInteractively()),
    vscode.commands.registerCommand('mlsysbook.renameLabelReferences', () => void renameLabelAcrossWorkspace()),
  );

  // Register all command groups
  registerBuildCommands(context);
  registerDebugCommands(context);
  registerPrecommitCommands(context);
  registerPublishCommands(context);
  registerContextMenuCommands(context);

  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(editor => navigatorProvider.refreshFromEditor(editor)),
    vscode.workspace.onDidSaveTextDocument(document => navigatorProvider.refreshFromDocument(document)),
    vscode.workspace.onDidChangeConfiguration(event => {
      if (event.affectsConfiguration('mlsysbook.navigatorVisibleEntryKinds')) {
        navigatorProvider.refreshView();
      }
    }),
    vscode.window.onDidChangeTextEditorSelection(event => {
      if (!vscode.workspace.getConfiguration('mlsysbook').get<boolean>('navigatorFollowCursor', true)) {
        return;
      }
      if (!event.textEditor.document.uri.fsPath.endsWith('.qmd')) {
        return;
      }
      const sectionItem = navigatorProvider.getSectionItemForLine(event.selections[0].active.line);
      if (!sectionItem) {
        return;
      }
      void navigatorTreeView.reveal(sectionItem, { focus: false, select: false, expand: true });
    }),
  );
}

export function deactivate(): void {
  // nothing to clean up
}
