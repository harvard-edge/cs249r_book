import * as vscode from 'vscode';
import { getRepoRoot } from './utils/workspace';
import { BuildTreeProvider } from './providers/buildTreeProvider';
import { DebugTreeProvider } from './providers/debugTreeProvider';
import { PrecommitTreeProvider } from './providers/precommitTreeProvider';
import { PublishTreeProvider } from './providers/publishTreeProvider';
import { MaintenanceTreeProvider } from './providers/maintenanceTreeProvider';
import { ConfigTreeProvider } from './providers/configTreeProvider';
import { ChapterNavigatorProvider } from './providers/chapterNavigatorProvider';
import { RunHistoryProvider } from './providers/runHistoryProvider';
import { QmdFoldingProvider } from './providers/qmdFoldingProvider';
import { QmdAutoFoldManager } from './providers/qmdAutoFoldManager';
import { QmdDiagnosticsManager, WorkspaceLabelIndex } from './validation/qmdDiagnostics';
import { QmdChunkHighlighter } from './providers/qmdChunkHighlighter';
import { QmdPythonValueResolver } from './providers/qmdPythonValueResolver';
import { QmdPythonHoverProvider, QmdPythonGhostText, QmdPythonCodeLensProvider } from './providers/qmdInlinePythonProviders';
import { renameLabelAcrossWorkspace } from './utils/labelRename';
import { registerBuildCommands } from './commands/buildCommands';
import { registerDebugCommands } from './commands/debugCommands';
import { registerPrecommitCommands } from './commands/precommitCommands';
import { registerPublishCommands } from './commands/publishCommands';
import { registerContextMenuCommands } from './commands/contextMenuCommands';
import {
  initializeRunManager,
  rerunLastCommand,
  rerunSavedCommand,
  revealRunTerminal,
  showLastFailureDetails,
  setExecutionModeInteractively,
  runInVisibleTerminal,
} from './utils/terminal';
import { ChapterOrderSource } from './types';

type NavigatorPresetId = 'all' | 'writing' | 'reference' | 'structure';

interface NavigatorPreset {
  label: string;
  id: NavigatorPresetId;
  kinds: string[];
}

const NAVIGATOR_PRESETS: NavigatorPreset[] = [
  { label: 'Show All', id: 'all', kinds: ['figures', 'tables', 'listings', 'equations', 'callouts'] },
  { label: 'Writing Focus', id: 'writing', kinds: ['figures', 'tables', 'listings', 'callouts'] },
  { label: 'Reference Focus', id: 'reference', kinds: ['figures', 'tables', 'equations'] },
  { label: 'Structure Focus', id: 'structure', kinds: ['listings', 'equations', 'callouts'] },
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
  const maintenanceProvider = new MaintenanceTreeProvider();
  const publishProvider = new PublishTreeProvider();
  const configProvider = new ConfigTreeProvider();
  const navigatorProvider = new ChapterNavigatorProvider();
  const runHistoryProvider = new RunHistoryProvider();
  const qmdFoldingProvider = new QmdFoldingProvider();
  const qmdAutoFoldManager = new QmdAutoFoldManager();
  const workspaceLabelIndex = new WorkspaceLabelIndex();
  const diagnosticsManager = new QmdDiagnosticsManager();
  diagnosticsManager.setWorkspaceIndex(workspaceLabelIndex);
  const chunkHighlighter = new QmdChunkHighlighter();
  chunkHighlighter.setWorkspaceIndex(workspaceLabelIndex);

  // Inline Python value resolution (hover, ghost text, CodeLens)
  let pythonResolver: QmdPythonValueResolver | undefined;
  let pythonHoverProvider: QmdPythonHoverProvider | undefined;
  let pythonGhostText: QmdPythonGhostText | undefined;
  let pythonCodeLensProvider: QmdPythonCodeLensProvider | undefined;
  try {
    pythonResolver = new QmdPythonValueResolver();
    pythonHoverProvider = new QmdPythonHoverProvider(pythonResolver);
    pythonGhostText = new QmdPythonGhostText(pythonResolver);
    pythonCodeLensProvider = new QmdPythonCodeLensProvider(pythonResolver);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    vscode.window.showWarningMessage(`MLSysBook: Python value resolver failed to initialize: ${msg}`);
  }
  const config = vscode.workspace.getConfiguration('mlsysbook');
  const defaultPreset = config.get<NavigatorPresetId>('defaultNavigatorPreset', 'all');
  const preset = NAVIGATOR_PRESETS.find(p => p.id === defaultPreset) ?? NAVIGATOR_PRESETS[0];
  void config.update('navigatorVisibleEntryKinds', preset.kinds, vscode.ConfigurationTarget.Workspace);
  navigatorProvider.refreshFromEditor(vscode.window.activeTextEditor);
  workspaceLabelIndex.start();
  diagnosticsManager.start();
  chunkHighlighter.start();
  pythonGhostText?.start();
  pythonCodeLensProvider?.start();
  qmdAutoFoldManager.start();

  context.subscriptions.push(
    vscode.window.createTreeView('mlsysbook.build', { treeDataProvider: buildProvider }),
    vscode.window.createTreeView('mlsysbook.debug', { treeDataProvider: debugProvider }),
    vscode.window.createTreeView('mlsysbook.runs', { treeDataProvider: runHistoryProvider }),
    vscode.window.createTreeView('mlsysbook.precommit', { treeDataProvider: precommitProvider }),
    vscode.window.createTreeView('mlsysbook.maintenance', { treeDataProvider: maintenanceProvider }),
    vscode.window.createTreeView('mlsysbook.publish', { treeDataProvider: publishProvider }),
    vscode.window.createTreeView('mlsysbook.config', { treeDataProvider: configProvider }),
    vscode.languages.registerFoldingRangeProvider(
      { pattern: '**/*.qmd' },
      qmdFoldingProvider,
    ),
    runHistoryProvider,
    qmdAutoFoldManager,
    workspaceLabelIndex,
    diagnosticsManager,
    chunkHighlighter,
  );
  if (pythonResolver && pythonHoverProvider && pythonGhostText && pythonCodeLensProvider) {
    context.subscriptions.push(
      pythonResolver,
      pythonGhostText,
      pythonCodeLensProvider,
      vscode.languages.registerHoverProvider(
        { pattern: '**/*.qmd' },
        pythonHoverProvider,
      ),
      vscode.languages.registerCodeLensProvider(
        { pattern: '**/*.qmd' },
        pythonCodeLensProvider,
      ),
    );
  }
  const navigatorTreeView = vscode.window.createTreeView('mlsysbook.navigator', { treeDataProvider: navigatorProvider });
  context.subscriptions.push(navigatorTreeView);
  context.subscriptions.push(
    navigatorTreeView.onDidChangeSelection(event => {
      const selected = event.selection[0] as vscode.TreeItem | undefined;
      if (!selected || selected.contextValue !== 'navigator-section') {
        return;
      }
      const command = selected.command;
      if (
        !command
        || command.command !== 'mlsysbook.openNavigatorLocation'
        || !Array.isArray(command.arguments)
      ) {
        return;
      }
      const [uri, line] = command.arguments;
      if (!(uri instanceof vscode.Uri) || typeof line !== 'number') {
        return;
      }
      void vscode.commands.executeCommand('mlsysbook.openNavigatorLocation', uri, line);
    }),
  );

  // Refresh command for build tree
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysbook.refreshBuildTree', () => buildProvider.refresh()),
    vscode.commands.registerCommand('mlsysbook.refreshNavigator', () => {
      navigatorProvider.refreshFromEditor(vscode.window.activeTextEditor);
    }),
    vscode.commands.registerCommand('mlsysbook.navigatorExpandAll', () => {
      navigatorProvider.expandAll();
    }),
    vscode.commands.registerCommand('mlsysbook.navigatorCollapseAll', () => {
      navigatorProvider.collapseAll();
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
    vscode.commands.registerCommand('mlsysbook.historyRerunCommand', (record) => {
      void rerunSavedCommand(record);
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
    vscode.commands.registerCommand('mlsysbook.setChapterOrderSource', async () => {
      const current = config.get<ChapterOrderSource>('chapterOrderSource', 'auto');
      const pick = await vscode.window.showQuickPick(
        [
          { label: 'Auto (PDF -> EPUB -> HTML)', id: 'auto' as ChapterOrderSource },
          { label: 'PDF', id: 'pdf' as ChapterOrderSource },
          { label: 'EPUB', id: 'epub' as ChapterOrderSource },
          { label: 'HTML', id: 'html' as ChapterOrderSource },
          { label: 'PDF Copyedit', id: 'pdfCopyedit' as ChapterOrderSource },
        ].map(item => ({
          ...item,
          description: item.id === current ? 'current default' : '',
        })),
        { placeHolder: 'Select chapter order source for build/debug chapter lists' },
      );
      if (!pick) { return; }
      await config.update('chapterOrderSource', pick.id, vscode.ConfigurationTarget.Workspace);
      buildProvider.refresh();
      vscode.window.showInformationMessage(`MLSysBook chapter order source set to: ${pick.id}`);
    }),
    vscode.commands.registerCommand('mlsysbook.setQmdVisualPreset', async () => {
      const pick = await vscode.window.showQuickPick(
        [
          { label: 'Subtle', id: 'subtle' },
          { label: 'Balanced', id: 'balanced' },
          { label: 'High Contrast', id: 'highContrast' },
        ],
        { placeHolder: 'Select QMD visual preset' },
      );
      if (!pick) { return; }
      await config.update('qmdVisualPreset', pick.id, vscode.ConfigurationTarget.Workspace);
    }),
    vscode.commands.registerCommand('mlsysbook.openSettings', async () => {
      await vscode.commands.executeCommand('workbench.action.openSettings', '@ext:mlsysbook.mlsysbook-workbench');
    }),
    vscode.commands.registerCommand('mlsysbook.renameLabelReferences', () => void renameLabelAcrossWorkspace()),

    // Section ID management (uses binder CLI)
    vscode.commands.registerCommand('mlsysbook.addSectionIds', () => {
      if (!root) { return; }
      const editor = vscode.window.activeTextEditor;
      const target = editor?.document.uri.fsPath.endsWith('.qmd')
        ? `--path ${editor.document.uri.fsPath}`
        : '--vol1';
      runInVisibleTerminal(
        `./book/binder fix headers add ${target} --force`,
        root,
        'Add Section IDs',
      );
    }),
    vscode.commands.registerCommand('mlsysbook.verifySectionIds', () => {
      if (!root) { return; }
      runInVisibleTerminal(
        './book/binder check headers --vol1',
        root,
        'Verify Section IDs',
      );
    }),
    vscode.commands.registerCommand('mlsysbook.validateCrossReferences', () => {
      if (!root) { return; }
      runInVisibleTerminal(
        './book/binder check labels --scope orphans --vol1',
        root,
        'Validate Cross-References',
      );
    }),
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
      if (event.affectsConfiguration('mlsysbook.chapterOrderSource')) {
        buildProvider.refresh();
      }
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
