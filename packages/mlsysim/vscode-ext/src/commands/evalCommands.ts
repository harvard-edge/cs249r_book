import * as vscode from 'vscode';
import { runInTerminal } from '../utils/terminal';
import { mlsysimCommand } from '../utils/mlsysimCli';

const HARDWARE_ITEMS = [
  'H100', 'H200', 'B200', 'A100', 'V100', 'MI300X', 'TPUv5p', 'T4',
  'MacBookM3Max', 'iPhone15Pro', 'Snapdragon8Gen3',
  'JetsonOrinNX', 'CoralDevBoard', 'ESP32', 'HimaxWE1',
];

const MODEL_ITEMS = [
  'GPT2', 'GPT3', 'GPT4', 'BERT_Base', 'BERT_Large',
  'Llama2_7B', 'Llama2_70B', 'Llama3_8B', 'Llama3_70B', 'Mamba',
  'ResNet50', 'MobileNetV2', 'YOLOv8Nano', 'AlexNet',
  'DLRM', 'StableDiffusion', 'KWS_DSCNN', 'WakeVision',
];

export function registerEvalCommands(context: vscode.ExtensionContext, root: string): void {
  // Quick Eval: model → hardware → batch/precision → run
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysim.evalQuick', async () => {
      const model = await vscode.window.showQuickPick(MODEL_ITEMS, {
        placeHolder: 'Select a model workload',
        title: 'MLSysim Quick Eval — Step 1: Model',
      });
      if (!model) { return; }

      const hardware = await vscode.window.showQuickPick(HARDWARE_ITEMS, {
        placeHolder: 'Select target hardware',
        title: 'MLSysim Quick Eval — Step 2: Hardware',
      });
      if (!hardware) { return; }

      const config = vscode.workspace.getConfiguration('mlsysim');
      const batchSize = config.get<number>('defaultBatchSize', 1);
      const precision = config.get<string>('defaultPrecision', 'fp16');
      const efficiency = config.get<number>('defaultEfficiency', 0.5);

      const cmd = mlsysimCommand(
        `eval ${model} ${hardware} --batch-size ${batchSize} --precision ${precision} --efficiency ${efficiency}`
      );
      runInTerminal(cmd, root, `Eval: ${model} on ${hardware}`);
    }),
  );

  // Evaluate a specific YAML file
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysim.evalYaml', async () => {
      const files = await vscode.window.showOpenDialog({
        canSelectMany: false,
        filters: { 'YAML': ['yaml', 'yml'] },
        title: 'Select scenario YAML',
      });
      if (!files || files.length === 0) { return; }

      const cmd = mlsysimCommand(`eval ${files[0].fsPath}`);
      runInTerminal(cmd, root, `Eval: ${files[0].fsPath.split('/').pop()}`);
    }),
  );

  // Evaluate current file
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysim.evalCurrentFile', (resource?: vscode.Uri) => {
      const filePath = resource?.fsPath ?? vscode.window.activeTextEditor?.document.uri.fsPath;
      if (!filePath || (!filePath.endsWith('.yaml') && !filePath.endsWith('.yml'))) {
        vscode.window.showWarningMessage('MLSysim: open a YAML scenario file first.');
        return;
      }
      const cmd = mlsysimCommand(`eval ${filePath}`);
      runInTerminal(cmd, root, `Eval: ${filePath.split('/').pop()}`);
    }),
  );

  // Optimize commands
  const optimizers = [
    { command: 'mlsysim.optimizeParallelism', sub: 'parallelism', label: 'Optimize Parallelism' },
    { command: 'mlsysim.optimizeBatching', sub: 'batching', label: 'Optimize Batching' },
    { command: 'mlsysim.optimizePlacement', sub: 'placement', label: 'Optimize Placement' },
  ];

  for (const opt of optimizers) {
    context.subscriptions.push(
      vscode.commands.registerCommand(opt.command, async () => {
        const files = await vscode.window.showOpenDialog({
          canSelectMany: false,
          filters: { 'YAML': ['yaml', 'yml'] },
          title: `Select scenario YAML for ${opt.label}`,
        });
        if (!files || files.length === 0) { return; }

        const cmd = mlsysimCommand(`optimize ${opt.sub} ${files[0].fsPath}`);
        runInTerminal(cmd, root, `${opt.label}: ${files[0].fsPath.split('/').pop()}`);
      }),
    );
  }

  // Export schema
  context.subscriptions.push(
    vscode.commands.registerCommand('mlsysim.exportSchema', () => {
      const cmd = mlsysimCommand('schema');
      runInTerminal(cmd, root, 'Export Schema');
    }),
  );
}
