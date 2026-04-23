import * as vscode from 'vscode';
import { CategoryTreeItem, ZooEntryTreeItem } from '../models/treeItems';

type TreeNode = CategoryTreeItem | ZooEntryTreeItem;

/** Hardware and Models Zoo — browse mlsysim registries */
export class ZooTreeProvider implements vscode.TreeDataProvider<TreeNode> {
  private _onDidChange = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  constructor(private projectRoot: string) {}

  refresh(): void {
    this._onDidChange.fire(undefined);
  }

  getTreeItem(element: TreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: TreeNode): TreeNode[] {
    if (!element) {
      return [
        new CategoryTreeItem('Hardware', 'hardware', 'server'),
        new CategoryTreeItem('Models', 'models', 'symbol-class'),
        new CategoryTreeItem('Systems', 'systems', 'server-process'),
        new CategoryTreeItem('Infrastructure', 'infra', 'globe'),
      ];
    }

    if (element instanceof CategoryTreeItem) {
      switch (element.categoryId) {
        case 'hardware':
          return this.getHardwareCategories();
        case 'models':
          return this.getModelCategories();
        case 'systems':
          return this.getSystemEntries();
        case 'infra':
          return this.getInfraEntries();

        // Hardware sub-categories
        case 'hw-cloud':
          return this.getHardwareDevices('Cloud', [
            ['H100 SXM5', '989 TFLOPS FP16, 80 GB HBM3, 3350 GB/s'],
            ['H200 SXM', '989 TFLOPS FP16, 141 GB HBM3e, 4800 GB/s'],
            ['B200 SXM', '2250 TFLOPS FP16, 192 GB HBM3e, 8000 GB/s'],
            ['A100 SXM', '312 TFLOPS FP16, 80 GB HBM2e, 2039 GB/s'],
            ['V100 SXM2', '125 TFLOPS FP16, 32 GB HBM2, 900 GB/s'],
            ['MI300X', '1307 TFLOPS FP16, 192 GB HBM3, 5300 GB/s'],
            ['TPU v5p', '459 TFLOPS BF16, 95 GB HBM2e, 2765 GB/s'],
            ['T4', '65 TFLOPS FP16, 16 GB GDDR6, 300 GB/s'],
          ]);
        case 'hw-workstation':
          return this.getHardwareDevices('Workstation', [
            ['MacBook M3 Max', '14 TFLOPS FP16, 128 GB Unified, 400 GB/s'],
          ]);
        case 'hw-mobile':
          return this.getHardwareDevices('Mobile', [
            ['iPhone 15 Pro (A17)', '35 TOPS INT8, 8 GB LPDDR5, 51.2 GB/s'],
            ['Snapdragon 8 Gen 3', '45 TOPS INT8, 24 GB LPDDR5X, 77 GB/s'],
          ]);
        case 'hw-edge':
          return this.getHardwareDevices('Edge', [
            ['Jetson Orin NX', '100 TOPS INT8, 16 GB LPDDR5, 102 GB/s'],
            ['Coral Dev Board', '4 TOPS INT8, 4 GB LPDDR4, 25.6 GB/s'],
          ]);
        case 'hw-tiny':
          return this.getHardwareDevices('Tiny', [
            ['ESP32-S3', '0.00048 TFLOPS, 0.5 MB SRAM, 0.9 GB/s'],
            ['Himax WE1', '0.0004 TFLOPS, 2 MB SRAM, 0.4 GB/s'],
          ]);

        // Model sub-categories
        case 'mdl-language':
          return this.getModelEntries([
            ['GPT-2', '124M params, Transformer'],
            ['GPT-3', '175B params, Transformer'],
            ['GPT-4', '1.8T params (est.), MoE Transformer'],
            ['BERT Base', '110M params, Transformer Encoder'],
            ['BERT Large', '340M params, Transformer Encoder'],
            ['Llama-2 7B', '7B params, Transformer'],
            ['Llama-2 70B', '70B params, Transformer'],
            ['Llama-3 8B', '8B params, Transformer'],
            ['Llama-3 70B', '70B params, Transformer'],
            ['Mamba', '2.8B params, SSM'],
          ]);
        case 'mdl-vision':
          return this.getModelEntries([
            ['ResNet-50', '25.6M params, CNN'],
            ['MobileNet V2', '3.4M params, CNN'],
            ['YOLO v8 Nano', '3.2M params, CNN'],
            ['AlexNet', '62.4M params, CNN'],
          ]);
        case 'mdl-other':
          return this.getModelEntries([
            ['DLRM', '540M params, Recommendation'],
            ['Stable Diffusion v1.5', '860M params, Diffusion'],
            ['KWS DSCNN', '23K params, TinyML'],
            ['WakeVision', '96K params, TinyML'],
          ]);
      }
    }

    return [];
  }

  private getHardwareCategories(): CategoryTreeItem[] {
    return [
      new CategoryTreeItem('Cloud', 'hw-cloud', 'cloud'),
      new CategoryTreeItem('Workstation', 'hw-workstation', 'device-desktop'),
      new CategoryTreeItem('Mobile', 'hw-mobile', 'device-mobile'),
      new CategoryTreeItem('Edge', 'hw-edge', 'circuit-board'),
      new CategoryTreeItem('Tiny', 'hw-tiny', 'pulse'),
    ];
  }

  private getModelCategories(): CategoryTreeItem[] {
    return [
      new CategoryTreeItem('Language Models', 'mdl-language', 'comment-discussion'),
      new CategoryTreeItem('Vision Models', 'mdl-vision', 'eye'),
      new CategoryTreeItem('Other', 'mdl-other', 'extensions'),
    ];
  }

  private getSystemEntries(): ZooEntryTreeItem[] {
    return [
      new ZooEntryTreeItem('Research 256', '256 GPUs, 10GbE', 'hardware', 'server-process'),
      new ZooEntryTreeItem('Frontier 8K', '8192 GPUs, IB NDR', 'hardware', 'server-process'),
      new ZooEntryTreeItem('Production 2K', '2048 GPUs, IB HDR', 'hardware', 'server-process'),
      new ZooEntryTreeItem('Mega 100K', '100K GPUs, IB NDR', 'hardware', 'server-process'),
    ];
  }

  private getInfraEntries(): ZooEntryTreeItem[] {
    return [
      new ZooEntryTreeItem('US Average', 'PUE 1.58, 386 gCO₂/kWh', 'hardware', 'globe'),
      new ZooEntryTreeItem('Quebec', 'PUE 1.10, 1.2 gCO₂/kWh', 'hardware', 'globe'),
      new ZooEntryTreeItem('Poland', 'PUE 1.40, 773 gCO₂/kWh', 'hardware', 'globe'),
    ];
  }

  private getHardwareDevices(_category: string, devices: string[][]): ZooEntryTreeItem[] {
    return devices.map(([name, specs]) =>
      new ZooEntryTreeItem(name, specs, 'hardware', 'chip')
    );
  }

  private getModelEntries(models: string[][]): ZooEntryTreeItem[] {
    return models.map(([name, specs]) =>
      new ZooEntryTreeItem(name, specs, 'model', 'symbol-class')
    );
  }
}
