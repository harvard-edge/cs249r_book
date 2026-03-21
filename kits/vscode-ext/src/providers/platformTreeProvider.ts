import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { CategoryTreeItem, LabTreeItem } from '../models/treeItems';

type TreeNode = CategoryTreeItem | LabTreeItem;

/** Platforms & Labs — browse hardware platforms and their labs */
export class PlatformTreeProvider implements vscode.TreeDataProvider<TreeNode> {
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
        new CategoryTreeItem('Arduino Nicla Vision', 'nicla', 'circuit-board'),
        new CategoryTreeItem('Seeed XIAO ESP32S3', 'xiao', 'circuit-board'),
        new CategoryTreeItem('Grove Vision AI V2', 'grove', 'circuit-board'),
        new CategoryTreeItem('Raspberry Pi', 'raspi', 'circuit-board'),
        new CategoryTreeItem('Shared Resources', 'shared', 'library'),
      ];
    }

    if (element instanceof CategoryTreeItem) {
      switch (element.categoryId) {
        case 'nicla':
          return this.getLabsForPlatform('arduino/nicla_vision', [
            ['Setup', 'setup'],
            ['Image Classification', 'image_classification'],
            ['Object Detection', 'object_detection'],
            ['Keyword Spotting', 'kws'],
            ['Motion Classification', 'motion_classification'],
          ]);
        case 'xiao':
          return this.getLabsForPlatform('seeed/xiao_esp32s3', [
            ['Setup', 'setup'],
            ['Image Classification', 'image_classification'],
            ['Object Detection', 'object_detection'],
            ['Keyword Spotting', 'kws'],
            ['Motion Classification', 'motion_classification'],
          ]);
        case 'grove':
          return this.getLabsForPlatform('seeed/grove_vision_ai_v2', [
            ['Setup & No-Code Apps', 'setup_and_no_code_apps'],
            ['Image Classification', 'image_classification'],
            ['Object Detection', 'object_detection'],
          ]);
        case 'raspi':
          return this.getLabsForPlatform('raspi', [
            ['Setup', 'setup'],
            ['Image Classification', 'image_classification'],
            ['Object Detection', 'object_detection'],
            ['Keyword Spotting', 'kws'],
            ['Motion Classification', 'motion_classification'],
            ['LLM Deployment', 'llm'],
            ['VLM Deployment', 'vlm'],
          ]);
        case 'shared':
          return this.getLabsForPlatform('shared', [
            ['KWS Feature Engineering', 'kws_feature_eng'],
            ['DSP Spectral Features', 'dsp_spectral_features_block'],
          ]);
      }
    }

    return [];
  }

  private getLabsForPlatform(platformPath: string, labs: string[][]): LabTreeItem[] {
    const contentsDir = path.join(this.projectRoot, 'kits', 'contents', platformPath);
    const items: LabTreeItem[] = [];

    for (const [label, dirName] of labs) {
      const labDir = path.join(contentsDir, dirName);
      if (!fs.existsSync(labDir)) { continue; }

      // Find the main .qmd file in the lab directory
      const qmdFiles = fs.readdirSync(labDir).filter(f => f.endsWith('.qmd'));
      if (qmdFiles.length > 0) {
        const mainQmd = path.join(labDir, qmdFiles[0]);
        items.push(new LabTreeItem(label, mainQmd, 'beaker'));
      } else {
        // Lab directory exists but no .qmd yet
        items.push(new LabTreeItem(`${label} (empty)`, labDir, 'circle-outline'));
      }
    }

    // Also add the platform hub .qmd if it exists
    const hubQmd = this.findHubQmd(contentsDir, platformPath);
    if (hubQmd) {
      items.unshift(new LabTreeItem('Platform Overview', hubQmd, 'home'));
    }

    return items;
  }

  private findHubQmd(contentsDir: string, platformPath: string): string | null {
    // Hub files are named after the platform: nicla_vision.qmd, raspi.qmd, etc.
    const dirName = path.basename(contentsDir);
    const hubPath = path.join(contentsDir, `${dirName}.qmd`);
    if (fs.existsSync(hubPath)) {
      return hubPath;
    }
    return null;
  }
}
