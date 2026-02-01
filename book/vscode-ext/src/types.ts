export type VolumeId = 'vol1' | 'vol2';
export type BuildFormat = 'html' | 'pdf' | 'epub';

export interface ChapterInfo {
  name: string;
  volume: VolumeId;
  dirPath: string;
  displayName: string;
}

export interface VolumeInfo {
  id: VolumeId;
  label: string;
  chapters: ChapterInfo[];
}

export interface PrecommitHook {
  id: string;
  label: string;
  command: string;
}

export interface ActionDef {
  id: string;
  label: string;
  command: string;
  icon?: string;
}

export interface QmdFileContext {
  volume: VolumeId;
  chapter: string;
}
