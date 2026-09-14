/** A recorded command run */
export interface CommandRunRecord {
  id: string;
  label: string;
  command: string;
  cwd: string;
  timestamp: string;
  status: 'started' | 'succeeded' | 'failed';
}

/** Hardware device from the mlsysim registry */
export interface HardwareDevice {
  name: string;
  category: string;
  tflops: number;
  memory_gb: number;
  bandwidth_gbs: number;
  tdp_w: number;
}

/** Model from the mlsysim registry */
export interface ModelInfo {
  name: string;
  category: string;
  params: string;
  architecture: string;
}

/** Test suite category */
export interface TestCategory {
  id: string;
  label: string;
  file: string;
  icon: string;
}
