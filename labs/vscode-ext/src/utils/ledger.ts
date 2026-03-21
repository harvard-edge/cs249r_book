import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { LedgerState } from '../types';

const LEDGER_DIR = path.join(os.homedir(), '.mlsys');
const LEDGER_PATH = path.join(LEDGER_DIR, 'ledger.json');

/** Read the Design Ledger from ~/.mlsys/ledger.json */
export function readLedger(): LedgerState | null {
  if (!fs.existsSync(LEDGER_PATH)) {
    return null;
  }
  try {
    const content = fs.readFileSync(LEDGER_PATH, 'utf-8');
    return JSON.parse(content) as LedgerState;
  } catch {
    return null;
  }
}

/** Check if ledger file exists */
export function ledgerExists(): boolean {
  return fs.existsSync(LEDGER_PATH);
}

/** Get the ledger file path */
export function getLedgerPath(): string {
  return LEDGER_PATH;
}

/** Reset the ledger by deleting the file */
export function resetLedger(): boolean {
  if (!fs.existsSync(LEDGER_PATH)) {
    return true;
  }
  try {
    fs.unlinkSync(LEDGER_PATH);
    return true;
  } catch {
    return false;
  }
}

/** Get ledger summary string for display */
export function getLedgerSummary(): string {
  const ledger = readLedger();
  if (!ledger) {
    return 'No ledger found';
  }

  const track = ledger.track ?? 'none';
  const chapter = ledger.current_chapter;
  const entries = ledger.history.length;

  return `Track: ${track} | Chapter: ${chapter} | Entries: ${entries}`;
}
