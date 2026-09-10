// ─── Level Definitions ─────────────────────────────────────────
// The backbone of StaffML: Bloom's Taxonomy mapped to ML systems roles.

export interface LevelDef {
  id: string;
  name: string;
  role: string;
  verb: string;
  example: string;
  color: string;
}

export const LEVELS: LevelDef[] = [
  {
    id: "L1",
    name: "Recall",
    role: "Entry",
    verb: "Can you name the thing?",
    example: "What does GPU HBM stand for?",
    color: "#60a5fa", // blue
  },
  {
    id: "L2",
    name: "Understand",
    role: "Junior",
    verb: "Can you explain why it matters?",
    example: "Why is memory bandwidth often more important than FLOPS for inference?",
    color: "#4ade80", // green
  },
  {
    id: "L3",
    name: "Apply",
    role: "Mid-Level",
    verb: "Can you use it to solve a problem?",
    example: "Calculate the minimum batch size to saturate an H100's memory bandwidth.",
    color: "#fbbf24", // amber
  },
  {
    id: "L4",
    name: "Analyze",
    role: "Senior",
    verb: "Can you compare trade-offs and diagnose?",
    example: "Your serving latency spiked 3x after switching FP16→FP8. Why?",
    color: "#fb923c", // orange
  },
  {
    id: "L5",
    name: "Evaluate",
    role: "Staff",
    verb: "Can you make architecture decisions under constraints?",
    example: "Design a serving stack for Llama-70B at 10K QPS on a $50K/month budget.",
    color: "#f87171", // red
  },
  {
    id: "L6+",
    name: "Architect",
    role: "Principal",
    verb: "Can you design novel systems and anticipate failure modes?",
    example: "Design fault-tolerant training for a 1T param model across 3 data centers.",
    color: "#c084fc", // purple
  },
];

export const LEVEL_IDS = LEVELS.map((l) => l.id);

export const LEVEL_MAP: Record<string, LevelDef> = Object.fromEntries(
  LEVELS.map((l) => [l.id, l])
);

export function getLevelDef(id: string): LevelDef {
  return LEVEL_MAP[id] || LEVELS[0];
}

export function getLevelLabel(id: string): string {
  const def = LEVEL_MAP[id];
  return def ? `${def.name} (${def.role})` : id;
}

export function getLevelShortLabel(id: string): string {
  const def = LEVEL_MAP[id];
  return def ? def.name : id;
}

/** Display name for UI — uses Bloom name, not L-number */
export function getLevelDisplayId(id: string): string {
  const def = LEVEL_MAP[id];
  return def ? def.name : id;
}
