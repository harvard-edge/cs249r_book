"use client";

import { getLevelDef } from "@/lib/levels";

export default function LevelBadge({ level, size = "sm" }: { level: string; size?: "sm" | "md" }) {
  const def = getLevelDef(level);
  const isSm = size === "sm";

  return (
    <span
      className={`inline-flex items-center gap-1 font-mono font-bold rounded-md border ${
        isSm ? "text-[10px] px-1.5 py-0.5" : "text-[12px] px-2 py-1"
      }`}
      style={{
        color: def.color,
        backgroundColor: `${def.color}12`,
        borderColor: `${def.color}30`,
      }}
    >
      {def.id}
      {!isSm && <span className="font-medium opacity-70">{def.name}</span>}
    </span>
  );
}
