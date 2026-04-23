"use client";

import { getLevelDef } from "@/lib/levels";
import { levelTooltip } from "@/lib/meta-descriptions";
import MetaTooltip from "@/components/MetaTooltip";

export default function LevelBadge({
  level,
  size = "sm",
  withTooltip = true,
}: {
  level: string;
  size?: "sm" | "md";
  /** Set false to opt out of the tooltip — useful inside other tooltipped controls */
  withTooltip?: boolean;
}) {
  const def = getLevelDef(level);
  const isSm = size === "sm";

  const badge = (
    <span
      className={`inline-flex items-center gap-1 font-semibold rounded-md border ${
        isSm ? "text-[10px] px-1.5 py-0.5" : "text-[12px] px-2 py-1"
      }`}
      style={{
        color: def.color,
        backgroundColor: `${def.color}12`,
        borderColor: `${def.color}30`,
      }}
    >
      <span className="font-mono font-bold">{def.id}</span>
      <span className="opacity-70">{def.name}</span>
    </span>
  );

  if (!withTooltip) return badge;
  const tip = levelTooltip(level);
  return (
    <MetaTooltip title={tip.title} body={tip.body}>
      {badge}
    </MetaTooltip>
  );
}
