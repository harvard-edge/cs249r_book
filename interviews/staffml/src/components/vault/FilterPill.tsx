import clsx from "clsx";

export default function FilterPill({ label, count, isActive, color, icon, onClick }: {
  label: string; count?: number; isActive: boolean;
  color?: string; icon?: React.ReactNode; onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={isActive}
      className={clsx(
        "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-semibold transition-all border",
        isActive
          ? "text-textPrimary border-borderHighlight"
          : "border-transparent text-textSecondary hover:text-textPrimary hover:bg-surface"
      )}
      style={isActive && color ? { backgroundColor: `${color}12`, borderColor: `${color}30`, color } : undefined}
    >
      {icon}
      {label}
      {count !== undefined && (
        <span className={clsx("font-mono text-[11px]", isActive ? "opacity-70" : "text-textMuted")}>
          {count}
        </span>
      )}
    </button>
  );
}
