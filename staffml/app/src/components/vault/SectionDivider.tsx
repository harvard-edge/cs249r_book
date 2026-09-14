export default function SectionDivider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3">
      <div className="h-px flex-1 bg-borderSubtle" />
      <span className="text-[11px] font-semibold uppercase tracking-widest text-textTertiary shrink-0">
        {label}
      </span>
      <div className="h-px flex-1 bg-borderSubtle" />
    </div>
  );
}
