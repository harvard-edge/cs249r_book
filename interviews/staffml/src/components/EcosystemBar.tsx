"use client";

/**
 * Thin top bar connecting StaffML to the MLSysBook ecosystem.
 * Shows on all pages so users know StaffML is part of the curriculum.
 */
export default function EcosystemBar() {
  return (
    <div className="bg-[#1a1a2e] text-white/80 text-[11px] px-4 py-1.5 flex items-center justify-between overflow-x-auto scrollbar-hide">
      <div className="flex items-center gap-4 shrink-0">
        <a
          href="https://mlsysbook.ai"
          className="font-bold text-white hover:text-white/90 transition-colors flex items-center gap-1.5 shrink-0"
        >
          <svg viewBox="0 0 20 20" className="w-3.5 h-3.5">
            <rect width="20" height="20" rx="3" fill="#a31f34" />
            <text x="10" y="14.5" textAnchor="middle" fill="white" fontSize="11" fontWeight="700" fontFamily="system-ui">M</text>
          </svg>
          MLSysBook
        </a>
        <span className="text-white/30 hidden sm:inline">|</span>
        <div className="hidden sm:flex items-center gap-3">
          <a href="https://mlsysbook.ai/vol1/" className="hover:text-white transition-colors">Vol I</a>
          <a href="https://mlsysbook.ai/vol2/" className="hover:text-white transition-colors">Vol II</a>
          <a href="https://mlsysbook.ai/labs/" className="hover:text-white transition-colors">Labs</a>
          <a href="https://mlsysbook.ai/tinytorch/" className="hover:text-white transition-colors">TinyTorch</a>
          <span className="text-white font-semibold">StaffML</span>
        </div>
      </div>
      <a
        href="https://github.com/harvard-edge/cs249r_book"
        target="_blank"
        rel="noopener noreferrer"
        className="hover:text-white transition-colors shrink-0 hidden md:inline"
      >
        Open Source
      </a>
    </div>
  );
}
