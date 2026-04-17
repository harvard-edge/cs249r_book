"use client";

/**
 * Phase-6 paper-prominence card — "Read the paper" above the fold on About.
 *
 * ARCHITECTURE.md §9 + §14 Phase 6: academic readers land on About wanting to
 * cite the work. v1 buried the paper link in the footer. This card surfaces
 * it near the hero with copy-able BibTeX tied to the release hash — so
 * citations are reproducible-by-release (C-3 academic integrity).
 */

import { useState } from "react";
import { FileText, Copy, Check } from "lucide-react";

export interface PaperCitationCardProps {
  paperUrl: string;
  releaseId: string;
  releaseHash?: string;   // short or full — displayed truncated to 16 chars.
  doi?: string;           // optional — register via Zenodo per ARCHITECTURE.md §15 #2
  className?: string;
}

function bibtex({
  releaseId,
  releaseHash,
  doi,
}: {
  releaseId: string;
  releaseHash?: string;
  doi?: string;
}): string {
  const year = new Date().getFullYear();
  const version = `v${releaseId}`;
  const hashLine = releaseHash ? `  note = {Release hash: ${releaseHash.slice(0, 16)}},\n` : "";
  const doiLine = doi ? `  doi = {${doi}},\n` : "";
  return (
`@misc{staffml${year},
  title = {StaffML: ML Systems Interview Preparation Question Corpus},
  author = {Janapa Reddi, Vijay and contributors},
  year = {${year}},
  version = {${version}},
${doiLine}${hashLine}  url = {https://staffml.mlsysbook.ai}
}`);
}

export default function PaperCitationCard({
  paperUrl,
  releaseId,
  releaseHash,
  doi,
  className,
}: PaperCitationCardProps) {
  const [copied, setCopied] = useState(false);
  const cite = bibtex({ releaseId, releaseHash, doi });

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(cite);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // no-op; user can still select + copy manually
    }
  };

  return (
    <section
      aria-labelledby="paper-citation-heading"
      className={`p-5 rounded-xl border border-accentBlue/30 bg-accentBlue/5 ${className ?? ""}`}
    >
      <div className="flex items-start gap-3 mb-3">
        <FileText className="w-5 h-5 text-accentBlue mt-0.5 flex-shrink-0" aria-hidden="true" />
        <div className="flex-1">
          <h2 id="paper-citation-heading" className="text-sm font-semibold text-textPrimary mb-1">
            Read the paper
          </h2>
          <p className="text-[13px] text-textSecondary leading-relaxed">
            StaffML is described in a research paper on corpus design and
            competency-backed question authoring for ML systems.
          </p>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 mb-4">
        <a
          href={paperUrl}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-accentBlue text-white text-[12px] font-medium hover:bg-accentBlue/90 transition-colors"
        >
          <FileText className="w-3.5 h-3.5" aria-hidden="true" />
          Download PDF
        </a>
        {doi && (
          <a
            href={`https://doi.org/${doi}`}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-border bg-surface text-[12px] text-textSecondary hover:text-textPrimary hover:bg-surfaceSubtle transition-colors"
          >
            DOI: {doi}
          </a>
        )}
      </div>

      <details className="group">
        <summary className="cursor-pointer list-none flex items-center gap-1.5 text-[11px] font-mono uppercase tracking-wide text-textTertiary hover:text-textSecondary">
          <span className="group-open:rotate-90 transition-transform">▸</span>
          <span>Cite this release</span>
        </summary>
        <div className="mt-3 relative">
          <pre className="text-[11px] leading-relaxed overflow-x-auto p-3 rounded-md bg-surfaceSubtle border border-borderSubtle text-textSecondary font-mono">
{cite}
          </pre>
          <button
            type="button"
            onClick={handleCopy}
            aria-label="Copy BibTeX to clipboard"
            className="absolute top-2 right-2 inline-flex items-center gap-1 px-2 py-1 rounded border border-border bg-surface hover:bg-surfaceSubtle text-[10px] text-textSecondary"
          >
            {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
            {copied ? "copied" : "copy"}
          </button>
        </div>
        {releaseHash && (
          <p className="mt-2 text-[10px] font-mono text-textTertiary">
            release_hash: <span className="text-textSecondary">{releaseHash}</span>
          </p>
        )}
      </details>
    </section>
  );
}
