"use client";

import { Network } from "lucide-react";
import taxonomy from "@/data/mlsys-taxonomy.json";

interface TaxonomyTagsProps {
  /** Subject identity (Question.taxonomy_subject). */
  subject?: string;
  /** Concept identities, most central first (Question.taxonomy_concepts). */
  concepts?: string[];
}

type Labels = {
  explorer: string;
  subjects: Record<string, { code: string; label: string; area: string }>;
  concepts: Record<string, { label: string; subject: string }>;
};

const labels = taxonomy as Labels;

/**
 * "Concepts tested" affordance, rendered AFTER the attempt beside the textbook
 * pointer. Names the ML Systems Taxonomy subject and concepts the question was
 * tagged with, each linking to its entry in the taxonomy explorer. Renders
 * nothing for an untagged question or an identity the labels file lacks.
 */
export default function TaxonomyTags({ subject, concepts }: TaxonomyTagsProps) {
  const s = subject ? labels.subjects[subject] : undefined;
  if (!s) return null;
  const known = (concepts ?? []).filter((c) => labels.concepts[c]);

  return (
    <div className="mt-3 rounded-lg border border-borderSubtle bg-surface p-4">
      <span className="flex items-center gap-1.5 text-[10px] font-mono uppercase text-textTertiary mb-2">
        <Network className="w-3 h-3" />
        Concepts tested
      </span>

      <a
        href={`${labels.explorer}#subject=${encodeURIComponent(subject!)}`}
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm font-medium text-textPrimary hover:underline underline-offset-2"
      >
        <span className="font-mono text-xs text-textTertiary mr-1.5">{s.code}</span>
        {s.label}
      </a>

      {known.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2 text-xs">
          {known.map((c) => (
            <a
              key={c}
              href={`${labels.explorer}#concept=${encodeURIComponent(c)}`}
              target="_blank"
              rel="noopener noreferrer"
              className="px-2 py-0.5 rounded-md border border-borderSubtle text-textSecondary hover:text-textPrimary hover:border-borderHighlight transition-colors"
            >
              {labels.concepts[c].label}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
