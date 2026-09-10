"use client";

import { BookOpen, ExternalLink } from "lucide-react";
import type { BookRef } from "@/lib/corpus";

interface BookRefCardProps {
  /** Topic-derived chapter pointers (Question.book_refs). */
  refs?: BookRef[];
}

/**
 * "Learn more in the textbook" affordance, rendered AFTER the attempt is
 * revealed. A go-deeper pointer tied to the question's topic — not an answer
 * key — derived at build time from schema/topic_chapter_map.yaml.
 *
 * Supersedes the older area-level ChapterLinks: refs are topic-granular (87
 * topics vs. 13 areas), carry an authored "why" line on the primary chapter,
 * and use the build-link-checked, 200-valid mlsysbook.ai chapter URL.
 * Renders nothing for the handful of topics with no chapter mapped.
 */
export default function BookRefCard({ refs }: BookRefCardProps) {
  if (!refs || refs.length === 0) return null;

  const primary = refs.find((r) => r.role === "primary") ?? refs[0];
  const alsoSee = refs.filter((r) => r !== primary);

  return (
    <div className="mt-3 rounded-lg border border-accentBlue/15 bg-accentBlue/5 p-4">
      <span className="flex items-center gap-1.5 text-[10px] font-mono uppercase text-accentBlue mb-2">
        <BookOpen className="w-3 h-3" />
        Learn more in the textbook
      </span>

      <a
        href={primary.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1 text-sm font-medium text-accentBlue hover:underline underline-offset-2"
      >
        Vol.{primary.vol} · {primary.title}
        <ExternalLink className="w-3 h-3 opacity-60" />
      </a>

      {primary.why && (
        <p className="mt-1 text-xs leading-relaxed text-textSecondary">
          {primary.why}
        </p>
      )}

      {alsoSee.length > 0 && (
        <div className="mt-2.5 flex flex-wrap items-center gap-2 text-xs">
          <span className="text-textTertiary">Also see:</span>
          {alsoSee.map((ch) => (
            <a
              key={`${ch.vol}-${ch.chapter}`}
              href={ch.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md
                         bg-accentBlue/5 text-accentBlue hover:bg-accentBlue/10
                         border border-accentBlue/10 transition-colors"
            >
              Vol.{ch.vol}: {ch.title}
              <ExternalLink className="w-2.5 h-2.5 opacity-50" />
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
