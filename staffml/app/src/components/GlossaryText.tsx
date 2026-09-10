"use client";

/**
 * GlossaryText — renders plain text with acronym hover tooltips.
 *
 * Scans the input text for known acronyms from the MLSysBook glossary and
 * wraps each match in a tooltip that shows the expansion and definition on
 * hover/focus. Non-acronym text passes through unchanged.
 *
 * Only annotates the first occurrence of each acronym per render to avoid
 * visual clutter.
 */

import { useMemo } from "react";
import MetaTooltip from "./MetaTooltip";
import { HighlightNumbers } from "./MarkdownText";
import { findAcronymsInText, type GlossaryEntry } from "@/lib/glossary";

export default function GlossaryText({
  text,
  className,
}: {
  text: string;
  className?: string;
}) {
  const segments = useMemo(() => annotate(text), [text]);

  if (segments.length === 1 && segments[0].type === "text") {
    return <HighlightNumbers text={text} />;
  }

  return (
    <span className={className}>
      {segments.map((seg, i) => {
        if (seg.type === "text") {
          return <HighlightNumbers key={i} text={seg.value} />;
        }
        const title = seg.entry.acronym
          ? `${seg.entry.display} — ${seg.entry.acronym}`
          : seg.entry.display;
        return (
          <MetaTooltip
            key={i}
            title={title}
            body={seg.entry.definition}
            side="bottom"
            className="inline"
          >
            <span className="border-b border-dotted border-textTertiary/50 cursor-help">
              {seg.value}
            </span>
          </MetaTooltip>
        );
      })}
    </span>
  );
}

type Segment =
  | { type: "text"; value: string }
  | { type: "acronym"; value: string; entry: GlossaryEntry };

function annotate(text: string): Segment[] {
  const matches = findAcronymsInText(text);
  if (matches.length === 0) return [{ type: "text", value: text }];

  const seen = new Set<string>();
  const unique = matches.filter((m) => {
    const key = m.entry.term;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  unique.sort((a, b) => a.start - b.start);

  const segments: Segment[] = [];
  let cursor = 0;

  for (const m of unique) {
    if (m.start < cursor) continue;
    if (m.start > cursor) {
      segments.push({ type: "text", value: text.slice(cursor, m.start) });
    }
    segments.push({
      type: "acronym",
      value: text.slice(m.start, m.end),
      entry: m.entry,
    });
    cursor = m.end;
  }

  if (cursor < text.length) {
    segments.push({ type: "text", value: text.slice(cursor) });
  }

  return segments;
}
