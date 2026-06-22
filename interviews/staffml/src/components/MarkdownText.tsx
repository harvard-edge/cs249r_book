"use client";

import React from "react";
import clsx from "clsx";
import GlossaryText from "./GlossaryText";
import MathText from "./MathText";

/**
 * Splits a string into alternating text and math segments. Math is matched
 * FIRST (before bold/code/glossary processing) so the LaTeX body — which can
 * contain `*`, `_`, `\`, and unit tokens — is never mangled by the markdown or
 * number-highlighter passes. Display math (`$$...$$`) is matched before inline
 * (`$...$`). A `$` immediately followed by a digit with no closing `$` on a
 * short span is left alone by requiring a non-greedy body and a closing `$`.
 */
const MATH_SPLIT = /(\$\$[^$]+\$\$|\$[^$\n]+\$)/g;

// A `$...$` segment is real math only if its body carries a LaTeX signal
// (backslash, caret, underscore, brace). Bare-$ currency — "$3M", "$2/hr and
// $48/day" — has none, so it renders as text instead of garbled KaTeX. ~5% of
// vault questions use bare-$ currency and would otherwise mis-render.
const MATH_BODY = /[\\^_{}]/;

function renderMathAware(text: string, glossary: boolean): React.ReactNode {
  const segments = text.split(MATH_SPLIT);
  return segments.map((seg, i) => {
    if (seg.startsWith("$$") && seg.endsWith("$$") && seg.length > 4 && MATH_BODY.test(seg.slice(2, -2))) {
      return <MathText key={`m${i}`} expr={seg.slice(2, -2).trim()} display />;
    }
    if (seg.startsWith("$") && seg.endsWith("$") && seg.length > 2 && MATH_BODY.test(seg.slice(1, -1))) {
      return <MathText key={`m${i}`} expr={seg.slice(1, -1).trim()} />;
    }
    return <MarkdownInline key={`t${i}`} text={seg} glossary={glossary} />;
  });
}

/**
 * Renders basic markdown-like text with KaTeX math (`$...$` / `$$...$$`),
 * bold, inline code, highlighted numbers/units, and optional glossary acronym
 * tooltips.
 */
export default function MarkdownText({
  text,
  className,
  glossary = true,
}: {
  text: string;
  className?: string;
  glossary?: boolean;
}) {
  if (!text) return null;
  return <span className={className}>{renderMathAware(text, glossary)}</span>;
}

/** The original (non-math) markdown rendering: bold / code / strikethrough. */
function MarkdownInline({
  text,
  glossary,
}: {
  text: string;
  glossary: boolean;
}) {
  if (!text) return null;

  // Split on **bold** markers and inline code `backticks`
  // Match **bold**, `code`, and ~~strikethrough~~ (double tilde only — single ~ means "approximately")
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|~~[^~]+~~)/g);

  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**")) {
          return (
            <span key={i} className="font-bold">
              {part.slice(2, -2)}
            </span>
          );
        }
        if (part.startsWith("`") && part.endsWith("`")) {
          return (
            <code
              key={i}
              className="text-accentBlue bg-accentBlue/10 px-1 py-0.5 rounded text-[12px]"
            >
              {part.slice(1, -1)}
            </code>
          );
        }
        if (part.startsWith("~~") && part.endsWith("~~")) {
          return (
            <span key={i} className="text-textTertiary line-through">
              {part.slice(2, -2)}
            </span>
          );
        }
        if (glossary) {
          return <GlossaryText key={i} text={part} />;
        }
        return <HighlightNumbers key={i} text={part} />;
      })}
    </>
  );
}

export function HighlightNumbers({ text }: { text: string }) {
  // Highlight numbers with units (e.g., "5 ms", "3.35 TB/s", "989 TFLOPS",
  // "120e12 FLOPs", "1.2×10^14 bytes"). The number group accepts:
  //   - plain integers/decimals with optional thousands commas: 1,000.5
  //   - scientific shorthand: 989e12, 1.5e-3
  //   - explicit ×10^N notation: 1.2×10^14
  // The unit group covers throughput, bandwidth, memory, latency, power,
  // energy, and ML-specific tokens; the optional /(s|byte|cycle|inf|J|W)
  // suffix handles compound rates like TFLOP/s, FLOPs/byte, FLOP/cycle.
  const parts = text.split(
    /([\d,]+(?:\.\d+)?(?:e-?\d+|×10\^-?\d+)?)\s*((?:TFLOPS|TOPS|GFLOPS|TFLOP|GFLOP|TB|GB|MB|KB|kB|Hz|MHz|GHz|ms|μs|ns|s|W|mW|μW|mJ|μJ|J|%|FLOPs|FLOP|Ops|bytes?|tokens?|MACs?|cycles?|frames?|samples?)(?:\/(?:s|byte|cycle|inf|J|W))?\b)/gi
  );

  if (parts.length === 1) return <>{text}</>;

  return (
    <>
      {parts.map((part, i) => {
        // Pattern: [before, number, unit, ...rest]
        // Groups come in triples: text, number, unit
        if (i % 3 === 1) {
          // This is the number part
          return (
            <span key={i} className="font-semibold text-textPrimary">
              {part}
            </span>
          );
        }
        if (i % 3 === 2) {
          // This is the unit part
          return (
            <span key={i} className="text-textTertiary">
              {" "}{part}
            </span>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}
