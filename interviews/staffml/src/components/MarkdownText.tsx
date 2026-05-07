"use client";

import React from "react";
import clsx from "clsx";

/**
 * Renders basic markdown-like text with bold, inline code, and highlighted numbers/units.
 * Extracted from NapkinMathDisplay for reuse across the app.
 */
export default function MarkdownText({ text, className }: { text: string; className?: string }) {
  if (!text) return null;

  // Split on **bold** markers and inline code `backticks`
  // Match **bold**, `code`, and ~~strikethrough~~ (double tilde only — single ~ means "approximately")
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|~~[^~]+~~)/g);

  return (
    <span className={className}>
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
        // Highlight numbers and units inline
        return <HighlightNumbers key={i} text={part} />;
      })}
    </span>
  );
}

function HighlightNumbers({ text }: { text: string }) {
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
