"use client";

import React from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

/**
 * Renders a single LaTeX math expression with KaTeX.
 *
 * SSR-safe: katex.renderToString produces a plain HTML string with no DOM or
 * browser API access, so it runs identically on the server and the client.
 * We inject it via dangerouslySetInnerHTML — the input is author-controlled
 * corpus content (scenario / napkin_math / realistic_solution / common_mistake),
 * not user input, and `throwOnError: false` makes KaTeX emit a styled error
 * span rather than crash on malformed LaTeX.
 */
export default function MathText({
  expr,
  display = false,
}: {
  expr: string;
  display?: boolean;
}) {
  let html: string;
  try {
    html = katex.renderToString(expr, {
      displayMode: display,
      throwOnError: false,
      output: "html",
    });
  } catch {
    // Defensive: if KaTeX throws despite throwOnError:false, fall back to the
    // raw expression wrapped in the original delimiters so nothing is lost.
    return (
      <span className="font-mono text-textSecondary">
        {display ? `$$${expr}$$` : `$${expr}$`}
      </span>
    );
  }
  const Tag = display ? "div" : "span";
  return (
    <Tag
      className={display ? "my-2 overflow-x-auto" : undefined}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
