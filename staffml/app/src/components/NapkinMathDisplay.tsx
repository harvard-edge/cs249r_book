"use client";

import clsx from "clsx";
import MarkdownText from "./MarkdownText";

/**
 * Renders napkin math text with structured formatting.
 */
export default function NapkinMathDisplay({ text }: { text: string }) {
  const steps = parseSteps(text);

  if (steps.length === 1 && !steps[0].isResult) {
    return (
      <div className="font-mono text-[13px] text-textSecondary leading-relaxed">
        <MarkdownText text={steps[0].text} />
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      {steps.map((step, i) => {
        const isHeader = step.text.startsWith('**') && step.text.includes(':');
        
        return (
          <div
            key={i}
            className={clsx(
              "font-mono text-[13px] leading-relaxed",
              step.isResult 
                ? "mt-4 p-3 bg-accentGreen/5 border-t border-accentGreen/20 text-accentGreen font-semibold" 
                : isHeader 
                ? "mt-4 mb-1 text-textPrimary font-bold" 
                : "pl-4 text-textSecondary relative before:content-['-'] before:absolute before:left-0 before:text-textTertiary/40"
            )}
          >
            <MarkdownText text={step.text} />
          </div>
        );
      })}
    </div>
  );
}

interface Step {
  text: string;
  isResult: boolean;
}

function parseSteps(raw: string): Step[] {
  const lines = raw.trim().split("\n").filter((l) => l.trim());

  return lines
    .map((line) => {
      const trimmedLine = line.trim();
      // A "section header" line is alone-on-its-line bold like
      // "**Conclusion & Interpretation:**". Treat that as a header
      // (bold subtitle), NOT as a result, so we don't conjure an empty
      // green callout when the actual result bullet is the next line.
      const isHeader = /^\*\*[^*]+:?\*\*\s*$/.test(trimmedLine);
      const isResult =
        !isHeader &&
        (trimmedLine.startsWith("=>") ||
          /\bresult:/i.test(trimmedLine));

      return {
        text: cleanStepText(trimmedLine),
        isResult,
      };
    })
    // After cleaning, some lines collapse to empty (e.g. a bare "**Result:**"
    // header that was stripped). Drop them rather than render empty bullets.
    .filter((step) => step.text.length > 0);
}

function cleanStepText(text: string): string {
  let out = text
    .replace(/^-\s*/, "") // strip leading dash
    .replace(/^\d+\.\s*/, "") // strip leading number
    .replace(/^=>\s*/, "") // strip =>
    .trim();

  // For result steps: strip the redundant "Conclusion & Interpretation:" and
  // "Result:" prefixes. The green callout already carries that signal, so
  // leaving the labels in produces noise like
  //   "Conclusion & Interpretation: **Result: Memory-Bound**. ..."
  // when "**Memory-Bound**. ..." is what the reader actually needs.

  // Strip "Conclusion & Interpretation:" / "Conclusion:" — bold and unbold.
  out = out
    .replace(/^\*\*conclusion(?:\s+&\s+interpretation)?:\*\*\s*/i, "")
    .replace(/^conclusion(?:\s+&\s+interpretation)?:\s*/i, "");

  // Strip the "Result:" label even when the *whole phrase* "Result: <verdict>"
  // is wrapped in one **bold** span (a common YAML pattern). Rewrite
  //   **Result: Memory-Bound**. ...   →   **Memory-Bound**. ...
  out = out.replace(/^\*\*result:\s*/i, "**");

  // Unbold form: "Result: Memory-Bound. ..." → "Memory-Bound. ..."
  out = out.replace(/^result:\s*/i, "");

  // Self-bold form: "**Result:**" already closed, then a verdict — drop the label.
  out = out.replace(/^\*\*result:?\*\*\s*/i, "");

  return out.trim();
}
