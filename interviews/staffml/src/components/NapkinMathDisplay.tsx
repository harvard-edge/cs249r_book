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
  
  return lines.map((line) => {
    const trimmedLine = line.trim();
    // Improved result detection
    const isResult = trimmedLine.startsWith("=>") || 
                     trimmedLine.toLowerCase().includes("result:") || 
                     trimmedLine.toLowerCase().includes("conclusion:");
                     
    return {
      text: cleanStepText(trimmedLine),
      isResult
    };
  });
}

function cleanStepText(text: string): string {
  return text
    .replace(/^-\s*/, "") // strip leading dash
    .replace(/^\d+\.\s*/, "") // strip leading number
    .replace(/^=>\s*/, "") // strip =>
    .trim();
}
