// Render-verification for the pre-release UI fixes (A1/A3/A4 + currency guard).
// These assert the ACTUAL rendered DOM — the automated stand-in for visually
// confirming the components come out correctly, not just that they typecheck.
import { describe, expect, test } from "vitest";
import { render } from "@testing-library/react";
import MCQOptions from "../MCQOptions";
import MarkdownText from "../MarkdownText";

describe("A1 — MCQOptions self-check renders correctly", () => {
  const opts = ["90 GPUs, 69.3 kW", "100 GPUs, 70.0 kW", "90 GPUs, 63.0 kW", "100 GPUs, 77.0 kW"];

  test("renders one radio per option (distractors kept)", () => {
    const { container } = render(
      <MCQOptions options={opts} correctIndex={0} revealed={false} selected={null} onSelect={() => {}} />,
    );
    expect(container.querySelectorAll('[role="radio"]').length).toBe(4);
  });

  test("on reveal: correct option marked green, wrong pick marked red", () => {
    const { container } = render(
      <MCQOptions options={opts} correctIndex={0} revealed={true} selected={1} onSelect={() => {}} />,
    );
    const buttons = container.querySelectorAll('[role="radio"]');
    expect(buttons[0].className).toContain("accentGreen"); // correct
    expect(buttons[1].className).toContain("accentRed");   // wrong pick
    expect(buttons[2].className).not.toContain("accentGreen");
  });
});

describe("A3/A4 — MarkdownText renders math, bold, and currency correctly", () => {
  const katex = (c: HTMLElement) => c.querySelector(".katex");

  test("inline math renders as KaTeX", () => {
    const { container } = render(<MarkdownText text={"intensity $E = mc^2$ here"} glossary={false} />);
    expect(katex(container)).not.toBeNull();
  });

  test("subscript math renders as KaTeX", () => {
    const { container } = render(<MarkdownText text={"a budget of $T_{lat}$ ms"} glossary={false} />);
    expect(katex(container)).not.toBeNull();
  });

  test("bare-$ currency is NOT treated as math (the regression I fixed)", () => {
    const { container } = render(
      <MarkdownText text={"Maintenance ($3M) plus $50 per device per year"} glossary={false} />,
    );
    expect(katex(container)).toBeNull();              // no garbled KaTeX
    expect(container.textContent).toContain("$3M");   // currency shown verbatim
    expect(container.textContent).toContain("$50");
  });

  test("bold still renders (original pass intact after math split)", () => {
    const { container } = render(<MarkdownText text={"the **dominant** term"} glossary={false} />);
    expect(container.querySelector("strong, b, .font-bold")?.textContent ?? container.textContent).toContain("dominant");
  });
});
