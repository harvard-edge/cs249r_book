"use client";

/**
 * Renders the diagram attached to a question.
 *
 * Architecture: visuals live as static SVG assets under
 * `interviews/vault/visuals/<track>/` and are mirrored to
 * `interviews/staffml/public/question-visuals/<track>/` at bundle
 * build time (`vault build --legacy-json`). Here we just load them
 * as `<img>` elements — browser caching, no inline-SVG sanitization,
 * and the HTTP request is a single static asset hit.
 *
 * v0.1.2: schema hardened (kind=svg only, caption required ≥5 chars,
 * alt required ≥10 chars). Click the image to open a fullscreen zoom
 * modal (react-medium-image-zoom) — ESC closes. The modal is the
 * accessibility-friendly way to inspect dense systems diagrams that
 * don't fit the practice column at default size.
 *
 * The component is deliberately minimal. The SVG itself carries the
 * semantic content (see `.claude/rules/svg-style.md` for the book's
 * SVG style system, which authors should follow). Styling of the
 * frame, caption, and responsive sizing lives here.
 */

import { useState } from "react";
import { ImageOff } from "lucide-react";
import Zoom from "react-medium-image-zoom";
import "react-medium-image-zoom/dist/styles.css";

export interface QuestionVisualProps {
  track: string;
  visual: {
    kind: "svg";
    path: string;
    alt: string;
    caption: string;
  };
}

export default function QuestionVisual({ track, visual }: QuestionVisualProps) {
  const [failed, setFailed] = useState(false);

  if (visual.kind !== "svg") return null;

  const src = `/question-visuals/${track}/${visual.path}`;

  if (failed) {
    return (
      <figure className="mt-5 p-4 rounded-lg border border-dashed border-accentRed/30 bg-accentRed/5 flex items-start gap-3">
        <ImageOff className="w-4 h-4 text-accentRed shrink-0 mt-0.5" />
        <div className="text-sm text-textSecondary">
          <span className="font-medium text-accentRed">Diagram failed to load.</span>{" "}
          {visual.alt}
        </div>
      </figure>
    );
  }

  return (
    <figure className="mt-5" data-testid="question-visual">
      <div className="rounded-lg border border-border bg-surface/40 p-3 flex items-center justify-center">
        <Zoom
          zoomMargin={32}
          classDialog="question-visual-zoom-dialog"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt={visual.alt}
            data-testid="question-visual-img"
            className="max-w-full max-h-[420px] h-auto w-auto cursor-zoom-in"
            onError={() => setFailed(true)}
            loading="lazy"
          />
        </Zoom>
      </div>
      <figcaption className="mt-2 text-[11px] font-mono text-textTertiary text-center">
        {visual.caption}
      </figcaption>
    </figure>
  );
}
