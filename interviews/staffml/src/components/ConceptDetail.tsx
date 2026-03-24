"use client";

import { X, ArrowRight, ArrowLeft, BookOpen, AlertCircle } from "lucide-react";
import clsx from "clsx";
import {
  type Concept,
  getPrereqs,
  getDependents,
  formatChapter,
} from "@/lib/taxonomy";

const LEVELS_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6+"];

interface Props {
  concept: Concept;
  onClose: () => void;
  onNavigate: (id: string) => void;
}

export default function ConceptDetail({ concept, onClose, onNavigate }: Props) {
  const prereqs = getPrereqs(concept.id);
  const dependents = getDependents(concept.id);

  return (
    <div className="h-full flex flex-col bg-surface border-l border-border overflow-auto">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <h2 className="text-base font-bold text-white leading-tight">
              {concept.name}
            </h2>
            <span className="text-[10px] font-mono text-textTertiary">
              {concept.id}
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-textTertiary hover:text-white transition-colors shrink-0"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Role badge + question count */}
        <div className="flex items-center gap-2 mt-2">
          <span
            className={clsx(
              "inline-block px-2 py-0.5 rounded text-[10px] font-medium",
              concept.role === "foundational"
                ? "bg-accentBlue/20 text-accentBlue"
                : concept.role === "competency"
                ? "bg-accentGreen/20 text-accentGreen"
                : "bg-surface text-textTertiary border border-border"
            )}
          >
            {concept.role}
          </span>
          <span
            className={clsx(
              "text-xs font-mono",
              concept.question_count === 0 ? "text-accentRed" : "text-textSecondary"
            )}
          >
            {concept.question_count === 0 ? (
              <span className="flex items-center gap-1">
                <AlertCircle className="w-3 h-3" /> 0 questions
              </span>
            ) : (
              `${concept.question_count} questions`
            )}
          </span>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-auto p-4 space-y-5">
        {/* Description */}
        <div>
          <h3 className="text-[10px] font-mono text-textTertiary uppercase tracking-wider mb-1">
            Description
          </h3>
          <p className="text-sm text-textSecondary leading-relaxed">
            {concept.description}
          </p>
        </div>

        {/* Level distribution */}
        {concept.question_count > 0 && (
          <div>
            <h3 className="text-[10px] font-mono text-textTertiary uppercase tracking-wider mb-2">
              Questions by Level
            </h3>
            <div className="flex gap-1">
              {LEVELS_ORDER.map((level) => {
                const count = concept.level_distribution[level] || 0;
                return (
                  <div
                    key={level}
                    className="flex-1 text-center"
                  >
                    <div
                      className={clsx(
                        "rounded-md py-1 text-[10px] font-mono border",
                        count > 0
                          ? "bg-accentBlue/10 border-accentBlue/30 text-accentBlue"
                          : "bg-surface border-border text-textTertiary"
                      )}
                    >
                      {count}
                    </div>
                    <div className="text-[9px] text-textTertiary mt-0.5">
                      {level}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Tracks */}
        <div>
          <h3 className="text-[10px] font-mono text-textTertiary uppercase tracking-wider mb-1">
            Tracks
          </h3>
          <div className="flex flex-wrap gap-1">
            {concept.tracks.map((t) => (
              <span
                key={t}
                className="inline-block bg-surface border border-border px-2 py-0.5 rounded text-xs text-textSecondary"
              >
                {t === "tinyml" ? "TinyML" : t.charAt(0).toUpperCase() + t.slice(1)}
              </span>
            ))}
          </div>
        </div>

        {/* Chapters */}
        <div>
          <h3 className="text-[10px] font-mono text-textTertiary uppercase tracking-wider mb-1">
            Source Chapters
          </h3>
          <div className="space-y-1">
            {concept.source_chapters.map((ch) => (
              <div key={ch} className="flex items-center gap-1.5 text-xs text-textSecondary">
                <BookOpen className="w-3 h-3 text-textTertiary shrink-0" />
                {formatChapter(ch)}
              </div>
            ))}
          </div>
        </div>

        {/* Prerequisites */}
        <div>
          <h3 className="text-[10px] font-mono text-textTertiary uppercase tracking-wider mb-1">
            Prerequisites ({prereqs.length})
          </h3>
          {prereqs.length === 0 ? (
            <p className="text-xs text-textTertiary italic">No prerequisites (root concept)</p>
          ) : (
            <div className="space-y-0.5">
              {prereqs.map((p) => (
                <button
                  key={p.id}
                  onClick={() => onNavigate(p.id)}
                  className="flex items-center gap-1.5 w-full text-left px-2 py-1.5 rounded hover:bg-surfaceHover transition-colors group"
                >
                  <ArrowLeft className="w-3 h-3 text-accentBlue shrink-0" />
                  <span className="text-xs text-textSecondary group-hover:text-white truncate">
                    {p.name}
                  </span>
                  <span className="text-[10px] font-mono text-textTertiary ml-auto shrink-0">
                    {p.question_count}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Dependents */}
        <div>
          <h3 className="text-[10px] font-mono text-textTertiary uppercase tracking-wider mb-1">
            Dependents ({dependents.length})
          </h3>
          {dependents.length === 0 ? (
            <p className="text-xs text-textTertiary italic">No dependents (leaf concept)</p>
          ) : (
            <div className="space-y-0.5">
              {dependents.map((d) => (
                <button
                  key={d.id}
                  onClick={() => onNavigate(d.id)}
                  className="flex items-center gap-1.5 w-full text-left px-2 py-1.5 rounded hover:bg-surfaceHover transition-colors group"
                >
                  <ArrowRight className="w-3 h-3 text-accentGreen shrink-0" />
                  <span className="text-xs text-textSecondary group-hover:text-white truncate">
                    {d.name}
                  </span>
                  <span className="text-[10px] font-mono text-textTertiary ml-auto shrink-0">
                    {d.question_count}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
