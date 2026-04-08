"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Send, Terminal, CheckCircle2, Github, Copy, Check } from "lucide-react";
import clsx from "clsx";
import { getAreas } from "@/lib/taxonomy";
import { buildContributeUrl } from "@/lib/issue-url";
import { track as trackEvent } from "@/lib/analytics";
import { useToast } from "@/components/Toast";

const TRACKS = ['cloud', 'edge', 'mobile', 'tinyml'];
const LEVELS = [
  { id: 'L1', name: 'Remember', desc: 'Recall facts and definitions' },
  { id: 'L2', name: 'Understand', desc: 'Explain concepts and tradeoffs' },
  { id: 'L3', name: 'Apply', desc: 'Use knowledge in new situations' },
  { id: 'L4', name: 'Analyze', desc: 'Break down systems, diagnose issues' },
  { id: 'L5', name: 'Evaluate', desc: 'Compare architectures, justify decisions' },
  { id: 'L6+', name: 'Create', desc: 'Design novel systems under constraints' },
];
const ZONES = [
  { id: 'recall', desc: 'Facts & definitions' },
  { id: 'analyze', desc: 'Tradeoffs & reasoning' },
  { id: 'design', desc: 'Architecture decisions' },
  { id: 'implement', desc: 'Napkin math & building' },
  { id: 'diagnosis', desc: 'Identify and explain failures' },
  { id: 'fluency', desc: 'Math from memory' },
  { id: 'evaluation', desc: 'Compare architectures' },
  { id: 'optimization', desc: 'Diagnose and fix bottlenecks' },
];

const STORAGE_KEY = 'staffml_contributions';

// Hard cap on the number of locally-stored contribution drafts. Without this,
// repeated submissions (or a malicious script spamming the form) would fill
// the ~5 MB localStorage quota, raising QuotaExceededError on every subsequent
// write across the app (theme switching, attempt logging, etc.) and bricking
// it. We keep the most recent 50 — anything older is dropped (LRU style).
const MAX_CONTRIBUTIONS = 50;

interface Contribution {
  id: string;
  track: string;
  level: string;
  zone: string;
  topic: string;
  scenario: string;
  answer: string;
  commonMistake: string;
  napkinMath: string;
  submittedAt: number;
}

function saveContribution(c: Contribution): void {
  try {
    const raw = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '[]');
    const all: Contribution[] = Array.isArray(raw) ? raw : [];
    all.push(c);
    // Drop oldest entries until we're under the cap.
    while (all.length > MAX_CONTRIBUTIONS) all.shift();
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  } catch {}
}

function getContributions(): Contribution[] {
  try {
    return JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
}

export default function ContributePage() {
  const [mounted, setMounted] = useState(false);
  const { show: showToast } = useToast();

  // Form state
  const [track, setTrack] = useState('cloud');
  const [level, setLevel] = useState('L3');
  const [zone, setZone] = useState('recall');
  const [topic, setTopic] = useState('');
  const [scenario, setScenario] = useState('');
  const [answer, setAnswer] = useState('');
  const [commonMistake, setCommonMistake] = useState('');
  const [napkinMath, setNapkinMath] = useState('');

  // UI state
  const [submitted, setSubmitted] = useState(false);
  const [pastCount, setPastCount] = useState(0);
  const [copied, setCopied] = useState(false);

  const areas = getAreas();
  const allTopics = areas.flatMap(a => a.topics.map(t => ({ id: t.id, name: t.name, area: a.name })));

  useEffect(() => {
    setMounted(true);
    setPastCount(getContributions().length);
  }, []);

  const canSubmit = scenario.trim().length > 20 && answer.trim().length > 20;

  const handleSubmit = () => {
    if (!canSubmit) return;

    const contribution: Contribution = {
      id: `contrib-${Date.now()}`,
      track, level, zone,
      topic: topic || 'uncategorized',
      scenario: scenario.trim(),
      answer: answer.trim(),
      commonMistake: commonMistake.trim(),
      napkinMath: napkinMath.trim(),
      submittedAt: Date.now(),
    };

    saveContribution(contribution);
    analyticsTrack(contribution);
    setSubmitted(true);
    setPastCount(p => p + 1);
    showToast({ type: 'success', title: 'Question saved!', description: 'Submit via GitHub to add it to the vault.' });
  };

  const analyticsTrack = (c: Contribution) => {
    trackEvent({ type: 'question_contributed', topic: c.topic, track: c.track });
  };

  const exportAsGitHubBody = (): string => {
    return [
      `**Track:** ${track}`,
      `**Level:** ${level}`,
      `**Zone:** ${zone}`,
      topic ? `**Topic:** ${topic}` : '',
      '',
      '### Scenario',
      scenario,
      '',
      '### Expected Answer',
      answer,
      '',
      commonMistake ? `### Common Mistake\n${commonMistake}\n` : '',
      napkinMath ? `### Napkin Math\n${napkinMath}\n` : '',
    ].filter(Boolean).join('\n');
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(exportAsGitHubBody());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const resetForm = () => {
    setScenario('');
    setAnswer('');
    setCommonMistake('');
    setNapkinMath('');
    setSubmitted(false);
  };

  if (!mounted) {
    return <div className="flex-1 flex items-center justify-center"><Terminal className="w-6 h-6 text-textTertiary animate-pulse" /></div>;
  }

  // Success state
  if (submitted) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-16">
        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="max-w-lg w-full text-center">
          <div className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6 border-2 border-accentGreen bg-accentGreen/10">
            <CheckCircle2 className="w-8 h-8 text-accentGreen" />
          </div>
          <h2 className="text-2xl font-bold text-textPrimary mb-2">Question Saved Locally</h2>
          <p className="text-sm text-textSecondary mb-6">
            To add it to the StaffML vault, submit it as a GitHub issue. Your question data has been saved to your browser.
          </p>
          <div className="flex items-center gap-3 justify-center flex-wrap">
            <a
              href={buildContributeUrl()}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-textPrimary text-background font-bold rounded-lg hover:opacity-90 transition-all text-sm"
            >
              <Github className="w-4 h-4" /> Submit on GitHub
            </a>
            <button
              onClick={copyToClipboard}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-surface border border-border text-textSecondary hover:text-textPrimary rounded-lg transition-colors text-sm"
            >
              {copied ? <><Check className="w-4 h-4" /> Copied!</> : <><Copy className="w-4 h-4" /> Copy as Markdown</>}
            </button>
            <button
              onClick={resetForm}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-surface border border-border text-textSecondary hover:text-textPrimary rounded-lg transition-colors text-sm"
            >
              Submit Another
            </button>
          </div>
          {pastCount > 1 && (
            <p className="text-[10px] font-mono text-textTertiary mt-6">{pastCount} questions contributed from this browser</p>
          )}
        </motion.div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col px-6 py-10">
      <div className="max-w-2xl mx-auto w-full">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          {/* Header */}
          <div className="flex items-center gap-3 mb-2">
            <Send className="w-7 h-7 text-accentBlue" />
            <h1 className="text-3xl font-extrabold text-textPrimary tracking-tight">Contribute a Question</h1>
          </div>
          <p className="text-sm text-textSecondary mb-8">
            Help build the ML systems interview question vault. Good questions are physics-grounded, quantitative, and test systems reasoning — not trivia.
          </p>

          {/* Classification row */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
            {/* Track */}
            <div>
              <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Track</label>
              <select
                value={track}
                onChange={(e) => setTrack(e.target.value)}
                className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-textPrimary focus:outline-none focus:border-accentBlue/50"
              >
                {TRACKS.map(t => (
                  <option key={t} value={t}>{t === 'tinyml' ? 'TinyML' : t.charAt(0).toUpperCase() + t.slice(1)}</option>
                ))}
              </select>
            </div>

            {/* Level */}
            <div>
              <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Difficulty</label>
              <select
                value={level}
                onChange={(e) => setLevel(e.target.value)}
                className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-textPrimary focus:outline-none focus:border-accentBlue/50"
              >
                {LEVELS.map(l => (
                  <option key={l.id} value={l.id}>{l.id} — {l.name}</option>
                ))}
              </select>
            </div>

            {/* Zone */}
            <div>
              <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Zone</label>
              <select
                value={zone}
                onChange={(e) => setZone(e.target.value)}
                className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-textPrimary focus:outline-none focus:border-accentBlue/50"
              >
                {ZONES.map(z => (
                  <option key={z.id} value={z.id}>{z.id} — {z.desc}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Topic */}
          <div className="mb-6">
            <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Topic (pick existing or type new)</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              list="topic-list"
              placeholder="e.g., roofline-analysis, kv-cache-management"
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-textPrimary focus:outline-none focus:border-accentBlue/50 placeholder:text-textTertiary/40"
            />
            <datalist id="topic-list">
              {allTopics.map(t => (
                <option key={t.id} value={t.id}>{t.name} ({t.area})</option>
              ))}
            </datalist>
          </div>

          {/* Scenario */}
          <div className="mb-6">
            <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">
              Interview Scenario <span className="text-accentRed">*</span>
            </label>
            <textarea
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              placeholder="The question as an interviewer would ask it. Be specific about hardware, constraints, and context."
              className="w-full min-h-[120px] bg-surface border border-border rounded-lg px-4 py-3 text-sm text-textPrimary font-mono resize-y focus:outline-none focus:border-accentBlue/50 placeholder:text-textTertiary/40 leading-relaxed"
              spellCheck="false"
            />
            <span className="text-[10px] text-textMuted mt-1 block">{scenario.length} chars (min 20)</span>
          </div>

          {/* Expected Answer */}
          <div className="mb-6">
            <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">
              Expected Answer <span className="text-accentRed">*</span>
            </label>
            <textarea
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              placeholder="The model answer with reasoning and real numbers. Explain WHY, not just WHAT."
              className="w-full min-h-[150px] bg-surface border border-border rounded-lg px-4 py-3 text-sm text-textPrimary font-mono resize-y focus:outline-none focus:border-accentBlue/50 placeholder:text-textTertiary/40 leading-relaxed"
              spellCheck="false"
            />
            <span className="text-[10px] text-textMuted mt-1 block">{answer.length} chars (min 20)</span>
          </div>

          {/* Common Mistake (optional) */}
          <div className="mb-6">
            <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Common Mistake (optional)</label>
            <textarea
              value={commonMistake}
              onChange={(e) => setCommonMistake(e.target.value)}
              placeholder="What do candidates typically get wrong?"
              className="w-full min-h-[80px] bg-surface border border-border rounded-lg px-4 py-3 text-sm text-textPrimary font-mono resize-y focus:outline-none focus:border-accentBlue/50 placeholder:text-textTertiary/40 leading-relaxed"
              spellCheck="false"
            />
          </div>

          {/* Napkin Math (optional) */}
          <div className="mb-8">
            <label className="text-[10px] font-mono text-textTertiary uppercase tracking-widest block mb-2">Napkin Math (optional)</label>
            <textarea
              value={napkinMath}
              onChange={(e) => setNapkinMath(e.target.value)}
              placeholder={"H100 bandwidth: 3.35 TB/s\nModel size (FP16): 140 GB\nLoad time: 140 / 3350 = 42 ms\n=> 42 ms minimum latency"}
              className="w-full min-h-[100px] bg-surface border border-border rounded-lg px-4 py-3 text-sm text-textPrimary font-mono resize-y focus:outline-none focus:border-accentBlue/50 placeholder:text-textTertiary/40 leading-relaxed"
              spellCheck="false"
            />
          </div>

          {/* Submit */}
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-textTertiary">
              Saved locally first, then submit via GitHub.
              {pastCount > 0 && ` (${pastCount} past contributions)`}
            </span>
            <button
              onClick={handleSubmit}
              disabled={!canSubmit}
              className="inline-flex items-center gap-2 px-6 py-3 bg-textPrimary text-background font-bold rounded-lg hover:opacity-90 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Save & Continue <Send className="w-4 h-4" />
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
