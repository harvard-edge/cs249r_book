"use client";

import Link from "next/link";
import { BookOpen, Github, Target, ArrowLeft, Layers, Package, Users, Crosshair, Calendar, Shuffle } from "lucide-react";
import { LEVELS } from "@/lib/levels";
import { getQuestions } from "@/lib/corpus";
import {
  QUESTION_COUNT_FORMATTED,
  TOPIC_COUNT,
  TRACK_COUNT,
  LEVEL_COUNT,
  VERSION,
  BUILD_DATE,
} from "@/lib/stats";
import PaperCitationCard from "@/components/PaperCitationCard";

const PAPER_URL = "https://mlsysbook.ai/staffml/downloads/StaffML-Paper.pdf";

// Release metadata is baked at build time. These are set by `vault publish`
// via NEXT_PUBLIC_VAULT_RELEASE (ARCHITECTURE.md §7.1 cutover contract) and
// are the citable identity of the corpus snapshot this bundle serves.
const RELEASE_ID = process.env.NEXT_PUBLIC_VAULT_RELEASE ?? VERSION ?? "0.9.0";
const RELEASE_HASH = process.env.NEXT_PUBLIC_VAULT_RELEASE_HASH;

export default function AboutPage() {
  // Pick a sample question to show on the page. Prefer the hand-picked L2
  // "Ridge Point Logic" as the canonical intro example; fall back to any L2
  // with napkin math if the corpus has drifted.
  const allQs = getQuestions();
  const sampleQ = allQs.find(q => q.id === 'global-0003')
    || allQs.find(q => q.level === 'L2' && q.details.napkin_math);

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-2xl mx-auto px-6 py-12">
        {/* Back */}
        <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-textTertiary hover:text-textSecondary transition-colors mb-8">
          <ArrowLeft className="w-3.5 h-3.5" /> Back to Vault
        </Link>

        {/* ─── Hero ─── */}
        <h1 className="text-3xl font-extrabold text-textPrimary tracking-tight mb-3">About StaffML</h1>
        <p className="text-[16px] text-textSecondary leading-relaxed mb-2">
          <strong className="text-textPrimary">{QUESTION_COUNT_FORMATTED} physics-grounded ML systems questions</strong>{' '}
          across {TOPIC_COUNT} topics and {TRACK_COUNT} deployment tracks, at {LEVEL_COUNT} difficulty levels
          from recall to system design. Backed by a 600-page open textbook. Free,
          open source, and runs entirely in your browser.
        </p>
        <div className="flex flex-wrap items-center gap-2 mt-4 mb-8">
          <span className="text-[11px] px-2.5 py-1 rounded-full border border-accentGreen/30 bg-accentGreen/5 text-accentGreen font-medium">No accounts</span>
          <span className="text-[11px] px-2.5 py-1 rounded-full border border-accentGreen/30 bg-accentGreen/5 text-accentGreen font-medium">No tracking</span>
          <span className="text-[11px] px-2.5 py-1 rounded-full border border-accentGreen/30 bg-accentGreen/5 text-accentGreen font-medium">100% free</span>
          <span className="text-[11px] px-2.5 py-1 rounded-full border border-accentGreen/30 bg-accentGreen/5 text-accentGreen font-medium">Open source</span>
        </div>

        {/* ─── Paper prominence (Phase 6 — above the fold) ─── */}
        {/* Academic readers land here wanting to cite the work. Card sits
            above the Numbers banner so the paper link + BibTeX are visible
            without scrolling (ARCHITECTURE.md §9, v2.1 closure of H-19 with
            evidence: after cutover, instrument About→Paper click-rate). */}
        <section className="mb-8">
          <PaperCitationCard
            paperUrl={PAPER_URL}
            releaseId={RELEASE_ID}
            releaseHash={RELEASE_HASH}
          />
        </section>

        {/* ─── Numbers banner ─── */}
        {/* The four numbers every visitor should see before they scroll.
            Derived from vault-manifest.json via src/lib/stats.ts, so they
            update automatically with each corpus regeneration. */}
        <section className="mb-10">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <StatCard value={QUESTION_COUNT_FORMATTED} label="questions" accent="blue" />
            <StatCard value={String(TOPIC_COUNT)} label="topics" accent="amber" />
            <StatCard value={String(TRACK_COUNT)} label="tracks" accent="green" />
            <StatCard value={String(LEVEL_COUNT)} label="levels" accent="purple" />
          </div>
        </section>

        {/* ─── Personal note (kept verbatim — this is the voice of the project) ─── */}
        <section className="mb-10">
          <div className="p-5 rounded-xl border border-borderSubtle bg-surface/50">
            <p className="text-[14px] text-textSecondary leading-relaxed mb-3 italic">
              &ldquo;Every semester, students come to my office hours with the same question:
              how do I prepare for ML interviews? Not the modeling side &mdash; the
              infrastructure. The compute, the memory, the hardware, the deployment.
              These interviews expect you to reason about all of it, often with numbers,
              on the spot. And most people aren&apos;t ready.
            </p>
            <p className="text-[14px] text-textSecondary leading-relaxed mb-3 italic">
              That&apos;s why I built StaffML. I wanted to give students and engineers
              a way to find out what they really know, and what they still need to
              learn. The questions come straight from the{' '}
              <a href="https://mlsysbook.ai" target="_blank" rel="noopener noreferrer" className="text-accentBlue hover:underline">Machine Learning Systems</a>{' '}
              textbook, but a textbook teaches you concepts one at a time, and quizzes
              test whether you remember them. Interviews challenge you to connect
              concepts across the entire system stack. StaffML does
              just that.
            </p>
            <p className="text-[14px] text-textSecondary leading-relaxed mb-4 italic">
              It&apos;s free because interview prep is just another form of education,
              and education works best when it&apos;s free and open access, and the world needs more
              AI engineers.&rdquo;
            </p>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-accentBlue/10 border border-accentBlue/20 flex items-center justify-center shrink-0">
                <span className="text-sm font-bold text-accentBlue">VR</span>
              </div>
              <div>
                <span className="text-[13px] font-bold text-textPrimary block">Vijay Janapa Reddi</span>
                <span className="text-[11px] text-textTertiary">Professor, Harvard University</span>
              </div>
            </div>
          </div>
        </section>

        {/* ─── What makes this different (promoted: this is the "why believe the scale" section) ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3">What makes StaffML different</h2>
          <div className="space-y-3">
            {[
              { title: "Textbook-grounded, not scraped", desc: `Every question traces back to a specific chapter of the Machine Learning Systems textbook. You are learning the ${TOPIC_COUNT} concepts a curriculum designer chose, not whatever the internet happened to cough up.` },
              { title: "Real hardware, real numbers", desc: "When a question asks about memory bandwidth, the numbers come from actual H100, A100, and Jetson datasheets. The math works on real silicon, not on round numbers that make the answer tidy." },
              { title: "Systems reasoning, not trivia", desc: "Questions ask you to estimate, diagnose, compare tradeoffs, and architect — the same skills tested in Staff+ interview loops. Nothing asks you to recite a definition." },
              { title: "Napkin math with feedback", desc: "Type your calculation, then compare it against the model answer. The app tells you whether you are in the right ballpark or off by a factor of ten." },
              { title: "Independently verified", desc: "Every question passed a second-pass math check by a separate model. The initial verification pass flagged an 8.3% error rate across the corpus. All flagged errors were corrected." },
            ].map(({ title, desc }) => (
              <div key={title} className="flex gap-3 items-start">
                <div className="w-1.5 h-1.5 rounded-full bg-accentBlue mt-2 shrink-0" />
                <div>
                  <span className="text-[13px] font-bold text-textPrimary">{title}</span>
                  <p className="text-[12px] text-textSecondary mt-0.5 leading-relaxed">{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ─── The textbook (compressed to a single epigraph + link) ─── */}
        <section className="mb-10">
          <div className="border-l-2 border-accentBlue/30 pl-4 py-1">
            <p className="text-[14px] text-textSecondary italic mb-2">
              AI is not magic &mdash; it is infrastructure, and infrastructure has laws.
            </p>
            <p className="text-[12px] text-textTertiary">
              StaffML is part of the{' '}
              <a href="https://mlsysbook.ai" target="_blank" rel="noopener noreferrer" className="text-accentBlue hover:underline font-medium">Machine Learning Systems</a>{' '}
              curriculum at Harvard University. Every topic links back to its source chapter.
            </p>
          </div>
        </section>

        {/* ─── Sample Question (moved up so the hook fires before the taxonomy deep-dive) ─── */}
        {sampleQ && (
          <section className="mb-10">
            <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
              <Target className="w-4.5 h-4.5 text-accentAmber" /> Try a Question
            </h2>
            <div className="p-4 rounded-xl border border-borderSubtle bg-surface/50">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border border-border bg-background text-textTertiary">{sampleQ.level}</span>
                <span className="text-[10px] font-mono text-textTertiary capitalize">{sampleQ.competency_area}</span>
                <span className="text-[10px] font-mono text-textMuted">{sampleQ.track}</span>
              </div>
              <p className="text-[14px] font-bold text-textPrimary mb-2">{sampleQ.title}</p>
              <p className="text-[13px] text-textSecondary leading-relaxed mb-3">{sampleQ.scenario.slice(0, 250)}{sampleQ.scenario.length > 250 ? '...' : ''}</p>
              <Link
                href={`/practice?q=${sampleQ.id}`}
                className="inline-flex items-center gap-1.5 text-sm font-bold text-accentBlue hover:underline"
              >
                Try this question →
              </Link>
            </div>
          </section>
        )}

        {/* ─── Who is this for? ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <Users className="w-4.5 h-4.5 text-accentBlue" /> Who is this for?
          </h2>
          <div className="space-y-2">
            {[
              { who: "Preparing for your first ML role?", action: "Start with L1–L2 recall questions to build your foundation.", href: "/practice?level=L1", cta: "Start Easy" },
              { who: "Working engineer targeting Staff+?", action: "Jump to L4–L6+ questions. Try the Mock Interview.", href: "/gauntlet", cta: "Mock Interview" },
              { who: "Short on time?", action: "Do the Daily Challenge — 3 questions, 5 minutes, same for everyone.", href: "/practice?daily=1", cta: "Daily Challenge" },
              { who: "Just curious about ML systems?", action: "Browse the question bank to see what the field looks like.", href: "/", cta: "Browse" },
            ].map(({ who, action, href, cta }) => (
              <div key={who} className="flex items-start gap-3 p-3 rounded-lg border border-borderSubtle bg-surface/50">
                <div className="flex-1">
                  <span className="text-[13px] font-bold text-textPrimary">{who}</span>
                  <p className="text-[12px] text-textTertiary mt-0.5">{action}</p>
                </div>
                <Link href={href} className="text-[11px] font-bold text-accentBlue hover:underline shrink-0 mt-1">{cta} →</Link>
              </div>
            ))}
          </div>
        </section>

        {/* ─── How questions are organized ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <Layers className="w-4.5 h-4.5 text-accentAmber" /> How Questions Are Organized
          </h2>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-4">
            Every question is tagged by <strong className="text-textPrimary">difficulty</strong> ({LEVEL_COUNT} levels
            from recall to system design), <strong className="text-textPrimary"> competency zone</strong> (what
            kind of thinking it tests), and <strong className="text-textPrimary">deployment track</strong> (Cloud,
            Edge, Mobile, or TinyML).
          </p>

          <h3 className="text-[13px] font-bold text-textPrimary mb-3">Difficulty Levels</h3>
          <div className="space-y-2.5 mb-6 ml-1">
            {LEVELS.map((level) => (
              <div key={level.id} className="flex gap-3 items-start">
                <div className="w-7 h-7 rounded-md flex items-center justify-center shrink-0 mt-0.5"
                  style={{ backgroundColor: level.color + "20", border: `1px solid ${level.color}40` }}>
                  <span className="text-[10px] font-bold font-mono" style={{ color: level.color }}>{level.id}</span>
                </div>
                <div className="flex-1">
                  <span className="text-[13px] font-bold text-textPrimary">{level.name}</span>
                  <span className="text-[11px] text-textTertiary ml-2">({level.role})</span>
                  <p className="text-[12px] text-textTertiary italic mt-0.5">&ldquo;{level.example}&rdquo;</p>
                </div>
              </div>
            ))}
          </div>

          <h3 className="text-[13px] font-bold text-textPrimary mb-3">Deployment Tracks</h3>
          <div className="grid grid-cols-2 gap-2">
            {[
              { name: "Cloud", desc: "GPU clusters, large-scale training and serving" },
              { name: "Edge", desc: "On-device inference, real-time constraints" },
              { name: "Mobile", desc: "Phones, power and thermal budgets" },
              { name: "TinyML", desc: "Microcontrollers, sub-milliwatt inference" },
            ].map((track) => (
              <div key={track.name} className="p-3 bg-surface border border-borderSubtle rounded-lg">
                <span className="text-[12px] font-bold text-textPrimary block">{track.name}</span>
                <span className="text-[11px] text-textTertiary">{track.desc}</span>
              </div>
            ))}
          </div>
        </section>

        {/* ─── How Questions Are Built ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <Package className="w-4.5 h-4.5 text-accentAmber" /> How Questions Are Built
          </h2>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-3">
            StaffML questions are constructed using LLM-assisted generation with structured
            prompts grounded in the{' '}
            <a href="https://mlsysbook.ai" target="_blank" rel="noopener noreferrer" className="text-accentBlue hover:underline">Machine Learning Systems</a>{' '}
            textbook and the{' '}
            <a href="https://github.com/harvard-edge/cs249r_book/tree/main/mlsysim" target="_blank" rel="noopener noreferrer" className="text-accentBlue hover:underline">MLSysIM</a>{' '}
            physics engine. Every hardware specification traces back to a centralized constants
            table maintained alongside the textbook.
          </p>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-3">
            Every question undergoes independent math verification by a separate model
            that rechecks all arithmetic and hardware specs. The initial
            verification pass found an 8.3% error rate across the corpus. All identified
            errors were corrected.
          </p>
          {/* Paper link is intentionally a low-key inline reference here, not
              a duplicate CTA card. The full citation card with PDF + BibTeX
              already sits above the fold (see <PaperCitationCard /> at the
              top of this page). Repeating the card created two competing
              "read the paper" surfaces; the inline link preserves the
              in-context pointer (this section IS the methodology) without
              the visual duplication. */}
          <p className="text-[14px] text-textSecondary leading-relaxed mb-4">
            The full methodology — backward design from textbook chapters, four-axis taxonomy,
            LLM-assisted generation pipeline, independent math verification, and the
            ikigai-inspired competency zone framework — is described in{' '}
            <a href={PAPER_URL} target="_blank" rel="noopener noreferrer" className="text-accentBlue hover:underline font-medium">our paper</a>.
          </p>
          <div className="p-4 rounded-xl border border-accentAmber/20 bg-accentAmber/5">
            <p className="text-[13px] text-textSecondary leading-relaxed">
              <strong className="text-textPrimary">Found an error?</strong>{' '}
              We take correctness seriously. If you spot a wrong number, a broken
              calculation, or a misleading scenario,{' '}
              <a href="https://github.com/harvard-edge/cs249r_book/issues/new?labels=staffml,bug&template=staffml-error.md&title=[StaffML]+Math+error+in+question+ID" target="_blank" rel="noopener noreferrer" className="text-accentBlue hover:underline font-medium">
                open an issue on GitHub
              </a>. Community verification is how we keep improving.
            </p>
          </div>
        </section>

        {/* ─── Open Source ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <Github className="w-4.5 h-4.5 text-textSecondary" /> Open Source
          </h2>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-3">
            The entire question corpus, taxonomy, and web application are open source.
            Contributions, feedback, and corrections are welcome.
          </p>
          <a
            href="https://github.com/harvard-edge/cs249r_book"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 bg-surface border border-border text-textSecondary hover:text-textPrimary rounded-lg text-sm font-medium transition-colors"
          >
            <Github className="w-4 h-4" /> View on GitHub
          </a>
          <p className="text-[11px] text-textTertiary font-mono mt-4">
            v{VERSION} &middot; built {BUILD_DATE.slice(0, 10)}
          </p>
        </section>

        {/* ─── Bottom CTA ─── */}
        {/* "Try a Random Question" is the primary CTA because it's the lowest-
            activation action on the page: no filter to pick, no level to
            choose, no commitment. A visitor who reaches the bottom of the
            About page and doesn't know what to do next gets a single button
            that puts them in front of a real question instantly. */}
        <div className="border-t border-border pt-8 pb-4 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/practice?random=1"
            className="inline-flex items-center gap-2 px-6 py-3 bg-accentBlue text-white font-bold rounded-lg text-sm hover:opacity-90 transition-opacity"
          >
            <Shuffle className="w-4 h-4" /> Try a Random Question
          </Link>
          <Link
            href="/practice?daily=1"
            className="inline-flex items-center gap-2 px-5 py-3 bg-surface border border-border text-textSecondary hover:text-textPrimary font-medium rounded-lg text-sm transition-colors"
          >
            <Calendar className="w-4 h-4" /> Daily Challenge
          </Link>
          <Link
            href="/gauntlet"
            className="inline-flex items-center gap-2 px-5 py-3 bg-surface border border-border text-textSecondary hover:text-textPrimary font-medium rounded-lg text-sm transition-colors"
          >
            <Crosshair className="w-4 h-4" /> Mock Interview
          </Link>
        </div>
      </div>
    </div>
  );
}

// ─── StatCard subcomponent ───────────────────────────────────
// Shared by the numbers banner. Lives here (local) rather than in
// src/components/ because it is only used on this page and its styling
// is tightly coupled to the about-page typography scale.
function StatCard({ value, label, accent }: {
  value: string;
  label: string;
  accent: "blue" | "amber" | "green" | "purple";
}) {
  const accentClass = {
    blue: "text-accentBlue",
    amber: "text-accentAmber",
    green: "text-accentGreen",
    purple: "text-accentPurple",
  }[accent];

  return (
    <div className="p-4 rounded-xl border border-borderSubtle bg-surface/50 text-center">
      <div className={`text-2xl font-extrabold tracking-tight ${accentClass}`}>{value}</div>
      <div className="text-[11px] font-medium text-textTertiary uppercase tracking-wider mt-1">{label}</div>
    </div>
  );
}
