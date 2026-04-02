"use client";

import Link from "next/link";
import { BookOpen, Github, Target, ArrowLeft, Layers, Package, Users, Crosshair, Calendar } from "lucide-react";
import { LEVELS } from "@/lib/levels";
import { getVaultStats } from "@/lib/taxonomy";
import { getQuestions } from "@/lib/corpus";
import manifest from "@/data/vault-manifest.json";

export default function AboutPage() {
  const stats = getVaultStats();

  // Pick a sample question to show on the page
  const allQs = getQuestions();
  const sampleQ = allQs.find(q => q.id === 'global-0003') // "The Ridge Point Logic" — a good L2 example
    || allQs.find(q => q.level === 'L2' && q.details.napkin_math);

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-2xl mx-auto px-6 py-12">
        {/* Back */}
        <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-textTertiary hover:text-textSecondary transition-colors mb-8">
          <ArrowLeft className="w-3.5 h-3.5" /> Back to Vault
        </Link>

        {/* ─── Hero ─── */}
        <h1 className="text-3xl font-extrabold text-textPrimary tracking-tight mb-2">About StaffML</h1>
        <p className="text-[15px] text-textSecondary leading-relaxed mb-3">
          {stats.totalQuestions.toLocaleString()} questions that test what ML systems interviews actually demand.
          Systems reasoning, napkin math, architectural tradeoffs — not trivia.
        </p>
        <div className="flex flex-wrap items-center gap-3 mb-10">
          <span className="text-[11px] px-2.5 py-1 rounded-full border border-accentGreen/30 bg-accentGreen/5 text-accentGreen font-medium">No accounts</span>
          <span className="text-[11px] px-2.5 py-1 rounded-full border border-accentGreen/30 bg-accentGreen/5 text-accentGreen font-medium">No tracking</span>
          <span className="text-[11px] px-2.5 py-1 rounded-full border border-accentGreen/30 bg-accentGreen/5 text-accentGreen font-medium">100% free</span>
          <span className="text-[11px] px-2.5 py-1 rounded-full border border-accentGreen/30 bg-accentGreen/5 text-accentGreen font-medium">Open source</span>
        </div>

        {/* ─── Personal note ─── */}
        <section className="mb-10">
          <div className="p-5 rounded-xl border border-borderSubtle bg-surface/50">
            <p className="text-[14px] text-textSecondary leading-relaxed mb-3 italic">
              &ldquo;Every semester, students come to my office hours with the same question:
              how do I prepare for ML interviews? ML is in everything now — the systems,
              the infrastructure, the hardware — and companies expect you to reason quantitatively
              about all of it. How much memory does a 70B model need? Why did your serving
              latency just spike? But there&apos;s no good way to practice this. Students read
              papers, take courses that teach concepts but not interview-ready thinking,
              or just hope for the best.
            </p>
            <p className="text-[14px] text-textSecondary leading-relaxed mb-4 italic">
              I&apos;ve been writing the Machine Learning Systems textbook to teach these ideas,
              but I&apos;ve come to realize that reading about systems and being ready to reason
              under pressure are different things. Sometimes the most valuable thing is discovering
              what you don&apos;t actually know — that&apos;s what motivates you to go learn it.
            </p>
            <p className="text-[14px] text-textSecondary leading-relaxed mb-4 italic">
              That&apos;s why I built StaffML. It&apos;s free because interview prep is
              just another form of education — and education shouldn&apos;t depend
              on what you can afford.&rdquo;
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

        {/* ─── Sample Question ─── */}
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

        {/* ─── What makes this different ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3">What makes StaffML different?</h2>
          <div className="space-y-3">
            {[
              { title: "Textbook-grounded", desc: "Every question traces back to a chapter of the Machine Learning Systems textbook. You're learning real concepts, not memorizing disconnected facts." },
              { title: "Real hardware specs", desc: "When a question asks about memory bandwidth, the numbers come from actual H100/A100 datasheets. The math works on real silicon." },
              { title: "Systems reasoning, not trivia", desc: "Questions ask you to estimate, diagnose, compare tradeoffs, and architect — the same skills tested in Staff+ interview loops." },
              { title: "Napkin math with feedback", desc: "Type your calculation, then compare against the model answer. The app tells you if you're in the right ballpark or way off." },
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

        {/* ─── The Book ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <BookOpen className="w-4.5 h-4.5 text-accentBlue" /> Built on the Textbook
          </h2>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-3">
            StaffML is part of the <a href="https://mlsysbook.ai" target="_blank" rel="noopener noreferrer" className="text-accentBlue hover:underline font-medium">Machine Learning Systems</a> curriculum
            at Harvard University. Every topic links back to its source chapter, so you can learn the concept first and then test yourself.
          </p>
          <p className="text-[13px] text-textTertiary italic">
            AI is not magic — it is infrastructure, and infrastructure has laws.
          </p>
        </section>

        {/* ─── How questions are organized (collapsed detail) ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <Layers className="w-4.5 h-4.5 text-accentAmber" /> How Questions Are Organized
          </h2>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-4">
            Every question is tagged by <strong className="text-textPrimary">difficulty</strong> (6 levels from recall to system design),
            <strong className="text-textPrimary"> competency zone</strong> (what kind of thinking it tests),
            and <strong className="text-textPrimary"> deployment track</strong> (Cloud, Edge, Mobile, or TinyML).
          </p>

          <details className="group mb-3">
            <summary className="text-[12px] font-medium text-accentBlue cursor-pointer hover:underline">
              Show difficulty levels
            </summary>
            <div className="space-y-2.5 mt-3 ml-1">
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
          </details>

          <details className="group">
            <summary className="text-[12px] font-medium text-accentBlue cursor-pointer hover:underline">
              Show deployment tracks
            </summary>
            <div className="grid grid-cols-2 gap-2 mt-3">
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
          </details>
        </section>

        {/* ─── Open Source + Research ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <Github className="w-4.5 h-4.5 text-textSecondary" /> Open Source
          </h2>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-3">
            The entire question corpus, taxonomy, and web application are open source.
            Contributions, feedback, and corrections are welcome.
          </p>
          <div className="flex flex-wrap items-center gap-3 mb-4">
            <a
              href="https://github.com/harvard-edge/cs249r_book"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 bg-surface border border-border text-textSecondary hover:text-textPrimary rounded-lg text-sm font-medium transition-colors"
            >
              <Github className="w-4 h-4" /> View on GitHub
            </a>
          </div>
          <p className="text-[11px] text-textTertiary font-mono">
            v{manifest.version} &middot; {manifest.questionCount.toLocaleString()} questions &middot; {manifest.conceptCount} topics &middot; built {manifest.buildDate.slice(0, 10)}
          </p>
        </section>

        {/* CTA */}
        <div className="border-t border-border pt-8 pb-4 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/practice"
            className="inline-flex items-center gap-2 px-6 py-3 bg-accentBlue text-white font-bold rounded-lg text-sm hover:opacity-90 transition-opacity"
          >
            <Target className="w-4 h-4" /> Start Practicing
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
