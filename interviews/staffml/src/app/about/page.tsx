"use client";

import Link from "next/link";
import { BookOpen, Github, Target, ArrowLeft, Brain, Layers, BarChart3, Package, Compass } from "lucide-react";
import { LEVELS } from "@/lib/levels";
import { getVaultStats, getZoneDefinitions } from "@/lib/taxonomy";
import manifest from "@/data/vault-manifest.json";

export default function AboutPage() {
  const stats = getVaultStats();
  const zoneDefs = getZoneDefinitions();

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-2xl mx-auto px-6 py-12">
        {/* Back */}
        <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-textTertiary hover:text-textSecondary transition-colors mb-8">
          <ArrowLeft className="w-3.5 h-3.5" /> Back to Vault
        </Link>

        {/* Header */}
        <h1 className="text-3xl font-extrabold text-textPrimary tracking-tight mb-2">About StaffML</h1>
        <p className="text-[15px] text-textSecondary leading-relaxed mb-10">
          A free, open-source interview preparation platform for ML systems engineers — built on research, not guesswork.
        </p>

        {/* ─── What ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <Target className="w-4.5 h-4.5 text-accentBlue" /> What is StaffML?
          </h2>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-3">
            StaffML is a question bank of <span className="font-mono font-semibold text-textPrimary">{stats.totalQuestions.toLocaleString()}</span> physics-grounded
            ML systems interview questions spanning <span className="font-mono font-semibold text-textPrimary">{stats.totalTopics}</span> topics
            across <span className="font-mono font-semibold text-textPrimary">{stats.totalAreas}</span> competency areas.
          </p>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-3">
            Every question is designed to test <em>systems reasoning</em>, not trivia. Questions ask you to calculate, diagnose,
            compare trade-offs, and architect solutions — the same skills evaluated in Staff+ ML engineering interviews
            at top companies.
          </p>
          <p className="text-[14px] text-textSecondary leading-relaxed">
            The platform is 100% client-side. No accounts, no tracking, no backend. Your progress lives in your browser
            and can be exported anytime.
          </p>
        </section>

        {/* ─── Methodology ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <Brain className="w-4.5 h-4.5 text-accentPurple" /> Methodology
          </h2>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-3">
            Questions are generated and validated using a research-driven pipeline:
          </p>
          <ol className="list-decimal list-inside text-[14px] text-textSecondary leading-relaxed space-y-2 ml-1">
            <li>
              <strong className="text-textPrimary">Textbook-grounded generation</strong> — Every question traces back to
              a specific chapter of the <a href="https://mlsysbook.ai" target="_blank" rel="noopener noreferrer" className="text-accentBlue hover:underline">Machine Learning Systems</a> textbook.
            </li>
            <li>
              <strong className="text-textPrimary">6-axis faceted classification</strong> — Each question is tagged across
              competency area, topic, difficulty level, deployment track, response mode, and archetype.
            </li>
            <li>
              <strong className="text-textPrimary">Napkin math validation</strong> — Quantitative questions include
              step-by-step calculations with real hardware specs (H100, A100, Cortex-M4) so answers are grounded in physics, not hand-waving.
            </li>
            <li>
              <strong className="text-textPrimary">Deduplication and quality control</strong> — The corpus is continuously
              deduplicated and reviewed for correctness, clarity, and pedagogical value.
            </li>
          </ol>
        </section>

        {/* ─── Bloom's Taxonomy ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <Layers className="w-4.5 h-4.5 text-accentAmber" /> Difficulty Levels
          </h2>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-4">
            Questions are classified using <strong className="text-textPrimary">Bloom&apos;s Taxonomy</strong>, an educational
            framework that measures <em>cognitive depth</em> — not just how hard a question feels, but what kind of thinking it requires.
          </p>
          <div className="space-y-3">
            {LEVELS.map((level) => (
              <div key={level.id} className="flex gap-3 items-start">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-0.5"
                  style={{ backgroundColor: level.color + "20", border: `1px solid ${level.color}40` }}>
                  <span className="text-[11px] font-bold font-mono" style={{ color: level.color }}>{level.id}</span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[13px] font-bold text-textPrimary">{level.name}</span>
                    <span className="text-[11px] text-textTertiary font-mono">({level.role})</span>
                  </div>
                  <p className="text-[13px] text-textSecondary leading-relaxed">{level.verb}</p>
                  <p className="text-[12px] text-textTertiary italic mt-0.5">&ldquo;{level.example}&rdquo;</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ─── Zones ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <Compass className="w-4.5 h-4.5 text-accentBlue" /> Competency Zones
          </h2>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-4">
            Every question exercises one of <strong className="text-textPrimary">11 competency zones</strong> — combinations of four fundamental skills:
            <strong className="text-textPrimary"> recall</strong> (facts), <strong className="text-textPrimary">analyze</strong> (tradeoffs),
            <strong className="text-textPrimary"> design</strong> (architecture), and <strong className="text-textPrimary">implement</strong> (napkin math).
            Zones map to interview round types: a phone screen tests <em>recall</em>, a debugging round tests <em>diagnosis</em>,
            and a system design round tests <em>evaluation</em> or <em>realization</em>.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {zoneDefs.map((z) => (
              <div key={z.id} className="flex gap-3 items-start p-3 rounded-lg border border-borderSubtle bg-surface/50">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 bg-accentBlue/10 border border-accentBlue/20">
                  <span className="text-[9px] font-bold font-mono text-accentBlue">{z.skills.length}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <span className="text-[13px] font-bold text-textPrimary capitalize">{z.id}</span>
                  <span className="text-[11px] text-textTertiary ml-2">{z.levels.join(', ')}</span>
                  <p className="text-[12px] text-textSecondary mt-0.5">{z.description}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ─── Tracks ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <BarChart3 className="w-4.5 h-4.5 text-accentGreen" /> Deployment Tracks
          </h2>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-3">
            ML systems roles aren&apos;t monolithic. Questions are tagged by deployment context so you can focus on what matters for your target role:
          </p>
          <div className="grid grid-cols-2 gap-3">
            {[
              { name: "Cloud", desc: "GPU clusters, data centers, large-scale training and serving" },
              { name: "Edge", desc: "On-device inference, latency-sensitive applications, accelerators" },
              { name: "Mobile", desc: "Phones, tablets, on-device ML with power and thermal constraints" },
              { name: "TinyML", desc: "Microcontrollers, sensors, sub-milliwatt inference" },
            ].map((track) => (
              <div key={track.name} className="p-3 bg-surface border border-borderSubtle rounded-lg">
                <span className="text-[13px] font-bold text-textPrimary block mb-1">{track.name}</span>
                <span className="text-[12px] text-textTertiary leading-relaxed">{track.desc}</span>
              </div>
            ))}
          </div>
        </section>

        {/* ─── Textbook ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <BookOpen className="w-4.5 h-4.5 text-accentBlue" /> The Book
          </h2>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-3">
            StaffML is built on <a href="https://mlsysbook.ai" target="_blank" rel="noopener noreferrer" className="text-accentBlue hover:underline font-medium">Machine Learning Systems</a>,
            a two-volume textbook by <strong className="text-textPrimary">Prof. Vijay Janapa Reddi</strong> at Harvard University.
            The textbook follows the Hennessy &amp; Patterson pedagogical model — the same approach behind
            the most widely-used computer architecture textbooks in the world.
          </p>
          <p className="text-[14px] text-textSecondary leading-relaxed">
            Every topic in the question bank links back to its source chapter, so you can learn the concept first and then test yourself.
            The thesis: <em>AI is not magic — it is infrastructure, and infrastructure has laws.</em>
          </p>
        </section>

        {/* ─── Why I Built This ─── */}
        <section className="mb-10">
          <div className="p-5 rounded-xl border border-borderSubtle bg-surface/50">
            <p className="text-[14px] text-textSecondary leading-relaxed mb-3 italic">
              &ldquo;Students kept asking me: &lsquo;How do I prepare for ML systems interviews?&rsquo;
              The existing resources were either pure trivia or too abstract. There was nothing that tested
              the quantitative, physics-grounded reasoning that real interviews demand — the kind where you
              need to know that HBM3 is 300x slower than an L1 read, and what that means for your serving architecture.
            </p>
            <p className="text-[14px] text-textSecondary leading-relaxed mb-4 italic">
              StaffML is my answer. Every question comes from the textbook, validated against real hardware specs,
              and designed to build the mental models that matter. It&apos;s free because interview prep shouldn&apos;t
              be a luxury.&rdquo;
            </p>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-accentBlue/10 border border-accentBlue/20 flex items-center justify-center shrink-0">
                <span className="text-sm font-bold text-accentBlue">VR</span>
              </div>
              <div>
                <span className="text-[13px] font-bold text-textPrimary block">Vijay Janapa Reddi</span>
                <span className="text-[11px] text-textTertiary">Associate Professor, Harvard University</span>
              </div>
            </div>
          </div>
        </section>

        {/* ─── Paper ─── */}
        <section className="mb-10">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <BookOpen className="w-4.5 h-4.5 text-accentAmber" /> Research
          </h2>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-3">
            The methodology behind StaffML — the ikigai zone model, 79-topic taxonomy, and IRT-based
            difficulty calibration — is documented in an accompanying research paper.
          </p>
          <p className="text-[12px] text-textTertiary font-mono mb-3">
            Reddi, V.J. (2026). &ldquo;StaffML: A Physics-Grounded Framework for ML Systems Interview Preparation.&rdquo;
          </p>
          {/*
            TODO: Replace with actual paper link when published:
            <a href="/paper.pdf" className="...">Download Paper (PDF)</a>
          */}
          <span className="text-[11px] text-textTertiary italic">Paper link coming soon.</span>
        </section>

        {/* ─── Open Source ─── */}
        <section className="mb-12">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <Github className="w-4.5 h-4.5 text-textSecondary" /> Open Source
          </h2>
          <p className="text-[14px] text-textSecondary leading-relaxed mb-3">
            StaffML is free and open source. The entire question corpus, taxonomy, and web application are available on GitHub.
            Contributions, feedback, and corrections are welcome.
          </p>
          <a
            href="https://github.com/harvard-edge/cs249r_book"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2.5 bg-surface border border-border text-textSecondary hover:text-textPrimary rounded-lg text-sm font-medium transition-colors"
          >
            <Github className="w-4 h-4" /> View on GitHub
          </a>
        </section>

        {/* Vault Version */}
        <section className="mb-12">
          <h2 className="text-lg font-bold text-textPrimary mb-3 flex items-center gap-2">
            <Package className="w-4.5 h-4.5 text-textSecondary" /> Vault Version
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
            {[
              { label: "Version", value: `v${manifest.version}` },
              { label: "Questions", value: manifest.questionCount.toLocaleString() },
              { label: "Chains", value: manifest.chainCount.toLocaleString() },
              { label: "Concepts", value: manifest.conceptCount.toLocaleString() },
            ].map(({ label, value }) => (
              <div key={label} className="p-3 bg-surface border border-borderSubtle rounded-lg text-center">
                <span className="text-[18px] font-bold font-mono text-textPrimary block">{value}</span>
                <span className="text-[11px] text-textTertiary">{label}</span>
              </div>
            ))}
          </div>
          <p className="text-[12px] text-textMuted font-mono">
            Built {manifest.buildDate.slice(0, 10)} &middot; Hash {manifest.contentHash} &middot; Taxonomy v{manifest.taxonomyVersion}
          </p>
        </section>

        {/* CTA */}
        <div className="border-t border-border pt-8 text-center">
          <Link
            href="/practice"
            className="inline-flex items-center gap-2 px-6 py-3 bg-accentBlue text-white font-bold rounded-lg text-sm hover:opacity-90 transition-opacity"
          >
            <Target className="w-4 h-4" /> Start Practicing
          </Link>
        </div>
      </div>
    </div>
  );
}
