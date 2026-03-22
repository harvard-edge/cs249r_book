"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Crosshair, BarChart3, Target, Cpu, Network, HardDrive, Zap, ArrowRight, BookOpen, Github, Terminal, Star } from "lucide-react";
import { getQuestions, getTracks, getCompetencyAreas } from "@/lib/corpus";

const features = [
  {
    icon: Crosshair,
    title: "The Gauntlet",
    description: "Timed mock interviews. Pick your track, level, and duration. Questions span all competency areas.",
    href: "/gauntlet",
    color: "text-accentRed",
    bgColor: "bg-accentRed/10",
    borderColor: "border-accentRed/30",
  },
  {
    icon: BarChart3,
    title: "Heat Map",
    description: "See your readiness by competency area. Green means ready, red means drill more.",
    href: "/heatmap",
    color: "text-accentGreen",
    bgColor: "bg-accentGreen/10",
    borderColor: "border-accentGreen/30",
  },
  {
    icon: Target,
    title: "Drill Mode",
    description: "Focused practice on one competency. Napkin math verification checks your estimates.",
    href: "/drill",
    color: "text-accentBlue",
    bgColor: "bg-accentBlue/10",
    borderColor: "border-accentBlue/30",
  },
];

const trackIcons: Record<string, typeof Cpu> = {
  cloud: Network,
  edge: Cpu,
  mobile: HardDrive,
  tinyml: Zap,
};

export default function Home() {
  const [stats, setStats] = useState({ questions: 0, tracks: 0, areas: 0 });
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const questions = getQuestions();
    const tracks = getTracks().filter(t => t !== "global");
    const areas = getCompetencyAreas();
    setStats({ questions: questions.length, tracks: tracks.length, areas: areas.length });
  }, []);

  if (!mounted) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Terminal className="w-6 h-6 text-textTertiary animate-pulse" />
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col">
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-dot-grid opacity-30" />
        <div className="absolute inset-0 bg-gradient-to-b from-accentBlue/5 via-transparent to-transparent" />

        <div className="relative max-w-5xl mx-auto px-6 pt-20 pb-16 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-surface border border-border text-xs text-textSecondary mb-8">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accentGreen opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-accentGreen" />
              </span>
              {stats.questions.toLocaleString()} questions across {stats.areas} competency areas
            </div>

            <h1 className="text-5xl sm:text-6xl font-extrabold tracking-tight text-white mb-6 leading-[1.1]">
              Prep for your<br />
              <span className="text-accentBlue">Staff ML Systems</span> interview
            </h1>

            <p className="text-lg text-textSecondary max-w-2xl mx-auto mb-10 leading-relaxed">
              Physics-grounded system design questions backed by real hardware constants.
              Not trivia — napkin math, bottleneck analysis, and architecture trade-offs.
            </p>

            <div className="flex items-center justify-center gap-4">
              <Link
                href="/gauntlet"
                className="inline-flex items-center gap-2 px-6 py-3 bg-white text-black font-bold rounded-lg hover:bg-gray-100 transition-all shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_30px_rgba(255,255,255,0.2)] text-sm"
              >
                Start The Gauntlet <ArrowRight className="w-4 h-4" />
              </Link>
              <Link
                href="/drill"
                className="inline-flex items-center gap-2 px-6 py-3 bg-surface border border-border text-textSecondary hover:text-white font-medium rounded-lg transition-colors text-sm"
              >
                <Target className="w-4 h-4" /> Quick Drill
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features */}
      <section className="max-w-5xl mx-auto px-6 pb-16 w-full">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {features.map((feature, i) => (
            <motion.div
              key={feature.href}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 + i * 0.1 }}
            >
              <Link href={feature.href} className="block group">
                <div className={`p-6 rounded-xl border ${feature.borderColor} ${feature.bgColor} hover:bg-opacity-20 transition-all`}>
                  <feature.icon className={`w-8 h-8 ${feature.color} mb-4`} />
                  <h3 className="text-lg font-bold text-white mb-2">{feature.title}</h3>
                  <p className="text-sm text-textSecondary leading-relaxed">{feature.description}</p>
                  <div className="mt-4 flex items-center gap-1 text-xs text-textTertiary group-hover:text-textSecondary transition-colors">
                    Try it <ArrowRight className="w-3 h-3" />
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Stats bar */}
      <section className="border-t border-border bg-surface/50">
        <div className="max-w-5xl mx-auto px-6 py-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { label: "Questions", value: stats.questions.toLocaleString() },
              { label: "Tracks", value: `${stats.tracks} domains` },
              { label: "Competencies", value: `${stats.areas} areas` },
              { label: "Levels", value: "L1 → L6+" },
            ].map((stat, i) => (
              <div key={i} className="text-center">
                <div className="text-2xl font-bold text-white font-mono">{stat.value}</div>
                <div className="text-xs text-textTertiary uppercase tracking-wider mt-1">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Moat section */}
      <section className="max-w-5xl mx-auto px-6 py-16 w-full">
        <h2 className="text-2xl font-bold text-white mb-8 text-center">Not another quiz app</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[
            { title: "Hardware Constants", desc: "Every question uses real specs: H100 bandwidth, A100 FLOPS, DDR5 latency. No made-up numbers." },
            { title: "Napkin Math", desc: "Type your calculation chain. We check if your final estimate is within tolerance — not exact match." },
            { title: "Textbook Backed", desc: "Each question links to the relevant MLSysBook chapter. Study the theory, then test it." },
            { title: "Bloom's Taxonomy", desc: "L1 = recall, L6+ = design novel systems. Questions are calibrated to real job levels." },
          ].map((item, i) => (
            <div key={i} className="p-5 rounded-xl border border-border bg-surface/50">
              <h3 className="text-sm font-bold text-white mb-2">{item.title}</h3>
              <p className="text-sm text-textSecondary leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer CTA */}
      <section className="border-t border-border bg-surface/30 mt-auto">
        <div className="max-w-5xl mx-auto px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <BookOpen className="w-5 h-5 text-textTertiary" />
            <span className="text-sm text-textSecondary">
              Built from the <a href="https://mlsysbook.ai" target="_blank" rel="noopener noreferrer" className="text-accentBlue hover:underline">ML Systems textbook</a> at Harvard
            </span>
          </div>
          <a
            href="https://github.com/harvard-edge/cs249r_book"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 bg-surface border border-border rounded-lg text-sm text-textSecondary hover:text-white transition-colors"
          >
            <Star className="w-4 h-4" /> Star on GitHub
          </a>
        </div>
      </section>
    </div>
  );
}
