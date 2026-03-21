"use client";

import { useState, useEffect, useRef } from "react";
import { getQuestions, getTracks, getLevels, Question } from "@/lib/corpus";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Terminal, Cpu, Network, HardDrive, Zap, Play, CheckCircle2, XCircle, Star, Github, Activity, ChevronRight, ChevronDown, ServerCrash, ExternalLink, BookOpen } from "lucide-react";
import clsx from "clsx";
import dynamic from "next/dynamic";
import { usePyodide } from "@/lib/pyodide";
import { motion, AnimatePresence } from "framer-motion";
import HardwareConfigurator from "@/components/HardwareConfigurator";

// Dynamically import Mermaid to avoid SSR/prerendering errors
const MermaidRenderer = dynamic(() => import("@/components/MermaidRenderer"), { ssr: false });
const AnimatedFlow = dynamic(() => import("@/components/AnimatedFlow"), { ssr: false });

// Helper component for the typing terminal effect
const TypewriterText = ({ text, delay = 0, onComplete }: { text: string, delay?: number, onComplete?: () => void }) => {
  const [displayedText, setDisplayedText] = useState("");

  useEffect(() => {
    let i = 0;
    const timeout = setTimeout(() => {
      const interval = setInterval(() => {
        setDisplayedText(text.substring(0, i));
        i++;
        if (i > text.length) {
          clearInterval(interval);
          if (onComplete) onComplete();
        }
      }, 10); 
      return () => clearInterval(interval);
    }, delay);
    return () => clearTimeout(timeout);
  }, [text, delay, onComplete]);

  return <span>{displayedText}</span>;
};

export default function Home() {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [activeQuestion, setActiveQuestion] = useState<Question | null>(null);
  const [tracks, setTracks] = useState<string[]>([]);
  const [levels, setLevels] = useState<string[]>([]);
  
  const [filterTrack, setFilterTrack] = useState<string>("cloud");
  const [expandedTracks, setExpandedTracks] = useState<string[]>(["cloud"]);
  const [filterScope, setFilterScope] = useState<string | null>(null);
  const [filterLevel, setFilterLevel] = useState<string>("L4");
  
  const [diagnosis, setDiagnosis] = useState("");
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evalPhase, setEvalPhase] = useState(0); 
  const [showAnswer, setShowAnswer] = useState(false);
  const [evalResult, setEvalResult] = useState<any>(null);

  const [completedCount, setCompletedCount] = useState(0);
  const [showStarTrap, setShowStarTrap] = useState(false);
  const [hasStarred, setHasStarred] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  const { isInitializing, pyodide, evaluateArchitecture } = usePyodide();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setIsMounted(true);
    setQuestions(getQuestions());
    setTracks(getTracks());
    setLevels(getLevels());
    
    try {
      const savedCount = window.localStorage.getItem("staffml_completed");
      const savedStar = window.localStorage.getItem("staffml_starred");
      if (savedCount) setCompletedCount(parseInt(savedCount, 10));
      if (savedStar === "true") setHasStarred(true);
    } catch (e) {
      console.warn("Failed to access localStorage");
    }
  }, []);

  useEffect(() => {
    if (!isMounted) return;
    
    let filtered = questions.filter(
      (q) => q.track === filterTrack && q.level === filterLevel
    );
    
    if (filterScope) {
      filtered = filtered.filter(q => q.scope === filterScope);
    }

    if (filtered.length > 0) {
      const randomQ = filtered[Math.floor(Math.random() * filtered.length)];
      setActiveQuestion(randomQ);
      setShowAnswer(false);
      setDiagnosis("");
      setEvalResult(null);
      setEvalPhase(0);
    } else {
      setActiveQuestion(null);
    }
  }, [filterTrack, filterScope, filterLevel, questions, isMounted]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        if (!isInitializing && !isEvaluating && diagnosis.trim() && !showAnswer && activeQuestion) {
          if (isSimulationMode) {
             handleEvaluate();
          } else {
             handleFlashcardFlip();
          }
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  });

  const isSimulationMode = activeQuestion?.level.includes("L4") || activeQuestion?.level.includes("L5") || activeQuestion?.level.includes("L6");

  const handleEvaluate = async () => {
    if (completedCount >= 3 && !hasStarred) {
      setShowStarTrap(true);
      return;
    }

    setIsEvaluating(true);
    setEvalPhase(1); 
    
    try {
      let resultObj;
      if (pyodide && activeQuestion) {
        const result = await evaluateArchitecture(diagnosis, activeQuestion.details.napkin_math || "");
        resultObj = result instanceof Map ? Object.fromEntries(result) : result;
      } else {
        throw new Error("Engine not ready");
      }
      
      setTimeout(() => setEvalPhase(2), 400); 
      
      setTimeout(() => {
        setEvalResult(resultObj);
        setEvalPhase(3); 
        setIsEvaluating(false);
        setShowAnswer(true);
        incrementProgress();
      }, 1000);

    } catch (e) {
      setEvalResult({ passed: true, feedback: "Evaluation bypass (Pyodide error)." });
      setIsEvaluating(false);
      setShowAnswer(true);
      setEvalPhase(3);
      incrementProgress();
    }
  };

  const handleFlashcardFlip = () => {
    if (completedCount >= 3 && !hasStarred) {
      setShowStarTrap(true);
      return;
    }
    
    // Create a mock evaluation result for flashcards to unify UI
    setEvalResult({ 
      passed: true, 
      feedback: "Knowledge Check Revealed. Compare your mental model against the ground truth below." 
    });
    
    setShowAnswer(true);
    incrementProgress();
  };

  const incrementProgress = () => {
    const newCount = completedCount + 1;
    setCompletedCount(newCount);
    try {
      window.localStorage.setItem("staffml_completed", newCount.toString());
    } catch (e) {}
  };

  const handleStarClick = () => {
    window.open("https://github.com/harvard-edge/MLSysBook", "_blank");
    setHasStarred(true);
    try {
      window.localStorage.setItem("staffml_starred", "true");
    } catch (e) {}
    setShowStarTrap(false);
  };

  const extractTelemetry = (text: string) => {
    const metrics: { label: string; value: string }[] = [];
    const pctMatch = text.match(/(\d+(?:\.\d+)?%)/g);
    if (pctMatch) pctMatch.slice(0, 2).forEach((m, i) => metrics.push({ label: `UTIL_${i}`, value: m }));
    const timeMatch = text.match(/(\d+(?:\.\d+)?(?:ms|s))/g);
    if (timeMatch) timeMatch.slice(0, 2).forEach((m, i) => metrics.push({ label: `LATENCY_${i}`, value: m }));
    const dataMatch = text.match(/(\d+(?:\.\d+)?\s*(?:GB|MB|TB|TFLOPS|TOPS))/g);
    if (dataMatch) dataMatch.slice(0, 2).forEach((m, i) => metrics.push({ label: `CAPACITY_${i}`, value: m }));
    return Array.from(new Map(metrics.map(item => [item.value, item])).values()).slice(0, 4);
  };

  const getScopesForTrack = (track: string) => {
    const trackQs = questions.filter(q => q.track === track);
    const scopes = new Set(trackQs.map(q => q.scope));
    return Array.from(scopes).sort();
  };

  const toggleTrack = (track: string) => {
    if (expandedTracks.includes(track)) {
      setExpandedTracks(expandedTracks.filter(t => t !== track));
      if (filterTrack === track) {
        setFilterScope(null);
      }
    } else {
      setExpandedTracks([...expandedTracks, track]);
    }
  };

  const getEditUrl = () => {
    if (!activeQuestion) return "https://github.com/harvard-edge/MLSysBook/tree/dev/interviews";
    // Construct approximate GitHub URL
    const fileMap: Record<string, string> = {
      "01 Single Machine": "01_single_machine.md",
      "03 Serving Stack": "03_serving_stack.md",
      "05 Visual Debugging": "05_visual_debugging.md",
      "02 Distributed Systems": "02_distributed_systems.md",
      "04 Production Ops": "04_production_ops.md",
      "Foundations": "foundations.md"
    };
    const fileName = fileMap[activeQuestion.scope] || "README.md";
    const trackPath = activeQuestion.track === "global" ? "" : `${activeQuestion.track}/`;
    return `https://github.com/harvard-edge/MLSysBook/edit/dev/interviews/${trackPath}${fileName}`;
  };

  const renderScenario = (text: string) => {
    if (!text) return null;
    const parts = text.split("```mermaid");
    const rawText = parts[0].replace(/- \*\*Interviewer:\*\*/, "").replace(/"/g, "").trim();
    const telemetry = extractTelemetry(rawText);

    // Hero Demo Override
    if (activeQuestion?.id === "cloud-network-the-communication-wa") {
      return (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="flex flex-col space-y-6"
        >
          <div className="flex gap-4 mb-2">
            <div className="flex items-baseline gap-2">
               <span className="text-[10px] font-mono text-textTertiary uppercase">Scale</span>
               <span className="text-xs font-mono text-textPrimary">512 GPUs</span>
            </div>
            <div className="flex items-baseline gap-2">
               <span className="text-[10px] font-mono text-textTertiary uppercase">Model</span>
               <span className="text-xs font-mono text-textPrimary">70B Params</span>
            </div>
            <div className="flex items-baseline gap-2">
               <span className="text-[10px] font-mono text-textTertiary uppercase">Payload</span>
               <span className="text-xs font-mono text-textPrimary">140 GB</span>
            </div>
          </div>

          <div className="prose max-w-none">
             <p className="text-textSecondary leading-relaxed">{rawText}</p>
          </div>

          <div className="panel w-full mt-4 p-4 shadow-xl">
            <div className="flex items-center justify-between mb-4 border-b border-border pb-4">
              <div className="flex items-center gap-2">
                <Activity className="w-3.5 h-3.5 text-textTertiary" />
                <span className="text-[10px] font-mono text-textSecondary uppercase tracking-widest">Live Topology Simulator</span>
              </div>
              <div className="flex items-center gap-3">
                 <span className="text-[10px] font-mono text-textTertiary uppercase">Interconnect:</span>
                 <select 
                   onChange={(e) => setDiagnosis(e.target.value === '10GbE' ? 'The system is bottlenecked by 10GbE.' : 'InfiniBand resolves the bottleneck.')}
                   className="bg-background border border-border text-textPrimary text-xs font-mono rounded px-3 py-1.5 focus:outline-none focus:border-accentBlue appearance-none"
                 >
                   <option value="10GbE">10 GbE (Legacy)</option>
                   <option value="InfiniBand">NDR InfiniBand (400 Gbps)</option>
                 </select>
              </div>
            </div>
            <AnimatedFlow interconnectType={diagnosis.includes('InfiniBand') ? 'InfiniBand' : '10GbE'} />
          </div>
        </motion.div>
      );
    }
    
    return (
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="flex flex-col space-y-6"
      >
        <div className="flex gap-4 mb-2">
          {telemetry.length > 0 ? telemetry.map((metric, i) => (
             <motion.div 
               initial={{ opacity: 0, x: -10 }}
               animate={{ opacity: 1, x: 0 }}
               transition={{ delay: i * 0.1 }}
               key={i} 
               className="flex items-baseline gap-2"
             >
               <span className="text-[10px] font-mono text-textTertiary uppercase">{metric.label}</span>
               <span className="text-xs font-mono text-textPrimary">{metric.value}</span>
             </motion.div>
          )) : (
             <div className="flex items-baseline gap-2">
                <span className="text-[10px] font-mono text-textTertiary uppercase">Status</span>
                <span className="text-xs font-mono text-textPrimary">Nominal</span>
             </div>
          )}
        </div>

        <div className="prose max-w-none">
           <p className="text-textSecondary leading-relaxed text-lg">{rawText}</p>
        </div>

        {parts.length > 1 && parts.slice(1).map((part, index) => {
          const splitPart = part.split("```");
          const mermaidCode = splitPart[0];
          return (
            <motion.div 
              initial={{ scale: 0.98, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.2 }}
              key={index} 
              className="panel w-full mt-4 p-6 shadow-xl"
            >
               <div className="flex items-center justify-between mb-4 border-b border-border pb-2">
                 <div className="flex items-center gap-2">
                   <Activity className="w-3.5 h-3.5 text-textTertiary" />
                   <span className="text-[10px] font-mono text-textSecondary uppercase tracking-widest">System Topology</span>
                 </div>
                 <div className="flex gap-1">
                   <div className="w-1.5 h-1.5 rounded-full bg-borderHighlight"></div>
                   <div className="w-1.5 h-1.5 rounded-full bg-borderHighlight"></div>
                   <div className="w-1.5 h-1.5 rounded-full bg-borderHighlight"></div>
                 </div>
               </div>
               <div className="flex justify-center opacity-90 hover:opacity-100 transition-opacity">
                 <MermaidRenderer chart={mermaidCode.trim()} />
               </div>
            </motion.div>
          );
        })}
      </motion.div>
    );
  };

  if (!isMounted) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <Terminal className="w-6 h-6 text-textTertiary animate-pulse" />
          <span className="font-mono text-xs text-textSecondary uppercase tracking-widest">Initializing Control Plane</span>
        </div>
      </div>
    );
  }

  return (
    <main className="flex h-screen w-full overflow-hidden text-sm bg-background font-sans selection:bg-accentBlue/30 selection:text-white">
      {/* Star Trap Modal */}
      <AnimatePresence>
      {showStarTrap && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md"
        >
          <motion.div 
            initial={{ scale: 0.95, y: 10 }}
            animate={{ scale: 1, y: 0 }}
            className="glass-panel border border-border rounded-2xl p-10 max-w-lg w-full shadow-2xl text-center relative overflow-hidden"
          >
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-accentBlue via-purple-500 to-accentBlue"></div>
            
            <div className="w-20 h-20 bg-surface border border-border rounded-full flex items-center justify-center mx-auto mb-6 shadow-[0_0_30px_rgba(0,112,243,0.3)]">
              <Star className="w-10 h-10 text-accentBlue" />
            </div>
            
            <h2 className="text-2xl font-bold text-white mb-3 tracking-tight">Level Up Unlocked</h2>
            <p className="text-textSecondary mb-8 leading-relaxed text-base">
              You've proven your architectural intuition. StaffML is powered entirely by the open-source <strong>MLSysBook</strong>. 
              To unlock the remaining 1,064 physics-based challenges and save your progress locally, please support the project with a star.
            </p>
            <button 
              onClick={handleStarClick}
              className="w-full bg-white text-black hover:bg-gray-200 font-bold py-4 px-4 rounded-xl flex items-center justify-center gap-3 transition-all shadow-[0_0_20px_rgba(255,255,255,0.15)] hover:shadow-[0_0_25px_rgba(255,255,255,0.25)] text-lg"
            >
              <Github className="w-6 h-6" /> Star on GitHub to Unlock
            </button>
            <button 
              onClick={() => setShowStarTrap(false)}
              className="mt-6 text-xs font-mono text-textTertiary hover:text-white transition-colors"
            >
              &gt; I ALREADY STARRED IT _
            </button>
          </motion.div>
        </motion.div>
      )}
      </AnimatePresence>

      {/* Sidebar */}
      <aside className="w-[300px] flex flex-col border-r border-border bg-surface z-20 shadow-xl relative">
        {/* Header / Logo */}
        <div className="h-24 flex flex-col justify-center px-8 border-b border-border bg-background/50 relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-r from-accentBlue/0 via-accentBlue/5 to-accentBlue/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></div>
          
          <div className="flex items-center relative z-10 tracking-tight select-none">
            {/* The Chevron - Optically aligned and sized to match the cap-height of 'Staff' */}
            <span 
              className="text-accentBlue font-mono font-black mr-2.5 drop-shadow-[0_0_12px_rgba(59,130,246,0.6)]" 
              style={{ fontSize: '1.6rem', lineHeight: '1', transform: 'translateY(-1px)' }}
            >
              &gt;
            </span>
            <span className="text-white text-2xl font-extrabold font-sans tracking-tight">Staff</span>
            <span className="text-white/70 text-2xl font-medium font-sans tracking-tight">ML</span>
          </div>
          
          <a href="https://mlsysbook.ai" target="_blank" rel="noopener noreferrer" className="text-[9px] text-textTertiary hover:text-accentBlue transition-colors mt-2 uppercase tracking-[0.25em] font-mono relative z-10 pl-[1.8rem] flex items-center gap-1 group-hover:underline">
            <BookOpen className="w-3 h-3 inline-block" /> Powered by MLSysBook
          </a>
        </div>

        <div className="flex-1 overflow-y-auto py-6 custom-scrollbar">
          {/* Tracks Accordion */}
          <div className="px-4 mb-8">
            <div className="px-2 mb-3 text-[10px] font-semibold text-textTertiary uppercase tracking-widest flex items-center gap-2">
              <ServerCrash className="w-3 h-3" /> Architecture Domains
            </div>
            <div className="space-y-1">
              {tracks.map((t) => {
                const isExpanded = expandedTracks.includes(t);
                const isActiveTrack = filterTrack === t;
                const scopes = getScopesForTrack(t);
                
                return (
                  <div key={t} className="flex flex-col">
                    <button
                      onClick={() => {
                        toggleTrack(t);
                        setFilterTrack(t);
                      }}
                      className={clsx(
                        "w-full flex items-center justify-between px-3 py-2 rounded-md text-[13px] font-medium capitalize transition-all",
                        isActiveTrack && !filterScope ? "bg-accentBlue/10 text-accentBlue" : "text-textPrimary hover:bg-surfaceHover"
                      )}
                    >
                      <span className="flex items-center gap-2">
                        {t === "cloud" && "☁️"}
                        {t === "edge" && "🤖"}
                        {t === "mobile" && "📱"}
                        {t === "tinyml" && "🔬"}
                        {t === "global" && "🌍"}
                        <span className={clsx("ml-1", t === "tinyml" ? "uppercase" : "capitalize")}>{t === "tinyml" ? "TinyML" : t}</span>
                      </span>
                      {isExpanded ? <ChevronDown className="w-3 h-3 text-textTertiary" /> : <ChevronRight className="w-3 h-3 text-textTertiary" />}
                    </button>
                    
                    {/* Sub-scopes Accordion */}
                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div 
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          className="overflow-hidden"
                        >
                          <div className="pl-6 pr-2 pt-1 pb-2 space-y-0.5 border-l border-border ml-4 mt-1">
                            {scopes.map(s => {
                              const isActiveScope = isActiveTrack && filterScope === s;
                              const qCount = questions.filter(q => q.track === t && q.scope === s).length;
                              return (
                                <button
                                  key={s}
                                  onClick={() => {
                                    setFilterTrack(t);
                                    setFilterScope(s);
                                  }}
                                  className={clsx(
                                    "w-full flex items-center justify-between px-2 py-1.5 rounded text-xs transition-colors",
                                    isActiveScope ? "text-accentBlue font-medium" : "text-textSecondary hover:text-textPrimary hover:bg-surfaceHover"
                                  )}
                                >
                                  <span className="truncate pr-2 text-left">{s}</span>
                                  <span className="text-[9px] font-mono text-textTertiary">{qCount}</span>
                                </button>
                              );
                            })}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Levels */}
          <div className="px-4">
            <div className="px-2 mb-3 text-[10px] font-semibold text-textTertiary uppercase tracking-widest flex items-center gap-2">
              <Activity className="w-3 h-3" /> Mastery Tier
            </div>
            <div className="space-y-1">
              {levels.map((l) => (
                <button
                  key={l}
                  onClick={() => setFilterLevel(l)}
                  className={clsx(
                    "w-full flex items-center justify-between px-3 py-2 rounded-md text-[13px] font-mono transition-all",
                    filterLevel === l ? "bg-borderHighlight text-textPrimary shadow-sm" : "text-textSecondary hover:bg-surfaceHover"
                  )}
                >
                  <div className="flex items-center">
                    <span className={clsx(
                      "w-1.5 h-1.5 rounded-full mr-3 shadow-[0_0_8px_rgba(255,255,255,0.5)]",
                      l.includes("L6") ? "bg-accentRed shadow-accentRed" : l.includes("L5") ? "bg-accentAmber shadow-accentAmber" : "bg-accentBlue shadow-accentBlue"
                    )} />
                    {l}
                  </div>
                  <span className="text-[9px] text-textTertiary">{questions.filter(q => q.level === l).length}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
        
        <div className="p-5 border-t border-border/50 bg-background/30">
           <div className="text-[10px] font-mono text-textTertiary mb-2 flex justify-between uppercase tracking-wider">
             <span>Simulations</span>
             <span className="text-white">{completedCount}</span>
           </div>
           {!hasStarred && (
             <div className="w-full bg-border h-1 rounded-full overflow-hidden">
               <div className="bg-accentBlue h-full transition-all duration-700 ease-out" style={{ width: `${Math.min((completedCount / 3) * 100, 100)}%` }}></div>
             </div>
           )}
        </div>
      </aside>

      {/* Main Workspace */}
      <section className="flex-1 flex flex-col min-w-0 bg-dot-grid relative">
        <div className="absolute inset-0 bg-background/90 pointer-events-none z-0"></div>
        
        {/* Topbar */}
        <header className="h-14 border-b border-border flex items-center px-8 justify-between bg-background/80 backdrop-blur-md z-10">
          <div className="flex items-center gap-2 text-textSecondary text-sm font-medium">
            <span className="capitalize">{filterTrack === "tinyml" ? "TinyML" : filterTrack}</span>
            {filterScope && (
              <>
                <ChevronRight className="w-4 h-4 text-textTertiary" />
                <span className="text-textSecondary">{filterScope}</span>
              </>
            )}
            <ChevronRight className="w-4 h-4 text-textTertiary" />
            <span className="text-textPrimary max-w-[400px] truncate">{activeQuestion?.title || "Select Scenario"}</span>
          </div>
          
          <div className="flex items-center gap-4">
            <a href={getEditUrl()} target="_blank" rel="noopener noreferrer" className="text-textTertiary hover:text-textPrimary flex items-center gap-1.5 text-xs transition-colors">
              <ExternalLink className="w-3.5 h-3.5" />
              Edit on GitHub
            </a>
            <div className="w-px h-4 bg-border"></div>
            {isInitializing && <span className="text-[10px] font-mono text-accentAmber uppercase flex items-center gap-1"><Zap className="w-3 h-3 animate-pulse"/> Booting</span>}
            <div className="flex items-center gap-2 px-2.5 py-1.5 rounded bg-surface border border-border shadow-sm">
              <div className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accentGreen opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-accentGreen"></span>
              </div>
              <span className="text-[10px] font-mono text-textSecondary uppercase">Ready</span>
            </div>
          </div>
        </header>

        {/* Editor Area */}
        {activeQuestion ? (
          <div className="flex-1 flex overflow-hidden z-10">
            {/* Left: Spec View */}
            <div className="flex-1 overflow-y-auto px-12 py-10 custom-scrollbar max-w-4xl">
              <motion.h1 
                key={activeQuestion.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="text-4xl font-medium text-textPrimary mb-8 tracking-tight"
              >
                {activeQuestion.title}
              </motion.h1>
              {renderScenario(activeQuestion.scenario)}
            </div>

            {/* Right: Terminal / IDE View */}
            <motion.div 
               initial={{ x: 50, opacity: 0 }}
               animate={{ x: 0, opacity: 1 }}
               className="w-[500px] border-l border-border bg-surface/90 backdrop-blur-2xl flex flex-col shadow-[-20px_0_40px_rgba(0,0,0,0.4)]"
            >
              <div className="h-10 border-b border-border flex items-center px-4 bg-background/50 justify-between">
                <span className="text-[10px] font-mono text-textTertiary uppercase tracking-widest flex items-center gap-2">
                  <Terminal className="w-3 h-3" /> {isSimulationMode ? "evaluation.py" : "flashcard.sys"}
                </span>
                <span className="text-[9px] font-mono text-textTertiary bg-surface px-1.5 py-0.5 rounded border border-border">Cmd + Enter</span>
              </div>
              
              <div className="flex-1 p-6 flex flex-col">
                
                {isSimulationMode ? (
                  <>
                    <HardwareConfigurator 
                       onStateChange={(state) => {
                         // Build diagnosis string automatically based on hardware choices
                         setDiagnosis(`I am configuring the system with:\n- ${state.compute} Compute\n- ${state.network} Interconnect\n- ${state.memory} Storage\n\n# Provide additional napkin math here...\n`);
                       }} 
                    />
                    <div className="relative flex-1 group min-h-[250px]">
                      <div className="absolute -inset-0.5 bg-gradient-to-br from-accentBlue/20 to-transparent rounded-lg blur opacity-0 group-focus-within:opacity-100 transition duration-500"></div>
                      <textarea
                        ref={textareaRef}
                        value={diagnosis}
                        onChange={(e) => setDiagnosis(e.target.value)}
                        placeholder="# Write your diagnosis and napkin math here...&#10;&#10;def analyze_bottleneck():&#10;    # The ridge point is 295 Ops/Byte...&#10;    return 'The system is memory-bound.'"
                        className="relative w-full h-full bg-background border border-border rounded-lg p-5 font-mono text-[13px] text-textPrimary resize-none focus:outline-none focus:border-accentBlue/50 transition-colors placeholder:text-textTertiary/50 leading-relaxed shadow-inner"
                        spellCheck="false"
                      />
                    </div>

                    <button
                      onClick={handleEvaluate}
                      disabled={isInitializing || isEvaluating || !diagnosis.trim() || showAnswer}
                      className="mt-6 w-full bg-textPrimary hover:bg-white disabled:opacity-50 disabled:bg-borderHighlight text-background font-semibold py-3 rounded-lg transition-all flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(255,255,255,0.1)] hover:shadow-[0_0_20px_rgba(255,255,255,0.2)]"
                    >
                      {isEvaluating ? (
                        <span className="flex items-center gap-2 text-white"><Zap className="w-4 h-4 animate-pulse text-accentAmber" /> Executing Pipeline...</span>
                      ) : showAnswer ? (
                        "Evaluation Complete"
                      ) : (
                        <>Run Simulation <Play className="w-4 h-4" /></>
                      )}
                    </button>
                  </>
                ) : (
                  <div className="flex-1 flex flex-col justify-center items-center text-center">
                     <div className="w-24 h-24 bg-background border border-border rounded-full flex items-center justify-center mb-6 shadow-inner relative">
                       <div className="absolute inset-0 bg-accentBlue/10 rounded-full animate-ping opacity-20"></div>
                       <Zap className="w-10 h-10 text-accentBlue relative z-10" />
                     </div>
                     <h3 className="text-xl font-medium text-textPrimary mb-3">Knowledge Check</h3>
                     <p className="text-textSecondary text-sm px-6 mb-8 leading-relaxed">This foundational scenario tests your baseline physics and system literacy. Formulate your answer mentally before flipping the card.</p>
                     
                     <button
                        onClick={handleFlashcardFlip}
                        disabled={showAnswer}
                        className="w-full bg-surfaceHover border border-borderHighlight hover:bg-white hover:text-black hover:border-transparent text-textPrimary font-semibold py-3.5 rounded-lg transition-all flex items-center justify-center gap-2 shadow-sm"
                      >
                        {showAnswer ? "Card Flipped" : "Reveal Ground Truth"}
                      </button>
                  </div>
                )}

                {/* Output Console - Animated Terminal Effect */}
                <AnimatePresence>
                {(isEvaluating || showAnswer) && (
                  <motion.div 
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: isSimulationMode ? "auto" : "100%", opacity: 1 }}
                    className={clsx("flex-1 overflow-y-auto custom-scrollbar", isSimulationMode ? "mt-6 border-t border-border pt-6" : "mt-0 pt-0")}
                  >
                    {isSimulationMode && (
                      <div className="font-mono text-xs text-textSecondary space-y-3 mb-8 bg-background p-4 rounded-lg border border-border shadow-inner">
                        {evalPhase >= 1 && <div><span className="text-accentBlue">❯</span> <TypewriterText text="Initializing Pyodide MLSys engine..." /></div>}
                        {evalPhase >= 2 && <div><span className="text-accentBlue">❯</span> <TypewriterText text="Mounting physics constants (constants.py)..." delay={200} /></div>}
                        {evalPhase >= 3 && evalResult && (
                          <motion.div 
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className={clsx("mt-4 pt-3 border-t border-border border-dashed", evalResult.passed ? "text-accentGreen" : "text-accentRed")}
                          >
                            <span className="font-bold uppercase tracking-wider text-[10px] mb-2 block">Status: {evalResult.passed ? 'PASSED' : 'FAILED'}</span>
                            <TypewriterText text={evalResult.feedback} />
                          </motion.div>
                        )}
                      </div>
                    )}

                    {showAnswer && (
                      <motion.div 
                         initial={{ opacity: 0, y: 10 }}
                         animate={{ opacity: 1, y: 0 }}
                         transition={{ delay: isSimulationMode ? 0.6 : 0.1 }}
                         className="space-y-6"
                      >
                        {activeQuestion?.details.common_mistake && (
                          <div className="border-l-4 border-accentRed pl-4">
                            <span className="text-[10px] font-mono text-accentRed uppercase mb-1.5 block flex items-center gap-1.5"><XCircle className="w-3.5 h-3.5"/> Anti-Pattern</span>
                            <p className="text-[15px] text-textSecondary leading-relaxed">{activeQuestion.details.common_mistake}</p>
                          </div>
                        )}

                        <div className="border-l-4 border-accentGreen pl-4">
                          <span className="text-[10px] font-mono text-accentGreen uppercase mb-1.5 block flex items-center gap-1.5"><CheckCircle2 className="w-3.5 h-3.5"/> Ground Truth</span>
                          <p className="text-[15px] text-textPrimary leading-relaxed">{activeQuestion?.details.realistic_solution}</p>
                        </div>

                        {activeQuestion?.details.napkin_math && (
                          <div className="bg-background border border-border p-5 rounded-xl shadow-inner relative overflow-hidden">
                            <div className="absolute top-0 left-0 w-1 h-full bg-accentBlue"></div>
                            <span className="text-[10px] font-mono text-accentBlue uppercase mb-3 block flex items-center gap-1.5"><Terminal className="w-3.5 h-3.5"/> Napkin Math</span>
                            <div className="font-mono text-[13px] text-textSecondary leading-relaxed whitespace-pre-wrap bg-surface/50 p-3 rounded border border-border/50">
                              {activeQuestion.details.napkin_math}
                            </div>
                          </div>
                        )}
                        
                        {activeQuestion?.details.deep_dive_title && (
                          <a href={activeQuestion.details.deep_dive_url} target="_blank" rel="noopener noreferrer" className="block w-full text-center py-3 px-4 rounded-lg bg-surface hover:bg-surfaceHover border border-border transition-colors text-xs font-mono text-textTertiary hover:text-textPrimary flex items-center justify-center gap-2">
                             <BookOpen className="w-4 h-4" />
                             Read Chapter: {activeQuestion.details.deep_dive_title}
                          </a>
                        )}
                      </motion.div>
                    )}
                  </motion.div>
                )}
                </AnimatePresence>
              </div>
            </motion.div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-textTertiary">
             Select a scenario from the sidebar.
          </div>
        )}
      </section>
    </main>
  );
}