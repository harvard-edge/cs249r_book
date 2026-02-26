import marimo

__generated_with = "0.10.15"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import sys
    import os
    from pathlib import Path
    
    # --- PATHS ---
    try:
        notebook_path = Path(os.path.abspath(__file__))
    except NameError:
        notebook_path = Path(os.getcwd()) / "labs" / "vol1" / "lab_00_the_map.py"
    
    project_root = notebook_path.parents[2]
    sys.path.append(str(project_root / "book" / "quarto"))
    sys.path.append(str(project_root))
    
    from labs.core.style import COLORS, LAB_CSS
    from labs.core.components import Card, StakeholderMessage
    from labs.core.state import DesignLedger
    
    ledger = DesignLedger()
    
    return COLORS, Card, DesignLedger, LAB_CSS, StakeholderMessage, ledger, mo, os, project_root, sys


@app.cell
def __(LAB_CSS, mo):
    # --- HERO SECTION: THE MANIFESTO ---
    mo.vstack([
        LAB_CSS,
        mo.md("# 🗺️ The Architect's Frontier"),
        mo.md(
            r"""
            ### **Software 1.0 is Dead.**
            For decades, software was built on explicit logic—human-written rules that computers followed blindly. Today, that paradigm is collapsing. We are entering the era of **AI Engineering**: the discipline of building stochastic systems with deterministic reliability.
            
            The world has enough "model builders." What it lacks are **System Architects**—engineers who can bridge the gap between a research notebook and a physical machine. 
            
            This laboratory is your **Diagnostic Flight Simulator**. Over the next 16 chapters, you will not just "run models"—you will navigate the physical trade-offs of the universe.
            """
        ),
        mo.md("---")
    ])


@app.cell
def __(mo):
    # --- THE CAMPAIGN ROADMAP ---
    mo.vstack([
        mo.md("## The 16-Chapter Campaign"),
        mo.md("Your journey is divided into four tactical phases. In each, you will confront a new **Physical Wall** that threatens your mission."),
        mo.hstack([
            mo.md("""
            #### **Phase I: Foundations**
            *Chapters 01–04*
            **The Goal:** Physics.
            **The Wall:** The Light Barrier.
            Analyze the 9-order-of-magnitude gap and the cost of moving bits.
            """),
            mo.md("""
            #### **Phase II: The Build**
            *Chapters 05–08*
            **The Goal:** Mechanics.
            **The Wall:** The Dispatch Tax.
            Audit the arithmetic intensity of layers and the cost of software overhead.
            """)
        ], gap=2),
        mo.hstack([
            mo.md("""
            #### **Phase III: Optimize**
            *Chapters 09–12*
            **The Goal:** Efficiency.
            **The Wall:** The Memory Wall.
            Find the Pareto Frontier using quantization, pruning, and tiling.
            """),
            mo.md("""
            #### **Phase IV: Deploy**
            *Chapters 13–16*
            **The Goal:** Production.
            **The Wall:** System Entropy.
            Manage drift and the 3-year TCO of a live, evolving fleet.
            """)
        ], gap=2),
        mo.md("---")
    ])


@app.cell
def __(mo):
    # --- TRACK SELECTION UI ---
    mo.md("## Choose Your Frontier")
    mo.md("Which physical regime will you master? Your choice here defines your narrative and constraints for the entire curriculum.")
    
    frontier_selector = mo.ui.radio(
        options={
            "☁️ THE CLOUD TITAN (Datacenter Scale)": "CLOUD",
            "🤖 THE EDGE GUARDIAN (Autonomous Systems)": "EDGE",
            "🕶️ THE MOBILE NOMAD (Augmented Reality)": "MOBILE",
            "👂 THE TINY PIONEER (Neural Hearables)": "TINY"
        },
        label="Select your Specialization:"
    )
    frontier_selector
    return (frontier_selector,)


@app.cell
def __(COLORS, Card, StakeholderMessage, frontier_selector, ledger, mo):
    # --- DYNAMIC BRIEFING LOGIC ---
    def render_briefing():
        if frontier_selector.value is None:
            return mo.md("_Select a frontier above to receive your mission briefing._")
        
        # Persistence: Save the choice immediately
        ledger.save(track=frontier_selector.value, chapter=0)
        
        _data = {
            "CLOUD": {
                "role": "LLM Serving Architect",
                "goal": "Maximize Llama-3-70B throughput on a single H100 node.",
                "wall": "The Memory Bandwidth Wall (HBM3)",
                "color": COLORS['BlueLine'],
                "persona": "The Visionary Founder",
                "quote": "We have the best model in the world, but our serving costs are 10x higher than our revenue. If you can't saturate the H100 cores, we're out of business by Chapter 13."
            },
            "EDGE": {
                "role": "AV Systems Lead",
                "goal": "Maintain a deterministic 10ms safety-critical vision loop on an NVIDIA Orin.",
                "wall": "The Determinism Wall (Tail Jitter)",
                "color": COLORS['RedLine'],
                "persona": "The Safety Director",
                "quote": "A 5ms delay in the perception loop means 15cm of extra stopping distance at highway speeds. I don't care about your 'average' latency. I care about the worst case."
            },
            "MOBILE": {
                "role": "AR Glasses Developer",
                "goal": "Run 60FPS AR translation on Meta Ray-Bans under a 2W thermal cap.",
                "wall": "The Thermal Wall (Thermodynamics)",
                "color": COLORS['OrangeLine'],
                "persona": "The UX Designer",
                "quote": "Users won't wear these if the frames burn their face. You have 2 Watts of thermal headroom. If you waste energy moving bits over Bluetooth, the app will throttle."
            },
            "TINY": {
                "role": "Neural Hearable Lead",
                "goal": "Real-time noise isolation in <10ms under 1mW of power.",
                "wall": "The Echo Wall (SRAM Capacity)",
                "color": COLORS['GreenLine'],
                "persona": "The Hardware Lead",
                "quote": "We have 256KB of on-chip memory. Every weight bit you save is a millisecond of audio buffer we gain. This is a battle for bytes."
            }
        }
        
        _f = _data[frontier_selector.value]
        
        return mo.vstack([
            mo.md(f"### 🎖️ MISSION GRANTED: {_f['role']}"),
            StakeholderMessage(_f['persona'], _f['quote'], _f['color']),
            Card("Strategic Overview", f"""
                - **Primary Mission:** {_f['goal']}
                - **Arch Nemesis:** <span style='color:{COLORS['RedLine']}; font-weight:bold;'>{_f['wall']}</span>
                - **Status:** Design Ledger Initialized.
            """),
            mo.md(f"""
                <div style='padding: 15px; background-color: {COLORS['BlueLine']}; color: white; border-radius: 8px; text-align: center; font-weight: bold; cursor: pointer;'>
                    COMMENCE ORIENTATION: Proceed to Lab 01
                </div>
            """)
        ])

    render_briefing()
