import asyncio
import argparse
import sys
from pathlib import Path
from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.types import TemplatedSystemInstructions, CapabilitiesConfig

# Define the sequential order of chapters to feed into progressive disclosure
CHAPTERS = [
    "introduction/introduction.qmd",
    "execution_descriptors/execution_descriptors.qmd",
    "control_topologies/control_topologies.qmd",
    "trajectory_fine_tuning/trajectory_fine_tuning.qmd",
    "reinforcement_learning/reinforcement_learning.qmd",
    "test_time_search/test_time_search.qmd",
    "context_working_sets/context_working_sets.qmd",
    "prefix_caching_paging/prefix_caching_paging.qmd",
    "trajectory_scheduling/trajectory_scheduling.qmd",
    "tool_interfaces/tool_interfaces.qmd",
    "sandboxing_isolation/sandboxing_isolation.qmd",
    "verification_recovery/verification_recovery.qmd",
    "multi_agent_coordination/multi_agent_coordination.qmd",
    "telemetry_evaluation/telemetry_evaluation.qmd",
    "conclusion/conclusion.qmd",
]

VOL3_DIR = Path("publishing/quarto/contents/vol3")

PASSES = {
    "pass1": {
        "name": "Systems Principles Check",
        "description": "Check whether we are covering all timeless principles and concepts. Ensure we focus on system implications, not just algorithmics. We are an MLSys book.",
        "needs_context": False
    },
    "pass2": {
        "name": "Progressive Disclosure",
        "description": "Ensure we maintain progressive disclosure. If a concept is assumed, verify it was introduced in a previous chapter. Do not assume future knowledge.",
        "needs_context": True
    },
    "pass3": {
        "name": "Footnotes & Progressive Disclosure",
        "description": "Audit footnotes. Are we using them correctly for progressive disclosure? Ensure footnotes don't introduce critical concepts that should be in the main text.",
        "needs_context": True
    },
    "pass4": {
        "name": "Cross-References",
        "description": "Add explicit Quarto cross-references (@sec-vol3-...) to other sections. Link concepts backward and forward where appropriate.",
        "needs_context": True
    }
}

async def run_agent_on_chapter(chapter_idx: int, pass_id: str):
    chapter_rel = CHAPTERS[chapter_idx]
    chapter_path = VOL3_DIR / chapter_rel
    if not chapter_path.exists():
        print(f"[ERROR] Chapter not found: {chapter_path}")
        return
        
    pass_info = PASSES[pass_id]
    
    # Build context for progressive disclosure passes
    context = ""
    if pass_info["needs_context"]:
        context = "### Context from Previous Chapters\n"
        if chapter_idx == 0:
            context += "This is the first chapter. No previous chapters exist.\n"
        else:
            for i in range(chapter_idx):
                prev_path = VOL3_DIR / CHAPTERS[i]
                if prev_path.exists():
                    # For a real run, reading the whole file might blow up context,
                    # but modern models have large contexts. 
                    # We inject the current text of previous chapters as read-only context.
                    content = prev_path.read_text(encoding="utf-8")
                    context += f"\n==== PREVIOUS CHAPTER: {CHAPTERS[i]} ====\n{content}\n"

    prompt = f"""
You are an expert Systems and Machine Learning Editor.
We are executing a targeted editorial pass on the book "Machine Learning Systems, Volume III: Agentic Machine Learning Systems".

TARGET FILE TO EDIT: {chapter_path.absolute()}

CURRENT PASS: {pass_info['name']}
PASS INSTRUCTIONS: {pass_info['description']}

{context}

YOUR TASK:
1. Use `view_file` to read the target file.
2. Evaluate the file against the PASS INSTRUCTIONS.
3. Use `edit_file` to apply your improvements directly to the file.
4. When finished, use `finish` to return a concise summary of the changes you made.
"""

    print(f"[{chapter_rel}] Starting agent for {pass_id}...")
    
    config = LocalAgentConfig(
        system_instructions=TemplatedSystemInstructions(
            identity="You are an expert Editor for a graduate-level CS textbook on Machine Learning Systems.",
            mandate="Strictly follow the pass instructions. Use file tools to read and edit the target chapter. Return a summary."
        ),
        capabilities=CapabilitiesConfig(
            enable_mcp_tools=False,
            enable_subagents=False
        )
    )
    
    async with Agent(config) as agent:
        response = await agent.chat(prompt)
        print(f"\n======================================================\n[{chapter_rel}] Finished:\n{await response.text()}\n======================================================\n")

async def main():
    parser = argparse.ArgumentParser(description="Orchestrate parallel agent passes over Volume III chapters.")
    parser.add_argument("pass_id", choices=PASSES.keys(), help="Which pass to execute (pass1, pass2, pass3, pass4)")
    args = parser.parse_args()
    
    pass_info = PASSES[args.pass_id]
    print(f"🚀 Starting {args.pass_id}: {pass_info['name']}")
    print(f"Instructions: {pass_info['description']}\n")
    
    # Run agents concurrently for all chapters
    tasks = []
    for idx in range(len(CHAPTERS)):
        tasks.append(run_agent_on_chapter(idx, args.pass_id))
        
    await asyncio.gather(*tasks)
    print(f"✅ Pass {args.pass_id} complete. You should now review and `git commit` before running the next pass.")

if __name__ == "__main__":
    asyncio.run(main())
