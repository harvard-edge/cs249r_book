"""
Agentic Infrastructure Design Loop (Conceptual Implementation)
============================================================
Vision: "AI designing AI infrastructure."

This script demonstrates how an advanced multi-agent system (e.g., powered
by multiple Gemini-capable models like gemini-3-pro-preview) can use MLSys·im 
to autonomously design, debate, and refine a datacenter cluster.

We simulate two agents:
1. The "Architect": Generates cluster configurations (YAML) to meet an SLA.
2. The "Critic/Evaluator": Runs MLSys·im, reads the physics output, and points out
   bottlenecks (e.g., "We hit the memory wall here, increase batch size or nodes.")

This is the exact loop that makes MLSys·im the de facto standard: it's not just a 
calculator for humans; it's a physics engine for autonomous AI engineers.
"""

import os
import yaml
import json
import time

# In a real environment, this would be google.generativeai or similar.
# import google.generativeai as genai

# We mock the LLM responses for the sake of the reproducible example in the repo.
class MockGeminiAgent:
    def __init__(self, role: str):
        self.role = role
        self.history = []

    def prompt(self, text: str, tools=None) -> str:
        """Simulates calling a frontier Gemini model."""
        print(f"\n[{self.role.upper()} AGENT] Thinking...")
        time.sleep(1)
        
        if "Initial Request" in text:
            return """
version: "1.0"
name: "Llama3 70B First Attempt"
workload:
  name: "Llama3_70B"
  batch_size: 256
hardware:
  name: "H100"
  nodes: 1
ops:
  region: "US_Avg"
  duration_days: 30.0
"""
        elif "FAIL" in text and "OOM" in text:
            print(f"[{self.role.upper()} AGENT] Noticed Memory Wall failure. Adjusting parallel nodes.")
            return """
version: "1.0"
name: "Llama3 70B Distributed Attempt"
workload:
  name: "Llama3_70B"
  batch_size: 256
hardware:
  name: "H100"
  nodes: 8
ops:
  region: "Quebec"
  duration_days: 30.0
"""
        return "Task Complete."


def run_agentic_loop():
    from mlsysim.cli.schemas import MlsysPlanSchema
    from mlsysim.core.evaluation import SystemEvaluator

    print("==================================================")
    print("🚀 INITIALIZING MLSYS·IM AGENTIC DESIGN LOOP")
    print("==================================================")
    
    architect = MockGeminiAgent(role="Architect")
    
    # The Goal SLA
    goal = "Design a cluster to serve Llama3_70B. Keep it under 10 nodes if possible. Minimize carbon."
    print(f"\n[USER] Goal: {goal}")

    iteration = 1
    max_iterations = 3
    current_prompt = f"Initial Request: {goal}. Output ONLY the YAML."

    while iteration <= max_iterations:
        print(f"\n--- Iteration {iteration} ---")
        
        # 1. Agent generates YAML
        yaml_str = architect.prompt(current_prompt).strip()
        print("Proposed Architecture YAML:")
        print(yaml_str)
        
        # 2. Execute against MLSys·im Physics Engine
        raw_data = yaml.safe_load(yaml_str)
        try:
            schema = MlsysPlanSchema(**raw_data)
            eval_obj = SystemEvaluator.evaluate(
                scenario_name=schema.name,
                model_obj=schema.model_obj,
                hardware_obj=schema.hardware_obj,
                batch_size=schema.workload.batch_size,
                precision=schema.hardware.precision,
                efficiency=schema.hardware.efficiency,
                fleet_obj=schema.fleet_obj,
                nodes=schema.hardware.nodes,
                duration_days=schema.ops.duration_days
            )
            
            result_dict = eval_obj.to_dict()
            
            # 3. Analyze output (The Critic)
            if result_dict["f_status"] == "FAIL":
                feedback = f"Feasibility FAIL. Summary: {eval_obj.feasibility.summary}. Please fix the OOM issue."
                print(f"[ENVIRONMENT] ❌ {feedback}")
                current_prompt = f"Previous YAML failed: {feedback}. Output a new corrected YAML."
            else:
                print("[ENVIRONMENT] ✅ Design is physically feasible.")
                print(f"   Throughput: {result_dict.get('p_throughput', 'N/A')}")
                print(f"   TCO ($):    ${result_dict.get('m_tco_usd', 0):,.2f}")
                print(f"   Carbon:     {result_dict.get('m_carbon_footprint', 0):.2f} tonnes")
                print("\n[SUCCESS] Agent reached optimal configuration.")
                break
                
        except Exception as e:
            print(f"[ENVIRONMENT] ❌ Crash evaluating YAML: {e}")
            current_prompt = f"YAML parsing or execution failed with error: {e}. Fix the schema."
            
        iteration += 1

if __name__ == "__main__":
    run_agentic_loop()
