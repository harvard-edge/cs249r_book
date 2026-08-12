import logging
from typing import Any

logger = logging.getLogger(__name__)

class DifferentialExplainer:
    """
    Automates the 'Why did this happen?' analysis by comparing two solver results.
    It identifies the binding constraints and mathematically explains the performance delta.
    """
    
    @staticmethod
    def compare_performance(baseline: Any, proposal: Any) -> str:
        """
        Compares two PerformanceProfile objects (or dicts containing them)
        and explains the speedup (or lack thereof) based on the Iron Law and Roofline regime.
        """
        try:
            # Handle if the input is a raw PerformanceProfile or wrapped in a dict
            b_prof = baseline["SingleNodeModel"] if isinstance(baseline, dict) and "SingleNodeModel" in baseline else baseline
            p_prof = proposal["SingleNodeModel"] if isinstance(proposal, dict) and "SingleNodeModel" in proposal else proposal
            
            if not hasattr(b_prof, "latency") or not hasattr(p_prof, "latency"):
                return "Cannot perform differential analysis: missing latency metrics."

            # We know it has latency via the hasattr check. Cast to Any to bypass pyright's dict inference.
            b_prof_any: Any = b_prof
            p_prof_any: Any = p_prof
            
            b_lat = getattr(b_prof_any, "latency").m_as("ms")
            p_lat = getattr(p_prof_any, "latency").m_as("ms")
            speedup = b_lat / p_lat if p_lat > 0 else 0

            # Bottleneck string matching can be tricky depending on how it's formatted in the engine
            b_neck = getattr(b_prof_any, "bottleneck", "Unknown")
            p_neck = getattr(p_prof_any, "bottleneck", "Unknown")
            
            # Normalize to simple "Memory" or "Compute"
            b_is_mem = "memory" in str(b_neck).lower()
            p_is_mem = "memory" in str(p_neck).lower()

            lines = [
                "📊 Differential Analysis: Proposal vs. Baseline",
                f"• Speedup: {speedup:.2f}x",
                f"• Baseline Regime: {b_neck}",
                f"• Proposal Regime: {p_neck}",
                ""
            ]

            if speedup < 1.0:
                lines.append("Conclusion: The proposed change decreased performance.")
                return "\n".join(lines)

            if b_is_mem and p_is_mem:
                lines.append("Analysis: The workload remained Memory Bound. The speedup is constrained strictly by the ratio of HBM bandwidth between the two configurations. Any additional compute capacity (FLOP/s) in the proposal was left unutilized.")
            elif not b_is_mem and not p_is_mem:
                lines.append("Analysis: The workload remained Compute Bound. The speedup is constrained strictly by the ratio of peak arithmetic throughput (FLOP/s) between the two configurations. Additional memory bandwidth was not the limiting factor.")
            elif b_is_mem and not p_is_mem:
                lines.append("Analysis: Regime Shift. The proposed hardware provided enough memory bandwidth to lift the memory wall, shifting the bottleneck to compute. The speedup is a blended result of higher bandwidth and the new compute ceiling.")
            elif not b_is_mem and p_is_mem:
                lines.append("Analysis: Regime Shift. The proposed hardware provided enough compute capacity to lift the compute wall, exposing a memory bandwidth bottleneck. The speedup is a blended result of higher compute and the new memory ceiling.")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Differential analysis failed: {e}")
            return f"Differential analysis unavailable: {str(e)}"
