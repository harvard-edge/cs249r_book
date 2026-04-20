"""
MLPerf EDU: Dual-Agent LLM Researcher.

Agent 1 (Architect) proposes architecture modifications based on loss traces.
Agent 2 (Critic) verifies proposals via AST analysis before execution.
"""

import ast
import os
import re
import subprocess
from dataclasses import dataclass


@dataclass
class CriticResult:
    """Structured output from the Critic agent."""
    passed: bool
    reason: str


class SecurityASTVerifier(ast.NodeVisitor):
    """Validates that proposed code contains no forbidden imports."""

    FORBIDDEN_IMPORTS = {'os', 'sys', 'subprocess', 'shutil', 'pathlib'}

    def __init__(self):
        self.is_secure = True
        self.violations = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in self.FORBIDDEN_IMPORTS:
                self.is_secure = False
                self.violations.append(f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.split('.')[0] in self.FORBIDDEN_IMPORTS:
            self.is_secure = False
            self.violations.append(f"Forbidden import from: {node.module}")
        self.generic_visit(node)


class UltimateAutonomousScientist:
    """
    Dual-Agent Actor/Critic research loop.

    The Architect (Agent 1) proposes model code modifications.
    The Critic (Agent 2) reviews them via AST analysis.
    Approved proposals are written to disk for re-import by the trainer.
    """

    def __init__(self, target_file_path: str, model_name: str):
        self.target_file = target_file_path
        self.model_name = model_name
        self.log_file = "checkpoints/research_log.md"
        self.context_buffer = []
        os.makedirs("checkpoints", exist_ok=True)

    def _write_and_commit(self, code: str, iteration: int):
        """Save the proposed code and create a git checkpoint."""
        version_file = f"checkpoints/{self.model_name}_v{iteration}.py"
        with open(version_file, "w") as f:
            f.write(code)
        with open(self.target_file, "w") as f:
            f.write(code)

        # Best-effort git commit
        try:
            subprocess.run(["git", "add", self.target_file],
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["git", "commit", "-m",
                            f"Auto-Scientist Iteration {iteration}: {self.model_name}"],
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass  # git not available

    def _verify_code_security(self, code: str) -> tuple[bool, list[str]]:
        """Validate proposed code via AST parsing and security checks."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]

        verifier = SecurityASTVerifier()
        verifier.visit(tree)
        return verifier.is_secure, verifier.violations

    def _spawn_critic(self, proposed_code: str) -> CriticResult:
        """Agent 2: The Critic reviews proposed code for correctness."""
        print(f"[{self.model_name}] 🧑‍🏫 Spawning Critic (Agent 2)...")

        critic_prompt = (
            f"You are a Senior Staff PyTorch Engineer reviewing code from a junior AI. "
            f"Review the following nn.Module. Check for:\n"
            f"1. Tensor shape mismatches in forward()\n"
            f"2. Missing .to(device) calls\n"
            f"3. Syntax errors\n"
            f"4. Dead code paths\n"
            f"If the code is safe to execute, reply EXACTLY with 'PASS'. "
            f"Otherwise, explain the error concisely.\n\n"
            f"{proposed_code}"
        )

        try:
            result = subprocess.run(
                ["gemini", "ask", critic_prompt],
                capture_output=True, text=True, timeout=60
            )
            response = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            # Gemini CLI not available — fall back to AST-only verification
            print(f"[{self.model_name}] ⚠️ Gemini CLI unavailable ({e}), using AST-only check")
            is_secure, violations = self._verify_code_security(proposed_code)
            if is_secure:
                return CriticResult(passed=True, reason="AST-only: no violations found")
            else:
                return CriticResult(passed=False, reason=f"AST violations: {violations}")

        if "PASS" in response[:20].upper():
            print(f"[{self.model_name}] ✅ Critic approved.")
            return CriticResult(passed=True, reason="Critic approved")
        else:
            print(f"[{self.model_name}] ❌ Critic rejected: {response[:200]}")
            self.context_buffer.append(f"Critic rejected: {response[:500]}")
            return CriticResult(passed=False, reason=response[:500])

    def execute_research_loop(self, loss_delta: list, validation_accuracy: list,
                              target_loss: float, iteration: int) -> bool:
        """
        Run one iteration of the Dual-Agent research loop.

        Args:
            loss_delta: Recent training loss values
            validation_accuracy: Recent validation loss values
            target_loss: Target convergence threshold
            iteration: Current NAS attempt number

        Returns:
            True if a new architecture was successfully written to disk.
        """
        print(f"\n[{self.model_name}] 🧪 Plateau detected. Running Dual-Agent NAS (iter {iteration})...")

        with open(self.target_file, "r") as f:
            current_code = f.read()

        # Build Architect prompt with loss context and experiment history
        prompt = (
            f"You are an elite PyTorch Systems Engineer (Agent 1).\n"
            f"Model: {self.model_name}\n\n"
            f"=== DIAGNOSTICS ===\n"
            f"Recent train loss: {loss_delta}\n"
            f"Recent val loss: {validation_accuracy}\n"
            f"Target: {target_loss}\n\n"
        )
        if self.context_buffer:
            prompt += (
                f"=== EXPERIMENT HISTORY (do NOT repeat these failures) ===\n"
                f"{self.context_buffer[-3:]}\n\n"
            )
        prompt += (
            f"Rewrite the following PyTorch nn.Module to improve convergence. "
            f"Keep the class name and constructor signature identical. "
            f"Wrap your code in ```python ... ``` blocks.\n\n"
            f"{current_code}"
        )

        # Invoke Architect (Agent 1)
        print(f"[{self.model_name}] 🤖 Spawning Architect (Agent 1)...")
        try:
            result = subprocess.run(
                ["gemini", "ask", prompt],
                capture_output=True, text=True, timeout=120
            )
            response = result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[{self.model_name}] ⚠️ Architect failed ({e}), skipping NAS iteration")
            return False

        # Extract Python code from response
        match = re.search(r'```python(.*?)```', response, re.DOTALL)
        if not match:
            print(f"[{self.model_name}] ⚠️ No code block found in Architect response")
            self.context_buffer.append("Architect produced no valid code block")
            return False

        new_code = match.group(1).strip()

        # Security verification (AST sandbox)
        is_secure, violations = self._verify_code_security(new_code)
        if not is_secure:
            print(f"[{self.model_name}] 🔒 Security check failed: {violations}")
            self.context_buffer.append(f"Security violation: {violations}")
            return False

        # Critic review
        critic_result = self._spawn_critic(new_code)
        if not critic_result.passed:
            return False

        # All checks passed — write to disk
        self._write_and_commit(new_code, iteration)
        print(f"[{self.model_name}] ✅ New architecture written. Ready for re-training.")

        # Log the iteration
        with open(self.log_file, "a") as f:
            f.write(f"\n## Iteration {iteration}: {self.model_name}\n")
            f.write(f"- Train loss: {loss_delta[-1]:.4f}\n")
            f.write(f"- Critic: {critic_result.reason}\n")
            f.write(f"- Status: APPROVED\n")

        return True


if __name__ == "__main__":
    agent = UltimateAutonomousScientist("reference/cloud/nano_moe.py", "nano-moe-12m")
    agent.execute_research_loop([3.4, 3.3, 3.3], [10.2, 10.1, 10.1], 1.5, 1)
