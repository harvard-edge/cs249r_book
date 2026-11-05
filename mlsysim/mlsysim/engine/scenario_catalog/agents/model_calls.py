"""Named agents scenarios for model calls.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....agents.registry import Agents as _Agents
from ... import scenario_types as _st


# --- Model calls and agent trajectories (added 2026-10-20) ---------------
# Per-token floors for one model call, then the calls a coding agent and a
# support agent make. Token counts are teaching assumptions, not
# measurements; the floors exclude attention, batching, and queueing.
Llama3_70B_Int8DecodeH100 = _st.DecodeScenario(
    name="Llama-3 70B, 8-bit weights, at batch 1 on one H100",
    description="Per-token floors for one unbatched call: compute_time = 2P / FP8 peak is the "
    "prefill floor per prompt token; memory_time = P x 1 byte / HBM bandwidth is the "
    "decode floor per output token.",
    workload=_Models.Language.Llama3_70B,
    hardware=_Hardware.Cloud.H100,
    precision="int8",  # scenario assumption: 8-bit weights
    peak_precision="fp8",  # scenario assumption: prefill at the FP8 tensor-core peak
)


CodingAgentResend30Turns = _st.TrajectoryResendScenario(
    name="Coding-agent trajectory that resends its history for 30 turns",
    description="An 8,000-token first prompt; each turn appends 1,500 tokens of model output "
    "and tool observation and generates 200 output tokens; the harness step budget "
    "sets the horizon.",
    floors=Llama3_70B_Int8DecodeH100,
    first_prompt=8000,  # scenario assumption: serialized first prompt, tokens
    appended=1500,  # scenario assumption: output plus observation appended per turn, tokens
    output_per_turn=200,  # scenario assumption: output tokens per call
    turns=_Agents.Coding.SWE_Bench_Runner.max_trajectory_steps,
)


CodingAgentResend60Turns = CodingAgentResend30Turns.variant(
    name="Coding-agent trajectory that resends its history for 60 turns", turns=2 * CodingAgentResend30Turns.turns
)


CodingAgentResend90Turns = CodingAgentResend30Turns.variant(
    name="Coding-agent trajectory that resends its history for 90 turns", turns=3 * CodingAgentResend30Turns.turns
)


CodingAgentFirstTurnCall = _st.ModelCallScenario(
    name="First call of the coding-agent trajectory",
    description="The 8,000-token first prompt and a 200-token reply at the 8-bit floors.",
    floors=Llama3_70B_Int8DecodeH100,
    prompt_tokens=CodingAgentResend30Turns.first_prompt,
    output_tokens=CodingAgentResend30Turns.output_per_turn,
)


CodingAgentWholeFileEdit = _st.ModelCallScenario(
    name="Coding-agent edit returned as a whole replacement file",
    description="The first prompt plus 350 tokens of format instructions; a 1,850-token "
    "replacement file; a 15 s call deadline.",
    floors=Llama3_70B_Int8DecodeH100,
    prompt_tokens=CodingAgentResend30Turns.first_prompt,
    instruction_tokens=350,  # scenario assumption: whole-file format instructions
    output_tokens=1850,  # scenario assumption: whole-file response tokens
    deadline=Q_(15, "second"),  # scenario assumption: call deadline
)


CodingAgentLocalizedEdit = CodingAgentWholeFileEdit.variant(
    name="Coding-agent edit returned as a localized edit",
    description="The first prompt plus 120 tokens of edit-format instructions; an 80-token "
    "edit; 95 percent apply and 60 percent of those pass the tests.",
    instruction_tokens=120,  # scenario assumption: localized-edit format instructions
    output_tokens=80,  # scenario assumption: localized-edit response tokens
    stage_success={
        "applies": 0.95,  # scenario assumption: edits whose anchor text matches
        "passes tests": 0.60,
    },  # scenario assumption: test pass rate given it applies
)


SupportAgentTurnCall = _st.ModelCallScenario(
    name="One support-agent turn at the 8-bit floors",
    description="The support workload's session context and reply length; the shared policy "
    "prefix can be served from cache.",
    floors=Llama3_70B_Int8DecodeH100,
    prompt_tokens=int(_Agents.Support.CustomerOps.session_context.to("token").magnitude),
    output_tokens=int(_Agents.Support.CustomerOps.reply_tokens.to("token").magnitude),
    shared_prefix_tokens=int(_Agents.Support.CustomerOps.shared_prefix.to("token").magnitude),
)


CanceledStreamDecodeSavings = _st.StreamCancelScenario(
    name="Streamed reply rejected at step 96 of a 4,096-token allowance",
    description="A stream monitor rejects the reply early; an honored cancel skips the "
    "remaining decode steps at the 8-bit decode floor.",
    floors=Llama3_70B_Int8DecodeH100,
    output_allowance=4096,  # scenario assumption: output allowance of the call
    reject_step=96,  # scenario assumption: decode step at which the monitor rejects
)


MultimodalContextAdmission = _st.ContextAdmissionScenario(
    name="Text request plus a screenshot against an 8,192-token window",
    description="A 6,000-token serialized text request with 1,024 tokens reserved for the "
    "reply; the vision front end charges 2,048 positions for the image.",
    window=8192,  # scenario assumption: combined context window, tokens
    output_reserve=1024,  # scenario assumption: reserved generation allowance
    text_tokens=6000,  # scenario assumption: serialized text request
    image_tokens=2048,  # scenario assumption: service-reported image charge
)


TemperatureNucleusExample = _st.SamplingScenario(
    name="Five hypothetical logits at temperature 1 and 0.5, then top-p 0.90",
    description="Logits [12, 10.5, 9, 6, 3]; halving the temperature sharpens the top token; "
    "top-p 0.90 at temperature 1 keeps two tokens.",
    logits=[12.0, 10.5, 9.0, 6.0, 3.0],  # scenario assumption: hypothetical logits
    base_temperature=1.0,
    sharp_temperature=0.5,  # scenario assumption
    top_p=0.90,  # scenario assumption
)


PatchCheckerPosterior = _st.CheckerPosteriorScenario(
    name="A fallible patch check over generated patches",
    description="18 percent of patches violate the concurrency property; the check passes "
    "65 percent of defective and 85 percent of valid patches.",
    defect_rate=0.18,  # scenario assumption
    false_accept_rate=0.65,  # scenario assumption: P(PASS | defective)
    valid_accept_rate=0.85,  # scenario assumption: P(PASS | valid)
)


PatchCheckerHalvedFalseAccept = PatchCheckerPosterior.variant(
    name="The same patch check with its false-accept rate halved",
    false_accept_rate=PatchCheckerPosterior.false_accept_rate / 2,
)
