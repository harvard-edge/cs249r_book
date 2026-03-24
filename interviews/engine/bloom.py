"""
Bloom's Revised Taxonomy mapping for question generation.

Based on: Anderson & Krathwohl (2001), "A Taxonomy for Learning,
Teaching, and Assessing" — the revised Bloom's taxonomy.

Each mastery level maps to specific cognitive verbs, question
templates (item models), and structural constraints. This is the
backbone of the Automatic Item Generation (AIG) approach from
Gierl & Haladyna (2013).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BloomLevel:
    """A single Bloom's taxonomy level with generation constraints."""

    level: str
    cognitive_process: str
    description: str
    verb_stems: list[str]
    question_templates: list[str]
    structural_requirements: dict[str, bool]
    distractor_strategy: str


# ---------------------------------------------------------------------------
# The six levels, mapped to StaffML's L1-L6+
# ---------------------------------------------------------------------------

BLOOM_LEVELS: dict[str, BloomLevel] = {
    "L1": BloomLevel(
        level="L1",
        cognitive_process="Remember",
        description="Retrieve relevant knowledge from long-term memory",
        verb_stems=[
            "define", "identify", "recall", "recognize",
            "state", "list", "name", "describe",
        ],
        question_templates=[
            # Pure recall — "what is the value of X?"
            (
                "What is the approximate {metric} of {hardware_component}? "
                "Express your answer in {units}."
            ),
            # Recognition — "which of these is true?"
            (
                "Which operation consumes more {resource}: "
                "{operation_a} or {operation_b}?"
            ),
            # Ratio recall — "how much faster/slower is X than Y?"
            (
                "Roughly how much {comparison} is {component_a} "
                "compared to {component_b}?"
            ),
        ],
        structural_requirements={
            "napkin_math": True,   # Even L1 should have a quick number
            "options": False,      # Optional MCQ
            "key_equation": False,
            "scenario_realistic": False,  # Can be direct questions
        },
        distractor_strategy=(
            "Off by order of magnitude. Distractors should be 10x or 100x "
            "wrong — testing whether students know the right ballpark."
        ),
    ),

    "L2": BloomLevel(
        level="L2",
        cognitive_process="Understand",
        description="Construct meaning from instructional messages — single-variable math",
        verb_stems=[
            "calculate", "compare", "contrast", "explain",
            "interpret", "estimate", "classify", "summarize",
        ],
        question_templates=[
            # Single-variable calculation — "given X, compute Y"
            (
                "A model has {param_count} parameters. How much {resource} "
                "does it require in {precision} precision?"
            ),
            # Ratio application — "given specs, compute the ratio"
            (
                "If an accelerator has {compute_spec} of compute and "
                "{bandwidth_spec} of memory bandwidth, what is its {derived_metric}?"
            ),
            # Unit conversion reasoning
            (
                "A {device} consumes {power} during inference. If your "
                "{power_source} has {capacity}, how long can you run continuously?"
            ),
        ],
        structural_requirements={
            "napkin_math": True,   # Required — this IS the answer
            "options": False,
            "key_equation": True,  # Should include the formula
            "scenario_realistic": False,
        },
        distractor_strategy=(
            "Unit confusion and off-by-2x errors. Distractors exploit "
            "mixing bytes/bits, forgetting ×2 for FP16, or confusing "
            "power (W) with energy (Wh)."
        ),
    ),

    "L3": BloomLevel(
        level="L3",
        cognitive_process="Apply",
        description="Carry out a procedure in a given situation — multi-step diagnosis",
        verb_stems=[
            "apply", "demonstrate", "solve", "use",
            "diagnose", "implement", "compute", "determine",
        ],
        question_templates=[
            # Scenario diagnosis — "you observe X, what's wrong?"
            (
                "You are {role} working on {task}. You observe {symptom}. "
                "{monitoring_data}. What is the most likely {bottleneck_type}?"
            ),
            # Multi-step calculation with real hardware
            (
                "You're running {workload} on {hardware}. "
                "Calculate the {metric} and determine whether this operation "
                "is {bound_type_a}-bound or {bound_type_b}-bound."
            ),
            # Design trade-off with numbers
            (
                "Your team must choose between {option_a} and {option_b} "
                "for {use_case}. {constraint}. Which is the better choice "
                "and why?"
            ),
        ],
        structural_requirements={
            "napkin_math": True,   # Required — multi-step chain
            "options": True,       # MCQ recommended at this level
            "key_equation": True,
            "scenario_realistic": True,  # Must be a real scenario
        },
        distractor_strategy=(
            "Correct reasoning about wrong bottleneck. Distractors should "
            "identify a real system component but misattribute the cause. "
            "E.g., blaming the GPU when the CPU is starving it."
        ),
    ),

    "L4": BloomLevel(
        level="L4",
        cognitive_process="Analyze",
        description="Break material into parts and determine relationships — system-level",
        verb_stems=[
            "analyze", "differentiate", "distinguish", "examine",
            "compare", "deconstruct", "attribute", "organize",
        ],
        question_templates=[
            # Cross-component analysis
            (
                "{setup_context}. Your profiler shows {metric_a} but "
                "{metric_b} tells a different story. What interaction "
                "between {component_a} and {component_b} explains this?"
            ),
            # Scaling analysis
            (
                "You benchmark {system} going from {scale_a} to {scale_b}. "
                "You expected {expected_result} but observed {actual_result}. "
                "What explains the sub-{expected} scaling?"
            ),
            # Cost/performance trade-off with multiple variables
            (
                "{business_context}. You must choose the {resource_type} "
                "degree: {option_list}. Calculate the optimal choice for "
                "{target_metric} of {target_value}."
            ),
        ],
        structural_requirements={
            "napkin_math": True,   # Required — detailed chain
            "options": False,      # Open-ended preferred
            "key_equation": True,
            "scenario_realistic": True,
        },
        distractor_strategy=(
            "True for a different system or regime. Distractors should be "
            "correct analysis for a different hardware, scale, or workload "
            "type — testing whether students can scope their reasoning."
        ),
    ),

    "L5": BloomLevel(
        level="L5",
        cognitive_process="Evaluate",
        description="Make judgments based on criteria — predict system behavior under stress",
        verb_stems=[
            "evaluate", "justify", "predict", "assess",
            "critique", "defend", "prioritize", "recommend",
        ],
        question_templates=[
            # Failure prediction
            (
                "{system_description}. Under {stress_condition}, the system "
                "didn't just {expected_degradation}; it {actual_failure}. "
                "Why did the system fail non-linearly?"
            ),
            # Design review
            (
                "Your team proposes {design_decision}. {constraints}. "
                "What failure mode does this introduce, and at what scale "
                "does it become catastrophic?"
            ),
            # Cross-layer optimization
            (
                "{optimization_context}. You implement {technique}. "
                "Under {condition_a}, {metric} improves. Under {condition_b}, "
                "it gets worse. Why does the optimization backfire?"
            ),
            # Operational judgment (per Chip Huyen review)
            (
                "Your model's {metric} dropped {amount} last {time_period}. "
                "Nothing changed in the model code. Walk me through your "
                "investigation: what do you check first, second, third?"
            ),
            # Production triage
            (
                "{production_system} is serving {traffic}. You receive an alert: "
                "{alert_description}. Your on-call playbook says {playbook_action}. "
                "Why is the playbook wrong for this specific failure, and what "
                "should you do instead?"
            ),
        ],
        structural_requirements={
            "napkin_math": True,   # Required — must quantify the failure
            "options": False,
            "key_equation": True,
            "scenario_realistic": True,
        },
        distractor_strategy=(
            "Correct solution at wrong scale. Distractors should be valid "
            "engineering approaches that work at a different scale but fail "
            "at the scale in the question. Tests systems thinking."
        ),
    ),

    "L6+": BloomLevel(
        level="L6+",
        cognitive_process="Create",
        description="Put elements together to form a novel whole — architect from constraints",
        verb_stems=[
            "design", "construct", "develop", "formulate",
            "propose", "architect", "synthesize", "derive",
        ],
        question_templates=[
            # Blank-slate architecture
            (
                "You are the {role} at {company}. {business_requirement}. "
                "Your constraints: {constraint_list}. Design the {system_type} "
                "from scratch. What are your first three architectural decisions "
                "and why?"
            ),
            # Chaos engineering scenario
            (
                "{system_running_normally}. At {time}, {chaos_event}. "
                "{cascading_effects}. How do you triage, and what architectural "
                "change prevents this class of failure?"
            ),
            # Novel trade-off synthesis
            (
                "{emerging_technology} promises {benefit}. Your existing "
                "system uses {current_approach}. {constraints}. Derive the "
                "break-even point where switching becomes worthwhile."
            ),
            # End-to-end operational ownership (per Chip Huyen review)
            (
                "You inherit a {system_description} that has been running for "
                "{duration}. The previous team left no runbooks. In your first "
                "week, you discover {problem_list}. Prioritize the fixes and "
                "explain why your ordering is correct."
            ),
        ],
        structural_requirements={
            "napkin_math": True,   # Required — cost/scale analysis
            "options": False,      # Never MCQ at this level
            "key_equation": False,
            "scenario_realistic": True,
        },
        distractor_strategy=(
            "Not applicable — L6+ questions are open-ended. The 'distractor' "
            "is the candidate's own biases and assumptions. The napkin math "
            "reveals whether their intuition matches the physics."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Competency areas from TOPIC_MAP.md
# ---------------------------------------------------------------------------

COMPETENCY_AREAS: list[str] = [
    "compute-analysis",
    "memory-systems",
    "numerical-representation",
    "model-architecture-cost",
    "latency-throughput",
    "power-thermal",
    "model-optimization",
    "deployment-serving",
    "monitoring-reliability",
    "security-privacy-fairness",
]


def get_bloom_level(level: str) -> BloomLevel:
    """Get the Bloom's level configuration for a given mastery level."""
    if level not in BLOOM_LEVELS:
        raise ValueError(
            f"Unknown level '{level}'. Valid: {list(BLOOM_LEVELS.keys())}"
        )
    return BLOOM_LEVELS[level]


def get_generation_prompt_context(level: str) -> str:
    """Build the Bloom's-specific portion of the generation prompt."""
    bl = get_bloom_level(level)
    verbs = ", ".join(bl.verb_stems[:5])
    templates = "\n".join(f"  - {t}" for t in bl.question_templates)

    return f"""## Cognitive Level: {bl.level} — {bl.cognitive_process}
{bl.description}

### Verb stems to use in the question:
{verbs}

### Question template patterns (adapt, don't copy verbatim):
{templates}

### Distractor strategy:
{bl.distractor_strategy}

### Structural requirements:
- Napkin math: {"REQUIRED" if bl.structural_requirements["napkin_math"] else "optional"}
- MCQ options: {"recommended" if bl.structural_requirements["options"] else "not needed"}
- Key equation: {"include" if bl.structural_requirements["key_equation"] else "optional"}
- Realistic scenario: {"REQUIRED" if bl.structural_requirements["scenario_realistic"] else "can be direct"}
"""
