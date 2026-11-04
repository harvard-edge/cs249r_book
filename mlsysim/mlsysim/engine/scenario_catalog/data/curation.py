"""Named data scenarios for curation.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....literature.registry import Literature as _Literature
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ... import scenario_types as _st


# --- Trajectory curation: triage, splits, admission, denominators, staging (added 2026-10-20) ---
RepairAgentFailureTriage = _st.FailureTriageScenario(
    name="A maintenance agent's month of failures, triaged by counterfactual probes",
    description="50,000 trajectories at 6 percent failing; probes attribute 70 percent to a tool schema that "
    "omits a required field and 20 percent to sandbox startup timeouts. Fine-tuning on every "
    "failure: Llama 3 70B, 4,096-token sequences, 3 epochs, one node of eight H100s.",
    runs=50_000,  # scenario assumption: trajectories per month
    failure_rate=0.06,  # scenario assumption: failing share
    probe_shares={
        "interface": 0.70,  # scenario assumption: schema probe removes the failure
        "runtime": 0.20,
    },  # scenario assumption: runtime probe removes the failure
    workload=_Models.Language.Llama3_70B,
    hardware=_Hardware.Cloud.H100,
    devices=8,  # scenario assumption: one node
    utilization=float(_Literature.Training.MfuLow),
    hourly_price=_Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate,
    sequence_tokens=4_096,  # scenario assumption: tokens per training sequence
    epochs=3,  # scenario assumption
)


EightAttemptRunSplit = _st.RunSplitLeakageScenario(
    name="Eight attempts per task, each run sent to training with probability 0.8",
    description="A random 80/20 split of runs rather than of tasks.",
    attempts=8,  # scenario assumption: attempts per task
    train_fraction=0.8,  # scenario assumption
)


TrajectoryMinHashScreen = _st.MinHashBandingScenario(
    name="A 128-hash MinHash screen in 32 bands of 4 rows over 100,000 trajectory records",
    description="Command n-grams and code-tree features per record; 32-bit hash values stored raw.",
    bands=32,  # scenario assumption
    rows=4,  # scenario assumption
    probe_similarities=[0.5, 0.8],  # scenario assumption: a loose and a near-duplicate pair
    hash_bytes=Q_(4, "byte"),  # 32-bit hash values
    records=100_000,  # scenario assumption
)


DemonstrationAdmissionOrder = _st.AdmissionOrderScenario(
    name="A demonstration route's five admission checks, process audit before tests and after",
    description="Record, use, process, outcome, and target checks, independent, on 100,000 candidates; "
    "the two orders swap the process audit and the test run.",
    checks={
        "record": (0.95, Q_(0.01, "second")),  # scenario assumption: pass rate, slot time
        "use": (0.90, Q_(0.04, "second")),  # scenario assumption
        "process": (0.80, Q_(0.20, "second")),  # scenario assumption
        "outcome": (0.40, Q_(8.0, "second")),  # scenario assumption
        "target": (0.75, Q_(2.0, "second")),
    },  # scenario assumption
    orders=[["record", "use", "process", "outcome", "target"], ["record", "use", "outcome", "process", "target"]],
    candidates=100_000,  # scenario assumption
)


DemonstrationAdmissionOrderSlowAudit = DemonstrationAdmissionOrder.variant(
    name="The same route with a slower, less selective process audit",
    description="The process audit takes 2 s and passes 95 percent, so the tests should run first.",
    checks={**DemonstrationAdmissionOrder.checks, "process": (0.95, Q_(2.0, "second"))},  # scenario assumption
)


RecoveryPoolDenominators = _st.CorpusDenominatorsScenario(
    name="A retained pool of 50,000 direct, recovery, and negative records",
    description="Recovery records are longer and only their corrections are targets; the demonstration "
    "route admits direct and recovery records.",
    classes={
        "direct": (32_500, Q_(7_200, "token"), Q_(1_200, "token")),  # scenario assumption
        "recovery": (12_500, Q_(18_500, "token"), Q_(2_000, "token")),  # scenario assumption
        "negative": (5_000, Q_(9_800, "token"), Q_(0, "token")),
    },  # scenario assumption
    focus="recovery",
    admitted=["direct", "recovery"],
)


EightAttemptGroupContrast = _st.AttemptGroupContrastScenario(
    name="Eight attempts per fixture at pass rates of 5, 50, and 95 percent",
    description="How often a group of attempts holds a success, and a success beside a failure.",
    attempts=8,  # scenario assumption
    pass_rates=[0.05, 0.50, 0.95],  # scenario assumption
)


VerifierTimeoutStagingBurst = _st.StagingBurstScenario(
    name="A collector's staging store during a verifier timeout burst",
    description="1,024 rollouts of 45 s feed 512 verifier slots; a defective fixture drives every test to "
    "its 120 s timeout; 8 MB record ceiling, 32 GB store, 30 s controller delay.",
    rollouts=1_024,  # scenario assumption
    rollout_time=Q_(45, "second"),  # scenario assumption
    slots=512,  # scenario assumption
    mean_service=Q_(16.875, "second"),  # scenario assumption
    timeout=Q_(120, "second"),  # scenario assumption
    record_ceiling=Q_(8, "MB"),  # scenario assumption
    store=Q_(32, "GB"),  # scenario assumption
    reaction=Q_(30, "second"),  # scenario assumption
)


CurationCohortDispositions = _st.CohortDispositionScenario(
    name="A cohort of 500 collection attempts by disposition at the cutoff",
    description="Fifteen selected and emitted; the rest assessed but not selected, ineligible, pending, "
    "or unknown. Environments, generation, tools, checks, storage, and review cost $300.",
    dispositions={
        "selected_emitted": 15,  # scenario assumption
        "assessed_not_selected": 470,  # scenario assumption
        "ineligible": 10,  # scenario assumption
        "pending": 3,  # scenario assumption
        "unknown": 2,
    },  # scenario assumption
    emitted="selected_emitted",
    cohort_cost=Q_(300, "USD"),  # scenario assumption
    yield_multiple=2.0,  # scenario assumption: the same cohort at double the yield
)
