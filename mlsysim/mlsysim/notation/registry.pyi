# Auto-generated type stubs for mlsysim.Notation / mlsysim.Symbols
from typing import Dict, Iterator, List
from .types import NotationSymbol

class NotationRegistry:
    def __getitem__(self, name: str) -> NotationSymbol: ...
    def __iter__(self) -> Iterator[NotationSymbol]: ...
    def __len__(self) -> int: ...
    def list(self) -> List[NotationSymbol]: ...
    def for_category(self, category: str) -> List[NotationSymbol]: ...
    def for_volume(self, volume: str) -> List[NotationSymbol]: ...

    ABudget: NotationSymbol  # A budget ($B$) [$B$ is batch size. The trajectory budget is bold $\mathbf{B}$, and the reasoning budget is $B_{\text{reason}}$.]
    ACommitHash: NotationSymbol  # A commit hash ($h$)
    AFourthTaxonomyAxis: NotationSymbol  # A fourth taxonomy axis; a grant or commit ($C$) [There is no $C$ axis; closure evidence levels are named in words. Costs carry subscripts ($C_{\text{task}}$, $C_{\text{eff}}$, $C_{\max}$), and the task contract is $\mathcal{C}$.]
    ASequenceOfSteps: NotationSymbol  # A sequence of steps ($\mathcal{T}$) [Calligraphic $\mathcal{T}$ with a subscript is a set of trajectories. Time is $T$.]
    AWorkerOrAgentCount: NotationSymbol  # A worker or agent count ($M$)
    AcceleratorCount: NotationSymbol  # Accelerator Count ($N_{\text{GPU}}$) [Integer]
    AcceleratorsPerNode: NotationSymbol  # Accelerators per Node, Node Count ($G_{\text{node}}$)
    AcceleratorsPerNodeNodes: NotationSymbol  # Accelerators per Node, Node Count ($N_{\text{nodes}}$)
    AcceptanceFraction: NotationSymbol  # Acceptance fraction ($\alpha$)
    AcceptanceProbability: NotationSymbol  # Acceptance probability ($p_{\text{acc}}$)
    AcceptedInfluences: NotationSymbol  # Accepted influences ($\text{Inf}(o)$)
    AcceptedTasks: NotationSymbol  # Accepted tasks ($N_{\text{acc}}$)
    AccuracyAtTime: NotationSymbol  # Accuracy at Time $t$ ($\text{Accuracy}(t)$) [Scalar]
    ActVariationalObjective: NotationSymbol  # ACT Variational Objective ($\mathcal{L}_{\text{ACT}}(\theta, \phi)$)
    ActionChunk: NotationSymbol  # Action chunk $\mathbf{A}_t$, $\mathbf{A}_{t:t+H}$, and its denoising iterate $\mathbf{A}_t^k$ ($\mathbf{A}$)
    ActionChunkHorizon: NotationSymbol  # Action Chunk Horizon ($H$) [Time steps]
    ActionChunkTrajectory: NotationSymbol  # Action Chunk Trajectory ($\mathbf{A}_{t:t+H}$) [Tensor]
    ActionDimension: NotationSymbol  # Action Dimension ($d_a$)
    ActionExecutionRate: NotationSymbol  # Action Execution Rate ($f_{\text{exec}}$) [$\text{Hz}$]
    ActionProposal: NotationSymbol  # Action proposal ($a_t$) [Text or tool call]
    ActionValueOrCosttogo: NotationSymbol  # Action value or cost-to-go ($Q_t^{\pi}(s_t, a_t)$)
    ActivationMemory: NotationSymbol  # Activation Memory ($M_{\text{activations}}$) [bytes]
    ActuationSelectionMatrix: NotationSymbol  # Actuation Selection Matrix ($\mathbf{S}$) [Matrix]
    ActuatorDutyCycle: NotationSymbol  # Actuator Duty Cycle ($D_{\text{cycle}}$) [$[0, 1]$]
    ActuatorResponseTime: NotationSymbol  # Actuator Response Time ($T_{\text{act}}$) [$\text{ms}$]
    AdjointRepresentation: NotationSymbol  # Adjoint Representation ($\mathbf{Ad}_T$) [Matrix]
    AdminLeaseTime: NotationSymbol  # Administrative Lease Time ($T_{\text{admin}}$) [seconds]
    Advantage: NotationSymbol  # Advantage; action sets ($A$) [Authority is written only with a level, $A_0$ to $A_3$. Advantage is $\hat{A}$; action sets are $\mathcal{A}_{\text{prop}}$ and $\mathcal{A}_{\text{perm}}$.]
    AdversarialPerturbationBudget: NotationSymbol  # Adversarial Perturbation Budget ($\epsilon_{\text{adv}}$)
    AgenticSystem: NotationSymbol  # Agentic system ($\mathcal{S}$) [Tuple]
    AllowableCollisionEnergy: NotationSymbol  # Allowable Collision Energy ($E_{k, \text{allow}}$) [$\text{Joules (J)}$]
    AmbientTemperature: NotationSymbol  # Ambient Temperature ($T_{\text{amb}}$) [degC]
    AngularVelocityAcceleration: NotationSymbol  # Angular Velocity & Acceleration ($\boldsymbol{\omega}$)
    ArithmeticIntensity: NotationSymbol  # Arithmetic Intensity ($I$) [FLOP/byte]
    ArrivalRate: NotationSymbol  # Arrival Rate ($\lambda_{\text{arr}}$) [requests/s]
    ArticulatedInertiaMatrix: NotationSymbol  # Articulated inertia matrix $\mathbf{M}(\mathbf{q}) \in \mathbb{R}^{n \times n}$ ($\mathbf{M}$) [Mass ($m, m_{\text{eff}}$), memory footprint ($M_{\text{weights}}, M_{\text{KV}}$), or fleet size ($N_{\text{fleet}}$)]
    AttackStepsPgd: NotationSymbol  # Attack Steps ($K_{\text{PGD}}$)
    AttainableComputeRate: NotationSymbol  # Attainable Compute Rate ($R_{\text{attain}}$) [FLOP/s]
    AttemptCost: NotationSymbol  # Attempt cost ($C_{\text{attempt}}$) [Dollars]
    AttentionHeadDimension: NotationSymbol  # Attention Head Dimension ($d_{\text{head}}$) [Integer. Per-head hidden dimension in attention layers.]
    AttentionHeads: NotationSymbol  # Attention heads; hidden size ($H$) [$H$ is the horizon in turns. Heads use $N_{\text{heads}}$ and $H_{\text{KV}}$; hidden size uses $d$.]
    AvailableObservations: NotationSymbol  # Available observations ($\mathcal{O}_{\text{avail}}$) [Specification]
    BackemfVoltageBemf: NotationSymbol  # Back-EMF Voltage ($e_{\text{bemf}}$) [$\text{V}$]
    Bandwidth: NotationSymbol  # Bandwidth ($\text{BW}$) [bytes/s]
    BarrierConstraintMatrix: NotationSymbol  # Barrier Constraint Rows ($\mathbf{A}_{\text{cbf}}$)
    BarrierConstraintVector: NotationSymbol  # Barrier Constraint Rows ($\mathbf{b}_{\text{cbf}}$)
    BarrierRelativeDegree: NotationSymbol  # Barrier Relative Degree ($r$) [Integer]
    BasePromptLength: NotationSymbol  # Base prompt length ($S_{\text{base}}$)
    BatchServiceTime: NotationSymbol  # Batch Service Time ($T_{\text{svc}}(B)$) [seconds]
    BatchSize: NotationSymbol  # Batch Size ($B$) [Bandwidth ($\text{BW}$), Body frame ($\mathcal{F}_B$), bus burst size ($B_k$), or input matrix $\mathbf{B}$]
    BeliefInvalidationHorizon: NotationSymbol  # Belief Invalidation Horizon ($T_{\text{inv}}$)
    BlendingTransitionWindow: NotationSymbol  # Blending Transition Window ($\tau_{\text{blend}}$) [$\text{s}$]
    BodyReferenceFrame: NotationSymbol  # Body Reference Frame ($\mathcal{F}_B$)
    BumplessTransferBlend: NotationSymbol  # Bumpless Transfer Blend ($\alpha(\hat{t})$) [$[0, 1]$]
    BumplessTransferBlendingFunctionAlphaInZero: NotationSymbol  # Bumpless transfer blending function $\alpha(\hat{t}) \in [0, 1]$ ($\alpha$)
    CallLatency: NotationSymbol  # Call latency ($T_{\text{call}}$) [Seconds]
    CameraIntrinsicCalibrationMatrix: NotationSymbol  # Camera intrinsic calibration matrix $\mathbf{K} \in \mathbb{R}^{3 \times 3}$ ($\mathbf{K}$) [Contact stiffness ($k_{\text{contact}}$), control gains ($K_p, K_d$), motor constants ($K_t, K_e$), ensembling window $K$, or diffusion step $k$]
    CameraframeThreedCoordinate: NotationSymbol  # Camera-Frame 3D Coordinate ($\mathbf{p}_c$)
    CapabilityGrant: NotationSymbol  # Capability grant ($\Gamma$)
    CartesianThreedPosition: NotationSymbol  # Cartesian 3D Position ($\mathbf{p}$)
    CenterOfMass: NotationSymbol  # Center of Mass (CoM) ($\mathbf{c}$) [$\text{m}$]
    CentroidalAngularMomentum: NotationSymbol  # Centroidal Angular Momentum ($\mathbf{h}_{\text{ang}} \in \mathbb{R}^3$) [$\text{kg}\cdot\text{m}^2/\text{s}$]
    CentroidalLinearMomentum: NotationSymbol  # Centroidal Linear Momentum ($\mathbf{h}_{\text{lin}} \in \mathbb{R}^3$) [$\text{kg}\cdot\text{m/s}$]
    CertifiedControlAction: NotationSymbol  # Certified Control Action ($\mathbf{u}^*$) [$\text{N}\cdot\text{m}$ / $\text{rad}$]
    CheckpointIntervalCkpt: NotationSymbol  # Checkpoint Interval ($T_{\text{ckpt}}$)
    CheckpointWriteTime: NotationSymbol  # Checkpoint Write Time ($T_{\text{write}}$) [seconds]
    ChunkLease: NotationSymbol  # Chunk Lease ($T_{\text{lease}}$)
    ChunkLeaseTime: NotationSymbol  # Chunk Lease Time ($T_{\text{lease}}$) [seconds]
    ChunkRenewalPeriod: NotationSymbol  # Chunk Renewal Period ($T_{\text{chunk}}$)
    ClassRateFunction: NotationSymbol  # Class-$\mathcal{K}$ Rate Function ($\kappa(\cdot)$)
    ClearDistance: NotationSymbol  # Clear Distance ($D_{\text{clear}}$) [$\text{m}$]
    ClockOffset: NotationSymbol  # Clock Offset ($\theta_{\text{offset}}$)
    ClockSynchronizationSkew: NotationSymbol  # Clock Synchronization Skew ($\delta_{\text{skew}}$) [$\text{ppm}, \mu\text{s/s}$]
    ClosureOperator: NotationSymbol  # Closure operator ($\mathcal{I}$) [Function]
    CodecDecodeTime: NotationSymbol  # Codec Decode Time ($T_{\text{decode}}$) [seconds]
    CodecEncodeTime: NotationSymbol  # Codec Encode Time ($T_{\text{encode}}$) [seconds]
    CoefficientOfRestitution: NotationSymbol  # Coefficient of Restitution ($e$)
    CognitivePolicyAction: NotationSymbol  # Cognitive Policy Action ($\mathbf{a}_t$) [Normalized]
    CohensKappa: NotationSymbol  # Cohen's kappa; a coherency coefficient ($\kappa$)
    CoherencyContentionCoefficient: NotationSymbol  # Contention and coherency coefficients ($\kappa_{\text{coh}}$)
    CollectivePayloadSize: NotationSymbol  # Collective Payload Size ($M$) [Bytes]
    CommandTransportTimeBus: NotationSymbol  # Command Transport Time ($T_{\text{bus}}$) [$\text{ms}$]
    CommitmentInstant: NotationSymbol  # Commitment Instant ($t_{\text{commit}}$)
    CommunicationIntensity: NotationSymbol  # Communication Intensity ($\text{CI}$)
    CommunicationTime: NotationSymbol  # Communication Time ($T_{\text{comm}}(N)$) [seconds]
    CommunicationcomputationRatio: NotationSymbol  # Communication-Computation Ratio ($\rho$) [Dimensionless]
    CompletionCriteria: NotationSymbol  # Completion criteria ($\mathcal{K}_{\text{comp}}$) [Set of checks]
    CompoundingCovariateDrift: NotationSymbol  # Compounding Covariate Drift ($\mathcal{O}(T^2 \epsilon)$) [Error bound]
    CompressedCommunicationTime: NotationSymbol  # Compressed Communication Time ($T_{\text{compressed}}$) [seconds]
    CompressionRatio: NotationSymbol  # Compression Ratio ($r_{\text{comp}}$) [Dimensionless]
    ConcurrentTrajectories: NotationSymbol  # Concurrent trajectories, admission rate ($N_{\text{traj}}$)
    ConcurrentTrajectoriesTwo: NotationSymbol  # Concurrent trajectories, admission rate ($\lambda_{\text{traj}}$)
    ConditionalDistribution: NotationSymbol  # Conditional Distribution ($p(y \mid x)$) [Conditional probability/density. Use this form for generic label relationships; reserve $P_0$ and $P_t$ for the training/current distributions of drift analysis.]
    Confidentiality: NotationSymbol  # Confidentiality ($\kappa(x)$)
    ContactForceVectorK: NotationSymbol  # Contact Force Vector ($\mathbf{f}_k$) [$\text{N}$]
    ContactPointJacobianK: NotationSymbol  # Contact Point Jacobian ($\mathbf{J}_k(\mathbf{q})$) [Mixed]
    ContactStiffness: NotationSymbol  # Contact Stiffness ($k_{\text{contact}}$)
    ContentionAndCoherencyCoefficients: NotationSymbol  # Contention and coherency coefficients ($\sigma_{\text{cont}}$)
    Context: NotationSymbol  # Context ($c_t$) [Tokens]
    ContextWindow: NotationSymbol  # Context window ($S_{\max}$) [Tokens]
    ContinuationCount: NotationSymbol  # Continuation count ($K$)
    ContinuousCurrent: NotationSymbol  # Continuous Motor Current ($I_{\text{cont}}$) [A]
    ContinuousPeakTorque: NotationSymbol  # Continuous / Peak Torque ($\tau_{\text{peak}}$) [$\text{N}\cdot\text{m}$]
    ControlAffineDriftField: NotationSymbol  # Control-Affine Drift and Input Fields ($\mathbf{f}(\mathbf{x})$)
    ControlAffineInputField: NotationSymbol  # Control-Affine Drift and Input Fields ($\mathbf{g}(\mathbf{x})$)
    ControlBarrierFunction: NotationSymbol  # Control Barrier Function (CBF) ($h(\mathbf{x})$) [Scalar]
    ControlSetpointJerk: NotationSymbol  # Control Setpoint Jerk ($\mathbf{j}(t)$) [$\text{m/s}^3, \text{N}\cdot\text{m/s}$]
    CoordinateReferenceFrames: NotationSymbol  # Coordinate Reference Frames ($\mathcal{F}_E$) [Reference]
    CoriolisCentrifugalMatrix: NotationSymbol  # Coriolis / Centrifugal Matrix ($\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})$)
    CoriolisMatrix: NotationSymbol  # Coriolis matrix $\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}) \in \mathbb{R}^{n \times n}$ ($\mathbf{C}$) [Safe set $\mathcal{C}$, state-space output matrix $\mathbf{C}$, capacitance ($C_{\text{th}}, C_{\text{bus}}$), CoM $\mathbf{c}$, or shield coverage $c_{\text{shield}}$]
    Correlation: NotationSymbol  # Correlation ($\rho$)
    CostCeiling: NotationSymbol  # Cost ceiling ($C_{\max}$) [Dollars]
    CostPerAcceptedTaskEff: NotationSymbol  # Cost per accepted task ($C_{\text{eff}}$) [Dollars/task]
    CoulombFrictionCoefficient: NotationSymbol  # Coulomb Friction Coefficient ($\mu$) [Dimensionless]
    CredibleBrakingDeceleration: NotationSymbol  # Credible Deceleration ($a_{\text{brake}}$) [$\text{m/s}^2$]
    CrossoverPoint: NotationSymbol  # Crossover Point ($n^*$) [bytes]
    CurrentDistribution: NotationSymbol  # Current Distribution ($P_t$) [Distribution]
    CvaeRecognitionEncoder: NotationSymbol  # CVAE Recognition Encoder ($q_\phi(\mathbf{z} \mid \mathbf{o})$)
    DataVolume: NotationSymbol  # Data Volume ($D_{\text{vol}}$) [bytes]
    DataparallelDegreeDp: NotationSymbol  # Data-Parallel Degree ($N_{\text{DP}}$)
    DatasetSize: NotationSymbol  # Dataset Size ($D$) [$\text{ms}$]
    DcBusBulkCapacitance: NotationSymbol  # DC Bus Bulk Capacitance ($C_{\text{bus}}$) [$\mu\text{F}$]
    DcBusDynamicRailDroop: NotationSymbol  # DC Bus Dynamic Rail Droop ($\Delta V_{\text{droop}}$) [$\text{V}$]
    Deadline: NotationSymbol  # Deadline ($T_{\text{deadline}}$)
    DecoderEvaluationCount: NotationSymbol  # Decoder Evaluation Count ($N_{\text{eval}}$)
    DetectionInterval: NotationSymbol  # Detection Interval ($t_{\text{detect}}$)
    DiagnosticCoverage: NotationSymbol  # Diagnostic Coverage ($C_{\text{diag}}$)
    DifferentialPrivacyBudget: NotationSymbol  # Differential Privacy Budget ($(\epsilon, \delta)$)
    DiffusionGaussianNoise: NotationSymbol  # Diffusion Gaussian Noise ($\boldsymbol{\epsilon}$)
    DiffusionMatchingLoss: NotationSymbol  # Diffusion Matching Loss ($\mathcal{L}_{\text{diff}}(\theta)$) [Scalar]
    DiffusionScheduleParameterK: NotationSymbol  # Diffusion Schedule Parameter ($\bar{\alpha}_k$) [$[0, 1]$]
    DirectionalCosineMatrix: NotationSymbol  # Directional Cosine Matrix ($\mathbf{R}$) [Matrix]
    DispatchtoadmissionIntervalDa: NotationSymbol  # Dispatch-to-Admission Interval ($\Delta_{\text{da}}$)
    DistillationTemperature: NotationSymbol  # Distillation Temperature ($T_{\text{distill}}$) [Dimensionless. Softmax temperature in knowledge distillation; the one use of the $T$ root that is not a time.]
    DistributedStepTime: NotationSymbol  # Distributed Step Time ($T_{\text{step}}(N)$) [seconds]
    Distribution: NotationSymbol  # Distribution ($p(x)$) [Probability mass/density for a random variable. Use lowercase $p(\cdot)$ for generic distributions to avoid collision with $P$ (parameter count).]
    DivergenceInstant: NotationSymbol  # Divergence Instant ($t_{\text{div}}$)
    DollarCostInfer: NotationSymbol  # Dollar Cost ($C_{\text{infer}}$)
    DollarCostTrain: NotationSymbol  # Dollar Cost ($C_{\text{train}}$)
    DuplicateeffectProbability: NotationSymbol  # Duplicate-effect probability ($p_{\text{dup}}$)
    EffectiveFleetThroughput: NotationSymbol  # Effective Fleet Throughput ($R_{\text{eff}}$) [FLOP/s]
    EffectiveImpactMass: NotationSymbol  # Effective Impact Mass ($m_{\text{eff}}$) [$\text{kg}$]
    EffectiveJointInertia: NotationSymbol  # Effective Joint Inertia ($J_{\text{eff}}$)
    EffectiveServiceRate: NotationSymbol  # Effective Service Rate ($\mu_{\text{eff}}(B)$) [requests/s]
    EfficiencyHw: NotationSymbol  # Efficiency ($\eta_{\text{hw}}$) [—]
    ElectricalTimeConstant: NotationSymbol  # Electrical Time Constant ($\tau_e$)
    ElementStorageSize: NotationSymbol  # Element Storage Size ($s_{\text{elem}}$) [bytes/element]
    EmittedTokenCountOut: NotationSymbol  # Emitted Token Count ($S_{\text{out}}$)
    EndToEndRequestLatency: NotationSymbol  # End-to-End Request Latency ($L_{\text{lat, total}}$) [seconds]
    EnergyPerByteMoved: NotationSymbol  # Energy per Byte Moved ($E_{\text{move}}$) [joules/byte]
    EnergyPerOperationCompute: NotationSymbol  # Energy per Operation ($E_{\text{compute}}$) [joules/FLOP]
    EnforcerComputeTime: NotationSymbol  # Enforcer Compute Time ($T_{\text{comp}}$)
    EnsemblingWindowLength: NotationSymbol  # Ensembling Window Length ($K$) [Time steps]
    Environment: NotationSymbol  # Environment ($\mathcal{E}_{\text{env}}$) [Specification]
    EnvironmentState: NotationSymbol  # Environment state ($s_t$)
    EnvironmentStateK: NotationSymbol  # Environment state ($s_k$)
    EssentialMatrix: NotationSymbol  # Essential Matrix ($\mathbf{E}$)
    EstimatedAdvantageK: NotationSymbol  # Estimated advantage ($\hat{A}(h_k, a_k)$)
    EstimatedValueK: NotationSymbol  # Estimated value ($\hat{V}(h_k)$) [Scalar]
    EstimationErrorCovariance: NotationSymbol  # Estimation Error Covariance ($\mathbf{P}$) [Matrix]
    EventProbability: NotationSymbol  # Event Probability ($\Pr(E)$) [Probability of an event $E$. Use for event statements such as $\Pr(\text{batch}=0)$; use $p(x)$ and $p(y \mid x)$ for distributions.]
    EvidenceHorizon: NotationSymbol  # Evidence Horizon ($\tau_{\text{ev}}$)
    EvidenceOrEventSets: NotationSymbol  # Evidence or event sets ($\mathcal{E}$)
    Expectation: NotationSymbol  # Expectation ($\mathbb{E}[\cdot]$)
    ExpectedCheckingTime: NotationSymbol  # Expected checking time ($T_{\text{check}}$)
    ExpectedIrreversibleLossA: NotationSymbol  # Expected irreversible loss ($L_A$)
    ExposureDuration: NotationSymbol  # Exposure Duration ($t_{\text{exp}}$)
    ExternalWait: NotationSymbol  # External wait ($T_{\text{wait}}$) [Seconds]
    FallbackHold: NotationSymbol  # Fallback Hold ($T_{\text{hold}}$)
    FittedSensitivity: NotationSymbol  # Fitted Sensitivity ($\lambda$) [Scalar]
    FleetEnergyProductivity: NotationSymbol  # Fleet Energy Productivity ($\rho_{\text{energy}}$)
    ForwardinvariantSafeSet: NotationSymbol  # Forward-Invariant Safe Set ($\mathcal{C}$) [Set]
    FreespaceDepth: NotationSymbol  # Free-Space Depth ($d_{\text{free}}$)
    FundamentalMatrix: NotationSymbol  # Fundamental Matrix ($\mathbf{F}$)
    GearReductionRatio: NotationSymbol  # Gear Reduction Ratio ($N_{\text{gear}}$) [Dimensionless]
    GearboxTransmissionEfficiency: NotationSymbol  # Transmission Efficiency ($\eta_{\text{gear}}$) [Dimensionless]
    GemmDimension: NotationSymbol  # GEMM dimension; memory root ($M_{\text{total}}$) ($M$)
    GeneralizedAccelerations: NotationSymbol  # Generalized Accelerations ($\ddot{\mathbf{q}}$) [$\text{rad/s}^2$ or $\text{m/s}^2$]
    GeneralizedCoordinates: NotationSymbol  # Generalized Coordinates ($\mathbf{q}$) [$\text{rad}$ or $\text{m}$]
    GeneralizedInertiaMatrix: NotationSymbol  # Generalized Inertia Matrix ($\mathbf{M}(\mathbf{q})$) [$\text{kg}\cdot\text{m}^2$]
    GeneralizedVelocities: NotationSymbol  # Generalized Velocities ($\dot{\mathbf{q}}$) [$\text{rad/s}$ or $\text{m/s}$]
    Goal: NotationSymbol  # Goal ($g$)
    GoodputRatio: NotationSymbol  # Goodput Ratio ($\eta_{\text{goodput}}$) [Dimensionless]
    Gradient: NotationSymbol  # Gradient $\mathbf{g}$ ($g$)
    GradientMemory: NotationSymbol  # Gradient Memory ($M_{\text{gradients}}$) [bytes]
    GradientStaleness: NotationSymbol  # Gradient Staleness ($\tau_{\text{stale}}$)
    GravitationalTorqueVector: NotationSymbol  # Gravitational Torque Vector ($\mathbf{g}(\mathbf{q})$) [$\text{N}\cdot\text{m}$]
    GroupSize: NotationSymbol  # Group size ($G$)
    HandoverTimeout: NotationSymbol  # Handover Timeout ($T_{\text{timeout}}$)
    HardwareAndCommoncauseRateOther: NotationSymbol  # Hardware and Common-Cause Rate ($\lambda_{\text{other}}$)
    HardwareWatchdogTimeout: NotationSymbol  # Hardware Watchdog Timeout ($t_{\text{watchdog}}$) [$\text{ms}$]
    HazardOnset: NotationSymbol  # Hazard Onset ($t_{\text{onset}}$)
    HazardousProposalRate: NotationSymbol  # Hazardous Proposal Rate ($\lambda_{\text{proposal}, h}$)
    HertzianContactStiffness: NotationSymbol  # Contact Stiffness ($k_H$) [$\text{N/m}$]
    HiddenDimension: NotationSymbol  # Hidden Dimension ($d$) [Permit]
    HiddenDimensionFour: NotationSymbol  # Hidden dimension; time; distribution $p(\cdot)$ ($p$) [Integer]
    HoldingTorque: NotationSymbol  # Continuous / Peak Torque ($\tau_{\text{hold}}$)
    HomogeneousTransform: NotationSymbol  # Homogeneous Transform ($\mathbf{T}_{AB}$) [Matrix]
    Horizon: NotationSymbol  # Horizon ($H$) [Time steps]
    HumanWalkingSpeedBound: NotationSymbol  # Human Walking-Speed Bound ($v_h$) [$\text{m/s}$]
    IdempotencyKey: NotationSymbol  # Idempotency key ($k_{\text{idem}}$) [String]
    ImagePlanePixelCoordinates: NotationSymbol  # Image Plane Pixel Coordinates ($\mathbf{u}_{\text{px}}$)
    InitialAccuracy: NotationSymbol  # Initial Accuracy ($\text{Accuracy}_0$) [Scalar]
    InitialAndCeilingError: NotationSymbol  # Initial and Ceiling Error ($e_{\max}$)
    InitialTrackingError: NotationSymbol  # Initial and Ceiling Error ($e_0$)
    IntegrityProvenance: NotationSymbol  # Integrity provenance ($\iota(x)$)
    IntentLease: NotationSymbol  # Intent Lease ($\mathcal{I}_{\text{intent}}$)
    IntentleaseExpiryIntentexp: NotationSymbol  # Intent-Lease Expiry ($t_{\text{intent\_exp}}$)
    InterconnectPortConflicts: NotationSymbol  # Interconnect Port Conflicts ($N_{\text{conflict}}$) [Count]
    InterconnectTransferTimeXfer: NotationSymbol  # Interconnect Transfer Time ($t_{\text{xfer}}$) [$\text{ms}$]
    IronLawComputeTime: NotationSymbol  # Iron-Law Term Times ($T_{\text{compute}}$) [seconds]
    IronLawMemoryTime: NotationSymbol  # Iron-Law Term Times ($T_{\text{memory}}$)
    IrreducibleLoss: NotationSymbol  # Irreducible Loss ($C_{\mathcal{L}}$)
    JerkLimit: NotationSymbol  # Jerk Limit ($j_{\max}$) [$\text{m/s}^3$]
    JouleanTemperatureRise: NotationSymbol  # Joulean Temperature Rise ($\Delta T$) [$\text{Kelvin (K)}$]
    KalmanGain: NotationSymbol  # Kalman Gain ($\mathbf{G}_k$)
    KineticInsetBarrierFunction: NotationSymbol  # Kinetic Inset Barrier Function ($h_{\text{kinetic}}(\mathbf{x})$) [Scalar]
    KvBlockSizeBlk: NotationSymbol  # KV block size ($S_{\text{blk}}$)
    KvBytesPerToken: NotationSymbol  # KV bytes per token ($m_{\text{token}}$) [Bytes/token]
    KvPayload: NotationSymbol  # KV payload ($M_{\text{KV}}$)
    KvcacheBytesPerToken: NotationSymbol  # KV-Cache Bytes per Token ($m_{\text{token}}$)
    KvcacheMemory: NotationSymbol  # KV-Cache Memory ($M_{\text{KV}}$) [bytes]
    Latency: NotationSymbol  # Latency ($L_{\text{lat}}$) [seconds]
    LatentStyleStateEmbedding: NotationSymbol  # Latent Style / State Embedding ($\mathbf{z}$) [Dimensionless]
    LayerIndex: NotationSymbol  # Layer Index ($\ell$) [Integer. Index for a layer. Use instead of bare $L$ when indexing layers, since $L$ collides with loss and latency. A numeric subscript marks a norm instead ($\ell_1$, $\ell_2$).]
    LayerInputWidth: NotationSymbol  # Layer Input Width ($d_{\text{in}}$)
    LayerOutputWidth: NotationSymbol  # Layer Output Width ($d_{\text{out}}$) [Integer. Input and output feature counts of a single layer.]
    LearnedModelParameters: NotationSymbol  # Learned model (policy) parameters, as in $\pi_\theta$ ($\theta$)
    LearningRate: NotationSymbol  # Learning Rate ($\eta$)
    LieDerivativeAlongControl: NotationSymbol  # Lie Derivative along Control ($L_{\mathbf{g}} h(\mathbf{x})$) [Vector]
    LieDerivativeAlongDrift: NotationSymbol  # Lie Derivative along Drift ($L_{\mathbf{f}} h(\mathbf{x})$) [$\text{s}^{-1}$]
    LinearBarrierDecayRate: NotationSymbol  # Linear Barrier Decay Rate ($\gamma_{\text{cbf}}$) [$\text{s}^{-1}$]
    LinearStatespaceMatrices: NotationSymbol  # Linear State-Space Matrices ($\mathbf{B}$) [Matrix]
    LinearStatespaceMatricesSys: NotationSymbol  # Linear State-Space Matrices ($\mathbf{A}_{\text{sys}}$)
    LinkBandwidth: NotationSymbol  # Link Bandwidth ($\beta$) [bytes/s]
    LittlesLawOccupancy: NotationSymbol  # Little's law occupancy; loss; latency ($L$)
    LoadInertia: NotationSymbol  # Load Inertia ($J_{\text{load}}$) [kg*m^2]
    LocalizationBound: NotationSymbol  # Localization Bound ($\delta_{\text{loc}}$) [$\text{m}$]
    LoggpLongmessageTime: NotationSymbol  # LogGP Long-Message Time ($T_{\text{long}}(n)$) [seconds]
    LogploggpNetworkLatency: NotationSymbol  # LogP/LogGP Network Latency ($L_{\text{LogGP}}$)
    LongmessageGapPerByte: NotationSymbol  # Long-Message Gap per Byte ($G$) [$\mathcal{G}$ is trajectory goodput. The goal is $g$ in the loop and $G$ in the task contract.]
    Loss: NotationSymbol  # Loss ($L$) [Latency]
    LossFunction: NotationSymbol  # Loss Function ($\mathcal{L}$) [Scalar. The objective function minimized during training.]
    LossMaskT: NotationSymbol  # Loss mask ($m_t$) [Binary $\{0, 1\}$]
    LossscaleFactor: NotationSymbol  # Loss-Scale Factor ($s_{\text{loss}}$) [Dimensionless]
    LqrControlWeightMatrix: NotationSymbol  # LQR Control Weighting Matrix ($\mathbf{R}_u$)
    LqrGainAndWeights: NotationSymbol  # LQR Gain and Weights ($\mathbf{K}_{\text{lqr}}$)
    LqrStateWeightMatrix: NotationSymbol  # LQR State Weighting Matrix ($\mathbf{Q}_x$)
    ManipulatorJacobian: NotationSymbol  # Manipulator Jacobian ($\mathbf{J}(\mathbf{q})$) [Mixed]
    MaskedFinetuningLossSft: NotationSymbol  # Masked fine-tuning loss ($\mathcal{L}_{\text{SFT}}$) [Cross-entropy]
    MeanTimeBetweenFailures: NotationSymbol  # Mean Time Between Failures ($\text{MTBF}$) [hours]
    MeanTimeToFailure: NotationSymbol  # Mean Time To Failure ($\text{MTTF}$) [hours]
    MeanTimeToRepair: NotationSymbol  # Mean Time To Repair ($\text{MTTR}$) [hours]
    MeasurementMatrix: NotationSymbol  # Transition and Measurement Maps ($\mathbf{H}$)
    MeasurementNoiseVector: NotationSymbol  # Process and Measurement Noise ($\mathbf{v}_{\text{meas}}(t)$) [Vector]
    MechanicalPlantState: NotationSymbol  # Mechanical Plant State ($\mathbf{x}$) [$\text{rad, rad/s}$]
    MemoryBusQueueDelay: NotationSymbol  # Memory Bus Queue Delay ($t_{\text{queue}}$) [$\text{ms}$]
    MemoryHeldDuringAWait: NotationSymbol  # Memory held during a wait ($O_{\text{mem}}$) [Byte-seconds]
    MessageInjectionGap: NotationSymbol  # Message Injection Gap ($g$) [$\text{failures/hour}$]
    MicrobatchCount: NotationSymbol  # Microbatch Count ($m$) [$\text{failures/hour}$]
    MlProductivityGoodputFactorsProgram: NotationSymbol  # ML Productivity Goodput Factors ($\eta_{\text{program}}$)
    MlProductivityGoodputFactorsRuntime: NotationSymbol  # ML Productivity Goodput Factors ($\eta_{\text{runtime}}$)
    MlProductivityGoodputFactorsScheduling: NotationSymbol  # ML Productivity Goodput Factors ($\eta_{\text{scheduling}}$)
    ModelFlopsUtilization: NotationSymbol  # Model FLOPs Utilization ($\text{MFU}$) [Dimensionless]
    ModelParameterCount: NotationSymbol  # Model parameter count $P$ (series core) ($P$)
    ModelParameters: NotationSymbol  # Model Parameters ($\theta$) [$\text{K/W}$]
    ModelTime: NotationSymbol  # Model time ($T_{\text{model}}$) [Seconds]
    MotorBackEmfConstant: NotationSymbol  # Motor Back-EMF Constant ($K_e$) [V/(rad/s)]
    MotorPhaseRmsCurrent: NotationSymbol  # Motor Phase RMS Current ($I_{\text{phase}}$) [$\text{Amperes (A)}$]
    MotorTorqueConstant: NotationSymbol  # Motor Torque Constant ($K_t$) [N*m/A]
    MultimodalFusedTokenStream: NotationSymbol  # Multimodal Fused Token Stream ($\mathbf{Z}$) [Tensor]
    MultimodalObservation: NotationSymbol  # Multimodal Observation ($\mathbf{o}_t$) [Mixed]
    NdeviceTime: NotationSymbol  # N-Device Time ($T_N$) [seconds]
    NetBrakingDeceleration: NotationSymbol  # Net Braking Deceleration ($a_{\text{net}}$)
    NetPhaseMargin: NotationSymbol  # Net Phase Margin ($\text{PM}_{\text{net}}$) [$\text{deg}$]
    NoisePredictionNetwork: NotationSymbol  # Noise Prediction Network ($\boldsymbol{\epsilon}_\theta(\mathbf{o}_t, \mathbf{A}_t^k, k)$)
    NoisePredictionNetworkTwo: NotationSymbol  # Noise Prediction Network ($\mathbf{A}^k$)
    NominalControlCommand: NotationSymbol  # Nominal Control Command ($\mathbf{u}_{\text{nom}}$) [$\text{N}\cdot\text{m}$ / $\text{rad}$]
    NonmaskableInterruptTimeNmi: NotationSymbol  # Non-Maskable Interrupt Time ($t_{\text{NMI}}$) [$\mu\text{s}$]
    NormalContactForce: NotationSymbol  # Normal Contact Force ($f_{k, \perp}$) [$\text{N}$]
    NumberOfAttentionHeads: NotationSymbol  # Number of Attention Heads ($N_{\text{heads}}$) [Integer. Number of attention heads in a multi-head attention layer.]
    NumberOfDevices: NotationSymbol  # Number of Devices ($N$) [Count]
    NumberOfKeyValueHeads: NotationSymbol  # Number of Key-Value Heads ($H_{\text{KV}}$) [Integer. Number of key-value heads in grouped-query or multi-query attention.]
    NumberOfLayers: NotationSymbol  # Number of Layers ($N_L$) [Integer. Total number of layers in a network.]
    ObservabilityMatrices: NotationSymbol  # Observability Matrices ($\mathbf{O}_{\text{obs}}$)
    ObservabilityMatricesNl: NotationSymbol  # Observability Matrices ($\mathbf{O}_{\text{NL}}$) [Matrix]
    Observation: NotationSymbol  # Observation ($o_t$) [Typed result]
    ObservationAge: NotationSymbol  # Observation Age ($t_{\text{age}}$) [$\text{ms}$]
    OpenloopCrossoverFrequency: NotationSymbol  # Open-Loop Crossover Frequency ($\omega_c$) [$\text{rad/s}$]
    OperationalOverheadFraction: NotationSymbol  # Operational Overhead Fraction ($f_{\text{overhead}}$) [Dimensionless]
    Operations: NotationSymbol  # Operations ($O$) [FLOPs]
    OptimalCheckpointInterval: NotationSymbol  # Optimal Checkpoint Interval ($T^\ast_{\text{ckpt}}$) [seconds]
    OptimalCheckpointIntervalCkpt: NotationSymbol  # Optimal Checkpoint Interval ($T^\ast_{\text{ckpt}}$) [seconds]
    OptimalInertiamatchingRatio: NotationSymbol  # Optimal Inertia-Matching Ratio ($N^*$) [Dimensionless]
    OptimizerstateMemory: NotationSymbol  # Optimizer-State Memory ($M_{\text{optimizer}}$) [bytes]
    OutputLength: NotationSymbol  # Output length ($K$) [Tokens]
    OutputLimit: NotationSymbol  # Output limit ($K_{\max}$) [Tokens]
    OverlappedTime: NotationSymbol  # Overlapped Time ($T_{\text{overlap}}$) [seconds]
    ParallelFraction: NotationSymbol  # Parallel fraction ($f_{\text{parallel}}$)
    ParameterStreamingTime: NotationSymbol  # Parameter Streaming Time ($t_{\text{stream}}$) [$\text{ms}$]
    ParameterizedNeuralPolicy: NotationSymbol  # Parameterized Neural Policy ($\pi_\theta(\mathbf{a} \mid \mathbf{o})$) [Distribution]
    Parameters: NotationSymbol  # Parameters ($P$) [Power ($P_{\text{elec}}, P_{\text{cont}}$), Riccati error covariance $\mathbf{P}(t)$, failure rate $p_{\text{fail}}$, or 3D point $\mathbf{p}$]
    PeakCurrent: NotationSymbol  # Peak Motor Current ($I_{\text{peak}}$) [A]
    PeakInternodePhaseJitter: NotationSymbol  # Peak Inter-Node Phase Jitter ($\Delta t_{\text{peak}}$) [$\mu\text{s}$]
    PeakRate: NotationSymbol  # Peak Rate ($R_{\text{peak}}$) [FLOP/s]
    PerceptionDepthCovariance: NotationSymbol  # Perception Depth Covariance ($\boldsymbol{\Sigma}_{\text{depth}}$) [$\text{m}^2$]
    PercomponentFailureRate: NotationSymbol  # Per-Component Failure Rate ($\lambda$) [failures/hour]
    PermessageLatency: NotationSymbol  # Per-Message Latency ($\alpha$) [CBF decay ($\gamma_{\text{cbf}}$), significance ($\alpha_{\text{sig}}$), angular acceleration ($\boldsymbol{\alpha}$), copper tempco ($\alpha_{\text{Cu}}$), or noise schedule ($\bar{\alpha}_k$)]
    PermissionloopPeriodTick: NotationSymbol  # Permission-Loop Period ($T_{\text{tick}}$) [$\text{ms}$]
    PermittedActions: NotationSymbol  # Permitted actions ($\mathcal{A}_{\text{perm}}$) [Set]
    PertokenDecodeTime: NotationSymbol  # Per-token decode time ($t_{\text{decode}}$) [Seconds]
    PertokenPricesCache: NotationSymbol  # Per-token prices ($c_{\text{cache}}$)
    PertokenPricesIn: NotationSymbol  # Per-token prices ($c_{\text{in}}$)
    PertokenPricesOut: NotationSymbol  # Per-token prices ($c_{\text{out}}$)
    PertrialProbabilitiesFail: NotationSymbol  # Per-Trial Probabilities ($p_{\text{fail}}$)
    PertrialProbabilitiesTarget: NotationSymbol  # Per-Trial Probabilities ($p_{\text{target}}$)
    PerturnAppend: NotationSymbol  # Per-turn append ($\Delta S_{\text{turn}}$)
    PhaseInductance: NotationSymbol  # Motor Phase Inductance ($L_{\text{phase}}$) [H]
    PhaseLagDegradation: NotationSymbol  # Phase Lag Degradation ($\Delta \phi$) [$\text{rad}$ / $\text{deg}$]
    PhaseResistance: NotationSymbol  # Motor Phase Resistance ($R_{\text{phase}}$) [Ohm]
    PhysicalWorldState: NotationSymbol  # Physical World State ($s_t$) [Mixed]
    PidDerivativeGain: NotationSymbol  # Derivative Gain ($K_d$)
    PidIntegralGain: NotationSymbol  # Integral Gain ($K_i$) [Mixed]
    PidProportionalGain: NotationSymbol  # Proportional Gain ($K_p$)
    PinholeCameraIntrinsics: NotationSymbol  # Pinhole Camera Intrinsics ($\mathbf{K}$) [Pixels]
    PipelineBubbleFraction: NotationSymbol  # Pipeline Bubble Fraction ($b_{\text{pipe}}$) [Dimensionless]
    PipelineparallelDegreePp: NotationSymbol  # Pipeline-Parallel Degree ($N_{\text{PP}}$)
    PointToPointMessageSize: NotationSymbol  # Point-to-Point Message Size ($n$) [$\text{m}$]
    PointToPointTransferTime: NotationSymbol  # Point-to-Point Transfer Time ($T(n)$)
    Policy: NotationSymbol  # Policy ($\pi_\theta$) [Distribution]
    PositionalTaskTolerance: NotationSymbol  # Positional Task Tolerance ($r_{\text{tol}}$)
    PowerRailInductance: NotationSymbol  # Power Rail Inductance / ESR ($L_{\text{bus}}$)
    PowerRailResistance: NotationSymbol  # Power Rail Inductance / ESR ($R_{\text{bus}}$) [$\mu\text{H}, \text{m}\Omega$]
    PrebrakeDelay: NotationSymbol  # Pre-Brake Delay ($\tau_{\text{delay}}$)
    PredictedOutput: NotationSymbol  # Predicted Output ($\hat{\mathbf{y}}$) [Vector. A model's output for one example, such as class probabilities; $\mathbf{y}$ is the label vector.]
    ProcessNoiseVector: NotationSymbol  # Process and Measurement Noise ($\mathbf{w}(t)$)
    ProfileAccelerationBound: NotationSymbol  # Profile Acceleration Bound ($a_{\max}$)
    PromptLength: NotationSymbol  # Prompt length; precision; a predicate; a count ($P$)
    PromptTokensSent: NotationSymbol  # Prompt tokens sent ($S_{\text{sent}}$)
    Proposed: NotationSymbol  # Proposed, permitted action ($a_{\text{prop}}$)
    ProposedPerm: NotationSymbol  # Proposed, permitted action ($a_{\text{perm}}$) [Tool call]
    ProposerInferenceTime: NotationSymbol  # Proposer Inference Time ($t_{\text{infer}}$) [$\text{ms}$]
    ProtectiveClearanceMargin: NotationSymbol  # Protective Clearance ($\delta_{\text{margin}}$) [$\text{m}$]
    PureTransportDelay: NotationSymbol  # Pure Transport Delay ($T_d$)
    ReasoningBudget: NotationSymbol  # Reasoning budget ($B_{\text{reason}}$) [Tokens]
    ReceiverOverheadRecv: NotationSymbol  # Receiver Overhead ($o_{\text{recv}}$) [seconds]
    ReducedImpactMass: NotationSymbol  # Reduced Impact Mass ($m_{\text{red}}$)
    ReflectedRotorInertia: NotationSymbol  # Reflected Rotor Inertia ($J_{\text{ref}}$) [$\text{kg}\cdot\text{m}^2$]
    RegularizationWeight: NotationSymbol  # Regularization weight ($\lambda$)
    ReliabilityFunctionSystem: NotationSymbol  # Reliability Function ($R_{\text{system}}(t)$) [Probability]
    RenewaljitterAllowance: NotationSymbol  # Renewal-Jitter Allowance ($T_{\text{jitter}}$)
    RepairCap: NotationSymbol  # Repair cap ($K_{\text{repair}}$) [Count]
    RequestAndReplylossProbabilities: NotationSymbol  # Request- and reply-loss probabilities ($q_{\text{req}}$)
    RequestAndReplylossProbabilitiesTwo: NotationSymbol  # Request- and reply-loss probabilities ($q_{\text{rep}}$)
    RequestConcurrency: NotationSymbol  # Request Concurrency ($Q_{\text{req}}$) [requests]
    RequestTimeInSystem: NotationSymbol  # Request Time in System ($T_{\text{lat}}$) [seconds]
    ResidentStopDistanceSuffix: NotationSymbol  # Resident Stop Distance ($d_{\text{suffix}}(v)$)
    ResponseThreshold: NotationSymbol  # Response Threshold ($\tau$) [Bare $\tau$ is a trajectory. Sampling temperature is named in words or written $\tau_{\text{temp}}$.]
    RigidtransformTranslation: NotationSymbol  # Rigid-Transform Translation ($\mathbf{t}$)
    RingAllReduceCompletionTime: NotationSymbol  # Collective Completion Time ($T_{\text{ring}}$)
    RingtreeCrossoverPayload: NotationSymbol  # Ring-Tree Crossover Payload ($M_{\text{crossover}}$)
    RollingShutterDistortion: NotationSymbol  # Rolling Shutter Distortion ($\Delta \mathbf{x}(y)$) [$\text{m}$]
    RollingShutterRowReadout: NotationSymbol  # Rolling Shutter Row Readout ($t_{\text{row}}$) [$\mu\text{s}$]
    RooflineRidgePoint: NotationSymbol  # Roofline Ridge Point ($I_{\text{ridge}}$) [FLOP/byte]
    RotorInertia: NotationSymbol  # Motor Rotor Inertia ($J_{\text{rotor}}$) [kg*m^2]
    RuntimeTime: NotationSymbol  # Runtime time ($T_{\text{runtime}}$) [Seconds]
    SafetyBoundaryManifold: NotationSymbol  # Safety Boundary Manifold ($\partial \mathcal{C}$) [Boundary]
    SampleStorageSize: NotationSymbol  # Sample Storage Size ($s_{\text{sample}}$) [bytes/sample]
    SamplingPeriod: NotationSymbol  # Sampling Period ($T_s$) [s]
    SamplingRate: NotationSymbol  # Sampling Frequency / Rate ($f_s$) [Hz]
    ScaleCoefficient: NotationSymbol  # Scale Coefficient ($A_{\mathcal{L}}$)
    ScaledResource: NotationSymbol  # Scaled Resource ($X$)
    ScalingEfficiency: NotationSymbol  # Scaling Efficiency ($\eta_{\text{scaling}}$) [Dimensionless]
    ScalingExponent: NotationSymbol  # Scaling Exponent ($\alpha_{\text{scale}}$)
    SenderOverhead: NotationSymbol  # Sender Overhead ($o_{\text{send}}$) [seconds]
    SensorIntegrationTimeSense: NotationSymbol  # Sensor Integration Time ($t_{\text{sense}}$) [$\text{ms}$]
    SensorLeverarmVector: NotationSymbol  # Sensor Lever-Arm Vector ($\mathbf{r}_{P/O}$) [$\text{m}$]
    SeqlockRetryProbability: NotationSymbol  # Seqlock Retry Probability ($p_{\text{retry}}$) [Ratio]
    SeqlockVulnerableInterval: NotationSymbol  # Seqlock Vulnerable Interval ($t_{\text{vulnerable}}$) [$\mu\text{s}$]
    SequenceLength: NotationSymbol  # Sequence Length ($S$)
    SequentialLoglikelihoodRatioM: NotationSymbol  # Sequential Log-Likelihood Ratio ($\Lambda_m$)
    ServingUtilization: NotationSymbol  # Serving Utilization ($\rho_{\text{serv}}$) [Dimensionless]
    ShieldInterceptionCoverage: NotationSymbol  # Shield Interception Coverage ($c_h$)
    SignificanceLevel: NotationSymbol  # Significance Level ($\alpha$)
    SingleDeviceComputeTime: NotationSymbol  # Single-Device Compute Time ($T_{\text{compute}}$) [seconds]
    SingledeviceTimeOne: NotationSymbol  # Single-Device Time ($T_1$) [seconds]
    SinglestepImitationError: NotationSymbol  # Single-Step Imitation Error ($\epsilon$) [$[0, 1]$]
    SmallConstant: NotationSymbol  # Small Constant ($\epsilon$)
    SpecialEuclideanGroup: NotationSymbol  # Special Euclidean Group ($SE(3)$) [Lie Group]
    SpecialOrthogonalGroup: NotationSymbol  # Special Orthogonal Group ($SO(3)$) [Lie Group]
    SpeedAtDetection: NotationSymbol  # Speed at Detection ($v$) [$\text{m/s}$]
    SpeedBound: NotationSymbol  # Speed Bound ($v_{\max}$)
    SprtLowerBoundary: NotationSymbol  # SPRT Lower Decision Boundary ($\Lambda_{\text{lo}}$)
    SprtUpperBoundary: NotationSymbol  # SPRT Upper Decision Boundary ($\Lambda_{\text{up}}$)
    StalesupportRate: NotationSymbol  # Stale-support rate ($\epsilon_{\text{stale}}$)
    StateAtTurn: NotationSymbol  # State at turn $t$ ($s_t$) [Record]
    StateDistribution: NotationSymbol  # State distribution ($d_\pi$)
    StateTransitionMatrix: NotationSymbol  # Transition and Measurement Maps ($\mathbf{F}$)
    StatisticalDivergence: NotationSymbol  # Statistical Divergence ($\mathcal{D}(P_t \| P_0)$) [Scalar $\ge 0$]
    StatisticalSignificanceLevel: NotationSymbol  # Statistical Significance Level ($\alpha_{\text{sig}}$) [Dimensionless]
    StatorThermalCapacitance: NotationSymbol  # Stator Thermal Capacitance ($C_{\text{th}}$) [$\text{J/K}$]
    StoppingDistance: NotationSymbol  # Stopping Distance ($d_{\text{stop}}$) [$\text{m}$]
    SustainedMemoryBandwidth: NotationSymbol  # Sustained Memory Bandwidth ($\text{BW}_{\text{mem}}$) [bytes/s]
    SynchronizationBlackout: NotationSymbol  # Synchronization Blackout ($T_{\text{blackout}}$)
    SynchronizationTime: NotationSymbol  # Synchronization Time ($T_{\text{sync}}(N)$) [seconds]
    SystemHazardRate: NotationSymbol  # System Hazard Rate ($\lambda_{\text{system}, h}$)
    TakeoverResponseTime: NotationSymbol  # Takeover Response Time ($T_{\Sigma, 1}$)
    TangentialContactShear: NotationSymbol  # Tangential Contact Shear ($\mathbf{f}_{k, \parallel}$) [$\text{N}$]
    TargetHazardRate: NotationSymbol  # Target Hazard Rate ($\lambda_{\text{target}}$)
    TaskContract: NotationSymbol  # Task contract ($\mathcal{C}$) [Tuple]
    TaskCost: NotationSymbol  # Task cost ($C_{\text{task}}$)
    TaskDeadline: NotationSymbol  # Deadline ($T_{\text{deadline}}$) [Seconds]
    TasksPerComparisonArm: NotationSymbol  # Tasks per comparison arm ($N_{\text{arm}}$)
    TelemetryFreshnessBound: NotationSymbol  # Telemetry Freshness Bound ($\Delta t_{\text{fresh}}$)
    Temperature: NotationSymbol  # Temperature; a threshold; an interval ($\tau$)
    TemperatureOrThreshold: NotationSymbol  # Temperature or threshold ($\tau$)
    TemperaturedependentResistanceWind: NotationSymbol  # Temperature-Dependent Resistance ($R(T_{\text{wind}})$) [$\Omega$]
    TemporalEnsemblingWeight: NotationSymbol  # Temporal Ensembling Weight ($w_i$) [Scalar]
    TensorparallelDegreeTp: NotationSymbol  # Tensor-Parallel Degree ($N_{\text{TP}}$)
    TestErrorRates: NotationSymbol  # Test error rates; bandwidth; regularization weights ($\alpha$)
    TheGoal: NotationSymbol  # The goal; a graph ($\mathcal{G}$)
    TheStateExposure: NotationSymbol  # The State exposure ($S$) [Bare $S$ is sequence length. State is written only with a level, $S_0$ to $S_3$.]
    ThermalResistance: NotationSymbol  # Thermal Resistance ($R_{\text{th}}$)
    ThermalTimeConstant: NotationSymbol  # Thermal Time Constant ($\tau_{\text{th}}$) [$\text{s}$]
    Time: NotationSymbol  # Time ($T$) [s]
    TimingJitter: NotationSymbol  # Timing Jitter ($\sigma_{\text{jitter}}$)
    TokenIndex: NotationSymbol  # Token index ($t$)
    ToolTime: NotationSymbol  # Tool time ($T_{\text{tool}}$) [Seconds]
    Top: NotationSymbol  # Top-$K$, key matrix $\mathbf{K}$ ($K$)
    TorqueproducingCurrent: NotationSymbol  # Torque-Producing Current ($i_q$)
    TotalEnergy: NotationSymbol  # Total Energy ($E_{\text{total}}$) [joules]
    TotalMemoryFootprint: NotationSymbol  # Total Memory Footprint ($M_{\text{total}}$) [bytes]
    TraceDecay: NotationSymbol  # Trace decay ($\lambda$)
    TrackingerrorBound: NotationSymbol  # Tracking-Error Bound ($\epsilon_{\text{track}}$) [$\text{m}$]
    TrainingCompute: NotationSymbol  # Training Compute ($O_{\text{train}}$)
    TrainingDistribution: NotationSymbol  # Training Distribution ($P_0$) [Distribution]
    Trajectory: NotationSymbol  # Trajectory ($\tau$) [Tuple]
    TrajectoryBudget: NotationSymbol  # Trajectory budget ($\mathbf{B}$) [Vector]
    TrajectoryDuration: NotationSymbol  # Trajectory duration ($T_{\text{traj}}$) [Seconds]
    TrajectoryGoodput: NotationSymbol  # Trajectory goodput ($\mathcal{G}$) [Fraction $[0, 1]$]
    TrajectoryReward: NotationSymbol  # Trajectory reward ($R(\tau)$) [Scalar]
    TrajectorySetsAll: NotationSymbol  # Trajectory sets ($\mathcal{T}_{\text{all}}$)
    TrajectorySetsSuccess: NotationSymbol  # Trajectory sets ($\mathcal{T}_{\text{success}}$) [Set]
    TransferTimeForPayload: NotationSymbol  # Transfer Time for Payload $x$ ($T_{\text{transfer}}(x)$) [seconds]
    TransformerKvcacheFootprint: NotationSymbol  # Transformer KV-Cache Footprint ($M_{\text{KV}}$) [bytes]
    TreeAllReduceCompletionTime: NotationSymbol  # Collective Completion Time ($T_{\text{tree}}$)
    TrichotomyMonitorState: NotationSymbol  # Trichotomy Monitor State ($\mathcal{T}_{\text{trichotomy}}$) [Enum]
    TruncatedSignedDistance: NotationSymbol  # Truncated Signed Distance ($\text{tsdf}(\mathbf{p}_{\text{vox}})$) [$[-1, +1]$]
    TsdfAccumulatedDistance: NotationSymbol  # TSDF Accumulated Volume ($D_t(\mathbf{p}_{\text{vox}})$)
    TsdfAccumulatedWeight: NotationSymbol  # TSDF Accumulated Volume ($W_t(\mathbf{p}_{\text{vox}})$) [Mixed]
    TsdfTruncationBand: NotationSymbol  # TSDF Truncation Band ($\mu_{\text{tsdf}}$) [$\text{m}$]
    TurnIndex: NotationSymbol  # Turn index ($t$) [Integer]
    TypeIAndTypeIiErrorRatesSig: NotationSymbol  # Type I and Type II Error Rates ($\alpha_{\text{sig}}$)
    TypeIAndTypeIiErrorRatesStat: NotationSymbol  # Type I and Type II Error Rates ($\beta_{\text{stat}}$)
    TypeIiErrorAllowance: NotationSymbol  # Type II Error Allowance ($\beta$)
    UnbrakedImpactForce: NotationSymbol  # Unbraked Impact Force ($F_{\text{unbraked}}$) [$\text{N}$]
    UnitQuaternion: NotationSymbol  # Unit Quaternion ($\mathbf{q}_{\text{quat}}$) [Dimensionless]
    UsefulComputetimeFraction: NotationSymbol  # Useful Compute-Time Fraction ($f_{\text{compute}}$) [Dimensionless]
    UsefulModelFlops: NotationSymbol  # Useful Model FLOPs ($O_{\text{useful}}$)
    Utilization: NotationSymbol  # Utilization; a recoverable fraction ($\rho$)
    ValidityIntervalOfAMemoryEntryBirth: NotationSymbol  # Validity interval of a memory entry ($g_{\text{birth}}(c)$)
    ValidityIntervalOfAMemoryEntryDeath: NotationSymbol  # Validity interval of a memory entry ($g_{\text{death}}(c)$)
    ValidityIntervalOfAMemoryEntryQ: NotationSymbol  # Validity interval of a memory entry ($g_q$)
    Var: NotationSymbol  # (none) ($N$)
    Velocity: NotationSymbol  # Velocity ($\mathbf{v}$, end-effector velocity $\mathbf{v}_e$) ($\mathbf{v}$)
    VerificationEvidence: NotationSymbol  # Verification evidence ($v_t$) [Pass/fail or score]
    VerifierFalseacceptance: NotationSymbol  # Verifier false-acceptance, false-rejection rates ($\epsilon_{\text{FA}}$)
    VerifierFalseacceptanceFr: NotationSymbol  # Verifier false-acceptance, false-rejection rates ($\epsilon_{\text{FR}}$)
    VirtualDampingMatrix: NotationSymbol  # Virtual Damping and Stiffness ($\mathbf{D}_{\text{virt}}$)
    VirtualPipelineStagesPerDevice: NotationSymbol  # Virtual Pipeline Stages per Device ($V$) [Integer]
    VirtualStiffnessMatrix: NotationSymbol  # Virtual Damping and Stiffness ($\mathbf{K}_{\text{virt}}$) [Matrix]
    VitPatchProjectionWeights: NotationSymbol  # ViT Projection & Positional Bias ($\mathbf{W}_{\text{patch}}$)
    VitProjectionPositionalBias: NotationSymbol  # ViT Projection & Positional Bias ($\mathbf{E}_{\text{pos}}$) [Matrix]
    Vocabulary: NotationSymbol  # Vocabulary ($\mathcal{V}$)
    WeightMemory: NotationSymbol  # Weight Memory ($M_{\text{weights}}$) [bytes]
    WindingTemperature: NotationSymbol  # Motor Winding Temperature ($T_{\text{winding}}$) [degC]
    WorkerCountWorkers: NotationSymbol  # Worker count ($N_{\text{workers}}$)
    WorldReferenceFrame: NotationSymbol  # World Reference Frame ($\mathcal{F}_W$)
    YoshikawaManipulability: NotationSymbol  # Yoshikawa Manipulability ($w(\mathbf{q})$) [Dimensionless]
    ZeroMomentPointZmp: NotationSymbol  # Zero Moment Point (ZMP) ($\mathbf{p}_{\text{ZMP}}$) [$\text{m}$]
    ZerofailureTrialCount: NotationSymbol  # Zero-Failure Trial Count ($n$)
    ZohTransportDelay: NotationSymbol  # ZOH Transport Delay ($\tau_{\text{ZOH}}$) [$\text{s}$]

Notation: NotationRegistry
Symbols: NotationRegistry
SYMBOLS: NotationRegistry
