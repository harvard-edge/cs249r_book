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

    AcceleratorCount: NotationSymbol  # number of accelerators one job runs on ($N_{\text{GPU}}$)
    AcceleratorsPerNode: NotationSymbol  # number of accelerators in one node ($G_{\text{node}}$)
    AcceptanceFraction: NotationSymbol  # share of a settled cohort of initiated tasks that ends accepted ($\alpha$) [dimensionless]
    AcceptanceProbability: NotationSymbol  # probability that one attempt at a task is accepted, across a workload ($p_{\text{acc}}$) [dimensionless]
    AcceptedInfluences: NotationSymbol  # set of sources a task policy accepts as influences on one operation ($\text{Inf}(o)$)
    AcceptedTasks: NotationSymbol  # number of distinct tasks in a cohort accepted under the declared contract ($N_{\text{acc}}$)
    AccuracyAtTime: NotationSymbol  # model accuracy after the model has been in service for time t since deployment or its last retraining ($\text{Accuracy}(t)$) [dimensionless]
    ActVariationalObjective: NotationSymbol  # training objective of Action Chunking with Transformers: chunk reconstruction error plus a weighted KL penalty on the latent style variable ($\mathcal{L}_{\text{ACT}}(\theta, \phi)$)
    ActionChunk: NotationSymbol  # sequence of future actions a policy predicts in one pass, one row per control step ($\mathbf{A}$)
    ActionChunkHorizon: NotationSymbol  # number of future action steps a policy emits in one forward pass ($H$)
    ActionChunkTrajectory: NotationSymbol  # action chunk written with its time window, the actions from step t up to the horizon ($\mathbf{A}_{t:t+H}$)
    ActionDimension: NotationSymbol  # number of components in one action vector ($d_a$)
    ActionExecutionRate: NotationSymbol  # rate at which the controller consumes the waypoints of an admitted action chunk ($f_{\text{exec}}$) [Hz]
    ActionObservation: NotationSymbol  # result an action returned, such as an exit code, captured output, a response body, or a status ($o_t$)
    ActionProposal: NotationSymbol  # action the model proposes on one turn, as text or a tool call ($a_t$)
    ActivationMemory: NotationSymbol  # memory holding the intermediate activations a model keeps for the backward pass or between serving stages ($M_{\text{activations}}$) [bytes]
    ActuationSelectionMatrix: NotationSymbol  # matrix that maps actuator torques onto the actuated joints of a floating-base robot, leaving the six base coordinates unactuated ($\mathbf{S}$) [dimensionless]
    ActuatorDutyCycle: NotationSymbol  # fraction of each burst cycle an actuator spends at peak power, on-time divided by cycle period ($D_{\text{cycle}}$) [dimensionless]
    ActuatorResponseTime: NotationSymbol  # time from a delivered stop command until the actuator produces retarding force, including gate drive and current rise ($T_{\text{act}}$) [s]
    AdjointRepresentation: NotationSymbol  # six-by-six matrix that carries twists and wrenches from one coordinate frame to another under a rigid transform ($\mathbf{Ad}_T$)
    AdminLeaseTime: NotationSymbol  # duration of the administrative lease a consensus protocol grants before it must be renewed ($T_{\text{admin}}$) [s]
    AdmittedAction: NotationSymbol  # action the runtime permits after checking a proposal ($a_{\text{perm}}$)
    Advantage: NotationSymbol  # estimate of how much better an action does than the policy's average from the same state ($A$) [dimensionless]
    AdversarialPerturbationBudget: NotationSymbol  # largest norm an adversary may give the perturbation it adds to an input ($\epsilon_{\text{adv}}$) [dimensionless]
    AgenticSystem: NotationSymbol  # policy together with its runtime harness, environment, memory, and verifiers ($\mathcal{S}$)
    AllRunsPassProbability: NotationSymbol  # probability that all k runs of a task pass, averaged over tasks ($\text{pass}^k$) [dimensionless]
    AllowableCollisionEnergy: NotationSymbol  # largest kinetic energy a robot may transfer to a person in transient collaborative contact, as set by the safety standard ($E_{k, \text{allow}}$) [J]
    AmbientTemperature: NotationSymbol  # temperature of the air around a motor, the reference for its winding temperature rise ($T_{\text{amb}}$) [degC]
    ArithmeticIntensity: NotationSymbol  # operations a workload performs per byte it moves through memory ($I$) [FLOP/byte]
    ArrivalRate: NotationSymbol  # average number of requests or jobs arriving per second ($\lambda_{\text{arr}}$) [arrivals/s]
    ArticulatedInertiaMatrix: NotationSymbol  # joint-space mass matrix of an articulated robot, mapping joint accelerations to the joint torques they require ($\mathbf{M}$) [kg*m^2]
    AssuranceClaim: NotationSymbol  # top-level safety claim of an assurance argument, whose probability of holding is assessed from its premises ($C$) [dimensionless]
    AttackStepsPgd: NotationSymbol  # number of inner gradient steps a projected gradient descent attack takes to build one adversarial example ($K_{\text{PGD}}$)
    AttainableComputeRate: NotationSymbol  # highest operation rate a workload can sustain on given hardware, capped by peak compute or by memory bandwidth times arithmetic intensity ($R_{\text{attain}}$) [FLOP/s]
    AttemptCost: NotationSymbol  # full monetary cost of one attempt at a task ($C_{\text{attempt}}$) [USD]
    AttemptPassProbability: NotationSymbol  # probability that one independent attempt on a given task passes its check ($p$) [dimensionless]
    AttentionHeadCount: NotationSymbol  # number of attention heads in one multi-head attention layer ($N_{\text{heads}}$)
    AttentionHeadDimension: NotationSymbol  # width of the query, key and value vectors in one attention head ($d_{\text{head}}$)
    AuthorityExposureLevel: NotationSymbol  # authority exposure level, what a task may do through its tools, from read access to irreversible external action ($A_0\text{ to }A_3$) [level]
    AvailableObservations: NotationSymbol  # specification of what the agent may see and how much, including truncation and redaction rules ($\mathcal{O}_{\text{avail}}$)
    BackEmfVoltage: NotationSymbol  # voltage the turning rotor induces in the stator windings, opposing the supply ($e_{\text{bemf}}$) [V]
    Bandwidth: NotationSymbol  # sustained rate at which bytes move across a memory or data path ($\text{BW}$) [bytes/s]
    BarrierConstraintMatrix: NotationSymbol  # matrix of stacked affine control-barrier constraints on the control input ($\mathbf{A}_{\text{cbf}}$)
    BarrierConstraintVector: NotationSymbol  # vector of bounds on the right side of the stacked control-barrier constraints ($\mathbf{b}_{\text{cbf}}$)
    BarrierFunction: NotationSymbol  # control-barrier certificate: a margin function of the state that is nonnegative exactly on the safe set ($h$) [m]
    BarrierRelativeDegree: NotationSymbol  # number of times the barrier function must be differentiated along the dynamics before the control input appears ($r$)
    BasePromptLength: NotationSymbol  # number of prompt tokens in the first call of a trajectory, before any turn appends to it ($S_{\text{base}}$)
    BatchServiceTime: NotationSymbol  # time an accelerator takes to process one batch once execution begins, as a function of batch size ($T_{\text{svc}}(B)$) [s]
    BatchSize: NotationSymbol  # number of examples or requests processed together in one step ($B$)
    BehaviorCloningHorizon: NotationSymbol  # number of control steps in an episode over which imitation errors compound ($S$)
    BeliefInvalidationHorizon: NotationSymbol  # time since the last supporting evidence after which a stored belief may no longer be used ($T_{\text{inv}}$) [s]
    BlendingTransitionWindow: NotationSymbol  # duration of the smooth blend from one latched command to the next during a bumpless transfer ($\tau_{\text{blend}}$) [s]
    BodyReferenceFrame: NotationSymbol  # coordinate frame attached to the mobile robot base ($\mathcal{F}_B$)
    BreakEvenTaskCount: NotationSymbol  # number of tasks an adaptation must serve before its per-task saving recovers its one-time cost ($N^*$) [tasks]
    BumplessTransferBlend: NotationSymbol  # blend weight that ramps from zero to one across a bumpless transfer, as a function of normalized time ($\alpha(\hat{t})$) [dimensionless]
    CachedInputTokenPrice: NotationSymbol  # price of one cached input token ($c_{\text{cache}}$) [USD/token]
    CallLatency: NotationSymbol  # elapsed time of one model call, from request preparation to normalized response ($T_{\text{call}}$) [s]
    CameraFramePoint: NotationSymbol  # position of a point expressed in the camera coordinate frame ($\mathbf{p}_c$) [m]
    CameraIntrinsicCalibrationMatrix: NotationSymbol  # three-by-three matrix of camera intrinsics, focal lengths and principal point, that maps camera-frame points to pixels ($\mathbf{K}$) [px]
    CandidatePoolSize: NotationSymbol  # number of candidates generated for one decision in best-of-N sampling, or the breadth of concurrent candidate streams ($N$)
    CapabilityGrant: NotationSymbol  # capability grant, the rights, objects, condition, and expiry that together describe one delegated authority ($\Gamma$)
    CartesianPosition: NotationSymbol  # position of a point in three-dimensional space ($\mathbf{p}$) [m]
    CentroidalAngularMomentum: NotationSymbol  # angular momentum of the whole robot about its center of mass ($\mathbf{h}_{\text{ang}} \in \mathbb{R}^3$) [kg*m^2/s]
    CentroidalLinearMomentum: NotationSymbol  # total linear momentum of the robot, its mass times its center-of-mass velocity ($\mathbf{h}_{\text{lin}} \in \mathbb{R}^3$) [kg*m/s]
    CertifiedControlAction: NotationSymbol  # control command after the safety filter has projected it onto the admissible set ($\mathbf{u}^*$)
    CheckpointInterval: NotationSymbol  # useful compute time between two completed checkpoints ($T_{\text{ckpt}}$) [s]
    CheckpointWriteTime: NotationSymbol  # time execution pauses to write one checkpoint ($T_{\text{write}}$) [s]
    ChunkLease: NotationSymbol  # longest time an admitted action chunk may keep driving the actuators without renewal ($T_{\text{lease}}$) [s]
    ChunkLeaseTime: NotationSymbol  # duration of the lease a distributed file system grants a chunk server to coordinate writes to one chunk ($T_{\text{lease}}$) [s]
    ChunkRenewalPeriod: NotationSymbol  # nominal interval between successive admissions of the chunk policy's proposals ($T_{\text{chunk}}$) [s]
    ClassKFunction: NotationSymbol  # extended class-K function that bounds how fast the barrier value may decrease ($\kappa$)
    ClassRateFunction: NotationSymbol  # extended class-K comparison function, written as a standalone label ($\kappa(\cdot)$)
    ClearDistance: NotationSymbol  # distance between the machine and the boundary it must stop before ($D_{\text{clear}}$) [m]
    ClockOffset: NotationSymbol  # difference between a follower clock's reading and the leader clock's at the same instant ($\theta_{\text{offset}}$) [s]
    ClockSynchronizationSkew: NotationSymbol  # rate at which two clocks drift apart, such as the application processor and the safety microcontroller ($\delta_{\text{skew}}$) [ppm]
    ClosureOperator: NotationSymbol  # runtime's check that maps each proposed action and the current system state to a permitted action or a rejection ($\mathcal{I}$)
    ClusterNodeCount: NotationSymbol  # number of nodes in a cluster or job ($N_{\text{nodes}}$)
    CodecDecodeTime: NotationSymbol  # time to decompress a compressed payload after it arrives ($T_{\text{decode}}$) [s]
    CodecEncodeTime: NotationSymbol  # time to compress a payload before it is sent ($T_{\text{encode}}$) [s]
    CognitivePolicyAction: NotationSymbol  # action the learned policy emits at step t, before any safety check ($\mathbf{a}_t$)
    CoherencyCoefficient: NotationSymbol  # universal scalability law's cost of pairwise reconciliation between workers ($\kappa_{\text{coh}}$) [dimensionless]
    CollectivePayloadSize: NotationSymbol  # size of the message each participant contributes to a collective operation ($M$) [bytes]
    CommandTransportTime: NotationSymbol  # time for a command to travel over the fieldbus from the permission path to the drive ($T_{\text{bus}}$) [s]
    CommandedJointTorque: NotationSymbol  # torque commanded at a single joint in one-joint derivations ($\tau$) [N*m]
    CommitmentInstant: NotationSymbol  # instant after which a receding-horizon plan's prefix is committed and can no longer be replaced ($t_{\text{commit}}$) [s]
    CommunicationComputeRatio: NotationSymbol  # ratio of per-step communication time to per-device compute time ($\rho$) [dimensionless]
    CommunicationIntensity: NotationSymbol  # network bytes a workload moves per floating-point operation it performs ($\text{CI}$) [bytes/FLOP]
    CommunicationTime: NotationSymbol  # time per training step spent moving gradients, activations and parameter shards between devices at scale N ($T_{\text{comm}}(N)$) [s]
    CompletionCriteria: NotationSymbol  # set of checks that decide when a task is done, at a stated closure evidence level ($\mathcal{K}_{\text{comp}}$)
    CompletionDeadline: NotationSymbol  # time by which a call, cycle, or task must complete ($T_{\text{deadline}}$) [s]
    ComponentAvailability: NotationSymbol  # probability that one component is up at a given instant ($A_{\text{component}}$) [dimensionless]
    ComponentFailureRate: NotationSymbol  # rate at which one component fails ($\lambda$) [failures/hour]
    CompressedCommunicationTime: NotationSymbol  # total time to encode, transfer and decode a compressed payload ($T_{\text{compressed}}$) [s]
    CompressionRatio: NotationSymbol  # uncompressed size of a payload divided by its compressed size ($r_{\text{comp}}$) [dimensionless]
    ConcurrentTrajectories: NotationSymbol  # mean number of admitted trajectories present at once ($N_{\text{traj}}$)
    ConditionalDistribution: NotationSymbol  # probability distribution of the label given an input, the input-to-label relationship that concept drift changes ($p(y \mid x)$)
    ConfidentialityLabel: NotationSymbol  # set of recipients permitted to receive a record ($\kappa(x)$)
    ConformalMiscoverageRate: NotationSymbol  # allowed probability that the conformal prediction set misses the true referent ($\epsilon$) [dimensionless]
    ConformalPredictionSet: NotationSymbol  # calibrated set of referent candidates that cannot be excluded at the miscoverage rate ($\mathcal{C}$)
    ContactPointForce: NotationSymbol  # force the environment exerts on the robot at one contact point ($\mathbf{f}_k$) [N]
    ContactPointJacobian: NotationSymbol  # matrix mapping joint velocities to the linear velocity of one contact point ($\mathbf{J}_k(\mathbf{q})$)
    ContactStiffness: NotationSymbol  # stiffness of a contact treated as a spring, force per unit penetration ($k_{\text{contact}}$) [N/m]
    ContentionCoefficient: NotationSymbol  # universal scalability law's share of work that serializes across workers ($\sigma_{\text{cont}}$) [dimensionless]
    ContextWindow: NotationSymbol  # hard per-call limit on prompt and output tokens together ($S_{\max}$)
    ContinuationCount: NotationSymbol  # number of independent continuations sampled from one fork condition in reinforcement learning ($K$)
    ContinuousCurrent: NotationSymbol  # largest motor current the winding can carry indefinitely without exceeding its temperature limit ($I_{\text{cont}}$) [A]
    ControlAffineDriftField: NotationSymbol  # unforced part of control-affine dynamics, the state rate when the control input is zero ($\mathbf{f}(\mathbf{x})$)
    ControlAffineInputField: NotationSymbol  # input matrix field of control-affine dynamics, mapping control inputs to state rates ($\mathbf{g}(\mathbf{x})$)
    ControlBarrierFunction: NotationSymbol  # control-barrier certificate evaluated at the state, whose superlevel set is the safe set ($h(\mathbf{x})$) [m]
    ControlSetpointJerk: NotationSymbol  # time derivative of the commanded acceleration or torque during a mode transition ($\mathbf{j}(t)$)
    ControllabilityMatrix: NotationSymbol  # block matrix of the input matrix and its products with powers of the system matrix, whose rank decides controllability ($\mathbf{C}_{\text{ctrb}}$)
    CoordinationOverhead: NotationSymbol  # added elapsed time of splitting work across workers, as a fraction of serial time ($h$) [dimensionless]
    CoriolisCentrifugalMatrix: NotationSymbol  # matrix of Coriolis and centrifugal effects in the manipulator equation, multiplying joint velocities ($\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})$)
    CoriolisMatrix: NotationSymbol  # matrix of Coriolis and centrifugal effects in the manipulator equation, multiplying joint velocities ($\mathbf{C}$)
    CorrelationCoefficient: NotationSymbol  # Pearson correlation coefficient between two random quantities ($\rho$) [dimensionless]
    CostCeiling: NotationSymbol  # dollar ceiling on a trajectory that the runtime enforces before dispatching the next call ($C_{\max}$) [USD]
    CostPerAcceptedTask: NotationSymbol  # total spend on all attempts in a cohort divided by the number of accepted tasks ($C_{\text{eff}}$) [USD/task]
    CoulombFrictionCoefficient: NotationSymbol  # ratio of the largest tangential friction force to the normal force at a contact ($\mu$) [dimensionless]
    CovariateDriftBound: NotationSymbol  # worst-case excess task cost of behavior cloning, growing with the square of the episode length ($\mathcal{O}(S^2 \epsilon)$)
    CredibleBrakingDeceleration: NotationSymbol  # validated low-tail deceleration the machine achieves under its stated conditions, not the datasheet peak ($a_{\text{brake}}$) [m/s^2]
    CrossoverFrequency: NotationSymbol  # angular frequency at which the open-loop gain of a feedback loop falls to one ($\omega_c$) [rad/s]
    CurrentDistribution: NotationSymbol  # data distribution a deployed model sees in service at time t ($P_t$)
    CvaeRecognitionEncoder: NotationSymbol  # approximate posterior of the latent style variable given an observation, learned by a conditional VAE encoder ($q_\phi(\mathbf{z} \mid \mathbf{o})$)
    DataParallelDegree: NotationSymbol  # number of model replicas that each process a different slice of the batch in data parallelism ($N_{\text{DP}}$)
    DataVolume: NotationSymbol  # total bytes a workload moves through memory or across a data path ($D_{\text{vol}}$) [bytes]
    DatasetSize: NotationSymbol  # number of training examples or tokens in a dataset ($D$)
    DcBusBulkCapacitance: NotationSymbol  # capacitance of the capacitor bank on the inverter DC bus that buffers transient load current ($C_{\text{bus}}$) [F]
    DcBusDynamicRailDroop: NotationSymbol  # transient drop in DC bus voltage under a step in motor current ($\Delta V_{\text{droop}}$) [V]
    DecodeInterval: NotationSymbol  # time a call spends generating its whole output ($T_{\text{decode}}$) [s]
    DecoderEvaluationCount: NotationSymbol  # number of dependent forward passes a generative action head makes per chunk, such as denoising or ODE-solver steps ($N_{\text{eval}}$)
    DelayPhaseLag: NotationSymbol  # phase lag that loop delays add at the crossover frequency, eroding the phase margin ($\Delta \phi$) [rad]
    DelegatedTaskGoal: NotationSymbol  # delegated task and its completion rule, fixed for the trajectory ($g$)
    DetectionInterval: NotationSymbol  # time a detector takes to recognize a hazard or an expiry, beyond what the observation age already charges ($t_{\text{detect}}$) [s]
    DeviceCount: NotationSymbol  # number of devices (accelerators) taking part in one distributed job, also the rank count of a collective ($N$)
    DiagnosticCoverage: NotationSymbol  # rate-weighted fraction of hardware faults whose hazardous effect a diagnostic mitigates ($C_{\text{diag}}$) [dimensionless]
    DiffusionCumulativeSchedule: NotationSymbol  # fraction of the clean signal variance retained after k diffusion steps ($\bar{\alpha}_k$) [dimensionless]
    DiffusionGaussianNoise: NotationSymbol  # standard Gaussian noise added to clean action chunks in the diffusion forward process ($\boldsymbol{\epsilon}$)
    DiffusionMatchingLoss: NotationSymbol  # mean squared error between the injected noise and the noise the network predicts, averaged over diffusion steps ($\mathcal{L}_{\text{diff}}(\theta)$)
    DirectionalCosineMatrix: NotationSymbol  # three-by-three rotation matrix giving the orientation of one frame relative to another ($\mathbf{R}$) [dimensionless]
    DispatchAdmissionInterval: NotationSymbol  # time from dispatching a proposal to its admission by the permission path ($\Delta_{\text{da}}$) [s]
    DispatchedTrajectorySet: NotationSymbol  # set of every trajectory dispatched ($\mathcal{T}_{\text{all}}$)
    DistillationTemperature: NotationSymbol  # softmax temperature that softens teacher and student output probabilities in knowledge distillation ($T_{\text{distill}}$) [dimensionless]
    DistributedStepTime: NotationSymbol  # wall-clock time of one training step on N devices ($T_{\text{step}}(N)$) [s]
    DivergenceInstant: NotationSymbol  # time at which an autonomous policy's rollout begins to leave its supported states ($t_{\text{div}}$) [s]
    DivergenceTolerance: NotationSymbol  # positive threshold on the trajectory-divergence residual that marks divergence ($\delta_{\text{tol}}$) [m]
    DuplicateEffectProbability: NotationSymbol  # probability that one blind retry of a non-idempotent call applies its effect twice ($p_{\text{dup}}$) [dimensionless]
    EffectiveFleetThroughput: NotationSymbol  # useful floating-point rate a fleet sustains after per-device, scaling and operational losses ($R_{\text{eff}}$) [FLOP/s]
    EffectiveImpactMass: NotationSymbol  # apparent mass the robot presents along the contact normal at a given contact point and configuration ($m_{\text{eff}}$) [kg]
    EffectiveJointInertia: NotationSymbol  # total inertia seen at one joint, link and payload plus reflected rotor inertia ($J_{\text{eff}}$) [kg*m^2]
    EffectiveServiceRate: NotationSymbol  # requests a batched server completes per second at a given batch size ($\mu_{\text{eff}}(B)$) [requests/s]
    EfficiencyHw: NotationSymbol  # fraction of peak compute rate the hardware actually sustains on a workload ($\eta_{\text{hw}}$) [dimensionless]
    ElectricalResistance: NotationSymbol  # electrical resistance of a circuit or winding ($R$) [Ohm]
    ElectricalTimeConstant: NotationSymbol  # time constant of current rise in a motor winding, inductance over resistance ($\tau_e$) [s]
    ElementStorageSize: NotationSymbol  # bytes needed to store one tensor element at its numerical precision ($s_{\text{elem}}$) [bytes/element]
    EmittedTokenCount: NotationSymbol  # number of tokens a decoder emits one after another for one output ($S_{\text{out}}$)
    EndEffectorReferenceFrame: NotationSymbol  # coordinate frame attached to the robot's end effector ($\mathcal{F}_E$)
    EndToEndRequestLatency: NotationSymbol  # time from a request leaving the client to its response returning, summed over every stage on the request path ($L_{\text{lat, total}}$) [s]
    EnergyPerByteMoved: NotationSymbol  # energy spent moving one byte between memory and the compute units ($E_{\text{move}}$) [J/byte]
    EnergyPerOperation: NotationSymbol  # energy spent executing one arithmetic operation ($E_{\text{compute}}$) [J/operation]
    EnforcerComputeTime: NotationSymbol  # time the enforcer takes to evaluate its constraints within one control period ($T_{\text{comp}}$) [s]
    EnsemblingWindowLength: NotationSymbol  # number of overlapping past chunks blended together in temporal ensembling ($K$)
    EnvironmentState: NotationSymbol  # full environment state of the decision process at a turn, including what the model never saw ($s_t$)
    EssentialMatrix: NotationSymbol  # three-by-three matrix relating corresponding normalized image points in two calibrated views ($\mathbf{E}$)
    EstimatedHistoryAdvantage: NotationSymbol  # estimate of how much an action taken after a given interaction history beats the policy value at that history ($\hat{A}(h_k, a_k)$) [dimensionless]
    EstimatedRemainingReturn: NotationSymbol  # estimated expected remaining return under a continuation policy, given the interaction history at a turn ($\hat{V}(h_k)$) [dimensionless]
    EstimationErrorCovariance: NotationSymbol  # covariance of the error between the true state and its estimate ($\mathbf{P}$)
    EventProbability: NotationSymbol  # probability that a stated event occurs ($\Pr(E)$) [dimensionless]
    EvidenceHorizon: NotationSymbol  # time for which a target estimate stays within its task tolerance without fresh evidence ($\tau_{\text{ev}}$) [s]
    ExpectedActionReturn: NotationSymbol  # expected return, or expected cost in the imitation bounds, from a step after taking an action in a state and then following the policy ($Q_t^{\pi}(s_t, a_t)$) [dimensionless]
    ExpectedCheckingTime: NotationSymbol  # expected verifier slot time spent per attempted record under a short-circuit check cascade ($T_{\text{check}}$) [s]
    ExpectedIrreversibleLoss: NotationSymbol  # expected consequence of the bad actions in an execution plan that cannot be recovered ($L_A$) [consequence units]
    ExposureDuration: NotationSymbol  # integration time of one camera frame ($t_{\text{exp}}$) [s]
    ExternalWait: NotationSymbol  # time spent waiting on a remote service or on a person's response ($T_{\text{wait}}$) [s]
    FallbackHold: NotationSymbol  # optional hold or fault-confirmation interval before the stop command reaches the drive ($T_{\text{hold}}$) [s]
    FittedSensitivity: NotationSymbol  # accuracy lost per unit of distribution divergence, fitted from labeled outcomes over a stated range ($\lambda$) [dimensionless]
    FleetEnergyProductivity: NotationSymbol  # useful floating-point operations a fleet delivers per joule of device, cooling and network energy ($\rho_{\text{energy}}$) [FLOP/J]
    ForwardInvariantSafeSet: NotationSymbol  # set of states on which the barrier certificate is nonnegative, kept invariant by the safety filter ($\mathcal{C}$)
    FreeSpaceDepth: NotationSymbol  # distance from a hull's surface to the nearest occupied or unknown cell ($d_{\text{free}}$) [m]
    FundamentalMatrix: NotationSymbol  # three-by-three matrix relating corresponding pixel points in two uncalibrated views ($\mathbf{F}$)
    GearReductionRatio: NotationSymbol  # ratio of motor speed to joint output speed across a transmission ($N_{\text{gear}}$) [dimensionless]
    GearboxTransmissionEfficiency: NotationSymbol  # fraction of input mechanical power a transmission delivers at its output ($\eta_{\text{gear}}$) [dimensionless]
    GeneralizedAccelerations: NotationSymbol  # second time derivative of the generalized coordinates ($\ddot{\mathbf{q}}$)
    GeneralizedCoordinates: NotationSymbol  # joint angles and prismatic extensions that fix the configuration of a mechanism ($\mathbf{q}$)
    GeneralizedInertiaMatrix: NotationSymbol  # joint-space mass matrix of an articulated robot, mapping joint accelerations to the joint torques they require ($\mathbf{M}(\mathbf{q})$) [kg*m^2]
    GeneralizedVelocities: NotationSymbol  # first time derivative of the generalized coordinates ($\dot{\mathbf{q}}$)
    GoalSet: NotationSymbol  # set of terminal poses within the positional and angular tolerances of a task ($\mathcal{G}_{\text{goal}}$)
    GoodputRatio: NotationSymbol  # fraction of allocated wall time left for productive steps after out-of-step operational losses ($\eta_{\text{goodput}}$) [dimensionless]
    GradientMemory: NotationSymbol  # memory holding the gradient of every trainable parameter during training ($M_{\text{gradients}}$) [bytes]
    GradientStaleness: NotationSymbol  # number of parameter updates between computing a gradient and applying it ($\tau_{\text{stale}}$)
    GrantExpiryTime: NotationSymbol  # time after which a capability grant admits nothing ($t_{\text{exp}}$) [s]
    GravitationalTorqueVector: NotationSymbol  # joint torques that gravity exerts on the links in a given configuration ($\mathbf{g}(\mathbf{q})$) [N*m]
    Gravity: NotationSymbol  # gravitational acceleration at the surface of the Earth ($g$) [m/s^2]
    GroupSize: NotationSymbol  # number of rollouts sampled per task for group-relative scoring ($G$)
    HandoverTimeout: NotationSymbol  # countdown an arbiter sets for an unacknowledged handover before falling back to a controlled stop ($T_{\text{timeout}}$) [s]
    HardwareCommonCauseRate: NotationSymbol  # rate of hazards from hardware faults and common causes that no shield coverage removes ($\lambda_{\text{other}}$) [1/h]
    HardwareWatchdogTimeout: NotationSymbol  # period of the hardware watchdog timer, after which missing heartbeats trigger fail-safe braking ($t_{\text{watchdog}}$) [s]
    HazardOnset: NotationSymbol  # interval from the last valid renewal to the physical onset of powered deceleration ($t_{\text{onset}}$) [s]
    HazardousProposalRate: NotationSymbol  # rate at which the policy emits hazardous action proposals of class h before the permission layer ($\lambda_{\text{proposal}, h}$) [1/h]
    HertzianContactStiffness: NotationSymbol  # effective spring constant of the body region struck in a collaborative contact ($k_H$) [N/m]
    HiddenDimension: NotationSymbol  # width of a model's hidden representation, the transformer model width or a recurrent network's hidden-state size ($d$)
    HoldingTorque: NotationSymbol  # joint torque a motor must sustain at standstill to hold a load ($\tau_{\text{hold}}$) [N*m]
    HomogeneousPixelCoordinates: NotationSymbol  # image point written in homogeneous coordinates, pixel column and row with a trailing one ($\tilde{\mathbf{m}}$) [px]
    HomogeneousTransform: NotationSymbol  # four-by-four matrix mapping coordinates from frame B into frame A ($\mathbf{T}_{AB}$)
    HumanWalkingSpeedBound: NotationSymbol  # speed bound for a person approaching the machine, as the safety standard states it ($v_h$) [m/s]
    HypothesisTestFalseNegativeRate: NotationSymbol  # probability that a hypothesis test misses a true effect ($\beta_{\text{stat}}$) [dimensionless]
    HypothesisTestSignificanceLevel: NotationSymbol  # probability that a hypothesis test rejects a true null hypothesis ($\alpha_{\text{sig}}$) [dimensionless]
    IdempotencyKey: NotationSymbol  # identifier for one intended operation, kept across retries so its effect applies once ($k_{\text{idem}}$)
    ImagePlanePixelCoordinates: NotationSymbol  # pixel coordinates of a point in the image plane ($\mathbf{u}_{\text{px}}$) [px]
    InertiaMatchingRatio: NotationSymbol  # gear ratio that maximizes load acceleration for a given motor torque, where reflected rotor inertia equals load inertia ($N^*$) [dimensionless]
    InferenceCost: NotationSymbol  # recurring cost of serving a model in production ($C_{\text{infer}}$) [USD]
    InitialAccuracy: NotationSymbol  # model accuracy at deployment, or right after the most recent retraining ($\text{Accuracy}_0$) [dimensionless]
    InitialTrackingError: NotationSymbol  # position error of an estimate at the moment its evidence is captured ($e_0$) [m]
    InitiatedTasks: NotationSymbol  # number of tasks in a settled cohort, whatever their outcome ($N$)
    InputDistribution: NotationSymbol  # probability distribution of a model's inputs, whose change is data drift ($p(x)$)
    InputTokenPrice: NotationSymbol  # price of one uncached input token ($c_{\text{in}}$) [USD/token]
    InstantaneousPhaseCurrent: NotationSymbol  # signed instantaneous current in one motor phase, positive from the inverter node into the winding ($i_{\text{phase}}$) [A]
    IntegrityProvenance: NotationSymbol  # set of sources whose content may have influenced a record ($\iota(x)$)
    IntentLease: NotationSymbol  # time-bounded authorization from the permission path that binds an admitted target, its expiry, and its effort ceilings ($\mathcal{I}_{\text{intent}}$)
    IntentLeaseExpiry: NotationSymbol  # instant at which an intent lease expires: the earliest of evidence decay, host silence, and the stopping deadline ($t_{\text{intent\_exp}}$) [s]
    InterAnnotatorAgreement: NotationSymbol  # Cohen's kappa, agreement between two annotators corrected for the agreement expected by chance ($\kappa$) [dimensionless]
    InterconnectPortConflicts: NotationSymbol  # number of simultaneous memory-channel access collisions among DMA, GPU and CPU masters ($N_{\text{conflict}}$)
    InterconnectTransferTime: NotationSymbol  # time to move sensor data across on-board links such as DMA, PCIe or SerDes ($t_{\text{xfer}}$) [s]
    IronLawComputeTime: NotationSymbol  # time spent computing in the iron law, operations divided by the sustained compute rate ($T_{\text{compute}}$) [s]
    IronLawMemoryTime: NotationSymbol  # time spent moving data in the iron law, bytes moved divided by bandwidth ($T_{\text{memory}}$) [s]
    IrreducibleLoss: NotationSymbol  # loss floor that adding parameters, data or compute does not remove within the fitted scaling regime ($C_{\mathcal{L}}$) [dimensionless]
    JerkLimit: NotationSymbol  # largest rate of change of acceleration a planned or stopping profile may command ($j_{\max}$) [m/s^3]
    JointTorque: NotationSymbol  # vector of torques the actuators apply at the joints ($\boldsymbol{\tau}$) [N*m]
    JouleanTemperatureRise: NotationSymbol  # temperature rise of a winding above its reference caused by resistive heating ($\Delta T$) [K]
    KalmanGain: NotationSymbol  # matrix weighting the measurement innovation in the Kalman filter state update ($\mathbf{G}_k$)
    KineticInsetBarrierFunction: NotationSymbol  # barrier certificate shrunk by the stopping distance at the current speed, so its relative degree is one ($h_{\text{kinetic}}(\mathbf{x})$) [m]
    KvBlockSize: NotationSymbol  # number of token positions in one fixed block of a paged KV cache ($S_{\text{blk}}$)
    KvBytesPerToken: NotationSymbol  # bytes of KV cache one token occupies across all layers ($m_{\text{token}}$) [bytes/token]
    KvCacheBytesPerToken: NotationSymbol  # bytes of key-value cache stored for each token held in context ($m_{\text{token}}$) [bytes/token]
    KvCacheFootprint: NotationSymbol  # memory a transformer key-value cache occupies during autoregressive decoding ($M_{\text{KV}}$) [bytes]
    KvCacheMemory: NotationSymbol  # memory the key-value cache occupies during autoregressive inference ($M_{\text{KV}}$) [bytes]
    KvHeadCount: NotationSymbol  # number of key-value heads in grouped-query or multi-query attention ($H_{\text{KV}}$)
    Latency: NotationSymbol  # fixed overhead time a step pays regardless of bytes moved or operations executed, such as kernel launch, dispatch or network round trip ($L_{\text{lat}}$) [s]
    LatencyBandwidthCrossoverSize: NotationSymbol  # message size at which per-message startup time equals transfer time in the alpha-beta model ($n^*$) [bytes]
    LatentStyleVariable: NotationSymbol  # latent variable of a conditional VAE policy that captures the demonstration style ($\mathbf{z}$)
    LayerCount: NotationSymbol  # number of layers in a network ($N_L$)
    LayerIndex: NotationSymbol  # position of one layer within a network, counted from the input ($\ell$)
    LayerInputWidth: NotationSymbol  # number of input features, or input channels, of one layer ($d_{\text{in}}$)
    LayerOutputWidth: NotationSymbol  # number of output features, or output channels, of one layer ($d_{\text{out}}$)
    LearningRate: NotationSymbol  # step size the optimizer multiplies the gradient by in each parameter update ($\eta$) [dimensionless]
    LeaseHoldTime: NotationSymbol  # mean time a trajectory holds a sandbox or other leased resource ($T_{\text{hold}}$) [s]
    LieDerivativeAlongControl: NotationSymbol  # sensitivity of the barrier rate to the control input, the barrier gradient applied to the input field ($L_{\mathbf{g}} h(\mathbf{x})$)
    LieDerivativeAlongDrift: NotationSymbol  # rate at which the barrier value changes under the unforced dynamics alone ($L_{\mathbf{f}} h(\mathbf{x})$) [m/s]
    LinearBarrierDecayRate: NotationSymbol  # rate constant of the linear class-K function in the barrier condition ($\gamma_{\text{cbf}}$) [1/s]
    LinearMomentum: NotationSymbol  # mass times speed of a moving body ($p$) [kg*m/s]
    LinkBandwidth: NotationSymbol  # bandwidth of one network link in each direction, in the alpha-beta communication model ($\beta$) [bytes/s]
    LoadInertia: NotationSymbol  # moment of inertia of the mechanical load a joint drives, on the output side of the transmission ($J_{\text{load}}$) [kg*m^2]
    LocalizationBound: NotationSymbol  # bound on the error of localization and state estimation ($\delta_{\text{loc}}$) [m]
    LoggpLongMessageTime: NotationSymbol  # time to send one long message of n bytes in the LogGP model ($T_{\text{long}}(n)$) [s]
    LogpNetworkLatency: NotationSymbol  # time a message spends in flight across the network in the LogP and LogGP models, excluding processor overhead ($L_{\text{LogGP}}$) [s]
    LongMessageGapPerByte: NotationSymbol  # time per byte to send a long message in the LogGP communication model ($G$) [s/byte]
    Loss: NotationSymbol  # value of the training objective a model minimizes ($L$) [dimensionless]
    LossFunction: NotationSymbol  # training objective that scores a model's predictions against the labels, the quantity training minimizes ($\mathcal{L}$)
    LossGradient: NotationSymbol  # gradient of the training loss with respect to one model parameter, as an optimizer update uses it ($g$) [loss per parameter unit]
    LossScaleFactor: NotationSymbol  # multiplier applied to the loss in reduced-precision training so small gradients stay representable ($s_{\text{loss}}$) [dimensionless]
    LqrControlWeightMatrix: NotationSymbol  # penalty matrix on control effort in the linear-quadratic regulator cost ($\mathbf{R}_u$)
    LqrFeedbackGain: NotationSymbol  # state-feedback gain matrix of the linear-quadratic regulator ($\mathbf{K}_{\text{lqr}}$)
    LqrStateWeightMatrix: NotationSymbol  # penalty matrix on state deviation in the linear-quadratic regulator cost ($\mathbf{Q}_x$)
    ManipulatorJacobian: NotationSymbol  # matrix mapping joint velocities to the spatial velocity of the end effector ($\mathbf{J}(\mathbf{q})$)
    MaskedFinetuningLoss: NotationSymbol  # mean cross-entropy over the token positions the loss mask selects, for one example ($\mathcal{L}_{\text{SFT}}$) [nats]
    Mass: NotationSymbol  # mass of a physical body ($m$) [kg]
    MassCenterPosition: NotationSymbol  # position of the robot's total center of mass ($\mathbf{c}$) [m]
    MeanTimeBetweenFailures: NotationSymbol  # average time between consecutive failures of a repairable component or system ($\text{MTBF}$) [hours]
    MeanTimeToFailure: NotationSymbol  # average operating time of a component before it fails, derived from its FIT rate ($\text{MTTF}$) [hours]
    MeanTimeToRepair: NotationSymbol  # average time to restore service after a failure ($\text{MTTR}$) [hours]
    MeasurementMatrix: NotationSymbol  # output matrix of a linear state-space model, mapping the state to sensor measurements ($\mathbf{H}$)
    MeasurementNoiseVector: NotationSymbol  # random disturbance added to the sensor output ($\mathbf{v}_{\text{meas}}(t)$)
    MechanicalPlantState: NotationSymbol  # state of the mechanical plant: generalized positions stacked with generalized velocities ($\mathbf{x}$)
    MemoryBusQueueDelay: NotationSymbol  # time a sensor transfer waits for contended shared memory channels on the SoC ($t_{\text{queue}}$) [s]
    MemoryEntryBirthGeneration: NotationSymbol  # source generation from which a derived memory entry is supported ($g_{\text{birth}}(c)$)
    MemoryEntryDeathGeneration: NotationSymbol  # source generation at which a later edit stops supporting a derived memory entry ($g_{\text{death}}(c)$)
    MemoryHoldDuringWait: NotationSymbol  # accelerator memory a paused trajectory withholds, multiplied by how long it waits ($O_{\text{mem}}$) [byte-seconds]
    MessageInjectionGap: NotationSymbol  # minimum interval between consecutive message injections at one node in the LogP and LogGP models ($g$) [s]
    MessageStartupLatency: NotationSymbol  # fixed startup time paid once per message in the alpha-beta communication model, independent of message size ($\alpha$) [s]
    MicrobatchCount: NotationSymbol  # number of microbatches one pipeline-parallel step splits its batch into ($m$)
    ModelFlopsUtilization: NotationSymbol  # useful model operations per second divided by the hardware's peak operation rate ($\text{MFU}$) [dimensionless]
    ModelParameters: NotationSymbol  # learned weights and biases of a model or policy ($\theta$)
    ModelTime: NotationSymbol  # time a turn spends in model calls ($T_{\text{model}}$) [s]
    MotorBackEmfConstant: NotationSymbol  # back-EMF voltage a motor generates per unit rotor speed ($K_e$) [V/(rad/s)]
    MotorPhaseRmsCurrent: NotationSymbol  # root-mean-square current in one stator phase of a brushless motor ($I_{\text{phase}}$) [A]
    MotorTorqueConstant: NotationSymbol  # torque a motor produces per ampere of torque-producing current ($K_t$) [N*m/A]
    MpcPredictionHorizon: NotationSymbol  # number of discrete control steps over which model predictive control optimizes ($N_{\text{mpc}}$)
    MultiDeviceRunTime: NotationSymbol  # wall-clock time for a fixed amount of productive work on N devices ($T_N$) [s]
    MultimodalFusedTokenStream: NotationSymbol  # concatenated sequence of vision, language and state tokens fed to an embodied transformer ($\mathbf{Z}$)
    MultimodalObservation: NotationSymbol  # robot's synchronized sensor observation at step t: images, depth, joint encoders and touch ($\mathbf{o}_t$)
    NetBrakingDeceleration: NotationSymbol  # deceleration left after a gravity or load bias is subtracted from the available braking effort ($a_{\text{net}}$) [m/s^2]
    NetPhaseMargin: NotationSymbol  # phase margin left after subtracting the loop's delay-induced phase lag ($\text{PM}_{\text{net}}$) [deg]
    NmiResponseTime: NotationSymbol  # worst-case response time of the non-maskable interrupt that handles motor overcurrent or gate-drive faults ($t_{\text{NMI}}$) [s]
    NoisePredictionNetwork: NotationSymbol  # network that predicts the noise added to an action chunk at a diffusion step, given the observation ($\boldsymbol{\epsilon}_\theta(\mathbf{o}_t, \mathbf{A}_t^k, k)$)
    NoisedActionChunk: NotationSymbol  # action chunk after k steps of diffusion noising, the iterate a denoising pass refines ($\mathbf{A}^k$)
    NominalControlCommand: NotationSymbol  # control command the learned policy proposes, before the safety filter checks it ($\mathbf{u}_{\text{nom}}$)
    NonlinearObservationSpace: NotationSymbol  # space spanned by the output functions and all their repeated Lie derivatives along the dynamics ($\mathbf{O}_{\text{NL}}$)
    NormalContactForce: NotationSymbol  # compressive component of a contact force along the contact normal ($f_{k, \perp}$) [N]
    ObservabilityMatrix: NotationSymbol  # stack of the output matrix times powers of the system matrix, whose rank decides observability ($\mathbf{O}_{\text{obs}}$)
    ObservationAge: NotationSymbol  # age of the state the permission path acts on, from the exposure midpoint at the sensor to the decision ($t_{\text{age}}$) [s]
    OperationalOverheadFraction: NotationSymbol  # fraction of allocated wall time lost outside productive training steps ($f_{\text{overhead}}$) [dimensionless]
    Operations: NotationSymbol  # total arithmetic operations a workload executes ($O$) [FLOP]
    OptimalCheckpointInterval: NotationSymbol  # checkpoint interval that minimizes expected overhead from checkpoint writes and lost work ($T^\ast_{\text{ckpt}}$) [s]
    OptimizerStateMemory: NotationSymbol  # memory holding the optimizer's per-parameter state, such as momentum, variance and master weights ($M_{\text{optimizer}}$) [bytes]
    OutputLength: NotationSymbol  # number of tokens a call generates ($K$)
    OutputLimit: NotationSymbol  # maximum number of tokens the runtime lets a call generate ($K_{\max}$)
    OutputTokenPrice: NotationSymbol  # price of one generated output token ($c_{\text{out}}$) [USD/token]
    OverlappedTime: NotationSymbol  # communication or coordination time per step hidden behind computation ($T_{\text{overlap}}$) [s]
    ParallelFraction: NotationSymbol  # share of the work that divides among workers ($f_{\text{parallel}}$) [dimensionless]
    ParameterStreamingTime: NotationSymbol  # time to stream model weights that do not stay resident into the compute units for one evaluation ($t_{\text{stream}}$) [s]
    ParameterizedNeuralPolicy: NotationSymbol  # distribution over actions given an observation, computed by a neural network with learned weights ($\pi_\theta(\mathbf{a} \mid \mathbf{o})$)
    Parameters: NotationSymbol  # number of trainable parameters in a model ($P$)
    PassAtK: NotationSymbol  # probability that at least one of k runs of a task passes, averaged over tasks ($\text{pass}@k$) [dimensionless]
    PathAccelerationTorqueCoefficient: NotationSymbol  # vector multiplying path acceleration in the path-parameterized manipulator torque equation ($\mathbf{a}_{\tau}(s)$) [N*m*s^2]
    PathCoriolisMatrix: NotationSymbol  # Coriolis and centrifugal matrix evaluated along a geometric path for time parameterization ($\mathbf{C}(\mathbf{q}(s), \mathbf{q}'(s))$)
    PathVelocitySquaredTorqueCoefficient: NotationSymbol  # vector multiplying squared path velocity in the path-parameterized manipulator torque equation ($\mathbf{b}_{\tau}(s)$) [N*m*s^2]
    PeakCurrent: NotationSymbol  # largest motor current permitted during a short burst ($I_{\text{peak}}$) [A]
    PeakInternodeTimingError: NotationSymbol  # largest timing disagreement between bus nodes during a synchronization blackout ($\Delta t_{\text{peak}}$) [s]
    PeakRate: NotationSymbol  # highest operation rate the hardware can execute, as specified for a given precision ($R_{\text{peak}}$) [FLOP/s]
    PeakTorque: NotationSymbol  # largest transient joint torque permitted during short bursts ($\tau_{\text{peak}}$) [N*m]
    PerStepErrorRate: NotationSymbol  # probability that one unchecked turn of a trajectory is wrong ($\epsilon$) [dimensionless]
    PerTokenDecodeTime: NotationSymbol  # time to generate one output token on a single stream ($t_{\text{decode}}$) [s]
    PerceptionDepthCovariance: NotationSymbol  # covariance of the three-dimensional position error of a point lifted from stereo, lidar or depth data ($\boldsymbol{\Sigma}_{\text{depth}}$) [m^2]
    PermissionLoopPeriod: NotationSymbol  # period of the permission loop, the longest a hazard may wait before it is checked ($T_{\text{tick}}$) [s]
    PermittedActions: NotationSymbol  # set of tools the task may call, with the limits on each ($\mathcal{A}_{\text{perm}}$)
    PhaseInductance: NotationSymbol  # inductance of one motor phase winding ($L_{\text{phase}}$) [H]
    PhaseResistance: NotationSymbol  # resistance of one motor phase winding ($R_{\text{phase}}$) [Ohm]
    PhysicalWorldState: NotationSymbol  # ground-truth state of the machine, its environment and their contacts at step t ($s_t$)
    PidDerivativeGain: NotationSymbol  # feedback gain on the velocity error, the damping of a PD or impedance controller ($K_d$) [N*s/m]
    PidIntegralGain: NotationSymbol  # feedback gain on the accumulated error in a PI or PID controller ($K_i$) [controller output per error-second]
    PidProportionalGain: NotationSymbol  # feedback gain on the position error, the stiffness of a PD or impedance controller ($K_p$) [N/m]
    PipelineBubbleFraction: NotationSymbol  # fraction of pipeline device time spent idle while the pipeline fills and drains ($b_{\text{pipe}}$) [dimensionless]
    PipelineParallelDegree: NotationSymbol  # number of pipeline stages a model's layers are split across ($N_{\text{PP}}$)
    PointToPointMessageSize: NotationSymbol  # size of one point-to-point message in the alpha-beta model ($n$) [bytes]
    PointToPointTransferTime: NotationSymbol  # time to send one point-to-point message of n bytes in the alpha-beta model ($T(n)$) [s]
    Policy: NotationSymbol  # learned model viewed as a mapping from context or observation to actions, with parameters theta ($\pi_\theta$)
    Pose: NotationSymbol  # rigid-body pose of a named frame, its rotation and position together ($\mathbf{T}$)
    PositionErrorCeiling: NotationSymbol  # largest position error a target's own motion may add before a measurement of it is too stale to act on ($e_{\max}$) [m]
    PositionalTaskTolerance: NotationSymbol  # radius within which a target estimate must lie for the task to succeed ($r_{\text{tol}}$) [m]
    PowerRailInductance: NotationSymbol  # lumped inductance of the supply loop feeding the inverter ($L_{\text{bus}}$) [H]
    PowerRailResistance: NotationSymbol  # lumped resistance of the supply path feeding the inverter: battery cells, harness and switch ($R_{\text{bus}}$) [Ohm]
    PreAdmissionAction: NotationSymbol  # action as proposed, before the runtime checks it ($a_{\text{prop}}$)
    PrebrakeDelay: NotationSymbol  # time from the capture of the frame that first shows a hazard to the onset of retarding force ($\tau_{\text{delay}}$) [s]
    PredictedOutput: NotationSymbol  # model output for one example, such as its vector of class probabilities ($\hat{\mathbf{y}}$)
    ProcessNoiseVector: NotationSymbol  # random disturbance acting on the state in the process model ($\mathbf{w}(t)$)
    ProfileAccelerationBound: NotationSymbol  # largest acceleration or deceleration a planned or interpolated profile may command ($a_{\max}$) [m/s^2]
    ProgramGoodput: NotationSymbol  # ratio of ideal to actual program execution time during productive work ($\eta_{\text{program}}$) [dimensionless]
    PromptTokensSent: NotationSymbol  # cumulative number of prompt tokens a trajectory sends across all its calls ($S_{\text{sent}}$)
    ProposalRejection: NotationSymbol  # outcome of withholding a proposal at the admission gate, recorded with its reason ($\bot$)
    ProposerInferenceTime: NotationSymbol  # time for one forward evaluation of the learned proposer ($t_{\text{infer}}$) [s]
    ProtectiveClearanceMargin: NotationSymbol  # fixed nonzero clearance kept between the machine envelope and the boundary after it stops ($\delta_{\text{margin}}$) [m]
    PureTransportDelay: NotationSymbol  # fixed delay in a feedback or communication channel ($T_d$) [s]
    QueryWorkspaceGeneration: NotationSymbol  # workspace generation a memory query requires ($g_q$)
    ReasoningBudget: NotationSymbol  # maximum number of reasoning tokens a call may spend before it answers ($B_{\text{reason}}$)
    ReceiverOverhead: NotationSymbol  # processor time spent completing the receipt of one message, during which it cannot compute ($o_{\text{recv}}$) [s]
    ReducedImpactMass: NotationSymbol  # two-body reduced mass of a collision between the robot and a body part ($m_{\text{red}}$) [kg]
    ReferencePosition: NotationSymbol  # commanded Cartesian position the tracked point should follow ($\mathbf{r}$) [m]
    ReflectedRotorInertia: NotationSymbol  # apparent inertia of a motor rotor seen at the joint output, scaled by the square of the gear ratio ($J_{\text{ref}}$) [kg*m^2]
    RegularizationWeight: NotationSymbol  # weight of the regularization penalty in a training objective ($\lambda$) [dimensionless]
    RenewalJitterAllowance: NotationSymbol  # allowance for variation in the interval between successive chunk admissions ($T_{\text{jitter}}$) [s]
    RepairCap: NotationSymbol  # maximum number of forward repair attempts on one failed step before the harness compensates or escalates ($K_{\text{repair}}$)
    ReplicaCount: NotationSymbol  # number of serving replicas among which a router chooses ($R$)
    ReplyLossProbability: NotationSymbol  # probability that a call's reply is lost in transit ($q_{\text{rep}}$) [dimensionless]
    RequestConcurrency: NotationSymbol  # average number of requests in flight inside a stable serving system ($Q_{\text{req}}$) [requests]
    RequestLossProbability: NotationSymbol  # probability that a call's request is lost in transit ($q_{\text{req}}$) [dimensionless]
    RequestTimeInSystem: NotationSymbol  # average time a request spends inside a serving system, waiting plus being served ($T_{\text{lat}}$) [s]
    RequiredComponentCount: NotationSymbol  # number of components a job needs available at the same time ($N$)
    ResidentStopDistance: NotationSymbol  # distance the resident stop travels from speed v to rest once braking begins ($d_{\text{suffix}}(v)$) [m]
    ResidualGate: NotationSymbol  # deterministic switch that admits a learned residual correction only inside the certified contact region ($g$) [dimensionless]
    ResponseThreshold: NotationSymbol  # score threshold at which a monitoring or selective-prediction rule acts, such as alerting, abstaining, or routing to manual review ($\tau$) [dimensionless]
    RestitutionCoefficient: NotationSymbol  # ratio of separation speed to approach speed in an impact ($e$) [dimensionless]
    Reward: NotationSymbol  # per-step reward an engineer writes for policy search ($r$) [dimensionless]
    RigidTransformTranslation: NotationSymbol  # translation part of a rigid transform between two frames ($\mathbf{t}$) [m]
    RingAllReduceCompletionTime: NotationSymbol  # time to complete an AllReduce over a ring of devices ($T_{\text{ring}}$) [s]
    RingTreeCrossoverPayload: NotationSymbol  # collective payload size above which ring AllReduce finishes faster than tree AllReduce ($M_{\text{crossover}}$) [bytes]
    RollingShutterDistortion: NotationSymbol  # apparent displacement of an image row caused by platform motion during rolling-shutter readout ($\Delta \mathbf{x}(y)$) [m]
    RollingShutterRowReadout: NotationSymbol  # time between the exposure of adjacent rows of a rolling-shutter sensor ($t_{\text{row}}$) [s]
    RolloutEnvironmentState: NotationSymbol  # full environment state at a turn of a reinforcement-learning rollout, where turns are indexed by k ($s_k$)
    RooflineRidgePoint: NotationSymbol  # arithmetic intensity at which a device's memory-bandwidth limit meets its compute limit ($I_{\text{ridge}}$) [FLOP/byte]
    RotorInertia: NotationSymbol  # moment of inertia of a motor rotor about its axis ($J_{\text{rotor}}$) [kg*m^2]
    RuntimeGoodput: NotationSymbol  # fraction of allocated resource time that becomes checkpoint-preserved productive progress ($\eta_{\text{runtime}}$) [dimensionless]
    RuntimeTime: NotationSymbol  # time a turn spends in the runtime's own work, such as assembling context, validating proposals, and running checks ($T_{\text{runtime}}$) [s]
    SafetyBoundaryManifold: NotationSymbol  # boundary of the safe set, the states where the barrier certificate is exactly zero ($\partial \mathcal{C}$)
    SampleStorageSize: NotationSymbol  # bytes needed to store one training or serving sample ($s_{\text{sample}}$) [bytes/sample]
    SamplingPeriod: NotationSymbol  # time between successive executions of a discrete controller ($T_s$) [s]
    SamplingRate: NotationSymbol  # number of controller executions or signal samples per second ($f_s$) [Hz]
    ScaleCoefficient: NotationSymbol  # coefficient of the reducible power-law term of an empirical scaling law ($A_{\mathcal{L}}$) [dimensionless]
    ScaledResource: NotationSymbol  # resource quantity a scaling law varies, such as parameters, training tokens or compute ($X$) [parameters, tokens or FLOP]
    ScalingEfficiency: NotationSymbol  # measured speedup on N devices divided by the ideal linear speedup N ($\eta_{\text{scaling}}$) [dimensionless]
    ScalingExponent: NotationSymbol  # power-law exponent at which loss falls as a training resource grows ($\alpha_{\text{scale}}$) [dimensionless]
    SchedulingGoodput: NotationSymbol  # fraction of fleet capacity-time during which a job holds all the resources it needs for synchronous progress ($\eta_{\text{scheduling}}$) [dimensionless]
    SearchDepth: NotationSymbol  # maximum depth a tree search expands ($d_{\max}$)
    SenderOverhead: NotationSymbol  # processor time spent initiating one message, during which it cannot compute ($o_{\text{send}}$) [s]
    SensorIntegrationTime: NotationSymbol  # sensor-side latency: half the exposure, analog-to-digital conversion and readout ($t_{\text{sense}}$) [s]
    SensorLeverArm: NotationSymbol  # displacement from the body rotation origin to a sensor mounted away from it ($\mathbf{r}_{P/O}$) [m]
    SeqlockRetryProbability: NotationSymbol  # probability that a reader of a sequence-locked buffer sees a torn update and must retry ($p_{\text{retry}}$) [dimensionless]
    SeqlockVulnerableInterval: NotationSymbol  # window during which a concurrent write can tear a sequence-locked read ($t_{\text{vulnerable}}$) [s]
    SequenceLength: NotationSymbol  # number of tokens in one input sequence ($S$)
    SequentialLogLikelihoodRatio: NotationSymbol  # cumulative log-likelihood ratio of the alternative to the null hypothesis after m trials of a sequential test ($\Lambda_m$) [dimensionless]
    ServiceRate: NotationSymbol  # rate at which one server, such as a worker, consumer, or model endpoint, completes items or calls ($\mu$) [items/s]
    ServingUtilization: NotationSymbol  # fraction of time a server is busy, offered load divided by service capacity ($\rho_{\text{serv}}$) [dimensionless]
    ShieldInterceptionCoverage: NotationSymbol  # verified probability that the shield intercepts a hazardous proposal of class h ($c_h$) [dimensionless]
    SignificanceLevel: NotationSymbol  # probability that a hypothesis test rejects a true null hypothesis ($\alpha$) [dimensionless]
    SingleDeviceRunTime: NotationSymbol  # wall-clock time for a fixed amount of productive work on one device ($T_1$) [s]
    SingleStepImitationError: NotationSymbol  # bound on the probability that the imitation policy makes an error at a single step ($\epsilon$) [dimensionless]
    SpecialEuclideanGroup: NotationSymbol  # group of rigid-body transformations in three dimensions, rotations combined with translations ($SE(3)$)
    SpecialOrthogonalGroup: NotationSymbol  # group of three-by-three rotation matrices, orthogonal with determinant one ($SO(3)$)
    SpeedAtDetection: NotationSymbol  # speed of the machine at the moment the hazard is detected ($v$) [m/s]
    SpeedBound: NotationSymbol  # largest speed the machine, a tracked object or an approaching body is credited with in a budget ($v_{\max}$) [m/s]
    SprtLowerBoundary: NotationSymbol  # lower stopping threshold of a sequential probability ratio test, below which the test accepts the null hypothesis ($\Lambda_{\text{lo}}$) [dimensionless]
    SprtUpperBoundary: NotationSymbol  # upper stopping threshold of a sequential probability ratio test, above which the test accepts the alternative ($\Lambda_{\text{up}}$) [dimensionless]
    StaleSupportRate: NotationSymbol  # share of decisions requiring current evidence that relied on superseded support ($\epsilon_{\text{stale}}$) [dimensionless]
    StateAtTurn: NotationSymbol  # context state the model saw on one turn ($s_t$)
    StateDistribution: NotationSymbol  # average distribution of states a policy visits over the horizon ($d_\pi$)
    StateExposureLevel: NotationSymbol  # state exposure level, what a task carries between turns, from nothing to durable shared state ($S_0\text{ to }S_3$) [level]
    StateSpaceInputMatrix: NotationSymbol  # input matrix of a linear state-space model, mapping control inputs to state rates ($\mathbf{B}$)
    StateSpaceSystemMatrix: NotationSymbol  # system matrix of a linear state-space model, governing the unforced state dynamics ($\mathbf{A}_{\text{sys}}$)
    StateTransitionMatrix: NotationSymbol  # discrete-time matrix that propagates the state one step forward in a Kalman filter ($\mathbf{F}$)
    StatisticalDivergence: NotationSymbol  # divergence of the current data distribution from the training distribution, a nonnegative measure of drift ($\mathcal{D}(P_t \| P_0)$) [dimensionless]
    StatisticalSignificanceLevel: NotationSymbol  # allowed probability of a type I error, rejecting a true null hypothesis ($\alpha_{\text{sig}}$) [dimensionless]
    StatorThermalCapacitance: NotationSymbol  # heat capacity of the motor stator and winding, energy per degree of temperature rise ($C_{\text{th}}$) [J/K]
    StoppingDistance: NotationSymbol  # distance the machine travels from hazard detection to rest, including travel during the pre-brake delay ($d_{\text{stop}}$) [m]
    StoppingDuration: NotationSymbol  # time from braking onset to standstill ($T_{\text{stop}}$) [s]
    SustainedMemoryBandwidth: NotationSymbol  # memory bandwidth achievable in practice under the target access pattern ($\text{BW}_{\text{mem}}$) [bytes/s]
    SynchronizationBlackout: NotationSymbol  # interval during which a node receives no synchronization message and runs on its own clock ($T_{\text{blackout}}$) [s]
    SynchronizationTime: NotationSymbol  # time per step lost to barriers, straggler waiting and other coordination at scale N ($T_{\text{sync}}(N)$) [s]
    SystemHazardRate: NotationSymbol  # rate of hazardous events of class h for the whole system, after the shield ($\lambda_{\text{system}, h}$) [1/h]
    SystemMtbf: NotationSymbol  # mean time between failures that interrupt the whole job or system ($\text{MTBF}_{\text{system}}$) [hours]
    SystemReliability: NotationSymbol  # probability that the whole system runs without failure up to time t ($R_{\text{system}}(t)$) [dimensionless]
    TakeoverResponseTime: NotationSymbol  # time a human operator needs to take over control after a handover request ($T_{\Sigma, 1}$) [s]
    TangentialContactShear: NotationSymbol  # friction component of a contact force, tangential to the contact plane ($\mathbf{f}_{k, \parallel}$) [N]
    TargetHazardRate: NotationSymbol  # tolerable dangerous failure rate assigned to a safety function by risk assessment ($\lambda_{\text{target}}$) [1/h]
    TargetTravelDistance: NotationSymbol  # straight-line distance from the measured end-effector position to the admitted target ($d$) [m]
    TargetTrialSuccessProbability: NotationSymbol  # per-trial success probability a test campaign must demonstrate ($p_{\text{target}}$) [dimensionless]
    TaskContract: NotationSymbol  # five-part specification of a delegated task: goal, environment, permitted actions, available observations, and completion criteria ($\mathcal{C}$)
    TaskContractGoal: NotationSymbol  # desired end state of a task, with explicit negative scope, as the first element of the task contract ($G$)
    TaskCost: NotationSymbol  # total monetary cost of all attempts and branches for one initiated task ($C_{\text{task}}$) [USD]
    TaskEnvironment: NotationSymbol  # specification of the reproducible place the work happens, such as base commit, image, configuration, and network rules ($\mathcal{E}_{\text{env}}$)
    TasksPerComparisonArm: NotationSymbol  # number of tasks per arm needed for a two-proportion release comparison ($N_{\text{arm}}$)
    TelemetryFreshnessBound: NotationSymbol  # largest age a measurement may reach before acting on it introduces intolerable error ($\Delta t_{\text{fresh}}$) [s]
    TemporalEnsemblingWeight: NotationSymbol  # weight given to the prediction of a past chunk at offset i when blending overlapping chunks ($w_i$) [dimensionless]
    TensorParallelDegree: NotationSymbol  # number of devices that shard each layer's tensors in tensor parallelism ($N_{\text{TP}}$)
    ThermalResistance: NotationSymbol  # lumped thermal resistance from the stator windings to ambient, temperature rise per watt ($R_{\text{th}}$) [K/W]
    ThermalTimeConstant: NotationSymbol  # characteristic time for a motor winding to heat or cool, thermal resistance times thermal capacitance ($\tau_{\text{th}}$) [s]
    Time: NotationSymbol  # total wall-clock time to complete a unit of work, from start to finish ($T$) [s]
    TimingJitter: NotationSymbol  # standard deviation of a nominally periodic event's timing about its schedule ($\sigma_{\text{jitter}}$) [s]
    TokenIndex: NotationSymbol  # position of a sampled policy token within a rollout ($t$)
    TokenLossMask: NotationSymbol  # indicator that a token position is a training target in masked fine-tuning ($m_t$) [dimensionless]
    TokenVocabulary: NotationSymbol  # set of tokens a model call samples from ($\mathcal{V}$)
    ToleratedFailureProbability: NotationSymbol  # largest probability with which a deployed model may miss its error tolerance, so a guarantee holds with confidence one minus it ($\epsilon$) [dimensionless]
    ToolTime: NotationSymbol  # time a tool takes to execute, such as a build or a test run ($T_{\text{tool}}$) [s]
    TorqueProducingCurrent: NotationSymbol  # quadrature-axis current of a field-oriented drive, the current component that produces torque ($i_q$) [A]
    TotalEnergy: NotationSymbol  # total energy a workload consumes ($E_{\text{total}}$) [J]
    TotalMemoryFootprint: NotationSymbol  # total memory a workload occupies on a device ($M_{\text{total}}$) [bytes]
    TraceDecay: NotationSymbol  # generalized advantage estimator's decay factor between zero and one ($\lambda$) [dimensionless]
    TrackingErrorBound: NotationSymbol  # bound on the distance between the commanded and the executed position of the tracked point ($\epsilon_{\text{track}}$) [m]
    TrainingCompute: NotationSymbol  # total floating-point operations needed to train a model ($O_{\text{train}}$) [FLOP]
    TrainingCost: NotationSymbol  # one-time cost of training and evaluating a model ($C_{\text{train}}$) [USD]
    TrainingDistribution: NotationSymbol  # data distribution the model was trained on, the baseline that drift is measured against ($P_0$)
    Trajectory: NotationSymbol  # ordered record of the states, actions, and observations of one delegated task ($\tau$)
    TrajectoryAdmissionRate: NotationSymbol  # rate at which trajectories are admitted, or arrive at a stage, per second ($\lambda_{\text{traj}}$) [trajectories/s]
    TrajectoryBudget: NotationSymbol  # ceilings on turns, tokens, wall-clock time, and dollars that the harness checks before every call ($\mathbf{B}$)
    TrajectoryDivergenceResidual: NotationSymbol  # signed error between measured and planned kinematics used to detect a diverging rollout ($e_{\text{div}}$) [m]
    TrajectoryDuration: NotationSymbol  # elapsed time of one trajectory between its declared endpoints ($T_{\text{traj}}$) [s]
    TrajectoryGoodput: NotationSymbol  # distinct accepted task completions per unit of elapsed time under the declared contract ($\mathcal{G}$) [tasks/s]
    TrajectoryReward: NotationSymbol  # reward a verifier assigns to a whole trajectory ($R(\tau)$) [dimensionless]
    TransferTimeForPayload: NotationSymbol  # time to communicate a payload of x bytes under the relevant link or collective model ($T_{\text{transfer}}(x)$) [s]
    TreeAllReduceCompletionTime: NotationSymbol  # time to complete an AllReduce over a binary tree of devices ($T_{\text{tree}}$) [s]
    TrialCount: NotationSymbol  # number of physical trials in a learning or evaluation campaign ($N$)
    TrialFailureProbability: NotationSymbol  # probability that one physical trial fails ($p_{\text{fail}}$) [dimensionless]
    TrichotomyMonitorState: NotationSymbol  # set of three verdicts a runtime monitor can return: known true, known false, or unknown ($\mathcal{T}_{\text{trichotomy}}$)
    TruncatedSignedDistance: NotationSymbol  # signed distance from a voxel to the nearest surface, divided by the truncation band and clamped to plus or minus one ($\text{tsdf}(\mathbf{p}_{\text{vox}})$) [dimensionless]
    TsdfAccumulatedDistance: NotationSymbol  # running weighted average of the truncated signed distance at a voxel after t frames ($D_t(\mathbf{p}_{\text{vox}})$) [dimensionless]
    TsdfAccumulatedWeight: NotationSymbol  # accumulated observation weight at a voxel after t frames ($W_t(\mathbf{p}_{\text{vox}})$) [dimensionless]
    TsdfTruncationBand: NotationSymbol  # distance from a surface within which signed distances are stored rather than clamped ($\mu_{\text{tsdf}}$) [m]
    TurnAppendLength: NotationSymbol  # number of tokens one turn adds to the next prompt ($\Delta S_{\text{turn}}$)
    TurnContext: NotationSymbol  # tokens the runtime assembles and sends to the model on one turn ($c_t$)
    TurnHorizon: NotationSymbol  # number of sequential turns an agent trajectory runs ($H$)
    TurnIndex: NotationSymbol  # position of a turn within a trajectory, counted from zero ($t$)
    UnbrakedImpactForce: NotationSymbol  # peak transient contact force when the robot strikes a body at speed without braking ($F_{\text{unbraked}}$) [N]
    UnitQuaternion: NotationSymbol  # four-component unit-norm vector representing an orientation ($\mathbf{q}_{\text{quat}}$) [dimensionless]
    UsefulComputeFraction: NotationSymbol  # share of the distributed step time spent on useful local arithmetic ($f_{\text{compute}}$) [dimensionless]
    UsefulModelFlops: NotationSymbol  # floating-point operations the model's work actually requires, excluding recomputation and padding ($O_{\text{useful}}$) [FLOP]
    ValidCandidateRate: NotationSymbol  # share of generated candidates that meet the specification, before any check ($p$) [dimensionless]
    Velocity: NotationSymbol  # linear velocity of a point or body ($\mathbf{v}$) [m/s]
    VerificationEvidence: NotationSymbol  # result of a check the runtime runs on the resulting state, as pass or fail or a score ($v_t$) [dimensionless]
    VerifiedTrajectorySet: NotationSymbol  # set of dispatched trajectories whose results passed verification ($\mathcal{T}_{\text{success}}$)
    VerifierFalseAcceptanceRate: NotationSymbol  # probability that a verifier passes a candidate that does not meet the specification ($\epsilon_{\text{FA}}$) [dimensionless]
    VerifierFalseRejectionRate: NotationSymbol  # probability that a verifier fails a candidate that meets the specification ($\epsilon_{\text{FR}}$) [dimensionless]
    VirtualDampingMatrix: NotationSymbol  # damping matrix an impedance controller emulates, mapping joint velocity to opposing torque ($\mathbf{D}_{\text{virt}}$)
    VirtualPipelineStagesPerDevice: NotationSymbol  # number of interleaved virtual pipeline stages assigned to each device ($V$)
    VirtualStiffnessMatrix: NotationSymbol  # stiffness matrix an impedance controller emulates, mapping displacement to restoring torque ($\mathbf{K}_{\text{virt}}$)
    VitPatchProjectionWeights: NotationSymbol  # learned linear map from a flattened image patch to a token embedding in a vision transformer ($\mathbf{W}_{\text{patch}}$)
    VitPositionEmbedding: NotationSymbol  # embedding added to each patch token that encodes its row and column in the image ($\mathbf{E}_{\text{pos}}$)
    WeightMemory: NotationSymbol  # memory holding a model's weights ($M_{\text{weights}}$) [bytes]
    WheelRadius: NotationSymbol  # rolling radius of a driven wheel ($r$) [m]
    WindingResistance: NotationSymbol  # electrical resistance of the stator winding at its current temperature ($R(T_{\text{wind}})$) [Ohm]
    WindingTemperature: NotationSymbol  # temperature of the motor winding copper ($T_{\text{winding}}$) [degC]
    WorkerCount: NotationSymbol  # number of equally capable workers, agents, threads, or verifier slots in one pool ($N_{\text{workers}}$)
    WorldReferenceFrame: NotationSymbol  # fixed coordinate frame attached to the world ($\mathcal{F}_W$)
    YoshikawaManipulability: NotationSymbol  # measure of distance from kinematic singularity, the square root of the determinant of J times J transpose ($w(\mathbf{q})$) [dimensionless]
    ZeroFailureTrialCount: NotationSymbol  # number of consecutive failure-free trials needed to demonstrate a target success probability ($n$)
    ZeroMomentPoint: NotationSymbol  # point on the ground where the net horizontal tipping moment of the contact forces vanishes ($\mathbf{p}_{\text{ZMP}}$) [m]
    ZohTransportDelay: NotationSymbol  # effective continuous-time delay a zero-order hold adds, half the sampling period ($\tau_{\text{ZOH}}$) [s]

Notation: NotationRegistry
Symbols: NotationRegistry
SYMBOLS: NotationRegistry
