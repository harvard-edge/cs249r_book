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

    AbsoluteConicImageEntry: NotationSymbol  # one entry of the image-of-the-absolute-conic matrix, indexed B_{ij} ($B$) [varies]
    AbsoluteConicImageMatrix: NotationSymbol  # symmetric matrix K^{-T} K^{-1}, the image of the absolute conic, from which camera calibration recovers the intrinsics ($\mathbf{B}$)
    AbsoluteConicVector: NotationSymbol  # six-vector of the distinct entries of the symmetric image-of-the-absolute-conic matrix ($\mathbf{b}$)
    AcceleratedEpisodeTime: NotationSymbol  # elapsed time of a fixed trajectory or episode after only its model-execution share is accelerated ($T'$) [s]
    AcceleratedFractionSpeedup: NotationSymbol  # speedup factor applied to the accelerated fraction of a workload in Amdahl's law, whether it comes from faster hardware or from more workers ($G$) [dimensionless]
    AccelerationNoiseDensityMatrix: NotationSymbol  # continuous power spectral density matrix of a white acceleration noise process ($\mathbf{Q}_c$)
    AccelerationNoiseSpectralDensity: NotationSymbol  # continuous power spectral density of a zero-mean white acceleration noise, per axis ($q_c$) [m^2/s^3]
    AccelerationWhiteNoise: NotationSymbol  # zero-mean continuous white acceleration noise driving a one-dimensional double-integrator model ($w$) [m/s^2]
    AcceleratorBatchTime: NotationSymbol  # time the accelerator takes to run one training step on one batch ($T_{\text{GPU}}$) [s]
    AcceleratorConsumptionRate: NotationSymbol  # batches per second the accelerator can consume, its compute rate divided by the operations per batch ($R_{\text{accel}}$) [batches/s]
    AcceleratorCount: NotationSymbol  # number of accelerators one job runs on ($N_{\text{GPU}}$)
    AcceleratorDataConsumptionRate: NotationSymbol  # rate at which one accelerator consumes training input data while computing at full speed ($R_{\text{consumption}}$) [bytes/s]
    AcceleratorGain: NotationSymbol  # speedup an optimization or accelerator gives the fraction of execution it applies to, in Amdahl's law ($G_{\text{accel}}$) [dimensionless]
    AcceleratorHourlyPrice: NotationSymbol  # price of one accelerator-hour, excluding energy ($R_{\text{GPU/hr}}$) [USD/h]
    AcceleratorMemoryBandwidth: NotationSymbol  # sustained rate at which bytes move between an accelerator's own memory, such as HBM, and its compute units ($\text{BW}_{\text{accelerator}}$) [bytes/s]
    AcceleratorRailVoltage: NotationSymbol  # voltage of a chip's supply rail at the load end of its power delivery network ($V_{\text{rail}}$) [V]
    AcceleratorSecondsPerQuery: NotationSymbol  # allocated accelerator-seconds divided by completed queries over an accounting period ($S_{\text{GPU/query}}$) [s/query]
    AcceleratorSupplyCurrent: NotationSymbol  # electrical current an accelerator draws from its power supply ($i$) [A]
    AcceleratorTargetQueryRate: NotationSymbol  # queries per second one accelerator sustains at the target latency objective, including batching and operating headroom ($R_{\text{GPU,target}}$) [queries/s]
    AcceleratorsPerNode: NotationSymbol  # number of accelerators in one node ($G_{\text{node}}$)
    AccelerometerBias: NotationSymbol  # constant offset of an accelerometer's output, which double integration turns into quadratically growing position error ($b_a$) [m/s^2]
    AcceptanceFraction: NotationSymbol  # share of a settled cohort of initiated tasks that ends accepted ($\alpha$) [dimensionless]
    AcceptanceIntervalLowerLimit: NotationSymbol  # lower limit of a confidence interval for an acceptance probability, such as the Wilson score interval ($L$) [dimensionless]
    AcceptanceIntervalUpperLimit: NotationSymbol  # upper limit of a confidence interval for an acceptance probability, such as the Wilson score interval ($U$) [dimensionless]
    AcceptanceProbability: NotationSymbol  # probability that one attempt at a task is accepted, across a workload ($p_{\text{acc}}$) [dimensionless]
    AcceptedInfluences: NotationSymbol  # set of sources a task policy accepts as influences on one operation ($\text{Inf}$)
    AcceptedInputSet: NotationSymbol  # set of inputs a selective predictor accepts, those whose acceptance score reaches the threshold ($\mathcal{A}_{\tau}$)
    AcceptedLabelCount: NotationSymbol  # number of accepted labels among the observations a sequential test has collected so far, indexed by that observation count ($d$)
    AcceptedOutcomePayoff: NotationSymbol  # monetary value of one accepted outcome, rejected outcomes valued at zero ($V$) [USD]
    AcceptedSampleCount: NotationSymbol  # number of accepted tasks among the independently sampled tasks of a single-run evaluation, the binomial count a confidence interval for acceptance is built from ($X$)
    AcceptedTaskMeanCost: NotationSymbol  # mean task-account cost of the accepted tasks in a settled cohort ($C_{\text{succ}}$) [USD]
    AcceptedTasks: NotationSymbol  # number of distinct tasks in a cohort accepted under the declared contract ($N_{\text{acc}}$)
    AcceptingStateSet: NotationSymbol  # set of accepting states of an output-constraint automaton, those at which a complete permitted string ends ($F$)
    AccessAttemptTime: NotationSymbol  # time at which a proposed access is checked against a capability grant ($t$) [s]
    AccountQueryDistribution: NotationSymbol  # empirical distribution of the queries one API account sends, over a chosen representation of inputs ($p_{\text{user}}$)
    AccumulatedProcessNoiseCovariance: NotationSymbol  # covariance that continuous process noise adds over an elapsed interval, the integral of the transition-propagated noise intensity ($\mathbf{Q}_d$)
    AccumulatedSyncTime: NotationSymbol  # gradient synchronization time paid once for an accumulated update ($T_{\text{sync}}$) [s]
    AccumulatedUpdateTime: NotationSymbol  # time of one optimizer update under gradient accumulation: all micro-batch passes, one synchronization and the step ($T_{\text{effective}}$) [s]
    AccumulationStepCount: NotationSymbol  # number of micro-batches whose gradients are accumulated before one optimizer update ($k$)
    AccuracyAfterDeployment: NotationSymbol  # model accuracy after the model has been in service for time t since deployment or its last retraining ($\text{Accuracy}$) [dimensionless]
    AccuracyDecayRate: NotationSymbol  # fitted rate at which deployed model accuracy decays exponentially with time since deployment or last retraining ($\gamma$) [1/day]
    AccuracyPayoffPerQuery: NotationSymbol  # monetary value of one query per unit change in accuracy fraction ($V$) [USD/query]
    AchievedBrakingDeceleration: NotationSymbol  # deceleration the machine actually achieves while braking, bounded above by tire-floor friction ($a$) [m/s^2]
    AchievedKernelBandwidth: NotationSymbol  # memory bandwidth a kernel actually achieves, the bytes it moves divided by its measured execution time ($\text{BW}_{\text{effective}}$) [bytes/s]
    AcquisitionEpoch: NotationSymbol  # instant a sensor transduced one observation, such as a camera frame's exposure midpoint or an encoder latch ($t_{\text{acq}}$) [s]
    ActVariationalObjective: NotationSymbol  # training objective of Action Chunking with Transformers: chunk reconstruction error plus a weighted KL penalty on the latent style variable ($\mathcal{L}_{\text{ACT}}$)
    ActionBinCount: NotationSymbol  # number of equal-width bins into which each continuous action dimension is discretized for action tokens ($B$)
    ActionChunk: NotationSymbol  # sequence of future actions a policy predicts in one pass, one row per control step ($\mathbf{A}$)
    ActionChunkHorizon: NotationSymbol  # number of future action steps a policy emits in one forward pass ($H$)
    ActionChunkIndex: NotationSymbol  # position of one action chunk in the sequence of chunks a policy generates, counted from the oldest one still in use ($k$)
    ActionDimension: NotationSymbol  # number of components in one action vector ($d_a$)
    ActionExecutionRate: NotationSymbol  # rate at which the controller consumes the waypoints of an admitted action chunk ($f_{\text{exec}}$) [Hz]
    ActionFilterCutoffFrequency: NotationSymbol  # cutoff frequency of the low-pass filter applied to policy actions before they reach the drives ($f_{\text{cutoff}}$) [Hz]
    ActionLowerBound: NotationSymbol  # componentwise lower limit of the valid action range a policy's actions are clipped to ($\mathbf{a}_{\min}$)
    ActionMagnitudeBound: NotationSymbol  # largest magnitude any component of a physical action may take, the symmetric actuator limit ($a_{\max}$) [varies]
    ActionRangeMaximum: NotationSymbol  # upper end of the declared range of one continuous action dimension, above which actions are clipped before binning ($a_{\max}$) [varies]
    ActionRangeMinimum: NotationSymbol  # lower end of the declared range of one continuous action dimension, below which actions are clipped before binning ($a_{\min}$) [varies]
    ActionRecoverableFraction: NotationSymbol  # fraction of an action's consequence that compensation can restore if the action proves bad ($\rho_{a}$) [dimensionless]
    ActionResidualVector: NotationSymbol  # vector of differences between a predicted and a demonstrated action in normalized action coordinates ($\mathbf{r}$)
    ActionSpace: NotationSymbol  # set of actions a policy may emit, the domain its action distribution is defined over ($\mathcal{A}$)
    ActionTokenPrefix: NotationSymbol  # action tokens of one response that precede the current action-token position ($a_{<j}$)
    ActionUpperBound: NotationSymbol  # componentwise upper limit of the valid action range a policy's actions are clipped to ($\mathbf{a}_{\max}$)
    ActivationCheckpointCount: NotationSymbol  # number of layers whose activations gradient checkpointing keeps ($n_{\text{ckpt}}$)
    ActivationFunction: NotationSymbol  # activation function of a neuron or layer, applied element-wise to the preactivation ($f$)
    ActivationInput: NotationSymbol  # scalar argument an activation function is applied to in its defining formula ($x$) [dimensionless]
    ActivationLipschitzBound: NotationSymbol  # bound on the absolute slope of an activation function over the input domain, its Lipschitz constant ($L_{\phi}$) [dimensionless]
    ActivationMatrix: NotationSymbol  # activation matrix of one layer for a batch, one example per row ($\mathbf{A}$)
    ActivationMemory: NotationSymbol  # memory holding the intermediate activations a model keeps for the backward pass or between serving stages ($M_{\text{activations}}$) [bytes]
    ActivationPayloadSize: NotationSymbol  # bytes of one activation tensor, batch by sequence by hidden width, that each tensor-parallel AllReduce carries ($M_{\text{act}}$) [bytes]
    ActivationStagingTime: NotationSymbol  # time to stage a validated plan for activation in the execution queue ($t_{\text{activate}}$) [s]
    ActivationVector: NotationSymbol  # activation vector of one layer for one example ($\mathbf{a}$)
    ActiveContactIndex: NotationSymbol  # position of one active contact among the robot's contacts ($k$)
    ActiveControlTorque: NotationSymbol  # active joint torque a contact controller applies, excluding its virtual damping term ($\mathbf{u}_{\text{active}}$) [N*m]
    ActiveInferencePower: NotationSymbol  # average power an accelerator draws while actively running inference ($P_{\text{active}}$) [W]
    ActiveLeaseCount: NotationSymbol  # mean number of sandbox leases held at once, by Little's law ($N_{\text{active}}$)
    ActiveMotionDuration: NotationSymbol  # part of a scheduled collection session left for attempts after startup, calibration and operator rest ($T_{\text{motion}}$) [s]
    ActiveParameterCount: NotationSymbol  # number of model parameters each token's forward pass uses, all parameters for a dense model and the active ones for a mixture of experts ($P_{\text{model}}$)
    ActivePrefillRequestSet: NotationSymbol  # set of waiting prefill requests that receive a prompt chunk in the current iteration ($\mathcal{P}_{\text{active}}$)
    ActiveSchemaSet: NotationSymbol  # set of tool schemas declared to the model when one scored call was proposed ($\mathcal{S}$)
    ActiveTargetTokenCount: NotationSymbol  # number of token positions in one example that the loss mask selects as training targets ($N_{\text{active}}$)
    ActorAccelerationBound: NotationSymbol  # validated bound on the acceleration of one tracked actor ($a_{k,\max}$) [m/s^2]
    ActorBodyRadius: NotationSymbol  # declared radius of the envelope covering a tracked person, limbs and carried objects included ($r$) [m]
    ActorUpdateDelayCount: NotationSymbol  # number of critic updates between successive actor and target-network updates in delayed policy updates ($d$)
    ActuatedJointAccelerations: NotationSymbol  # accelerations of the actuated joints of a floating-base robot, the joint block of its generalized accelerations ($\ddot{\mathbf{q}}_j$)
    ActuatedJointBiasForces: NotationSymbol  # actuated-joint block of the generalized bias forces of a floating-base robot ($\mathbf{h}_j$)
    ActuatedJointCoordinates: NotationSymbol  # actuated joint angles of a floating-base robot, the joint block of its generalized coordinates ($\mathbf{q}_j$)
    ActuationSelectionBaseBlock: NotationSymbol  # block of the actuation selection matrix belonging to the six floating-base coordinates, identically zero because the base is unactuated ($\mathbf{S}_b$) [dimensionless]
    ActuationSelectionMatrix: NotationSymbol  # matrix that maps actuator torques onto the actuated joints of a floating-base robot, leaving the six base coordinates unactuated ($\mathbf{S}$) [dimensionless]
    ActuatorAxisCount: NotationSymbol  # number of actuator axes, each with its own drive, sharing one power bus or fieldbus ($N_{\text{axes}}$)
    ActuatorBiasEffort: NotationSymbol  # steady force or torque from gravity, friction or payload that an actuator must overcome before it can brake ($u_{\text{bias}}$) [varies]
    ActuatorCyclePeriod: NotationSymbol  # period of one burst-and-recovery cycle, on-time plus off-time ($t_{\text{cycle}}$) [s]
    ActuatorDutyCycle: NotationSymbol  # fraction of each burst cycle an actuator spends at peak power, on-time divided by cycle period ($D_{\text{cycle}}$) [dimensionless]
    ActuatorEffort: NotationSymbol  # force or torque one actuator applies, the scalar control input of a single axis ($u$) [varies]
    ActuatorOffTime: NotationSymbol  # recovery duration between actuator bursts in one duty cycle ($t_{\text{off}}$) [s]
    ActuatorOnTime: NotationSymbol  # duration an actuator runs at burst power in one duty cycle ($t_{\text{on}}$) [s]
    ActuatorResponseTime: NotationSymbol  # time from a delivered stop command until the actuator produces retarding force, including gate drive and current rise ($T_{\text{act}}$) [s]
    ActuatorTorqueLowerBound: NotationSymbol  # per-joint lower bounds of the verified actuator torque envelope ($\mathbf{u}_{\min}$) [N*m]
    ActuatorTorqueUpperBound: NotationSymbol  # per-joint upper bounds of the verified actuator torque envelope ($\mathbf{u}_{\max}$) [N*m]
    AdamFirstMoment: NotationSymbol  # per-parameter exponential moving average of gradients that Adam keeps, its first moment ($m$) [varies]
    AdamMomentDecayRate: NotationSymbol  # exponential decay factor of an Adam moment estimate, between zero and one ($\beta$) [dimensionless]
    AdamSecondMoment: NotationSymbol  # per-parameter exponential moving average of squared gradients that Adam keeps, its second moment ($v$) [varies]
    AdaptationEnergyAllowance: NotationSymbol  # energy a device's admission policy lets background adaptation spend before the next charging opportunity ($E_{\text{slack}}$) [J]
    AdaptationInvestmentCost: NotationSymbol  # incremental one-time cost of an intervention such as an adaptation, covering data curation, training compute and offline evaluation ($C_{\text{invest}}$) [USD]
    AdaptationLatencyIncrease: NotationSymbol  # worst-case increase in foreground latency when an adaptation update runs concurrently ($\Delta L_{\text{lat,fg}}^{\text{wc}}$) [s]
    AdaptationModule: NotationSymbol  # learned adaptation module with parameters psi that predicts the environment latent vector from a window of recent onboard observations ($g_{\psi}$)
    AdaptationPeakMemory: NotationSymbol  # peak working memory an on-device adaptation update needs: resident weights, trainable state, and the largest sum of saved activations and operator workspace over its execution ($M_{\text{peak}}$) [bytes]
    AdaptationPowerIncrement: NotationSymbol  # additional sustained power a running adaptation update adds to the device's draw ($P_{\text{update}}$) [W]
    AdaptationWindowEnergy: NotationSymbol  # predicted energy one on-device adaptation window consumes ($E_{\text{update}}$) [J]
    AdaptedHiddenActivation: NotationSymbol  # hidden representation after a residual adapter's output is added to it ($\mathbf{h}'$)
    AdaptedTaskCost: NotationSymbol  # expected operating cost per assigned task under the adapted configuration, including failed attempts, retries and review ($C_{\text{task, adapt}}$) [USD/task]
    AdapterActivationFunction: NotationSymbol  # element-wise nonlinearity between a residual adapter's down-projection and up-projection ($\sigma$)
    AdapterDownProjectionMatrix: NotationSymbol  # first low-rank adapter matrix, which projects a layer's input down to the adapter rank, the LoRA down-projection ($\mathbf{A}$)
    AdapterIndex: NotationSymbol  # identifier of one low-rank adapter among those a multi-tenant server holds ($k$)
    AdapterInitializationStandardDeviation: NotationSymbol  # standard deviation of the Gaussian from which a low-rank adapter's down-projection is initialized ($\sigma$) [dimensionless]
    AdapterRank: NotationSymbol  # rank of a low-rank adapter update, equivalently the bottleneck width of a residual adapter, which sets its trainable parameter count ($r$)
    AdapterScalingFactor: NotationSymbol  # fixed scaling factor a low-rank adapter's output is multiplied by, divided by the adapter rank ($\gamma$) [dimensionless]
    AdapterUpProjectionMatrix: NotationSymbol  # second low-rank adapter matrix, which projects the rank-sized intermediate up to the layer's output width, the LoRA up-projection ($\mathbf{B}$)
    AdjacentNeighboringDataset: NotationSymbol  # neighboring copy of a differential privacy input dataset that differs from it in exactly one record ($X'$)
    AdjacentQueryResult: NotationSymbol  # value a query returns on the adjacent dataset of a neighboring pair, before any noise is added ($v'$) [varies]
    AdmissibleBacklogCount: NotationSymbol  # number of transaction backlogs that can queue ahead of a safety read before the read exceeds its time ceiling ($n$)
    AdmissibleControlSet: NotationSymbol  # set of control inputs the actuators can physically deliver, bounded by torque, voltage, traction and deceleration limits ($\mathcal{U}_{\text{adm}}$)
    AdmissibleDisturbanceSet: NotationSymbol  # set of disturbance values a falsification search may inject ($\mathcal{W}$)
    AdmissionCheckCount: NotationSymbol  # number of admission predicates an admission route evaluates in sequence ($n_{\text{chk}}$)
    AdmissionCheckIndex: NotationSymbol  # position of one admission predicate in the execution order of an admission route ($i$)
    AdmissionDecisionEvent: NotationSymbol  # event at which the runtime admits a proposed external effect, recording its authorization and budget reservation decision ($e_{\text{auth}}$)
    AdmissionInclusivePrebrakeDelay: NotationSymbol  # pre-brake delay plus the dispatch-to-admission interval, the complete delay from capture to brake onset ($T_{\text{true}}$) [s]
    AdmissionRouteYield: NotationSymbol  # probability that a candidate record passes every admission predicate of its route, the product of the conditional pass probabilities ($Y$) [dimensionless]
    AdmittedAttemptCap: NotationSymbol  # largest number of admitted attempts a gate allows, a safety property that a finite history can violate ($N$)
    AdmittedCommand: NotationSymbol  # command the permission path has admitted for one step and alone may write to the drives ($u$)
    AdmittedContactForce: NotationSymbol  # contact force admitted for one control interval, the smallest of the proposed force, the energy-limited force and the force cap ($F^*$) [N]
    AdmittedControlCommand: NotationSymbol  # control command the permission path applies to the actuators after admitting, clipping or replacing a policy proposal ($\mathbf{u}$)
    AdmittedSpeedCommand: NotationSymbol  # speed command the speed governor admits, the candidate speed clamped to the stopping-envelope ceiling ($v^*$) [m/s]
    AdmittedTargetPosition: NotationSymbol  # target position an intent lease admitted, against which later target evidence is compared ($\hat{\mathbf{p}}_{\text{target}}$) [m]
    AdmittedWorkUnitCount: NotationSymbol  # number of units of work, such as generation rounds, a search has admitted so far ($m$)
    AdvantageStabilizer: NotationSymbol  # small positive constant added to the group reward standard deviation in the group-relative advantage so the coefficient stays defined when every reward in a group ties ($\epsilon_{\text{stab}}$) [dimensionless]
    AdversarialDisturbance: NotationSymbol  # disturbance input chosen adversarially in a differential game, standing for unmodeled friction, gusts, obstacles or model error ($\mathbf{d}$)
    AdversarialExample: NotationSymbol  # adversarial example, an input an attacker forms by adding a bounded perturbation to a clean input so the model mispredicts ($x_{\text{adv}}$)
    AdversarialInputLength: NotationSymbol  # number of characters in an adversarial input to a pattern matcher ($n$)
    AdversarialPenaltyWeight: NotationSymbol  # weight of the adversary loss relative to the task loss in an adversarial debiasing objective ($\lambda_{\text{adv}}$) [dimensionless]
    AdversarialPerturbation: NotationSymbol  # perturbation an adversary adds to an input to change a model's prediction ($\boldsymbol{\delta}_{\text{adv}}$)
    AdversarialPerturbationBudget: NotationSymbol  # largest norm an adversary may give the perturbation it adds to an input ($\epsilon_{\text{adv}}$) [dimensionless]
    AdversaryLoss: NotationSymbol  # loss an adversary head minimizes when predicting a protected attribute from a model's latent representation ($\mathcal{L}_{\text{adv}}$)
    AffordanceFrame: NotationSymbol  # coordinate frame rigidly attached to one interaction site of an object, such as a mug handle ($\mathcal{F}_{\text{affordance}}$)
    AffordanceSuccessEstimate: NotationSymbol  # learned estimate of the probability that a skill succeeds if executed from the current state, its affordance score ($V_{\text{policy}}$) [dimensionless]
    AgeBlindTravel: NotationSymbol  # distance the base travels during the observation age, before the observation is dispatched ($\Delta x_{\text{blind}}$) [m]
    AgeCheckInstant: NotationSymbol  # current instant on the permission path's clock at which the age of evidence is checked ($t_{\text{curr}}$) [s]
    AggregateSlotWork: NotationSymbol  # total slot time a task-precedence graph needs, the sum of its vertex durations ($W$) [s]
    AggregatedRecoveryDataset: NotationSymbol  # dataset that dataset aggregation grows each round with states the learner visited, each labeled with the expert's recovery action ($\mathcal{D}$)
    AggregatedSurrogateLoss: NotationSymbol  # best-in-class surrogate imitation loss on the states the learner visited, aggregated over the rounds so far ($\epsilon_N$) [dimensionless]
    AggregationLearnedPolicy: NotationSymbol  # policy trained in a round of corrective dataset aggregation, one of the sequence the bound selects from ($\hat{\pi}$)
    AggregationLearnerStateDistribution: NotationSymbol  # distribution of states the policy learned in one round of dataset aggregation visits when it executes ($d_{\hat{\pi}_i}$)
    AggregationRoundCount: NotationSymbol  # number of rounds of corrective dataset aggregation (DAgger iterations) run so far ($N$)
    AggregationRoundIndex: NotationSymbol  # position of one round of dataset aggregation, counted from one ($i$)
    AggregationRoundTrajectorySet: NotationSymbol  # set of states the learner's policy visits in the trajectories collected during one round of dataset aggregation ($\mathcal{T}$)
    AiStandaloneSensitivity: NotationSymbol  # fraction of actual positives, such as tumors, that the AI assistant detects on its own ($S_{\text{AI}}$) [dimensionless]
    AisleTravelSpeed: NotationSymbol  # speed the mobile base is set to travel in a warehouse aisle, chosen below the derived dry-floor speed ceiling ($v_{\text{aisle}}$) [m/s]
    AisleWidth: NotationSymbol  # clear width of the warehouse aisle the machine travels ($w$) [m]
    AlertCheckCount: NotationSymbol  # number of independent monitoring checks evaluated, such as models times metrics ($N_{\text{tests}}$)
    AlertFalsePositiveRate: NotationSymbol  # probability that one monitoring check fires when nothing is wrong ($\alpha_{\text{fp}}$) [dimensionless]
    AlgorithmTaxonomyAxis: NotationSymbol  # Algorithm axis of the D·A·M taxonomy, viewed as the set of techniques and concepts it covers so that axis intersections name zones of the taxonomy ($\mathsf{Algorithm}$)
    AliasedVibrationFrequency: NotationSymbol  # apparent frequency at which vibration above a sensor's sampled bandwidth appears after sampling ($f_{\text{alias}}$) [Hz]
    AliasingHarmonicIndex: NotationSymbol  # integer multiple of the sampling rate nearest a vibration frequency, which folds it into the observed band ($k$)
    AllGatherCompletionTime: NotationSymbol  # time to complete an AllGather over a ring of devices, each starting with one shard and ending with the full tensor ($T_{\text{allgather}}$) [s]
    AllOnesVector: NotationSymbol  # vector whose entries are all one, used to sum over a dimension ($\mathbf{1}$)
    AllReduceCompletionTime: NotationSymbol  # time to complete one AllReduce collective over a group of devices, whatever algorithm implements it ($T_{\text{allreduce}}$) [s]
    AllReduceMinimumBandwidthTime: NotationSymbol  # smallest byte-transfer time any point-to-point AllReduce can achieve, with full-duplex per-rank bandwidth and no in-network reduction or compression ($T_{\text{bandwidth}}^{\text{min}}$) [s]
    AllRunsPassEstimate: NotationSymbol  # unbiased estimate, for one task, of the probability that all k runs pass, the share of k-subsets of the graded runs in which every run passed ($\widehat{r}$) [dimensionless]
    AllToAllCompletionTime: NotationSymbol  # time to complete one AllToAll collective, in which every device exchanges a distinct share of its payload with every other device ($T_{\text{alltoall}}$) [s]
    AllToAllExchangeTime: NotationSymbol  # elapsed time of one AllToAll exchange among the ranks of a communication group ($T$) [s]
    AllocatedBlockCount: NotationSymbol  # number of voxel blocks a sparse map allocates, those that hold data ($N_{\text{blocks}}$)
    AllocatedHazardRate: NotationSymbol  # share of the tolerable hazard rate that risk allocation assigns to one failure mode ($\lambda_{\text{alloc}}$) [1/h]
    AllocatedWallTime: NotationSymbol  # total wall-clock time for which a training job holds its allocated resources, productive or not ($T_{\text{wall}}$) [s]
    AllowableRailDroop: NotationSymbol  # largest transient drop in DC bus voltage that the bus capacitance is sized to keep the bus within ($\Delta V_{\text{droop, max}}$) [V]
    AllowableRailExcursion: NotationSymbol  # largest rail-voltage excursion a chip tolerates, the nominal voltage times the tolerance fraction ($\Delta V_{\max}$) [V]
    AllowableWindingTemperatureRise: NotationSymbol  # largest winding temperature rise above ambient that the continuous torque rating allows ($\Delta T_{\max}$) [K]
    AllowedTokenSet: NotationSymbol  # set of tokens a decoding mask leaves unmasked at a constraint state, the viable ordinary tokens plus an end-of-sequence token when the state accepts ($\mathcal{V}_{\text{allow}}$)
    AllowedViolationFraction: NotationSymbol  # fraction of eligible observations allowed to violate one reliability target over a window, its error budget ($V^{\text{allow}}$) [dimensionless]
    AlphabetByte: NotationSymbol  # one byte of the alphabet a byte-level constraint automaton reads ($b$)
    AlternateModelInput: NotationSymbol  # any second input compared with a model input, as in a Lipschitz bound over pairs of inputs ($x'$)
    AmbientDimension: NotationSymbol  # number of dimensions of the Euclidean space a ball or a collection of embedding vectors lives in ($D$)
    AmbientTemperature: NotationSymbol  # temperature of the air around a motor, the reference for its winding temperature rise ($T_{\text{amb}}$) [degC]
    AmortizedTurnCost: NotationSymbol  # mean monetary cost per task turn over one compaction cycle, including that cycle's compaction pass ($\bar{C}_{\text{turn}}$) [USD]
    AngularAcceleration: NotationSymbol  # angular acceleration vector of a rigid body ($\boldsymbol{\alpha}$) [rad/s^2]
    AngularClearance: NotationSymbol  # angle an approaching joint can still rotate before its tool point reaches the obstacle ($\theta_{\text{clear}}$) [rad]
    AngularClearanceRate: NotationSymbol  # time derivative of the angular clearance of an approaching joint ($\dot{\theta}_{\text{clear}}$) [rad/s]
    AngularFrequency: NotationSymbol  # angular frequency of a sinusoidal signal, the free variable of a frequency response ($\omega$) [rad/s]
    AngularSpeed: NotationSymbol  # magnitude of a rigid body's angular velocity ($\omega$) [rad/s]
    AngularVelocity: NotationSymbol  # angular velocity of a rigid body, the axis and rate of its rotation ($\boldsymbol{\omega}$) [rad/s]
    AngularVelocityDerivative: NotationSymbol  # time derivative of the angular velocity vector, written to define the angular acceleration ($\dot{\boldsymbol{\omega}}$) [rad/s^2]
    AnnualQueryVolume: NotationSymbol  # number of queries a deployed model serves in one year ($Q_{\text{annual}}$)
    AnonymitySetSize: NotationSymbol  # smallest number of records that share every quasi-identifier value in a k-anonymous release ($k_{\text{anon}}$)
    AntiWindupTrackingTimeConstant: NotationSymbol  # tracking time constant of back-calculation anti-windup, setting how fast the actuator mismatch discharges the integrator ($T_t$) [s]
    AntialiasingFilterVariance: NotationSymbol  # isotropic variance added to a projected Gaussian covariance as a low-pass filter against aliasing ($s$) [pixels^2]
    ApiAccount: NotationSymbol  # user account of a model API whose query stream an extraction detector scores and whose quota it adjusts ($u$)
    ApparentContrast: NotationSymbol  # contrast of an object as seen through the atmosphere at a given range ($C$) [dimensionless]
    AppearanceRealityGap: NotationSymbol  # appearance component of the reality gap: discrepancy between rendered and real images, weighted by what the policy is sensitive to ($\Delta_{\text{vis}}$) [varies]
    AppliedForce: NotationSymbol  # force acting on a body along the path over which it moves ($F$) [N]
    AppliedLoadCycleCount: NotationSymbol  # number of load cycles a component has experienced at one load level, in a cumulative fatigue-damage sum ($n$)
    ApproachChannelCount: NotationSymbol  # number of approach channels, each a zone and activation condition with its own approach-speed ceiling ($K_a$)
    ApproachDirection: NotationSymbol  # unit vector along which the gripper approaches a grasp affordance ($\hat{\mathbf{a}}_{\text{approach}}$) [dimensionless]
    ApproachPosition: NotationSymbol  # position of the end effector along its approach direction into a contact ($x$) [m]
    ApproachPredicate: NotationSymbol  # motion predicate that holds when the tool center point lies within the positional and angular tolerances of the pregrasp pose ($P_{\text{approach}}$)
    ApproachSpeed: NotationSymbol  # speed at which the gripper approaches the latch during a demonstration ($v_{\text{app}}$) [m/s]
    ApprovalCatchFraction: NotationSymbol  # share of the defective changes missed by the canary that a human approval gate before the pivot catches ($q$) [dimensionless]
    ApprovalHeldTrajectories: NotationSymbol  # mean number of trajectories held awaiting a human approval decision at once ($N_{\text{held}}$)
    ApproximatingDistribution: NotationSymbol  # distribution the code is built for, the second argument of a KL divergence ($q$)
    ArbitrationFieldBitCount: NotationSymbol  # number of bits of frame i sent at the nominal rate: identifier and control bits ($B_{\text{arb},i}$)
    ArcObstacleDistance: NotationSymbol  # distance along the circular arc a velocity pair commands to the nearest obstacle on it ($d_{\text{arc}}$) [m]
    ArchitectureResourceCost: NotationSymbol  # resource cost of a candidate architecture, such as its latency or memory, that a search constrains ($C$)
    ArchitectureSearchSpace: NotationSymbol  # set of candidate architectures a neural architecture search may choose from ($\mathcal{A}$)
    ArcsineEffectSize: NotationSymbol  # Cohen's effect size between two success probabilities, the difference of their arcsine-square-root transforms ($h$) [dimensionless]
    ArgumentLegConfidence: NotationSymbol  # probability, as assessed confidence, that one independent leg of a multi-legged assurance argument holds ($r$) [dimensionless]
    ArgumentLegCount: NotationSymbol  # number of independent legs of a multi-legged assurance argument, any one of which suffices for the claim ($m$)
    ArithmeticIntensity: NotationSymbol  # operations a workload performs per byte it moves through memory ($I$) [FLOP/byte]
    ArmStoppingDeceleration: NotationSymbol  # deceleration the arm must achieve after braking onset to stop within the latch force limit ($a_{\text{arm}}$) [m/s^2]
    ArmSuccessProbability: NotationSymbol  # hypothesized per-trial success probability of one arm of a two-arm comparison, indexed by arm ($p$) [dimensionless]
    ArmingSettleDuration: NotationSymbol  # time a fault-injection trial lets the plant settle in its nominal envelope before the fault is injected ($t_{\text{settle}}$) [s]
    ArrivalOffsetTravel: NotationSymbol  # distance the base travels between a sample's capture and its arrival timestamp ($\Delta x$) [m]
    ArrivalRate: NotationSymbol  # average number of requests or jobs arriving per second ($\lambda_{\text{arr}}$) [arrivals/s]
    ArrivalStampOffset: NotationSymbol  # time by which a sample's arrival timestamp trails its capture timestamp ($\Delta t_{\text{recv}}$) [s]
    ArrivalTimestamp: NotationSymbol  # instant a sensor sample arrives in host memory, when its direct memory access transfer completes ($t_{\text{recv}}$) [s]
    ArtifactVersion: NotationSymbol  # immutable version identifier, such as a content hash, that pins one version of a code, data, configuration or environment artifact ($v$)
    AssessedContactForceLimit: NotationSymbol  # largest contact force a site assessment admits for one human body region, contact type and geometry in a collaborative contact ($F_{\text{limit}}$) [N]
    AssignedJobGpu: NotationSymbol  # one GPU assigned to a job, indexed over the job's allocation ($g$)
    AssociationCostEntry: NotationSymbol  # entry of the data-association cost matrix for pairing track i with detection j ($c$) [dimensionless]
    AssuranceClaim: NotationSymbol  # top-level safety claim of an assurance argument, whose probability of holding is assessed from its premises ($C$) [dimensionless]
    AsymptoticTightBound: NotationSymbol  # asymptotic tight bound: grows at the same rate as its argument, up to constant factors ($\Theta$)
    AtmosphericExtinctionCoefficient: NotationSymbol  # rate per unit path length at which the atmosphere attenuates light by scattering and absorption ($\beta_{\text{atm}}$) [1/m]
    AttackStepsPgd: NotationSymbol  # number of inner gradient steps a projected gradient descent attack takes to build one adversarial example ($K_{\text{PGD}}$)
    AttackedTrialCount: NotationSymbol  # number of independent attacked trials or episodes run against a defense, in the zero-success rule-of-three bound ($n$)
    AttainableComputeRate: NotationSymbol  # highest operation rate a workload can sustain on given hardware, capped by peak compute or by memory bandwidth times arithmetic intensity ($R_{\text{attain}}$) [FLOP/s]
    AttemptCleanupCost: NotationSymbol  # charge for state reset, storage teardown and compensating transactions after one attempt ($C_{\text{cleanup}}$) [USD]
    AttemptCost: NotationSymbol  # full monetary cost of one attempt at a task ($C_{\text{attempt}}$) [USD]
    AttemptCycleDuration: NotationSymbol  # duration of one attempt plus its reset in a collection session ($t_{\text{cyc}}$) [s]
    AttemptDuration: NotationSymbol  # duration of one teleoperated demonstration attempt ($t_{\text{att}}$) [s]
    AttemptGenerationTime: NotationSymbol  # accelerator time one rollout attempt spends in generation, in accelerator-seconds ($T_{\text{gen}}$) [s]
    AttemptPassProbability: NotationSymbol  # probability that one independent attempt on a given task passes its check ($p$) [dimensionless]
    AttemptRuntimeOverheadCost: NotationSymbol  # share of orchestrator, scheduler and shared telemetry overhead apportioned to one attempt ($C_{\text{runtime}}$) [USD]
    AttemptVerificationCost: NotationSymbol  # charge for post-execution checks and test suites of one attempt not already billed as sandbox time ($C_{\text{verify}}$) [USD]
    AttentionHeadCount: NotationSymbol  # number of attention heads in one multi-head attention layer ($N_{\text{heads}}$)
    AttentionHeadDimension: NotationSymbol  # width of the query, key and value vectors in one attention head ($d_{\text{head}}$)
    AttentionHeadOutput: NotationSymbol  # output of one attention head before the output projection, the attention weights applied to the value matrix ($\mathbf{O}$)
    AttentionKeyMatrix: NotationSymbol  # matrix of key vectors, one row per token, in an attention layer ($\mathbf{K}$)
    AttentionOutputMatrix: NotationSymbol  # output of an attention layer after the output projection recombines the heads, one row per token ($\mathbf{O}$)
    AttentionOutputProjectionMatrix: NotationSymbol  # learned matrix that projects the concatenated outputs of all attention heads back to the model width ($\mathbf{W}_{O}$)
    AttentionQueryMatrix: NotationSymbol  # matrix of query vectors, one row per token, in an attention layer ($\mathbf{Q}$)
    AttentionScoreEntry: NotationSymbol  # attention score of query i against key j, one entry of the score matrix ($S_{ij}$) [dimensionless]
    AttentionScoreMatrix: NotationSymbol  # matrix of attention scores, the scaled dot products of every query with every key before the softmax ($\mathbf{S}$)
    AttentionScoreRow: NotationSymbol  # row of the attention score matrix for one query token, the logits its softmax normalizes ($\mathbf{S}_{i, \cdot}$)
    AttentionScoreSummand: NotationSymbol  # attention score of query i against key k, the term a softmax normalizing sum runs over ($S_{ik}$) [dimensionless]
    AttentionScoreTile: NotationSymbol  # block of attention scores between query tile i and key tile j, computed on chip in tiled attention ($\mathbf{S}_{ij}$)
    AttentionValuesMatrix: NotationSymbol  # matrix of value vectors, one row per token, that attention weights combine into the output ($\mathbf{V}$)
    AttentionValuesProjectionMatrix: NotationSymbol  # learned matrix that projects token representations into values ($\mathbf{W}_{V}$)
    AttentionValuesSourceMatrix: NotationSymbol  # representations attention draws its values from, before a head-specific value projection ($\mathbf{X}_{V}$)
    AttentionValuesVector: NotationSymbol  # value vector of one token, one row of the value matrix ($\mathbf{v}$)
    AttentionWeightEntry: NotationSymbol  # attention weight query i places on key j, one entry of the attention weight matrix ($A_{ij}$) [dimensionless]
    AttentionWeightMatrix: NotationSymbol  # matrix of attention weights, the row-wise softmax of scaled query-key scores ($\mathbf{A}$)
    AttentionWeightVector: NotationSymbol  # attention weights one query token places on every key, one row of the attention weight matrix ($\mathbf{a}$)
    AttitudeError: NotationSymbol  # angular error in an estimated attitude, which leaks gravity into the horizontal acceleration ($\theta_{\text{att}}$) [rad]
    AttributionFeatureCount: NotationSymbol  # number of input features a feature-attribution method assigns contributions to ($n_{\text{feat}}$)
    AuditGroupRecordCount: NotationSymbol  # number of records from one group in a fairness audit's confusion matrix, the sum of its four cells ($n_{\text{records}}$)
    AuthorityExposureLevel: NotationSymbol  # authority exposure level, what a task may do through its tools, from read access to irreversible external action ($A$) [dimensionless]
    AutocorrelationLagIndex: NotationSymbol  # number of samples separating the two values whose autocorrelation is taken ($k$)
    AutoencoderReconstructionError: NotationSymbol  # reconstruction error of an autoencoder on one record, the squared distance between the record and its reconstruction, used as an anomaly score ($\mathcal{L}_{\text{rec}}$) [varies]
    AutomatedResetTime: NotationSymbol  # time an automated fixture takes to restore a physical rig to its initial configuration between trials ($t_{\text{reset,auto}}$) [s]
    AutomatonStateSet: NotationSymbol  # finite set of states of an output-constraint automaton ($Q$)
    AutomatonTransitionFunction: NotationSymbol  # transition function of an output-constraint automaton, mapping a state and a byte to the next state ($\delta$)
    AutonomousExecutionComputeCost: NotationSymbol  # direct token and compute cost of running a proposed action autonomously, paid under either escalation decision ($C_{\text{compute}}$) [USD]
    AutoscalingDecisionTime: NotationSymbol  # time an autoscaler takes to decide on a scaling action once a load change is detected ($T_{\text{decide}}$) [s]
    AutoscalingDetectionTime: NotationSymbol  # time an autoscaler's metrics take to reveal a load change ($T_{\text{detect}}$) [s]
    AutoscalingResponseTime: NotationSymbol  # time an autoscaler takes from a load change to new capacity serving, detection plus decision plus provisioning plus warmup ($T_{\text{response}}$) [s]
    AvailabilityObjective: NotationSymbol  # availability dimension of a service's SLO vector: its indicator, target, window and owner ($S_{\text{availability}}$)
    AvailableBrakingDistance: NotationSymbol  # clear distance minus the localization bound and the protective clearance margin, the distance braking and blind travel may use ($D_{\text{avail}}$) [m]
    AvailableDeceleration: NotationSymbol  # deceleration a machine actually has available, the smaller of its credible braking deceleration and the traction limit friction coefficient times gravity ($a$) [m/s^2]
    AvailableObservations: NotationSymbol  # specification of what the agent may see and how much, including truncation and redaction rules ($\mathcal{O}_{\text{avail}}$)
    AverageCopperLossPower: NotationSymbol  # copper loss averaged over a duty cycle or sustained operation ($\bar{P}_{\text{loss}}$) [W]
    AverageOnlineRegret: NotationSymbol  # average regret of the online learner over the rounds of dataset aggregation, relative to the best fixed policy in hindsight ($\gamma_N$) [dimensionless]
    AveragePower: NotationSymbol  # power drawn averaged over a stated period ($P_{\text{avg}}$) [W]
    AverageStateVisitationDistribution: NotationSymbol  # average over the steps of a finite horizon of the distribution of states a policy visits, the policy written as superscript ($\bar{d}$)
    AxiBurstLength: NotationSymbol  # number of data beats in one burst of an AXI interconnect transaction ($A_{\text{len}}$)
    AxisGridPointCount: NotationSymbol  # number of grid points along each state coordinate of a regular grid ($N$)
    AzimuthAngle: NotationSymbol  # azimuth angle of a direction, measured in the horizontal plane about the vertical axis ($\theta_{\text{az}}$) [rad]
    BackEmfVoltage: NotationSymbol  # voltage the turning rotor induces in the stator windings, opposing the supply ($e_{\text{bemf}}$) [V]
    BacklashAngle: NotationSymbol  # angular lost motion of a gear train when the drive reverses, its backlash ($\theta_{\text{backlash}}$) [deg]
    BackwardEulerEigenvalue: NotationSymbol  # eigenvalue of the backward (implicit) Euler transition matrix ($\lambda_{\text{BE}}$) [dimensionless]
    BackwardEulerTransitionMatrix: NotationSymbol  # discrete-time transition matrix that backward (implicit) Euler integration produces, the inverse of identity minus the system matrix times the step ($\mathbf{A}_{\text{BE}}$)
    BackwardPassTime: NotationSymbol  # time of the full backward pass over all layers in one training step ($T_{\text{backward}}$) [s]
    BackwardReachableTube: NotationSymbol  # set of states from which a worst-case disturbance can force the plant into failure before the horizon despite the best control ($\mathcal{R}$)
    BalanceDivergenceTime: NotationSymbol  # time constant over which an unrecovered lean of a balancing biped grows by a factor of e, the reciprocal of the pendulum natural frequency ($\tau_{\text{fall}}$) [s]
    BallRadius: NotationSymbol  # radius of a Euclidean ball, also the radial coordinate when an integral switches to spherical coordinates ($r$) [dimensionless]
    BallVolume: NotationSymbol  # volume of a Euclidean ball in a space of the ambient dimension, as a function of its radius ($V_D$)
    Bandwidth: NotationSymbol  # sustained rate at which bytes move across a memory or data path ($\text{BW}$) [bytes/s]
    BankConflictPenalty: NotationSymbol  # time one memory bank or row conflict adds to a queued transfer, about a precharge plus an activation ($\Delta t_{\text{conflict}}$) [s]
    BarrierConstraintBound: NotationSymbol  # bound on the right side of one affine control-barrier constraint, one entry of the stacked bound vector ($b_{\text{cbf}}$) [varies]
    BarrierConstraintCount: NotationSymbol  # number of control-barrier constraints an enforcement record parameterizes ($K_b$)
    BarrierConstraintNormal: NotationSymbol  # normal vector of a single affine control-barrier constraint on the control input ($\mathbf{a}_{\text{cbf}}$) [varies]
    BarrierFunction: NotationSymbol  # control-barrier certificate: a margin function of the state that is nonnegative exactly on the safe set ($h$) [m]
    BarrierFunctionRate: NotationSymbol  # time derivative of the control barrier function along the closed-loop dynamics ($\dot{h}$) [m/s]
    BarrierHalfSpaceControlSet: NotationSymbol  # set of control inputs that satisfy the control barrier function inequality at the current state, before actuator limits are imposed ($\mathcal{U}_{\text{safe}}$)
    BarrierMargin: NotationSymbol  # margin a barrier value must keep above zero, in normalized distance ($\delta$) [dimensionless]
    BarrierNormalizationScale: NotationSymbol  # positive scale fixed in the enforcement contract that divides the barrier value to log a dimensionless margin ($h_{\text{scale}}$) [m]
    BarrierProjectionLagrangian: NotationSymbol  # Lagrangian of the single-constraint CBF-QP projection: the squared deviation from the nominal command plus the multiplier times the constraint residual ($L_{\text{Lag}}$)
    BarrierProjectionMultiplier: NotationSymbol  # Lagrange multiplier of the barrier half-space constraint in the CBF-QP projection ($\lambda$) [varies]
    BarrierRelativeDegree: NotationSymbol  # number of times the barrier function must be differentiated along the dynamics before the control input appears ($r$)
    BaseAbscissaVelocity: NotationSymbol  # velocity of a mobile base along the first axis of the fixed planar frame, the time derivative of its abscissa ($\dot{x}$) [m/s]
    BaseAccelerationCommand: NotationSymbol  # acceleration the mobile base channel is commanded to follow ($a$) [m/s^2]
    BaseBatchSize: NotationSymbol  # reference batch size at which the base learning rate was tuned ($B_{\text{base}}$)
    BaseCameraRotation: NotationSymbol  # rotation part of the camera-to-base extrinsic transform ($\mathbf{R}_{\text{base}}^{\text{cam}}$) [dimensionless]
    BaseCameraTransform: NotationSymbol  # rigid transform mapping camera coordinates into the robot base frame ($\mathbf{T}_{\text{base}}^{\text{cam}}$)
    BaseCameraTranslation: NotationSymbol  # translation part of the camera-to-base extrinsic transform, the camera center in the base frame ($\mathbf{t}_{\text{base}}^{\text{cam}}$) [m]
    BaseFlangeTransform: NotationSymbol  # rigid transform mapping tool-flange coordinates into the robot base frame, from forward kinematics ($\mathbf{T}_{\text{base}}^{\text{flange}}$)
    BaseForwardSpeed: NotationSymbol  # forward linear speed of the mobile base along its heading, the speed input of the unicycle model ($v$) [m/s]
    BaseFramePoint: NotationSymbol  # position of a point expressed in the robot's base coordinate frame ($\mathbf{p}_{\text{base}}$) [m]
    BaseHeadingRate: NotationSymbol  # rate of change of a mobile base's heading angle ($\dot{\theta}_{\text{head}}$) [rad/s]
    BaseJointInertiaCoupling: NotationSymbol  # block of the generalized inertia matrix through which actuated-joint accelerations load the floating-base equations ($\mathbf{M}_{bj}$) [varies]
    BaseLateralPosition: NotationSymbol  # lateral position of the mobile base, its offset from the reference line such as an aisle centerline ($y_{\text{base}}$) [m]
    BaseLateralVelocity: NotationSymbol  # lateral velocity of a mobile base in the plane ($\dot{y}$) [m/s]
    BaseLayerCount: NotationSymbol  # number of layers in the base model before compound scaling ($N_{L,0}$)
    BaseLearningRate: NotationSymbol  # learning rate tuned at the reference batch size before linear scaling ($\eta_{\text{base}}$) [dimensionless]
    BaseLongitudinalPosition: NotationSymbol  # longitudinal position of the mobile base along the reference direction, such as distance along an aisle ($x_{\text{base}}$) [m]
    BaseLongitudinalVelocity: NotationSymbol  # forward velocity of a mobile base in the plane ($\dot{x}$) [m/s]
    BaseOdometrySpeed: NotationSymbol  # speed at which the mobile base closes on the target, as wheel odometry reports it ($v$) [m/s]
    BaseOrdinateVelocity: NotationSymbol  # velocity of a mobile base along the second axis of the fixed planar frame, the time derivative of its ordinate ($\dot{y}$) [m/s]
    BasePosition: NotationSymbol  # position of the mobile base in the estimation frame ($\mathbf{p}_{\text{base}}$) [m]
    BasePromptLength: NotationSymbol  # number of prompt tokens in the first call of a trajectory, before any turn appends to it ($S_{\text{base}}$)
    BaseStoppingDistance: NotationSymbol  # stopping distance from the body's budget rows alone: braking distance, localization and clearance overhead, and brake-onset travel ($d_{\text{base}}$) [m]
    BaseToCameraTransform: NotationSymbol  # rigid transform carrying robot-base coordinates into the camera's optical frame, the camera extrinsics ($\mathbf{T}_{\text{cam} \leftarrow \text{base}}$)
    BaseTrainingComputeCost: NotationSymbol  # compute cost of a training run in the absence of failures ($C_{\text{compute}}$) [USD]
    BaseTrajectoryTokenCount: NotationSymbol  # number of output tokens the base trajectory generates, without any continuation branches ($N_{\text{base}}$)
    BaseTravelDistance: NotationSymbol  # distance the machine has traveled toward the boundary it must stop before, as a function of time during a stop ($d$) [m]
    BaseTravelSpeed: NotationSymbol  # speed at which the mobile base travels while a sensor captures a measurement ($v_{\text{base}}$) [m/s]
    BaseWidthMultiplier: NotationSymbol  # width multiplier of the base model before compound scaling ($w_{\text{mult},0}$) [dimensionless]
    BaseWristTransform: NotationSymbol  # homogeneous transform mapping wrist-frame points into the base frame ($\mathbf{T}_{\text{base} \leftarrow \text{wrist}}$)
    BaseYawRateBias: NotationSymbol  # persistent bias in the commanded yaw rate of the mobile base, a steering drift ($\omega_{\text{bias}}$) [rad/s]
    BaselineAcceptanceEstimate: NotationSymbol  # observed acceptance share of the baseline configuration over the tasks of a paired comparison, its accepted count divided by the task count ($\hat{p}_{\text{base}}$) [dimensionless]
    BaselineControllerAction: NotationSymbol  # action the validated baseline controller applies at step t while a candidate runs in shadow mode ($\mathbf{a}_t^{\text{base}}$)
    BaselineConversionRate: NotationSymbol  # conversion rate of the control variant before the change an A/B test evaluates ($p_0$) [dimensionless]
    BaselineFeatureMean: NotationSymbol  # mean of the Gaussian fitted to the baseline sample of a monitored feature ($\mu_{\text{base}}$) [varies]
    BaselineFeatureStandardDeviation: NotationSymbol  # standard deviation of the Gaussian fitted to the baseline sample of a monitored feature ($\sigma_{\text{base}}$) [varies]
    BaselineIntervalLowerLimit: NotationSymbol  # lower limit of the single-proportion Wilson interval for the baseline configuration's acceptance proportion ($l_{\text{base}}$) [dimensionless]
    BaselineIntervalUpperLimit: NotationSymbol  # upper limit of the single-proportion Wilson interval for the baseline configuration's acceptance proportion ($u_{\text{base}}$) [dimensionless]
    BaselineLowerLimitDistance: NotationSymbol  # downward distance from the baseline configuration's observed acceptance share to the lower limit of its Wilson interval ($\delta^-_{\text{base}}$) [dimensionless]
    BaselineOnlyAcceptedTaskCount: NotationSymbol  # number of tasks in a paired comparison that the baseline accepts and the candidate does not, one of the two discordant counts ($b$)
    BaselineOnlyCorrectCount: NotationSymbol  # number of evaluation examples the baseline model classifies correctly and the compared model gets wrong ($n_{\text{base}}$)
    BaselinePolicyAcceptanceProbability: NotationSymbol  # acceptance probability of the baseline routing policy against which an alternative is compared ($p_{X}$) [dimensionless]
    BaselinePolicyCost: NotationSymbol  # expected additional monetary cost of the baseline routing policy ($C_{X}$) [USD]
    BaselineStateDistribution: NotationSymbol  # distribution of states the baseline controller visits while a candidate runs in shadow mode ($d^{\pi_{\text{base}}}$)
    BaselineTaskCost: NotationSymbol  # expected operating cost per assigned task under the baseline configuration, including failed attempts, retries and review ($C_{\text{task, base}}$) [USD/task]
    BaselineToolCallDistribution: NotationSymbol  # distribution of the baseline configuration's tool calls over the kinds of tool call, under one declared action-counting policy ($p_{b}$)
    BaselineTrackingError: NotationSymbol  # tracking error between commanded and measured plant motion during the steady-state arming of a fault-injection trial ($e_{\text{track}}$) [m]
    BaselineTrainingDuration: NotationSymbol  # wall-clock duration of a training run before failure and recovery overhead ($T_{\text{hours}}$) [h]
    BaselineTrajectoryGoodput: NotationSymbol  # trajectory goodput of the baseline configuration over a matched observation interval and workload ($\mathcal{G}_{b}$) [tasks/s]
    BaselineUpperLimitDistance: NotationSymbol  # upward distance from the baseline configuration's observed acceptance share to the upper limit of its Wilson interval ($\delta^+_{\text{base}}$) [dimensionless]
    BatchArrivalRate: NotationSymbol  # rate at which batches arrive at a server when each batch is treated as one super-request, the request arrival rate divided by the batch size ($\lambda_{\text{batch}}$) [batches/s]
    BatchAxisLabel: NotationSymbol  # einsum index label naming the batch axis of an attention tensor ($b$)
    BatchBufferMemory: NotationSymbol  # host memory one batch occupies while it sits in a pipeline buffer ($M_{\text{batch}}$) [bytes]
    BatchCostPerAcceptedTask: NotationSymbol  # expected cost per accepted task when a full pool of candidates is generated and checked in parallel ($C_{\text{eff,batch}}$) [USD/task]
    BatchDataVolume: NotationSymbol  # bytes of input data one batch carries to the accelerator ($D_{\text{vol}}^{(\text{batch})}$) [bytes]
    BatchExampleIndex: NotationSymbol  # position of one example within a training batch whose per-example losses are aggregated ($n$)
    BatchFetchTime: NotationSymbol  # time to fetch one batch from storage ($T_{\text{fetch}}$) [s]
    BatchFillArrivalCount: NotationSymbol  # number of arrivals after a batch's triggering request needed to fill the batch, one less than the maximum batch size ($k$)
    BatchFixedOverheadTime: NotationSymbol  # part of a batch's service time paid once per batch whatever its size, such as weight loading and kernel launch ($T_{\text{fixed}}$) [s]
    BatchFlushStartTime: NotationSymbol  # time at which a group-commit batch starts its shared durable storage operation ($t_{\text{start}}$) [s]
    BatchFormationDelay: NotationSymbol  # time a request waits for its batch to fill before the batch is dispatched ($L_{\text{lat,form}}$) [s]
    BatchIndex: NotationSymbol  # position of one batch in the sequence of batches a data loader or runtime delivers ($k$)
    BatchLabelComponent: NotationSymbol  # one-hot target indicator for class j on example i of a batch ($y_{ij}$) [dimensionless]
    BatchLatencyEnvelope: NotationSymbol  # conservative bound on a request's batching latency, the full batching window plus the service time of its 99th-percentile batch ($L_{\text{lat,envelope}}$) [s]
    BatchLoss: NotationSymbol  # mean loss over the examples of one mini-batch ($\mathcal{L}_{\text{batch}}$) [dimensionless]
    BatchMaxOutputLength: NotationSymbol  # largest number of output tokens any request in a batch generates ($S_{\text{max}}$)
    BatchMaximumDeviationMultiple: NotationSymbol  # number of standard deviations by which the longest output in a batch is assumed to exceed the mean output length ($k$) [dimensionless]
    BatchMeanOutputLength: NotationSymbol  # mean number of output tokens per request over the requests of a batch ($\bar{S}$) [tokens]
    BatchNormMean: NotationSymbol  # mean of one feature's activations over a mini-batch, as batch normalization computes it ($\mu_{\mathcal{B}}$) [dimensionless]
    BatchNormOutput: NotationSymbol  # activation batch normalization outputs after its learned scale and shift ($y$) [dimensionless]
    BatchNormRunningMean: NotationSymbol  # per-channel running mean of activations that batch normalization stores for inference ($\boldsymbol{\mu}_{\mathcal{B}}$)
    BatchNormRunningStandardDeviation: NotationSymbol  # per-channel square root of the running variance plus the numerical stabilizer that batch normalization uses at inference ($\boldsymbol{\sigma}_{\mathcal{B}}$)
    BatchNormStandardDeviation: NotationSymbol  # standard deviation of one feature's activations over a mini-batch ($\sigma_{\mathcal{B}}$) [dimensionless]
    BatchOperations: NotationSymbol  # floating-point operations of the forward and backward pass for one batch ($O_{\text{batch}}$) [FLOP]
    BatchPredictedProbability: NotationSymbol  # predicted probability for class j on example i of a batch ($\hat{y}_{ij}$) [dimensionless]
    BatchProcessingTime: NotationSymbol  # time the preprocessing stage spends on one batch ($T_{\text{process}}$) [s]
    BatchRolloutCount: NotationSymbol  # number of rollouts in one training batch, across all of its tasks and groups ($N_{\text{batch}}$)
    BatchScalingFactor: NotationSymbol  # factor by which a training recipe multiplies its batch size, the multiplier the linear and square-root learning-rate scaling rules apply ($k$) [dimensionless]
    BatchServiceTime: NotationSymbol  # time an accelerator takes to process one batch once execution begins, as a function of batch size ($T_{\text{svc}}$) [s]
    BatchSize: NotationSymbol  # number of examples or requests processed together in one step ($B$)
    BatchTransferTime: NotationSymbol  # time to copy one batch of input data from host memory to the device that consumes it ($T_{\text{transfer}}$) [s]
    BatchedDecodeStepTime: NotationSymbol  # time of one decode step that serves a batch of streams with one weight read, the larger of the weight-transfer and arithmetic times ($t_{\text{step}}$) [s]
    BatchedEventIndex: NotationSymbol  # position of one event or record within a group-commit batch ($i$)
    BatchedRequestLatency: NotationSymbol  # latency a request accrues in a batching server, its batching wait plus the service time of its batch ($L_{\text{lat,batch}}$) [s]
    BatchingLatencyBudget: NotationSymbol  # share of a mean latency objective allotted to batching wait and batch service ($L_{\text{lat,budget}}$) [s]
    BatchingQueueWait: NotationSymbol  # time a request waits in a batching server's queue before its batch begins service, formation included ($L_{\text{lat,wait}}$) [s]
    BatchingWindow: NotationSymbol  # length of the batching window, the longest a dynamic batcher holds requests before dispatching a batch ($T_{\text{window}}$) [s]
    BatteryCellResistance: NotationSymbol  # internal resistance of a battery pack's cells, one term of the lumped source resistance ($R_{\text{cell}}$) [Ohm]
    BeamDivergenceAngle: NotationSymbol  # full angle by which a lidar beam spreads as it propagates ($\delta\theta_{\text{div}}$) [rad]
    BeamWidth: NotationSymbol  # largest number of partial candidates a beam search retains for the next step ($W$)
    BearingRay: NotationSymbol  # viewing ray through an image pixel, the back-projection of its homogeneous coordinates through the inverse intrinsics, along which every depth fits the pixel ($\mathbf{r}$) [dimensionless]
    BeatFrequency: NotationSymbol  # frequency of the beat tone from mixing an FMCW echo with the transmitted chirp, proportional to range ($f_b$) [Hz]
    BehaviorCloningHorizon: NotationSymbol  # number of control steps in an episode over which imitation errors compound ($S$)
    BehaviorCloningLoss: NotationSymbol  # behavior-cloning objective, the expected imitation error under the collecting policy's empirical state-action occupancy ($\mathcal{L}_{\text{BC}}$)
    BehaviorDistribution: NotationSymbol  # next-token distribution under which training tokens were actually sampled, such as an older policy ($b$)
    BehaviorPolicyVersion: NotationSymbol  # version number of the policy checkpoint that generated a rollout, counted in trainer updates ($\text{ver}_b$)
    BeliefErrorBound: NotationSymbol  # upper bound on the discrepancy between a stored belief and the true state after a given elapsed time without new evidence ($E$) [varies]
    BeliefInvalidationHorizon: NotationSymbol  # time since the last supporting evidence after which a stored belief may no longer be used ($T_{\text{inv}}$) [s]
    BeliefState: NotationSymbol  # probability distribution over environment states conditional on the interaction history, the belief state of a partially observable decision process ($b$)
    BellmanBackupTarget: NotationSymbol  # regression target of a critic update: the reward plus the discounted value estimate at the next state ($y$) [dimensionless]
    BellmanErrorLoss: NotationSymbol  # squared Bellman error of a Q-function against its bootstrapped target, the standard critic loss ($\mathcal{L}_{\text{Bellman}}$)
    BelowThresholdIndicator: NotationSymbol  # indicator over a sparse weight matrix that is one at each entry whose magnitude is at or below the pruning threshold ($\mathbf{1}_{\{|(\mathbf{W}_{\text{sparse}})_{ij}| \leq \delta_{\text{prune}}\}}$)
    BestEffortWireBitCount: NotationSymbol  # largest non-preemptible wire occupancy of a best-effort frame, in bits, including framing overhead ($B_{\text{BE,wire}}$)
    BetterArmSuccessProbability: NotationSymbol  # per-trial success probability of the better policy in a two-policy power calculation ($p_{\text{high}}$) [dimensionless]
    BetweenTaskStandardDeviation: NotationSymbol  # standard deviation across the task population of a fixed system's per-task acceptance probability, whose square is the between-task variance of an acceptance estimate ($\sigma_{\text{task}}$) [dimensionless]
    BevPillarFeature: NotationSymbol  # feature of one bird's-eye-view pillar, the sum of the frustum features that fall inside it, as a function of the pillar ($\mathbf{V}_{\text{BEV}}$)
    BevPillarXIndex: NotationSymbol  # first ground-plane grid coordinate of one pillar of a bird's-eye-view grid ($X$)
    BevPillarYIndex: NotationSymbol  # second ground-plane grid coordinate of one pillar of a bird's-eye-view grid ($Y$)
    BiasFeedbackFactor: NotationSymbol  # factor by which a group's disparity is multiplied over one feedback interval, fitted from observed disparities ($\alpha_{\text{fb}}$) [dimensionless]
    BiasVector: NotationSymbol  # vector of learned biases one layer adds to its weighted sums, one per neuron or output channel ($\mathbf{b}$)
    BilledAllocationTime: NotationSymbol  # billed allocation time of a spot job, including checkpointing, recovery and work discarded after interruptions ($T_{\text{total}}$) [s]
    BinaryActivationVector: NotationSymbol  # vector of binarized activations packed into one machine word ($\mathbf{a}$)
    BinaryClassifier: NotationSymbol  # binary classifier viewed as a function that maps an individual's input to a decision of zero or one ($h$)
    BinaryWeightVector: NotationSymbol  # vector of binarized weights packed into one machine word ($\mathbf{w}$)
    BinomialSuccessCount: NotationSymbol  # random number of successful trials among the independent trials of an evaluation record, binomially distributed ($X$)
    BipedTiltAcceleration: NotationSymbol  # second time derivative of a balancing body's lean from vertical ($\ddot{\theta}_{\text{tilt}}$) [rad/s^2]
    BisectionBandwidth: NotationSymbol  # total bandwidth across the narrowest cut that divides a network's nodes into two equal halves ($\text{BW}_{\text{bisect}}$) [bytes/s]
    BitFlippedFloatNumber: NotationSymbol  # number a floating-point bit pattern represents after one exponent bit flips ($x_{\text{corrupt}}$) [dimensionless]
    BitTime: NotationSymbol  # duration of one bit on the bus, charged once for arbitration at the boundary of a response-time window ($T_{\text{bit}}$) [s]
    BlackoutClockConversionError: NotationSymbol  # clock-conversion error accumulated over a synchronization blackout with a one-sigma jitter excursion, drift rate times blackout plus jitter ($\Delta t_{\text{one-sigma scenario}}$) [s]
    BlendDeadline: NotationSymbol  # internal scheduling instant by which a preferred successor trajectory should clear validation, earlier than the commitment instant ($t_{\text{blend}}$) [s]
    BlendedCommandJerk: NotationSymbol  # time derivative of a blended commanded acceleration during a bumpless transfer ($j_{\text{cmd}}$) [varies]
    BlendedControlCommand: NotationSymbol  # command the enforcer proposes during a bumpless transfer, the blend of the two latched endpoints, checked before dispatch ($\mathbf{u}_{\text{blend}}$)
    BlendedInputTokenPrice: NotationSymbol  # average price per input token when a share of input tokens is billed at the cached rate and the rest at the uncached rate ($c_{\text{in}}$) [USD/token]
    BlendingTransitionWindow: NotationSymbol  # duration of the smooth blend from one latched command to the next during a bumpless transfer ($\tau_{\text{blend}}$) [s]
    BlindTravelDistance: NotationSymbol  # distance a machine travels during the pre-brake delay, before retarding force begins ($d_{\text{blind}}$) [m]
    BlockCheckTime: NotationSymbol  # time to run the check on one generated block ($T_{\text{ver}}$) [s]
    BlockCoordinateX: NotationSymbol  # integer x component of a voxel block coordinate ($b_x$)
    BlockCoordinateY: NotationSymbol  # integer y component of a voxel block coordinate ($b_y$)
    BlockCoordinateZ: NotationSymbol  # integer z component of a voxel block coordinate ($b_z$)
    BlockEdgeVoxelCount: NotationSymbol  # number of voxels along one edge of a cubic voxel block ($n_{\text{blk}}$)
    BlockMaximum: NotationSymbol  # largest of the observations of a monitored quantity in one block, indexed by the block size ($M$) [varies]
    BlockMaximumLevel: NotationSymbol  # level of the block maximum at which its limiting distribution is evaluated ($z$) [varies]
    BlockMaximumLimitDistribution: NotationSymbol  # limiting cumulative distribution function of the normalized block maximum, a member of the generalized extreme value family ($G$)
    BlockMaximumNormalizingScale: NotationSymbol  # positive scale constant, indexed by block size, that divides the centered block maximum so it converges in distribution ($a$) [varies]
    BlockMaximumNormalizingShift: NotationSymbol  # shift constant, indexed by block size, subtracted from the block maximum before normalization ($b$) [varies]
    BlockMetadataSize: NotationSymbol  # bytes of metadata stored per allocated voxel block ($s_{\text{meta}}$) [bytes]
    BlockQuantizationScale: NotationSymbol  # quantization scale computed independently for one block of a block-quantized tensor ($s_b$) [varies]
    BlockRangeMaximum: NotationSymbol  # largest value in one block of a block-wise quantized tensor, the upper end of that block's range ($x_{\max,i}$) [varies]
    BlockRangeMinimum: NotationSymbol  # smallest value in one block of a block-wise quantized tensor, the lower end of that block's range ($x_{\min,i}$) [varies]
    BlockSampleCount: NotationSymbol  # number of independent observations in one block whose maximum is taken ($n$)
    BlockedWorkerCount: NotationSymbol  # mean number of worker slots held by calls blocked on a stalled dependency ($N_{\text{blocked}}$)
    BlockingCallTimeout: NotationSymbol  # hard timeout after which a call blocked on an unresponsive dependency gives up the worker slot it occupies ($T_{\text{out}}$) [s]
    BlockingTaskIndex: NotationSymbol  # index of a lower-priority task whose critical sections can block the task under analysis ($k$)
    BodyAcceleration: NotationSymbol  # time derivative of the speed of a moving body ($\dot{v}$) [m/s^2]
    BodyDisplacement: NotationSymbol  # displacement of a body along the line of an applied force ($x$) [m]
    BodyEndEffectorTransform: NotationSymbol  # rigid transform mapping end-effector coordinates into the robot base frame, the arm's forward kinematics ($\mathbf{T}_{BE}$)
    BodyFramePoint: NotationSymbol  # position of a perceived point expressed in the robot body frame ($\mathbf{p}_{\text{body}}$) [m]
    BodyFramePointCovariance: NotationSymbol  # covariance of a perceived point's position error expressed in the robot body frame, the spatial covariance an observation contract carries ($\boldsymbol{\Sigma}_{\text{body}}$) [m^2]
    BodyLidarTransform: NotationSymbol  # calibrated mounting extrinsic of a lidar, mapping lidar coordinates into the body frame ($\mathbf{T}_{bl}$)
    BodyReferenceFrame: NotationSymbol  # coordinate frame attached to the mobile robot base ($\mathcal{F}_B$)
    BodyRegionMass: NotationSymbol  # effective mass of a human body region, here the hand, in the ISO/TS 15066 body model ($m_H$) [kg]
    BodySpeed: NotationSymbol  # magnitude of the linear velocity of a moving body or machine ($v$) [m/s]
    BoltzmannConstant: NotationSymbol  # Boltzmann constant, relating a temperature to the thermal energy scale ($k_B$) [J/K]
    BoltzmannStateEnergy: NotationSymbol  # energy of one microstate whose weight the Boltzmann distribution gives as the exponential of minus energy over Boltzmann constant times temperature ($E$) [J]
    BoltzmannTemperature: NotationSymbol  # absolute temperature of a system whose states follow the Boltzmann distribution ($T$) [K]
    BootstrapQuantileFunction: NotationSymbol  # empirical quantile function of the bootstrap replicate statistics, returning the smallest replicate value whose empirical distribution reaches a level ($Q$)
    BootstrapReplicateCount: NotationSymbol  # number of bootstrap resamples drawn to approximate a statistic's sampling distribution ($B_{\text{boot}}$)
    BootstrapReplicateIndex: NotationSymbol  # position of one bootstrap resample among the replicates ($r$)
    BottomedOutContactStiffness: NotationSymbol  # contact stiffness after a compliant cover bottoms out against the rigid structure behind it, effectively unbounded ($k_{\infty}$) [varies]
    BouligandContingentCone: NotationSymbol  # set-valued map giving, at each point of a closed set, the Bouligand contingent cone: the directions along which the set can be entered or followed to first order ($T_{\mathcal{C}}$)
    BoundaryConditionVector: NotationSymbol  # vector of the boundary positions, velocities and accelerations a trajectory segment must match at its endpoints ($\mathbf{b}$)
    BoundaryLayerTime: NotationSymbol  # stretched time of the boundary layer, physical time divided by the time-scale separation ratio ($t_{\text{fast}}$) [s]
    BoundedRegimeTimestamp: NotationSymbol  # reference-clock timestamp at which the plant enters its safe bounded regime in a fault trial ($t_{\text{bounded}}$) [s]
    BoxMaximumHorizontalCoordinate: NotationSymbol  # horizontal pixel coordinate of a bounding box's right edge ($x_{\text{max}}$) [pixels]
    BoxMaximumVerticalCoordinate: NotationSymbol  # larger vertical pixel coordinate of a bounding box's edges ($y_{\text{max}}$) [pixels]
    BoxMinimumHorizontalCoordinate: NotationSymbol  # horizontal pixel coordinate of a bounding box's left edge ($x_{\text{min}}$) [pixels]
    BoxMinimumVerticalCoordinate: NotationSymbol  # smaller vertical pixel coordinate of a bounding box's edges ($y_{\text{min}}$) [pixels]
    BrakeChopperResistance: NotationSymbol  # resistance of the absorption resistor a dynamic brake chopper switches across the bus ($R_{\text{brake}}$) [Ohm]
    BrakeCommandForce: NotationSymbol  # contact force against the latch at the instant the brake command issues, after the contact loop responds ($F_{\text{cmd}}$) [N]
    BrakeResistorPulseEnergy: NotationSymbol  # energy the braking resistor can absorb in one adiabatic pulse ($E_{\text{pulse}}$) [J]
    BrakingDistance: NotationSymbol  # distance a machine travels from brake onset to rest ($d_{\text{brake}}$) [m]
    BrakingForce: NotationSymbol  # retarding force applied to bring a moving machine to rest ($F_{\text{brake}}$) [N]
    BrakingOnsetContactForce: NotationSymbol  # contact force reached against the latch plate at physical braking onset ($F_{\text{cmd}}$) [N]
    BrakingWork: NotationSymbol  # mechanical work active brakes perform against a moving mass across a displacement, braking force times that displacement ($W_{\text{brake}}$) [J]
    BreadthCriticalPathLatency: NotationSymbol  # critical-path latency of one decision when test-time compute samples a pool of candidates concurrently along the breadth axis ($T^{\text{breadth}}$) [s]
    BreadthPeakKvCacheMemory: NotationSymbol  # peak memory the key-value caches of a breadth allocation occupy when its concurrent candidate streams share one prompt prefix ($M_{\text{KV}}^{\text{breadth}}$) [bytes]
    BreakEvenAcceptanceProbability: NotationSymbol  # per-attempt acceptance probability at which an attempt at the original cost matches the cost per accepted task of a price cut ($p'$) [dimensionless]
    BreakEvenBrakingDeceleration: NotationSymbol  # credible braking deceleration at which the braking travel of a stopping budget exactly consumes the room left for braking ($a$) [m/s^2]
    BreakEvenParseFailureRate: NotationSymbol  # parse failure rate at which constrained decoding and free sampling with one repair have equal expected latency ($p_{\text{break}}$) [dimensionless]
    BreakEvenTaskCount: NotationSymbol  # number of tasks an adaptation must serve before its per-task saving recovers its one-time cost ($N^*$)
    BreakEvenUtilization: NotationSymbol  # fraction of reference capacity above which a dedicated deployment costs less per hour than a per-token API at the same delivered output ($u^*$) [dimensionless]
    BreakerCooldownInterval: NotationSymbol  # time an open circuit breaker waits before it admits a half-open probe ($T_{\text{cool}}$) [s]
    BreakerFailureFraction: NotationSymbol  # share of the classified outcomes in a circuit breaker's window that count as infrastructure failures ($r$) [dimensionless]
    BreakerMinimumSampleCount: NotationSymbol  # smallest number of classified outcomes a circuit breaker's window must hold before it may trip ($n_{\min}$)
    BreakerOutcomeIndex: NotationSymbol  # position of one classified call outcome within a circuit breaker's rolling window ($i$)
    BreakerOutcomeWindow: NotationSymbol  # rolling window of recent classified call outcomes in one circuit breaker's scope ($W$)
    BreakerTripThreshold: NotationSymbol  # failure fraction at or above which a circuit breaker trips to open ($r_{\text{trip}}$) [dimensionless]
    BridgeTraceResistance: NotationSymbol  # resistance of the circuit-board traces in a half-bridge's shoot-through path ($R_{\text{trace}}$) [Ohm]
    BroadcastColumnElement: NotationSymbol  # one element of the broadcast column tensor ($A$) [dimensionless]
    BroadcastColumnTensor: NotationSymbol  # column tensor of shape (3, 1) that broadcasting stretches across columns ($\mathbf{A}$)
    BroadcastCompletionTime: NotationSymbol  # time for one root device to send a payload to every other device in a collective group ($T_{\text{broadcast}}$) [s]
    BroadcastRowElement: NotationSymbol  # one element of the broadcast row tensor ($b$) [dimensionless]
    BroadcastRowTensor: NotationSymbol  # row tensor of shape (1, 4) that broadcasting stretches across rows ($\mathbf{B}$)
    BroadcastTableSize: NotationSymbol  # size in bytes of the small table a broadcast join sends to every worker ($s_{\text{table}}$) [bytes]
    BspLearningRate: NotationSymbol  # learning rate a bulk-synchronous data-parallel run uses at the same effective batch size ($\eta_{\text{BSP}}$) [dimensionless]
    BsplineBasisFunction: NotationSymbol  # basis function of a B-spline, indexed by its control point and its polynomial degree ($B$) [dimensionless]
    BudgetEvaluationTime: NotationSymbol  # wall-clock time at which a task's budget account is evaluated, the argument of its settled charges and reserved funds in the conservation invariant ($\tau$) [s]
    BudgetFreeBalance: NotationSymbol  # unreserved monetary balance of one budget account node, from which new reservations and allocations are drawn ($F$) [USD]
    BudgetNodeIndex: NotationSymbol  # index of one node, the root task or a delegated child, in a delegation tree whose budget account the runtime keeps ($k$)
    BudgetReservedFunds: NotationSymbol  # monetary funds a budget account withholds for outstanding liabilities, delegated allocations and protected finalization ($R$) [USD]
    BudgetSettledCharges: NotationSymbol  # gross monetary charges settled into one budget account, of a task or of a node in its delegation tree ($E$) [USD]
    BudgetedAcceptanceProbability: NotationSymbol  # budgeted acceptance of a fixed complete policy: the probability that one trial is accepted and its total monetary charge stays within a stated ceiling ($q_{\pi}$) [dimensionless]
    BufferDependencyEdgeSet: NotationSymbol  # set of directed edges of a buffer dependency graph, each meaning a packet in one buffer needs the next buffer to progress ($E$)
    BufferDependencyGraph: NotationSymbol  # directed graph whose vertices are switch packet buffers and whose edges say which buffer a packet needs next ($G$)
    BufferDependencyVertexSet: NotationSymbol  # set of vertices, the packet buffers or virtual-lane queues, of a buffer dependency graph ($V$)
    BufferIndex: NotationSymbol  # position of one staging buffer in a double-buffered or multi-buffered pipeline, where one buffer fills while the next is consumed ($i$)
    BulkDecouplingCapacitance: NotationSymbol  # bulk electrolytic or polymer decoupling capacitance that holds a rail until the regulator loop responds ($C_{\text{bulk}}$) [F]
    BumplessTransferBlend: NotationSymbol  # blend weight that ramps from zero to one across a bumpless transfer, as a function of normalized time ($\alpha$) [dimensionless]
    BumplessTransferBlendAcceleration: NotationSymbol  # second time derivative of the bumpless-transfer blend weight, zero at both ends of a fixed-endpoint blend ($\ddot{\alpha}$) [1/s^2]
    BumplessTransferBlendRate: NotationSymbol  # first time derivative of the bumpless-transfer blend weight, zero at both ends of a fixed-endpoint blend ($\dot{\alpha}$) [1/s]
    BumplessTransferBlendSlope: NotationSymbol  # derivative of the bumpless-transfer blend weight with respect to normalized progress, whose peak sets the shortest blend window under a rate ceiling ($\alpha'$) [dimensionless]
    BumplessTransferProgress: NotationSymbol  # elapsed time since a bumpless transfer began, divided by the blending window, from zero to one ($\hat{t}$) [dimensionless]
    BumplessTransferStartInstant: NotationSymbol  # instant at which a bumpless-transfer blend begins, the zero of its normalized progress ($t_{\text{start}}$) [s]
    BurstArrivalCount: NotationSymbol  # number of tasks that arrive together in one burst ($N_{\text{burst}}$)
    BurstSurvivalTime: NotationSymbol  # time a burst takes to heat a winding from its starting temperature to the insulation limit ($t_{\text{limit}}$) [s]
    BusBitRate: NotationSymbol  # bit rate of a classical single-rate fieldbus ($R_{\text{bit}}$) [bits/s]
    BusCapacitorEnergy: NotationSymbol  # energy stored in the DC bus capacitor bank ($E_C$) [J]
    BusCapacitorVoltage: NotationSymbol  # voltage across the local inverter bus capacitor bank ($V_C$) [V]
    BusSerializationTime: NotationSymbol  # time to serialize a sensor frame across the camera bus to the host ($t_{\text{bus}}$) [s]
    ButterflyAllReduceTime: NotationSymbol  # time to complete an AllReduce by recursive halving-doubling, the butterfly algorithm ($T_{\text{butterfly}}$) [s]
    ByteAlphabet: NotationSymbol  # alphabet of byte values, zero to 255, that a byte-level constraint automaton reads ($\Sigma$)
    ByteString: NotationSymbol  # finite string of bytes over the alphabet of a byte-level constraint automaton ($u$)
    ByzantineClientCount: NotationSymbol  # number of Byzantine, faulty or malicious, clients a robust aggregation rule tolerates ($f$)
    ByzantineFaultCount: NotationSymbol  # number of Byzantine faulty components, which may send arbitrary conflicting messages, that an agreement protocol tolerates ($f$)
    CableFlexCycleCount: NotationSymbol  # number of bending cycles one attempt imposes on the wrist's internal cable bundle ($n_{\text{cable}}$)
    CableRatedFatigueLife: NotationSymbol  # rated fatigue life of the wrist's internal flex cable bundle, in bending cycles ($L_{\text{cable}}$)
    CacheMissStallFraction: NotationSymbol  # fraction of the unloaded permission tick spent stalled on shared-cache misses ($f$) [dimensionless]
    CachedInputTokenCount: NotationSymbol  # number of input tokens in a call that the provider bills at the cached input rate ($n_{\text{cache}}$)
    CachedInputTokenPrice: NotationSymbol  # price of one cached input token ($c_{\text{cache}}$) [USD/token]
    CalibratedAcceptanceEstimate: NotationSymbol  # calibrated estimate of the probability that a search has found an acceptable result, after a given number of admitted units of work ($\hat{p}$) [dimensionless]
    CalibrationBinCount: NotationSymbol  # number of confidence intervals predictions are grouped into to estimate calibration error ($N_{\text{bins}}$)
    CalibrationBinExampleSet: NotationSymbol  # set of evaluation examples whose predicted confidence falls in one calibration bin, indexed by the bin ($B_m$)
    CalibrationBinIndex: NotationSymbol  # position of one confidence bin among the calibration bins, counted from one ($m$)
    CalibrationConstraintMatrix: NotationSymbol  # matrix stacking the two homography constraints of every calibration view, one row per constraint ($\mathbf{V}$)
    CalibrationDataDistribution: NotationSymbol  # joint distribution of inputs and labels from which held-out calibration examples are drawn ($p_{\text{cal}}$)
    CalibrationDataset: NotationSymbol  # set of representative examples run through a model to estimate quantization ranges ($\mathcal{C}_{\text{cal}}$)
    CalibrationExampleCount: NotationSymbol  # number of held-out calibration examples, and so of nonconformity scores, used to set a conformal prediction threshold ($n_{\text{cal}}$)
    CalibrationModelPoint: NotationSymbol  # point of a planar calibration model in homogeneous world coordinates ($\mathbf{M}$)
    CalibrationPoseIndex: NotationSymbol  # position of one pose in the sequence of hand-eye calibration poses ($k$)
    CalibrationRotationAxisCount: NotationSymbol  # number of non-parallel rotation axes hand-eye calibration motions must span ($N$)
    CalibrationSceneIndex: NotationSymbol  # position of one labeled scene within a conformal calibration set ($j$)
    CalibrationTemperature: NotationSymbol  # positive scalar fitted on held-out calibration data that divides a model's logits before softmax to calibrate its confidence ($T_{\text{cal}}$) [dimensionless]
    CalibrationViewCount: NotationSymbol  # number of distinct orientations of a planar calibration rig observed during camera calibration ($N$)
    CallLatency: NotationSymbol  # elapsed time of one model call, from request preparation to normalized response ($T_{\text{call}}$) [s]
    CallPostprocessingTime: NotationSymbol  # time after generation until a model call's normalized response is ready: response transfer, deserialization and schema validation ($T_{\text{post}}$) [s]
    CallPreparationTime: NotationSymbol  # time a runtime spends preparing a model call: assembling context, serializing the payload and dispatching it ($T_{\text{prep}}$) [s]
    CallTailProbability: NotationSymbol  # probability that one call exceeds its per-call tail-latency percentile, such as the 99th ($p_{\text{tail}}$) [dimensionless]
    CallUsefulWorkTime: NotationSymbol  # useful execution time one call performs inside its envelope, excluding setup ($W$) [s]
    CameraCenterArcPosition: NotationSymbol  # position of a rotating camera's optical center along its arc, whose finite change over an exposure is the travel that blurs the image ($x_{\text{rot}}$) [m]
    CameraExtrinsicRotation: NotationSymbol  # rotation part of a camera's extrinsic calibration, taking camera-frame vectors into the body frame ($\mathbf{R}_{bc}$) [dimensionless]
    CameraExtrinsicTransform: NotationSymbol  # rigid transform of a camera's extrinsic calibration, mapping camera-frame coordinates into the body frame ($\mathbf{T}_{bc}$)
    CameraExtrinsicTranslation: NotationSymbol  # translation of the camera-to-body extrinsic, the camera origin expressed in the body frame ($\mathbf{p}_{bc}$) [m]
    CameraFrameDepth: NotationSymbol  # depth of a point along the camera's optical axis, the third camera-frame coordinate ($Z_c$) [m]
    CameraFrameHorizontalCoordinate: NotationSymbol  # horizontal coordinate of a point in a camera's optical frame ($X_c$) [m]
    CameraFramePeriod: NotationSymbol  # time between successive frames of a camera at its sampling rate ($T_{\text{sample}}$) [s]
    CameraFramePoint: NotationSymbol  # position of a point expressed in the camera coordinate frame ($\mathbf{p}_c$) [m]
    CameraFramePointCovariance: NotationSymbol  # covariance of a point's position error in the camera's optical frame, after unprojection of its pixel and depth measurement ($\boldsymbol{\Sigma}_c$) [m^2]
    CameraFrameVerticalCoordinate: NotationSymbol  # vertical coordinate of a point in a camera's optical frame ($Y_c$) [m]
    CameraIdentifierEmbedding: NotationSymbol  # learned embedding added to every patch token of one camera that identifies which camera it came from ($\mathbf{e}_{\text{cam}}$)
    CameraImage: NotationSymbol  # camera image as a pixel array of height by width by color channels, the input a vision encoder reads ($\mathbf{I}$)
    CameraImuTimeOffset: NotationSymbol  # uncalibrated time offset between a camera's exposure and the inertial unit's sampling ($\Delta t_{\text{imu}}$) [s]
    CameraIntrinsicCalibrationMatrix: NotationSymbol  # three-by-three matrix of camera intrinsics, focal lengths and principal point, that maps camera-frame points to pixels ($\mathbf{K}$) [pixels]
    CameraIntrinsicInverseTranspose: NotationSymbol  # transpose of the inverse of a camera's intrinsic matrix, which carries an essential matrix into pixel coordinates on the left of the fundamental matrix ($\mathbf{K}^{-\top}$) [1/pixel]
    CameraLateralStandardDeviation: NotationSymbol  # standard deviation of an unprojected point position along the first camera axis ($\sigma_{x_c}$) [m]
    CameraMountTiltAngle: NotationSymbol  # orientation angle of a camera's mount relative to its calibrated extrinsic pose ($\theta_{\text{ext}}$) [deg]
    CameraObjectTransform: NotationSymbol  # rigid transform giving a target object's pose in the camera frame ($\mathbf{T}_{\text{camera}}^{\text{object}}$)
    CameraOpticalCenter: NotationSymbol  # position of a camera's optical center, the point through which all its projection rays pass ($\mathbf{o}$) [m]
    CameraReferenceFrame: NotationSymbol  # coordinate frame attached to a camera sensor ($\mathcal{F}_{\text{cam}}$)
    CameraRotationColumn: NotationSymbol  # one column of the rotation matrix of a camera pose relative to the calibration model, numbered r_1 to r_3 ($\mathbf{r}$) [dimensionless]
    CameraSampleTimestamp: NotationSymbol  # timestamp of a camera exposure on the clock used to fuse it with joint data ($t_{\text{cam}}$) [s]
    CameraSkewCoefficient: NotationSymbol  # skew entry of a camera intrinsic matrix, coupling the image axes ($\gamma$) [pixels]
    CameraTargetTransform: NotationSymbol  # rigid transform mapping calibration-target coordinates into the camera frame, from target unprojection ($\mathbf{T}_{\text{cam}}^{\text{target}}$)
    CameraToWorldRotation: NotationSymbol  # rotation part of the camera-to-world extrinsic transform, the camera's orientation in the world frame ($\mathbf{R}_{WC}$) [dimensionless]
    CameraToWorldTransform: NotationSymbol  # rigid transform mapping camera-frame coordinates into the world frame, the camera's pose in the world ($\mathbf{T}_{WC}$)
    CameraToWorldTranslation: NotationSymbol  # translation part of the camera-to-world extrinsic transform, the camera center's position in the world frame ($\mathbf{t}_{WC}$) [m]
    CampaignActiveTime: NotationSymbol  # total active operating time summed over the trials of a campaign ($T_{\text{active}}$) [s]
    CanHighLineVoltage: NotationSymbol  # voltage on the high line of a CAN differential pair ($V_{\text{CANH}}$) [V]
    CanLowLineVoltage: NotationSymbol  # voltage on the low line of a CAN differential pair ($V_{\text{CANL}}$) [V]
    CanaryAllocationFraction: NotationSymbol  # share of eligible tasks that a canary's allocation rule assigns to the candidate configuration ($\rho$) [dimensionless]
    CanaryInterventionRate: NotationSymbol  # rate of enforcer or operator interventions per operating hour of a candidate policy in a canary trial ($\lambda_{\text{canary}}$) [1/h]
    CanaryLabelDelay: NotationSymbol  # time from a canary task's admission until its acceptance label is available to the sequential monitor ($t_{\text{label}}$) [s]
    CanarySpeedBound: NotationSymbol  # largest speed the enforcer permits a candidate policy during a restricted canary trial, set below the nominal rating ($v_{\text{canary}}$) [m/s]
    CanaryStageDuration: NotationSymbol  # time a canary rollout stage needs to collect its planned observations at its traffic share ($T_{\text{stage}}$) [s]
    CanaryTrafficFraction: NotationSymbol  # share of production traffic a canary stage routes to the candidate ($p_{\text{stage}}$) [dimensionless]
    CandidateAcceptanceEstimate: NotationSymbol  # observed acceptance share of the candidate configuration over the tasks of a paired comparison, its accepted count divided by the task count ($\hat{p}_{\text{cand}}$) [dimensionless]
    CandidateActionVector: NotationSymbol  # candidate action vector over which a regression objective is minimized ($\hat{\mathbf{a}}$)
    CandidateArchitecture: NotationSymbol  # one candidate network architecture drawn from a search space ($a$)
    CandidateAvailabilityEstimate: NotationSymbol  # unbiased estimate of pass@k for one task, the share of claim-size subsets of its graded runs that contain at least one pass ($\widehat{q}$) [dimensionless]
    CandidateAvailabilityProbability: NotationSymbol  # probability that at least one of a pool of independent candidates passes its check, indexed by the pool size or cap ($q$) [dimensionless]
    CandidateCheckingCost: NotationSymbol  # cost in dollars of checking one generated candidate against the acceptance criterion ($v$) [USD]
    CandidateClientCount: NotationSymbol  # number of clients a federated server invites to a round, over-selected above the number of updates it needs ($K_{\text{candidates}}$)
    CandidateControlCommand: NotationSymbol  # control command a candidate policy would have proposed while it runs in shadow mode, logged but never dispatched ($\mathbf{u}^{\text{cand}}$) [varies]
    CandidateControlCommandDerivative: NotationSymbol  # time derivative of the candidate policy's shadow-mode control command, its command rate ($\dot{\mathbf{u}}^{\text{cand}}$) [varies]
    CandidateDefectRate: NotationSymbol  # share of generated candidates that violate the specified property, before any check ($\epsilon$) [dimensionless]
    CandidateGenerationCost: NotationSymbol  # cost in dollars of generating one candidate, before it is checked ($g$) [USD]
    CandidateIndex: NotationSymbol  # position of one candidate within a pool of candidates ($i$)
    CandidateIntervalLowerLimit: NotationSymbol  # lower limit of the single-proportion Wilson interval for the candidate configuration's acceptance proportion ($l_{\text{cand}}$) [dimensionless]
    CandidateIntervalUpperLimit: NotationSymbol  # upper limit of the single-proportion Wilson interval for the candidate configuration's acceptance proportion ($u_{\text{cand}}$) [dimensionless]
    CandidateLowerLimitDistance: NotationSymbol  # downward distance from the candidate configuration's observed acceptance share to the lower limit of its Wilson interval ($\delta^-_{\text{cand}}$) [dimensionless]
    CandidateOnlyAcceptedTaskCount: NotationSymbol  # number of tasks in a paired comparison that the candidate accepts and the baseline does not, one of the two discordant counts ($c$)
    CandidatePolicyAcceptanceProbability: NotationSymbol  # acceptance probability of the alternative routing policy under consideration ($p_{Y}$) [dimensionless]
    CandidatePolicyAction: NotationSymbol  # action a candidate policy computes at step t in shadow mode, logged but never dispatched ($\mathbf{a}_t^{\text{cand}}$)
    CandidatePolicyCost: NotationSymbol  # expected additional monetary cost of the alternative routing policy ($C_{Y}$) [USD]
    CandidatePoolSize: NotationSymbol  # number of candidates generated for one decision in best-of-N sampling, or the breadth of concurrent candidate streams ($N$)
    CandidateProposal: NotationSymbol  # candidate proposal a learned proposer publishes for one step, such as a waypoint chunk or spline, before the permission path admits it ($p$)
    CandidateRoutingPolicy: NotationSymbol  # alternative routing policy compared with a baseline policy by a router ($Y$)
    CandidateScoreVector: NotationSymbol  # normalized detector scores of the candidate referents, nonnegative and summing to one ($\mathbf{c}$) [dimensionless]
    CandidateScoringCost: NotationSymbol  # cost of running the active model's inference on one candidate sample ($c_{\text{infer}}$) [USD/sample]
    CandidateScoringError: NotationSymbol  # error a learned scorer adds to one candidate's true utility, independent across candidates with zero mean ($\epsilon$) [dimensionless]
    CandidateSkill: NotationSymbol  # one low-level skill from a robot's library that a language-grounded planner may select ($a$)
    CandidateSpeedCommand: NotationSymbol  # speed a candidate proposal commands before the speed governor clamps it to the envelope ceiling ($v_{\text{cand}}$) [m/s]
    CandidateTailCompletionTime: NotationSymbol  # empirical 99th-percentile task completion time of the candidate configuration across an evaluation cohort ($\widehat{T}_{c,0.99}$) [s]
    CandidateToolCallDistribution: NotationSymbol  # distribution of the candidate configuration's tool calls over the kinds of tool call, under one declared action-counting policy ($p_{c}$)
    CandidateTrackingErrorStandardDeviation: NotationSymbol  # standard deviation of the tracking error a candidate policy shows in a canary trial ($\sigma_{\text{track}}$) [m]
    CandidateTrajectoryGoodput: NotationSymbol  # trajectory goodput of the candidate configuration over a declared observation interval ($\mathcal{G}_{c}$) [tasks/s]
    CandidateUpperLimitDistance: NotationSymbol  # upward distance from the candidate configuration's observed acceptance share to the upper limit of its Wilson interval ($\delta^+_{\text{cand}}$) [dimensionless]
    CandidateUtility: NotationSymbol  # genuine task utility, the true quality, of one candidate, which a scorer estimates with error ($U$) [dimensionless]
    CandidateUtilityMean: NotationSymbol  # mean true utility of the candidates in a pool ($\mu$) [dimensionless]
    CandidateUtilityStandardDeviation: NotationSymbol  # standard deviation of the true utility of the candidates in a pool ($\sigma_{U}$) [dimensionless]
    CandidateViolationCount: NotationSymbol  # number of verified safety or contract violations, such as unauthorized modifications or privilege escalations, the candidate configuration incurs in an evaluation cohort ($V_{c}$)
    CandidateWinCount: NotationSymbol  # number of discordant pairs won by one configuration, the variable over which the exact McNemar tail probability sums ($j$)
    CapabilityGrant: NotationSymbol  # capability grant, the rights, objects, condition, and expiry that together describe one delegated authority ($\Gamma$)
    CaptureAgeEmbedding: NotationSymbol  # Fourier-feature embedding of the time elapsed since an observation's capture ($\mathbf{e}_{\Delta t}$)
    CaptureSampleIndex: NotationSymbol  # index of the IMU sample whose capture time is nearest a delayed observation's capture epoch ($t_0$)
    CaptureSampleKalmanGain: NotationSymbol  # Kalman gain computed at the IMU sample nearest a delayed observation's capture epoch ($\mathbf{G}_{t_0}$)
    CaptureSampleMeasurementMatrix: NotationSymbol  # measurement matrix evaluated at the IMU sample nearest a delayed observation's capture epoch ($\mathbf{H}_{t_0}$)
    CaptureSamplePredictedCovariance: NotationSymbol  # error covariance stored in the ring buffer for the IMU sample nearest a delayed observation's capture epoch ($\mathbf{P}_{t_0|t_0-1}$)
    CaptureSamplePredictedEstimate: NotationSymbol  # state prediction stored in the ring buffer for the IMU sample nearest a delayed observation's capture epoch ($\hat{\mathbf{x}}_{t_0|t_0-1}$)
    CaptureSampleUpdatedCovariance: NotationSymbol  # error covariance at the capture sample after the delayed observation is applied there ($\mathbf{P}_{t_0|t_0}$)
    CaptureSampleUpdatedEstimate: NotationSymbol  # state estimate at the capture sample after the delayed observation is applied there ($\hat{\mathbf{x}}_{t_0|t_0}$)
    CaptureTimestamp: NotationSymbol  # capture timestamp of an observation: the midpoint of the sensor's exposure or firing window, the instant the observation describes ($t_{\text{capture}}$) [s]
    CapturedSensorObservation: NotationSymbol  # raw sensory input, such as pixels or a point cloud, captured at the capture instant and fed to a policy ($\mathbf{z}_{t_0}$)
    CartesianAcceleration: NotationSymbol  # acceleration of a point in three-dimensional space, the second time derivative of its position ($\ddot{\mathbf{p}}$) [m/s^2]
    CartesianPosition: NotationSymbol  # position of a point in three-dimensional space ($\mathbf{p}$) [m]
    CartesianXCoordinate: NotationSymbol  # first Cartesian component of a three-dimensional position ($x$) [m]
    CartesianYCoordinate: NotationSymbol  # second Cartesian component of a three-dimensional position ($y$) [m]
    CartesianZCoordinate: NotationSymbol  # third Cartesian component of a three-dimensional position ($z$) [m]
    CascadeEscalationProbability: NotationSymbol  # probability that an edge-cloud cascade escalates an input to cloud inference, measured on the deployment input distribution ($p_{\text{escalate}}$) [dimensionless]
    CascadeGateCost: NotationSymbol  # mean monetary cost of the acceptance gate at one tier of a model cascade ($g$) [USD]
    CascadeGateDuration: NotationSymbol  # mean duration of the acceptance gate at one tier of a model cascade ($v$) [s]
    CascadeInferenceEnergy: NotationSymbol  # average energy an edge-cloud cascade spends per input, local inference plus escalated transmission and cloud inference ($E_{\text{cascade}}$) [J]
    CascadeTaskCost: NotationSymbol  # monetary cost of one task through a gated model cascade, every reached tier's generation and gate ($C_{\text{cascade}}$) [USD]
    CascadeTaskDuration: NotationSymbol  # elapsed time of one task through a serial gated model cascade ($T_{\text{cascade}}$) [s]
    CascadeTierCost: NotationSymbol  # mean monetary cost of generation and associated execution at one tier of a model cascade, conditional on the task reaching that tier, excluding its acceptance gate ($c^{\text{tier}}$) [USD]
    CascadeTierDuration: NotationSymbol  # mean duration of the generation step at one tier of a model cascade, conditional on the task reaching that tier, excluding its acceptance gate ($t$) [s]
    CascadeTierIndex: NotationSymbol  # position of one tier in a model cascade, counted from one for the cheapest tier ($j$)
    CascadedBarrierCertificate: NotationSymbol  # one certificate in the cascade of a high-order control barrier function, the previous certificate's time derivative plus a class-K function of it, indexed by order ($\psi$) [varies]
    CascadedBarrierClassKFunction: NotationSymbol  # class-K function applied to one certificate of a high-order control barrier function cascade to form the next, indexed by stage ($\alpha$)
    CascadedBarrierDerivative: NotationSymbol  # time derivative of one certificate of a high-order control barrier function cascade along the closed-loop dynamics ($\dot{\psi}$) [varies]
    CaseAmbientThermalResistance: NotationSymbol  # thermal resistance from the motor frame and housing to ambient air ($R_{\text{th,case-amb}}$) [K/W]
    CaseTemperature: NotationSymbol  # temperature of a motor or package case ($T_{\text{case}}$) [degC]
    CaseThermalCapacitance: NotationSymbol  # heat capacity of the motor frame and housing node in a multi-node motor thermal network ($C_{\text{case}}$) [J/K]
    CaseThermalTimeConstant: NotationSymbol  # thermal time constant of the motor case stage of a multi-node motor thermal model ($\tau_{\text{th,case}}$) [s]
    CatalogDefinitionTokens: NotationSymbol  # total tokens of tool definitions a staging policy sends across all its model calls, excluding other request content ($S_{\text{catalog}}$) [tokens]
    CatalogStagingCallCount: NotationSymbol  # number of model calls whose requests stage tool definitions, over which a catalog's token cost is counted ($N$)
    CategoryRecordCount: NotationSymbol  # number of retained records in one trajectory category of a curated corpus ($n$)
    CausalInterventionMask: NotationSymbol  # binary mask over a recorded intervention trajectory that is one where a step may serve as an imitation target and zero elsewhere ($\mathcal{M}$) [dimensionless]
    CausalMediator: NotationSymbol  # mediator variable of a structural causal model, a variable on a causal path from the protected attribute to the decision ($M$)
    CausalTokenPosition: NotationSymbol  # position of one token within a causal prefill, equal to the number of earlier tokens whose keys and values it attends to ($s$)
    CbsRuntimeBudget: NotationSymbol  # runtime a Constant Bandwidth Server task declares it may execute within each period, beyond which the kernel throttles it ($Q$) [s]
    CellOccupancyProbability: NotationSymbol  # probability that one cell of an occupancy map is occupied ($p$) [dimensionless]
    CenterCount: NotationSymbol  # number of centers a k-center selection places, each a retained sample ($k$)
    CentralizedTrainingEnergy: NotationSymbol  # total energy of training centrally, every client uploading raw data plus one cloud training run ($E_{\text{centralized}}$) [J]
    CentroidalAngularMomentum: NotationSymbol  # angular momentum of the whole robot about its center of mass ($\mathbf{h}_{\text{ang}}$) [kg*m^2/s]
    CentroidalAngularMomentumDerivative: NotationSymbol  # time derivative of the centroidal angular momentum, the net external moment about the center of mass ($\dot{\mathbf{h}}_{\text{ang}}$) [N*m]
    CentroidalLinearMomentum: NotationSymbol  # total linear momentum of the robot, its mass times its center-of-mass velocity ($\mathbf{h}_{\text{lin}}$) [kg*m/s]
    CentroidalLinearMomentumDerivative: NotationSymbol  # time derivative of the centroidal linear momentum, the net external force on the robot ($\dot{\mathbf{h}}_{\text{lin}}$) [N]
    CentroidalMomentum: NotationSymbol  # six-component centroidal momentum of a robot, its linear momentum stacked with its angular momentum about the center of mass ($\mathbf{h}_G$)
    CentroidalMomentumMatrix: NotationSymbol  # configuration-dependent matrix mapping generalized velocities to the six-component centroidal momentum ($\mathbf{A}_G$)
    CentroidalMomentumMatrixDerivative: NotationSymbol  # time derivative of the centroidal momentum matrix, a function of configuration and generalized velocity ($\dot{\mathbf{A}}_G$)
    CentroidalMomentumRate: NotationSymbol  # rate of change of the six-component centroidal momentum, equal to the net external wrench about the center of mass ($\dot{\mathbf{h}}_G$)
    CertifiedCommandWindow: NotationSymbol  # sequence of control commands the safety filter admitted, from one step through the end of an action-chunk horizon ($\mathbf{u}^*_{t:t+H-1}$)
    CertifiedControlAction: NotationSymbol  # control command after the safety filter has projected it onto the admissible set ($\mathbf{u}^*$)
    CertifiedRadius: NotationSymbol  # radius within which no input perturbation can change a certified prediction, in the norm the certificate states ($R$) [varies]
    ChainRuleInput: NotationSymbol  # input variable of the scalar chain-rule illustration v = 3u with u = x squared ($x$) [dimensionless]
    ChainRuleIntermediate: NotationSymbol  # intermediate variable u = x squared of the scalar chain-rule illustration ($u$) [dimensionless]
    ChainRuleOutput: NotationSymbol  # output variable v = 3u of the scalar chain-rule illustration ($v$) [dimensionless]
    ChainedOperationCount: NotationSymbol  # number of consecutive operations in a chain that each read and write the same intermediate tensor, the chain kernel fusion merges ($n_{\text{ops}}$)
    ChanceAnnotatorAgreement: NotationSymbol  # share of items on which two annotators would agree by chance, given each annotator's label frequencies ($p_e$) [dimensionless]
    ChannelAuthorityState: NotationSymbol  # discrete authority state of one actuator channel: autonomous, handover pending, blend, human manual, fallback or protective ($S$)
    ChannelCommand: NotationSymbol  # scalar command of one actuator channel, such as the base acceleration or one joint torque ($u$) [varies]
    ChannelStoredEnergy: NotationSymbol  # energy held in flight by the waves inside a delayed communication channel ($E_{\text{channel}}$) [J]
    CharacteristicPolynomial: NotationSymbol  # characteristic polynomial of a square matrix, the determinant of lambda times identity minus the matrix ($p$)
    CharacteristicPolynomialCoefficient: NotationSymbol  # coefficient of one power of lambda in the characteristic polynomial of a square matrix ($c$) [varies]
    ChassisThermalResistance: NotationSymbol  # thermal resistance of conduction through a metal enclosure to the ambient air ($R_{\text{th,chassis}}$) [K/W]
    CheapConfigurationAttemptCost: NotationSymbol  # all-in monetary cost of one attempt by the cheaper, lightweight agent configuration ($c_{\text{small}}$) [USD]
    CheapConfigurationPassProbability: NotationSymbol  # probability that one attempt by the cheaper, lightweight agent configuration is accepted ($p_{\text{small}}$) [dimensionless]
    CheckIntervalStepCount: NotationSymbol  # number of generation steps between consecutive verification checks, the size of one checked block ($m$)
    CheckPassProbability: NotationSymbol  # probability that one admission predicate passes a candidate record, conditional on every earlier predicate in the route having passed ($p^{\text{chk}}$) [dimensionless]
    CheckSlotTime: NotationSymbol  # expected verifier slot time one admission predicate consumes for a record that reaches it ($T$) [s]
    CheckedSignal: NotationSymbol  # trajectory signal a pre-execution checker compares with its limit, such as one joint's position, velocity or torque ($x$) [varies]
    CheckedSignalLimit: NotationSymbol  # limit a pre-execution checker holds one trajectory signal under, such as a joint position, contracted velocity or torque limit ($L_{\text{limit}}$) [varies]
    CheckpointAttemptDuration: NotationSymbol  # duration of one uninterrupted compute-plus-checkpoint attempt, the checkpoint interval plus the write time ($T_{\text{att}}$) [s]
    CheckpointExtraPauseTime: NotationSymbol  # time training pauses per checkpoint beyond the write itself, for memory allocation, coordination and serialization ($T_{\text{pause}}$) [s]
    CheckpointInterval: NotationSymbol  # useful compute time between two completed checkpoints ($T_{\text{ckpt}}$) [s]
    CheckpointLoadTime: NotationSymbol  # time to read the latest checkpoint state from storage back into accelerator memory during recovery ($T_{\text{load}}$) [s]
    CheckpointLogPosition: NotationSymbol  # authoritative log position to which a published checkpoint binds its record and workspace snapshots ($j_{\text{ckpt}}$)
    CheckpointOrdinalIndex: NotationSymbol  # position of one checkpoint among those a trajectory publishes, counted from one ($m$)
    CheckpointOverheadFraction: NotationSymbol  # fraction of training time lost to saving checkpoints, excluding replay of lost work ($f_{\text{ckpt}}$) [dimensionless]
    CheckpointOverheadRatio: NotationSymbol  # approximate overhead per unit of useful execution time from checkpoint writes, lost work and restarts, as a function of the checkpoint interval ($w$) [dimensionless]
    CheckpointRecoveryOverheadTime: NotationSymbol  # total time a trajectory loses over a failure trace to checkpoint pauses, checkpoint restores and suffix reconstruction, as a function of the checkpoint stride ($T_{\text{overhead}}$) [s]
    CheckpointRestoreTime: NotationSymbol  # fixed time to restore the latest checkpoint after one crash, before any suffix reconstruction ($T_{\text{restore}}$) [s]
    CheckpointStride: NotationSymbol  # number of turns between consecutive checkpoints of a trajectory, the checkpoint stride ($k$)
    CheckpointSuffixReadTime: NotationSymbol  # time to fetch and verify a trajectory's checkpoint and its log suffix during resume ($T_{\text{read}}$) [s]
    CheckpointWasteFraction: NotationSymbol  # fraction of training time wasted on checkpoint writes and on replaying work lost to failures, as a function of the checkpoint interval ($W$) [dimensionless]
    CheckpointWriteTime: NotationSymbol  # time execution pauses to write one checkpoint ($T_{\text{write}}$) [s]
    ChildBatchSize: NotationSymbol  # number of children a search step proposes in parallel in one round when it needs only one surviving child ($n_{\text{batch}}$)
    ChildBudgetAllocation: NotationSymbol  # monetary allocation one delegated child node's budget account holds as its ceiling, set to the amount its parent delegated ($C_{\max,C}$) [USD]
    ChildCapabilityGrant: NotationSymbol  # capability grant derived from a parent grant, which may permit no more than its parent ($\Gamma_{\text{child}}$)
    ChildFreeBalance: NotationSymbol  # unreserved monetary balance of one delegated child node's budget account ($F_{C}$) [USD]
    ChildPruningProbability: NotationSymbol  # probability that an intermediate check prunes one proposed child of a search node, independently of its siblings ($p_{\text{prune}}$) [dimensionless]
    ChildReservedFunds: NotationSymbol  # monetary funds one delegated child node's budget account withholds for its outstanding operations and descendants ($R_{C}$) [USD]
    ChildSettledCharges: NotationSymbol  # gross monetary charges settled into one delegated child node's budget account ($E_{C}$) [USD]
    ChinchillaIrreducibleLoss: NotationSymbol  # constant term of Hoffmann et al.'s parametric loss fit, the loss no amount of parameters or tokens removes ($E$) [dimensionless]
    ChirpDuration: NotationSymbol  # time over which one FMCW radar chirp sweeps its bandwidth ($T_c$) [s]
    ChirpSlope: NotationSymbol  # rate at which an FMCW radar chirp's frequency rises, bandwidth over chirp duration ($S$) [Hz/s]
    ChopperClampVoltage: NotationSymbol  # bus voltage at which a dynamic brake chopper switches its absorption resistor across the bus ($V_{\text{clamp}}$) [V]
    ChopperTurnOffVoltage: NotationSymbol  # lower hysteresis threshold at which the brake chopper disconnects its resistor ($V_{\text{clamp,off}}$) [V]
    ChopperTurnOnVoltage: NotationSymbol  # upper hysteresis threshold at which the brake chopper switches its resistor onto the bus ($V_{\text{clamp,on}}$) [V]
    ChunkActionDimension: NotationSymbol  # number of scalar components in one action chunk, the action dimension times the chunk horizon ($d_{\text{chunk}}$)
    ChunkEnsemblingWeight: NotationSymbol  # weight temporal ensembling gives the prediction of one overlapping action chunk, indexed by that chunk's position ($w$) [dimensionless]
    ChunkExecutionHorizon: NotationSymbol  # number of actions of each predicted chunk the controller executes before the policy is queried again ($H_{\text{exec}}$)
    ChunkImitationLoss: NotationSymbol  # behavioral cloning objective that sums a per-step regression penalty over every step of a demonstrated action chunk ($\mathcal{L}_{\text{chunk}}$) [dimensionless]
    ChunkLease: NotationSymbol  # longest time an admitted action chunk may keep driving the actuators without renewal ($T_{\text{lease}}$) [s]
    ChunkReactionLatency: NotationSymbol  # worst-case delay from a new observation until a revised chunk proposal built from it is available, one renewal period plus one proposer inference ($T_{\text{reaction}}$) [s]
    ChunkRenewalPeriod: NotationSymbol  # nominal interval between successive admissions of the chunk policy's proposals ($T_{\text{chunk}}$) [s]
    ChunkRenewalRate: NotationSymbol  # rate at which the chunk policy starts a new evaluation from fresh sensory frames ($f_{\text{replan}}$) [Hz]
    ChunkStepIndex: NotationSymbol  # position of one step within an action chunk, counted from zero ($i$)
    ClampedControlCommand: NotationSymbol  # control command after independent per-axis clamping, which ignores the coupled barrier constraint ($\mathbf{u}_{\text{clip}}$)
    ClassCount: NotationSymbol  # number of classes a classifier distinguishes, or vocabulary entries it scores, the length of its softmax output ($K$)
    ClassIndex: NotationSymbol  # index of an output class ($j$)
    ClassKFunction: NotationSymbol  # extended class-K function that bounds how fast the barrier value may decrease ($\kappa$)
    ClassKFunctionSet: NotationSymbol  # class of continuous strictly increasing functions that are zero at zero, the class-K functions ($\mathcal{K}$)
    ClassProbabilityVector: NotationSymbol  # vector of class probabilities over a classifier's classes, nonnegative and summing to one, a point of the probability simplex ($\mathbf{p}$)
    ClassificationThreshold: NotationSymbol  # score threshold above which a classifier outputs the positive class ($\gamma_{\text{cls}}$) [dimensionless]
    CleanLabelSampleCount: NotationSymbol  # number of training samples with clean labels needed to reach a target error ($D_{\text{clean}}$)
    CleanTrainingDataset: NotationSymbol  # original training dataset before an attacker adds poisoning data ($\mathcal{S}$)
    CleanupGraphEdgeSet: NotationSymbol  # set of directed edges of a cleanup graph, each requiring one cleanup operation to settle before another dispatches ($E$)
    CleanupGraphVertexSet: NotationSymbol  # set of vertices of a cleanup graph, one per admitted cleanup operation ($V$)
    ClearDistance: NotationSymbol  # distance between the machine and the boundary it must stop before ($D_{\text{clear}}$) [m]
    ClearanceAcceleration: NotationSymbol  # second time derivative of the clearance distance to the boundary ($\ddot{D}_{\text{clear}}$) [m/s^2]
    ClearanceAllowance: NotationSymbol  # one allowance of the stopping budget, such as the localization allowance or the side clearance, against which a depth error scale is compared ($C$) [m]
    ClearanceExhaustionTime: NotationSymbol  # time for a constant heading offset's lateral drift to consume the side clearance ($t_{\text{impact}}$) [s]
    ClearanceInset: NotationSymbol  # separation a consumer holds between an estimated obstacle position and the machine's envelope so the true obstacle stays at least the required clearance away, except with a stated tail probability ($C_{\text{inset}}$) [m]
    ClearanceQueryTime: NotationSymbol  # time one clearance query takes at one evaluation sphere of the machine's body ($t_{\text{query}}$) [s]
    ClearanceRate: NotationSymbol  # rate of change of the clearance distance to the boundary, negative while approaching ($\dot{D}_{\text{clear}}$) [m/s]
    ClearanceSpeedClamp: NotationSymbol  # largest speed at which the stopping distance still fits within the available clearance, solved from the clearance inequality ($v_{\text{clamp}}$) [m/s]
    ClearanceTimeBudget: NotationSymbol  # time available at constant speed before the clearance remaining after the braking distance is used up, the budget the teleoperation delays must fit ($T_{\text{total}}$) [s]
    ClfClassKFunction: NotationSymbol  # extended class-K function of the Lyapunov value that sets the minimum rate at which a control Lyapunov function must decrease ($\alpha_V$)
    ClfDescentControlSet: NotationSymbol  # set of admissible inputs that satisfy the control Lyapunov decrease condition at a state, a closed affine half-space ($\mathcal{U}_{\text{CLF}}$)
    ClientIndex: NotationSymbol  # position of one client among the clients of a federated round or secure aggregation ($k$)
    ClientLocalDataset: NotationSymbol  # local dataset held on one federated client, indexed by client ($\mathcal{D}$)
    ClientLocalTrainingEnergy: NotationSymbol  # energy one federated client spends on local training in one round ($E_{\text{client}}$) [J]
    ClientObjectiveWeight: NotationSymbol  # weight a client's local loss receives in the federated global objective, indexed by client ($w$) [dimensionless]
    ClientSampleCount: NotationSymbol  # number of training samples on one federated client, written with the client index ($n$)
    ClientUpdateUploadEnergy: NotationSymbol  # energy one federated client spends uploading its model update for one round ($E_{\text{upload}}$) [J]
    ClippedSurrogateObjective: NotationSymbol  # per-token clipped surrogate: the smaller of the ratio-weighted coefficient and its clipped counterpart ($\mathcal{M}$) [dimensionless]
    ClippingRangeLowerBound: NotationSymbol  # lower end of the real-valued range a quantizer represents, below which values are clipped ($\alpha$) [varies]
    ClippingRangeUpperBound: NotationSymbol  # upper end of the real-valued range a quantizer represents, above which values are clipped ($\beta$) [varies]
    ClockConversionBound: NotationSymbol  # bound on the error of converting a sensor's capture timestamp to the clock of the safety microcontroller that hosts the permission path ($\epsilon_{\text{sync}}$) [s]
    ClockConversionIntercept: NotationSymbol  # intercept of the affine mapping that converts a capture-clock time to the safety microcontroller's clock ($b_{\text{clk}}$) [s]
    ClockConversionSlope: NotationSymbol  # slope of the affine mapping that converts a capture-clock time to the safety microcontroller's clock ($a$) [dimensionless]
    ClockFrequency: NotationSymbol  # rate at which a processor's clock ticks, setting how many switching cycles occur per second ($f$) [Hz]
    ClockOffset: NotationSymbol  # difference between a follower clock's reading and the leader clock's at the same instant ($\theta_{\text{offset}}$) [s]
    ClockPeriod: NotationSymbol  # period of the clock of a synchronous digital pipeline, the time one stage has to settle ($T_{\text{clk}}$) [s]
    ClockSynchronizationSkew: NotationSymbol  # rate at which two clocks drift apart, such as the application processor and the safety microcontroller ($\delta_{\text{skew}}$) [ppm]
    ClosedLoopTrajectoryDensity: NotationSymbol  # joint density of an executed closed-loop trajectory under a candidate policy, its permission path and the physical plant, given the trial's environment, plant and permission-path parameters ($p_{\text{cl}}$)
    ClosingSpeedBound: NotationSymbol  # largest speed at which two moving bodies approach each other ($v_{\text{close}}$) [m/s]
    CloudInferenceEnergy: NotationSymbol  # energy a cloud-hosted model spends serving one inference ($E_{\text{cloud}}$) [J]
    CloudOffloadEnergy: NotationSymbol  # energy a device spends transmitting an input to the cloud for remote inference ($E_{\text{transmit}}$) [J]
    CloudTrainingRunEnergy: NotationSymbol  # energy of one centralized training run in the cloud ($E_{\text{cloud,train}}$) [J]
    ClusterCount: NotationSymbol  # number of clusters, and centroids, a k-means clustering maintains ($k$)
    ClusterHeadroom: NotationSymbol  # number of accelerators in a cluster free to run a deferrable job at a given time ($N$)
    ClusterNodeCount: NotationSymbol  # number of nodes in a cluster or job ($N_{\text{nodes}}$)
    ClusterPower: NotationSymbol  # electrical power a training cluster draws, accelerators plus hosts ($P_{\text{cluster}}$) [W]
    CoarseCellCount: NotationSymbol  # number of coarse cells, each with its centroid and inverted list, into which an inverted-file vector index partitions its vectors ($C$)
    CodecDecodeTime: NotationSymbol  # time to decompress a compressed payload after it arrives ($T_{\text{decode}}$) [s]
    CodecEncodeTime: NotationSymbol  # time to compress a payload before it is sent ($T_{\text{encode}}$) [s]
    CodecOverheadTime: NotationSymbol  # compute time a compression scheme spends encoding a payload before sending and decoding it after arrival ($T_{\text{overhead}}$) [s]
    CognitivePolicyAction: NotationSymbol  # action the learned policy emits at step t, before any safety check ($\mathbf{a}$)
    CoherencyCoefficient: NotationSymbol  # universal scalability law's cost of pairwise reconciliation between workers ($\kappa_{\text{coh}}$) [dimensionless]
    ColdCopperLossPower: NotationSymbol  # copper loss of a winding at its reference resistance, before copper's resistance rise ($P_0$) [W]
    ColdResetCycleTime: NotationSymbol  # environment-slot cycle time when each episode ends with a cold reset ($T_{\text{cycle,cold}}$) [s]
    CollaborativeContactForce: NotationSymbol  # peak force the robot applies to a human body region in a collaborative contact, as a function of approach speed, the robot effective mass and the body-region stiffness ($F_{\text{contact}}$) [N]
    CollectedPhotocharge: NotationSymbol  # total electric charge a photodiode accumulates from incident light over one exposure ($Q$) [A*s]
    CollectionCohortCost: NotationSymbol  # total monetary spend of one collection cohort up to an accounting cutoff: environments, generation, sandboxes, verification, review, redaction and storage ($C_{\text{cohort}}$) [USD]
    CollectionStateVisitationDistribution: NotationSymbol  # empirical distribution of states, or state-action pairs, the collecting policy visited while a dataset was logged ($p^{\pi_{\text{col}}}$)
    CollectiveCallStartupLatency: NotationSymbol  # effective fixed startup time one collective call pays, including software and launch overhead, as measured by a matched collective benchmark ($\alpha_{\text{eff}}$) [s]
    CollectivePayloadSize: NotationSymbol  # size of the message each participant contributes to a collective operation ($M$) [bytes]
    CollectiveTensorSize: NotationSymbol  # byte size of the full tensor acted on by a collective operation ($M_{\text{coll}}$) [bytes]
    CollisionHullCount: NotationSymbol  # number of evaluation spheres whose union covers the machine's body ($N_{\text{hulls}}$)
    CollocationDefect: NotationSymbol  # mismatch at one collocation point between the state derivative of the interpolating spline and the plant dynamics, driven to zero by direct collocation ($\Delta$) [varies]
    ColumnAccessLatency: NotationSymbol  # latency from a DRAM column read command to its data on the bus, the only step a row hit pays ($t_{\text{CL}}$) [s]
    ColumnModeComponentVector: NotationSymbol  # column of the column-mode factor matrix for one rank-one component of a CP decomposition ($\mathbf{b}_s$)
    ColumnModeFactorMatrix: NotationSymbol  # factor matrix for the column mode of a tensor decomposition, one row per column index and one column per rank component ($\mathbf{B}$)
    ColumnModeFactorSlice: NotationSymbol  # row of the column-mode factor matrix for one column index, holding its weight on every rank component ($\mathbf{b}$)
    ColumnParallelWeightMatrix: NotationSymbol  # weight matrix of the first layer of a Megatron tensor-parallel pair, split by columns across devices ($\mathbf{A}$)
    CombinedBatch: NotationSymbol  # set of training examples formed by the union of every data-parallel worker's minibatch for one update ($B_{\text{total}}$)
    CombinedPlanValidationAllowance: NotationSymbol  # elapsed timing allowance for combined host geometry validation and local controller admission of a whole candidate plan ($t_{\text{validate}}$) [s]
    CommandComponentCount: NotationSymbol  # number of components of one command vector, in the order and units the static authority manifest fixes ($N_c$)
    CommandRateCeiling: NotationSymbol  # largest rate of change the enforcer allows a channel command, such as the jerk bound of an acceleration command or the torque rate of a torque command ($r$) [varies]
    CommandTransportTime: NotationSymbol  # time for a command to travel over the fieldbus from the permission path to the drive ($T_{\text{bus}}$) [s]
    CommandedActuatorEffort: NotationSymbol  # force or torque a controller commands from one actuator, before clipping to its realizable envelope ($u_{\text{cmd}}$) [varies]
    CommandedApproachAcceleration: NotationSymbol  # acceleration commanded along the approach direction toward an obstacle, positive when it increases the approach speed ($a_{\text{cmd}}$) [m/s^2]
    CommandedJointConfiguration: NotationSymbol  # joint configuration the controller commands, as opposed to the one the encoders measure ($\mathbf{q}_{\text{cmd}}$) [varies]
    CommandedJointCount: NotationSymbol  # number of joints a proposal or trajectory record commands, fixed by the configured count ($N_j$)
    CommandedJointTorque: NotationSymbol  # torque commanded at a single joint in one-joint derivations ($\tau$) [N*m]
    CommandedJointTorqueSlew: NotationSymbol  # time derivative of the torque at a single joint, its torque slew ($\dot{\tau}$) [N*m/s]
    CommandedTrackedPosition: NotationSymbol  # commanded reference position the tracked point should follow along one axis ($r_{\text{cmd}}$) [m]
    CommandedTractiveForce: NotationSymbol  # tractive force a commanded drive torque asks of a wheel at its contact patch, drive torque divided by wheel radius ($F_{\text{cmd}}$) [N]
    CommitBatchRecordCount: NotationSymbol  # number of records one group-commit batch makes durable with a single storage operation ($N_{\text{batch}}$)
    CommitmentInstant: NotationSymbol  # instant after which a receding-horizon plan's prefix is committed and can no longer be replaced ($t_{\text{commit}}$) [s]
    CommitmentPathPosition: NotationSymbol  # position along the admitted path at the commitment instant, the entry position of the matched stop ($q_{c}$) [m]
    CommonPositionShift: NotationSymbol  # number of positions by which a query and a key are both shifted, as when a packed example sits at an offset ($\Delta$)
    CommunicationComputeRatio: NotationSymbol  # ratio of per-step communication time to per-device compute time ($\rho$) [dimensionless]
    CommunicationIntensity: NotationSymbol  # network bytes a workload moves per floating-point operation it performs ($\text{CI}$) [bytes/FLOP]
    CommunicationSlowdownFactor: NotationSymbol  # factor by which the communication part of a step lengthens, such as on an oversubscribed fabric, while computation stays fixed ($r$) [dimensionless]
    CommunicationTime: NotationSymbol  # time per training step spent moving gradients, activations and parameter shards between devices at scale N ($T_{\text{comm}}$) [s]
    CommunicationTimeFraction: NotationSymbol  # share of each training step a workload spends communicating ($f$) [dimensionless]
    CompactedContextLength: NotationSymbol  # number of input tokens in a context after compaction replaces earlier turns with a summary ($S_{\text{comp}}$)
    CompactionBoundaryCount: NotationSymbol  # number of successive compaction boundaries over which retention is measured ($k$)
    CompactionBoundaryIndex: NotationSymbol  # position of one compaction boundary in a trajectory, counted from one ($i$)
    CompactionCycleTurnCount: NotationSymbol  # number of task turns between successive compaction passes ($n_{\text{cycle}}$)
    CompactionCycleTurnIndex: NotationSymbol  # position of one task turn within a compaction cycle, counted from one ($i$)
    CompactionLossProbability: NotationSymbol  # probability that one compaction round independently loses a given required constraint or fact ($\rho$) [dimensionless]
    CompactionPassCost: NotationSymbol  # total monetary cost of one compaction pass, billed on the same boundary as a task turn ($C_{\text{comp}}$) [USD]
    CompactionPathCost: NotationSymbol  # remaining token cost of a trajectory that compacts its context, as a function of the remaining call count ($C_{\text{compact}}$) [USD]
    CompactionPaybackCallCount: NotationSymbol  # fewest remaining calls, including the first compacted one, for which compaction costs no more than keeping the full context ($N_{\min}$)
    CompactionPerCallSaving: NotationSymbol  # monetary saving on each call after the first compacted one, the cached-input price of the tokens compaction removed ($s$) [USD]
    CompactionTargetLength: NotationSymbol  # serialized input length, in tokens, that a compaction pass reduces the request to, the low watermark ($S_{\text{low}}$)
    CompactionTriggerLength: NotationSymbol  # serialized input length, in tokens, at which the runtime triggers a compaction pass, the high watermark ($S_{\text{high}}$)
    CompactionUpfrontCost: NotationSymbol  # one-time extra cost of compacting a context relative to keeping it: the summary's output tokens plus re-sending the compacted context as fresh input ($U$) [USD]
    ComparedPeriodAccuracy: NotationSymbol  # accuracy of a model over one of two compared periods, indexed 1 for the baseline and 2 after drift ($a$) [dimensionless]
    ComparisonArmTrialCount: NotationSymbol  # number of independent trials in each arm of a two-arm comparison ($n_{\text{arm}}$)
    ComparisonContext: NotationSymbol  # shared context and evaluation criterion under which two trajectory records are compared ($c$)
    ComparisonSampleEmpiricalCdf: NotationSymbol  # empirical cumulative distribution function of the second of two samples a two-sample Kolmogorov-Smirnov test compares, the sample of size m ($G_{m}$)
    ComparisonSampleSize: NotationSymbol  # number of observations in the second of the two samples a two-sample drift test compares ($m$)
    ComparisonTrajectory: NotationSymbol  # solution of the scalar comparison differential equation that lower-bounds the barrier value in the comparison lemma ($z$)
    ComparisonTrajectoryDerivative: NotationSymbol  # time derivative of the comparison trajectory in the comparison lemma ($\dot{z}$)
    CompensatedEnvironmentState: NotationSymbol  # state of the environment after a saga forward step and its compensating step have both executed ($\sigma_0'$)
    CompensationPostcondition: NotationSymbol  # one declared postcondition predicate of a compensation contract, true or false of an environment state ($\phi$)
    CompensationPostconditionSet: NotationSymbol  # declared set of postcondition predicates on which a compensation claims to restore the environment ($\Phi$)
    CompetingKeyCount: NotationSymbol  # number of keys other than the target key that one attention query normalizes over ($n$)
    CompetingKeyLogit: NotationSymbol  # scaled query-key logit of one key other than the target in one attention head ($z$) [dimensionless]
    CompilationBreakevenCount: NotationSymbol  # number of executions whose per-execution savings repay one compilation ($N_{\text{breakeven}}$)
    CompilationTime: NotationSymbol  # time one compilation or recompilation of a model takes ($T_{\text{compile}}$) [s]
    CompiledExecutionTime: NotationSymbol  # time for one execution of a model after compilation ($T_{\text{compiled}}$) [s]
    CompilerBackendCount: NotationSymbol  # number of hardware backends a compiler stack generates code for ($N_{\text{backend}}$)
    CompilerFrontendCount: NotationSymbol  # number of source frontends, such as capture mechanisms or languages, a compiler stack supports ($M$)
    ComplementarityNormalForce: NotationSymbol  # normal force at a unilateral rigid contact in the complementarity model, nonnegative and nonzero only when the gap is closed ($\lambda_n$) [N]
    CompletedRequestCount: NotationSymbol  # number of requests a system has completed by a given time ($N$)
    CompletedRoundIndex: NotationSymbol  # position of one completed round among those a federated run executes, as summed in the time to target and indexed in the per-round traffic budget ($r$)
    CompletionCriteria: NotationSymbol  # set of checks that decide when a task is done, at a stated closure evidence level ($\mathcal{K}_{\text{comp}}$)
    CompletionDeadline: NotationSymbol  # time by which a call, cycle, or task must complete ($T_{\text{deadline}}$) [s]
    CompletionOffset: NotationSymbol  # time from the start of a tick to the completion of that tick's work, where the controller feeds the watchdog ($c$) [s]
    CompliantElementPreload: NotationSymbol  # force a compliant element already carries at rest, before it begins to deflect ($F_{\text{preload}}$) [N]
    CompliantElementStiffness: NotationSymbol  # stiffness of a passive compliant element placed in series with a contact, force per unit deflection ($k_{\text{spring}}$) [N/m]
    ComponentAvailability: NotationSymbol  # probability that one component is up at a given instant ($A_{\text{component}}$) [dimensionless]
    ComponentFailureRate: NotationSymbol  # rate at which one component fails ($\lambda$) [failures/hour]
    ComponentHazardRate: NotationSymbol  # instantaneous failure rate of a component as a function of its age, the hazard function ($h$) [failures/hour]
    ComponentMttf: NotationSymbol  # mean time to failure of one component, the reciprocal of its constant failure rate ($\text{MTTF}_{\text{component}}$) [hours]
    ComponentReliability: NotationSymbol  # probability that one component of a system runs without failure up to a given time, its survival function ($R_{\text{single}}$) [dimensionless]
    ComposedChainTransform: NotationSymbol  # homogeneous transform from frame C to frame A obtained by composing a chain of transforms ($\mathbf{T}_{AC}$)
    ComposedFunctionInput: NotationSymbol  # input of a composition of layer functions ($x$) [dimensionless]
    ComposedFunctionOutput: NotationSymbol  # output of a composition of layer functions, whose derivative the chain rule computes ($y$) [dimensionless]
    ComposedMechanismCount: NotationSymbol  # number of differentially private mechanism applications whose privacy loss is composed, such as the training steps of a DP-SGD run ($k$)
    CompoundScalingCoefficient: NotationSymbol  # exponent that sets how much compute compound scaling adds to a base model ($\phi$) [dimensionless]
    CompressedCommunicationTime: NotationSymbol  # total time to encode, transfer and decode a compressed payload ($T_{\text{compressed}}$) [s]
    CompressedGradientUpdate: NotationSymbol  # compressed update a worker transmits at one step, the compressor applied to its gradient plus error-feedback residual ($\mathbf{v}$)
    CompressedGradientUpdateElement: NotationSymbol  # value the compressor transmits for one gradient element at one step ($v$) [varies]
    CompressedOnlyCorrectCount: NotationSymbol  # number of evaluation examples the compressed model classifies correctly and the baseline model gets wrong ($n_{\text{comp}}$)
    CompressionRatio: NotationSymbol  # uncompressed size of a payload divided by its compressed size ($r_{\text{comp}}$) [dimensionless]
    ComputationalGraph: NotationSymbol  # computational graph that evaluates a function, a pair of nodes and edges ($G$)
    ComputeBudgetConstraint: NotationSymbol  # constraint function that is zero when training compute, six times parameters times tokens, equals the fixed budget ($g$)
    ComputeChipPower: NotationSymbol  # steady-state power the proposal-path compute silicon draws during a mission ($P_{\text{chip}}$) [W]
    ComputeConstrainedLagrangian: NotationSymbol  # Lagrangian of minimizing scaling-law loss over parameters and tokens at a fixed training compute budget ($\mathcal{L}_{\text{Lagrange}}$)
    ComputeConstraintMultiplier: NotationSymbol  # Lagrange multiplier attached to the fixed training compute budget in the compute-optimal allocation ($\lambda$) [varies]
    ComputeDemandVolume: NotationSymbol  # quantity of AI compute demanded at a given unit cost, the volume axis of a demand curve ($Q$) [FLOP]
    ComputeLimitedSampleRate: NotationSymbol  # samples per second the accelerators could process if input data were always ready ($R_{\text{compute}}$) [samples/s]
    ComputeOptimalTokenCount: NotationSymbol  # number of training tokens that minimizes loss for a fixed compute budget ($D_{\text{opt}}$)
    ComputeUnitCost: NotationSymbol  # monetary cost of one unit of AI compute, the price axis of a demand curve in a rebound analysis ($P$) [USD/FLOP]
    ConcordantTaskCount: NotationSymbol  # number of sampled tasks on which the baseline and candidate configurations receive the same verdict in a paired comparison ($n$)
    ConcurrentTrajectories: NotationSymbol  # mean number of admitted trajectories present at once ($N_{\text{traj}}$)
    ConditionalDamageProbability: NotationSymbol  # true conditional probability that a proposed action produces the modeled damaging outcome, given observable features ($p$) [dimensionless]
    ConditionalDistribution: NotationSymbol  # probability distribution of the label given an input, the input-to-label relationship that concept drift changes ($p(y \mid x)$)
    ConditionalFlowMap: NotationSymbol  # map that carries a source noise sample to its point on the straight conditional path at a given flow time ($\psi$)
    ConditionalVelocityField: NotationSymbol  # velocity along one conditional flow-matching path given its endpoint sample ($v$)
    ConditioningTokenPrefix: NotationSymbol  # token prefix a language-model policy conditions on when choosing the next token, such as the context delivered before one sampled token of a rollout ($x$)
    ConductorCurrent: NotationSymbol  # electrical current flowing through a power-distribution conductor ($I$) [A]
    ConeLimitStepSize: NotationSymbol  # positive step length along a direction, taken to zero in the definition of the contingent cone ($\tau$) [varies]
    ConfidenceCalibrationDataset: NotationSymbol  # held-out labeled examples on which a post-hoc confidence calibration, such as a softmax temperature, is fitted ($\mathcal{D}_{\text{cal}}$)
    ConfidenceCriticalQuantile: NotationSymbol  # positive standard-Normal critical value for a two-sided test at the chosen confidence level ($z_{\alpha/2}$) [dimensionless]
    ConfidenceLevel: NotationSymbol  # confidence level of a one-sided statistical bound, one minus its significance level ($C$) [dimensionless]
    ConfidentialityLabel: NotationSymbol  # set of recipients permitted to receive a record ($\kappa$)
    ConfigurationSpace: NotationSymbol  # set of all configurations of a mechanism, one point per assignment of its joint positions ($\mathcal{C}$)
    ConfirmationHitCount: NotationSymbol  # number of associated frames a tentative track needs within the confirmation window ($M$)
    ConfirmationWindowCount: NotationSymbol  # number of most recent frames an M-of-N track-confirmation rule examines ($N_{\text{conf}}$)
    ConfluentVandermondeMatrix: NotationSymbol  # matrix of powers of the segment duration that maps polynomial coefficients to endpoint position, velocity and acceleration ($\mathbf{M}$)
    ConformalMiscoverageRate: NotationSymbol  # allowed probability that the conformal prediction set misses the true referent ($\epsilon$) [dimensionless]
    ConformalPredictionSet: NotationSymbol  # calibrated set of referent candidates that cannot be excluded at the miscoverage rate ($\mathcal{C}$)
    ConformalQuantileRank: NotationSymbol  # position, among the calibration scores sorted in ascending order, of the score that becomes the conformal prediction threshold ($k_{\epsilon}$)
    ConformalScoreQuantile: NotationSymbol  # nonconformity-score threshold calibrated on held-out data, the finite-sample corrected one-minus-miscoverage quantile of the calibration scores ($\widehat{q}_{1-\epsilon}$) [dimensionless]
    ConservativePenaltyWeight: NotationSymbol  # weight of the conservative penalty that pushes down values of unsupported actions in conservative Q-learning ($\alpha_{\text{CQL}}$) [dimensionless]
    ConservativeSamplingDistribution: NotationSymbol  # action distribution from which the conservative Q-learning regularizer samples actions whose values it pushes down ($\mu$)
    ConsoleCommunicationTime: NotationSymbol  # time the operator console and link add to a takeover: video transport to the operator plus the returning protocol exchange ($T_{\text{comm}}$) [s]
    ConsolidationOverheadTime: NotationSymbol  # expected time lost per checkpoint to consolidating an incremental delta chain into a new base plus the expected restore of the chain, as a function of the consolidation period ($T_{\text{overhead}}$) [s]
    ConsolidationStride: NotationSymbol  # number of incremental checkpoints between consolidations of a delta chain into a new full base ($c$)
    ConstantDivisorGrpoObjective: NotationSymbol  # sampled GRPO objective that divides by one constant length shared across the group instead of each rollout's own token count ($\widehat{\mathcal{J}}_{\text{const}}$) [dimensionless]
    ConstantHazardRate: NotationSymbol  # constant rate of hazardous failures per operating hour of a machine, the rate a failure-free test exposure bounds from above ($\lambda$) [1/h]
    ConstrainedCallLatency: NotationSymbol  # latency of generating a tool call under constrained decoding, per-token decode time plus per-token masking time over every output token ($T_{\text{constrained}}$) [s]
    ConstrainedDecodingAutomaton: NotationSymbol  # deterministic finite automaton over bytes that tracks whether a generated prefix can still complete to a string of the permitted output language ($\mathcal{D}$)
    ConstrainedParameterVector: NotationSymbol  # parameter vector a projected gradient descent step updates and then projects back onto the constraint set ($\mathbf{w}$)
    ConstrainedTestEnvelope: NotationSymbol  # set of states a constrained physical test permits, enforced by tethers, clamps, geofences or the permission path ($\mathcal{S}$)
    ConstraintLagrangeMultiplier: NotationSymbol  # Lagrange multiplier attached to an inequality constraint in constrained optimization, nonnegative and nonzero only when the constraint is active ($\lambda$) [varies]
    ConstraintRetentionFraction: NotationSymbol  # fraction of a set of critical constraints retained without interruption across a number of compaction boundaries ($\eta_{\mathcal{C}}$) [dimensionless]
    ContactApproachSpeed: NotationSymbol  # speed at which the robot approaches and strikes a contact surface ($v_{\text{approach}}$) [m/s]
    ContactCount: NotationSymbol  # number of active contacts between a robot and its environment ($N_c$)
    ContactDeformationDuration: NotationSymbol  # duration over which a physical contact pad deforms during impact ($\Delta t_{\text{def}}$) [s]
    ContactDirectionSpeed: NotationSymbol  # speed of the end effector along the contact direction during a contact interval ($v_c$) [m/s]
    ContactDirectionVelocity: NotationSymbol  # velocity of the robot's contact point along the contact direction ($\dot{x}$) [m/s]
    ContactDriftAcceleration: NotationSymbol  # part of the normal acceleration at a contact that does not depend on the contact force, from applied torques, Coriolis, gravity and Jacobian rate terms ($a_{\text{drift}}$) [m/s^2]
    ContactForce: NotationSymbol  # force between the robot and the environment at a contact, along the contact direction ($F_{\text{contact}}$) [N]
    ContactForceBarrierFunction: NotationSymbol  # barrier certificate on contact force, the hard force limit minus the contact force magnitude ($h_{\text{force}}$) [N]
    ContactForceCap: NotationSymbol  # smaller of the deliverable actuator force and the contact-force ceiling after sensing and delay allowances ($F_{\text{cap}}$) [N]
    ContactForceCeiling: NotationSymbol  # largest contact force an intent lease admits, the force part of its wrench ceiling ($F_{\max}$) [N]
    ContactForceLimit: NotationSymbol  # hard limit on the contact force magnitude during intentional contact ($F_{\text{limit}}$) [N]
    ContactForceMagnitude: NotationSymbol  # magnitude of a contact force at the gripper or tool ($F$) [N]
    ContactForceRegularizationWeight: NotationSymbol  # weight on the squared deviation of contact forces from their nominal values in a whole-body control objective ($w_f$) [varies]
    ContactForceRiseTime: NotationSymbol  # time contact force takes to rise from onset to its peak or target ($t_{\text{rise}}$) [s]
    ContactGapAcceleration: NotationSymbol  # second time derivative of the contact gap function, the relative normal acceleration at a contact ($\ddot{\phi}$) [m/s^2]
    ContactGapFunction: NotationSymbol  # signed distance between two contact geometries as a function of the configuration: positive when separated, zero in contact, negative when penetrating ($\phi$) [m]
    ContactGapVelocity: NotationSymbol  # first time derivative of the contact gap function, the rate at which the gap closes or opens ($\dot{\phi}$) [m/s]
    ContactImpulse: NotationSymbol  # impulse a contact transfers, the time integral of contact force over the contact event ($I$) [N*s]
    ContactJacobianBaseBlock: NotationSymbol  # block of the Jacobian of contact point k that multiplies the floating-base velocities ($\mathbf{J}_{k,b}$)
    ContactJacobianJointBlock: NotationSymbol  # block of the Jacobian of contact point k that multiplies the actuated-joint velocities ($\mathbf{J}_{k,j}$)
    ContactLoopResponseTime: NotationSymbol  # time the contact-control loop takes to respond after its tripwire fires ($T_c$) [s]
    ContactNaturalTimescale: NotationSymbol  # natural time scale of a mass-spring contact, the square root of effective mass over contact stiffness ($T_{\text{contact}}$) [s]
    ContactNormal: NotationSymbol  # unit vector normal to the contact plane at one contact, pointing into the robot ($\hat{\mathbf{n}}$) [dimensionless]
    ContactNormalDeflection: NotationSymbol  # position of the contacting body along the contact normal, whose change from first touch is the contact deflection ($x$) [m]
    ContactNormalJacobian: NotationSymbol  # Jacobian of a contact gap function, the row that maps generalized velocities to the normal velocity of the contact ($\mathbf{J}_n$)
    ContactNormalJacobianDerivative: NotationSymbol  # time derivative of the contact normal Jacobian, a function of configuration and generalized velocity ($\dot{\mathbf{J}}_n$)
    ContactNormalPosition: NotationSymbol  # position of a tool or link along the contact normal, whose change from first contact is the contact penetration ($x$) [m]
    ContactNormalVelocity: NotationSymbol  # relative velocity of two bodies at a contact along the contact normal ($v_n$) [m/s]
    ContactPatchRegion: NotationSymbol  # region of the ground plane covered by one active contact patch, projected onto the support plane ($\mathcal{A}$)
    ContactPointForce: NotationSymbol  # force the environment exerts on the robot at one contact point ($\mathbf{f}$) [N]
    ContactPointIndex: NotationSymbol  # position of one contact point among the contacts of a grasp ($i$)
    ContactPointJacobian: NotationSymbol  # matrix mapping joint velocities to the linear velocity of one contact point ($\mathbf{J}_{\text{c}}$)
    ContactPointJacobianDerivative: NotationSymbol  # time derivative of the Jacobian of one contact point ($\dot{\mathbf{J}}_{\text{c}}$)
    ContactPointOffset: NotationSymbol  # position of a grasp contact relative to the reference point of the grasped object, the moment arm of its force ($\mathbf{r}$) [m]
    ContactPointPosition: NotationSymbol  # position of the robot's contact point along the contact direction ($x$) [m]
    ContactPointTorque: NotationSymbol  # moment the environment exerts on the robot at one contact patch, the torque part of the contact wrench ($\boldsymbol{\tau}$)
    ContactReactionForce: NotationSymbol  # reaction force a stiff contact exerts when the robot's contact point deviates into it ($F$) [N]
    ContactResponseLatency: NotationSymbol  # time from the contact tripwire firing until the arm begins to decelerate, during which it keeps moving at its approach speed ($t_{\text{resp}}$) [s]
    ContactStiffness: NotationSymbol  # stiffness of a contact treated as a spring, force per unit penetration ($k_{\text{contact}}$) [N/m]
    ContactTorqueCeiling: NotationSymbol  # largest contact torque an intent lease admits, the torque part of its wrench ceiling ($\tau_{\max}$) [N*m]
    ContactTripwireForce: NotationSymbol  # contact force at which the permission path ends an attempt on the machine, set below the structural damage limit ($F_{\text{trip}}$) [N]
    ContactWrench: NotationSymbol  # contact wrench: the force and moment the environment applies at a contact, stacked into one vector ($\mathbf{w}$) [varies]
    ContactWrenchCeiling: NotationSymbol  # largest contact force and torque an intent lease permits at the tool, stacked as a wrench ($\mathbf{w}_{\max}$) [varies]
    ContactWrenchEnvelope: NotationSymbol  # allowable contact wrench a grasp affordance may exert, three force and three torque components ($\mathbf{w}_{\text{contact}}$) [varies]
    ContactWrenchTransform: NotationSymbol  # six-by-six spatial force transform that carries the wrench at contact k into the center-of-mass frame ($\mathbf{X}_k^*$)
    ContactYieldDeadline: NotationSymbol  # time from first contact until a drive at constant velocity into a linear-elastic contact reaches the structural force limit ($t_{\text{yield}}$) [s]
    ContainerInnerRadius: NotationSymbol  # inner radius of a carried liquid container ($R$) [m]
    ContainmentForceCeiling: NotationSymbol  # hard ceiling on commanded force that an operational containment of a residual claim imposes ($F_{\max}$) [N]
    ContendingApplicationCount: NotationSymbol  # number of applications contending for a shared cluster, which sets each application's fair share as one over it ($N$)
    ContentionCoefficient: NotationSymbol  # universal scalability law's share of work that serializes across workers ($\sigma_{\text{cont}}$) [dimensionless]
    ContextChunkTokenCount: NotationSymbol  # number of tokens in one device's chunk of the sequence under context parallelism ($n$)
    ContextHiddenRepresentation: NotationSymbol  # final hidden representation a vision-language-action backbone hands to its action decoder ($\mathbf{h}_{\text{context}}$)
    ContextParallelDegree: NotationSymbol  # number of devices that each hold a slice of the sequence in context parallelism ($c$)
    ContextParallelRankIndex: NotationSymbol  # position of one device in a context-parallel ring, which is also the index of the query chunk that device holds ($i$)
    ContextSaveTime: NotationSymbol  # time to save the interrupted thread's register context on interrupt entry ($t_{\text{context\_save}}$) [s]
    ContextWindow: NotationSymbol  # hard per-call limit on prompt and output tokens together ($S_{\max}$)
    ContinuationCount: NotationSymbol  # number of independent continuations sampled from one fork condition in reinforcement learning ($K$)
    ContinuationIndex: NotationSymbol  # position of one sampled continuation among those launched from a fork condition ($i$)
    ContinuationReturnStandardDeviation: NotationSymbol  # standard deviation of the bounded return of one sampled continuation ($\sigma$) [dimensionless]
    ContinuityCorrectedCrossProduct: NotationSymbol  # numerator of the paired-verdict correlation, the concordant-minus-discordant cross product with a continuity correction when it is positive ($A$) [dimensionless]
    ContinuousActionComponent: NotationSymbol  # continuous value of one dimension of an action, before it is discretized into a token bin ($a$) [varies]
    ContinuousActionSpace: NotationSymbol  # set of continuous actions a robot policy may emit, a subset of real space of the action dimension ($\mathcal{A}$)
    ContinuousActionVector: NotationSymbol  # continuous action vector of the robot's decision process, written without a step index: the argument of a policy, value function, reward or action distribution, or an action recorded from a demonstrator or logging routine ($\mathbf{a}$)
    ContinuousCurrent: NotationSymbol  # largest motor current the winding can carry indefinitely without exceeding its temperature limit ($I_{\text{cont}}$) [A]
    ContinuousFactorialFunction: NotationSymbol  # Euler's gamma function, the continuous extension of the factorial ($\Gamma$)
    ContinuousMeasurementNoiseIntensity: NotationSymbol  # intensity matrix of continuous-time white measurement noise, the weight of the delta function in its covariance kernel ($\mathbf{R}_{\text{cont}}$) [varies]
    ContinuousMotionModel: NotationSymbol  # motion model giving the state's rate of change for a given state and input ($\mathbf{f}$)
    ContinuousOutcome: NotationSymbol  # value of a continuous random vector, such as a physical state, at which its probability density is evaluated or integrated ($\mathbf{x}$)
    ContinuousRandomVariable: NotationSymbol  # continuous random variable, such as a physical state, described by a probability density ($\mathbf{X}$)
    ContinuousStateTransitionMatrix: NotationSymbol  # state transition matrix of a continuous-time linear system, the matrix exponential of the system matrix times elapsed time ($\boldsymbol{\Phi}$)
    ContinuousTorque: NotationSymbol  # largest motor torque a winding can sustain indefinitely at its temperature limit under the stated cooling ($\tau_{\text{cont}}$) [N*m]
    ContractedBarrierFunction: NotationSymbol  # barrier certificate evaluated at the reference position and contracted by the tracking-error bound, so the actual body keeps nonnegative clearance ($h_{\text{contracted}}$) [m]
    ControlAffineDriftField: NotationSymbol  # unforced part of control-affine dynamics, the state rate when the control input is zero ($\mathbf{f}$)
    ControlAffineInputField: NotationSymbol  # input matrix field of control-affine dynamics, mapping control inputs to state rates ($\mathbf{g}$)
    ControlCommand: NotationSymbol  # control command vector sent toward the actuators, whatever its source or stage ($\mathbf{u}$)
    ControlCommandSpace: NotationSymbol  # normed space of the control commands a policy outputs ($\mathcal{U}$)
    ControlConversionRate: NotationSymbol  # observed conversion rate of the control variant of an A/B test ($\hat{p}_A$) [dimensionless]
    ControlCycleIndex: NotationSymbol  # position of one control cycle, one re-solve of a receding-horizon controller, in the sequence of cycles ($i$)
    ControlInput: NotationSymbol  # control input applied to the plant, the command a controller or safety filter sends to the actuators ($\mathbf{u}$)
    ControlInputDimension: NotationSymbol  # number of components of the control input vector ($m$)
    ControlInputLowerLimit: NotationSymbol  # per-component lower limit of the control input the actuators can deliver ($\mathbf{u}_{\min}$)
    ControlInputSet: NotationSymbol  # set of control inputs the controller may choose from, a subset of the input space ($\mathcal{U}$)
    ControlInputUpperLimit: NotationSymbol  # per-component upper limit of the control input the actuators can deliver ($\mathbf{u}_{\max}$)
    ControlInputWindow: NotationSymbol  # control input signal over the short interval from t to t plus Delta t, the decision variable of one Bellman step ($\mathbf{u}_{[t, t+\Delta t]}$)
    ControlSampleIndex: NotationSymbol  # index of one sampling period of a discrete-time controller, counted from zero ($k$)
    ControlSampleSize: NotationSymbol  # number of observations in the control variant of an A/B test ($m_A$)
    ControlSetpointPeriod: NotationSymbol  # time between successive setpoints the controller consumes, the deadline a synchronous setpoint producer must meet ($\Delta t_{\text{ctrl}}$) [s]
    ControlStepIndex: NotationSymbol  # position of one discrete control step, or setpoint, in a robot's control timeline ($t$)
    ControlTaskIndex: NotationSymbol  # position of one task in a whole-body control task hierarchy, in priority order ($i$)
    ControllabilityMatrix: NotationSymbol  # block matrix of the input matrix and its products with powers of the system matrix, whose rank decides controllability ($\mathbf{C}_{\text{ctrb}}$)
    ControlledStopCommand: NotationSymbol  # resident fallback command that executes the state-matched controlled stop ($\mathbf{u}_{\text{stop}}$)
    ControllerReactionDelay: NotationSymbol  # time the staging admission controller needs to stop dispatch after the high watermark is crossed ($T_{\text{react}}$) [s]
    ConversationTurnIndex: NotationSymbol  # position of one turn within a multi-turn conversation ($i$)
    ConverterLockoutVoltage: NotationSymbol  # input voltage at which a DC-DC converter shuts off, its undervoltage lockout threshold ($V_{\text{uvlo}}$) [V]
    ConveyorSpeed: NotationSymbol  # speed of the conveyor belt that carries a workpiece past the pick station ($v_{\text{conv}}$) [m/s]
    ConveyorSpeedBound: NotationSymbol  # largest speed at which the takeaway conveyor carries a target ($v_{\text{conv}}$) [m/s]
    ConvolutionActivation: NotationSymbol  # activation at one spatial position and channel of a convolution layer's feature map ($H$) [dimensionless]
    ConvolutionTimeInstant: NotationSymbol  # earlier time instant between zero and t, the variable a convolution integral over past noise runs over ($t'$) [s]
    CookedTokenKeyVector: NotationSymbol  # key vector of the token "cooked" in the chef-cooked-the-meal example ($\mathbf{k}_{\text{cooked}}$)
    CookedTokenQueryVector: NotationSymbol  # query vector of the token "cooked" in the chef-cooked-the-meal example ($\mathbf{q}_{\text{cooked}}$)
    CoolantHeatLoad: NotationSymbol  # heat per unit time that a liquid cooling loop must carry away from a rack, the load that sets its coolant mass flow ($P_{\text{thermal}}$) [W]
    CoolantMassFlowRate: NotationSymbol  # mass of coolant, liquid or air, passing through a cooling loop per second ($\dot{m}$) [kg/s]
    CoolantTemperature: NotationSymbol  # temperature of the liquid or air coolant in a cooling loop, whose finite change is its rise from inlet to outlet ($T$) [K]
    CoolingEnergy: NotationSymbol  # cooling energy attributable to a fleet's equipment over a stated interval ($E_{\text{cooling}}$) [J]
    CoordinateFrameEmbedding: NotationSymbol  # learned embedding of an observation's sensor-to-body transform ($\mathbf{e}_{\text{frame}}$)
    CoordinationLatency: NotationSymbol  # exposed time a request spends fanning out to services and waiting to join their results on its critical path ($L_{\text{lat,coordination}}$) [s]
    CoordinationOverhead: NotationSymbol  # added elapsed time of splitting work across workers, as a fraction of serial time ($h$) [dimensionless]
    CopperIronThermalResistance: NotationSymbol  # thermal resistance across the winding insulation from the copper node to the stator iron node ($R_{\text{th,cu-iron}}$) [K/W]
    CopperReferenceTemperature: NotationSymbol  # reference temperature at which a winding has resistance R_0, the base of the copper temperature coefficient ($T_0$) [degC]
    CopperTemperatureCoefficient: NotationSymbol  # fractional rise in copper's electrical resistance per kelvin of temperature rise ($\alpha_{\text{Cu}}$) [1/K]
    CopperThermalCapacitance: NotationSymbol  # heat capacity of the stator winding copper node in a multi-node motor thermal network ($C_{\text{cu}}$) [J/K]
    CopperThermalTimeConstant: NotationSymbol  # thermal time constant of the stator winding copper stage of a multi-node motor thermal model ($\tau_{\text{th,cu}}$) [s]
    CoreAmbientThermalResistance: NotationSymbol  # convective thermal resistance from a motor's stator core to the ambient air ($R_{\text{th,ca}}$) [K/W]
    CoreRailVoltageDroop: NotationSymbol  # transient drop in a processor core supply rail when the load current steps, from package inductance and resistance ($\Delta V_{\text{droop}}$) [V]
    CoreThermalCapacitance: NotationSymbol  # heat capacity of a motor's laminated stator core and housing, the slow node of the two-node thermal model ($C_{\text{core}}$) [J/K]
    CoresetBudget: NotationSymbol  # number of examples a coreset selection method is asked to retain ($K$)
    CoriolisCentrifugalMatrix: NotationSymbol  # matrix of Coriolis and centrifugal effects in the manipulator equation, multiplying joint velocities ($\mathbf{C}$)
    CorrectCandidateTrialPassProbability: NotationSymbol  # probability that a fixed correct candidate passes one verification trial ($p_{g}$) [dimensionless]
    CorrectClassIndex: NotationSymbol  # index of an example's correct class ($c$)
    CorrectClassLogit: NotationSymbol  # logit of the correct class ($z_{c}$) [dimensionless]
    CorrectClassProbability: NotationSymbol  # predicted probability of the correct class ($\hat{y}_{c}$) [dimensionless]
    CorrectnessReward: NotationSymbol  # binary outcome of the correctness check on a fully assessed trajectory, one for pass and zero for fail ($R_{\text{corr}}$) [dimensionless]
    CorrelationCoefficient: NotationSymbol  # Pearson correlation coefficient between two random quantities ($\rho$) [dimensionless]
    CorruptedAttentionScoreMatrix: NotationSymbol  # pre-softmax attention score matrix computed from a query matrix that silent data corruption has altered ($\mathbf{S}_{\text{corrupt}}$)
    CorruptedModelWeight: NotationSymbol  # value of a model weight after a hardware fault flips one bit of its stored word ($w'$) [dimensionless]
    CostAnomalyScore: NotationSymbol  # number of historical standard deviations by which the current period's cost differs from its historical mean ($Z$) [dimensionless]
    CostBudgetUsedFraction: NotationSymbol  # fraction of a trajectory's dollar allowance already spent or reserved ($\rho_C$) [dimensionless]
    CostCeiling: NotationSymbol  # dollar ceiling on a trajectory that the runtime enforces before dispatching the next call ($C_{\max}$) [USD]
    CostPerAcceptedTask: NotationSymbol  # total spend on all attempts in a cohort divided by the number of accepted tasks ($C_{\text{eff}}$) [USD/task]
    CostPerEmittedArtifact: NotationSymbol  # total cohort spend divided by the number of valid artifacts emitted for one consumer ($C_{\text{emit}}$) [USD/item]
    CostToGo: NotationSymbol  # expected cumulative stage cost from a state onward under a policy, the control-theory counterpart of the negated state value ($C$) [dimensionless]
    CoulombFrictionCoefficient: NotationSymbol  # ratio of the largest tangential friction force to the normal force at a contact ($\mu$) [dimensionless]
    CoulombFrictionForce: NotationSymbol  # tangential friction force at a contact, set-valued between minus and plus the friction coefficient times the normal force ($F_{\text{friction}}$) [N]
    CountedFailureIndicator: NotationSymbol  # indicator that one classified call outcome counts as an infrastructure failure against a breaker's scope ($e$) [dimensionless]
    CovariateDriftBound: NotationSymbol  # worst-case excess task cost of behavior cloning, growing with the square of the episode length ($\mathcal{O}(S^2 \epsilon)$)
    CoverageFactor: NotationSymbol  # number of standard deviations a calibrated confidence bound adds when a covariance is turned into a spatial safety margin ($k$) [dimensionless]
    CpuEnergy: NotationSymbol  # energy the host CPUs consume over a stated interval ($E_{\text{CPU}}$) [J]
    CredibleBrakingDeceleration: NotationSymbol  # validated low-tail deceleration the machine achieves under its stated conditions, not the datasheet peak ($a_{\text{brake}}$) [m/s^2]
    CreditWindowCapacity: NotationSymbol  # fixed number of credits a credit-based flow-control channel holds, its bound on admitted in-flight items ($N_{\text{win}}$)
    CriticActionReturnEstimate: NotationSymbol  # action-value estimate a learned critic network with parameters phi predicts for a state and action ($Q_{\phi}$)
    CriticApproximationError: NotationSymbol  # error of a learned critic's action-value estimate relative to the true value, at a given state and action ($\epsilon$) [dimensionless]
    CriticModel: NotationSymbol  # learned model in PPO-based RLHF that predicts the return, the cumulative discounted reward, following each generated token ($M_{\text{value}}$)
    CriticParameters: NotationSymbol  # learned weights of an actor-critic method's critic network ($\phi$)
    CriticReturnEstimate: NotationSymbol  # remaining-return estimate a learned critic network with parameters phi predicts for an interaction history ($V_{\phi}$) [dimensionless]
    CriticalBatchSize: NotationSymbol  # batch size beyond which further growth yields diminishing reductions in optimization steps per added sample, estimated by the gradient noise scale ($B^{*}$)
    CriticalConstraint: NotationSymbol  # one critical constraint drawn from the task contract, such as a negative scope of the goal or a limit on a permitted action ($c$)
    CriticalPathDuration: NotationSymbol  # duration of the longest source-to-sink path through a dependency graph, summing the durations of the vertices on it ($T_{\text{crit}}$) [s]
    CriticalSectionIndex: NotationSymbol  # index of one critical section of a lower-priority task in the priority ceiling protocol blocking bound ($s$)
    CriticalSectionWcet: NotationSymbol  # worst-case execution time of one critical section of a lower-priority task, indexed by task and critical section ($C^{\text{cs}}$) [s]
    CriticalTrackingErrorBound: NotationSymbol  # critical bound on the distance between commanded and measured position, larger than the tracking-error bound, above which the enforcer stops ($\epsilon_{\text{crit}}$) [m]
    CriticalTransitDistance: NotationSymbol  # move distance at which a rest-to-rest profile first reaches its speed ceiling, separating triangular from trapezoidal profiles ($d_{\text{crit}}$) [m]
    CrossEntropyLevelThreshold: NotationSymbol  # intermediate threshold on the performance metric at one cross-entropy iteration, the elite quantile of the current batch ($\gamma$) [varies]
    CrossEntropyLoss: NotationSymbol  # cross-entropy between a model's predicted class probabilities and the ground-truth labels ($\mathcal{L}_{\text{CE}}$)
    CrossoverFrequency: NotationSymbol  # angular frequency at which the open-loop gain of a feedback loop falls to one ($\omega_c$) [rad/s]
    CruiseSpeed: NotationSymbol  # constant speed the machine holds, with zero acceleration, when a stop begins ($v_{c}$) [m/s]
    CumulativeArrivals: NotationSymbol  # cumulative number of requests that have arrived at a system by a given time ($A$) [requests]
    CumulativeCompletions: NotationSymbol  # cumulative number of requests a system has completed by a given time ($C$) [requests]
    CumulativeIncidentCount: NotationSymbol  # random number of catastrophic incidents a fleet records over a cumulative operating duration, a Poisson counting process ($N$)
    CumulativeLogLikelihoodRatio: NotationSymbol  # cumulative log-likelihood ratio of the specified degradation against the baseline after the labels observed so far in a sequential test ($\Lambda$) [dimensionless]
    CumulativeScheduleStepIndex: NotationSymbol  # running diffusion step in the product that forms the cumulative noise schedule ($s$)
    CumulativeTransmittance: NotationSymbol  # fraction of light along a viewing ray not yet absorbed by the Gaussians in front of a given one ($T$) [dimensionless]
    CumulativeWorkCost: NotationSymbol  # cumulative cost of the work a search has admitted so far, indexed by the number of admitted units ($C$) [USD]
    CurrentBinProportion: NotationSymbol  # proportion of current production observations falling in one bin of the population stability index ($A$) [dimensionless]
    CurrentDistribution: NotationSymbol  # data distribution a deployed model sees in service at time t ($P_t$)
    CurrentFeatureMean: NotationSymbol  # mean of the Gaussian fitted to the current serving sample of a monitored feature ($\mu_{\text{curr}}$) [varies]
    CurrentFeatureStandardDeviation: NotationSymbol  # standard deviation of the Gaussian fitted to the current serving sample of a monitored feature ($\sigma_{\text{curr}}$) [varies]
    CurrentLayerWidth: NotationSymbol  # number of units in layer l ($n_{\ell}$)
    CurrentModelParameters: NotationSymbol  # model parameters before one optimizer update ($\theta_{\text{old}}$)
    CurrentMonotonicTime: NotationSymbol  # current instant on the monotonic hardware clock when a consumer evaluates a stored record ($t_{\text{now,mono}}$) [s]
    CurrentNodeIndex: NotationSymbol  # index of the computational-graph node being evaluated or differentiated ($j$)
    CurrentPeriodCost: NotationSymbol  # observed cost of the current accounting period, such as one day ($C_{\text{current}}$) [USD]
    CurrentStateDigest: NotationSymbol  # digest of the canonicalized workspace projection captured at the current observation step ($h_{\text{state},t}$)
    CurrentStateEstimate: NotationSymbol  # estimate of the mechanical plant state propagated forward to the current instant ($\hat{\mathbf{x}}_{t_{\text{current}}}$)
    CurrentTime: NotationSymbol  # current wall-clock time at which a consumer evaluates a stored record ($t_{\text{now}}$) [s]
    CurrentWallClockTime: NotationSymbol  # current wall-clock time at which a monitor evaluates staleness ($t_{\text{current}}$) [s]
    CutCrossingDataVolume: NotationSymbol  # bytes an exchange must send across a fabric's minimum bisection cut ($V_{\text{cut}}$) [bytes]
    CvaeEncoderParameters: NotationSymbol  # learned weights of the conditional VAE encoder that infers the latent style variable from a demonstration ($\phi$)
    CvaeRecognitionEncoder: NotationSymbol  # approximate posterior of the latent style variable given an observation, learned by a conditional VAE encoder ($q_\phi$)
    CycleAveragePowerCeiling: NotationSymbol  # largest cycle-average copper loss that keeps a winding's equilibrium at the engineering ceiling ($\bar{P}_{\max}$) [W]
    CycleGuardTime: NotationSymbol  # guard time reserved in a fieldbus cycle budget against timing uncertainty ($t_{\text{guard}}$) [s]
    CylinderForce: NotationSymbol  # force a hydraulic cylinder delivers, supply pressure times piston area ($F_{\text{cyl}}$) [N]
    DailyAccountQueryCount: NotationSymbol  # number of queries one API account issues in a day, compared against a static extraction-alert threshold ($q_{\text{daily}}$)
    DailyQueryVolume: NotationSymbol  # number of queries, requests or transactions a deployed model serves per day ($Q_{\text{day}}$) [queries/day]
    DamageEventCount: NotationSymbol  # number of trials in a campaign that ended in physical damage or collision ($N_{\text{damage}}$)
    DamageProbabilityTolerance: NotationSymbol  # largest true conditional damage probability at which a policy allows automatic execution ($\epsilon$) [dimensionless]
    DamageProbabilityUpperBound: NotationSymbol  # upper confidence bound on the conditional probability that an action produces the modeled damage ($p_U$) [dimensionless]
    DamagingOutcomeIndicator: NotationSymbol  # binary indicator that a proposed action would produce the modeled damaging outcome if executed automatically ($Y$) [dimensionless]
    DampedLeastSquaresPseudoinverse: NotationSymbol  # damped least-squares inverse of the manipulator Jacobian, the Jacobian transpose times the inverse of the Jacobian Gram matrix plus the squared damping times the identity ($\mathbf{J}_{\lambda}^\dagger$)
    DampedStepConfiguration: NotationSymbol  # joint configuration a damped-least-squares inverse-kinematics step moves to ($\mathbf{q}_\lambda$)
    DampingRatio: NotationSymbol  # damping ratio of a second-order system, damping over twice the square root of stiffness times mass ($\zeta$) [dimensionless]
    DataAcquisitionCost: NotationSymbol  # cost of obtaining the raw samples of a training dataset, by scraping, licensing or collection ($C_{\text{acquire}}$) [USD]
    DataDebtGrowthRate: NotationSymbol  # net fractional growth of the data-debt remediation backlog per planning period ($r$) [varies]
    DataEgressCost: NotationSymbol  # cost of transferring data out of a storage region or provider over the network ($C_{\text{egress}}$) [USD]
    DataFieldBitCount: NotationSymbol  # number of bits of frame i sent at the data-phase rate: payload and CRC ($B_{\text{data},i}$)
    DataInformationContent: NotationSymbol  # learning signal a body of training data carries, measured as the gain it produces in a fixed evaluation metric ($\mathcal{I}$)
    DataLabelingCost: NotationSymbol  # cost of labeling the samples of a training dataset ($C_{\text{label}}$) [USD]
    DataLoaderWorkerCount: NotationSymbol  # number of data-loader worker processes on one host that each read, decode, augment and enqueue batches ($W$)
    DataMovementEnergyFraction: NotationSymbol  # share of a workload's total energy that data movement contributes ($q$) [dimensionless]
    DataParallelDegree: NotationSymbol  # number of model replicas that each process a different slice of the batch in data parallelism ($N_{\text{DP}}$)
    DataPhaseBitRate: NotationSymbol  # bit rate of the CAN FD data phase after the bit rate switch ($R_{\text{data}}$) [bits/s]
    DataPipelineProcessingCost: NotationSymbol  # cost of running data and feature pipelines over a processed data volume ($C_{\text{process}}$) [USD]
    DataProcessingCost: NotationSymbol  # accelerator compute cost of training passes over a dataset ($C_{\text{process}}$) [USD]
    DataScaleCoefficient: NotationSymbol  # fitted coefficient of the training-token term of a two-term scaling law in parameters and tokens ($B$) [dimensionless]
    DataScalingExponent: NotationSymbol  # fitted power-law exponent at which loss falls as the training dataset or token count grows ($\beta$) [dimensionless]
    DataScalingLossFactor: NotationSymbol  # dataset size raised to minus the data scaling exponent, the factor to which a power-law fit makes loss proportional ($D^{-\beta}$) [dimensionless]
    DataStallFraction: NotationSymbol  # fraction of a job's held time its accelerators spend waiting for input data, the data stall ratio ($f_{\text{stall}}$) [dimensionless]
    DataStorageCost: NotationSymbol  # cost of keeping a training dataset in storage over its retention period ($C_{\text{store}}$) [USD]
    DataSupplyRate: NotationSymbol  # samples per second the storage and input pipeline can deliver to training ($R_{\text{data}}$) [samples/s]
    DataTaxonomyAxis: NotationSymbol  # Data axis of the D·A·M taxonomy, viewed as the set of techniques and concepts it covers so that axis intersections name zones of the taxonomy ($\mathsf{Data}$)
    DataVolume: NotationSymbol  # total bytes a workload moves through memory or across a data path ($D_{\text{vol}}$) [bytes]
    DatasetQuery: NotationSymbol  # query function a differentially private mechanism evaluates on a dataset, whose output is released only after calibrated noise is added ($f$)
    DatasetQueryResult: NotationSymbol  # value a query returns on one dataset of an adjacent pair, before any noise is added ($v$) [varies]
    DatasetShardCount: NotationSymbol  # number of shards a training dataset is split into on shared storage ($K_{\text{shards}}$)
    DatasetSize: NotationSymbol  # number of training examples or tokens in a dataset ($D$)
    DatasetTransferTime: NotationSymbol  # ideal line-rate time to move a dataset of a given size across a network or storage path, data volume divided by bandwidth ($T$) [s]
    DcBusBulkCapacitance: NotationSymbol  # capacitance of the capacitor bank on the inverter DC bus that buffers transient load current ($C_{\text{bus}}$) [F]
    DcBusOvervoltageThreshold: NotationSymbol  # largest DC bus voltage the enforcer tolerates before it stops, the overvoltage threshold ($V_{\text{max\_safe}}$) [V]
    DcBusVoltage: NotationSymbol  # voltage of the DC bus that feeds a motor inverter ($V_{\text{bus}}$) [V]
    DcRailVoltage: NotationSymbol  # voltage of the shared DC power rail that feeds the compute and the motor inverters ($V_{\text{rail}}$) [V]
    DeadTimeCycleCount: NotationSymbol  # number of delay cycles a timer's dead-time generator inserts between complementary outputs ($N_{\text{dtg}}$)
    DeadTimeDqVoltageError: NotationSymbol  # voltage disturbance dead time introduces in the rotor dq frame, which compensation cancels with feedforward voltage ($\Delta V_{d,q}$) [V]
    DeadTimeSafetyMargin: NotationSymbol  # engineering buffer added to the switch delays when sizing an inverter's dead time ($t_{\text{margin}}$) [s]
    DeadTimeTickPeriod: NotationSymbol  # duration of one delay cycle of a timer's dead-time generator ($T_{\text{dtg}}$) [s]
    DeadTimeVoltageError: NotationSymbol  # average phase-voltage error dead time introduces, whose sign follows the phase current ($\Delta V_{\text{phase}}$) [V]
    DeadlineInstant: NotationSymbol  # clock instant by which an observation or action must be complete ($t_{\text{deadline}}$) [s]
    DebiasingTotalLoss: NotationSymbol  # objective an encoder and task head minimize in adversarial debiasing, the task loss minus the weighted adversary loss ($\mathcal{L}_{\text{total}}$)
    DebounceFrameCount: NotationSymbol  # number of consecutive perception frames a newly observed class or mode must persist before the system commits to it ($N$)
    DecisionProcessState: NotationSymbol  # complete environment state of a decision process, including interaction history and variables the agent cannot observe ($s$)
    DecisionScore: NotationSymbol  # score a model assigns to an input, larger values favoring a positive decision, compared against a decision threshold and not necessarily a calibrated probability ($s_{\text{dec}}$) [dimensionless]
    DecisionStepIndex: NotationSymbol  # position of one decision step in a finite-horizon episode, counted from zero to one less than the horizon ($k$)
    DecodeArithmeticIntensity: NotationSymbol  # arithmetic intensity of the weight-dominated part of an LLM decode step, operations per byte of weights and activations moved, as a function of batch size ($I_{\text{decode}}$) [FLOP/byte]
    DecodeBatchSize: NotationSymbol  # number of sequences decoding in one iteration, each contributing one token ($B_{\text{dec}}$)
    DecodeInterval: NotationSymbol  # time a call spends generating its whole output ($T_{\text{decode}}$) [s]
    DecodeRequestSet: NotationSymbol  # set of requests decoding in one serving iteration, each contributing one token position to the iteration ($\mathcal{D}$)
    DecodeTokenTimeFloor: NotationSymbol  # lower bound on the time to generate one token at batch one, bytes streamed per step divided by memory bandwidth ($T_{\text{token}}$) [s]
    DecoderEvaluationCount: NotationSymbol  # number of dependent forward passes a generative action head makes per chunk, such as denoising or ODE-solver steps ($N_{\text{eval}}$)
    DecoderPassTime: NotationSymbol  # time of one dependent forward pass of a generative action head, one denoising or ODE-solver step ($t_{\text{pass}}$) [s]
    DecodingConstraintState: NotationSymbol  # state of the parser or output-constraint automaton that constrains decoding, indexed by decoding step ($q$)
    DecomposedTensor: NotationSymbol  # multidimensional weight tensor that a tensor decomposition approximates ($\mathcal{Y}$)
    DecomposedTensorEntry: NotationSymbol  # one entry of a decomposed three-mode tensor, at row i, column j and depth t ($y_{ijt}$) [dimensionless]
    DecouplingCapacitance: NotationSymbol  # capacitance of one decoupling capacitor or parallel capacitor branch of a power delivery network ($C$) [F]
    DecouplingDriftVector: NotationSymbol  # state-dependent part of the stacked output derivatives that does not depend on the input, the repeated drift Lie derivatives of the outputs ($\mathbf{b}$)
    DecouplingMatrix: NotationSymbol  # decoupling matrix of input-output feedback linearization, mapping the control input to the stacked highest output derivatives ($\mathbf{E}$)
    DecouplingMatrixElement: NotationSymbol  # one entry of the decoupling matrix, a Lie derivative of an output along one input column ($E$) [varies]
    DedicatedHourlyFixedCost: NotationSymbol  # complete hourly fixed cost of a dedicated deployment: amortized hardware leases, failover headroom and operations ($C_{\text{fixed}}$) [USD/h]
    DedicatedVariableTokenCost: NotationSymbol  # variable operating cost per output token on a dedicated deployment ($c_{\text{ded}}$) [USD/token]
    DefectiveActionProbability: NotationSymbol  # probability that one proposed action, such as a fleet configuration change, is wrong ($p_{\text{bad}}$) [dimensionless]
    DefinitionSerializedSize: NotationSymbol  # mean number of tokens one staged tool definition contributes to the final serialized request ($S$) [tokens]
    DeflectionComplianceAllowance: NotationSymbol  # clearance reserved for structural deflection, payload deflection and drivetrain compliance before any tracking error is observable ($\delta_{\text{deflect}}$) [m]
    DegradedFrictionCoefficient: NotationSymbol  # tire-floor friction coefficient on a degraded surface ($\mu_{\text{fault}}$) [dimensionless]
    DegradedStepTime: NotationSymbol  # time of one training step on an allocation degraded by a straggler ($T_{\text{slow}}$) [s]
    DelassusOperator: NotationSymbol  # Delassus operator: contact Jacobian times inverse generalized inertia times Jacobian transpose, the apparent inverse inertia reflected at the contacts ($\mathbf{A}_{\text{Delassus}}$) [1/kg]
    DelayPhaseLag: NotationSymbol  # phase lag that loop delays add at the crossover frequency, eroding the phase margin ($\Delta \phi$) [rad]
    DelayTravelDistance: NotationSymbol  # distance the machine travels during an uncorrected sensing-to-actuation delay, the motion a delayed observation cannot yet correct ($d_{\text{delay}}$) [m]
    DelayWindowIntegrationTime: NotationSymbol  # dummy time variable of integration over a past time window, distinct from the current time it ends at ($\sigma$) [s]
    DelayedObservation: NotationSymbol  # measurement vector of a delayed observation, applied at the IMU sample nearest its capture epoch ($\mathbf{y}_{t_0}$)
    DelegatedAllocationAmount: NotationSymbol  # monetary amount a parent node delegates to one child, moved from the parent's free balance into its reservations ($b$) [USD]
    DelegationTokenMultiple: NotationSymbol  # ratio of the tokens a delegated multi-agent design spends per attempt to the tokens a single agent spends per attempt ($r$) [dimensionless]
    DeliveredActuatorTorque: NotationSymbol  # torque an actuator actually delivers at a given time, as measured in telemetry ($\tau_{\text{act}}$) [N*m]
    DeliveredCapacityFraction: NotationSymbol  # fraction of a dedicated deployment's latency-feasible reference output rate that the workload actually delivers ($u$) [dimensionless]
    DemandCap: NotationSymbol  # finite cap on a modeled demand, such as an enforced turn limit, above which demand is not admitted ($x_{\max}$) [varies]
    DemandPriceElasticity: NotationSymbol  # magnitude of the price elasticity of demand, the percentage change in quantity demanded per percentage change in unit cost ($\epsilon_{\text{elast}}$) [dimensionless]
    DemandThreshold: NotationSymbol  # demand level at which a tail probability is evaluated ($x$) [varies]
    DemonstratedTargetA: NotationSymbol  # regression target a demonstration gives when it picks the first of two identical mugs ($\mathbf{y}_A$) [m]
    DemonstratedTargetB: NotationSymbol  # regression target a demonstration gives when it picks the second of two identical mugs ($\mathbf{y}_B$) [m]
    DemonstrationActionDistribution: NotationSymbol  # conditional distribution of the actions demonstrators took given an observation, the target a behavioral cloning decoder models ($p$)
    DemonstrationActionLabel: NotationSymbol  # action recorded in a demonstration or log, the supervisory label paired with the state or observation at which it was taken ($\mathbf{a}$)
    DemonstrationArchiveActionDistribution: NotationSymbol  # conditional distribution of the actions recorded in a robot demonstration archive at one observation, whose separate modes a squared-error cloning loss averages ($p_{\text{demo}}$)
    DemonstrationChunkDistribution: NotationSymbol  # distribution of clean demonstrated action chunks that a generative action decoder learns to sample from ($p_{\text{data}}$)
    DemonstrationDataset: NotationSymbol  # dataset of recorded observation-action demonstration pairs that an imitation loss averages over ($\mathcal{D}_{\text{demo}}$)
    DemonstrationRetentionRate: NotationSymbol  # fraction of collection attempts that survive task failure and quality filtering as usable demonstrations ($R$) [dimensionless]
    DemonstratorAction: NotationSymbol  # action vector the demonstrator recorded at one step, the reference a policy prediction is scored against ($\mathbf{a}^{*}$)
    DemonstratorActionComponent: NotationSymbol  # one component of the action a demonstrator recorded, indexed by component and written with a star ($a$) [varies]
    DenseEvaluationEnergy: NotationSymbol  # energy one evaluation of a periodically evaluated dense model consumes ($E_{\text{eval}}$) [J]
    DenseEvaluationRate: NotationSymbol  # number of times per second a periodically evaluated dense model runs ($f_{\text{eval}}$) [1/s]
    DenseVoxelGridMemory: NotationSymbol  # memory a dense voxel grid over a workspace occupies, voxels times bytes per voxel ($M_{\text{dense}}$) [bytes]
    DepartmentQuotaLimit: NotationSymbol  # total capacity limit of a department in a hierarchical quota tree ($Q_{\text{department}}$)
    DependencyGraphEdgeSet: NotationSymbol  # set of directed edges of a task-precedence graph, each meaning the head vertex needs a precondition the tail vertex supplies ($E$)
    DependencyGraphVertex: NotationSymbol  # one vertex of a task-precedence graph, a bounded unit of execution that occupies one worker slot ($v$)
    DependencyGraphVertexSet: NotationSymbol  # set of vertices of a task-precedence graph, each a bounded unit of execution such as an agent trajectory or a runtime check ($V$)
    DepletedBusVoltage: NotationSymbol  # DC bus voltage at the depleted end of the battery's operating range ($V_{\text{low}}$) [V]
    DeploymentAge: NotationSymbol  # time a model has been in service since deployment or its most recent retraining ($t$) [s]
    DeploymentDistribution: NotationSymbol  # joint distribution of inputs and labels a deployed model must serve, the population a verification guarantee is stated over ($p_{\text{deploy}}$)
    DeploymentDuration: NotationSymbol  # length of time one model version stays in production serving traffic ($T_{\text{deployment}}$) [s]
    DepthBinCount: NotationSymbol  # number of discrete depth hypotheses a lift-splat network scores along each camera ray ($D$)
    DepthBinDistribution: NotationSymbol  # categorical distribution over the depth bins of one feature cell's ray, the softmax of its depth logits ($\boldsymbol{\alpha}_{u, v}$)
    DepthBinIndex: NotationSymbol  # index of one discrete depth hypothesis along a camera ray ($k$)
    DepthBinLogits: NotationSymbol  # logits a lift-splat network predicts over the depth bins of one feature cell's ray ($\mathbf{d}_{u, v}$)
    DepthBinProbability: NotationSymbol  # probability one feature cell's depth distribution assigns to one depth bin, as a function of the bin index ($\alpha_{u, v}$) [dimensionless]
    DepthCalibrationScale: NotationSymbol  # multiplicative scale of the affine calibration that converts a relative-depth network's output to metric depth ($s$) [m]
    DepthCalibrationShift: NotationSymbol  # additive shift of the affine calibration that converts a relative-depth network's output to metric depth ($t$) [m]
    DepthCriticalPathLatency: NotationSymbol  # critical-path latency of one decision when test-time compute extends a single candidate along the depth axis ($T^{\text{depth}}$) [s]
    DepthFrameRate: NotationSymbol  # rate at which a depth camera streams range frames ($f_{\text{cam}}$) [Hz]
    DepthHypothesis: NotationSymbol  # metric depth of one discrete depth hypothesis along a camera ray, indexed by bin ($d$) [m]
    DepthModeComponentVector: NotationSymbol  # column of the depth-mode factor matrix for one rank-one component of a CP decomposition ($\mathbf{c}_s$)
    DepthModeFactorMatrix: NotationSymbol  # factor matrix for the depth mode of a tensor decomposition, one row per depth index and one column per rank component ($\mathbf{C}$)
    DepthModeFactorSlice: NotationSymbol  # row of the depth-mode factor matrix for one depth index, holding its weight on every rank component ($\mathbf{c}$)
    DepthRayCount: NotationSymbol  # number of rays one depth frame casts into a volumetric map, one per depth pixel ($N_{\text{rays}}$)
    DepthScalingRatio: NotationSymbol  # ratio by which compound scaling multiplies network depth for each unit of the scaling coefficient ($\alpha$) [dimensionless]
    DepthStandardDeviation: NotationSymbol  # standard deviation of the depth a stereo or range sensor reports for one point ($\sigma_z$) [m]
    DerivativeControlEffort: NotationSymbol  # derivative-action component of a PID command, the derivative gain matrix times the error rate or the filtered measured velocity ($\mathbf{u}_{d}$)
    DerivativeFilterTimeConstant: NotationSymbol  # time constant of the first-order low-pass filter applied to a PID derivative term, the inverse of its roll-off frequency ($T_{f}$) [s]
    DerivativeGainMatrix: NotationSymbol  # matrix of derivative feedback gains acting on the tracking error rate in joint-space PD or PID control ($\mathbf{K}_{d}$)
    DerivativeOperator: NotationSymbol  # differential operator d of an ordinary derivative, written over the differential in a Leibniz quotient ($d$)
    DerivativeSplineControlPoint: NotationSymbol  # control point of the derivative of a B-spline, a scaled difference of adjacent control points ($\mathbf{v}$) [varies]
    DerivedMemoryEntry: NotationSymbol  # derived memory entry, such as a summary, embedding or consolidated lesson, whose validity is tracked by source generation ($c$)
    DerivedRecord: NotationSymbol  # record a transformation derives from labeled input records, carrying the union of their provenance and the intersection of their confidentiality labels ($y$)
    DesignLatencyQuantile: NotationSymbol  # quantile of a latency distribution at one minus the per-handoff exceedance probability, the design tail a replacement deadline is budgeted against ($Q_{1-\epsilon}$) [s]
    DesiredContactPosition: NotationSymbol  # setpoint position of the robot's contact point along the contact direction ($x_{\text{des}}$) [m]
    DesiredContactVelocity: NotationSymbol  # setpoint velocity of the robot's contact point along the contact direction ($\dot{x}_{\text{des}}$) [m/s]
    DesiredTaskAcceleration: NotationSymbol  # desired task-space acceleration of the end effector in an impedance law ($\ddot{\mathbf{x}}_d$) [varies]
    DesiredTaskPosition: NotationSymbol  # desired task-space position of the end effector in an impedance law ($\mathbf{x}_d$) [varies]
    DesiredTaskSpaceAcceleration: NotationSymbol  # commanded acceleration of a whole-body control task's operational-space coordinates, the reference acceleration plus proportional-derivative correction of the task tracking error, which the whole-body QP tries to realize ($\ddot{\mathbf{x}}^{\text{des}}$)
    DesiredTaskVelocity: NotationSymbol  # desired task-space velocity of the end effector in an impedance law ($\dot{\mathbf{x}}_d$) [varies]
    DestinationRankIndex: NotationSymbol  # position of the rank that receives a chunk in a collective exchange, such as AllToAll ($j$)
    DetectionCount: NotationSymbol  # number of candidate detections one perception frame yields ($n_{\text{det}}$)
    DetectionInterval: NotationSymbol  # time a detector takes to recognize a hazard or an expiry, beyond what the observation age already charges ($t_{\text{detect}}$) [s]
    DetectionMeasurementNoiseCovariance: NotationSymbol  # measurement noise covariance of candidate detection j in data association ($\boldsymbol{\Sigma}_{\text{meas},j}$)
    DetectionRangeShortfall: NotationSymbol  # extra distance a machine travels before it reliably detects a person of one group, compared with the best-detected group, at the same speed ($\Delta d_{\text{detect}}$) [m]
    DetectionTimestamp: NotationSymbol  # reference-clock timestamp at which a fault trial observes the NMI or diagnostic flag ($t_{\text{det}}$) [s]
    DetectorMissProbability: NotationSymbol  # probability that a hazard detector fails to report a hazard it encounters ($p_{\text{miss}}$) [dimensionless]
    DeterministicPolicy: NotationSymbol  # deterministic policy, the actor network with parameters theta that maps a state to one action ($\mu_{\theta}$)
    DevelopmentIterationCount: NotationSymbol  # number of development iterations that each force a recompilation ($N_{\text{dev}}$)
    DeviationDisadvantage: NotationSymbol  # largest extra expected cost of one deviation from the reference action followed by reference recovery, at any learner-reachable state ($u$) [dimensionless]
    DeviceCount: NotationSymbol  # number of devices (accelerators) taking part in one distributed job, also the rank count of a collective ($N$)
    DeviceEnergy: NotationSymbol  # energy the compute devices of a fleet consume over a stated interval ($E_{\text{device}}$) [J]
    DeviceHardwareTime: NotationSymbol  # time the device spends executing a step's kernels, compute plus memory time when they do not overlap ($T_{\text{hw}}$) [s]
    DeviceMemoryCapacity: NotationSymbol  # total memory a device provides for a model's weights and activations ($M_{\text{device}}$) [bytes]
    DevicePower: NotationSymbol  # electrical power one device draws ($P_{\text{device}}$) [W]
    DeviceRequestCapacity: NotationSymbol  # largest number of requests a device can hold in service at once, its concurrent request slots ($Q_{\text{req,max}}$)
    DeviceSilentCorruptionProbability: NotationSymbol  # probability that one device suffers at least one residual silent data corruption, an error the hardware does not correct, over a stated observation interval ($p$) [dimensionless]
    DiagnosticCoverage: NotationSymbol  # rate-weighted fraction of hardware faults whose hazardous effect a diagnostic mitigates ($C_{\text{diag}}$) [dimensionless]
    DictionaryCoefficientMatrix: NotationSymbol  # block-sparse coefficient matrix recording which dictionary patterns are active in each stored example ($\mathbf{C}$)
    DifferentiabilityClass: NotationSymbol  # class of functions whose derivatives up to the order written as its superscript exist and are continuous ($C$)
    DifferentialEntropy: NotationSymbol  # differential entropy of a continuous random variable, the negative expected log density ($h$) [nats]
    DiffusionCumulativeSchedule: NotationSymbol  # fraction of the clean signal variance retained after k diffusion steps ($\bar{\alpha}$) [dimensionless]
    DiffusionGaussianNoise: NotationSymbol  # standard Gaussian noise added to clean action chunks in the diffusion forward process ($\boldsymbol{\epsilon}$)
    DiffusionMatchingLoss: NotationSymbol  # mean squared error between the injected noise and the noise the network predicts, averaged over diffusion steps ($\mathcal{L}_{\text{diff}}$)
    DiffusionNoiseVariance: NotationSymbol  # variance of the Gaussian noise the forward diffusion process adds at one step, indexed by step ($\beta$) [dimensionless]
    DiffusionStepCount: NotationSymbol  # number of discrete steps in the forward diffusion noise schedule ($N$)
    DiffusionStepIndex: NotationSymbol  # position of one step in the diffusion noising or denoising sequence ($k$)
    DiffusionStepRetention: NotationSymbol  # fraction of the signal variance one forward diffusion step retains, one minus that step's noise variance ($\alpha$) [dimensionless]
    DiffusionTimeCoordinate: NotationSymbol  # normalized continuous position along the forward diffusion process, from clean data at zero to noise at one ($t_{\text{diff}}$) [dimensionless]
    DirectAxisCurrent: NotationSymbol  # direct-axis current of a field-oriented drive, along the rotor field, which produces no torque ($i_d$) [A]
    DirectAxisInductance: NotationSymbol  # stator inductance along the direct axis of the rotor dq frame ($L_d$) [H]
    DirectAxisVoltage: NotationSymbol  # direct-axis voltage the inverter applies in a field-oriented drive ($v_d$) [V]
    DirectionalCosineMatrix: NotationSymbol  # three-by-three rotation matrix giving the orientation of one frame relative to another ($\mathbf{R}$) [dimensionless]
    DirectionalDerivativeStep: NotationSymbol  # scalar step along a direction at which a directional derivative is taken in the limit of zero ($\epsilon$) [s]
    DisadvantagedGroupClassificationThreshold: NotationSymbol  # classification threshold applied only to the disadvantaged group in a group-specific thresholding intervention ($\gamma_{\text{cls},g=\text{B}}$) [dimensionless]
    DiscordantPairCount: NotationSymbol  # number of paired tasks on which exactly one of the two compared configurations is accepted, the sum of the two discordant counts ($n_{\text{disc}}$)
    DiscordantPairFraction: NotationSymbol  # share of paired tasks on which exactly one of the two compared configurations is accepted ($\psi$) [dimensionless]
    DiscountFactor: NotationSymbol  # factor by which a reward arriving one turn later is weighted in the return, between zero and one ($\gamma$) [dimensionless]
    DiscountedReturn: NotationSymbol  # discounted sum of the rewards an episode collects from a given step onward, the return of reinforcement learning ($G$) [dimensionless]
    DiscountedReturnStepOffset: NotationSymbol  # number of control steps after the current decision at which a robot task reward enters the discounted return a policy-search objective sums ($j$)
    DiscountedStateVisitationDistribution: NotationSymbol  # normalized discounted distribution of states a policy visits, one minus the discount times the discounted sum of its per-step state probabilities ($d$)
    DiscreteLqrFeedbackGain: NotationSymbol  # state-feedback gain matrix of discrete-time LQR, from the discrete Riccati solution ($\mathbf{K}_{\text{lqr},d}$)
    DiscreteLqrQuadraticObjective: NotationSymbol  # infinite-horizon quadratic cost of discrete-time LQR: the sum over control steps of the weighted squared state plus the weighted squared control effort ($J_d$) [varies]
    DiscreteOutcome: NotationSymbol  # one outcome of a discrete random variable, the value at which a probability distribution is evaluated or summed ($x$)
    DiscretePlantDynamics: NotationSymbol  # discrete-time dynamics of the physical plant: the map from the current state and applied command to the next state ($f$)
    DiscreteRandomVariable: NotationSymbol  # discrete random variable, the uncertain quantity whose outcomes a distribution describes ($X$)
    DiscreteSampleIndex: NotationSymbol  # position of one sample in a fixed-period discrete-time update, written in brackets as x[k] ($k$)
    DiscreteStateSpaceInputMatrix: NotationSymbol  # discrete-time input matrix of a sampled linear model under zero-order hold, mapping a held input to the next state ($\mathbf{B}_{d}$)
    DiscreteStateSpaceSystemMatrix: NotationSymbol  # discrete-time system matrix of a sampled linear model under zero-order hold, the matrix exponential of the system matrix times the sampling period ($\mathbf{A}_{d}$)
    DiscretizationBinCount: NotationSymbol  # number of bins into which each state coordinate is partitioned when the state space is gridded ($n_{\text{bin}}$)
    DisengagedTakeoverResponseTime: NotationSymbol  # time an out-of-loop human operator, not monitoring the machine, needs to take over control after a handover request, including console video transport and handshake ($T_{\Sigma,2}$) [s]
    DiskBandwidth: NotationSymbol  # sustained rate at which a storage disk delivers bytes ($\text{BW}_{\text{disk}}$) [bytes/s]
    DisparityStandardDeviation: NotationSymbol  # standard deviation of the disparity a stereo matcher measures ($\sigma_d$) [pixels]
    DispatchAdmissionCeiling: NotationSymbol  # longest dispatch-to-admission interval the spare stopping clearance can pay for at the machine's speed ($\Delta_{\text{da,max}}$) [s]
    DispatchAdmissionInterval: NotationSymbol  # time from dispatching a proposal to its admission by the permission path ($\Delta_{\text{da}}$) [s]
    DispatchObservationAge: NotationSymbol  # elapsed physical time from sensor exposure midpoint to dispatch of the processed observation ($t_{\text{age}}$) [s]
    DispatchedOperationCount: NotationSymbol  # number of operations a framework dispatches, each paying its own launch overhead ($N_{\text{ops}}$)
    DisplayPipelineLatency: NotationSymbol  # delay from a physical event to its appearance on the teleoperator's display ($t_{\text{disp}}$) [s]
    DispreferredReferenceRelativeScore: NotationSymbol  # reference-relative score of the dispreferred trajectory record of one comparison pair ($\Delta_l$) [dimensionless]
    DispreferredTrajectory: NotationSymbol  # trajectory record ranked lower in one preference comparison ($\tau_l$)
    DistanceEstimateStandardDeviation: NotationSymbol  # standard deviation of the scalar distance estimate ($\sigma$) [m]
    DistanceMeasurement: NotationSymbol  # one scalar range measurement of the remaining distance, from LiDAR or stereo ($z$) [m]
    DistanceNormOrder: NotationSymbol  # order of the vector norm in which query-candidate distances are measured ($p$) [dimensionless]
    DistanceQueryPoint: NotationSymbol  # point of state space whose distance to a set is measured ($\mathbf{y}$)
    DistillationClassProbability: NotationSymbol  # probability that the softmax at the distillation temperature assigns to one class ($p^{(T_{\text{distill}})}$) [dimensionless]
    DistillationLoss: NotationSymbol  # knowledge-distillation training objective, a weighted sum of hard-label cross-entropy and the temperature-scaled divergence from the teacher ($\mathcal{L}_{\text{distill}}$)
    DistillationLossWeight: NotationSymbol  # weight in [0, 1] given to the soft-target distillation term, with the rest on the hard-label term ($\gamma_{\text{KD}}$) [dimensionless]
    DistillationTemperature: NotationSymbol  # softmax temperature that softens teacher and student output probabilities in knowledge distillation ($T_{\text{distill}}$) [dimensionless]
    DistortedImageAbscissa: NotationSymbol  # first normalized image coordinate after radial lens distortion ($x_d$) [dimensionless]
    DistortedImageOrdinate: NotationSymbol  # second normalized image coordinate after radial lens distortion ($y_d$) [dimensionless]
    DistributedStepTime: NotationSymbol  # wall-clock time of one training step on N devices ($T_{\text{step}}$) [s]
    DisturbanceAccelerationBound: NotationSymbol  # bound on the unmodeled acceleration that disturbance forces impart to an unobserved body ($a_{\text{dist}}$) [m/s^2]
    DisturbanceBasisCoefficient: NotationSymbol  # coefficient of one basis function in a parameterized disturbance trajectory, a component of the disturbance parameter vector ($\psi$) [varies]
    DisturbanceBasisCount: NotationSymbol  # number of basis functions composing a parameterized disturbance trajectory ($K$)
    DisturbanceBasisFunction: NotationSymbol  # time-dependent basis function, such as a B-spline or Fourier mode, from which a disturbance trajectory is composed ($B$) [dimensionless]
    DisturbanceBudget: NotationSymbol  # largest peak magnitude a falsification disturbance trajectory may take, derived from the operating envelope ($\Delta_{w}$) [varies]
    DisturbanceDimension: NotationSymbol  # number of components of the disturbance input vector ($p$)
    DisturbanceForceBound: NotationSymbol  # qualified upper bound on the magnitude of the external disturbance force acting on the plant ($d_{\max}$) [N]
    DisturbanceParameterSet: NotationSymbol  # compact set of disturbance parameter vectors a falsification search ranges over ($\Psi$)
    DisturbanceParameterVector: NotationSymbol  # vector of basis-function coefficients that parameterizes a falsification disturbance trajectory ($\boldsymbol{\psi}$) [varies]
    DisturbanceTrajectory: NotationSymbol  # time-varying physical disturbance a falsification search injects, parameterized by smooth basis functions ($w_{\text{dist}}$) [varies]
    DivergenceInstant: NotationSymbol  # time at which an autonomous policy's rollout begins to leave its supported states ($t_{\text{div}}$) [s]
    DivergenceMeasure: NotationSymbol  # divergence function that measures how far one probability distribution lies from another, with the specific measure left open ($\mathcal{D}$)
    DivergenceTolerance: NotationSymbol  # positive threshold on the trajectory-divergence residual that marks divergence ($\delta_{\text{tol}}$) [m]
    DmaQueueTime: NotationSymbol  # time a frame spends in direct memory access transfer and the driver queue ($t_{\text{DMA}}$) [s]
    DocumentPayloadFraction: NotationSymbol  # fraction of retrievable untrusted documents that carry an injected payload, each read treated as an independent draw ($q$) [dimensionless]
    DomainRandomizationDistribution: NotationSymbol  # distribution from which domain randomization samples simulator parameter vectors ($\mathcal{D}_{\text{DR}}$)
    DomainRandomizationObjective: NotationSymbol  # expected return of a policy averaged over the distribution of simulator parameters that domain randomization samples ($J_{\text{DR}}$) [dimensionless]
    DominantComponentEnergy: NotationSymbol  # larger of the compute and data-movement energy contributions of a workload, the energy-roofline term ($E_{\text{dominant}}$) [J]
    DominantShare: NotationSymbol  # a tenant's largest share of fleet capacity across resource types, the quantity dominant resource fairness equalizes ($d$) [dimensionless]
    DoorOpenReward: NotationSymbol  # reward the latch-opening task pays on a step at which the door is open ($r_{\text{open}}$) [dimensionless]
    DoubleIntegratorPosition: NotationSymbol  # position of a one-dimensional double-integrator model, as a function of time ($p$) [m]
    DoubleIntegratorPositionDerivative: NotationSymbol  # time derivative of the double-integrator position, equal to its velocity ($\dot{p}$) [m/s]
    DoubleIntegratorVelocity: NotationSymbol  # velocity of a one-dimensional double-integrator model, as a function of time ($v$) [m/s]
    DoubleIntegratorVelocityDerivative: NotationSymbol  # time derivative of the double-integrator velocity, its acceleration, equal to the acceleration noise ($\dot{v}$) [m/s^2]
    DoubleTreeAllReduceTime: NotationSymbol  # time to complete an AllReduce over two complementary pipelined binary trees, each carrying half the payload ($T_{\text{double-tree}}$) [s]
    DownstreamStageLatency: NotationSymbol  # latency of the next stage of a sense-to-actuation pipeline, as a random variable ($Y$) [s]
    DpoLoss: NotationSymbol  # direct preference optimization loss, the expected negative log-sigmoid of the scaled difference of reference-relative scores over comparison pairs ($\mathcal{L}_{\text{DPO}}$) [dimensionless]
    DraftAcceptanceRate: NotationSymbol  # fraction of draft-model tokens the target model accepts in speculative decoding ($\alpha$) [dimensionless]
    DraftCandidateToken: NotationSymbol  # one token a draft proposes in a speculative-sampling iteration, written with its draft position ($\tilde{x}$)
    DraftModelDistribution: NotationSymbol  # next-token probability distribution of the draft model in speculative decoding ($q_{\phi}$)
    DraftPositionAcceptanceProbability: NotationSymbol  # probability that exact speculative sampling accepts the draft candidate at one position, the target-to-draft probability ratio capped at one ($\alpha_{\text{spec}, j}$) [dimensionless]
    DraftSamplingDistribution: NotationSymbol  # normalized next-token distribution the draft sampler draws from at a prefix in speculative sampling, after temperature, masks and other sampling transformations ($q$)
    DraftTokenAcceptanceProbability: NotationSymbol  # probability that the target model accepts one drafted token in speculative decoding, taken as independent across draft positions ($\alpha_{\text{acc}}$) [dimensionless]
    DraftTokenCostRatio: NotationSymbol  # time cost of generating one draft token as a fraction of one target-model decoding step in speculative decoding ($\beta$) [dimensionless]
    DraftTokenCount: NotationSymbol  # number of tokens a draft model proposes for the target model to verify in one speculative-decoding step ($\gamma$)
    DraftedToken: NotationSymbol  # one token the draft model proposes in a speculative-decoding round, numbered by its position ($T$)
    DragonflyGlobalChannelCount: NotationSymbol  # number of global channels, links to other groups, on each router of a dragonfly topology ($h$)
    DragonflyGroupRouterCount: NotationSymbol  # number of routers in one fully connected group of a dragonfly topology ($a$)
    DragonflyTerminalPortCount: NotationSymbol  # number of terminal (endpoint-facing) ports on each router of a dragonfly topology ($p$)
    DrainSourceVoltage: NotationSymbol  # voltage between the drain and source of a power switch ($V_{\text{ds}}$) [V]
    DriftElapsedTime: NotationSymbol  # time elapsed since a mobile base with a heading error or yaw-rate bias left its reference line, over which its lateral drift accumulates ($t_{\text{drift}}$) [s]
    DriftImpactTime: NotationSymbol  # time until a persistent heading bias carries a mobile base across its lateral clearance into an obstacle ($t_{\text{impact}}$) [s]
    DriftLieDerivativeOperator: NotationSymbol  # Lie derivative operator along the drift field, mapping a scalar function of the state to its rate of change under the unforced dynamics ($L_{\mathbf{f}}$)
    DriftSpeedBound: NotationSymbol  # bound on the unobserved speed at which a target drifts relative to its last measured position ($v_{\text{drift}}$) [m/s]
    DriftTestSampleSize: NotationSymbol  # number of observations in one of the two samples a two-sample drift test compares ($n$)
    DriftVelocityBound: NotationSymbol  # bound on the unmodeled velocity at which an unobserved estimate drifts from the true state ($v_{\text{drift}}$) [m/s]
    DriveBrakingForceCeiling: NotationSymbol  # largest retarding force a machine's current-limited drives can put on the floor ($F$) [N]
    DriveBrakingTorqueCeiling: NotationSymbol  # largest braking torque one current-limited drive applies at its wheel, the torque constant times the drive's current limit ($\tau$) [N*m]
    DrivenWheelLoadFraction: NotationSymbol  # fraction of a vehicle's total weight carried by its driven wheels ($\phi_{\text{drive}}$) [dimensionless]
    DriverInterruptLatency: NotationSymbol  # time the host driver takes to service the device interrupt before stamping a packet ($t_{\text{driver\_irq}}$) [s]
    DrivetrainAntiresonanceFrequency: NotationSymbol  # antiresonance frequency of a two-inertia drivetrain, set by its torsional stiffness and the load inertia alone ($f_{\text{anti}}$) [Hz]
    DrivetrainResonanceFrequency: NotationSymbol  # resonance frequency of a two-inertia drivetrain, set by its torsional stiffness and both the load and reflected rotor inertia ($f_{\text{res}}$) [Hz]
    DroopedSupplyVoltage: NotationSymbol  # supply voltage a core sees at the bottom of a droop event ($V_{\text{droop}}$) [V]
    DropletDiameter: NotationSymbol  # diameter of one airborne water droplet or particle ($d_{\text{droplet}}$) [m]
    DropoutProbability: NotationSymbol  # probability that a dropout layer zeroes each element during training ($p_{\text{drop}}$) [dimensionless]
    DualNumberInfinitesimal: NotationSymbol  # nonzero infinitesimal unit of dual numbers, whose square is zero ($\epsilon$) [dimensionless]
    DuplicateEffectProbability: NotationSymbol  # probability that one blind retry of a non-idempotent call applies its effect twice ($p_{\text{dup}}$) [dimensionless]
    DurableAcknowledgmentTime: NotationSymbol  # time at which a group-commit batch's durable acknowledgment becomes available ($t_{\text{ack}}$) [s]
    DurableStoreWriteBandwidth: NotationSymbol  # sustained rate at which the collector writes records to durable storage ($\text{BW}_{\text{store}}$) [bytes/s]
    DutyCyclePeriod: NotationSymbol  # length of one wake-and-sleep cycle of a duty-cycled device ($T_{\text{period}}$) [s]
    DynamicPower: NotationSymbol  # power a chip draws switching its transistors, proportional to switched capacitance, supply voltage squared and clock frequency ($P_{\text{dynamic}}$) [W]
    DynamicReachableSet: NotationSymbol  # set of states reachable from the measured state within a given remaining time under the stated limits ($\mathcal{R}$)
    DynamicsRealityGap: NotationSymbol  # dynamics component of the reality gap: discrepancy between simulated and physical actuator response and contact mechanics under identical commands ($\Delta_{\text{dyn}}$) [varies]
    EagerExecutionTime: NotationSymbol  # time for one execution of a model in eager mode ($T_{\text{eager}}$) [s]
    EarlierBaseFlangeTransform: NotationSymbol  # base-to-flange forward-kinematics transform at the earlier of two hand-eye calibration poses ($\mathbf{T}_{\text{base}, k}^{\text{flange}}$)
    EarlierCameraTargetTransform: NotationSymbol  # camera-to-target transform at the earlier of two hand-eye calibration poses ($\mathbf{T}_{\text{cam}, k}^{\text{target}}$)
    EarlierControlStepIndex: NotationSymbol  # control step before the current step t, the running index over earlier steps in a union, sum or for-all condition ($t'$)
    EarlierWorkerCount: NotationSymbol  # number of workers that have already chosen their write sets when the next worker chooses, the running index of the write-collision product ($k$)
    EarliestConflictTime: NotationSymbol  # earliest time at which a candidate corridor is predicted to intersect an actor forward occupancy set ($t_{\text{conflict}}$) [s]
    EarthReferenceFrame: NotationSymbol  # earth-fixed coordinate frame, the optional geodetic root of a robot's frame tree ($\mathcal{F}_{\text{earth}}$)
    EchoFactor: NotationSymbol  # number of times data echoing reuses each sample the input pipeline produces ($e$) [dimensionless]
    EchoRoundTripTime: NotationSymbol  # delay of a radar echo behind its transmitted chirp, the round trip to the reflector ($t_{\text{rt}}$) [s]
    EditGrantRights: NotationSymbol  # set of operations the edit-phase capability grant permits ($\mathcal{R}_{\Gamma_{\text{edit}}}$)
    EditPhaseCapabilityGrant: NotationSymbol  # capability grant for the edit phase of a repair, read and write over the repository ($\Gamma_{\text{edit}}$)
    EffectiveAdmissionCeiling: NotationSymbol  # largest allocation a team can be admitted to before any reclamation, its cap or the department's unused capacity, whichever is smaller ($Q_{\text{effective}}$)
    EffectiveApiOutputTokenPrice: NotationSymbol  # effective per-token API charge per output token, including the input tokens associated with it at the workload's fixed mix and cache hit rate ($c_{\text{API}}$) [USD/token]
    EffectiveBrakingDelay: NotationSymbol  # pre-brake delay plus the equivalent delay the jerk ramp adds before full credible deceleration, the delay term of the speed ceiling ($T_{\text{eff}}$) [s]
    EffectiveBranchingFactor: NotationSymbol  # expected number of children of an expanded search node that survive pruning ($b_{\text{eff}}$) [dimensionless]
    EffectiveConfidentialityLabel: NotationSymbol  # set of recipients permitted to receive a handle argument, the intersection of the handle's own confidentiality label and that of the object it designates ($\kappa_{\text{eff}}$)
    EffectiveFleetThroughput: NotationSymbol  # useful floating-point rate a fleet sustains after per-device, scaling and operational losses ($R_{\text{eff}}$) [FLOP/s]
    EffectiveImpactMass: NotationSymbol  # apparent mass the robot presents along the contact normal at a given contact point and configuration ($m_{\text{eff}}$) [kg]
    EffectiveIntegrityProvenance: NotationSymbol  # integrity provenance a handle argument is checked with, the union of the handle's own provenance and that of the object it designates ($\iota_{\text{eff}}$)
    EffectiveJointInertia: NotationSymbol  # total inertia seen at one joint, link and payload plus reflected rotor inertia ($J_{\text{eff}}$) [kg*m^2]
    EffectiveLossResistance: NotationSymbol  # effective resistance that converts torque-producing current into winding loss ($R_{\text{eff}}$) [Ohm]
    EffectiveSampleSize: NotationSymbol  # number of independent observations that carry the same information about a mean as a serially correlated stream ($N_{\text{eff}}$)
    EffectiveSeriesStiffness: NotationSymbol  # effective stiffness of a contact whose two sides deflect in series, the reciprocal of the sum of their compliances ($k_{\text{eff}}$) [N/m]
    EffectiveServiceRate: NotationSymbol  # requests a batched server completes per second at a given batch size ($\mu_{\text{eff}}$) [requests/s]
    EffectiveSpotTrainingCost: NotationSymbol  # spot price scaled by billed time over productive time, the cost of each hour of retained training progress ($C_{\text{effective}}$) [USD/hour]
    EffectiveStorageThroughput: NotationSymbol  # storage throughput a training job actually achieves, after record size, layout, caching and shuffling losses ($R_{\text{storage,eff}}$) [bytes/s]
    EffectiveWasteFraction: NotationSymbol  # padding-waste fraction that would produce a given realized continuous-batching speedup ($W_{\text{effective}}$) [dimensionless]
    EfficiencyHw: NotationSymbol  # fraction of peak compute rate the hardware actually sustains on a workload ($\eta_{\text{hw}}$) [dimensionless]
    EgressArrivalRate: NotationSymbol  # aggregate rate at which traffic arrives at one switch egress port ($\lambda_{\text{arr}}$) [bytes/s]
    EgressBufferCapacity: NotationSymbol  # largest amount of traffic one switch egress buffer can hold ($Q_{\text{max}}$) [bytes]
    EgressPortUtilization: NotationSymbol  # ratio of the traffic arrival rate at one switch egress port to the rate the port drains it ($\rho_{\text{egress}}$) [dimensionless]
    EgressPrice: NotationSymbol  # price of transferring one gigabyte out over the network ($R_{\text{egress/GB}}$) [USD/GB]
    EgressQueueOccupancy: NotationSymbol  # amount of traffic queued at one switch egress port at a given time ($Q$) [bytes]
    EgressServiceRate: NotationSymbol  # rate at which one switch egress port drains queued traffic ($\mu$) [bytes/s]
    ElapsedPlanningPeriods: NotationSymbol  # number of planning periods elapsed since the data-debt backlog was measured ($n$)
    ElasticJointTorque: NotationSymbol  # vector of torques transmitted through the elastic joints, stiffness times the motor-link angle difference ($\mathbf{z}$) [N*m]
    ElasticMaximumWorkerCount: NotationSymbol  # largest number of workers an elastic job declares it can use, the upper end of its elastic range ($N_{\text{max}}$)
    ElasticMinimumWorkerCount: NotationSymbol  # smallest number of workers an elastic job declares it can run on, the lower end of its elastic range ($N_{\text{min}}$)
    ElasticModalFrequencyMatrix: NotationSymbol  # matrix whose eigenvalues are the natural frequencies of the elastic joint modes in a frozen configuration, the square root of stiffness times inverse inertia ($\boldsymbol{\Omega}$) [rad/s]
    ElasticPotentialEnergy: NotationSymbol  # elastic potential energy a deflected spring stores ($E$) [J]
    ElasticStrainEnergy: NotationSymbol  # elastic strain energy stored in a compressed linear contact spring, one half stiffness times compression squared ($U_{\text{elastic}}$) [J]
    ElasticTorqueSecondDerivative: NotationSymbol  # second time derivative of the elastic joint torque vector ($\ddot{\mathbf{z}}$) [N*m/s^2]
    ElbowSwivelAngle: NotationSymbol  # angle of the elbow on its orbit around the shoulder-to-wrist axis of a redundant arm ($\psi$) [rad]
    ElectricalAngularSpeed: NotationSymbol  # electrical angular speed of a motor, the rotor speed times the pole-pair count ($\omega_e$) [rad/s]
    ElectricalResistance: NotationSymbol  # electrical resistance of a circuit or winding ($R$) [Ohm]
    ElectricalRotorAngle: NotationSymbol  # electrical angle of a motor rotor, the mechanical angle times the pole-pair count ($\theta_e$) [rad]
    ElectricalTimeConstant: NotationSymbol  # time constant of current rise in a motor winding, inductance over resistance ($\tau_e$) [s]
    ElectricityPrice: NotationSymbol  # price of electrical energy ($R_{\$/\text{kWh}}$) [USD/(W*h)]
    ElementStorageSize: NotationSymbol  # bytes needed to store one tensor element at its numerical precision ($s_{\text{elem}}$) [bytes/element]
    ElementaryCharge: NotationSymbol  # electric charge of one electron, the elementary charge ($q$) [A*s]
    ElementaryOperationCount: NotationSymbol  # number of elementary operations a computational graph needs to evaluate a function once, its forward work ($W$)
    ElevationAngle: NotationSymbol  # elevation angle of a direction above the horizontal plane ($\phi$) [rad]
    EligibleCachedPrefixLength: NotationSymbol  # number of a call's input tokens in a resident prefix eligible for the cached-input rate ($S_{\text{hit}}$)
    EligibleTargetPositionSet: NotationSymbol  # set of token positions in one serialized example that the declared objective selects as imitation targets ($\mathcal{P}_{\text{elig}}$)
    EligibleVectorCount: NotationSymbol  # number of stored vectors eligible for one vector search after scope filtering, the vectors an exact scan streams ($N$)
    EliteSampleFraction: NotationSymbol  # fraction of each cross-entropy batch kept as elite samples, which sets the intermediate threshold quantile ($\beta$) [dimensionless]
    EmbeddingDimension: NotationSymbol  # number of components in each embedding vector stored in a vector index ($d$)
    EmbeddingShardCount: NotationSymbol  # number of shards an embedding table is partitioned into ($N_{\text{shards}}$)
    EmergencyStopContact: NotationSymbol  # positive-break contact of an emergency-stop switch, one per redundant channel of the safety circuit ($S$)
    EmergencyStopCount: NotationSymbol  # number of emergency protective stops recorded across a campaign's trials ($n_{\text{estop}}$)
    EmittedArtifactCount: NotationSymbol  # number of valid training artifacts a collection cohort emits for one designated consumer by an accounting cutoff ($N_{\text{emit}}$)
    EmittedTokenCount: NotationSymbol  # number of tokens a decoder emits one after another for one output ($S_{\text{out}}$)
    EmpiricalBellmanOperator: NotationSymbol  # empirical Bellman backup operator that maps an action-value estimate to its one-step target, reward plus discounted next value ($\mathcal{B}$)
    EmpiricalExceedanceFraction: NotationSymbol  # share of the recorded observations that exceed the threshold, an estimate of the exceedance probability ($\hat{\zeta}_{u}$) [dimensionless]
    EmpiricalTimestepAverage: NotationSymbol  # empirical average over the sampled timesteps of a batch of rollouts, the sample estimate of an expectation ($\hat{\mathbb{E}}$)
    EmpiricalTransferDiscrepancy: NotationSymbol  # largest weighted state error observed between paired simulated and physical runs, the empirical estimate of the transfer discrepancy ($\hat{\epsilon}_{\text{gap}}$) [dimensionless]
    EmptyByteString: NotationSymbol  # empty byte string, the string of length zero ($\epsilon$)
    EncodedLatentState: NotationSymbol  # latent state estimate a learned neural encoder produces from sensor observations ($\hat{\mathbf{z}}$)
    EncoderAngle: NotationSymbol  # joint angle an encoder reports ($\theta_{\text{enc}}$) [rad]
    EncoderBitCount: NotationSymbol  # number of bits an absolute encoder resolves per revolution ($n$)
    EncoderLatentRepresentation: NotationSymbol  # intermediate representation a shared encoder computes from an input, consumed by the heads that sit on top of it ($\mathbf{z}$)
    EncoderVelocityEstimate: NotationSymbol  # joint velocity estimated by differencing successive encoder angle samples ($\dot{\theta}_{\text{enc}}$) [rad/s]
    EncoderVelocityNoise: NotationSymbol  # standard deviation of the velocity estimate's error caused by encoder quantization ($\sigma_{\dot{\theta}_{\text{enc}}}$) [rad/s]
    EndEffectorCameraTransform: NotationSymbol  # rigid transform mapping wrist-camera coordinates into the end-effector frame, the camera's mounting extrinsic ($\mathbf{T}_{EC}$)
    EndEffectorFirstAxisPosition: NotationSymbol  # end-effector position along the first axis of the declared Cartesian frame ($x$) [m]
    EndEffectorPitchAngle: NotationSymbol  # pitch angle of the end effector under a declared rotation convention ($\phi_{\text{pitch}}$) [rad]
    EndEffectorPose: NotationSymbol  # pose of the end-effector frame expressed in the world frame, a homogeneous transform ($\mathbf{T}_{WE}$)
    EndEffectorRollAngle: NotationSymbol  # roll angle of the end effector under a declared rotation convention ($\phi_{\text{roll}}$) [rad]
    EndEffectorSecondAxisPosition: NotationSymbol  # end-effector position along the second axis of the declared Cartesian frame ($y$) [m]
    EndEffectorTaskAcceleration: NotationSymbol  # task-space acceleration of the end effector that an impedance law regulates ($\ddot{\mathbf{x}}$) [varies]
    EndEffectorTaskPosition: NotationSymbol  # task-space position of the end effector that an impedance law regulates ($\mathbf{x}$) [varies]
    EndEffectorTaskVelocity: NotationSymbol  # task-space velocity of the end effector that an impedance law regulates ($\dot{\mathbf{x}}$) [varies]
    EndEffectorThirdAxisPosition: NotationSymbol  # end-effector position along the third axis of the declared Cartesian frame ($z$) [m]
    EndEffectorTwist: NotationSymbol  # six-vector of end-effector angular and linear velocity in a declared spatial or body twist convention ($\mathbf{V}_E$) [varies]
    EndEffectorVelocity: NotationSymbol  # velocity of the end effector in operational space, the image of the joint rates under the manipulator Jacobian ($\mathbf{v}_e$)
    EndEffectorYawAngle: NotationSymbol  # yaw angle of the end effector under a declared rotation convention ($\phi_{\text{yaw}}$) [rad]
    EndToEndRequestLatency: NotationSymbol  # time from a request leaving the client to its response returning, summed over every stage on the request path ($L_{\text{lat,total}}$) [s]
    EnergyCrossoverIntensity: NotationSymbol  # arithmetic intensity at which compute energy and data-movement energy are equal ($I_{\text{crossover}}$) [FLOP/byte]
    EnergyLimitedContactForce: NotationSymbol  # largest contact force the energy tank allowance permits over one control interval ($F_E$) [N]
    EnergyPerByteMoved: NotationSymbol  # energy spent moving one byte between memory and the compute units ($E_{\text{move}}$) [J/byte]
    EnergyPerOperation: NotationSymbol  # energy spent executing one arithmetic operation ($E_{\text{compute}}$) [J/operation]
    EnergyTankBarrierFunction: NotationSymbol  # barrier certificate on a passivity energy tank, its level minus the reserve floor ($h_{\text{energy}}$) [J]
    EnergyTankLevel: NotationSymbol  # energy held in a passivity filter energy tank, credits from realized damping plus the initial reserve less active work ($E_{\text{tank}}$) [J]
    EnergyTankPower: NotationSymbol  # rate of change of the energy tank, realized damping power credited minus active power debited ($\dot{E}_{\text{tank}}$) [W]
    EnergyTankReserve: NotationSymbol  # reserve floor of a passivity energy tank, below which no active work is admitted ($E_{\text{min}}$) [J]
    EnforcedCommandLabel: NotationSymbol  # admitted command vector at a control step, used as the imitation label ($\mathbf{a}_{\text{enf}, t}$)
    EnforcedCommandTap: NotationSymbol  # third stream of the four-stream action tap, the command the permission check admitted ($a_{\text{enf}}$)
    EnforcerComputeTime: NotationSymbol  # time the enforcer takes to evaluate its constraints within one control period ($T_{\text{comp}}$) [s]
    EnforcerDeadline: NotationSymbol  # deadline, measured from the start of the permission tick, by which the enforcer must finish its evaluation, nested inside the permission-loop period ($T_{\text{deadline,enf}}$) [s]
    EngineerHourlyCost: NotationSymbol  # fully loaded cost of one hour of engineering time ($C_{\text{engineer}}$) [USD/h]
    EngineeringTemperatureCeiling: NotationSymbol  # working winding-temperature limit chosen below the insulation rating to leave room for thermal-model error ($T_{\text{ceil}}$) [degC]
    EnsembleSpreadThreshold: NotationSymbol  # threshold on the spread of a policy ensemble's normalized actions, squared to bound the covariance trace ($\sigma_{\text{thresh}}$) [dimensionless]
    EnsemblingDecayRate: NotationSymbol  # decay factor per chunk index of exponential temporal-ensembling weights ($m$) [dimensionless]
    EnsemblingWindowLength: NotationSymbol  # number of overlapping past chunks blended together in temporal ensembling ($K$)
    EntityCount: NotationSymbol  # number of entities, such as users or items, a recommendation table or feature store holds rows for ($N_{\text{entities}}$)
    EntityEmbedding: NotationSymbol  # embedding row of one entity, such as a user or item, in a recommendation embedding table ($\mathbf{e}$)
    EntropyTemperature: NotationSymbol  # weight on the policy entropy in the maximum-entropy objective, trading reward maximization against exploration ($\alpha$) [dimensionless]
    EnvelopeCallCount: NotationSymbol  # number of sequential calls a trajectory makes that each need an execution envelope ($N$)
    EnvelopeSetupTime: NotationSymbol  # time to prepare a sandbox envelope until it is ready for useful work ($T_{\text{setup}}$) [s]
    EnvironmentCycleTime: NotationSymbol  # time one environment slot takes per episode when it runs trajectories serially: turns of model and tool time, then assessment, then reset ($T_{\text{cycle}}$) [s]
    EnvironmentFixture: NotationSymbol  # one environment fixture, the reproducible task setup a rollout runs against, as assigned to a task group and split ($f$)
    EnvironmentLatentDimension: NotationSymbol  # number of components of the environment latent vector ($d_z$)
    EnvironmentLatentVector: NotationSymbol  # low-dimensional latent vector an encoder computes from the privileged simulator state, summarizing the environment's hidden physical parameters for the actor ($\mathbf{z}$)
    EnvironmentParameters: NotationSymbol  # external physical variables of one trial or rollout, such as floor friction, illuminance, obstacle geometry and air temperature ($e$)
    EnvironmentResetTime: NotationSymbol  # time to restore an environment slot to its fixture state between episodes ($T_{r}$) [s]
    EnvironmentResponseFunction: NotationSymbol  # response of the environment to a deployed model, mapping the current data distribution and the model's decisions to the next data distribution ($g$)
    EnvironmentRolloutStepCount: NotationSymbol  # number of steps each parallel environment contributes to one on-policy rollout ($h$)
    EnvironmentTransitionDistribution: NotationSymbol  # distribution the environment induces over the next state given the current state and the action taken ($p$)
    EpipolarLine: NotationSymbol  # homogeneous coefficients of the epipolar line in one image, on which the correspondence of a point in the other image must lie ($\mathbf{l}$)
    EpochCount: NotationSymbol  # number of full passes a training run makes over its dataset ($N_{\text{epochs}}$)
    EpochIndex: NotationSymbol  # position of the current epoch in a training run, counted from one ($n_{\text{epoch}}$)
    EquilibriumWindingTemperature: NotationSymbol  # temperature a winding approaches asymptotically under a constant loss power ($T_\infty$) [degC]
    EquivalentSeriesStiffness: NotationSymbol  # equivalent stiffness of a compliant element and a strike plate acting in series, the reciprocal of the sum of their compliances ($k_{\text{eq}}$) [N/m]
    ErasureDataFragmentCount: NotationSymbol  # number of data fragments an erasure code splits an object into, any that many of the stored fragments recovering the original ($d$)
    ErasureParityFragmentCount: NotationSymbol  # number of parity fragments an erasure code adds to the data fragments of an object ($m$)
    ErroneousPredictionCost: NotationSymbol  # average business cost incurred by one erroneous prediction ($C_{\text{error}}$) [USD]
    ErrorBudgetConsumption: NotationSymbol  # share of one reliability target's error budget a service has used: the fraction of eligible observations that violate the target divided by the violating fraction the target allows ($B$) [dimensionless]
    ErrorCovarianceDerivative: NotationSymbol  # time derivative of the estimation error covariance, given by the Riccati or Lyapunov differential equation ($\dot{\mathbf{P}}$)
    ErrorFeedbackResidual: NotationSymbol  # per-worker accumulator of compression error not yet transmitted, added to the next gradient before it is compressed ($\mathbf{e}$)
    ErrorFeedbackResidualElement: NotationSymbol  # error-feedback residual of one gradient element, the untransmitted remainder carried to the next step ($e$) [varies]
    ErrorGrowthRate: NotationSymbol  # maximum rate at which the true state can diverge from a stored estimate, the slope of a linear error-growth law ($r$) [m/s]
    ErrorRecoveryCost: NotationSymbol  # cost of recovering from one detected error, independent of the check interval ($C_{\text{recovery}}$) [varies]
    EscalationDecision: NotationSymbol  # choice between automatic execution and human review for one proposed action ($d$)
    EscalationLossMatrix: NotationSymbol  # loss of each escalation decision under each damaging-outcome value, one row per decision and one column per outcome ($\mathbf{L}$)
    EscalationThresholdProbability: NotationSymbol  # estimated damage probability above which human review has lower modeled expected cost than automatic execution, the review-to-damage cost ratio ($p_{\text{threshold}}$) [dimensionless]
    EssentialMatrix: NotationSymbol  # three-by-three matrix relating corresponding normalized image points in two calibrated views ($\mathbf{E}$)
    EstimateEpoch: NotationSymbol  # instant a published state estimate describes, which prediction may advance past the evidence epoch ($t_{\text{est}}$) [s]
    EstimatedAcceptanceProbability: NotationSymbol  # sample estimate of a configuration's acceptance probability from an evaluation, such as accepted trials over assigned trials or the task-averaged share of accepted runs ($\hat{p}$) [dimensionless]
    EstimatedAccuracy: NotationSymbol  # accuracy of a model estimated from a finite evaluation set, the fraction of evaluated examples it gets right ($\hat{p}$) [dimensionless]
    EstimatedCandidateUtility: NotationSymbol  # utility a learned scorer reports for one candidate, its true utility plus an independent scoring error ($\hat{U}$) [dimensionless]
    EstimatedCleanActionChunk: NotationSymbol  # estimate of the clean action chunk that the reverse process recovers from a noisy chunk and the predicted noise ($\hat{\mathbf{A}}$)
    EstimatedCoriolisMatrix: NotationSymbol  # model estimate of the Coriolis and centrifugal matrix that computed torque control uses ($\hat{\mathbf{C}}$)
    EstimatedDamageProbability: NotationSymbol  # runtime estimate of the probability that a proposed action produces the modeled damaging outcome, given observable features ($\hat{p}$) [dimensionless]
    EstimatedDivergenceInstant: NotationSymbol  # estimate of the divergence instant: the first threshold crossing found by searching back from the takeover ($\hat{t}_{\text{div}}$) [s]
    EstimatedForceSensorBias: NotationSymbol  # estimator's estimate of the force-sensor bias ($\hat{b}$) [N]
    EstimatedGravitationalTorque: NotationSymbol  # model estimate of the gravitational joint torques that computed torque control uses ($\hat{\mathbf{g}}$)
    EstimatedGripContactForce: NotationSymbol  # estimator's estimate of the true normal contact force ($\hat{x}$) [N]
    EstimatedHistoryAdvantage: NotationSymbol  # estimate of how much an action taken after a given interaction history beats the policy value at that history ($\hat{A}$) [dimensionless]
    EstimatedInertiaMatrix: NotationSymbol  # model estimate of the joint-space inertia matrix that computed torque control uses ($\hat{\mathbf{M}}$) [kg*m^2]
    EstimatedPairedAcceptanceDifference: NotationSymbol  # observed difference between the candidate's and the baseline's acceptance shares on the same sampled tasks, the discordant count difference divided by the task count ($\hat{\Delta}$) [dimensionless]
    EstimatedPlantState: NotationSymbol  # estimate of the mechanical plant state reconstructed from sensor observations ($\hat{\mathbf{x}}$)
    EstimatedPosition: NotationSymbol  # estimate of a point's position that a filter propagates under its nominal model ($\hat{\mathbf{p}}$) [m]
    EstimatedRemainingReturn: NotationSymbol  # estimated expected remaining return under a continuation policy, given the interaction history at a turn ($\hat{V}$) [dimensionless]
    EstimatedUtilityStandardDeviation: NotationSymbol  # standard deviation of the utility a scorer reports, combining the spread of true utility and of scoring error ($s$) [dimensionless]
    EstimatedVelocity: NotationSymbol  # estimate of a body's velocity that a filter propagates under its nominal model ($\hat{\mathbf{v}}$) [m/s]
    EstimatedWindingTemperature: NotationSymbol  # estimate of the motor winding temperature from the thermal model ($\hat{T}_{\text{wind}}$) [degC]
    EstimationErrorCovariance: NotationSymbol  # covariance of the error between the true state and its estimate ($\mathbf{P}$)
    EstimatorDrawCount: NotationSymbol  # number of independent draws or simulation runs a Monte Carlo or importance-sampling estimator averages ($N$)
    EstimatorDrawStandardDeviation: NotationSymbol  # standard deviation of one draw of a Monte Carlo or importance-sampling estimator ($\sigma$) [dimensionless]
    EstimatorState: NotationSymbol  # true value of the state vector a recursive estimator tracks, such as a track's position and velocity ($\mathbf{x}$)
    EstimatorStateVector: NotationSymbol  # state vector an estimator reconstructs from measurements, such as a tracked object's position and velocity ($\mathbf{x}$)
    EstimatorStepIndex: NotationSymbol  # index of one discrete step of a recursive estimator ($k$)
    EulerIntegralFirstShape: NotationSymbol  # first shape argument of the regularized incomplete beta function ($a$) [dimensionless]
    EulerIntegralSecondShape: NotationSymbol  # second shape argument of the regularized incomplete beta function ($b$) [dimensionless]
    EulerIntegralVariable: NotationSymbol  # integration variable of the incomplete beta integral, a probability level between zero and the upper limit ($t$) [dimensionless]
    EvaluatedExceedance: NotationSymbol  # exceedance level at which a generalized Pareto distribution function is evaluated ($y$) [varies]
    EvaluationPredicate: NotationSymbol  # binary predicate that maps a realized trial trajectory to its task outcome, zero or one ($\phi$)
    EvaluationSampleCount: NotationSymbol  # number of examples in the evaluation set over which an accuracy or calibration metric is computed ($N$)
    EvaluationTrialCount: NotationSymbol  # number of trials assigned to an evaluation cohort, including incomplete, timed-out, crashed and budget-exhausted runs ($N$)
    EvaluationTrialIndex: NotationSymbol  # position of one trial within an evaluation campaign, counted from one, the index of per-trial outcomes and of sums over the campaign's trials ($i$)
    EventContrastThreshold: NotationSymbol  # threshold on the change in logarithmic intensity at which an event-camera pixel emits an event ($C_{\text{th}}$) [dimensionless]
    EventDetectionPower: NotationSymbol  # constant power drawn by the circuitry that watches for input events in an event-driven design ($P_{\text{detect}}$) [W]
    EventIndicator: NotationSymbol  # indicator of an event, one when the braced condition holds and zero otherwise ($\mathbf{1}$)
    EventPacket: NotationSymbol  # packet an event-camera pixel emits when its log-intensity change crosses the contrast threshold: address, timestamp and polarity ($e$)
    EventPolarity: NotationSymbol  # sign of an event camera event, plus one when radiance increased and minus one when it decreased ($p$) [dimensionless]
    EventProcessingEnergy: NotationSymbol  # energy an event-driven design spends processing one event ($E_{\text{event}}$) [J]
    EvidenceCaptureInstant: NotationSymbol  # instant at which the sensor evidence behind an observation or proposal was captured, on a monotonic clock ($t_{\text{capture}}$) [s]
    EvidenceEpoch: NotationSymbol  # capture instant of the most recent sensor measurement that constrained a stored belief ($t_0$) [s]
    EvidenceHorizon: NotationSymbol  # time for which a target estimate stays within its task tolerance without fresh evidence ($\tau_{\text{ev}}$) [s]
    EvidenceLowerBound: NotationSymbol  # evidence lower bound, the expected reconstruction log-likelihood minus the KL divergence of the latent posterior from the prior, which lower-bounds the log-marginal likelihood ($\mathcal{L}_{\text{ELBO}}$)
    EvidenceRecordEvent: NotationSymbol  # logged event that durably records evidence received about a dispatched operation, whose status may still be pending, partial or unknown ($e_{\text{obs}}$)
    ExactFailureLowerLimit: NotationSymbol  # Clopper-Pearson exact lower confidence limit on the per-trial failure probability, from inverting the binomial distribution of the observed failure count ($p_{L}$) [dimensionless]
    ExactFailureUpperLimit: NotationSymbol  # Clopper-Pearson exact upper confidence limit on the per-trial failure probability, from inverting the binomial distribution of the observed failure count ($p_{U}$) [dimensionless]
    ExampleAveragedBatchLoss: NotationSymbol  # batch loss that gives every example equal weight, the unweighted mean of per-example masked token losses ($\mathcal{L}_{\text{example}}$) [nats]
    ExampleIndex: NotationSymbol  # index of an example within a batch ($i$)
    ExampleLabel: NotationSymbol  # the label paired with one input, treated as a random variable ($y$)
    ExceedanceReturnLevel: NotationSymbol  # level of the monitored quantity exceeded with the target probability per sampled cycle ($z_\epsilon$) [varies]
    ExceedanceSampleSet: NotationSymbol  # set of observed exceedances over a threshold, the sample a generalized Pareto fit uses ($\mathcal{Y}_u$)
    ExceedanceThreshold: NotationSymbol  # high threshold on a performance metric above which peaks-over-threshold modeling fits exceedances ($u$) [varies]
    ExcessDistributionFunction: NotationSymbol  # cumulative distribution function of the excess over a threshold, conditional on exceeding it ($F_{u}$)
    ExcessForcePenaltyWeight: NotationSymbol  # weight that converts contact force above the tripwire into a reward penalty, in reward per newton ($\lambda_f$) [1/N]
    ExcessSupportUpperEndpoint: NotationSymbol  # upper endpoint of the support of the threshold excess, infinite for a nonnegative shape ($y_{\max}$) [varies]
    ExecutedActionStepCount: NotationSymbol  # number of steps of a predicted action chunk executed before the policy replans, the executed prefix ($T_a$)
    ExecutedToolCall: NotationSymbol  # one tool call the harness executed on a turn, indexed by turn and by call within the turn ($a$)
    ExecutedTrialCount: NotationSymbol  # number of verification trials actually run on one candidate when verification stops at the first rejection ($N_{\text{trial}}$)
    ExecutionHorizon: NotationSymbol  # number of leading steps of each predicted action chunk the controller executes before the next chunk replaces it, the execution prefix ($H_{\text{exec}}$)
    ExecutionNondeterminism: NotationSymbol  # run-to-run execution nondeterminism of a training job, such as random seeds, data ordering and the order of floating-point accumulation ($\xi$)
    ExecutionSwitchTime: NotationSymbol  # time one system call spends transferring execution to another thread or process ($T_{\text{switch}}$) [s]
    ExogenousDataDistribution: NotationSymbol  # exogenous stationary distribution from which a passive learner's samples are drawn independently ($\mathcal{D}$)
    ExpectedCategoryFrequency: NotationSymbol  # number of observations one category of a categorical feature is expected to hold under the baseline distribution, in a chi-square goodness-of-fit test ($n_i^{\text{exp}}$) [dimensionless]
    ExpectedCheckingTime: NotationSymbol  # expected verifier slot time spent per attempted record under a short-circuit check cascade ($T_{\text{check}}$) [s]
    ExpectedCostLimit: NotationSymbol  # upper bound a constrained reinforcement learning objective places on expected constraint cost ($\delta$) [varies]
    ExpectedDecisionRisk: NotationSymbol  # expected loss of an escalation decision under the estimated damage probability ($\mathcal{R}$) [USD]
    ExpectedHistoryActionReturn: NotationSymbol  # expected remaining return under a policy after taking a given action at a given interaction history and then following the policy ($Q$) [dimensionless]
    ExpectedIrreversibleLoss: NotationSymbol  # expected consequence of the bad actions in an execution plan that cannot be recovered ($L_A$) [varies]
    ExpectedNormalMaximum: NotationSymbol  # expected maximum of as many independent standard normal draws as there are candidates in the pool ($m_{N}$) [dimensionless]
    ExpectedQueryDistribution: NotationSymbol  # reference distribution of queries expected from legitimate use of an API, against which an account's query distribution is compared ($p_{\text{expected}}$)
    ExpectedRemainingReturn: NotationSymbol  # expected remaining return under a policy from a given interaction history at a decision step ($V$) [dimensionless]
    ExpectedReturnObjective: NotationSymbol  # expected return of a policy as a function of its parameters, the objective reinforcement learning maximizes ($J$) [dimensionless]
    ExpectedRunCompletionTime: NotationSymbol  # expected elapsed time to complete all useful steps when every failed block is regenerated from its start, as a function of block size ($T$) [s]
    ExpectedTaskCostSum: NotationSymbol  # expected sum over the horizon of the normalized immediate task costs a policy incurs ($J$) [dimensionless]
    ExperienceReplayBuffer: NotationSymbol  # buffer of past transitions, each a state, action, reward and next state, from which an off-policy learner samples minibatches ($\mathcal{D}$)
    ExperimentCount: NotationSymbol  # number of experiments a team conducts over an accounting period ($N_{\text{exp}}$)
    ExperimentUserIndex: NotationSymbol  # index of one user, the experimental unit of an online experiment ($i$)
    ExpertCapacityFactor: NotationSymbol  # multiplier on an expert's uniform share of token assignments that sets how many assignments it may accept per step ($C_{\text{cap}}$) [dimensionless]
    ExpertCount: NotationSymbol  # number of expert subnetworks in one mixture-of-experts layer ($N_{\text{experts}}$)
    ExpertDiscountedStateVisitationDistribution: NotationSymbol  # normalized discounted distribution of states the expert demonstrator policy visits ($d^{\pi^*}$)
    ExpertIndex: NotationSymbol  # index of one expert in a mixture-of-experts layer ($e$)
    ExpertSelectionProbability: NotationSymbol  # probability that a given token is routed to one particular expert ($p_e$) [dimensionless]
    ExpertStateVisitationDistribution: NotationSymbol  # distribution of states the expert demonstrator visits, the expected fraction of time steps its demonstrations spend in each ($p^{\pi^*}$)
    ExpertTokenCount: NotationSymbol  # number of token assignments a router sends to one expert in one layer ($B_e$)
    ExploitCandidateFraction: NotationSymbol  # fraction of generated candidates that contain an unmodeled edge case or exploit through which an invalid candidate passes the verifier ($p_{\text{exp}}$) [dimensionless]
    ExploitTrajectory: NotationSymbol  # unintended trajectory that earns at least the reward of the intended solution while serving the intended objective far worse ($\tau_{\text{hack}}$)
    ExploratoryStateVisitationDistribution: NotationSymbol  # distribution of states visited by an exploratory collection policy that deliberately seeks out uncovered regions ($p^{\pi_{\text{expl}}}$)
    ExponentBitIndex: NotationSymbol  # position of one bit in the exponent field of a floating-point number, counted from zero so that the bit has place value two to this power ($k$)
    ExponentialSeriesArgument: NotationSymbol  # argument of a power-series expansion of the exponential function ($u$) [dimensionless]
    ExponentialStabilityConstant: NotationSymbol  # positive constant in the quadratic bounds that certify exponential stability, written with an index 1 to 3 ($k$) [varies]
    ExposedCommunicationTime: NotationSymbol  # communication and coordination time per step left on the critical path after overlap with computation ($T_{\text{comm,exposed}}$) [s]
    ExposedUnitCount: NotationSymbol  # number of exposure units, such as requests or users, a rollout exposes to a change during its period ($E_{\text{exposure}}$)
    ExposureDuration: NotationSymbol  # integration time of one camera frame ($t_{\text{exp}}$) [s]
    ExposureMidpointTime: NotationSymbol  # time at the midpoint of a camera frame's optical exposure, the instant its image represents ($t_{\text{mid}}$) [s]
    ExposureMotionDisplacement: NotationSymbol  # bound on how far a camera's optical center moves during one exposure, from linear and rotational motion ($\Delta x_{\text{motion}}$) [m]
    ExposureStartTime: NotationSymbol  # time at which a camera latches its trigger and begins exposing a frame ($t_{\text{start}}$) [s]
    ExposureTravelDistance: NotationSymbol  # distance the machine travels during one camera exposure, its speed times the exposure duration ($\Delta x$) [m]
    ExtendedClassKInfinityFamily: NotationSymbol  # class of extended class-K functions that are unbounded, from which the barrier decay function is chosen ($\mathcal{K}_\infty$)
    ExtendedDepthLength: NotationSymbol  # number of additional output tokens a call generates when test-time compute extends one candidate along the depth axis ($K_{\text{ext}}$)
    ExternalContactTorque: NotationSymbol  # joint torques produced by external contact forces acting on the manipulator ($\boldsymbol{\tau}_{\text{ext}}$) [N*m]
    ExternalDispatchAction: NotationSymbol  # dispatch of an admitted operation to an external endpoint, which is not its completion ($a_{\text{ext}}$)
    ExternalDisturbanceForce: NotationSymbol  # external disturbance force and unmodeled friction acting on the plant, as a function of time ($d$) [N]
    ExternalDisturbanceLoad: NotationSymbol  # constant external load acting on the controlled joints, such as gravity sag, cable friction or a payload offset ($\mathbf{d}_{\text{ext}}$)
    ExternalJointTorque: NotationSymbol  # external torque applied at a joint's output by the environment ($\tau_{\text{ext}}$) [N*m]
    ExternalObjectWrench: NotationSymbol  # external wrench acting on a grasped object, such as its weight and inertial loads, that the contact forces must balance ($\mathbf{w}_{\text{ext}}$) [varies]
    ExternalWait: NotationSymbol  # time spent waiting on a remote service or on a person's response ($T_{\text{wait}}$) [s]
    ExternalWrench: NotationSymbol  # external force and moment (wrench) the environment applies at the end effector ($\mathbf{F}_{\text{ext}}$) [varies]
    ExtractionAnomalyClassifier: NotationSymbol  # trained classifier that maps an account's query features (volume, distribution, timing regularity) to an extraction anomaly score ($f_{\theta}$)
    ExtractionQueryCount: NotationSymbol  # number of API queries an attacker needs to extract a substitute model ($n_{\text{queries}}$)
    ExtrinsicCalibrationCovariance: NotationSymbol  # covariance of a sensor's calibrated mounting extrinsic ($\boldsymbol{\Sigma}_{\text{ext}}$) [m^2]
    ExtrinsicCovarianceJacobian: NotationSymbol  # Jacobian that projects the extrinsic covariance into world coordinates ($\mathbf{J}_{\text{ext}}$)
    ExtrinsicLateralError: NotationSymbol  # transverse position error an extrinsic orientation error induces at a given range ($\Delta y$) [m]
    ExtrinsicMisalignmentAngle: NotationSymbol  # angular misalignment of a sensor's extrinsic calibration relative to its true mounting ($\theta_{\text{ext}}$) [rad]
    EyeInHandCameraMotion: NotationSymbol  # relative camera motion between two eye-in-hand calibration poses, from target unprojection ($\mathbf{B}$)
    EyeInHandFlangeMotion: NotationSymbol  # relative flange motion between two eye-in-hand calibration poses, from forward kinematics ($\mathbf{A}$)
    EyeInHandUnknownTransform: NotationSymbol  # unknown camera-to-flange transform that eye-in-hand calibration solves for in the hand-eye equation ($\mathbf{X}$)
    EyeToHandCameraPose: NotationSymbol  # absolute transform mapping camera coordinates into the target frame at one eye-to-hand calibration pose ($\mathbf{B}'$)
    EyeToHandFlangePose: NotationSymbol  # absolute transform mapping base coordinates into the flange frame at one eye-to-hand calibration pose ($\mathbf{A}'$)
    EyeToHandTargetTransform: NotationSymbol  # unknown transform of the calibration target in the flange frame that eye-to-hand calibration solves for alongside the camera ($\mathbf{Z}$)
    EyeToHandUnknownTransform: NotationSymbol  # unknown camera-to-base transform that eye-to-hand calibration solves for ($\mathbf{X}'$)
    FacilityEnergy: NotationSymbol  # energy a data center facility draws over a stated period, its IT energy scaled by power usage effectiveness ($E_{\text{facility}}$) [J]
    FacilityHeatRemovalCapacity: NotationSymbol  # largest rate at which an installed facility and cooling configuration can remove heat ($\dot{Q}_{\text{remove,max}}$) [W]
    FacilityPower: NotationSymbol  # total electrical power a data center facility draws from the grid, its IT load plus cooling, power conversion, lighting and other overhead ($P_{\text{facility}}$) [W]
    FacilityWaterConsumption: NotationSymbol  # volume of water a facility consumes over a period, mostly through evaporative cooling ($W_{\text{water}}$) [m^3]
    FactoredIntermediateActivation: NotationSymbol  # intermediate activation vector produced by the first factor of a low-rank layer and consumed by the second ($\mathbf{z}$)
    FactorizationRank: NotationSymbol  # number of rank-one components a low-rank matrix or tensor factorization keeps ($k$)
    FailureDetectionTime: NotationSymbol  # time from a hardware or process fault until the system classifies it as a failure ($T_{\text{detect}}$) [s]
    FailureEventCost: NotationSymbol  # cost of one failure during training: the compute it wastes plus recovery overhead ($C_{\text{per-failure}}$) [USD]
    FailureFlagLatency: NotationSymbol  # time from a physical failure to a fault flag the enforcer can read, the hardware detection latency ($\delta_{\text{flag}}$) [s]
    FailureMarginFunction: NotationSymbol  # Lipschitz function of the state that is negative inside the failure or collision region and nonnegative in the safe set ($l$)
    FailureModeOccurrenceCount: NotationSymbol  # number of executions in which one failure mode occurs over the period a trace-sampling policy covers ($n_{f}$)
    FailureObservationInterval: NotationSymbol  # length of the time interval over which failures are counted under the Poisson failure model ($T$) [s]
    FailureOverheadFraction: NotationSymbol  # additional training time lost to failures and recovery, as a fraction of the baseline duration ($F_{\text{fail}}$) [dimensionless]
    FailureRecoveryTime: NotationSymbol  # time from a failure until useful training work resumes: detection, restart, checkpoint load and warmup ($T_{\text{recovery}}$) [s]
    FailureRegion: NotationSymbol  # subset of scenario parameters whose outcome violates the safety requirement ($F$)
    FailureSet: NotationSymbol  # set of states in failure or collision, the zero-sublevel set of the failure margin function ($\mathcal{F}_{\text{fail}}$)
    FailureUpperConfidenceLimit: NotationSymbol  # one-sided upper confidence limit on the per-trial failure probability from a finite campaign ($p_{\text{fail, upper}}$) [dimensionless]
    FairShareStabilizingConstant: NotationSymbol  # small positive constant added to recent usage in the fair-share priority so a user with no recent usage gets finite priority ($\epsilon_{\text{stab}}$) [dimensionless]
    FairnessGapTolerance: NotationSymbol  # largest allowed gap between protected groups in the fairness metric a release gate or monitor enforces ($\epsilon_{\text{fair}}$) [dimensionless]
    FakeQuantizedOutput: NotationSymbol  # value after quantizing and immediately dequantizing, which stays in floating point but carries the rounding error ($x_{\text{fake}}$) [varies]
    FallbackCost: NotationSymbol  # forward-looking cost of compensating existing side effects and rebuilding to a declared readiness endpoint ($C_{\text{fallback}}$) [varies]
    FallbackDeadline: NotationSymbol  # latest permissible wait after a handover request before the fallback must begin, from the stopping inequality at the current speed and clearance ($T_{\text{deadline,fallback}}$) [s]
    FallbackHold: NotationSymbol  # optional hold or fault-confirmation interval before the stop command reaches the drive ($T_{\text{hold}}$) [s]
    FalseAcceptHarmCost: NotationSymbol  # additional harm or remediation cost of a gate false accept not already captured in the execution cost record ($C_{\text{harm}}$) [USD]
    FalseApprovalCount: NotationSymbol  # number of reviewers in an ensemble that approve a candidate known to be invalid ($N_{\text{fa}}$)
    FalseApprovalOutcomeCount: NotationSymbol  # a particular number of falsely approving reviewers, the value at which the false-approval count's distribution is evaluated or summed ($k$)
    FalseApprovalPropensity: NotationSymbol  # probability, intrinsic to one candidate, that a randomly chosen reviewer falsely approves it, treated as a random variable over the candidate population ($Q$) [dimensionless]
    FalseApprovalPropensityLevel: NotationSymbol  # a particular level of a candidate's false-approval propensity, the variable integrated over in its beta density ($q$) [dimensionless]
    FalseDiscoveryRateTarget: NotationSymbol  # level at which the Benjamini-Hochberg procedure bounds the expected false-discovery proportion ($q$) [dimensionless]
    FalsificationInitialStateSet: NotationSymbol  # set of initial plant states a falsification search may choose from ($\mathcal{X}_0$)
    FanoutBackendCount: NotationSymbol  # number of parallel backend sub-requests one user request fans out to and must wait for ($m$)
    FarthestCandidateDistance: NotationSymbol  # distance from a query vector to the farthest stored candidate vector, as a function of collection size and dimension ($D_{\max}$)
    FatTreeHostCount: NotationSymbol  # number of hosts a non-blocking fat-tree built from switches of a given radix supports ($N_{\text{hosts}}$)
    FatigueLifeCycleCount: NotationSymbol  # number of load cycles a component survives at one load level before fatigue failure, from its fatigue data ($N$)
    FaultContainmentLatency: NotationSymbol  # measured time from fault injection to the plant's entry into its safe bounded regime ($\Delta t_{\text{bounded}}$) [s]
    FaultDetectionLatency: NotationSymbol  # measured time from fault injection to the detection flag in a fault-injection trial ($\Delta t_{\text{det}}$) [s]
    FaultDomainIsolation: NotationSymbol  # fault-domain isolation of a signal handoff: whether a fault on one side can propagate across it, one of the five handoff limits ($\mathcal{F}$)
    FaultInjectionDuration: NotationSymbol  # length of time an injected fault is held, such as a drive-bus droop ($\Delta t_{\text{duration}}$) [s]
    FaultStartTimestamp: NotationSymbol  # reference-clock timestamp recorded when a fault is injected, the origin of a trial's detection and takeover times ($t_{\text{fault\_start}}$) [s]
    FaultTakeoverLatency: NotationSymbol  # measured time from fault injection to hardware preemption of the motor driver in a fault-injection trial ($\Delta t_{\text{takeover}}$) [s]
    FaultTakeoverTimestamp: NotationSymbol  # reference-clock timestamp at which the hardware multiplexer preempts the motor driver in a fault trial ($t_{\text{takeover}}$) [s]
    FaultToleranceCost: NotationSymbol  # cost of the fault-tolerance mechanisms deployed for a training run ($C_{\text{ft}}$) [USD]
    FaultToleranceInvestment: NotationSymbol  # level of investment in a fault-tolerance mechanism, the decision variable against which marginal costs are compared ($x$) [varies]
    FeasibleOutputTokenRate: NotationSymbol  # output tokens per second a dedicated deployment can generate while meeting its latency objective ($\mu_{\text{out}}$) [tokens/s]
    FeasibleTrajectoryProjection: NotationSymbol  # operator that maps a learned trajectory proposal to the nearest trajectory in the feasible set, under a weighted tracking and jerk cost ($\Pi_{\mathcal{K}}$)
    FeasibleTrajectorySet: NotationSymbol  # set of joint trajectories that satisfy the boundary conditions, collision, speed, torque and stopping constraints of a projection ($\mathcal{K}$)
    FeatureCategoryCount: NotationSymbol  # number of categories a categorical feature takes, the number of terms of a chi-square goodness-of-fit statistic ($k$)
    FeatureCategoryIndex: NotationSymbol  # position of one category among the categories of a categorical feature ($i$)
    FeatureCellVector: NotationSymbol  # feature vector a vision backbone computes at one cell of its feature map, indexed by the cell's pixel coordinates ($\mathbf{f}_{u, v}$)
    FeatureChannelCount: NotationSymbol  # number of channels of each feature vector in a vision backbone's output feature map ($C$)
    FeatureCount: NotationSymbol  # number of features a feature store maintains for each entity ($N_{\text{features}}$)
    FeatureFreshnessLatency: NotationSymbol  # delay from a source event to the moment a feature value reflecting it is available for serving ($T_{\text{freshness}}$) [s]
    FeatureMapColumnCoordinate: NotationSymbol  # column coordinate of a position in a feature map, as a loop nest or parallel schedule traverses it ($x$)
    FeatureMapGrowthRate: NotationSymbol  # number of new feature maps each DenseNet layer adds to the concatenated input of later layers ($g$)
    FeatureMapHeight: NotationSymbol  # height, in pixels, of the image or feature map a convolution layer reads ($H_{\text{img}}$)
    FeatureMapRowCoordinate: NotationSymbol  # row coordinate of a position in a feature map, as a loop nest or parallel schedule traverses it ($y$)
    FeatureMapWidth: NotationSymbol  # width, in pixels, of the image or feature map a convolution layer reads ($W_{\text{img}}$)
    FeatureRecordTime: NotationSymbol  # time at which a feature record became available, which a point-in-time join requires to be no later than the observation timestamp ($t$) [s]
    FeatureRetentionWindow: NotationSymbol  # length of feature history a feature store keeps so it can answer point-in-time queries ($T_{\text{retention}}$) [s]
    FeatureUpdateInterval: NotationSymbol  # interval between successive updates of one feature value in a feature store ($T_{\text{update}}$) [s]
    FeatureUpdateLag: NotationSymbol  # time since a served feature value was last materialized, the behavior it omits ($T_{\text{staleness}}$) [s]
    FeatureUpdateTimestamp: NotationSymbol  # time at which a feature record was last updated ($t_{\text{feat}}$) [s]
    FeatureViewingDistance: NotationSymbol  # distance from a rotating sensor head's axis to an observed feature ($r$) [m]
    FederatedRoundCount: NotationSymbol  # number of communication rounds a federated training run performs, or needs to reach a quality target ($R$)
    FederatedRoundIndex: NotationSymbol  # position of one communication round in a federated training run ($t$)
    FederatedTrainingEnergy: NotationSymbol  # total energy of training a model by federated learning, summed over all rounds, clients and server aggregation ($E_{\text{federated}}$) [J]
    FeedForwardOperations: NotationSymbol  # floating-point operations of the two linear transformations of one transformer feed-forward block ($O_{\text{FFN}}$) [FLOP]
    FeedForwardWidth: NotationSymbol  # width of the hidden layer inside a transformer block's feed-forward network ($d_{\text{ff}}$)
    FeedbackCheckDuration: NotationSymbol  # time the external check of one feedback round takes to run and return its observation ($t_{\text{check}}$) [s]
    FeedbackCriticalPathLatency: NotationSymbol  # critical-path latency of one decision when test-time compute revises a candidate over sequential feedback turns, each ending in an external check ($T^{\text{feedback}}$) [s]
    FeedbackIntervalCount: NotationSymbol  # number of fixed feedback intervals, such as retraining cycles, elapsed since the disparity was first measured ($k$)
    FeedbackStepIndex: NotationSymbol  # position of one deployment period in the feedback loop between a model's decisions and the data it later sees ($t$)
    FeedforwardJointTorque: NotationSymbol  # vector of joint torques the dynamics model predicts a reference trajectory needs, sent ahead of feedback ($\boldsymbol{\tau}_{\text{ff}}$) [N*m]
    FeedthroughMatrix: NotationSymbol  # feedthrough matrix of a linear state-space model, mapping the control input directly to the output ($\mathbf{D}$)
    FewShotStepCount: NotationSymbol  # number of gradient steps a few-shot adaptation policy permits ($n_{\text{steps}}$)
    FewShotSupportSet: NotationSymbol  # local support set of labeled examples a few-shot adaptation trains on ($\mathcal{D}_{\text{shot}}$)
    FiberPropagationSpeed: NotationSymbol  # speed at which light propagates through optical fiber, roughly two-thirds of its vacuum value ($c_{\text{fiber}}$) [m/s]
    FieldbusCycleTime: NotationSymbol  # period of one fieldbus communication cycle ($T_{\text{cycle}}$) [s]
    FilmEncounterRate: NotationSymbol  # rate at which a machine encounters fluid films along its transit routes ($\lambda_{\text{film}}$) [1/h]
    FilterSolveBudget: NotationSymbol  # execution time allocated to the safety filter solver within one loop period ($T_{\text{budget}}$) [s]
    FilterSolveTime: NotationSymbol  # execution time of the safety filter solver or projection in one control tick ($T_{\text{solve}}$) [s]
    FilterTapCount: NotationSymbol  # number of taps (coefficients) of a finite impulse response filter ($N$)
    FilterWeight: NotationSymbol  # one weight of a convolution filter, as listed in a small worked example ($w$) [dimensionless]
    FilteredScreenSpaceCovariance: NotationSymbol  # screen-space covariance of a projected Gaussian after an isotropic low-pass variance is added to prevent aliasing ($\boldsymbol{\Sigma}_{2D}'$) [pixels^2]
    FinalBusVoltage: NotationSymbol  # DC bus voltage at the end of a regenerative stop when bus capacitance alone absorbs the returned energy ($V_{\text{final}}$) [V]
    FinalGeneratorLatentVector: NotationSymbol  # latent vector of the last of the m synthetic records a generator produces ($\mathbf{z}_{m}$)
    FinalLayerFunction: NotationSymbol  # function the last layer of a deep network computes ($f_{N_L}$)
    FinalLayerWidth: NotationSymbol  # number of units in a network's last layer ($n_{N_L}$)
    FinalMiniBatchActivation: NotationSymbol  # activation of the last example in a mini-batch of batch size B ($a_{B}$) [dimensionless]
    FinalRecurrentHiddenState: NotationSymbol  # hidden state a recurrent network holds after the last step of a sequence ($\mathbf{h}_{S}$)
    FinalWatchdogToken: NotationSymbol  # watchdog token after the last checkpoint stage of a tick, which must match its expected value before the watchdog is serviced ($T_{\text{final}}$)
    FingerNormalForce: NotationSymbol  # normal force one finger of a grasp presses against the object at its contact ($f_n$) [N]
    FingerTangentialForce: NotationSymbol  # tangential friction force one finger of a grasp carries at its contact, bounded by the friction coefficient times the normal force ($\mathbf{f}_t$) [N]
    FinishedWorkerCount: NotationSymbol  # number of parallel workers that have returned an admissible result by a given time ($j$)
    FiniteChange: NotationSymbol  # finite change in the quantity it prefixes, between two states, configurations or points in time ($\Delta$)
    FiniteSampleGeneralizationBound: NotationSymbol  # error term of the dataset-aggregation guarantee that covers finite-sample generalization and the selection of one policy from the sequence ($\delta_N$) [dimensionless]
    FiniteTimeHorizon: NotationSymbol  # length of the finite time window over which an optimization, rollout, test or temporal-logic operator is evaluated ($T$) [s]
    FirstAdjacentCheckIndex: NotationSymbol  # label of the first of two adjacent independent admission predicates whose order is compared ($a$)
    FirstAdjacentCheckPassProbability: NotationSymbol  # pass probability of the first of two adjacent independent admission predicates ($p_{a}$) [dimensionless]
    FirstAdjacentCheckSlotTime: NotationSymbol  # expected verifier slot time of the first of two adjacent independent admission predicates ($T_{a}$) [s]
    FirstCatalogDefinitionCount: NotationSymbol  # number of tool definitions each call stages under the first of two compared fixed catalogs ($n_{\text{def},A}$)
    FirstCatalogDefinitionSize: NotationSymbol  # mean number of tokens one tool definition of the first of two compared catalogs contributes to the serialized request ($S_{A}$) [tokens]
    FirstClassCoverageFloor: NotationSymbol  # minimum number of examples of the first class, A, that a coreset must retain ($a$)
    FirstClientMaskedUpdate: NotationSymbol  # masked update the first client, labeled A, uploads in a two-client secure-aggregation illustration ($\mathbf{y}_A$)
    FirstClientModelParameters: NotationSymbol  # model parameters of the first client, labeled A, in a two-client secure-aggregation illustration ($\theta_A$)
    FirstComparedGroupLabel: NotationSymbol  # first of two group labels compared in a group-fairness condition, one value of the protected group attribute ($a$)
    FirstComparedItem: NotationSymbol  # first of two items in a paired comparison, the one whose preference probability is stated ($i$)
    FirstDiscardedSingularNumber: NotationSymbol  # largest singular value a rank-r truncation discards, which equals the spectral-norm reconstruction error ($\sigma_{r+1}$) [dimensionless]
    FirstErrorStepIndex: NotationSymbol  # control step at which an imitating policy first takes an action different from the expert's, the variable summed over in the behavioral-cloning compounding-error bound ($k$)
    FirstFloatingPointAddend: NotationSymbol  # first of three floating-point numbers added in either grouping to show that floating-point addition is not associative ($a$) [dimensionless]
    FirstGroupClassificationThreshold: NotationSymbol  # score threshold applied to the first compared group in a group-specific thresholding policy ($\tau_{\text{thr},a}$) [dimensionless]
    FirstGroupQualifiedFraction: NotationSymbol  # fraction of the first compared group's applicants who are qualified, that group's base rate of the positive outcome ($q_a$) [dimensionless]
    FirstGroupSampleApprovalRate: NotationSymbol  # observed share of the first compared group's sampled applicants who are approved ($\hat{p}_a$) [dimensionless]
    FirstGroupSampleSize: NotationSymbol  # number of sampled decisions from the first compared group in a two-proportion test ($n_a$)
    FirstGroupThresholdClassifier: NotationSymbol  # decision rule applied to the first compared group, one when the score reaches that group's classification threshold ($h_a$)
    FirstIncidentTime: NotationSymbol  # operating time from the start of observation until the first catastrophic incident, exponentially distributed under a constant rate ($t_1$) [s]
    FirstInputNodePrimal: NotationSymbol  # value of the first input node of a computational graph, holding the first input ($v_{1-N}$) [dimensionless]
    FirstJaccardSet: NotationSymbol  # first of two sets compared by Jaccard similarity, such as one document's shingles ($\mathcal{A}$)
    FirstMistakeEvent: NotationSymbol  # event that a learned policy first disagrees with the expert at a given step, having agreed at every earlier step ($\mathcal{E}$)
    FirstRankComputeTime: NotationSymbol  # local compute time of the first rank (rank 0) in one synchronous training step ($T_{\text{rank}_0}$) [s]
    FirstRaterCategoryFraction: NotationSymbol  # fraction of the cases the first of two raters assigns to one rating category ($p_{y,1}$) [dimensionless]
    FirstRingTraceElement: NotationSymbol  # first of the four elements of a rank's gradient vector in the four-GPU ring AllReduce trace, each element one chunk ($a$) [varies]
    FirstSummandStandardDeviation: NotationSymbol  # standard deviation of the first of two independent zero-mean Gaussian variables in the variance-addition identity ($\sigma_a$) [dimensionless]
    FirstTwinCriticReturnEstimate: NotationSymbol  # action-value estimate of the first of the two independently parameterized critics in clipped double Q-learning ($Q_{\phi_1}$) [dimensionless]
    FittedSensitivity: NotationSymbol  # accuracy lost per unit of distribution divergence, fitted from labeled outcomes over a stated range ($\lambda$) [dimensionless]
    FixedClearanceAllowance: NotationSymbol  # sum of the fixed distance allowances of the stopping budget: localization bound, protective clearance margin and tracking-error bound ($\delta$) [m]
    FlangeCameraTransform: NotationSymbol  # rigid transform mapping camera coordinates into the tool-flange frame, the extrinsic of a wrist-mounted camera ($\mathbf{T}_{\text{flange}}^{\text{cam}}$)
    FlangePose: NotationSymbol  # pose of a robot arm's mounting flange, before any tool offset ($\mathbf{T}_{\text{flange}}$)
    FlangeReferenceFrame: NotationSymbol  # coordinate frame attached to a manipulator's mounting flange, before any tool ($\mathcal{F}_{\text{flange}}$)
    FlangeTargetTransform: NotationSymbol  # rigid transform mapping calibration-target coordinates into the tool-flange frame ($\mathbf{T}_{\text{flange}}^{\text{target}}$)
    FlattenedFilterWeights: NotationSymbol  # column vector of a filter's weights, flattened so im2col can apply the filter as a matrix product ($\mathbf{w}_{\text{col}}$)
    FleetCommunicationTaxonomyAxis: NotationSymbol  # Communication axis of the C³ taxonomy, data movement across the network fabric, viewed as the set of techniques it covers so that axis intersections name zones of the taxonomy ($C_2$)
    FleetComputeTaxonomyAxis: NotationSymbol  # Compute axis of the C³ taxonomy, local arithmetic and memory within a node, viewed as the set of techniques it covers so that axis intersections name zones of the taxonomy ($C_1$)
    FleetCoordinationTaxonomyAxis: NotationSymbol  # Coordination axis of the C³ taxonomy, synchronization, failure recovery and scheduling across the fleet, viewed as the set of techniques it covers so that axis intersections name zones of the taxonomy ($C_3$)
    FleetEnergyProductivity: NotationSymbol  # useful floating-point operations a fleet delivers per joule of device, cooling and network energy ($\rho_{\text{energy}}$) [FLOP/J]
    FleetManufacturingDistribution: NotationSymbol  # distribution of unit-to-unit physical variation across the manufactured robots of a fleet, from which qualification units are sampled ($\mathcal{M}$)
    FleetResourceCapacity: NotationSymbol  # total fleet capacity of one resource type ($C_{r}$) [varies]
    FleetRobotCount: NotationSymbol  # number of robots in a deployed fleet ($N_{\text{fleet}}$)
    FlippedFloatExponentField: NotationSymbol  # unsigned integer stored in the exponent bits of a floating-point number after one exponent bit flips ($e'$) [dimensionless]
    FloatEncodedNumber: NotationSymbol  # real number that a finite normal binary floating-point bit pattern represents ($x$) [dimensionless]
    FloatExponentField: NotationSymbol  # unsigned integer stored in the exponent bits of a binary floating-point number, before the bias is subtracted ($e$) [dimensionless]
    FloatMantissaField: NotationSymbol  # unsigned integer stored in the mantissa (fraction) bits of a binary floating-point number ($m$) [dimensionless]
    FloatSignBit: NotationSymbol  # sign bit of a binary floating-point number, zero for positive and one for negative ($s$) [dimensionless]
    FloatingBaseAccelerations: NotationSymbol  # accelerations of the unactuated floating base, the base block of the generalized accelerations ($\ddot{\mathbf{q}}_b$)
    FloatingBaseBiasForces: NotationSymbol  # floating-base block of the generalized bias forces (Coriolis, centrifugal and gravity terms on the base coordinates) ($\mathbf{h}_b$)
    FloatingBaseInertiaBlock: NotationSymbol  # block of the generalized inertia matrix coupling floating-base accelerations to floating-base generalized forces ($\mathbf{M}_{bb}$) [varies]
    FloatingBasePose: NotationSymbol  # pose of the unactuated floating base of a legged robot, its position and orientation, the base block of the generalized coordinates ($\mathbf{q}_b$)
    FlopPrice: NotationSymbol  # price of executing one floating-point operation on rented or owned accelerators ($c_{\text{FLOP}}$) [USD/FLOP]
    FlowMatchingLoss: NotationSymbol  # mean squared error between the learned velocity field and the conditional path velocity, averaged over flow time, noise and demonstration samples ($\mathcal{L}_{\text{FM}}$)
    FlowProbabilityPath: NotationSymbol  # time-indexed family of densities that flow matching transports from source noise at flow time zero to the demonstration distribution at flow time one ($\rho$)
    FlowSampleVector: NotationSymbol  # point in action-chunk space that flow matching transports, subscripted by flow time: source noise at zero, a demonstration chunk at one ($\mathbf{y}$)
    FlowTimeCoordinate: NotationSymbol  # normalized continuous position along a flow-matching probability path, from source noise at zero to demonstration data at one ($t$) [dimensionless]
    FocalLength: NotationSymbol  # focal length of a pinhole camera expressed in pixels, the scale from normalized image coordinates to pixels ($f$) [pixels]
    FollowerForwardWave: NotationSymbol  # forward wave variable at the follower side of a teleoperation channel, the leader's forward wave after the channel delay ($\mathbf{u}_s$)
    FollowerJointPositions: NotationSymbol  # joint positions of a teleoperation follower arm ($\mathbf{q}_{\text{follower}}$)
    FollowerMechanicalLatency: NotationSymbol  # time for the follower's actuators and links to reach a commanded acceleration ($t_{\text{mech}}$) [s]
    FollowerReturnWave: NotationSymbol  # returning wave variable at the follower side of a teleoperation channel, which reaches the leader after the channel delay ($\mathbf{w}_s$)
    ForceGuidedRelay: NotationSymbol  # force-guided relay of a redundant emergency-stop channel, whose monitored contacts block restart if it fails to open ($K$)
    ForceReadingInnovation: NotationSymbol  # innovation of a scalar force-sensor reading, the reading minus its prediction ($\nu$) [N]
    ForceSensorBias: NotationSymbol  # slowly drifting additive bias of a strain-gauge force sensor ($b$) [N]
    ForceSensorReading: NotationSymbol  # raw force-sensor reading, the sum of the true contact force and the sensor bias ($y$) [N]
    ForegroundLatency: NotationSymbol  # measured latency of the foreground application's interactive work on a device ($L_{\text{lat,fg}}$) [s]
    ForgetDataset: NotationSymbol  # training records whose influence machine unlearning must remove, the forget set ($D_{\text{forget}}$)
    ForkBoundarySet: NotationSymbol  # set of completed-turn boundaries of a trajectory at which continuations are forked ($\mathcal{B}_{\text{fork}}$)
    ForkedTrajectoryTokenCount: NotationSymbol  # total output tokens of a trajectory and all continuation branches forked under a given fork schedule ($N_{\text{total}}$)
    FormatComplianceScore: NotationSymbol  # score between zero and one measuring how well a trajectory's tool calls and output follow the required format ($\Phi$) [dimensionless]
    FormatEfficiency: NotationSymbol  # fraction of the bytes a storage format reads that carry the features a job selects ($\eta_{\text{format}}$) [dimensionless]
    FormatRewardWeight: NotationSymbol  # non-negative weight on the format-compliance term of a shaped trajectory reward ($\lambda_{\text{fmt}}$) [dimensionless]
    ForwardDiffusionDistribution: NotationSymbol  # distribution defined by the fixed forward diffusion process that adds Gaussian noise to a clean action chunk step by step ($q$)
    ForwardEulerEigenvalue: NotationSymbol  # eigenvalue of the forward Euler transition matrix ($\lambda_{\text{FE}}$) [dimensionless]
    ForwardEulerTransitionMatrix: NotationSymbol  # discrete-time transition matrix that forward Euler integration produces, identity plus the system matrix times the step ($\mathbf{A}_{\text{FE}}$)
    ForwardInvariantSafeSet: NotationSymbol  # set of states on which the barrier certificate is nonnegative, kept invariant by the safety filter ($\mathcal{C}$)
    ForwardKinematicsMap: NotationSymbol  # forward kinematics: the map from joint positions to the positions of points on the robot's links in the base frame ($\mathbf{FK}$)
    ForwardOccupancySet: NotationSymbol  # region a tracked actor could occupy a given time after its last measurement ($\mathcal{R}$)
    ForwardPassTokenCount: NotationSymbol  # number of token positions, decode and prefill together, that one forward pass of the serving engine processes ($n$)
    ForwardPosteriorMean: NotationSymbol  # mean of the Gaussian posterior over the previous diffusion step given the noisy chunk and the clean chunk ($\tilde{\boldsymbol{\mu}}$)
    ForwardPosteriorVariance: NotationSymbol  # variance of the Gaussian posterior over the previous diffusion step given the noisy chunk and the clean chunk ($\tilde{\beta}$) [dimensionless]
    FourthRingTraceElement: NotationSymbol  # fourth of the four elements of a rank's gradient vector in the four-GPU ring AllReduce trace, each element one chunk ($d$) [varies]
    FrameBitCount: NotationSymbol  # length of one EtherCAT frame on the wire, in bits ($B_{\text{frame}}$)
    FrameBlockingTime: NotationSymbol  # longest time a fieldbus frame waits behind a lower-priority frame already transmitting, written B_i for frame i ($B$) [s]
    FrameExposureStartTime: NotationSymbol  # instant at which the first row of a rolling-shutter frame begins its exposure ($t_{\text{start}}$) [s]
    FrameIndex: NotationSymbol  # position of one frame in a fieldbus message set ($i$)
    FramePayloadSize: NotationSymbol  # payload length of one fieldbus frame ($B_{\text{payload}}$) [bytes]
    FramePeriod: NotationSymbol  # release period of one periodic fieldbus frame, written T_i for frame i ($T$) [s]
    FrameReadoutTime: NotationSymbol  # time a camera takes to read its pixels out after exposure ($t_{\text{readout}}$) [s]
    FrameReleaseJitter: NotationSymbol  # release jitter of one periodic fieldbus frame, written J_j for frame j ($J$) [s]
    FrameResponseTime: NotationSymbol  # worst-case time from the release of a fieldbus frame to the end of its transmission, written R_i for frame i ($R$) [s]
    FrameSerializationTime: NotationSymbol  # time to serialize frame i on a dual-rate fieldbus: arbitration bits, data bits, stuffing and interframe spacing ($t_{\text{frame},i}$) [s]
    FrameTransmissionTime: NotationSymbol  # time one fieldbus frame occupies the bus, written C_i for frame i in a response-time analysis ($C$) [s]
    FrameWireBitCount: NotationSymbol  # complete number of bits frame i occupies on the wire: identifier, payload, CRC, stuffing and interframe spacing ($B_{\text{wire},i}$)
    FreeConfigurationSpace: NotationSymbol  # set of configurations at which the robot touches no obstacle, the configuration space minus the obstacle set ($\mathcal{C}_{\text{free}}$)
    FreeCreditCount: NotationSymbol  # number of credits a producer holds that are not attached to any admitted item ($N_{\text{free}}$)
    FreeSpaceDepth: NotationSymbol  # distance from a hull's surface to the nearest occupied or unknown cell ($d_{\text{free}}$) [m]
    FreeSpaceLogOddsIncrement: NotationSymbol  # negative log-odds increment applied to each voxel a range ray traverses before its return ($L_{\text{free}}$) [dimensionless]
    FreeSurfaceTiltAngle: NotationSymbol  # angle by which the free surface of a carried liquid tilts from horizontal under steady acceleration ($\theta_{\text{tilt}}$) [rad]
    FreeboardHeight: NotationSymbol  # height from the free surface of a carried liquid to the rim of its container ($h_{f}$) [m]
    FreshEvidenceTime: NotationSymbol  # capture time of the fresh physical evidence that renews a belief's evidence epoch ($t_{\text{fresh}}$) [s]
    FreshInputTokenCount: NotationSymbol  # number of a trajectory's input tokens billed at the fresh-input rate under ideal prefix reuse, the initial prompt plus every appended increment once ($S_{\text{fresh}}$)
    FreshnessObjective: NotationSymbol  # freshness dimension of a service's SLO vector: its indicator, target, window and owner ($S_{\text{freshness}}$)
    FrictionCoefficientFloor: NotationSymbol  # lowest friction coefficient a braking or contact premise assumes the surface provides ($\mu_{\min}$) [dimensionless]
    FrictionPyramidMatrix: NotationSymbol  # matrix of linear inequalities whose feasible set is an inner polyhedral approximation of one contact's friction cone ($\mathbf{F}$)
    FrobeniusNorm: NotationSymbol  # label marking a matrix norm as the Frobenius norm, the square root of the sum of squared entries ($F$)
    FrozenBackboneFunction: NotationSymbol  # function the frozen pretrained layers compute, to which a residual adapter's output is added ($f$)
    FrozenFeatureEncoder: NotationSymbol  # fixed pretrained feature encoder that maps an input to the embedding stored for replay ($f$)
    FrustumFeature: NotationSymbol  # feature lifted to one depth hypothesis of a camera ray, the cell's feature weighted by that bin's depth probability ($\mathbf{c}_{u, v, k}$)
    FullBatchLoss: NotationSymbol  # training loss averaged over all examples of one effective batch ($\mathcal{L}_B$)
    FullDatasetSize: NotationSymbol  # number of samples in the full dataset before any selection removes some ($D_{\text{total}}$)
    FullPrecisionWeight: NotationSymbol  # one model weight at full precision, before quantization maps it to an integer code ($w$) [dimensionless]
    FullSnapshotWriteVolume: NotationSymbol  # cumulative bytes a trajectory writes when every checkpoint is a full snapshot ($D_{\text{vol,full}}$) [bytes]
    FullSpeedTravelTime: NotationSymbol  # total time the machine keeps moving at its detection speed before full braking: pre-brake delay, takeover wait and half the jerk ramp ($b$) [s]
    FunctionInputAdjoint: NotationSymbol  # reverse-mode adjoint of an input variable, the derivative of the output with respect to it ($\bar{x}$) [dimensionless]
    FunctionInputDimension: NotationSymbol  # number of inputs of a differentiated function, the column count of its Jacobian ($N$)
    FunctionInputElement: NotationSymbol  # one element of the input vector of a layer or elementwise function ($x$) [dimensionless]
    FunctionInputTangent: NotationSymbol  # forward-mode tangent of an input variable, the seed of the forward sweep ($\dot{x}$) [dimensionless]
    FunctionInputVariable: NotationSymbol  # scalar input variable of an illustrative function being traced or differentiated ($x$) [dimensionless]
    FunctionInputVector: NotationSymbol  # vector a layer or differentiated function takes as input ($\mathbf{x}$)
    FunctionJacobian: NotationSymbol  # matrix of partial derivatives of a function's outputs with respect to its inputs ($\mathbf{J}$)
    FunctionOutputDimension: NotationSymbol  # number of outputs of a differentiated function, the row count of its Jacobian ($M$)
    FunctionOutputElement: NotationSymbol  # one element of the output vector of a layer or differentiated function ($y$) [dimensionless]
    FunctionOutputVariable: NotationSymbol  # scalar output of an illustrative function or computational graph ($z$) [dimensionless]
    FunctionOutputVector: NotationSymbol  # vector a layer or differentiated function produces as output ($\mathbf{y}$)
    FunctionSecondInputVariable: NotationSymbol  # second scalar input of an illustrative two-input function ($y$) [dimensionless]
    FundamentalMatrix: NotationSymbol  # three-by-three matrix relating corresponding pixel points in two uncalibrated views ($\mathbf{F}$)
    FusedConvolutionBias: NotationSymbol  # per-channel bias vector of a convolution after the following batch normalization is folded into it ($\mathbf{b}_{\text{fused}}$)
    FusedConvolutionWeight: NotationSymbol  # convolution weight tensor after the following batch normalization is folded into it ($\mathbf{W}_{\text{fused}}$)
    FusedTokenCount: NotationSymbol  # total number of vision, language and state tokens in the fused input sequence of an embodied transformer ($N_{\text{total}}$)
    FusionCandidate: NotationSymbol  # candidate record that one or more retrieval methods return, scored by reciprocal rank fusion from its ranks in their lists ($x_{\text{cand}}$)
    FusionInputList: NotationSymbol  # ranked list of candidates one retrieval method returns, one input to reciprocal rank fusion ($\mathcal{L}$)
    FusionInputRank: NotationSymbol  # 1-based rank of a candidate within one retrieval method's ranked list, as reciprocal rank fusion reads it ($r$)
    FutureControlTrajectory: NotationSymbol  # control input as a function of time from a given instant onward, the future a definition of inevitable collision quantifies over ($\mathbf{u}_{[t, \infty)}$)
    FutureTimeInstant: NotationSymbol  # any time instant at or after a given one, the variable a statement about the whole future quantifies over ($t'$) [s]
    GaeAdvantageEstimate: NotationSymbol  # generalized advantage estimate at a decision step, the discounted and trace-decayed sum of the temporal-difference residuals from that step to the end of the episode ($\widehat{A}^{\text{GAE}}$) [dimensionless]
    GateCurrent: NotationSymbol  # current the gate driver sources or sinks at a power switch's gate ($I_g$) [A]
    GateDriverSkew: NotationSymbol  # mismatch in propagation delay between the isolated gate drivers of a half-bridge ($t_{\text{prop, skew}}$) [s]
    GatePropagationDelay: NotationSymbol  # propagation delay of a logic gate, written as a function of its supply voltage ($t_{\text{prop}}$) [s]
    GateSourceCapacitance: NotationSymbol  # capacitance between the gate and source of a power switch ($C_{\text{gs}}$) [F]
    GateSourceCharge: NotationSymbol  # share of a power switch's gate charge that charges the gate-source capacitance ($Q_{\text{gs}}$) [A*s]
    GateSourceVoltage: NotationSymbol  # voltage between the gate and source of a power switch ($V_{\text{gs}}$) [V]
    GaussianCovarianceMatrix: NotationSymbol  # covariance matrix of a multivariate Gaussian distribution ($\boldsymbol{\Sigma}$) [varies]
    GaussianDistribution: NotationSymbol  # Gaussian (normal) distribution with the mean and covariance given as its arguments ($\mathcal{N}$)
    GaussianFirstAxisScale: NotationSymbol  # positive scale of one three-dimensional Gaussian primitive along its first principal axis ($s_{x}$) [m]
    GaussianMeanVector: NotationSymbol  # mean vector of a multivariate Gaussian distribution ($\boldsymbol{\mu}$)
    GaussianMechanismNoiseVector: NotationSymbol  # Gaussian noise vector a Gaussian mechanism adds to a vector-valued query output ($\mathbf{z}$)
    GaussianMechanismStandardDeviation: NotationSymbol  # standard deviation of the Gaussian noise a Gaussian mechanism adds to a query output, in the query's units ($\sigma_{\text{mech}}$) [varies]
    GaussianOpacityLogit: NotationSymbol  # unbounded logit of one Gaussian primitive whose sigmoid is its stored opacity ($o$) [dimensionless]
    GaussianPrimitiveCovariance: NotationSymbol  # covariance matrix of one three-dimensional Gaussian primitive, setting its ellipsoidal shape and orientation ($\boldsymbol{\Sigma}$) [m^2]
    GaussianPrimitiveDensity: NotationSymbol  # unnormalized density of one three-dimensional Gaussian primitive of a splatted scene, evaluated at a point in space ($G$) [dimensionless]
    GaussianPrimitiveIndex: NotationSymbol  # index of one Gaussian primitive, in the front-to-back depth order when blending ($i$)
    GaussianPrimitiveMean: NotationSymbol  # centroid of one three-dimensional Gaussian primitive of a splatted scene, in world coordinates ($\boldsymbol{\mu}$) [m]
    GaussianPrimitiveOpacity: NotationSymbol  # stored opacity of one Gaussian primitive, between zero and one, the sigmoid of its opacity logit ($\alpha$) [dimensionless]
    GaussianPrimitiveScaleMatrix: NotationSymbol  # diagonal matrix of the positive axis scales of one three-dimensional Gaussian primitive ($\mathbf{S}$)
    GaussianSecondAxisScale: NotationSymbol  # positive scale of one three-dimensional Gaussian primitive along its second principal axis ($s_{y}$) [m]
    GaussianStandardDeviation: NotationSymbol  # standard deviation of a Gaussian distribution, or of each coordinate of an isotropic one ($\sigma$) [varies]
    GaussianThirdAxisScale: NotationSymbol  # positive scale of one three-dimensional Gaussian primitive along its third principal axis ($s_{z}$) [m]
    GaussianViewDependentColor: NotationSymbol  # view-dependent color one Gaussian primitive contributes, given by its spherical-harmonic coefficients ($c$) [dimensionless]
    GearFaceWidth: NotationSymbol  # face width of a gear tooth in the Lewis bending equation ($b$) [m]
    GearModule: NotationSymbol  # module of a gear, pitch diameter divided by tooth count ($m$) [m]
    GearRatedFatigueLife: NotationSymbol  # rated fatigue life of the wrist strain-wave gear, in torque-reversal stress cycles ($L_{\text{gear}}$)
    GearReductionRatio: NotationSymbol  # ratio of motor speed to joint output speed across a transmission ($N_{\text{gear}}$) [dimensionless]
    GearTangentialLoad: NotationSymbol  # tangential force transmitted across one gear-tooth contact ($W_t$) [N]
    GearToothBendingStress: NotationSymbol  # bending stress at the root of a gear tooth, by the Lewis equation ($\sigma_b$) [N/m^2]
    GearTorqueReversalCount: NotationSymbol  # number of torque reversals one attempt imposes on the wrist strain-wave gear ($n_{\text{gear}}$)
    GearboxTransmissionEfficiency: NotationSymbol  # fraction of input mechanical power a transmission delivers at its output ($\eta_{\text{gear}}$) [dimensionless]
    GemmAccumulatorScale: NotationSymbol  # scalar multiplier applied to the existing output matrix before the product is added in a general matrix multiply ($\beta$) [dimensionless]
    GemmColumnDimension: NotationSymbol  # number of columns of the right operand and of the output in a matrix multiply ($N$)
    GemmInnerDimension: NotationSymbol  # shared inner dimension of a general matrix multiply, the number of products summed per output element ($K$)
    GemmLeftOperand: NotationSymbol  # left input matrix of a general matrix multiply, of size M by K ($\mathbf{A}$)
    GemmLeftOperandElement: NotationSymbol  # one entry of the left operand of a matrix multiply, indexed by row and reduction position ($A$) [dimensionless]
    GemmLeftOperandEntry: NotationSymbol  # entry in row i and column k of the left input matrix of a matrix multiply ($A_{ik}$) [dimensionless]
    GemmOutputElement: NotationSymbol  # one entry of the output of a matrix multiply, indexed by row and column ($C$) [dimensionless]
    GemmOutputEntry: NotationSymbol  # entry in row i and column j of the output matrix of a matrix multiply ($C_{ij}$) [dimensionless]
    GemmOutputMatrix: NotationSymbol  # output matrix of a general matrix multiply, of size M by N, which the operation updates in place ($\mathbf{C}$)
    GemmProductScale: NotationSymbol  # scalar multiplier applied to the matrix product in a general matrix multiply ($\alpha$) [dimensionless]
    GemmReductionIndex: NotationSymbol  # index of the reduction position a matrix-multiply loop is summing over ($k$)
    GemmRightOperand: NotationSymbol  # right input matrix of a general matrix multiply, of size K by N ($\mathbf{B}$)
    GemmRightOperandElement: NotationSymbol  # one entry of the right operand of a matrix multiply, indexed by reduction position and column ($B$) [dimensionless]
    GemmRightOperandEntry: NotationSymbol  # entry in row k and column j of the right input matrix of a matrix multiply ($B_{kj}$) [dimensionless]
    GemmRowDimension: NotationSymbol  # number of rows of the left operand and of the output in a matrix multiply ($M$)
    GemmTileColumnDimension: NotationSymbol  # number of output columns in one tile of a tiled matrix multiply ($N_{\text{tile}}$)
    GemmTileInnerDimension: NotationSymbol  # length of the reduction dimension covered by one tile of a tiled matrix multiply ($K_{\text{tile}}$)
    GemmTileRowDimension: NotationSymbol  # number of output rows in one tile of a tiled matrix multiply ($M_{\text{tile}}$)
    GemvArithmeticIntensity: NotationSymbol  # arithmetic intensity of a matrix-vector multiply that reads each weight once, counting weight traffic only ($I_{\text{GEMV}}$) [FLOP/byte]
    GeneralPurposeEnergyProductivity: NotationSymbol  # useful floating-point operations general-purpose silicon delivers per joule on a given workload ($\rho_{\text{energy,general}}$) [FLOP/J]
    GeneralizedAccelerationLowerBound: NotationSymbol  # lower bound on generalized accelerations that keeps joint velocities and positions within limits over one integration step ($\ddot{\mathbf{q}}_{\min}$)
    GeneralizedAccelerationUpperBound: NotationSymbol  # upper bound on generalized accelerations that keeps joint velocities and positions within limits over one integration step ($\ddot{\mathbf{q}}_{\max}$)
    GeneralizedAccelerations: NotationSymbol  # second time derivative of the generalized coordinates ($\ddot{\mathbf{q}}$)
    GeneralizedBiasForces: NotationSymbol  # generalized bias forces of the equations of motion, the Coriolis, centrifugal and gravity terms together ($\mathbf{h}$)
    GeneralizedCoordinateCount: NotationSymbol  # number of generalized coordinates of a mechanism, its degrees of freedom (joints for a serial arm) ($n$)
    GeneralizedCoordinates: NotationSymbol  # joint angles and prismatic extensions that fix the configuration of a mechanism ($\mathbf{q}$)
    GeneralizedInertiaMatrix: NotationSymbol  # joint-space mass matrix of an articulated robot, mapping joint accelerations to the joint torques they require ($\mathbf{M}$) [kg*m^2]
    GeneralizedParetoDistribution: NotationSymbol  # generalized Pareto cumulative distribution function of the excess over a high threshold ($H$)
    GeneralizedParetoLikelihood: NotationSymbol  # likelihood of the generalized Pareto scale and shape given the recorded threshold exceedances ($L$)
    GeneralizedVelocities: NotationSymbol  # first time derivative of the generalized coordinates ($\dot{\mathbf{q}}$)
    GeneratedResponse: NotationSymbol  # response a language model generates for a prompt, as scored by a reward model ($y$)
    GeneratedSampleCount: NotationSymbol  # number of candidates generated for one task when estimating pass@k, from which subsets of k runs are drawn ($n$)
    GeneratedTokenCap: NotationSymbol  # largest number of generated tokens the runtime lets one rollout produce, enforced by hard termination, that normalizes a length penalty ($N_{\text{cap}}$)
    GenerationStepErrorRate: NotationSymbol  # rate of rare errors per useful generation step in a verification cadence model, assumed constant ($\lambda_{\text{error}}$) [dimensionless]
    GenerativeActionModelDistribution: NotationSymbol  # probability distribution that a learned generative action model with parameters theta defines over action chunks and its latent or noisy intermediates ($p_\theta$)
    GeneratorLatentVector: NotationSymbol  # latent vector sampled from standard Gaussian noise and mapped by a generator to one synthetic record ($\mathbf{z}$)
    GenericRandomVariable: NotationSymbol  # arbitrary real-valued random variable in a general probability inequality ($X$)
    GeometricStateRepresentation: NotationSymbol  # geometric state representation computed solely from raw sensory data, such as bounding boxes or metric geometry ($\mathbf{Z}_{\text{geom}}$)
    GevLocationParameter: NotationSymbol  # location parameter of the generalized extreme value distribution ($\mu$) [varies]
    GevScaleParameter: NotationSymbol  # scale parameter of the generalized extreme value distribution ($\sigma$) [varies]
    GlobalBatchGradient: NotationSymbol  # gradient over the global batch, the average of every data-parallel worker's minibatch gradient that one synchronous update applies ($\mathbf{g}_{\text{global}}$)
    GlobalBatchSize: NotationSymbol  # number of examples one synchronous optimizer update averages over across all data-parallel workers ($B_{\text{global}}$)
    GlobalDeviationPenalty: NotationSymbol  # penalty function measuring how far a personalized client model deviates from the global model ($\mathcal{R}$)
    GlobalModelParameters: NotationSymbol  # parameters of the aggregated global model in federated learning, the reference personalized clients stay near ($\theta_{\text{global}}$)
    GoalConfiguration: NotationSymbol  # joint configuration a planned trajectory must reach at its final time ($\mathbf{q}_{\text{goal}}$)
    GoalSet: NotationSymbol  # set of terminal poses within the positional and angular tolerances of a task ($\mathcal{G}_{\text{goal}}$)
    GoodputRelativeTolerance: NotationSymbol  # declared largest relative shortfall of the candidate's trajectory goodput below the baseline's that a release gate allows ($\epsilon_{\mathcal{G}}$) [dimensionless]
    GpdCumulativeDistribution: NotationSymbol  # cumulative distribution function of the generalized Pareto distribution of threshold exceedances ($F_{\text{GPD}}$) [dimensionless]
    GpdScaleParameter: NotationSymbol  # scale parameter of the generalized Pareto distribution fitted to exceedances over a threshold ($\tilde{\sigma}$) [varies]
    GpuAvailability: NotationSymbol  # probability that one GPU is up at a randomly chosen instant ($A_{\text{GPU}}$) [dimensionless]
    GpuEnergy: NotationSymbol  # energy the GPU boards consume over a stated interval, as board telemetry reports it ($E_{\text{GPU}}$) [J]
    GpuFailureRate: NotationSymbol  # rate at which one GPU fails, the reciprocal of its mean time to failure ($\lambda_{\text{GPU}}$) [failures/hour]
    GpuMttf: NotationSymbol  # mean time to failure of one GPU, its average operating time before it fails ($\text{MTTF}_{\text{GPU}}$) [hours]
    GpuTopologicalDistance: NotationSymbol  # topological distance between two GPUs, 0 on the same node, 1 in the same rack and 2 across racks ($d$) [dimensionless]
    GradientClippingNorm: NotationSymbol  # largest L2 norm a per-example gradient may keep after clipping in differentially private training ($C$) [varies]
    GradientCovarianceMatrix: NotationSymbol  # covariance matrix of per-example gradients about the true gradient, whose trace is the total gradient variance ($\Sigma$)
    GradientElementCount: NotationSymbol  # number of elements in the gradient vector a worker communicates ($d_{\text{grad}}$)
    GradientMemory: NotationSymbol  # memory holding the gradient of every trainable parameter during training ($M_{\text{gradients}}$) [bytes]
    GradientPayload: NotationSymbol  # bytes of one full set of gradients that a data-parallel collective reduces each step, the parameter count times the gradient's communication width ($M_{\text{grad}}$) [bytes]
    GradientStaleness: NotationSymbol  # number of parameter updates between computing a gradient and applying it ($\tau_{\text{stale}}$)
    GrantAllowedAccesses: NotationSymbol  # set of accesses a capability grant allows, written as a function of the grant ($\mathcal{A}$)
    GrantCondition: NotationSymbol  # condition predicate of a capability grant, which an access must satisfy, such as binding use to one task or recipient ($\phi_{\Gamma}$)
    GrantExpiryTime: NotationSymbol  # time after which a capability grant admits nothing ($t_{\text{exp}}$) [s]
    GrantIssueTime: NotationSymbol  # time at which a capability grant is issued, the reference for its expiry ($T$) [s]
    GrantRights: NotationSymbol  # set of operations a capability grant permits its holder to perform ($\mathcal{R}_{\Gamma}$)
    GraphEdgeSet: NotationSymbol  # set of edges, the data dependencies between nodes, of a computational graph ($E$)
    GraphNodeIndex: NotationSymbol  # index of any node of a computational graph in topological order ($i$)
    GraphNodeSet: NotationSymbol  # set of nodes (variables) of a computational graph ($V$)
    GraspContactDisplacement: NotationSymbol  # displacement from the grasped object's reference origin to one grasp contact, whose cross product with the contact force gives that contact's moment in the wrench-space test for force closure ($\mathbf{d}$) [m]
    GraspForceCeiling: NotationSymbol  # largest grasp contact force the object tolerates without crushing or deforming ($f_{\max}$) [N]
    GraspForceFloor: NotationSymbol  # smallest grasp contact force that holds the object against slip ($f_{\min}$) [N]
    GraspMap: NotationSymbol  # linear map from contact forces to the wrench they exert on the grasped object ($\mathbf{G}$)
    GraspOrientation: NotationSymbol  # rotation matrix giving the gripper orientation of a target grasp pose ($\mathbf{R}_{\text{grasp}}$) [dimensionless]
    GraspPose: NotationSymbol  # target pose of the gripper for one grasp affordance, its rotation and position in the robot base frame ($\mathbf{T}_{\text{grasp}}$)
    GraspPosition: NotationSymbol  # position of a target grasp pose in the robot base frame ($\mathbf{p}_{\text{grasp}}$) [m]
    GraspVerticalAcceleration: NotationSymbol  # upward acceleration an arm imposes on a grasped object while moving it ($a_z$) [m/s^2]
    GravitationalTorqueVector: NotationSymbol  # joint torques that gravity exerts on the links in a given configuration ($\mathbf{g}$) [N*m]
    Gravity: NotationSymbol  # gravitational acceleration at the surface of the Earth ($g$) [m/s^2]
    GravityCenterHeight: NotationSymbol  # height of a vehicle's center of gravity above the floor ($h_{\text{cg}}$) [m]
    GravityVector: NotationSymbol  # gravitational acceleration as a vector, pointing down with magnitude the surface gravity ($\mathbf{g}_0$) [m/s^2]
    GridFeedCapacity: NotationSymbol  # largest electrical power the utility grid connection can deliver to a facility ($P_{\text{grid}}$) [W]
    GridSideCellCount: NotationSymbol  # number of cells along each dimension of a cubic volumetric grid ($N$)
    GripContactForce: NotationSymbol  # true normal contact force a gripper applies to its workpiece, the hidden state of the force-plus-bias example ($x$) [N]
    GripperContactAngle: NotationSymbol  # angle between the gripper and the workpiece contact surface during a contact task ($\theta_{\text{contact}}$) [deg]
    GripperOpeningCommand: NotationSymbol  # normalized gripper opening command between closed and fully open ($a_{\text{gripper}}$) [dimensionless]
    GroundPlaneAbscissa: NotationSymbol  # first horizontal coordinate of a point on the ground plane of a mobile robot's map frame ($x$) [m]
    GroundPlaneOrdinate: NotationSymbol  # second horizontal coordinate of a point on the ground plane of a mobile robot's map frame ($y$) [m]
    GroupCommitWaitTime: NotationSymbol  # time one event waits under group commit, from entering the batching queue until its batch's durable acknowledgment ($L_{\text{lat}, i}$) [s]
    GroupContrastProbability: NotationSymbol  # probability that a group of independent rollouts on one task contains at least one pass and at least one failure, so its reward contrast is nonzero ($p_{\text{contrast}}$) [dimensionless]
    GroupDisparity: NotationSymbol  # nonnegative magnitude of the performance disparity between one group and a specified reference group ($\Delta_g$) [dimensionless]
    GroupMeanReward: NotationSymbol  # mean terminal reward over the rollouts of one task group, the GRPO baseline ($\mu_{R}$) [dimensionless]
    GroupRelativeAdvantage: NotationSymbol  # standardized coefficient GRPO attaches to every sampled token of one rollout: its reward minus the group mean, divided by the group reward dispersion ($\hat{A}$) [dimensionless]
    GroupRewardStandardDeviation: NotationSymbol  # population standard deviation, with divisor equal to the group size, of the terminal rewards in one task group ($\sigma_{R}$) [dimensionless]
    GroupRolloutIndex: NotationSymbol  # position of one rollout within the group sampled for a task ($i$)
    GroupSize: NotationSymbol  # number of rollouts sampled per task for group-relative scoring ($G$)
    GrpoSampledObjective: NotationSymbol  # sampled GRPO objective: the group average of per-rollout token means of the clipped surrogate minus the weighted reference penalty ($\widehat{\mathcal{J}}_{\text{GRPO}}$) [dimensionless]
    GuardBandDuration: NotationSymbol  # duration of the guard band a time-aware shaper closes before each scheduled traffic window ($t_{\text{guard}}$) [s]
    GuardedContactPredicate: NotationSymbol  # motion predicate that holds when the measured grasp contact force lies between the force floor that prevents slip and the ceiling that avoids crushing ($P_{\text{contact}}$)
    GuestImageMemory: NotationSymbol  # memory of the full image of one sandbox guest ($M_{\text{guest}}$) [bytes]
    GuestKernelBootTime: NotationSymbol  # time to initialize the guest kernel of one sandbox instance ($T_{\text{kernel}}$) [s]
    HalfPrecisionGemvIntensity: NotationSymbol  # arithmetic intensity of a batch-one matrix-vector multiply whose 16-bit weights stream from memory ($I_{\text{GEMV, 16-bit}}$) [FLOP/byte]
    HammingDataBitCount: NotationSymbol  # number of data bits one Hamming codeword protects ($n$)
    HandForceCriterion: NotationSymbol  # peak contact force a chosen criterion allows on the human hand ($F_h$) [N]
    HandleAffordanceFrame: NotationSymbol  # affordance frame attached to the handle of the mug ($\mathcal{F}_{\text{handle}}$)
    HandoffExceedanceProbability: NotationSymbol  # allowed probability that one plan handoff misses its admission deadline ($\epsilon$) [dimensionless]
    HandoverTimeout: NotationSymbol  # countdown an arbiter sets for an unacknowledged handover before falling back to a controlled stop ($T_{\text{timeout}}$) [s]
    HandoverTimeoutCeiling: NotationSymbol  # configured protocol ceiling on the handover timeout, which the clearance-derived fallback deadline may shorten but never extend ($T_{\text{config}}$) [s]
    HandshakeProtocolLatency: NotationSymbol  # time the four-phase takeover handshake takes, from request through commit notification ($T_{\text{handshake}}$) [s]
    HardwareCommonCauseRate: NotationSymbol  # rate of hazards from hardware faults and common causes that no shield coverage removes ($\lambda_{\text{other}}$) [1/h]
    HardwareParameterVector: NotationSymbol  # physical parameter vector of the real machine, in the coordinates of the simulator parameters ($\boldsymbol{\xi}_{\text{real}}$) [varies]
    HardwareServiceLife: NotationSymbol  # length of time a piece of hardware stays in service, over which its embodied carbon is amortized ($T_{\text{lifetime}}$) [s]
    HarmInclusiveTaskCost: NotationSymbol  # monetary cost of one initiated cascade task including the harm of gate false accepts ($C_{\text{with\ harm}}$) [USD]
    HarmInstant: NotationSymbol  # instant at which a machine's unbraked trajectory would reach the point of harm ($t_{\text{harm}}$) [s]
    HarmLeadTime: NotationSymbol  # predicted interval until a specified harmful event under a stated continuation of the current motion and disturbance assumptions ($t_{\text{harm}}$) [s]
    HarmonicDriveTorsionalStiffness: NotationSymbol  # torsional stiffness of a strain-wave (harmonic) drive, torque per unit of angular wind-up ($k_c$) [N*m/rad]
    HarmonicPeriodMultiple: NotationSymbol  # integer multiple relating each longer task period to every shorter one in a harmonic task set ($m$)
    HarnessConnectorResistance: NotationSymbol  # resistance of the wiring harness and connectors between a battery and a bus node ($R_{\text{harness}}$) [Ohm]
    HashBucketCount: NotationSymbol  # number of buckets in the voxel-block hash table ($N_{\text{buckets}}$)
    HashEntrySize: NotationSymbol  # bytes per entry of the voxel-block hash table ($s_{\text{entry}}$) [bytes]
    HashRingServer: NotationSymbol  # one server placed on a consistent-hashing ring ($s$)
    HashRingServerCount: NotationSymbol  # number of servers placed on a consistent-hashing ring ($N$)
    HashRingServerSet: NotationSymbol  # set of servers placed on a consistent-hashing ring ($\mathcal{S}_{\text{srv}}$)
    HashedGridMemory: NotationSymbol  # memory a sparse block-hashed voxel map occupies ($M_{\text{hashed}}$) [bytes]
    HazardClassIndex: NotationSymbol  # index of one defined hazard class ($h$)
    HazardOnset: NotationSymbol  # interval from the last valid renewal to the physical onset of powered deceleration ($t_{\text{onset}}$) [s]
    HazardOnsetBound: NotationSymbol  # upper bound on the interval from the last valid renewal to the onset of powered deceleration, set by the lease path ($t_{\text{bound}}$) [s]
    HazardousProposalRate: NotationSymbol  # rate at which the policy emits hazardous action proposals of class h before the permission layer ($\lambda_{\text{proposal}, h}$) [1/h]
    HazardousWorldState: NotationSymbol  # physical world state in which the same commanded trajectory causes damage or injury, the state a safety filter must detect ($s_B$)
    HeadAxisLabel: NotationSymbol  # einsum index label naming the attention-head axis of an attention tensor ($h$)
    HeadByteAllocation: NotationSymbol  # bytes of the observation budget reserved for the start of a tool's output, captured into a linear buffer that fills once ($M_{\text{head}}$) [bytes]
    HeadComponentIndex: NotationSymbol  # position of one component within a query or key vector of one attention head ($m$)
    HeadDimensionAxisLabel: NotationSymbol  # einsum index label naming the per-head feature axis of an attention tensor ($d$)
    HeadKeyProjectionMatrix: NotationSymbol  # learned key projection matrix of one attention head ($\mathbf{W}_{K,i}$)
    HeadQueryProjectionMatrix: NotationSymbol  # learned query projection matrix of one attention head ($\mathbf{W}_{Q,i}$)
    HeadSamplingProbability: NotationSymbol  # probability with which head sampling retains each trace, decided before the trial's outcome is known ($\theta$) [dimensionless]
    HeadValuesProjectionMatrix: NotationSymbol  # learned value projection matrix of one attention head ($\mathbf{W}_{V,i}$)
    HeadingAngle: NotationSymbol  # heading angle of a planar mobile base in the unicycle model ($\theta_{\text{head}}$) [rad]
    HeadingCurvature: NotationSymbol  # heading error accumulated per unit distance driven, from wheel-diameter mismatch ($\kappa$) [rad/m]
    HealthyStepTime: NotationSymbol  # time of one training step after a straggler has been replaced ($T_{\text{healthy}}$) [s]
    HeartbeatAge: NotationSymbol  # age of the latest heartbeat on the receiver's local clock: local time minus the heartbeat's translated timestamp ($\Delta t_{\text{heartbeat}}$) [s]
    HeartbeatInterval: NotationSymbol  # interval at which each rank sends a liveness heartbeat to the coordinator ($H$) [s]
    HeartbeatReceiptTime: NotationSymbol  # reading of the receiving microcontroller's local clock at which a heartbeat's age is evaluated ($t_{\text{local}}$) [s]
    HeartbeatSequenceCounter: NotationSymbol  # strictly increasing sequence counter a heartbeat packet carries, used to detect loss and replay ($k$)
    HeartbeatTimestamp: NotationSymbol  # monotonic hardware timestamp a heartbeat packet carries, translated into the receiver's clock domain ($t^{\text{hb}}$) [s]
    HeatDissipationPower: NotationSymbol  # power the hardware at one signal handoff dissipates as heat, the energy-and-heat limit of the handoff matrix ($P_{\text{heat}}$) [W]
    HermiteBasisFunction: NotationSymbol  # cubic Hermite basis polynomial weighting an endpoint position or velocity in spline interpolation ($h$)
    HermiteInterpolationFraction: NotationSymbol  # fraction of a telemetry tick elapsed at the instant being interpolated ($\lambda$) [dimensionless]
    HermiteSegmentProgress: NotationSymbol  # elapsed time since the start of an interpolation segment divided by the segment duration, from zero to one ($s$) [dimensionless]
    HertzianContactCoefficient: NotationSymbol  # coefficient of the Hertzian contact law, the normal force divided by the deformation raised to the three-halves power ($k_{c}$) [varies]
    HertzianContactStiffness: NotationSymbol  # effective spring constant of the body region struck in a collaborative contact ($k_H$) [N/m]
    HiddenActivation: NotationSymbol  # activation vector a hidden layer outputs and passes to the next layer ($\mathbf{h}$)
    HiddenActivationMatrix: NotationSymbol  # matrix of hidden-layer activations after the nonlinearity, one row per example in the batch ($\mathbf{H}$)
    HiddenDimension: NotationSymbol  # width of a model's hidden representation, the transformer model width or a recurrent network's hidden-state size ($d$)
    HiddenUnitOutput: NotationSymbol  # output of one hidden unit of a small network ($h$) [dimensionless]
    HighOccupancyBinCount: NotationSymbol  # number of bins whose occupancy is at or above a threshold, in the balls-into-bins argument for two-choice routing ($m$)
    HigherPriorityFrames: NotationSymbol  # function giving the set of frames whose priority is higher than that of a given frame ($hp$)
    HigherPriorityTasks: NotationSymbol  # function giving the set of tasks or interrupt handlers whose priority exceeds that of the given task or interrupt ($hp$)
    HighestSignalFrequency: NotationSymbol  # highest frequency present in a sampled signal ($f_{\max}$) [Hz]
    HnswLayerIndex: NotationSymbol  # level of one layer in an HNSW graph, counted from layer 0 which holds every vector ($L$)
    HnswLevelDraw: NotationSymbol  # random number drawn uniformly from (0, 1] for each vector inserted into an HNSW graph, which sets the highest layer the vector joins ($U$) [dimensionless]
    HnswLevelNormalization: NotationSymbol  # normalization factor of HNSW layer assignment, the reciprocal of the natural log of the per-layer connection limit ($m_\ell$) [dimensionless]
    HnswMaximumConnectionCount: NotationSymbol  # largest number of bidirectional links each node keeps in one layer of an HNSW graph ($M$)
    HnswSearchBeamWidth: NotationSymbol  # number of closest candidates the layer-0 beam search of an HNSW query keeps in its priority queue, its primary recall knob ($ef_{\text{search}}$)
    HoldingTorque: NotationSymbol  # joint torque a motor must sustain at standstill to hold a load ($\tau_{\text{hold}}$) [N*m]
    HomogeneousCameraFramePoint: NotationSymbol  # homogeneous coordinates of a point expressed in the camera frame ($\tilde{\mathbf{p}}_C$) [m]
    HomogeneousPixelCoordinates: NotationSymbol  # image point written in homogeneous coordinates, pixel column and row with a trailing one ($\tilde{\mathbf{m}}$) [pixels]
    HomogeneousPointCoordinates: NotationSymbol  # homogeneous coordinates of a 3D point, its position with a trailing one, so a four-by-four transform acts by multiplication ($\tilde{\mathbf{p}}$) [m]
    HomogeneousTransform: NotationSymbol  # four-by-four matrix mapping coordinates from frame B into frame A ($\mathbf{T}_{AB}$)
    HomogeneousWorldFramePoint: NotationSymbol  # homogeneous coordinates of a point expressed in the world frame ($\tilde{\mathbf{p}}_W$) [m]
    HomographyColumn: NotationSymbol  # one column of a planar homography matrix, numbered h_1 to h_3 ($\mathbf{h}$)
    HomographyColumnEntry: NotationSymbol  # one entry of a homography column, h_{ik} being entry k of column i ($h$) [varies]
    HomographyConstraintVector: NotationSymbol  # six-vector built from two homography columns so that their quadratic form in the conic image is linear in its entries, written v_{ij} ($\mathbf{v}$)
    HomographyScaleFactor: NotationSymbol  # arbitrary nonzero scale relating a planar homography to the intrinsics times the pose columns ($\lambda$) [dimensionless]
    HomomorphicCiphertext: NotationSymbol  # ciphertext, the encrypted form of a plaintext under a public key ($c$)
    HomomorphicFunction: NotationSymbol  # function a homomorphic encryption scheme evaluates over ciphertexts, whose decrypted result equals the function applied to the plaintext ($f$)
    HomomorphicPublicKey: NotationSymbol  # public key of a homomorphic encryption scheme, used to encrypt plaintexts and to evaluate functions over ciphertexts ($pk$)
    HomomorphicSecretKey: NotationSymbol  # secret key of a homomorphic encryption scheme, needed to decrypt ciphertexts ($sk$)
    HorizonStepIndex: NotationSymbol  # position of one discrete control step within a model-predictive-control horizon, from zero to one less than the horizon ($k$)
    HorizontalFocalLength: NotationSymbol  # camera focal length expressed in pixels along the horizontal image axis ($f_x$) [pixels]
    HostArrivalTimestamp: NotationSymbol  # time at which the host operating system stamps a sensor packet on arrival ($t_{\text{host}}$) [s]
    HostHeartbeatRate: NotationSymbol  # rate at which the application processor sends liveness pulses to the safety microcontroller ($f_{\text{heartbeat}}$) [Hz]
    HostLinkBandwidth: NotationSymbol  # aggregate bandwidth, in one direction, of the host links that move KV state between a serving node's accelerator memory and host memory ($\text{BW}_{\text{link}}$) [bytes/s]
    HostPower: NotationSymbol  # electrical power one host server draws for its CPUs and other non-accelerator components ($P_{\text{host}}$) [W]
    HostPreparationTime: NotationSymbol  # time the host takes to prepare one batch for the accelerator, from read through batching ($T_{\text{prep}}$) [s]
    HostRuntimeCount: NotationSymbol  # number of host runtimes, agent applications, that must connect to independently maintained tool services ($N_{\text{host}}$)
    HostSoftwareTime: NotationSymbol  # host-side software time a step spends in the interpreter, dispatch and kernel launch ($T_{\text{sw}}$) [s]
    HostTransferBandwidth: NotationSymbol  # sustained bandwidth of the host-to-accelerator transfer path ($\text{BW}_{\text{host}}$) [bytes/s]
    HotTierHitProbability: NotationSymbol  # probability that a request to a tiered storage hierarchy is served by the hot (fastest) tier, its cache hit rate ($h$) [dimensionless]
    HotTierServiceTime: NotationSymbol  # time to serve one request that hits the hot tier of a storage hierarchy ($T_{\text{hot}}$) [s]
    HourlyAccountQueryCount: NotationSymbol  # number of queries one API account issues in an hour, compared against a static extraction-alert threshold ($q_{\text{hourly}}$)
    HourlyComponentFailureRate: NotationSymbol  # per-component failure rate expressed in failures per hour, converted from a FIT rating ($\lambda_{\text{hour}}$) [failures/hour]
    HuberLossThreshold: NotationSymbol  # residual magnitude at which the Huber penalty switches from quadratic to linear growth ($\delta$) [dimensionless]
    HullQueryTime: NotationSymbol  # time of one clearance query at the center of one collision hull ($t_{\text{query}}$) [s]
    HullSphereRadius: NotationSymbol  # radius of one spherical collision hull ($r_{\text{hull}}$) [m]
    HumanCommandRate: NotationSymbol  # time derivative of the operator's command vector, the slew the stabilization test bounds ($\dot{\mathbf{u}}_{\text{human}}$) [varies]
    HumanControlCommand: NotationSymbol  # control command vector the human operator issues during manual control ($\mathbf{u}_{\text{human}}$)
    HumanFallbackCost: NotationSymbol  # monetary cost of sending one rejected ticket through the human fallback workflow ($C_{h}$) [USD]
    HumanReactionTime: NotationSymbol  # neuromuscular reaction time of a human teleoperator, from perceiving an obstacle to issuing a corrective command ($T_{\text{human}}$) [s]
    HumanResolutionProbability: NotationSymbol  # probability that the human fallback workflow resolves a ticket the automated tier rejected ($h$) [dimensionless]
    HumanReviewCost: NotationSymbol  # cost of human review that an escalated action or a rejected attempt triggers, inspection and labor included ($C_{\text{human}}$) [USD]
    HumanStandaloneSensitivity: NotationSymbol  # fraction of actual positives, such as tumors, that human readers detect without AI assistance ($S_{\text{human}}$) [dimensionless]
    HumanWalkingSpeedBound: NotationSymbol  # speed bound for a person approaching the machine, as the safety standard states it ($v_h$) [m/s]
    HybridIterationTime: NotationSymbol  # time of one serving iteration that runs decode tokens and prefill chunks together ($T_{\text{step}}$) [s]
    HydraulicSupplyPressure: NotationSymbol  # pressure of the hydraulic fluid supplied to a cylinder ($p$) [N/m^2]
    HypothesisIndex: NotationSymbol  # index naming one of the two hypotheses of a test, zero for the null and one for the alternative ($h$)
    HypothesisTestFalseNegativeRate: NotationSymbol  # probability that a hypothesis test misses a true effect ($\beta_{\text{stat}}$) [dimensionless]
    HypothesisTestSignificanceLevel: NotationSymbol  # probability that a hypothesis test rejects a true null hypothesis ($\alpha_{\text{sig}}$) [dimensionless]
    HypothesizedAcceptanceProbability: NotationSymbol  # acceptance probability a sequential probability ratio test specifies under one of its two simple hypotheses, indexed 0 for the baseline and 1 for the degraded alternative ($p$) [dimensionless]
    HypothesizedTrialFailureProbability: NotationSymbol  # per-trial failure probability a sequential probability ratio test specifies under one of its two hypotheses, indexed 0 for the unacceptable null and 1 for the acceptable alternative ($p$) [dimensionless]
    IdealAcceleratorTime: NotationSymbol  # accelerator-time a job's work needs when every device runs at the ideal rate with no scheduling, runtime or program loss ($T_{\text{ideal}}$) [s]
    IdealMemoryQueueDelay: NotationSymbol  # time a safety read waits behind the transactions queued ahead of it counting only their transfer time, before bank conflicts, refresh and turnaround ($t_{\text{queue,ideal}}$) [s]
    IdempotentAlgebraElement: NotationSymbol  # element of a linear associative algebra that remains unchanged when raised to any positive integer power, Peirce's idempotent ($x$) [dimensionless]
    IdentityMatrix: NotationSymbol  # square matrix with ones on the diagonal and zeros elsewhere ($\mathbf{I}$)
    IllustrativeFunction: NotationSymbol  # function used as a worked example of graph tracing or differentiation ($f$)
    ImageChannelCount: NotationSymbol  # number of color channels of a camera image, three for RGB ($C$)
    ImageChannelIndex: NotationSymbol  # channel coordinate of an element in an image tensor ($c$)
    ImageColumnIndex: NotationSymbol  # column (width) coordinate of an element in an image tensor ($w$)
    ImageEpipole: NotationSymbol  # epipole of one view: the image of the other camera's optical center, where the baseline meets that image plane ($\mathbf{e}$) [pixels]
    ImageHeight: NotationSymbol  # number of pixel rows in an image or its segmentation mask ($H$)
    ImageRowIndex: NotationSymbol  # row (height) coordinate of an element in an image tensor ($h$)
    ImageTensorElement: NotationSymbol  # value of an image tensor at a given height, width and channel coordinate ($X$)
    ImageWidth: NotationSymbol  # number of pixel columns in an image or its segmentation mask ($W$)
    ImaginaryUnit: NotationSymbol  # imaginary unit, the square root of minus one ($j$) [dimensionless]
    ImitationLearnedPolicy: NotationSymbol  # policy learned by imitation from expert demonstrations or expert-labeled states, compared against the expert policy ($\hat{\pi}$)
    ImitationTargetCommand: NotationSymbol  # supervised imitation target command for one recorded step, chosen per interval from the logged fields or supplied by an expert ($\mathbf{u}_{\text{target}}$)
    ImmediateTaskCostScore: NotationSymbol  # normalized cost, between zero and one, of taking one action in one state of a delegated task, in the imitation-learning bounds ($C$) [dimensionless]
    ImpactInstant: NotationSymbol  # instant at which a collision occurs, the reference from which an incident timeline counts time backward ($T$) [s]
    ImpactRiseTime: NotationSymbol  # time from first contact to the first impact force peak, a quarter of the natural period of the contact spring-mass system ($t_{\text{rise}}$) [s]
    ImpactSpeed: NotationSymbol  # speed of a moving mass along the contact normal at the instant it strikes an obstacle ($v_{\text{impact}}$) [m/s]
    ImpedanceControlledPosition: NotationSymbol  # measured position of the point an impedance controller regulates ($x$) [m]
    ImpedanceControlledVelocity: NotationSymbol  # measured velocity of the point an impedance controller regulates ($\dot{x}$) [m/s]
    ImpedanceEquilibriumPosition: NotationSymbol  # virtual equilibrium position an impedance controller pulls the controlled point toward ($x_{\text{des}}$) [m]
    ImpedanceEquilibriumVelocity: NotationSymbol  # velocity of the virtual equilibrium an impedance controller tracks ($\dot{x}_{\text{des}}$) [m/s]
    ImpedanceRestoringForce: NotationSymbol  # force an impedance-controlled actuator exerts, stiffness times position error plus damping times velocity error ($F$) [N]
    ImplicitPolicyEnergyFunction: NotationSymbol  # learned scalar energy that implicit behavioral cloning assigns to an observation-action pair, low for demonstrated actions and high for counterfactual ones ($E_\theta$)
    ImportanceProposalDistribution: NotationSymbol  # proposal distribution an importance sampler draws scenarios from, reweighted by the likelihood ratio ($q$)
    ImportanceSampledFailureProbability: NotationSymbol  # importance-sampling estimate of the operational failure probability, the weighted fraction of draws that fail ($\hat{p}_F$) [dimensionless]
    ImuSampleTime: NotationSymbol  # capture time of one IMU sample ($t_{\text{IMU}}$) [s]
    InContactViableSet: NotationSymbol  # set of contact states in which both the energy-tank and the contact-force barriers are nonnegative ($\mathcal{C}_{\text{contact}}$)
    IncidenceAngle: NotationSymbol  # angle between an incoming ray and the surface normal ($\theta_{\text{inc}}$) [rad]
    IncidentOpticalPower: NotationSymbol  # optical power of the light falling on a photodiode ($P_{\text{opt}}$) [W]
    IncomingChannelCommand: NotationSymbol  # channel command of the source taking over at a bumpless transfer ($u_{\text{new}}$) [varies]
    IncomingCommandDerivative: NotationSymbol  # time derivative of the incoming source command during a bumpless transfer ($\dot{u}_{\text{new}}$) [varies]
    IncorrectCandidateTrialPassProbability: NotationSymbol  # probability that a fixed incorrect candidate passes one verification trial ($p_{b}$) [dimensionless]
    IncrementalSnapshotWriteVolume: NotationSymbol  # cumulative bytes a trajectory writes when each checkpoint after the base is an incremental delta ($D_{\text{vol,delta}}$) [bytes]
    IncumbentControlCommand: NotationSymbol  # control command the incumbent controller proposes and the enforcer admits while a candidate runs in shadow mode ($\mathbf{u}^{\text{incumbent}}$) [varies]
    IncumbentInterventionRate: NotationSymbol  # rate of enforcer or operator interventions per operating hour of the incumbent controller ($\lambda_{\text{incumbent}}$) [1/h]
    IncumbentTrackingErrorStandardDeviation: NotationSymbol  # standard deviation of the tracking error of the incumbent controller under comparable conditions ($\sigma_{\text{incumbent}}$) [m]
    IndependentAttemptCount: NotationSymbol  # number of independent attempts or runs at one task, such as the attempts a retry budget allows or the runs pass@k and pass^k score ($k$)
    IndependentEventProbability: NotationSymbol  # probability of one of several independent, identical events, such as one component failing or one worker straggling in a step ($p$) [dimensionless]
    IndependentSampleRate: NotationSymbol  # rate at which statistically independent samples of a monitored quantity occur ($f$) [Hz]
    IndexGraphNodeSet: NotationSymbol  # set of nodes, one per stored record, of a vector index graph such as HNSW ($V$)
    IndexedDocumentCount: NotationSymbol  # number of documents in the corpus a lexical index covers ($N$)
    IndicatorFunction: NotationSymbol  # function equal to one when its condition holds and zero otherwise ($\mathbb{I}$)
    InducedGateVoltage: NotationSymbol  # parasitic gate-source voltage that Miller displacement current induces across the turn-off gate resistance ($V_{\text{gs,induced}}$) [V]
    InductiveVoltage: NotationSymbol  # voltage across a coil, inductance times the rate of change of its current ($v$) [V]
    InequalityConstraintFunction: NotationSymbol  # constraint function of an optimization problem, whose nonpositive values define the feasible set ($g$) [varies]
    InertiaMatchingRatio: NotationSymbol  # gear ratio that maximizes load acceleration for a given motor torque, where reflected rotor inertia equals load inertia ($N^*$) [dimensionless]
    InertiaMatrixDerivative: NotationSymbol  # time derivative of the joint-space inertia matrix along a motion ($\dot{\mathbf{M}}$) [kg*m^2/s]
    InertialIntegrationTime: NotationSymbol  # time over which an inertial sensor's output is integrated without an external correction ($T$) [s]
    InferenceActiveTime: NotationSymbol  # time a duty-cycled device spends actively running inference in each period ($t_{\text{inference}}$) [s]
    InferenceCost: NotationSymbol  # recurring cost of serving a model in production ($C_{\text{infer}}$) [USD]
    InferenceDutyCycle: NotationSymbol  # fraction of time an accelerator actively runs inference rather than waiting in standby ($\alpha_{\text{duty}}$) [dimensionless]
    InferenceEnergy: NotationSymbol  # energy consumed to complete one inference request ($E_{\text{inf}}$) [J]
    InferenceLatency: NotationSymbol  # time the model takes to compute a request's output ($L_{\text{lat,infer}}$) [s]
    InferenceLoadSocPower: NotationSymbol  # power an application SoC dissipates with continuous proposal inference running ($P_{\text{load}}$) [W]
    InflatedHullRadius: NotationSymbol  # radius of a collision hull after inflating it by a covariance-derived margin, as a function of time ($r_{\text{inflated}}$) [m]
    InflightRolloutReservation: NotationSymbol  # staging bytes that rollouts already running can still reserve after the dispatcher stops issuing work ($M_{\text{inflight}}$) [bytes]
    InfluenceFunction: NotationSymbol  # influence function, the approximate change in a test example's loss when a training example's weight in the training loss is increased infinitesimally ($I$)
    InformationTechnologyEnergy: NotationSymbol  # energy delivered to a data center's IT equipment over a stated period ($E_{\text{IT}}$) [J]
    InformationTechnologyHeatLoad: NotationSymbol  # rate at which a facility's information technology equipment (servers, accelerators, network gear) produces heat ($\dot{Q}_{\text{IT}}$) [W]
    InformationTechnologyPower: NotationSymbol  # electrical power delivered to a data center's IT equipment: servers, accelerators, storage and networking ($P_{\text{IT}}$) [W]
    InformativeTaskFraction: NotationSymbol  # probability that a fixed number of independent attempts on one task contains at least one success and at least one failure, the informative-task fraction ($p_{\text{inform}}$) [dimensionless]
    IngestionPricePerSample: NotationSymbol  # provider price charged to ingest one telemetry sample ($C_{\text{ingest/sample}}$) [USD/sample]
    InitialAccuracy: NotationSymbol  # model accuracy at deployment, or right after the most recent retraining ($\text{Accuracy}_0$) [dimensionless]
    InitialBusVoltage: NotationSymbol  # DC bus voltage at the moment a regenerative deceleration begins ($V_{\text{initial}}$) [V]
    InitialPositionStandardDeviation: NotationSymbol  # standard deviation of the position estimate at the instant measurements stop ($\sigma_{p,0}$) [m]
    InitialPositionVelocityCovariance: NotationSymbol  # cross-covariance of the position and velocity estimates at the instant measurements stop, the off-diagonal entry of the initial covariance ($\sigma_{pv,0}$) [m^2/s]
    InitialProposalDistribution: NotationSymbol  # first proposal of a cross-entropy search, set to the operational distribution ($q_{\boldsymbol{\psi}_0}$)
    InitialStateDistribution: NotationSymbol  # distribution from which a decision process draws its starting state ($\rho_0$)
    InitialTrackingError: NotationSymbol  # position error of an estimate at the moment its evidence is captured ($e_0$) [m]
    InitialVelocityStandardDeviation: NotationSymbol  # standard deviation of the velocity error of an estimate, here at the onset of occlusion ($\sigma_{v,0}$) [m/s]
    InitiatedTasks: NotationSymbol  # number of tasks in a settled cohort, whatever their outcome ($N$)
    InjectionBandwidth: NotationSymbol  # rate at which one endpoint can inject bytes into the fabric ($\text{BW}_{\text{inject}}$) [bytes/s]
    InjectionTriggerEpoch: NotationSymbol  # monotonic test-clock epoch at which the orchestrator asserts the fault-injection trigger ($t_{\text{inject}}$) [s]
    InnerChainTransform: NotationSymbol  # homogeneous transform from frame C to frame B, the inner factor of a composed chain ($\mathbf{T}_{BC}$)
    InnerLayerFunction: NotationSymbol  # inner function of a composition, the layer applied before the outer one ($g$)
    InnerProductLeftComponent: NotationSymbol  # one component of the first vector in a dot product ($a$) [dimensionless]
    InnerProductLeftVector: NotationSymbol  # first of two vectors whose dot product measures their similarity ($\mathbf{a}$)
    InnerProductRightComponent: NotationSymbol  # one component of the second vector in a dot product ($v$) [dimensionless]
    InnerProductRightVector: NotationSymbol  # second of two vectors whose dot product measures their similarity ($\mathbf{b}$)
    InnovationBias: NotationSymbol  # systematic offset in a measurement innovation, such as the arrival-stamp travel of a misdated update ($b$) [varies]
    InnovationCovariance: NotationSymbol  # covariance of the measurement innovation ($\mathbf{S}$)
    InnovationVariance: NotationSymbol  # variance of a scalar measurement innovation ($S$) [varies]
    InputCaptureTime: NotationSymbol  # time to capture one unit of streaming input, such as an audio or video frame ($T_{\text{capture}}$) [s]
    InputChannelCount: NotationSymbol  # number of channels in the feature map a convolution layer reads ($C_{\text{in}}$)
    InputChannelIndex: NotationSymbol  # position of one input channel of a convolution layer ($c$)
    InputColumnLieDerivativeOperator: NotationSymbol  # Lie derivative operator along one column of the input matrix field, the field of a single control input ($L_{\mathbf{g}_j}$)
    InputConsumerAcceleratorCount: NotationSymbol  # number of accelerators that each consume a distinct local input batch, so devices cooperating on one batch count once ($N_{\text{GPU}}$)
    InputCoordinateCount: NotationSymbol  # number of coordinates of an input vector, such as the pixels of an image ($d$)
    InputDistribution: NotationSymbol  # probability distribution of a model's inputs, whose change is data drift ($p(x)$)
    InputEventRate: NotationSymbol  # number of relevant input changes per second that make an event-driven design fire ($r_{\text{event}}$) [1/s]
    InputFeatureIndex: NotationSymbol  # index of an input feature of a neuron or layer ($i$)
    InputFeatureVector: NotationSymbol  # input feature vector of one example, written as a row vector ($\mathbf{x}$)
    InputLieDerivativeOperator: NotationSymbol  # Lie derivative operator along the input matrix field, mapping a scalar function of the state to its sensitivity to the control input ($L_{\mathbf{g}}$)
    InputNodePrimal: NotationSymbol  # value of the input node of a computational graph that holds input i ($v_{i-N}$) [dimensionless]
    InputPipelineRate: NotationSymbol  # batches per second the host-side input pipeline supplies ($R_{\text{pipeline}}$) [batches/s]
    InputPipelineRatio: NotationSymbol  # time the input pipeline takes to produce a batch divided by the time the accelerator takes to train on it ($r_{\text{pipe}}$) [dimensionless]
    InputPipelineServiceRate: NotationSymbol  # end-to-end rate at which the input pipeline delivers data, set by its slowest stage under bounded buffering ($r_{\text{pipeline}}$) [bytes/s]
    InputPipelineStageIndex: NotationSymbol  # position of one stage in the input pipeline service chain from storage read to host-to-device transfer ($j$)
    InputQuantizationScale: NotationSymbol  # quantization scale of a layer's input activation tensor ($s_X$) [varies]
    InputResolution: NotationSymbol  # input image resolution of a convolutional model ($r$)
    InputStageServiceRate: NotationSymbol  # measured rate at which one stage of the input pipeline service chain delivers data, in output-equivalent bytes per second ($r$) [bytes/s]
    InputTokenPrice: NotationSymbol  # price of one uncached input token ($c_{\text{in}}$) [USD/token]
    InputTokenSequence: NotationSymbol  # sequence of input tokens a language model conditions on, positions one through the sequence length ($x_{1:S}$)
    InputVariableIndex: NotationSymbol  # index of one input variable of a scalar graph function ($m$)
    InsetAvailableBrakingDistance: NotationSymbol  # clear distance minus the localization bound, protective margin and tracking-error bound, the distance left for delay travel and braking ($D_{\text{avail}}$) [m]
    InstanceReplenishTime: NotationSymbol  # time to prepare one clean sandbox instance for the standby reserve, such as by snapshot restore ($T_{\text{init}}$) [s]
    InstantaneousPhaseCurrent: NotationSymbol  # signed instantaneous current in one motor phase, positive from the inverter node into the winding ($i_{\text{phase}}$) [A]
    InstantaneousRequestOccupancy: NotationSymbol  # number of requests inside a system at one instant, arrivals so far minus completions so far ($Q$) [requests]
    IntegerWeightCode: NotationSymbol  # integer code symmetric group quantization stores for one weight, its rounded and clamped ratio to the group scale ($q$) [dimensionless]
    IntegralControlEffort: NotationSymbol  # integral-action component of a PID command, the integral gain matrix times the integrator state ($\mathbf{u}_{\text{int}}$)
    IntegralGainMatrix: NotationSymbol  # matrix of integral feedback gains acting on the accumulated tracking error in joint-space PID control ($\mathbf{K}_{i}$)
    IntegratedPixelResponse: NotationSymbol  # value a pixel records, the scene radiance it receives integrated over the exposure window under a linear response ($I_{\text{pixel}}$) [varies]
    IntegrationCandidateCommitHash: NotationSymbol  # content hash of the immutable integration candidate, the worker's commit or a staged merge of it with the captured head ($h_{\text{candidate}}$)
    IntegrationDummyTime: NotationSymbol  # dummy time variable of integration inside a time integral, distinct from its limits ($t'$) [s]
    IntegrationHeadCommitHash: NotationSymbol  # content hash of the integration branch head captured for this integration attempt ($h_{\text{head}}$)
    IntegrationPoint: NotationSymbol  # point of a Euclidean space that a volume or Gaussian integral ranges over ($\mathbf{y}$)
    IntegrationPointCoordinate: NotationSymbol  # one coordinate of a point that a separable Euclidean integral ranges over ($y$) [dimensionless]
    IntegrationTimeInstant: NotationSymbol  # time instant running over an integration interval, the dummy variable when the interval's end is the time t ($s$) [s]
    IntegratorState: NotationSymbol  # internal state of a controller's integral action, the time integral of the tracking error ($\mathbf{z}$)
    IntegratorStateRate: NotationSymbol  # time derivative of the integral-action state, the tracking error plus any anti-windup correction ($\dot{\mathbf{z}}$)
    IntegratorUnwindDelay: NotationSymbol  # delay a wound-up integrator adds before the actuator command leaves its saturation rail and net braking begins ($T_{\text{unwind}}$) [s]
    IntegrityProvenance: NotationSymbol  # set of sources whose content may have influenced a record ($\iota$)
    IntegrityTrustLabel: NotationSymbol  # two-level integrity label, trusted or untrusted, that marks whether a record's dependencies are qualified for a specified use, propagated by join ($\iota$)
    IntentAcknowledgmentEvent: NotationSymbol  # event at which the log durably acknowledges an operation's recoverable intent record, and the records before it, under the declared failure domain ($e_{\text{flush}}$)
    IntentAdministrativeLease: NotationSymbol  # longest the application processor may hold an intent-lease goal without renewing it, the host-silence bound ($T_{\text{admin}}$) [s]
    IntentDemonstrationDataset: NotationSymbol  # demonstration dataset that pairs observations with the operator's mapped setpoints, recorded before the permission path filters them ($\mathcal{D}_{\text{intent}}$)
    IntentLease: NotationSymbol  # time-bounded authorization from the permission path that binds an admitted target, its expiry, and its effort ceilings ($\mathcal{I}_{\text{intent}}$)
    IntentLeaseExpiry: NotationSymbol  # instant at which an intent lease expires: the earliest of evidence decay, host silence, and the stopping deadline ($t_{\text{intent\_exp}}$) [s]
    IntentProposalPeriod: NotationSymbol  # time between successive target proposals of the intent model ($T_{\text{period}}$) [s]
    InterAnnotatorAgreement: NotationSymbol  # Cohen's kappa, agreement between two annotators corrected for the agreement expected by chance ($\kappa$) [dimensionless]
    InterFeedInterval: NotationSymbol  # time between consecutive watchdog feeds when the controller feeds once after each successful tick ($T_{\text{feed}}$) [s]
    InterGroupColumnDelay: NotationSymbol  # minimum spacing between consecutive column commands to different DRAM bank groups ($t_{\text{CCD\_S}}$) [s]
    InterKernelGapTime: NotationSymbol  # time the device sits idle between consecutive kernels on a profiler timeline ($T_{\text{gap}}$) [s]
    InterNodeContextParallelDegree: NotationSymbol  # number of nodes a sequence is split across by ring attention in hierarchical context parallelism ($c_{\text{inter}}$)
    InterQueryIntervalDeviation: NotationSymbol  # standard deviation of the time between one account's successive API queries, a timing-regularity feature for extraction detection ($\sigma_{\text{inter-query}}$)
    InterSensorClockSkew: NotationSymbol  # uncorrected time offset between the clocks of two sensors whose exposures are fused as if simultaneous ($\Delta t_{\text{skew}}$) [s]
    InteractionHistory: NotationSymbol  # transcript of delivered prompts, emitted actions and observations up to a turn of a rollout, the history a policy conditions on ($h$)
    InteractionPortForce: NotationSymbol  # force at a physical interaction port, signed so that this force times the port velocity is the power entering the system ($F$) [N]
    InteractionPortVelocity: NotationSymbol  # velocity at a physical interaction port, signed so that the port force times this velocity is the power entering the system ($v$) [m/s]
    InterarrivalVariationCoefficient: NotationSymbol  # coefficient of variation of the interarrival times of a renewal arrival stream, their standard deviation divided by their mean ($c_{a}$) [dimensionless]
    InterfaceMaterialThermalResistance: NotationSymbol  # thermal resistance through the thermal interface material between a package and its heat path ($R_{\text{th,TIM}}$) [K/W]
    InterferenceSlowdownFactor: NotationSymbol  # factor by which cross-workload interference lengthens the memory-stall share of a victim task ($r_{\text{slow}}$) [dimensionless]
    InterframeSpacingTime: NotationSymbol  # minimum idle spacing that must follow frame i on the bus ($t_{\text{interframe},i}$) [s]
    InterleavedPipelineBubbleFraction: NotationSymbol  # fraction of pipeline device time spent idle when each device runs several interleaved virtual stages ($b_{\text{interleaved}}$) [dimensionless]
    IntermediateTensorSize: NotationSymbol  # bytes in the intermediate tensor that a sequence of operations passes between its steps ($M_{\text{tensor}}$) [bytes]
    InternalDissipationPower: NotationSymbol  # rate at which a passive system dissipates energy internally, through damping, friction or resistance, as a non-negative function of the state ($g$) [W]
    IntervalCompletionTime: NotationSymbol  # elapsed time, failed attempts included, until one compute-plus-checkpoint interval completes ($T_{\text{total}}$) [s]
    IntervalMean: NotationSymbol  # mean of an interarrival or service time in a queueing model ($\bar{T}$) [s]
    IntervalMeanOccupancy: NotationSymbol  # average number of requests inside a system over a finite observation interval ($\bar{Q}$) [requests]
    IntervalNormalQuantile: NotationSymbol  # positive standard-normal quantile that sets the width of a two-sided confidence interval at its stated nominal confidence, written bare inside interval formulas ($z$) [dimensionless]
    IntervalStandardDeviation: NotationSymbol  # standard deviation of an interarrival or service time in a queueing model ($\sigma$) [s]
    IntervalTransitionMatrix: NotationSymbol  # state transition matrix across an interval of estimator steps, the product of the one-step transition matrices ($\boldsymbol{\Phi}$)
    IntervalVariationCoefficient: NotationSymbol  # coefficient of variation of an interarrival or service time, its standard deviation divided by its mean ($c$) [dimensionless]
    InterventionDensity: NotationSymbol  # number of interventions per unit of exposure, such as per kilometer traveled or per thousand control steps ($D_{\text{int}}$) [varies]
    InterventionRate: NotationSymbol  # rate of recorded interventions per hour of active autonomous operation, total interventions over total active time ($\lambda_{\text{int}}$) [1/h]
    InterventionWindowDuration: NotationSymbol  # duration of the rolling window over which the permission path sums the intervention magnitude ($W$) [s]
    IntraClusterCorrelation: NotationSymbol  # correlation of outcomes between users in the same randomization cluster ($\rho_{\text{ICC}}$) [dimensionless]
    IntraGroupColumnDelay: NotationSymbol  # minimum spacing between consecutive column commands to the same DRAM bank group ($t_{\text{CCD\_L}}$) [s]
    IntraNodeContextParallelDegree: NotationSymbol  # number of devices inside one node that split attention heads by an all-to-all transpose in hierarchical context parallelism ($c_{\text{intra}}$)
    IntrusionDistance: NotationSymbol  # distance a body part can intrude past the sensing field before detection ($C$) [m]
    IntrusionDistanceAllowance: NotationSymbol  # distance a body part may reach past a safeguard before detection, added in the ISO 13855 separation formula ($C$) [m]
    InvalidatedArtifactNode: NotationSymbol  # artifact node of a candidate plan's dependency graph whose supporting evidence fails, the root of a revision traversal ($v$)
    InverseDepth: NotationSymbol  # reciprocal of a point's depth along the optical axis, the parameterization that lets a landmark at infinity enter an estimator at zero ($\rho$) [1/m]
    InverseDepthLandmarkParameters: NotationSymbol  # six-component parameter vector of a visual landmark in inverse-depth form: optical center at first observation, azimuth, elevation and inverse depth ($\mathbf{y}$)
    InverseKinematicsDamping: NotationSymbol  # damping length that damped-least-squares inverse kinematics adds to bound the joint step near a singularity ($\lambda$) [varies]
    InverseSensorLogOdds: NotationSymbol  # log-odds of occupancy given only the current observation, the evidence the inverse sensor model adds to a voxel ($L_{\text{sensor}}$) [dimensionless]
    InverterCurrentLimit: NotationSymbol  # largest current an inverter bridge lets flow, set to protect its transistors and the stator ($I_{\max}$) [A]
    InverterDeadTime: NotationSymbol  # dead time a PWM timer inserts between turning off one switch of an inverter half-bridge and turning on its complement, so the two never conduct together ($t_{\text{dead}}$) [s]
    IoTransferTime: NotationSymbol  # time to move a workload's data over a storage or network link, the input/output stage of a pipelined step ($T_{\text{IO}}$) [s]
    IronCaseThermalResistance: NotationSymbol  # thermal resistance from the stator iron node to the motor frame and housing node ($R_{\text{th,iron-case}}$) [K/W]
    IronLawComputeTime: NotationSymbol  # time spent computing in the iron law, operations divided by the sustained compute rate ($T_{\text{compute}}$) [s]
    IronLawMemoryTime: NotationSymbol  # time spent moving data in the iron law, bytes moved divided by bandwidth ($T_{\text{memory}}$) [s]
    IronLossPower: NotationSymbol  # eddy-current and hysteresis loss power dissipated in the stator iron laminations ($P_{\text{iron}}$) [W]
    IronThermalCapacitance: NotationSymbol  # heat capacity of the stator iron lamination node in a multi-node motor thermal network ($C_{\text{iron}}$) [J/K]
    IrreducibleLoss: NotationSymbol  # loss floor that adding parameters, data or compute does not remove within the fitted scaling regime ($C_{\mathcal{L}}$) [dimensionless]
    IsolatedEnforcerExecutionTime: NotationSymbol  # execution time of the permission path running with zero contention from proposal-side workloads ($t_{\text{iso}}$) [s]
    IsolatedJobRunTime: NotationSymbol  # running time of one job on its own allocation with no contention from other jobs ($T$) [s]
    IspProcessingTime: NotationSymbol  # time the image signal processor spends debayering and rectifying a frame ($t_{\text{ISP}}$) [s]
    IsrExecutionTime: NotationSymbol  # execution time of one interrupt service routine, indexed by interrupt source ($t_{\text{ISR}}$) [s]
    IterationCost: NotationSymbol  # cost of experimentation and maintenance after a model's initial training ($C_{\text{iter}}$) [USD]
    IterationTokenBudget: NotationSymbol  # largest number of tokens, decode and prefill together, a hybrid scheduler admits into one iteration ($N_{\text{budget}}$)
    JaccardSimilarity: NotationSymbol  # Jaccard similarity of two sets, the size of their intersection divided by the size of their union ($J$) [dimensionless]
    JacobianLeftSingularMatrix: NotationSymbol  # orthogonal matrix whose columns are the left singular vectors of the manipulator Jacobian, the task-space directions ($\mathbf{U}$)
    JacobianLeftSingularVector: NotationSymbol  # left singular vector of the manipulator Jacobian, a unit task-space direction ($\mathbf{u}$)
    JacobianPseudoinverse: NotationSymbol  # Moore-Penrose pseudoinverse of the manipulator Jacobian, giving the minimum-norm joint velocity for a tool velocity ($\mathbf{J}^\dagger$)
    JacobianRightSingularMatrix: NotationSymbol  # orthogonal matrix whose columns are the right singular vectors of the manipulator Jacobian, the joint-space directions ($\mathbf{V}$)
    JacobianRightSingularVector: NotationSymbol  # right singular vector of the manipulator Jacobian, a unit joint-space direction ($\mathbf{v}$)
    JacobianSingularNumber: NotationSymbol  # singular value of the manipulator Jacobian, the tool speed a unit joint rate produces along one singular direction ($\sigma$) [varies]
    JacobianSingularNumberMatrix: NotationSymbol  # rectangular diagonal matrix of the singular values of the manipulator Jacobian ($\boldsymbol{\Sigma}$) [varies]
    JawAperture: NotationSymbol  # opening between the gripper jaws that a grasp affordance requires ($w_{\text{aperture}}$) [m]
    JensenShannonDivergence: NotationSymbol  # Jensen-Shannon divergence between two probability distributions, a symmetric bounded measure of how far they differ ($\mathcal{D}_{\text{JS}}$)
    JensenShannonMixtureDistribution: NotationSymbol  # equal-weight average of the current and training distributions, the reference against which Jensen-Shannon divergence measures each ($\bar{P}$)
    JerkLimit: NotationSymbol  # largest rate of change of acceleration a planned or stopping profile may command ($j_{\max}$) [m/s^3]
    JerkWeightMatrix: NotationSymbol  # positive semidefinite matrix weighting joint jerk in the regularization term of a trajectory projection ($\mathbf{W}_j$) [varies]
    JitterBufferPlayoutDelay: NotationSymbol  # fixed delay by which a time-sorted jitter buffer's playhead trails real time ($T_{\text{buf}}$) [s]
    JobAbsoluteDeadline: NotationSymbol  # absolute deadline of one job of a real-time task, its release time plus the task's relative deadline ($d$) [s]
    JobAvailability: NotationSymbol  # probability that every component a job needs is up at the same instant ($A_{\text{all}}$) [dimensionless]
    JobDuration: NotationSymbol  # wall-clock time one job or run spends executing, from start to finish ($T_{\text{job}}$) [s]
    JobIndex: NotationSymbol  # index of one job in a scheduler's workload ($j$)
    JobQueueWaitTime: NotationSymbol  # time one job waits in the scheduler's queue before it starts running ($W$) [s]
    JobReleaseTime: NotationSymbol  # instant at which one job of a periodic real-time task is released ($r$) [s]
    JobRestartTime: NotationSymbol  # time for the scheduler to allocate resources and launch replacement processes after a failure ($T_{\text{restart}}$) [s]
    JobSlowdown: NotationSymbol  # ratio of a job's queueing-plus-running time to its isolated runtime ($S$) [dimensionless]
    JobStartupTime: NotationSymbol  # time from a job's complete allocation to its first retained step, spent loading, initializing and compiling ($T_{\text{startup}}$) [s]
    JointAcceleration: NotationSymbol  # angular acceleration of one joint ($\ddot{q}$) [varies]
    JointAngle: NotationSymbol  # angle of one joint ($\theta_{\text{joint}}$) [rad]
    JointAngleLowerLimit: NotationSymbol  # lowest angle a joint may be commanded to, its lower reach limit ($q_{\min}$) [rad]
    JointAngleUpperLimit: NotationSymbol  # highest angle a joint may be commanded to, its upper reach limit ($q_{\max}$) [rad]
    JointAngularAcceleration: NotationSymbol  # angular acceleration of a single arm joint ($\dot{\omega}$) [rad/s^2]
    JointAngularRate: NotationSymbol  # angular rate of one revolute joint ($\omega$) [rad/s]
    JointAngularSpeed: NotationSymbol  # angular speed of a single arm joint ($\omega$) [rad/s]
    JointBaseInertiaCoupling: NotationSymbol  # block of the generalized inertia matrix through which floating-base accelerations load the actuated-joint equations ($\mathbf{M}_{jb}$) [varies]
    JointBiasTorque: NotationSymbol  # steady gravitational or dynamic torque acting on a joint, signed along its motion, that braking must overcome ($\tau_{\text{bias}}$) [N*m]
    JointCoordinate: NotationSymbol  # one entry of the generalized coordinates, a single joint's angle or extension ($q$) [varies]
    JointCount: NotationSymbol  # number of joints of a manipulator, the length of its joint-position vector ($n$)
    JointFrictionTorque: NotationSymbol  # total friction torque of a joint referred to its output link ($\tau_{\text{fric}}$) [N*m]
    JointIndex: NotationSymbol  # position of one joint axis of a manipulator, from one to the number of joints ($i$)
    JointJerk: NotationSymbol  # jerk of one joint's trajectory, the third time derivative of its position ($j_{\text{joint}}$) [rad/s^3]
    JointLoadTorque: NotationSymbol  # joint-side torque from link and payload inertia and external loads, apart from the reflected rotor inertia, in a single-joint model ($\tau_{\text{load}}$) [N*m]
    JointOutputImpedance: NotationSymbol  # mechanical impedance a joint presents at its output flange, external torque over joint velocity ($Z_{\text{out}}$) [N*m*s/rad]
    JointOutputTorque: NotationSymbol  # torque a joint delivers at the transmission output ($\tau_{\text{out}}$) [N*m]
    JointPathCurvature: NotationSymbol  # second derivative of a joint-space path with respect to its path coordinate ($\mathbf{q}''$)
    JointPathTangent: NotationSymbol  # derivative of a joint-space path with respect to its path coordinate, the path's tangent in joint space ($\mathbf{q}'$)
    JointPosition: NotationSymbol  # position of a single joint along its axis, an angle for a revolute joint, one component of the generalized coordinates ($q$) [rad]
    JointPositionLowerLimit: NotationSymbol  # per-joint lower bounds on the generalized coordinates ($\mathbf{q}_{\min}$) [varies]
    JointPositionSetpoint: NotationSymbol  # setpoint joint positions a joint-level PID loop tracks, changed in steps by a planner or policy handover ($\mathbf{q}^*$)
    JointPositionTarget: NotationSymbol  # joint position a command interface asks the drive's position loop to reach ($q_{\text{des}}$) [varies]
    JointPositionUpperLimit: NotationSymbol  # per-joint upper bounds on the generalized coordinates ($\mathbf{q}_{\max}$) [varies]
    JointReflectedRotorInertia: NotationSymbol  # diagonal entry of the reflected rotor inertia matrix: one joint's rotor inertia times the square of its gear ratio ($J_{m}$) [kg*m^2]
    JointSampleTimestamp: NotationSymbol  # timestamp of the joint-angle sample paired with a camera exposure ($t_{\text{joint}}$) [s]
    JointTorque: NotationSymbol  # vector of torques the actuators apply at the joints ($\boldsymbol{\tau}$) [N*m]
    JointTorqueConstant: NotationSymbol  # output torque a joint delivers per ampere of its motor's torque-producing current, including gear ratio and transmission efficiency ($K_t$) [N*m/A]
    JointTorqueLimit: NotationSymbol  # largest torque the actuator of one joint can deliver, indexed by joint ($\tau_{\max}$) [N*m]
    JointTorqueLimits: NotationSymbol  # vector of the largest torques each joint actuator may deliver, its saturation limits ($\boldsymbol{\tau}_{\max}$) [N*m]
    JointTorqueLowerLimit: NotationSymbol  # per-joint lower bounds on the torque the actuators can deliver ($\boldsymbol{\tau}_{\min}$)
    JointTorqueSlew: NotationSymbol  # time derivative of the joint torque vector, the torque slew ($\dot{\boldsymbol{\tau}}$) [N*m/s]
    JointTorqueSlewLimit: NotationSymbol  # per-joint bounds on the magnitude of the joint torque slew ($\dot{\boldsymbol{\tau}}_{\max}$) [N*m/s]
    JointTorqueTarget: NotationSymbol  # joint torque a command interface asks the drive to produce, delivered as a current reference ($\tau_{\text{des}}$) [N*m]
    JointTorqueUpperLimit: NotationSymbol  # per-joint upper bounds on the torque the actuators can deliver ($\boldsymbol{\tau}_{\max}$) [N*m]
    JointTorsionalStiffnessMatrix: NotationSymbol  # diagonal matrix of the torsional stiffnesses of elastic joint transmissions, mapping motor-link angle difference to elastic torque ($\mathbf{K}$) [N*m/rad]
    JointTrackingError: NotationSymbol  # tracking error vector between the desired and the measured joint positions ($\mathbf{e}$)
    JointVelocity: NotationSymbol  # angular velocity of one joint ($\dot{q}$) [varies]
    JointVelocityCeiling: NotationSymbol  # largest joint velocity the enforcer admits, the certified velocity ceiling ($\dot{q}_{\max}$) [rad/s]
    JointVelocityLimit: NotationSymbol  # per-joint bounds on the magnitude of the generalized velocities ($\dot{\mathbf{q}}_{\max}$) [varies]
    JointVelocitySetpoint: NotationSymbol  # time derivative of the joint-position setpoint, impulsive when the setpoint steps ($\dot{\mathbf{q}}^*$)
    JointVelocityTarget: NotationSymbol  # joint velocity a command interface asks the drive's velocity loop to track ($\dot{q}_{\text{des}}$) [varies]
    JouleanTemperatureRise: NotationSymbol  # temperature rise of a winding above its reference caused by resistive heating ($\Delta T$) [K]
    JunctionAmbientThermalResistance: NotationSymbol  # lumped thermal resistance from a silicon junction to the ambient air, the sum of the junction-to-case, interface-material and chassis paths ($R_{\text{th,JA}}$) [K/W]
    JunctionCaseThermalResistance: NotationSymbol  # thermal resistance from a silicon junction to the case of its package ($R_{\text{th,JC}}$) [K/W]
    JunctionTemperature: NotationSymbol  # temperature of a silicon die at its transistor junctions ($T_{\text{j}}$) [degC]
    KalmanBucyObserverGain: NotationSymbol  # gain of a continuous-time state observer that weights the innovation, optimal for the Kalman-Bucy filter ($\mathbf{L}$)
    KalmanGain: NotationSymbol  # matrix weighting the measurement innovation in the Kalman filter state update ($\mathbf{G}$)
    KeepContextPathCost: NotationSymbol  # remaining token cost of a trajectory that keeps its full uncompacted context, as a function of the remaining call count ($C_{\text{keep}}$) [USD]
    KendallMarkovianCode: NotationSymbol  # Kendall-notation code marking a Markovian, memoryless arrival or service process in a queue model name such as M/M/1 ($M$)
    KernelActiveTime: NotationSymbol  # time a resource spends executing useful kernels, the union of its kernel-active intervals ($T_{\text{compute}}$) [s]
    KernelColumnOffset: NotationSymbol  # column offset of one position inside a convolution kernel window ($n$)
    KernelExecutionTime: NotationSymbol  # time a kernel spends executing on the device, as measured on a profiler timeline ($T_{\text{kernel}}$) [s]
    KernelHeight: NotationSymbol  # height of a convolution kernel, in input positions ($K_h$)
    KernelHorizontalLoopOffset: NotationSymbol  # horizontal kernel offset a direct-convolution loop nest iterates over ($k_{x}$)
    KernelRowOffset: NotationSymbol  # row offset of one position inside a convolution kernel window ($m$)
    KernelSize: NotationSymbol  # side length, in input positions, of a square convolution kernel ($K$)
    KernelVerticalLoopOffset: NotationSymbol  # vertical kernel offset a direct-convolution loop nest iterates over ($k_{y}$)
    KernelWeight: NotationSymbol  # one convolution kernel weight, connecting one input channel at one kernel offset to one output channel ($W$) [dimensionless]
    KernelWidth: NotationSymbol  # width of a convolution kernel, in input positions ($K_w$)
    KeyAxisLabel: NotationSymbol  # einsum index label naming the key-position axis of an attention tensor ($j$)
    KeyBufferPosition: NotationSymbol  # position of the key token within a packed attention buffer ($j$)
    KeyComponent: NotationSymbol  # one component of one token's key vector ($k_{jm}$) [dimensionless]
    KeyLeftSingularMatrix: NotationSymbol  # orthonormal matrix whose columns are the left singular vectors of a key activation matrix ($\mathbf{U}$)
    KeyLeftSingularVector: NotationSymbol  # one left singular vector of a key activation matrix, a column of its left singular matrix ($\mathbf{u}$)
    KeyProjectionMatrix: NotationSymbol  # learned matrix that projects token representations into keys ($\mathbf{W}_{K}$)
    KeyRightSingularMatrix: NotationSymbol  # orthonormal matrix whose columns are the right singular vectors of a key activation matrix ($\mathbf{V}$)
    KeyRightSingularVector: NotationSymbol  # one right singular vector of a key activation matrix, a column of its right singular matrix ($\mathbf{v}$)
    KeySingularNumber: NotationSymbol  # one singular value of a key activation matrix, indexed in decreasing order ($\sigma$) [dimensionless]
    KeySingularSpectrumMatrix: NotationSymbol  # rectangular diagonal matrix holding the singular values of a key activation matrix in decreasing order ($\mathbf{\Sigma}$)
    KeySourceMatrix: NotationSymbol  # representations attention draws its keys from, before a head-specific key projection ($\mathbf{X}_{K}$)
    KeySpectrumLength: NotationSymbol  # number of singular values of a key activation matrix, the smaller of its sequence length and head dimension ($q$)
    KeyVector: NotationSymbol  # key vector of one token, one row of the key matrix ($\mathbf{k}$)
    KineticEnergy: NotationSymbol  # kinetic energy a moving body carries, one half its mass times its speed squared (plus rotational terms where stated) ($E_k$) [J]
    KineticEnergyCeiling: NotationSymbol  # largest kinetic energy an operational containment allows the machine under fault ($E_{\max}$) [J]
    KineticInsetBarrierFunction: NotationSymbol  # barrier certificate shrunk by the stopping distance at the current speed, so its relative degree is one ($h_{\text{kinetic}}$) [m]
    KineticInsetBarrierRate: NotationSymbol  # time derivative of the kinetic stopping inset barrier along the dynamics ($\dot{h}_{\text{kinetic}}$) [m/s]
    KlDivergence: NotationSymbol  # Kullback-Leibler divergence, the expected extra code length from encoding one distribution with a code built for another ($\mathcal{D}_{\text{KL}}$)
    KlPenaltyCoefficient: NotationSymbol  # weight on the Kullback-Leibler divergence from the reference model that RLHF subtracts from the reward ($\beta_{\text{KL}}$) [dimensionless]
    KlSampleStatistic: NotationSymbol  # nonnegative per-token statistic, the reference-to-current probability ratio minus its log minus one, whose expectation under the current distribution is the KL divergence ($d$) [dimensionless]
    KnownComponentStopDistance: NotationSymbol  # stopping travel the known components of the stopping budget add up to at a given speed on a dry floor, delay travel, braking, jerk ramps and fixed overheads ($d_{\text{budget}}$) [m]
    KolmogorovSmirnovCriticalThreshold: NotationSymbol  # critical threshold that the two-sample Kolmogorov-Smirnov statistic must exceed to reject equal distributions at a given significance level ($c_{\text{crit}}$) [dimensionless]
    KolmogorovSmirnovStatistic: NotationSymbol  # two-sample Kolmogorov-Smirnov statistic, the largest absolute difference between two empirical cumulative distribution functions ($\mathcal{D}_{\text{KS}}$) [dimensionless]
    KroneckerIndicator: NotationSymbol  # Kronecker delta, one when its two indices are equal and zero otherwise ($\delta$) [dimensionless]
    KrumClientCount: NotationSymbol  # total number of client updates a Krum-style robust aggregation rule selects among ($n$)
    KrylovSubspace: NotationSymbol  # Krylov subspace of a matrix and a starting subspace, the span of the subspace and its images under successive powers of the matrix ($\mathcal{K}$)
    KsCriticalCoefficient: NotationSymbol  # asymptotic coefficient of the two-sample Kolmogorov-Smirnov rejection rule, a function of the significance level alone ($c$)
    KvBlockSize: NotationSymbol  # number of token positions in one fixed block of a paged KV cache ($S_{\text{blk}}$)
    KvBlockTailWaste: NotationSymbol  # number of unused token positions in the last block of one sequence under paged KV cache allocation ($\Delta_{\text{tail}}$)
    KvCacheBytesPerToken: NotationSymbol  # bytes of key-value cache stored for each token held in context ($m_{\text{token}}$) [bytes/token]
    KvCacheHandoffTime: NotationSymbol  # time to transfer one request's KV cache from a prefill worker to a decode worker in a disaggregated deployment ($T_{\text{handoff}}$) [s]
    KvCacheMemory: NotationSymbol  # memory the key-value cache occupies during autoregressive inference ($M_{\text{KV}}$) [bytes]
    KvElementStorageSize: NotationSymbol  # bytes needed to store one key or value element of the KV cache at its storage precision ($s_{\text{elem,KV}}$) [bytes/element]
    KvHeadCount: NotationSymbol  # number of key-value heads in grouped-query or multi-query attention ($H_{\text{KV}}$)
    KvPoolHeadroomFraction: NotationSymbol  # fraction of the KV pool a capacity plan withholds from allocation to absorb bursty activations and fragmentation ($f_{\text{head}}$) [dimensionless]
    KvPoolMemory: NotationSymbol  # accelerator memory of a serving node left for KV cache after model weights and the runtime reserve are allocated ($M_{\text{pool}}$) [bytes]
    KvRecomputeTime: NotationSymbol  # time to rebuild a paused sequence's KV state by running prefill over its retained tokens again ($T_{\text{recompute}}$) [s]
    KvReloadTime: NotationSymbol  # time to move a paused sequence's KV state back from host memory to accelerator memory over the host links ($T_{\text{reload}}$) [s]
    KvTileCount: NotationSymbol  # number of key-value tiles the sequence is split into in tiled attention ($T_c$)
    KvTileIndex: NotationSymbol  # index of one key-value tile in tiled attention ($j$)
    KvTileSize: NotationSymbol  # number of key and value rows in one tile of tiled attention ($B_c$)
    LaSalleInvariantSet: NotationSymbol  # compact positively invariant set of states on which the Lyapunov derivative is nonpositive, the set LaSalle's principle starts from ($\Omega$)
    LabelCategoryIndex: NotationSymbol  # position of one label category among a classifier's output classes ($i$)
    LabelVector: NotationSymbol  # true label of one example written as a vector, such as a one-hot class vector ($\mathbf{y}$)
    LabeledRecord: NotationSymbol  # record that carries integrity-provenance and confidentiality labels, such as a context item, a derived value or a proposal argument ($x$)
    LabeledSampleCount: NotationSymbol  # number of samples in a dataset that carry labels ($D_{\text{labeled}}$)
    LabelingRoundIndex: NotationSymbol  # position of one on-policy data-collection and retraining round ($k$)
    LaggedModelInput: NotationSymbol  # model input features as they existed a lag before the current time, as point-in-time retrieval reconstructs them for training ($x_{t-\Delta}$)
    LaggedStateDigest: NotationSymbol  # digest of the canonicalized workspace projection captured a recurrence period before the current observation step ($h_{\text{state},t-k}$)
    LandmarkDepth: NotationSymbol  # distance of a landmark along its bearing ray from the optical center where it was first observed ($d$) [m]
    LanguageInstruction: NotationSymbol  # high-level natural-language instruction a language-grounded planner must carry out ($c_{\text{instr}}$)
    LanguageSkillPrior: NotationSymbol  # probability a language model assigns that a skill contributes to completing an instruction, the semantic prior of a language-grounded planner ($p_{\text{LLM}}$)
    LaplaceLikelihoodScale: NotationSymbol  # scale parameter of the isotropic Laplace likelihood a decoder assigns to each action-chunk entry, which turns maximum likelihood into an l1 loss ($b$) [varies]
    LaplaceNoiseSample: NotationSymbol  # value of a zero-mean Laplace noise variable at which its density is evaluated ($x$) [varies]
    LaplaceNoiseScale: NotationSymbol  # scale parameter of the Laplace noise a differentially private mechanism adds, the query sensitivity divided by the privacy budget ($b$) [varies]
    LaplaceVariable: NotationSymbol  # complex frequency variable of the Laplace transform, standing for differentiation in time ($s$) [1/s]
    LargestDrivenWheelLoadShare: NotationSymbol  # largest fraction of the machine's loaded weight carried by a single driven wheel, which sets the friction a credible deceleration needs ($\varphi$) [dimensionless]
    LargestEigenvalue: NotationSymbol  # largest eigenvalue of a symmetric matrix, applied to that matrix ($\lambda_{\max}$) [varies]
    LargestInvariantSet: NotationSymbol  # largest invariant set contained in the states where the Lyapunov derivative is zero, to which LaSalle's principle says trajectories converge ($\mathcal{M}$)
    LargestSingularNumber: NotationSymbol  # largest singular value of a matrix, its spectral norm ($\sigma_{\max}$) [dimensionless]
    LarsGlobalLearningRate: NotationSymbol  # global learning rate that layer-wise adaptive rate scaling multiplies by each layer's trust ratio ($\eta_{\text{global}}$) [dimensionless]
    LastCandidateEmbeddingVector: NotationSymbol  # last stored candidate vector of a nearest-neighbor collection, whose index is the collection size ($\mathbf{x}_N$)
    LastFunctionInputComponent: NotationSymbol  # last component of the input vector of a scalar graph function ($x_N$) [dimensionless]
    LastMeasurementTime: NotationSymbol  # capture time of the most recent measurement that updated a track or map cell ($t_{\text{last}}$) [s]
    LastRankComputeTime: NotationSymbol  # local compute time of the last rank (rank N-1) in one synchronous training step ($T_{\text{rank}_{N-1}}$) [s]
    LatchApproachSpeed: NotationSymbol  # speed at which the arm presses into the cage-door latch plate ($v_{\text{latch}}$) [m/s]
    LatchClearance: NotationSymbol  # lateral clearance of the latch that a paired observation-action displacement must stay within ($c$) [m]
    LatchContactForce: NotationSymbol  # magnitude of the contact force between a gripper and the latch, measured at one control step ($f$) [N]
    LatchForceLimit: NotationSymbol  # structural force limit of the door latch the arm pushes against ($F_{\text{lim}}$) [N]
    LatchPlateStiffness: NotationSymbol  # spring stiffness of the cage-door latch plate the arm compresses ($k_{\text{latch}}$) [N/m]
    LatchStallForce: NotationSymbol  # contact force the latch plate develops when the arm keeps pressing during an unbraked stall ($F$) [N]
    LatchStrengthFloor: NotationSymbol  # declared lower bound on a latch's structural yield strength, the force the interaction must not exceed ($F_{\text{floor}}$) [N]
    LateHandoffProbability: NotationSymbol  # probability that one plan handoff exceeds a stated latency budget, such as the p99 budget ($p$) [dimensionless]
    Latency: NotationSymbol  # fixed overhead time a step pays regardless of bytes moved or operations executed, such as kernel launch, dispatch or network round trip ($L_{\text{lat}}$) [s]
    LatencyBandwidthCrossoverSize: NotationSymbol  # message size at which per-message startup time equals transfer time in the alpha-beta model ($n^*$) [bytes]
    LatencyBoundedBatchSize: NotationSymbol  # largest batch size whose half-window wait plus batch service time fits the batching share of the latency objective ($B_{\text{SLO}}$)
    LatencyObjective: NotationSymbol  # latency dimension of a service's SLO vector: its indicator, target, window and owner ($S_{\text{latency}}$)
    LatencyQuantileFunction: NotationSymbol  # quantile of a latency distribution at percentile p ($Q_p$) [s]
    LatencyRewardExponent: NotationSymbol  # exponent on the latency ratio in a hardware-aware search reward, setting how hard missing the latency target is penalized ($\beta_{\text{lat}}$) [dimensionless]
    LatencySlo: NotationSymbol  # latency target a serving system's service-level objective sets for a request ($L_{\text{lat,SLO}}$) [s]
    LatentActionComponent: NotationSymbol  # one component of the unbounded latent action of a squashed-Gaussian policy, before tanh squashing ($u$) [dimensionless]
    LatentComponentMean: NotationSymbol  # mean of one coordinate of the diagonal Gaussian latent posterior a conditional VAE encoder outputs ($\mu_{\phi, j}$) [dimensionless]
    LatentComponentStandardDeviation: NotationSymbol  # standard deviation of one coordinate of the diagonal Gaussian latent posterior a conditional VAE encoder outputs ($\sigma_{\phi, j}$) [dimensionless]
    LatentDimension: NotationSymbol  # number of coordinates of the latent style variable of a conditional VAE ($d_z$)
    LatentGaussianPolicyDensity: NotationSymbol  # Gaussian density that a squashed-Gaussian policy assigns to an unbounded latent action given the state, before the tanh change of variables ($\mu$)
    LatentKlWeight: NotationSymbol  # weight on the KL divergence between the latent posterior and prior in a conditional variational autoencoder objective ($\beta$) [dimensionless]
    LatentPosteriorDistribution: NotationSymbol  # distribution over the latent style variable that a conditional VAE encoder infers from a demonstration chunk and its observation ($q$)
    LatentPosteriorMean: NotationSymbol  # mean of the diagonal Gaussian latent posterior that a conditional VAE encoder outputs for an observation and demonstration chunk ($\boldsymbol{\mu}_\phi$)
    LatentPosteriorStandardDeviation: NotationSymbol  # per-coordinate standard deviations of the diagonal Gaussian latent posterior that a conditional VAE encoder outputs ($\boldsymbol{\sigma}_\phi$)
    LatentStateEncoder: NotationSymbol  # learned encoder, with parameters phi, that maps sensor observations to a latent state estimate ($q_\phi$)
    LatentStyleVariable: NotationSymbol  # latent variable of a conditional VAE policy that captures the demonstration style ($\mathbf{z}$)
    LatentVariableSpace: NotationSymbol  # space of values the latent style variable of a conditional VAE can take, the domain its marginal likelihood integrates over ($\mathcal{Z}$)
    LaterBaseFlangeTransform: NotationSymbol  # base-to-flange forward-kinematics transform at the later of two hand-eye calibration poses ($\mathbf{T}_{\text{base}, k+1}^{\text{flange}}$)
    LaterCameraTargetTransform: NotationSymbol  # camera-to-target transform at the later of two hand-eye calibration poses ($\mathbf{T}_{\text{cam}, k+1}^{\text{target}}$)
    LaterRolloutTurnIndex: NotationSymbol  # position of the later turn of a pair of reinforcement-learning rollout turns compared in a covariance ($j$)
    LaterStepIndex: NotationSymbol  # position of a later decision step summed over in a return, from the current step to the end of the episode ($j$)
    LateralClearance: NotationSymbol  # lateral clearance between a mobile base and the obstacles on either side of its lane ($W_{\text{clear}}$) [m]
    LayerAllReduceTime: NotationSymbol  # time to AllReduce the gradients of one layer ($T_{\text{layer comm}}$) [s]
    LayerBackwardTime: NotationSymbol  # backward compute time of one layer, the window in which the previous layer's gradient AllReduce can hide ($T_{\text{layer backward}}$) [s]
    LayerCount: NotationSymbol  # number of layers in a network ($N_L$)
    LayerExposedCommunicationTime: NotationSymbol  # amount by which one layer's AllReduce overruns the backward computation it overlaps, left on the critical path ($T_{\text{exposed}}$) [s]
    LayerFunction: NotationSymbol  # function one layer, operator or recurrent cell computes, as a whole map from its inputs to its output ($f$)
    LayerIndex: NotationSymbol  # position of one layer within a network, counted from the input ($\ell$)
    LayerInputComponent: NotationSymbol  # one component, or feature, of a layer's input vector ($x$) [dimensionless]
    LayerInputMatrix: NotationSymbol  # input to a layer in matrix form, one row per example or token, or a single-channel image ($\mathbf{X}$)
    LayerInputVector: NotationSymbol  # vector of activations one layer receives as input ($\mathbf{x}$)
    LayerInputWidth: NotationSymbol  # number of input features, or input channels, of one layer ($d_{\text{in}}$)
    LayerIntermediateStateMemory: NotationSymbol  # bytes of saved intermediate state, other than activation outputs, one example leaves at one layer for the backward pass ($s_{\ell}$) [bytes]
    LayerNonlinearity: NotationSymbol  # elementwise nonlinearity written inside a convolution or recurrent layer's update equation, kept apart from the whole-layer map f ($\phi$)
    LayerNormMean: NotationSymbol  # mean over the features of one input vector, as layer normalization computes it ($\mu_{\text{LN}}$) [dimensionless]
    LayerNormStandardDeviation: NotationSymbol  # standard deviation over the features of one input vector, as layer normalization computes it ($\sigma_{\text{LN}}$) [dimensionless]
    LayerOutputMatrix: NotationSymbol  # output matrix of a layer's matrix multiplication of the input matrix by the weight matrix, one row per example ($\mathbf{Y}$)
    LayerOutputVector: NotationSymbol  # vector of activations one layer produces ($\mathbf{y}$)
    LayerOutputWidth: NotationSymbol  # number of output features, or output channels, of one layer ($d_{\text{out}}$)
    LayerParameterPayload: NotationSymbol  # bytes of one sharding unit's parameters, one layer, that a fully sharded schedule gathers or reduce-scatters ($M_{\text{layer}}$) [bytes]
    LayerSavedTensorCount: NotationSymbol  # number of activation tensors one layer retains for its backward pass ($n_{\text{saved}}$)
    LayerSelectionIndex: NotationSymbol  # position of one layer's parameter block among a network's layers, from one to the layer count, as task-adaptive sparse updates select layers ($i$)
    LayerWeightEntry: NotationSymbol  # one scalar entry of a layer's weight matrix, picked out by a layer superscript and numeric row and column subscripts ($W$) [dimensionless]
    LayerWeights: NotationSymbol  # weights of one layer viewed as a single vector, whose norm layer-wise adaptive rate scaling compares with the layer's gradient norm ($\mathbf{w}$)
    LayerWidth: NotationSymbol  # number of units in the layer at a given position of the network, with n_0 the input width ($n$)
    LcResonantFrequency: NotationSymbol  # resonant frequency of a branch of inductance and capacitance, one over two pi root LC ($f_0$) [Hz]
    LeaderForwardWave: NotationSymbol  # forward wave variable at the leader side of a teleoperation channel, effort plus impedance-scaled flow over the square root of twice the impedance ($\mathbf{u}_m$)
    LeaderJointPositions: NotationSymbol  # joint positions of a teleoperation leader arm ($\mathbf{q}_{\text{leader}}$)
    LeaderPortEffort: NotationSymbol  # effort, force or torque, at the leader-side port of a teleoperation channel ($\mathbf{F}_m$)
    LeaderPortFlow: NotationSymbol  # flow, velocity, at the leader-side port of a teleoperation channel ($\dot{\mathbf{x}}_m$)
    LeaderPortPower: NotationSymbol  # instantaneous power entering a teleoperation channel at its leader-side port, effort times flow ($P_m$) [W]
    LeaderReturnWave: NotationSymbol  # returning wave variable at the leader side of a teleoperation channel, effort minus impedance-scaled flow over the square root of twice the impedance ($\mathbf{w}_m$)
    LeaderWristRollAngle: NotationSymbol  # wrist roll angle commanded through a teleoperation leader device ($\theta_{\text{roll}}$) [deg]
    LeadingSupportDistance: NotationSymbol  # horizontal distance from a vehicle's center of gravity forward to its leading ground support ($d_{\text{lead}}$) [m]
    LeakageCurrent: NotationSymbol  # current that leaks through a chip's transistors when they are not switching ($I_{\text{leak}}$) [A]
    LeakagePower: NotationSymbol  # static power a chip dissipates through leakage current regardless of switching ($P_{\text{leakage}}$) [W]
    LearnedActionProposal: NotationSymbol  # action proposal a learned policy emits, the raw input a kinodynamic projection adjusts into a feasible trajectory ($\hat{\mathbf{a}}_{\text{learned}}$)
    LearnedNominalTrajectory: NotationSymbol  # joint configuration the learned proposal calls for at each instant, the nominal trajectory a projection tracks ($\hat{\mathbf{q}}_{\text{learned}}$)
    LearnedRemainingReturnEstimate: NotationSymbol  # learned value function that approximates the policy's expected remaining return at an interaction history, used to form temporal-difference residuals ($\widehat{V}$) [dimensionless]
    LearnedVelocityField: NotationSymbol  # neural velocity field that flow matching trains and integrates, given a point, flow time and the observation history ($v_\theta$)
    LearnerStateDisagreementRate: NotationSymbol  # probability that a learned policy's action differs from the reference policy's, averaged over the states the learner itself visits ($\epsilon_\pi$) [dimensionless]
    LearnerStateVisitationDistribution: NotationSymbol  # distribution of states the learned policy visits under its own closed-loop actions, the expected fraction of time steps its rollouts spend in each ($p^{\pi}$)
    LearningCurveSampleCount: NotationSymbol  # number of training samples a model is trained on at one point of its learning curve ($n$)
    LearningRate: NotationSymbol  # step size the optimizer multiplies the gradient by in each parameter update ($\eta$) [dimensionless]
    LeaseAdmissionInstant: NotationSymbol  # instant at which the permission path admits an intent lease ($t_{\text{adm}}$) [s]
    LeaseDirtyMemory: NotationSymbol  # memory of the private pages one active sandbox lease writes under copy-on-write ($M_{\text{dirty}}$) [bytes]
    LeaseExpiryDelaySubtotal: NotationSymbol  # subtotal of capture-to-brake-onset delay for a lease-expiry stop, excluding dispatch-to-admission time ($\tau_{\text{delay,sub}}$) [s]
    LeaseHoldTime: NotationSymbol  # mean time a trajectory holds a sandbox or other leased resource ($T_{\text{hold}}$) [s]
    LeasePathTravel: NotationSymbol  # distance the machine travels during the chunk lease, control tick and fieldbus transport ($d_{\text{lease}}$) [m]
    LeaseRefreshInstant: NotationSymbol  # local hardware time of the most recent execution-lease renewal accepted after authentication, freshness, state and evidence checks ($t_{\text{refresh}}$) [s]
    LeftEvasionAction: NotationSymbol  # demonstrated action that steers a mobile manipulator around an obstacle on the left ($\mathbf{a}_{\text{left}}$)
    LeftSingularVectorMatrix: NotationSymbol  # matrix whose columns are the left singular vectors of a weight matrix ($\mathbf{P}$)
    LengthPenaltyWeight: NotationSymbol  # non-negative weight on the normalized generated-length penalty of a shaped trajectory reward ($\lambda_{\text{len}}$) [dimensionless]
    LeverArmLength: NotationSymbol  # distance from a rotating joint to the point it carries, which converts a joint rotation into tool travel ($r_{\text{arm}}$) [m]
    LeverArmSweepError: NotationSymbol  # tool displacement caused by joint rotation acting through a lever arm over a stale interval ($e_{\text{rot}}$) [m]
    LewisFormFactor: NotationSymbol  # Lewis form factor of a gear tooth, set by its profile and tooth count ($Y$) [dimensionless]
    LexicalDocument: NotationSymbol  # one indexed document that a lexical ranking function such as BM25 scores against a query ($x$)
    LexicalFrequencySaturation: NotationSymbol  # tuning parameter of BM25 lexical ranking that sets how quickly repeated occurrences of a query word stop adding to a document's score ($k_1$) [dimensionless]
    LexicalLengthNormalization: NotationSymbol  # tuning parameter in [0, 1] of BM25 lexical ranking that sets how strongly a document's score is penalized for its length relative to the collection mean ($b$) [dimensionless]
    LexicalQuery: NotationSymbol  # lexical query as the multiset of analyzer words a ranking function such as BM25 scores documents against ($Q$)
    LidarAzimuthAngle: NotationSymbol  # azimuth angle along which a scanning lidar fires a pulse, measured in the sensor frame ($\theta_{\text{az}}$) [rad]
    LidarBeamDirection: NotationSymbol  # unit vector along which a lidar emits one pulse, a function of its azimuth and elevation ($\mathbf{d}$) [dimensionless]
    LidarElevationAngle: NotationSymbol  # elevation angle along which a scanning lidar fires a pulse, measured in the sensor frame ($\phi$) [rad]
    LidarFramePoint: NotationSymbol  # position of a lidar return expressed in the lidar's own frame ($\mathbf{p}_{\text{lidar}}$) [m]
    LidarReferenceFrame: NotationSymbol  # coordinate frame attached to a lidar sensor, in which its returns are first expressed ($\mathcal{F}_{\text{lidar}}$)
    LidarScanRate: NotationSymbol  # number of full azimuth sweeps a spinning lidar completes per second ($f_{\text{scan}}$) [Hz]
    LidarSweepPeriod: NotationSymbol  # time a spinning lidar takes to complete one full azimuth sweep ($T_{\text{sweep}}$) [s]
    LidarSweepStartTime: NotationSymbol  # instant at which a lidar sweep begins, the reference epoch its returns are de-skewed to ($t_{\text{sweep}}$) [s]
    LieDerivativeAlongControl: NotationSymbol  # sensitivity of the barrier rate to the control input, the barrier gradient applied to the input field ($L_{\mathbf{g}} h$)
    LieDerivativeAlongDrift: NotationSymbol  # rate at which the barrier value changes under the unforced dynamics alone ($L_{\mathbf{f}} h$) [m/s]
    LieDerivativeOrder: NotationSymbol  # number of times the drift Lie derivative is applied in a repeated Lie derivative, below the relative degree ($k$)
    LikelihoodRatioWeight: NotationSymbol  # importance weight of one draw, the operational density over the proposal density at that draw ($w$) [dimensionless]
    LimitMarginFunction: NotationSymbol  # margin by which a checked trajectory signal stays below its limit at each instant ($M$) [varies]
    LinearAcceleration: NotationSymbol  # linear acceleration of a body or sensor in an inertial frame ($\mathbf{a}$) [m/s^2]
    LinearBarrierDecayRate: NotationSymbol  # rate constant of the linear class-K function in the barrier condition ($\gamma_{\text{cbf}}$) [1/s]
    LinearMomentum: NotationSymbol  # mass times speed of a moving body ($p$) [kg*m/s]
    LinearizedVirtualInput: NotationSymbol  # new input of a feedback-linearized system, which the linearizing law makes equal to the stacked highest output derivatives ($\mathbf{v}$)
    LinkAngularAcceleration: NotationSymbol  # angular acceleration of a driven link on the output side of its transmission ($\ddot{\theta}_{\text{link}}$) [rad/s^2]
    LinkBandwidth: NotationSymbol  # bandwidth of one network link in each direction, in the alpha-beta communication model ($\beta$) [bytes/s]
    LinkBitRate: NotationSymbol  # line bit rate of one Ethernet-based fieldbus link ($R_{\text{link}}$) [bits/s]
    LinkLength: NotationSymbol  # length of one link of a serial arm, indexed by its position in the chain ($l$) [m]
    LinkOutputPosition: NotationSymbol  # angular position of the driven link at the transmission output ($q_{\text{out}}$) [rad]
    LintFalseAcceptFraction: NotationSymbol  # fraction of the tasks a cheap replacement gate exits that violate the independently assessed operational criterion ($\phi$) [dimensionless]
    LipschitzConstant: NotationSymbol  # Lipschitz constant of a function, the largest ratio of the change in its output to the change in its input ($K$) [varies]
    LiveAutomatonStateSet: NotationSymbol  # set of automaton states from which some finite sequence of ordinary vocabulary tokens reaches an accepting state ($Q_{\text{live}}$)
    LoadCurrent: NotationSymbol  # current the loads draw from a bus node ($I_{\text{load}}$) [A]
    LoadInertia: NotationSymbol  # moment of inertia of the mechanical load a joint drives, on the output side of the transmission ($J_{\text{load}}$) [kg*m^2]
    LoadStepCurrentDeficit: NotationSymbol  # initial current deficit between load demand and battery source current when a load step begins, which the bus capacitance must supply ($\Delta I_{\text{step}}$) [A]
    LoadStepDeficitDuration: NotationSymbol  # time over which the current deficit of a load step decreases linearly to zero ($\Delta t_{\text{step}}$) [s]
    LoadedEnforcerExecutionTime: NotationSymbol  # execution time of one permission evaluation measured under operational proposal-side load, the isolated execution time plus shared-hardware interference ($t_{\text{enf}}$) [s]
    LoadedJunctionTemperature: NotationSymbol  # steady-state junction temperature under continuous proposal-inference power ($T_{\text{load}}$) [degC]
    LocalBatchSize: NotationSymbol  # number of examples one data-parallel worker processes for one optimizer update ($B_{\text{local}}$)
    LocalEpochCount: NotationSymbol  # number of passes each federated client makes over its local data in one round ($E$)
    LocalGradient: NotationSymbol  # gradient vector one data-parallel worker computes from its local batch in one step, its contribution to the gradient collective ($\mathbf{g}$)
    LocalInferenceEnergy: NotationSymbol  # energy a device spends running one inference on its own hardware ($E_{\text{local}}$) [J]
    LocalPathTime: NotationSymbol  # complete time to serve one request on the local device, with no network transit ($T_{\text{local}}$) [s]
    LocalRankIndex: NotationSymbol  # position of an accelerator within its node, its local rank, which in a rail-optimized fabric also names the rail it attaches to ($i$)
    LocalTokenCount: NotationSymbol  # number of tokens in one rank's local batch entering a mixture-of-experts layer ($B_{\text{tok}}$)
    LocalUpdateCount: NotationSymbol  # number of optimizer updates each worker takes on its own between synchronizations in local SGD ($H_{\text{local}}$)
    LocalityPenaltyFunction: NotationSymbol  # penalty a placement's locality cost assigns to a GPU pair as a function of their topological distance ($w$) [dimensionless]
    LocalizationBound: NotationSymbol  # bound on the error of localization and state estimation ($\delta_{\text{loc}}$) [m]
    LogFailureRatio: NotationSymbol  # natural logarithm of the probability that a claim-size subset of graded runs contains no pass, accumulated factor by factor ($s_{\text{fail}}$) [dimensionless]
    LogOddsLowerClamp: NotationSymbol  # lower bound at which a voxel's stored log-odds is clamped to prevent saturation lockup ($L_{\min}$) [dimensionless]
    LogOddsUpperClamp: NotationSymbol  # upper bound at which a voxel's stored log-odds is clamped to prevent saturation lockup ($L_{\max}$) [dimensionless]
    LogPositionIndex: NotationSymbol  # position of one entry or boundary in a trajectory's durable append-only log, counted from one ($j$)
    LoggedEvent: NotationSymbol  # one event recorded in a trajectory's durable append-only log, numbered by its log position ($e$)
    LoggedEventPrefix: NotationSymbol  # ordered sequence of the events in a trajectory's durable append-only log, from the first entry through a given log position ($\mathbf{e}_{1:j}$)
    LoggpLongMessageTime: NotationSymbol  # time to send one long message of n bytes in the LogGP model ($T_{\text{long}}$) [s]
    LogisticProbability: NotationSymbol  # probability between zero and one taken as the argument of the logit function, the output of a logistic sigmoid ($p_{\text{sig}}$) [dimensionless]
    LogitMask: NotationSymbol  # vector of additive logit masks over the vocabulary for one parser state, zero on valid tokens and minus infinity elsewhere ($\mathbf{M}$)
    LogitMaskEntry: NotationSymbol  # additive mask value for one vocabulary token, zero if the token is valid in the parser state and minus infinity otherwise ($M$) [dimensionless]
    LogitStorageSize: NotationSymbol  # bytes stored per logit in the vector a model emits at each decode step ($s_{\text{logit}}$) [bytes/element]
    LogitVector: NotationSymbol  # vector of logits a classifier outputs, the unnormalized score for every class before softmax ($\mathbf{z}$)
    LogpNetworkLatency: NotationSymbol  # time a message spends in flight across the network in the LogP and LogGP models, excluding processor overhead ($L_{\text{LogGP}}$) [s]
    LonePassAdvantage: NotationSymbol  # group-relative coefficient of the single passing rollout in a binary-reward group with exactly one pass ($\hat{A}_{\text{pass}}$) [dimensionless]
    LongMessageGapPerByte: NotationSymbol  # time per byte to send a long message in the LogGP communication model ($G$) [s/byte]
    LongestInterruptMaskTime: NotationSymbol  # longest interval during which a kernel runs with CPU interrupts disabled ($t_{\text{irq\_disabled,max}}$) [s]
    LongitudinalFrictionCoefficient: NotationSymbol  # longitudinal tire-road friction coefficient as a function of slip ratio, tractive force over normal load ($\mu_x$) [dimensionless]
    LongitudinalSlipRatio: NotationSymbol  # longitudinal slip ratio of a wheel, the normalized difference between its circumferential speed and the vehicle forward speed ($\lambda$) [dimensionless]
    LongitudinalTractionStiffness: NotationSymbol  # longitudinal traction stiffness of a tire, the slope of tractive force against slip ratio in the linear low-slip range ($C_x$) [N]
    LongitudinalVehicleSpeed: NotationSymbol  # forward longitudinal speed of a wheeled vehicle over the ground ($v_x$) [m/s]
    LookaheadStepIndex: NotationSymbol  # number of steps ahead of the current step at which a temporal-difference residual enters the generalized advantage sum ($l$)
    LoopAcceptanceProbability: NotationSymbol  # probability that a bounded agent loop produces an accepted candidate within its turn cap, a function of the cap ($p_{\text{loop}}$) [dimensionless]
    LoopGainScalingFactor: NotationSymbol  # scalar multiplier inserted at the loop break to test how much gain change the closed loop tolerates ($\kappa$) [dimensionless]
    LoopNestDepth: NotationSymbol  # number of nested loops in a computation's loop nest ($n_{\text{loops}}$)
    LoopOrderCount: NotationSymbol  # number of ways to order the loops of a loop nest, ignoring dependences and equivalent orderings ($N_{\text{order}}$)
    LoopParallelizationChoiceCount: NotationSymbol  # number of ordered ways to choose the loops of a loop nest that run in parallel, before legality constraints ($\mathcal{P}_{\text{parallel}}$)
    LoopTurnCost: NotationSymbol  # monetary cost of one turn of a bounded agent loop: inference, tool execution, sandbox lease and assessment ($C$) [USD]
    LossAdjointSeed: NotationSymbol  # reverse-mode adjoint of the loss, set to one to seed the backward sweep ($\bar{\mathcal{L}}$) [dimensionless]
    LossFunction: NotationSymbol  # training objective that scores a model's predictions against the labels, the quantity training minimizes ($\mathcal{L}$)
    LossGradient: NotationSymbol  # gradient of the training loss with respect to one model parameter, as an optimizer update uses it ($g$) [varies]
    LossGradientVector: NotationSymbol  # gradient of the training loss with respect to all model parameters, written as one vector ($\mathbf{g}$)
    LossHessian: NotationSymbol  # matrix of second derivatives of the loss with respect to every pair of model parameters ($\mathbf{H}$)
    LossLowerBound: NotationSymbol  # greatest lower bound of the training objective over all parameter values ($\mathcal{L}_{\inf}$) [dimensionless]
    LossScaleFactor: NotationSymbol  # multiplier applied to the loss in reduced-precision training so small gradients stay representable ($s_{\text{loss}}$) [dimensionless]
    LossTimeDerivative: NotationSymbol  # rate of change of the training loss along a continuous-time parameter trajectory ($\dot{\mathcal{L}}$)
    LostMotionDeflection: NotationSymbol  # load-dependent angle by which the link lags the motor-predicted position through backlash and compliance ($\delta_{\text{lost}}$) [rad]
    LostSuffixTurnCount: NotationSymbol  # number of completed turns past the most recent checkpoint when one crash in a trace occurs, as a function of the checkpoint stride ($H_{\text{suffix},i}$)
    LostWorkReplayTime: NotationSymbol  # time to recompute the training steps lost since the last checkpoint after a restart ($T_{\text{replay}}$) [s]
    LotSamplingProbability: NotationSymbol  # probability that Poisson sampling includes each record in a DP-SGD lot, the expected lot size divided by the dataset size ($q$) [dimensionless]
    LowRankLeftFactor: NotationSymbol  # tall factor of a low-rank weight factorization, with one row per weight-matrix row and one column per retained rank ($\mathbf{U}$)
    LowRankRightFactor: NotationSymbol  # second factor of a low-rank weight factorization, with one row per weight-matrix column and one column per retained rank ($\mathbf{V}$)
    LowerPriorityFrames: NotationSymbol  # function giving the set of frames whose priority is lower than that of a given frame ($lp$)
    LowerPriorityTasks: NotationSymbol  # function giving the set of tasks whose priority is lower than that of the given task ($lp$)
    LowerRatioClippingWidth: NotationSymbol  # distance below one at which the clip-higher surrogate stops penalizing a falling probability ratio ($\epsilon_{\text{low}}$) [dimensionless]
    LqrControlWeightMatrix: NotationSymbol  # penalty matrix on control effort in the linear-quadratic regulator cost ($\mathbf{R}_u$)
    LqrCostToGoMatrix: NotationSymbol  # symmetric solution of the regulator algebraic Riccati equation, whose quadratic form is the optimal LQR cost-to-go ($\mathbf{P}$)
    LqrFeedbackGain: NotationSymbol  # state-feedback gain matrix of the linear-quadratic regulator ($\mathbf{K}_{\text{lqr}}$)
    LqrQuadraticObjective: NotationSymbol  # infinite-horizon quadratic cost the linear-quadratic regulator minimizes: the integral of the weighted squared state plus the weighted squared control effort ($J$) [varies]
    LqrStateWeightMatrix: NotationSymbol  # penalty matrix on state deviation in the linear-quadratic regulator cost ($\mathbf{Q}_x$)
    LshBandCount: NotationSymbol  # number of bands locality-sensitive hashing splits a MinHash signature into ($L$)
    LshCandidateProbability: NotationSymbol  # probability that a pair of records with a given Jaccard similarity matches in at least one band of a banded LSH index and so becomes a candidate ($p_{\text{candidate}}$) [dimensionless]
    LshRowsPerBand: NotationSymbol  # number of MinHash signature rows in each locality-sensitive hashing band ($r$)
    LyapunovAnalysisDomain: NotationSymbol  # open region of state space containing the equilibrium, on which the dynamics and a Lyapunov candidate are analyzed ($\mathcal{D}$)
    LyapunovFunction: NotationSymbol  # scalar function of the state, positive away from the equilibrium, whose decrease along trajectories certifies stability ($V$)
    LyapunovFunctionDerivative: NotationSymbol  # time derivative of a Lyapunov function along the system trajectories, its Lie derivative along the dynamics ($\dot{V}$)
    LyapunovPositivityMargin: NotationSymbol  # positive weight on the squared state norm added to a learned Lyapunov candidate so it is positive away from the origin ($\epsilon_{\text{pos}}$) [varies]
    LyapunovSublevelSet: NotationSymbol  # set of states in the analysis domain where the Lyapunov function is at most a level c, an inner estimate of the domain of attraction when trajectories in it converge ($\Omega_c$)
    MacAccumulator: NotationSymbol  # running sum that a multiply-accumulate operation adds each product into ($\text{acc}$) [dimensionless]
    MachineTaxonomyAxis: NotationSymbol  # Machine axis of the D·A·M taxonomy, viewed as the set of techniques and concepts it covers so that axis intersections name zones of the taxonomy ($\mathsf{Machine}$)
    MahalanobisDistance: NotationSymbol  # Mahalanobis distance between candidate detection j and the predicted measurement of track i ($d_M$) [dimensionless]
    ManifestTrackIndex: NotationSymbol  # position of one confirmed or coasting track in a tracker manifest ($k$)
    ManipulatorBaseFrame: NotationSymbol  # coordinate frame at the manipulator base in which targets, tolerances and error envelopes are expressed ($\mathcal{F}_{\text{base}}$)
    ManipulatorJacobian: NotationSymbol  # matrix mapping joint velocities to the spatial velocity of the end effector ($\mathbf{J}$)
    ManualLease: NotationSymbol  # execution lease of a human operator in manual mode: the longest silence between authenticated heartbeats before manual authority lapses ($T_{\text{lease,man}}$) [s]
    ManualResetTime: NotationSymbol  # time a human technician takes to restore a physical rig to its initial configuration between trials ($t_{\text{reset,manual}}$) [s]
    ManualReviewCost: NotationSymbol  # cost of sending one item to manual review ($C_{\text{manual}}$) [USD/item]
    ManualReviewFraction: NotationSymbol  # fraction of items a confidence threshold routes to manual review ($r_{\text{manual}}$) [dimensionless]
    MapCovarianceJacobian: NotationSymbol  # Jacobian that projects the map-frame pose covariance into world coordinates ($\mathbf{J}_{\text{map}}$)
    MapOdometryTransform: NotationSymbol  # rigid transform mapping odometry-frame coordinates into the map frame, corrected at relocalization ($\mathbf{T}_{\text{map}}^{\text{odom}}$)
    MapParameterEstimate: NotationSymbol  # maximum a posteriori point estimate of model parameters, the mode of the posterior ($\theta_{\text{MAP}}$)
    MapPoseCovariance: NotationSymbol  # covariance of the map-frame pose correction, which jumps at relocalization ($\boldsymbol{\Sigma}_{\text{map}}$)
    MapReferenceFrame: NotationSymbol  # globally corrected map coordinate frame, which may jump at relocalization ($\mathcal{F}_{\text{map}}$)
    MappedCommandTap: NotationSymbol  # second stream of the four-stream action tap, the kinematically mapped command before it crosses to the safety microcontroller ($a_{\text{cmd}}$)
    MarginalFalseApprovalProbability: NotationSymbol  # probability that one model reviewer approves a candidate known to be invalid, averaged over the candidate population ($p$) [dimensionless]
    MarginalRequestServiceTime: NotationSymbol  # additional batch service time that one more request in the batch adds ($t_{\text{req}}$) [s]
    MarginalStoppingThreshold: NotationSymbol  # smallest gain in the calibrated probability of an acceptable result per unit of added cost for which a search keeps admitting work ($\epsilon_{\text{stop}}$) [1/USD]
    MarginalVelocityField: NotationSymbol  # marginal velocity field of a flow-matching probability path, the average of conditional velocities through a point ($u$)
    MarginalizedModelParameters: NotationSymbol  # value of the model parameters that the marginal likelihood integrates over, a dummy variable ranging over the parameter space ($\theta'$)
    MarkovDecisionProcess: NotationSymbol  # Markov decision process of continuous control, the tuple of state space, action space, transition dynamics, reward function, discount factor and initial state distribution ($\mathcal{M}$)
    MaskTokenIndex: NotationSymbol  # index of one vocabulary token whose logit a grammar mask keeps or removes ($i$)
    MaskedFinetuningLoss: NotationSymbol  # mean cross-entropy over the token positions the loss mask selects, for one example ($\mathcal{L}_{\text{SFT}}$) [nats]
    MaskedImitationDataset: NotationSymbol  # training dataset of plant-state, imitation-target and causal-mask triples built from logged intervention episodes ($\mathcal{D}_{\text{train}}$)
    MaskedImitationLoss: NotationSymbol  # imitation-learning objective over recorded trajectories in which a binary causal mask selects which steps' targets train the policy ($\mathcal{L}_{\text{imitation}}$)
    MaskedLogitVector: NotationSymbol  # vector of next-token logits after the grammar constraint mask is added, unchanged on valid tokens and minus infinity elsewhere ($\mathbf{z}'$) [dimensionless]
    MaskedTokenLogit: NotationSymbol  # logit of one vocabulary token at a decoding step after the constraint mask, equal to its logit when allowed and minus infinity otherwise ($z'$) [dimensionless]
    Mass: NotationSymbol  # mass of a physical body ($m$) [kg]
    MassCenterAcceleration: NotationSymbol  # acceleration of the robot's total center of mass ($\ddot{\mathbf{c}}$) [m/s^2]
    MassCenterHeight: NotationSymbol  # height of a body's center of mass above its support or contact point ($z_c$) [m]
    MassCenterPosition: NotationSymbol  # position of the robot's total center of mass ($\mathbf{c}$) [m]
    MasterProcessingTime: NotationSymbol  # time the EtherCAT master spends processing one cycle frame ($t_{\text{master}}$) [s]
    MatchingRowCount: NotationSymbol  # number of rows that satisfy a selection predicate and are returned by an index lookup ($K$)
    MatrixColumnIndex: NotationSymbol  # column index of an entry in a matrix ($j$)
    MatrixEigenvalue: NotationSymbol  # eigenvalue of a square matrix, a root of its characteristic polynomial ($\lambda$) [varies]
    MatrixMultiplyDestination: NotationSymbol  # destination matrix of a matrix multiply-accumulate instruction, which receives the product plus the accumulator input ($\mathbf{D}$)
    MatrixRowIndex: NotationSymbol  # position of a row within a matrix ($i$)
    MaxBatchSize: NotationSymbol  # largest batch a dynamic batcher forms, the size at which it dispatches without waiting for the window to close ($B_{\text{max}}$)
    MaxBatchingWindow: NotationSymbol  # longest batching window a traffic-adaptive batcher may choose, the cap on its window when arrivals are slow ($T_{\text{max}}$) [s]
    MaxReceivedTokenCount: NotationSymbol  # largest number of tokens any one rank receives from other ranks in an expert-dispatch exchange ($B_{\text{max}}$)
    MaxResidentContextCount: NotationSymbol  # largest number of contexts of mean length whose KV cache fits in a node's KV pool at once, the memory bound on decode batch size ($B_{\max}$)
    MaximumActuatorEffort: NotationSymbol  # upper limit of the force or torque an actuator can realize ($u_{\max}$) [varies]
    MaximumBreakerCooldownInterval: NotationSymbol  # largest cooldown a circuit breaker reaches when repeated failed probes keep doubling it ($T_{\text{cool}}^{\max}$) [s]
    MaximumCompletionOffset: NotationSymbol  # longest justified time from a tick's start to its completion, including declared execution, blocking and interference ($c_{\max}$) [s]
    MaximumDepthHypothesis: NotationSymbol  # farthest depth of the discrete depth bins a lift-splat network scores ($d_{\max}$) [m]
    MaximumEntropyObjective: NotationSymbol  # expected discounted sum of rewards plus the temperature-weighted entropy of the policy at each visited state, the objective maximum-entropy reinforcement learning maximizes ($J$) [dimensionless]
    MaximumLoadCurrentStep: NotationSymbol  # largest step in load current a chip's rail must absorb ($\Delta I_{\max}$) [A]
    MaximumLogit: NotationSymbol  # largest logit in one example's logit vector ($z_{\max}$) [dimensionless]
    MaximumObservationAge: NotationSymbol  # largest observation age a consumer tolerates before acting on the observation is no longer safe or stable ($t_{\text{age},\max}$) [s]
    MaximumPlateIndentation: NotationSymbol  # largest compression of the latch strike plate, reached when the braking arm comes to rest ($x_{\max}$) [m]
    MaximumRetryBackoffInterval: NotationSymbol  # cap on the backoff interval of any single retry under capped exponential backoff ($b_{\max}$) [s]
    MaximumTurnOffDelay: NotationSymbol  # longest turn-off delay of an inverter power switch over temperature and load ($t_{\text{off, max}}$) [s]
    McnemarTestStatistic: NotationSymbol  # McNemar test statistic, the squared difference of the two discordant counts divided by their sum ($\chi^2$) [dimensionless]
    McuCaptureTime: NotationSymbol  # capture instant of an observation expressed on the safety microcontroller's clock ($t_{\text{MCU,capture}}$) [s]
    McuDispatchTime: NotationSymbol  # instant an observation is dispatched, expressed on the safety microcontroller's clock ($t_{\text{MCU,dispatch}}$) [s]
    MealTokenKeyVector: NotationSymbol  # key vector of the token "meal" in the chef-cooked-the-meal example ($\mathbf{k}_{\text{meal}}$)
    MealTokenQueryVector: NotationSymbol  # query vector of the token "meal" in the chef-cooked-the-meal example ($\mathbf{q}_{\text{meal}}$)
    MeanAcceptedDraftLength: NotationSymbol  # mean number of draft tokens the target model accepts per speculative-decoding round ($\bar{m}$) [tokens]
    MeanAttemptCost: NotationSymbol  # expected all-inclusive monetary cost of one attempt at a task, the same for every attempt under the independent-trials model ($c$) [USD]
    MeanClusterSize: NotationSymbol  # average number of users in one randomization cluster ($m$) [dimensionless]
    MeanContextLength: NotationSymbol  # mean number of tokens in the contexts a serving node keeps resident ($\bar{S}$) [tokens]
    MeanDeliveryPayloadLength: NotationSymbol  # mean number of tokens one message delivery between agents carries ($L$) [tokens]
    MeanDocumentLength: NotationSymbol  # mean length, in analyzer words, of the documents in a lexically indexed collection ($\overline{|x|}$) [words]
    MeanHoldTime: NotationSymbol  # mean time one item holds a resource, such as a turn holding its KV cache slot, the residence term of Little's law ($\bar{T}_{\text{hold}}$) [s]
    MeanLatchContactForce: NotationSymbol  # average recorded latch contact force under the declared averaging procedure ($\bar{F}$) [N]
    MeanMessageQueueDelay: NotationSymbol  # mean time a safety message waits in a queue ($\bar{T}_q$) [s]
    MeanNeesStatistic: NotationSymbol  # sample mean of the normalized estimation error squared over independent trials ($\bar{\epsilon}$) [dimensionless]
    MeanQueryDistance: NotationSymbol  # common mean distance between the query and each candidate vector at a given ambient dimension ($\mu_D$) [dimensionless]
    MeanRolloutLength: NotationSymbol  # mean number of sampled policy tokens per rollout over a group or microbatch, used as a shared loss divisor ($\bar{N}$) [tokens]
    MeanSlotOccupancy: NotationSymbol  # long-run average number of KV cache slots held resident in accelerator memory, the occupancy term of Little's law ($\bar{N}_{\text{slot}}$) [items]
    MeanSquaredErrorCloningLoss: NotationSymbol  # behavioral cloning objective that averages the squared distance between the policy's action and the demonstrated action over the dataset ($\mathcal{L}_{\text{MSE}}$) [dimensionless]
    MeanStoredTokensPerRecord: NotationSymbol  # average number of stored tokens per retained record of one trajectory category, context and targets together, under the active tokenizer ($\bar{S}$) [tokens]
    MeanTargetTokensPerRecord: NotationSymbol  # average number of loss-selected target tokens per record of one trajectory category admitted into an imitation fine-tuning artifact ($\bar{w}$) [tokens]
    MeanTimeBetweenFailures: NotationSymbol  # average time between consecutive failures of a repairable component or system ($\text{MTBF}$) [hours]
    MeanTimeToFailure: NotationSymbol  # average operating time of a component before it fails, derived from its FIT rate ($\text{MTTF}$) [hours]
    MeanTimeToRepair: NotationSymbol  # average time to restore service after a failure ($\text{MTTR}$) [hours]
    MeanVerifierServiceTime: NotationSymbol  # mean time one verifier slot spends assessing one record ($\bar{T}_{\text{svc}}$) [s]
    MeasuredAxisPosition: NotationSymbol  # measured position of the tracked point along one axis, compared against its commanded reference ($x_{\text{meas}}$) [m]
    MeasuredContactForce: NotationSymbol  # force vector the environment exerts on the end effector during intentional contact, as sensed ($\mathbf{F}_{\text{contact}}$) [N]
    MeasuredJointVelocities: NotationSymbol  # joint velocities measured by the encoders, part of the measured response ($\dot{\mathbf{q}}_{\text{meas}}$)
    MeasuredPixelDepth: NotationSymbol  # depth a calibrated depth image reports at a pixel ($Z_{\text{meas}}$) [m]
    MeasuredResponseTap: NotationSymbol  # fourth stream of the four-stream action tap, the plant response measured by joint encoders and current shunts ($a_{\text{meas}}$)
    MeasuredSeamAccelerations: NotationSymbol  # encoder-derived estimate of the joint accelerations at activation, with its own error bound ($\ddot{\mathbf{q}}_{\text{meas}}$) [rad/s^2]
    MeasuredSeamConfiguration: NotationSymbol  # encoder estimate of the joint configuration at activation, carried with its error bound into the seam check ($\mathbf{q}_{\text{meas}}$)
    MeasuredSeamVelocities: NotationSymbol  # encoder-derived estimate of the joint rates at activation, with its own error bound ($\dot{\mathbf{q}}_{\text{meas}}$) [rad/s]
    MeasuredTrackedPosition: NotationSymbol  # measured Cartesian position of the tracked point, which the tracking error compares against the reference ($\mathbf{p}_{\text{meas}}$) [m]
    MeasurementCovariance: NotationSymbol  # covariance of the error a perception output reports for its measurement ($\boldsymbol{\Sigma}_z$)
    MeasurementDimension: NotationSymbol  # number of components in a measurement vector ($m$)
    MeasurementErrorStandardDeviation: NotationSymbol  # standard deviation of a measurement error under an assumed Gaussian model, the unit of a sigma gate ($\sigma$) [varies]
    MeasurementFunction: NotationSymbol  # nonlinear function mapping a state to the measurement a sensor would report ($h$)
    MeasurementHistory: NotationSymbol  # sequence of measurements an estimator has received from the first through the k-th sample ($\mathbf{y}_{1:k}$)
    MeasurementInnovation: NotationSymbol  # difference between a measurement and the measurement the predicted state implies ($\boldsymbol{\nu}$)
    MeasurementMatrix: NotationSymbol  # output matrix of a linear state-space model, mapping the state to sensor measurements ($\mathbf{H}$)
    MeasurementNoiseCovariance: NotationSymbol  # covariance of the noise on a sensor measurement vector, such as a pixel position with depth or a Kalman filter measurement ($\boldsymbol{\Sigma}_{\text{meas}}$) [varies]
    MeasurementNoiseVector: NotationSymbol  # random disturbance added to the sensor output ($\mathbf{v}_{\text{meas}}$)
    MeasurementVector: NotationSymbol  # measurement vector a sensor or detector reports at one step ($\mathbf{y}$)
    MeasurementVectorDerivative: NotationSymbol  # time derivative of the measurement vector of a state-space model ($\dot{\mathbf{y}}$)
    MeasurementWindowEnd: NotationSymbol  # instant at which an energy measurement window closes ($t_{\text{end}}$) [s]
    MechanicalEnergy: NotationSymbol  # total mechanical energy of a second-order system, its kinetic energy plus the potential energy stored in its spring or proportional feedback ($E$) [J]
    MechanicalEnergyDerivative: NotationSymbol  # time derivative of the total mechanical energy along system trajectories, the net power into the system ($\dot{E}$) [W]
    MechanicalImpedanceMap: NotationSymbol  # mechanical impedance, the map from an imposed motion to the force that answers it ($Z$) [varies]
    MechanicalPlantState: NotationSymbol  # state of the mechanical plant: generalized positions stacked with generalized velocities ($\mathbf{x}$)
    MechanicalWork: NotationSymbol  # mechanical work a force does on a body over the distance it acts ($W$) [J]
    MechanismAcceptanceProbability: NotationSymbol  # probability that one attempt at a task is accepted with an added mechanism, such as a check, a sandbox boundary or a prompt cache, in place ($p'_{\text{acc}}$) [dimensionless]
    MechanismAttemptCost: NotationSymbol  # full monetary cost of one attempt at a task with an added mechanism, such as a check, a sandbox boundary or a prompt cache, in place ($C'_{\text{attempt}}$) [USD]
    MechanismLogMoment: NotationSymbol  # log moment of a mechanism's privacy loss at a given moment order, the worst case over adjacent datasets of the log expected likelihood ratio raised to that order ($\alpha_{\mathcal{M}}$)
    MechanismOutcome: NotationSymbol  # one output a randomized mechanism can release, the value at which its output probabilities or densities on adjacent datasets are compared ($o$)
    MechanismOutputSet: NotationSymbol  # set of possible outputs of a randomized mechanism over which a differential privacy bound must hold ($\mathcal{S}$)
    MechanismOutputVector: NotationSymbol  # vector output a Gaussian mechanism releases, the vector-valued query plus Gaussian noise ($\mathbf{y}$)
    MechanismRenyiPrivacyBudget: NotationSymbol  # Renyi differential privacy bound of the i-th mechanism in a composition, at a common Renyi order ($\epsilon_{\text{R},i}$) [dimensionless]
    MemoryBusQueueDelay: NotationSymbol  # time a sensor transfer waits for contended shared memory channels on the SoC ($t_{\text{queue}}$) [s]
    MemoryEntryBirthGeneration: NotationSymbol  # source generation from which a derived memory entry is supported ($g_{\text{birth}}$)
    MemoryEntryDeathGeneration: NotationSymbol  # source generation at which a later edit stops supporting a derived memory entry ($g_{\text{death}}$)
    MemoryHierarchyLevelCount: NotationSymbol  # number of levels in a memory hierarchy, such as registers, shared memory and DRAM ($N_{\text{mem}}$)
    MemoryHoldDuringWait: NotationSymbol  # accelerator memory a paused trajectory withholds, multiplied by how long it waits ($O_{\text{mem}}$) [bytes*s]
    MemorySubsystemEnergy: NotationSymbol  # energy the memory subsystem consumes over a stated interval, outside any other measured component ($E_{\text{memory}}$) [J]
    MergedAttentionOutput: NotationSymbol  # attention output accumulated over two key blocks taken together, the block partial outputs rescaled to the merged softmax statistics and added ($\mathbf{O}^{\text{new}}$)
    MergedSoftmaxMaximum: NotationSymbol  # largest score over two softmax blocks taken together, the maximum online softmax carries forward after merging a block ($m^{\text{new}}$) [dimensionless]
    MergedSoftmaxNormalizer: NotationSymbol  # softmax denominator over two blocks taken together, the block sums rescaled to the merged maximum and added ($z^{\text{new}}$) [dimensionless]
    MessageInjectionGap: NotationSymbol  # minimum interval between consecutive message injections at one node in the LogP and LogGP models ($g$) [s]
    MessageQueueDelay: NotationSymbol  # time one safety message waits in a queue before it is served, a random variable ($T_q$) [s]
    MessageStartupLatency: NotationSymbol  # fixed startup time paid once per message in the alpha-beta communication model, independent of message size ($\alpha$) [s]
    MessageTimestamp: NotationSymbol  # transport timestamp a publisher stamps on a message, which a republisher can refresh without new evidence ($t_{\text{msg}}$) [s]
    MetricDepth: NotationSymbol  # depth in meters recovered from a relative-depth prediction by its affine calibration ($z_{\text{metric}}$) [m]
    MicroUnitPrefix: NotationSymbol  # SI unit prefix micro, one millionth, written before a unit such as seconds ($\text{µ}$)
    MicrobatchCount: NotationSymbol  # number of microbatches one pipeline-parallel step splits its batch into ($m$)
    MicrobatchGradient: NotationSymbol  # gradient of one micro-batch's loss with respect to all model parameters ($\mathbf{g}$)
    MicrobatchIndex: NotationSymbol  # position of one micro-batch among those accumulated into an optimizer update ($j$)
    MicrobatchPassTime: NotationSymbol  # measured forward and backward time of one micro-batch, indexed by micro-batch ($T_{\text{micro},j}$) [s]
    MicrobatchSize: NotationSymbol  # number of examples in one micro-batch, the unit run through a single forward and backward pass ($B_{\text{micro}}$)
    MicrocontrollerClockTime: NotationSymbol  # time read on the safety microcontroller's monotonic clock ($t_{\text{MCU}}$) [s]
    MillerCapacitance: NotationSymbol  # non-linear gate-drain (reverse transfer, Miller) capacitance of a power switch ($C_{\text{gd}}$) [F]
    MillerCharge: NotationSymbol  # gate-drain (Miller) charge a power switch's gate must exchange while its drain voltage traverses the rail ($Q_{\text{gd}}$) [A*s]
    MillerDisplacementCurrent: NotationSymbol  # displacement current a drain-voltage slew forces through a power switch's Miller capacitance ($I_{\text{miller}}$) [A]
    MillerPlateauDuration: NotationSymbol  # time a power switch's gate spends on the Miller plateau, the Miller charge over the gate current ($t_{\text{plateau}}$) [s]
    MillerPlateauVoltage: NotationSymbol  # gate-source voltage at which a power switch's gate dwells while its drain voltage traverses the rail ($V_{\text{plateau}}$) [V]
    MinHashPermutation: NotationSymbol  # uniformly random permutation of the feature vocabulary whose minimum over a set gives one MinHash component ($\sigma$)
    MinHashPermutationCount: NotationSymbol  # number of independent random permutations, and so of components, in one MinHash signature as used for screening trajectory records ($n_{\sigma}$)
    MinHashSketchSize: NotationSymbol  # number of hash values in one MinHash signature ($s$)
    MiniBatchActivation: NotationSymbol  # activation of one example in a mini-batch, for a single feature or channel, as batch normalization sees it ($a$) [dimensionless]
    MiniBatchSet: NotationSymbol  # set of examples processed together as one mini-batch ($\mathcal{B}$)
    MinibatchGradient: NotationSymbol  # gradient of the mean loss over one minibatch with respect to all model parameters, the stochastic gradient estimate one SGD update uses ($\mathbf{g}$)
    MinimumActuatorEffort: NotationSymbol  # lower limit of the force or torque an actuator can realize ($u_{\min}$) [varies]
    MinimumCellSupportCount: NotationSymbol  # declared minimum number of retained records or demonstrations a scenario-table cell needs to count as supported ($N_{\min}$)
    MinimumChunkSetpointCount: NotationSymbol  # smallest number of setpoints an action chunk must carry to keep the actuators supplied until its replacement arrives ($h_{\min}$)
    MinimumCompletionOffset: NotationSymbol  # shortest justified time from a tick's start to its completion, including healthy paths a campaign did not exercise ($c_{\min}$) [s]
    MinimumDepthHypothesis: NotationSymbol  # nearest depth of the discrete depth bins a lift-splat network scores ($d_{\min}$) [m]
    MinimumDetectableEffect: NotationSymbol  # smallest absolute difference in the outcome metric that an A/B test is designed to detect ($\delta_{\text{MDE}}$) [dimensionless]
    MinimumPrefetchBufferCount: NotationSymbol  # smallest number of batches the prefetch queue must hold so a tail-latency I/O delay does not stall the accelerator ($Q_{\text{prefetch,min}}$)
    MinimumRailVoltage: NotationSymbol  # lowest voltage the DC rail reaches under load transients ($V_{\text{rail,min}}$) [V]
    MinimumRoundTripLatency: NotationSymbol  # smallest possible round-trip time over a network path, set by the propagation speed of the medium ($L_{\text{lat,min}}$) [s]
    MinimumStableBatchSize: NotationSymbol  # batch size at which a batched server's effective service rate equals the arrival rate, the boundary that stable batch sizes must exceed ($B_{\text{min}}$)
    MinimumTransitTime: NotationSymbol  # shortest nominal time for a rest-to-rest move of a given distance under acceleration and speed bounds ($t_{\min}$) [s]
    MinimumTurnOnDelay: NotationSymbol  # shortest turn-on delay of an inverter power switch over temperature and load ($t_{\text{on, min}}$) [s]
    MisclassifiedFraction: NotationSymbol  # fraction of items a model misclassifies at a given confidence threshold ($r_{\text{error}}$) [dimensionless]
    MisroutedQueryFraction: NotationSymbol  # fraction of queries in each class that a router sends to the wrong architecture ($m$) [dimensionless]
    MissAcceptanceProbability: NotationSymbol  # probability that a human reviewer accepts an AI assistant's missed detection instead of correcting it ($\alpha$) [dimensionless]
    MissPathServiceTime: NotationSymbol  # total time to serve one request that misses the hot tier of a storage hierarchy, including the failed hot-tier lookup and the colder-tier access ($T_{\text{cold}}$) [s]
    MissionDuration: NotationSymbol  # duration of one mission of a mobile robot, the unit of operation over which a per-mission failure probability is allocated ($T_{\text{mission}}$) [s]
    MissionEnergy: NotationSymbol  # total energy a machine draws to complete one physical mission ($E_{\text{mission}}$) [J]
    MissionExceedanceBudget: NotationSymbol  # allowed probability that at least one handoff of a mission misses its admission deadline ($\epsilon_{\text{mission}}$) [dimensionless]
    MissionHandoffCount: NotationSymbol  # number of successive plan handoffs a mission makes, each of which must meet its admission deadline ($K$)
    MixtureComponentCount: NotationSymbol  # number of components of a mixture distribution ($M$)
    MixtureComponentCovariance: NotationSymbol  # covariance matrix of one component of a Gaussian mixture, output by a mixture density network for the given input ($\boldsymbol{\Sigma}_m$)
    MixtureComponentIndex: NotationSymbol  # position of one component of a mixture distribution, from one to the component count ($m$)
    MixtureComponentMean: NotationSymbol  # mean vector of one component of a Gaussian mixture, output by a mixture density network for the given input ($\boldsymbol{\mu}_m$)
    MixtureComponentWeight: NotationSymbol  # nonnegative weight of one component of a mixture distribution, the weights over all components summing to one ($w_m$) [dimensionless]
    MixupSample: NotationSymbol  # virtual training input MixUp forms as a weighted combination of two real inputs ($\tilde{x}$)
    MixupWeight: NotationSymbol  # weight MixUp gives the first of two inputs and labels it combines ($\lambda_{\text{mix}}$) [dimensionless]
    MlProductivityGoodput: NotationSymbol  # fraction of a fleet's capacity-time that becomes useful work at the ideal rate, the product of scheduling, runtime and program goodput ($\text{MPG}$) [dimensionless]
    MlpWeightArithmeticIntensity: NotationSymbol  # arithmetic intensity of a dense layer evaluated on one input vector, counting only weight bytes read from the memory level analyzed ($I_{\text{MLP}}$) [FLOP/byte]
    MobileBaseBarrierFunction: NotationSymbol  # barrier certificate of the mobile base, clear distance minus delay travel, braking travel and the fixed allowances ($h_{\text{base}}$) [m]
    ModelCallIndex: NotationSymbol  # position of one model call among the calls whose staged tool definitions are counted ($i$)
    ModelCallRequest: NotationSymbol  # bounded request one model call sends: prompt tokens, model identifier, output limit, deadline, stop sequences and sampling parameters ($\mathcal{R}$)
    ModelDownloadEnergy: NotationSymbol  # energy one federated client spends downloading the global model in one round ($E_{\text{download}}$) [J]
    ModelFlopsUtilization: NotationSymbol  # useful model operations per second divided by the hardware's peak operation rate ($\text{MFU}$) [dimensionless]
    ModelGenerationIndex: NotationSymbol  # position of a model in a chain of models each trained on the previous one's synthetic output ($n$)
    ModelInput: NotationSymbol  # input of one example, the data a model receives to produce a prediction ($x$)
    ModelLifetimeCost: NotationSymbol  # total cost of one model version over its deployment, the one-time training cost plus the serving cost of every query it answers ($C_{\text{total}}$) [USD]
    ModelParameters: NotationSymbol  # learned weights and biases of a model or policy ($\theta$)
    ModelPointAbscissa: NotationSymbol  # first world coordinate of a point on a planar calibration model ($X$) [m]
    ModelPointHeight: NotationSymbol  # world coordinate of a calibration model point normal to the model plane, zero on the plane ($Z$) [m]
    ModelPointOrdinate: NotationSymbol  # second world coordinate of a point on a planar calibration model ($Y$) [m]
    ModelPrediction: NotationSymbol  # prediction a model outputs for one example, compared with its label by the loss ($\hat{y}$) [dimensionless]
    ModelPredictor: NotationSymbol  # trained model viewed as a function that maps an input to its prediction ($f$)
    ModelScore: NotationSymbol  # score a classifier assigns to one individual, read as the predicted probability of the positive outcome ($s$) [dimensionless]
    ModelStepOperations: NotationSymbol  # useful model floating-point operations of one training step, forward plus backward, excluding recomputation ($O_{\text{model}}$) [FLOP]
    ModelTime: NotationSymbol  # time a turn spends in model calls ($T_{\text{model}}$) [s]
    ModelTimeFraction: NotationSymbol  # share of a trajectory's baseline elapsed time spent in model execution, the fraction a faster model can shorten ($f$) [dimensionless]
    ModelWeight: NotationSymbol  # value of one learnable weight of a model, such as the weight on one input of a neuron ($w$) [dimensionless]
    MomentOrder: NotationSymbol  # order of the moment of the privacy loss that the moments accountant bounds, optimized when converting to an (epsilon, delta) guarantee ($\lambda_{\text{mom}}$) [dimensionless]
    MonitorStartupTime: NotationSymbol  # time to start the virtual machine monitor for one sandbox instance ($T_{\text{vmm}}$) [s]
    MonitoredFairnessMetricCount: NotationSymbol  # number of fairness metrics a monitor computes for each protected group ($m$)
    MonitoredFeatureLevel: NotationSymbol  # one value of a monitored feature, the argument at which its distribution or empirical cumulative distribution function is evaluated ($x_{\text{feat}}$) [varies]
    MonitoredGroupCount: NotationSymbol  # number of protected groups a fairness monitor keeps separate statistics for ($k$)
    MonitoredMetric: NotationSymbol  # metric a statistical process control chart monitors over time ($X$) [varies]
    MonitoredMetricMean: NotationSymbol  # established historical mean of a monitored metric, the center line of its control chart ($\mu$) [varies]
    MonitoredMetricStandardDeviation: NotationSymbol  # established historical standard deviation of a monitored metric, which sets its control limits ($\sigma$) [varies]
    MonitoringAlertCost: NotationSymbol  # monitoring cost of evaluating alert rules over an accounting window ($C_{\text{alert}}$) [USD]
    MonitoringComputeCost: NotationSymbol  # monitoring cost of query and dashboard compute over an accounting window ($C_{\text{compute}}$) [USD]
    MonitoringCostComponent: NotationSymbol  # any one of the four monitoring cost components (ingestion, storage, compute, alert) over an accounting window ($C_*$) [USD]
    MonitoringIngestionCost: NotationSymbol  # monitoring cost of ingesting telemetry over an accounting window ($C_{\text{ingest}}$) [USD]
    MonitoringStorageCost: NotationSymbol  # monitoring cost of storing retained telemetry over an accounting window ($C_{\text{storage}}$) [USD]
    MonotonicClockInstant: NotationSymbol  # instant on a qualified monotonic hardware clock ($t$) [s]
    MonthlyOptimizationSavings: NotationSymbol  # monthly savings an optimization yields, net of added operating and maintenance costs ($C_{\text{save,monthly}}$) [USD/month]
    MotionBlurLength: NotationSymbol  # length in the image of a uniform motion blur ($L$) [pixels]
    MotionCyclePeriod: NotationSymbol  # duration of one closed motion cycle of the operator at the teleoperation port ($T_{\text{cycle}}$) [s]
    MotionModelInput: NotationSymbol  # input vector that drives an estimator's motion model, such as one IMU sample ($\mathbf{u}$)
    MotorBackEmfConstant: NotationSymbol  # back-EMF voltage a motor generates per unit rotor speed ($K_e$) [V/(rad/s)]
    MotorConstant: NotationSymbol  # torque a motor produces per square root of the resistive power it dissipates ($K_m$) [varies]
    MotorPhaseRmsCurrent: NotationSymbol  # root-mean-square current in one stator phase of a brushless motor ($I_{\text{phase}}$) [A]
    MotorRotorAccelerations: NotationSymbol  # second time derivative of the reflected motor rotor coordinates ($\ddot{\boldsymbol{\theta}}_{\text{motor}}$) [rad/s^2]
    MotorRotorCoordinates: NotationSymbol  # vector of motor rotor angles reflected through the gear reduction to the link side, one per elastic joint ($\boldsymbol{\theta}_{\text{motor}}$) [rad]
    MotorRotorVelocities: NotationSymbol  # time derivative of the reflected motor rotor coordinates ($\dot{\boldsymbol{\theta}}_{\text{motor}}$) [rad/s]
    MotorShaftDamping: NotationSymbol  # viscous damping coefficient on a motor shaft ($D_{\text{motor}}$) [N*m*s/rad]
    MotorShaftPosition: NotationSymbol  # angular position of the motor shaft at the transmission input ($q_{\text{motor}}$) [rad]
    MotorTorque: NotationSymbol  # torque a motor produces at its shaft, before the transmission ($\tau_{\text{motor}}$) [N*m]
    MotorTorqueConstant: NotationSymbol  # torque a motor produces per ampere of torque-producing current ($K_t$) [N*m/A]
    MotorViscousDamping: NotationSymbol  # viscous damping coefficient on one reflected motor rotor, a diagonal entry of the motor damping matrix ($B_{m}$) [N*m*s/rad]
    MotorViscousDampingMatrix: NotationSymbol  # diagonal matrix of viscous damping coefficients acting on the reflected motor rotor velocities ($\mathbf{B}_m$) [N*m*s/rad]
    MpcControlSequence: NotationSymbol  # sequence of control inputs over a model-predictive-control horizon, one per step, that the optimizer chooses ($\mathbf{u}_{0:N_{\text{mpc}}-1}$)
    MpcControlSequenceTail: NotationSymbol  # control inputs of a model predictive control sequence after its first, which receding-horizon execution discards ($\mathbf{u}_{1:N_{\text{mpc}}-1}$)
    MpcHorizonObjective: NotationSymbol  # finite-horizon objective of model predictive control from a given state, the terminal cost plus the summed stage costs ($J$)
    MpcPredictionHorizon: NotationSymbol  # number of discrete control steps over which model predictive control optimizes ($N_{\text{mpc}}$)
    MpcTerminalCost: NotationSymbol  # cost a model predictive control objective charges the final predicted state of its horizon, the terminal cost ($\ell_f$)
    MpcTerminalCostMatrix: NotationSymbol  # matrix weighting the quadratic cost of the final predicted state in a model predictive control objective ($\mathbf{P}_{N_{\text{mpc}}}$)
    MultiAgentAcceptanceProbability: NotationSymbol  # probability that one attempt by a delegated multi-agent configuration is accepted ($p_{\text{multi}}$) [dimensionless]
    MultiDeviceRunTime: NotationSymbol  # wall-clock time for a fixed amount of productive work on N devices ($T_N$) [s]
    MultimodalFusedTokenStream: NotationSymbol  # concatenated sequence of vision, language and state tokens fed to an embodied transformer ($\mathbf{Z}$)
    MultimodalObservation: NotationSymbol  # robot's synchronized sensor observation at step t: images, depth, joint encoders and touch ($\mathbf{o}$)
    MutatingToolAction: NotationSymbol  # mutation one tool call performs on an external system, the action whose outcome a timeout can leave unknown ($a$)
    MutualInformation: NotationSymbol  # mutual information between two random variables, how much observing one reduces uncertainty about the other ($I$)
    NaturalLogarithmBase: NotationSymbol  # Euler's number, the base of the natural logarithm, written as a factor rather than as an exponential base ($e$) [dimensionless]
    NearestCandidateDistance: NotationSymbol  # distance from a query vector to the nearest stored candidate vector, as a function of collection size and dimension ($D_{\min}$)
    NeesStatistic: NotationSymbol  # normalized estimation error squared: the state error weighted by the inverse reported covariance ($\epsilon$) [dimensionless]
    NegativeGateSupplyVoltage: NotationSymbol  # negative supply rail of a gate driver, toward which it pulls the gate at turn-off ($V_{\text{ee}}$) [V]
    NeighborCount: NotationSymbol  # number of nearest neighbors a similarity search returns for one query ($k$)
    NeighborUserIndex: NotationSymbol  # index of one peer user in another user's recorded connection set ($j$)
    NeighboringDataset: NotationSymbol  # dataset input to a randomized mechanism in the differential privacy definition, compared against a copy that differs in one record ($X$)
    NetAngularBrakingDeceleration: NotationSymbol  # angular deceleration a joint can produce after a gravity or load bias torque is subtracted from its peak torque ($\alpha_{\text{net}}$) [rad/s^2]
    NetBrakingDeceleration: NotationSymbol  # deceleration left after a gravity or load bias is subtracted from the available braking effort ($a_{\text{net}}$) [m/s^2]
    NetworkBandwidth: NotationSymbol  # sustained rate at which a network path delivers bytes between machines ($\text{BW}_{\text{network}}$) [bytes/s]
    NetworkDelayStandardDeviation: NotationSymbol  # standard deviation of the network delay a heartbeat message experiences ($\sigma_d$) [s]
    NetworkEnergy: NotationSymbol  # energy the network fabric, its switches, links and adapters, consumes over a stated interval ($E_{\text{network}}$) [J]
    NetworkFunction: NotationSymbol  # whole neural network viewed as a function of its input and its current parameters, the composition of all its layers ($f_{\text{net}}$)
    NetworkLatency: NotationSymbol  # time a request and its response spend crossing the network, including propagation, routing and protocol handling ($L_{\text{lat,network}}$) [s]
    NetworkOutput: NotationSymbol  # scalar output of a neuron or of a small network ($y$) [dimensionless]
    NetworkTrafficVolume: NotationSymbol  # bytes one worker moves across the network over a stated interval, such as one training step ($M$) [bytes]
    NeuralLyapunovFactor: NotationSymbol  # state-dependent matrix a neural network with parameters phi outputs, whose outer product shapes a learned Lyapunov candidate ($\mathbf{L}_{\phi}$)
    NeuronActivation: NotationSymbol  # activation output of one neuron, the activation function applied to its preactivation ($a$) [dimensionless]
    NeuronBias: NotationSymbol  # bias term added to one neuron's weighted sum ($b$) [dimensionless]
    NeuronConnectionWeight: NotationSymbol  # weight on the connection from input feature i to neuron j of a layer ($w_{ij}$) [dimensionless]
    NeuronIndex: NotationSymbol  # index of a receiving neuron within a layer ($j$)
    NeuronPreactivation: NotationSymbol  # preactivation of one neuron, its weighted input sum plus bias before the activation function ($z$) [dimensionless]
    NextDecisionProcessState: NotationSymbol  # environment state of a decision process that a transition leads to after an action is taken in the current state ($s'$)
    NextFailureTime: NotationSymbol  # random time from the start of an attempt until the next failure, exponentially distributed with mean the system MTBF ($F$) [s]
    NextProprioceptiveObservation: NotationSymbol  # proprioceptive part of the next step's observation, formed from the measured plant response ($\mathbf{o}_{\text{proprio}, t+1}$)
    NextRobotAction: NotationSymbol  # action the policy takes at the next state of a transition, inside the Bellman relation ($\mathbf{a}'$)
    NextTokenDistribution: NotationSymbol  # probability distribution a language model assigns over the next token given its context ($p$)
    NextTokenEntropy: NotationSymbol  # entropy of a policy's next-token distribution after a delivered prefix ($\mathcal{H}$) [nats]
    NextTokenLogitFunction: NotationSymbol  # function a language model with parameters theta computes, mapping input tokens and the generated prefix to next-token logits ($\mathbf{f}_\theta$)
    NgramLength: NotationSymbol  # number of consecutive tokens in one n-gram ($n$)
    NicMttf: NotationSymbol  # mean operating time of one network interface card before it fails ($\text{MTTF}_{\text{nic}}$) [hours]
    NicsPerNode: NotationSymbol  # number of network interface cards in one node ($n_{\text{nic}}$)
    NodeAcceleratorCount: NotationSymbol  # number of accelerators in one serving node ($n_{\text{acc}}$)
    NodeHbmCapacity: NotationSymbol  # total high-bandwidth memory across all accelerators of one serving node ($M_{\text{HBM}}$) [bytes]
    NodeMtbf: NotationSymbol  # mean time between failures that stop one compute node ($\text{MTBF}_{\text{node}}$) [hours]
    NodeStragglerProbability: NotationSymbol  # probability that a single node or link is delayed enough to straggle in one collective step ($p_{\text{straggle}}$) [dimensionless]
    NoisePredictionNetwork: NotationSymbol  # network that predicts the noise added to an action chunk at a diffusion step, given the observation ($\boldsymbol{\epsilon}_\theta$)
    NoisedActionChunk: NotationSymbol  # action chunk after k steps of diffusion noising, the iterate a denoising pass refines ($\mathbf{A}^k$)
    NoisyClassProbability: NotationSymbol  # one entry of the noise-shaped class-probability vector, the returned probability of one class ($\tilde{p}$) [dimensionless]
    NoisyClassProbabilityVector: NotationSymbol  # class-probability vector an API returns after adding Gaussian noise to the model's scores and projecting the result back onto the probability simplex ($\tilde{\mathbf{p}}$)
    NoisyLabelSampleCount: NotationSymbol  # number of training samples with noisy labels needed to reach a target error ($D_{\text{noisy}}$)
    NominalBitRate: NotationSymbol  # bit rate of the CAN FD arbitration phase ($R_{\text{nom}}$) [bits/s]
    NominalBitTime: NotationSymbol  # duration of one bit in the CAN FD arbitration phase, the reciprocal of the nominal bit rate ($T_{\text{bit,nom}}$) [s]
    NominalBrakingDeceleration: NotationSymbol  # braking deceleration computed from bare chassis mass, ignoring the reflected rotor inertia ($a_{\text{nominal}}$) [m/s^2]
    NominalClearance: NotationSymbol  # separation between an obstacle's estimated position and the machine's envelope ($C_{\text{nominal}}$) [m]
    NominalClockFrequency: NotationSymbol  # nominal clock frequency of a processor cluster before any thermal throttling ($f_{\text{nom}}$) [Hz]
    NominalCommandWindow: NotationSymbol  # sequence of nominal control commands from one step through the end of an action-chunk horizon ($\mathbf{u}_{\text{nom}, t:t+H-1}$)
    NominalContactForces: NotationSymbol  # nominal stacked contact forces a whole-body control objective regularizes toward ($\mathbf{f}_{\text{nom}}$) [N]
    NominalControlCommand: NotationSymbol  # control command the learned policy proposes, before the safety filter checks it ($\mathbf{u}_{\text{nom}}$)
    NominalFrameInterval: NotationSymbol  # nominal interval between successive frames of a sensor stream ($T_{\text{nominal}}$) [s]
    NominalFrictionCoefficient: NotationSymbol  # tire-floor friction coefficient on the nominal, undegraded surface ($\mu_{\text{nom}}$) [dimensionless]
    NominalOscillatorFrequency: NotationSymbol  # nominal frequency of a clock's oscillator, the reference for its fractional frequency error ($f_{\text{nom}}$) [Hz]
    NominalPlanExpiry: NotationSymbol  # instant, on the permission clock, at which the nominal setpoints of an admitted trajectory record end ($t_{\text{plan\_exp}}$) [s]
    NominalRatedSpeed: NotationSymbol  # speed rating of the machine in normal, non-derated operation ($v_{\text{nominal}}$) [m/s]
    NominalReferenceClearance: NotationSymbol  # clearance a reference trajectory keeps from an obstacle, before the tracking-error inset is subtracted ($c$) [m]
    NominalSupplyVoltage: NotationSymbol  # nominal regulated supply voltage of a core rail ($V_{\text{nom}}$) [V]
    NominalTickPeriod: NotationSymbol  # nominal interval between consecutive tick starts of the task that feeds a watchdog ($T_{\text{period}}$) [s]
    NominalTrackingTolerance: NotationSymbol  # largest baseline tracking error a fault-injection bench may show before the trial is declared invalid ($\epsilon_{\text{nom}}$) [m]
    NonconformityScore: NotationSymbol  # conformal nonconformity score, how poorly a candidate label agrees with an input, larger meaning less conformity ($s$)
    NoninferiorityRatioMargin: NotationSymbol  # largest relative excess of a candidate intervention rate over the incumbent rate that a non-inferiority test still accepts, fixed before the trial ($\delta$) [dimensionless]
    NoninferiorityTolerance: NotationSymbol  # declared largest loss in acceptance probability that a candidate may show against its baseline and still count as noninferior ($\delta_{\text{tol}}$) [dimensionless]
    NonlinearBiasTorque: NotationSymbol  # vector of velocity-dependent and gravitational generalized forces in the manipulator equation, the Coriolis and centrifugal terms plus gravity ($\mathbf{h}$) [N*m]
    NonlinearObservationSpace: NotationSymbol  # space spanned by the output functions and all their repeated Lie derivatives along the dynamics ($\mathbf{O}_{\text{NL}}$)
    NonlinearOutputFunction: NotationSymbol  # output function of a nonlinear state-space model, mapping the state to the measurement vector ($\mathbf{h}$)
    NonzeroCount: NotationSymbol  # number of nonzero values a sparse matrix format stores ($n_{\text{nz}}$)
    NonzeroParameterBudget: NotationSymbol  # largest number of nonzero weights a pruned model may keep ($k_{\text{nz}}$)
    NormWeightMatrix: NotationSymbol  # positive semidefinite matrix that weights a quadratic norm ($\mathbf{W}$) [varies]
    NormalContactForce: NotationSymbol  # compressive component of a contact force along the contact normal ($f_{k, \perp}$) [N]
    NormalContactImpulse: NotationSymbol  # normal impulse an idealized rigid impact transfers across a contact, the effective normal mass times the jump in normal velocity ($P_n$) [N*s]
    NormalizationScale: NotationSymbol  # learned factor a normalization layer multiplies each normalized activation by ($\gamma$) [dimensionless]
    NormalizationScaleVector: NotationSymbol  # vector of learned per-channel or per-feature factors a normalization layer multiplies its normalized output by ($\boldsymbol{\gamma}$)
    NormalizationShift: NotationSymbol  # learned offset a normalization layer adds to each scaled activation ($\beta$) [dimensionless]
    NormalizationShiftVector: NotationSymbol  # vector of learned per-channel or per-feature offsets a normalization layer adds to its scaled output ($\boldsymbol{\beta}$)
    NormalizedActionCommand: NotationSymbol  # action command rescaled to a fixed range relative to the actuator's maximum travel ($\tilde{a}$) [dimensionless]
    NormalizedActionResidual: NotationSymbol  # one component of the difference between a predicted and a demonstrated action, in normalized action coordinates ($e$) [dimensionless]
    NormalizedActivation: NotationSymbol  # activation after batch normalization subtracts the batch mean and divides by the batch standard deviation ($\hat{a}$) [dimensionless]
    NormalizedCandidateScore: NotationSymbol  # normalized detector score of one candidate referent ($c$) [dimensionless]
    NormalizedElapsedTimeRatio: NotationSymbol  # elapsed time of a fixed work partition split across workers divided by its serial elapsed time, the reciprocal of the speedup ($D$) [dimensionless]
    NormalizedGearRatio: NotationSymbol  # gear ratio divided by the inertia-matching ratio ($\tilde{N}$) [dimensionless]
    NormalizedImageAbscissa: NotationSymbol  # first normalized image coordinate of a point, its camera-frame first coordinate divided by its depth ($x_n$) [dimensionless]
    NormalizedImageOrdinate: NotationSymbol  # second normalized image coordinate of a point, its camera-frame second coordinate divided by its depth ($y_n$) [dimensionless]
    NormalizedMetricBaseline: NotationSymbol  # normalized value two tracked metrics share before a change, the baseline their later values are expressed against ($v$) [dimensionless]
    NormalizedPredicateMargin: NotationSymbol  # atomic predicate margin divided by its declared positive scale, so margins of different units can be compared ($\bar{f}$) [dimensionless]
    NormalizedRadialDistance: NotationSymbol  # distance of a normalized image point from the principal axis, the radius the distortion polynomial is written in ($r$) [dimensionless]
    NormalizedStatusEnvelope: NotationSymbol  # normalized record of a model call's observed outcome: a status, the returned payload and its metadata ($\mathcal{N}$)
    NormalizerTokenIndex: NotationSymbol  # token of the vocabulary that a softmax normalizer sums over ($w$)
    NormalizingClassIndex: NotationSymbol  # index that runs over every output class in the softmax denominator ($k$)
    NormalizingKeyIndex: NotationSymbol  # index that runs over every key token in the denominator of an attention softmax ($m$)
    NormalizingNextState: NotationSymbol  # candidate next environment state that the normalizing sum of a belief update ranges over ($\tilde{s}$)
    NoticeByteAllocation: NotationSymbol  # bytes of the observation budget reserved for the notice that reports how many bytes were omitted ($M_{\text{notice}}$) [bytes]
    NucleusProbabilityMass: NotationSymbol  # cumulative probability mass of the smallest high-probability token set that nucleus sampling draws from ($p$) [dimensionless]
    NullSpaceProjector: NotationSymbol  # projection matrix onto the null space of a task Jacobian, N = I - J^dagger J ($\mathbf{N}$) [dimensionless]
    NullSpaceVelocityBias: NotationSymbol  # arbitrary joint-velocity vector that the null-space projector of a redundant arm filters into self-motion ($\dot{\mathbf{q}}_0$)
    NumericBitWidth: NotationSymbol  # number of bits used to represent one numeric value, such as an arithmetic operand or a quantized weight or activation ($b$)
    NumericalStabilityConstant: NotationSymbol  # small positive constant added to a denominator or variance so the computation never divides by zero ($\epsilon$) [dimensionless]
    ObjectHandle: NotationSymbol  # typed handle passed as a tool argument that designates a host-held object, carrying its own labels for what influenced its selection ($h$)
    ObjectWrench: NotationSymbol  # net wrench, force and moment together, that the contact forces of a grasp exert on the object ($\mathbf{w}_{\text{obj}}$) [varies]
    ObservabilityMatrix: NotationSymbol  # stack of the output matrix times powers of the system matrix, whose rank decides observability ($\mathbf{O}_{\text{obs}}$)
    ObservableModelOutput: NotationSymbol  # model output an adversary can observe, such as a prediction, score or generated text ($O$)
    ObservationAge: NotationSymbol  # age of the state the permission path acts on, from the exposure midpoint at the sensor to the decision ($t_{\text{age}}$) [s]
    ObservationAgeBudget: NotationSymbol  # configured upper bound on observation age above which a consumer refuses a state ($t_{\text{age},\max}$) [s]
    ObservationAgeDisplacement: NotationSymbol  # displacement the robot accumulates between an observation's capture and the end of its observation age ($\Delta \mathbf{p}$) [m]
    ObservationBinding: NotationSymbol  # function that binds a staged observation to the artifact it describes, as an artifact, kind, revision, range and content identifier tuple ($\lambda$)
    ObservationByteBudget: NotationSymbol  # maximum number of bytes of captured tool output staged as one observation ($M_{\text{obs},\max}$) [bytes]
    ObservationCount: NotationSymbol  # number of observations in a dataset a model is fitted to ($N$)
    ObservationDataset: NotationSymbol  # dataset or sequence of observations whose likelihood a model with given parameters evaluates, as in a Bayesian posterior ($\mathcal{D}$)
    ObservationEncodingTime: NotationSymbol  # time to encode one observation, visual backbone and proprioception, once per policy proposal ($t_{\text{enc}}$) [s]
    ObservationFunction: NotationSymbol  # observation model of an estimator, mapping a state to the measurement the sensors would report ($h$)
    ObservationHistory: NotationSymbol  # stacked history of recent visual and proprioceptive observations that a generative action decoder is conditioned on ($\mathbf{O}$)
    ObservationHistoryLength: NotationSymbol  # number of past observation steps a policy's temporal encoder conditions on ($H_{\text{hist}}$)
    ObservationHistorySequence: NotationSymbol  # sequence of the robot's observations from the first step through step t, the history a policy may condition on ($\mathbf{o}_{0:t}$)
    ObservationHistoryWindow: NotationSymbol  # sliding window of the robot's most recent onboard observations, from the history length before the current step up to the current step ($\mathbf{o}_{t-H_{\text{hist}}:t}$)
    ObservationIntervalEnd: NotationSymbol  # time at which an observation interval closes, its excluded end ($t_{b}$) [s]
    ObservationIntervalLength: NotationSymbol  # length of the observation interval over which arrivals, completions and in-system requests are counted, starting at time zero ($T_{\text{obs}}$) [s]
    ObservationIntervalStart: NotationSymbol  # time at which an observation interval opens, its included start ($t_{a}$) [s]
    ObservationPairingLag: NotationSymbol  # uncompensated delay between when an observation was physically true and the action a logger pairs it with ($\Delta t_{\text{lag}}$) [s]
    ObservationPayload: NotationSymbol  # payload an observation contract carries: the raw sensor tensor or a derived geometric bounding envelope ($\mathbf{T}_{\text{payload}}$)
    ObservationResponseLatency: NotationSymbol  # latency from an observation to the actuator's response along a blocking sensor-to-actuation path, the sum of sensing, planning and actuator response times ($T_{\text{total}}$) [s]
    ObservationTimestamp: NotationSymbol  # time at which a labeled observation, such as a churn event, occurred ($t_{\text{obs}}$) [s]
    ObservationTokenCount: NotationSymbol  # number of tokens one call devotes to immediate tool observations and user steering, the observation buffer of the context budget ($S_{\text{obs}}$)
    ObservationTokenSequence: NotationSymbol  # token sequence an observation tokenizer emits: sensory feature tokens concatenated with time, frame and covariance embeddings ($\mathbf{z}_{\text{token}}$)
    ObservedAnnotatorAgreement: NotationSymbol  # share of items on which two annotators assign the same label ($p_o$) [dimensionless]
    ObservedCategoryCount: NotationSymbol  # number of current observations that fall in one category of a categorical feature, in a chi-square goodness-of-fit test ($n_i^{\text{obs}}$)
    ObservedCycleCount: NotationSymbol  # number of operating cycles, each one observation of the monitored quantity, over which exceedances are counted ($N$)
    ObservedDataPoint: NotationSymbol  # one observation of a dataset, such as a sensor reading, indexed by its position ($\mathbf{x}$)
    ObservedFailureCount: NotationSymbol  # number of failures, catastrophic incidents, a test campaign observes over its trials or its operating time ($k$)
    ObservedFailureFraction: NotationSymbol  # observed fraction of a campaign's trials that fail, the sample estimate of the per-trial failure probability ($\hat{p}$) [dimensionless]
    ObservedFreeSpace: NotationSymbol  # part of the spatial memory map whose cells are observed free and still supported by current evidence ($\mathcal{M}_{\text{free}}$)
    ObservedLabelCount: NotationSymbol  # number of task labels a sequential test has observed so far ($n$)
    ObservedOccupiedSpace: NotationSymbol  # part of the spatial memory map whose cells are observed occupied ($\mathcal{M}_{\text{occ}}$)
    ObservedPixelCoordinates: NotationSymbol  # homogeneous pixel coordinates of a robot-link point as observed in the camera image ($\tilde{\mathbf{m}}_{\text{observed}}$) [pixels]
    ObservedPlantState: NotationSymbol  # plant state as independent truth instrumentation records it over a trial window ($\mathbf{x}_{\text{obs}}$)
    ObservedPointRange: NotationSymbol  # distance from a sensor to an observed point along the viewing ray ($r$) [m]
    ObservedRelativeSpeed: NotationSymbol  # speed of an observed entity relative to the machine ($v_{\text{rel}}$) [m/s]
    ObservedSignificanceLevel: NotationSymbol  # p-value of a hypothesis test, the probability under the null hypothesis of a result at least as extreme as the one observed ($p_{\text{val}}$) [dimensionless]
    ObservedSuccessFraction: NotationSymbol  # fraction of a campaign's trials that succeed, the sample estimate of the per-trial success probability ($\hat{p}$) [dimensionless]
    ObservedTrialFailureCount: NotationSymbol  # number of failures observed among the trials of a test campaign, fixed-sample or sequential ($k$)
    ObservedViolationFraction: NotationSymbol  # fraction of eligible observations over a window that violate one reliability target ($V$) [dimensionless]
    ObservedVolume: NotationSymbol  # volume of the observed, navigable space a Euclidean distance field stores ($V_{\text{obs}}$) [m^3]
    ObstacleConfigurationSet: NotationSymbol  # set of configurations at which the robot intersects an obstacle ($\mathcal{C}_{\text{obs}}$)
    ObstacleCostWeight: NotationSymbol  # weight that sets how strongly trajectory optimization trades obstacle proximity against smoothness ($\lambda_{\text{obs}}$) [varies]
    ObstaclePoint: NotationSymbol  # position of one obstacle point stored in the spatial memory map ($\mathbf{o}$) [m]
    ObstaclePosition: NotationSymbol  # position of an obstacle along the line of approach ($x_{\text{obstacle}}$) [m]
    ObstacleProximityCost: NotationSymbol  # cost a configuration incurs for its proximity to obstacles, large near or inside them ($c_{\text{obs}}$) [varies]
    ObstacleRiskRadius: NotationSymbol  # radius that conservatively contains a previously localized obstacle after its evidence has aged, initial uncertainty plus speed bound times age ($r_{\text{risk}}$) [m]
    ObstacleSeparationDistance: NotationSymbol  # distance between the machine and the nearest obstacle or boundary at a given time, the signal a clearance predicate scores ($d$) [m]
    ObstacleStateSet: NotationSymbol  # set of plant states at which the robot overlaps an obstacle or crosses a constraint boundary ($\mathcal{X}_{\text{obs}}$)
    ObstacleSurfaceArea: NotationSymbol  # total surface area of the obstacles in a mapped workspace ($A_{\text{surface}}$) [m^2]
    OccupancyCoverageMultiplier: NotationSymbol  # multiplier on the propagated track deviation that sets the spatial coverage of a forward occupancy set ($k_\sigma$) [dimensionless]
    OccupancyLogOdds: NotationSymbol  # natural logarithm of the odds of a voxel's occupancy probability, the quantity an occupancy map stores per cell ($L$) [dimensionless]
    OccupancyThreshold: NotationSymbol  # occupancy level at or above which a bin counts as heavily loaded in the balls-into-bins argument ($k$)
    OccupiedLogOddsIncrement: NotationSymbol  # positive log-odds increment applied to the voxel that contains a range return ($L_{\text{occ}}$) [dimensionless]
    OccupiedScenarioCellCount: NotationSymbol  # number of scenario-table cells containing recorded data, without asserting a declared support minimum ($N_{\text{supported}}$)
    OdometryBaseTransform: NotationSymbol  # rigid transform mapping base-frame coordinates into the odometry frame ($\mathbf{T}_{\text{odom}}^{\text{base}}$)
    OdometryCovarianceJacobian: NotationSymbol  # Jacobian that projects the odometry pose covariance into world coordinates ($\mathbf{J}_{\text{odom}}$)
    OdometryNoiseStandardDeviation: NotationSymbol  # standard deviation an odometry speed disturbance adds to the distance estimate over one predict interval ($\sigma_{w}$) [m]
    OdometryPoseCovariance: NotationSymbol  # covariance of the odometry-frame pose estimate, which grows with drift ($\boldsymbol{\Sigma}_{\text{odom}}$)
    OdometryReferenceFrame: NotationSymbol  # odometric coordinate frame, locally continuous but drifting ($\mathcal{F}_{\text{odom}}$)
    OfflineTransitionDataset: NotationSymbol  # fixed dataset of logged transitions, each a state, action, reward and next state, that an offline reinforcement learner trains on without further interaction ($\mathcal{D}_{\text{offline}}$)
    OffloadTailNetworkTime: NotationSymbol  # tail network round-trip time of an off-board computation at the required reliability level ($t_{\text{net}}^{\text{tail}}$) [s]
    OffloadTailProcessingTime: NotationSymbol  # tail remote processing time of an off-board computation at the required reliability level ($t_{\text{proc}}^{\text{tail}}$) [s]
    OffloadTransactionWindow: NotationSymbol  # allocated time for network transfer and remote processing within one offloaded transaction ($W$) [s]
    OhmUnit: NotationSymbol  # SI unit of electrical resistance, the ohm, written after a prefix as in milliohms ($\Omega$)
    OilBulkModulus: NotationSymbol  # bulk modulus of a hydraulic fluid, its resistance to compression ($\beta$) [N/m^2]
    OnDieDecouplingCapacitance: NotationSymbol  # decoupling capacitance fabricated on the silicon die, the fastest tier of a decoupling hierarchy ($C_{\text{die}}$) [F]
    OpenLoopTransferFunction: NotationSymbol  # open-loop transfer function of a feedback loop, evaluated along the imaginary axis for frequency response ($L$) [dimensionless]
    OpenLoopTransferMatrix: NotationSymbol  # open-loop transfer function matrix of a multi-input feedback loop, broken at the plant input ($\mathbf{L}$)
    OperationChargeBound: NotationSymbol  # enforced upper bound on the maximum monetary charge one billable operation can incur, reserved before it is dispatched ($\bar{c}$) [USD]
    OperationDispatchOverhead: NotationSymbol  # host time to dispatch one operation, including framework dispatch and kernel launch ($t_{\text{dispatch}}$) [s]
    OperationRecipient: NotationSymbol  # recipient a proposed operation would send its argument values to, as resolved by the host ($r$)
    OperationReconciliationTime: NotationSymbol  # time to reconcile a trajectory's outstanding external operations during resume ($T_{\text{rec}}$) [s]
    OperationalFailureProbability: NotationSymbol  # probability under the operational distribution that a scenario falls in the failure region ($p_F$) [dimensionless]
    OperationalOverheadFraction: NotationSymbol  # fraction of allocated wall time lost outside productive training steps ($f_{\text{overhead}}$) [dimensionless]
    OperationalScenarioDistribution: NotationSymbol  # declared operational distribution of a scenario's uncertain input parameters, the population a failure probability refers to ($p$)
    OperationalTaxonomyDimension: NotationSymbol  # one of the three operational dimensions of the C-cubed taxonomy, indexed 1 to 3 as compute, communication and coordination ($C$)
    OperationalTestDistribution: NotationSymbol  # declared joint distribution of trial conditions an evaluation campaign samples, the population its success probability describes ($\mathcal{D}$)
    Operations: NotationSymbol  # total arithmetic operations a workload executes ($O$) [FLOP]
    OperationsPerSample: NotationSymbol  # arithmetic operations spent training on one sample in one pass ($O_{\text{sample}}$) [FLOP]
    OperatorCognitiveTime: NotationSymbol  # time an operator needs to decide on a response to a takeover stimulus, the cognitive stage of a simple reaction ($T_{\text{cognitive}}$) [s]
    OperatorMotorTime: NotationSymbol  # time an operator needs to move the controls once a response is chosen, the motor stage of a simple reaction ($T_{\text{motor}}$) [s]
    OperatorResponseLatency: NotationSymbol  # time a human operator takes to perceive, decide and move the leader device ($t_{\text{hum}}$) [s]
    OperatorSensoryTime: NotationSymbol  # time an operator needs to perceive a takeover stimulus, the perceptual stage of a simple reaction ($T_{\text{sensory}}$) [s]
    OperatorStabilizationInstant: NotationSymbol  # time after a takeover at which the operator's corrections settle into a steady return toward the nominal corridor ($t_{\text{stabilize}}$) [s]
    OperatorWindowTime: NotationSymbol  # time inside the window of a timed temporal operator, the variable over which robustness is minimized, maximized or bounded ($t'$) [s]
    OperatorWorkspaceMemory: NotationSymbol  # temporary operator workspace and staging buffers live at a given point of an update's execution ($M_{\text{workspace}}$) [bytes]
    OpticalFrequency: NotationSymbol  # frequency of the light striking a photodiode ($\nu$) [Hz]
    OpticalWavelength: NotationSymbol  # wavelength of the light striking an image sensor's photodiodes ($\lambda_{\text{wl}}$) [m]
    OptimalCheckpointInterval: NotationSymbol  # checkpoint interval that minimizes expected overhead from checkpoint writes and lost work ($T^\ast_{\text{ckpt}}$) [s]
    OptimalCostToGoDerivative: NotationSymbol  # time derivative of the optimal cost-to-go along closed-loop trajectories ($\dot{V}$)
    OptimalCostToGoFunction: NotationSymbol  # optimal cost-to-go of an optimal-control problem: the least remaining cost from a state, the value function of the Hamilton-Jacobi-Bellman equation ($V$)
    OptimalModelParameters: NotationSymbol  # model parameters that minimize the training loss for a given architecture ($\theta^*$)
    OptimalStateReturn: NotationSymbol  # largest expected discounted return attainable from a state of a Markov decision process, the fixed point of the Bellman optimality equation ($V_{\text{RL}}$) [dimensionless]
    OptimizationBreakevenTime: NotationSymbol  # time for an optimization's monthly net savings to repay its one-time cost ($T_{\text{breakeven}}$) [months]
    OptimizationImplementationCost: NotationSymbol  # one-time cost of implementing and validating an optimization ($C_{\text{optimization}}$) [USD]
    OptimizationIterate: NotationSymbol  # decision variable of an optimization problem, whose successive values are the numerical iterates ($x$) [varies]
    OptimizerStateMemory: NotationSymbol  # memory holding the optimizer's per-parameter state, such as momentum, variance and master weights ($M_{\text{optimizer}}$) [bytes]
    OptimizerStepIndex: NotationSymbol  # position of one optimizer update in the sequence of training steps ($t$)
    OptimizerStepTime: NotationSymbol  # time the optimizer takes to apply one parameter update ($T_{\text{update}}$) [s]
    OptimizerUpdateCount: NotationSymbol  # number of optimizer updates a training run takes, the horizon over which a convergence bound averages ($T$)
    OptimumMissFactor: NotationSymbol  # factor by which a chosen checkpoint interval or stride differs from its optimum, the argument of the flat-minimum overhead ratio ($x$) [dimensionless]
    OrderedTestRank: NotationSymbol  # rank of one test's p-value in the list ordered from smallest to largest, as Benjamini-Hochberg uses it ($r$)
    OrientationTolerance: NotationSymbol  # largest angular orientation error a task accepts ($\theta_{\text{tol}}$) [rad]
    OrphanSweepInterval: NotationSymbol  # time between successive runs of a sweep that deletes environments no obligation owns ($T$) [s]
    OscillationPriorStateDigest: NotationSymbol  # digest of the canonicalized workspace projection captured two observation steps before the current one ($h_{\text{state},t-2}$)
    OscillatorAcceleration: NotationSymbol  # acceleration of the undamped harmonic oscillator used to compare discretization schemes ($\ddot{q}$) [varies]
    OscillatorNaturalFrequency: NotationSymbol  # natural angular frequency of the undamped harmonic oscillator used to compare discretization schemes ($\omega$) [rad/s]
    OscillatorPosition: NotationSymbol  # position coordinate of the undamped harmonic oscillator used to compare discretization schemes ($q$) [varies]
    OscillatorStepVelocity: NotationSymbol  # velocity of the undamped test oscillator at one step of the semi-implicit (symplectic) Euler update ($v$) [varies]
    OscillatorVelocity: NotationSymbol  # velocity of the undamped harmonic oscillator used to compare discretization schemes, the time derivative of its position ($\dot{q}$) [varies]
    OutcomeAlphabet: NotationSymbol  # finite set of values a discrete random variable can take, over which its entropy sums ($\mathcal{X}$)
    OutcomeCheckPassProbability: NotationSymbol  # probability that a candidate passes the outcome-verification predicate of an admission route ($p_{\text{out}}$) [dimensionless]
    OutcomeCheckSlotTime: NotationSymbol  # expected verifier slot time of the outcome-verification predicate for a record that reaches it ($T_{\text{out}}$) [s]
    OutcomePairPrivacyLoss: NotationSymbol  # privacy loss of a mechanism at one released outcome for one ordered adjacent pair of datasets, the log ratio of that outcome's probabilities on the first dataset and on the second ($\mathcal{L}_{\text{priv},(X,X')}^{(o)}$) [dimensionless]
    OutcomeStandardDeviation: NotationSymbol  # standard deviation of one user's outcome metric under the common-variance Normal approximation of an A/B test ($\sigma$) [dimensionless]
    OutgoingChannelCommand: NotationSymbol  # channel command of the source being retired at a bumpless transfer ($u_{\text{prior}}$) [varies]
    OutgoingCommandDerivative: NotationSymbol  # time derivative of the outgoing source command during a bumpless transfer ($\dot{u}_{\text{prior}}$) [varies]
    OutputBearingDamping: NotationSymbol  # viscous damping coefficient of a joint's output bearings ($D_{\text{joint}}$) [N*m*s/rad]
    OutputChannel: NotationSymbol  # one measured output of a multi-output system, a component of the measurement vector ($y$) [varies]
    OutputChannelCount: NotationSymbol  # number of channels, one per filter, in the feature map a convolution layer produces ($C_{\text{out}}$)
    OutputChannelIndex: NotationSymbol  # position of one output channel of a convolution layer ($k$)
    OutputComponentFunction: NotationSymbol  # one scalar component of the output map of a nonlinear system, a function of the state ($h$)
    OutputDimension: NotationSymbol  # number of measured outputs of a state-space model, the length of the output vector ($p$)
    OutputFeatureMapHeight: NotationSymbol  # number of output rows a convolution produces ($H_{\text{out}}$)
    OutputFeatureMapWidth: NotationSymbol  # number of output columns a convolution produces ($W_{\text{out}}$)
    OutputIndex: NotationSymbol  # position of one measured output channel of a multi-output system, from one to the output dimension ($\ell$)
    OutputLayerBiasVector: NotationSymbol  # bias vector of a network's final linear output layer ($\mathbf{b}_{\text{out}}$)
    OutputLayerWeightMatrix: NotationSymbol  # weight matrix of a network's final linear output layer, applied after the last hidden activation ($\mathbf{W}_{\text{out}}$)
    OutputLength: NotationSymbol  # number of tokens a call generates ($K$)
    OutputLengthStandardDeviation: NotationSymbol  # standard deviation of the number of output tokens per request across a workload ($\sigma$) [tokens]
    OutputLimit: NotationSymbol  # maximum number of tokens the runtime lets a call generate ($K_{\max}$)
    OutputLogit: NotationSymbol  # raw score a model outputs for one class before the softmax ($z$) [dimensionless]
    OutputNodeAdjoint: NotationSymbol  # adjoint of the output node of a computational graph, which reverse mode seeds with one ($\bar{v}_W$) [dimensionless]
    OutputNodePrimal: NotationSymbol  # value of the final output node of a computational graph, the function value ($v_W$) [dimensionless]
    OutputProjectionOperations: NotationSymbol  # floating-point operations of the output projection that maps attention output back to the hidden width, for one layer ($O_{\text{out}}$) [FLOP]
    OutputQuantizationScale: NotationSymbol  # quantization scale of a layer's output activation tensor ($s_Y$) [varies]
    OutputRelativeDegree: NotationSymbol  # relative degree of an output of a control-affine system, the number of differentiations before the input appears ($r$)
    OutputToken: NotationSymbol  # token a language model generates at one output position ($y$)
    OutputTokenPrefix: NotationSymbol  # output tokens generated before the current position, the prefix the next token conditions on ($y_{<t}$)
    OutputTokenPrice: NotationSymbol  # price of one generated output token ($c_{\text{out}}$) [USD/token]
    OutputTokenSequence: NotationSymbol  # sequence of tokens one call generates, positions one through the output length ($y_{1:K}$)
    OutputTorqueCeiling: NotationSymbol  # largest joint output torque a drive can deliver under its present current limit and thermal state ($\tau_{\text{out,max}}$) [N*m]
    OutstandingItemCount: NotationSymbol  # number of admitted items still holding a credit, from admission to their capacity-release boundary ($N_{\text{out}}$)
    OverheadCameraEmbedding: NotationSymbol  # camera-identifier embedding of the overhead scene camera ($\mathbf{e}_{\text{overhead}}$)
    OverlappedBackwardCommunicationTime: NotationSymbol  # time of the backward pass plus gradient communication when each layer's AllReduce overlaps the next layer's backward computation ($T_{\text{pipelined}}$) [s]
    OverlappedTime: NotationSymbol  # communication or coordination time per step hidden behind computation ($T_{\text{overlap}}$) [s]
    OverlappingGaussianSet: NotationSymbol  # set of Gaussian primitives that overlap one pixel, ordered front to back by depth ($\mathcal{N}$)
    OversubscriptionRatio: NotationSymbol  # ratio of endpoint-facing to upward bandwidth at a fabric tier, written R:1 ($R$) [dimensionless]
    OwnershipAcquisitionTime: NotationSymbol  # time a replacement worker takes to acquire a trajectory's ownership fence before resuming it ($T_{\text{own}}$) [s]
    PackageDecouplingCapacitance: NotationSymbol  # decoupling capacitance mounted on the chip package ($C_{\text{pkg}}$) [F]
    PackageInductance: NotationSymbol  # lumped inductance of the package power-delivery path to a processor die ($L_{\text{pkg}}$) [H]
    PackageResistance: NotationSymbol  # lumped resistance of the package power-delivery path to a processor die ($R_{\text{pkg}}$) [Ohm]
    PackageThermalCapacitance: NotationSymbol  # lumped heat capacity of a die and package, energy per degree of temperature rise ($C_{\text{th}}$) [J/K]
    PackageThermalTimeConstant: NotationSymbol  # characteristic time for a die and package to heat or cool after a power step, junction-to-ambient thermal resistance times thermal capacitance ($\tau_{\text{th}}$) [s]
    PackedExampleIndex: NotationSymbol  # position of one example among the examples packed into a training buffer ($k$)
    PackedExampleLength: NotationSymbol  # number of tokens one packed example occupies in a training buffer ($s$)
    PackedExampleOffset: NotationSymbol  # zero-based buffer position at which one packed example begins ($b$)
    PacketBuffer: NotationSymbol  # one packet buffer or virtual-lane queue at a switch port, a vertex of a buffer dependency graph ($b$)
    PaddingWasteFraction: NotationSymbol  # fraction of the compute static batching performs that is padding for requests that already finished ($W$) [dimensionless]
    PageRankDampingFactor: NotationSymbol  # damping factor of personalized PageRank, the weight on following graph edges rather than teleporting to the personalization vector ($d$) [dimensionless]
    PageRankScoreVector: NotationSymbol  # personalized PageRank score vector over the symbols of a repository graph, the unique fixed point of the damped update ($\mathbf{r}$)
    PairedAcceptanceDifference: NotationSymbol  # difference between a candidate's and a baseline's acceptance probabilities over the task population, measured on the same tasks ($\Delta$) [dimensionless]
    PairedDifferenceLowerLimit: NotationSymbol  # lower endpoint of a confidence interval for the paired acceptance difference between a candidate and its baseline ($L_{\Delta}$) [dimensionless]
    PairedDifferenceUpperLimit: NotationSymbol  # upper endpoint of a confidence interval for the paired acceptance difference between a candidate and its baseline ($U_{\Delta}$) [dimensionless]
    PairedOutcomeCellCount: NotationSymbol  # number of paired tasks in one cell of the baseline-by-candidate outcome table, indexed by the baseline outcome then the candidate outcome, 1 for accepted and 0 for rejected ($n$)
    PairedRandomVariable: NotationSymbol  # second arbitrary real-valued random variable, paired with the first in a general probability inequality ($Y$)
    PairedTaskIndex: NotationSymbol  # index of a second periodic task compared with task i, such as the task whose period is longer ($j$)
    PairedTimeInstant: NotationSymbol  # second point in continuous time at which a random process is evaluated, paired with t in a covariance kernel ($s$) [s]
    PairedVerdictCorrelation: NotationSymbol  # estimated correlation (phi coefficient) between the baseline and candidate verdicts on the same task, with a continuity-corrected numerator when positive ($\hat{\phi}$) [dimensionless]
    PairingLagCeiling: NotationSymbol  # largest observation-to-action pairing lag that keeps the resulting displacement within the task clearance ($\Delta t_{\text{lag,max}}$) [s]
    PairwiseCoordinationCoefficient: NotationSymbol  # coordination overhead each pair of workers adds, as a fraction of serial time, in an assumed quadratic overhead model ($\beta_{\text{pair}}$) [dimensionless]
    PairwiseMaskSecret: NotationSymbol  # shared random mask vector two clients derive from a common secret, which one adds and the other subtracts so that it cancels in the secure-aggregation sum ($\mathbf{s}$)
    ParallelEnvironmentCount: NotationSymbol  # number of simulated environments stepped in parallel to collect one on-policy rollout ($n_{\text{env}}$)
    ParallelFraction: NotationSymbol  # share of the work that divides among workers ($f_{\text{parallel}}$) [dimensionless]
    ParallelLoopCount: NotationSymbol  # number of loops of a loop nest selected to run in parallel ($k_{\text{parallel}}$)
    ParallelRigCount: NotationSymbol  # number of physical rigs that run trials of one learning campaign in parallel ($N_{\text{rig}}$)
    ParameterPayload: NotationSymbol  # bytes of the full parameter set at the communication precision, the payload a fully sharded schedule gathers ($M_{\text{param}}$) [bytes]
    ParameterRandomizationRanges: NotationSymbol  # ranges over which domain randomization samples each simulator parameter, recorded in the simulation provenance with their source ($\mathcal{P}$)
    ParameterScaleCoefficient: NotationSymbol  # fitted coefficient of the parameter-count term of a two-term scaling law in parameters and tokens ($A$) [dimensionless]
    ParameterScalingExponent: NotationSymbol  # power-law exponent at which a two-term scaling-law loss falls as parameter count grows ($\alpha$) [dimensionless]
    ParameterServerCount: NotationSymbol  # number of parameter servers across which a parameter-server system shards the model state ($K_{\text{srv}}$)
    ParameterSpace: NotationSymbol  # set of all values the parameters of a model may take, over which a posterior is integrated ($\Theta$)
    ParameterSpaceDirection: NotationSymbol  # fixed direction in a model's parameter space onto which a gradient estimate is projected to study its variance as a scalar ($u$)
    ParameterStreamingTime: NotationSymbol  # time to stream model weights that do not stay resident into the compute units for one evaluation ($t_{\text{stream}}$) [s]
    ParameterVelocity: NotationSymbol  # rate of change of a model's parameters under continuous-time gradient flow, the parameter velocity ($\dot{\theta}$)
    ParameterizedPredictor: NotationSymbol  # trained model written as a function of its parameters, where training optimizes over those parameters ($f_{\theta}$)
    Parameters: NotationSymbol  # number of trainable parameters in a model ($P$)
    ParametricProposalDistribution: NotationSymbol  # proposal distribution from a parametric family, selected by its parameter vector in the cross-entropy method ($q_{\boldsymbol{\psi}}$)
    ParasiticLoopInductance: NotationSymbol  # unintended inductance of the current loop that feeds an accelerator's core rail, which turns current slew into voltage droop ($L_{\text{parasitic}}$) [H]
    ParentCapabilityGrant: NotationSymbol  # capability grant from which a narrower child grant is derived ($\Gamma_{\text{parent}}$)
    ParentFreeBalance: NotationSymbol  # unreserved monetary balance of the delegating parent node's budget account ($F_{P}$) [USD]
    ParentNodeIndex: NotationSymbol  # index of a parent node, one of the inputs to the node being evaluated ($k$)
    ParentReservedFunds: NotationSymbol  # monetary funds the delegating parent node's budget account withholds, including allocations delegated to its children ($R_{P}$) [USD]
    ParentSettledCharges: NotationSymbol  # gross monetary charges settled into the delegating parent node's budget account ($E_{P}$) [USD]
    ParetoMinimumDemand: NotationSymbol  # smallest demand a Pareto tail model allows, its scale parameter ($x_{\min}$) [varies]
    ParetoTailExponent: NotationSymbol  # tail index of a Pareto model, the exponent at which the probability of exceeding a demand falls with that demand ($\alpha_{\text{tail}}$) [dimensionless]
    ParetoTaskDemand: NotationSymbol  # unconstrained demand of a task, such as its step count, modeled as a continuous Pareto random variable ($X$) [varies]
    ParticipatingClientCount: NotationSymbol  # number of clients whose updates one federated round aggregates ($N_{\text{clients}}$)
    PartnerBitIndex: NotationSymbol  # bit position whose mask pairs each rank with its partner, rank index XOR two to this power, in one round of recursive halving-doubling ($k$)
    PassAtKEstimate: NotationSymbol  # unbiased estimate of pass@k for one task from a set of generated candidates, one minus the share of size-k subsets that contain no passing candidate ($\widehat{\operatorname{pass@}k}$) [dimensionless]
    PassedTrialCount: NotationSymbol  # number of executed trials scored as passing ($N_{\text{pass}}$)
    PassingRunCount: NotationSymbol  # number of one task's fully graded repeated runs that pass, the count from which an all-runs-pass consistency estimate draws its passing subsets ($s$)
    PassingSampleCount: NotationSymbol  # number of the candidates generated for one task in a pass@k estimate that pass the test criterion ($c$)
    PassiveModelPrediction: NotationSymbol  # prediction a passively deployed model outputs for an input ($\hat{Y}$)
    PastSynchronizationStepIndex: NotationSymbol  # summation index over past synchronization steps in a clock servo's integral term ($j$)
    PatchColumnIndex: NotationSymbol  # column of one patch in the grid of patches a vision transformer cuts an image into ($w$)
    PatchRowIndex: NotationSymbol  # row of one patch in the grid of patches a vision transformer cuts an image into ($h$)
    PatchTokenCount: NotationSymbol  # number of patch tokens a vision transformer cuts one image into, the image area divided by the squared patch size ($N$)
    PathAcceleration: NotationSymbol  # second time derivative of the path coordinate, the acceleration along a geometric path ($\ddot{s}$) [1/s^2]
    PathAccelerationInput: NotationSymbol  # path acceleration held constant over one grid interval, the control input of the discretized path-parameterization problem ($u$) [1/s^2]
    PathAccelerationLowerBound: NotationSymbol  # smallest path acceleration the joint torque and acceleration limits admit at a given path position and path velocity ($\ddot{s}_{\min}$) [1/s^2]
    PathAccelerationTorqueCoefficient: NotationSymbol  # vector multiplying path acceleration in the path-parameterized manipulator torque equation ($\mathbf{a}_{\tau}$) [N*m*s^2]
    PathAccelerationUpperBound: NotationSymbol  # largest path acceleration the joint torque and acceleration limits admit at a given path position and path velocity ($\ddot{s}_{\max}$) [1/s^2]
    PathCoordinate: NotationSymbol  # scalar coordinate along a geometric path, from zero at its start to one at its end, that time parameterization maps to time ($s$) [dimensionless]
    PathDelayAsymmetry: NotationSymbol  # difference between the forward and reverse one-way delays of a clock-synchronization exchange, which biases the estimated clock offset by half its value ($\Delta t_{\text{asym}}$) [s]
    PathGridPointCount: NotationSymbol  # number of grid points along a geometric path at which a time-parameterization algorithm checks its constraints ($N_{\text{grid}}$)
    PathVelocity: NotationSymbol  # rate at which the path coordinate advances in time, the speed along a geometric path that time parameterization chooses ($\dot{s}$) [1/s]
    PathVelocitySquaredTorqueCoefficient: NotationSymbol  # vector multiplying squared path velocity in the path-parameterized manipulator torque equation ($\mathbf{b}_{\tau}$) [N*m*s^2]
    PayloadGravityTorque: NotationSymbol  # torque the weight of a carried payload exerts about one joint axis ($\tau_{g}$) [N*m]
    PayloadLeverArm: NotationSymbol  # distance from a joint axis to the center of gravity of the mass it carries, the arm of the gravity moment ($r$) [m]
    PcieLinkFailureRate: NotationSymbol  # rate at which the PCIe connection between one GPU and its host fails ($\lambda_{\text{PCIe}}$) [failures/hour]
    PdnBranchCurrent: NotationSymbol  # current through one lumped source-path branch of a power delivery network, distinct from the load current local decoupling supplies ($I_{\text{branch}}$) [A]
    PdnBranchResistance: NotationSymbol  # parasitic resistance of one lumped source-path branch of a chip's power delivery network ($R_{\text{PDN}}$) [Ohm]
    PdnTargetImpedance: NotationSymbol  # impedance a power delivery network must stay below across its frequency band, allowed excursion over the largest current step ($Z_{\text{target}}$) [Ohm]
    PeakBridgeJerk: NotationSymbol  # largest jerk of one joint over a seam bridge, found at its endpoints or its interior stationary point ($j_{\text{peak}}$) [rad/s^3]
    PeakContactForce: NotationSymbol  # largest contact force reached during one contact event ($F_{\text{peak}}$) [N]
    PeakCopperLossPower: NotationSymbol  # copper loss a winding dissipates during a burst at peak current ($P_{\text{peak}}$) [W]
    PeakCurrent: NotationSymbol  # largest motor current permitted during a short burst ($I_{\text{peak}}$) [A]
    PeakMemoryBandwidth: NotationSymbol  # highest memory bandwidth the hardware specification states for a device ($\text{BW}_{\text{peak}}$) [bytes/s]
    PeakPenetrationDepth: NotationSymbol  # largest penetration depth reached during an impact, where the approach velocity has fallen to zero ($\delta_{\max}$) [m]
    PeakRate: NotationSymbol  # highest operation rate the hardware can execute, as specified for a given precision ($R_{\text{peak}}$) [FLOP/s]
    PeakRegenerativePower: NotationSymbol  # largest electrical power a regenerative stop returns to the DC bus ($P_{\text{regen, peak}}$) [W]
    PeakRotorAngularSpeed: NotationSymbol  # largest motor rotor angular speed from which a regenerative stop begins ($\omega_{\max}$) [rad/s]
    PeakSlipRatio: NotationSymbol  # slip ratio at which the longitudinal friction coefficient reaches its maximum ($\lambda_{\text{peak}}$) [dimensionless]
    PeakStopDeceleration: NotationSymbol  # largest deceleration a stopping profile reaches, at the midpoint for the quintic polynomial stop ($a_{\text{peak}}$) [m/s^2]
    PeakTorque: NotationSymbol  # largest transient joint torque permitted during short bursts ($\tau_{\text{peak}}$) [N*m]
    PeakTractionCoefficient: NotationSymbol  # largest longitudinal friction coefficient a tire reaches over slip ratio, setting the peak traction ($\mu_{\text{peak}}$) [dimensionless]
    PearsonFitStatistic: NotationSymbol  # chi-square goodness-of-fit statistic, the sum over categories of squared differences between observed and expected counts divided by the expected counts ($\chi^2$) [dimensionless]
    PeerReviewerIndex: NotationSymbol  # index of the other reviewer of a pair whose errors are compared with reviewer j's ($i$)
    PeerToolCallIndex: NotationSymbol  # index of a second tool call in the same batch, checked against call i for noninterference ($j$)
    PenaltyContactDamping: NotationSymbol  # damping coefficient of a penalty (Kelvin-Voigt) contact model, the force per unit penetration rate added to the spring force ($c$) [N*s/m]
    PenaltyContactStiffness: NotationSymbol  # stiffness of the compliant penalty spring a physics simulator uses to compute contact force from penetration depth ($k_{\text{pen}}$) [N/m]
    PenaltyPenetrationDepth: NotationSymbol  # largest penetration depth a penalty-contact simulator allows at a given approach speed, which fixes its effective spring stiffness ($\delta_{\text{pen}}$) [m]
    PenaltyRestitutionDuration: NotationSymbol  # duration of the restitution phase of a simulated penalty contact, from peak penetration to separation ($\Delta t_{\text{res}}$) [s]
    PendulumNaturalFrequency: NotationSymbol  # natural frequency of the linear inverted pendulum, the square root of gravitational acceleration over center-of-mass height, at which an unrecovered lean grows exponentially ($\omega_0$) [rad/s]
    PenetrationDepth: NotationSymbol  # depth by which one body penetrates or compresses another at a contact, measured along the contact normal ($\delta$) [m]
    PenetrationRate: NotationSymbol  # time rate of change of the penetration depth at a contact ($\dot{\delta}$) [m/s]
    PenultimateLayerWidth: NotationSymbol  # number of units in the layer before a network's last layer ($n_{N_L-1}$)
    PerDeviceComputeTime: NotationSymbol  # compute time each device spends per step after the work is divided evenly across the devices of a job ($T_{\text{compute,local}}$) [s]
    PerDeviceWeightReadTime: NotationSymbol  # time one device takes to read its share of the active weights from memory in one decode step ($T_{\text{read}}$) [s]
    PerExperimentComputeCost: NotationSymbol  # compute cost of one experiment ($C_{\text{compute,per-exp}}$) [USD]
    PerExperimentEngineeringCost: NotationSymbol  # engineering cost of one experiment ($C_{\text{engineer,per-exp}}$) [USD]
    PerFrameDetectionProbability: NotationSymbol  # probability that a person detector reports a person who is in view in one camera frame ($p_{\text{frame}}$) [dimensionless]
    PerGpuNetworkFailureRate: NotationSymbol  # share of a node's network-path failure rate charged to each of its GPUs ($\lambda_{\text{network/GPU}}$) [failures/hour]
    PerGpuPowerFailureRate: NotationSymbol  # share of a node's power-supply failure rate charged to each of its GPUs, counting only supplies whose single failure stops the node ($\lambda_{\text{power/GPU}}$) [failures/hour]
    PerGpuTotalFailureRate: NotationSymbol  # failure rate of one GPU's whole failure domain, the sum of the rates of the subsystems in series with it ($\lambda_{\text{total/GPU}}$) [failures/hour]
    PerStepCostBound: NotationSymbol  # largest incremental task cost a policy can incur in one step after it has drifted into states the expert never visited ($C_{\max}$) [dimensionless]
    PerStepErrorRate: NotationSymbol  # probability that one unchecked turn of a trajectory is wrong ($\epsilon$) [dimensionless]
    PerStreamTokenRate: NotationSymbol  # number of output tokens one decode stream emits per second ($R$) [tokens/s]
    PerTaskRunCount: NotationSymbol  # number of independent runs an evaluation gives each sampled task ($R$)
    PerTestSignificanceLevel: NotationSymbol  # significance threshold applied to each test of a family after a multiple-testing correction such as Bonferroni or Sidak ($\alpha_{\text{sig}}'$) [dimensionless]
    PerTokenDecodeTime: NotationSymbol  # time to generate one output token on a single stream ($t_{\text{decode}}$) [s]
    PerTokenKlEstimate: NotationSymbol  # non-negative per-token estimate of the Kullback-Leibler divergence of the current policy from the reference policy ($\hat{\mathcal{D}}$) [nats]
    PerTokenMaskingTime: NotationSymbol  # time constrained decoding adds to each output token to compute and apply the grammar mask ($t_{\text{mask}}$) [s]
    PerTokenPrefillTime: NotationSymbol  # time prefill spends per prompt token at the compute-bound floor, about two operations per parameter at peak rate ($t_{\text{prefill}}$) [s]
    PerWorkerBatchSize: NotationSymbol  # number of examples one worker contributes to one optimizer update, including gradient accumulation ($B_{\text{per-worker}}$)
    PerceptionDepthCovariance: NotationSymbol  # covariance of the three-dimensional position error of a point lifted from stereo, lidar or depth data ($\boldsymbol{\Sigma}_{\text{depth}}$) [m^2]
    PerceptionDispatchTime: NotationSymbol  # time to package a perception output with its covariance and dispatch it to consumers between processes ($t_{\text{IPC}}$) [s]
    PerceptionErrorScale: NotationSymbol  # order-of-magnitude scale of perception position error: travel during the observation age plus stereo depth noise plus extrinsic error ($\epsilon_{\text{scale}}$) [m]
    PeriodicPeakWindingTemperature: NotationSymbol  # highest winding temperature reached in the periodic steady state of a burst schedule ($T_{\text{peak}}$) [degC]
    PeriodicTaskCount: NotationSymbol  # number of periodic tasks in a real-time task set ($n$)
    PermanentMagnetFluxLinkage: NotationSymbol  # magnetic flux from the rotor magnets that links each stator winding ($\lambda_{\text{pm}}$) [V*s]
    PermissionLoopPeriod: NotationSymbol  # period of the permission loop, the longest a hazard may wait before it is checked ($T_{\text{tick}}$) [s]
    PermissionPathKernel: NotationSymbol  # conditional distribution of the admitted command given the state and the policy's proposal, set by the permission-path parameters ($G_\kappa$)
    PermissionPathParameters: NotationSymbol  # configuration of the permission path in one rollout, such as lease durations and barrier thresholds ($\kappa$)
    PermissionTickWaitTime: NotationSymbol  # time a staged plan waits for the next discrete tick of the permission loop ($t_{\text{wait}}$) [s]
    PermissionTickWcet: NotationSymbol  # declared worst-case execution time of the permission tick running in isolation, the falsification threshold a loaded cycle must not exceed ($T_{\text{wcet}}$) [s]
    PermittedActions: NotationSymbol  # set of tools the task may call, with the limits on each ($\mathcal{A}_{\text{perm}}$)
    PermittedUsePassProbability: NotationSymbol  # probability that a candidate passes the permitted-use predicate (provenance, licensing, access rights, split compatibility) ($p_{\text{use}}$) [dimensionless]
    PersonApproachTravel: NotationSymbol  # distance a person travels toward the robot during the robot's reaction and stopping time ($S_h$) [m]
    PersonPositionUncertainty: NotationSymbol  # position uncertainty of the person from the sensing system ($Z_d$) [m]
    PersonalizationPenaltyWeight: NotationSymbol  # weight of the penalty that keeps a personalized client model near the global model ($\lambda_{\text{reg}}$) [dimensionless]
    PersonalizationVector: NotationSymbol  # nonnegative vector over the symbols of a repository graph, summing to one, toward which personalized PageRank teleports ($\mathbf{p}$)
    PerturbationNoiseStandardDeviation: NotationSymbol  # standard deviation of the isotropic Gaussian noise that perturbs the clean signal in Tweedie's formula ($\sigma$) [varies]
    PerturbationNormOrder: NotationSymbol  # order of the vector norm that measures a perturbation, such as 2 for Euclidean length or infinity for the largest coordinate ($p$) [dimensionless]
    PerturbedPose: NotationSymbol  # rigid-body pose after a small random twist perturbation is composed with the mean pose in its local frame ($\mathbf{T}'$)
    PerturbedSignalDimension: NotationSymbol  # number of coordinates of the signal vector in the Gaussian perturbation model of Tweedie's formula ($D$)
    PerturbedSignalVector: NotationSymbol  # signal vector in the Gaussian perturbation model of Tweedie's formula, the noisy measurement, with subscript 0 for the clean value it perturbs ($\mathbf{x}$)
    PfcWatchdogTimeout: NotationSymbol  # pause duration after which a switch PFC watchdog declares a stalled queue and triggers recovery ($T_{\text{watchdog}}$) [s]
    PhaseInductance: NotationSymbol  # inductance of one motor phase winding ($L_{\text{phase}}$) [H]
    PhaseJoulePower: NotationSymbol  # resistive heating power dissipated in one motor phase winding, the RMS phase current squared times the phase resistance ($P_{\text{phase}}$) [W]
    PhaseMargin: NotationSymbol  # phase margin of a feedback loop: the additional phase lag at gain crossover that would bring the loop to instability ($\Phi_m$) [deg]
    PhaseResistance: NotationSymbol  # resistance of one motor phase winding ($R_{\text{phase}}$) [Ohm]
    PhotodiodeQuantumEfficiency: NotationSymbol  # fraction of incident photons a photodiode converts into collected electrons, as a function of wavelength ($\eta_{\text{QE}}$) [dimensionless]
    PhysicalPlantDynamics: NotationSymbol  # dynamics of the real machine and environment, the operator a physical trial evaluates a policy against ($f_{\text{real}}$)
    PhysicalRestoreTime: NotationSymbol  # time to return the machine and fixture to the trial's starting configuration, part of the reset latency ($t_{\text{physical\_restore}}$) [s]
    PhysicalReturnLevel: NotationSymbol  # return level of the physical machine, bounded by the simulated return level plus the transfer discount ($z_{T, \text{real}}$) [varies]
    PhysicalStateTrajectory: NotationSymbol  # state trajectory the physical machine produces under declared conditions and a control input history ($s_{\text{real}}$) [varies]
    PhysicalStlRobustness: NotationSymbol  # quantitative robustness of a signal temporal logic formula on the physical machine's trajectory ($\rho_{\text{real}}$) [varies]
    PhysicalTrajectorySet: NotationSymbol  # set of trajectories the physical machine can produce under given declared conditions ($\mathcal{S}_{\text{real}}$)
    PhysicalWorldState: NotationSymbol  # ground-truth state of the machine, its environment and their contacts at step t ($s$)
    PidDerivativeGain: NotationSymbol  # feedback gain on the velocity error, the damping of a PD or impedance controller ($K_d$) [N*s/m]
    PidIntegralGain: NotationSymbol  # feedback gain on the accumulated error in a PI or PID controller ($K_i$) [varies]
    PidProportionalGain: NotationSymbol  # feedback gain on the position error, the stiffness of a PD or impedance controller ($K_p$) [N/m]
    PidProportionalGainMatrix: NotationSymbol  # proportional gain matrix of multi-joint PD or PID feedback, the stiffness acting on the position error ($\mathbf{K}_p$)
    PipelineBubbleFraction: NotationSymbol  # fraction of pipeline device time spent idle while the pipeline fills and drains ($b_{\text{pipe}}$) [dimensionless]
    PipelineBufferMemory: NotationSymbol  # total host memory the prefetch, processing and transfer buffers of an input pipeline require ($M_{\text{required}}$) [bytes]
    PipelineClockSkew: NotationSymbol  # difference in clock arrival time between the launching and capturing registers of a pipeline stage ($t_{\text{skew}}$) [s]
    PipelineFlushDelay: NotationSymbol  # time a processor takes to flush its pipeline when it takes an interrupt ($t_{\text{hw\_pipe}}$) [s]
    PipelineIterationTime: NotationSymbol  # steady-state interval between batches an overlapped input pipeline delivers, the largest stage time ($T_{\text{iteration}}$) [s]
    PipelineParallelDegree: NotationSymbol  # number of pipeline stages a model's layers are split across ($N_{\text{PP}}$)
    PipelineTimeSlot: NotationSymbol  # one equal-length time slot of a pipeline schedule grid, written with its slot number as a subscript ($t$)
    PipelinedExecutionTime: NotationSymbol  # execution time of a workload whose data movement and computation overlap, set by the slower of the two plus fixed latency ($T_{\text{pipelined}}$) [s]
    PistonArea: NotationSymbol  # area of a hydraulic cylinder's piston face ($A$) [m^2]
    PixelColumnCoordinate: NotationSymbol  # horizontal pixel coordinate, the column, of a point in the image ($u$) [pixels]
    PixelLastEventTime: NotationSymbol  # time at which an event-camera pixel last emitted an event, the reference for its next contrast comparison ($t_{\text{last}}$) [s]
    PixelOpacityContribution: NotationSymbol  # opacity a Gaussian contributes at one pixel during alpha blending, its stored opacity weighted by its projected footprint at that pixel ($\alpha'$) [dimensionless]
    PixelRowCoordinate: NotationSymbol  # vertical pixel coordinate, the row, of a point in the image ($v$) [pixels]
    PixelRowStandardDeviation: NotationSymbol  # standard deviation of the measured pixel row coordinate ($\sigma_v$) [pixels]
    PixelStandardDeviation: NotationSymbol  # standard deviation of a measured image position, in pixels ($\sigma_u$) [pixels]
    PlaintextInput: NotationSymbol  # plaintext input that is encrypted before computation over ciphertexts ($m$)
    PlanAction: NotationSymbol  # one action of an execution plan, as the expected irreversible loss sums over the plan's actions ($a$)
    PlanarHomographyMatrix: NotationSymbol  # three-by-three homography mapping points of a planar calibration model to image points ($\mathbf{H}$)
    PlanarPositionEmbedding: NotationSymbol  # position embedding of an image patch built from a row component and a column component, a function of the patch's row and column ($\mathbf{E}_{\text{pos}}^{2D}$)
    PlanarSpecialEuclideanGroup: NotationSymbol  # group of rigid-body transformations in the plane, rotations combined with translations, the configuration space of a planar mobile base ($SE(2)$)
    PlanckConstant: NotationSymbol  # Planck constant, relating a photon's energy to its frequency ($h$) [J*s]
    PlanningOperator: NotationSymbol  # action of a symbolic planner defined by its preconditions, add list and delete list, as in STRIPS ($o$)
    PlanningStageLatency: NotationSymbol  # time the planning stage of a blocking sensor-to-actuation path takes, including its queueing and execution delay ($T_{\text{plan}}$) [s]
    PlanningStartInstant: NotationSymbol  # instant at which the planner samples state and begins a replacement ($t_{\text{plan}}$) [s]
    PlantDynamicsFunction: NotationSymbol  # vector field of a plant's equations of motion, giving the state rate from the state, control input and plant parameters ($\mathbf{f}$)
    PlantDynamicsParameters: NotationSymbol  # physical parameters of a plant's dynamics, such as resistances, stiffness and friction, that drift over operating time ($\theta_{\text{plant}}$)
    PlantOutcomeFunction: NotationSymbol  # nonlinear map from an action applied to the plant to the physical outcome it produces ($g$)
    PlantParameters: NotationSymbol  # physical and hardware properties of the machine in one trial or rollout, such as tolerances, wear, payload mass, state of charge and calibration ($m$)
    PlantStateAcceleration: NotationSymbol  # second time derivative of the mechanical plant state, the acceleration a force produces in a double-integrator plant ($\ddot{\mathbf{x}}$)
    PlantStateComponentCount: NotationSymbol  # number of sampled plant-state components an authority log entry records, fixed by the static manifest ($N_s$)
    PlantStateDimension: NotationSymbol  # number of components of the plant state vector ($n$)
    PlantStateRate: NotationSymbol  # time derivative of the mechanical plant state ($\dot{\mathbf{x}}$)
    PlantStateSpace: NotationSymbol  # set of all states the mechanical plant can occupy, positions with velocities ($\mathcal{X}$)
    PlantTransitionDensity: NotationSymbol  # distribution of the next state given the state and the admitted command, for a plant with the given parameters ($p_m$)
    PlantTransitionKernel: NotationSymbol  # stochastic transition kernel of the physical plant: the distribution of the next state given the state, the applied command and the trial's plant and environment parameters ($\mathcal{T}$)
    PlatformResourceCapacity: NotationSymbol  # capacity of one resource pool of a shared platform, the weight of that pool in platform utilization ($\text{Cap}$) [varies]
    PlatformResourceIndex: NotationSymbol  # index of one resource pool of a shared platform ($i$)
    PlatformTotalCost: NotationSymbol  # total cost of building and running a shared ML platform, including development, infrastructure and maintenance ($C_{\text{platform}}$) [USD]
    PlatformUtilization: NotationSymbol  # capacity-weighted average utilization of a shared platform's resources over an interval ($U_{\text{platform}}$) [dimensionless]
    PointClearanceMargin: NotationSymbol  # smallest distance a candidate point must keep from every stored obstacle point to pass the clearance predicate ($d_{\text{margin}}$) [m]
    PointToPointMessageSize: NotationSymbol  # size of one point-to-point message in the alpha-beta model ($n$) [bytes]
    PointToPointTransferTime: NotationSymbol  # time to send one point-to-point message of n bytes in the alpha-beta model ($T$) [s]
    PoisonedModel: NotationSymbol  # model trained on the union of the original training data and the attacker's poisoning data ($f_{\mathcal{S} \cup \mathcal{S}_{\text{poison}}}$)
    PoisoningDataset: NotationSymbol  # set of crafted or mislabeled examples an attacker adds to a training set ($\mathcal{S}_{\text{poison}}$)
    PoisoningTargetDataset: NotationSymbol  # validation or target dataset on which a poisoning attacker seeks to raise the trained model's loss ($\mathcal{S}_{\text{test}}$)
    PoissonSubintervalCount: NotationSymbol  # number of equal subintervals operating time is divided into when the Poisson model is derived as a limit of Bernoulli trials ($n$)
    Policy: NotationSymbol  # learned model viewed as a mapping from context or observation to actions, with parameters theta ($\pi_\theta$)
    PolicyAction: NotationSymbol  # action a policy chooses given the state it observes ($a$)
    PolicyActionStandardDeviation: NotationSymbol  # standard deviation, given the state, of one action component of a diagonal Gaussian policy, indexed by action dimension ($\sigma^{\pi}$) [varies]
    PolicyActionWindow: NotationSymbol  # action chunk written with its time window, the actions a policy predicts from one control step through the last step of its horizon ($\mathbf{A}_{t:t+H-1}$)
    PolicyCheckpoint: NotationSymbol  # policy checkpoint that generated one model invocation of a rollout, indexed by invocation ($v$)
    PolicyEnsembleCovariance: NotationSymbol  # covariance of the actions a policy ensemble proposes, each component normalized by its declared scale ($\boldsymbol{\Sigma}_{\pi}$)
    PolicyEntropy: NotationSymbol  # entropy of a stochastic policy's action distribution at a given state, the quantity maximum-entropy reinforcement learning rewards ($\mathcal{H}$) [nats]
    PolicyGaussianMean: NotationSymbol  # mean of the Gaussian over unbounded latent actions that a squashed-Gaussian policy network outputs for a state ($\boldsymbol{\mu}_{\theta}$)
    PolicyGaussianMeanComponent: NotationSymbol  # one component of the mean a squashed-Gaussian policy outputs for a state, for one action dimension ($\mu^{\pi}$) [dimensionless]
    PolicyGaussianStandardDeviation: NotationSymbol  # per-dimension standard deviation of the Gaussian over unbounded latent actions that a squashed-Gaussian policy network outputs, through its logarithm, for a state ($\boldsymbol{\sigma}_{\theta}$)
    PolicyGradientBaseline: NotationSymbol  # offset subtracted from the return in a policy-gradient estimator, a function of the interaction history that does not depend on the sampled action ($b$) [dimensionless]
    PolicyGradientEstimate: NotationSymbol  # likelihood-ratio estimate of the gradient of a policy's expected reward with respect to its parameters, computed from one sampled trajectory ($\widehat{g}$)
    PolicyHistoryActionReturn: NotationSymbol  # expected remaining return under the current policy after taking a given action at a given interaction history and then following that policy ($Q^{\pi}$) [dimensionless]
    PolicyHistoryAdvantage: NotationSymbol  # expected return of taking a given action at a given interaction history under the current policy minus that policy's expected remaining return at the same history ($A^{\pi}$) [dimensionless]
    PolicyImitationLoss: NotationSymbol  # imitation loss of a policy averaged over demonstration pairs labeled with one chosen action tap ($\mathcal{L}_{\text{policy}}$)
    PolicyLipschitzBound: NotationSymbol  # Lipschitz bound of a learned policy over a specified operating domain, the largest ratio of action change to state change ($L_{\text{lip}}$) [varies]
    PolicyObservedState: NotationSymbol  # state or observation a policy conditions its choice of action on ($s$)
    PolicyPassRate: NotationSymbol  # number of forward passes per second a learned policy can run, at most sustained memory bandwidth over weight memory when weight streaming dominates ($f$) [Hz]
    PolicyProbabilityRatio: NotationSymbol  # ratio of a sampled token's probability under the current policy to its probability under the behavior policy that generated it ($\rho$) [dimensionless]
    PolicyRemainingReturn: NotationSymbol  # expected remaining return under the current policy from a given interaction history at a decision step ($V^{\pi}$) [dimensionless]
    PolicyRequestLength: NotationSymbol  # number of tokens in the request a context policy produces at one decision step ($S_{\pi}$)
    PolicyStateActionDistribution: NotationSymbol  # distribution of state-action pairs a policy visits at a step, its trajectory marginal ($\rho_{\pi_\theta}$)
    PolicyTrajectoryDistribution: NotationSymbol  # distribution over state-action trajectories that a policy with parameters theta induces together with the environment dynamics ($p_\theta$)
    PolicyTti: NotationSymbol  # time to intervention (TTI): elapsed time from granting a policy authority to the first permission-path override or operator takeover ($T_{\text{TTI}}$) [s]
    PolicyUpdateStepTime: NotationSymbol  # wall-clock time of one training step of a reinforcement-learning loop, from one parameter update to the next ($T_{\text{step}}$) [s]
    PolyakAveragingCoefficient: NotationSymbol  # fraction of the active network's parameters blended into the target network's at each update in Polyak averaging ($\tau$) [dimensionless]
    PolynomialCoefficientVector: NotationSymbol  # vector stacking the coefficients of a trajectory polynomial, lowest power first ($\mathbf{a}$)
    PolynomialTrajectoryCoefficient: NotationSymbol  # coefficient of one power of time in a polynomial joint trajectory segment ($a$) [varies]
    PomdpActionSet: NotationSymbol  # set of actions available to the agent in a partially observable Markov decision process ($\mathsf{A}$)
    PomdpObservationFunction: NotationSymbol  # observation law of a partially observable Markov decision process, the probability of receiving an observation after an action leads to a given next state ($\mathsf{O}$)
    PomdpObservationSet: NotationSymbol  # set of possible observations in a partially observable Markov decision process ($\Omega$)
    PomdpRewardFunction: NotationSymbol  # reward function of a partially observable Markov decision process, the modeled objective signal for taking an action in a state ($\mathsf{R}$)
    PomdpStateSpace: NotationSymbol  # set of environment states of a partially observable Markov decision process, including remote state and concurrent activity the agent cannot see ($\mathsf{S}$)
    PomdpTransitionFunction: NotationSymbol  # transition law of a partially observable Markov decision process, giving the distribution of the next environment state after an action in a state ($\mathsf{T}$)
    PoolResidentMemory: NotationSymbol  # host resident memory of a copy-on-write sandbox pool, guest pages only ($M_{\text{pool}}$) [bytes]
    PooledAcceptanceProbability: NotationSymbol  # average of the acceptance probabilities of the two system versions a release comparison tests, used as the pooled variance estimate ($\bar{p}$) [dimensionless]
    PooledConversionRate: NotationSymbol  # conversion rate pooled over both variants of an A/B test, weighted by their sample sizes ($\hat{p}$) [dimensionless]
    PooledSampleApprovalRate: NotationSymbol  # observed approval share pooled over both compared groups, the sample-size-weighted mean of the two group approval rates ($\hat{p}$) [dimensionless]
    PopulationStabilityBinCount: NotationSymbol  # number of fixed bins over which the population stability index sums ($n_{\text{bins}}$)
    PopulationStabilitySmoothingConstant: NotationSymbol  # small constant added to the bin proportions of the population stability index before renormalizing, so an empty bin does not make a logarithm undefined ($\epsilon_{\text{smooth}}$) [dimensionless]
    PortCycleNetEnergy: NotationSymbol  # net mechanical energy a delayed teleoperation port delivers over one closed motion cycle ($\Delta E$) [J]
    PortEffort: NotationSymbol  # effort variable at a mechanical port, the force or torque that multiplies the port flow to give power ($\mathbf{F}$)
    PortFlow: NotationSymbol  # flow variable at a mechanical port, the velocity that multiplies the port effort to give power ($\dot{\mathbf{x}}$)
    PortInput: NotationSymbol  # input of a system at its power port, an effort variable whose inner product with the port output is the power supplied ($\mathbf{u}$)
    PortOutput: NotationSymbol  # output of a system at its power port, a flow variable whose inner product with the port input is the power supplied ($\mathbf{y}$)
    PortStoredEnergy: NotationSymbol  # energy stored in a physical system behind an interaction port, as a function of time ($E$) [J]
    PortVelocity: NotationSymbol  # velocity of the leader-follower port through which reflected contact force does work ($v$) [m/s]
    Pose: NotationSymbol  # rigid-body pose of a named frame, its rotation and position together ($\mathbf{T}$)
    PoseTwistPerturbation: NotationSymbol  # zero-mean Gaussian twist vector that perturbs a mean pose in its local frame to model pose uncertainty ($\boldsymbol{\epsilon}$)
    PoseTwistPerturbationMatrix: NotationSymbol  # four-by-four matrix form of a pose twist perturbation, exponentiated to perturb a mean pose ($\hat{\boldsymbol{\epsilon}}$)
    PositionComponent: NotationSymbol  # component of a Cartesian position along one coordinate axis ($p$) [m]
    PositionConfidenceEllipsoidRadius: NotationSymbol  # Mahalanobis radius of the 95 percent confidence ellipsoid of a three-dimensional Gaussian position estimate, whose square is the chi-square quantile for three degrees of freedom at confidence 0.95 ($\chi_{3,0.95}$) [dimensionless]
    PositionErrorBound: NotationSymbol  # worst-case position error of a target estimate as a function of the time elapsed since its evidence was captured ($E$) [m]
    PositionErrorCeiling: NotationSymbol  # largest position error a target's own motion may add before a measurement of it is too stale to act on ($e_{\max}$) [m]
    PositionErrorCovariance: NotationSymbol  # position block of an estimator's error covariance, the spread of its position estimate ($\mathbf{P}_{pp}$)
    PositionErrorEnvelope: NotationSymbol  # conservative bound on the position error of a target estimate along one coordinate axis of the base frame ($e$) [m]
    PositionLossMask: NotationSymbol  # loss-mask indicator at the token position named by its index subscript ($m$) [dimensionless]
    PositionUncertaintyRadius: NotationSymbol  # radius of the region that may contain a previously localized obstacle, given its position uncertainty ($r$) [m]
    PositionalTaskTolerance: NotationSymbol  # radius within which a target estimate must lie for the task to succeed ($r_{\text{tol}}$) [m]
    PositiveClassInput: NotationSymbol  # clean input of the positive class, the labeled point whose perturbation neighborhood an illustrative decision boundary may cut through ($x_{+}$)
    PositiveIntegerExponent: NotationSymbol  # positive integer power to which an algebra element is raised in the definition of an idempotent element ($n$)
    PostContactDrivenStroke: NotationSymbol  # displacement the arm drives into a compliant load path after the contact tripwire fires, travel during the response latency plus the braking stroke ($\Delta x_{\text{driven}}$) [m]
    PostImpactNormalVelocity: NotationSymbol  # normal contact velocity just after an impact, minus the restitution coefficient times the pre-impact normal velocity ($v_n^+$) [m/s]
    PostRevocationDelay: NotationSymbol  # delay from revoking authority to the onset of braking: the permission tick, bus transport and brake onset ($\tau_{\text{rev}}$) [s]
    PosteriorCovarianceMatrix: NotationSymbol  # covariance matrix of a Gaussian state estimate after a measurement update ($\boldsymbol{\Sigma}_{\text{post}}$) [varies]
    PosteriorDifferentialEntropy: NotationSymbol  # differential entropy of a state estimate after a measurement update ($h_{\text{post}}$) [nats]
    PostprocessingLatency: NotationSymbol  # time spent turning a model's output into the response ($L_{\text{lat,post}}$) [s]
    PotentialOutcome: NotationSymbol  # potential outcome of one experimental unit as a function of the treatment assignment it receives ($Y^{\text{pot}}$)
    PowerCriticalQuantile: NotationSymbol  # positive standard-Normal critical value corresponding to the chosen statistical power of a test ($z_{\beta}$) [dimensionless]
    PowerLimitBase: NotationSymbol  # positive base of a power whose exponent tends to zero, so the scaled power tends to its logarithm ($a$) [dimensionless]
    PowerRailInductance: NotationSymbol  # lumped inductance of the supply loop feeding the inverter ($L_{\text{bus}}$) [H]
    PowerRailResistance: NotationSymbol  # lumped resistance of the supply path feeding the inverter: battery cells, harness and switch ($R_{\text{bus}}$) [Ohm]
    PoweredHoldCommand: NotationSymbol  # resident fallback command that holds the machine at standstill under energized position control ($\mathbf{u}_{\text{hold}}$)
    PoweredHoldState: NotationSymbol  # plant state the drives maintain under closed-loop position control during a powered hold at standstill ($\mathbf{x}_{\text{hold}}$)
    PoweredServerCount: NotationSymbol  # number of servers a facility keeps powered and active at the same time ($N$)
    PpoClippedSurrogateObjective: NotationSymbol  # clipped surrogate objective proximal policy optimization maximizes, the empirical mean of the smaller of the ratio-weighted advantage and its clipped counterpart ($\mathcal{L}^{\text{CLIP}}$) [dimensionless]
    PqApproximateDistance: NotationSymbol  # approximate squared distance between a search query embedding and a product-quantized stored vector, summed from per-subspace lookup-table entries ($\tilde{d}$)
    PqCentroidCode: NotationSymbol  # index of one centroid in a product-quantization subspace codebook, stored as the one-byte code of the subvector it is nearest ($c^{\text{pq}}$)
    PqSubspaceIndex: NotationSymbol  # index of one product-quantization subspace, from 1 to the subvector count ($j$)
    PqSubvectorCount: NotationSymbol  # number of subvectors product quantization splits each stored vector into, which is also the number of one-byte codes it stores per vector ($m$)
    PreForwardEnvironmentState: NotationSymbol  # state of the environment before a forward step of a saga executes ($\sigma_0$)
    PreImpactNormalVelocity: NotationSymbol  # normal contact velocity just before an impact, negative while the bodies approach ($v_n^-$) [m/s]
    PreImpactSeparationDistance: NotationSymbol  # separation distance kept between the machine and a body before any contact, which must cover the stopping distance ($d_{\text{sep}}$) [m]
    PrePivotCatchRate: NotationSymbol  # share of defective actions that a check placed before the pivot catches ($v$) [dimensionless]
    PreTriggerWindowDuration: NotationSymbol  # length of the ring-buffer window before an intervention trigger that the logger freezes and keeps ($T_{\text{pre}}$) [s]
    PreactivationMatrix: NotationSymbol  # preactivation matrix of one layer for a batch, one example per row ($\mathbf{Z}$)
    PreactivationVector: NotationSymbol  # preactivation vector of one layer for one example, the logit vector at the output layer ($\mathbf{z}$)
    PrebrakeDelay: NotationSymbol  # time from the capture of the frame that first shows a hazard to the onset of retarding force ($\tau_{\text{delay}}$) [s]
    PrebrakeDelayCeiling: NotationSymbol  # largest pre-brake delay that still lets the machine stop within its clearance, clearance minus margins and braking distance divided by speed ($\tau_{\text{delay},\max}$) [s]
    PredecessorGraphVertex: NotationSymbol  # vertex at the tail of a precedence edge, whose output the edge's head vertex needs before it can start ($u$)
    PredicateLipschitzConstant: NotationSymbol  # Lipschitz constant of a signal temporal logic predicate's margin with respect to the weighted state norm, the largest margin change per unit of normalized state discrepancy ($L_{\phi}$) [varies]
    PredicateNormalizationScale: NotationSymbol  # positive scale, in the unit of one atomic predicate, by which its margin is divided to make it dimensionless ($c$) [varies]
    PredictedActionComponent: NotationSymbol  # one component of an action a policy predicts in open-loop scoring, indexed by component ($\hat{a}$) [varies]
    PredictedClassProbability: NotationSymbol  # probability a classifier assigns to one label category for a given input ($p$) [dimensionless]
    PredictedConfidence: NotationSymbol  # confidence a classifier reports for its predicted class on one example, its largest class probability ($\hat{p}$) [dimensionless]
    PredictedDecision: NotationSymbol  # binary decision a classifier outputs for a randomly drawn individual, one for a positive prediction ($\hat{Y}$) [dimensionless]
    PredictedEnvironmentLatent: NotationSymbol  # estimate of the environment latent vector that the adaptation module predicts from onboard observation history ($\hat{\mathbf{z}}$)
    PredictedErrorCovariance: NotationSymbol  # error covariance at step k predicted from measurements through step k minus one, before the update ($\mathbf{P}_{k|k-1}$)
    PredictedForceReading: NotationSymbol  # force-sensor reading the estimator predicts from its force and bias estimates ($\hat{y}$) [N]
    PredictedOutput: NotationSymbol  # model output for one example, such as its vector of class probabilities ($\hat{\mathbf{y}}$)
    PredictedOutputComponent: NotationSymbol  # one entry of a model's prediction, such as the predicted probability for one class ($\hat{y}$) [dimensionless]
    PredictedPolicyAction: NotationSymbol  # action a policy predicts on a recorded demonstrator state in open-loop scoring, never executed ($\hat{\mathbf{a}}$)
    PredictedProtectedAttribute: NotationSymbol  # protected attribute an adversary head predicts from a latent representation ($\hat{a}$)
    PredictedRelativeDepth: NotationSymbol  # depth a monocular network predicts up to an unknown scale and shift ($z_{\text{pred}}$) [dimensionless]
    PredictedStateEmbedding: NotationSymbol  # learned embedding of the state estimate propagated forward to the instant of actuation ($\mathbf{e}_{\text{state}}$)
    PredictedStateEstimate: NotationSymbol  # state estimate at step k predicted from measurements through step k minus one, before the update ($\hat{\mathbf{x}}_{k|k-1}$)
    PredictionHeadFunction: NotationSymbol  # prediction function of a small trainable head applied to a stored embedding ($g$)
    PredictionStepIndex: NotationSymbol  # position of one predicted step within a model predictive control horizon ($i$)
    PredictiveEntropy: NotationSymbol  # entropy of a model's predicted class distribution for one input, a measure of its uncertainty ($H_{\text{pred}}$) [nats]
    PreemptionFanoutCount: NotationSymbol  # number of lower-priority jobs one admission displaces at each level of a preemption cascade ($k$)
    PreemptionRate: NotationSymbol  # number of forced preemptions a job suffers per hour of held time ($\lambda_{\text{preempt}}$) [hours^-1]
    PreemptionTaxTime: NotationSymbol  # held time one forced preemption costs a job, the work discarded since its last checkpoint plus reload and warmup ($T_{\text{preempt}}$) [hours]
    PreferenceMargin: NotationSymbol  # preference scale times the difference between the preferred and dispreferred records' reference-relative scores ($z$) [dimensionless]
    PreferencePairDataset: NotationSymbol  # dataset of curated preference comparisons, each a context with a preferred and a dispreferred trajectory record ($\mathcal{D}$)
    PreferenceScale: NotationSymbol  # positive coefficient in direct preference optimization that scales the score difference and sets how strongly the objective resists drift from the reference policy ($\beta$) [dimensionless]
    PreferenceScoredPositionSet: NotationSymbol  # set of policy-generated token positions of one trajectory that a preference objective scores ($\mathcal{A}$)
    PreferredReferenceRelativeScore: NotationSymbol  # reference-relative score of the preferred trajectory record of one comparison pair ($\Delta_w$) [dimensionless]
    PreferredTrajectory: NotationSymbol  # trajectory record ranked higher in one preference comparison ($\tau_w$)
    PrefetchBufferCount: NotationSymbol  # number of buffers holding batches loaded ahead of use ($N_{\text{prefetch}}$)
    PrefillAllocatedTokenCount: NotationSymbol  # number of prompt tokens the scheduler allocates to one waiting prefill request in one iteration ($C$)
    PrefillArithmeticIntensity: NotationSymbol  # operations per byte of weights loaded for the linear projections of the prefill phase ($I_{\text{prefill}}$) [FLOP/byte]
    PrefillChunkSize: NotationSymbol  # largest number of prompt tokens chunked prefill processes for one request in one iteration ($C_{\text{chunk}}$)
    PrefillOperations: NotationSymbol  # arithmetic operations one evaluation's prefill executes over all of its input tokens ($O_{\text{prefill}}$) [FLOP]
    PrefillRequestIndex: NotationSymbol  # index of a waiting prefill request within the active prefill set ($i$)
    PrefillTime: NotationSymbol  # time a model call spends in prefill, processing its whole prompt ($T_{\text{prefill}}$) [s]
    PrefixHitRate: NotationSymbol  # token-weighted share of prompt tokens over a workload window whose compatible key-value state was attached without recomputation ($\rho_{\text{prefix}}$) [dimensionless]
    PregraspPosition: NotationSymbol  # position the tool center point must reach before contact, on the approach axis short of the grasp ($\mathbf{p}_{\text{pre}}$) [m]
    PreprocessingLatency: NotationSymbol  # time spent preparing a request's input before inference ($L_{\text{lat,pre}}$) [s]
    PreprocessingServiceTime: NotationSymbol  # average time between batches the preprocessing workers deliver, across all workers ($T_{\text{preprocessing}}$) [s]
    PreprocessingThroughput: NotationSymbol  # batches per second the host workers can transform into model-ready tensors ($R_{\text{preprocessing}}$) [batches/s]
    PresaturationTorqueCommand: NotationSymbol  # joint torque vector a controller commands before actuator saturation ($\boldsymbol{\tau}_{\text{cmd}}$)
    PreviousLayerWidth: NotationSymbol  # number of units in the layer before layer l ($n_{\ell-1}$)
    PreviousPeriodCost: NotationSymbol  # observed cost of the accounting period before the current one ($C_{\text{previous}}$) [USD]
    PreviousRangeObservationHistory: NotationSymbol  # sequence of range observations from the first step through the step before the current one ($z_{1:t-1}$)
    PreviousUpdatedErrorCovariance: NotationSymbol  # error covariance at step k minus one after that step's update ($\mathbf{P}_{k-1|k-1}$)
    PreviousUpdatedStateEstimate: NotationSymbol  # state estimate at step k minus one after that step's update, the input to the next prediction ($\hat{\mathbf{x}}_{k-1|k-1}$)
    PrimitiveOperation: NotationSymbol  # elementary operation a computational graph node applies to the values of its parent nodes ($\phi$)
    PrincipalPointColumn: NotationSymbol  # horizontal pixel coordinate of the principal point, where the optical axis meets the image ($c_x$) [pixels]
    PrincipalPointRow: NotationSymbol  # vertical pixel coordinate of the principal point, where the optical axis meets the image ($c_y$) [pixels]
    PriorCovarianceMatrix: NotationSymbol  # covariance matrix of a Gaussian state estimate before a measurement update ($\boldsymbol{\Sigma}_{\text{prior}}$) [varies]
    PriorDifferentialEntropy: NotationSymbol  # differential entropy of a state estimate before a measurement update ($h_{\text{prior}}$) [nats]
    PriorStandardDeviation: NotationSymbol  # standard deviation of an isotropic zero-mean Gaussian prior over model parameters ($\sigma_0$) [varies]
    PriorTurnCount: NotationSymbol  # number of turns whose appended tokens a call's prompt already carries, counted from zero on the first call ($j$)
    PrivacyAdditiveSlack: NotationSymbol  # additive slack term of approximate differential privacy, the probability mass by which the multiplicative bound may be exceeded ($\delta$) [dimensionless]
    PrivacyBudget: NotationSymbol  # privacy-loss bound of differential privacy: the log of the largest factor by which one record may change the probability of any output ($\epsilon$) [dimensionless]
    PrivacyLossVariable: NotationSymbol  # privacy loss random variable: the log of the ratio between a mechanism's output probabilities or densities on two adjacent datasets, evaluated at the observed output ($\mathcal{L}_{\text{priv}}$)
    PrivacyNoiseMultiplier: NotationSymbol  # ratio of the Gaussian noise standard deviation to the gradient clipping norm in differentially private training ($\sigma$) [dimensionless]
    PrivatizedQuery: NotationSymbol  # privatized query: the value a differentially private mechanism releases, the query on the dataset plus Laplace or Gaussian noise ($\tilde{f}$)
    PrivilegeTrapTime: NotationSymbol  # time one system call spends crossing a privilege boundary in hardware ($T_{\text{trap}}$) [s]
    PrivilegedSimulatorState: NotationSymbol  # privileged simulator state at step t, the ground-truth quantities such as friction and contact wrenches that a deployed policy cannot measure ($s_{\text{priv}}$)
    PrivilegedStateDimension: NotationSymbol  # number of real components in the privileged simulator state ($d_{\text{priv}}$)
    PrivilegedStateEncoder: NotationSymbol  # learned encoder that compresses the privileged simulator state into the environment latent vector ($E$)
    ProbabilityFunction: NotationSymbol  # probability mass or density function of a random variable, applied to its argument ($p$)
    ProbabilitySimplex: NotationSymbol  # probability simplex: the set of vectors with nonnegative entries that sum to one, the valid probability distributions over a finite set of classes ($\Delta$)
    ProbedCellCount: NotationSymbol  # number of nearest coarse cells an inverted-file index query scans, its recall knob ($n_{\text{probe}}$)
    ProcessConditionCheckTime: NotationSymbol  # expected verifier slot time of the process-conditions predicate for a record that reaches it ($T_{\text{proc}}$) [s]
    ProcessConditionPassProbability: NotationSymbol  # probability that a candidate passes the process-conditions predicate, the audit of its execution history against operational constraints and tool policies ($p_{\text{proc}}$) [dimensionless]
    ProcessNoiseCovariance: NotationSymbol  # covariance of the process noise that the predict step adds to the error covariance ($\mathbf{Q}$)
    ProcessNoiseIntensity: NotationSymbol  # intensity (power spectral density) matrix of continuous-time white process noise, the weight of the delta function in its covariance kernel ($\mathbf{Q}$)
    ProcessNoiseVector: NotationSymbol  # random disturbance acting on the state in the process model ($\mathbf{w}$)
    ProcessedDataVolume: NotationSymbol  # data volume data and feature pipelines process in an accounting period ($D_{\text{vol,processed}}$) [bytes]
    ProcessedSampleCount: NotationSymbol  # total number of training samples a run processes, counting repeated passes over the data ($N_{\text{samples}}$)
    ProcessingBufferCount: NotationSymbol  # number of buffers holding batches under transformation ($N_{\text{processing}}$)
    ProcessingPrice: NotationSymbol  # price of processing one gigabyte through data and feature pipelines ($R_{\text{process/GB}}$) [USD/GB]
    ProductFactorIndex: NotationSymbol  # position of one factor in a product or sum that runs from zero to one less than the claim size ($j$)
    ProductionExecutionCount: NotationSymbol  # number of production executions, such as inference requests or training steps, that reuse one compiled program ($N_{\text{prod}}$)
    ProductionModelCount: NotationSymbol  # number of production models a fleet or shared platform operates ($N_{\text{models}}$)
    ProductionOnlyFailureCount: NotationSymbol  # number of paired requests on which the production model fails and the shadow model succeeds ($n_{\text{prod}}$)
    ProfileAccelerationBound: NotationSymbol  # largest acceleration or deceleration a planned or interpolated profile may command ($a_{\max}$) [m/s^2]
    ProfileJerk: NotationSymbol  # time derivative of the acceleration of a one-dimensional motion profile ($\dot{a}$) [m/s^3]
    ProgramGoodput: NotationSymbol  # ratio of ideal to actual program execution time during productive work ($\eta_{\text{program}}$) [dimensionless]
    ProgramInvariantPredicate: NotationSymbol  # predicate on program states that holds initially and is preserved by every transition, the inductive invariant of software verification ($P$)
    ProgramState: NotationSymbol  # state of a program in formal software verification, the valuation of its variables ($s$)
    ProgramStateSpace: NotationSymbol  # set of all states a program can reach or represent in formal software verification ($\Sigma$)
    ProjectedSeamState: NotationSymbol  # reference state the active trajectory is projected to reach at the replanning seam, on which the replacement is conditioned ($\hat{\mathbf{x}}_{\text{seam}}$)
    ProjectedTrajectoryRecord: NotationSymbol  # trajectory control record that a deterministic reducer rebuilds by folding logged events, indexed by the log position it reflects ($R$)
    ProjectedTurnGradientContribution: NotationSymbol  # contribution of one turn's action to a likelihood-ratio policy-gradient estimate, projected onto a fixed parameter-space direction ($Z$) [dimensionless]
    ProjectionBallRadius: NotationSymbol  # radius of the norm ball onto which projected gradient descent projects its iterate ($\epsilon$) [varies]
    ProjectionConstraintSet: NotationSymbol  # convex constraint set onto which projected gradient descent projects each updated iterate ($\mathcal{C}$)
    ProjectiveDistanceSample: NotationSymbol  # signed distance from a voxel to the measured return along the sensing ray, positive in front of the return, before truncation ($s$) [m]
    ProjectiveRayScale: NotationSymbol  # free positive scale along a camera ray: every multiple of a homogeneous pixel direction projects to the same pixel ($\lambda$) [varies]
    ProjectiveScaleFactor: NotationSymbol  # arbitrary nonzero scale of a homogeneous image point in the projection equation ($s$) [dimensionless]
    PromotedFusionInputRank: NotationSymbol  # better 1-based rank a candidate moves to within one retrieval method's ranked list, compared with its earlier rank there, when reasoning about how reciprocal rank fusion scores respond ($r'$)
    PromptLength: NotationSymbol  # number of tokens in one request's prompt, all of which prefill processes ($S_{\text{prompt}}$)
    PromptToken: NotationSymbol  # one token of the serialized prompt a call sends, indexed by its position ($x$)
    PromptTokensSent: NotationSymbol  # cumulative number of prompt tokens a trajectory sends across all its calls ($S_{\text{sent}}$)
    PropagatedPositionStandardDeviation: NotationSymbol  # standard deviation of the position estimate as a function of time since measurements stopped ($\sigma_p$) [m]
    PropensityPriorApprovalShape: NotationSymbol  # first shape parameter of the beta distribution assumed for the false-approval propensity, the pseudo-count weighting approval ($a$) [dimensionless]
    PropensityPriorRejectionShape: NotationSymbol  # second shape parameter of the beta distribution assumed for the false-approval propensity, the pseudo-count weighting correct rejection ($b$) [dimensionless]
    ProportionTestStatistic: NotationSymbol  # test statistic of a two-proportion z-test, the difference of two sample proportions divided by its pooled standard error ($z$) [dimensionless]
    ProportionalGainMatrix: NotationSymbol  # matrix of proportional feedback gains acting on the tracking error vector in joint-space PD or PID control ($\mathbf{K}_{p}$)
    ProportionalPotentialEnergy: NotationSymbol  # artificial quadratic potential of the tracking error that proportional feedback descends, one half the error weighted by the proportional gain matrix ($\mathcal{L}_{\text{pos}}$)
    ProposalDeserializationTime: NotationSymbol  # time for the safety microcontroller to deserialize a received candidate proposal ($t_{\text{deser}}$) [s]
    ProposalExpiryInstant: NotationSymbol  # monotonic instant after which a proposal lapses and may no longer be executed ($t_{\text{valid}}$) [s]
    ProposalFirstWaypoint: NotationSymbol  # first waypoint of an incoming action proposal, which must match the robot's rest configuration before authority is restored ($\mathbf{a}_0$) [varies]
    ProposalLinkTransportTime: NotationSymbol  # time for a candidate proposal to cross the link from the host to the safety microcontroller ($t_{\text{link}}$) [s]
    ProposalParameterVector: NotationSymbol  # parameter vector of a parametric proposal distribution, updated by the cross-entropy method ($\boldsymbol{\psi}$) [varies]
    ProposalPayload: NotationSymbol  # typed candidate-motion record a learned proposer sends to the real-time permission path, with its evidence capture time, expiry and provenance ($\mathcal{P}$)
    ProposedAccess: NotationSymbol  # access a tool call proposes, naming an operation, a target and its arguments, which an enforcement point admits or refuses under a capability grant ($r$)
    ProposedContactForce: NotationSymbol  # constant contact force a proposal asks for over one control interval ($F_{\text{nom}}$) [N]
    ProposedEndEffectorPoseSequence: NotationSymbol  # end-effector target poses, position and orientation, at every setpoint of a proposed chunk ($\mathbf{X}_{1:H}$)
    ProposedGripperCommandSequence: NotationSymbol  # gripper closure targets between zero and one at every setpoint of a proposed chunk ($\mathbf{g}_{1:H}$)
    ProposedJointPositionSequence: NotationSymbol  # candidate joint positions at every setpoint of a proposed chunk, one row per setpoint ($\mathbf{Q}_{1:H}$)
    ProposedJointVelocitySequence: NotationSymbol  # candidate joint velocity setpoints of a proposed chunk, one row per setpoint ($\dot{\mathbf{Q}}_{1:H}$)
    ProposedOperation: NotationSymbol  # operation a proposed tool call would perform, the unit a reference monitor admits or refuses ($o$)
    ProposerConfidenceScore: NotationSymbol  # confidence score a learned proposer attaches to a proposal, from the method its calibration profile names ($\hat{p}$) [dimensionless]
    ProposerEvaluationRate: NotationSymbol  # rate of fresh proposer evaluations when every evaluation streams the weights ($f$) [Hz]
    ProposerInferenceTime: NotationSymbol  # time for one forward evaluation of the learned proposer ($t_{\text{infer}}$) [s]
    ProprioceptiveInputSpace: NotationSymbol  # normed space of the inputs a continuous-control policy network accepts ($\mathcal{X}$)
    ProprioceptiveInputVector: NotationSymbol  # low-dimensional proprioceptive vector a continuous-control policy network takes as input: joint positions and velocities, inertial rates, projected gravity and recent actions ($\mathbf{x}$)
    ProprioceptiveTokenEmbeddings: NotationSymbol  # token embeddings of the robot's joint and force state fed to an embodied transformer ($\mathbf{z}_{\text{proprio}}$)
    ProtectedAttribute: NotationSymbol  # protected attribute of one example, such as its demographic group, used as the target of an adversary in adversarial debiasing ($a$)
    ProtectedGroupMembership: NotationSymbol  # protected group a randomly drawn individual belongs to, as a random variable over group labels ($G$)
    ProtectedTrainingRecord: NotationSymbol  # training record of one protected individual, the unit whose leakage a privacy claim bounds ($R$)
    ProtectionSwitchResistance: NotationSymbol  # on-state resistance of the protection switches in a battery's supply path ($R_{\text{switch}}$) [Ohm]
    ProtectiveClearanceMargin: NotationSymbol  # fixed nonzero clearance kept between the machine envelope and the boundary after it stops ($\delta_{\text{margin}}$) [m]
    ProtectiveSeparationDistance: NotationSymbol  # minimum distance a speed-and-separation monitor must keep between a person and a robot ($S_p$) [m]
    ProtectiveSpringStiffness: NotationSymbol  # stiffness of a protective linear spring sized to absorb a design force ($k_{\text{spring}}$) [N/m]
    ProtectiveSpringTravel: NotationSymbol  # compression of a protective linear spring under a given force, its travel ($\delta$) [m]
    ProxyTrajectoryReward: NotationSymbol  # reward the training loop computes for a trajectory, viewed as an imperfect proxy for the intended objective ($\hat{R}$) [dimensionless]
    PrunedWeightMatrix: NotationSymbol  # weight matrix of a layer after pruning, with the removed weights set to zero ($\hat{\mathbf{W}}$)
    PruningMask: NotationSymbol  # binary matrix, the same shape as a weight matrix, whose ones mark the weights pruning keeps ($\mathbf{M}$)
    PruningMaskEntry: NotationSymbol  # one entry of the pruning mask, 1 if the weight at row i and column j is kept and 0 if removed ($M_{i,j}$) [dimensionless]
    PruningThreshold: NotationSymbol  # weight magnitude at or below which magnitude-based pruning removes a weight ($\delta_{\text{prune}}$) [dimensionless]
    PsuMttf: NotationSymbol  # mean operating time of one power supply unit before it fails ($\text{MTTF}_{\text{psu}}$) [hours]
    PtpExchangeTimestamp: NotationSymbol  # local-clock timestamp of one event of a Precision Time Protocol delay exchange, numbered 1 to 4 in exchange order ($t$) [s]
    PtpForwardPathDelay: NotationSymbol  # one-way propagation delay of the path from the leader clock to the follower clock ($T_{\text{prop}, M \to S}$) [s]
    PtpHardwareTimestamp: NotationSymbol  # time stamped in hardware in the Precision Time Protocol timebase shared by the application processor and the safety microcontroller ($t_{\text{PTP}}$) [s]
    PtpPropagationDelay: NotationSymbol  # one-way network path delay between two clocks estimated from a symmetric two-way Precision Time Protocol exchange ($T_{\text{prop}}$) [s]
    PtpReversePathDelay: NotationSymbol  # one-way propagation delay of the path from the follower clock to the leader clock ($T_{\text{prop}, S \to M}$) [s]
    PublicationEpoch: NotationSymbol  # instant a serialized state record is committed to a transport bus or shared-memory buffer ($t_{\text{pub}}$) [s]
    PufChallenge: NotationSymbol  # challenge presented to a physical unclonable function, a bit string whose indexed entries are the challenge bits ($C$)
    PufResponse: NotationSymbol  # response a physical unclonable function returns to a challenge, which differs between devices of the same design ($R$)
    PufResponseBit: NotationSymbol  # single output bit of an arbiter physical unclonable function, set by which of two racing signals arrives first ($r$)
    PulseRoundTripTime: NotationSymbol  # time from emitting a lidar pulse to receiving its reflection ($\Delta t$) [s]
    PureNoiseActionChunk: NotationSymbol  # action chunk at the final forward diffusion step, pure Gaussian noise from which reverse denoising starts ($\mathbf{A}^N$)
    PureTransportDelay: NotationSymbol  # fixed delay in a feedback or communication channel ($T_d$) [s]
    PwmPeriod: NotationSymbol  # period of one pulse-width-modulation cycle of an inverter timer ($T_{\text{pwm}}$) [s]
    QuadratureAxisInductance: NotationSymbol  # stator inductance along the quadrature axis of the rotor dq frame, which sets the torque-current time constant ($L_q$) [H]
    QuadratureAxisVoltage: NotationSymbol  # quadrature-axis voltage the inverter applies in a field-oriented drive ($v_q$) [V]
    QualifiedAdmissionRate: NotationSymbol  # fraction of qualified applicants admitted, the same in every group under equal opportunity, the shared true positive rate ($r$) [dimensionless]
    QualityObjective: NotationSymbol  # model-quality dimension of a service's SLO vector: its indicator, target, window and owner ($S_{\text{quality}}$)
    QuantileLevel: NotationSymbol  # probability level at which a quantile function is evaluated, strictly between zero and one ($u$) [dimensionless]
    QuantileProbabilityLevel: NotationSymbol  # cumulative probability at which a quantile of a distribution is taken ($q$) [dimensionless]
    QuantileThresholdCandidate: NotationSymbol  # candidate value of a statistic over which an empirical quantile's infimum is taken ($x$) [varies]
    QuantizationBitWidth: NotationSymbol  # number of bits in each integer code a quantizer produces ($b$)
    QuantizationBlockSize: NotationSymbol  # number of tensor elements in one block of block-wise quantization, all sharing one scale ($b_{\text{block}}$)
    QuantizationScale: NotationSymbol  # real-valued step between adjacent integer levels of a uniform quantizer ($s$) [varies]
    QuantizationStep: NotationSymbol  # one quantization step of the quantity it prefixes, the smallest change a sensor resolves ($\delta$)
    QuantizationZeroPoint: NotationSymbol  # integer code that an affine quantizer assigns to real zero ($z$) [dimensionless]
    QuantizedGradient: NotationSymbol  # gradient vector after symmetric INT8 quantization, one integer code per element ($\mathbf{g}_{\text{int8}}$)
    QuantizedIntegerCode: NotationSymbol  # integer code a quantizer assigns to one real value ($x_q$) [dimensionless]
    QuantizedWeightEntry: NotationSymbol  # one integer-quantized weight at output row i and reduction index k of an integer matrix multiply ($W_{ik}$) [dimensionless]
    QuantizerInput: NotationSymbol  # full-precision real value, a weight or activation, that a quantizer maps to an integer code ($x$) [varies]
    QuarterCycleCompressionTime: NotationSymbol  # time from first contact to maximum compression in an unbraked linear-spring impact, a quarter of the oscillation period ($t_{\text{peak}}$) [s]
    QueryAxisLabel: NotationSymbol  # einsum index label naming the query-position axis of an attention tensor ($i$)
    QueryBlockCount: NotationSymbol  # number of query tiles the sequence is split into in tiled attention ($T_r$)
    QueryBlockIndex: NotationSymbol  # index of one query tile in tiled attention ($i$)
    QueryBlockSize: NotationSymbol  # number of query rows in one tile of tiled attention ($B_r$)
    QueryBoundaryCount: NotationSymbol  # number of policy query boundaries in an episode when each chunk executes several control steps before the next query ($S_{\text{eff}}$)
    QueryBufferPosition: NotationSymbol  # position of the query token within a packed attention buffer ($i$)
    QueryComponent: NotationSymbol  # one component of one token's query vector ($q_{im}$) [dimensionless]
    QueryPrice: NotationSymbol  # price an API charges for one query ($p_{\text{query}}$) [USD/query]
    QueryProjectionMatrix: NotationSymbol  # learned matrix that projects token representations into queries ($\mathbf{W}_{Q}$)
    QuerySensitivity: NotationSymbol  # global sensitivity of a query function: the largest change in its output when one protected record changes, in the l1 norm, or in the l2 norm when subscripted 2 ($\Delta$)
    QuerySourceMatrix: NotationSymbol  # representations attention draws its queries from, before a head-specific query projection ($\mathbf{X}_{Q}$)
    QueryVector: NotationSymbol  # query vector of one token, one row of the query matrix ($\mathbf{q}$)
    QueryWord: NotationSymbol  # one analyzer word of a lexical query ($q$)
    QueryWordDocumentFrequency: NotationSymbol  # number of documents in a lexically indexed corpus that contain a given query word ($n$)
    QueryWordFrequency: NotationSymbol  # number of times one query word occurs in one indexed document, as the lexical analyzer counts it ($f$)
    QueryWorkspaceGeneration: NotationSymbol  # workspace generation a memory query requires ($g_q$)
    QueueArrivalTime: NotationSymbol  # time at which one event or record enters a group-commit batching queue ($t_{\text{arr}, i}$) [s]
    QueueServerCount: NotationSymbol  # number of identical servers in an M/M/c queue ($c$)
    QueueServiceVariationCoefficient: NotationSymbol  # coefficient of variation of a queue's service times, their standard deviation divided by their mean, in the lowercase Kingman notation ($c_{s}$) [dimensionless]
    QueueSystemCapacity: NotationSymbol  # largest number of requests a queueing system can hold at once, waiting plus in service, beyond which arrivals are rejected ($K$)
    QueueWaitTime: NotationSymbol  # mean time a request waits in a queue before its service begins, excluding service time ($W_{q}$) [s]
    QueuedJobCount: NotationSymbol  # number of jobs waiting in a queue, not in service, as an arrival sees it ($N_{\text{queue}}$)
    QueuedTransactionBytes: NotationSymbol  # bytes one memory transaction queued ahead of a safety read carries, the per-transaction term of the DRAM queueing delay ($D_{\text{vol}}$) [bytes]
    QueuedTransactionCount: NotationSymbol  # number of interconnect transactions admitted ahead of a safety-critical request in a shared queue ($K$)
    QueuedTransactionSet: NotationSymbol  # set of memory transactions queued ahead of a safety read at the memory controller ($\mathcal{Q}$)
    QueueingLatency: NotationSymbol  # time a request waits in queues before it is served ($L_{\text{lat,queue}}$) [s]
    QuiescentJunctionTemperature: NotationSymbol  # steady-state junction temperature under baseline (quiescent) power, toward which the junction relaxes once a burst ends ($T_{\text{j},\text{target}}$) [degC]
    QuiescentSocPower: NotationSymbol  # power an application SoC dissipates with only the permission loop, idle and memory power running ($P_{\text{base}}$) [W]
    QuinticBlendCoefficient: NotationSymbol  # coefficient vector of one power of normalized time in a quintic blend polynomial, indexed by that power ($\mathbf{a}$) [varies]
    QuotaReductionSensitivity: NotationSymbol  # coefficient that sets how steeply an account's effective query quota shrinks, exponentially, with its anomaly score ($\gamma_{\text{rate}}$) [dimensionless]
    RadarRangeResolution: NotationSymbol  # smallest separation in range at which an FMCW radar distinguishes two reflectors, set by its sweep bandwidth ($\Delta R$) [m]
    RadialBasisKernelCoefficient: NotationSymbol  # coefficient of a radial basis function kernel, the factor multiplying the squared distance between two inputs in the exponent ($\gamma$) [varies]
    RadialDistortionCoefficient: NotationSymbol  # coefficient of one power of radius in a radial lens distortion polynomial, numbered k_1, k_2 ($k$) [dimensionless]
    RadialRange: NotationSymbol  # radial distance from an active ranging sensor, lidar or radar, to the surface that reflects its signal ($R$) [m]
    RadialRangeStandardDeviation: NotationSymbol  # standard deviation of a lidar's measured radial range, set by receiver timing jitter ($\sigma_R$) [m]
    RadiationWavelength: NotationSymbol  # wavelength of the electromagnetic radiation a sensor emits or receives ($\lambda$) [m]
    RailVoltageTolerance: NotationSymbol  # allowed rail-voltage excursion as a fraction of the nominal voltage ($\delta_{\text{tol}}$) [dimensionless]
    RandomEnvironmentState: NotationSymbol  # environment state at a turn of a partially observable decision process, viewed as a random variable over its state space ($S$)
    RandomRetainedElementCount: NotationSymbol  # number of gradient elements that random-k sparsification selects uniformly at random and transmits ($K_{\text{rand}}$)
    RandomSampleCollectionSize: NotationSymbol  # total number of examples that must be collected and labeled at random to obtain a target number from one subgroup in expectation ($D_{\text{eval,total}}$)
    RandomVectorDimension: NotationSymbol  # number of components of a continuous random vector, the dimension of the space its density is defined on ($d$)
    RandomizationClusterCount: NotationSymbol  # number of clusters to which treatment is assigned in graph cluster randomization ($K$)
    RandomizedMechanism: NotationSymbol  # randomized algorithm that maps a dataset to a released output, the object a differential privacy guarantee constrains ($\mathcal{M}$)
    RandomizedSmoothingNoise: NotationSymbol  # random Gaussian noise vector randomized smoothing adds to an input before the base classifier sees it ($\boldsymbol{\delta}$)
    RangeObservation: NotationSymbol  # one range observation a sensor returns at a time step ($z$)
    RangeObservationHistory: NotationSymbol  # sequence of range observations from the first step through the current step ($z_{1:t}$)
    RankComponentIndex: NotationSymbol  # index of one rank-one component in a CP decomposition, from 1 to the factorization rank ($s$)
    RankConstrainedMatrix: NotationSymbol  # candidate matrix of rank at most a given value that a low-rank approximation minimum ranges over ($\mathbf{A}$)
    RankContribution: NotationSymbol  # data one rank contributes to a collective operation, its input buffer, indexed by rank ($x$) [varies]
    RankCountBinaryExponent: NotationSymbol  # integer exponent for which the rank count equals two raised to it, the condition algorithms such as butterfly AllReduce require ($m$) [dimensionless]
    RankFusionSmoothingConstant: NotationSymbol  # positive constant reciprocal rank fusion adds to each rank before taking the reciprocal, which compresses score differences between adjacent ranks ($k_{\text{rrf}}$) [dimensionless]
    RankIndex: NotationSymbol  # position of one rank, a device or worker, within a collective communication group, counted from zero ($i$)
    RankReductionResult: NotationSymbol  # result a reducing collective leaves on one rank, indexed by rank ($y$) [varies]
    RankingFunction: NotationSymbol  # nonnegative integer measure of remaining admissible steps that every admitted step decreases by at least one, bounding how many steps can be admitted ($V$)
    RatedActuatorTorque: NotationSymbol  # rated torque of the actuator under test, the reference the stall counter-torque is scaled from ($\tau_{\text{rated}}$) [N*m]
    RatedCurrentEquilibriumTemperature: NotationSymbol  # winding temperature at thermal equilibrium under continuous rated current, the heat-soaked starting point of a burst ($T_{\text{eq}}$) [degC]
    RatedFatigueLife: NotationSymbol  # cumulative load-cycle count at which ten percent of a population of identical components is expected to have failed ($L_{10}$)
    RatedJointTorque: NotationSymbol  # vector of the torque each joint drive is rated to deliver repeatedly, against which a checker tests a candidate's inverse-dynamics torques ($\boldsymbol{\tau}_{\text{rated}}$) [N*m]
    RatingCategory: NotationSymbol  # one category of the categorical rating that two raters assign to the same cases, the variable the chance-agreement sum runs over ($y$)
    RatioClippingWidth: NotationSymbol  # half-width of the interval around one to which the clipped surrogate limits the probability ratio ($\epsilon_{\text{clip}}$) [dimensionless]
    RationaleTokens: NotationSymbol  # rationale tokens a policy generates before the action tokens of one response ($z$)
    RawClearanceBarrierFunction: NotationSymbol  # barrier function taken directly as the measured clearance to an obstacle, whose relative degree to acceleration inputs is two ($h_{\text{raw}}$) [m]
    RawDataUploadEnergy: NotationSymbol  # energy one client spends uploading its raw training data for centralized training ($E_{\text{raw}}$) [J]
    RawHardwareFaultRate: NotationSymbol  # rate of hazardous hardware faults before any diagnostic mitigates them, the sum of the component fault rates ($\lambda_{\text{raw}}$) [1/h]
    RawPipelineInput: NotationSymbol  # raw input record before feature transformation, as received by the training or serving pipeline ($x$)
    RawRecordCount: NotationSymbol  # number of logged records, such as control steps, in a telemetry stream before any correction for serial correlation ($N$)
    RawSensoryData: NotationSymbol  # raw sensory data a representation is computed from, treated as a random variable when bounding the task information it carries ($\mathbf{S}$)
    RayReturnRange: NotationSymbol  # distance from the sensor to the measured return along one range ray ($r_{\text{hit}}$) [m]
    RayStepCount: NotationSymbol  # number of voxel cells one ray steps through, its length divided by the voxel pitch ($N_{\text{steps}}$)
    RayTraversalLength: NotationSymbol  # line-of-sight distance one range ray traverses from the sensor origin to the observed surface point ($d_{\text{ray}}$) [m]
    RayVoxelRange: NotationSymbol  # distance from the sensor to a voxel center along the range ray that passes through it ($r_{\text{voxel}}$) [m]
    RayleighDissipationFunction: NotationSymbol  # Rayleigh dissipation function, one half the velocity error weighted by the damping matrix, whose gradient is the viscous damping force ($\mathcal{D}$)
    ReachabilityCostate: NotationSymbol  # costate of the Hamilton-Jacobi-Isaacs equation, the gradient of the reachability value function with respect to the state ($\mathbf{p}$)
    ReachabilityHamiltonian: NotationSymbol  # Hamiltonian of the Hamilton-Jacobi-Isaacs equation: the max-min over control and disturbance of the costate dotted with the dynamics ($\mathcal{H}$)
    ReachabilityHorizonInstant: NotationSymbol  # time instant inside a reachability horizon, between the current time and the terminal time, at which a trajectory is tested for entering the failure set ($t'$) [s]
    ReachabilityLevelFunction: NotationSymbol  # worst-case minimum of the failure margin function along trajectories from a state and time, under the best control against the worst disturbance ($V$)
    ReachabilityTerminalTime: NotationSymbol  # final instant of the finite horizon over which a reachability or safety guarantee is computed ($T$) [s]
    ReachableArtifactNode: NotationSymbol  # artifact node reachable by a directed path from a given node of a candidate plan's dependency graph ($u$)
    ReachableDistance: NotationSymbol  # distance a rest-to-rest profile can cover within a given remaining time under acceleration and speed bounds ($d_{\text{reach}}$) [m]
    ReachableWorkspace: NotationSymbol  # set of end-effector poses the manipulator can reach given its link dimensions, joint travel limits and mounting ($\mathcal{W}$)
    ReadEndSequenceCount: NotationSymbol  # value of a sequence-lock counter a reader loads after copying the shared slot ($c_{\text{end}}$)
    ReadStartSequenceCount: NotationSymbol  # value of a sequence-lock counter a reader loads before copying the shared slot ($c_{\text{start}}$)
    ReadTimeCeiling: NotationSymbol  # longest time a safety read may take, from the sensing allowance or what the declared execution bound leaves for the read ($R$) [s]
    RealDataset: NotationSymbol  # sensitive source dataset a synthetic-data generator is trained on ($\mathcal{S}_{\text{real}}$)
    RealTimeTaskIndex: NotationSymbol  # position of one periodic task in a real-time task set, the index of its period, deadline and execution time ($i$)
    RealityGapVector: NotationSymbol  # vector of the discrepancies between a simulator and the physical machine, one component per channel: dynamics, sensing, appearance and timing ($\boldsymbol{\Delta}_{\text{sim2real}}$) [varies]
    RealizedActuatorEffort: NotationSymbol  # force or torque an actuator actually realizes, the commanded effort clipped to its envelope ($u_{\text{act}}$) [varies]
    RealizedArrivalCount: NotationSymbol  # one possible value of the number of requests that arrive after the trigger in a batching window, used as a summation variable ($n$)
    RealizedDamageOutcome: NotationSymbol  # realized value, zero or one, of the damaging-outcome indicator ($y$) [dimensionless]
    RealizedDampingPower: NotationSymbol  # conservative lower bound on the damping power the actuators actually realize, credited to the energy tank ($P_d$) [W]
    RealizedDeviceComputeRate: NotationSymbol  # floating-point operation rate one device actually achieves on its local work during a training step ($R$) [FLOP/s]
    RealizedFalseAlarmProbability: NotationSymbol  # realized probability that an untruncated sequential test selects the degradation when the baseline holds ($\alpha_{*}$) [dimensionless]
    RealizedMissedDegradationProbability: NotationSymbol  # realized probability that an untruncated sequential test selects the baseline when the specified degradation holds ($\beta_{*}$) [dimensionless]
    RealizedServiceTime: NotationSymbol  # length of one realized service interval, a sample of the service time ($s$) [s]
    ReceiptSampleIndex: NotationSymbol  # index of the IMU sample whose capture time is nearest a delayed observation's receipt ($t_{\text{recv}}$)
    ReceiveBandwidth: NotationSymbol  # rate at which one rank can receive bytes from the fabric ($\text{BW}_{\text{rx}}$) [bytes/s]
    ReceiverOverhead: NotationSymbol  # processor time spent completing the receipt of one message, during which it cannot compute ($o_{\text{recv}}$) [s]
    ReceiverTimingJitter: NotationSymbol  # standard deviation of the time at which a lidar receiver's discriminator registers a returning pulse ($\sigma_{\Delta t}$) [s]
    RecentUsageShare: NotationSymbol  # share of cluster capacity a user or account has recently consumed, as fair-share scheduling measures it ($F_{\text{actual}}$) [dimensionless]
    RecommenderItemCount: NotationSymbol  # number of catalog items a static recommendation system precomputes predictions for ($N_{\text{items}}$)
    RecommenderUserCount: NotationSymbol  # number of users whose predictions a static recommendation system precomputes ($N_{\text{users}}$)
    ReconcilableToolInterface: NotationSymbol  # three-part interface a mutating tool over an endpoint without deduplication ships: an actuation, a verification probe against the pre-dispatch baseline, and an optional compensation ($\mathbb{T}_{a}$)
    ReconstructedGradient: NotationSymbol  # gradient vector a receiver reconstructs by multiplying the quantized codes by their scale ($\hat{\mathbf{g}}$)
    ReconstructedWeight: NotationSymbol  # approximate weight a kernel reconstructs from a weight's integer code times its group scale ($\hat{w}$) [dimensionless]
    RecordAdequacyPassProbability: NotationSymbol  # probability that a candidate passes the record-adequacy and fidelity predicate of an admission route ($p_{\text{record}}$) [dimensionless]
    RecordSizeCeiling: NotationSymbol  # largest size in bytes one trajectory record may reach, against which staging reservations and storage writes are dimensioned ($s_{\max}$) [bytes]
    RecordedAuthorityState: NotationSymbol  # active discrete authority state recorded in the per-tick evidence tuple ($S_{\text{auth}}$)
    RecoveryCompletionInstant: NotationSymbol  # time at which a post-takeover recovery completes and the episode returns to the nominal corridor ($t_{\text{rec}}$) [s]
    RecoveryPoseTolerance: NotationSymbol  # largest distance allowed between a new proposal's first waypoint and the robot's measured rest configuration when proposal authority is restored ($\epsilon_{\text{sync}}$) [varies]
    RecoveryTargetLogPosition: NotationSymbol  # log position up to which a recovery reconstructs record and workspace state ($j_{\text{target}}$)
    RecoveryWarmupTime: NotationSymbol  # time after a restart to refill data pipelines, compile kernels and return training to steady throughput ($T_{\text{warmup}}$) [s]
    RecurrentHiddenBias: NotationSymbol  # bias vector a recurrent network adds in its hidden-state update ($\mathbf{b}_{\text{h}}$)
    RecurrentHiddenState: NotationSymbol  # fixed-size state vector a recurrent network carries from one time step to the next ($\mathbf{h}$)
    RecurrentInputWeights: NotationSymbol  # weight matrix a recurrent network applies to the current input at every step ($\mathbf{W}_{\text{hx}}$)
    RecurrentOutputWeights: NotationSymbol  # weight matrix that maps a recurrent network's hidden state to its output ($\mathbf{W}_{\text{yh}}$)
    RecurrentStateWeights: NotationSymbol  # weight matrix a recurrent network applies to its previous hidden state at every step ($\mathbf{W}_{\text{hh}}$)
    ReduceScatterCompletionTime: NotationSymbol  # time to complete a ReduceScatter over a ring of devices, each ending with the reduced values of its own shard ($T_{\text{reducescatter}}$) [s]
    ReducedBoundaryConditionVector: NotationSymbol  # boundary-condition vector of the reduced system for the three highest polynomial coefficients, after the initial-condition terms are substituted ($\mathbf{b}_{\text{red}}$)
    ReducedImpactMass: NotationSymbol  # two-body reduced mass of a collision between the robot and a body part ($m_{\text{red}}$) [kg]
    ReducibleLoss: NotationSymbol  # part of a scaling-law loss that more parameters or data can remove, the loss minus its irreducible floor ($\hat{\mathcal{L}}$)
    ReductionInputCount: NotationSymbol  # number of values one reduction combines ($n$)
    RedundantComponentCount: NotationSymbol  # number of interchangeable components in a parallel redundant system, any one of which can provide the service ($N_{\text{red}}$)
    RedundantReplicaCount: NotationSymbol  # number of independent replicas of a service in a redundancy availability calculation, any one of which can serve the load ($k$)
    RedundantServiceAvailability: NotationSymbol  # probability that at least one of a service's independent replicas is up at a randomly chosen instant ($A_{\text{system}}$) [dimensionless]
    ReferenceActionDisadvantage: NotationSymbol  # extra expected task cost of taking an action instead of the reference policy's action at one step and then following the reference ($A_t^{\pi^*}$) [dimensionless]
    ReferenceAssessedFailure: NotationSymbol  # event that a trial fails its specified criterion according to an independent reference assessment, the conditioning event of a grader's false-acceptance rate ($F$)
    ReferenceBinIndex: NotationSymbol  # position of one reference bin among the bins a continuous feature is discretized into for the population stability index ($b$)
    ReferenceBinProportion: NotationSymbol  # proportion of reference (training-time) observations falling in one bin of the population stability index ($E$) [dimensionless]
    ReferenceCostToGo: NotationSymbol  # expected task cost from a step through the end of the horizon after taking an action in a state and then following the reference policy ($Q_t^{\pi^*}$) [dimensionless]
    ReferenceDisagreementProbability: NotationSymbol  # probability that a policy's action in a given state differs from the reference policy's action there ($\mathcal{L}$) [dimensionless]
    ReferenceDistribution: NotationSymbol  # distribution whose samples are encoded, the first argument of a KL divergence ($p$)
    ReferenceJointAccelerations: NotationSymbol  # second time derivative of the reference joint trajectory, the feedforward acceleration of computed torque control ($\ddot{\mathbf{q}}^{\text{ref}}$)
    ReferenceJointPosition: NotationSymbol  # position of one joint along the active reference trajectory ($q_{\text{reference}}$) [rad]
    ReferenceJointPositions: NotationSymbol  # time-varying reference trajectory of joint positions that a tracking controller follows ($\mathbf{q}^{\text{ref}}$)
    ReferenceJointVelocities: NotationSymbol  # time derivative of the reference joint trajectory ($\dot{\mathbf{q}}^{\text{ref}}$)
    ReferenceLogProbabilitySum: NotationSymbol  # sum of the frozen reference policy's log-probabilities over a trajectory's selected tokens, precomputed and cached ($\Delta_{\text{ref}}$) [dimensionless]
    ReferenceMixingProbability: NotationSymbol  # probability with which a data-collection round executes the reference action instead of the learner's ($\lambda$) [dimensionless]
    ReferenceNextTokenDistribution: NotationSymbol  # next-token distribution a frozen reference policy assigns at a given prefix ($q$)
    ReferencePosition: NotationSymbol  # commanded Cartesian position the tracked point should follow ($\mathbf{r}$) [m]
    ReferenceProbabilityRatio: NotationSymbol  # ratio of a sampled token's probability under the frozen reference policy to its probability under the current policy ($z$) [dimensionless]
    ReferenceRelativeScore: NotationSymbol  # sum over a trajectory's scored token positions of the log ratio of the trained policy's to the reference policy's token probabilities ($\Delta_\theta$) [dimensionless]
    ReferenceRequestLength: NotationSymbol  # number of tokens in the uncompacted reference request that carries the complete raw history ($S_{\text{ref}}$)
    ReferenceSampleEmpiricalCdf: NotationSymbol  # empirical cumulative distribution function of the first of two samples a two-sample Kolmogorov-Smirnov test compares ($F$)
    ReferenceStateDisagreementRate: NotationSymbol  # probability that a learned policy's action differs from the reference policy's, averaged over the states the reference visits ($\epsilon_E$) [dimensionless]
    ReferenceStateDistribution: NotationSymbol  # average distribution of states the reference policy visits over the horizon ($d_{\pi^*}$)
    ReferenceStepDisagreementRate: NotationSymbol  # probability that a learner's action disagrees with the reference policy at one step, averaged over the reference policy's state distribution at that step ($\epsilon$) [dimensionless]
    ReferenceWindingResistance: NotationSymbol  # winding resistance at the reference temperature, the base of copper's linear resistance law ($R_0$) [Ohm]
    ReflectedContactForce: NotationSymbol  # contact force measured at the follower arm and reflected back to the leader device ($F_{\text{contact}}$) [N]
    ReflectedRotorInertia: NotationSymbol  # apparent inertia of a motor rotor seen at the joint output, scaled by the square of the gear ratio ($J_{\text{ref}}$) [kg*m^2]
    ReflectedRotorInertiaMatrix: NotationSymbol  # diagonal matrix of motor rotor inertias reflected through each gear reduction to the link side ($\mathbf{J}_m$) [kg*m^2]
    ReflectedTransmissionInputInertia: NotationSymbol  # apparent inertia of the motor rotor and transmission input stage seen at the joint output ($J_{\text{ref,in}}$) [kg*m^2]
    ReflectionAngle: NotationSymbol  # angle between a specularly reflected ray and the surface normal ($\theta_{\text{refl}}$) [rad]
    ReflexCollisionBudget: NotationSymbol  # time the permission tick reserves for the every-tick local collision check over all hulls ($T_{\text{reflex\_budget}}$) [s]
    RegeneratedBrakingEnergy: NotationSymbol  # share of a machine's kinetic energy that regenerative braking returns to the DC bus ($E_{\text{regen}}$) [J]
    RegenerativeBrakingTorque: NotationSymbol  # torque the motor applies to decelerate its load during a regenerative stop ($\tau_{\text{brake}}$) [N*m]
    RegenerativeRecoveryEfficiency: NotationSymbol  # fraction of mechanical kinetic energy a decelerating motor returns to the DC bus as electrical energy ($\eta_{\text{regen}}$) [dimensionless]
    RegisterSetupTime: NotationSymbol  # time a register's input must be stable before the capturing clock edge ($t_{\text{setup}}$) [s]
    RegisteredDescriptorCount: NotationSymbol  # number of tool descriptors registered in a toolkit catalog, all of which a call carries without selection ($n_{\text{reg}}$)
    RegularizedEulerIntegral: NotationSymbol  # regularized incomplete beta function, the incomplete Euler integral of the first kind divided by the complete one ($I$)
    RejectionDecodeStep: NotationSymbol  # decode step at which a stream validator detects the first schema violation and rejects the output ($t_{\text{cut}}$)
    RelationalSelection: NotationSymbol  # relational-algebra selection, the operator that keeps only the rows of a table satisfying a predicate ($\sigma$)
    RelativeContrastTolerance: NotationSymbol  # positive tolerance below which the relative contrast between farthest and nearest candidate distances falls with probability tending to one ($\epsilon_{\text{contrast}}$) [dimensionless]
    RelativeDeviationBound: NotationSymbol  # bound on the relative deviation of each query-candidate distance from their common mean ($\delta$) [dimensionless]
    RelativeImprovementFraction: NotationSymbol  # relative fraction by which an intervention lowers the cost per attempt or raises the acceptance probability, in a comparison of the two ($\delta$) [dimensionless]
    RelativeStandardError: NotationSymbol  # standard deviation of an estimate divided by the quantity it estimates ($r$) [dimensionless]
    RelevantRecordSet: NotationSymbol  # set of records labeled relevant to one evaluation query within the eligible snapshot ($\mathcal{D}^*$)
    ReliabilityTargetIndex: NotationSymbol  # position of one reliability target, such as availability, latency, freshness, or model quality, among the targets an evaluation contract budgets ($i$)
    ReliableDetectionFrameCount: NotationSymbol  # number of consecutive camera frames a person must stay in view before the probability that the detector has missed them in every frame falls to the allowed miss probability ($N_{\text{frames}}$)
    RemainingCallCount: NotationSymbol  # number of ordinary model calls a trajectory still makes after a compaction decision point, including the first compacted call ($N_{\text{rem}}$)
    RemainingDistanceEstimate: NotationSymbol  # estimate of the remaining distance to the rack-end upright ($\hat{s}$) [m]
    RemainingExecutionLease: NotationSymbol  # time left on the active execution lease: the lease length minus the time elapsed since its last accepted renewal ($T_{\text{lease,rem}}$) [s]
    RemainingHarmDistance: NotationSymbol  # distance remaining to the point of harm when the observation that distinguishes a hazardous state from a safe one arrives ($d_{\text{rem}}$) [m]
    RemainingIntentTime: NotationSymbol  # time remaining until the intent lease expires, measured from now ($T_{\text{rem}}$) [s]
    RemainingReturn: NotationSymbol  # sum of the rewards a rollout receives from one turn to the end of its horizon, the reward-to-go ($R_{\geq k}$) [dimensionless]
    RemainingUprightDistance: NotationSymbol  # remaining distance from the mobile base to the rack-end upright, the scalar state of the worked Kalman filter ($s$) [m]
    RemoteHardwareEfficiency: NotationSymbol  # fraction of its peak compute rate the remote server sustains on an offloaded request ($\eta_{\text{hw,remote}}$) [dimensionless]
    RemotePathTime: NotationSymbol  # complete time to serve one request remotely: transmission, network round trip and remote compute ($T_{\text{remote}}$) [s]
    RemotePeakRate: NotationSymbol  # peak compute rate of the remote server that would execute an offloaded request ($R_{\text{peak,remote}}$) [FLOP/s]
    RenderedPixelColor: NotationSymbol  # color rendered at one pixel by front-to-back alpha blending of the Gaussians that overlap it ($C$) [dimensionless]
    RenewalJitterAllowance: NotationSymbol  # allowance for variation in the interval between successive chunk admissions ($T_{\text{jitter}}$) [s]
    RenyiDivergence: NotationSymbol  # Renyi divergence of the given order between two probability distributions ($\mathcal{D}_{\alpha_{\text{R}}}$)
    RenyiOrder: NotationSymbol  # order of the Renyi divergence at which a Renyi differential privacy guarantee is stated, a number greater than one ($\alpha_{\text{R}}$) [dimensionless]
    RenyiPrivacyBudget: NotationSymbol  # Renyi differential privacy bound: the largest Renyi divergence of the given order between a mechanism's outputs on adjacent datasets ($\epsilon_{\text{R}}$) [dimensionless]
    RepairAttemptCost: NotationSymbol  # cost of one bounded repair attempt toward a declared readiness endpoint ($C_{\text{repair}}$) [varies]
    RepairCap: NotationSymbol  # maximum number of forward repair attempts on one failed step before the harness compensates or escalates ($K_{\text{repair}}$)
    RepairCapabilityGrant: NotationSymbol  # capability grant a repair task holds over its repository ($\Gamma_{\text{repair}}$)
    RepairFirstCost: NotationSymbol  # total cost of attempting one bounded repair and falling back only if it fails ($C_{\text{repair\ first}}$) [varies]
    RepairPatchArtifact: NotationSymbol  # one patch artifact in a repair candidate's dependency graph, numbered by its subscript ($P$)
    RepairSuccessProbability: NotationSymbol  # probability that one bounded repair attempt reaches the declared readiness endpoint, measured per failure class and feedback policy ($p$) [dimensionless]
    RepairTurnLatency: NotationSymbol  # latency one repair turn adds after a malformed tool call, including the re-sent context and the regenerated call ($T_{\text{repair}}$) [s]
    ReparameterizationNoise: NotationSymbol  # standard normal noise vector that the reparameterization trick scales and shifts into a latent sample, so gradients reach the encoder ($\boldsymbol{\epsilon}$)
    RepeatedRunCount: NotationSymbol  # number of repeated runs of one benchmark configuration, such as runs under different random seeds ($k$)
    RepeatedRunMean: NotationSymbol  # mean of a measured benchmark metric across repeated runs of the same configuration ($\mu_{\text{run}}$) [varies]
    RepeatedRunStandardDeviation: NotationSymbol  # standard deviation of a measured benchmark metric across repeated runs of the same configuration ($\sigma_{\text{run}}$) [varies]
    ReplacementLatency: NotationSymbol  # time from the capture of the observation a candidate trajectory rests on to that candidate's admission into the execution queue ($L$) [s]
    ReplacementTimingReserve: NotationSymbol  # time reserved for scheduling slack before preferred replacement admission and the physical decision-to-stop-onset allowance ($T_{\text{reserve}}$) [s]
    ReplanRequestOffset: NotationSymbol  # time after the active chunk begins executing at which the planner dispatches the request for its successor ($T_{\text{request}}$) [s]
    ReplayBuffer: NotationSymbol  # fixed-size buffer of past training examples that replay-based continual learning samples batches from ($\mathcal{M}$)
    ReplayBufferTransitionCount: NotationSymbol  # number of transitions an off-policy replay buffer holds ($n_{\text{replay}}$)
    ReplayedErrorCovariance: NotationSymbol  # error covariance at the receipt sample after propagating the corrected covariance across the replay interval ($\mathbf{P}_{t_{\text{recv}}}$)
    ReplayedStateEstimate: NotationSymbol  # state estimate at the receipt sample after replaying the buffered inputs from the capture sample ($\hat{\mathbf{x}}_{t_{\text{recv}}}$)
    ReplicaAvailability: NotationSymbol  # probability that one serving replica is up at a randomly chosen instant ($A_{\text{single}}$) [dimensionless]
    ReplicaColdStartTime: NotationSymbol  # time from requesting a new serving replica until it can serve, provisioning plus loading plus warmup ($T_{\text{cold-start}}$) [s]
    ReplicaCount: NotationSymbol  # number of serving replicas among which a router chooses ($R$)
    ReplicaModelLoadTime: NotationSymbol  # time to fetch a model's artifacts and load its weights onto a new serving replica's devices ($T_{\text{load}}$) [s]
    ReplicaProvisioningTime: NotationSymbol  # time to allocate and start the hardware a new serving replica runs on ($T_{\text{provision}}$) [s]
    ReplicaWarmupTime: NotationSymbol  # time a newly loaded serving replica spends compiling and warming caches before it serves at full speed ($T_{\text{warmup}}$) [s]
    ReplyLossProbability: NotationSymbol  # probability that a call's reply is lost in transit ($q_{\text{rep}}$) [dimensionless]
    RepositoryEdgeSourceIndex: NotationSymbol  # index of the symbol a directed edge of a repository symbol graph leaves ($i$)
    RepositoryEdgeTargetIndex: NotationSymbol  # index of the symbol a directed edge of a repository symbol graph enters ($j$)
    RepositoryEdgeWeight: NotationSymbol  # nonnegative weight of the directed edge from one symbol to another in a repository symbol graph ($A$) [dimensionless]
    RepositoryEdgeWeightMatrix: NotationSymbol  # matrix of directed edge weights of a repository symbol graph, one row per source symbol ($\mathbf{A}$)
    RepositoryFileCount: NotationSymbol  # number of files in a shared repository from which workers draw their write sets ($n$)
    RepositoryGraphEdgeSet: NotationSymbol  # set of directed edges of a repository symbol graph, whose direction is declared before ranking ($E$)
    RepositoryGraphVertexSet: NotationSymbol  # set of vertices of a repository symbol graph, one per symbol definition or reference ($V$)
    RepositoryTransitionMatrix: NotationSymbol  # row-stochastic transition matrix of a repository symbol graph, its row-normalized edge weights with dangling rows set to the personalization vector ($\mathbf{P}_{\text{graph}}$)
    ReprojectionResidual: NotationSymbol  # pixel distance between where calibration and joint encoders place a robot-link point and where the camera observes it ($\epsilon_{\text{reproj}}$) [pixels]
    RepublicationIndex: NotationSymbol  # position of one republication in the sequence a transport node emits from a cached message ($k$)
    RepublicationPeriod: NotationSymbol  # period at which a transport node republishes a cached message under fresh timestamps ($T_{\text{rep}}$) [s]
    RequantizationScale: NotationSymbol  # scale that converts an integer matrix multiply's accumulator into the output tensor's quantized range ($s_{\text{eff}}$) [dimensionless]
    RequestCompressionRatio: NotationSymbol  # token count of an uncompacted reference request divided by that of the request a context policy produces at the same decision step ($\gamma_{\text{req}}$) [dimensionless]
    RequestConcurrency: NotationSymbol  # average number of requests in flight inside a stable serving system ($Q_{\text{req}}$) [requests]
    RequestLossProbability: NotationSymbol  # probability that a call's request is lost in transit ($q_{\text{req}}$) [dimensionless]
    RequestOutputLength: NotationSymbol  # number of tokens one request generates during decode ($S$)
    RequestPromptTokens: NotationSymbol  # serialized prompt token sequence a model call request carries: messages, role markers, tool schemas and delimiters ($\mathbf{x}$)
    RequestQueueTime: NotationSymbol  # time a request, such as a model call or a queued job, waits in a queue before its service begins ($T_{\text{queue}}$) [s]
    RequestResidenceTime: NotationSymbol  # time one request spends inside a system from arrival to completion ($W$) [s]
    RequestResponseTime: NotationSymbol  # time one request spends in a serving system, the random variable whose distribution gives tail latency ($t$) [s]
    RequestSeenBatchSize: NotationSymbol  # size of the batch a given request lands in, counting itself and the other arrivals in its window ($B_{\text{req}}$)
    RequestServiceTime: NotationSymbol  # time a server spends serving one request, call or job once its service begins, excluding the wait in queue ($T_{\text{svc}}$) [s]
    RequestTimeInSystem: NotationSymbol  # average time a request spends inside a serving system, waiting plus being served ($T_{\text{lat}}$) [s]
    RequestedActionTap: NotationSymbol  # first stream of the four-stream action tap, the raw policy or operator request before kinematic mapping ($a_{\text{req}}$)
    RequiredClearance: NotationSymbol  # physical clearance the true obstacle must keep from the machine's envelope, such as the base's side clearance ($C_{\text{safe}}$) [m]
    RequiredComponentCount: NotationSymbol  # number of components a job needs available at the same time ($N$)
    RequiredDetectionRange: NotationSymbol  # smallest range at which a forward detector must report a hazard ahead of the machine for the stop to fit ($d_{\text{detect}}$) [m]
    RequiredInputBandwidth: NotationSymbol  # input data rate the storage and data pipeline must deliver so the accelerators reach a target fraction of time computing ($\text{BW}_{\text{required}}$) [bytes/s]
    RequiredMemoryBandwidth: NotationSymbol  # memory bandwidth a workload needs to sustain a target operation rate at its arithmetic intensity ($\text{BW}_{\text{req}}$) [bytes/s]
    RequiredPremiseClaim: NotationSymbol  # one required premise of a conjunctive assurance argument, a child claim that must hold for its parent claim to hold ($P$)
    RequiredPremiseConfidence: NotationSymbol  # probability, as assessed confidence, that one required premise of a conjunctive assurance argument holds ($p$) [dimensionless]
    RequiredPremiseCount: NotationSymbol  # number of independent required premises that jointly support a claim in a conjunctive assurance argument ($n$)
    RequiredResultCount: NotationSymbol  # number of interchangeable admissible results a task contract requires before a fan-out may stop ($k$)
    RequiredScenarioCellCount: NotationSymbol  # number of cells in the scenario table that deployment requires ($N_{\text{req}}$)
    RequiredScenarioSet: NotationSymbol  # set of scenario-table cells, the operating conditions deployment requires the dataset to support ($\mathcal{S}_{\text{req}}$)
    RequiredServiceRate: NotationSymbol  # service rate a server must sustain so peak arrivals keep utilization at the safe ceiling ($\mu_{\text{required}}$) [requests/s]
    RequiredTaskPairCount: NotationSymbol  # number of task pairs a paired comparison needs for its noninferiority gate to clear if the observed difference holds, scaled from the current count by the squared ratio of half-widths ($N'$)
    RequiredUsersPerVariant: NotationSymbol  # number of users each variant of an A/B test needs to detect the minimum detectable effect at the chosen confidence and power ($n$)
    ResampledReplacementToken: NotationSymbol  # token the target model samples from the normalized positive residual to replace the first rejected draft token, written for the fourth position of the illustrated round ($T_4'$)
    ReservedDeviceMemory: NotationSymbol  # device memory held back for the runtime, the operating system and the foreground application, unavailable to background adaptation ($M_{\text{reserve}}$) [bytes]
    ResetCalibrationTime: NotationSymbol  # time to recalibrate sensors and joint encoders between trials, part of the reset latency ($t_{\text{calibration}}$) [s]
    ResetDuration: NotationSymbol  # duration of the scene reset between demonstration attempts ($t_{\text{rst}}$) [s]
    ResetLatency: NotationSymbol  # time to restore a physical test cell between trials: physical restoration, recalibration and telemetry flush ($T_{\text{reset}}$) [s]
    ResidentStopDistance: NotationSymbol  # distance the resident stop travels from speed v to rest once braking begins ($d_{\text{suffix}}$) [m]
    ResidentWeightMemory: NotationSymbol  # memory the frozen backbone weights occupy once staged into working memory for an adaptation update ($M_{\text{weights,resident}}$) [bytes]
    ResidualAdapterFunction: NotationSymbol  # trainable residual adapter transformation added to a hidden representation, two linear projections around a nonlinearity ($A$)
    ResidualBlockOutput: NotationSymbol  # output of a residual block, the residual function plus the identity shortcut ($\mathbf{y}$)
    ResidualCorrectionCeiling: NotationSymbol  # largest norm the gated residual correction may add to the base policy's action ($\Delta_{\max}$) [varies]
    ResidualDynamicsPerturbation: NotationSymbol  # generalized force left uncancelled when computed torque control uses estimated rather than true inertia, Coriolis and gravity terms ($\Delta \mathbf{f}$)
    ResidualFrequencyError: NotationSymbol  # bound on the follower clock's frequency error left after servo correction ($\Delta f_{\text{res}}$) [Hz]
    ResidualFunction: NotationSymbol  # function the weight layers of a residual block compute, which the identity shortcut adds its input to ($\mathcal{F}$)
    ResidualGate: NotationSymbol  # deterministic switch that admits a learned residual correction only inside the certified contact region ($g$) [dimensionless]
    ResidualPolicyTorque: NotationSymbol  # joint torque a learned residual policy with parameters theta adds to the analytical computed-torque command ($\boldsymbol{\tau}_{\text{res},\theta}$)
    ResidualSamplingDistribution: NotationSymbol  # normalized positive part of the target minus draft distribution, from which speculative sampling draws the replacement token at the first rejection ($p_{\text{res}}$)
    ResolutionScalingRatio: NotationSymbol  # ratio by which compound scaling multiplies input resolution for each unit of the scaling coefficient ($\gamma$) [dimensionless]
    ResourceChangeRate: NotationSymbol  # mean rate at which other actors change one observed shared resource, modeled as a Poisson process ($r_{\text{chg}}$) [1/s]
    ResourceCostBudget: NotationSymbol  # largest resource cost a searched architecture may have ($C_{\text{max}}$) [varies]
    ResourceTypeIndex: NotationSymbol  # index of one resource type, such as GPUs, CPU cores or memory, in a multi-resource allocation ($r$)
    ResourceUtilization: NotationSymbol  # utilization of one platform resource over an interval ($U$) [dimensionless]
    ResponseActionTokens: NotationSymbol  # action tokens of one policy response, written with a position index for a single token ($a$)
    ResponseMetadata: NotationSymbol  # metadata a model call's response carries: kind, raw stop reason, token usage, request identifiers, latency and serving parameters ($\mathbf{m}$)
    ResponsePayload: NotationSymbol  # payload a model call returns, which may be complete, partial or absent ($\mathbf{y}$)
    ResponseReward: NotationSymbol  # scalar score a learned reward model assigns to one generated response in reinforcement learning from human feedback ($r$) [dimensionless]
    ResponseThreshold: NotationSymbol  # score threshold at which a monitoring or selective-prediction rule acts, such as alerting, abstaining, or routing to manual review ($\tau$) [dimensionless]
    ResponseTimeDistributionFunction: NotationSymbol  # cumulative distribution function of the time a request spends in a serving system ($F$)
    ResponseTimeIterationIndex: NotationSymbol  # iteration number of the fixed-point recurrence that computes a task's worst-case response time ($\ell$)
    ResponseTimeStandardDeviation: NotationSymbol  # standard deviation of the time a request spends in a serving system ($\sigma_t$) [s]
    RestConfiguration: NotationSymbol  # joint configuration the robot is confirmed to hold at rest after a fault stop ($\mathbf{q}_{\text{rest}}$) [varies]
    RestitutionCoefficient: NotationSymbol  # ratio of separation speed to approach speed in an impact ($e$) [dimensionless]
    RestoreWorkerCount: NotationSymbol  # number of restores a host runs concurrently to replenish a sandbox pool ($w$)
    RestrictedAccelerationBound: NotationSymbol  # largest acceleration a signed restricted release case admits, validated with the rest of its configuration before release ($a_{\text{restricted}}$) [m/s^2]
    RestrictedSpeedBound: NotationSymbol  # largest speed a signed restricted release case admits, validated with the rest of its configuration before release ($v_{\text{restricted}}$) [m/s]
    ResultActionTime: NotationSymbol  # time to deliver an inference result to the application and act on it, the last stage of a streaming pipeline ($T_{\text{action}}$) [s]
    ResumeTime: NotationSymbol  # end-to-end time to resume a trajectory after a failure: ownership, reading, folding, workspace restore, reconciliation and serving-state prefill ($T_{\text{resume}}$) [s]
    RetainedDigestCount: NotationSymbol  # number of previous workspace digests a semantic watchdog retains for recurrence checks ($m$)
    RetainedGradientElementCount: NotationSymbol  # number of largest-magnitude gradient elements top-k sparsification keeps and transmits ($K_{\text{top}}$)
    RetainedProductiveTime: NotationSymbol  # equivalent training time represented by the progress a spot job retains ($T_{\text{productive}}$) [s]
    RetainedSpectralEnergyFraction: NotationSymbol  # fraction of a matrix's total squared Frobenius norm, its activation energy, that a rank-r truncated approximation preserves ($\eta_r$) [dimensionless]
    RetainingCallCount: NotationSymbol  # number of later model calls in a trajectory that retain one staged observation in context ($H_{\text{rem}}$)
    RetiringChunkJointVelocity: NotationSymbol  # reference velocity of one joint along the retiring trajectory chunk at a seam ($\dot{q}_A$) [rad/s]
    RetiringChunkReferenceState: NotationSymbol  # reference plant state along the retiring trajectory chunk at a seam, as a function of time ($\mathbf{x}_A$)
    RetrainingComputeCost: NotationSymbol  # compute cost of executing one retraining run ($C_{\text{compute}}$) [USD]
    RetrainingDeploymentCost: NotationSymbol  # cost of rolling a retrained model into production ($C_{\text{deployment}}$) [USD]
    RetrainingInterval: NotationSymbol  # time between successive retrainings of a deployed model ($T$) [s]
    RetrainingRiskCost: NotationSymbol  # expected cost of a production regression caused by a retrained model ($C_{\text{risk}}$) [USD]
    RetrainingRunCost: NotationSymbol  # fixed total cost of one retraining run, the sum of compute, validation, deployment and risk costs ($C$) [USD]
    RetrainingValidationCost: NotationSymbol  # cost of evaluating a retrained model before it is released ($C_{\text{validation}}$) [USD]
    RetrievalCollectionSize: NotationSymbol  # number of stored candidate vectors a nearest-neighbor query is compared against ($N$)
    RetrievalCutoffCount: NotationSymbol  # number of top-ranked records a retrieval request asks for, the cutoff at which ranking stops and at which precision and recall at k are measured ($k$)
    RetrievalMethodIndex: NotationSymbol  # index of one retrieval method, such as lexical or dense, whose ranked list rank fusion combines ($i$)
    RetrievedRecordSet: NotationSymbol  # set of distinct records a retriever returns up to a cutoff ($\mathcal{R}$)
    RetryAttemptIndex: NotationSymbol  # position of one retry among the retries of an operation, counted from zero ($j$)
    RetryBackoffInterval: NotationSymbol  # upper end of the interval from which a retry's jittered delay is drawn under capped exponential backoff ($b$) [s]
    RetryJitterDelay: NotationSymbol  # delay a client waits before one retry, drawn uniformly between zero and that retry's backoff interval (full jitter) ($d$) [s]
    ReturnLevel: NotationSymbol  # level of a performance metric exceeded on average once per return period, written with the return period as subscript ($z$) [varies]
    ReturnPeriodTrialCount: NotationSymbol  # return period of an extreme value, the number of trials over which its level is exceeded once on average ($T$)
    ReturnStepOffset: NotationSymbol  # number of control steps after the current step at which a reward arrives, summed over in a discounted return ($k$)
    ReturnedClassLabel: NotationSymbol  # label of one class an API returns in its truncated top-k output, paired with its probability ($c$)
    ReusedPrefixLength: NotationSymbol  # number of tokens of one request's prompt whose compatible KV state is attached from a prefix cache without recomputation ($S_{\text{match}}$)
    ReverseModeWorkConstant: NotationSymbol  # constant factor bounding the work of reverse-mode gradient evaluation as a multiple of one function evaluation ($C$) [dimensionless]
    ReverseTransitionCovariance: NotationSymbol  # covariance of the Gaussian reverse transition at one diffusion step ($\boldsymbol{\Sigma}$)
    ReverseTransitionMean: NotationSymbol  # mean of the learned Gaussian reverse transition from one diffusion step to the previous, given the noisy chunk, observation history and step ($\boldsymbol{\mu}_\theta$)
    RevertFidelity: NotationSymbol  # share of a caught defective action's consequence that reverting it avoids ($\rho_{\text{revert}}$) [dimensionless]
    ReviewerFalseApprovalIndicator: NotationSymbol  # indicator that one reviewer approves a candidate known to violate its acceptance criterion, one when it falsely approves and zero otherwise ($X$) [dimensionless]
    ReviewerIndex: NotationSymbol  # index of one reviewer in a review ensemble ($j$)
    RevisionPassProbability: NotationSymbol  # probability that a revised candidate passes its check, given that the initial candidate failed and its diagnostics were fed back ($p_{\text{rev}}$) [dimensionless]
    Reward: NotationSymbol  # per-step reward an engineer writes for policy search ($r$) [dimensionless]
    RewardFunction: NotationSymbol  # function that assigns a scalar reward to taking an action in a state of a decision process ($r$) [dimensionless]
    RewardModel: NotationSymbol  # learned reward model that scores a generated response to a prompt, trained from human preference comparisons ($r$)
    RightEvasionAction: NotationSymbol  # demonstrated action that steers a mobile manipulator around an obstacle on the right ($\mathbf{a}_{\text{right}}$)
    RightSingularVectorMatrix: NotationSymbol  # matrix whose columns are the right singular vectors of a weight matrix ($\mathbf{Q}$)
    RigidGangJob: NotationSymbol  # a rigid distributed job whose devices must be granted together as one gang ($J$)
    RigidImpactInstant: NotationSymbol  # time at which an idealized rigid impact occurs, across which velocity jumps ($t_{\text{impact}}$) [s]
    RigidTransformTranslation: NotationSymbol  # translation part of a rigid transform between two frames ($\mathbf{t}$) [m]
    RingAllReduceBandwidthTime: NotationSymbol  # byte-transfer term of ring AllReduce time, the time spent moving data excluding startup costs ($T_{\text{bandwidth}}$) [s]
    RingAllReduceCompletionTime: NotationSymbol  # time to complete an AllReduce over a ring of devices ($T_{\text{ring}}$) [s]
    RingAllReduceStartupTime: NotationSymbol  # startup term of ring AllReduce time, the per-step startup latency paid over all sequential steps ($T_{\text{latency}}$) [s]
    RingBufferHorizon: NotationSymbol  # oldest observation age an out-of-sequence ring buffer can still roll back to ($T_{\text{buffer}}$) [s]
    RingBufferSlotCount: NotationSymbol  # number of IMU-sample slots an out-of-sequence ring buffer holds ($N_{\text{buffer}}$)
    RingStepIndex: NotationSymbol  # position of one step within a phase of ring AllReduce, counted from one ($s$)
    RingTreeCrossoverPayload: NotationSymbol  # collective payload size above which ring AllReduce finishes faster than tree AllReduce ($M_{\text{crossover}}$) [bytes]
    RmsMotorTorque: NotationSymbol  # root-mean-square motor torque over one duty cycle, the torque whose square sets the cycle's copper heating ($\tau_{\text{rms}}$) [N*m]
    RmspropSquaredGradientAverage: NotationSymbol  # per-parameter running average of squared gradients that RMSprop divides the step by ($s$) [varies]
    RobotAction: NotationSymbol  # action vector a learned robot policy emits, written without a step index, as the argument of a policy, value function or action distribution ($\mathbf{a}$)
    RobotApproachSpeed: NotationSymbol  # speed of the robot toward the person, the robot term of the speed-and-separation distance ($v_r$) [m/s]
    RobotApproachVelocity: NotationSymbol  # rate at which the robot position advances toward an obstacle along the line of approach, the velocity integrated from actuation in a clearance barrier ($v_{\text{robot}}$) [m/s]
    RobotObservation: NotationSymbol  # observation vector a robot policy receives from its sensors, written without a step index ($\mathbf{o}$)
    RobotPosition: NotationSymbol  # position of the robot along the line of approach to an obstacle ($x_{\text{robot}}$) [m]
    RobotPositionUncertainty: NotationSymbol  # position uncertainty of the robot ($Z_r$) [m]
    RobotReactionTime: NotationSymbol  # time from the separation sensor detecting a person to the onset of robot braking, in the speed-and-separation distance ($T_r$) [s]
    RobotReactionTravel: NotationSymbol  # distance a robot travels during its reaction time before stopping begins ($S_r$) [m]
    RobotStoppingTravel: NotationSymbol  # distance a robot travels while stopping ($S_s$) [m]
    RobotStructuralStiffness: NotationSymbol  # stiffness of the manipulator side of a contact, its pads, linkages and joint elasticity acting as one series spring ($k_{\text{robot}}$) [N/m]
    RobotTransitVolume: NotationSymbol  # region the robot sweeps along a candidate corridor up to a given elapsed time ($\mathcal{T}_{\text{robot}}$)
    RollingShutterDistortion: NotationSymbol  # apparent displacement of an image row caused by platform motion during rolling-shutter readout ($\Delta \mathbf{x}$) [m]
    RollingShutterFrameSkew: NotationSymbol  # time between the exposures of the first and last rows of one rolling-shutter frame ($\Delta t_{\text{skew}}$) [s]
    RollingShutterRowReadout: NotationSymbol  # time between the exposure of adjacent rows of a rolling-shutter sensor ($t_{\text{row}}$) [s]
    RolloutAction: NotationSymbol  # response the policy samples at a turn of a reinforcement-learning rollout, a proposed tool call or a final answer ($a$)
    RolloutActionSequence: NotationSymbol  # sequence of actions a policy takes in a rollout from its first step to its horizon ($\mathbf{a}_{0:T}$)
    RolloutAdmissionAge: NotationSymbol  # wall-clock time from sampling a rollout's first prompt token to its admission into the training batch ($\Delta T_{\text{rollout}}$) [s]
    RolloutAssessmentTime: NotationSymbol  # time to run the final assessment of one completed rollout inside its environment slot ($T_{v}$) [s]
    RolloutDuration: NotationSymbol  # wall-clock time one rollout has run since its first prompt token was sampled ($T_{\text{rollout}}$) [s]
    RolloutEnvironmentState: NotationSymbol  # full environment state at a turn of a reinforcement-learning rollout, where turns are indexed by k ($s$)
    RolloutHorizon: NotationSymbol  # number of control steps in one executed rollout, the index of its final state ($T$)
    RolloutLoss: NotationSymbol  # loss of a simulated rollout, the sum over its steps of a per-step loss on the state and parameters ($\mathcal{L}$)
    RolloutPolicyParameters: NotationSymbol  # parameters of the policy that collected the current batch of rollouts, frozen while the updated parameters are optimized ($\theta_{\text{old}}$)
    RolloutPolicyTokenCount: NotationSymbol  # number of tokens the policy sampled in one rollout, excluding prompt, tool-output and environment tokens ($N$)
    RolloutStateSequence: NotationSymbol  # sequence of world states a rollout visits from its first step to its horizon ($s_{0:T}$)
    RolloutStepCount: NotationSymbol  # last step index of a simulated rollout, the number of steps after the initial state over which its loss is summed ($N$)
    RolloutStepIndex: NotationSymbol  # position of one control step in a rollout, counted from zero to the horizon ($k$)
    RolloutSurrogateObjective: NotationSymbol  # contribution of one rollout to the GRPO reward surrogate, its coefficient times the mean of its token log probabilities ($\mathcal{L}$) [dimensionless]
    RolloutTurnIndex: NotationSymbol  # position of an agent-environment turn within a reinforcement-learning rollout, counted from zero ($k$)
    RooflineRidgePoint: NotationSymbol  # arithmetic intensity at which a device's memory-bandwidth limit meets its compute limit ($I_{\text{ridge}}$) [FLOP/byte]
    RotaryRotationMatrix: NotationSymbol  # rotation matrix a rotary position embedding applies to a query or key vector at one position ($\mathbf{R}$)
    RotatingBodyAngularSpeed: NotationSymbol  # angular speed of a spinning component about its axis ($\omega$) [rad/s]
    RotatingBodyInertia: NotationSymbol  # moment of inertia of a rotating component about its axis ($J$) [kg*m^2]
    RotationOriginAcceleration: NotationSymbol  # linear acceleration of the rotation origin of a rigid body ($\mathbf{a}_O$) [m/s^2]
    RotationOriginVelocity: NotationSymbol  # linear velocity of the rotation origin of a rigid body ($\mathbf{v}_O$) [m/s]
    RotationalKineticEnergy: NotationSymbol  # kinetic energy stored in a machine's spinning wheels, gearboxes and motor rotors ($E_{k,\text{rot}}$) [J]
    RotationalSpringStiffness: NotationSymbol  # stiffness of a spring element in a mechanical impedance ($K$) [N*m/rad]
    RotorAngularSpeed: NotationSymbol  # angular speed of a motor rotor about its axis ($\omega_m$) [rad/s]
    RotorInertia: NotationSymbol  # moment of inertia of a motor rotor about its axis ($J_{\text{rotor}}$) [kg*m^2]
    RotorPolePairCount: NotationSymbol  # number of magnetic pole pairs on a motor rotor ($N_{\text{poles}}$)
    RoundCommunicationVolume: NotationSymbol  # total bytes a federated client moves in one round: download, upload, protocol and retry traffic ($M_r$) [bytes]
    RoundDownloadVolume: NotationSymbol  # bytes of model state a federated client downloads in one round ($M_{\text{download},r}$) [bytes]
    RoundDuration: NotationSymbol  # elapsed wall-clock duration of one completed federated round, including selection, local work, transfer, aggregation and waiting ($t_r$) [s]
    RoundEmittedTokenCount: NotationSymbol  # number of tokens one speculative-decoding round emits, its accepted draft prefix plus one token sampled from the target model ($N_{\text{emit}}$)
    RoundProtocolVolume: NotationSymbol  # bytes of authentication and cryptographic-setup traffic a federated client exchanges in one round ($M_{\text{protocol},r}$) [bytes]
    RoundRetryVolume: NotationSymbol  # bytes a federated client spends on failed or repeated transfers in one round ($M_{\text{retry},r}$) [bytes]
    RoundSampleCount: NotationSymbol  # total number of training samples across the clients participating in one federated round ($n_{\text{round}}$)
    RoundUploadVolume: NotationSymbol  # bytes of client update a federated client uploads in one round ($M_{\text{upload},r}$) [bytes]
    RoundedClassProbability: NotationSymbol  # class probability an API returns after rounding the model's score to a fixed number of decimal places ($p^{\text{rnd}}$) [dimensionless]
    RoundingDecimalCount: NotationSymbol  # number of decimal places to which an API rounds the class probabilities it returns ($d_{\text{dec}}$)
    RoutedExpertCount: NotationSymbol  # number of experts a mixture-of-experts router sends each token to ($k_{\text{experts}}$)
    RouterGateProbability: NotationSymbol  # probability a mixture-of-experts router assigns to the expert it selects for a token ($p_{\text{gate}}$) [dimensionless]
    RowActivationDelay: NotationSymbol  # minimum time from activating a DRAM row to the first column access to it, the row-to-column delay ($t_{\text{RCD}}$) [s]
    RowActiveTime: NotationSymbol  # minimum time a DRAM row must stay open after activation before it may be precharged ($t_{\text{RAS}}$) [s]
    RowAttitudeIncrement: NotationSymbol  # rotation a rolling-shutter camera accumulates between the first row's exposure and a given row's ($\Delta \boldsymbol{\theta}_{\text{att}}$) [rad]
    RowConflictCount: NotationSymbol  # number of DRAM row conflicts among the transactions queued ahead of a read, each a request to a different row of a bank whose row is already open, paying a precharge plus an activation ($N_{\text{conflict}}$)
    RowCycleTime: NotationSymbol  # minimum time between consecutive row activations to the same DRAM bank, the row active time plus the precharge time ($t_{\text{RC}}$) [s]
    RowExposureStartTime: NotationSymbol  # instant at which a given row of a rolling-shutter frame begins its exposure, as a function of the row index ($t_{\text{scan}}$) [s]
    RowModeComponentVector: NotationSymbol  # column of the row-mode factor matrix for one rank-one component of a CP decomposition ($\mathbf{a}_s$)
    RowModeFactorMatrix: NotationSymbol  # factor matrix for the row mode of a tensor decomposition, one row per row index and one column per rank component ($\mathbf{A}$)
    RowModeFactorSlice: NotationSymbol  # row of the row-mode factor matrix for one row index, holding its weight on every rank component ($\mathbf{a}$)
    RowParallelOutputMatrix: NotationSymbol  # output of the row-parallel layer of a Megatron tensor-parallel pair, the sum of the per-device partial products ($\mathbf{Z}$)
    RowParallelWeightMatrix: NotationSymbol  # weight matrix of the second layer of a Megatron tensor-parallel pair, split by rows across devices ($\mathbf{B}$)
    RowPrechargeTime: NotationSymbol  # minimum time a DRAM bank needs to close its open row by precharge before another row can be activated ($t_{\text{RP}}$) [s]
    RunIndexSubset: NotationSymbol  # subset of the graded runs' indices of the claim size, over which the pass@k and pass^k estimators average ($J$)
    RuntimeGoodput: NotationSymbol  # fraction of allocated resource time that becomes checkpoint-preserved productive progress ($\eta_{\text{runtime}}$) [dimensionless]
    RuntimeReserveMemory: NotationSymbol  # accelerator memory a serving runtime reserves for dynamic activations and scratch buffers, excluded from the KV pool ($M_{\text{runtime}}$) [bytes]
    SCurveAccelerationPhaseTime: NotationSymbol  # duration of the constant-acceleration hold plus one jerk ramp of a rest-to-rest double-S profile without cruise, half its duration less one ramp ($T_{\text{acc}}$) [s]
    SafeApproachSpeedEnvelope: NotationSymbol  # largest speed from which a machine can still brake to rest before a stopping boundary, as a function of its clearance to that boundary ($v_{\text{safe}}$) [m/s]
    SafeControlSet: NotationSymbol  # set of control inputs that satisfy every active barrier constraint and the actuator limits at the current state ($\mathcal{U}_{\text{safe}}$)
    SafeOverrideCommand: NotationSymbol  # command the permission path substitutes for a proposal it overrides during constrained testing ($\mathbf{u}_{\text{safe}}$)
    SafeSeparationDistance: NotationSymbol  # smallest separation from an obstacle that a clearance requirement allows ($d_{\text{safe}}$) [m]
    SafeSetProjection: NotationSymbol  # minimal-change projection that maps a proposed control command to the closest command keeping the state in the forward-invariant safe set ($\mathcal{P}_{\mathcal{C}}$)
    SafeStateConstraintSet: NotationSymbol  # set of plant states that satisfy the state constraints of a model predictive control problem, such as joint limits and obstacle margins ($\mathcal{X}_{\text{safe}}$)
    SafeStateSet: NotationSymbol  # set of plant states a model-predictive controller keeps every predicted step inside, carrying the collision margins perception supplies ($\mathcal{X}_{\text{safe}}$)
    SafeWorldState: NotationSymbol  # physical world state in which a commanded trajectory executes safely, one of two states a safety filter must tell apart ($s_A$)
    SafeguardSeparationDistance: NotationSymbol  # minimum distance at which ISO 13855 positions a safeguard from a hazard, approach speed times overall stopping time plus an intrusion allowance ($S$) [m]
    SafeguardStoppingTime: NotationSymbol  # overall stopping time of a safeguarded machine in the ISO 13855 separation formula, from the safeguard's actuation to the end of the hazard ($T$) [s]
    SafetyBoundaryManifold: NotationSymbol  # boundary of the safe set, the states where the barrier certificate is exactly zero ($\partial \mathcal{C}$)
    SagaCompensatingStep: NotationSymbol  # compensating operation of a saga, a new forward operation that remediates an earlier step's established external effect ($C$)
    SagaFailedForwardStep: NotationSymbol  # forward step of a linear saga whose failure starts the reverse-order unwind ($F_k^{\text{fail}}$)
    SagaForwardStep: NotationSymbol  # one separately committed forward operation of a saga, which may produce an external effect ($F$)
    SagaStepIndex: NotationSymbol  # position of one step in a saga's sequence of forward steps, used for the step that fails ($k$)
    SampleAcquisitionPrice: NotationSymbol  # price of obtaining one raw training sample ($c_{\text{sample}}$) [USD/sample]
    SampleGradientStandardDeviation: NotationSymbol  # bound on the standard deviation of a single-sample stochastic gradient about the true gradient, so its square bounds the per-sample gradient variance ($\sigma$) [varies]
    SampleLabelingPrice: NotationSymbol  # price of labeling one training sample ($c_{\text{label}}$) [USD/sample]
    SampleStatistic: NotationSymbol  # evaluation statistic computed from the sampled units, recomputed on each bootstrap resample ($\widehat{\theta}$) [varies]
    SampleStorageSize: NotationSymbol  # bytes needed to store one training or serving sample ($s_{\text{sample}}$) [bytes/sample]
    SampledPolicyToken: NotationSymbol  # token the policy sampled at one generation position of a rollout ($w$)
    SampledSimulationParameterVector: NotationSymbol  # simulator parameter vector drawn from the domain randomization distribution ($\tilde{\boldsymbol{\xi}}$) [varies]
    SampledUnit: NotationSymbol  # one independently sampled unit of an evaluation, such as a paired task record, that a bootstrap resamples whole ($y$)
    SampledUnitCount: NotationSymbol  # number of independently sampled units in an evaluation, which each bootstrap resample redraws with replacement ($m$)
    SamplingParameters: NotationSymbol  # sampling configuration of one model call, such as temperature, top-k and top-p, that sets the decoding rule ($\mathbf{\Theta}_{\text{sample}}$)
    SamplingPeriod: NotationSymbol  # time between successive executions of a discrete controller ($T_s$) [s]
    SamplingRate: NotationSymbol  # number of controller executions or signal samples per second ($f_s$) [Hz]
    SamplingTemperature: NotationSymbol  # positive scalar that divides a language model's logits before the softmax at sampling time, sharpening the distribution below one and flattening it above ($\tau_{\text{temp}}$) [dimensionless]
    SandboxColdBootTime: NotationSymbol  # time to cold-boot one sandbox instance: monitor startup, guest kernel boot, and user-space setup ($T_{\text{cold}}$) [s]
    SaturatedJointTorque: NotationSymbol  # joint torque the actuators actually deliver, the commanded torque clipped to the actuator limits ($\boldsymbol{\tau}_{\text{act}}$)
    SavedEngineeringTime: NotationSymbol  # engineering time a shared platform saves for one model in one accounting period ($T_{\text{saved}}$) [h]
    ScalarFieldSize: NotationSymbol  # size in bytes of one scalar field of a wire record, whose offset must be a multiple of it under natural alignment ($k$) [bytes]
    ScalarGraphFunction: NotationSymbol  # scalar-valued function of many inputs, evaluated by a computational graph, whose gradient reverse-mode differentiation computes ($f$)
    ScalarKalmanGain: NotationSymbol  # weight between zero and one that the scalar Kalman update gives the measurement ($g$) [dimensionless]
    ScalarTrackingError: NotationSymbol  # tracking error of a single-degree-of-freedom plant, reference minus measured position ($e$) [varies]
    ScalarTrackingErrorAcceleration: NotationSymbol  # second time derivative of the tracking error of a single-degree-of-freedom plant ($\ddot{e}$) [varies]
    ScalarTrackingErrorRate: NotationSymbol  # time derivative of the tracking error of a single-degree-of-freedom plant ($\dot{e}$) [varies]
    ScaleCoefficient: NotationSymbol  # coefficient of the reducible power-law term of an empirical scaling law ($A_{\mathcal{L}}$) [dimensionless]
    ScaledBatchSize: NotationSymbol  # batch size a training recipe is scaled to under the linear scaling rule ($B_{\text{new}}$)
    ScaledJointStiffnessMatrix: NotationSymbol  # joint torsional stiffness matrix after nondimensional scaling, of order one, so that the physical stiffness is it divided by the squared time-scale ratio ($\tilde{\mathbf{K}}$) [dimensionless]
    ScaledLearningRate: NotationSymbol  # learning rate used at a new batch size under a batch-size scaling rule (linear or square root) ($\eta_{\text{new}}$) [dimensionless]
    ScaledResource: NotationSymbol  # resource quantity a scaling law varies, such as parameters, training tokens or compute ($X$) [varies]
    ScaledSerialFraction: NotationSymbol  # share of a scaled parallel run's execution time that is serial, which sets the scaled speedup in Gustafson's law ($s$) [dimensionless]
    ScalingCurveWorkerCount: NotationSymbol  # number of workers at which a job's scaling-efficiency curve is evaluated, compared with a reference allocation ($k$)
    ScalingEfficiency: NotationSymbol  # measured speedup on N devices divided by the ideal linear speedup N ($\eta_{\text{scaling}}$) [dimensionless]
    ScalingExponent: NotationSymbol  # power-law exponent at which loss falls as a training resource grows ($\alpha_{\text{scale}}$) [dimensionless]
    ScalingLawLoss: NotationSymbol  # loss a trained model reaches as a function of the resource scaled, such as parameters, training tokens or compute, in an empirical power-law scaling law ($\mathcal{L}$)
    ScanCapabilityGrant: NotationSymbol  # read-only capability grant over one source file, derived from the edit grant ($\Gamma_{\text{scan}}$)
    ScatteringCrossSection: NotationSymbol  # effective area with which one particle scatters incident radiation ($\sigma_s$) [m^2]
    ScenarioCellRecordCount: NotationSymbol  # number of eligible retained records in one scenario-table cell under a declared counting unit ($N_{\text{cell}}$)
    SceneRadiance: NotationSymbol  # scene radiance arriving at an image-sensor pixel, as a function of time and pixel address ($I$) [varies]
    ScheduleFinishTime: NotationSymbol  # makespan of one schedule of a dependency graph, the elapsed time from its first start to its last finish ($T_{\text{finish}}$) [s]
    ScheduledSessionDuration: NotationSymbol  # scheduled wall-clock length of one data-collection session ($T_{\text{sched}}$) [s]
    SchedulingClassWeight: NotationSymbol  # weight of one service class in weighted deficit round-robin, the multiplier on the per-round credit quantum ($w$) [dimensionless]
    SchedulingGoodput: NotationSymbol  # fraction of fleet capacity-time during which a job holds all the resources it needs for synchronous progress ($\eta_{\text{scheduling}}$) [dimensionless]
    SchedulingIntervalLength: NotationSymbol  # length of a time interval over which a task set's processor demand is evaluated, the argument of the demand bound function ($t$) [s]
    ScoreNoiseStandardDeviation: NotationSymbol  # standard deviation of the Gaussian noise an API adds to each returned class probability as an extraction defense ($\sigma$) [dimensionless]
    ScoreNoiseVector: NotationSymbol  # Gaussian noise vector an API adds to a model's class-probability vector to make repeated extraction measurements less useful ($\mathbf{z}$)
    ScoreRefreshInterval: NotationSymbol  # number of training epochs between two re-scorings of the candidate pool ($k_{\text{refresh}}$)
    ScoredPrefixLength: NotationSymbol  # number of tokens in the growing candidate prefix that a process scorer reads after one step ($S_{\text{step}}$)
    ScoredStepCount: NotationSymbol  # number of steps in a candidate whose growing prefix a process scorer evaluates, once after each step ($J$)
    ScoredStepIndex: NotationSymbol  # position of one step of a candidate that a process scorer evaluates, counted from one ($j$)
    ScoredToolCall: NotationSymbol  # one proposed tool call in a schema-robustness evaluation, indexed by its position among the scored calls ($a$)
    ScoredToolCallCount: NotationSymbol  # number of proposed tool calls an evaluation scores for schema following and parametric recall ($N$)
    ScoringErrorStandardDeviation: NotationSymbol  # standard deviation of the error a learned scorer adds to each candidate's true utility ($\sigma$) [dimensionless]
    ScreenSpaceCovariance: NotationSymbol  # two-by-two covariance of a Gaussian primitive projected onto the image plane ($\boldsymbol{\Sigma}_{2D}$) [pixels^2]
    ScreenedFeature: NotationSymbol  # one feature, such as a shingle, drawn from a record's feature set ($x$)
    ScreenedFeatureSet: NotationSymbol  # finite feature set extracted from one trajectory record, such as its shingles, compared with another by Jaccard similarity ($X$)
    ScreeningMissRate: NotationSymbol  # fraction of injected payloads a screening classifier fails to catch ($m$) [dimensionless]
    SeamActualAccelerations: NotationSymbol  # joint accelerations of the arm at the activation seam, the initial acceleration a projected trajectory must match ($\ddot{\mathbf{q}}_{\text{actual}}$) [rad/s^2]
    SeamActualConfiguration: NotationSymbol  # joint configuration the arm occupies at the activation seam, the initial condition a projected trajectory must match ($\mathbf{q}_{\text{actual}}$)
    SeamActualVelocities: NotationSymbol  # joint rates of the arm at the activation seam, the initial velocity a projected trajectory must match ($\dot{\mathbf{q}}_{\text{actual}}$) [rad/s]
    SeamBlendReference: NotationSymbol  # reference the quintic blend commands across a trajectory seam, as a function of normalized blend time ($\mathbf{s}$) [varies]
    SeamBridgeDuration: NotationSymbol  # duration of the polynomial bridge that carries a joint from a retiring trajectory chunk onto its successor at a replanning seam ($T_{\text{blend}}$) [s]
    SeamBridgePosition: NotationSymbol  # joint position along the polynomial bridge that carries a joint from a retiring trajectory chunk onto its successor ($q_{\text{blend}}$) [rad]
    SeamInstant: NotationSymbol  # instant at which a replacement trajectory takes over from the retiring one, the anchor of future-state conditioning ($t_{\text{seam}}$) [s]
    SeamStepTorque: NotationSymbol  # inertial torque a joint must supply to close a velocity step at a seam within one servo period ($\tau_{\text{request}}$) [N*m]
    SearchBranchingFactor: NotationSymbol  # number of candidate extensions each expanded node of a search tree proposes before any pruning ($b$)
    SearchDepth: NotationSymbol  # maximum depth a tree search expands ($d_{\max}$)
    SearchDepthIndex: NotationSymbol  # depth of one level of a search tree, counted from the root ($d$)
    SearchQueryEmbedding: NotationSymbol  # embedding of a search query, the full-precision vector a vector index compares against its stored vectors ($\mathbf{q}$)
    SecondAdjacentCheckIndex: NotationSymbol  # label of the second of two adjacent independent admission predicates whose order is compared ($b$)
    SecondAdjacentCheckPassProbability: NotationSymbol  # pass probability of the second of two adjacent independent admission predicates ($p_{b}$) [dimensionless]
    SecondAdjacentCheckSlotTime: NotationSymbol  # expected verifier slot time of the second of two adjacent independent admission predicates ($T_{b}$) [s]
    SecondCatalogDefinitionCount: NotationSymbol  # number of tool definitions each call stages under the second of two compared fixed catalogs ($n_{\text{def},B}$)
    SecondCatalogDefinitionSize: NotationSymbol  # mean number of tokens one tool definition of the second of two compared catalogs contributes to the serialized request ($S_{B}$) [tokens]
    SecondClassCoverageFloor: NotationSymbol  # minimum number of examples of the second class, B, that a coreset must retain ($b$)
    SecondClientMaskedUpdate: NotationSymbol  # masked update the second client, labeled B, uploads in a two-client secure-aggregation illustration ($\mathbf{y}_B$)
    SecondClientModelParameters: NotationSymbol  # model parameters of the second client, labeled B, in a two-client secure-aggregation illustration ($\theta_B$)
    SecondComparedGroupLabel: NotationSymbol  # second of two group labels compared in a group-fairness condition, one value of the protected group attribute ($b$)
    SecondComparedItem: NotationSymbol  # second of two items in a paired comparison, the one the first is preferred over ($j$)
    SecondFloatingPointAddend: NotationSymbol  # second of three floating-point numbers added in either grouping to show that floating-point addition is not associative ($b$) [dimensionless]
    SecondGroupClassificationThreshold: NotationSymbol  # score threshold applied to the second compared group in a group-specific thresholding policy ($\tau_{\text{thr},b}$) [dimensionless]
    SecondGroupQualifiedFraction: NotationSymbol  # fraction of the second compared group's applicants who are qualified, that group's base rate of the positive outcome ($q_b$) [dimensionless]
    SecondGroupSampleApprovalRate: NotationSymbol  # observed share of the second compared group's sampled applicants who are approved ($\hat{p}_b$) [dimensionless]
    SecondGroupSampleSize: NotationSymbol  # number of sampled decisions from the second compared group in a two-proportion test ($n_b$)
    SecondJaccardSet: NotationSymbol  # second of two sets compared by Jaccard similarity, such as another document's shingles ($\mathcal{B}$)
    SecondRankComputeTime: NotationSymbol  # local compute time of the second rank (rank 1) in one synchronous training step ($T_{\text{rank}_1}$) [s]
    SecondRaterCategoryFraction: NotationSymbol  # fraction of the cases the second of two raters assigns to one rating category ($p_{y,2}$) [dimensionless]
    SecondRingTraceElement: NotationSymbol  # second of the four elements of a rank's gradient vector in the four-GPU ring AllReduce trace, each element one chunk ($b$) [varies]
    SecondSummandStandardDeviation: NotationSymbol  # standard deviation of the second of two independent zero-mean Gaussian variables in the variance-addition identity ($\sigma_b$) [dimensionless]
    SecondTwinCriticReturnEstimate: NotationSymbol  # action-value estimate of the second of the two independently parameterized critics in clipped double Q-learning ($Q_{\phi_2}$) [dimensionless]
    SecurityLabelLattice: NotationSymbol  # information-flow security lattice, the set of security labels together with their ordering, join and meet ($\mathbb{L}$)
    SecurityLabelSet: NotationSymbol  # set of security labels an information-flow lattice orders ($\mathcal{S}_{\text{lab}}$)
    SelectedActuationCommand: NotationSymbol  # actuation command the fallback arbiter selects on a tick from the certified command and the resident fallback commands ($\mathbf{u}_{\text{act}}$)
    SelectedDescriptorCount: NotationSymbol  # number of tool descriptors a selection step places in one call's context ($n_{\text{sel}}$)
    SelectedLayerSet: NotationSymbol  # subset of layer indices whose parameters a task-adaptive sparse update trains ($\mathcal{S}$)
    SelectionOverheadSize: NotationSymbol  # number of tokens of overhead a tool-selection step adds to a call ($S_{\text{sel}}$)
    SelectionScoreFunction: NotationSymbol  # function that scores a candidate sample for selection into training ($f$)
    SelectionTime: NotationSymbol  # wall-clock time spent scoring and filtering a candidate pool to choose a training subset ($T_{\text{selection}}$) [s]
    SelectiveAcceptanceScore: NotationSymbol  # score a selective predictor computes for a request, larger when there is stronger evidence that answering it will be correct ($\kappa$)
    SelfConsistencySampleCount: NotationSymbol  # number of reasoning paths or answers sampled for one query to check self-consistency ($k$)
    SelfWatchdogTimeout: NotationSymbol  # interval within which the permission loop must complete all its checks and feed the safety microcontroller's own hardware watchdog before it trips ($T_{\text{self}}$) [s]
    SenderOverhead: NotationSymbol  # processor time spent initiating one message, during which it cannot compute ($o_{\text{send}}$) [s]
    SensingRealityGap: NotationSymbol  # sensing component of the reality gap: discrepancy between simulated and physical sensor transduction, bias, noise and dropout ($\Delta_{\text{sens}}$) [varies]
    SensingStageLatency: NotationSymbol  # time the sensing stage of a blocking sensor-to-actuation path takes, including its queueing and transport delay ($T_{\text{sense}}$) [s]
    SensorErrorBound: NotationSymbol  # bound on the norm of the error between the sensed and the true plant state ($\epsilon$) [varies]
    SensorFeatureTokens: NotationSymbol  # feature tokens an encoder computes from an observation's sensory payload ($\mathbf{z}_{\text{sensor}}$)
    SensorLeverArm: NotationSymbol  # displacement from the body rotation origin to a sensor mounted away from it ($\mathbf{r}_{P/O}$) [m]
    SensorLeverArmLength: NotationSymbol  # length of the lever arm from a rotation center to a mounted sensor ($r$) [m]
    SensorLinkTransportTime: NotationSymbol  # time a frame spends on the sensor link after streaming readout, such as MIPI CSI-2 ($t_{\text{transport}}$) [s]
    SensorNoiseBound: NotationSymbol  # bound on the norm of the sensor measurement noise added to a policy input ($\epsilon$) [varies]
    SensorNoiseDensity: NotationSymbol  # white-noise density of an inertial sensor, its random error per square root of measurement bandwidth ($N_{\text{IMU}}$) [varies]
    SensorNoiseStandardDeviation: NotationSymbol  # standard deviation of the additive Gaussian noise on one sensor channel ($\sigma$) [varies]
    SensorObservationFunction: NotationSymbol  # function that produces the robot's observation from the physical state and the environment parameters ($g$)
    SensorPatternDictionary: NotationSymbol  # dictionary matrix whose columns are the recurring basis patterns of sensor traces ($\mathbf{\Phi}_{\text{dict}}$)
    SensorPitchAngle: NotationSymbol  # pitch angle of a sensor mounting relative to the mobile base frame ($\theta_{\text{pitch}}$) [rad]
    SensorPointAcceleration: NotationSymbol  # linear acceleration of a sensor point fixed to a rotating rigid body ($\mathbf{a}_P$) [m/s^2]
    SensorPointVelocity: NotationSymbol  # linear velocity of a sensor point fixed to a rotating rigid body ($\mathbf{v}_P$) [m/s]
    SensorReadoutTime: NotationSymbol  # time for an image sensor to read out all rows of one frame ($t_{\text{readout}}$) [s]
    SensorReferenceFrame: NotationSymbol  # coordinate frame of the transducer in which a measurement is natively expressed ($\mathcal{F}_{\text{sensor}}$)
    SensorRowIndex: NotationSymbol  # index of a pixel row of a rolling-shutter image sensor, counted in readout order from the first row ($y$)
    SensorTraceMatrix: NotationSymbol  # matrix of stored sensor traces, one column per example, that dictionary learning factorizes ($\mathbf{X}$)
    SequenceCounterReading: NotationSymbol  # value read from a seqlock sequence counter, S_1 before and S_2 after the bounded copy ($S$)
    SequenceIterationOperations: NotationSymbol  # operations one decode iteration spends on one sequence of a batch ($c_{\text{decode}}$) [FLOP]
    SequenceLength: NotationSymbol  # number of tokens in one input sequence ($S$)
    SequenceToken: NotationSymbol  # one token of a serialized input sequence, written with its position as an index ($x$)
    SequentialAttemptIndex: NotationSymbol  # position of one attempt in a sequential retry sequence, counted from zero, so attempt j runs only after j rejections ($j$)
    SequentialBackwardCommunicationTime: NotationSymbol  # time of the backward pass plus gradient communication when communication starts only after the backward pass ends ($T_{\text{sequential}}$) [s]
    SequentialCandidateCount: NotationSymbol  # number of candidates sequential generation starts on one task before the first one passes or the cap is reached, indexed by the cap ($M$)
    SequentialCostPerAcceptedTask: NotationSymbol  # expected cost per accepted task when candidates are generated and checked one at a time until the first passes ($C_{\text{eff,seq}}$) [USD/task]
    SequentialExecutionTime: NotationSymbol  # execution time when data movement, computation and fixed overhead run one after another with no overlap, the sum of the iron-law terms ($T_{\text{sequential}}$) [s]
    SequentialLogLikelihoodRatio: NotationSymbol  # cumulative log-likelihood ratio of the alternative to the null hypothesis after m trials of a sequential test ($\Lambda_m$) [dimensionless]
    SequentialTestObservation: NotationSymbol  # one observation a sequential test collects, indexed in arrival order ($X$)
    SequentialTrialCount: NotationSymbol  # number of trials a sequential test has observed so far, the index of its running log-likelihood ratio ($m$)
    SerialAutocorrelation: NotationSymbol  # lag-one autocorrelation coefficient of a stationary first-order autoregressive telemetry series ($\rho$) [dimensionless]
    SerialCallCount: NotationSymbol  # number of model calls a trajectory makes one after another, each exposed to the per-call latency tail ($n$)
    SerialFraction: NotationSymbol  # share of a fixed workload's single-device execution time that does not shrink when the work is spread across devices, the serial term of Amdahl's law ($r$) [dimensionless]
    SerialLinkClockFrequency: NotationSymbol  # clock frequency of the serial inter-chip link that carries proposals to the safety microcontroller ($f_{\text{link}}$) [Hz]
    SerialReferenceTime: NotationSymbol  # elapsed time of a fixed work partition executed serially on a single thread, the reference a parallel speedup is measured against ($T_{\text{serial}}$) [s]
    SerializationLatency: NotationSymbol  # exposed time a request spends converting data to a wire representation and reconstructing it at the receiver ($L_{\text{lat,serialization}}$) [s]
    SerializedChunkVolume: NotationSymbol  # bytes of one serialized action chunk sent over the inter-chip link ($D_{\text{vol,chunk}}$) [bytes]
    SerializedTrainingExample: NotationSymbol  # one serialized training example, the ordered token sequence a fine-tuning loader builds from a trajectory ($X$)
    SeriesComponentCount: NotationSymbol  # number of components in a series reliability model, every one of which must survive for the system to survive ($N_{\text{series}}$)
    SeriesElasticStiffness: NotationSymbol  # torsional stiffness of the spring in a series elastic actuator ($k$) [N*m/rad]
    SeriesPsuCount: NotationSymbol  # number of power supply units in one node whose failure alone stops the node ($n_{\text{psu}}$)
    ServerAggregationEnergy: NotationSymbol  # energy the server spends aggregating client updates in one federated round ($E_{\text{aggregation}}$) [J]
    ServerIndex: NotationSymbol  # index of one server or replica a router can send a request to ($i$)
    ServerPower: NotationSymbol  # electrical power one complete server draws, its accelerators, CPUs, memory and fans together ($P_{\text{server}}$) [W]
    ServerRoutingWeight: NotationSymbol  # routing weight of one server, proportional to its capacity ($w$) [varies]
    ServiceRate: NotationSymbol  # rate at which one server, such as a worker, consumer, or model endpoint, completes items or calls ($\mu$) [items/s]
    ServiceTimeStandardDeviation: NotationSymbol  # standard deviation of the service time of jobs or requests in a queueing model ($\sigma$) [s]
    ServiceTimeVariationCoefficient: NotationSymbol  # coefficient of variation of service time, its standard deviation divided by its mean ($c_{s}$) [dimensionless]
    ServingCostPerQuery: NotationSymbol  # average cost of serving one query, including idle or reserved capacity billed at the chosen serving boundary ($C_{\text{serving}}$) [USD/query]
    ServingEmpiricalCdf: NotationSymbol  # empirical cumulative distribution function of a feature in the current serving sample ($F_{P_t}$)
    ServingPathPredictor: NotationSymbol  # serving code path viewed as a function from a raw input to the model's output, including its preprocessing and runtime ($f_{\text{serve}}$)
    ServingTransformFunction: NotationSymbol  # transformation the serving pipeline applies to a raw input to produce model features ($f_{\text{serve}}$)
    ServingUtilization: NotationSymbol  # fraction of time a server is busy, offered load divided by service capacity ($\rho_{\text{serv}}$) [dimensionless]
    ServoFrequencyCorrection: NotationSymbol  # frequency correction a clock servo adds to the follower clock at one synchronization step ($\Delta f$) [Hz]
    SessionAttemptCount: NotationSymbol  # number of full attempt-and-reset cycles one scheduled collection session fits ($N_{\text{att}}$)
    SessionRequestCount: NotationSymbol  # number of independent requests a single user session issues, each of which may hit the latency tail ($N_{\text{session}}$)
    SetMemberPoint: NotationSymbol  # point of a set over which the infimum of a point-to-set distance ranges ($\mathbf{z}$)
    SetpointSampleIndex: NotationSymbol  # position of one discrete setpoint sample in a sampled trajectory ($k$)
    SetpointStepInterval: NotationSymbol  # time between consecutive setpoints of an action chunk, the spacing at which the controller consumes them ($\Delta t_{\text{step}}$) [s]
    SettledOperationCharge: NotationSymbol  # final monetary charge settled for one billable operation, posted from authenticated usage evidence ($c$) [USD]
    SettledTrajectoryCost: NotationSymbol  # settled cumulative monetary spend of a trajectory through a turn, from reported usage ($C$) [USD]
    SetupShareCap: NotationSymbol  # largest share of a fresh call's elapsed time that envelope setup may occupy ($\phi$) [dimensionless]
    ShadowOnlyFailureCount: NotationSymbol  # number of paired requests on which the shadow model fails and the production model succeeds ($n_{\text{shadow}}$)
    ShannonEntropy: NotationSymbol  # average uncertainty of a probability distribution, the expected negative log probability of its outcomes ($H$) [bits]
    ShapedObservationSize: NotationSymbol  # number of tokens one staged observation occupies after shaping ($S_{\text{shaped}}$)
    ShapedTotalReward: NotationSymbol  # shaped reward that subtracts torque-effort, torque-slew and winding-temperature penalties from a task reward ($r_{\text{total}}$) [dimensionless]
    ShapingPotential: NotationSymbol  # potential function over decision-process states whose discounted difference, added to the reward, leaves the optimal policy unchanged ($\Phi$) [dimensionless]
    ShardIndex: NotationSymbol  # index of one shard of a partitioned table ($s$)
    ShardReaderCount: NotationSymbol  # number of workers that each choose one dataset shard at random to read next, in a shard-collision estimate ($n_{\text{workers}}$)
    SharedBusBandwidth: NotationSymbol  # sustained transfer rate of the shared on-chip interconnect or memory bus that queued transactions drain through ($\text{BW}_{\text{bus}}$) [bytes/s]
    SharedHardwareInterferenceDelay: NotationSymbol  # cumulative delay proposal-side activity inflicts on the enforcer through shared DRAM channels, caches, interconnects, clock trees, power delivery and thermal budgets ($T_{\text{interf,shared}}$) [s]
    SharedPoolWorkerCount: NotationSymbol  # number of worker slots in a pool shared across external dependencies, which one stalled dependency can exhaust ($N_{\text{pool}}$)
    SharedPrefixFraction: NotationSymbol  # share of each request's KV blocks that a prefix shared across requests covers ($f$) [dimensionless]
    SharedPrefixLength: NotationSymbol  # number of tokens in a prompt prefix that several requests share ($S_{\text{prefix}}$)
    SharedPrefixRequestCount: NotationSymbol  # number of requests that share an identical prompt prefix ($n_{\text{req}}$)
    ShellRelativeThickness: NotationSymbol  # thickness of the outer shell of a ball as a fraction of its radius ($\epsilon$) [dimensionless]
    ShieldInterceptionCoverage: NotationSymbol  # verified probability that the shield intercepts a hazardous proposal of class h ($c_h$) [dimensionless]
    ShiftedMpcControlSequence: NotationSymbol  # candidate control sequence built from the previous optimal sequence shifted by one step, with the terminal feedback input appended ($\tilde{\mathbf{u}}_{0:N_{\text{mpc}}-1}$)
    ShiftedTestDistribution: NotationSymbol  # operational test distribution after a change in trial conditions, such as a loaded tote rack or a moved strike plate ($\mathcal{D}'$)
    ShingleLength: NotationSymbol  # number of consecutive characters or words in one shingle used for near-duplicate detection ($w$)
    ShipDecision: NotationSymbol  # release decision of a declared gate, one when every predicate of the gate holds and zero otherwise ($\mathbf{1}_{\text{ship}}$) [dimensionless]
    ShootThroughCurrent: NotationSymbol  # prospective current through both switches of an inverter half-bridge when they conduct simultaneously ($I_{\text{shoot-through}}$) [A]
    SiblingPolicyAction: NotationSymbol  # any action available at the same state as a given action, the variable a sum over that state's actions runs over ($a'$)
    SiblingTeamAllocation: NotationSymbol  # current allocation of one sibling team under the same department quota ($U_{\text{allocated}}$)
    SightlineRange: NotationSymbol  # distance from a sensor to an observed object along the line of sight, the path length over which the atmosphere attenuates its signal ($d$) [m]
    SigmoidFunction: NotationSymbol  # logistic sigmoid function, one over one plus e to the minus x ($\sigma$)
    SignalDimension: NotationSymbol  # number of real-valued components in one sample of a signal trajectory ($n$)
    SignalHandoff: NotationSymbol  # one of the five canonical signal handoffs of the machine, from sensing to perception through enforcer to actuator, at which interface limits are budgeted ($\mathcal{H}$)
    SignalPropagationDelay: NotationSymbol  # time for a signal to propagate along the fieldbus cabling between nodes ($t_{\text{prop}}$) [s]
    SignalTrajectory: NotationSymbol  # trajectory of a system's real-valued signals over continuous time, the trace a signal temporal logic formula is evaluated on ($s$) [varies]
    SignatureBandCount: NotationSymbol  # number of bands a locality-sensitive hashing index splits a MinHash signature into for trajectory screening ($b$)
    SignificanceCriticalQuantile: NotationSymbol  # standard-normal quantile at one minus half the significance level, the two-sided critical value of a release test ($z_{1-\alpha_{\text{sig}}/2}$) [dimensionless]
    SignificanceLevel: NotationSymbol  # probability that a hypothesis test rejects a true null hypothesis ($\alpha$) [dimensionless]
    SiliconThermalTimeConstant: NotationSymbol  # thermal time constant of a semiconductor junction, the characteristic time for the die to heat during a compute burst ($\tau_{\text{th,silicon}}$) [s]
    SimplexProjection: NotationSymbol  # projection that maps a vector to the closest point of the probability simplex, a valid distribution over the classes ($\Pi_{\Delta^{K-1}}$)
    SimulatedPlantDynamics: NotationSymbol  # numerical simulator model substituted for the physical machine and environment ($f_{\text{sim}}$)
    SimulatedReturnLevel: NotationSymbol  # return level fitted to simulated runs ($z_{T, \text{sim}}$) [varies]
    SimulatedStateTrajectory: NotationSymbol  # state trajectory a simulator produces under declared conditions and a control input history ($s_{\text{sim}}$) [varies]
    SimulatedStlRobustness: NotationSymbol  # quantitative robustness of a signal temporal logic formula on a simulated trajectory ($\rho_{\text{sim}}$) [varies]
    SimulatedTrajectorySet: NotationSymbol  # set of trajectories a simulator produces under a given parameter vector ($\mathcal{S}_{\text{sim}}$)
    SimulatedTransportDelay: NotationSymbol  # transport delay from observation to actuation that the simulator models, typically idealized to zero or one step ($T_{d,\text{sim}}$) [s]
    SimulatedUnbrakedImpactForce: NotationSymbol  # peak unbraked impact force a penalty-contact simulator computes for a mass striking a contact at speed ($F_{\text{unbraked,sim}}$) [N]
    SimulationParameterSet: NotationSymbol  # bounded set of simulator parameter vectors a training scheme considers, the support of domain randomization and the uncertainty set of its minimax form ($\Xi$)
    SimulationParameterVector: NotationSymbol  # vector of physical parameters that defines one simulated environment: link masses, centers of mass, friction, joint damping, transport delays and sensor noise ($\boldsymbol{\xi}$) [varies]
    SimulatorStepFunction: NotationSymbol  # one-step transition function of a differentiable discrete simulator, mapping the state and parameters at a step to the next state ($\mathbf{F}$)
    SingleAgentAcceptanceProbability: NotationSymbol  # probability that one attempt by a single-agent configuration is accepted ($p_{\text{single}}$) [dimensionless]
    SingleDeviceContextLength: NotationSymbol  # longest sequence one device can hold in memory without context parallelism ($S_{\text{single}}$)
    SingleDeviceRunTime: NotationSymbol  # wall-clock time for a fixed amount of productive work on one device ($T_1$) [s]
    SingleJointAccelerationLimit: NotationSymbol  # largest acceleration one joint may be commanded to reach, indexed by joint ($\ddot{q}_{i,\max}$) [varies]
    SingleJointInertia: NotationSymbol  # configuration-dependent inertia one joint presents in a scalar one-joint derivation, the diagonal counterpart of the generalized inertia matrix ($M$) [kg*m^2]
    SingleJointVelocityLimit: NotationSymbol  # largest speed one joint may be commanded to move at, indexed by joint ($\dot{q}_{i,\max}$) [varies]
    SingleLinkBandwidth: NotationSymbol  # bandwidth of one network link in one direction, written in the core bandwidth notation outside alpha-beta formulas ($\text{BW}_{\text{link}}$) [bytes/s]
    SingleStepImitationError: NotationSymbol  # bound on the probability that the imitation policy makes an error at a single step ($\epsilon$) [dimensionless]
    SingularSpectrumMatrix: NotationSymbol  # diagonal matrix holding the singular values of a weight matrix in decreasing order ($\mathbf{\Sigma}$)
    SisaShardCount: NotationSymbol  # number of independent shards, and so of constituent models, that SISA training partitions the data into ($K$)
    SisaTrainingShard: NotationSymbol  # one isolated partition of the training data in SISA training, on which one constituent model is trained ($S$)
    SixthHarmonicTorqueRipple: NotationSymbol  # torque ripple at six times the electrical frequency that dead-time voltage distortion produces in a three-phase motor ($\tau_{\text{ripple}}$) [N*m]
    SkewDisplacementEnvelope: NotationSymbol  # worst-case relative displacement two moving bodies can accumulate across the epoch skew between their state records ($d_{\text{skew}}$) [m]
    SkillSelectionDistribution: NotationSymbol  # distribution over candidate skills given the current state and instruction, proportional to language relevance times affordance ($p$)
    SlaveDriveCount: NotationSymbol  # number of slave drives an EtherCAT frame traverses ($S$)
    SlaveDriveIndex: NotationSymbol  # position of one slave drive along an EtherCAT ring ($s$)
    SlaveForwardingDelay: NotationSymbol  # forwarding delay slave s adds to the EtherCAT frame ($t_{\text{slave},s}$) [s]
    SlipAccelerationBound: NotationSymbol  # bound on the residual acceleration of a tracked object slipping relative to its carrier ($a_{\text{slip}}$) [m/s^2]
    SlipRegularizationSpeed: NotationSymbol  # small positive speed floor in the slip-ratio denominator that keeps it defined near standstill ($\epsilon_{\text{reg}}$) [m/s]
    SloVector: NotationSymbol  # service-level objective vector of one service, one objective per reliability dimension ($\mathbf{S}$)
    SlowdownPersistenceUpdateCount: NotationSymbol  # number of upcoming useful training updates over which a straggler's slowdown is expected to persist ($K$)
    SmallChange: NotationSymbol  # small change, increment or error in the quantity it prefixes ($\delta$)
    SmallPerturbation: NotationSymbol  # small change or error in the quantity it prefixes, treated to first order ($\delta$)
    SmallestJacobianSingularNumber: NotationSymbol  # smallest singular value of the manipulator Jacobian, the tool speed a unit joint rate produces in the weakest direction ($\sigma_{\min}$) [m/rad]
    SmallestKeySingularNumber: NotationSymbol  # smallest singular value of a key activation matrix, the last in decreasing order ($\sigma_{\min(S, d_{\text{head}})}$) [dimensionless]
    SmoothedClassifier: NotationSymbol  # smoothed classifier of randomized smoothing, which returns the class the base classifier predicts most often when Gaussian noise is added to the input ($g$)
    SmoothedRunnerUpProbability: NotationSymbol  # probability that the base classifier predicts the second most likely class when randomized-smoothing noise is added to the input ($p_{B}$) [dimensionless]
    SmoothedTargetAction: NotationSymbol  # next action the target actor proposes after clipped Gaussian noise is added and the result is clipped to the action range ($\tilde{\mathbf{a}}'$)
    SmoothedTopClassProbability: NotationSymbol  # probability that the base classifier predicts the most likely class when randomized-smoothing noise is added to the input ($p_{A}$) [dimensionless]
    SmoothingNoiseClipBound: NotationSymbol  # largest magnitude a component of the target smoothing noise may keep after clipping ($c$) [varies]
    SmoothingNoiseStandardDeviation: NotationSymbol  # standard deviation of the isotropic Gaussian noise randomized smoothing adds to each input coordinate ($\sigma$) [varies]
    SmoothnessConstant: NotationSymbol  # Lipschitz constant of the loss gradient, bounding how fast the gradient can change as the parameters move ($L_s$) [dimensionless]
    SnapshotReadTime: NotationSymbol  # time to read the safety snapshot itself once no traffic waits ahead of it ($t_{\text{own}}$) [s]
    SnapshotResetCycleTime: NotationSymbol  # environment-slot cycle time when each episode ends with a snapshot reset ($T_{\text{cycle,snap}}$) [s]
    SoftExpectedActionReturn: NotationSymbol  # soft action value of maximum-entropy reinforcement learning: the reward for taking an action in a state plus the discounted soft state value of the next state ($Q$) [dimensionless]
    SoftExpectedRemainingReturn: NotationSymbol  # soft state value of maximum-entropy reinforcement learning: the expected soft action value under the policy minus the temperature-weighted log-probability of its action ($V$) [dimensionless]
    SoftenedStudentProbabilityVector: NotationSymbol  # vector of class probabilities the student model outputs from a softmax at the distillation temperature ($\mathbf{p}_{\text{student}}^{(T_{\text{distill}})}$)
    SoftenedTeacherProbabilityVector: NotationSymbol  # vector of class probabilities the teacher model outputs from a softmax at the distillation temperature ($\mathbf{p}_{\text{teacher}}^{(T_{\text{distill}})}$)
    SoftmaxFunction: NotationSymbol  # softmax function, which maps a vector of logits to a probability vector that sums to one ($\sigma$)
    SoftmaxInputScore: NotationSymbol  # one entry of the score vector the online softmax normalizes ($x$) [dimensionless]
    SoftmaxInputVector: NotationSymbol  # vector of scores an online softmax normalizes block by block ($\mathbf{x}$)
    SoftmaxNormalizer: NotationSymbol  # sum of the max-shifted exponentials of a softmax row or block, the softmax denominator, kept as a running sum by online softmax ($z$) [dimensionless]
    SoftmaxOutputProbability: NotationSymbol  # one entry of a softmax output, the probability it assigns to one position ($p$) [dimensionless]
    SoftmaxRowMaximum: NotationSymbol  # largest score in a softmax row or block, subtracted before exponentiation to prevent overflow ($m$) [dimensionless]
    SoftmaxRunningMaximum: NotationSymbol  # largest score seen so far in a block of the online softmax, subtracted before exponentiation ($m$) [dimensionless]
    SoftmaxVectorLength: NotationSymbol  # number of scores in the vector an online softmax normalizes, or in one block of it ($n$)
    SontagStabilizingControl: NotationSymbol  # stabilizing feedback built from a smooth control Lyapunov function by Sontag's universal formula ($\mathbf{u}_{\text{Sontag}}$)
    SourceCurrent: NotationSymbol  # current arriving at a bus node through the source path of battery, harness and switches ($I_{\text{source}}$) [A]
    SourceEventTime: NotationSymbol  # time at which the source event a feature value reflects occurred ($T_{\text{event}}$) [s]
    SourceFramePoint: NotationSymbol  # coordinates of a point expressed in the source frame of a transform ($\mathbf{p}_B$) [m]
    SourceSinkPathSet: NotationSymbol  # set of all source-to-sink paths through an execution dependency graph ($\Pi$)
    SourceVoltage: NotationSymbol  # unloaded voltage of the supply that feeds a bus node through the source path ($V_{\text{source}}$) [V]
    SourceWriteShardGrant: NotationSymbol  # child capability grant proposed for a shard with write access over the source directory ($\Gamma_{c}$)
    SpareStoppingClearance: NotationSymbol  # clearance the finished stopping budget leaves inside the clear distance, the travel that can pay for the dispatch-to-admission interval ($d_{\text{spare}}$) [m]
    SparseMatrixEntryCount: NotationSymbol  # number of entries of a sparse matrix counted in dense form, zeros included, against which CSR storage is compared ($n_{\text{elem}}$)
    SparseMatrixRowCount: NotationSymbol  # number of rows of a sparse matrix, which sets the length of the CSR row-pointer array ($n_{\text{rows}}$)
    SparseProductInputComponent: NotationSymbol  # one component of the dense vector that a sparse matrix-vector product multiplies ($v$) [dimensionless]
    SparseWeightMatrix: NotationSymbol  # weight matrix with many zero entries, as stored for sparse execution ($\mathbf{W}_{\text{sparse}}$)
    SparsityPenaltyWeight: NotationSymbol  # weight of the L1 penalty on weight magnitudes that pushes weights toward zero before pruning ($\lambda_{\text{L1}}$) [dimensionless]
    SparsityRatio: NotationSymbol  # fraction of a weight matrix's entries that are zero ($\rho_{\text{sparse}}$) [dimensionless]
    SpatialClearancePredicate: NotationSymbol  # motion predicate that holds when a candidate point keeps at least a clearance margin from every obstacle in the spatial memory map ($P_{\text{clear}}$)
    SpatialColumnIndex: NotationSymbol  # column coordinate of one output position of a convolution ($j$)
    SpatialCovarianceEmbedding: NotationSymbol  # learned embedding of an observation's spatial covariance ($\mathbf{e}_{\Sigma}$)
    SpatialErrorVector: NotationSymbol  # total error of an estimated point's position: optical noise, transform-chain propagation and uncalibrated mounting bias ($\boldsymbol{\epsilon}_{\text{spatial}}$) [m]
    SpatialHashFunction: NotationSymbol  # spatial hash function mapping a voxel block coordinate to a hash-table bucket index ($h$)
    SpatialMemoryMap: NotationSymbol  # occupancy map in spatial memory that stores the obstacle geometry around the machine ($\mathcal{M}$)
    SpatialPositionCount: NotationSymbol  # number of pixel positions in an image or feature map that a convolution filter is applied at ($N_{\text{pix}}$)
    SpatialRowIndex: NotationSymbol  # row coordinate of one output position of a convolution ($i$)
    SpatialTransformation: NotationSymbol  # spatial transformation, such as a translation, applied to an image or feature map ($\mathcal{T}$)
    SpecialEuclideanAlgebra: NotationSymbol  # Lie algebra of the special Euclidean group, the four-by-four twist matrices that form its tangent space at the identity ($\mathfrak{se}(3)$)
    SpecialEuclideanGroup: NotationSymbol  # group of rigid-body transformations in three dimensions, rotations combined with translations ($SE(3)$)
    SpecialOrthogonalGroup: NotationSymbol  # group of three-by-three rotation matrices, orthogonal with determinant one ($SO(3)$)
    SpecializedEnergyProductivity: NotationSymbol  # useful floating-point operations specialized, domain-specific silicon delivers per joule on a workload suited to its design ($\rho_{\text{energy,specific}}$) [FLOP/J]
    SpecificHeatCapacity: NotationSymbol  # heat needed to raise one kilogram of a coolant by one kelvin at constant pressure ($c_p$) [J/(kg*K)]
    SpectrallyNormalizedWeightMatrix: NotationSymbol  # weight matrix divided by its spectral norm, so its own spectral norm is one ($\overline{\mathbf{W}}$)
    SpeculativeDecodingSpeedup: NotationSymbol  # decoding speedup speculative decoding gives one isolated generation stream, verified tokens per round over target-step equivalents per round ($S_{\text{spec}}$) [dimensionless]
    SpeculativeDraftLength: NotationSymbol  # number of tokens a draft model proposes for the target model to verify in one speculative-decoding round ($K$)
    SpeculativeDraftTokenCount: NotationSymbol  # number of tokens a draft proposes autoregressively for the target to verify in one speculative-sampling iteration ($k$)
    SpeedAtDetection: NotationSymbol  # speed of the machine at the moment the hazard is detected ($v$) [m/s]
    SpeedBound: NotationSymbol  # largest speed the machine, a tracked object or an approaching body is credited with in a budget ($v_{\max}$) [m/s]
    SpendingWindowDuration: NotationSymbol  # duration of the trailing window over which spending is measured or new liabilities are limited ($T_{\text{win}}$) [s]
    SphereSurfaceArea: NotationSymbol  # surface area of the sphere bounding a ball of the ambient dimension, as a function of its radius ($S_{D-1}$)
    SplineControlPoint: NotationSymbol  # control point of a B-spline trajectory, one vertex of the control polygon whose convex hull contains the curve ($\mathbf{p}$) [varies]
    SplineDegree: NotationSymbol  # polynomial degree of the basis functions of a B-spline ($p$)
    SplineFinalControlIndex: NotationSymbol  # index of the last control point of a B-spline, one less than its number of control points ($L$)
    SplineKnotTime: NotationSymbol  # knot of a B-spline trajectory, the timestamp at which adjacent polynomial pieces join, indexed by its position in the knot sequence ($\xi$) [s]
    SplitAssignmentMap: NotationSymbol  # function that assigns a task group to one data partition: training, development or final test ($s$)
    SplitMessageCount: NotationSymbol  # number of equal messages a payload is split into when sent one after another instead of fused into one message ($N_{\text{msg}}$)
    SpotHourlyPrice: NotationSymbol  # price per billed hour of a fixed spot or preemptible allocation ($C_{\text{spot}}$) [USD/hour]
    SpringBrakeEngagementTime: NotationSymbol  # time from de-energizing a spring-applied holding brake until it engages and holds ($t_{\text{engage}}$) [s]
    SpringCompressionForce: NotationSymbol  # force to which a spring is compressed, which sets the potential energy it stores ($F$) [N]
    SprtExpectedLogIncrement: NotationSymbol  # expected increment of a sequential probability ratio test's log-likelihood ratio per labeled observation, under the hypothesis its index names ($m$) [nats]
    SprtLowerBoundary: NotationSymbol  # lower stopping threshold of a sequential probability ratio test, below which the test accepts the null hypothesis ($\Lambda_{\text{lo}}$) [dimensionless]
    SprtLowerLogBoundary: NotationSymbol  # lower stopping boundary on the log-likelihood ratio of a release sequential probability ratio test, at or below which the test selects the baseline hypothesis ($b_{\text{low}}$) [nats]
    SprtStoppingSampleCount: NotationSymbol  # random number of labeled observations a sequential probability ratio test takes before its statistic first crosses a boundary, without truncation ($N_{\text{stop}}$)
    SprtUpperBoundary: NotationSymbol  # upper stopping threshold of a sequential probability ratio test, above which the test accepts the alternative ($\Lambda_{\text{up}}$) [dimensionless]
    SprtUpperLogBoundary: NotationSymbol  # upper stopping boundary on the log-likelihood ratio of a release sequential probability ratio test, at or above which the test selects the degraded hypothesis ($u$) [nats]
    SquareGemmDimension: NotationSymbol  # side length of the square matrices in a matrix multiply, the shared row, column and inner dimension ($N$)
    SquaredGaussianSumQuantile: NotationSymbol  # quantile function of the chi-squared distribution, the sum of squares of independent standard normals ($\chi$)
    SquaredPathVelocity: NotationSymbol  # square of the path velocity, the state variable that makes the path-parameterization constraints linear ($x$) [1/s^2]
    SquaredRadiusSubstitution: NotationSymbol  # integration variable equal to the squared radius, introduced to reduce a radial integral to a gamma function ($u$) [dimensionless]
    SquashedActionComponent: NotationSymbol  # one component of a normalized action after tanh squashing, in the open interval from minus one to one ($a$) [dimensionless]
    SramCapacity: NotationSymbol  # capacity of the fast on-chip SRAM available to one kernel ($M_{\text{SRAM}}$) [words]
    StabilizationErrorThreshold: NotationSymbol  # calibrated limit on the post-takeover tracking-error norm below which operator control counts as stabilized ($\epsilon_{\text{stable}}$) [varies]
    StabilizationSlewCeiling: NotationSymbol  # channel-specific ceiling on the operator command slew below which operator control counts as stabilized ($j_{\text{stable}}$) [varies]
    StackedContactForces: NotationSymbol  # vector stacking the forces at all active contacts, three components per contact ($\mathbf{f}_{\text{c}}$) [N]
    StageConditionalAcceptanceProbability: NotationSymbol  # probability that one stage of a sequential policy, a cascade tier or a loop turn, accepts the task, conditional on the task reaching that stage ($q$) [dimensionless]
    StageEpisodeCount: NotationSymbol  # number of trajectories one stage of a scheduled collection session produces ($N_{\text{episodes}}$)
    StageReachProbability: NotationSymbol  # probability that a task reaches one stage of a sequential policy, a cascade tier or a loop turn, the product of the rejection probabilities of the stages before it ($r$) [dimensionless]
    StagedDefinitionCount: NotationSymbol  # number of tool definitions one model call stages in its request ($n_{\text{def}}$)
    StagingExhaustionTime: NotationSymbol  # time until an initially empty staging store is fully reserved while records arrive faster than stalled verifiers drain them ($T_{\text{exhaust}}$) [s]
    StagingStoreCapacity: NotationSymbol  # total bytes of the staging store that holds collected records awaiting assessment ($M_{\text{stage}}$) [bytes]
    StaleSupportRate: NotationSymbol  # share of decisions requiring current evidence that relied on superseded support ($\epsilon_{\text{stale}}$) [dimensionless]
    StalenessAdjustedLearningRate: NotationSymbol  # learning rate of a stale-synchronous run reduced from the bulk-synchronous rate to compensate for bounded staleness ($\eta'$) [dimensionless]
    StalenessBound: NotationSymbol  # largest number of iterations a stale-synchronous worker may run ahead of the slowest worker before it blocks ($s$)
    StallCounterTorque: NotationSymbol  # step counter-torque a dynamometer applies to an actuator in a mechanical-stall fault trial ($\tau_{\text{stall}}$) [N*m]
    StallTravelBound: NotationSymbol  # bound on the travel from the last valid lease renewal to rest when the proposer stalls (fault F1), as a function of speed ($d_{\text{F1}}$) [m]
    StandardDeviationMultiple: NotationSymbol  # number of standard deviations above the mean at which a one-sided Gaussian exceedance is evaluated ($k$) [dimensionless]
    StandardNormalCumulativeDistribution: NotationSymbol  # cumulative distribution function of the standard normal distribution ($\Phi$)
    StandardNormalQuantile: NotationSymbol  # standard-normal quantile function, the value below which a standard normal variable falls with a given probability ($z_q$)
    StandbyInstanceMemory: NotationSymbol  # private memory one idle standby instance keeps resident ($M_{\text{standby}}$) [bytes]
    StandbyPower: NotationSymbol  # power a device draws in standby, when it is not running inference ($P_{\text{idle}}$) [W]
    StandbyReserveCount: NotationSymbol  # number of clean, unassigned sandbox instances a pool keeps ready ($N_{\text{reserve}}$)
    StandstillSpeedThreshold: NotationSymbol  # speed below which firmware treats a joint as stopped and replaces the kinetic barrier with a static hold check ($\epsilon_{\text{vel}}$) [rad/s]
    StateConstraintMatrix: NotationSymbol  # matrix of stacked linear inequality constraints on the plant state ($\mathbf{H}_x$)
    StateConstraintVector: NotationSymbol  # vector of bounds on the right side of the stacked linear state constraints ($\mathbf{b}_x$)
    StateCotangentSpace: NotationSymbol  # cotangent space at a given plant state, the space of linear functionals on state perturbations that the differentials of output functions live in ($T_{\mathbf{x}}^*$)
    StateDimension: NotationSymbol  # number of components in the state vector a recursive estimator tracks ($n$)
    StateDistribution: NotationSymbol  # average distribution of states a policy visits over the horizon ($d_\pi$)
    StateEstimate: NotationSymbol  # estimate of the state vector that a recursive estimator maintains ($\hat{\mathbf{x}}$)
    StateEstimateDerivative: NotationSymbol  # time derivative of the state estimate of a continuous-time estimator ($\dot{\hat{\mathbf{x}}}$)
    StateEstimationError: NotationSymbol  # error between the true plant state and its estimate ($\mathbf{e}$)
    StateExposureLevel: NotationSymbol  # state exposure level, what a task carries between turns, from nothing to durable shared state ($S$) [dimensionless]
    StateNormWeighting: NotationSymbol  # declared reference scales and dimensionless task weights that turn a mixed-unit state error into a dimensionless weighted norm ($W$)
    StateOperation: NotationSymbol  # operation that maps a system's stored state to a new state, such as one pipeline write ($f$)
    StateParameterSensitivity: NotationSymbol  # sensitivity of a simulated state to the parameters, the Jacobian of the state at a step with respect to the parameters ($\mathbf{S}$)
    StateRecurrencePeriod: NotationSymbol  # number of observation steps between the current workspace digest and the earlier digest it matches ($\ell$)
    StateScalarFunction: NotationSymbol  # generic smooth scalar function of the plant state, used to define its differential ($\phi$)
    StateSpaceDimension: NotationSymbol  # number of continuous coordinates in the task-relevant state ($d_s$)
    StateSpaceInputMatrix: NotationSymbol  # input matrix of a linear state-space model, mapping control inputs to state rates ($\mathbf{B}$)
    StateSpaceSystemMatrix: NotationSymbol  # system matrix of a linear state-space model, governing the unforced state dynamics ($\mathbf{A}_{\text{sys}}$)
    StateTokenCount: NotationSymbol  # number of proprioceptive state tokens the robot's configuration contributes to an embodied transformer's input ($K_s$)
    StateTrackingWeightMatrix: NotationSymbol  # positive semidefinite matrix weighting the deviation of a projected trajectory from the learned nominal trajectory ($\mathbf{W}_q$) [varies]
    StateTransitionMatrix: NotationSymbol  # discrete-time matrix that propagates the state one step forward in a Kalman filter ($\mathbf{F}$)
    StatelessCursorPayload: NotationSymbol  # serialized continuation state a stateless cursor carries: a query descriptor, the last evaluated sort key and an issuance timestamp ($d$)
    StaticBatchingOperations: NotationSymbol  # operations static (traditional) batching performs on one batch, counting the padded iterations of sequences that already finished ($O_{\text{traditional}}$) [FLOP]
    StaticChassisPower: NotationSymbol  # baseline power the chassis draws regardless of compute: sensors, cooling, holding currents ($P_{\text{chassis\_static}}$) [W]
    StaticDataDistribution: NotationSymbol  # fixed data distribution from which a static benchmark draws its held-out examples ($p_{\text{data}}$)
    StationaryConditionPolynomial: NotationSymbol  # polynomial in the worker count whose positive root is the stationary point of the normalized elapsed-time ratio in the pairwise overhead model ($P$) [dimensionless]
    StatisticalDivergence: NotationSymbol  # divergence of the current data distribution from the training distribution, a nonnegative measure of drift ($\mathcal{D}(P_t \| P_0)$) [dimensionless]
    StatisticalFalseNegativeRate: NotationSymbol  # allowed probability of a type II error, a release comparison missing a true difference of the target size ($\beta$) [dimensionless]
    StatisticalHypothesis: NotationSymbol  # a statistical hypothesis about a population, such as the null hypothesis H_0 or the alternative H_1 ($H$)
    StatisticalPowerQuantile: NotationSymbol  # standard-normal quantile at the target statistical power of a release test, one minus its type II error rate ($z_{1-\beta}$) [dimensionless]
    StatisticalSignificanceLevel: NotationSymbol  # allowed probability of a type I error, rejecting a true null hypothesis ($\alpha_{\text{sig}}$) [dimensionless]
    StatorCoreTemperature: NotationSymbol  # temperature of the stator core in a two-node motor thermal model ($T_{\text{core}}$) [degC]
    StatorIronTemperature: NotationSymbol  # temperature of the stator iron lamination node of a multi-node motor thermal network ($T_{\text{iron}}$) [degC]
    StatorThermalCapacitance: NotationSymbol  # heat capacity of the motor stator and winding, energy per degree of temperature rise ($C_{\text{th}}$) [J/K]
    StatusPollCount: NotationSymbol  # number of intermediate model-directed status calls made while an asynchronous job runs ($N_{\text{poll}}$)
    SteadyDeceleration: NotationSymbol  # constant deceleration a carried container undergoes, which tilts the free surface of the liquid inside ($a$) [m/s^2]
    SteadyStateAvailability: NotationSymbol  # probability that a component, replica or repairable system is up at a randomly chosen instant, its long-run fraction of up time ($A$) [dimensionless]
    SteadyStateCurrent: NotationSymbol  # current a winding approaches in steady state after a voltage step ($I_{\text{final}}$) [A]
    SteadyStateErrorCovariance: NotationSymbol  # steady-state covariance of the state-estimation error, the solution of the filter algebraic Riccati equation ($\mathbf{P}_{\text{est}}$)
    SteadyStateIntegratorState: NotationSymbol  # value the integral-action state settles at, where the integral command cancels a constant disturbance ($\mathbf{z}_{\text{ss}}$)
    SteadyStateTemperatureRise: NotationSymbol  # steady-state temperature rise a single thermal node approaches under constant heat input, heat input times thermal resistance ($\Delta T_{\max}$) [K]
    SteadyStateTrackingError: NotationSymbol  # tracking error vector at rest once transients decay, the offset a constant disturbance leaves under PD control ($\mathbf{e}_{\text{ss}}$)
    SteeringTransferTime: NotationSymbol  # finite time within which some input must steer the state from the initial state to the target state, in the definition of controllability ($T$) [s]
    StepFailureHazard: NotationSymbol  # negative natural logarithm of the per-step success probability, the per-step failure hazard ($a$) [dimensionless]
    StepGenerationCost: NotationSymbol  # cost of generating one useful step, assumed the same for every step ($C_{\text{gen}}$) [varies]
    StepGenerationTime: NotationSymbol  # time to generate one step in a complete-block verification model ($T_{\text{step}}$) [s]
    StepIoTime: NotationSymbol  # time the input pipeline needs to read, decompress, decode and stage the data for one training step ($T_{\text{IO}}$) [s]
    StepResponsePhaseLag: NotationSymbol  # phase by which a physical actuator's step response lags the simulated response at a test frequency, positive for lag ($\Delta \phi$) [deg]
    StepSettlingTime: NotationSymbol  # time a step response takes to enter and stay within its settling band ($t_{\text{settle}}$) [s]
    StepSuccessProbability: NotationSymbol  # probability that one generation step succeeds, independently of the others ($p$) [dimensionless]
    StepTestVoltage: NotationSymbol  # small voltage step applied across a held motor winding to measure its electrical time constant ($V_{\text{step}}$) [V]
    StepTimePenalty: NotationSymbol  # constant reward penalty charged on every control step of an episode, which rewards finishing sooner ($c_{\text{time}}$) [dimensionless]
    StepTokenLogit: NotationSymbol  # unnormalized logit the model assigns to one vocabulary token at one decoding step, before any constraint mask ($z_{t,v}$) [dimensionless]
    StereoBaseline: NotationSymbol  # distance between the optical centers of a stereo camera pair ($b$) [m]
    StereoDisparity: NotationSymbol  # horizontal pixel offset between the images of one scene point in the two views of a rectified stereo pair ($d$) [pixels]
    StlEventuallyDeadline: NotationSymbol  # upper time bound of a signal temporal logic Eventually operator, the deadline within which the property must become true ($\tau$) [s]
    StlFormula: NotationSymbol  # signal temporal logic formula a signal must satisfy, such as a safety invariant or a reachability deadline ($\phi$)
    StlIntervalEnd: NotationSymbol  # end offset, relative to the evaluation time, of the time interval of a timed temporal-logic operator ($b$) [s]
    StlIntervalStart: NotationSymbol  # start offset, relative to the evaluation time, of the time interval of a timed temporal-logic operator ($a$) [s]
    StlPredicateFunction: NotationSymbol  # real-valued function of a signal whose sign decides an atomic temporal-logic predicate ($f$) [varies]
    StlRobustness: NotationSymbol  # signed margin by which a signal satisfies (positive) or violates (negative) a signal temporal logic formula at a time, its quantitative robustness ($\rho$) [varies]
    StlUntilOperator: NotationSymbol  # until operator of signal temporal logic over a time interval: the left formula holds until a time within the interval at which the right formula holds ($\mathcal{U}_{[a, b]}$)
    StockoutNormalQuantile: NotationSymbol  # standard normal quantile at one minus the stockout probability target, the reserve's safety factor ($z_{1-\epsilon}$) [dimensionless]
    StockoutProbabilityTarget: NotationSymbol  # largest acceptable probability that an arriving task finds the standby reserve empty ($\epsilon$) [dimensionless]
    StopAxisAcceleration: NotationSymbol  # acceleration along the admitted path during a stop, negative while braking, in one-dimensional stop derivations ($\ddot{q}_{\text{stop}}$) [m/s^2]
    StopAxisPosition: NotationSymbol  # distance travelled along the admitted path during a stop, in one-dimensional stop derivations ($q_{\text{stop}}$) [m]
    StopCompletionInstant: NotationSymbol  # instant, on the permission clock, at which the matched stop of an admitted trajectory record reaches rest ($t_{\text{stop\_end}}$) [s]
    StopDecisionComputeBound: NotationSymbol  # bound on the stop-decision computation and any still-required input transfer that a hazard-triggered stop spends inside the pre-brake delay ($T_{\text{compute}}$) [s]
    StopProgress: NotationSymbol  # time elapsed since a stop began divided by the stop duration, from zero to one ($u$) [dimensionless]
    StopSequenceSet: NotationSymbol  # set of token patterns that make a serving engine halt decoding when the output matches one ($\mathcal{S}_{\text{stop}}$)
    StoppingBoundaryClearance: NotationSymbol  # remaining distance from the machine to a stopping boundary along its direction of travel, after sensing and actuation latencies are charged ($z$) [m]
    StoppingCrossoverSpeed: NotationSymbol  # speed at which travel during the pre-brake delay equals braking travel in the elementary stopping budget, twice the credible deceleration times the pre-brake delay ($v^{*}$) [m/s]
    StoppingDeadline: NotationSymbol  # latest instant at which authority can be revoked and a worst-case stop still ends within the verified clearance ($t_{\text{deadline,stop}}$) [s]
    StoppingDistance: NotationSymbol  # distance the machine travels from hazard detection to rest, including travel during the pre-brake delay ($d_{\text{stop}}$) [m]
    StoppingDistanceBudget: NotationSymbol  # distance available for a stop, the clearance to the stop line less the reserve held back for uncertainty and residual clearance ($d_{\text{budget}}$) [m]
    StoppingDuration: NotationSymbol  # time from braking onset to standstill ($T_{\text{stop}}$) [s]
    StorageAccessFactor: NotationSymbol  # fraction of peak storage throughput a workload's access pattern achieves ($F_{\text{access}}$) [dimensionless]
    StorageBandwidth: NotationSymbol  # sustained rate at which the storage system delivers bytes to the input pipeline ($\text{BW}_{\text{storage}}$) [bytes/s]
    StorageCapacityPrice: NotationSymbol  # provider price charged to retain one byte of telemetry for one unit of time ($C_{\text{storage/byte-time}}$) [USD/(byte*s)]
    StorageFunction: NotationSymbol  # storage function of a passive system, a non-negative function of the state equal to the energy it holds internally ($V$) [J]
    StorageFunctionDerivative: NotationSymbol  # time derivative of a passive system's storage function along its trajectories ($\dot{V}$) [W]
    StorageIngestBandwidth: NotationSymbol  # aggregate rate at which a storage tier absorbs writes, such as a checkpoint burst ($\text{BW}_{\text{ingest}}$) [bytes/s]
    StoragePrice: NotationSymbol  # price of keeping one gigabyte in storage for one month ($c_{\text{storage}}$) [USD/GB/month]
    StorageReadTime: NotationSymbol  # time to read one batch's raw data from storage at the effective storage throughput ($T_{\text{read}}$) [s]
    StorageRetentionTime: NotationSymbol  # length of time a dataset is kept in storage ($T_{\text{months}}$) [months]
    StorageThroughput: NotationSymbol  # rate at which storage can deliver training data, bounded by disk and network bandwidth ($R_{\text{storage}}$) [bytes/s]
    StoredDataVolume: NotationSymbol  # bytes a retained dataset occupies in storage ($D_{\text{vol,store}}$) [bytes]
    StoredEmbedding: NotationSymbol  # embedding a frozen feature encoder computes for one example, stored in place of the raw input for replay ($\mathbf{z}$)
    StoredIndexVector: NotationSymbol  # one vector stored in a vector index, scored against a search query embedding ($\mathbf{x}$)
    StoredSystemState: NotationSymbol  # stored state an operation reads and writes, such as a register, counter or table ($s$)
    StoredTokenVolume: NotationSymbol  # total stored tokens of one trajectory category, its record count times the mean stored tokens per record ($V$) [tokens]
    StragglerStepTime: NotationSymbol  # time of one synchronous training step on N devices when the slowest rank sets the pace and communication does not overlap compute ($T_{\text{step,straggler}}$) [s]
    StreamAcceptanceProbability: NotationSymbol  # stationary probability that one monitored task is accepted, the Bernoulli parameter a sequential release test compares at two hypothesized values ($p$) [dimensionless]
    StreamingEndToEndLatency: NotationSymbol  # time from capturing one streaming input to acting on its result, summed over capture, transfer, inference and action ($T_{\text{streaming}}$) [s]
    StrideCheckpointCount: NotationSymbol  # number of checkpoints a policy takes over a trace, written with its stride as subscript ($N$)
    StrongConfigurationAttemptCost: NotationSymbol  # all-in monetary cost of one attempt by the stronger, more expensive agent configuration ($c_{\text{big}}$) [USD]
    StrongConfigurationPassProbability: NotationSymbol  # probability that one attempt by the stronger, more expensive agent configuration is accepted ($p_{\text{big}}$) [dimensionless]
    StruckObjectMass: NotationSymbol  # mass of the object a robot strikes in a contact or collision, infinite for an immovable frame ($m_{\text{obj}}$) [kg]
    StructuredSparsityGroupSize: NotationSymbol  # number of consecutive values in one group of an N:M structured-sparsity pattern ($m_{\text{group}}$)
    StructuredSparsityNonzeroCount: NotationSymbol  # number of values kept nonzero in each group of an N:M structured-sparsity pattern ($n_{\text{keep}}$)
    StudentProbabilityVector: NotationSymbol  # vector of class probabilities the student model outputs ($\mathbf{p}_{\text{student}}$)
    StuffBitTime: NotationSymbol  # time frame i spends on inserted stuff bits ($t_{\text{stuff},i}$) [s]
    SubgroupIndex: NotationSymbol  # label of one population subgroup whose outcomes are compared against a reference group ($g$)
    SubgroupSampleSize: NotationSymbol  # number of labeled examples from one subgroup used to estimate that subgroup's error rate ($n$)
    SubintervalFailureProbability: NotationSymbol  # probability of a failure within one short subinterval of operating time, the rate times the subinterval length ($p$) [dimensionless]
    SublevelSetThreshold: NotationSymbol  # level of the Lyapunov function that bounds a sublevel set ($c$) [varies]
    SubsequentArrivalTime: NotationSymbol  # time from the request that opened a batching window until a later arrival, indexed by arrival order ($\tau$) [s]
    SubsetAllPassIndicator: NotationSymbol  # indicator that every run in a subset of graded runs passed ($I_J^{\text{all}}$) [dimensionless]
    SubsetAnyPassIndicator: NotationSymbol  # indicator that at least one run in a subset of graded runs passed ($I_J$) [dimensionless]
    SubsetDatasetSize: NotationSymbol  # number of samples a selection method retains from the full dataset for training ($D_{\text{subset}}$)
    SuccessCountIndex: NotationSymbol  # possible number of successes over which a binomial tail probability sums ($j$)
    SuccessLowerConfidenceLimit: NotationSymbol  # one-sided lower confidence limit on the per-trial success probability from a finite campaign ($p_{\text{lower}}$) [dimensionless]
    SuccessfulTrialCount: NotationSymbol  # number of successful trials observed in a campaign ($k$)
    SuccessorActionVector: NotationSymbol  # continuous action vector taken in the successor state of a transition, the next action a Bellman backup evaluates ($\mathbf{a}'$)
    SuccessorPlanningState: NotationSymbol  # discrete world state a symbolic task planner reaches after applying one operator, the current state with the operator's delete list removed and its add list added ($s'$)
    SuffixFoldTime: NotationSymbol  # time to fold a trajectory's logged event suffix into its record during resume ($T_{\text{fold}}$) [s]
    SummaryOutputLength: NotationSymbol  # number of output tokens the summarization call generates during context compaction ($K_{\text{sum}}$)
    SummaryTokenEmbedding: NotationSymbol  # learned summary token a vision transformer prepends to its patch tokens ($\mathbf{x}_{\text{class}}$)
    SupervisoryMtbi: NotationSymbol  # mean time between interventions (MTBI): total active operating time of a campaign divided by its count of interventions of one class ($T_{\text{MTBI}}$) [s]
    SupervisoryTakeoverLatency: NotationSymbol  # total time from a takeover demand to the first checked human command, the human sensory, cognitive and motor times plus console transport and handshake ($T_{\Sigma}$) [s]
    SupplyVoltage: NotationSymbol  # supply voltage that powers a digital circuit, the quantity dynamic voltage and frequency scaling lowers ($V$) [V]
    SupportPolygon: NotationSymbol  # support polygon of a robot on coplanar contacts, the convex hull of its active contact patches projected onto the ground plane ($\mathcal{P}_{\text{support}}$)
    SupportSetSize: NotationSymbol  # number of examples in a few-shot support set, counted over all classes ($n_{\text{shot}}$)
    SustainableAttemptRate: NotationSymbol  # rollout attempts per second a collector sustains, the minimum of its generation, assessment and storage ceilings ($R_{\text{attempt}}$) [1/s]
    SustainableCopperLossPower: NotationSymbol  # largest copper loss a winding can dissipate at thermal steady state without exceeding its temperature limit ($P_{\text{loss,ss}}$) [W]
    SustainableHoldingTorque: NotationSymbol  # largest joint torque a motor can hold at stall without its winding exceeding the insulation limit ($\tau_{\text{hold,max}}$) [N*m]
    SustainedComputeRate: NotationSymbol  # operation rate an accelerator or system actually sustains on a workload ($R_{\text{sustained}}$) [FLOP/s]
    SustainedDropoutFrameCount: NotationSymbol  # number of consecutive sensor frames lost in a sustained dropout ($N_{\text{drop}}$)
    SustainedMemoryBandwidth: NotationSymbol  # memory bandwidth achievable in practice under the target access pattern ($\text{BW}_{\text{mem}}$) [bytes/s]
    SweepCandidateCostFraction: NotationSymbol  # cost of one sweep candidate as a fraction of the final training run's cost ($f$) [dimensionless]
    SweepCandidateCount: NotationSymbol  # number of candidate models a hyperparameter or architecture sweep trains before the final run ($k$)
    SwitchBreakdownVoltage: NotationSymbol  # voltage rating of an inverter's power switches, above which they can fail ($V_{\text{breakdown}}$) [V]
    SwitchInputCapacitance: NotationSymbol  # input capacitance of a power switch seen at its gate, gate-source plus gate-drain capacitance ($C_{\text{iss}}$) [F]
    SwitchOnResistance: NotationSymbol  # drain-source resistance of a power switch in its on state ($R_{\text{ds(on)}}$) [Ohm]
    SwitchRadix: NotationSymbol  # number of ports on one network switch, its radix ($k$)
    SwitchedCapacitance: NotationSymbol  # capacitance of the wires and gates a circuit charges when it switches ($C$) [F]
    SwitchingActivityFactor: NotationSymbol  # average fraction of a chip's gate capacitance that switches each clock cycle ($\alpha$) [dimensionless]
    SwitchingEnergy: NotationSymbol  # dynamic energy a circuit dissipates charging and discharging its switched capacitance ($E$) [J]
    SwitchingTransitionTime: NotationSymbol  # duration of one switching transition of a power switch, over which the gate charge is transferred ($t_{\text{sw}}$) [s]
    SymbolicPlanningState: NotationSymbol  # discrete world state of a symbolic task planner, the set of grounded predicates that currently hold ($s$)
    SymmetricClippingThreshold: NotationSymbol  # largest magnitude a symmetric quantizer represents, so values are clipped to plus or minus this bound ($\alpha_{\text{sym}}$) [varies]
    SymmetricMessageOverhead: NotationSymbol  # processor time spent initiating or completing one message in the symmetric LogP model, where sender and receiver overheads are equal, during which the processor cannot compute ($o$) [s]
    SymplecticEulerTransitionMatrix: NotationSymbol  # discrete-time transition matrix that semi-implicit (symplectic) Euler integration produces ($\mathbf{A}_{\text{symp}}$)
    SyncMessageInterval: NotationSymbol  # interval between successive clock-synchronization messages, over which residual frequency error accumulates as offset ($\Delta t_{\text{interval}}$) [s]
    SyncPairingDisplacement: NotationSymbol  # displacement the arm travels during an uncompensated lag between an observation and the action it is paired with ($\Delta \mathbf{x}_{\text{sync}}$) [m]
    SynchronizationBlackout: NotationSymbol  # interval during which a node receives no synchronization message and runs on its own clock ($T_{\text{blackout}}$) [s]
    SynchronizationClearanceMargin: NotationSymbol  # extra clearance that covers the travel of an observed entity during the clock-synchronization error bound ($d_{\text{sync}}$) [m]
    SynchronizationErrorBound: NotationSymbol  # bound on the error of converting a timestamp from one node's clock to another's ($\epsilon_{\text{sync}}$) [s]
    SynchronizationStepIndex: NotationSymbol  # index of one synchronization step of a clock servo ($k$)
    SynchronizationTime: NotationSymbol  # time per step lost to barriers, straggler waiting and other coordination at scale N ($T_{\text{sync}}$) [s]
    SyntheticDataGenerator: NotationSymbol  # generative model with learned parameters that maps latent noise vectors to synthetic data records ($G_{\theta}$)
    SyntheticDataset: NotationSymbol  # set of synthetic records a trained generator produces from sampled latent vectors ($\mathcal{S}_{\text{synthetic}}$)
    SystemCallFilterTime: NotationSymbol  # time one system call spends being validated or emulated in software ($T_{\text{filter}}$) [s]
    SystemCallOverheadTime: NotationSymbol  # time a sandbox envelope adds to one mediated system call, its system-call tax ($T_{\text{syscall}}$) [s]
    SystemFailureRate: NotationSymbol  # rate at which failures interrupt the whole job or system, the sum of its components' failure rates ($\lambda_{\text{system}}$) [failures/hour]
    SystemHazardRate: NotationSymbol  # rate of hazardous events of class h for the whole system, after the shield ($\lambda_{\text{system}, h}$) [1/h]
    SystemInstructionTokenCount: NotationSymbol  # number of tokens one call devotes to static system instructions, one term of the context budget partition ($S_{\text{system}}$)
    SystemMatrixPowerIndex: NotationSymbol  # exponent of a power of the system matrix, the number of propagation steps a controllability column block represents ($k$)
    SystemMemoryBandwidth: NotationSymbol  # sustained bandwidth of host system memory ($\text{BW}_{\text{system}}$) [bytes/s]
    SystemMtbf: NotationSymbol  # mean time between failures that interrupt the whole job or system ($\text{MTBF}_{\text{system}}$) [hours]
    SystemReliability: NotationSymbol  # probability that the whole system runs without failure up to time t ($R_{\text{system}}$) [dimensionless]
    TableRowCount: NotationSymbol  # number of rows in a database table ($N$)
    TailByteAllocation: NotationSymbol  # bytes of the observation budget reserved for the end of a tool's output, kept in a ring buffer that overwrites its oldest bytes ($M_{\text{tail}}$) [bytes]
    TailMomentOrder: NotationSymbol  # order of a raw moment of a random quantity, the power to which it is raised before taking the expectation ($k$) [dimensionless]
    TailPerformanceMetric: NotationSymbol  # continuous performance metric of a trial whose upper tail extreme value theory models, such as braking-distance overshoot ($X$) [varies]
    TailRequestLatency: NotationSymbol  # 99th-percentile time a request spends in a serving system ($T_{\text{lat},\text{p99}}$) [s]
    TailRequestSeenBatchSize: NotationSymbol  # 99th percentile of the batch size a request lands in ($B_{\text{req,p99}}$)
    TailRiskDistributionFunction: NotationSymbol  # cumulative distribution function of the monitored quantity whose tail is modeled ($F$)
    TailRiskObservation: NotationSymbol  # monitored physical quantity whose rare large values a tail model describes, such as a contact load or a control-loop latency, treated as a random variable ($X$) [varies]
    TailRiskObservationLevel: NotationSymbol  # level of the monitored quantity at which its distribution or density is evaluated, or one observed value of it ($x$) [varies]
    TailShapeParameter: NotationSymbol  # shape parameter of an extreme value tail model, positive for a power-law tail, zero for an exponential tail and negative for a bounded tail ($\xi$) [dimensionless]
    TailStepIoTime: NotationSymbol  # 99th-percentile time the input pipeline takes to fetch the data for one training step from storage ($T_{\text{IO,p99}}$) [s]
    TailSupportUpperEndpoint: NotationSymbol  # right endpoint of the support of the monitored quantity's distribution, possibly infinite ($x_F$) [varies]
    TakeoverInstant: NotationSymbol  # time at which a human or the safety enforcer takes control from the learned policy, latched at a permission tick ($t_{\text{takeover}}$) [s]
    TakeoverResponseTime: NotationSymbol  # time a human operator needs to take over control after a handover request ($T_{\Sigma, 1}$) [s]
    TakeoverStateVisitationDistribution: NotationSymbol  # distribution of states visited while an expert recovers the machine after a takeover, conditioned on the prior policy having failed ($p^{\text{takeover}}$)
    TakeoverTrackingError: NotationSymbol  # tracking error vector of the plant against the operator's command after a takeover, whose norm the stabilization test bounds ($\mathbf{e}_{\text{track}}$)
    TangentDirection: NotationSymbol  # direction in state space tested for membership in a contingent cone ($\mathbf{v}$)
    TangentSeedVector: NotationSymbol  # tangent vector seeded on a function's inputs for one forward-mode pass ($\dot{\mathbf{x}}$)
    TangentialContactShear: NotationSymbol  # friction component of a contact force, tangential to the contact plane ($\mathbf{f}_{k, \parallel}$) [N]
    TangentialSpeed: NotationSymbol  # speed at which a feature at a given range sweeps across a rotating sensor's view, angular rate times range ($v_{\text{tangential}}$) [m/s]
    TapeIntermediateResult: NotationSymbol  # intermediate result of the autograd tape example, twice its input ($y_{\text{tape}}$) [dimensionless]
    TargetAcceptanceDifference: NotationSymbol  # difference in task acceptance probability between two system versions that a release comparison is planned to detect ($\Delta$) [dimensionless]
    TargetActionReturnEstimate: NotationSymbol  # target estimate of the action value, held fixed during one value-fitting update to form the Bellman target ($\hat{Q}$) [dimensionless]
    TargetActorParameters: NotationSymbol  # parameters of the target actor network, a slowly Polyak-averaged copy of the actor parameters ($\theta'$)
    TargetBatchSize: NotationSymbol  # batch size a traffic-adaptive batcher sizes its window to fill at the current arrival rate ($B_{\text{target}}$)
    TargetCompletionQuantile: NotationSymbol  # required probability that a request completes within its latency deadline, the completion quantile a service level states ($q$) [dimensionless]
    TargetComputeRate: NotationSymbol  # operation rate a workload is required to sustain, from which the memory bandwidth it needs follows ($R_{\text{ops}}$) [FLOP/s]
    TargetComputeTimeFraction: NotationSymbol  # target fraction of wall-clock time an accelerator spends computing rather than waiting for input, used to size input delivery ($\eta_{\text{target}}$) [dimensionless]
    TargetCriticParameters: NotationSymbol  # parameters of the target critic network, a slowly Polyak-averaged copy of the critic parameters ($\phi'$)
    TargetCriticReturnEstimate: NotationSymbol  # action-value estimate of the target critic network, the slowly Polyak-averaged copy of the critic that evaluates the Bellman target ($Q_{\phi'}$) [dimensionless]
    TargetDeterministicPolicy: NotationSymbol  # slowly updated copy of the deterministic actor, used to choose the next action in the critic's Bellman target ($\mu_{\theta'}$)
    TargetError: NotationSymbol  # test error a model is required to reach ($\epsilon$) [dimensionless]
    TargetExceedanceProbability: NotationSymbol  # per-sample probability of a catastrophic exceedance that a design margin is sized against ($\epsilon$) [dimensionless]
    TargetFairShare: NotationSymbol  # share of cluster capacity a user or account is entitled to under fair-share scheduling ($F_{\text{target}}$) [dimensionless]
    TargetFramePoint: NotationSymbol  # coordinates of a point expressed in the destination frame of a transform ($\mathbf{p}_A$) [m]
    TargetHazardRate: NotationSymbol  # tolerable dangerous failure rate assigned to a safety function by risk assessment ($\lambda_{\text{target}}$) [1/h]
    TargetImpedanceDampingMatrix: NotationSymbol  # damping an impedance or admittance controller makes the end effector present, mapping velocity error to force ($\mathbf{D}_d$) [varies]
    TargetImpedanceInertiaMatrix: NotationSymbol  # inertia an impedance or admittance controller makes the end effector present to external force ($\mathbf{M}_d$) [varies]
    TargetImpedanceStiffnessMatrix: NotationSymbol  # stiffness an impedance or admittance controller makes the end effector present, mapping position error to restoring force ($\mathbf{K}_d$) [varies]
    TargetInferenceLatency: NotationSymbol  # inference latency a searched architecture is required to meet ($L_{\text{lat,target}}$) [s]
    TargetKeyAttentionWeight: NotationSymbol  # softmax attention weight one query assigns to the target key, its share of attention ($\alpha_{\text{target}}$) [dimensionless]
    TargetKeyLogit: NotationSymbol  # scaled query-key logit of the target key in one attention head, the key whose share of attention is tracked ($z_{\text{target}}$) [dimensionless]
    TargetModelDistribution: NotationSymbol  # next-token probability distribution of the target model in speculative decoding ($p_{\theta}$)
    TargetOrientation: NotationSymbol  # rotation matrix of the target orientation a goal set is centered on ($\mathbf{R}_{\text{target}}$) [dimensionless]
    TargetPlantState: NotationSymbol  # target plant state an input must reach in finite time in the definition of controllability ($\mathbf{x}_{f}$)
    TargetPolicyEntropy: NotationSymbol  # minimum average policy entropy that automatic temperature tuning holds the policy to ($\overline{\mathcal{H}}$) [nats]
    TargetPose: NotationSymbol  # pose an intent proposal asks the end effector to reach, in the robot base frame ($\mathbf{T}_{\text{target}}$)
    TargetPosition: NotationSymbol  # position of the target an end effector is commanded toward, in the robot base frame ($\mathbf{p}_{\text{target}}$) [m]
    TargetPositionComponent: NotationSymbol  # component of the target position along one coordinate axis ($p_{\text{target},i}$) [m]
    TargetRange: NotationSymbol  # distance from a camera to the target it observes, measured along the viewing direction ($z$) [m]
    TargetRepeatedRunConsistency: NotationSymbol  # repeated-run consistency, the probability that all k recurrences of a task pass, that a recurring workload is required to hold ($p_{\text{target}}$) [dimensionless]
    TargetSamplingDistribution: NotationSymbol  # normalized next-token distribution of the target model at a prefix in speculative sampling, after temperature, masks and other sampling transformations, the distribution exact speculative sampling preserves ($p$)
    TargetServingUtilization: NotationSymbol  # serving utilization a system is sized not to exceed, chosen below one to keep latency headroom ($\rho_{\text{target}}$) [dimensionless]
    TargetSmoothingNoise: NotationSymbol  # zero-mean Gaussian noise vector added to the target actor's action before clipping, in target policy smoothing ($\boldsymbol{\epsilon}$)
    TargetSmoothingNoiseStandardDeviation: NotationSymbol  # standard deviation of each component of the Gaussian noise added in target policy smoothing ($\sigma$) [varies]
    TargetTokenVolume: NotationSymbol  # total loss-selected target tokens of one trajectory category, its record count times the mean target tokens per record ($W$) [tokens]
    TargetTravelDistance: NotationSymbol  # straight-line distance from the measured end-effector position to the admitted target ($d$) [m]
    TargetTrialFailureProbability: NotationSymbol  # per-trial failure probability that a zero-failure campaign is sized to bound ($p_{*}$) [dimensionless]
    TargetTrialSuccessProbability: NotationSymbol  # per-trial success probability a test campaign must demonstrate ($p_{\text{target}}$) [dimensionless]
    TargetTwinCriticReturnEstimate: NotationSymbol  # action-value estimate of the target copy of the i-th twin critic, minimized over both critics in the clipped Bellman target ($Q_{\phi'_i}$) [dimensionless]
    TargetUnitQuaternion: NotationSymbol  # unit quaternion of the target orientation a goal set is centered on ($\mathbf{q}_{\text{quat,target}}$) [dimensionless]
    TargetUpdateCount: NotationSymbol  # number of client updates a federated round waits for and aggregates ($K_{\text{target}}$)
    TargetUtilizationBatchSize: NotationSymbol  # smallest batch size whose serving utilization stays at or below the target utilization, under the linear batch service model ($B_{\text{util}}$)
    TargetXPosition: NotationSymbol  # x coordinate of a proposed or admitted target position in the robot base frame ($x$) [m]
    TargetYPosition: NotationSymbol  # y coordinate of a proposed or admitted target position in the robot base frame ($p_y$) [m]
    TargetZPosition: NotationSymbol  # z coordinate of a proposed or admitted target position in the robot base frame ($z$) [m]
    TaskAcceptanceLabel: NotationSymbol  # binary label recording whether one monitored task was accepted, one for acceptance ($x$) [dimensionless]
    TaskBlockingTime: NotationSymbol  # longest time a real-time task can wait for a lower-priority task to release a shared resource ($B$) [s]
    TaskContract: NotationSymbol  # five-part specification of a delegated task: goal, environment, permitted actions, available observations, and completion criteria ($\mathcal{C}$)
    TaskContractGoal: NotationSymbol  # desired end state of a task, with explicit negative scope, as the first element of the task contract ($G$)
    TaskCost: NotationSymbol  # total monetary cost of all attempts and branches for one initiated task ($C_{\text{task}}$) [USD]
    TaskCount: NotationSymbol  # number of tasks in a whole-body control hierarchy or a multi-task learning problem ($K$)
    TaskCoverageProbability: NotationSymbol  # probability that at least one attempt in a retry budget is accepted on a task, its coverage ($Q$) [dimensionless]
    TaskCriticalSections: NotationSymbol  # function giving the set of critical sections a task executes whose priority ceiling is at or above the priority of the task under analysis ($cs$)
    TaskDerivativeGain: NotationSymbol  # derivative feedback gain matrix on the task-space velocity error of task i ($\mathbf{K}_{d,i}$)
    TaskDistance: NotationSymbol  # task-specific distance between a model's prediction and the correct label, the error measure a verification guarantee bounds ($d_{\text{task}}$)
    TaskEnvironment: NotationSymbol  # specification of the reproducible place the work happens, such as base commit, image, configuration, and network rules ($\mathcal{E}_{\text{env}}$)
    TaskErrorTolerance: NotationSymbol  # largest task distance between a prediction and its label that still counts as an acceptable output ($\tau_{\text{tol}}$) [dimensionless]
    TaskGroupMap: NotationSymbol  # function that sends an environment fixture to the task group that holds all its attempts, paraphrases and solution-equivalent variants ($g$)
    TaskIndex: NotationSymbol  # position of one task among the tasks an evaluation samples, the index of its per-task acceptance probability ($i$)
    TaskInterferenceTime: NotationSymbol  # processor time that higher-priority tasks and interrupt handlers take from a real-time task within its response window ($I$) [s]
    TaskJacobian: NotationSymbol  # Jacobian of a whole-body control task's coordinates with respect to the generalized coordinates ($\mathbf{J}$)
    TaskJacobianDerivative: NotationSymbol  # time derivative of a whole-body control task's Jacobian ($\dot{\mathbf{J}}$)
    TaskJacobianPseudoinverse: NotationSymbol  # Moore-Penrose pseudoinverse of a whole-body control task's Jacobian, mapping a task acceleration to the minimum-norm generalized acceleration that produces it ($\mathbf{J}^\dagger$)
    TaskKinematicMap: NotationSymbol  # kinematic map of one whole-body control task, from generalized coordinates to that task's operational-space coordinates ($f$)
    TaskLoss: NotationSymbol  # loss that scores the primary task prediction against the label in a multi-objective training setup ($\mathcal{L}_{\text{task}}$)
    TaskLossGradient: NotationSymbol  # gradient of one task's loss with respect to the shared parameters in multi-task learning, indexed by task ($\nabla \mathcal{L}$)
    TaskPeriod: NotationSymbol  # interval between successive releases of a periodic real-time task, indexed by task ($T$) [s]
    TaskProportionalGain: NotationSymbol  # proportional feedback gain matrix on the task-space position error of task i ($\mathbf{K}_{p,i}$)
    TaskReadCount: NotationSymbol  # number of untrusted documents or items one task reads ($n_{\text{read}}$)
    TaskRelativeDeadline: NotationSymbol  # longest time after its release by which a job of a real-time task must complete ($D$) [s]
    TaskRelevantOutcome: NotationSymbol  # task-relevant random variable, such as a latch state or material compliance, whose information a representation may retain or lose ($\mathbf{Y}_{\text{task}}$)
    TaskReward: NotationSymbol  # task-progress term of a shaped reward, before effort and hardware penalties are subtracted ($r_{\text{task}}$) [dimensionless]
    TaskSetUtilization: NotationSymbol  # share of processor time a set of periodic real-time tasks demands, the sum of each task's execution time over its period ($U$) [dimensionless]
    TaskSpaceAcceleration: NotationSymbol  # acceleration of the operational-space coordinates of a whole-body control task ($\ddot{\mathbf{x}}$)
    TaskSpaceCoordinates: NotationSymbol  # operational-space coordinates of a whole-body control task, such as an end-effector pose, the center-of-mass position or the torso orientation ($\mathbf{x}$)
    TaskSpaceDimension: NotationSymbol  # number of operational-space coordinates of one whole-body control task ($m$)
    TaskSpaceReference: NotationSymbol  # reference operational-space coordinates a high-level policy proposes for whole-body control task i ($\mathbf{x}_i^{\text{ref}}$)
    TaskSpaceReferenceAcceleration: NotationSymbol  # feedforward reference acceleration of a whole-body control task's operational-space coordinates, the second derivative of the proposed task reference ($\ddot{\mathbf{x}}^{\text{ref}}$)
    TaskSpaceReferenceVelocity: NotationSymbol  # reference velocity of a whole-body control task's operational-space coordinates, the first derivative of the proposed task reference ($\dot{\mathbf{x}}^{\text{ref}}$)
    TaskSpaceVelocity: NotationSymbol  # velocity of the operational-space coordinates of a whole-body control task ($\dot{\mathbf{x}}$)
    TaskSpecificLoss: NotationSymbol  # loss function of one task in multi-task learning, indexed by task ($\mathcal{L}$)
    TaskStateTokenCount: NotationSymbol  # number of tokens one call devotes to task state, such as constraints, progress records and source or test excerpts ($S_{\text{state}}$)
    TaskToleranceSemiAxes: NotationSymbol  # principal semi-axes of the ellipsoid within which terminal position error must lie ($\mathbf{r}$) [m]
    TaskToleranceSemiAxis: NotationSymbol  # semi-axis of the task-tolerance ellipsoid along one coordinate axis ($r$) [m]
    TaskTurnCost: NotationSymbol  # total monetary cost of one task turn's model call, its cached and uncached input tokens and its output tokens ($C_{\text{call}}$) [USD]
    TaskWeight: NotationSymbol  # finite weight on one task's error in a weighted whole-body control objective ($w$) [dimensionless]
    TasksPerComparisonArm: NotationSymbol  # number of tasks per arm needed for a two-proportion release comparison ($N_{\text{arm}}$)
    TeamQuotaCap: NotationSymbol  # maximum allocation a team may hold under hierarchical quotas, as opposed to its guaranteed share ($Q_{\text{team}}$)
    TelemetryFlushTime: NotationSymbol  # time to flush onboard telemetry buffers between trials, part of the reset latency ($t_{\text{telemetry\_flush}}$) [s]
    TelemetryFreshnessBound: NotationSymbol  # largest age a measurement may reach before acting on it introduces intolerable error ($\Delta t_{\text{fresh}}$) [s]
    TelemetryRetentionDuration: NotationSymbol  # length of time stored telemetry samples are retained ($T_{\text{retention}}$) [s]
    TelemetrySamplingFrequency: NotationSymbol  # number of samples recorded per second for one telemetry series ($f_{\text{sample}}$) [Hz]
    TelemetrySeriesCount: NotationSymbol  # number of distinct telemetry time series after label expansion ($N_{\text{series}}$)
    TelemetryTickInterval: NotationSymbol  # time between two consecutive joint-telemetry ticks of the safety microcontroller ($\Delta t_s$) [s]
    TeleoperationCommandLatency: NotationSymbol  # time for a teleoperation command to cross from the leader to the follower controller ($t_{\text{cmd}}$) [s]
    TeleoperationDownlinkTime: NotationSymbol  # network transport time of a teleoperation link from the robot to the operator station ($T_{\text{downlink}}$) [s]
    TeleoperationRoundTripLatency: NotationSymbol  # time from a physical contact disturbance to the follower's corrective actuator response in a teleoperation loop ($t_{\text{loop}}$) [s]
    TeleoperationRoundTripTime: NotationSymbol  # network round-trip transport time of a teleoperation link, the downlink transport time plus the uplink transport time ($T_{\text{RTT}}$) [s]
    TeleoperationStateVisitationDistribution: NotationSymbol  # distribution of states visited during cautious human teleoperation, the occupancy a conservative demonstration archive covers ($p^{\pi_{\text{teleop}}}$)
    TeleoperationUplinkTime: NotationSymbol  # network transport time of a teleoperation link from the operator station to the robot ($T_{\text{uplink}}$) [s]
    TeleoperationVideoLatency: NotationSymbol  # time a teleoperation video path spends on camera exposure, encoding and operator display processing, excluding network transport ($T_{\text{video}}$) [s]
    TemperatureDualObjective: NotationSymbol  # dual objective of automatic entropy tuning, whose gradient with respect to the temperature drives policy entropy toward its target ($\mathcal{L}$)
    TemperatureStepSize: NotationSymbol  # step size of the dual gradient descent update of the entropy temperature ($\lambda_{\alpha}$) [1/nats^2]
    TemplateSharedMemory: NotationSymbol  # memory of the golden template image that all pool instances share copy-on-write ($M_{\text{template}}$) [bytes]
    TemporalDifferenceResidual: NotationSymbol  # sampled one-step temporal-difference residual: the turn reward plus the next history's value minus the current history's value ($\delta$) [dimensionless]
    TemporalEnsemblingWeight: NotationSymbol  # weight given to the prediction of a past chunk at offset i when blending overlapping chunks ($w_i$) [dimensionless]
    TenantIndex: NotationSymbol  # index of one tenant sharing a fleet ($i$)
    TenantResourceAllocation: NotationSymbol  # amount of one resource type allocated to one tenant ($a_{i,r}$) [varies]
    TensorAxisCount: NotationSymbol  # number of axes, or dimensions, of an array or tensor ($n$)
    TensorColumnModeSize: NotationSymbol  # size of the column mode of a decomposed three-mode tensor ($J$)
    TensorCoreAccumulatorMatrix: NotationSymbol  # accumulator matrix added to the product in one Tensor Core matrix multiply-accumulate ($\mathbf{C}_{\text{acc}}$)
    TensorCoreLeftInput: NotationSymbol  # left input matrix of one Tensor Core matrix multiply-accumulate ($\mathbf{A}_{\text{tc}}$)
    TensorCoreOutputMatrix: NotationSymbol  # result matrix of one Tensor Core matrix multiply-accumulate ($\mathbf{D}_{\text{tc}}$)
    TensorCoreRightInput: NotationSymbol  # right input matrix of one Tensor Core matrix multiply-accumulate ($\mathbf{B}_{\text{tc}}$)
    TensorDepthModeSize: NotationSymbol  # size of the depth mode of a decomposed three-mode tensor ($T$)
    TensorElementCount: NotationSymbol  # number of elements in one tensor ($N$)
    TensorModeIndex: NotationSymbol  # position of one axis (mode) of a tensor, the axis a mode-n product multiplies along ($n_{\text{mode}}$)
    TensorParallelDegree: NotationSymbol  # number of devices that shard each layer's tensors in tensor parallelism ($N_{\text{TP}}$)
    TensorParallelLayerTime: NotationSymbol  # time of one transformer layer under tensor parallelism, its compute divided across the tensor-parallel devices plus its two AllReduces ($T_{\text{inference}}$) [s]
    TensorRowModeSize: NotationSymbol  # size of the row mode of a decomposed three-mode tensor ($I$)
    TerminalFeedbackGain: NotationSymbol  # gain matrix of the local linear feedback law that keeps the terminal set invariant in model predictive control ($\mathbf{K}_f$)
    TerminalInvariantSet: NotationSymbol  # set of plant states from which an invariant fallback control law keeps the system safe indefinitely, the required endpoint of a receding horizon ($\mathcal{X}_{f}$)
    TerminalPlantState: NotationSymbol  # predicted plant state at the final step of a model-predictive-control horizon ($\mathbf{x}_N$)
    TerminalTransitTime: NotationSymbol  # time a terminal move needs under the actuator velocity and acceleration limits ($t_{\text{transit}}$) [s]
    TerminalVoltageLimit: NotationSymbol  # largest magnitude of winding terminal voltage the inverter can apply ($V_{\max}$) [V]
    TerminationProtocol: NotationSymbol  # termination protocol of a trial: its execution deadline, goal region and outcome predicate ($\kappa$)
    TestExample: NotationSymbol  # one test example, an input with its label, at which an influence function measures the change in loss ($z_{\text{test}}$)
    TestFamilySize: NotationSymbol  # number of hypothesis tests in a family whose error rate a multiple-testing correction controls ($k$)
    TestPhaseCapabilityGrant: NotationSymbol  # capability grant proposed for the test phase of a repair, read and execute over the repository ($\Gamma_{\text{test}}$)
    TestsDirectoryShardGrant: NotationSymbol  # child capability grant proposed for a test shard over the tests directory ($\Gamma_{a}$)
    TextTokenCount: NotationSymbol  # number of language tokens a natural-language instruction contributes to an embodied transformer's input ($S_{\text{text}}$)
    TextTokenEmbeddings: NotationSymbol  # token embeddings of the language instruction fed to an embodied transformer ($\mathbf{z}_{\text{text}}$)
    ThermalExpansionCoefficient: NotationSymbol  # fractional change in a material's length per kelvin of temperature change ($\alpha$) [1/K]
    ThermalPenaltyWeight: NotationSymbol  # weight of the winding-temperature excess over its limit in a shaped reward, the thermal penalty ($\lambda_T$) [1/degC]
    ThermalPowerLimit: NotationSymbol  # largest power a device can sustain without exceeding its thermal limits ($P_{\text{thermal}}$) [W]
    ThermalResistance: NotationSymbol  # lumped thermal resistance from the stator windings to ambient, temperature rise per watt ($R_{\text{th}}$) [K/W]
    ThermalTimeConstant: NotationSymbol  # characteristic time for a motor winding to heat or cool, thermal resistance times thermal capacitance ($\tau_{\text{th}}$) [s]
    ThirdFloatingPointAddend: NotationSymbol  # third of three floating-point numbers added in either grouping to show that floating-point addition is not associative ($c$) [dimensionless]
    ThirdRingTraceElement: NotationSymbol  # third of the four elements of a rank's gradient vector in the four-GPU ring AllReduce trace, each element one chunk ($c$) [varies]
    ThresholdExceedanceCount: NotationSymbol  # number of trials whose performance metric exceeds the threshold ($N_u$)
    ThresholdExceedanceRate: NotationSymbol  # probability per trial that a performance metric exceeds the threshold, estimated as exceedances over trials ($\zeta_u$) [dimensionless]
    ThresholdExcess: NotationSymbol  # amount by which the monitored quantity exceeds the threshold, the argument of the excess distribution or one recorded exceedance ($y$) [varies]
    ThresholdExcessVariable: NotationSymbol  # excess of the monitored quantity over the threshold, conditional on exceeding it, treated as a random variable ($Y$) [varies]
    ThresholdedSparsityRatio: NotationSymbol  # fraction of a weight matrix's entries whose magnitude is at or below the pruning threshold ($\rho_{\text{sparse},\delta}$) [dimensionless]
    ThrottleClearTemperature: NotationSymbol  # junction temperature below which a thermal governor restores the nominal clock after throttling ($T_{\text{clear}}$) [degC]
    ThrottleOnsetJunctionTemperature: NotationSymbol  # junction temperature at the moment throttling begins, the start of the cooling curve ($T_{\text{j},\text{start}}$) [degC]
    ThrottleRecoveryTime: NotationSymbol  # time a throttled die takes to cool from the trip temperature to the clear temperature, during which the clock stays reduced ($t_{\text{recov}}$) [s]
    ThrottleTripTemperature: NotationSymbol  # junction temperature at or above which a hardware thermal governor reduces the processor clock ($T_{\text{trip}}$) [degC]
    ThrottledClockFrequency: NotationSymbol  # reduced clock frequency a thermal governor imposes while the junction is above its trip temperature ($f_{\text{throt}}$) [Hz]
    TickCycleCount: NotationSymbol  # number of clock cycles one permission tick needs, its execution time times the nominal clock frequency ($N$)
    TickIndex: NotationSymbol  # position of one tick in the sequence of consecutive permission ticks ($n$)
    TickPeriodJitterBound: NotationSymbol  # bound on how far the interval between consecutive tick starts deviates from the nominal tick period ($J_{\text{phase}}$) [s]
    TieredRequestServiceTime: NotationSymbol  # time to serve one request through a tiered storage hierarchy, a random variable that equals the hot-tier time on a hit and the miss-path time on a miss ($T$) [s]
    TightenedDiscreteSafeSet: NotationSymbol  # safe set shrunk by the worst-case inter-sample drift, so that checking the barrier at sample instants keeps the state safe between them ($\mathcal{C}_{\text{discrete}}$)
    TileRowIndex: NotationSymbol  # index of the tile row that an output block occupies in a tiled matrix multiply ($m$)
    TiltAngle: NotationSymbol  # angular deflection of a body from vertical ($\theta_{\text{tilt}}$) [rad]
    Time: NotationSymbol  # total wall-clock time to complete a unit of work, from start to finish ($T$) [s]
    TimeBudgetUsedFraction: NotationSymbol  # fraction of a trajectory's wall-clock allowance already elapsed ($\rho_T$) [dimensionless]
    TimeInstant: NotationSymbol  # a point in continuous wall-clock time, the variable over which time-varying quantities are integrated ($t$) [s]
    TimeScaleSeparationRatio: NotationSymbol  # small ratio of the fast elastic time scale to the slow rigid-body time scale in a singular perturbation model ($\epsilon$) [dimensionless]
    TimeStepIndex: NotationSymbol  # position of one step in a sequence that a recurrent network or decoder processes one step at a time ($t$)
    TimeToResult: NotationSymbol  # wall-clock time from a job's submission to its result, its queue wait plus its run time ($T_{\text{result}}$) [s]
    TimeToTarget: NotationSymbol  # wall-clock time a federated run takes to reach a held-out quality target, the sum of its completed round durations ($T_{\text{target}}$) [s]
    TimeoutSafetyMultiplier: NotationSymbol  # number of delay standard deviations a failure-detection timeout allows beyond the heartbeat interval ($k$) [dimensionless]
    TimeoutTrialCount: NotationSymbol  # number of trials in a campaign that end by reaching their execution deadline ($n_{\text{timeout}}$)
    TimerClockPeriod: NotationSymbol  # period of the clock that increments a microcontroller timer's counter ($T_{\text{timer\_clk}}$) [s]
    TimestampJitterDelay: NotationSymbol  # random variation in the delay between a physical event and its timestamp, from scheduling or transport ($\delta_{\text{jitter}}$) [s]
    TimestampQuantizationBound: NotationSymbol  # bound on the quantization error of one hardware timestamp ($T_{\text{quant}}$) [s]
    TimestampSkew: NotationSymbol  # skew between the time stamps of two records or clocks ($t_{\text{skew}}$) [s]
    TimingJitter: NotationSymbol  # standard deviation of a nominally periodic event's timing about its schedule ($\sigma_{\text{jitter}}$) [s]
    TimingRealityGap: NotationSymbol  # timing component of the reality gap: discrepancy between the simulated and physical observation-to-actuation latency and its jitter ($\Delta_{\text{lat}}$) [s]
    TokenAuthenticationCode: NotationSymbol  # message authentication code carried by a chained-caveat token, recomputed as each caveat is appended ($\sigma$)
    TokenAveragedBatchLoss: NotationSymbol  # batch loss that averages over all selected target tokens of a batch, so each example counts in proportion to its selected-token count ($\mathcal{L}_{\text{token}}$) [nats]
    TokenBucketDepth: NotationSymbol  # largest number of admission tokens a token-bucket limiter holds, the burst it lets one source send ($b$)
    TokenBucketRefillRate: NotationSymbol  # rate at which a token-bucket limiter adds admission tokens, the largest sustained packet rate it admits ($r$) [1/s]
    TokenBudgetUsedFraction: NotationSymbol  # fraction of a trajectory's token allowance already used or reserved ($\rho_N$) [dimensionless]
    TokenByteDecoding: NotationSymbol  # function mapping an ordinary vocabulary token to the fixed nonempty byte string it decodes to ($s$)
    TokenCaveat: NotationSymbol  # restriction a holder appends to a chained-caveat token, such as a path prefix or an earlier expiry ($c$)
    TokenDispatchTime: NotationSymbol  # time for the AllToAll that sends tokens from their source ranks to the ranks hosting their selected experts ($T_{\text{dispatch}}$) [s]
    TokenIdentifierSize: NotationSymbol  # bytes one sampled token identifier occupies when sampling runs on the accelerator and only the identifier is transferred ($s_{\text{id}}$) [bytes/token]
    TokenIndex: NotationSymbol  # position of a token within a rollout or a serialized training example ($t$)
    TokenPrefix: NotationSymbol  # tokens of a serialized sequence that precede the current position, the context an autoregressive model conditions on ($x_{<t}$)
    TokenVocabulary: NotationSymbol  # set of tokens a model call samples from ($\mathcal{V}$)
    TokenizationTime: NotationSymbol  # time one CPU core takes to tokenize one batch ($T_{\text{tokenize}}$) [s]
    ToleranceEllipsoidMatrix: NotationSymbol  # positive-definite weighting matrix of the ellipsoidal norm that bounds terminal position error, diagonal in the inverse squared tolerance semi-axes ($\mathbf{E}$)
    ToleratedFailureProbability: NotationSymbol  # largest probability with which a deployed model may miss its error tolerance, so a guarantee holds with confidence one minus it ($\epsilon$) [dimensionless]
    ToleratedPositionError: NotationSymbol  # largest position error a fusion pipeline tolerates, which with the speed bound sets the allowable clock skew ($\epsilon_{\text{pos}}$) [m]
    TombstonedNodeSet: NotationSymbol  # set of nodes in a vector index graph whose records are tombstoned, excluded from results but still linked in the graph ($V_{\text{tomb}}$)
    TombstonedRecordFraction: NotationSymbol  # fraction of the records in a vector index graph that are tombstoned but not yet compacted away ($\rho_{\text{dead}}$) [dimensionless]
    ToolArgumentKey: NotationSymbol  # one parameter name that a proposed tool call emits ($p$)
    ToolCallCategory: NotationSymbol  # one kind of tool call, such as a test run or a lint call, over which a configuration's tool-call distribution is defined ($y$)
    ToolCallCost: NotationSymbol  # monetary charge for executing one tool call, such as a sandbox run or a paid third-party request ($c_{\text{tool}}$) [USD]
    ToolCallIndex: NotationSymbol  # index of one tool call within a batch the runtime may dispatch concurrently ($i$)
    ToolCallReadSet: NotationSymbol  # set of resources one tool call declares it reads, the read half of its resource footprint ($R$)
    ToolCallWriteSet: NotationSymbol  # set of resources one tool call declares it writes, the write half of its resource footprint ($W$)
    ToolCartesianPose: NotationSymbol  # Cartesian pose of the tool or end effector written as a task-space vector of position and orientation ($\mathbf{x}$)
    ToolCartesianVelocity: NotationSymbol  # spatial velocity of the tool or end effector in Cartesian task space ($\dot{\mathbf{x}}$)
    ToolCenterPointAxis: NotationSymbol  # unit vector along the z axis of the tool center point frame, the direction the gripper points ($\hat{\mathbf{z}}_{\text{TCP}}$) [dimensionless]
    ToolCenterPointFrame: NotationSymbol  # coordinate frame of the tool center point, its origin between the gripper fingertips ($\mathcal{F}_{\text{TCP}}$)
    ToolCenterPointPose: NotationSymbol  # pose of the tool center point frame, its rotation and position in the robot base frame ($\mathbf{T}_{\text{TCP}}$)
    ToolCenterPointPosition: NotationSymbol  # position of the tool center point, midway between the gripper fingertips ($\mathbf{p}_{\text{TCP}}$) [m]
    ToolClearanceDistance: NotationSymbol  # distance from the tool point to the obstacle surface it approaches ($d_{\text{tool}}$) [m]
    ToolDeclarationTokenCount: NotationSymbol  # number of tokens one call devotes to API schemas and tool declarations, one term of the context budget partition ($S_{\text{tools}}$)
    ToolDescriptorSize: NotationSymbol  # number of tokens one tool descriptor occupies in a call's context ($S_{\text{desc}}$)
    ToolPlanarAbscissa: NotationSymbol  # first coordinate of the tool position in the plane of a planar arm, along the axis from which the shoulder angle is measured ($x$) [m]
    ToolPlanarOrdinate: NotationSymbol  # second coordinate of the tool position in the plane of a planar arm ($y$) [m]
    ToolPositionError: NotationSymbol  # error in the computed tool or fingertip position ($e_{\text{tool}}$) [m]
    ToolServiceCount: NotationSymbol  # number of independently maintained tool services that host runtimes integrate with ($N_{\text{tool}}$)
    ToolSpeedLimit: NotationSymbol  # largest speed the tool center point of an arm may move at, its TCP speed limit ($v_{\text{TCP}}$) [m/s]
    ToolTargetState: NotationSymbol  # state of the external system or resource a tool operation reads or mutates, such as a remote file or service record ($s$)
    ToolTime: NotationSymbol  # time a tool takes to execute, such as a build or a test run ($T_{\text{tool}}$) [s]
    ToolTipPose: NotationSymbol  # pose of the tool tip, the flange pose composed with the tool offset ($\mathbf{T}_{\text{tool}}$)
    TopClassCount: NotationSymbol  # number of highest-scoring classes a top-k prediction returns or checks for the correct label ($k$)
    TopMagnitudeIndicator: NotationSymbol  # indicator vector that is one at each gradient element whose magnitude is at least the largest-k threshold and zero elsewhere ($\mathbf{1}_{|\mathbf{g}| \geq |\mathbf{g}|_{(K_{\text{top}})}}$)
    TopTokenSampleCount: NotationSymbol  # number of highest-probability tokens top-k sampling keeps as candidates at each decoding step ($k$)
    TopologicalPlaceNode: NotationSymbol  # one topological place node of a 3D scene graph, a free-space clearance sphere linked to its neighbors ($p^{\text{place}}$)
    TopologyEndpointCount: NotationSymbol  # number of endpoints, accelerators or hosts, that one network topology connects, the size variable of its scaling laws ($N$)
    TorqueCutoffCommand: NotationSymbol  # fallback command that removes actuator torque and drops the spring brakes ($\mathbf{u}_{\text{cutoff}}$)
    TorqueEffortPenaltyWeight: NotationSymbol  # weight of the squared joint-torque norm in a shaped reward, the effort penalty ($\lambda_{\tau}$) [1/(N*m)^2]
    TorqueProducingCurrent: NotationSymbol  # quadrature-axis current of a field-oriented drive, the current component that produces torque ($i_q$) [A]
    TorqueRateLimit: NotationSymbol  # largest rate of change of joint output torque a drive accepts ($\dot{\tau}_{\max}$) [N*m/s]
    TorqueRegularizationWeight: NotationSymbol  # weight on the squared joint-torque regularization term of a whole-body control objective ($w_\tau$) [varies]
    TorqueSlewPenaltyWeight: NotationSymbol  # weight of the squared rate of change of joint torque in a shaped reward, the slew penalty ($\lambda_{\dot{\tau}}$) [s^2/(N*m)^2]
    TorusFirstAxisChipCount: NotationSymbol  # number of chips along the first axis of a torus pod ($X$)
    TorusSecondAxisChipCount: NotationSymbol  # number of chips along the second axis of a torus pod ($Y$)
    TorusThirdAxisChipCount: NotationSymbol  # number of chips along the third axis of a torus pod ($Z$)
    TotalChipPower: NotationSymbol  # total power a chip dissipates, the sum of dynamic switching power and static leakage power ($P_{\text{total}}$) [W]
    TotalCollisionQueryTime: NotationSymbol  # total time of the clearance queries for every collision hull in one permission tick ($T_{\text{query\_total}}$) [s]
    TotalDataCost: NotationSymbol  # total cost of a training dataset over its lifecycle ($C_{\text{total}}$) [USD]
    TotalDelayPhaseShift: NotationSymbol  # phase that the combined compute, bus and zero-order-hold delays contribute near crossover ($\phi_{\text{total}}$) [rad]
    TotalEnergy: NotationSymbol  # total energy a workload consumes ($E_{\text{total}}$) [J]
    TotalExecutedTrialCount: NotationSymbol  # total number of trials a test campaign executed ($N_{\text{total}}$)
    TotalFailureRecoveryTime: NotationSymbol  # wall-clock time from a failure until training is productive again, including replay of the work lost since the last checkpoint: detection, restart, checkpoint load and replay ($T_{\text{recovery}}$) [s]
    TotalGateCharge: NotationSymbol  # total charge the gate driver moves into or out of a power switch's gate during one transition ($Q_g$) [A*s]
    TotalInterventionCount: NotationSymbol  # total number of interventions recorded across an evaluation campaign, human takeovers, permission-path clips and stops, and tripwires ($N_{\text{int}}$)
    TotalKineticEnergy: NotationSymbol  # total kinetic energy of a machine, its translational energy plus the rotational energy of its powertrain ($E_{k,\text{total}}$) [J]
    TotalMemoryFootprint: NotationSymbol  # total memory a workload occupies on a device ($M_{\text{total}}$) [bytes]
    TotalPipelineLatency: NotationSymbol  # end-to-end latency of the physical sensing-to-actuation pipeline, from observation capture to actuation ($L_{\text{lat,total}}$) [s]
    TotalTrainingRunCost: NotationSymbol  # expected total cost of one training run: base compute, expected failure cost and the cost of fault-tolerance mechanisms ($C_{\text{total}}$) [USD]
    TraceCrashCount: NotationSymbol  # number of work-losing crashes in one failure trace ($N_{\text{crash}}$)
    TraceCrashIndex: NotationSymbol  # position of one crash among those in a failure trace ($i$)
    TraceDecay: NotationSymbol  # generalized advantage estimator's decay factor between zero and one ($\lambda$) [dimensionless]
    TraceIntermediateAdjoint: NotationSymbol  # reverse-mode adjoint of an intermediate variable, the derivative of the output with respect to it ($\bar{v}$) [dimensionless]
    TraceIntermediateTangent: NotationSymbol  # forward-mode tangent of an intermediate variable, its directional derivative ($\dot{v}$) [dimensionless]
    TraceIntermediateVariable: NotationSymbol  # intermediate variable in the evaluation trace of a differentiated program, its primal value ($v$) [dimensionless]
    TrackIndex: NotationSymbol  # index of one active track in a multi-object tracker ($i$)
    TrackLastUpdateTime: NotationSymbol  # time of the most recent observation associated with a track ($t_{\text{last}}$) [s]
    TrackLossTimeout: NotationSymbol  # time a coasting track may go without an associated observation before it is terminated ($T_{\text{lost}}$) [s]
    TrackPositionDeviation: NotationSymbol  # root-mean-square position deviation of a track covariance propagated forward over an elapsed time ($\sigma_{\text{track}}$) [m]
    TrackPredictedErrorCovariance: NotationSymbol  # error covariance of a track predicted to frame t from observations through frame t minus one ($\mathbf{P}_{t|t-1}$)
    TrackPredictedStateEstimate: NotationSymbol  # state of a track predicted to frame t from observations through frame t minus one ($\hat{\mathbf{x}}_{t|t-1}$)
    TrackedEvidenceHorizon: NotationSymbol  # evidence horizon of a target whose steady drift a tracker cancels, leaving only slip acceleration ($\tau_{\text{ev,tracked}}$) [s]
    TrackerFrameIndex: NotationSymbol  # index of one perception frame in a multi-object tracker ($t$)
    TrackingErrorAcceleration: NotationSymbol  # second time derivative of the tracking error vector, the acceleration error ($\ddot{\mathbf{e}}$)
    TrackingErrorAllowance: NotationSymbol  # part of the swept-volume clearance left for tracking error after the structural deflection and compliance allowance is deducted, a ceiling on the tracking threshold ($\epsilon_{\text{pos}}$) [m]
    TrackingErrorBound: NotationSymbol  # bound on the distance between the commanded and the executed position of the tracked point ($\epsilon_{\text{track}}$) [m]
    TrackingErrorDistance: NotationSymbol  # instantaneous tracking error, the distance between the measured and the commanded position of the tracked point ($e$) [m]
    TrackingErrorRate: NotationSymbol  # time derivative of the tracking error vector, the velocity error of a feedback controller ($\dot{\mathbf{e}}$)
    TrackingErrorVector: NotationSymbol  # tracking error vector of a feedback controller, the commanded reference state minus the measured state ($\mathbf{e}$)
    TrackingRelaxationSlack: NotationSymbol  # nonnegative slack variable that relaxes the control Lyapunov (tracking) constraint in a CLF-CBF quadratic program while the barrier constraint stays hard ($\delta$) [varies]
    TrackingSlackPenaltyWeight: NotationSymbol  # weight on the squared tracking-relaxation slack in the objective of a CLF-CBF quadratic program ($p$) [varies]
    TrackingStartInstant: NotationSymbol  # instant, on the permission clock, at which nominal tracking of an admitted trajectory record begins ($t_{\text{start}}$) [s]
    TractionLimitedDeceleration: NotationSymbol  # deceleration the machine can achieve once floor traction is taken into account, the smaller of the credible deceleration and friction coefficient times gravity ($a$) [m/s^2]
    TractiveForce: NotationSymbol  # longitudinal tractive force the ground transmits to a wheel at its contact patch, a function of slip ratio ($F_x$) [N]
    TrainableStateMemory: NotationSymbol  # memory holding the trainable parameters of an adaptation update together with their gradients and optimizer state ($M_{\text{train}}$) [bytes]
    TrainerUpdateTime: NotationSymbol  # time the trainer spends computing and applying one parameter update after its rollouts are collected ($T_{\text{train}}$) [s]
    TrainingCompute: NotationSymbol  # total floating-point operations needed to train a model ($O_{\text{train}}$) [FLOP]
    TrainingCost: NotationSymbol  # cost of training and evaluating a model, one-time or per period of retraining ($C_{\text{train}}$) [USD]
    TrainingDataset: NotationSymbol  # training dataset, the collection of examples a model is trained on ($D$)
    TrainingDistribution: NotationSymbol  # data distribution the model was trained on, the baseline that drift is measured against ($P_0$)
    TrainingEmpiricalCdf: NotationSymbol  # empirical cumulative distribution function of a feature in the training-baseline sample ($F_{P_0}$)
    TrainingExample: NotationSymbol  # one training example, an input together with its label, treated as a point whose weight in the training loss can change ($z$)
    TrainingFailureCount: NotationSymbol  # number of failures that occur during one training run ($N_{\text{failures}}$)
    TrainingFeatureMean: NotationSymbol  # mean of one feature fitted on the training data, as a cleaning or normalization parameter ($\mu$) [varies]
    TrainingFeatureStandardDeviation: NotationSymbol  # standard deviation of one feature fitted on the training data, as a cleaning or normalization parameter ($\sigma$) [varies]
    TrainingLabelMask: NotationSymbol  # indicator that admits a logged command as an imitation training target (one) or masks it out (zero) ($\mathcal{M}_{\text{train}}$) [dimensionless]
    TrainingLoss: NotationSymbol  # loss a model incurs on its training data, the objective its weights are fit to ($\mathcal{L}_{\text{train}}$)
    TrainingPathPredictor: NotationSymbol  # training code path viewed as a function from a raw input to the model's output, as offline validation evaluates it ($f_{\text{train}}$)
    TrainingPipelineFunction: NotationSymbol  # function that maps a training pipeline's versioned inputs and its execution nondeterminism to the model output it produces ($f$)
    TrainingRunCount: NotationSymbol  # number of training runs that reuse one curated dataset, such as sweeps and ablations ($N_{\text{runs}}$)
    TrainingSampleThroughput: NotationSymbol  # samples per second a training job actually processes ($R_{\text{train}}$) [samples/s]
    TrainingSchemaKeySet: NotationSymbol  # set of parameter names that appeared in the tool schemas of the training corpus ($\mathcal{K}_{\text{train}}$)
    TrainingServingSkew: NotationSymbol  # expected absolute difference between the serving-path and training-path model scores on the same inputs, over a stated comparison-input distribution ($S_{\text{skew}}$) [dimensionless]
    TrainingSplitFraction: NotationSymbol  # fraction of individual traces a random trace-level split assigns to the training partition ($f_{\text{train}}$) [dimensionless]
    TrainingStepCount: NotationSymbol  # number of training steps over which a per-step quantity is summed or averaged ($K$)
    TrainingStepEnergy: NotationSymbol  # energy a fleet consumes executing one distributed training step ($E_{\text{step}}$) [J]
    TrainingStepOperations: NotationSymbol  # arithmetic operations one training step executes across all devices, for every token in the global batch ($O_{\text{step}}$) [FLOP]
    TrainingStepTime: NotationSymbol  # wall-clock time of one training step, bounded below by the slower of compute and input I/O ($T_{\text{step}}$) [s]
    TrainingThroughput: NotationSymbol  # batches per second a training system completes, the slowest of preprocessing, transfer and compute ($R_{\text{training}}$) [batches/s]
    TrainingTime: NotationSymbol  # wall-clock time to complete one training run ($T_{\text{train}}$) [s]
    TrainingTransformFunction: NotationSymbol  # transformation the training pipeline applies to a raw input to produce model features ($f_{\text{train}}$)
    Trajectory: NotationSymbol  # ordered record of the states, actions, and observations of one delegated task ($\tau$)
    TrajectoryAction: NotationSymbol  # action the agent takes at one turn of a trajectory or decision process, an element of its action set ($a$)
    TrajectoryAdmissionRate: NotationSymbol  # rate at which trajectories are admitted, or arrive at a stage, per second ($\lambda_{\text{traj}}$) [trajectories/s]
    TrajectoryBudget: NotationSymbol  # ceilings on turns, tokens, wall-clock time, and dollars that the harness checks before every call ($\mathbf{B}$)
    TrajectoryCallCount: NotationSymbol  # number of model calls one admitted trajectory makes, including retries, reflection loops and delegated sub-agent calls ($N_{\text{call}}$)
    TrajectoryCategoryIndex: NotationSymbol  # index of one trajectory category in a curated corpus, such as direct successes, recovery traces or negatives ($j$)
    TrajectoryCheckpointCount: NotationSymbol  # number of checkpoints a trajectory publishes over its run ($N_{\text{ckpt}}$)
    TrajectoryDivergenceResidual: NotationSymbol  # signed error between measured and planned kinematics used to detect a diverging rollout ($e_{\text{div}}$) [m]
    TrajectoryDuration: NotationSymbol  # elapsed time of one trajectory between its declared endpoints ($T_{\text{traj}}$) [s]
    TrajectoryGoodput: NotationSymbol  # distinct accepted task completions per unit of elapsed time under the declared contract ($\mathcal{G}$) [tasks/s]
    TrajectoryObservation: NotationSymbol  # observation the environment returns to the agent after an action, an element of the decision process's observation set ($o$)
    TrajectoryOptimizationObjective: NotationSymbol  # objective functional continuous trajectory optimization minimizes over a joint trajectory, balancing smoothness against obstacle proximity ($\mathcal{U}$) [varies]
    TrajectoryRecordDuration: NotationSymbol  # time span an admitted trajectory record must cover: request offset, design-tail replacement latency, admission reserve and stop ($T_{\text{record}}$) [s]
    TrajectoryRecordFrame: NotationSymbol  # coordinate frame in which a trajectory record expresses its references ($\mathcal{F}$)
    TrajectoryReturn: NotationSymbol  # discounted sum of the rewards a trajectory collects, a function of the whole trajectory ($R$) [dimensionless]
    TrajectoryReward: NotationSymbol  # reward a verifier assigns to a whole trajectory ($R$) [dimensionless]
    TrajectorySampleCount: NotationSymbol  # number of time samples in a candidate trajectory that a checker evaluates ($N$)
    TrajectoryState: NotationSymbol  # context state the model saw at one turn of a trajectory, written with an explicit turn index ($s$)
    TrajectoryStateSpace: NotationSymbol  # set of system states that simulated and physical trajectories take values in ($\mathcal{S}$)
    TrajectoryTimeLimit: NotationSymbol  # elapsed wall-clock time a trajectory budget allows, the trajectory deadline ($T_{\max}$) [s]
    TrajectoryTokenCost: NotationSymbol  # total token charge of one trajectory's calls under a declared tariff, fresh input, cached input and output ($C_{\text{session}}$) [USD]
    TrajectoryTokenLimit: NotationSymbol  # largest cumulative number of tokens, input plus output, the harness lets one trajectory consume ($N_{\max}$)
    TrajectoryTurnLimit: NotationSymbol  # largest number of turns the harness admits for one trajectory, the turn allowance of its budget ($H_{\max}$)
    TransceiverDelay: NotationSymbol  # signal delay through a fieldbus node transceiver ($t_{\text{transceiver}}$) [s]
    TransductionInstant: NotationSymbol  # instant at which a sensor converted the physical quantity it measures into a signal ($t_{\text{transduction}}$) [s]
    TransferBufferCount: NotationSymbol  # number of buffers staging batches for transfer to the accelerator ($N_{\text{transfer}}$)
    TransferDiscrepancyBound: NotationSymbol  # bound on the sim-to-real transfer discrepancy, the largest weighted state error between real and simulated trajectories over the horizon ($\epsilon_{\text{gap}}$) [dimensionless]
    TransferTimeForPayload: NotationSymbol  # time to communicate a payload of x bytes under the relevant link or collective model ($T_{\text{transfer}}$) [s]
    TransferredDataVolume: NotationSymbol  # data volume transferred out over the network in an accounting period ($D_{\text{vol,transfer}}$) [bytes]
    TransformGraph: NotationSymbol  # directed graph whose vertices are a robot's coordinate frames and whose edges are the rigid transforms between them ($\mathcal{G}$)
    TransformGraphEdgeSet: NotationSymbol  # set of directed edges of a robot's transform graph, each a time-varying rigid transform between two frames ($\mathcal{E}$)
    TransformGraphFrame: NotationSymbol  # one coordinate frame, a vertex of a robot's transform graph ($\mathcal{F}$)
    TransformGraphVertexSet: NotationSymbol  # set of coordinate frames that are the vertices of a robot's transform graph ($\mathcal{V}$)
    TransformRotationBlock: NotationSymbol  # rotation block of the homogeneous transform from frame B to frame A ($\mathbf{R}_{AB}$)
    TransformTranslationBlock: NotationSymbol  # translation block of the homogeneous transform from frame B to frame A ($\mathbf{t}_{AB}$) [m]
    TransformationInputCount: NotationSymbol  # number of labeled input records a transformation derives one record from ($k$)
    TransformedCartesianPosition: NotationSymbol  # position of a 3D point after a rigid transform carries it into another coordinate frame ($\mathbf{p}'$) [m]
    TransformedHomogeneousPointCoordinates: NotationSymbol  # homogeneous coordinates of a 3D point after a rigid transform carries it into another coordinate frame ($\tilde{\mathbf{p}}'$) [m]
    TransistorThresholdVoltage: NotationSymbol  # gate-source voltage at which a transistor begins to conduct, its threshold voltage ($V_{\text{th}}$) [V]
    TransitionStorageSize: NotationSymbol  # bytes one stored reinforcement-learning transition occupies in a rollout store or replay buffer ($b_{\text{tr}}$) [bytes]
    TranslationComponent: NotationSymbol  # shift along one spatial axis, one component of a translation vector ($v$) [pixels]
    TranslationVector: NotationSymbol  # vector by which an image or feature map is shifted ($\mathbf{v}$)
    TranslationalDriftError: NotationSymbol  # displacement from unobserved translational motion during a stale interval ($e_{\text{trans}}$) [m]
    TranslationalJerk: NotationSymbol  # rate of change of acceleration of a moving body along its path, the third time derivative of its position ($j$) [m/s^3]
    TranslationalKineticEnergy: NotationSymbol  # kinetic energy of a machine's translating chassis and payload, one half its mass times its speed squared ($E_{k,\text{trans}}$) [J]
    TransmissionEnergyPerByte: NotationSymbol  # energy a device spends transmitting one byte of data over a wireless link ($E_{\text{tx}}$) [J/byte]
    TransmissionInputInertia: NotationSymbol  # inertia on the input side of a transmission, the rotor plus the gear input stage ($J_{\text{in}}$) [kg*m^2]
    TransmissionTorsionalStiffness: NotationSymbol  # torsional stiffness of a joint's transmission between rotor and output link ($K_{\text{joint}}$) [N*m/rad]
    TransverseStandardDeviation: NotationSymbol  # standard deviation of an unprojected point's position perpendicular to the viewing ray ($\sigma_\perp$) [m]
    TravelPosition: NotationSymbol  # position of a body along the path over which a force acts ($x$) [m]
    TraverseDistance: NotationSymbol  # distance driven along a straight traverse ($d$) [m]
    TreatedNeighborFraction: NotationSymbol  # fraction of one user's recorded connections that receive the treatment, its network exposure ($e$) [dimensionless]
    TreatmentAssignment: NotationSymbol  # treatment assignment of one experimental unit, one if treated and zero for control ($w$) [dimensionless]
    TreatmentConversionRate: NotationSymbol  # observed conversion rate of the treatment variant of an A/B test ($\hat{p}_B$) [dimensionless]
    TreatmentSampleSize: NotationSymbol  # number of observations in the treatment variant of an A/B test ($m_B$)
    TreeAllReduceCompletionTime: NotationSymbol  # time to complete an AllReduce over a binary tree of devices ($T_{\text{tree}}$) [s]
    TreeLikeAllReduceTime: NotationSymbol  # modeled time of a tree-like AllReduce, such as a simple reduce-then-broadcast tree or recursive doubling, that forwards the full message in each of its logarithmic rounds ($T_{\text{tree-like}}$) [s]
    TreeSearchExplorationWeight: NotationSymbol  # positive weight on the exploration bonus of the PUCT selection rule, trading exploitation of high-scoring edges against exploration of under-sampled ones ($c_{\text{explore}}$) [dimensionless]
    TreeSearchLeafScore: NotationSymbol  # score of a newly reached partial candidate in a tree search, supplied by a process scorer or a check, that each visit backs up along its path ($V$) [dimensionless]
    TreeSearchMeanReturn: NotationSymbol  # running mean of the leaf scores backed up through one edge of a tree search, the edge's empirical mean return ($Q$) [dimensionless]
    TreeSearchPriorDistribution: NotationSymbol  # prior distribution over the extensions available at a partial state, such as the base model's proposal probability for each reasoning step, that steers a tree search ($p_{\text{prior}}$)
    TreeSearchVisitCount: NotationSymbol  # number of times a tree search has traversed one edge, an extension taken from a partial state ($N_{\text{visit}}$)
    TrialCampaignWallTime: NotationSymbol  # wall-clock duration of a physical learning campaign, trials times rollout plus reset time, divided by the rig count ($T_{\text{wall}}$) [s]
    TrialCompletionIndicator: NotationSymbol  # binary outcome of one trial, one when the goal is reached within the execution deadline, interventions allowed ($Y$) [dimensionless]
    TrialCondition: NotationSymbol  # condition of one physical trial: initial plant state, environment parameters, plant parameters, computational configuration and termination protocol ($c$)
    TrialControlStepCount: NotationSymbol  # number of control cycles executed in one trial, indexed by trial ($N_{\text{steps}, i}$)
    TrialCount: NotationSymbol  # number of physical trials in a learning or evaluation campaign ($N$)
    TrialFailureIndicator: NotationSymbol  # outcome of one trial, one if it fails and zero if it succeeds ($x$) [dimensionless]
    TrialFailureProbability: NotationSymbol  # probability that one physical trial fails ($p_{\text{fail}}$) [dimensionless]
    TrialIndex: NotationSymbol  # position of one physical trial within a learning campaign, counted from one ($n$)
    TrialInterventionCount: NotationSymbol  # number of supervisory interventions, human or automatic, during one trial, indexed by trial ($n_{\text{int}}$)
    TrialLoadReversalCount: NotationSymbol  # number of drivetrain load reversals one physical trial causes ($n_{\text{rev}}$)
    TrialPassProbability: NotationSymbol  # probability that a fixed candidate passes one verification trial under a specified protocol ($p_{\text{trial}}$) [dimensionless]
    TrialPathDistance: NotationSymbol  # path distance the machine traverses during one trial, indexed by trial ($d$) [m]
    TrialResetLatency: NotationSymbol  # time to restore the robot and workspace for the next trial after one completes or aborts: physical restore, recalibration and telemetry flush ($T_{\text{reset}}$) [s]
    TrialResetTime: NotationSymbol  # time to return a physical rig and its fixture to the initial configuration between trials ($t_{\text{reset}}$) [s]
    TrialRolloutTime: NotationSymbol  # duration of one physical trial's autonomous execution on the machine, excluding the reset ($t_{\text{rollout}}$) [s]
    TrialRunTime: NotationSymbol  # active execution time of one trial, before its reset ($t_{\text{run}}$) [s]
    TrialSuccessProbability: NotationSymbol  # true probability that one trial succeeds under the declared evaluation protocol and test distribution ($p$) [dimensionless]
    TriggeredWindowArrivalCount: NotationSymbol  # number of requests that arrive after the request that opened a batching window and before the window closes ($Q_{\text{arr}}$)
    TrueClearance: NotationSymbol  # actual separation between an obstacle and the machine's envelope ($C_{\text{true}}$) [m]
    TrueLabelComponent: NotationSymbol  # one entry of an example's true label, such as the one-hot target for one class or a scalar regression target ($y$) [dimensionless]
    TrueOutcome: NotationSymbol  # true binary outcome of a randomly drawn individual, such as whether an applicant repays, one for the positive class ($Y$) [dimensionless]
    TrueOutcomeLevel: NotationSymbol  # one value of a binary true outcome, zero or one, on which a fairness condition is evaluated ($v$)
    TrueReferentScore: NotationSymbol  # normalized score a detector assigns to the ground-truth referent of one calibration scene ($c_{y_j}$) [dimensionless]
    TruncatedDistanceSample: NotationSymbol  # projective distance sample at a voxel clamped to plus or minus the truncation band, in meters ($d_{\mu}$) [m]
    TruncatedKeyApproximation: NotationSymbol  # rank-r truncated singular value decomposition approximation of a key activation matrix, the optimal approximation of that rank in spectral and Frobenius norm ($\hat{\mathbf{K}}_r$)
    TruncatedKeyLeftSingularMatrix: NotationSymbol  # matrix of the first r left singular vectors of a key activation matrix ($\mathbf{U}_r$)
    TruncatedKeyRightSingularMatrix: NotationSymbol  # matrix of the first r right singular vectors of a key activation matrix ($\mathbf{V}_r$)
    TruncatedKeySpectrumMatrix: NotationSymbol  # diagonal matrix of the r largest singular values of a key activation matrix ($\mathbf{\Sigma}_r$)
    TruncationBandVolume: NotationSymbol  # volume of the shell around observed surfaces within the truncation band ($V_{\text{band}}$) [m^3]
    TruncationRank: NotationSymbol  # number of leading singular components a truncated singular value decomposition keeps ($r$)
    TruthInstrumentSet: NotationSymbol  # set of independent truth instruments (sensors and digitizers) that observe the plant separately from the system under test ($\mathcal{I}_{\text{truth}}$)
    TsdfAccumulatedDistance: NotationSymbol  # running weighted average of the truncated signed distance at a voxel after t frames ($D$) [dimensionless]
    TsdfAccumulatedWeight: NotationSymbol  # accumulated observation weight at a voxel after t frames ($W$) [dimensionless]
    TsdfObservationWeight: NotationSymbol  # weight one new depth observation receives when it is fused into a voxel's running distance average ($w$) [dimensionless]
    TsdfTruncationBand: NotationSymbol  # distance from a surface within which signed distances are stored rather than clamped ($\mu_{\text{tsdf}}$) [m]
    TsdfWeightCap: NotationSymbol  # largest value the accumulated fusion weight of a voxel may reach, which keeps old evidence from dominating ($W_{\max}$) [dimensionless]
    TuckerCoreTensor: NotationSymbol  # small dense core tensor of a Tucker decomposition, multiplied along each mode by a factor matrix ($\mathcal{G}$)
    TurnAppendLength: NotationSymbol  # number of tokens one turn adds to the next prompt ($\Delta S_{\text{turn}}$)
    TurnBoundaryDelay: NotationSymbol  # time an admitted model call can postpone the next turn boundary, the time to generate its full output allowance ($t_{\text{boundary}}$) [s]
    TurnBudgetUsedFraction: NotationSymbol  # fraction of a trajectory's turn allowance already used ($\rho_H$) [dimensionless]
    TurnContext: NotationSymbol  # tokens the runtime assembles and sends to the model on one turn ($c$)
    TurnContributionDeviationBound: NotationSymbol  # uniform upper bound on the standard deviation of each turn's projected contribution to a policy-gradient estimate ($\sigma$) [dimensionless]
    TurnCovarianceBoundScale: NotationSymbol  # constant multiplying the geometric decay in an assumed bound on the covariance between two turns' projected policy-gradient contributions ($C$) [dimensionless]
    TurnCovarianceDecayRatio: NotationSymbol  # factor, between zero and one, by which an assumed bound on the covariance between two turns' projected policy-gradient contributions shrinks per turn of separation ($\alpha_{\text{corr}}$) [dimensionless]
    TurnFailureRate: NotationSymbol  # expected number of work-losing crashes per completed turn of a trajectory ($\lambda_{\text{fail}}$) [dimensionless]
    TurnGeneratedTokenCount: NotationSymbol  # number of output tokens each turn of a trajectory generates, taken as constant across turns in a token-cost inventory ($n_{\text{gen}}$)
    TurnHorizon: NotationSymbol  # number of sequential turns an agent trajectory runs ($H$)
    TurnIndex: NotationSymbol  # position of a turn within a trajectory, counted from zero or from one as the chapter states ($t$)
    TurnInputTokenCost: NotationSymbol  # inference charge for the input tokens of one turn of an attempt, under the provider's rate card including caching discounts ($C_{\text{in}}$) [USD]
    TurnOffGateCurrent: NotationSymbol  # current the gate driver sinks from a power switch's gate during turn-off ($I_{g,\text{off}}$) [A]
    TurnOffGateResistance: NotationSymbol  # gate resistance in the turn-off path of a power switch's gate driver ($R_{g,\text{off}}$) [Ohm]
    TurnOnGateResistance: NotationSymbol  # gate resistance in the turn-on path of a power switch's gate driver ($R_{g,\text{on}}$) [Ohm]
    TurnOutputTokenCost: NotationSymbol  # inference charge for the output tokens one turn of an attempt generates ($C_{\text{out}}$) [USD]
    TurnPayloadProbability: NotationSymbol  # probability that one turn of a trajectory encounters attacker-influenced content, independently across turns ($q_{\text{turn}}$) [dimensionless]
    TurnReadCount: NotationSymbol  # number of untrusted documents one turn of a trajectory reads ($r$)
    TurnReconstructionTime: NotationSymbol  # time to reconstruct one logged turn of a recovery suffix by reading, validating, folding and applying its records ($T_{\text{recon}}$) [s]
    TurnReexecutionTime: NotationSymbol  # time to re-execute one unlogged turn by repeating its model calls and tool invocations ($T_{\text{exec}}$) [s]
    TurnReward: NotationSymbol  # reward a rollout receives immediately after the action at one turn ($r$) [dimensionless]
    TurnSandboxCost: NotationSymbol  # charge for the wall-clock lifetime of an attempt's sandbox environment during one turn, idle waits included ($C_{\text{sandbox}}$) [USD]
    TurnStateGrowth: NotationSymbol  # bytes of checkpointed state, workspace files and record fields, that one turn adds ($s_{\text{turn}}$) [bytes]
    TurnToolFeeCost: NotationSymbol  # billable external tool, API and data-lookup fees one turn of an attempt incurs ($C_{\text{tools}}$) [USD]
    TustinEigenvalue: NotationSymbol  # eigenvalue of the bilinear (Tustin) transition matrix ($\lambda_{\text{Tustin}}$) [dimensionless]
    TustinTransitionMatrix: NotationSymbol  # discrete-time transition matrix that the bilinear (Tustin, trapezoidal) transform produces ($\mathbf{A}_{\text{Tustin}}$)
    TwinCriticReturnEstimate: NotationSymbol  # action-value estimate of the i-th of two independently parameterized critics, each a noisy estimate of the true action value ($Q_{\phi_i}$) [dimensionless]
    TwistCoordinates: NotationSymbol  # twist coordinates of a rigid-body pose increment: three translation coordinates stacked with three rotation coordinates ($\boldsymbol{\xi}$)
    TwistCovarianceMatrix: NotationSymbol  # six-by-six covariance of a pose perturbation in local tangent (twist) coordinates ($\boldsymbol{\Sigma}_{\boldsymbol{\xi}}$) [varies]
    TwistMatrix: NotationSymbol  # four-by-four matrix form of a twist, an element of the Lie algebra that the exponential map turns into a rigid transform ($\hat{\boldsymbol{\xi}}$)
    TwistRotationAngle: NotationSymbol  # rotation angle of a twist, the magnitude of its rotation coordinates ($\theta_{\text{rot}}$) [rad]
    TwistRotationCoordinates: NotationSymbol  # rotation part of a twist coordinate vector: rotation axis times rotation angle of a finite pose increment ($\boldsymbol{\omega}$) [rad]
    TwistTranslationMatrix: NotationSymbol  # three-by-three matrix that maps the translation coordinates of a twist to the translation of its exponential, the left Jacobian of SO(3) ($\mathbf{V}$)
    TwistWedgeOperator: NotationSymbol  # wedge operator of rigid-body motion, which maps six twist coordinates to their four-by-four matrix in the Lie algebra of the special Euclidean group ($\wedge$)
    UnacceptedTaskMeanCost: NotationSymbol  # mean task-account cost of the unaccepted tasks in a settled cohort: failures, cancellations and timeouts ($C_{\text{fail}}$) [USD]
    UnassistedCompletionCount: NotationSymbol  # number of trials in a campaign that reach the goal with no supervisory intervention ($N_{\text{unassisted}}$)
    UnassistedSuccessIndicator: NotationSymbol  # binary indicator that one trial reached its goal within the deadline with no intervention, the per-trial term of the autonomous completion rate ($A$) [dimensionless]
    UnbiasedAngularBrakingDeceleration: NotationSymbol  # angular deceleration a joint produces from its peak torque with no bias torque acting ($\alpha_{\text{nom}}$) [rad/s^2]
    UnbrakedImpactForce: NotationSymbol  # peak transient contact force when the robot strikes a body at speed without braking ($F_{\text{unbraked}}$) [N]
    UncachedInputTokenCount: NotationSymbol  # number of input tokens in a call that the provider bills at the uncached input rate ($n_{\text{in}}$)
    UncachedPromptLength: NotationSymbol  # number of tokens of a newly arriving request's prompt that have no cached KV state and must be prefilled ($S_{\text{new}}$)
    UncertainScenarioParameters: NotationSymbol  # uncertain input parameters of one simulated scenario, drawn from an operational or proposal distribution ($x$)
    UnconstrainedCallLatency: NotationSymbol  # latency of a tool call generated by free sampling without constrained decoding, including at most one repair turn when it fails to parse ($T_{\text{free}}$) [s]
    UnconstrainedParseFailureRate: NotationSymbol  # fraction of tool calls generated without constrained decoding whose output fails to parse ($p_{\text{err}}$) [dimensionless]
    UndampedNaturalFrequency: NotationSymbol  # undamped natural angular frequency of a second-order closed-loop error equation, the square root of stiffness over mass ($\omega_n$) [rad/s]
    UnfoldedConvolutionOutput: NotationSymbol  # convolution output computed as the im2col product of the unfolded patch matrix and the flattened filter ($\mathbf{Y}_{\text{col}}$)
    UnfoldedPatchMatrix: NotationSymbol  # matrix im2col builds by flattening each receptive-field patch of the input into one row ($\mathbf{X}_{\text{col}}$)
    UniformDistribution: NotationSymbol  # uniform distribution over the set or interval given as its argument ($\mathcal{U}$)
    UniformLoopTurnCost: NotationSymbol  # monetary cost of one turn of a bounded agent loop when every turn costs the same, including inference, tools, sandbox lease and assessment ($C_{\text{turn}}$) [USD]
    UnitAliasMultiplier: NotationSymbol  # number of base units in one unit alias, the conversion multiplier of a unit-conventions table row ($k$) [dimensionless]
    UnitImpulseFunction: NotationSymbol  # Dirac delta, the idealized unit impulse of zero width and unit area ($\delta$) [1/s]
    UnitQuaternion: NotationSymbol  # four-component unit-norm vector representing an orientation ($\mathbf{q}_{\text{quat}}$) [dimensionless]
    UnitQuaternionScalarComponent: NotationSymbol  # scalar part of a unit orientation quaternion ($q_w$) [dimensionless]
    UnitQuaternionSphere: NotationSymbol  # unit three-sphere, the set of four-component vectors of unit length in which unit quaternions live ($S^3$)
    UnitQuaternionXComponent: NotationSymbol  # x component of the vector part of a unit orientation quaternion ($q_x$) [dimensionless]
    UnitQuaternionYComponent: NotationSymbol  # y component of the vector part of a unit orientation quaternion ($q_y$) [dimensionless]
    UnitQuaternionZComponent: NotationSymbol  # z component of the vector part of a unit orientation quaternion ($q_z$) [dimensionless]
    UnitTemperatureProbabilityVector: NotationSymbol  # vector of next-token probabilities the softmax assigns at unit sampling temperature ($\mathbf{p}_{\tau_{\text{temp}}=1}$)
    UnlabeledPool: NotationSymbol  # pool of unlabeled candidate samples an active learner scores to choose what to label ($U$)
    UnloadedTickExecutionTime: NotationSymbol  # execution time of the permission tick measured without proposal-side load ($t_{\text{unloaded}}$) [s]
    UnmitigatedHardwareFaultRate: NotationSymbol  # rate of hardware faults whose hazardous effect no diagnostic mitigates, the rate compared with the target hazard rate ($\lambda_{\text{unmitigated}}$) [1/h]
    UnmodeledCommandPathDelay: NotationSymbol  # unmodeled delay added to the command path between the permission path and the drive, beyond the budgeted fieldbus time ($\Delta t_{\text{bus}}$) [s]
    UnobservableForceOffset: NotationSymbol  # arbitrary offset trajectory that, subtracted from the contact force and added to the sensor bias, leaves the force sensor's reading unchanged ($\Delta$) [N]
    UnobservedUnknownSpace: NotationSymbol  # part of the spatial memory map whose cells are unobserved or whose evidence has expired, which supplies no clearance evidence ($\mathcal{M}_{\text{unknown}}$)
    UnprojectionJacobian: NotationSymbol  # Jacobian of a camera-frame point with respect to its pixel coordinates and depth, which carries measurement covariance into Cartesian covariance ($\mathbf{J}_{\text{unproj}}$)
    UnrecoverableDamageCost: NotationSymbol  # cost of the unrecoverable damage a modeled external failure causes when an action executes automatically ($C_{\text{blast}}$) [USD]
    UnreservedStopDelay: NotationSymbol  # delay between stop dispatch and physical stop onset that the planned schedule did not reserve, such as an uncharged communication or actuator delay ($\delta$) [s]
    UnshardedLocalExecutionTime: NotationSymbol  # time one device needs to run a replica's whole unsharded local workload, its data movement, computation and host dispatch together, the reference the Fleet Law divides across devices ($T_{\text{compute}}$) [s]
    UnsquashedLatentAction: NotationSymbol  # unbounded latent action sampled from a squashed-Gaussian policy before the tanh squashing that bounds it ($\mathbf{u}$)
    UntilPrefixTime: NotationSymbol  # time between the evaluation time and an until operator's satisfaction time, over which the left formula must keep holding ($t''$) [s]
    UntrackedEvidenceHorizon: NotationSymbol  # evidence horizon of a target that drifts at a bounded velocity with no tracker compensating ($\tau_{\text{ev,untracked}}$) [s]
    UpdatedErrorCovariance: NotationSymbol  # error covariance at step k after incorporating that step's measurement ($\mathbf{P}_{k|k}$)
    UpdatedModelParameters: NotationSymbol  # model parameters after one optimizer update ($\theta_{\text{new}}$)
    UpdatedParameterSubset: NotationSymbol  # subset of the model parameters a few-shot adaptation policy allows to change ($\theta_{\text{updated}}$)
    UpdatedSoftmaxRunningMaximum: NotationSymbol  # running maximum of an online softmax after a new block of scores has been seen, replacing the previous running maximum ($m'$) [dimensionless]
    UpdatedStateEstimate: NotationSymbol  # state estimate at step k after incorporating that step's measurement ($\hat{\mathbf{x}}_{k|k}$)
    UpperRatioClippingWidth: NotationSymbol  # distance above one at which the clip-higher surrogate stops rewarding a rising probability ratio ($\epsilon_{\text{high}}$) [dimensionless]
    UprightPosition: NotationSymbol  # position of the rack-end upright in the estimation frame ($\mathbf{p}_{\text{upright}}$) [m]
    UprightRange: NotationSymbol  # range from the mobile base to the rack upright, the norm of their position difference ($r$) [m]
    UpstreamGradientElement: NotationSymbol  # one element of the upstream gradient vector ($\bar{y}$) [dimensionless]
    UpstreamGradientVector: NotationSymbol  # gradient of the loss with respect to an operation's output, passed backward into it ($\mathbf{v}$)
    UpstreamStageLatency: NotationSymbol  # latency of one stage of a sense-to-actuation pipeline, as a random variable ($X$) [s]
    UsableDemonstrationYield: NotationSymbol  # number of validated, policy-ready demonstrations one scheduled collection session yields ($Y_{\text{usable}}$)
    UsefulAcceleratorHourCost: NotationSymbol  # price of one accelerator-hour of useful work at the ideal rate, the capacity price divided by ML productivity goodput ($C_{\text{useful}}$) [USD/h]
    UsefulComputeFraction: NotationSymbol  # share of the distributed step time spent on useful local arithmetic ($f_{\text{compute}}$) [dimensionless]
    UsefulModelFlops: NotationSymbol  # floating-point operations the model's work actually requires, excluding recomputation and padding ($O_{\text{useful}}$) [FLOP]
    UsefulStepCount: NotationSymbol  # number of useful generation steps a task needs, excluding regenerated work ($N$)
    UsefulTrainingTime: NotationSymbol  # wall-clock time a training job spends on committed, productive steps that contribute to the final model ($T_{\text{useful}}$) [s]
    UserConnectionSet: NotationSymbol  # set of users recorded as connected to one user in the social or interaction graph ($\mathcal{N}$)
    UserSpaceSetupTime: NotationSymbol  # time to set up the user-space runtime of one sandbox instance, such as language runtimes and libraries ($T_{\text{user}}$) [s]
    UvloThresholdVoltage: NotationSymbol  # DC bus voltage below which the gate driver enters under-voltage lockout and disables gate bias ($V_{\text{UVLO}}$) [V]
    VacuumLightSpeed: NotationSymbol  # speed of light in vacuum, the upper limit on how fast an electrical signal crosses a chip or link ($c$) [m/s]
    ValidCandidateRate: NotationSymbol  # share of generated candidates that meet the specification, before any check ($p_{\text{valid}}$) [dimensionless]
    ValidTokenSet: NotationSymbol  # set of vocabulary tokens whose decoded strings keep the output a viable prefix of the target grammar from a parser state ($\mathcal{V}_{\text{valid}}$)
    ValidatedMotionDuration: NotationSymbol  # duration of validated demonstration motion a collection session stage produces ($T_{\text{yield}}$) [s]
    ValidatedProposalEvent: NotationSymbol  # event at which the runtime holds a validated, complete proposal for an external effect, first in the write-ahead dispatch ordering ($e_{\text{prop}}$)
    ValidationDataset: NotationSymbol  # held-out validation dataset of input-label pairs on which a model is scored ($\mathcal{D}_{\text{val}}$)
    ValidationGateThreshold: NotationSymbol  # threshold on the squared Mahalanobis distance above which a detection-track pairing is rejected ($\gamma_{\text{gate}}$) [dimensionless]
    ValidationLoss: NotationSymbol  # loss a trained model incurs on held-out validation data ($\mathcal{L}_{\text{val}}$)
    VanishingTimeOffset: NotationSymbol  # small positive time offset on either side of an instant, taken to zero in a limit ($\epsilon$) [s]
    VariationalDistribution: NotationSymbol  # tractable distribution, with variational parameters phi, fitted to approximate an intractable posterior by maximizing the evidence lower bound ($q_\phi$)
    VariationalParameters: NotationSymbol  # parameters of a variational distribution, adjusted to maximize the evidence lower bound ($\phi$)
    VectorAngle: NotationSymbol  # angle between two vectors in the geometric form of their dot product ($\phi$) [rad]
    VectorElementCount: NotationSymbol  # number of elements in each operand vector of an element-wise vector operation ($N_{\text{vec}}$)
    Velocity: NotationSymbol  # linear velocity of a point or body ($\mathbf{v}$) [m/s]
    VelocitySaturationExponent: NotationSymbol  # technology-dependent exponent of the alpha-power law, by which drain current grows with gate overdrive above threshold ($\alpha$) [dimensionless]
    VerificationCheckCost: NotationSymbol  # cost of running one verification check on generated work ($C_{\text{ver}}$) [varies]
    VerificationCheckCount: NotationSymbol  # number of verification checks over a task ($K$)
    VerificationPoolCapacity: NotationSymbol  # number of trajectories per second a verification pool can finish when every trajectory needs its full set of trials ($\mu_{\text{verify}}$) [trajectories/s]
    VerificationScheduleCost: NotationSymbol  # expected total cost of a task's generation, verification checks and error recovery as a function of the check interval ($C$) [varies]
    VerificationTrialCount: NotationSymbol  # number of verification trials an all-pass rule requires of one candidate, counting the first ($n_{\text{trial}}$)
    VerificationTrialIndex: NotationSymbol  # number of verification trials already run on a candidate, the index of a sum over trials ($j$)
    VerificationTrialTime: NotationSymbol  # mean time one verification trial occupies a worker slot, including environment provisioning, execution and cleanup ($T_{\text{trial}}$) [s]
    VerifiedDemonstrationDataset: NotationSymbol  # demonstration dataset that pairs observations with the setpoints the permission path admitted ($\mathcal{D}_{\text{verified}}$)
    VerifierFalseAcceptanceRate: NotationSymbol  # probability that a verifier passes a candidate that does not meet the specification ($\epsilon_{\text{FA}}$) [dimensionless]
    VerifierFalseRejectionRate: NotationSymbol  # probability that a verifier fails a candidate that meets the specification ($\epsilon_{\text{FR}}$) [dimensionless]
    VerifierPassProbability: NotationSymbol  # probability that a verifier passes one candidate drawn from the generator, valid or not ($q$) [dimensionless]
    VerifierPrecision: NotationSymbol  # probability that a candidate meets the specification given that the verifier passed it, the precision of a passing verdict ($\Pi_{V}$) [dimensionless]
    VerifierTimeout: NotationSymbol  # hard timeout after which one verifier execution is stopped, the service time of every assessment during a stall ($T_{\text{out}}$) [s]
    VerifierTrueAcceptanceRate: NotationSymbol  # probability that a verifier passes a candidate that meets the specification ($u$) [dimensionless]
    VerifierVerdict: NotationSymbol  # verdict of the training verifier on a rollout, written with a plus superscript for the event that it accepts ($V$)
    VersionSkewBound: NotationSymbol  # largest number of model versions by which deployment regions may differ under a bounded-staleness consistency policy ($k_{\text{skew}}$)
    VertexSlotTime: NotationSymbol  # time one vertex of a task-precedence graph occupies a worker slot, including model inference and tool execution ($d$) [s]
    VerticalFocalLength: NotationSymbol  # camera focal length expressed in pixels along the vertical image axis ($f_y$) [pixels]
    VibrationFrequency: NotationSymbol  # frequency of a mechanical vibration acting on a sensor ($f_{\text{vib}}$) [Hz]
    VirtualAccelerationCommand: NotationSymbol  # synthetic joint-acceleration command that computed torque control turns into torque through the inverse dynamics model ($\ddot{\mathbf{q}}^*$)
    VirtualDampingMatrix: NotationSymbol  # damping matrix an impedance controller emulates, mapping joint velocity to opposing torque ($\mathbf{D}_{\text{virt}}$)
    VirtualPipelineStagesPerDevice: NotationSymbol  # number of interleaved virtual pipeline stages assigned to each device ($V$)
    ViscousDampingMatrix: NotationSymbol  # positive semidefinite matrix of viscous and frictional damping that maps joint velocity to dissipative generalized force ($\mathbf{D}$)
    VisionBackboneTime: NotationSymbol  # time the perception vision backbone takes to process one frame ($t_{\text{backbone}}$) [s]
    VisionTokenEmbeddings: NotationSymbol  # token embeddings an embodied transformer computes from camera images ($\mathbf{z}_{\text{vision}}$)
    VisualConditioningEmbedding: NotationSymbol  # conditioning features a visual encoder extracts from the observation history once per action chunk, consumed by every denoising evaluation ($\mathbf{E}$)
    VisualEncodingTime: NotationSymbol  # time to extract the visual conditioning features once per action chunk, before the first denoising evaluation ($t_{\text{vis}}$) [s]
    VitImagePatch: NotationSymbol  # flattened image patch vector that a vision transformer projects into a token ($\mathbf{x}_p$)
    VitPatchProjectionWeights: NotationSymbol  # learned linear map from a flattened image patch to a token embedding in a vision transformer ($\mathbf{W}_{\text{patch}}$)
    VitPatchSize: NotationSymbol  # side length of one square image patch a vision transformer tokenizes ($p_{\text{patch}}$) [pixels]
    VitPatchTokenEmbedding: NotationSymbol  # latent token embedding a vision transformer forms from one image patch ($\mathbf{z}$)
    VocabularyToken: NotationSymbol  # one token of a model's vocabulary, the variable a next-token distribution, a logit or a sum over candidate tokens ranges over ($v$)
    VocabularyTokenLogit: NotationSymbol  # logit a language model produces for one vocabulary token at one sequence position, before the softmax ($o$) [dimensionless]
    VolumetricFlowRate: NotationSymbol  # volume of fluid passing a valve or line per unit time ($\dot{V}$) [m^3/s]
    VoxelBlockCoordinate: NotationSymbol  # integer coordinates of the voxel block that contains a point ($\mathbf{b}$)
    VoxelBlockEdgeLength: NotationSymbol  # edge length of one cubic voxel block, the voxel pitch times the voxels per edge ($L_{\text{blk}}$) [m]
    VoxelCenterPosition: NotationSymbol  # position of the center of one voxel of a volumetric map, the point at which a signed-distance field is evaluated ($\mathbf{p}_{\text{vox}}$)
    VoxelIndex: NotationSymbol  # index of one voxel of a volumetric grid ($i$)
    VoxelOccupancyState: NotationSymbol  # binary occupancy state of one voxel, one when occupied and zero when free ($m$) [dimensionless]
    VoxelPitch: NotationSymbol  # edge length of one cubic voxel or grid cell ($r$) [m]
    VoxelStorageSize: NotationSymbol  # bytes one voxel of a volumetric map stores: distance, weight and evidence epoch ($s_{\text{voxel}}$) [bytes]
    VoxelUpdateTraffic: NotationSymbol  # bytes of read-modify-write memory traffic one ray step spends on one traversed voxel ($s_{\text{rw}}$) [bytes]
    WarmupEpochCount: NotationSymbol  # number of epochs a curriculum takes to ramp from the easiest samples to the full dataset ($N_{\text{warmup}}$)
    WarmupStepCount: NotationSymbol  # number of initial optimizer steps over which learning-rate warmup ramps the rate from its base to its scaled value ($W$)
    WatchdogClockFrequency: NotationSymbol  # frequency of the independent oscillator that decrements a hardware watchdog's counter ($f_{\text{wdg}}$) [Hz]
    WatchdogCounterReading: NotationSymbol  # count held in a hardware watchdog's down-counter register, falling at the watchdog clock rate from its reload count ($W$)
    WatchdogEarliestServiceTime: NotationSymbol  # time after the last reload before which a windowed watchdog rejects a service attempt as too early, the lower end of its service window ($T_{\text{lower}}$) [s]
    WatchdogLatestServiceTime: NotationSymbol  # time after the last reload by which a windowed watchdog must be serviced, past which it trips the safe state ($T_{\text{upper}}$) [s]
    WatchdogLowerWindowCount: NotationSymbol  # counter value below which a windowed watchdog's refresh is too late ($W_{\text{lower}}$)
    WatchdogReloadCount: NotationSymbol  # count loaded into a hardware watchdog's down-counter at initialization and on each valid refresh ($W_{\text{init}}$)
    WatchdogServiceTime: NotationSymbol  # time after the last reload at which software services a windowed watchdog ($t_{\text{service}}$) [s]
    WatchdogToken: NotationSymbol  # rolling CRC token a control tick passes through its checkpoint stages, indexed by stage, which the watchdog refresh routine checks ($T$)
    WatchdogUpperWindowCount: NotationSymbol  # counter value above which a windowed watchdog rejects a refresh as too early ($W_{\text{upper}}$)
    WaveImpedance: NotationSymbol  # characteristic impedance of a wave-variable transformation, the ratio of effort to flow it treats as matched ($b$) [N*s/m]
    WeightElementStorageSize: NotationSymbol  # bytes needed to store one model weight at the precision the weights are held in ($s_{\text{elem,W}}$) [bytes/element]
    WeightMatrix: NotationSymbol  # matrix of learned weights one layer multiplies its input by, or the kernel tensor of a convolution layer ($\mathbf{W}$)
    WeightMatrixColumnCount: NotationSymbol  # number of columns of a layer's weight matrix ($n$)
    WeightMatrixEntry: NotationSymbol  # entry in row i and column j of a weight matrix ($W_{ij}$) [dimensionless]
    WeightMatrixRowCount: NotationSymbol  # number of rows of a layer's weight matrix ($m$)
    WeightMemory: NotationSymbol  # memory holding a model's weights ($M_{\text{weights}}$) [bytes]
    WeightOnlyStepTime: NotationSymbol  # weight-only estimate of the time of one serving forward pass, the larger of the weight-read time and the time to perform two weight operations per parameter per token position at the sustained rate ($t$) [s]
    WeightPenaltyStrength: NotationSymbol  # coefficient that scales an l1 or l2 penalty on weight magnitude added to the training loss, equivalent to weight decay under plain gradient descent ($\lambda_{\text{reg}}$) [dimensionless]
    WeightQuantizationScale: NotationSymbol  # quantization scale of a layer's weight tensor ($s_W$) [varies]
    WeightRangeBound: NotationSymbol  # largest weight magnitude a serving-time range check accepts, set from the distribution of the trained weights ($W_{\text{max}}$) [dimensionless]
    WeightVector: NotationSymbol  # weight vector of one neuron, one weight per input ($\mathbf{w}$)
    WeightedNormArgument: NotationSymbol  # vector whose weighted quadratic norm a definition states ($\mathbf{z}$) [varies]
    WheelAngularSpeed: NotationSymbol  # angular speed of a driven wheel about its axle ($\omega$) [rad/s]
    WheelDiameter: NotationSymbol  # outer diameter of a driven wheel ($D_{\text{wheel}}$) [m]
    WheelInertia: NotationSymbol  # moment of inertia of a driven wheel and what turns with it, about the axle ($J_{\text{wheel}}$) [kg*m^2]
    WheelNormalLoad: NotationSymbol  # normal load a wheel presses onto the ground at its contact patch ($F_z$) [N]
    WheelRadius: NotationSymbol  # rolling radius of a driven wheel ($r$) [m]
    WheelRevolutionCount: NotationSymbol  # number of revolutions a wheel turns over a test campaign, the wear it accumulates against a bearing rated life ($N_{\text{rev}}$)
    WheelTrackWidth: NotationSymbol  # distance between the contact points of the two drive wheels of a differential-drive base ($b$) [m]
    WhiteNoiseProcess: NotationSymbol  # zero-mean continuous Gaussian white-noise process, written with its mean and spectral density matrix ($\mathcal{WN}$)
    WholeBodyDecisionVariables: NotationSymbol  # decision vector of a whole-body control quadratic program: generalized accelerations, stacked contact forces and joint torques ($\mathbf{z}$)
    WholeRepositoryShardGrant: NotationSymbol  # child capability grant proposed for a test shard over the whole repository ($\Gamma_{b}$)
    WidthMultiplier: NotationSymbol  # channel count of a scaled model relative to its base model ($w_{\text{mult}}$) [dimensionless]
    WidthScalingRatio: NotationSymbol  # ratio by which compound scaling multiplies network width for each unit of the scaling coefficient ($\beta$) [dimensionless]
    WilsonIntervalCenter: NotationSymbol  # center of the Wilson score interval for an acceptance probability, the observed share shrunk toward one half ($\tilde{p}$) [dimensionless]
    WindingCoreThermalResistance: NotationSymbol  # conductive thermal resistance from a motor's winding to its stator core ($R_{\text{th,wc}}$) [K/W]
    WindingCurrent: NotationSymbol  # electric current flowing through a motor winding, which heats it at the rate current squared times resistance ($I$) [A]
    WindingJouleHeat: NotationSymbol  # resistive energy dissipated as heat in motor windings over an interval, the integral of current squared times resistance ($E_{\text{thermal}}$) [J]
    WindingLossPower: NotationSymbol  # resistive power dissipated in the motor windings ($P_{\text{loss}}$) [W]
    WindingTemperature: NotationSymbol  # temperature of the motor winding copper ($T_{\text{winding}}$) [degC]
    WindingTemperatureLimit: NotationSymbol  # largest winding temperature the insulation class and derating policy allow ($T_{\text{limit}}$) [degC]
    WindingTerminalVoltage: NotationSymbol  # signed voltage the inverter applies across the motor winding terminals ($V_{\text{terminal}}$) [V]
    WindingThermalCapacitance: NotationSymbol  # heat capacity of a motor's copper winding alone, the fast node of the two-node thermal model ($C_{\text{wind}}$) [J/K]
    WindowArrivalCount: NotationSymbol  # number of requests that arrive during one batching window ($k$)
    WindowArrivalOffset: NotationSymbol  # time at which a request arrives, measured from the start of its batching window ($t$) [s]
    WindowSpendingLimit: NotationSymbol  # largest total of admitted charge bounds permitted within any trailing spending window ($C_{\text{win}}$) [USD]
    WithinTaskRunStandardDeviation: NotationSymbol  # square root of the expected variance of one run's acceptance outcome across repeated runs of the same task, averaged over the task population ($\sigma_{\text{run}}$) [dimensionless]
    WorkedTraceProduct: NotationSymbol  # output of the reverse-mode worked example, the product of its two intermediates ($c$) [dimensionless]
    WorkedTraceProductAdjoint: NotationSymbol  # reverse-mode adjoint of the worked example output, seeded to one ($\bar{c}$) [dimensionless]
    WorkedTraceSine: NotationSymbol  # intermediate equal to the sine of the input in the reverse-mode worked example ($b$) [dimensionless]
    WorkedTraceSineAdjoint: NotationSymbol  # reverse-mode adjoint of the sine intermediate in the worked example ($\bar{b}$) [dimensionless]
    WorkedTraceSquare: NotationSymbol  # intermediate equal to the square of the input in the reverse-mode worked example ($a$) [dimensionless]
    WorkedTraceSquareAdjoint: NotationSymbol  # reverse-mode adjoint of the square intermediate in the worked example ($\bar{a}$) [dimensionless]
    WorkerBaseCommitHash: NotationSymbol  # content hash of the commit a worker started from ($h_{\text{base}}$)
    WorkerCandidateCommitHash: NotationSymbol  # content hash of the candidate commit a worker returns ($h_{\text{worker}}$)
    WorkerCompletionDistribution: NotationSymbol  # cumulative distribution function of one parallel worker's completion time, the probability it returns an admissible result within a given time ($F$) [dimensionless]
    WorkerCompletionTime: NotationSymbol  # time a parallel worker takes from dispatch to return an admissible result ($T$) [s]
    WorkerCount: NotationSymbol  # number of equally capable workers, agents, threads, or verifier slots in one pool ($N_{\text{workers}}$)
    WorkerIndex: NotationSymbol  # position of one data-parallel worker among the devices of a job, counted from one ($k$)
    WorkerOverheadSlope: NotationSymbol  # normalized coordination overhead added by each worker beyond the first, in a linear overhead model of a fixed work partition ($\gamma$) [dimensionless]
    WorkerTransformTime: NotationSymbol  # time one preprocessing worker takes to transform one batch ($T_{\text{transform}}$) [s]
    WorkerWrittenFileCount: NotationSymbol  # number of distinct files each worker writes, its write-set size ($w$)
    WorkloadMeanOutputLength: NotationSymbol  # mean number of output tokens per request across a workload's output-length distribution ($\mu$) [tokens]
    WorkloadRequestIndex: NotationSymbol  # index of one request among the requests of a measurement window ($i$)
    WorkspaceExtentX: NotationSymbol  # length of the workspace along the x axis ($L_x$) [m]
    WorkspaceExtentY: NotationSymbol  # length of the workspace along the y axis ($L_y$) [m]
    WorkspaceExtentZ: NotationSymbol  # length of the workspace along the z axis ($L_z$) [m]
    WorkspaceRestoreTime: NotationSymbol  # time to restore a trajectory's workspace files during resume ($T_{\text{ws}}$) [s]
    WorkspaceVolume: NotationSymbol  # physical volume of the operational workspace a map covers ($V_{\text{ws}}$) [m^3]
    WorldBaseTransform: NotationSymbol  # rigid transform mapping mobile-base coordinates into the world frame, the first link of the world, base, end-effector and camera chain ($\mathbf{T}_{WB}$)
    WorldBodyTransform: NotationSymbol  # rigid transform giving the vehicle body frame's pose in the world frame, mapping body coordinates into world coordinates ($\mathbf{T}_{wb}$)
    WorldFramePoint: NotationSymbol  # position of an observed point expressed in the world frame ($\mathbf{p}_{\text{world}}$) [m]
    WorldLidarTransform: NotationSymbol  # rigid transform giving a lidar's pose in the world frame, mapping lidar coordinates into world coordinates ($\mathbf{T}_{wl}$)
    WorldMapTransform: NotationSymbol  # rigid transform mapping map-frame coordinates into the world frame ($\mathbf{T}_{\text{world}}^{\text{map}}$)
    WorldPointCovariance: NotationSymbol  # covariance of an observation's position in world coordinates, compounded across the frame tree ($\boldsymbol{\Sigma}_{\text{world}}$)
    WorldReferenceFrame: NotationSymbol  # fixed coordinate frame attached to the world ($\mathcal{F}_W$)
    WorldStateSpace: NotationSymbol  # set of all possible states of the machine, its environment and the decision process ($\mathcal{S}$)
    WorldToCameraRotation: NotationSymbol  # rotation part of the extrinsic transform from the world frame to the camera optical frame ($\mathbf{R}_{CW}$) [dimensionless]
    WorldToCameraTransform: NotationSymbol  # rigid extrinsic transform that maps world-frame coordinates into the camera optical frame ($\mathbf{T}_{CW}$)
    WorldToCameraTranslation: NotationSymbol  # translation part of the extrinsic transform from the world frame to the camera optical frame ($\mathbf{t}_{CW}$) [m]
    WornPlantTransitionDistribution: NotationSymbol  # transition distribution of a physical plant late in a learning campaign, after many trials have worn it ($p_{n \gg 1}$)
    WorseArmSuccessProbability: NotationSymbol  # per-trial success probability of the worse policy in a two-policy power calculation ($p_{\text{low}}$) [dimensionless]
    WorstCaseExecutionTime: NotationSymbol  # worst-case execution time of one real-time task in response-time analysis, indexed by task ($C$) [s]
    WorstCaseInterruptLatency: NotationSymbol  # worst-case time from a hardware interrupt request to entry into its interrupt handler ($t_{\text{irq,lat}}$) [s]
    WorstCasePermissionCycleTime: NotationSymbol  # worst-case execution time of one permission-path cycle, from state acquisition to the final drive transfer ($T_{\text{exec}}$) [s]
    WorstCaseResponseTime: NotationSymbol  # longest time from a real-time task's release to its completion, indexed by task ($R$) [s]
    WristAngularVelocity: NotationSymbol  # angular velocity of a manipulator's wrist ($\boldsymbol{\omega}_{\text{wrist}}$) [rad/s]
    WristCameraEmbedding: NotationSymbol  # camera-identifier embedding of the wrist camera ($\mathbf{e}_{\text{wrist}}$)
    WristLinearVelocity: NotationSymbol  # linear velocity of a manipulator's wrist ($\mathbf{v}_{\text{wrist}}$) [m/s]
    WristObjectTransform: NotationSymbol  # homogeneous transform mapping object-frame points into the wrist frame, as the wrist camera estimates it ($\mathbf{T}_{\text{wrist} \leftarrow \text{object}}$)
    WristSkewDisplacement: NotationSymbol  # displacement of a wrist-camera observation caused by skew between camera and joint timestamps ($\Delta \mathbf{p}_{\text{skew}}$) [m]
    XAxisAngularRate: NotationSymbol  # component of a body's angular velocity about its x axis ($\omega_x$) [rad/s]
    XAxisRotationCoordinate: NotationSymbol  # component about the x axis of the rotation coordinates of a twist ($\omega_x$) [rad]
    XForceCeiling: NotationSymbol  # largest contact force along the x axis that an intent proposal requests or a lease admits ($F_x$) [N]
    XPositionErrorEnvelope: NotationSymbol  # conservative bound on the position error of a target estimate along the x axis of the base frame ($e_x$) [m]
    XTaskToleranceSemiAxis: NotationSymbol  # semi-axis of the task-tolerance ellipsoid along the x axis of the base frame ($r_x$) [m]
    XTorqueCeiling: NotationSymbol  # largest contact torque about the x axis that an intent proposal requests or a lease admits ($\tau_x$) [N*m]
    YAxisAngularRate: NotationSymbol  # component of a body's angular velocity about its y axis ($\omega_y$) [rad/s]
    YAxisRotationCoordinate: NotationSymbol  # component about the y axis of the rotation coordinates of a twist ($\omega_y$) [rad]
    YForceCeiling: NotationSymbol  # largest contact force along the y axis that an intent proposal requests or a lease admits ($F_y$) [N]
    YPositionErrorEnvelope: NotationSymbol  # conservative bound on the position error of a target estimate along the y axis of the base frame ($e_y$) [m]
    YTaskToleranceSemiAxis: NotationSymbol  # semi-axis of the task-tolerance ellipsoid along the y axis of the base frame ($r_y$) [m]
    YTorqueCeiling: NotationSymbol  # largest contact torque about the y axis that an intent proposal requests or a lease admits ($\tau_y$) [N*m]
    YawRate: NotationSymbol  # angular velocity of a body about its vertical (z) axis, its yaw rate ($\omega_z$) [rad/s]
    YoshikawaManipulability: NotationSymbol  # measure of distance from kinematic singularity, the square root of the determinant of J times J transpose ($w$) [dimensionless]
    ZAxisRotationCoordinate: NotationSymbol  # component about the z axis of the rotation coordinates of a twist ($\omega_z$) [rad]
    ZForceCeiling: NotationSymbol  # largest contact force along the z axis that an intent proposal requests or a lease admits ($F_z$) [N]
    ZPositionErrorEnvelope: NotationSymbol  # conservative bound on the position error of a target estimate along the z axis of the base frame ($e_z$) [m]
    ZTaskToleranceSemiAxis: NotationSymbol  # semi-axis of the task-tolerance ellipsoid along the z axis of the base frame ($r_z$) [m]
    ZTorqueCeiling: NotationSymbol  # largest contact torque about the z axis that an intent proposal requests or a lease admits ($\tau_z$) [N*m]
    ZTransformVariable: NotationSymbol  # complex frequency variable of the z-transform, standing for a one-sample advance in discrete time ($z$) [dimensionless]
    ZeroEntryIndicator: NotationSymbol  # indicator over a sparse weight matrix that is one at each exactly-zero entry ($\mathbf{1}_{\{(\mathbf{W}_{\text{sparse}})_{ij} = 0\}}$)
    ZeroFailureTrialCount: NotationSymbol  # number of consecutive failure-free trials needed to demonstrate a target success probability ($n$)
    ZeroFailureUpperConfidenceLimit: NotationSymbol  # one-sided upper confidence limit at the 95 percent level on the per-trial failure probability after a campaign in which every trial passed, the quantity the rule of three approximates ($p_{\text{fail, upper}}^{95\%}$) [dimensionless]
    ZeroMatrix: NotationSymbol  # matrix or vector whose entries are all zero ($\mathbf{0}$)
    ZeroMomentPoint: NotationSymbol  # point on the ground where the net horizontal tipping moment of the contact forces vanishes ($\mathbf{p}_{\text{ZMP}}$) [m]
    ZeroVector: NotationSymbol  # vector whose entries are all zero, such as the mean of zero-centered Gaussian noise ($\mathbf{0}$)
    ZohHoldTravel: NotationSymbol  # distance the plant drifts during one zero-order-hold interval before the next check can act ($d_{\text{ZOH}}$) [m]
    ZohPhaseShift: NotationSymbol  # phase a zero-order hold contributes at the crossover frequency, equivalent to half a sample period of delay ($\phi_{\text{ZOH}}$) [rad]
    ZohTransportDelay: NotationSymbol  # effective continuous-time delay a zero-order hold adds, half the sampling period ($\tau_{\text{ZOH}}$) [s]

Notation: NotationRegistry
Symbols: NotationRegistry
SYMBOLS: NotationRegistry
