import sys
import re
import yaml
from pathlib import Path

# Add project root to sys.path so we can import mlsysim
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "mlsysim"))
from mlsysim.notation.types import NotationSymbol

with open(PROJECT_ROOT / "mlsysim/mlsysim/notation/registry.yml") as f:
    raw_data = yaml.safe_load(f)

# Explicit renames and fixes
RENAME_MAP = {
    # Stutter removals
    "WeightMemoryWeights": "WeightMemory",
    "ActivationMemoryActivations": "ActivationMemory",
    "GradientMemoryGradients": "GradientMemory",
    "BatchServiceTimeSvc": "BatchServiceTime",
    "RequestTimeInSystemLat": "RequestTimeInSystem",
    "NumberOfKeyvalueHeadsKv": "NumberOfKeyValueHeads",
    "AcceleratorCountGpu": "AcceleratorCount",
    "InitialAccuracyZero": "InitialAccuracy",
    "TrainingDistributionZero": "TrainingDistribution",
    "SingledeviceComputeTime": "SingleDeviceComputeTime",
    "ScalingExponentScale": "ScalingExponent",
    "InitialAndCeilingErrorZero": "InitialTrackingError",
    "LinearBarrierDecayRateCbf": "LinearBarrierDecayRate",
    "VitProjectionPositionalBiasPatch": "VitPatchProjectionWeights",
    "CredibleDecelerationBrake": "CredibleBrakingDeceleration",
    "TransmissionEfficiencyGear": "GearboxTransmissionEfficiency",
    "ContinuousPeakTorqueHold": "HoldingTorque",
    "ContactStiffnessH": "HertzianContactStiffness",
    "TruncatedSignedDistanceVox": "TruncatedSignedDistance",
    "TsdfAccumulatedVolumeVox": "TsdfAccumulatedDistance",
    "TsdfAccumulatedVolumeVoxTwo": "TsdfAccumulatedWeight",

    # Casing fixes
    "EndtoendRequestLatency": "EndToEndRequestLatency",
    "PointtopointTransferTime": "PointToPointTransferTime",
    "PointtopointMessageSize": "PointToPointMessageSize",
    "HumanWalkingspeedBound": "HumanWalkingSpeedBound",
    "IronlawTermTimesCompute": "IronLawComputeTime",
    "IronlawTermTimesMemory": "IronLawMemoryTime",
    "CollectiveCompletionTimeRing": "RingAllReduceCompletionTime",
    "CollectiveCompletionTimeTree": "TreeAllReduceCompletionTime",

    # Multi-variable splits & Two/Three proper semantic names
    "LayerInputAndOutputWidths": "LayerInputWidth",
    "LayerInputAndOutputWidthsTwo": "LayerOutputWidth",
    "PidFeedbackGains": "PidProportionalGain",
    "PidFeedbackGainsTwo": "PidDerivativeGain",
    "PidFeedbackGainsThree": "PidIntegralGain",
    "CoordinateReferenceFramesW": "WorldReferenceFrame",
    "CoordinateReferenceFramesB": "BodyReferenceFrame",
    "SprtBoundariesUp": "SprtUpperBoundary",
    "SprtBoundariesLo": "SprtLowerBoundary",
    "LqrGainAndWeightsX": "LqrStateWeightMatrix",
    "LqrGainAndWeightsU": "LqrControlWeightMatrix",
    "DeadlineTwo": "TaskDeadline",
    "CheckpointWriteTimeTwo": "CheckpointWriteTime",
    "CheckpointIntervalCkptTwo": "OptimalCheckpointInterval",
    "ContentionAndCoherencyCoefficientsTwo": "CoherencyContentionCoefficient",
    "SamplingPeriodAndRate": "SamplingPeriod",
    "SamplingPeriodAndRateTwo": "SamplingRate",
    "TorqueAndBackemfConstants": "MotorTorqueConstant",
    "TorqueAndBackemfConstantsTwo": "MotorBackEmfConstant",
    "ContinuousPeakCurrent": "ContinuousCurrent",
    "ContinuousPeakCurrentTwo": "PeakCurrent",
    "PhaseInductanceAndResistance": "PhaseInductance",
    "PhaseInductanceAndResistanceTwo": "PhaseResistance",
    "WindingAndAmbientTemperature": "WindingTemperature",
    "WindingAndAmbientTemperatureTwo": "AmbientTemperature",
    "RotorAndLoadInertia": "RotorInertia",
    "RotorAndLoadInertiaTwo": "LoadInertia",
    "FundamentalAndEssentialMatrices": "FundamentalMatrix",
    "FundamentalAndEssentialMatricesTwo": "EssentialMatrix",
    "PowerRailInductanceEsrBus": "PowerRailInductance",
    "PowerRailInductanceEsrBusTwo": "PowerRailResistance",
    "VirtualDampingAndStiffness": "VirtualDampingMatrix",
    "VirtualDampingAndStiffnessTwo": "VirtualStiffnessMatrix",
    "BarrierConstraintRowsCbf": "BarrierConstraintMatrix",
    "BarrierConstraintRowsCbfTwo": "BarrierConstraintVector",
    "ProcessAndMeasurementNoise": "ProcessNoiseVector",
    "ProcessAndMeasurementNoiseTwo": "MeasurementNoiseVector",
    "TransitionAndMeasurementMaps": "StateTransitionMatrix",
    "TransitionAndMeasurementMapsTwo": "MeasurementMatrix",
    "ControlaffineDriftAndInputFields": "ControlAffineDriftField",
    "ControlaffineDriftAndInputFieldsTwo": "ControlAffineInputField",
}

# Entries to drop completely because they are malformed fragments from broken table parses
DROP_MACROS = {
    "\\BatchSizeTwo", "\\BatchSizeThree",
    "\\DatasetSizeTwo", "\\DatasetSizeThree",
    "\\ParametersTwo",
    "\\TimeTwo",
    "\\HiddenDimensionTwo", "\\HiddenDimensionThree",
    "\\LearningRateTwo",
    "\\SequenceLengthTwo",
    "\\GradientTwo",
    "\\GoalTwo",
    "\\MeanTimeBetweenFailuresTwo", "\\MeanTimeBetweenFailuresThree",
    "\\ArrivalRateTwo",
    "\\TestErrorRatesTwo",
    "\\TopTwo",
    # Malformed table fragments:
    "\\ActionValueOrCosttogoTwo",      # latex: a)
    "\\EstimatedAdvantageKTwo",        # latex: a_k)
    "\\DifferentialPrivacyBudgetTwo",  # latex: \delta)
    "\\DiffusionGaussianNoiseTwo",     # latex: \mathbf{I})
    "\\NoisePredictionNetworkThree",   # latex: k)
    "\\ActVariationalObjectiveTwo",    # latex: \phi)
    "\\CvaeRecognitionEncoderTwo",     # latex: \mathbf{A})
    "\\ImagePlanePixelCoordinatesTwo", # latex: v]^\top
    "\\ImagePlanePixelCoordinatesThree",# latex: \tilde{\mathbf{m}} (covered by main image coords)
    "\\CameraframeThreedCoordinateTwo",# latex: Y_c
    "\\CameraframeThreedCoordinateThree",# latex: Z_c]^\top
    "\\CoefficientOfRestitutionTwo",   # latex: 1]
    "\\CoriolisCentrifugalMatrixTwo",  # latex: \dot{\mathbf{q}})
    "\\ActuatorTorqueVectorTwo",       # duplicate torque
    "\\AngularVelocityAccelerationTwo",# duplicate accel
    "\\TakeoverResponseTimeThree",     # duplicate takeover
    "\\NetBrakingDecelerationTwo",     # duplicate deceleration
}

clean_symbols = []
seen_macros = set()

for v in raw_data["variables"]:
    raw_macro = v.get("macro", "").strip()
    if not raw_macro:
        continue

    if raw_macro in DROP_MACROS:
        continue

    macro_name = raw_macro.lstrip("\\")
    clean_macro_name = RENAME_MAP.get(macro_name, macro_name)
    clean_macro = f"\\{clean_macro_name}"

    if clean_macro in seen_macros:
        continue
    seen_macros.add(clean_macro)

    latex = v.get("latex", "").strip()
    defn = v.get("definition", "").strip()
    unit = v.get("unit", "") or ""
    notes = v.get("notes", "") or ""
    category = v.get("category", "") or "General Notation"
    volume = v.get("volume", "core") or "core"

    # Fix flipped columns in VarTwo and VarThree
    if macro_name == "VarTwo":
        clean_macro = "\\ChunkLeaseTime"
        latex = "T_{\\text{lease}}"
        defn = "Chunk Lease Time"
        unit = "seconds"
        notes = "Time lease granted to a chunk server in distributed filesystems."
        category = "Distributed Systems Notation"
        volume = "vol2"

    elif macro_name == "VarThree":
        clean_macro = "\\AdminLeaseTime"
        latex = "T_{\\text{admin}}"
        defn = "Administrative Lease Time"
        unit = "seconds"
        notes = "Administrative lease timeout in consensus state."
        category = "Distributed Systems Notation"
        volume = "vol2"

    # Fix \Time
    elif clean_macro == "\\Time":
        latex = "T"
        defn = "Time"
        unit = "s"
        notes = "Execution or wall-clock time (e.g. training time, serving latency term in the Iron Law)."
        category = "The Iron Law of ML Systems"
        volume = "core"

    # Specific cleanups for definition and latex
    if clean_macro == "\\LayerInputWidth":
        latex = "d_{\\text{in}}"
        defn = "Layer Input Width"
    elif clean_macro == "\\LayerOutputWidth":
        latex = "d_{\\text{out}}"
        defn = "Layer Output Width"
    elif clean_macro == "\\OptimalCheckpointInterval":
        latex = "T^\\ast_{\\text{ckpt}}"
        defn = "Optimal Checkpoint Interval"
        unit = "seconds"
    elif clean_macro == "\\CheckpointWriteTime":
        latex = "T_{\\text{write}}"
        defn = "Checkpoint Write Time"
        unit = "seconds"
    elif clean_macro == "\\PidProportionalGain":
        latex = "K_p"
        defn = "Proportional Gain"
    elif clean_macro == "\\PidDerivativeGain":
        latex = "K_d"
        defn = "Derivative Gain"
    elif clean_macro == "\\PidIntegralGain":
        latex = "K_i"
        defn = "Integral Gain"
    elif clean_macro == "\\WorldReferenceFrame":
        latex = "\\mathcal{F}_W"
        defn = "World Reference Frame"
    elif clean_macro == "\\BodyReferenceFrame":
        latex = "\\mathcal{F}_B"
        defn = "Body Reference Frame"
    elif clean_macro == "\\SprtUpperBoundary":
        latex = "\\Lambda_{\\text{up}}"
        defn = "SPRT Upper Decision Boundary"
    elif clean_macro == "\\SprtLowerBoundary":
        latex = "\\Lambda_{\\text{lo}}"
        defn = "SPRT Lower Decision Boundary"
    elif clean_macro == "\\LqrStateWeightMatrix":
        latex = "\\mathbf{Q}_x"
        defn = "LQR State Weighting Matrix"
    elif clean_macro == "\\LqrControlWeightMatrix":
        latex = "\\mathbf{R}_u"
        defn = "LQR Control Weighting Matrix"
    elif clean_macro == "\\SamplingPeriod":
        latex = "T_s"
        defn = "Sampling Period"
        unit = "s"
    elif clean_macro == "\\SamplingRate":
        latex = "f_s"
        defn = "Sampling Frequency / Rate"
        unit = "Hz"
    elif clean_macro == "\\MotorTorqueConstant":
        latex = "K_t"
        defn = "Motor Torque Constant"
        unit = "N*m/A"
    elif clean_macro == "\\MotorBackEmfConstant":
        latex = "K_e"
        defn = "Motor Back-EMF Constant"
        unit = "V/(rad/s)"
    elif clean_macro == "\\ContinuousCurrent":
        latex = "I_{\\text{cont}}"
        defn = "Continuous Motor Current"
        unit = "A"
    elif clean_macro == "\\PeakCurrent":
        latex = "I_{\\text{peak}}"
        defn = "Peak Motor Current"
        unit = "A"
    elif clean_macro == "\\PhaseInductance":
        latex = "L_{\\text{phase}}"
        defn = "Motor Phase Inductance"
        unit = "H"
    elif clean_macro == "\\PhaseResistance":
        latex = "R_{\\text{phase}}"
        defn = "Motor Phase Resistance"
        unit = "Ohm"
    elif clean_macro == "\\WindingTemperature":
        latex = "T_{\\text{winding}}"
        defn = "Motor Winding Temperature"
        unit = "degC"
    elif clean_macro == "\\AmbientTemperature":
        latex = "T_{\\text{amb}}"
        defn = "Ambient Temperature"
        unit = "degC"
    elif clean_macro == "\\RotorInertia":
        latex = "J_{\\text{rotor}}"
        defn = "Motor Rotor Inertia"
        unit = "kg*m^2"
    elif clean_macro == "\\LoadInertia":
        latex = "J_{\\text{load}}"
        defn = "Load Inertia"
        unit = "kg*m^2"
    elif clean_macro == "\\FundamentalMatrix":
        latex = "\\mathbf{F}"
        defn = "Fundamental Matrix"
    elif clean_macro == "\\EssentialMatrix":
        latex = "\\mathbf{E}"
        defn = "Essential Matrix"

    # Validate against Pydantic model
    sym = NotationSymbol(
        macro=clean_macro,
        latex=latex,
        definition=defn,
        unit=unit,
        notes=notes,
        category=category,
        volume=volume
    )
    clean_symbols.append(sym)

print(f"Purification complete: {len(raw_data['variables'])} raw entries -> {len(clean_symbols)} clean validated symbols.")

# Output to clean symbols.yaml
data_dict = {
    "symbols": [
        {
            "macro": s.macro,
            "latex": s.latex,
            "definition": s.definition,
            "unit": s.unit,
            "notes": s.notes,
            "category": s.category,
            "volume": s.volume
        }
        for s in clean_symbols
    ]
}

out_path = PROJECT_ROOT / "mlsysim/mlsysim/notation/data/symbols.yaml"
with open(out_path, "w", encoding="utf-8") as f:
    yaml.dump(data_dict, f, sort_keys=False, allow_unicode=True)

print(f"Wrote clean symbols to {out_path}")
