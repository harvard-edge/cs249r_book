#!/usr/bin/env python3
r"""Sanitize mlsysim/notation/data/symbols.yaml.

Removes table scrape artifacts, relational fragments (e.g. e \in [0 -> e),
unclosed brackets, and non-math markdown text.
"""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYMBOLS_PATH = PROJECT_ROOT / "mlsysim/mlsysim/notation/data/symbols.yaml"

# Macro cleanup mappings: (macro_name, clean_latex, keep_or_drop)
CLEANUPS = {
    r"\CoefficientOfRestitution": ("e", True),
    r"\ImagePlanePixelCoordinates": (r"\mathbf{u}_{\text{px}}", True),
    r"\CameraframeThreedCoordinate": (r"\mathbf{p}_c", True),
    r"\DiffusionGaussianNoise": (r"\boldsymbol{\epsilon}", True),
    r"\CvaeRecognitionEncoder": (r"q_\phi(\mathbf{z} \mid \mathbf{o})", True),
    r"\PhysicalWorldState": (r"s_t", True),
    r"\GeneralizedCoordinates": (r"\mathbf{q}", True),
    r"\GeneralizedVelocities": (r"\dot{\mathbf{q}}", True),
    r"\GeneralizedAccelerations": (r"\ddot{\mathbf{q}}", True),
    r"\ActuationSelectionMatrix": (r"\mathbf{S}", True),
    r"\ContactForceVectorK": (r"\mathbf{f}_k", True),
    r"\CenterOfMass": (r"\mathbf{c}", True),
    r"\CartesianThreedPosition": (r"\mathbf{p}", True),
    r"\RigidtransformTranslation": (r"\mathbf{t}", True),
    r"\DirectionalCosineMatrix": (r"\mathbf{R}", True),
    r"\UnitQuaternion": (r"\mathbf{q}_{\text{quat}}", True),
    r"\HomogeneousTransform": (r"\mathbf{T}_{AB}", True),
    r"\SensorLeverarmVector": (r"\mathbf{r}_{P/O}", True),
    r"\PinholeCameraIntrinsics": (r"\mathbf{K}", True),
    r"\EstimationErrorCovariance": (r"\mathbf{P}", True),
    r"\ForwardinvariantSafeSet": (r"\mathcal{C}", True),
    r"\ControlBarrierFunction": (r"h(\mathbf{x})", True),
    r"\NominalControlCommand": (r"\mathbf{u}_{\text{nom}}", True),
    r"\CertifiedControlAction": (r"\mathbf{u}^*", True),
    r"\ControlSetpointJerk": (r"\mathbf{j}(t)", True),
    r"\CognitivePolicyAction": (r"\mathbf{a}_t", True),
    r"\LatentStyleStateEmbedding": (r"\mathbf{z}", True),
    r"\ZeroMomentPointZmp": (r"\mathbf{p}_{\text{ZMP}}", True),
    # Drop table artifacts and text
    r"\PassAt": (None, False),
    r"\PassToThe": (None, False),
    r"\RemainingTtextlease": (None, False),
    r"\Ttextlease": (None, False),
    r"\Expiry": (None, False),
    r"\VarFour": (None, False),
    r"\SafetyIntegrityLevel": (None, False),
    r"\FourstageAuthorityChainNom": (None, False),
    r"\ActuatorTorqueVector": (None, False),
    r"\MechanicalPlantStateMathbfxMathbfqtop": (None, False),
    r"\StateZero": (None, False),
    r"\AuthorityZero": (None, False),
}

def main():
    with open(SYMBOLS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    symbols = data.get("symbols", [])
    new_symbols = []
    cleaned_count = 0
    dropped_count = 0

    for s in symbols:
        macro = s.get("macro")
        if macro in CLEANUPS:
            new_latex, keep = CLEANUPS[macro]
            if not keep:
                dropped_count += 1
                continue
            s["latex"] = new_latex
            cleaned_count += 1
        new_symbols.append(s)

    data["symbols"] = new_symbols

    with open(SYMBOLS_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"Sanitized symbols.yaml: cleaned {cleaned_count} entries, dropped {dropped_count} junk entries. Total symbols: {len(new_symbols)}")

if __name__ == "__main__":
    main()
