#!/usr/bin/env python3
"""Replace ``defaults.*`` with domain registry paths in book QMD and Python."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]

# Longest symbols first to avoid partial replacements.
_SYMBOL_MAP: list[tuple[str, str]] = [
    ("PCIE_SWITCH_MTTF_HOURS", "Systems.Reliability.PcieSwitch.mttf_hours"),
    ("GRID_INTERCONNECTION_QUEUE_US_GW", "Infrastructure.Capacity.GridInterconnectionQueueGw"),
    ("CHECKPOINT_WRITE_BW_GBS", "Systems.Reliability.Recovery.checkpoint_write_bw_gbs"),
    ("OVERHEAD_FAILURE_RECOVERY", "Literature.Overheads.FailureRecovery"),
    ("OVERHEAD_PIPELINE_BUBBLE", "Literature.Overheads.PipelineBubble"),
    ("CHINCHILLA_TOKENS_PER_PARAM", "Literature.Chinchilla.TokensPerParam"),
    ("CHINCHILLA_COMPUTE_CONSTANT", "Literature.Chinchilla.ComputeConstant"),
    ("CRITICAL_BATCH_SIZE_DEFAULT", "Literature.BatchSize.Default"),
    ("CRITICAL_BATCH_SIZE_BERT", "Literature.BatchSize.Bert"),
    ("CRITICAL_BATCH_SIZE_GPT3", "Literature.BatchSize.Gpt3"),
    ("MEMORY_BIT_ERROR_RATE_PER_BIT", "Monitoring.MemoryBitErrorRatePerBit"),
    ("TARGET_CLUSTER_UTILIZATION", "Systems.Orchestration.target_cluster_utilization"),
    ("AVERAGE_RESEARCHER_JOB_DAYS", "Systems.Orchestration.average_researcher_job_days"),
    ("REFERENCE_MFU_SUSTAINED", "calibration.REFERENCE_MFU_SUSTAINED"),
    ("DEFAULT_SCALING_EFFICIENCY", "calibration.DEFAULT_SCALING_EFFICIENCY"),
    ("DEFAULT_OVERLAP_EFFICIENCY", "calibration.DEFAULT_OVERLAP_EFFICIENCY"),
    ("DEFAULT_COMPUTE_EFFICIENCY", "calibration.DEFAULT_COMPUTE_EFFICIENCY"),
    ("LEAD_TIME_SUBSTATION_MONTHS", "Infrastructure.Capacity.SubstationLeadTimeMonths"),
    ("TOR_SWITCH_MTTF_HOURS", "Systems.Reliability.TorSwitch.mttf_hours"),
    ("MFU_INFERENCE_BATCH1", "Literature.Training.MfuInferenceBatch1"),
    ("MFU_INFERENCE_BATCHED", "Literature.Training.MfuInferenceBatched"),
    ("SCALING_EFF_1024GPU", "Literature.Scaling.Eff1024Gpu"),
    ("SCALING_EFF_8192GPU", "Literature.Scaling.Eff8192Gpu"),
    ("SCALING_EFF_256GPU", "Literature.Scaling.Eff256Gpu"),
    ("SCALING_EFF_32GPU", "Literature.Scaling.Eff32Gpu"),
    ("OVERHEAD_MAINTENANCE", "Literature.Overheads.Maintenance"),
    ("OVERHEAD_CHECKPOINT", "Literature.Overheads.Checkpoint"),
    ("LEAD_TIME_GPU_MONTHS", "Infrastructure.Capacity.GpuLeadTimeMonths"),
    ("HEARTBEAT_TIMEOUT_S", "Systems.Reliability.Recovery.heartbeat_timeout_s"),
    ("RESCHEDULE_TIME_S", "Systems.Reliability.Recovery.reschedule_time_s"),
    ("PSI_CRITICAL_THRESHOLD", "Monitoring.PsiCriticalThreshold"),
    ("PSI_REVIEW_THRESHOLD", "Monitoring.PsiReviewThreshold"),
    ("PSI_WARN_THRESHOLD", "Monitoring.PsiWarnThreshold"),
    ("ALLREDUCE_FACTOR", "Literature.Communication.RingAllreduceFactor"),
    ("MFU_TRAINING_HIGH", "Literature.Training.MfuHigh"),
    ("MFU_TRAINING_LOW", "Literature.Training.MfuLow"),
    ("P_SDC_PER_GPU_HR", "Systems.Reliability.SdcRatePerGpuHr"),
    ("GPU_MTTF_HOURS", "Systems.Reliability.Gpu.mttf_hours"),
    ("NIC_MTTF_HOURS", "Systems.Reliability.Nic.mttf_hours"),
    ("PSU_MTTF_HOURS", "Systems.Reliability.Psu.mttf_hours"),
    ("CABLE_MTTF_HOURS", "Systems.Reliability.Cable.mttf_hours"),
    ("HBM_MTTF_HOURS", "Systems.Reliability.Hbm.mttf_hours"),
    ("QUEUE_DISCIPLINE", "Systems.Orchestration.queue_discipline"),
    ("KS_TEST_COEFFICIENT", "Monitoring.KsTestCoefficient"),
]

_IMPORT_DEFAULTS = re.compile(
    r"^from mlsysim\.core import defaults\s*$",
    re.MULTILINE,
)
_IMPORT_DEFAULTS_MODULE = re.compile(
    r"^from mlsysim\.core\.defaults import (\w+)\s*$",
    re.MULTILINE,
)
_IMPORT_DEFAULTS_SYMBOL = re.compile(
    r"^from mlsysim\.core\.defaults import ([A-Z][A-Z0-9_]+)\s*$",
    re.MULTILINE,
)


def _symbol_replacements(text: str) -> str:
    """Returns a mapping of legacy symbols to their new registry locations."""
    for old, new in _SYMBOL_MAP:
        text = text.replace(f"defaults.{old}", new)
    return text


def _import_remap(symbol: str) -> str:
    """Remaps legacy import statements to their new registry paths."""
    for old, new in _SYMBOL_MAP:
        if old == symbol:
            return new
    return f"UNKNOWN.{symbol}"


def migrate_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    text = _symbol_replacements(original)

    def _replace_symbol_import(m: re.Match[str]) -> str:
        """Performs AST replacement for migrated symbols."""
        sym = m.group(1)
        target = _import_remap(sym)
        if target.startswith("calibration."):
            return (
                f"from mlsysim.engine import calibration\n"
                f"{sym} = calibration.{target.split('.', 1)[1]}"
            )
        if target.startswith("Monitoring."):
            return f"from mlsysim.ops.monitoring import Monitoring\n{sym} = Monitoring.{target.split('.', 1)[1]}"
        return f"{sym} = {target}"

    text = _IMPORT_DEFAULTS_SYMBOL.sub(_replace_symbol_import, text)
    text = _IMPORT_DEFAULTS.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    roots = [
        _REPO / "book/quarto/contents",
        _REPO / "book/labs",
        _REPO / "labs",
        _REPO / "mlsysim/tests",
    ]
    changed = 0
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix not in (".qmd", ".py", ".md"):
                continue
            if "migrate_defaults" in path.name:
                continue
            if migrate_file(path):
                print(path.relative_to(_REPO))
                changed += 1
    print(f"Updated {changed} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
