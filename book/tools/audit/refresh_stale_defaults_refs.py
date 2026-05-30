#!/usr/bin/env python3
"""Rewrite stale ``mlsysim.core.defaults.*`` paths in audit YAML and tooling."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]

# Longest keys first.
_REPLACEMENTS: list[tuple[str, str]] = [
    ("mlsysim.core.defaults.CHINCHILLA_DATA_STARVED_TOKENS_PER_PARAM", "Literature.Chinchilla.TokensPerParam"),
    ("mlsysim.core.defaults.CHINCHILLA_TOKENS_PER_PARAM", "Literature.Chinchilla.TokensPerParam"),
    ("mlsysim.core.defaults.CHINCHILLA_COMPUTE_CONSTANT", "Literature.Chinchilla.ComputeConstant"),
    ("mlsysim.core.defaults.REFERENCE_MFU_SUSTAINED", "mlsysim.engine.calibration.REFERENCE_MFU_SUSTAINED"),
    ("mlsysim.core.defaults.MFU_TRAINING_LOW/HIGH", "Literature.Training.MfuLow, Literature.Training.MfuHigh"),
    ("mlsysim.core.defaults.MFU_TRAINING_HIGH", "Literature.Training.MfuHigh"),
    ("mlsysim.core.defaults.MFU_TRAINING_LOW", "Literature.Training.MfuLow"),
    ("mlsysim.core.defaults.GPU_MTTF_HOURS", "Systems.Reliability.Gpu.mttf_hours"),
    ("add mlsysim.core.defaults.PUE_STATE_OF_ART", "add Infrastructure.PUE_STATE_OF_ART"),
    ("mlsysim.core.defaults.PUE_STATE_OF_ART", "Infrastructure.PUE_STATE_OF_ART"),
    ("mlsysim.core.defaults.PUE_LIQUID_COOLED", "Infrastructure.PUE_LIQUID_COOLED"),
    ("mlsysim.core.defaults.PUE_BEST_AIR", "Infrastructure.PUE_BEST_AIR"),
    ("mlsysim.core.defaults.PUE_TYPICAL", "Infrastructure.PUE_TYPICAL"),
    ("mlsysim.core.defaults.PUE_LEGACY", "Infrastructure.PUE_LEGACY"),
    ("mlsysim.core.defaults.CARBON_PER_GPU_HR_KG", "Infrastructure.Pricing.Fleet.CarbonPerGpuHr.rate"),
    ("mlsysim.core.defaults.CARBON_US_AVG_GCO2_KWH", "Infrastructure.Grids.US_Avg.carbon_intensity_g_kwh"),
    ("mlsysim.core.defaults.CLOUD_GPU_TRAINING_PER_HOUR", "Infrastructure.Pricing.Cloud.GpuTrainingPerHour.rate"),
    ("mlsysim.core.defaults.CLOUD_GPU_INFERENCE_PER_HOUR", "Infrastructure.Pricing.Cloud.GpuInferencePerHour.rate"),
    ("mlsysim.core.defaults.CLOUD_EGRESS_PER_GB", "Infrastructure.Pricing.Cloud.EgressPerGB.rate"),
    ("mlsysim.core.defaults.CLOUD_ELECTRICITY_PER_KWH", "Infrastructure.Pricing.Cloud.ElectricityPerKwh.rate"),
    ("mlsysim.core.defaults.TPU_V4_PER_HOUR", "Infrastructure.Pricing.Cloud.TpuV4PerHour.rate"),
    ("mlsysim.core.defaults.STORAGE_COST_S3_STD", "Infrastructure.Pricing.Storage.S3StandardPerTbMonth.rate"),
    ("mlsysim.core.defaults.STORAGE_COST_NVME_LOW", "Infrastructure.Pricing.Storage.NvmeLowPerTbMonth.rate"),
    ("mlsysim.core.defaults.STORAGE_COST_NVME_HIGH", "Infrastructure.Pricing.Storage.NvmeHighPerTbMonth.rate"),
    ("mlsysim.core.defaults.RETRIEVAL_COST_GLACIER", "Infrastructure.Pricing.Storage.GlacierRetrievalPerGB.rate"),
    ("mlsysim.core.defaults.KS_TEST_COEFFICIENT", "Ops.Monitoring.KsTestCoefficient"),
    ("mlsysim.core.defaults.INFINIBAND_NDR_BW_GBS", "Systems.Fabrics.InfiniBand_NDR.bandwidth"),
    ("mlsysim.core.defaults.IB_NDR_LATENCY_US", "mlsysim.systems.registry.IB_NDR_LATENCY_US"),
    ("mlsysim.core.defaults.FrameworkOverhead.python_dispatch", "mlsysim.engine.calibration.KERNEL_LAUNCH_LATENCY_US"),
    ("mlsysim.core.defaults.DriftThresholds.psi_significant", "Ops.Monitoring.PsiWarnThreshold"),
    ("mlsysim.core.defaults.KWSCaseStudy", "mlsysim.core.constants  # TODO: KWS case study registry"),
    ("e.g. mlsysim.core.defaults", "Infrastructure.Pricing / Systems.Orchestration"),
    ("defaults.RETRIEVAL_COST_GLACIER", "Infrastructure.Pricing.Storage.GlacierRetrievalPerGB.rate"),
    ("defaults.PUE_LIQUID_COOLED", "Infrastructure.PUE_LIQUID_COOLED"),
    ("defaults.PUE_BEST_AIR", "Infrastructure.PUE_BEST_AIR"),
    ("defaults.PUE_TYPICAL", "Infrastructure.PUE_TYPICAL"),
    ("defaults.PUE_LEGACY", "Infrastructure.PUE_LEGACY"),
    ("defaults.CARBON_US_AVG_GCO2_KWH", "Infrastructure.Grids.US_Avg.carbon_intensity_g_kwh"),
    ("defaults.CARBON_QUEBEC_GCO2_KWH", "Infrastructure.Grids.Quebec.carbon_intensity_g_kwh"),
    ("defaults.CARBON_IOWA_GCO2_KWH", "Infrastructure.Grids.Iowa.carbon_intensity_g_kwh"),
    ("defaults.CARBON_POLAND_GCO2_KWH", "Infrastructure.Grids.Poland.carbon_intensity_g_kwh"),
    ("defaults.CARBON_NORWAY_GCO2_KWH", "Infrastructure.Grids.Norway.carbon_intensity_g_kwh"),
    ("defaults.CARBON_EU_AVG_GCO2_KWH", "Infrastructure.Grids.EU_Avg.carbon_intensity_g_kwh"),
    ("defaults.CARBON_FRANCE_GCO2_KWH", "Infrastructure.Grids.France.carbon_intensity_g_kwh"),
    ("defaults.INFINIBAND_NDR_BW_GBS", "Systems.Fabrics.InfiniBand_NDR.bandwidth"),
    ("defaults.INFINIBAND_HDR_BW_GBS", "Systems.Fabrics.InfiniBand_HDR.bandwidth"),
    ("defaults.INFINIBAND_XDR_BW_GBS", "Systems.Fabrics.InfiniBand_XDR.bandwidth"),
    ("defaults.ETHERNET_400G_BW_GBS", "Systems.Fabrics.Ethernet_400G.bandwidth"),
    ("defaults.ETHERNET_800G_BW_GBS", "Systems.Fabrics.Ethernet_800G.bandwidth"),
    ("defaults.ROCE_100G_BW_GBS", "Systems.Fabrics.RoCE_100G.bandwidth"),
    ("defaults.IB_NDR_LATENCY_US", "mlsysim.systems.registry.IB_NDR_LATENCY_US"),
    ("defaults.IB_HDR_LATENCY_US", "mlsysim.systems.registry.IB_HDR_LATENCY_US"),
    ("defaults.ROCE_LATENCY_US", "mlsysim.systems.registry.ROCE_LATENCY_US"),
    ("defaults.TCP_LATENCY_US", "mlsysim.systems.registry.TCP_LATENCY_US"),
    ("defaults.TPU_POD_CHIPS", "Systems.Pods.TPUv4_4096.chips"),
    ("defaults.TPU_POD_MEM", "Systems.Pods.TPUv4_4096.memory"),
    ("defaults.TPU_POD_POWER", "Systems.Pods.TPUv4_4096.power"),
]

_GLOB_DIRS = [
    _REPO / "book/tools/audits/mlsysim_constants",
    _REPO / "book/tools/audit",
]


def _refresh_text(text: str) -> tuple[str, int]:
    count = 0
    for old, new in _REPLACEMENTS:
        if old in text:
            n = text.count(old)
            text = text.replace(old, new)
            count += n
    return text, count


def main() -> None:
    total = 0
    files = 0
    for base in _GLOB_DIRS:
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix not in (".yaml", ".yml", ".py", ".md", ".tex", ".qmd"):
                continue
            if path.name == "refresh_stale_defaults_refs.py":
                continue
            original = path.read_text(encoding="utf-8")
            updated, n = _refresh_text(original)
            if n:
                path.write_text(updated, encoding="utf-8")
                print(f"{path.relative_to(_REPO)}: {n}")
                total += n
                files += 1
    print(f"Updated {files} file(s), {total} replacement(s).")


if __name__ == "__main__":
    main()
