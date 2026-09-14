"""Characterization tests for MLSysIM Pint registry and unit aliases."""

from __future__ import annotations

import pytest

from mlsysim.core.units import (
    Bparam,
    GB,
    GFLOPs,
    GW,
    GiB,
    KiB,
    Gbps,
    Kparam,
    L,
    MB,
    MiB,
    Mparam,
    MJ,
    MS,
    NS,
    PFLOP,
    Q_,
    TB,
    TFLOP,
    TOPS,
    Tparam,
    US,
    USD,
    ZFLOP,
    byte,
    count,
    hour,
    joule,
    kilogram,
    kilowatt,
    kilojoule,
    km,
    kWh,
    kJ,
    liter,
    gigawatt,
    megawatt,
    metric_ton,
    milliwatt,
    minute,
    mJ,
    MW,
    MWh,
    microwatt,
    microjoule,
    param,
    pJ,
    second,
    uJ,
    ureg,
    uW,
    watt,
    Wh,
)


def test_decimal_data_scales():
    assert Q_("1 GB").to("byte").magnitude == pytest.approx(1e9)
    assert Q_("1 TB").to("byte").magnitude == pytest.approx(1e12)


def test_binary_data_scales():
    assert Q_("1 GiB").to("byte").magnitude == 1073741824


def test_flop_rate_dimensions():
    assert Q_("1 TFLOP/s").to(TFLOP / second).magnitude == pytest.approx(1)
    assert Q_("1 PFLOP/s").to(PFLOP / second).magnitude == pytest.approx(1)
    assert Q_("1 ZFLOP/s").to(ZFLOP / second).magnitude == pytest.approx(1)
    assert Q_("1 TOPS").to(TOPS).magnitude == pytest.approx(1)


def test_gbps_to_gigabytes_per_second():
    assert Q_("1 Gbps").to("GB/s").magnitude == pytest.approx(0.125)


def test_energy_aliases():
    assert Q_("1 kWh").to("J").magnitude == pytest.approx(3.6e6)
    assert Q_("1 MWh").to("kWh").magnitude == pytest.approx(1000)
    assert Wh == ureg.watt_hour


def test_param_scales():
    assert Q_("1 Bparam").to("param").magnitude == pytest.approx(1e9)


def test_legacy_time_aliases_match_si():
    assert Q_(1, MS).to("second").magnitude == pytest.approx(1e-3)
    assert Q_(1, US).to("second").magnitude == pytest.approx(1e-6)
    assert Q_(1, NS).to("second").magnitude == pytest.approx(1e-9)


def test_exported_aliases_match_registry():
    assert kJ == ureg.kilojoule
    assert MJ == ureg.megajoule
    assert mJ == ureg.millijoule
    assert uJ == ureg.microjoule
    assert kilojoule == ureg.kilojoule
    assert microjoule == ureg.microjoule
    assert GW == ureg.gigawatt
    assert MW == ureg.megawatt
    assert uW == ureg.microwatt
    assert gigawatt == ureg.gigawatt
    assert kilowatt == ureg.kilowatt
    assert microwatt == ureg.microwatt
    assert milliwatt == ureg.milliwatt
    assert kWh == ureg.kilowatt_hour
    assert kilogram == ureg.kilogram
    assert metric_ton == ureg.metric_ton
    assert km == ureg.kilometer
    assert L == ureg.liter
    assert liter == ureg.liter
    assert pJ == ureg.picojoule
    assert minute == ureg.minute


def test_mlsysim_units_file_loaded():
    """Custom units from mlsysim_units.txt must parse like inline defines."""
    assert Q_("2.5 TFLOP").to(TFLOP).magnitude == pytest.approx(2.5)
    assert Q_("3 Bparam").to(Bparam).magnitude == pytest.approx(3)


def test_registry_yaml_strings_still_parse():
    assert Q_("80 GiB").to(GiB).magnitude == pytest.approx(80)
    assert Q_("3.35 TB/s").to(TB / second).magnitude == pytest.approx(3.35)


def test_gpt3_training_energy_is_quantity():
    from mlsysim import Models

    energy = Models.Language.GPT3.training_energy_mwh
    assert energy.to(MWh).magnitude == pytest.approx(1287, rel=1e-3)


def test_energy_anchor_household_year_is_quantity():
    from mlsysim import ReferenceStats

    household_year = ReferenceStats.EnergyAnchors.USHouseholdAnnualElectricity
    assert household_year.to(MWh).magnitude == pytest.approx(10.7)
    assert household_year.provenance.ref


def test_training_and_recommendation_model_anchors_are_quantities():
    from mlsysim import Models

    assert Models.Language.GPT2.training_dataset_size.to(GB).magnitude == pytest.approx(40)
    assert Models.Recommendation.DLRM.parameter_range_min.to(Bparam).magnitude == pytest.approx(100)
    assert Models.Recommendation.DLRM.parameter_range_max.to(Tparam).magnitude == pytest.approx(10)


def test_emissions_transatlantic_flight_co2_anchor():
    from mlsysim import ReferenceStats
    from mlsysim.core.provenance import scalar_value

    anchor = ReferenceStats.EmissionsAnchors.TransatlanticRoundTripCo2Kg
    assert scalar_value(anchor) == pytest.approx(1000)
    assert anchor.provenance.ref


def test_clinical_imaging_photo_size_anchor():
    from mlsysim import ReferenceStats

    anchor = ReferenceStats.ClinicalImaging.RetinalPhotoSize
    assert anchor.to(MB).magnitude == pytest.approx(5.0)
    assert anchor.provenance.ref


def test_oura_sleep_study_anchors():
    from mlsysim import ReferenceStats
    from mlsysim.core.provenance import scalar_value

    study = ReferenceStats.OuraSleepStudy

    assert scalar_value(study.Participants) == pytest.approx(106)
    assert scalar_value(study.RecordingNights) == pytest.approx(440)
    assert study.RecordingHours.to(hour).magnitude == pytest.approx(3444)
    assert scalar_value(study.CrossValidationFolds) == pytest.approx(5)
    assert scalar_value(study.AccelOnlyAccuracy) == pytest.approx(0.57)
    assert scalar_value(study.EnhancedAccuracy) == pytest.approx(0.79)
    assert scalar_value(study.PsgScorerAgreementLow) == pytest.approx(0.82)
    assert scalar_value(study.PsgScorerAgreementHigh) == pytest.approx(0.83)
    assert study.EnhancedAccuracy.provenance.ref


def test_storage_training_corpus_anchor():
    from mlsysim import ReferenceStats
    from mlsysim.core.units import byte, day, minute, param

    corpus = ReferenceStats.StorageTrainingCorpus

    assert corpus.TrainingTokens.to(count).magnitude == pytest.approx(1.5e12)
    assert corpus.CompressedSource.to(TB).magnitude == pytest.approx(3.0)
    assert corpus.TokenIdBytes.to(byte).magnitude == pytest.approx(4.0)
    assert corpus.TokenizedText.to(TB).magnitude == pytest.approx(6.0)
    assert corpus.TrainingWindow.to(day).magnitude == pytest.approx(30.0)
    assert corpus.CheckpointInterval.to(minute).magnitude == pytest.approx(10.0)
    assert corpus.CheckpointBytesPerParameter.to(byte / param).magnitude == pytest.approx(14.0)
    assert corpus.CompressedSource.provenance.ref


def test_model_loading_anchors():
    from mlsysim import ReferenceStats

    loading = ReferenceStats.ModelLoading

    assert loading.StableDiffusionV15CheckpointSize.to(GB).magnitude == pytest.approx(5.0)
    assert loading.StableDiffusionV15PickleLoadTime.to(second).magnitude == pytest.approx(15.0)
    assert loading.StableDiffusionV15SafetensorsLoadTime.to(second).magnitude == pytest.approx(1.5)
    assert loading.PcieSwapReferenceModelSize.to(GB).magnitude == pytest.approx(10.0)
    assert loading.StableDiffusionV15CheckpointSize.provenance.ref


def test_serving_profile_anchors():
    from mlsysim import ReferenceStats

    profile = ReferenceStats.ServingProfiles

    assert profile.H100VendorMemoryBudget.to(GiB).magnitude == pytest.approx(80.0)
    assert float(profile.PrecisionDividendTensorParallelDegree) == pytest.approx(8.0)
    assert float(profile.PrecisionDividendContextLengthTokens) == pytest.approx(4096.0)
    assert not hasattr(profile, "PrecisionDividendGpuMemoryBudget")
    assert float(profile.PrecisionDividendBaselinePolicyBatchLimit) == pytest.approx(4.0)
    assert float(profile.PrecisionDividendOptimizedPolicyBatchLimit) == pytest.approx(32.0)
    assert float(profile.PrecisionDividendSpeculationBatchThreshold) == pytest.approx(16.0)
    assert float(profile.HeterogeneousRoutingH100Servers) == pytest.approx(10.0)
    assert float(profile.HeterogeneousRoutingA100Servers) == pytest.approx(20.0)
    assert float(profile.HeterogeneousRoutingH100CapacityQps) == pytest.approx(1000.0)
    assert float(profile.HeterogeneousRoutingA100CapacityQps) == pytest.approx(600.0)
    assert float(profile.HeterogeneousRoutingTargetQps) == pytest.approx(15000.0)
    assert profile.PrecisionDividendTensorParallelDegree.provenance.ref


def test_checkpoint_archetype_anchors():
    from mlsysim import ReferenceStats
    from mlsysim.core.units import byte, param

    ckpt = ReferenceStats.CheckpointArchetypes

    assert ckpt.MixedPrecisionOptimizerBytesPerParameter.to(byte / param).magnitude == pytest.approx(12.0)
    assert ckpt.Dense20BTransformerCheckpointSize.to(GB).magnitude == pytest.approx(240.0)
    assert ckpt.EmbeddingHeavyRecommenderCheckpointSize.to(TB).magnitude == pytest.approx(4.0)
    assert ckpt.MediumVisionTransformerCheckpointSize.to(GB).magnitude == pytest.approx(1.2)
    assert ckpt.Dense20BTransformerCheckpointSize.provenance.ref


def test_edge_device_spectrum_anchors():
    from mlsysim import ReferenceStats

    spectrum = ReferenceStats.EdgeDeviceSpectrum

    assert spectrum.TinyRamLow.to(KiB).magnitude == pytest.approx(32.0)
    assert spectrum.MicrocontrollerSram.to(KiB).magnitude == pytest.approx(256.0)
    assert spectrum.FlagshipSmartphoneRamHigh.to(GiB).magnitude == pytest.approx(16.0)
    assert spectrum.CortexMClock.to(ureg.megahertz).magnitude == pytest.approx(48.0)
    assert spectrum.MobileClassClock.to(ureg.gigahertz).magnitude == pytest.approx(3.0)
    assert float(spectrum.TinyCpuThroughputMips) == pytest.approx(10.0)
    assert float(spectrum.MobileCpuThroughputMips) == pytest.approx(100_000.0)
    assert spectrum.SensorPowerLow.to(microwatt).magnitude == pytest.approx(10.0)
    assert spectrum.MicrocontrollerBoardCost.to(USD).magnitude == pytest.approx(10.0)
    assert spectrum.LowEndEdgeRam.to(MB).magnitude == pytest.approx(512.0)
    assert spectrum.LowEndEdgeCompute.to(GFLOPs / second).magnitude == pytest.approx(1.0)
    assert spectrum.FlagshipPhonePowerHigh.to(watt).magnitude == pytest.approx(5.0)
    assert spectrum.IotMicrocontrollerComputeLow.to(TOPS).magnitude == pytest.approx(0.03)
    assert spectrum.TinyRamLow.provenance.ref


def test_platform_threshold_anchors():
    from mlsysim import Platforms

    assert Platforms.Cloud.compute_threshold.to(TFLOP / second).magnitude == pytest.approx(
        1000.0
    )
    assert Platforms.Cloud.bandwidth_threshold.to(GB / second).magnitude == pytest.approx(
        1000.0
    )
    assert Platforms.Edge.compute_threshold.to(PFLOP / second).magnitude == pytest.approx(1.0)
    assert Platforms.Edge.bandwidth_threshold.to(GB / second).magnitude == pytest.approx(270.0)
    assert Platforms.Tiny.compute_threshold.to(TOPS).magnitude == pytest.approx(1.0)
    assert Platforms.Tiny.power_threshold.to(milliwatt).magnitude == pytest.approx(1.0)


def test_mobilenetv2_variant_model_profiles():
    from mlsysim import Models
    from mlsysim.core.units import param

    assert Models.Vision.MobileNetV2.parameters.to(param).magnitude == pytest.approx(3_504_872)
    assert Models.Vision.EfficientNetB0.parameters.to(param).magnitude == pytest.approx(5_300_000)
    assert Models.Vision.MobileNetV2_Alpha0_5.parameters.to(param).magnitude == pytest.approx(1_968_680)
    assert Models.Vision.MobileNetV2_Alpha0_5FeatureExtractor.parameters.to(param).magnitude == pytest.approx(687_680)
    assert Models.Vision.MobileNetV2.metadata.provenance.ref
    assert Models.Vision.EfficientNetB0.metadata.provenance.ref
    assert Models.Vision.MobileNetV2_Alpha0_5.metadata.provenance.ref
