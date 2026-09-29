import pytest

from mlsysim import Infrastructure


def test_facility_cooling_pue_profiles():
    assert Infrastructure.FacilityCooling.Legacy.pue == pytest.approx(1.58)
    assert Infrastructure.FacilityCooling.StateOfArt.pue == pytest.approx(1.10)
    assert Infrastructure.FacilityCooling.SimpleAir.pue == pytest.approx(1.50)
    # Uptime Institute 2022 Global Data Center Survey: average annual PUE 1.55.
    assert Infrastructure.FacilityCooling.Typical.pue == pytest.approx(1.55)
    # The duplicate survey-average anchor was folded into Typical (hard delete).
    assert not hasattr(Infrastructure.FacilityCooling, "UptimeSurvey2022Average")


def test_cloud_gpu_price_anchors():
    assert Infrastructure.Pricing.Cloud.LowCostGpuPerHour.rate.magnitude == pytest.approx(0.50)
    assert Infrastructure.Pricing.Cloud.A10GInferencePerHour.rate.magnitude == pytest.approx(0.75)
