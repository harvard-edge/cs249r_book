import pytest

from mlsysim import Infrastructure


def test_facility_cooling_pue_profiles():
    assert Infrastructure.FacilityCooling.Legacy.pue == pytest.approx(1.58)
    assert Infrastructure.FacilityCooling.StateOfArt.pue == pytest.approx(1.10)
    assert Infrastructure.FacilityCooling.SimpleAir.pue == pytest.approx(1.50)
