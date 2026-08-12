import pytest

from mlsysim import Datasets
from mlsysim.core.units import byte, second


def test_mswc_audio_encoding_is_registry_backed():
    mswc = Datasets.MSWC

    assert mswc.sample_duration.to(second).magnitude == pytest.approx(1.0)
    assert mswc.sample_rate.to(1 / second).magnitude == pytest.approx(16_000.0)
    assert mswc.sample_width.to(byte).magnitude == pytest.approx(2.0)
