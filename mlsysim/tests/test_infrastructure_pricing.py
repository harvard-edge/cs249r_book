from __future__ import annotations

import pytest

from mlsysim import Infrastructure
from mlsysim.core.units import GB, USD, ureg


def test_storage_pricing_round_number_anchors():
    s3_low = Infrastructure.Pricing.Storage.S3StandardLowPerTbMonth.rate
    glacier = Infrastructure.Pricing.Storage.GlacierStandardPerTbMonth.rate

    assert s3_low.to(USD / (GB * ureg.month)).magnitude == pytest.approx(0.02)
    assert glacier.to(USD / (GB * ureg.month)).magnitude == pytest.approx(0.004)


def test_monitoring_pricing_anchors():
    pricing = Infrastructure.Pricing.Monitoring

    assert pricing.IngestionPerMillionDatapoints.rate.to(USD).magnitude == pytest.approx(0.30)
    assert pricing.StoragePerGbMonth.rate.to(USD / (GB * ureg.month)).magnitude == pytest.approx(1.00)
    assert pricing.QueryPerRequest.rate.to(USD).magnitude == pytest.approx(0.02)
