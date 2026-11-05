"""Machine serialization preserves fractional unit exponents across Pint versions."""

import pytest
from pydantic import BaseModel

from mlsysim import Q_
from mlsysim.core.types import Quantity, serialize_quantity, validate_quantity
from mlsysim.labs.experiments import v2_02_experiments, v2_03_experiments, v2_10_experiments


class QuantityRecord(BaseModel):
    amount: Quantity


@pytest.fixture(
    params=[
        "0.014 degree / hertz ** 0.5 / second",
        "0.00016 g_0 / hertz ** 0.5",
        "9.81 meter / second ** 2",
        "1.2e-8 micrometer / second ** 1.5",
        "3.5 GiB",
        "2.5 USD / kilowatt_hour",
        "1.2345678901234567 meter ** 0.3333333333333333 / second ** 1.5",
        "8 bit",
        "1 byte",
        "120 TFLOP",
        "1000 param",
        "1.25 USD",
        "0.75 EUR",
        "0.5 dimensionless",
    ]
)
def quantity(request):
    return Q_(request.param)


def assert_physical_roundtrip(original, restored):
    # Matching the magnitude alone would miss the fractional-Hz parse failure.
    assert restored.dimensionality == original.dimensionality
    assert restored.units == original.units
    assert restored.to(original.units).magnitude == original.magnitude


def test_quantity_serializer_roundtrip_preserves_unit_exponents(quantity):
    human_display = format(quantity, "~P")
    payload = serialize_quantity(quantity)
    assert format(quantity, "~P") == human_display
    assert payload.isascii()
    assert_physical_roundtrip(quantity, validate_quantity(payload))


def test_public_model_dump_roundtrip_preserves_unit_exponents(quantity):
    record = QuantityRecord(amount=quantity)
    restored = QuantityRecord.model_validate(record.model_dump())
    assert_physical_roundtrip(quantity, restored.amount)
    restored_json = QuantityRecord.model_validate_json(record.model_dump_json())
    assert_physical_roundtrip(quantity, restored_json.amount)


@pytest.mark.parametrize("module", [v2_02_experiments, v2_03_experiments, v2_10_experiments])
def test_lab_quantity_record_roundtrip_preserves_unit_exponents(quantity, module):
    payload = module.serialize_quantity(quantity)
    assert payload["unit"].isascii()
    assert_physical_roundtrip(quantity, module.deserialize_quantity(payload))
