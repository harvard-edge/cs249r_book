"""Guards for the 2026-10-15 round-2 registry additions (B200 die-to-die link,
effective collective-launch alpha ranges)."""

import pytest

from mlsysim import Hardware, Systems
from mlsysim.core.units import ureg

SF = Systems.SwitchFabric
RANGES = ["Nvlink", "PcieGen5", "Ndr", "Hdr", "Roce"]


def test_b200_die_to_die_bandwidth_is_vendor_figure():
    bw = Hardware.Cloud.B200.die_to_die_bandwidth
    assert bw.to(ureg.TB / ureg.second).magnitude == pytest.approx(10.0)


def test_monolithic_parts_have_no_die_to_die_link():
    assert Hardware.Cloud.H100.die_to_die_bandwidth is None


def test_b200_die_to_die_exceeds_package_nvlink():
    # The die-to-die link must be faster than the package's off-package NVLink
    # total, or the two dies could not present as one logical GPU.
    b200 = Hardware.Cloud.B200
    assert b200.die_to_die_bandwidth > b200.nvlink.bandwidth


@pytest.mark.parametrize("key", RANGES)
def test_effective_alpha_range_is_ordered(key):
    lo = getattr(SF, f"EffectiveAlpha{key}Low")
    hi = getattr(SF, f"EffectiveAlpha{key}High")
    assert 0 * ureg.microsecond < lo < hi


def test_point_anchors_sit_inside_their_ranges():
    assert SF.EffectiveAlphaNdrLow <= SF.AlphaNdr <= SF.EffectiveAlphaNdrHigh
    assert SF.EffectiveAlphaRoceLow <= SF.AlphaRoce <= SF.EffectiveAlphaRoceHigh
