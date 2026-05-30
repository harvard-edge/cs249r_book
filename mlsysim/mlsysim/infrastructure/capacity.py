"""Datacenter and grid build-out lead times."""

from ..core.provenance import sourced
from ..core.registry import Registry
from ..core import provenance_catalog as pc


class Capacity(Registry):
    GpuLeadTimeMonths = sourced(6, pc.BOOK_CAPACITY_LEAD_TIMES, name="GPU lead time (months)", description="Typical GPU procurement lead time.")
    SubstationLeadTimeMonths = sourced(24, pc.BOOK_CAPACITY_LEAD_TIMES, name="Substation lead time (months)", description="Typical substation construction lead time.")
    GridInterconnectionQueueGw = sourced(2000, pc.BOOK_CAPACITY_LEAD_TIMES, name="US grid interconnection queue (GW)", description="Illustrative US interconnection queue size.")
