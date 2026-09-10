"""Software runtime and framework overhead assumptions."""

from ..core import provenance_catalog as pc
from ..core.provenance import sourced, sourced_qty
from ..core.registry import Registry
from ..core.units import ureg


class RuntimeOverheads(Registry):
    """Reusable software-runtime latency anchors for framework overhead examples."""

    PythonDispatch = sourced_qty(
        10 * ureg.microsecond,
        pc.FRAMEWORK_RUNTIME_OVERHEAD_REFERENCE,
        name="Python dispatch overhead",
        description="Reference Python-to-runtime dispatch overhead for tiny eager operations.",
    )
    KernelLaunch = sourced_qty(
        5 * ureg.microsecond,
        pc.FRAMEWORK_RUNTIME_OVERHEAD_REFERENCE,
        name="Kernel launch overhead",
        description="Reference accelerator kernel-launch overhead for tiny operations.",
    )
    TinyMemoryAccess = sourced_qty(
        1 * ureg.microsecond,
        pc.FRAMEWORK_RUNTIME_OVERHEAD_REFERENCE,
        name="Tiny memory access overhead",
        description="Reference memory-access overhead term used in small-operation fusion models.",
    )


class MemoryProtection(Registry):
    """Reusable memory-protection policy overheads."""

    EccParityOverhead = sourced(
        0.125,
        pc.MEMORY_PROTECTION_OVERHEADS,
        name="ECC parity overhead",
        description="Reference SECDED-style ECC parity fraction for protected memory.",
    )
    NoEccOverhead = sourced(
        0.0,
        pc.MEMORY_PROTECTION_OVERHEADS,
        name="No-ECC bandwidth overhead",
        description="Reference overhead for memory configurations without ECC protection.",
    )
