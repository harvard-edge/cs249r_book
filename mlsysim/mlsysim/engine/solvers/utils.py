from __future__ import annotations

from ...systems.types import NetworkFabric, Node


def _intra_node_latency(node: Node):
    """Resolve NVLink latency from the accelerator, falling back to tech defaults."""
    nvlink = node.accelerator.nvlink
    if nvlink and nvlink.latency is not None:
        return nvlink.latency
    from ...hardware.tech import Tech
    return Tech.Interconnect.NVLink.latency


def _inter_node_latency(fabric: NetworkFabric):
    """Resolve fabric latency, falling back to the reference NDR fabric default."""
    if fabric.latency is not None:
        return fabric.latency
    from ...systems.registry import Systems
    return Systems.Fabrics.InfiniBand_NDR.latency


__all__ = ["_intra_node_latency", "_inter_node_latency"]
