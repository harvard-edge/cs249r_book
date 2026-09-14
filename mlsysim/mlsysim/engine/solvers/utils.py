from __future__ import annotations

from ...systems.types import NetworkFabric, Node


def _intra_node_latency(node: Node):
    """Resolve the per-hop latency for intra-node (NVLink) communication.

    Implements the instance -> tech-class fallback: prefer the latency on the
    accelerator's own NVLink spec (instance data); when the spec is absent or
    carries no latency, fall back to the NVLink technology-class default in
    ``Hardware.Tech`` (a generation-level constant). Used by the distributed
    solvers as the alpha term for TP/intra-node collectives.

    Parameters
    ----------
    node : Node
        The node whose accelerator interconnect is being priced.

    Returns
    -------
    Quantity
        Per-message latency (time units, typically microseconds).
    """
    nvlink = node.accelerator.nvlink
    if nvlink and nvlink.latency is not None:
        return nvlink.latency
    from ...hardware.tech import Tech
    return Tech.Interconnect.NVLink.latency


def _inter_node_latency(fabric: NetworkFabric):
    """Resolve the per-hop latency for inter-node (fabric) communication.

    Prefer the latency on the fabric spec itself; when unset, fall back to
    the reference InfiniBand NDR fabric in the Systems registry (the
    package's canonical datacenter fabric). Used by the distributed solvers
    as the alpha term for DP/EP cross-node collectives.

    Parameters
    ----------
    fabric : NetworkFabric
        The fabric whose latency is being priced.

    Returns
    -------
    Quantity
        Per-message latency (time units, typically microseconds).
    """
    if fabric.latency is not None:
        return fabric.latency
    from ...systems.registry import Systems
    return Systems.Fabrics.InfiniBand_NDR.latency


__all__ = ["_intra_node_latency", "_inter_node_latency"]
