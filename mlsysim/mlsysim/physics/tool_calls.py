"""
Tool calling: the equations behind a mediated call, its retry, and its catalog.

Pure arithmetic (added 2026-10-20). A call whose request or reply may be lost
times out with one probability and, under one blind retry, executes twice
with another; a model that polls a pending job makes one status call per
interval before completion; a tool catalog staged on every call costs its
descriptors, and selection pays once the descriptors it leaves out exceed its
own overhead; a shared protocol turns pairwise adapters into one per party.
The solvers in engine/solvers/tool_calls.py compose these.
"""

import math as _math


def calc_timeout_probability(request_loss: float, reply_loss: float) -> float:
    """
    Probability that one attempt times out when request and reply are lost independently.

        Pr(timeout) = 1 - (1 - q)(1 - r)
    """
    return 1 - (1 - request_loss) * (1 - reply_loss)


def calc_blind_retry_duplicate_probability(request_loss: float, reply_loss: float) -> float:
    """
    Probability that one blind retry after a timeout executes the effect a second time.

    The original request is delivered, its reply is lost, and the retry is
    delivered: Pr(dup) = (1 - q) r (1 - q). Losing the request alone leaves a
    harmless timeout; lost messages are taken never to arrive later.
    """
    return (1 - request_loss) * reply_loss * (1 - request_loss)


def calc_pending_polls(job_duration, poll_interval) -> int:
    """
    Status checks that find a job still pending.

    Checks start one interval after dispatch and stop at completion; a check
    that lands exactly at completion already sees the result, so the pending
    checks are ceil(T / dt) - 1.
    """
    ratio = (job_duration / poll_interval).to("").magnitude
    return _math.ceil(ratio) - 1


def calc_selection_break_even(selection_overhead: float, descriptor_tokens: float) -> float:
    """
    Unused descriptors selection must leave out before it saves context.

        (N_reg - N_sel) s_desc > s_over   =>   break-even at s_over / s_desc
    """
    return selection_overhead / descriptor_tokens


def calc_integration_adapters(hosts: int, services: int):
    """
    Adapters needed to connect every host to every service: pairwise, and through one protocol.

    Pairwise integration needs one adapter per pair (N x M); a shared protocol
    needs one client per host and one server per service (N + M).
    """
    return hosts * services, hosts + services
