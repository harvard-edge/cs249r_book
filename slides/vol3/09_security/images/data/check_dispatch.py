#!/usr/bin/env python3
"""Run the chapter's dispatch check (Agent Security, Sec. 9.6.4, listing 2) on
four proposals from the repair task, and print what the monitor returns.

The functions Label, meet, join, reject, and check_dispatch are copied from
the listing unchanged. Everything below the marked line is the smallest
harness that lets them run: a registry of two tools, host records for the
README, the private fixture, and a fixture handle, and one task grant.
The slide "The chapter's monitor, run on four proposals" shows this output.

    python3 check_dispatch.py > check_dispatch_output.txt
"""

from dataclasses import dataclass
from functools import reduce

PUBLIC = None  # readers marker: no recipient restriction

@dataclass(frozen=True)
class Label:
    sources: frozenset  # integrity provenance: possible influences
    readers: frozenset | None  # confidentiality: permitted recipients

def meet(a, b):  # intersection, with PUBLIC as its identity
    return b if a is PUBLIC else a if b is PUBLIC else a & b

def join(labels):  # conservative derivation rule
    return Label(frozenset().union(*(l.sources for l in labels)),
                 reduce(meet, (l.readers for l in labels)))

def reject(rule):  # names the rule, never the blocked content
    return {"status": "rejected_before_dispatch", "rule": rule}

def check_dispatch(proposal, registry, records, task):
    tool = registry.get(proposal.tool)
    if tool is None:
        return reject("unregistered_tool")
    recipient = tool.resolve_recipient(proposal.args)  # host-resolved
    if recipient is None:
        return reject("unresolved_recipient")
    if not task.grant.permits(tool.name, recipient):
        return reject("outside_task_grant")
    labels = [proposal.invocation_label]  # all selected model inputs
    for name, value in proposal.args.items():
        label = records.label_of(task.id, value)  # host metadata only
        if label is None and proposal.model_wrote(name):
            label = proposal.invocation_label  # inherits all its inputs
        if label is None:
            return reject("missing_provenance")
        if tool.is_handle(name):
            obj = records.resolve_handle(task.id, value)
            if obj is None or tool.name not in obj.permitted_ops:
                return reject("handle_not_usable_here")
            label = join([label, obj.label])  # dereference-time labels
        labels.append(label)
    x = join(labels)
    if x.readers is not PUBLIC and recipient not in x.readers:
        return reject("release: recipient not permitted")
    if not tool.accepts_influence(task, x.sources):
        return reject("evidence: source not accepted for this use")
    return {"status": "admitted", "recipient": recipient, "label": x}

# ---------------------------------------------------------------- harness
README = Label(frozenset({"readme_7"}), PUBLIC)
FIXTURE = Label(frozenset({"fixture"}), frozenset({"local_test_runner"}))
USER = Label(frozenset({"user"}), PUBLIC)


@dataclass
class Obj:
    label: Label
    permitted_ops: frozenset


class Records:
    def __init__(self):
        self.handles = {"h_fixture": Obj(FIXTURE, frozenset({"run_tests", "package_search"}))}

    def label_of(self, task_id, value):
        return Label(frozenset(), PUBLIC) if value in self.handles else None

    def resolve_handle(self, task_id, value):
        return self.handles.get(value)


class Tool:
    def __init__(self, name, recipient, handle_args=(), accepted=None):
        self.name, self._r, self._h, self._acc = name, recipient, set(handle_args), accepted

    def resolve_recipient(self, args):
        return self._r

    def is_handle(self, name):
        return name in self._h

    def accepts_influence(self, task, sources):
        return self._acc is None or sources <= self._acc


class Grant:
    def permits(self, tool, recipient):
        return (tool, recipient) in {("package_search", "search_endpoint"),
                                     ("run_tests", "local_test_runner"),
                                     ("record_test_pass", "acceptance_record")}


class Task:
    id, grant = "repair_42", Grant()


@dataclass
class Proposal:
    tool: str
    args: dict
    invocation_label: Label

    def model_wrote(self, name):
        return True


REGISTRY = {
    "package_search": Tool("package_search", "search_endpoint", handle_args=("q_handle",)),
    "run_tests": Tool("run_tests", "local_test_runner", handle_args=("fixture",)),
    "record_test_pass": Tool("record_test_pass", "acceptance_record",
                             accepted=frozenset({"protected_runner"})),
}

saw_readme = join([USER, README])
saw_both = join([USER, README, FIXTURE])
CASES = [
    ("1 search, planner saw README only",
     Proposal("package_search", {"query": "connection pool timeout documentation"}, saw_readme)),
    ("2 search, planner also saw the fixture",
     Proposal("package_search", {"query": "dependency diagnosis ..."}, saw_both)),
    ("3 search with a fixture handle",
     Proposal("package_search", {"q_handle": "h_fixture"}, saw_readme)),
    ("4 record README claim as a test pass",
     Proposal("record_test_pass", {"claim": "claimed_tests_passed: true"}, saw_readme)),
]

if __name__ == "__main__":
    for title, p in CASES:
        r = check_dispatch(p, REGISTRY, Records(), Task())
        print(f"# {title}")
        print(f"{p.tool}({', '.join(f'{k}={v!r}' for k, v in p.args.items())})")
        print({k: v for k, v in r.items() if k != "label"})
