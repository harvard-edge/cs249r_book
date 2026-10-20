#!/usr/bin/env python3
"""Emit real OpenTelemetry spans for the chapter's timing trial (Napkin Math 14.1).

The work itself is stood in for by sleep(): each operation sleeps for the
interval the chapter's Table 2 gives, the two checks run in parallel threads,
and synthesis starts 0.5 s after the slower check ends (the chapter's
unattributed gap). The spans, their identities, and their timestamps are what
the OpenTelemetry Python SDK actually records for that run.

Needs: pip install opentelemetry-sdk==1.27.0 (not in the book venv).
Run from this folder: python3 trace_run.py
Writes trace_run.json (one record per span) and trace_run_console.txt (the SDK's
ConsoleSpanExporter output, from which the slide quotes one span).
"""

import json
import threading
import time
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (ConsoleSpanExporter,
                                            SimpleSpanProcessor)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import \
    InMemorySpanExporter

out = Path(__file__).parent
mem = InMemorySpanExporter()
console_file = open(out / "trace_run_console.txt", "w")
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(mem))
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=console_file)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("repair-agent-timing-trial")

# Napkin Math 14.1, Table 2 (hypothetical intervals, seconds)
RETRIEVE, PROPOSE, TESTS, LINT, GAP, SYNTH = 0.2, 3.0, 6.0, 2.0, 0.5, 2.0


def op(name, seconds, **attrs):
    with tracer.start_as_current_span(name, attributes=attrs):
        time.sleep(seconds)


def check(ctx, name, seconds, tool):
    token = trace.context_api.attach(ctx)
    try:
        op(name, seconds, **{"gen_ai.operation.name": "execute_tool",
                             "gen_ai.tool.name": tool})
    finally:
        trace.context_api.detach(token)


with tracer.start_as_current_span("invoke_agent repair_agent",
                                  attributes={"gen_ai.operation.name": "invoke_agent"}):
    op("retrieve supporting_context", RETRIEVE,
       **{"gen_ai.operation.name": "retrieval"})
    op("chat propose_checks", PROPOSE, **{"gen_ai.operation.name": "chat"})
    ctx = trace.context_api.get_current()
    threads = [threading.Thread(target=check, args=(ctx, "execute_tool run_tests", TESTS, "run_tests")),
               threading.Thread(target=check, args=(ctx, "execute_tool run_lint", LINT, "run_lint"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    time.sleep(GAP)                      # uninstrumented bookkeeping
    op("chat synthesize_report", SYNTH, **{"gen_ai.operation.name": "chat"})

provider.shutdown()
console_file.close()
recs = []
for s in mem.get_finished_spans():
    recs.append(dict(name=s.name,
                     span_id=format(s.context.span_id, "016x"),
                     parent_id=format(s.parent.span_id, "016x") if s.parent else None,
                     trace_id=format(s.context.trace_id, "032x"),
                     start_ns=s.start_time, end_ns=s.end_time,
                     attributes=dict(s.attributes)))
(out / "trace_run.json").write_text(json.dumps(recs, indent=1))
print(f"wrote {len(recs)} spans")
