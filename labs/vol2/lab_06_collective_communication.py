import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import html as html_lib
    import sys
    from pathlib import Path

    import numpy as np

    if sys.platform == "emscripten":
        import micropip

        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        _labs_dir = Path(__file__).resolve().parents[1]
        if str(_labs_dir) not in sys.path:
            sys.path.insert(0, str(_labs_dir))
        from bootstrap import native_bootstrap

        native_bootstrap(__file__)

    import plotly.graph_objects as go
    from mlsysim import Hardware, Systems, ureg
    from mlsysim.physics import (
        calc_hierarchical_allreduce_time,
        calc_ring_allreduce_time,
        calc_tree_allreduce_time,
    )
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS as LAB_CSS,
        COLORS,
        ChapterRecap,
        DesignLedger,
        InstructorMetadata,
        LabMetadata,
        apply_plotly_theme,
        build_lab_report,
        chapter_recap,
        instructor_adoption_card,
        lab_header,
        report_export,
        scenario_brief,
    )

    H100 = Hardware.Cloud.H100
    JETSON = Hardware.Edge.JetsonOrinNX
    IB_NDR = Systems.Fabrics.InfiniBand_NDR
    NVLINK = H100.nvlink
    ETHERNET = Systems.Fabrics.Ethernet_100G

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()

    return (
        COLORS,
        ChapterRecap,
        ETHERNET,
        H100,
        IB_NDR,
        InstructorMetadata,
        JETSON,
        LAB_CSS,
        LabMetadata,
        NVLINK,
        Systems,
        apply_plotly_theme,
        build_lab_report,
        calc_hierarchical_allreduce_time,
        calc_ring_allreduce_time,
        calc_tree_allreduce_time,
        chapter_recap,
        go,
        html_lib,
        instructor_adoption_card,
        lab_header,
        ledger,
        mo,
        np,
        report_export,
        scenario_brief,
        ureg,
    )


@app.cell
def _(ChapterRecap, InstructorMetadata, LabMetadata):
    metadata = LabMetadata(
        lab_id="v2_06_collective_communication",
        title="Collective Communication",
        volume="Volume II",
        chapter="Chapter 6",
        book_anchor="Volume II, Chapter 6: Collective Communication",
        lab_version="1.0.0",
        updated_at="2026-06-02",
        release_channel="preview",
        mlsysim_version="0.1.2",
    )

    recap = ChapterRecap(
        emphasis=(
            "Collective communication is an algorithmic systems layer, not a single "
            "opaque communication overhead term."
        ),
        key_terms=("AllReduce", "AllGather", "ReduceScatter", "alpha-beta model", "hierarchy"),
        ml_concept="Distributed training synchronizes gradients, activations, or expert tokens.",
        systems_translation=(
            "The ML operation becomes a topology-aware communication schedule whose cost "
            "depends on message size, participant count, bandwidth, latency, and overlap."
        ),
        what_to_watch="Watch when the binding term moves from bandwidth to latency or hierarchy.",
        common_trap="Assuming more bandwidth always fixes collective communication.",
        suggested_reading="Volume II, Chapter 6: Collective Communication.",
    )

    instructor = InstructorMetadata(
        why_assign=(
            "Students learn that communication algorithms and topology determine whether "
            "distributed training speedups are real."
        ),
        where_it_fits="After Distributed Training and before Fault Tolerance.",
        assignment_prompt=(
            "Complete the Collective Communication lab, choose a communication strategy, "
            "and submit the downloaded report."
        ),
        expected_report=(
            "Track, collective operation, algorithm choice, binding term, optimization choice, "
            "residual risk, and MLSysIM result snapshot."
        ),
        rubric=(
            "Identifies the active alpha or beta term.",
            "Compares ring, tree, and hierarchical collectives with evidence.",
            "Explains residual risk such as topology mismatch, overlap assumption, or compression error.",
        ),
        misconceptions=(
            "All communication is the same overhead.",
            "Ring AllReduce is always best.",
            "Compression is free if it reduces bytes.",
        ),
        discussion_prompts=(
            "When does hierarchy dominate a flat collective?",
            "Which assumption would make the selected strategy fail?",
            "How does this lab explain non-linear distributed training scaling?",
        ),
        extensions=(
            "Add topology oversubscription.",
            "Add gradient compression error.",
            "Compare Ethernet and InfiniBand placement.",
        ),
        setup_notes=("Runs in browser/WASM with MLSysIM and mlsysbook_labs wheels.",),
    )
    return instructor, metadata, recap


@app.cell(hide_code=True)
def _(LAB_CSS, chapter_recap, lab_header, metadata, mo, recap, scenario_brief):
    mo.vstack(
        [
            LAB_CSS,
            lab_header(
                metadata,
                "Explore how collective algorithms, fabric bandwidth, latency, and hierarchy change distributed training decisions.",
                chips=("Report-ready", "Track-aware", "MLSysIM-backed"),
            ),
            chapter_recap(recap),
            scenario_brief(
                "Training Cluster Communication Decision",
                "Platform team preparing a model-parallel training run",
                "Choose a collective strategy that minimizes exposed communication without hiding residual risk.",
                {
                    "Workload": "Gradient synchronization for a large transformer training step",
                    "Default fabric": "8 GPUs per node with NVLink, nodes connected by InfiniBand NDR",
                    "Decision": "Ring, tree, hierarchical, overlap, or compression",
                },
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    partA_prediction = mo.ui.radio(
        options={
            "A) The number of GPU operations": "ops",
            "B) The bytes each participant must exchange": "bytes",
            "C) The optimizer choice only": "optimizer",
            "D) The dataset size only": "dataset",
        },
        label="For an AllReduce, what primarily determines communication time once the topology is fixed?",
    )

    partB_prediction = mo.ui.radio(
        options={
            "A) Ring is always best": "ring",
            "B) Tree is always best": "tree",
            "C) The answer depends on message size and participant count": "depends",
            "D) Algorithm choice only matters for Ethernet": "ethernet",
        },
        label="Which collective algorithm should win across all message sizes?",
    )

    partC_prediction = mo.ui.radio(
        options={
            "A) Hierarchy helps when inter-node bandwidth/latency is the scarce resource": "hierarchy",
            "B) Hierarchy only helps for one GPU": "single",
            "C) Hierarchy removes communication entirely": "free",
            "D) Hierarchy is unrelated to topology": "unrelated",
        },
        label="When should a hierarchical collective help?",
    )

    partD_prediction = mo.ui.radio(
        options={
            "A) Compression reduces bytes but may add quality or compute risk": "risk",
            "B) Compression is always free": "free",
            "C) Overlap changes math accuracy": "accuracy",
            "D) Overlap makes bandwidth irrelevant": "irrelevant",
        },
        label="What is the main systems risk when using compression or overlap?",
    )
    return partA_prediction, partB_prediction, partC_prediction, partD_prediction


@app.cell(hide_code=True)
def _(mo):
    n_gpus = mo.ui.slider(start=2, stop=1024, value=64, step=2, label="Participants")
    message_gb = mo.ui.slider(start=0.1, stop=80.0, value=16.0, step=0.1, label="Message size per participant (GB)")
    fabric = mo.ui.dropdown(
        options={"InfiniBand NDR": "ib", "100G Ethernet": "eth", "NVLink-only": "nvlink"},
        value="InfiniBand NDR",
        label="Fabric to compare",
    )
    gpus_per_node = mo.ui.slider(start=1, stop=8, value=8, step=1, label="GPUs per node")
    overlap_pct = mo.ui.slider(start=0, stop=80, value=30, step=5, label="Communication hidden by overlap (%)")
    compression_ratio = mo.ui.slider(start=1, stop=8, value=1, step=1, label="Compression ratio")
    partA_reflection = mo.ui.text_area(
        label="Part A reflection",
        placeholder="Name the binding term at this point: bytes, latency rounds, or topology.",
        full_width=True,
    )
    partB_reflection = mo.ui.text_area(
        label="Part B reflection",
        placeholder="Where does the preferred collective change, and why?",
        full_width=True,
    )
    partC_reflection = mo.ui.text_area(
        label="Part C reflection",
        placeholder="When does hierarchy help, and what topology assumption makes that true?",
        full_width=True,
    )
    partD_reflection = mo.ui.text_area(
        label="Part D reflection",
        placeholder="What residual risk remains after compression or overlap?",
        full_width=True,
    )
    student_id = mo.ui.text(label="Student identifier", placeholder="Optional")
    decision_text = mo.ui.text_area(
        label="Final communication decision",
        placeholder="Choose a strategy and explain the residual risk.",
        full_width=True,
    )
    return (
        compression_ratio,
        decision_text,
        fabric,
        gpus_per_node,
        message_gb,
        n_gpus,
        overlap_pct,
        partA_reflection,
        partB_reflection,
        partC_reflection,
        partD_reflection,
        student_id,
    )


@app.cell(hide_code=True)
def _(
    ETHERNET,
    H100,
    IB_NDR,
    JETSON,
    NVLINK,
    apply_plotly_theme,
    build_lab_report,
    calc_hierarchical_allreduce_time,
    calc_ring_allreduce_time,
    calc_tree_allreduce_time,
    compression_ratio,
    decision_text,
    fabric,
    go,
    gpus_per_node,
    html_lib,
    instructor,
    instructor_adoption_card,
    ledger,
    message_gb,
    metadata,
    mo,
    n_gpus,
    np,
    overlap_pct,
    partA_prediction,
    partA_reflection,
    partB_prediction,
    partB_reflection,
    partC_prediction,
    partC_reflection,
    partD_prediction,
    partD_reflection,
    recap,
    report_export,
    student_id,
    ureg,
):
    fabric_map = {
        "ib": IB_NDR,
        "InfiniBand NDR": IB_NDR,
        "eth": ETHERNET,
        "100G Ethernet": ETHERNET,
        "nvlink": NVLINK,
        "NVLink-only": NVLINK,
    }

    fabric_labels = {
        "ib": "InfiniBand NDR",
        "InfiniBand NDR": "InfiniBand NDR",
        "eth": "100G Ethernet",
        "100G Ethernet": "100G Ethernet",
        "nvlink": "NVLink-only",
        "NVLink-only": "NVLink-only",
    }

    def _fabric_values():
        active = fabric_map[fabric.value]
        return active.bandwidth, active.latency

    def _fabric_label():
        return fabric_labels.get(fabric.value, str(fabric.value))

    def _times(gpus=None, size_gb=None, compression=None):
        gpus = int(gpus or n_gpus.value)
        size_gb = float(size_gb or message_gb.value)
        compression = max(1, int(compression or compression_ratio.value))
        msg = (size_gb / compression) * ureg.GB
        bw, lat = _fabric_values()
        ring = calc_ring_allreduce_time(msg, gpus, bw, lat).m_as("ms")
        tree = calc_tree_allreduce_time(msg, gpus, bw, lat).m_as("ms")
        nodes = max(1, int(np.ceil(gpus / max(1, int(gpus_per_node.value)))))
        hierarchy = calc_hierarchical_allreduce_time(
            msg,
            nodes,
            max(1, int(gpus_per_node.value)),
            NVLINK.bandwidth,
            bw,
            NVLINK.latency,
            lat,
        ).m_as("ms")
        exposed = min(ring, tree, hierarchy) * (1 - overlap_pct.value / 100.0)
        return {
            "ring_ms": ring,
            "tree_ms": tree,
            "hierarchical_ms": hierarchy,
            "exposed_best_ms": exposed,
            "message_gb_after_compression": size_gb / compression,
        }

    def _bar_chart(values):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Ring", "Tree", "Hierarchical", "Exposed best"],
                y=[
                    values["ring_ms"],
                    values["tree_ms"],
                    values["hierarchical_ms"],
                    values["exposed_best_ms"],
                ],
                marker_color=["#A51C30", "#1F407A", "#247A4D", "#9A5B00"],
            )
        )
        fig.update_layout(height=320, yaxis_title="Collective time (ms)", showlegend=False)
        return apply_plotly_theme(fig)

    def _frontier_chart():
        sizes = np.geomspace(0.1, 80, 28)
        ring = []
        tree = []
        hierarchy = []
        for size in sizes:
            vals = _times(size_gb=float(size), compression=1)
            ring.append(vals["ring_ms"])
            tree.append(vals["tree_ms"])
            hierarchy.append(vals["hierarchical_ms"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sizes, y=ring, mode="lines", name="Ring"))
        fig.add_trace(go.Scatter(x=sizes, y=tree, mode="lines", name="Tree"))
        fig.add_trace(go.Scatter(x=sizes, y=hierarchy, mode="lines", name="Hierarchical"))
        fig.update_layout(
            height=330,
            xaxis_title="Message size per participant (GB)",
            yaxis_title="Collective time (ms)",
            xaxis_type="log",
        )
        return apply_plotly_theme(fig)

    def _current_values():
        vals = _times()
        best = min(("Ring", vals["ring_ms"]), ("Tree", vals["tree_ms"]), ("Hierarchical", vals["hierarchical_ms"]), key=lambda x: x[1])
        vals["best_algorithm"] = best[0]
        vals["best_ms"] = best[1]
        vals["failure_state"] = "SLA violation" if vals["exposed_best_ms"] > 1000 else "feasible"
        return vals

    def _reading_guide(title, steps, interpretation):
        def _inline_markup(text):
            parts = html_lib.escape(text).split("**")
            return "".join(f"<strong>{part}</strong>" if idx % 2 else part for idx, part in enumerate(parts))

        step_items = "".join(f"<li>{_inline_markup(step)}</li>" for step in steps)
        return mo.Html(
            f"""
<div class="mlsysbook-callout">
  <strong>{html_lib.escape(title)}</strong>
  <ol style="margin: 8px 0 8px 20px; padding: 0; line-height: 1.55;">
    {step_items}
  </ol>
  <div><strong>How to read the result:</strong> {_inline_markup(interpretation)}</div>
</div>
"""
        )

    def build_part_a():
        vals = _current_values()
        if partA_prediction.value is None:
            return mo.vstack(
                [
                    mo.md("## Part A - Collective Operation Anatomy"),
                    mo.callout(
                        mo.md(
                            "This part starts from a concrete training-step baseline: **64 participants**, "
                            "**16 GB** per participant, and **InfiniBand NDR**. First predict what controls "
                            "the communication time, then the controls and bar chart will unlock."
                        ),
                        kind="info",
                    ),
                    partA_prediction,
                    mo.callout(mo.md("Select a prediction to unlock the collective communication instruments."), kind="warn"),
                ]
            )
        return mo.vstack(
            [
                mo.md("## Part A - Collective Operation Anatomy"),
                partA_prediction,
                mo.callout(
                    mo.md(
                        f"You predicted **{partA_prediction.value}**. The actual binding quantity is usually bytes moved through the fabric, plus latency rounds."
                    ),
                    kind="success" if partA_prediction.value == "bytes" else "warn",
                ),
                mo.callout(
                    mo.md(
                        "AllReduce is a data movement operation: every participant contributes "
                        "a tensor and receives the reduced result. The active systems question is "
                        "how many bytes move over which fabric, and how many latency rounds are required."
                    ),
                    kind="info",
                ),
                _reading_guide(
                    "Start with one concrete communication point",
                    (
                        "Use the default **64 participants**, **16 GB** per participant, and **InfiniBand NDR** as the baseline training step.",
                        "Change only the fabric dropdown first. That isolates the interconnect decision.",
                        "Then double or halve the message size. That shows when bytes dominate the answer.",
                    ),
                    "each bar is MLSysIM's modeled collective time in milliseconds. The shortest bar is the fastest schedule for the current point; the gap between bars is the penalty for choosing the wrong schedule.",
                ),
                mo.accordion(
                    {
                        "Math Peek: Ring AllReduce Alpha-Beta Model": mo.md(
                            "Ring AllReduce time is `2 * (N - 1) / N * M / bandwidth + 2 * (N - 1) * latency`. "
                            "This lab uses MLSysIM's `calc_ring_allreduce_time`, `calc_tree_allreduce_time`, "
                            "and `calc_hierarchical_allreduce_time` functions."
                        )
                    }
                ),
                mo.hstack([n_gpus, message_gb, fabric], justify="start"),
                mo.as_html(_bar_chart(vals)),
                mo.md(
                    f"At **{n_gpus.value} participants**, **{message_gb.value:.1f} GB**, and **{_fabric_label()}**, "
                    f"the current best is **{vals['best_algorithm']}** at **{vals['best_ms']:.1f} ms** before overlap. "
                    f"Failure state: **{vals['failure_state']}**."
                ),
                mo.callout(
                    mo.md(
                        "**Reflection checkpoint:** Explain whether this point is bandwidth-bound, "
                        "latency-bound, or topology-bound, and name the evidence you used."
                    ),
                    kind="info",
                ),
                partA_reflection,
            ]
        )

    def build_part_b():
        if partB_prediction.value is None:
            return mo.vstack(
                [
                    mo.md("## Part B - Algorithm And Topology Frontier"),
                    mo.callout(
                        mo.md(
                            "This part asks whether one collective algorithm dominates, or whether the right "
                            "algorithm changes as message size, participant count, and topology change."
                        ),
                        kind="info",
                    ),
                    partB_prediction,
                    mo.callout(mo.md("Select a prediction before comparing collective algorithms."), kind="warn"),
                ]
            )
        return mo.vstack(
            [
                mo.md("## Part B - Algorithm And Topology Frontier"),
                partB_prediction,
                mo.callout(
                    mo.md(
                        "Ring, tree, and hierarchical collectives expose different alpha-beta trade-offs. "
                        "Ring often uses bandwidth efficiently for large messages; tree can help when latency rounds dominate; "
                        "hierarchy exploits fast intra-node links before paying inter-node cost."
                    ),
                    kind="info",
                ),
                _reading_guide(
                    "Read the frontier as a decision boundary",
                    (
                        "Keep participant count fixed and choose the fabric you want to study.",
                        "Scan left to right across message sizes instead of focusing on one point.",
                        "Look for curve crossings. A crossing means the preferred algorithm changes.",
                    ),
                    "the x-axis is message size on a log scale, and the y-axis is modeled collective time. Lower curves are better; a flat-looking region usually means latency rounds are more visible than bandwidth.",
                ),
                mo.hstack([n_gpus, fabric, gpus_per_node], justify="start"),
                mo.as_html(_frontier_chart()),
                mo.callout(
                    mo.md(
                        "**Reflection checkpoint:** Pick one crossover or non-crossover and explain "
                        "the system reason. Do not just name the winning line."
                    ),
                    kind="info",
                ),
                partB_reflection,
            ]
        )

    def build_part_c():
        vals = _current_values()
        if partC_prediction.value is None:
            return mo.vstack(
                [
                    mo.md("## Part C - Hierarchy As A Systems Decision"),
                    mo.callout(
                        mo.md(
                            "This part treats a cluster as two networks: fast links inside a node and slower "
                            "links across nodes. The question is whether hierarchy matches the hardware."
                        ),
                        kind="info",
                    ),
                    partC_prediction,
                    mo.callout(mo.md("Select a prediction before inspecting the hierarchy result."), kind="warn"),
                ]
            )
        hierarchy_gain = vals["ring_ms"] / vals["hierarchical_ms"] if vals["hierarchical_ms"] else 0
        return mo.vstack(
            [
                mo.md("## Part C - Hierarchy As A Systems Decision"),
                partC_prediction,
                _reading_guide(
                    "Separate intra-node and inter-node communication",
                    (
                        "Start at **8 GPUs per node** so hierarchy can use fast NVLink within each node.",
                        "Reduce GPUs per node toward 1 and watch hierarchy lose structure.",
                        "Switch the fabric dropdown to see how much the inter-node link controls the result.",
                    ),
                    "hierarchical collectives are useful when the system has two communication regimes. If the fast local link is not actually available, the hierarchy assumption is weak.",
                ),
                mo.hstack([n_gpus, gpus_per_node, fabric], justify="start"),
                mo.callout(
                    mo.md(
                        f"With **{gpus_per_node.value} GPUs per node**, hierarchy is "
                        f"**{hierarchy_gain:.2f}x** faster than flat ring for this scenario."
                    ),
                    kind="success" if hierarchy_gain > 1 else "warn",
                ),
                mo.as_html(_bar_chart(vals)),
                mo.callout(
                    mo.md(
                        "**Reflection checkpoint:** State the hardware assumption that makes hierarchy "
                        "a good or bad decision here."
                    ),
                    kind="info",
                ),
                partC_reflection,
            ]
        )

    def build_part_d():
        vals = _current_values()
        if partD_prediction.value is None:
            return mo.vstack(
                [
                    mo.md("## Part D - Overlap, Compression, And Residual Risk"),
                    mo.callout(
                        mo.md(
                            "This part separates reducing exposed time from eliminating risk. Compression "
                            "shrinks bytes; overlap hides time; both require validation."
                        ),
                        kind="info",
                    ),
                    partD_prediction,
                    mo.callout(mo.md("Select a prediction before using overlap and compression controls."), kind="warn"),
                ]
            )
        return mo.vstack(
            [
                mo.md("## Part D - Overlap, Compression, And Residual Risk"),
                partD_prediction,
                _reading_guide(
                    "Treat optimization knobs as claims that need evidence",
                    (
                        "Move compression first. That changes the modeled bytes but introduces possible accuracy or convergence risk.",
                        "Move overlap second. That hides communication only if the schedule has useful work to run at the same time.",
                        "Compare the exposed best bar against the raw collective bars. That separates real communication cost from hidden cost.",
                    ),
                    "compression and overlap can improve the exposed time without removing the systems risk. A release decision must say which assumption you still need to validate.",
                ),
                mo.hstack([overlap_pct, compression_ratio], justify="start"),
                mo.callout(
                    mo.md(
                        f"Compression reduces the modeled message to **{vals['message_gb_after_compression']:.2f} GB**, "
                        f"and overlap hides **{overlap_pct.value}%** of the best collective. "
                        "The residual risk is that overlap may not be schedulable and compression may harm convergence."
                    ),
                    kind="warn" if compression_ratio.value > 1 or overlap_pct.value > 50 else "info",
                ),
                mo.as_html(_bar_chart(vals)),
                mo.callout(
                    mo.md(
                        "**Reflection checkpoint:** Write the risk you would put in a design review "
                        "before shipping this optimization."
                    ),
                    kind="info",
                ),
                partD_reflection,
            ]
        )

    def build_synthesis():
        vals = _current_values()
        decision = decision_text.value or f"Use {vals['best_algorithm']} and validate overlap on the real topology."
        result_snapshot = {
            "participants": n_gpus.value,
            "message_gb": message_gb.value,
            "fabric": fabric.value,
            "fabric_label": _fabric_label(),
            "gpus_per_node": gpus_per_node.value,
            "overlap_pct": overlap_pct.value,
            "compression_ratio": compression_ratio.value,
            **vals,
        }
        ledger.save(
            chapter=6,
            design={
                "lab_id": metadata.lab_id,
                "decision": decision,
                "binding_constraint": "communication latency/bandwidth",
                "result_snapshot": result_snapshot,
            },
        )
        report = build_lab_report(
            metadata,
            student_id=student_id.value or "",
            track="cloud/fleet",
            scenario="collective communication",
            recap=recap,
            predictions={
                "operation_anatomy": partA_prediction.value,
                "algorithm_frontier": partB_prediction.value,
                "hierarchy": partC_prediction.value,
                "overlap_compression": partD_prediction.value,
            },
            knob_settings={
                "participants": n_gpus.value,
                "message_gb": message_gb.value,
                "fabric": _fabric_label(),
                "gpus_per_node": gpus_per_node.value,
                "overlap_pct": overlap_pct.value,
                "compression_ratio": compression_ratio.value,
            },
            binding_constraints={
                "best_algorithm": vals["best_algorithm"],
                "best_ms": round(vals["best_ms"], 3),
                "failure_state": vals["failure_state"],
            },
            decisions={"communication_strategy": decision},
            reflections={
                "part_a_operation_anatomy": partA_reflection.value
                or "The fastest collective depends on message size, participant count, fabric, and hierarchy.",
                "part_b_algorithm_frontier": partB_reflection.value,
                "part_c_hierarchy": partC_reflection.value,
                "part_d_overlap_compression": partD_reflection.value,
            },
            residual_risk="Topology mismatch, unschedulable overlap, or compression-induced convergence loss.",
            result_snapshot=result_snapshot,
        )
        return mo.vstack(
            [
                mo.md("## Synthesis - Communication Design Review"),
                student_id,
                decision_text,
                mo.callout(
                    mo.md(
                        f"Recommended strategy: **{vals['best_algorithm']}**. "
                        "Submit the report after writing a short decision rationale."
                    ),
                    kind="success",
                ),
                report_export(report),
                instructor_adoption_card(instructor),
            ]
        )

    tabs = mo.ui.tabs(
        {
            "Part A - Operation Anatomy": build_part_a(),
            "Part B - Algorithm Frontier": build_part_b(),
            "Part C - Hierarchy": build_part_c(),
            "Part D - Overlap And Compression": build_part_d(),
            "Synthesis": build_synthesis(),
        }
    )
    tabs
    return


@app.cell(hide_code=True)
def _(metadata, mo):
    mo.Html(
        f"""
<div class="lab-hud" style="display:flex; gap:18px; align-items:center; padding:12px 18px; border:1px solid #D9DEE8; border-radius:8px; background:#F6F8FB; margin-top:20px;">
  <span><strong>LEDGER</strong></span>
  <span>{metadata.lab_id}</span>
  <span>Lab v{metadata.lab_version}</span>
  <span>Report schema {metadata.report_schema_version}</span>
</div>
"""
    )
    return



# ─── TRACK-AWARE MIGRATION SHELL ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(ledger, mo):
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        legacy_migration_panel,
    )

    _metadata = get_lab_metadata("vol2/lab_06_collective_communication.py")
    _saved_track = ledger.get_track()
    _track_id = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    _profile = get_track_profile(_track_id)
    _variant = get_lab_track_variant(_metadata.lab_id, _profile.track_id)
    mo.vstack([
        ACADEMIC_LAB_CSS,
        legacy_migration_panel(_metadata, _profile, _variant),
    ])
    return

if __name__ == "__main__":
    app.run()
