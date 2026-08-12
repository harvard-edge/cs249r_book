import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


# ===========================================================================
# ZONE A: OPENING
# ===========================================================================


@app.cell
async def _():
    import sys
    from pathlib import Path
    import html as html_lib
    import marimo as mo

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
    from mlsysim.labs.state import DesignLedger
    from mlsysim.solvers import CompressionModel
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        ChapterRecap,
        DEFAULT_TRACK_ID,
        LabMetadata,
        big_takeaways,
        build_lab_report,
        chapter_recap,
        decision_flow,
        get_lab_track_variant,
        get_track_profile,
        lab_header,
        lab_map,
        learning_objectives,
        report_export_panel,
        resolve_mlsysim_ref,
        scenario_brief,
        scenario_thread,
        track_arc_context,
        track_context,
        track_selector,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()

    return (
        ACADEMIC_LAB_CSS,
        COLORS,
        ChapterRecap,
        CompressionModel,
        DEFAULT_TRACK_ID,
        LAB_CSS,
        LabMetadata,
        apply_plotly_theme,
        big_takeaways,
        build_lab_report,
        chapter_recap,
        decision_flow,
        get_lab_track_variant,
        get_track_profile,
        go,
        html_lib,
        lab_header,
        lab_map,
        learning_objectives,
        ledger,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        scenario_brief,
        scenario_thread,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(ChapterRecap, LabMetadata):
    lab_metadata = LabMetadata(
        lab_id="v1_10_compression_paradox",
        title="The Compression Paradox",
        volume="Volume I",
        chapter="Chapter 10: Model Compression",
        book_anchor="Volume I, Chapter 10",
        updated_at="2026-06-13",
    )
    lab_learning_objectives = (
        "Diagnose which compression method is feasible for a track-specific deployment target.",
        "Distinguish storage savings from real sparse-kernel speedup.",
        "Decide when a dense student is safer by weighing teacher quality against deployable student constraints.",
        "Validate a release recipe against size, quality, speed, hardware support, and residual risk.",
    )
    chapter_10_recap = ChapterRecap(
        emphasis=(
            "Compression helps only when it reduces the resource that actually binds the "
            "deployment and the runtime can exploit the representation."
        ),
        key_terms=(
            "quantization",
            "calibration",
            "structured pruning",
            "unstructured pruning",
            "distillation",
            "Pareto frontier",
        ),
        ml_concept=(
            "A model can be made smaller with lower precision, fewer effective weights, "
            "or a smaller student architecture."
        ),
        systems_translation=(
            "The deployment result depends on quality guardrails, supported kernels, memory "
            "or flash budgets, latency, energy, and validation tests."
        ),
        what_to_watch=(
            "A candidate can be smaller but still fail because it is unsupported, too slow, "
            "or below the track quality guardrail."
        ),
        common_trap="Treating compression ratio as the same thing as deployable speedup.",
        suggested_reading="Volume I, Chapter 10: Model Compression.",
    )
    lab_big_takeaways = (
        "Compression helps only if it attacks the binding resource for the selected track.",
        "Runtime and hardware support are part of the compression contract.",
        "A release recipe needs validation evidence and a named residual risk.",
    )
    return chapter_10_recap, lab_big_takeaways, lab_learning_objectives, lab_metadata


@app.cell(hide_code=True)
def _(DEFAULT_TRACK_ID, ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else DEFAULT_TRACK_ID
    v1_10_track_picker = track_selector(default=_default_track)
    return (v1_10_track_picker,)


@app.cell
def _(get_lab_track_variant, get_track_profile, v1_10_track_picker):
    v1_10_track_id = v1_10_track_picker.value
    v1_10_track_profile = get_track_profile(v1_10_track_id)
    v1_10_variant = get_lab_track_variant("v1_10_compression_paradox", v1_10_track_id)
    return v1_10_track_id, v1_10_track_profile, v1_10_variant


@app.cell
def _(CompressionModel, resolve_mlsysim_ref, v1_10_variant):
    def v1_10_size_limit_for(hardware, limit_ref: str):
        if limit_ref == "memory.flash_capacity":
            return hardware.memory.flash_capacity or hardware.memory.capacity
        if limit_ref == "storage.capacity" and hardware.storage is not None:
            return hardware.storage.capacity
        return hardware.memory.capacity

    def v1_10_method_label(method_key: str) -> str:
        labels = {
            "int8_quantization": "INT8 quantization",
            "structured_pruning": "structured pruning",
            "distillation": "distillation",
            "no_compression": "no compression / keep baseline",
        }
        return labels.get(method_key, method_key.replace("_", " "))

    def v1_10_quant_label(bit_width: int) -> str:
        return f"FP{bit_width} quantization" if bit_width >= 16 else f"INT{bit_width} quantization"

    def v1_10_build_candidate_configs(bit_widths: tuple[int, ...]) -> tuple[dict[str, object], ...]:
        configs = [{"label": "FP32 baseline", "method": "quantization", "target_bitwidth": 32}]
        configs.extend(
            {
                "label": v1_10_quant_label(int(bit_width)),
                "method": "quantization",
                "target_bitwidth": int(bit_width),
            }
            for bit_width in bit_widths
        )
        configs.extend(
            (
                {
                    "label": "50% structured pruning",
                    "method": "pruning",
                    "sparsity": 0.5,
                    "sparsity_type": "structured",
                },
                {
                    "label": "90% unstructured pruning",
                    "method": "pruning",
                    "sparsity": 0.9,
                    "sparsity_type": "unstructured",
                },
                {
                    "label": "50% N:M pruning",
                    "method": "pruning",
                    "sparsity": 0.5,
                    "sparsity_type": "n_m",
                },
                {
                    "label": "90% structured pruning",
                    "method": "pruning",
                    "sparsity": 0.9,
                    "sparsity_type": "structured",
                },
            )
        )
        return tuple(configs)

    def v1_10_candidate_to_row(candidate) -> dict[str, object]:
        return {
            "label": candidate.label,
            "method": candidate.method,
            "target_bitwidth": candidate.target_bitwidth,
            "sparsity_pct": round(candidate.sparsity * 100),
            "sparsity_type": candidate.sparsity_type,
            "compressed_size_mb": round(candidate.compressed_size_gb.to("MB").magnitude, 4),
            "compression_ratio": round(candidate.compression_ratio, 3),
            "accuracy_drop_pct": round(abs(candidate.estimated_accuracy_delta) * 100, 3),
            "inference_speedup": round(candidate.inference_speedup, 3),
            "hardware_supported": candidate.hardware_supported,
            "feasible": candidate.feasible,
            "pareto_status": candidate.pareto_status,
            "binding_constraint": candidate.binding_constraint,
            "guardrail_violations": tuple(candidate.guardrail_violations),
        }

    v1_10_model = resolve_mlsysim_ref(v1_10_variant.model_ref)
    v1_10_hardware = resolve_mlsysim_ref(v1_10_variant.hardware_ref)
    v1_10_defaults = dict(v1_10_variant.defaults)
    v1_10_bit_widths = tuple(int(bit_width) for bit_width in v1_10_defaults["bit_widths"])
    v1_10_validation_tests = tuple(str(test) for test in v1_10_defaults["validation_tests"])
    v1_10_size_limit = v1_10_size_limit_for(
        v1_10_hardware,
        str(v1_10_defaults.get("size_limit_ref", "memory.capacity")),
    )
    v1_10_candidate_configs = v1_10_build_candidate_configs(v1_10_bit_widths)
    v1_10_compression_solver = CompressionModel()
    v1_10_compression_sweep = v1_10_compression_solver.sweep(
        v1_10_model,
        v1_10_hardware,
        list(v1_10_candidate_configs),
        size_limit=v1_10_size_limit,
        max_accuracy_drop=float(v1_10_defaults["max_accuracy_drop"]),
        min_speedup=float(v1_10_defaults["min_speedup"]),
        require_hardware_support=bool(v1_10_defaults["require_hardware_support"]),
    )
    v1_10_candidate_rows = tuple(
        v1_10_candidate_to_row(candidate) for candidate in v1_10_compression_sweep.candidates
    )
    _quant_candidates = [
        candidate
        for candidate in v1_10_compression_sweep.candidates
        if candidate.method == "quantization" and candidate.target_bitwidth in v1_10_bit_widths
    ]
    _feasible_quant_bits = [
        int(candidate.target_bitwidth)
        for candidate in _quant_candidates
        if candidate.feasible and candidate.target_bitwidth is not None
    ]
    v1_10_lowest_feasible_bit_width = min(_feasible_quant_bits) if _feasible_quant_bits else 0
    v1_10_best_candidate_label = v1_10_compression_sweep.best_candidate_label or "No feasible solver candidate"
    v1_10_best_candidate_row = next(
        (row for row in v1_10_candidate_rows if row["label"] == v1_10_compression_sweep.best_candidate_label),
        None,
    )
    v1_10_dominated_label = next(
        (row["label"] for row in v1_10_candidate_rows if row["pareto_status"] == "dominated"),
        "FP32 baseline",
    )
    v1_10_unsupported_label = next(
        (row["label"] for row in v1_10_candidate_rows if not row["hardware_supported"]),
        "90% unstructured pruning",
    )
    v1_10_rejected_candidate_row = next(
        (
            row
            for row in v1_10_candidate_rows
            if not row["feasible"] and row["label"] != "FP32 baseline"
        ),
        None,
    )
    v1_10_rejected_method_label = (
        str(v1_10_rejected_candidate_row["label"])
        if v1_10_rejected_candidate_row
        else "no rejected solver candidate"
    )
    v1_10_carry_forward_implication = (
        "Use Lab 11 roofline and hardware-acceleration evidence to verify that "
        "the selected representation maps to a supported fast path."
    )
    v1_10_aggressive_label = next(
        (
            row["label"]
            for row in v1_10_candidate_rows
            if row["binding_constraint"] == "quality"
        ),
        "90% structured pruning",
    )
    v1_10_recipe_prediction_options = {
        f"Dominated recipe: {v1_10_dominated_label}": "dominated",
        f"Unsupported recipe: {v1_10_unsupported_label}": "unsupported",
        f"Overly aggressive recipe: {v1_10_aggressive_label}": "aggressive",
        f"Best feasible recipe: {v1_10_best_candidate_label}": "best",
    }

    def v1_10_recipe_summary(quant_bit: int, prune_choice: str, distillation_choice: str) -> str:
        quant = "no quantization" if int(quant_bit) == 0 else v1_10_quant_label(int(quant_bit))
        prune_labels = {
            "none": "no pruning",
            "structured_50": "50% structured pruning",
            "unstructured_90": "90% unstructured pruning",
            "nm_50": "50% N:M pruning",
            "structured_90": "90% structured pruning",
        }
        distill = "dense-student fallback" if distillation_choice == "on" else "no distillation"
        return f"{quant}; {prune_labels.get(prune_choice, prune_choice)}; {distill}"

    def v1_10_prune_config(prune_choice: str) -> dict[str, object] | None:
        configs = {
            "structured_50": {"sparsity": 0.5, "sparsity_type": "structured"},
            "unstructured_90": {"sparsity": 0.9, "sparsity_type": "unstructured"},
            "nm_50": {"sparsity": 0.5, "sparsity_type": "n_m"},
            "structured_90": {"sparsity": 0.9, "sparsity_type": "structured"},
        }
        return configs.get(prune_choice)

    def v1_10_evaluate_recipe(
        quant_bit: int,
        prune_choice: str,
        distillation_choice: str,
        calibration_choice: str,
        validation_test: str,
    ) -> dict[str, object]:
        failures: list[str] = []
        warnings: list[str] = []
        solver_checks: list[dict[str, object]] = []
        selected_recipe = v1_10_recipe_summary(quant_bit, prune_choice, distillation_choice)

        if int(quant_bit) > 0:
            quant_candidate = v1_10_compression_solver.candidate(
                v1_10_model,
                v1_10_hardware,
                label=v1_10_quant_label(int(quant_bit)),
                method="quantization",
                target_bitwidth=int(quant_bit),
                size_limit=v1_10_size_limit,
                max_accuracy_drop=float(v1_10_defaults["max_accuracy_drop"]),
                min_speedup=float(v1_10_defaults["min_speedup"]),
                require_hardware_support=bool(v1_10_defaults["require_hardware_support"]),
            )
            quant_row = v1_10_candidate_to_row(quant_candidate)
            solver_checks.append(quant_row)
            failures.extend(quant_candidate.guardrail_violations)

        prune_config = v1_10_prune_config(prune_choice)
        if prune_config is not None:
            prune_candidate = v1_10_compression_solver.candidate(
                v1_10_model,
                v1_10_hardware,
                label=f"{round(float(prune_config['sparsity']) * 100)}% {prune_config['sparsity_type']} pruning",
                method="pruning",
                sparsity=float(prune_config["sparsity"]),
                sparsity_type=str(prune_config["sparsity_type"]),
                size_limit=v1_10_size_limit,
                max_accuracy_drop=float(v1_10_defaults["max_accuracy_drop"]),
                min_speedup=float(v1_10_defaults["min_speedup"]),
                require_hardware_support=bool(v1_10_defaults["require_hardware_support"]),
            )
            prune_row = v1_10_candidate_to_row(prune_candidate)
            solver_checks.append(prune_row)
            failures.extend(prune_candidate.guardrail_violations)

        if int(quant_bit) == 0 and prune_config is None and distillation_choice == "off":
            failures.append("recipe: no compression method selected, so the primary metric does not improve")
        if int(quant_bit) and int(quant_bit) < 8 and calibration_choice != "qat":
            failures.append("calibration: sub-8-bit precision requires QAT before release")
        if distillation_choice == "on":
            warnings.append(
                "distillation: dense-student physics is not modeled by CompressionModel; prove quality with validation"
            )
            validation_lower = validation_test.lower()
            if not any(token in validation_lower for token in ("quality", "replay", "regression", "signal", "recall")):
                failures.append("validation: distillation selected without an explicit quality or replay test")
        if not validation_test:
            failures.append("validation: no release validation test selected")

        if solver_checks:
            primary_metric_gain = (
                f"{max(float(row['inference_speedup']) for row in solver_checks):.2f}x max speedup; "
                f"{min(float(row['compressed_size_mb']) for row in solver_checks):.3g} MB smallest solver anchor"
            )
        else:
            primary_metric_gain = "distillation recommendation only; no solver speedup claimed"
        binding_constraint = failures[0].split(":", 1)[0] if failures else "none"
        return {
            "selected_recipe": selected_recipe,
            "solver_checks": tuple(solver_checks),
            "release_ok": not failures,
            "failures": tuple(dict.fromkeys(failures)),
            "warnings": tuple(warnings),
            "primary_metric_gain": primary_metric_gain,
            "binding_constraint": binding_constraint,
            "validation_test": validation_test,
        }

    def v1_10_evaluate_distillation(
        teacher_quality: str,
        student_scale_pct: int,
        validation_test: str,
    ) -> dict[str, object]:
        teacher_penalty = {
            "validated": 0.002,
            "brittle": 0.012,
            "weak": 0.030,
        }.get(teacher_quality, 0.012)
        student_scale = max(float(student_scale_pct) / 100, 0.05)
        capacity_penalty = max(0.0, 0.40 - student_scale) * 0.05
        estimated_accuracy_drop = teacher_penalty + capacity_penalty
        compression_ratio = 1 / student_scale
        latency_speedup = compression_ratio
        validation_lower = validation_test.lower()
        failures: list[str] = []
        if estimated_accuracy_drop > float(v1_10_defaults["max_accuracy_drop"]):
            failures.append(
                f"quality: distillation drop {estimated_accuracy_drop:.3g} exceeds "
                f"{float(v1_10_defaults['max_accuracy_drop']):.3g}"
            )
        if not any(token in validation_lower for token in ("quality", "replay", "regression", "signal", "recall")):
            failures.append("validation: dense student needs quality, replay, signal, or regression evidence")
        return {
            "teacher_quality": teacher_quality,
            "student_scale_pct": int(student_scale_pct),
            "estimated_accuracy_drop_pct": round(estimated_accuracy_drop * 100, 3),
            "compression_ratio": round(compression_ratio, 2),
            "latency_speedup": round(latency_speedup, 2),
            "release_ok": not failures,
            "failures": tuple(failures),
            "binding_constraint": failures[0].split(":", 1)[0] if failures else "none",
        }

    return (
        v1_10_aggressive_label,
        v1_10_best_candidate_label,
        v1_10_best_candidate_row,
        v1_10_bit_widths,
        v1_10_carry_forward_implication,
        v1_10_candidate_configs,
        v1_10_candidate_rows,
        v1_10_candidate_to_row,
        v1_10_compression_solver,
        v1_10_compression_sweep,
        v1_10_defaults,
        v1_10_dominated_label,
        v1_10_evaluate_distillation,
        v1_10_evaluate_recipe,
        v1_10_hardware,
        v1_10_lowest_feasible_bit_width,
        v1_10_method_label,
        v1_10_model,
        v1_10_quant_label,
        v1_10_rejected_candidate_row,
        v1_10_rejected_method_label,
        v1_10_recipe_prediction_options,
        v1_10_size_limit,
        v1_10_unsupported_label,
        v1_10_validation_tests,
    )


@app.cell
def _(COLORS, html_lib, mo):
    def v1_10_fmt_size_mb(value: float) -> str:
        if value >= 1024:
            return f"{value / 1024:.2f} GB"
        if value >= 1:
            return f"{value:.2f} MB"
        return f"{value * 1024:.1f} KB"

    def v1_10_fmt_bool(value: bool) -> str:
        return "yes" if value else "no"

    def v1_10_status_color(ok: bool) -> str:
        return COLORS["GreenLine"] if ok else COLORS["RedLine"]

    def v1_10_part_banner(letter: str, title: str, why: str, color: str, duration: str = "10-12 min"):
        return mo.Html(
            f"""
<div style="margin: 12px 0;">
  <div style="display:flex; align-items:center; gap:12px;">
    <div style="background:{color}; color:white; border-radius:50%; width:32px; height:32px;
                display:inline-flex; align-items:center; justify-content:center; font-size:0.9rem;
                font-weight:800; flex-shrink:0;">{html_lib.escape(letter)}</div>
    <div style="flex:1; height:2px; background:{COLORS['Border']};"></div>
    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                text-transform:uppercase; letter-spacing:0.12em;">
      Part {html_lib.escape(letter)} &middot; {html_lib.escape(duration)}
    </div>
  </div>
  <div style="font-size:1.5rem; font-weight:800; color:{COLORS['Text']};
              margin-top:8px; line-height:1.2;">{html_lib.escape(title)}</div>
  <div style="color:{COLORS['TextSec']}; font-size:0.92rem; margin-top:6px;
              line-height:1.55; max-width:780px;">{html_lib.escape(why)}</div>
</div>
"""
        )

    def v1_10_metric_cards(cards: tuple[tuple[str, str, str, str], ...]):
        card_html = []
        for title, value, caption, color in cards:
            card_html.append(
                f"""
<div style="padding:14px 16px; border:1px solid {COLORS['Border']}; border-radius:8px;
            background:white; border-top:3px solid {color}; min-width:170px; flex:1;">
  <div style="color:{COLORS['TextMuted']}; font-size:0.75rem; font-weight:700;
              text-transform:uppercase; letter-spacing:0.08em;">{html_lib.escape(title)}</div>
  <div style="font-size:1.45rem; font-weight:800; color:{color}; margin-top:4px;">
    {html_lib.escape(value)}
  </div>
  <div style="font-size:0.78rem; color:{COLORS['TextSec']}; line-height:1.45;">
    {html_lib.escape(caption)}
  </div>
</div>
"""
            )
        return mo.Html(
            f"""
<div style="display:flex; gap:12px; flex-wrap:wrap; margin:14px 0;">
  {''.join(card_html)}
</div>
"""
        )

    def v1_10_reveal_card(title: str, predicted: str, actual: str, consequence: str, ok: bool = False):
        color = COLORS["GreenLine"] if ok else COLORS["OrangeLine"]
        return mo.Html(
            f"""
<div style="background:white; border:1px solid {COLORS['Border']}; border-left:4px solid {color};
            border-radius:8px; padding:16px 18px; margin:14px 0;">
  <div style="font-size:0.72rem; font-weight:800; color:{color}; text-transform:uppercase;
              letter-spacing:0.12em; margin-bottom:6px;">Prediction vs. Reality</div>
  <div style="font-weight:800; color:{COLORS['Text']}; margin-bottom:8px;">
    {html_lib.escape(title)}
  </div>
  <div style="color:{COLORS['TextSec']}; line-height:1.65;">
    <strong>You predicted:</strong> {html_lib.escape(predicted)}<br/>
    <strong>Actual:</strong> {html_lib.escape(actual)}<br/>
    <strong>Consequence:</strong> {html_lib.escape(consequence)}
  </div>
</div>
"""
        )

    def v1_10_checkpoint_card(title: str, fields: tuple[tuple[str, str], ...], color: str):
        field_html = "".join(
            f"""
<div style="display:flex; gap:8px; align-items:flex-start; padding:5px 0;">
  <div style="min-width:160px; color:{COLORS['TextMuted']}; font-weight:800; font-size:0.76rem;
              text-transform:uppercase; letter-spacing:0.06em;">{html_lib.escape(label)}</div>
  <div style="color:{COLORS['Text']}; line-height:1.45;">{html_lib.escape(value)}</div>
</div>
"""
            for label, value in fields
        )
        return mo.Html(
            f"""
<div style="background:white; border:1px solid {COLORS['Border']}; border-left:4px solid {color};
            border-radius:8px; padding:15px 18px; margin:14px 0;">
  <div style="font-size:0.72rem; font-weight:800; color:{color}; text-transform:uppercase;
              letter-spacing:0.12em; margin-bottom:6px;">Checkpoint</div>
  <div style="font-weight:800; color:{COLORS['Text']}; margin-bottom:6px;">
    {html_lib.escape(title)}
  </div>
  {field_html}
</div>
"""
        )

    def v1_10_failure_card(title: str, failures: tuple[str, ...] | list[str], recovery: str):
        failure_items = "".join(
            f"<li>{html_lib.escape(str(failure))}</li>" for failure in failures
        )
        return mo.Html(
            f"""
<div style="background:#FFF7F7; border:1px solid #F3B8B8; border-left:4px solid {COLORS['RedLine']};
            border-radius:8px; padding:16px 18px; margin:14px 0;">
  <div style="font-size:0.72rem; font-weight:800; color:{COLORS['RedLine']};
              text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px;">Failure State</div>
  <div style="font-weight:800; color:{COLORS['Text']}; margin-bottom:6px;">
    {html_lib.escape(title)}
  </div>
  <ul style="margin:8px 0 8px 18px; color:{COLORS['TextSec']}; line-height:1.55;">
    {failure_items}
  </ul>
  <div style="color:{COLORS['TextSec']}; line-height:1.55;">
    <strong>Recovery:</strong> {html_lib.escape(recovery)}
  </div>
</div>
"""
        )

    def v1_10_candidate_table(rows, columns, highlight_label: str = ""):
        header_html = "".join(
            f"<th style='text-align:left; padding:9px 10px; border-bottom:1px solid {COLORS['Border']};'>{html_lib.escape(label)}</th>"
            for _, label in columns
        )
        body_html = []
        for row in rows:
            feasible = bool(row.get("feasible", False))
            bg = "#F8FFFB" if feasible else "#FFF7F7"
            if row.get("label") == highlight_label:
                bg = "#FFF9E8"
            cells = []
            for key, _ in columns:
                value = row.get(key, "")
                if isinstance(value, bool):
                    rendered = v1_10_fmt_bool(value)
                elif isinstance(value, (tuple, list)):
                    rendered = "<br/>".join(html_lib.escape(str(item)) for item in value) or "none"
                elif key == "compressed_size_mb" and value != "":
                    rendered = v1_10_fmt_size_mb(float(value))
                elif key == "accuracy_drop_pct" and value != "":
                    rendered = f"{float(value):.2f}%"
                elif key == "inference_speedup" and value != "":
                    rendered = f"{float(value):.2f}x"
                elif key == "compression_ratio" and value != "":
                    rendered = f"{float(value):.2f}x"
                else:
                    rendered = html_lib.escape(str(value))
                cells.append(
                    f"<td style='vertical-align:top; padding:9px 10px; border-bottom:1px solid {COLORS['Border']};'>{rendered}</td>"
                )
            body_html.append(
                f"<tr style='background:{bg}; border-left:3px solid {v1_10_status_color(feasible)};'>{''.join(cells)}</tr>"
            )
        return mo.Html(
            f"""
<div style="overflow-x:auto; margin:14px 0; border:1px solid {COLORS['Border']};
            border-radius:8px; background:white;">
  <table style="border-collapse:collapse; width:100%; font-size:0.84rem;">
    <thead style="background:{COLORS['Surface2']}; color:{COLORS['Text']};">
      <tr>{header_html}</tr>
    </thead>
    <tbody>{''.join(body_html)}</tbody>
  </table>
</div>
"""
        )

    def v1_10_release_gate_card(result: dict[str, object]):
        release_ok = bool(result["release_ok"])
        color = COLORS["GreenLine"] if release_ok else COLORS["RedLine"]
        status = "Release gate passes" if release_ok else "Release gate blocks shipment"
        failures = tuple(result.get("failures", ()))
        warnings = tuple(result.get("warnings", ()))
        failure_html = (
            "".join(f"<li>{html_lib.escape(str(item))}</li>" for item in failures)
            if failures
            else "<li>All solver-backed guardrails pass for the selected anchor checks.</li>"
        )
        warning_html = "".join(f"<li>{html_lib.escape(str(item))}</li>" for item in warnings)
        if warning_html:
            warning_html = f"<div style='margin-top:8px;'><strong>Residual warning:</strong><ul>{warning_html}</ul></div>"
        return mo.Html(
            f"""
<div style="background:white; border:1px solid {COLORS['Border']}; border-left:4px solid {color};
            border-radius:8px; padding:16px 18px; margin:14px 0;">
  <div style="font-size:0.72rem; font-weight:800; color:{color}; text-transform:uppercase;
              letter-spacing:0.12em; margin-bottom:6px;">Release Gate</div>
  <div style="font-weight:800; color:{COLORS['Text']}; margin-bottom:8px;">{status}</div>
  <div style="color:{COLORS['TextSec']}; line-height:1.6;">
    <strong>Recipe:</strong> {html_lib.escape(str(result['selected_recipe']))}<br/>
    <strong>Primary metric evidence:</strong> {html_lib.escape(str(result['primary_metric_gain']))}<br/>
    <strong>Validation test:</strong> {html_lib.escape(str(result['validation_test']))}
    <ul style="margin:8px 0 0 18px;">{failure_html}</ul>
    {warning_html}
  </div>
</div>
"""
        )

    return (
        v1_10_candidate_table,
        v1_10_checkpoint_card,
        v1_10_failure_card,
        v1_10_fmt_size_mb,
        v1_10_metric_cards,
        v1_10_part_banner,
        v1_10_release_gate_card,
        v1_10_reveal_card,
        v1_10_status_color,
    )


@app.cell(hide_code=True)
def _(ACADEMIC_LAB_CSS, LAB_CSS, lab_header, lab_metadata, mo):
    mo.vstack(
        [
            LAB_CSS,
            ACADEMIC_LAB_CSS,
            lab_header(
                lab_metadata,
                (
                    "Use solver-backed compression candidates to decide what can actually "
                    "ship on the selected deployment track."
                ),
                chips=("Compression", "Guardrails", "Calibration", "Sparsity", "Release gate"),
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    chapter_10_recap,
    chapter_recap,
    lab_learning_objectives,
    lab_map,
    learning_objectives,
    mo,
    scenario_brief,
    track_arc_context,
    track_context,
    v1_10_track_picker,
    v1_10_track_profile,
    v1_10_variant,
):
    mo.vstack(
        [
            learning_objectives(lab_learning_objectives),
            chapter_recap(chapter_10_recap),
            v1_10_track_picker,
            track_context(v1_10_track_profile),
            track_arc_context(v1_10_track_profile, "v1_10_compression_paradox"),
            scenario_brief(
                "Scenario Brief",
                stakeholder=v1_10_variant.stakeholder,
                objective=v1_10_variant.objective,
                constraints={
                    "Workload": v1_10_variant.workload_summary,
                    "Primary metric": v1_10_variant.primary_metric,
                    "Release guardrail": v1_10_variant.guardrail_metric,
                    "Required validation": ", ".join(v1_10_variant.defaults["validation_tests"]),
                },
            ),
            lab_map(
                (
                    {
                        "part_id": "A",
                        "part": "A",
                        "concept": "Smaller Is Not Automatically Faster",
                        "question": "Which method still wins after track guardrails and hardware support are applied?",
                    },
                    {
                        "part_id": "B",
                        "part": "B",
                        "concept": "Pruning Has Structure",
                        "question": "When do zeros become speed instead of storage-only sparsity?",
                    },
                    {
                        "part_id": "C",
                        "part": "C",
                        "concept": "Distillation Trades Teacher Quality For Student Constraints",
                        "question": "When is a dense student deployable, and what teacher risk follows it?",
                    },
                    {
                        "part_id": "D",
                        "part": "D",
                        "concept": "Compression Strategy Depends On Binding Constraint And Evidence",
                        "question": "Which recipe survives the release gate?",
                    },
                    {
                        "part_id": "Synthesis",
                        "part": "Synthesis",
                        "concept": "Compression Contract",
                        "question": "What lesson carries into hardware acceleration?",
                    },
                ),
                {},
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        """
<div class="mlsysbook-panel mlsysbook-launch-panel">
  <div class="mlsysbook-section-label">Instructions</div>
  <h2>Recommended Reading</h2>
  <ul class="mlsysbook-list">
    <li><strong>Chapter 10: Model Compression</strong> -- quantization, calibration, pruning structure, distillation, and the Pareto frontier.</li>
    <li><strong>Chapter 11 preview</strong> -- hardware acceleration explains why supported kernels control speedup.</li>
  </ul>
</div>
"""
    )
    return


# ===========================================================================
# ZONE B: WIDGET DEFINITIONS
# ===========================================================================


@app.cell(hide_code=True)
def _(
    mo,
    v1_10_best_candidate_label,
    v1_10_bit_widths,
    v1_10_defaults,
    v1_10_lowest_feasible_bit_width,
    v1_10_method_label,
    v1_10_recipe_prediction_options,
    v1_10_validation_tests,
):
    _candidate_method_keys = tuple(v1_10_defaults["candidate_methods"]) + ("no_compression",)
    partA_pred = mo.ui.radio(
        options={
            "INT8 quantization": "int8_quantization",
            "Structured pruning": "structured_pruning",
            "Distillation": "distillation",
            "No compression / keep baseline": "no_compression",
        },
        label="Which single method is most likely to win for this track?",
    )
    partA_method = mo.ui.dropdown(
        options={v1_10_method_label(method): method for method in _candidate_method_keys},
        value=v1_10_method_label(_candidate_method_keys[0]),
        label="Method to inspect",
    )

    partC_pred = mo.ui.radio(
        options={
            "About 1x": "1x",
            "About 2x": "2x",
            "About 5x": "5x",
            "About 10x": "10x",
        },
        label="At 90 percent sparsity, what speedup do you expect?",
    )
    partC_sparsity_pct = mo.ui.slider(start=0, stop=90, value=90, step=10, label="Sparsity (%)")
    partC_sparsity_type = mo.ui.dropdown(
        options={
            "Unstructured": "unstructured",
            "Structured": "structured",
            "N:M / 2:4": "n_m",
        },
        value="Unstructured",
        label="Sparsity structure",
    )
    partC_decision = mo.ui.radio(
        options={
            "Use pruning": "prune",
            "Use distillation": "distill",
            "Reject pruning for this track": "reject",
        },
        value="Reject pruning for this track",
        label="Checkpoint decision",
    )
    partC_distill_pred = mo.ui.radio(
        options={
            "Teacher quality blocks release": "teacher_quality",
            "Student is too small": "student_capacity",
            "Validation coverage blocks release": "validation_coverage",
            "Dense student is ready to ship": "ship",
        },
        label="Which distillation risk most likely blocks release?",
    )
    partC_teacher_quality = mo.ui.dropdown(
        options={
            "Validated teacher": "validated",
            "Brittle teacher": "brittle",
            "Weak teacher": "weak",
        },
        value="Validated teacher",
        label="Teacher quality",
    )
    partC_student_scale = mo.ui.slider(
        start=20,
        stop=80,
        value=50,
        step=10,
        label="Student size (% of teacher)",
    )
    partC_distill_decision = mo.ui.radio(
        options={
            "Ship dense student": "ship",
            "Use as fallback only": "fallback",
            "Reject distillation": "reject",
        },
        value="Use as fallback only",
        label="Distillation checkpoint",
    )

    partD_pred = mo.ui.radio(
        options=v1_10_recipe_prediction_options,
        label="Which recipe survives this track's release gate?",
    )
    partD_quant_bit = mo.ui.dropdown(
        options={"No quantization": 0} | {f"{bit_width}-bit": int(bit_width) for bit_width in v1_10_bit_widths},
        value=f"{v1_10_lowest_feasible_bit_width}-bit" if v1_10_lowest_feasible_bit_width else "No quantization",
        label="Quantization",
    )
    partD_prune_choice = mo.ui.dropdown(
        options={
            "No pruning": "none",
            "50% structured pruning": "structured_50",
            "90% unstructured pruning": "unstructured_90",
            "50% N:M pruning": "nm_50",
            "90% structured pruning": "structured_90",
        },
        value="No pruning",
        label="Pruning",
    )
    partD_distillation = mo.ui.radio(
        options={"No dense-student fallback": "off", "Add dense-student fallback": "on"},
        value="No dense-student fallback",
        label="Distillation",
        inline=True,
    )
    partD_calibration = mo.ui.dropdown(
        options={
            "Layerwise PTQ": "layerwise_ptq",
            "Channelwise PTQ": "channelwise_ptq",
            "QAT required": "qat",
        },
        value="Channelwise PTQ",
        label="Calibration / QAT",
    )
    partD_validation = mo.ui.dropdown(
        options={test: test for test in v1_10_validation_tests},
        value=v1_10_validation_tests[0],
        label="Validation test",
    )
    partD_residual_risk = mo.ui.dropdown(
        options={
            "Unsupported fast path could force runtime fallback": "unsupported fast path could force runtime fallback",
            "Quality loss may appear only on track validation data": "quality loss may appear only on track validation data",
            "Sustained load may change latency, energy, or thermal behavior": "sustained load may change latency, energy, or thermal behavior",
        },
        value="Unsupported fast path could force runtime fallback",
        label="Residual risk",
    )
    return (
        partA_method,
        partA_pred,
        partC_decision,
        partC_pred,
        partC_distill_decision,
        partC_distill_pred,
        partC_sparsity_pct,
        partC_sparsity_type,
        partC_student_scale,
        partC_teacher_quality,
        partD_calibration,
        partD_distillation,
        partD_pred,
        partD_prune_choice,
        partD_quant_bit,
        partD_residual_risk,
        partD_validation,
    )


# ===========================================================================
# ZONE C: CONCEPT MODULE TABS
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    decision_flow,
    go,
    mo,
    partA_method,
    partA_pred,
    partC_decision,
    partC_distill_decision,
    partC_distill_pred,
    partC_pred,
    partC_sparsity_pct,
    partC_sparsity_type,
    partC_student_scale,
    partC_teacher_quality,
    partD_calibration,
    partD_distillation,
    partD_pred,
    partD_prune_choice,
    partD_quant_bit,
    partD_residual_risk,
    partD_validation,
    scenario_thread,
    v1_10_best_candidate_label,
    v1_10_best_candidate_row,
    v1_10_bit_widths,
    v1_10_carry_forward_implication,
    v1_10_candidate_rows,
    v1_10_candidate_table,
    v1_10_candidate_to_row,
    v1_10_checkpoint_card,
    v1_10_compression_solver,
    v1_10_compression_sweep,
    v1_10_defaults,
    v1_10_evaluate_distillation,
    v1_10_evaluate_recipe,
    v1_10_failure_card,
    v1_10_fmt_size_mb,
    v1_10_hardware,
    v1_10_lowest_feasible_bit_width,
    v1_10_method_label,
    v1_10_metric_cards,
    v1_10_model,
    v1_10_part_banner,
    v1_10_quant_label,
    v1_10_rejected_candidate_row,
    v1_10_rejected_method_label,
    v1_10_release_gate_card,
    v1_10_recipe_prediction_options,
    v1_10_reveal_card,
    v1_10_size_limit,
    v1_10_status_color,
    v1_10_track_profile,
    v1_10_variant,
):
    _track_label = v1_10_track_profile.label
    _guardrail = v1_10_variant.guardrail_metric
    _primary_metric = v1_10_variant.primary_metric
    _max_drop_pct = float(v1_10_defaults["max_accuracy_drop"]) * 100
    _min_speedup = float(v1_10_defaults["min_speedup"])
    _candidate_columns = (
        ("label", "Candidate"),
        ("compressed_size_mb", "Size"),
        ("compression_ratio", "Ratio"),
        ("accuracy_drop_pct", "Drop"),
        ("inference_speedup", "Speedup"),
        ("hardware_supported", "HW"),
        ("feasible", "Feasible"),
        ("binding_constraint", "Binding"),
    )

    def _prediction_label(options, value):
        for label, stored in options.items():
            if stored == value:
                return label
        return str(value)

    def _rows_for_method(method_key: str):
        if method_key == "int8_quantization":
            return tuple(
                row for row in v1_10_candidate_rows if row["method"] == "quantization" and row["target_bitwidth"] == 8
            )
        if method_key == "structured_pruning":
            return tuple(
                row for row in v1_10_candidate_rows if row["method"] == "pruning" and row["sparsity_type"] == "structured"
            )
        if method_key == "no_compression":
            return tuple(row for row in v1_10_candidate_rows if row["label"] == "FP32 baseline")
        return ()

    def build_part_a():
        items = [
            v1_10_part_banner(
                "A",
                "Smaller Is Not Automatically Faster",
                (
                    "The common prior is to choose the highest compression ratio. The solver shows "
                    "that the winning method depends on track guardrails and supported runtime paths."
                ),
                COLORS["BlueLine"],
            ),
            scenario_thread(
                f"{_track_label} feasibility review",
                (
                    f"{v1_10_variant.stakeholder} needs {_primary_metric} without violating "
                    f"{_guardrail}. The active model is {v1_10_model.name} on {v1_10_hardware.name}."
                ),
                callout=(
                    f"Size limit: {v1_10_size_limit.to('MB').magnitude:.3g} MB; "
                    f"max quality drop: {_max_drop_pct:.2f}%; min speedup: {_min_speedup:.2f}x."
                ),
            ),
            mo.md(
                """
### Scenario

Compression is not one operation. A method must reduce the resource that binds this
track, preserve quality, meet the speedup requirement, and run on a supported fast path.
"""
            ),
            partA_pred,
        ]
        if partA_pred.value is None:
            items.append(mo.callout(mo.md("Select a method prediction to reveal the solver-backed candidate table."), kind="warn"))
            return mo.vstack(items)

        items.extend(
            [
                mo.hstack([partA_method], justify="start"),
                decision_flow(
                    "Part A decision path",
                    (
                        "Pick a method",
                        "Resolve model and hardware from track",
                        "Run CompressionModel.sweep",
                        "Reject failed guardrails",
                    ),
                ),
            ]
        )
        selected_rows = _rows_for_method(partA_method.value)
        if not selected_rows and partA_method.value == "distillation":
            selected_rows = tuple(row for row in v1_10_candidate_rows if row["feasible"])
        highlight = selected_rows[0]["label"] if selected_rows else v1_10_best_candidate_label
        items.append(v1_10_candidate_table(v1_10_candidate_rows, _candidate_columns, highlight))

        best_row = v1_10_best_candidate_row or {}
        items.append(
            v1_10_metric_cards(
                (
                    (
                        "Best feasible",
                        v1_10_best_candidate_label,
                        str(best_row.get("binding_constraint", "none")),
                        COLORS["GreenLine"] if best_row else COLORS["RedLine"],
                    ),
                    (
                        "Size limit",
                        f"{v1_10_size_limit.to('MB').magnitude:.3g} MB",
                        str(v1_10_defaults["size_limit_ref"]),
                        COLORS["BlueLine"],
                    ),
                    (
                        "Hardware support",
                        "required" if v1_10_defaults["require_hardware_support"] else "optional",
                        "unsupported fast paths are infeasible",
                        COLORS["OrangeLine"],
                    ),
                )
            )
        )
        predicted = _prediction_label(
            {
                "INT8 quantization": "int8_quantization",
                "Structured pruning": "structured_pruning",
                "Distillation": "distillation",
                "No compression / keep baseline": "no_compression",
            },
            partA_pred.value,
        )
        consequence = (
            "The best candidate is feasible only because every guardrail passes."
            if best_row
            else "No candidate survived all guardrails; the track needs a new recipe or model."
        )
        items.append(
            v1_10_reveal_card(
                "Single-method feasibility",
                predicted,
                v1_10_best_candidate_label,
                consequence,
                ok=partA_pred.value in str(v1_10_best_candidate_label).lower().replace(" ", "_"),
            )
        )
        items.append(
            v1_10_checkpoint_card(
                "Part A feasibility decision",
                (
                    ("Predicted method", predicted),
                    ("Actual best feasible", v1_10_best_candidate_label),
                    ("Binding resource", _primary_metric),
                    ("Required validation", ", ".join(v1_10_variant.defaults["validation_tests"])),
                ),
                COLORS["BlueLine"],
            )
        )
        if partA_method.value == "distillation":
            items.append(
                mo.callout(
                    mo.md(
                        "**Distillation note.** `CompressionModel` does not invent dense-student physics. "
                        "Use distillation as a recommendation when solver-backed shrink-in-place "
                        "candidates fail hardware, quality, or speed guardrails."
                    ),
                    kind="warn",
                )
            )
        elif selected_rows and not selected_rows[0]["feasible"]:
            items.append(
                v1_10_failure_card(
                    "Selected method is infeasible for this track",
                    selected_rows[0]["guardrail_violations"],
                    "Change the method, choose a supported precision, or relax the recipe only after validation.",
                )
            )
        items.append(
            mo.accordion(
                {
                    "Math Peek: Compression feasibility": mo.md(
                        """
Compression ratio:

$$
\\text{ratio} = \\frac{\\text{original size}}{\\text{compressed size}}
$$

Solver release condition:

$$
\\text{feasible} =
\\text{size_ok} \\land \\text{quality_ok} \\land \\text{speed_ok} \\land \\text{hardware_supported}
$$

Source model: `CompressionModel.sweep()` evaluates all listed candidates against the
track's size limit, maximum accuracy drop, minimum speedup, and hardware support flag.
"""
                    )
                }
            )
        )
        return mo.vstack(items)

    def build_part_b():
        selected_candidate = v1_10_compression_solver.candidate(
            v1_10_model,
            v1_10_hardware,
            label=f"{partC_sparsity_pct.value}% {partC_sparsity_type.value} pruning",
            method="pruning",
            sparsity=float(partC_sparsity_pct.value) / 100,
            sparsity_type=partC_sparsity_type.value,
            size_limit=v1_10_size_limit,
            max_accuracy_drop=float(v1_10_defaults["max_accuracy_drop"]),
            min_speedup=float(v1_10_defaults["min_speedup"]),
            require_hardware_support=bool(v1_10_defaults["require_hardware_support"]),
        )
        selected_row = v1_10_candidate_to_row(selected_candidate)
        pruning_rows = tuple(row for row in v1_10_candidate_rows if row["method"] == "pruning")
        unstructured_90 = next(
            row for row in pruning_rows if row["sparsity_type"] == "unstructured" and row["sparsity_pct"] == 90
        )
        items = [
            v1_10_part_banner(
                "B",
                "Pruning Has Structure",
                (
                    "The naive prior says 90 percent sparse should mean about 10x faster. "
                    "The runtime only sees speed when the sparsity has exploitable structure."
                ),
                COLORS["RedLine"],
            ),
            scenario_thread(
                f"{_track_label} pruning escalation",
                (
                    f"{v1_10_variant.stakeholder} asks whether pruning can rescue the release. "
                    "The answer depends on zero structure, not only zero count."
                ),
                callout=f"Validation risk: {', '.join(v1_10_variant.defaults['validation_tests'])}.",
            ),
            mo.md(
                """
### Scenario

Unstructured zeros often save storage only. Structured pruning can create real
speedup, but it trades regularity against quality. N:M patterns only help when
the hardware supports the exact sparse-kernel shape.
"""
            ),
            partC_pred,
        ]
        if partC_pred.value is None:
            items.append(mo.callout(mo.md("Select a sparsity speedup prediction to unlock the pruning evidence."), kind="warn"))
            return mo.vstack(items)

        items.extend(
            [
                mo.hstack([partC_sparsity_pct, partC_sparsity_type, partC_decision], justify="start"),
                v1_10_candidate_table(
                    pruning_rows + (selected_row,),
                    (
                        ("label", "Pattern"),
                        ("compressed_size_mb", "Size"),
                        ("compression_ratio", "Ratio"),
                        ("accuracy_drop_pct", "Drop"),
                        ("inference_speedup", "Speedup"),
                        ("hardware_supported", "HW"),
                        ("feasible", "Feasible"),
                        ("binding_constraint", "Binding"),
                    ),
                    selected_row["label"],
                ),
            ]
        )
        items.append(
            v1_10_metric_cards(
                (
                    (
                        "Selected speedup",
                        f"{float(selected_row['inference_speedup']):.2f}x",
                        f"{partC_sparsity_type.value} at {partC_sparsity_pct.value}% sparsity",
                        v1_10_status_color(bool(selected_row["feasible"])),
                    ),
                    (
                        "90% unstructured",
                        f"{float(unstructured_90['inference_speedup']):.2f}x",
                        "storage savings without a fast path",
                        COLORS["RedLine"],
                    ),
                    (
                        "Checkpoint",
                        str(partC_decision.value),
                        "name the deployment risk, not only the ratio",
                        COLORS["BlueLine"],
                    ),
                )
            )
        )
        items.append(
            v1_10_reveal_card(
                "Sparsity speedup",
                _prediction_label(
                    {
                        "About 1x": "1x",
                        "About 2x": "2x",
                        "About 5x": "5x",
                        "About 10x": "10x",
                    },
                    partC_pred.value,
                ),
                f"90% unstructured pruning is {float(unstructured_90['inference_speedup']):.2f}x",
                (
                    "Zeros only become runtime speed when the kernel can exploit their structure."
                ),
                ok=partC_pred.value == "1x",
            )
        )
        items.append(
            v1_10_checkpoint_card(
                "Part B structure decision",
                (
                    ("Predicted 90% speedup", _prediction_label({"About 1x": "1x", "About 2x": "2x", "About 5x": "5x", "About 10x": "10x"}, partC_pred.value)),
                    ("Actual unstructured", f"{float(unstructured_90['inference_speedup']):.2f}x"),
                    ("Checkpoint choice", str(partC_decision.value)),
                    ("Validation risk", ", ".join(v1_10_variant.defaults["validation_tests"])),
                ),
                COLORS["RedLine"],
            )
        )
        if not selected_row["feasible"]:
            items.append(
                v1_10_failure_card(
                    "Selected sparse recipe fails",
                    selected_row["guardrail_violations"],
                    "Lower sparsity, switch to structured or N:M support, or use a dense student fallback.",
                )
            )
        items.append(
            mo.accordion(
                {
                    "Math Peek: Pruning ratio and hardware branch": mo.md(
                        """
Theoretical pruning ratio:

$$
\\text{pruning_ratio} = \\frac{1}{1 - s}
$$

Runtime branch:

$$
\\text{speedup} =
\\begin{cases}
1.0, & \\text{unstructured on dense kernels} \\\\
\\frac{1}{1-s}, & \\text{structured pruning} \\\\
2.0, & \\text{supported 2:4 N:M near 50 percent sparsity}
\\end{cases}
$$

Source model: `CompressionModel.candidate()` normalizes the sparsity type, checks
hardware support, and returns guardrail violations for the selected pattern.
"""
                    )
                }
            )
        )
        return mo.vstack(items)

    def build_part_c():
        distill_result = v1_10_evaluate_distillation(
            partC_teacher_quality.value,
            int(partC_student_scale.value),
            partD_validation.value,
        )
        items = [
            v1_10_part_banner(
                "C",
                "Distillation Trades Teacher Quality For Student Constraints",
                (
                    "A dense student can be easier to deploy than a sparse teacher, but the "
                    "student can only inherit what the teacher and distillation data can teach."
                ),
                COLORS["OrangeLine"],
            ),
            scenario_thread(
                f"{_track_label} dense-student review",
                (
                    f"{v1_10_variant.stakeholder} asks whether a smaller dense student can "
                    f"replace {v1_10_model.name} while preserving {_guardrail}."
                ),
                callout="The trade moves cost from repeated inference into teacher choice, soft-target generation, and validation.",
            ),
            mo.md(
                """
### Scenario

Distillation trains a smaller dense student from a larger teacher. It can remove
deployment cost without depending on sparse kernels, but it is not lossless: a
weak teacher, biased soft targets, or an undersized student can fail the release
quality guardrail.
"""
            ),
            partC_distill_pred,
        ]
        if partC_distill_pred.value is None:
            items.append(mo.callout(mo.md("Select a distillation-risk prediction to unlock the dense-student evidence."), kind="warn"))
            return mo.vstack(items)

        items.extend(
            [
                mo.hstack([partC_teacher_quality, partC_student_scale, partC_distill_decision], justify="start"),
                mo.hstack([partD_validation], justify="start"),
            ]
        )
        items.append(
            v1_10_metric_cards(
                (
                    (
                        "Student size",
                        f"{int(partC_student_scale.value)}%",
                        f"{float(distill_result['compression_ratio']):.2f}x smaller dense model",
                        COLORS["BlueLine"],
                    ),
                    (
                        "Quality drop",
                        f"{float(distill_result['estimated_accuracy_drop_pct']):.2f}%",
                        f"guardrail allows {_max_drop_pct:.2f}%",
                        v1_10_status_color(bool(distill_result["release_ok"])),
                    ),
                    (
                        "Latency speedup",
                        f"{float(distill_result['latency_speedup']):.2f}x",
                        "dense student estimate, not sparse-kernel physics",
                        COLORS["OrangeLine"],
                    ),
                )
            )
        )
        items.append(
            v1_10_reveal_card(
                "Distillation release risk",
                _prediction_label(
                    {
                        "Teacher quality blocks release": "teacher_quality",
                        "Student is too small": "student_capacity",
                        "Validation coverage blocks release": "validation_coverage",
                        "Dense student is ready to ship": "ship",
                    },
                    partC_distill_pred.value,
                ),
                "release passes" if distill_result["release_ok"] else "; ".join(distill_result["failures"]),
                (
                    "Distillation trades teacher quality and one-time training cost for a deployable dense student."
                ),
                ok=(partC_distill_pred.value == "ship") == bool(distill_result["release_ok"]),
            )
        )
        items.append(
            v1_10_checkpoint_card(
                "Part C distillation decision",
                (
                    ("Teacher quality", str(partC_teacher_quality.value)),
                    ("Student constraint", f"{int(partC_student_scale.value)}% of teacher"),
                    ("Checkpoint choice", str(partC_distill_decision.value)),
                    ("Validation test", str(partD_validation.value)),
                ),
                COLORS["OrangeLine"],
            )
        )
        if not distill_result["release_ok"]:
            items.append(
                v1_10_failure_card(
                    "Dense-student release fails",
                    distill_result["failures"],
                    "Improve the teacher, choose a larger student, or select a validation test that directly covers quality and recall.",
                )
            )
        items.append(
            mo.accordion(
                {
                    "Math Peek: Distillation capacity and teacher risk": mo.md(
                        """
Dense-student compression:

$$
\\text{student compression ratio} = \\frac{1}{\\text{student scale}}
$$

Local risk model:

$$
\\Delta \\text{quality} =
\\text{teacher penalty} + \\max(0, 0.40 - \\text{student scale}) \\cdot 0.05
$$

Source model: `CompressionModel` has no trained-student physics, so this
notebook-local risk card only checks the chapter claim that teacher quality and
student capacity limit deployability. The final release still requires the
track validation test selected for Part D.
"""
                    )
                }
            )
        )
        return mo.vstack(items)

    def build_part_d():
        recipe_result = v1_10_evaluate_recipe(
            int(partD_quant_bit.value),
            partD_prune_choice.value,
            partD_distillation.value,
            partD_calibration.value,
            partD_validation.value,
        )
        items = [
            v1_10_part_banner(
                "D",
                "Recipe Frontier And Release Gate",
                (
                    "A leaderboard candidate is not a release. A compression deployment is a "
                    "recipe plus validation evidence and residual risk."
                ),
                COLORS["GreenLine"],
                duration="12-15 min",
            ),
            scenario_thread(
                f"{_track_label} release review",
                (
                    "The review board wants the recipe you would ship, the guardrail that "
                    "could block it, and the validation test you will run first."
                ),
                callout=f"Primary metric: {_primary_metric}; guardrail: {_guardrail}.",
            ),
            partD_pred,
        ]
        if partD_pred.value is None:
            items.append(mo.callout(mo.md("Select the release-gate prediction to unlock the recipe frontier."), kind="warn"))
            return mo.vstack(items)

        fig = go.Figure()
        for row in v1_10_candidate_rows:
            color = COLORS["GreenLine"] if row["feasible"] else COLORS["RedLine"]
            symbol = "diamond" if row["pareto_status"] == "frontier" else "circle"
            fig.add_trace(
                go.Scatter(
                    x=[row["compressed_size_mb"]],
                    y=[row["inference_speedup"]],
                    mode="markers+text",
                    marker=dict(size=12, color=color, symbol=symbol),
                    text=[row["label"]],
                    textposition="top center",
                    name=row["label"],
                    showlegend=False,
                )
            )
        fig.update_layout(
            height=390,
            xaxis=dict(title="Compressed size (MB)", type="log"),
            yaxis=dict(title="Inference speedup (x)"),
        )
        apply_plotly_theme(fig)
        items.extend(
            [
                mo.as_html(fig),
                v1_10_candidate_table(
                    v1_10_candidate_rows,
                    (
                        ("label", "Candidate"),
                        ("compressed_size_mb", "Size"),
                        ("accuracy_drop_pct", "Drop"),
                        ("inference_speedup", "Speedup"),
                        ("pareto_status", "Pareto"),
                        ("feasible", "Feasible"),
                        ("binding_constraint", "Binding"),
                    ),
                    v1_10_best_candidate_label,
                ),
                mo.hstack(
                    [partD_quant_bit, partD_prune_choice, partD_distillation],
                    justify="start",
                ),
                mo.hstack(
                    [partD_calibration, partD_validation, partD_residual_risk],
                    justify="start",
                ),
                v1_10_release_gate_card(recipe_result),
            ]
        )
        items.append(
            v1_10_reveal_card(
                "Release recipe",
                _prediction_label(v1_10_recipe_prediction_options, partD_pred.value),
                f"Best feasible solver recipe: {v1_10_best_candidate_label}",
                (
                    "The release gate accepts only candidates whose size, quality, speed, "
                    "hardware support, calibration, and validation story all survive."
                ),
                ok=partD_pred.value == "best",
            )
        )
        items.append(
            v1_10_checkpoint_card(
                "Part D release decision",
                (
                    ("Selected recipe", str(recipe_result["selected_recipe"])),
                    ("Rejected method", v1_10_rejected_method_label),
                    ("Quality guardrail", _guardrail),
                    ("Carry forward", v1_10_carry_forward_implication),
                ),
                COLORS["GreenLine"],
            )
        )
        if not recipe_result["release_ok"]:
            items.append(
                v1_10_failure_card(
                    "Recipe validation failed",
                    recipe_result["failures"],
                    "Adjust the recipe controls until the release gate card turns green.",
                )
            )
        items.append(
            mo.accordion(
                {
                    "Math Peek: Multi-constraint recipe feasibility": mo.md(
                        """
Release condition:

$$
\\text{ship} =
\\text{size_ok} \\land \\text{quality_ok} \\land \\text{speed_ok}
\\land \\text{hardware_supported} \\land \\text{validation_selected}
$$

Dominance condition:

$$
a \\prec b \\quad \\text{if candidate a is no worse in size, speed, and quality, and better in at least one.}
$$

Source model: `CompressionModel.sweep()` marks dominated and frontier candidates.
The local recipe gate reuses `CompressionModel.candidate()` for each selected
quantization or pruning anchor and adds calibration and validation checks.
"""
                    )
                }
            )
        )
        return mo.vstack(items)

    def build_synthesis():
        recipe_result = v1_10_evaluate_recipe(
            int(partD_quant_bit.value),
            partD_prune_choice.value,
            partD_distillation.value,
            partD_calibration.value,
            partD_validation.value,
        )
        rejected_reason = (
            "; ".join(str(item) for item in v1_10_rejected_candidate_row.get("guardrail_violations", ()))
            if v1_10_rejected_candidate_row
            else "no rejected solver candidate for this track"
        )
        return mo.vstack(
            [
                v1_10_part_banner(
                    "S",
                    "Synthesis",
                    (
                        "The invariant across the parts is that compression is a deployment "
                        "contract, not a model-size scoreboard."
                    ),
                    COLORS["BlueLine"],
                    duration="5-8 min",
                ),
                mo.Html(
                    f"""
<div style="background:#F7FAFF; border:1px solid #C9D8EE; border-left:4px solid {COLORS['BlueLine']};
            border-radius:8px; padding:22px 26px; margin:16px 0;">
  <div style="font-size:0.72rem; font-weight:800; color:{COLORS['BlueLine']};
              text-transform:uppercase; letter-spacing:0.12em; margin-bottom:10px;">
    Key Takeaways
  </div>
  <div style="font-size:0.92rem; color:{COLORS['Text']}; line-height:1.75;">
    <div><strong>1. Compression helps only if it attacks the binding resource.</strong>
      The best candidate for {_track_label} is {v1_10_best_candidate_label}, not necessarily the smallest candidate.</div>
    <div style="margin-top:8px;"><strong>2. Hardware support is part of the contract.</strong>
      Rejected method: {v1_10_rejected_method_label} ({rejected_reason}).</div>
    <div style="margin-top:8px;"><strong>3. Final recipes require validation and residual risk.</strong>
      This recipe records {_primary_metric} as the binding resource, {_guardrail} as the quality guardrail, and "{partD_residual_risk.value}" as the risk to test.</div>
    <div style="margin-top:8px;"><strong>4. Carry-forward implication.</strong>
      {v1_10_carry_forward_implication}</div>
  </div>
</div>
"""
                ),
                mo.Html(
                    f"""
<div style="display:flex; gap:16px; margin:8px 0 16px 0; flex-wrap:wrap;">
  <div style="flex:1; min-width:280px; background:white; border:1px solid {COLORS['Border']};
              border-radius:8px; padding:18px 22px;">
    <div style="font-size:0.7rem; font-weight:800; color:{COLORS['BlueLine']};
                text-transform:uppercase; letter-spacing:0.12em; margin-bottom:8px;">
      What's Next
    </div>
    <div style="font-size:0.88rem; color:{COLORS['TextSec']}; line-height:1.6;">
      <strong>Lab 11:</strong> Hardware acceleration and roofline analysis explain why a supported fast path controls whether compression turns into speedup.
    </div>
  </div>
  <div style="flex:1; min-width:280px; background:white; border:1px solid {COLORS['Border']};
              border-radius:8px; padding:18px 22px;">
    <div style="font-size:0.7rem; font-weight:800; color:{COLORS['GreenLine']};
                text-transform:uppercase; letter-spacing:0.12em; margin-bottom:8px;">
      Carry-Forward Ledger
    </div>
    <div style="font-size:0.88rem; color:{COLORS['TextSec']}; line-height:1.6;">
      The ledger saves the selected recipe, rejected method, binding resource, quality guardrail, validation test, and residual risk for later labs.
    </div>
  </div>
</div>
"""
                ),
                mo.accordion(
                    {
                        "Self-check": mo.md(
                            """
1. Why can a smaller model still be infeasible?
2. Why does unstructured sparsity usually save storage but not latency?
3. What teacher or student constraint can make distillation unsafe?
4. What validation evidence would you demand before release?
"""
                        )
                    }
                ),
            ]
        )

    tabs = mo.ui.tabs(
        {
            "Part A: Smaller vs. Faster": build_part_a(),
            "Part B: Pruning Structure": build_part_b(),
            "Part C: Distillation Contract": build_part_c(),
            "Part D: Strategy Gate": build_part_d(),
            "Synthesis": build_synthesis(),
        }
    )
    tabs
    return


# ===========================================================================
# ZONE D: REPORT AND LEDGER
# ===========================================================================


@app.cell(hide_code=True)
def _(
    big_takeaways,
    build_lab_report,
    lab_big_takeaways,
    lab_learning_objectives,
    lab_metadata,
    mo,
    partA_method,
    partA_pred,
    partC_decision,
    partC_distill_decision,
    partC_distill_pred,
    partC_pred,
    partC_sparsity_pct,
    partC_sparsity_type,
    partC_student_scale,
    partC_teacher_quality,
    partD_calibration,
    partD_distillation,
    partD_pred,
    partD_prune_choice,
    partD_quant_bit,
    partD_residual_risk,
    partD_validation,
    report_export_panel,
    v1_10_best_candidate_label,
    v1_10_carry_forward_implication,
    v1_10_candidate_rows,
    v1_10_compression_sweep,
    v1_10_evaluate_recipe,
    v1_10_lowest_feasible_bit_width,
    v1_10_rejected_candidate_row,
    v1_10_rejected_method_label,
    v1_10_track_profile,
    v1_10_variant,
):
    _recipe_result = v1_10_evaluate_recipe(
        int(partD_quant_bit.value),
        partD_prune_choice.value,
        partD_distillation.value,
        partD_calibration.value,
        partD_validation.value,
    )
    predictions = {
        "Part A - predicted method": partA_pred.value,
        "Part B - predicted 90 percent sparsity speedup": partC_pred.value,
        "Part C - predicted distillation risk": partC_distill_pred.value,
        "Part D - predicted release recipe": partD_pred.value,
    }
    recorded_predictions = {key: value for key, value in predictions.items() if value is not None}
    report = build_lab_report(
        lab_metadata,
        track=v1_10_track_profile.track_id,
        scenario=v1_10_variant.scenario_id,
        learning_objectives=lab_learning_objectives,
        predictions=recorded_predictions,
        knob_settings={
            "method_inspected": partA_method.value,
            "sparsity_type": partC_sparsity_type.value,
            "sparsity_pct": partC_sparsity_pct.value,
            "pruning_checkpoint": partC_decision.value,
            "distillation_teacher_quality": partC_teacher_quality.value,
            "distillation_student_scale_pct": int(partC_student_scale.value),
            "distillation_checkpoint": partC_distill_decision.value,
            "recipe_quantization": int(partD_quant_bit.value),
            "recipe_pruning": partD_prune_choice.value,
            "recipe_distillation": partD_distillation.value,
            "recipe_calibration": partD_calibration.value,
        },
        binding_constraints={
            "binding_resource": v1_10_variant.primary_metric,
            "quality_guardrail": v1_10_variant.guardrail_metric,
            "recipe_binding_constraint": _recipe_result["binding_constraint"],
            "rejected_method": v1_10_rejected_method_label,
            "release_ok": _recipe_result["release_ok"],
        },
        evidence_summary={
            "track": v1_10_track_profile.label,
            "primary_metric": v1_10_variant.primary_metric,
            "guardrail_metric": v1_10_variant.guardrail_metric,
            "best_candidate_label": v1_10_best_candidate_label,
            "rejected_method": v1_10_rejected_method_label,
            "rejected_method_reason": (
                tuple(v1_10_rejected_candidate_row.get("guardrail_violations", ()))
                if v1_10_rejected_candidate_row
                else ()
            ),
            "lowest_feasible_bit_width": v1_10_lowest_feasible_bit_width,
            "frontier_labels": tuple(v1_10_compression_sweep.frontier_labels),
            "dominated_labels": tuple(v1_10_compression_sweep.dominated_labels),
            "release_gate": _recipe_result,
        },
        result_snapshot={
            "compression_candidates": v1_10_candidate_rows,
            "best_candidate_label": v1_10_best_candidate_label,
            "release_gate": _recipe_result,
        },
        final_decision={
            "selected_recipe": _recipe_result["selected_recipe"],
            "binding_resource": v1_10_variant.primary_metric,
            "rejected_method": v1_10_rejected_method_label,
            "quality_guardrail": v1_10_variant.guardrail_metric,
            "validation_test": partD_validation.value,
            "residual_risk": partD_residual_risk.value,
            "carry_forward_implication": v1_10_carry_forward_implication,
            "pruning_checkpoint": partC_decision.value,
            "distillation_checkpoint": partC_distill_decision.value,
        },
        big_takeaways=lab_big_takeaways,
        reflections={
            "compression_contract": (
                "The selected recipe must pass size, quality, speed, hardware support, "
                "calibration, and validation checks."
            )
        },
        residual_risk=partD_residual_risk.value,
        source_trace={
            "track_profile": v1_10_track_profile.track_id,
            "scenario_id": v1_10_variant.scenario_id,
            "model_ref": v1_10_variant.model_ref,
            "hardware_ref": v1_10_variant.hardware_ref,
            "variant_defaults": dict(v1_10_variant.defaults),
            "solver": "CompressionModel.sweep and CompressionModel.candidate",
            "notebook_local_logic": "UI checkpoint cards, recipe composition, and dense-student risk card; CompressionModel owns quantization and pruning physics.",
        },
    )
    mo.vstack(
        [
            big_takeaways(lab_big_takeaways),
            mo.Html(
                """
<div class="mlsysbook-panel mlsysbook-report-panel">
  <h2>Download Report</h2>
  <p class="mlsysbook-source-summary">
    This report records the solver-backed candidate frontier, structured predictions,
    selected recipe, release-gate result, validation test, and residual risk.
  </p>
</div>
"""
            ),
            report_export_panel(report),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    partA_method,
    partA_pred,
    partC_decision,
    partC_distill_decision,
    partC_distill_pred,
    partC_student_scale,
    partC_sparsity_pct,
    partC_sparsity_type,
    partC_teacher_quality,
    partD_calibration,
    partD_distillation,
    partD_pred,
    partD_prune_choice,
    partD_quant_bit,
    partD_residual_risk,
    partD_validation,
    v1_10_best_candidate_label,
    v1_10_carry_forward_implication,
    v1_10_candidate_rows,
    v1_10_evaluate_recipe,
    v1_10_lowest_feasible_bit_width,
    v1_10_rejected_candidate_row,
    v1_10_rejected_method_label,
    v1_10_track_id,
    v1_10_variant,
):
    _recipe_result = v1_10_evaluate_recipe(
        int(partD_quant_bit.value),
        partD_prune_choice.value,
        partD_distillation.value,
        partD_calibration.value,
        partD_validation.value,
    )
    complete = all(
        prediction.value is not None
        for prediction in (partA_pred, partC_pred, partC_distill_pred, partD_pred)
    )
    ledger.save(
        track=v1_10_track_id,
        chapter=10,
        design={
            "lab": "model_compress",
            "lab_id": "v1_10_compression_paradox",
            "completed": complete,
            "track_id": v1_10_track_id,
            "scenario_id": v1_10_variant.scenario_id,
            "binding_resource": v1_10_variant.primary_metric,
            "quality_guardrail": v1_10_variant.guardrail_metric,
            "predicted_method": partA_pred.value,
            "method_inspected": partA_method.value,
            "best_candidate_label": v1_10_best_candidate_label,
            "rejected_method": v1_10_rejected_method_label,
            "rejected_method_reason": (
                tuple(v1_10_rejected_candidate_row.get("guardrail_violations", ()))
                if v1_10_rejected_candidate_row
                else ()
            ),
            "selected_precision": int(partD_quant_bit.value),
            "lowest_feasible_bit_width": int(v1_10_lowest_feasible_bit_width or 0),
            "predicted_pruning_speedup": partC_pred.value,
            "selected_sparsity_type": partC_sparsity_type.value,
            "selected_sparsity_pct": int(partC_sparsity_pct.value),
            "sparsity_checkpoint": partC_decision.value,
            "predicted_distillation_risk": partC_distill_pred.value,
            "distillation_teacher_quality": partC_teacher_quality.value,
            "distillation_student_scale_pct": int(partC_student_scale.value),
            "distillation_checkpoint": partC_distill_decision.value,
            "selected_recipe": _recipe_result["selected_recipe"],
            "recipe_quantization": int(partD_quant_bit.value),
            "recipe_pruning": partD_prune_choice.value,
            "recipe_distillation": partD_distillation.value,
            "recipe_calibration": partD_calibration.value,
            "binding_constraint": _recipe_result["binding_constraint"],
            "validation_test": partD_validation.value,
            "residual_risk": partD_residual_risk.value,
            "release_ok": _recipe_result["release_ok"],
            "primary_metric_gain": _recipe_result["primary_metric_gain"],
            "carry_forward_implication": v1_10_carry_forward_implication,
            "compression_candidates": v1_10_candidate_rows,
        },
    )
    status = "COMPLETE" if complete else "IN PROGRESS"
    mo.Html(
        f"""
<div class="lab-hud">
  <span class="hud-label">LAB</span>
  <span class="hud-value">10 &middot; Model Compression</span>
  <span style="flex:1;"></span>
  <span class="hud-label">CH</span>
  <span class="hud-value">10</span>
  <span class="hud-label">TRACK</span>
  <span class="hud-value">{v1_10_track_id}</span>
  <span class="hud-label">STATUS</span>
  <span class="hud-active">{status}</span>
</div>
"""
    )
    return


if __name__ == "__main__":
    app.run()
