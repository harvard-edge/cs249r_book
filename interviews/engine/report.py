"""
Interactive HTML report generator for StaffML corpus analysis.

Generates a self-contained HTML report with:
1. UMAP embedding visualization (interactive plotly)
2. Coverage heatmap (track × level)
3. Topic cluster analysis
4. Gap detection results
5. Corpus statistics

Usage:
    python3 -m engine.cli report --html
    # Opens interviews/_reports/corpus_report.html
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Optional

import numpy as np


def generate_html_report(
    output_path: Optional[Path] = None,
    open_browser: bool = True,
) -> Path:
    """Generate a comprehensive HTML report of the corpus.

    Returns the path to the generated report.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False

    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    # ── Load corpus ───────────────────────────────────────────────────────
    corpus_path = Path(__file__).parent.parent / "corpus.json"
    if not corpus_path.exists():
        raise FileNotFoundError("corpus.json not found. Run build_corpus.py first.")

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    # Normalize levels
    for q in corpus:
        if q.get("level") in ("L6", "L6%2B"):
            q["level"] = "L6+"

    # ── Build text for embedding ──────────────────────────────────────────
    from .embed import corpus_to_texts, embed_texts
    texts = corpus_to_texts(corpus)

    # Try sentence-transformer embeddings (nomic), fall back to TF-IDF
    try:
        embeddings_full = embed_texts(texts)
        used_st = True
    except Exception:
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        embeddings_full = vectorizer.fit_transform(texts).toarray()
        used_st = False

    # ── UMAP reduction ────────────────────────────────────────────────────
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        embedding_2d = reducer.fit_transform(embeddings_full)
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        embedding_2d = pca.fit_transform(embeddings_full)

    # ── BERTopic auto-clustering ──────────────────────────────────────────
    bertopic_html = ""
    discovered_topics = []
    try:
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import CountVectorizer

        # Custom vectorizer that strips stopwords + short tokens — prevents
        # "the", "at", "is" from dominating topic representations
        vectorizer = CountVectorizer(
            stop_words="english",
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),  # Capture bigrams like "kv cache", "tensor core"
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]{2,}\b",  # Min 3 chars, starts with letter
        )

        # Boost discriminative terms with BM25-style weighting
        ctfidf = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

        topic_model = BERTopic(
            embedding_model=None,  # We provide pre-computed embeddings
            umap_model=umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42) if HAS_UMAP else None,
            hdbscan_model=KMeans(n_clusters=20, random_state=42),
            vectorizer_model=vectorizer,
            ctfidf_model=ctfidf,
            nr_topics=20,
            verbose=False,
        )
        topics, probs = topic_model.fit_transform(texts, embeddings=embeddings_full)

        # Rename topics from "0_battery_power_energy" to readable labels
        # Use the top 2-3 terms to create a human name
        topic_info = topic_model.get_topic_info()
        label_map = {}
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                continue
            # Get the top terms for this topic
            top_terms = topic_model.get_topic(tid)
            if top_terms:
                # Take top 2-3 most discriminative terms
                top_words = [w for w, _ in top_terms[:3]]
                label = " / ".join(top_words).title()
            else:
                label = f"Cluster {tid}"
            label_map[tid] = f"{label} ({row['Count']}q)"
            discovered_topics.append({
                "id": tid,
                "name": label,
                "count": row["Count"],
                "representation": row.get("Representation", []),
            })

        topic_model.set_topic_labels(label_map)

        # --- Actionable cluster summary (replaces noisy bar charts) ---
        # Sort clusters by size, build a clean ranked table
        sorted_topics = sorted(discovered_topics, key=lambda t: -t["count"])

        cluster_rows = ""
        for t in sorted_topics:
            terms = ", ".join(t.get("representation", [])[:5]) if t.get("representation") else t["name"]
            cluster_rows += f"""
            <tr>
                <td style="font-weight:600;">{t['name']}</td>
                <td style="text-align:center;">{t['count']}</td>
                <td style="color:#666;font-size:0.9em;">{terms}</td>
            </tr>"""

        bertopic_html += f"""
        <div class="section">
            <h2>Discovered Topic Clusters</h2>
            <div class="desc">BERTopic automatically grouped all {total} questions into {len(discovered_topics)} semantic clusters. Larger clusters may be over-represented; small or missing clusters are generation targets.</div>
            <table style="width:100%;border-collapse:collapse;margin-top:15px;">
                <thead>
                    <tr style="border-bottom:2px solid #ddd;text-align:left;">
                        <th style="padding:8px 12px;">Cluster</th>
                        <th style="padding:8px 12px;text-align:center;">Questions</th>
                        <th style="padding:8px 12px;">Defining Terms</th>
                    </tr>
                </thead>
                <tbody>
                    {cluster_rows}
                </tbody>
            </table>
        </div>
        """

        # --- Document map: keep this, it's the useful visual ---
        try:
            fig_bt_docs = topic_model.visualize_documents(
                texts,
                embeddings=embeddings_full,
                custom_labels=True,
                hide_annotations=True,
            )
            bertopic_html += f"""
            <div class="section">
                <h2>Question Landscape</h2>
                <div class="desc">Each point is a question. Color = topic cluster. Dense areas = well-covered topics. Empty space = generation opportunities. Hover any point for details.</div>
                <div id="bertopic-docs"></div>
            </div>
            <script>Plotly.newPlot('bertopic-docs', {fig_bt_docs.to_json()}.data, {fig_bt_docs.to_json()}.layout, {{responsive: true}});</script>
            """
        except Exception:
            pass

    except ImportError:
        pass  # BERTopic not installed
    except Exception:
        pass  # BERTopic failed, continue without it

    # ── Textbook concept overlay (the "target space") ───────────────────
    # Embed chapter concepts from TOPIC_MAP.md and GENERATION_TARGETS
    # into the same space as questions, so we can see coverage gaps
    from .embed import extract_concepts_from_topic_map

    concept_texts = extract_concepts_from_topic_map()

    # Also pull from generate.py's GENERATION_TARGETS for richer coverage
    try:
        sys_path = str(Path(__file__).parent.parent)
        import importlib.util
        spec = importlib.util.spec_from_file_location("generate_targets", Path(__file__).parent.parent / "generate.py")
        gen_mod = importlib.util.module_from_spec(spec)
        # Don't execute (has rich imports), just extract the dict manually
    except Exception:
        pass

    # Add manually curated high-level concepts per competency area
    competency_concepts = [
        "GPU roofline model: compute-bound vs memory-bound analysis",
        "VRAM memory accounting: model weights, optimizer states, activations, KV-cache",
        "Mixed precision training: FP16, BF16, loss scaling, underflow",
        "Transformer scaling laws: compute-optimal training, Chinchilla",
        "LLM serving latency: TTFT, TPOT, continuous batching, PagedAttention",
        "Power and cooling: GPU TDP, PUE, liquid cooling economics",
        "Quantization: INT8, INT4, calibration, accuracy trade-offs",
        "Distributed training: data parallelism, tensor parallelism, pipeline parallelism",
        "AllReduce communication: ring, tree, InfiniBand, NVLink bandwidth",
        "Model compression: pruning, distillation, knowledge transfer",
        "Edge deployment: TOPS/W, thermal envelope, DVFS, real-time deadlines",
        "Mobile inference: NPU delegation, ANR timeouts, CoreML, TFLite",
        "TinyML: SRAM tensor arena, flash storage, CMSIS-NN, Cortex-M SIMD",
        "Microcontroller constraints: no FPU, watchdog timers, DMA, interrupt-driven",
        "Fleet management: OTA updates, firmware versioning, A/B partitions",
        "Data drift detection: KL divergence, PSI, training-serving skew",
        "Feature store consistency: online vs offline, time semantics",
        "Federated learning: on-device privacy, differential privacy",
        "Prompt injection and model security: adversarial attacks, guardrails",
        "Production ML debugging: accuracy regression triage, silent failures",
        "Kubernetes autoscaling for ML: cold start, GPU scheduling",
        "Checkpoint and fault tolerance: async checkpointing, recovery time",
        "Energy harvesting and duty cycling for TinyML",
        "Sensor fusion and calibration for edge perception",
        "Video pipeline optimization: ISP, frame rate, resolution trade-offs",
    ]

    # Combine and deduplicate
    all_concepts = list(set(concept_texts + competency_concepts))

    # Embed concepts with the same model
    try:
        concept_embeddings = embed_texts(all_concepts)
        # Project into same UMAP space
        concept_2d = reducer.transform(concept_embeddings)
        has_concept_overlay = True
    except Exception:
        has_concept_overlay = False

    # ── Data prep ─────────────────────────────────────────────────────────
    tracks = [q.get("track", "unknown") for q in corpus]
    levels = [q.get("level", "?") for q in corpus]
    titles = [q.get("title", "?") for q in corpus]
    topics = [q.get("topic", "?") for q in corpus]
    scenarios = [q.get("scenario", "")[:120] + "..." for q in corpus]

    all_tracks = sorted(set(tracks))
    all_levels = ["L1", "L2", "L3", "L4", "L5", "L6+"]

    track_colors = {
        "cloud": "#4a90c4", "edge": "#c87b2a",
        "mobile": "#3d9e5a", "tinyml": "#a31f34",
        "global": "#999",
    }
    level_colors = {
        "L1": "#2ecc71", "L2": "#3498db", "L3": "#27ae60",
        "L4": "#2980b9", "L5": "#f39c12", "L6+": "#e74c3c",
    }

    # ── Figure 1: UMAP by Track ───────────────────────────────────────────
    fig_umap_track = go.Figure()
    for track in all_tracks:
        mask = [i for i, t in enumerate(tracks) if t == track]
        if not mask:
            continue
        fig_umap_track.add_trace(go.Scatter(
            x=[embedding_2d[i, 0] for i in mask],
            y=[embedding_2d[i, 1] for i in mask],
            mode="markers",
            name=track,
            marker=dict(
                size=6,
                color=track_colors.get(track, "#999"),
                opacity=0.7,
            ),
            text=[f"<b>{titles[i]}</b><br>{levels[i]} · {topics[i]}<br>{scenarios[i]}" for i in mask],
            hoverinfo="text",
        ))

    # Overlay textbook concepts as red diamonds — shows target space
    if has_concept_overlay:
        # Compute distance from each concept to its nearest question
        from scipy.spatial.distance import cdist
        dists = cdist(concept_2d, embedding_2d, metric="euclidean")
        min_dists = dists.min(axis=1)
        # Normalize: 0 = on top of a question, 1 = far from any question
        max_dist = max(min_dists.max(), 0.001)
        gap_scores = min_dists / max_dist

        # Color by coverage: green = well-covered, red = gap
        concept_colors = [
            f"rgb({int(200 * g)}, {int(200 * (1-g))}, 50)" for g in gap_scores
        ]

        fig_umap_track.add_trace(go.Scatter(
            x=concept_2d[:, 0].tolist(),
            y=concept_2d[:, 1].tolist(),
            mode="markers+text",
            name="📍 Textbook Concepts",
            marker=dict(
                size=12,
                symbol="diamond",
                color=concept_colors,
                line=dict(width=2, color="black"),
                opacity=0.9,
            ),
            text=[c[:40] for c in all_concepts],
            textposition="top center",
            textfont=dict(size=8, color="#333"),
            hovertext=[
                f"<b>CONCEPT:</b> {c}<br>"
                f"<b>Coverage:</b> {'🟢 Well-covered' if g < 0.3 else '🟡 Partial' if g < 0.6 else '🔴 GAP'}<br>"
                f"<b>Gap score:</b> {g:.2f}"
                for c, g in zip(all_concepts, gap_scores)
            ],
            hoverinfo="text",
        ))

    method = "UMAP" if HAS_UMAP else "PCA"
    fig_umap_track.update_layout(
        title=f"Coverage Map: Questions + Textbook Concepts ({method})",
        xaxis_title=f"{method} 1",
        yaxis_title=f"{method} 2",
        template="plotly_white",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ── Gap priority chart — which concepts need questions most? ──────────
    fig_gaps = None
    if has_concept_overlay:
        # Sort by gap score descending — biggest gaps first
        gap_data = sorted(
            zip(all_concepts, gap_scores.tolist()),
            key=lambda x: -x[1]
        )
        top_gaps = gap_data[:20]

        fig_gaps = go.Figure(data=[go.Bar(
            y=[c[:50] for c, _ in reversed(top_gaps)],
            x=[g for _, g in reversed(top_gaps)],
            orientation="h",
            marker_color=[
                "#e74c3c" if g > 0.6 else "#f39c12" if g > 0.3 else "#2ecc71"
                for _, g in reversed(top_gaps)
            ],
            text=[f"{g:.0%}" for _, g in reversed(top_gaps)],
            textposition="auto",
        )])
        fig_gaps.update_layout(
            title="Top 20 Coverage Gaps (concepts furthest from any question)",
            xaxis_title="Gap Score (1.0 = no nearby questions)",
            template="plotly_white",
            height=550,
            margin=dict(l=300),
        )

    # ── Figure 2: UMAP by Level ───────────────────────────────────────────
    fig_umap_level = go.Figure()
    for level in all_levels:
        mask = [i for i, l in enumerate(levels) if l == level]
        if not mask:
            continue
        fig_umap_level.add_trace(go.Scatter(
            x=[embedding_2d[i, 0] for i in mask],
            y=[embedding_2d[i, 1] for i in mask],
            mode="markers",
            name=level,
            marker=dict(
                size=6,
                color=level_colors.get(level, "#999"),
                opacity=0.7,
            ),
            text=[f"<b>{titles[i]}</b><br>{tracks[i]} · {topics[i]}<br>{scenarios[i]}" for i in mask],
            hoverinfo="text",
        ))

    fig_umap_level.update_layout(
        title=f"Question Embedding Space ({method}) — Colored by Level",
        xaxis_title=f"{method} 1",
        yaxis_title=f"{method} 2",
        template="plotly_white",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ── 3D Coverage Cube ────────────────────────────────────────────────
    from collections import defaultdict
    try:
        from .taxonomy import normalize_tag, get_area_for_tag, ALL_TAGS, TAXONOMY
        has_taxonomy = True
    except Exception:
        has_taxonomy = False

    fig_cube = None
    fig_cube_heatmaps = None

    if has_taxonomy:
        cube = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for q in corpus:
            t = q.get("track", "?")
            l = q.get("level", "?")
            if l in ("L6", "L6%2B"): l = "L6+"
            tag = normalize_tag(q.get("topic", ""))
            area = get_area_for_tag(tag) if tag in ALL_TAGS else "unmapped"
            cube[t][area][l] += 1

        cube_tracks = ["cloud", "edge", "mobile", "tinyml"]
        cube_levels = ["L1", "L2", "L3", "L4", "L5", "L6+"]
        cube_areas = list(TAXONOMY.keys())
        target_per_cell = 3

        # Build 3D scatter: each cell is a bubble sized by question count
        x_vals, y_vals, z_vals, sizes, colors, hover_texts = [], [], [], [], [], []
        track_map = {t: i for i, t in enumerate(cube_tracks)}
        level_map = {l: i for i, l in enumerate(cube_levels)}
        area_map = {a: i for i, a in enumerate(cube_areas)}

        for t in cube_tracks:
            for a in cube_areas:
                for l in cube_levels:
                    count = cube[t][a].get(l, 0)
                    x_vals.append(track_map[t])
                    y_vals.append(level_map[l])
                    z_vals.append(area_map[a])
                    sizes.append(max(4, min(count * 3, 30)))
                    if count == 0:
                        colors.append("rgba(231,76,60,0.8)")   # Red = empty
                    elif count < target_per_cell:
                        colors.append("rgba(243,156,18,0.8)")  # Orange = thin
                    else:
                        colors.append("rgba(46,204,113,0.8)")  # Green = covered

                    hover_texts.append(
                        f"<b>{t} / {a} / {l}</b><br>"
                        f"Questions: {count}<br>"
                        f"{'🔴 EMPTY' if count == 0 else '🟡 THIN' if count < target_per_cell else '🟢 COVERED'}"
                    )

        fig_cube = go.Figure(data=[go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
            ),
            text=hover_texts,
            hoverinfo="text",
        )])

        fig_cube.update_layout(
            title="3D Coverage Cube: Track × Level × Competency",
            scene=dict(
                xaxis=dict(
                    title="Track",
                    tickvals=list(range(len(cube_tracks))),
                    ticktext=cube_tracks,
                ),
                yaxis=dict(
                    title="Level",
                    tickvals=list(range(len(cube_levels))),
                    ticktext=cube_levels,
                ),
                zaxis=dict(
                    title="Competency",
                    tickvals=list(range(len(cube_areas))),
                    ticktext=[a[:12] for a in cube_areas],
                ),
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
            ),
            template="plotly_white",
            height=700,
        )

        # Also build per-track heatmaps (competency × level) as a subplot grid
        from plotly.subplots import make_subplots
        fig_cube_heatmaps = make_subplots(
            rows=1, cols=4,
            subplot_titles=[f"{t.upper()}" for t in cube_tracks],
            horizontal_spacing=0.05,
        )

        for col, t in enumerate(cube_tracks, 1):
            z_data = []
            z_text = []
            for a in cube_areas:
                row_data = []
                row_text = []
                for l in cube_levels:
                    count = cube[t][a].get(l, 0)
                    row_data.append(count)
                    row_text.append(str(count) if count > 0 else "·")
                z_data.append(row_data)
                z_text.append(row_text)

            fig_cube_heatmaps.add_trace(
                go.Heatmap(
                    z=z_data,
                    x=cube_levels,
                    y=[a[:14] for a in cube_areas],
                    text=z_text,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    colorscale=[
                        [0, "#e74c3c"],
                        [0.1, "#e67e22"],
                        [0.3, "#f1c40f"],
                        [0.5, "#2ecc71"],
                        [1.0, "#1a5276"],
                    ],
                    showscale=(col == 4),
                    zmin=0,
                    zmax=20,
                    hovertemplate="%{y} / %{x}: %{z} questions<extra></extra>",
                ),
                row=1, col=col,
            )

        fig_cube_heatmaps.update_layout(
            title="Coverage by Track × Competency × Level (target: ≥3 per cell)",
            height=500,
            template="plotly_white",
        )

    # ── Figure 3: Coverage Heatmap ────────────────────────────────────────
    heatmap_data = []
    heatmap_text = []
    display_tracks = [t for t in all_tracks if t != "unknown"]

    for track in display_tracks:
        row = []
        text_row = []
        for level in all_levels:
            count = sum(1 for t, l in zip(tracks, levels) if t == track and l == level)
            row.append(count)
            text_row.append(str(count))
        heatmap_data.append(row)
        heatmap_text.append(text_row)

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=all_levels,
        y=display_tracks,
        text=heatmap_text,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        colorscale=[
            [0, "#e74c3c"],      # 0 = red
            [0.05, "#e67e22"],   # low = orange
            [0.15, "#f1c40f"],   # medium = yellow
            [0.4, "#2ecc71"],    # good = green
            [1.0, "#1a5276"],    # saturated = dark blue
        ],
        hovertemplate="Track: %{y}<br>Level: %{x}<br>Questions: %{z}<extra></extra>",
    ))

    fig_heatmap.update_layout(
        title="Coverage Heatmap: Track × Level",
        xaxis_title="Mastery Level",
        yaxis_title="Deployment Track",
        template="plotly_white",
        height=350,
    )

    # ── Figure 4: Level Distribution ──────────────────────────────────────
    level_counts = {l: sum(1 for lv in levels if lv == l) for l in all_levels}

    fig_levels = go.Figure(data=[go.Bar(
        x=list(level_counts.keys()),
        y=list(level_counts.values()),
        marker_color=[level_colors.get(l, "#999") for l in level_counts.keys()],
        text=list(level_counts.values()),
        textposition="auto",
    )])
    fig_levels.update_layout(
        title="Questions by Mastery Level",
        xaxis_title="Level",
        yaxis_title="Count",
        template="plotly_white",
        height=350,
    )

    # ── Figure 5: Topic Treemap ───────────────────────────────────────────
    topic_counts: dict[str, int] = {}
    for q in corpus:
        t = q.get("topic", "unknown")
        topic_counts[t] = topic_counts.get(t, 0) + 1

    # Top 30 topics
    top_topics = sorted(topic_counts.items(), key=lambda x: -x[1])[:30]
    fig_topics = go.Figure(data=[go.Treemap(
        labels=[t[0] for t in top_topics],
        parents=["" for _ in top_topics],
        values=[t[1] for t in top_topics],
        textinfo="label+value",
        marker=dict(colorscale="Blues", line=dict(width=1)),
    )])
    fig_topics.update_layout(
        title="Top 30 Topics by Question Count",
        height=500,
    )

    # ── Figure 6: Napkin Math Coverage ────────────────────────────────────
    has_napkin = sum(1 for q in corpus if q.get("details", {}).get("napkin_math"))
    has_options = sum(1 for q in corpus if q.get("details", {}).get("options"))
    total = len(corpus)

    fig_quality = go.Figure(data=[go.Bar(
        x=["Has Napkin Math", "Has MCQ Options", "Total Questions"],
        y=[has_napkin, has_options, total],
        marker_color=["#f39c12", "#3498db", "#2ecc71"],
        text=[f"{has_napkin} ({has_napkin/total:.0%})", f"{has_options} ({has_options/total:.0%})", str(total)],
        textposition="auto",
    )])
    fig_quality.update_layout(
        title="Question Quality Indicators",
        yaxis_title="Count",
        template="plotly_white",
        height=350,
    )

    # ── Assemble HTML ─────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StaffML Corpus Report — {timestamp}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background: #f5f5f5; color: #333; }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white; padding: 40px 60px; margin-bottom: 30px;
        }}
        .header h1 {{ font-size: 2.2em; font-weight: 700; margin-bottom: 8px; }}
        .header .subtitle {{ font-size: 1.1em; opacity: 0.8; }}
        .header .stats {{
            display: flex; gap: 40px; margin-top: 20px;
        }}
        .header .stat {{ text-align: center; }}
        .header .stat .number {{ font-size: 2.4em; font-weight: 700; color: #4ecdc4; }}
        .header .stat .label {{ font-size: 0.85em; opacity: 0.7; text-transform: uppercase; letter-spacing: 1px; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 0 30px 60px; }}
        .section {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
        .section h2 {{ font-size: 1.4em; margin-bottom: 5px; color: #1a1a2e; }}
        .section .desc {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
        .footer {{ text-align: center; padding: 30px; color: #999; font-size: 0.85em; }}
        @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} .header .stats {{ flex-wrap: wrap; }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>StaffML Corpus Report</h1>
        <div class="subtitle">Automatic Item Generation Pipeline — Coverage Analysis</div>
        <div class="stats">
            <div class="stat">
                <div class="number">{total}</div>
                <div class="label">Total Questions</div>
            </div>
            <div class="stat">
                <div class="number">{len(set(tracks)) - (1 if 'unknown' in set(tracks) else 0)}</div>
                <div class="label">Tracks</div>
            </div>
            <div class="stat">
                <div class="number">{len(set(topics))}</div>
                <div class="label">Unique Topics</div>
            </div>
            <div class="stat">
                <div class="number">{has_napkin/total:.0%}</div>
                <div class="label">With Napkin Math</div>
            </div>
        </div>
    </div>

    <div class="container">

        <!-- ============ SECTION 1: WHAT TO DO NEXT ============ -->
        <div class="section" style="border-left:4px solid #e74c3c;">
            <h2 style="color:#e74c3c;">1. Where To Generate Next</h2>
            <div class="desc">The coverage heatmap shows questions per track × level. <b style="color:#e74c3c;">Red cells need questions urgently.</b> Run <code>python3 generate.py --level L1,L2 --budget 20</code> to fill the biggest gaps.</div>
            <div id="heatmap"></div>
            <div id="levels" style="margin-top:20px;"></div>
            {"<div id='gaps' style='margin-top:20px;'></div>" if fig_gaps else ""}
        </div>

        <!-- ============ SECTION 2: QUESTION LANDSCAPE ============ -->
        <div class="section">
            <h2>2. Question Landscape</h2>
            <div class="desc">Every question plotted by semantic similarity. <b>Dense clusters</b> = well-covered topics. <b>Empty space</b> = generation opportunities. Hover any point for the question title and details.</div>
            <div id="umap-track"></div>
        </div>

        <div class="section">
            <h2 style="font-size:1.1em;">Same map, colored by mastery level</h2>
            <div class="desc">If one color dominates a region, that topic only has questions at one difficulty level — it needs questions at other levels too.</div>
            <div id="umap-level"></div>
        </div>

        <!-- ============ SECTION 2.5: 3D COVERAGE CUBE ============ -->
        {"<div class='section' style='border-left:4px solid #9b59b6;'><h2 style=\"color:#9b59b6;\">Coverage Cube (3D)</h2><div class=\"desc\">Every bubble is a track × level × competency cell. <b style=\"color:#e74c3c;\">Red = empty</b>, <b style=\"color:#f39c12;\">orange = thin (&lt;3)</b>, <b style=\"color:#2ecc71;\">green = covered (≥3)</b>. Rotate to find the holes.</div><div id=\"cube3d\"></div></div>" if fig_cube else ""}

        {"<div class='section'><h2>Coverage Grid: Competency × Level per Track</h2><div class=\"desc\">Each cell shows question count. Red = 0, green = well-covered. Compare across tracks to see which competencies are under-served where.</div><div id=\"cube-heatmaps\"></div></div>" if fig_cube_heatmaps else ""}

        <!-- ============ SECTION 3: TOPIC CLUSTERS + LANDSCAPE ============ -->
        {bertopic_html}

        <!-- ============ SECTION 4: QUALITY ============ -->
        <div class="section" style="border-left:4px solid #f39c12;">
            <h2 style="color:#f39c12;">3. Quality Health Check</h2>
            <div class="desc">How well-constructed are the existing questions? Bars show what percentage have napkin math, MCQ options, etc.</div>
            <div id="quality"></div>
        </div>

        <!-- ============ SECTION 5: TOPIC TREEMAP ============ -->
        <div class="section">
            <h2>4. Topic Distribution</h2>
            <div class="desc">The 30 most common topic tags by question count. Oversized boxes may be over-saturated; missing topics need generation.</div>
            <div id="topics"></div>
        </div>

    </div>

    <div class="footer">
        Generated {timestamp} · StaffML Question Generation Engine v0.1.0 ·
        Embedding: {"nomic-embed-text-v1.5 (8192 tokens)" if used_st else "TF-IDF (fallback)"} ·
        {"BERTopic: " + str(len(discovered_topics)) + " clusters discovered" if discovered_topics else "BERTopic: not available"} ·
        Grounded in AIG (Gierl 2013), Bloom's Taxonomy (Anderson 2001), ECD (Mislevy 2003)
    </div>

    <script>
        var umap_track = {fig_umap_track.to_json()};
        var umap_level = {fig_umap_level.to_json()};
        var heatmap = {fig_heatmap.to_json()};
        var levels = {fig_levels.to_json()};
        var topics = {fig_topics.to_json()};
        var quality = {fig_quality.to_json()};

        Plotly.newPlot('umap-track', umap_track.data, umap_track.layout, {{responsive: true}});
        Plotly.newPlot('umap-level', umap_level.data, umap_level.layout, {{responsive: true}});
        Plotly.newPlot('heatmap', heatmap.data, heatmap.layout, {{responsive: true}});
        Plotly.newPlot('levels', levels.data, levels.layout, {{responsive: true}});
        Plotly.newPlot('topics', topics.data, topics.layout, {{responsive: true}});
        Plotly.newPlot('quality', quality.data, quality.layout, {{responsive: true}});
        {f"var gaps_chart = {fig_gaps.to_json()}; Plotly.newPlot('gaps', gaps_chart.data, gaps_chart.layout, {{responsive: true}});" if fig_gaps else ""}
        {f"var cube3d = {fig_cube.to_json()}; Plotly.newPlot('cube3d', cube3d.data, cube3d.layout, {{responsive: true}});" if fig_cube else ""}
        {f"var cube_hm = {fig_cube_heatmaps.to_json()}; Plotly.newPlot('cube-heatmaps', cube_hm.data, cube_hm.layout, {{responsive: true}});" if fig_cube_heatmaps else ""}
    </script>
</body>
</html>"""

    # ── Write ─────────────────────────────────────────────────────────────
    if output_path is None:
        reports_dir = Path(__file__).parent.parent / "_reports"
        reports_dir.mkdir(exist_ok=True)
        output_path = reports_dir / "corpus_report.html"

    output_path.write_text(html, encoding="utf-8")

    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{output_path.resolve()}")

    return output_path
