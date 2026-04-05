#!/usr/bin/env python3
"""Topic graph explorer — visualize and query the StaffML taxonomy.

Usage:
    python3 graph.py                              # Full graph SVG
    python3 graph.py --topic kv-cache-management  # Neighborhood of one topic
    python3 graph.py --area compute               # Subgraph for one area
    python3 graph.py --track tinyml               # Only topics relevant to a track
    python3 graph.py --query "what leads to 3d-parallelism"
    python3 graph.py --path roofline-analysis flash-attention
    python3 graph.py --stats
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

try:
    import networkx as nx
except ImportError:
    print("pip install networkx")
    sys.exit(1)

DATA_PATH = Path(__file__).parent / "taxonomy_data.yaml"

# ── Colors ───────────────────────────────────────────────────

AREA_COLORS = {
    "compute": "#cfe2f3", "memory": "#d4edda", "latency": "#fdebd0",
    "precision": "#e8d5f5", "power": "#f9d6d5", "architecture": "#d5e8d4",
    "optimization": "#fff2cc", "parallelism": "#dae8fc",
    "networking": "#e1d5e7", "deployment": "#f8cecc",
    "reliability": "#c8e6c9", "data": "#b3e5fc", "cross-cutting": "#f5f5f5",
}

EDGE_STYLES = {
    "prerequisite": {"color": "#c44",     "style": "solid",  "label": "requires"},
    "broader":      {"color": "#4a90c4",  "style": "dashed", "label": "broader"},
    "narrower":     {"color": "#3d9e5a",  "style": "dashed", "label": "narrower"},
    "related":      {"color": "#999",     "style": "dotted", "label": "related"},
}


# ── Load ─────────────────────────────────────────────────────

def load_taxonomy(path: Path = DATA_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_graph(data: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for t in data["topics"]:
        G.add_node(t["id"], **{
            "name": t["name"],
            "area": t["area"],
            "tracks": t.get("tracks", []),
            "description": t.get("description", ""),
        })
    for t in data["topics"]:
        for edge in t.get("edges", []):
            G.add_edge(t["id"], edge["target"], edge_type=edge["edge_type"],
                       note=edge.get("note", ""))
    return G


# ── Queries ──────────────────────────────────────────────────

def neighborhood(G: nx.DiGraph, topic_id: str, radius: int = 2) -> nx.DiGraph:
    """Return the subgraph within `radius` hops of a topic (any direction)."""
    undirected = G.to_undirected()
    nodes = nx.single_source_shortest_path_length(undirected, topic_id, cutoff=radius)
    return G.subgraph(nodes.keys()).copy()


def area_subgraph(G: nx.DiGraph, area: str) -> nx.DiGraph:
    nodes = [n for n, d in G.nodes(data=True) if d.get("area") == area]
    return G.subgraph(nodes).copy()


def track_subgraph(G: nx.DiGraph, track: str) -> nx.DiGraph:
    nodes = [n for n, d in G.nodes(data=True) if track in d.get("tracks", [])]
    return G.subgraph(nodes).copy()


def prerequisite_path(G: nx.DiGraph, source: str, target: str) -> list[str] | None:
    """Find shortest prerequisite-only path from source to target."""
    prereq_edges = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get("edge_type") == "prerequisite"]
    H = nx.DiGraph(prereq_edges)
    try:
        return nx.shortest_path(H, source, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Try reverse direction
        try:
            return nx.shortest_path(H, target, source)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None


def what_leads_to(G: nx.DiGraph, topic_id: str) -> list[str]:
    """All transitive prerequisites of a topic.

    Edge convention: 3d-parallelism → data-parallelism means
    '3d-parallelism requires data-parallelism', so prerequisites
    are descendants in the prerequisite subgraph.
    """
    prereq_graph = nx.DiGraph(
        [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "prerequisite"]
    )
    try:
        return list(nx.descendants(prereq_graph, topic_id))
    except nx.NetworkXError:
        return []


def what_depends_on(G: nx.DiGraph, topic_id: str) -> list[str]:
    """All topics that transitively require this topic.

    Edge convention: child → prereq, so dependents are ancestors.
    """
    prereq_graph = nx.DiGraph(
        [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "prerequisite"]
    )
    try:
        return list(nx.ancestors(prereq_graph, topic_id))
    except nx.NetworkXError:
        return []


# ── DOT Export ───────────────────────────────────────────────

def to_dot(G: nx.DiGraph, title: str = "StaffML Topic Taxonomy",
           highlight: str | None = None) -> str:
    """Export graph as Graphviz DOT."""
    lines = [
        f'digraph "{title}" {{',
        '  rankdir=LR;',
        '  node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8];',
        f'  label="{title}";',
        '  labelloc=t;',
        '',
    ]

    # Group by area
    areas = defaultdict(list)
    for n, d in G.nodes(data=True):
        areas[d.get("area", "unknown")].append(n)

    for area, nodes in sorted(areas.items()):
        color = AREA_COLORS.get(area, "#ffffff")
        lines.append(f'  subgraph cluster_{area.replace("-", "_")} {{')
        lines.append(f'    label="{area}";')
        lines.append(f'    style=filled; color="{color}"; fillcolor="{color}30";')
        for n in nodes:
            d = G.nodes[n]
            name = d.get("name", n)
            fc = '"#ffffff"' if n == highlight else f'"{color}"'
            penwidth = "3" if n == highlight else "1"
            lines.append(
                f'    "{n}" [label="{name}", fillcolor={fc}, penwidth={penwidth}];'
            )
        lines.append('  }')
        lines.append('')

    # Edges
    for u, v, d in G.edges(data=True):
        etype = d.get("edge_type", "related")
        style = EDGE_STYLES.get(etype, EDGE_STYLES["related"])
        lines.append(
            f'  "{u}" -> "{v}" '
            f'[color="{style["color"]}", style={style["style"]}, '
            f'label="{style["label"]}"];'
        )

    lines.append('}')
    return '\n'.join(lines)


# ── Stats ────────────────────────────────────────────────────

def print_stats(G: nx.DiGraph):
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    edge_types = defaultdict(int)
    for _, _, d in G.edges(data=True):
        edge_types[d.get("edge_type", "unknown")] += 1
    print(f"\nEdge types:")
    for et, cnt in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"  {et}: {cnt}")

    areas = defaultdict(int)
    for _, d in G.nodes(data=True):
        areas[d.get("area", "unknown")] += 1
    print(f"\nTopics per area:")
    for area, cnt in sorted(areas.items()):
        print(f"  {area}: {cnt}")

    # Connectivity
    prereq_G = nx.DiGraph(
        [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "prerequisite"]
    )
    prereq_G.add_nodes_from(G.nodes())
    components = list(nx.weakly_connected_components(prereq_G))
    print(f"\nPrerequisite graph:")
    print(f"  Connected components: {len(components)}")
    print(f"  Largest component: {max(len(c) for c in components)} nodes")

    if nx.is_directed_acyclic_graph(prereq_G):
        longest = nx.dag_longest_path(prereq_G)
        print(f"  DAG: yes (no cycles)")
        print(f"  Longest prerequisite chain ({len(longest)} topics):")
        print(f"    {' → '.join(longest)}")
    else:
        print(f"  WARNING: Cycles detected in prerequisite graph!")

    # Track coverage
    tracks = defaultdict(int)
    for _, d in G.nodes(data=True):
        for t in d.get("tracks", []):
            tracks[t] += 1
    print(f"\nTopics per track:")
    for t, cnt in sorted(tracks.items()):
        print(f"  {t}: {cnt}")

    # Most connected topics
    print(f"\nMost connected topics (degree):")
    by_degree = sorted(G.degree(), key=lambda x: -x[1])[:10]
    for n, deg in by_degree:
        name = G.nodes[n].get("name", n)
        print(f"  {name}: {deg} connections")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Explore the StaffML topic taxonomy")
    parser.add_argument("--topic", help="Show neighborhood of a specific topic")
    parser.add_argument("--area", help="Show only one competency area")
    parser.add_argument("--track", help="Show only topics for a track")
    parser.add_argument("--path", nargs=2, metavar=("FROM", "TO"),
                        help="Find prerequisite path between two topics")
    parser.add_argument("--query", help="'what leads to X' or 'what needs X'")
    parser.add_argument("--radius", type=int, default=2,
                        help="Neighborhood radius (default: 2)")
    parser.add_argument("--stats", action="store_true", help="Print graph statistics")
    parser.add_argument("--output", default=None, help="Output DOT file path")
    parser.add_argument("--format", choices=["dot", "svg", "png"], default="dot",
                        help="Output format (default: dot)")
    args = parser.parse_args()

    data = load_taxonomy()
    G = build_graph(data)

    if args.stats:
        print_stats(G)
        return

    if args.path:
        path = prerequisite_path(G, args.path[0], args.path[1])
        if path:
            print(f"Prerequisite path ({len(path)} topics):")
            for i, p in enumerate(path):
                name = G.nodes[p].get("name", p)
                prefix = "  └─" if i == len(path) - 1 else "  ├─"
                print(f"{prefix} {name} ({p})")
        else:
            print(f"No prerequisite path between {args.path[0]} and {args.path[1]}")
        return

    if args.query:
        q = args.query.lower()
        if "leads to" in q or "what leads" in q:
            topic = q.split("leads to")[-1].strip().strip('"').strip("'")
            prereqs = what_leads_to(G, topic)
            if prereqs:
                print(f"Prerequisites for '{topic}' ({len(prereqs)} topics):")
                for p in sorted(prereqs):
                    name = G.nodes[p].get("name", p)
                    print(f"  ← {name} ({p})")
            else:
                print(f"'{topic}' has no prerequisites (it's a root topic)")
        elif "needs" in q or "depends on" in q:
            topic = q.split("needs")[-1].split("depends on")[-1].strip().strip('"')
            deps = what_depends_on(G, topic)
            if deps:
                print(f"Topics that depend on '{topic}' ({len(deps)} topics):")
                for d in sorted(deps):
                    name = G.nodes[d].get("name", d)
                    print(f"  → {name} ({d})")
            else:
                print(f"Nothing depends on '{topic}' (it's a leaf topic)")
        return

    # Build subgraph for visualization
    title = "StaffML Topic Taxonomy"
    highlight = None

    if args.topic:
        if args.topic not in G:
            print(f"Topic '{args.topic}' not found. Available: {sorted(G.nodes())}")
            sys.exit(1)
        G = neighborhood(G, args.topic, args.radius)
        title = f"Neighborhood of {G.nodes[args.topic].get('name', args.topic)}"
        highlight = args.topic
    elif args.area:
        G = area_subgraph(G, args.area)
        title = f"Competency Area: {args.area}"
    elif args.track:
        G = track_subgraph(G, args.track)
        title = f"Track: {args.track}"

    dot = to_dot(G, title=title, highlight=highlight)

    if args.output:
        outpath = Path(args.output)
        if args.format == "dot":
            outpath.write_text(dot)
            print(f"Wrote {outpath}")
        else:
            # Use graphviz CLI if available
            import subprocess
            dot_path = outpath.with_suffix(".dot")
            dot_path.write_text(dot)
            result = subprocess.run(
                ["dot", f"-T{args.format}", str(dot_path), "-o", str(outpath)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print(f"Wrote {outpath}")
                dot_path.unlink()
            else:
                print(f"graphviz error: {result.stderr}")
                print(f"DOT file saved at {dot_path}")
    else:
        print(dot)


if __name__ == "__main__":
    main()
