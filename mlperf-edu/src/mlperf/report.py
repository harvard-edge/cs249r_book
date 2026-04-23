#!/usr/bin/env python3
"""MLPerf EDU: Per-Run HTML Report Generator

Generates a self-contained, audit-ready HTML report for each benchmark run.
This is the human-readable interface between benchmarking, auditing, and learning.

Each report answers:
  - What was run?
  - How was it run?
  - What are the results?
  - Can I trust this?

Usage:
    python -m mlperf.report --submission submissions/nanogpt_20260411_143500.json
    
    # Or import and use programmatically:
    from mlperf.report import generate_report
    generate_report("submissions/nanogpt_20260411_143500.json")
"""

import json
import os
import hashlib
import datetime
from pathlib import Path

try:
    from src.mlperf.fingerprint import detect_hardware, format_fingerprint
except ImportError:
    try:
        from mlperf.fingerprint import detect_hardware, format_fingerprint
    except ImportError:
        detect_hardware = None
        format_fingerprint = None


def _compute_file_hash(filepath):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def _format_metric(value, metric_type):
    """Format a metric value for display."""
    if metric_type in ('loss', 'mse'):
        return f"{value:.4f}"
    elif metric_type in ('accuracy', 'top1', 'retrieval', 'pass1', 'trace'):
        return f"{value*100:.1f}%" if value < 1.0 else f"{value:.1f}%"
    elif metric_type == 'reward':
        return f"{value:.1f}"
    elif metric_type in ('latency_ms',):
        return f"{value:.2f} ms"
    elif metric_type in ('throughput_qps',):
        return f"{value:.1f} QPS"
    elif metric_type in ('power_watts',):
        return f"{value:.1f} W"
    elif metric_type in ('energy_joules',):
        return f"{value:.1f} J"
    return str(value)


CSS = """
:root {
    --bg: #0f172a;
    --surface: #1e293b;
    --surface2: #334155;
    --border: #475569;
    --text: #e2e8f0;
    --text-dim: #94a3b8;
    --accent: #38bdf8;
    --green: #4ade80;
    --red: #f87171;
    --amber: #fbbf24;
    --purple: #a78bfa;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
    max-width: 1200px;
    margin: 0 auto;
}

h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }
h2 { font-size: 1.3rem; font-weight: 600; margin: 1.5rem 0 0.75rem; color: var(--accent); }
h3 { font-size: 1.05rem; font-weight: 600; margin: 1rem 0 0.5rem; }

.header {
    display: flex; justify-content: space-between; align-items: flex-start;
    border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; margin-bottom: 2rem;
}
.header-meta { text-align: right; color: var(--text-dim); font-size: 0.85rem; }

.badge {
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 4px;
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
}
.badge-pass { background: rgba(74, 222, 128, 0.15); color: var(--green); border: 1px solid var(--green); }
.badge-fail { background: rgba(248, 113, 113, 0.15); color: var(--red); border: 1px solid var(--red); }
.badge-division { background: rgba(56, 189, 248, 0.15); color: var(--accent); border: 1px solid var(--accent); }

.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; }

.card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 1.25rem; transition: border-color 0.2s;
}
.card:hover { border-color: var(--accent); }
.card-label { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
.card-value { font-size: 1.5rem; font-weight: 700; margin-top: 0.25rem; }
.card-sub { font-size: 0.8rem; color: var(--text-dim); margin-top: 0.25rem; }

table {
    width: 100%; border-collapse: collapse; margin: 0.75rem 0;
    background: var(--surface); border-radius: 8px; overflow: hidden;
}
th { background: var(--surface2); text-align: left; padding: 0.6rem 1rem; font-size: 0.8rem;
     color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
td { padding: 0.6rem 1rem; border-top: 1px solid var(--border); font-size: 0.9rem; }

.delta-positive { color: var(--green); }
.delta-negative { color: var(--red); }
.delta-neutral { color: var(--text-dim); }

.trace-box {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
    word-break: break-all; color: var(--text-dim);
}

.section { margin-bottom: 2rem; }

.interpretation {
    background: rgba(56, 189, 248, 0.08); border-left: 3px solid var(--accent);
    padding: 1rem 1.25rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
    font-size: 0.9rem; line-height: 1.7;
}

.footer {
    margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border);
    text-align: center; color: var(--text-dim); font-size: 0.8rem;
}

@media (max-width: 768px) {
    body { padding: 1rem; }
    .header { flex-direction: column; }
    .header-meta { text-align: left; margin-top: 0.5rem; }
    .grid { grid-template-columns: 1fr; }
}
"""


def generate_report(submission_path, output_path=None, baseline_path=None):
    """Generate a self-contained HTML report from a submission JSON.
    
    Args:
        submission_path: Path to the submission JSON file
        output_path: Optional output path for the HTML report
        baseline_path: Optional path to a baseline submission for comparison
    
    Returns:
        Path to the generated HTML report
    """
    with open(submission_path) as f:
        data = json.load(f)
    
    baseline = None
    if baseline_path and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
    
    # Extract fields with safe defaults
    workload = data.get('workload', data.get('task', 'unknown'))
    division = data.get('division', 'unknown')
    scenario = data.get('scenario', 'Offline')
    hardware = data.get('hardware', data.get('hardware_fingerprint', {}))
    timestamp = data.get('timestamp', datetime.datetime.now().isoformat())
    seed = data.get('seed', 42)
    
    # Metrics
    metrics = data.get('metrics', {})
    latency = metrics.get('latency_p50_ms', metrics.get('latency_ms'))
    throughput = metrics.get('throughput_qps')
    accuracy = metrics.get('accuracy', metrics.get('loss', metrics.get('top1')))
    accuracy_type = 'loss' if 'loss' in metrics else 'accuracy' if 'accuracy' in metrics else 'top1'
    power = metrics.get('power_avg_watts')
    energy = metrics.get('energy_joules')
    
    # Training metrics
    training = data.get('training', {})
    epochs = training.get('epochs', data.get('epochs'))
    final_loss = training.get('final_loss', training.get('final_train_loss'))
    final_val = training.get('final_val_loss', training.get('final_val_metric'))
    training_time = training.get('total_time_s', training.get('wall_time_s'))
    
    # Config
    config = data.get('config', {})
    batch_size = config.get('batch_size', data.get('batch_size'))
    lr = config.get('learning_rate', data.get('learning_rate'))
    optimizer = config.get('optimizer', data.get('optimizer', 'AdamW'))
    
    # Hashes
    hashes = data.get('hashes', data.get('integrity', {}))
    dataset_hash = hashes.get('dataset', hashes.get('dataset_hash', 'N/A'))
    model_hash = hashes.get('model', hashes.get('checkpoint_hash', 'N/A'))
    log_hash = hashes.get('log', hashes.get('log_hash', 'N/A'))
    
    # Compliance
    compliance = data.get('compliance', {})
    target_met = compliance.get('target_met', None)
    target_value = compliance.get('target', data.get('target_quality'))
    run_count = compliance.get('run_count', 1)
    
    # Hardware details — prefer submission data, fall back to auto-detection
    if isinstance(hardware, dict) and hardware:
        hw_cpu = hardware.get('cpu', hardware.get('chip', 'N/A'))
        hw_mem = hardware.get('memory_gb', 'N/A')
        hw_os = hardware.get('os', 'N/A')
        hw_gpu = hardware.get('gpu', hardware.get('accelerator', 'N/A'))
        hw_backend = hardware.get('backend', 'N/A')
        hw_fingerprint_id = hardware.get('fingerprint_hash', 'N/A')
        hw_caches = hardware.get('cache_sizes', {})
    elif detect_hardware is not None:
        # Auto-detect if submission didn't include hardware
        hw_info = detect_hardware()
        hw_cpu = hw_info.get('chip', 'N/A')
        hw_mem = hw_info.get('memory_gb', 'N/A')
        hw_os = hw_info.get('os', 'N/A')
        hw_gpu = hw_info.get('gpu', 'N/A')
        hw_backend = hw_info.get('backend', 'N/A')
        hw_fingerprint_id = hw_info.get('fingerprint_hash', 'N/A')
        hw_caches = hw_info.get('cache_sizes', {})
    else:
        hw_cpu = str(hardware) if hardware else 'N/A'
        hw_mem = hw_os = hw_gpu = hw_backend = hw_fingerprint_id = 'N/A'
        hw_caches = {}
    
    # Generate comparison deltas
    def _delta_html(current, baseline_val, lower_is_better=True):
        if baseline_val is None or current is None:
            return ''
        pct = ((current - baseline_val) / abs(baseline_val)) * 100 if baseline_val != 0 else 0
        improved = (pct < 0) if lower_is_better else (pct > 0)
        cls = 'delta-positive' if improved else 'delta-negative'
        arrow = '↓' if pct < 0 else '↑'
        return f' <span class="{cls}">({arrow} {abs(pct):.1f}%)</span>'
    
    # Build compliance status
    if target_met is True:
        compliance_badge = '<span class="badge badge-pass">✓ PASS</span>'
    elif target_met is False:
        compliance_badge = '<span class="badge badge-fail">✗ FAIL</span>'
    else:
        compliance_badge = '<span class="badge badge-division">—</span>'
    
    # File hash for this report
    report_hash = _compute_file_hash(submission_path)
    
    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLPerf EDU Report — {workload}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>

<!-- Section 1: Run Overview -->
<div class="header">
    <div>
        <h1>MLPerf EDU — {workload}</h1>
        <span class="badge badge-division">{division}</span>
        <span class="badge badge-division">{scenario}</span>
        {compliance_badge}
    </div>
    <div class="header-meta">
        <div>{timestamp}</div>
        <div>Seed: {seed}</div>
        <div>Report: {report_hash}</div>
    </div>
</div>

<!-- Section 2: Top-Level Metrics -->
<div class="section">
<h2>§1 Metrics Summary</h2>
<div class="grid">
"""
    
    # Add metric cards dynamically
    if latency is not None:
        bl = baseline.get('metrics', {}).get('latency_p50_ms') if baseline else None
        html += f"""    <div class="card">
        <div class="card-label">Latency (p50)</div>
        <div class="card-value">{latency:.2f} ms{_delta_html(latency, bl, True)}</div>
        <div class="card-sub">p90: {metrics.get('latency_p90_ms', 'N/A')}, p99: {metrics.get('latency_p99_ms', 'N/A')}</div>
    </div>
"""
    
    if throughput is not None:
        bl = baseline.get('metrics', {}).get('throughput_qps') if baseline else None
        html += f"""    <div class="card">
        <div class="card-label">Throughput</div>
        <div class="card-value">{throughput:.1f} QPS{_delta_html(throughput, bl, False)}</div>
    </div>
"""
    
    if accuracy is not None:
        bl = baseline.get('metrics', {}).get(accuracy_type) if baseline else None
        lower = accuracy_type in ('loss', 'mse')
        html += f"""    <div class="card">
        <div class="card-label">{accuracy_type.upper()}</div>
        <div class="card-value">{_format_metric(accuracy, accuracy_type)}{_delta_html(accuracy, bl, lower)}</div>
        <div class="card-sub">Target: {target_value if target_value else 'N/A'}</div>
    </div>
"""
    
    if power is not None:
        html += f"""    <div class="card">
        <div class="card-label">Power</div>
        <div class="card-value">{power:.1f} W</div>
        <div class="card-sub">Energy: {energy:.1f} J</div>
    </div>
"""
    
    html += """</div>
</div>

<!-- Section 3: Configuration -->
<div class="section">
<h2>§2 Configuration</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
"""
    
    config_rows = [
        ('Workload', workload),
        ('Division', division),
        ('Scenario', scenario),
        ('Epochs', epochs),
        ('Batch Size', batch_size),
        ('Learning Rate', lr),
        ('Optimizer', optimizer),
        ('Seed', seed),
    ]
    for k, v in config_rows:
        if v is not None:
            html += f"<tr><td>{k}</td><td>{v}</td></tr>\n"
    
    html += """</table>
</div>

<!-- Section 4: Hardware -->
<div class="section">
<h2>§3 Hardware Fingerprint</h2>
<table>
<tr><th>Component</th><th>Value</th></tr>
"""
    hw_rows = [
        ('CPU/Chip', hw_cpu),
        ('GPU/Accelerator', hw_gpu),
        ('Memory', f"{hw_mem} GB" if hw_mem != 'N/A' else 'N/A'),
        ('OS', hw_os),
        ('Backend', hw_backend),
        ('Fingerprint', hw_fingerprint_id),
    ]
    # Add cache sizes if detected
    for cache_name, cache_key in [('L1D Cache', 'l1d'), ('L2 Cache', 'l2'), ('L3 Cache', 'l3')]:
        val = hw_caches.get(cache_key)
        if val:
            if val >= 1024 * 1024:
                hw_rows.append((cache_name, f"{val // (1024*1024)} MB"))
            else:
                hw_rows.append((cache_name, f"{val // 1024} KB"))
    
    for k, v in hw_rows:
        html += f"<tr><td>{k}</td><td>{v}</td></tr>\n"
    
    html += """</table>
</div>
"""
    
    # Section 5: Training curve (if available)
    training_curve = data.get('training_curve', training.get('curve', []))
    if training_curve:
        html += """
<!-- Section 5: Convergence -->
<div class="section">
<h2>§4 Convergence Behavior</h2>
<table>
<tr><th>Epoch</th><th>Train Loss</th><th>Val Loss</th></tr>
"""
        # Show first 5 and last 5 if long
        show = training_curve
        if len(show) > 10:
            show = training_curve[:5] + [{'epoch': '...', 'train': '...', 'val': '...'}] + training_curve[-5:]
        for entry in show:
            if isinstance(entry, dict):
                html += f"<tr><td>{entry.get('epoch', '—')}</td><td>{entry.get('train', entry.get('train_loss', '—'))}</td><td>{entry.get('val', entry.get('val_loss', '—'))}</td></tr>\n"
        html += "</table>\n</div>\n"
    
    # Section 6: Baseline comparison table
    if baseline:
        html += """
<!-- Section 6: Baseline Comparison -->
<div class="section">
<h2>§5 Baseline Comparison</h2>
<table>
<tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr>
"""
        bl_metrics = baseline.get('metrics', {})
        for key in sorted(set(list(metrics.keys()) + list(bl_metrics.keys()))):
            curr = metrics.get(key)
            bl_val = bl_metrics.get(key)
            if curr is not None and bl_val is not None:
                lower = key in ('loss', 'mse', 'latency_p50_ms', 'latency_p90_ms', 'latency_p99_ms')
                pct = ((curr - bl_val) / abs(bl_val)) * 100 if bl_val != 0 else 0
                improved = (pct < 0) if lower else (pct > 0)
                cls = 'delta-positive' if improved else 'delta-negative'
                arrow = '↓' if pct < 0 else '↑'
                html += f"<tr><td>{key}</td><td>{bl_val}</td><td>{curr}</td>"
                html += f'<td class="{cls}">{arrow} {abs(pct):.1f}%</td></tr>\n'
        html += "</table>\n</div>\n"
    
    # Section 7: Execution trace (audit)
    html += f"""
<!-- Section 7: Execution Trace (Audit) -->
<div class="section">
<h2>§6 Execution Trace</h2>
<div class="trace-box">
<strong>Dataset Hash:</strong> {dataset_hash}<br>
<strong>Model Hash:</strong> {model_hash}<br>
<strong>Log Hash:</strong> {log_hash}<br>
<strong>Report Hash:</strong> {report_hash}<br>
<strong>Run Count:</strong> {run_count}<br>
<strong>Submission File:</strong> {os.path.basename(submission_path)}
</div>
</div>
"""
    
    # Section 8: Interpretation
    interpretation_parts = []
    if final_loss is not None and final_val is not None:
        gap = abs(final_loss - final_val) / max(abs(final_loss), 1e-8)
        if gap > 0.3:
            interpretation_parts.append(
                f"The train-val gap ({final_loss:.4f} vs {final_val:.4f}) suggests overfitting. "
                "Consider regularization, data augmentation, or early stopping."
            )
        else:
            interpretation_parts.append(
                f"The train-val gap is narrow ({final_loss:.4f} vs {final_val:.4f}), "
                "indicating good generalization on this dataset."
            )
    
    if latency is not None and throughput is not None:
        interpretation_parts.append(
            f"At {latency:.1f}ms latency and {throughput:.0f} QPS, "
            "this workload is suitable for interactive serving scenarios."
            if latency < 100 else
            f"At {latency:.1f}ms latency, this may be too slow for real-time serving. "
            "Consider batching or model optimization."
        )
    
    if interpretation_parts:
        html += """
<!-- Section 8: Interpretation -->
<div class="section">
<h2>§7 Interpretation</h2>
<div class="interpretation">
"""
        for p in interpretation_parts:
            html += f"<p>{p}</p>\n"
        html += "</div>\n</div>\n"
    
    html += f"""
<div class="footer">
    <p>MLPerf EDU — Pedagogical Evaluation Framework for AI Systems Benchmarking</p>
    <p>Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Report hash: {report_hash}</p>
</div>
</body>
</html>
"""
    
    # Write output
    if output_path is None:
        base = os.path.splitext(submission_path)[0]
        output_path = f"{base}_report.html"
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✅ Report generated: {output_path}")
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="MLPerf EDU: Generate audit-ready HTML report from submission JSON"
    )
    parser.add_argument('--submission', required=True, help="Path to submission JSON")
    parser.add_argument('--output', help="Output HTML path (default: <submission>_report.html)")
    parser.add_argument('--baseline', help="Optional baseline submission for comparison")
    args = parser.parse_args()
    
    generate_report(args.submission, args.output, args.baseline)


if __name__ == '__main__':
    main()
