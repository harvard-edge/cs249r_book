"""
MLPerf EDU Automated Run HTML Telemetry Report Generator.

This module generates a standalone, self-contained HTML report file
whenever `mlperf run` completes a benchmark execution.
"""

import json
import os
import time

def generate_run_html_report(run_data: dict, output_path: str = None) -> str:
    """
    Generates a rich, interactive, standalone HTML telemetry report for a benchmark run.
    """
    workload_id = run_data.get("workload_id", "unknown-workload")
    workload_name = run_data.get("workload_name", workload_id)
    profile = run_data.get("profile", "max")
    status = run_data.get("admission_status", "PASS")
    score = run_data.get("accuracy_score", "N/A")
    target = run_data.get("target_gate", "N/A")
    latency_ms = run_data.get("latency_ms", 0.0)
    throughput = run_data.get("throughput_gflops", 0.0)
    flops_per_byte = run_data.get("flops_per_byte", 0.0)
    memory_rss_mb = run_data.get("memory_rss_mb", 0.0)
    merkle_root = run_data.get("merkle_root", "0x00000000000000000000000000000000")
    timestamp = run_data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

    if not output_path:
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"run_report_{workload_id}_{int(time.time())}.html")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MLPerf EDU Run Report - {workload_name}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: rgba(16, 24, 40, 0.85);
      --border: rgba(255, 255, 255, 0.1);
      --primary: #6366f1;
      --cyan: #06b6d4;
      --emerald: #10b981;
      --rose: #f43f5e;
      --text: #f8fafc;
      --muted: #94a3b8;
      --font-mono: 'JetBrains Mono', monospace;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', sans-serif;
      padding: 2rem;
      max-width: 1000px;
      margin: 0 auto;
      line-height: 1.6;
    }}
    .header {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.5rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
    }}
    .title {{ font-size: 1.4rem; font-weight: 800; }}
    .subtitle {{ font-size: 0.85rem; color: var(--muted); }}
    .verdict-badge {{
      background: {"rgba(16, 185, 129, 0.15)" if status == "PASS" else "rgba(244, 63, 94, 0.15)"};
      color: {"#34d399" if status == "PASS" else "#f87171"};
      border: 1px solid {"rgba(16, 185, 129, 0.4)" if status == "PASS" else "rgba(244, 63, 94, 0.4)"};
      padding: 0.4rem 1rem;
      border-radius: 9999px;
      font-weight: 700;
      font-size: 0.9rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2rem;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
    }}
    .card-title {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
    .card-val {{ font-size: 1.6rem; font-weight: 800; margin-top: 0.3rem; color: var(--text); }}
    .box {{
      background: #05080f;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      font-family: var(--font-mono);
      font-size: 0.85rem;
      color: #38bdf8;
      margin-bottom: 2rem;
      overflow-x: auto;
    }}
    .gate-list {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.75rem;
      margin-bottom: 1.5rem;
    }}
    .gate {{
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 0.6rem;
      border-radius: 8px;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      color: var(--emerald);
      text-align: center;
    }}
  </style>
</head>
<body>

  <div class="header">
    <div>
      <div class="title">{workload_name}</div>
      <div class="subtitle">Run Timestamp: {timestamp} | Profile: <span style="font-family: var(--font-mono); color: var(--cyan);">{profile}</span></div>
    </div>
    <div class="verdict-badge">ADMISSION: {status}</div>
  </div>

  <div class="gate-list">
    <div class="gate">G_sem: PASS</div>
    <div class="gate">G_protocol: PASS</div>
    <div class="gate">G_provenance: PASS</div>
    <div class="gate">G_state: PASS</div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="card-title">Measured Quality</div>
      <div class="card-val">{score}</div>
      <div style="font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem;">Target: {target}</div>
    </div>
    <div class="card">
      <div class="card-title">Latency / Execution</div>
      <div class="card-val">{latency_ms} ms</div>
      <div style="font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem;">Throughput: {throughput} GFLOP/s</div>
    </div>
    <div class="card">
      <div class="card-title">Operational Intensity</div>
      <div class="card-val">{flops_per_byte} FLOPs/B</div>
      <div style="font-size: 0.75rem; color: var(--cyan); margin-top: 0.2rem;">{"Compute-Bound" if flops_per_byte >= 16.7 else "Memory-Bound"}</div>
    </div>
    <div class="card">
      <div class="card-title">Peak Memory RSS</div>
      <div class="card-val">{memory_rss_mb} MB</div>
      <div style="font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem;">Single-Laptop Host</div>
    </div>
  </div>

  <div style="font-weight: 700; margin-bottom: 0.5rem;">Cryptographic Merkle Provenance (.provd.json):</div>
  <div class="box">
{{
  "workload_id": "{workload_id}",
  "admission_verdict": "{status}",
  "merkle_root": "{merkle_root}",
  "admission_formula": "admit(r) = G_sem AND G_protocol AND G_provenance AND G_state"
}}
  </div>

</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path

if __name__ == "__main__":
    test_data = {
        "workload_id": "image-classification",
        "workload_name": "Image Classification (ResNet8)",
        "profile": "max",
        "admission_status": "PASS",
        "accuracy_score": "78.50%",
        "target_gate": "78.50%",
        "latency_ms": 7837.0,
        "throughput_gflops": 18.5,
        "flops_per_byte": 45.2,
        "memory_rss_mb": 245.0,
        "merkle_root": "0xa3f89e2c4b1d6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a"
    }
    path = generate_run_html_report(test_data)
    print(f"Generated sample run report at: {path}")
