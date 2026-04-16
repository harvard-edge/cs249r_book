#!/usr/bin/env python3
"""Fix training for remaining workloads and regenerate full figure."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, time
import torch, torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# Load existing data
data_path = "/Users/VJ/GitHub/mlperf-edu/paper/figures/training_data.json"
if os.path.exists(data_path):
    with open(data_path) as f:
        saved = json.load(f)
    results = {k: (v["train"], v["val"]) for k, v in saved.items()}
    print(f"Loaded existing data: {list(results.keys())}")
else:
    results = {}

# ─── Train missing workloads ────────────────────────────────────────────────

def train_nanogpt():
    from reference.cloud.nanogpt_core import GPT
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("nanogpt-12m", batch_size=8)
    model = GPT(vocab_size=65, n_layer=6, n_head=6, n_embd=384, max_seq_len=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    t_l, v_l = [], []
    for ep in range(25):
        model.train(); el=0; n=0
        for batch in train_ld:
            x = batch[0].to(device) if isinstance(batch, (list,tuple)) else batch.to(device)
            inp, tgt = x[:, :-1], x[:, 1:]
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); n += 1
        t_l.append(el/max(n,1))
        model.eval(); vl=0; vn=0
        with torch.no_grad():
            for batch in val_ld:
                x = batch[0].to(device) if isinstance(batch, (list,tuple)) else batch.to(device)
                inp, tgt = x[:, :-1], x[:, 1:]
                logits = model(inp)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                vl += loss.item(); vn += 1
        v_l.append(vl/max(vn,1))
        print(f"  GPT ep{ep}: t={t_l[-1]:.3f} v={v_l[-1]:.3f}")
    return t_l, v_l

def train_moe():
    from reference.cloud.nano_moe import NanoMoEWhiteBox
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("nano-moe-12m", batch_size=8)
    model = NanoMoEWhiteBox().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    t_l, v_l = [], []
    for ep in range(25):
        model.train(); el=0; n=0
        for batch in train_ld:
            x = batch[0].to(device) if isinstance(batch, (list,tuple)) else batch.to(device)
            inp, tgt = x[:, :-1], x[:, 1:]
            out = model(inp)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); n += 1
        t_l.append(el/max(n,1))
        model.eval(); vl=0; vn=0
        with torch.no_grad():
            for batch in val_ld:
                x = batch[0].to(device) if isinstance(batch, (list,tuple)) else batch.to(device)
                inp, tgt = x[:, :-1], x[:, 1:]
                out = model(inp)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                vl += loss.item(); vn += 1
        v_l.append(vl/max(vn,1))
        print(f"  MoE ep{ep}: t={t_l[-1]:.4f} v={v_l[-1]:.4f}")
    return t_l, v_l

def train_dlrm():
    from reference.cloud.micro_dlrm import MicroDLRMWhiteBox
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("micro-dlrm-1m", batch_size=256)
    model = MicroDLRMWhiteBox().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_l, v_l = [], []
    for ep in range(20):
        model.train(); el=0; n=0
        for batch in train_ld:
            # batch might be a list of tensors
            if isinstance(batch, (list, tuple)):
                x = torch.stack(batch[:-1], dim=1).to(device) if len(batch) > 2 else batch[0].to(device)
                y = batch[-1].to(device) if len(batch) > 1 else batch[0].to(device)
            else:
                x, y = batch.to(device), None
            try:
                x_dev = x.to(device) if hasattr(x, 'to') else torch.tensor(x).to(device)
                y_dev = y.to(device) if y is not None and hasattr(y, 'to') else y
                out = model(x_dev)
                if y_dev is not None:
                    loss = F.binary_cross_entropy_with_logits(out.squeeze(), y_dev.float())
                else:
                    loss = out.mean()
                opt.zero_grad(); loss.backward(); opt.step()
                el += loss.item(); n += 1
            except Exception as e:
                print(f"  DLRM batch error: {e}")
                break
        if n == 0: return None, None
        t_l.append(el/n)
        model.eval(); vl=0; vn=0
        with torch.no_grad():
            for batch in val_ld:
                try:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0].to(device) if hasattr(batch[0], 'to') else torch.tensor(batch[0]).to(device)
                        y = batch[-1].to(device) if hasattr(batch[-1], 'to') else torch.tensor(batch[-1]).to(device)
                    else:
                        x, y = batch.to(device), None
                    out = model(x)
                    if y is not None:
                        loss = F.binary_cross_entropy_with_logits(out.squeeze(), y.float())
                    else:
                        loss = out.mean()
                    vl += loss.item(); vn += 1
                except:
                    break
        v_l.append(vl/max(vn,1))
        print(f"  DLRM ep{ep}: t={t_l[-1]:.4f} v={v_l[-1]:.4f}")
    return t_l, v_l

def train_gcn():
    from reference.cloud.micro_gnn import MicroGCN
    from reference.dataset_factory import get_dataloaders
    data = get_dataloaders("micro-gcn", batch_size=1)
    features = data['x'].to(device)
    labels = data['y'].to(device)
    adj = data['adj_norm'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    model = MicroGCN(in_features=data['n_features'], hidden=64, num_classes=data['n_classes']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    t_l, v_l = [], []
    for ep in range(200):
        model.train()
        out = model(features, adj)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        t_l.append(loss.item())
        model.eval()
        with torch.no_grad():
            out = model(features, adj)
            vloss = F.cross_entropy(out[val_mask], labels[val_mask])
        v_l.append(vloss.item())
        if ep % 40 == 0: print(f"  GCN ep{ep}: t={t_l[-1]:.4f} v={v_l[-1]:.4f}")
    return t_l, v_l

def train_bert():
    from reference.cloud.micro_bert import MicroBERT
    from reference.dataset_factory import get_dataloaders
    result = get_dataloaders("micro-bert", batch_size=32)
    if isinstance(result, dict):
        train_ld = result.get('train_loader')
        val_ld = result.get('val_loader')
        vocab_size = result.get('vocab_size', 5004)
    elif isinstance(result, tuple) and len(result) == 3:
        train_ld, val_ld, vocab_size = result
    else:
        train_ld, val_ld = result
        vocab_size = 5004
    model = MicroBERT(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_l, v_l = [], []
    for ep in range(20):
        model.train(); el=0; n=0
        for batch in train_ld:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x); loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); n += 1
        t_l.append(el/max(n,1))
        model.eval(); vl=0; vn=0
        with torch.no_grad():
            for batch in val_ld:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x); loss = F.cross_entropy(out, y)
                vl += loss.item(); vn += 1
        v_l.append(vl/max(vn,1))
        print(f"  BERT ep{ep}: t={t_l[-1]:.4f} v={v_l[-1]:.4f}")
    return t_l, v_l

def train_resnet():
    from reference.edge.resnet_core import ResNet18Local
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("resnet18", batch_size=64)
    model = ResNet18Local(num_classes=100).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    t_l, v_l = [], []
    for ep in range(20):
        model.train(); el=0; n=0
        for batch in train_ld:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x); loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); n += 1
        t_l.append(el/max(n,1)); sched.step()
        model.eval(); vl=0; vn=0
        with torch.no_grad():
            for batch in val_ld:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x); loss = F.cross_entropy(out, y)
                vl += loss.item(); vn += 1
        v_l.append(vl/max(vn,1))
        print(f"  ResNet ep{ep}: t={t_l[-1]:.4f} v={v_l[-1]:.4f}")
    return t_l, v_l

def train_mobilenet():
    from reference.mobile.mobilenet_core import MobileNetV2Local
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("mobilenetv2", batch_size=64)
    model = MobileNetV2Local(num_classes=100).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_l, v_l = [], []
    for ep in range(20):
        model.train(); el=0; n=0
        for batch in train_ld:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x); loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); n += 1
        t_l.append(el/max(n,1))
        model.eval(); vl=0; vn=0
        with torch.no_grad():
            for batch in val_ld:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x); loss = F.cross_entropy(out, y)
                vl += loss.item(); vn += 1
        v_l.append(vl/max(vn,1))
        print(f"  MobileV2 ep{ep}: t={t_l[-1]:.4f} v={v_l[-1]:.4f}")
    return t_l, v_l

def train_ae():
    from reference.tiny.anomaly_detection_ae import AnomalyDetectionAE
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("anomaly-ae", batch_size=64)
    model = AnomalyDetectionAE(input_dim=784).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_l, v_l = [], []
    for ep in range(20):
        model.train(); el=0; n=0
        for batch in train_ld:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            x = x.view(x.size(0), -1)
            out = model(x)
            recon = out[0] if isinstance(out, tuple) else out
            loss = F.mse_loss(recon, x)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); n += 1
        t_l.append(el/max(n,1))
        model.eval(); vl=0; vn=0
        with torch.no_grad():
            for batch in val_ld:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                x = x.view(x.size(0), -1)
                out = model(x)
                recon = out[0] if isinstance(out, tuple) else out
                loss = F.mse_loss(recon, x)
                vl += loss.item(); vn += 1
        v_l.append(vl/max(vn,1))
        print(f"  AE ep{ep}: t={t_l[-1]:.4f} v={v_l[-1]:.4f}")
    return t_l, v_l

# ─── Run missing workloads ──────────────────────────────────────────────────

missing = {
    "nanogpt": train_nanogpt,
    "nano-moe": train_moe,
    "micro-dlrm": train_dlrm,
    "micro-gcn": train_gcn,
    "micro-bert": train_bert,
    "resnet18": train_resnet,
    "mobilenetv2": train_mobilenet,
    "anomaly-ae": train_ae,
}

for name, fn in missing.items():
    if name in results and results[name][0] is not None:
        print(f"✓ {name} already trained ({len(results[name][0])} epochs)")
        continue
    print(f"\n{'='*50}\nTraining: {name}\n{'='*50}")
    t0 = time.time()
    try:
        tr, vl = fn()
        if tr is not None:
            results[name] = (tr, vl)
            print(f"  ✓ {name}: {time.time()-t0:.1f}s, {len(tr)} epochs")
        else:
            print(f"  ✗ {name}: returned None")
    except Exception as e:
        print(f"  ✗ {name} FAILED: {e}")
        import traceback; traceback.print_exc()

# Agent/RL placeholders
for k in ["micro-rl", "nanorag", "nanocodegen", "nanoreact"]:
    results.setdefault(k, (None, None))

# Save
save_data = {}
for k, v in results.items():
    if v is not None and v[0] is not None:
        save_data[k] = {"train": v[0], "val": v[1]}
with open(data_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved data for {len(save_data)} workloads")

# ─── Generate 4x4 figure ───────────────────────────────────────────────────

plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(4, 4, figsize=(14, 12))
fig.suptitle("MLPerf EDU — Training Convergence for All 16 Workloads", 
             fontsize=14, fontweight='bold', y=0.99)

workloads = [
    ("NanoGPT", "Cloud", "nanogpt", True),
    ("Nano-MoE", "Cloud", "nano-moe", False),
    ("Micro-DLRM", "Cloud", "micro-dlrm", True),
    ("Micro-Diff", "Cloud", "micro-diffusion", False),
    ("Micro-GCN", "Cloud", "micro-gcn", False),
    ("Micro-BERT", "Cloud", "micro-bert", False),
    ("Micro-LSTM", "Cloud", "micro-lstm", False),
    ("Micro-RL", "Cloud", "micro-rl", False),
    ("ResNet-18", "Edge", "resnet18", True),
    ("MobileNetV2", "Edge", "mobilenetv2", False),
    ("DS-CNN", "Tiny", "dscnn", True),
    ("Anomaly AE", "Tiny", "anomaly-ae", False),
    ("VWW", "Tiny", "vww", False),
    ("NanoRAG", "Agent", "nanorag", False),
    ("NanoCodeGen", "Agent", "nanocodegen", False),
    ("NanoReAct", "Agent", "nanoreact", True),
]

div_colors = {'Cloud': '#2196F3', 'Edge': '#4CAF50', 'Tiny': '#FF9800', 'Agent': '#9C27B0'}

for idx, (ax, (name, div, key, core)) in enumerate(zip(axes.flat, workloads)):
    c = div_colors[div]
    star = " [core]" if core else ""
    
    if key in results and results[key] is not None and results[key][0] is not None:
        tr, vl = results[key]
        epochs = range(len(tr))
        ax.plot(epochs, tr, color=c, linewidth=1.5, label='Train', alpha=0.9)
        ax.plot(epochs, vl, color='#E53935', linewidth=1.5, linestyle='--', label='Val', alpha=0.9)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel("Loss", fontsize=7)
        ax.legend(fontsize=6, loc='upper right', framealpha=0.8)
        ax.grid(True, alpha=0.2, linewidth=0.5)
    else:
        label = "Agent workload\n(non-traditional\ntraining loop)" if div == "Agent" else "RL workload\n(reward, not loss)"
        ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=8, 
                style='italic', transform=ax.transAxes, color='gray')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linestyle(':'); s.set_alpha(0.3)
    
    ax.set_title(f"({chr(97+idx)}) {name}{star}", fontsize=8, fontweight='bold', color=c)
    ax.tick_params(labelsize=6)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = "/Users/VJ/GitHub/mlperf-edu/paper/figures/all_training_curves.pdf"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.savefig(out_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
print(f"\nFigure saved: {out_path}")
plt.close()
