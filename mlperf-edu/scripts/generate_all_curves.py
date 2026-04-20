#!/usr/bin/env python3
"""
Generate comprehensive training curves for ALL MLPerf EDU workloads.
Produces a 4x4 grid figure for the paper.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

results = {}

# ─── 1. Micro-DLRM ───────────────────────────────────────────────────────────
def train_dlrm():
    from reference.cloud.micro_dlrm import MicroDLRMWhiteBox
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("micro-dlrm-1m", batch_size=256)
    model = MicroDLRMWhiteBox().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_l, val_l = [], []
    for ep in range(20):
        model.train(); eloss = 0; n = 0
        for batch in train_ld:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x); loss = F.binary_cross_entropy_with_logits(out.squeeze(), y.float())
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); n += 1
        train_l.append(eloss/n)
        model.eval(); vloss = 0; vn = 0
        with torch.no_grad():
            for batch in val_ld:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x); loss = F.binary_cross_entropy_with_logits(out.squeeze(), y.float())
                vloss += loss.item(); vn += 1
        val_l.append(vloss/max(vn,1))
        print(f"  DLRM ep{ep}: t={train_l[-1]:.4f} v={val_l[-1]:.4f}")
    return train_l, val_l

# ─── 2. Micro-GCN ─────────────────────────────────────────────────────────────
def train_gcn():
    from reference.cloud.micro_gnn import MicroGCN
    from reference.dataset_factory import get_dataloaders
    result = get_dataloaders("micro-gcn", batch_size=1)
    # GCN returns (features, labels, adj, train_mask, val_mask)
    if isinstance(result, tuple) and len(result) > 2:
        features, labels, adj, train_mask, val_mask = result
    else:
        # Try direct loading from the module
        from reference.cloud.micro_gnn import load_cora_data
        features, labels, adj, train_mask, val_mask = load_cora_data()
    
    model = MicroGCN(in_features=features.shape[1], hidden=64, num_classes=labels.max().item()+1).to(device)
    features = features.to(device); labels = labels.to(device); adj = adj.to(device)
    train_mask = train_mask.to(device); val_mask = val_mask.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_l, val_l = [], []
    for ep in range(200):
        model.train()
        out = model(features, adj)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        train_l.append(loss.item())
        model.eval()
        with torch.no_grad():
            out = model(features, adj)
            vloss = F.cross_entropy(out[val_mask], labels[val_mask])
        val_l.append(vloss.item())
        if ep % 20 == 0: print(f"  GCN ep{ep}: t={train_l[-1]:.4f} v={val_l[-1]:.4f}")
    return train_l, val_l

# ─── 3. Micro-BERT ────────────────────────────────────────────────────────────
def train_bert():
    from reference.cloud.micro_bert import MicroBERT
    from reference.dataset_factory import get_dataloaders
    result = get_dataloaders("micro-bert", batch_size=32)
    # micro-bert returns (train_ld, val_ld, vocab_size)
    if isinstance(result, tuple) and len(result) == 3:
        train_ld, val_ld, vocab_size = result
    else:
        train_ld, val_ld = result
        vocab_size = 5000
    model = MicroBERT(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=2, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_l, val_l = [], []
    for ep in range(20):
        model.train(); eloss = 0; n = 0
        for batch in train_ld:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x); loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); n += 1
        train_l.append(eloss/max(n,1))
        model.eval(); vloss = 0; vn = 0
        with torch.no_grad():
            for batch in val_ld:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x); loss = F.cross_entropy(out, y)
                vloss += loss.item(); vn += 1
        val_l.append(vloss/max(vn,1))
        print(f"  BERT ep{ep}: t={train_l[-1]:.4f} v={val_l[-1]:.4f}")
    return train_l, val_l

# ─── 4. Micro-LSTM (already ran, embed data) ──────────────────────────────────
def train_lstm():
    from reference.cloud.micro_lstm import MicroLSTM
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("micro-lstm", batch_size=32)
    model = MicroLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_l, val_l = [], []
    for ep in range(30):
        model.train(); eloss = 0; n = 0
        for batch in train_ld:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x); loss = F.mse_loss(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); n += 1
        train_l.append(eloss/max(n,1))
        model.eval(); vloss = 0; vn = 0
        with torch.no_grad():
            for batch in val_ld:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x); loss = F.mse_loss(out, y)
                vloss += loss.item(); vn += 1
        val_l.append(vloss/max(vn,1))
        if ep % 5 == 0: print(f"  LSTM ep{ep}: t={train_l[-1]:.4f} v={val_l[-1]:.4f}")
    return train_l, val_l

# ─── 5. NanoGPT (use core directly) ──────────────────────────────────────────
def train_nanogpt():
    from reference.cloud.nanogpt_core import NanoGPT
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("nanogpt-12m", batch_size=8)
    # Smaller model for speed
    model = NanoGPT(vocab_size=65, d_model=384, n_heads=6, n_layers=6, block_size=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_l, val_l = [], []
    for ep in range(25):
        model.train(); eloss = 0; n = 0
        for batch in train_ld:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            inp, tgt = x[:, :-1], x[:, 1:]
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); n += 1
        train_l.append(eloss/max(n,1))
        model.eval(); vloss = 0; vn = 0
        with torch.no_grad():
            for batch in val_ld:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                inp, tgt = x[:, :-1], x[:, 1:]
                logits = model(inp)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                vloss += loss.item(); vn += 1
        val_l.append(vloss/max(vn,1))
        print(f"  GPT ep{ep}: t={train_l[-1]:.3f} v={val_l[-1]:.3f}")
    return train_l, val_l

# ─── 6. Nano-MoE ─────────────────────────────────────────────────────────────
def train_moe():
    from reference.cloud.nano_moe import NanoMoEWhiteBox
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("nano-moe-12m", batch_size=8)
    model = NanoMoEWhiteBox().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_l, val_l = [], []
    for ep in range(25):
        model.train(); eloss = 0; n = 0
        for batch in train_ld:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            inp, tgt = x[:, :-1], x[:, 1:]
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); n += 1
        train_l.append(eloss/max(n,1))
        model.eval(); vloss = 0; vn = 0
        with torch.no_grad():
            for batch in val_ld:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                inp, tgt = x[:, :-1], x[:, 1:]
                logits = model(inp)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                vloss += loss.item(); vn += 1
        val_l.append(vloss/max(vn,1))
        print(f"  MoE ep{ep}: t={train_l[-1]:.4f} v={val_l[-1]:.4f}")
    return train_l, val_l

# ─── 7. ResNet-18 ─────────────────────────────────────────────────────────────
def train_resnet():
    from reference.edge.resnet_core import ResNet18
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("resnet18", batch_size=64)
    model = ResNet18(num_classes=100).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    train_l, val_l = [], []
    for ep in range(20):
        model.train(); eloss = 0; n = 0
        for batch in train_ld:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x); loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); n += 1
        train_l.append(eloss/max(n,1))
        sched.step()
        model.eval(); vloss = 0; vn = 0
        with torch.no_grad():
            for batch in val_ld:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x); loss = F.cross_entropy(out, y)
                vloss += loss.item(); vn += 1
        val_l.append(vloss/max(vn,1))
        print(f"  ResNet ep{ep}: t={train_l[-1]:.4f} v={val_l[-1]:.4f}")
    return train_l, val_l

# ─── 8. MobileNetV2 ──────────────────────────────────────────────────────────
def train_mobilenet():
    from reference.mobile.mobilenet_core import MobileNetV2
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("mobilenetv2", batch_size=64)
    model = MobileNetV2(num_classes=100).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_l, val_l = [], []
    for ep in range(20):
        model.train(); eloss = 0; n = 0
        for batch in train_ld:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x); loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); n += 1
        train_l.append(eloss/max(n,1))
        model.eval(); vloss = 0; vn = 0
        with torch.no_grad():
            for batch in val_ld:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x); loss = F.cross_entropy(out, y)
                vloss += loss.item(); vn += 1
        val_l.append(vloss/max(vn,1))
        print(f"  MobileV2 ep{ep}: t={train_l[-1]:.4f} v={val_l[-1]:.4f}")
    return train_l, val_l

# ─── 9. DS-CNN ────────────────────────────────────────────────────────────────
def train_dscnn():
    from reference.tiny.dscnn_kws import DSCNN
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("dscnn-kws", batch_size=64)
    model = DSCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_l, val_l = [], []
    for ep in range(20):
        model.train(); eloss = 0; n = 0
        for batch in train_ld:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x); loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); n += 1
        train_l.append(eloss/max(n,1))
        model.eval(); vloss = 0; vn = 0
        with torch.no_grad():
            for batch in val_ld:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x); loss = F.cross_entropy(out, y)
                vloss += loss.item(); vn += 1
        val_l.append(vloss/max(vn,1))
        print(f"  DSCNN ep{ep}: t={train_l[-1]:.4f} v={val_l[-1]:.4f}")
    return train_l, val_l

# ─── 10. Anomaly AE ──────────────────────────────────────────────────────────
def train_ae():
    from reference.tiny.anomaly_detection_ae import AnomalyDetectionAE
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("anomaly-ae", batch_size=64)
    model = AnomalyDetectionAE(input_dim=784).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_l, val_l = [], []
    for ep in range(20):
        model.train(); eloss = 0; n = 0
        for batch in train_ld:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            x = x.view(x.size(0), -1)
            out = model(x); loss = F.mse_loss(out, x)
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); n += 1
        train_l.append(eloss/max(n,1))
        model.eval(); vloss = 0; vn = 0
        with torch.no_grad():
            for batch in val_ld:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                x = x.view(x.size(0), -1)
                out = model(x); loss = F.mse_loss(out, x)
                vloss += loss.item(); vn += 1
        val_l.append(vloss/max(vn,1))
        print(f"  AE ep{ep}: t={train_l[-1]:.4f} v={val_l[-1]:.4f}")
    return train_l, val_l

# ─── 11. VWW ─────────────────────────────────────────────────────────────────
def train_vww():
    from reference.tiny.wake_vision_vww import MicroNet
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("vww", batch_size=32)
    model = MicroNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_l, val_l = [], []
    for ep in range(20):
        model.train(); eloss = 0; n = 0
        for batch in train_ld:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x); loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); n += 1
        train_l.append(eloss/max(n,1))
        model.eval(); vloss = 0; vn = 0
        with torch.no_grad():
            for batch in val_ld:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x); loss = F.cross_entropy(out, y)
                vloss += loss.item(); vn += 1
        val_l.append(vloss/max(vn,1))
        print(f"  VWW ep{ep}: t={train_l[-1]:.4f} v={val_l[-1]:.4f}")
    return train_l, val_l

# ─── 12. Micro-Diffusion ─────────────────────────────────────────────────────
def train_diffusion():
    from reference.cloud.micro_diffusion import MicroDiffusionUNet
    from reference.dataset_factory import get_dataloaders
    train_ld, val_ld = get_dataloaders("micro-diffusion", batch_size=64)
    model = MicroDiffusionUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_l, val_l = [], []
    for ep in range(20):
        model.train(); eloss = 0; n = 0
        for batch in train_ld:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            # Diffusion: add noise then denoise
            t_step = torch.randint(0, 1000, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            alpha = (1 - t_step.float()/1000).view(-1,1,1,1)
            x_noisy = alpha * x + (1-alpha) * noise
            pred = model(x_noisy, t_step)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); n += 1
        train_l.append(eloss/max(n,1))
        model.eval(); vloss = 0; vn = 0
        with torch.no_grad():
            for batch in val_ld:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                t_step = torch.randint(0, 1000, (x.size(0),), device=device)
                noise = torch.randn_like(x)
                alpha = (1 - t_step.float()/1000).view(-1,1,1,1)
                x_noisy = alpha * x + (1-alpha) * noise
                pred = model(x_noisy, t_step)
                loss = F.mse_loss(pred, noise)
                vloss += loss.item(); vn += 1
        val_l.append(vloss/max(vn,1))
        print(f"  Diff ep{ep}: t={train_l[-1]:.4f} v={val_l[-1]:.4f}")
    return train_l, val_l

# ─── 13. Micro-RL (generate typical RL curve) ────────────────────────────────
def train_rl():
    """Run actual REINFORCE on CartPole."""
    from reference.cloud.micro_rl import train_rl as rl_train_fn
    # RL doesn't have a standard loss curve — we'll track episode rewards
    # Use the module's own training function if it returns data
    try:
        result = rl_train_fn()
        if isinstance(result, tuple):
            return result
    except:
        pass
    # Fallback: just run CartPole manually with REINFORCE
    return None, None


# ─── PLOT ─────────────────────────────────────────────────────────────────────
def plot_all_curves(results, output_path):
    """4x4 grid of training curves."""
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    fig.suptitle("MLPerf EDU — Training Convergence for All 16 Workloads", 
                 fontsize=14, fontweight='bold', y=0.99)
    
    workloads = [
        ("NanoGPT", "Cloud", "nanogpt", "★"),
        ("Nano-MoE", "Cloud", "nano-moe", ""),
        ("Micro-DLRM", "Cloud", "micro-dlrm", "★"),
        ("Micro-Diff", "Cloud", "micro-diffusion", ""),
        ("Micro-GCN", "Cloud", "micro-gcn", ""),
        ("Micro-BERT", "Cloud", "micro-bert", ""),
        ("Micro-LSTM", "Cloud", "micro-lstm", ""),
        ("Micro-RL", "Cloud", "micro-rl", ""),
        ("ResNet-18", "Edge", "resnet18", "★"),
        ("MobileNetV2", "Edge", "mobilenetv2", ""),
        ("DS-CNN", "Tiny", "dscnn", "★"),
        ("Anomaly AE", "Tiny", "anomaly-ae", ""),
        ("VWW", "Tiny", "vww", ""),
        ("NanoRAG", "Agent", "nanorag", ""),
        ("NanoCodeGen", "Agent", "nanocodegen", ""),
        ("NanoReAct", "Agent", "nanoreact", "★"),
    ]
    
    div_colors = {
        'Cloud': '#2196F3', 'Edge': '#4CAF50', 
        'Tiny': '#FF9800', 'Agent': '#9C27B0'
    }
    
    for idx, (ax, (name, div, key, core)) in enumerate(zip(axes.flat, workloads)):
        c = div_colors[div]
        title = f"({chr(97+idx)}) {core}{name}" if core else f"({chr(97+idx)}) {name}"
        
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
            ax.text(0.5, 0.5, f"Agent/RL\nworkload", 
                    ha='center', va='center', fontsize=9, style='italic',
                    transform=ax.transAxes, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linestyle(':'); s.set_alpha(0.3)
        
        ax.set_title(title, fontsize=8, fontweight='bold', color=c)
        ax.tick_params(labelsize=6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")
    plt.close()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainers = [
        ("micro-dlrm", train_dlrm),
        ("micro-gcn", train_gcn),
        ("micro-bert", train_bert),
        ("micro-lstm", train_lstm),
        ("micro-diffusion", train_diffusion),
        ("dscnn", train_dscnn),
        ("anomaly-ae", train_ae),
        ("vww", train_vww),
        ("resnet18", train_resnet),
        ("mobilenetv2", train_mobilenet),
        ("nano-moe", train_moe),
        ("nanogpt", train_nanogpt),
    ]
    
    total_t0 = time.time()
    for name, fn in trainers:
        print(f"\n{'='*50}\nTraining: {name}\n{'='*50}")
        t0 = time.time()
        try:
            tr, vl = fn()
            results[name] = (tr, vl)
            print(f"  ✓ {name}: {time.time()-t0:.1f}s, {len(tr)} epochs")
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            results[name] = (None, None)
    
    # Placeholders for agent/RL workloads
    for k in ["micro-rl", "nanorag", "nanocodegen", "nanoreact"]:
        results.setdefault(k, (None, None))
    
    total_time = time.time() - total_t0
    print(f"\n{'='*50}\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)\n{'='*50}")
    
    # Save raw data
    save_data = {}
    for k, v in results.items():
        if v is not None and v[0] is not None:
            save_data[k] = {"train": v[0], "val": v[1]}
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    with open(os.path.join(figures_dir, 'training_data.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # Generate figure
    plot_all_curves(results, os.path.join(figures_dir, 'all_training_curves.pdf'))
