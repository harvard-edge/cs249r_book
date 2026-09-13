#!/usr/bin/env python3
"""
Generate high-resolution labeled and unlabeled isometric blueprint chapter openers
for Volume 3 (Agentic Machine Learning Systems) Chapters 01 - 18.
"""

import os
import math
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
BRAIN = '/Users/VJ/.gemini/antigravity-cli/brain/6702a6d9-86c1-48b2-93d2-f2c949a6e416'

FONT_PATH = '/System/Library/Fonts/Helvetica.ttc'
if not os.path.exists(FONT_PATH):
    FONT_PATH = '/System/Library/Fonts/Supplemental/Arial.ttf'

try:
    font = ImageFont.truetype(FONT_PATH, 16)
except Exception:
    font = ImageFont.load_default()

def draw_pill(draw, text, cx, cy, border_color='#1A4D3E', text_color='#1A4D3E'):
    try:
        font_pill = ImageFont.truetype(FONT_PATH, 15)
    except Exception:
        font_pill = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font_pill)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    px, py = 12, 6
    bx0, by0 = cx - tw // 2 - px, cy - th // 2 - py
    bx1, by1 = cx + tw // 2 + px, cy + th // 2 + py
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=8, fill='white', outline=border_color, width=2)
    draw.text((cx - tw // 2, cy - th // 2 - 1), text, fill=text_color, font=font_pill)
    return (bx0, by0, bx1, by1)

def draw_leader(draw, p_box, p_target, color='#1A4D3E'):
    draw.line([p_box, p_target], fill=color, width=2)
    r = 3.5
    draw.ellipse([p_target[0] - r, p_target[1] - r, p_target[0] + r, p_target[1] + r], fill=color)

def process_chapter(slug, base_filename, labels):
    base_path = os.path.join(BRAIN, base_filename)
    im_base = Image.open(base_path).convert('RGB')
    
    # Standard 1400 x 800 blueprint canvas
    canvas = Image.new('RGB', (1400, 800), (255, 255, 255))
    dx = (1400 - im_base.width) // 2
    dy = (800 - im_base.height) // 2
    canvas.paste(im_base, (dx, dy))
    
    unlabeled = canvas.copy()
    labeled = unlabeled.copy()
    draw = ImageDraw.Draw(labeled)
    
    for item in labels:
        text = item['text']
        px, py = item['pill']
        tx = dx + item['targ'][0]
        ty = dy + item['targ'][1]
        border = item.get('border', '#0F172A')
        tcolor = item.get('color', '#0F172A')
        
        try:
            font_pill = ImageFont.truetype(FONT_PATH, 15)
        except Exception:
            font_pill = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font_pill)
        tw = (bbox[2] - bbox[0]) // 2 + 12
        if tx < px:
            anchor = (px - tw, py)
        else:
            anchor = (px + tw, py)
            
        draw_leader(draw, anchor, (tx, ty), color=border)
        draw_pill(draw, text, px, py, border_color=border, text_color=tcolor)
        
    ch_dir = os.path.join(BASE, slug)
    png_dir = os.path.join(ch_dir, 'images', 'png')
    webp_dir = os.path.join(ch_dir, 'images', 'webp')
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(webp_dir, exist_ok=True)
    
    parts = slug.split('_', 1)
    prefix = parts[1] if len(parts) > 1 else parts[0]
    
    unlabeled_png = os.path.join(png_dir, f'cover_{prefix}_blueprint.png')
    unlabeled_webp = os.path.join(webp_dir, f'cover_{prefix}_blueprint.webp')
    labeled_png = os.path.join(png_dir, f'cover_{prefix}_blueprint_labeled_print.png')
    labeled_webp = os.path.join(webp_dir, f'cover_{prefix}_blueprint_labeled.webp')
    
    unlabeled.save(unlabeled_png, 'PNG')
    unlabeled.save(unlabeled_webp, 'WEBP', quality=95)
    labeled.save(labeled_png, 'PNG')
    labeled.save(labeled_webp, 'WEBP', quality=95)
    
    brain_proof = os.path.join(BRAIN, f'final_{slug}_labeled.png')
    labeled.save(brain_proof, 'PNG')
    print(f'Processed {slug} -> {labeled_png}')

# =============================================================================
# CHAPTER LABEL DEFINITIONS (ALL 18 CHAPTERS)
# =============================================================================

# Chapter 01: Introduction — The Von Neumann Agent Architecture
labels_01_intro = [
    {'text': 'stochastic processor socket', 'pill': (240, 160), 'targ': (640, 380), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'hbm3e memory stacks', 'pill': (240, 360), 'targ': (580, 360)},
    {'text': 'pcie tool expansion bus', 'pill': (240, 560), 'targ': (350, 480), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'l1 sram ring buffer', 'pill': (700, 80), 'targ': (690, 240), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'l2 paged block-table cache', 'pill': (1160, 200), 'targ': (880, 270), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'l3 episodic storage vault', 'pill': (1180, 420), 'targ': (1070, 400), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'isolated actuation interface', 'pill': (1160, 620), 'targ': (940, 550), 'border': '#9B2226', 'color': '#9B2226'},
]

# Chapter 02: The Stochastic Processor Contract
labels_02_processor = [
    {'text': 'draft model execution core', 'pill': (240, 220), 'targ': (320, 390), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'speculative candidate token bus', 'pill': (260, 460), 'targ': (540, 450), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'memory-bandwidth streaming', 'pill': (660, 80), 'targ': (680, 240), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'target model verification engine', 'pill': (1160, 180), 'targ': (950, 300), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'crimson rollback diversion gate', 'pill': (1180, 420), 'targ': (880, 550), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'parallel acceptance retirement', 'pill': (1180, 640), 'targ': (780, 640), 'border': '#2EC4B6', 'color': '#2EC4B6'},
]

# Chapter 03: Test-Time Deliberation and Search
labels_03_deliberation = [
    {'text': 'root decision state', 'pill': (220, 560), 'targ': (340, 570), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'active exploratory branch', 'pill': (240, 240), 'targ': (540, 380), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'optimal high-reward path', 'pill': (600, 80), 'targ': (570, 440), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'pruned dead-end containment cap', 'pill': (980, 80), 'targ': (760, 330), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'speculative rollout branch', 'pill': (1180, 380), 'targ': (960, 440), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'value-heuristic telemetry rail', 'pill': (740, 740), 'targ': (800, 650)},
]

# Chapter 04: Context Working Sets and Attention Budgets
labels_04_working_sets = [
    {'text': 'permanent attention sink tokens', 'pill': (260, 140), 'targ': (690, 220), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'active sliding window working set', 'pill': (240, 360), 'targ': (770, 240), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'rotating context carousel', 'pill': (260, 580), 'targ': (620, 440)},
    {'text': 'quadratic attention envelope', 'pill': (1160, 200), 'targ': (770, 360), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'eviction discharge chute', 'pill': (1180, 440), 'targ': (960, 430), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'working set budget boundary', 'pill': (760, 740), 'targ': (800, 490), 'border': '#9B2226', 'color': '#9B2226'},
]

# Chapter 05: Virtual Memory & PagedAttention
labels_05_virtual_memory = [
    {'text': 'logical virtual blocks', 'pill': (240, 180), 'targ': (480, 300), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'unmapped sequence streams', 'pill': (240, 420), 'targ': (410, 440), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'radix block table mmu', 'pill': (680, 80), 'targ': (680, 420), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'physical hbm page frames', 'pill': (1160, 200), 'targ': (940, 460), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'shared prefix blocks', 'pill': (1180, 440), 'targ': (810, 480), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'zero-fragmentation page allocation', 'pill': (760, 740), 'targ': (730, 660), 'border': '#2EC4B6', 'color': '#2EC4B6'},
]

# Chapter 06: Episodic & External Memory Systems
labels_06_episodic = [
    {'text': 'on-chip context cache tier', 'pill': (260, 140), 'targ': (630, 200), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'optical vertical elevator', 'pill': (240, 380), 'targ': (580, 420)},
    {'text': 'semantic similarity bus', 'pill': (260, 620), 'targ': (580, 580), 'border': '#D97706', 'color': '#D97706'},
    {'text': '3d hnsw vector embedding lattice', 'pill': (1180, 260), 'targ': (780, 400), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'persistent episodic vault', 'pill': (1180, 520), 'targ': (780, 620), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'long-term memory consolidation', 'pill': (760, 740), 'targ': (690, 680), 'border': '#D97706', 'color': '#D97706'},
]

# Chapter 07: State Checkpointing & Time-Travel Recovery
labels_07_checkpointing = [
    {'text': 'transaction execution stream', 'pill': (240, 480), 'targ': (380, 620), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'verified checkpoint anchor', 'pill': (400, 160), 'targ': (560, 490), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'execution fault alarm sensor', 'pill': (880, 80), 'targ': (840, 340), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'committed transaction bus', 'pill': (1180, 220), 'targ': (960, 250), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'compensating rollback conduit', 'pill': (1180, 520), 'targ': (850, 580), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'deterministic state rewind path', 'pill': (680, 740), 'targ': (660, 620), 'border': '#9B2226', 'color': '#9B2226'},
]

# Chapter 08: Tool Actuation & The Environment Interface
labels_08_actuation = [
    {'text': 'agent token serialization decoder', 'pill': (240, 240), 'targ': (330, 390), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'central mcp system call distributor', 'pill': (480, 100), 'targ': (630, 460), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'isolated bash terminal executor', 'pill': (900, 80), 'targ': (840, 270), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'sql database query docking bay', 'pill': (680, 60), 'targ': (680, 180)},
    {'text': 'hardware physical actuator bay', 'pill': (1200, 260), 'targ': (1090, 410)},
    {'text': 'json-schema validation iris', 'pill': (1200, 480), 'targ': (960, 580), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'schema violation purge valve', 'pill': (760, 740), 'targ': (800, 700), 'border': '#9B2226', 'color': '#9B2226'},
]

# Chapter 09: System Virtualization & Isolation
labels_09_virtualization = [
    {'text': 'isolated autonomous execution core', 'pill': (240, 260), 'targ': (670, 440), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'air-gapped virtual network bridge', 'pill': (240, 540), 'targ': (520, 650), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'microvm containment bell jar', 'pill': (700, 70), 'targ': (690, 240), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'sandboxed device emulator', 'pill': (1160, 180), 'targ': (920, 230)},
    {'text': 'seccomp-bpf syscall filter grate', 'pill': (1180, 380), 'targ': (760, 550), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'ephemeral copy-on-write rootfs', 'pill': (1180, 600), 'targ': (840, 620), 'border': '#2EC4B6', 'color': '#2EC4B6'},
]

# Chapter 10: Asynchronous Interrupts & Human-in-the-Loop Escrow
labels_10_interrupts = [
    {'text': 'optical interrupt tripwire line', 'pill': (240, 180), 'targ': (600, 310), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'apic preemption controller', 'pill': (240, 380), 'targ': (650, 370), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'continuous batch gpu core', 'pill': (260, 600), 'targ': (660, 610), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'zero-idle pcie offload bus', 'pill': (700, 80), 'targ': (820, 320), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'host dram state escrow vault', 'pill': (1180, 240), 'targ': (980, 340), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'human authorization terminal', 'pill': (1180, 520), 'targ': (900, 580), 'border': '#D97706', 'color': '#D97706'},
]

# Chapter 11: Trajectory Scheduling & Cluster Resource Management
labels_11_scheduling = [
    {'text': 'prompt activation tensor stream', 'pill': (240, 160), 'targ': (380, 260), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'compute-bound prefill blade bank', 'pill': (240, 420), 'targ': (470, 380), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'optical cxl interconnect fabric', 'pill': (700, 80), 'targ': (680, 450), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'streaming autoregressive tokens', 'pill': (1180, 200), 'targ': (1060, 400), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'memory-bound decode blade bank', 'pill': (1180, 460), 'targ': (930, 490), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'phase-interference isolation', 'pill': (720, 740), 'targ': (760, 580), 'border': '#9B2226', 'color': '#9B2226'},
]

# Chapter 12: Trajectory Data Flywheels & Synthesis
labels_12_flywheel = [
    {'text': 'raw multi-turn trajectory intake', 'pill': (220, 200), 'targ': (230, 270), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'syntax validation scanner', 'pill': (240, 440), 'targ': (440, 380), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'environment state-diff verifier', 'pill': (660, 80), 'targ': (650, 420), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'deduplication & decontamination gantry', 'pill': (960, 80), 'targ': (890, 360), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'validated policy training cassettes', 'pill': (1180, 260), 'targ': (1070, 340), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'automated trajectory conveyor', 'pill': (700, 740), 'targ': (640, 520)},
]

# Chapter 13: Supervised Trajectory Fine-Tuning (SFT)
labels_13_sft = [
    {'text': 'interleaved trajectory track', 'pill': (240, 220), 'targ': (420, 440), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'observation masking plates (zero loss)', 'pill': (240, 460), 'targ': (430, 380), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'thought & action gradient illumination', 'pill': (560, 80), 'targ': (580, 480), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'gradient backpropagation rollers', 'pill': (1180, 220), 'targ': (840, 450), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'policy parameter update inductors', 'pill': (1180, 460), 'targ': (940, 410), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'teacher-forcing loss pipeline', 'pill': (740, 740), 'targ': (740, 560), 'border': '#2EC4B6', 'color': '#2EC4B6'},
]

# Chapter 14: Reinforcement Learning with Verifiable Rewards (RLVR)
labels_14_rlvr = [
    {'text': 'step-level process reward calipers', 'pill': (240, 200), 'targ': (480, 350), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'intermediate reasoning state rail', 'pill': (240, 440), 'targ': (520, 440), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'negative penalty grounding line', 'pill': (300, 680), 'targ': (580, 560), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'environment outcome test chamber', 'pill': (1180, 200), 'targ': (880, 360), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'dual-reward gradient backprop manifold', 'pill': (1180, 440), 'targ': (720, 500), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'verified positive scalar reinforcement', 'pill': (760, 740), 'targ': (750, 610), 'border': '#2EC4B6', 'color': '#2EC4B6'},
]

# Chapter 15: Multi-Agent Topologies & Distributed Consensus
labels_15_multi_agent = [
    {'text': 'autonomous specialized agent nodes', 'pill': (240, 220), 'targ': (420, 420), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'proposal transmission ring', 'pill': (240, 460), 'targ': (560, 320), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'shared blackboard memory repository', 'pill': (700, 80), 'targ': (690, 420), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'lock-free atomic read-write channels', 'pill': (1180, 200), 'targ': (850, 270), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'state synchronization update bus', 'pill': (1180, 440), 'targ': (760, 470), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'distributed consensus arbitration ring', 'pill': (740, 740), 'targ': (690, 600), 'border': '#D97706', 'color': '#D97706'},
]

# Chapter 16: Distributed Telemetry, Tracing & Evaluation
labels_16_observability = [
    {'text': 'llm prefill latency span', 'pill': (240, 220), 'targ': (570, 350), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'tool execution wait span', 'pill': (240, 440), 'targ': (620, 410), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'verified checkpoint anchor', 'pill': (260, 640), 'targ': (640, 520), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'failure flame-graph anomaly column', 'pill': (1180, 200), 'targ': (870, 320), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'timecode latency caliper', 'pill': (1180, 440), 'targ': (760, 510)},
    {'text': '3d gantt trajectory flight recorder', 'pill': (720, 740), 'targ': (600, 450), 'border': '#0090B0', 'color': '#0090B0'},
]

# Chapter 17: System Tokenomics, Capacity Sizing & Cost Modeling
labels_17_tokenomics = [
    {'text': 'incoming task prompt stream', 'pill': (220, 260), 'targ': (300, 460), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'complexity difficulty classifier', 'pill': (240, 500), 'targ': (540, 470), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'low-cost local model routing path', 'pill': (660, 80), 'targ': (680, 340), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'lightweight edge model socket', 'pill': (1160, 160), 'targ': (760, 260), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'frontier model accelerator cluster', 'pill': (1180, 360), 'targ': (1000, 420), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'high-throughput frontier bus', 'pill': (1180, 560), 'targ': (840, 480), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'real-time token cost & latency meters', 'pill': (700, 740), 'targ': (900, 540), 'border': '#D97706', 'color': '#D97706'},
]

# Chapter 18: Conclusion — The Autonomous Frontier & Systems Invariants
labels_18_conclusion = [
    {'text': 'digital agent computing matrix', 'pill': (240, 240), 'targ': (440, 470), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'formal invariant closure ring', 'pill': (680, 70), 'targ': (700, 380), 'border': '#2EC4B6', 'color': '#2EC4B6'},
    {'text': 'verified trajectory beam', 'pill': (640, 720), 'targ': (730, 450), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'physical boundary gateway', 'pill': (1180, 240), 'targ': (900, 360), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'open autonomous frontier horizon', 'pill': (1180, 500), 'targ': (1040, 290), 'border': '#D97706', 'color': '#D97706'},
]

if __name__ == '__main__':
    print('Starting Volume 3 Chapter Opener Generation...')
    process_chapter('01_introduction', 'vol3_ch01_opt_a_motherboard_1789305798861.jpg', labels_01_intro)
    process_chapter('02_processor', 'vol3_ch02_opt_b_speculative_decoding_1789305855071.jpg', labels_02_processor)
    process_chapter('03_deliberation', 'vol3_ch03_opt_a_tree_manifold_1789305887191.jpg', labels_03_deliberation)
    process_chapter('04_working_sets', 'vol3_ch04_opt_a_sliding_window_ring_1789305962313.jpg', labels_04_working_sets)
    process_chapter('05_virtual_memory', 'vol3_ch05_opt_a_paged_kv_mmu_1789306214410.jpg', labels_05_virtual_memory)
    process_chapter('06_episodic_memory', 'vol3_ch06_opt_a_hierarchical_memory_tower_1789306272011.jpg', labels_06_episodic)
    process_chapter('07_checkpointing', 'vol3_ch07_opt_a_distributed_saga_rollback_1789306331658.jpg', labels_07_checkpointing)
    process_chapter('08_actuation', 'vol3_ch08_opt_a_mcp_system_call_bus_1789306419505.jpg', labels_08_actuation)
    process_chapter('09_virtual_memory' if False else '09_virtualization', 'vol3_ch09_opt_a_microvm_containment_1789306492233.jpg', labels_09_virtualization)
    process_chapter('10_interrupts', 'vol3_ch10_opt_a_apic_state_escrow_1789306563113.jpg', labels_10_interrupts)
    process_chapter('11_scheduling', 'vol3_ch11_opt_a_prefill_decode_disaggregation_1789306645329.jpg', labels_11_scheduling)
    process_chapter('12_data_flywheel', 'vol3_ch12_opt_b_trajectory_distillation_line_1789306805102.jpg', labels_12_flywheel)
    process_chapter('13_sft', 'vol3_ch13_opt_a_loss_masking_loom_1789306869293.jpg', labels_13_sft)
    process_chapter('14_rlvr', 'vol3_ch14_opt_b_process_vs_outcome_engine_1789306990581.jpg', labels_14_rlvr)
    process_chapter('15_multi_agent', 'vol3_ch15_opt_b_blackboard_memory_ring_1789307134078.jpg', labels_15_multi_agent)
    process_chapter('16_observability', 'vol3_ch16_opt_a_trajectory_flight_recorder_1789307205556.jpg', labels_16_observability)
    process_chapter('17_tokenomics', 'vol3_ch17_opt_a_dynamic_routing_switchboard_1789307309143.jpg', labels_17_tokenomics)
    process_chapter('18_conclusion', 'vol3_ch18_opt_b_invariant_closure_horizon_1789307677035.jpg', labels_18_conclusion)
    print('Finished all 18 chapters!')
