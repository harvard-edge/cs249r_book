#!/usr/bin/env python3
"""
Generate high-resolution labeled and unlabeled isometric blueprint chapter openers
for Volume 3 (Agentic Machine Learning Systems) Chapters 01 - 18.
Standard: Wave 1 Instrument-Grade Cute Robot Hero Mascots on Floating Plinths.
"""

import os
import math
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
BRAIN = '/Users/VJ/.gemini/antigravity-cli/brain/6702a6d9-86c1-48b2-93d2-f2c949a6e416'

FONT_PATH = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
if not os.path.exists(FONT_PATH):
    FONT_PATH = '/System/Library/Fonts/Helvetica.ttc'

def draw_pill(draw, text, cx, cy, border_color='#0F172A', text_color='#0F172A'):
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

def draw_leader(draw, p_box, p_target, color='#0F172A'):
    draw.line([p_box, p_target], fill=color, width=2)
    r = 3.5
    draw.ellipse([p_target[0] - r, p_target[1] - r, p_target[0] + r, p_target[1] + r], fill=color)

def process_chapter(slug, base_filename, labels):
    base_path = os.path.join(BRAIN, base_filename)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Missing base image: {base_path}")
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
# CHAPTER CONFIGURATIONS & LABEL DEFINITIONS (ALL 18 CHAPTERS)
# Palette semantics:
#   Cyan    (#0090B0): Model core, stochastic processor, active trajectories
#   Mint    (#2EC4B6): Memory, verified states, cache, return loop
#   Amber   (#D97706): Buses, tools, exploration, coordination
#   Crimson (#9B2226): Barriers, faults, gates, eviction, masks
# =============================================================================

chapters = [
    # 01 Introduction
    {
        'slug': '01_introduction',
        'image': 'vol3_ch01_opt_a_cute_anatomy_1789320258297.jpg',
        'labels': [
            {'text': 'stochastic processor core', 'pill': (240, 240), 'targ': (620, 330), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'l1 sram ring buffer', 'pill': (240, 440), 'targ': (590, 430), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'isolated actuation bus', 'pill': (240, 620), 'targ': (480, 520), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'glowing sensory visor', 'pill': (680, 80), 'targ': (680, 220), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'hbm3e memory stacks', 'pill': (1160, 220), 'targ': (760, 360), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'l2 paged block cache', 'pill': (1180, 420), 'targ': (800, 450), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'pcie tool expansion bus', 'pill': (1160, 620), 'targ': (880, 550), 'border': '#D97706', 'color': '#D97706'},
        ]
    },
    # 02 Processor
    {
        'slug': '02_processor',
        'image': 'vol3_ch02_opt_b_speculative_duo_1789320293420.jpg',
        'labels': [
            {'text': 'draft model execution core', 'pill': (240, 280), 'targ': (470, 410), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'speculative token proposal stream', 'pill': (240, 520), 'targ': (540, 440), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'memory-bandwidth streaming', 'pill': (660, 80), 'targ': (680, 220), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'target verification engine', 'pill': (1160, 180), 'targ': (720, 300), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'crimson rollback diversion gate', 'pill': (1180, 400), 'targ': (790, 370), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'parallel token acceptance pool', 'pill': (1160, 620), 'targ': (760, 480), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    # 03 Deliberation
    {
        'slug': '03_deliberation',
        'image': 'vol3_ch03_opt_b_stylus_rollout_1789320338507.jpg',
        'labels': [
            {'text': 'root decision state', 'pill': (240, 560), 'targ': (520, 460), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'laser rollout drafting stylus', 'pill': (240, 340), 'targ': (500, 370), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'deliberation tree architect', 'pill': (660, 80), 'targ': (580, 240), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'optimal high-reward trajectory', 'pill': (1180, 220), 'targ': (740, 330), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'pruned dead-end branch', 'pill': (1180, 440), 'targ': (840, 430), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'speculative tree expansion', 'pill': (1160, 640), 'targ': (880, 510), 'border': '#D97706', 'color': '#D97706'},
        ]
    },
    # 04 Working Sets
    {
        'slug': '04_working_sets',
        'image': 'vol3_ch04_opt_a_sink_carousel_1789320414341.jpg',
        'labels': [
            {'text': 'permanent attention sink tokens', 'pill': (260, 180), 'targ': (640, 290), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'rotating context carousel', 'pill': (240, 420), 'targ': (530, 440), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'working set budget boundary', 'pill': (240, 620), 'targ': (480, 540), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'quadratic attention envelope', 'pill': (1160, 200), 'targ': (800, 300), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'active sliding window tokens', 'pill': (1180, 420), 'targ': (770, 390), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'eviction discharge chute', 'pill': (1160, 640), 'targ': (870, 470), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    # 05 Virtual Memory
    {
        'slug': '05_virtual_memory',
        'image': 'vol3_ch05_opt_a_page_librarian_1789320442303.jpg',
        'labels': [
            {'text': 'logical virtual blocks', 'pill': (240, 200), 'targ': (510, 270), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'shared prefix block pool', 'pill': (240, 420), 'targ': (560, 360), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'radix block table mmu', 'pill': (240, 640), 'targ': (640, 520), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'pagedattention block librarian', 'pill': (700, 80), 'targ': (680, 310), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'physical hbm page frames', 'pill': (1180, 240), 'targ': (860, 330), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'zero-fragmentation page allocation', 'pill': (1180, 520), 'targ': (820, 480), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    # 06 Episodic Memory
    {
        'slug': '06_episodic_memory',
        'image': 'vol3_ch06_opt_a_vector_lattice_1789320488153.jpg',
        'labels': [
            {'text': 'nearest-neighbor query prism', 'pill': (240, 300), 'targ': (590, 370), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'approximate search agent', 'pill': (240, 540), 'targ': (520, 430), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'semantic similarity vector beam', 'pill': (660, 80), 'targ': (660, 340), 'border': '#D97706', 'color': '#D97706'},
            {'text': '3d hnsw vector embedding lattice', 'pill': (1180, 220), 'targ': (840, 320), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'clustered memory hierarchy', 'pill': (1180, 440), 'targ': (820, 200), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'persistent episodic plinth', 'pill': (1180, 640), 'targ': (760, 560), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    # 07 Checkpointing
    {
        'slug': '07_checkpointing',
        'image': 'vol3_ch07_opt_a_rewind_track_1789320516299.jpg',
        'labels': [
            {'text': 'forward transaction stream', 'pill': (240, 420), 'targ': (580, 440), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'compensating rollback siding', 'pill': (240, 620), 'targ': (480, 500), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'checkpoint rewind portal', 'pill': (700, 80), 'targ': (720, 300), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'fault detection laser tripwire', 'pill': (1180, 220), 'targ': (620, 380), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'deterministic state rewind track', 'pill': (1180, 440), 'targ': (710, 410), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'immutable snapshot state', 'pill': (1180, 640), 'targ': (810, 480), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    # 08 Actuation
    {
        'slug': '08_actuation',
        'image': 'vol3_ch08_opt_a_mcp_tool_switchboard_1789320637600.jpg',
        'labels': [
            {'text': 'agent actuation client', 'pill': (240, 320), 'targ': (490, 390), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'json-schema validation port', 'pill': (240, 560), 'targ': (590, 410), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'central mcp switchboard junction', 'pill': (660, 80), 'targ': (680, 360), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'database query console bay', 'pill': (1180, 200), 'targ': (760, 310), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'sandboxed terminal execution rack', 'pill': (1180, 420), 'targ': (820, 390), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'external tool environment bus', 'pill': (1180, 620), 'targ': (870, 490), 'border': '#D97706', 'color': '#D97706'},
        ]
    },
    # 09 Virtualization
    {
        'slug': '09_virtualization',
        'image': 'vol3_ch09_opt_c_cow_sandbox_1789320728670.jpg',
        'labels': [
            {'text': 'isolated sandboxed rover', 'pill': (240, 320), 'targ': (710, 340), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'microvm containment barrier', 'pill': (240, 540), 'targ': (590, 420), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'immutable base image layer', 'pill': (660, 80), 'targ': (660, 480), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'copy-on-write scratch layer', 'pill': (1180, 220), 'targ': (740, 370), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'seccomp-bpf syscall drain', 'pill': (1180, 460), 'targ': (860, 460), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'air-gapped virtual boundary', 'pill': (1180, 640), 'targ': (820, 540), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    # 10 Interrupts
    {
        'slug': '10_interrupts',
        'image': 'vol3_ch10_opt_b_interrupt_pause_1789320747092.jpg',
        'labels': [
            {'text': 'asynchronous pause barrier', 'pill': (240, 320), 'targ': (530, 480), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'optical interrupt tripwire', 'pill': (240, 540), 'targ': (580, 490), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'state escrow capsule', 'pill': (700, 80), 'targ': (640, 260), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'preempted agent worker', 'pill': (1180, 240), 'targ': (640, 400), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'frozen in-flight token bubbles', 'pill': (1180, 440), 'targ': (730, 360), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'human authorization checkpoint', 'pill': (1180, 640), 'targ': (770, 510), 'border': '#D97706', 'color': '#D97706'},
        ]
    },
    # 11 Scheduling
    {
        'slug': '11_scheduling',
        'image': 'vol3_ch11_opt_b_preemption_railway_1789320776809.jpg',
        'labels': [
            {'text': 'latency-critical prefill train', 'pill': (240, 300), 'targ': (520, 420), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'anti-starvation bypass switch', 'pill': (240, 540), 'targ': (620, 460), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'cluster switchyard operator', 'pill': (660, 80), 'targ': (640, 350), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'throughput-bound decode train', 'pill': (1180, 220), 'targ': (820, 410), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'preemption railway switch', 'pill': (1180, 440), 'targ': (720, 430), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'cxl memory-pool ballast', 'pill': (1180, 640), 'targ': (760, 530), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    # 12 Data Flywheel
    {
        'slug': '12_data_flywheel',
        'image': 'vol3_ch12_opt_a_synthetic_garden_1789320824888.jpg',
        'labels': [
            {'text': 'synthetic trajectory gardener', 'pill': (240, 300), 'targ': (740, 340), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'verified policy sprout', 'pill': (240, 520), 'targ': (570, 430), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'simulation terrarium cloche', 'pill': (660, 80), 'targ': (710, 260), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'decontamination filter base', 'pill': (1180, 240), 'targ': (820, 410), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'automated curriculum nursery', 'pill': (1180, 460), 'targ': (630, 470), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'data flywheel distillation plinth', 'pill': (1180, 660), 'targ': (740, 560), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    # 13 SFT (UPGRADED Wave 1 Standard)
    {
        'slug': '13_sft',
        'image': 'vol3_ch13_upg_a_optical_mask_bench_1789324786290.jpg',
        'labels': [
            {'text': 'observation context mask (zero loss)', 'pill': (240, 320), 'targ': (460, 450), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'trajectory token filmstrip', 'pill': (240, 540), 'targ': (480, 510), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'teacher-forcing inspector', 'pill': (660, 80), 'targ': (610, 310), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'policy action token illumination', 'pill': (1180, 220), 'targ': (670, 350), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'gradient backprop focal optics', 'pill': (1180, 440), 'targ': (640, 390), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'loss masking optical bench', 'pill': (1180, 640), 'targ': (730, 480), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    # 14 RLVR (UPGRADED Wave 1 Standard)
    {
        'slug': '14_rlvr',
        'image': 'vol3_ch14_upg_a_laser_test_chamber_1789324813406.jpg',
        'labels': [
            {'text': 'candidate solution crystal', 'pill': (240, 320), 'targ': (510, 440), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'formal verification test agent', 'pill': (240, 540), 'targ': (450, 430), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'dual-laser diffraction verifier', 'pill': (660, 80), 'targ': (640, 360), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'environment outcome test chamber', 'pill': (1180, 200), 'targ': (740, 330), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'fault containment bypass lock', 'pill': (1180, 420), 'targ': (820, 420), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'scalar binary reward conduit', 'pill': (1180, 620), 'targ': (680, 530), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    # 15 Multi-Agent (UPGRADED Wave 1 Standard)
    {
        'slug': '15_multi_agent',
        'image': 'vol3_ch15_upg_a_ring_bus_deck_1789324897563.jpg',
        'labels': [
            {'text': 'circulating amber data packets', 'pill': (240, 340), 'targ': (540, 430), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'specialized worker consoles', 'pill': (240, 560), 'targ': (440, 460), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'elevated coordinator bot', 'pill': (660, 80), 'targ': (820, 230), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'central consensus core', 'pill': (1180, 220), 'targ': (680, 370), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'circular optical ring bus', 'pill': (1180, 440), 'targ': (750, 450), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'state synchronization broadcast', 'pill': (1180, 640), 'targ': (720, 530), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    # 16 Observability
    {
        'slug': '16_observability',
        'image': 'vol3_ch16_opt_a_3d_flight_recorder_1789322565895.jpg',
        'labels': [
            {'text': 'distributed tracer bot', 'pill': (240, 320), 'targ': (520, 480), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'telemetry flight recorder box', 'pill': (240, 540), 'targ': (640, 470), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'execution anomaly spike', 'pill': (660, 80), 'targ': (740, 300), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': '3d trajectory ribbon span', 'pill': (1180, 220), 'targ': (660, 320), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'checkpoint milestone beacon', 'pill': (1180, 440), 'targ': (600, 350), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'latency flame-graph tier', 'pill': (1180, 640), 'targ': (770, 420), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    # 17 Tokenomics (UPGRADED Wave 1 Standard)
    {
        'slug': '17_tokenomics',
        'image': 'vol3_ch17_upg_a_pareto_spectrometer_1789324934337.jpg',
        'labels': [
            {'text': 'collimated prompt input ray', 'pill': (240, 300), 'targ': (500, 410), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'dynamic routing prism operator', 'pill': (240, 520), 'targ': (590, 330), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'quartz beam-splitter spectrometer', 'pill': (660, 80), 'targ': (580, 440), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'heavy-compute frontier waveguide', 'pill': (1180, 220), 'targ': (770, 420), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'slender low-cost edge fiber', 'pill': (1180, 440), 'targ': (660, 520), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'pareto capacity arbitrage stage', 'pill': (1180, 640), 'targ': (660, 460), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    # 18 Conclusion (UPGRADED Wave 1 Standard)
    {
        'slug': '18_conclusion',
        'image': 'vol3_ch18_upg_a_monumental_torus_1789324970868.jpg',
        'labels': [
            {'text': 'sovereign scout agent', 'pill': (240, 340), 'targ': (600, 480), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'floating systems plinth', 'pill': (240, 580), 'targ': (500, 550), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'concentric optical waveguides', 'pill': (660, 80), 'targ': (730, 240), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'monumental torus portal', 'pill': (1180, 220), 'targ': (780, 340), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'autonomous frontier metropolis', 'pill': (1180, 440), 'targ': (840, 410), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'invariant closure gateway', 'pill': (1180, 640), 'targ': (800, 490), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
]

if __name__ == '__main__':
    print('Starting Volume 3 Chapter Opener Generation (All 18 Chapters)...')
    for ch in chapters:
        process_chapter(ch['slug'], ch['image'], ch['labels'])
    print('Finished generating all 18 Volume 3 chapter openers!')
