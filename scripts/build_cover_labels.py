#!/usr/bin/env python3
"""
Generate publication-quality labeled isometric blueprint chapter openers
for MLSysBook Volume 3 (18 chapters) and Volume 4 (17 chapters).

Key Quality Standards:
1. Pure vector supersampling: Renders labels, leader lines, and targets on a 2x
   scaled overlay (2800x1600) and downsamples with Lanczos filtering to 1400x800,
   guaranteeing perfectly antialiased lines without staircase raster jaggies.
2. Zero semantic color leaks: Component labels never contain literal color names
   (e.g., "speculative rollback diversion gate", not "crimson rollback diversion gate").
3. Zero line crossings: Geometric routing strictly avoids leader line intersections.
4. Physical element anchoring: Targets terminate directly on physical hardware/modules
   rather than bare plinth floor space.
"""

import os
import re
import math
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FONT_ARIAL_BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
if not os.path.exists(FONT_ARIAL_BOLD):
    FONT_ARIAL_BOLD = '/System/Library/Fonts/Helvetica.ttc'

FONT_HELVETICA = '/System/Library/Fonts/Helvetica.ttc'
if not os.path.exists(FONT_HELVETICA):
    FONT_HELVETICA = '/System/Library/Fonts/Supplemental/Arial.ttf'

COLOR_WORDS = {
    'red', 'crimson', 'blue', 'cyan', 'green', 'amber',
    'yellow', 'orange', 'purple', 'white', 'black', 'pink', 'gray', 'grey'
}

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# =============================================================================
# CHAPTER CONFIGURATIONS: VOLUME 3 (AGENTIC SYSTEMS)
# =============================================================================

VOL3_CHAPTERS = [
    {
        'slug': '01_introduction',
        'prefix': 'introduction',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'stochastic processor socket', 'pill': (680, 80), 'targ': (695, 200), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'isolated actuation interface', 'pill': (240, 220), 'targ': (490, 360), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'l2 paged block cache', 'pill': (240, 440), 'targ': (680, 440), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'l1 sram buffer ring', 'pill': (240, 620), 'targ': (560, 550), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'hbm3e memory stacks', 'pill': (1160, 320), 'targ': (820, 380), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'pcie tool expansion bus', 'pill': (1160, 620), 'targ': (880, 550), 'border': '#D97706', 'color': '#D97706'},
        ]
    },
    {
        'slug': '02_processor',
        'prefix': 'processor',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'draft model execution core', 'pill': (240, 280), 'targ': (560, 420), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'speculative token proposal stream', 'pill': (240, 560), 'targ': (700, 530), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'memory-bandwidth streaming', 'pill': (660, 80), 'targ': (680, 220), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'target verification engine', 'pill': (1160, 180), 'targ': (720, 300), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'speculative rollback diversion gate', 'pill': (1180, 400), 'targ': (890, 400), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'parallel token acceptance pool', 'pill': (1160, 600), 'targ': (825, 440), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '03_deliberation',
        'prefix': 'deliberation',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'deliberation tree architect', 'pill': (660, 80), 'targ': (580, 240), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'laser rollout drafting stylus', 'pill': (240, 320), 'targ': (575, 350), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'root decision state', 'pill': (240, 540), 'targ': (595, 425), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'optimal high-reward trajectory', 'pill': (1180, 220), 'targ': (740, 330), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'pruned dead-end branch', 'pill': (1180, 440), 'targ': (840, 430), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'speculative tree expansion', 'pill': (1160, 640), 'targ': (780, 470), 'border': '#D97706', 'color': '#D97706'},
        ]
    },
    {
        'slug': '04_working_sets',
        'prefix': 'working_sets',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'permanent attention sink tokens', 'pill': (260, 200), 'targ': (480, 405), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'rotating context carousel', 'pill': (240, 420), 'targ': (530, 440), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'working set budget boundary', 'pill': (240, 620), 'targ': (480, 540), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'quadratic attention envelope', 'pill': (1160, 200), 'targ': (800, 300), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'active sliding window tokens', 'pill': (1180, 420), 'targ': (880, 320), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'eviction discharge chute', 'pill': (1160, 640), 'targ': (870, 470), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '05_virtual_memory',
        'prefix': 'virtual_memory',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'logical virtual blocks', 'pill': (240, 200), 'targ': (360, 220), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'shared prefix block pool', 'pill': (240, 400), 'targ': (490, 280), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'radix block table mmu', 'pill': (240, 600), 'targ': (581, 310), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'pagedattention block librarian', 'pill': (680, 80), 'targ': (680, 240), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'physical hbm page frames', 'pill': (1180, 200), 'targ': (1040, 280), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'zero-fragmentation page allocation', 'pill': (1180, 420), 'targ': (1014, 350), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '06_episodic_memory',
        'prefix': 'episodic_memory',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'nearest-neighbor query prism', 'pill': (240, 220), 'targ': (505, 420), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'semantic similarity query beam', 'pill': (240, 360), 'targ': (470, 480), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'approximate search agent', 'pill': (240, 540), 'targ': (380, 520), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'clustered memory hierarchy', 'pill': (1180, 200), 'targ': (750, 220), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': '3d hnsw vector embedding lattice', 'pill': (1180, 440), 'targ': (780, 480), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'persistent episodic plinth', 'pill': (1160, 700), 'targ': (680, 720), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '07_checkpointing',
        'prefix': 'checkpointing',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'forward transaction stream', 'pill': (240, 420), 'targ': (580, 440), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'compensating rollback siding', 'pill': (240, 620), 'targ': (480, 500), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'checkpoint rewind portal', 'pill': (700, 80), 'targ': (720, 300), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'immutable snapshot state', 'pill': (1180, 220), 'targ': (860, 240), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'deterministic state rewind track', 'pill': (1180, 420), 'targ': (920, 360), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'fault detection laser tripwire', 'pill': (1180, 620), 'targ': (735, 480), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '08_actuation',
        'prefix': 'actuation',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'json-schema validation port', 'pill': (500, 80), 'targ': (680, 280), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'central mcp switchboard junction', 'pill': (850, 80), 'targ': (780, 180), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'agent actuation client', 'pill': (240, 320), 'targ': (480, 310), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'database query console bay', 'pill': (1180, 200), 'targ': (880, 260), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'sandboxed terminal execution rack', 'pill': (1180, 420), 'targ': (998, 359), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'external tool environment bus', 'pill': (1180, 620), 'targ': (850, 450), 'border': '#D97706', 'color': '#D97706'},
        ]
    },
    {
        'slug': '09_virtualization',
        'prefix': 'virtualization',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'isolated sandboxed rover', 'pill': (720, 80), 'targ': (720, 200), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'copy-on-write scratch layer', 'pill': (240, 320), 'targ': (620, 320), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'immutable base image layer', 'pill': (240, 560), 'targ': (660, 520), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'microvm containment barrier', 'pill': (1180, 220), 'targ': (880, 310), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'seccomp-bpf syscall drain', 'pill': (1180, 460), 'targ': (850, 482), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'air-gapped virtual boundary', 'pill': (1180, 640), 'targ': (820, 560), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '10_interrupts',
        'prefix': 'interrupts',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'state escrow capsule', 'pill': (700, 80), 'targ': (705, 140), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'asynchronous pause barrier', 'pill': (240, 320), 'targ': (530, 480), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'continuous execution pipeline', 'pill': (240, 540), 'targ': (440, 640), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'preempted agent worker', 'pill': (1180, 220), 'targ': (740, 290), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'frozen in-flight token bubbles', 'pill': (1180, 420), 'targ': (893, 310), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'human authorization checkpoint', 'pill': (1180, 620), 'targ': (710, 520), 'border': '#D97706', 'color': '#D97706'},
        ]
    },
    {
        'slug': '11_scheduling',
        'prefix': 'scheduling',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'latency-critical prefill train', 'pill': (240, 240), 'targ': (520, 420), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'anti-starvation bypass switch', 'pill': (240, 480), 'targ': (620, 460), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'cluster switchyard operator', 'pill': (660, 80), 'targ': (640, 350), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'throughput-bound decode train', 'pill': (1180, 220), 'targ': (820, 410), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'switchyard preemption lever', 'pill': (1180, 440), 'targ': (775, 415), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'cxl memory-pool ballast', 'pill': (1180, 640), 'targ': (650, 550), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '12_data_flywheel',
        'prefix': 'data_flywheel',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'synthetic trajectory gardener', 'pill': (720, 80), 'targ': (720, 260), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'simulation terrarium cloche', 'pill': (240, 240), 'targ': (420, 260), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'verified policy sprout', 'pill': (240, 480), 'targ': (413, 358), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'automated curriculum nursery', 'pill': (1180, 240), 'targ': (990, 420), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'decontamination filter base', 'pill': (1180, 480), 'targ': (960, 485), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'data flywheel distillation plinth', 'pill': (1180, 680), 'targ': (740, 690), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '13_sft',
        'prefix': 'sft',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'observation context mask (zero loss)', 'pill': (240, 320), 'targ': (480, 460), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'trajectory token filmstrip', 'pill': (240, 540), 'targ': (510, 485), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'teacher-forcing inspector', 'pill': (660, 80), 'targ': (610, 310), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'policy action token illumination', 'pill': (1180, 220), 'targ': (670, 350), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'gradient backprop focal optics', 'pill': (1180, 440), 'targ': (640, 390), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'loss masking optical bench', 'pill': (1180, 640), 'targ': (730, 480), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '14_rlvr',
        'prefix': 'rlvr',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'formal verification test agent', 'pill': (240, 260), 'targ': (460, 260), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'candidate solution crystal', 'pill': (240, 500), 'targ': (560, 420), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'dual-laser diffraction verifier', 'pill': (660, 80), 'targ': (640, 360), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'environment outcome test chamber', 'pill': (1180, 200), 'targ': (740, 330), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'fault containment bypass lock', 'pill': (1180, 420), 'targ': (980, 480), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'scalar binary reward conduit', 'pill': (1180, 620), 'targ': (680, 530), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '15_multi_agent',
        'prefix': 'multi_agent',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'elevated coordinator bot', 'pill': (660, 80), 'targ': (820, 230), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'circulating token packets', 'pill': (240, 340), 'targ': (495, 436), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'specialized worker console', 'pill': (240, 560), 'targ': (410, 490), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'central consensus core', 'pill': (1180, 220), 'targ': (680, 370), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'circular optical ring bus', 'pill': (1180, 440), 'targ': (840, 470), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'state synchronization ring', 'pill': (1180, 640), 'targ': (700, 620), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '16_observability',
        'prefix': 'observability',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': '3d trajectory trace ribbon', 'pill': (240, 240), 'targ': (480, 220), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'distributed tracer bot', 'pill': (240, 480), 'targ': (578, 434), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'execution anomaly spike', 'pill': (830, 70), 'targ': (830, 175), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'checkpoint milestone beacon', 'pill': (1180, 240), 'targ': (695, 282), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'latency flame-graph span', 'pill': (1180, 440), 'targ': (880, 360), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'telemetry flight recorder box', 'pill': (1180, 640), 'targ': (800, 480), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '17_tokenomics',
        'prefix': 'tokenomics',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'collimated prompt input ray', 'pill': (240, 240), 'targ': (460, 290), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'dynamic routing prism operator', 'pill': (240, 460), 'targ': (580, 350), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'pareto capacity arbitrage stage', 'pill': (240, 660), 'targ': (640, 430), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'quartz beam-splitter spectrometer', 'pill': (660, 70), 'targ': (660, 260), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'heavy-compute frontier waveguide', 'pill': (1180, 240), 'targ': (892, 352), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'slender low-cost edge fiber', 'pill': (1180, 480), 'targ': (863, 479), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '18_conclusion',
        'prefix': 'conclusion',
        'font': FONT_ARIAL_BOLD,
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

# =============================================================================
# CHAPTER CONFIGURATIONS: VOLUME 4 (PHYSICAL AI & ROBOTICS)
# =============================================================================

VOL4_CHAPTERS = [
    {
        'slug': '01_boundary',
        'prefix': 'boundary',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'brain', 'pill': (240, 150), 'targ': (240, 320), 'border': '#1A4D3E', 'color': '#1A4D3E'},
            {'text': 'causal boundary', 'pill': (890, 220), 'targ': (890, 560), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'nervous system', 'pill': (695, 360), 'targ': (695, 520), 'border': '#1A4D3E', 'color': '#1A4D3E'},
            {'text': 'body', 'pill': (1150, 170), 'targ': (1150, 320), 'border': '#1A4D3E', 'color': '#1A4D3E'},
            {'text': 'sensor feedback return', 'pill': (480, 700), 'targ': (480, 560), 'border': '#1A4D3E', 'color': '#1A4D3E'},
        ]
    },
    {
        'slug': '02_body',
        'prefix': 'body',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'stator windings', 'pill': (300, 520), 'targ': (620, 455), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'planetary gearset', 'pill': (240, 390), 'targ': (590, 385), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'harmonic drive transmission', 'pill': (260, 260), 'targ': (560, 275), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'optical encoder disc', 'pill': (440, 100), 'targ': (675, 155), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'kinematic arm linkage', 'pill': (880, 80), 'targ': (820, 210), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'stopping envelope', 'pill': (1180, 150), 'targ': (1110, 210), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'contact friction cones', 'pill': (1220, 310), 'targ': (1100, 310), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'parallel jaw gripper', 'pill': (1200, 460), 'targ': (1030, 310), 'border': '#1C4E4F', 'color': '#1C4E4F'},
        ]
    },
    {
        'slug': '03_brain',
        'prefix': 'brain',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'multimodal perception rays', 'pill': (240, 140), 'targ': (420, 190), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'cognitive token lattice', 'pill': (260, 360), 'targ': (640, 270), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'beveled glass housing', 'pill': (680, 70), 'targ': (760, 210), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'proposal aperture', 'pill': (1120, 360), 'targ': (800, 360), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'prospective action chunk', 'pill': (1160, 600), 'targ': (990, 510), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'thermal dissipation fins', 'pill': (440, 700), 'targ': (760, 520), 'border': '#1C4E4F', 'color': '#1C4E4F'},
        ]
    },
    {
        'slug': '04_nervous',
        'prefix': 'nervous',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'deterministic 1 khz clock', 'pill': (440, 80), 'targ': (710, 220), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'jitter-bound envelope', 'pill': (960, 90), 'targ': (830, 160), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'seqlock proposal mailbox', 'pill': (200, 480), 'targ': (420, 400), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'cbf reflex filter', 'pill': (420, 660), 'targ': (690, 372), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'shielded fieldbus ring', 'pill': (1180, 210), 'targ': (1050, 260), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'machined heatsink fins', 'pill': (1180, 440), 'targ': (850, 400), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'sensor feedback return', 'pill': (960, 720), 'targ': (750, 560), 'border': '#1C4E4F', 'color': '#1C4E4F'},
        ]
    },
    {
        'slug': '05_data',
        'prefix': 'data',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'bilateral haptic master', 'pill': (240, 260), 'targ': (440, 360), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'overhead sensor mast', 'pill': (560, 80), 'targ': (690, 130), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'stereo vision cone', 'pill': (880, 160), 'targ': (760, 240), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': '6-dof follower arm', 'pill': (1160, 260), 'targ': (960, 340), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'manipulation workpiece', 'pill': (1180, 520), 'targ': (816, 457), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'action stream conduit', 'pill': (240, 580), 'targ': (580, 500), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'timestamp sync logger', 'pill': (700, 740), 'targ': (710, 600), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '06_training',
        'prefix': 'training',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'virtual physics digital twin', 'pill': (240, 160), 'targ': (420, 310), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'randomized friction cones', 'pill': (240, 480), 'targ': (394, 490), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'domain randomization prism', 'pill': (700, 80), 'targ': (700, 320), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'physical quadruped plant', 'pill': (1160, 200), 'targ': (1000, 360), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'dynamometer treadmill', 'pill': (1180, 480), 'targ': (940, 600), 'border': '#1A4D3E', 'color': '#1A4D3E'},
            {'text': 'real-time sensor telemetry', 'pill': (760, 740), 'targ': (770, 560), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '07_evaluation',
        'prefix': 'evaluation',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'optical mocap tower', 'pill': (200, 180), 'targ': (348, 275), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'retroreflective constellation', 'pill': (480, 80), 'targ': (618, 290), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'dynamometer baseplate', 'pill': (260, 640), 'targ': (570, 440), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'safety perimeter rail', 'pill': (190, 480), 'targ': (380, 480), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'confidence ledger cylinder', 'pill': (1160, 680), 'targ': (885, 545), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'tracking ray cone', 'pill': (1160, 240), 'targ': (850, 310), 'border': '#1C4E4F', 'color': '#1C4E4F'},
        ]
    },
    {
        'slug': '08_perception',
        'prefix': 'perception',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'stereo optical baseline', 'pill': (360, 70), 'targ': (620, 140), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'micro-lidar point cloud', 'pill': (880, 80), 'targ': (720, 230), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'metrology step artifact', 'pill': (200, 260), 'targ': (550, 360), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'extrinsic calibration frame', 'pill': (220, 500), 'targ': (635, 415), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'latency drift vector', 'pill': (480, 720), 'targ': (712, 545), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'spatial covariance ellipsoid', 'pill': (1160, 600), 'targ': (840, 510), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'edge preprocessor box', 'pill': (1160, 340), 'targ': (980, 400), 'border': '#1C4E4F', 'color': '#1C4E4F'},
        ]
    },
    {
        'slug': '09_memory',
        'prefix': 'memory',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'octree voxel grid', 'pill': (180, 320), 'targ': (341, 438), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'occlusion barrier', 'pill': (560, 70), 'targ': (568, 224), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'spatial belief decay', 'pill': (820, 90), 'targ': (752, 240), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'temporal lease ledger', 'pill': (1220, 160), 'targ': (1079, 278), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'stale token eviction', 'pill': (1160, 710), 'targ': (958, 567), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '10_intent',
        'prefix': 'intent',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'task token projection', 'pill': (740, 60), 'targ': (735, 120), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'kinematic reachability dome', 'pill': (200, 160), 'targ': (380, 180), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'spatial tolerance cylinder', 'pill': (200, 440), 'targ': (690, 380), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'aerospace bracket workpiece', 'pill': (320, 680), 'targ': (640, 500), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'countdown lease ring dial', 'pill': (1180, 160), 'targ': (765, 265), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'insertion affordance cone', 'pill': (1180, 360), 'targ': (886, 280), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'optical expiration tripwire beam', 'pill': (1180, 560), 'targ': (940, 500), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '11_planning',
        'prefix': 'planning',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'bldc outrunner cutaway', 'pill': (220, 180), 'targ': (380, 285), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'executed 3d polynomial spline', 'pill': (240, 440), 'targ': (540, 350), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'aerodynamic thrust cones', 'pill': (400, 720), 'targ': (560, 520), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'spatial obstacle ring', 'pill': (740, 80), 'targ': (730, 170), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'prospective waypoint chunk', 'pill': (1180, 180), 'targ': (880, 205), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'hover-recovery stopping suffix', 'pill': (1180, 360), 'targ': (1010, 260), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'terminal landing perch', 'pill': (1180, 540), 'targ': (1040, 480), 'border': '#1C4E4F', 'color': '#1C4E4F'},
        ]
    },
    {
        'slug': '12_enforcement',
        'prefix': 'enforcement',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'forward-invariant safe set', 'pill': (240, 200), 'targ': (450, 400), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'omnidirectional mecanum base', 'pill': (240, 420), 'targ': (520, 450), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'emergency brake interlock', 'pill': (280, 640), 'targ': (635, 470), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'prohibited hazard boundary', 'pill': (1160, 180), 'targ': (980, 200), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'candidate proposal vector', 'pill': (1160, 340), 'targ': (910, 250), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'cbf-qp projected vector', 'pill': (1160, 500), 'targ': (910, 325), 'border': '#D97706', 'color': '#D97706'},
        ]
    },
    {
        'slug': '13_placement',
        'prefix': 'placement',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': '3d memory stacks', 'pill': (160, 140), 'targ': (270, 240), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'cognitive npu cluster', 'pill': (180, 340), 'targ': (520, 350), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'thermal dissipation fins', 'pill': (360, 60), 'targ': (468, 144), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'deterministic mcu enclave', 'pill': (1160, 260), 'targ': (925, 356), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'shared crossbar bus', 'pill': (380, 720), 'targ': (538, 464), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'qos isolation trench', 'pill': (720, 750), 'targ': (588, 564), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '14_intervention',
        'prefix': 'intervention',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'vehicle perception mast', 'pill': (260, 140), 'targ': (595, 180), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'independent double-wishbone', 'pill': (220, 340), 'targ': (470, 390), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'steer-by-wire rack & pinion', 'pill': (240, 520), 'targ': (540, 410), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'tire contact friction ellipses', 'pill': (360, 720), 'targ': (690, 540), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'electric traction motor', 'pill': (1160, 160), 'targ': (825, 240), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'timestamped intervention relay', 'pill': (1180, 300), 'targ': (900, 280), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'human steering override', 'pill': (1180, 440), 'targ': (730, 275), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'kinetic authority clutch', 'pill': (1180, 580), 'targ': (685, 310), 'border': '#D97706', 'color': '#D97706'},
        ]
    },
    {
        'slug': '15_verification',
        'prefix': 'verification',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'virtual simulation crucible', 'pill': (300, 680), 'targ': (455, 559), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'processor-in-the-loop', 'pill': (420, 240), 'targ': (588, 474), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'hardware-in-the-loop dyno', 'pill': (760, 680), 'targ': (748, 331), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'fault injection needles', 'pill': (1160, 120), 'targ': (926, 187), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'surveillance telemetry', 'pill': (880, 60), 'targ': (768, 144), 'border': '#1C4E4F', 'color': '#1C4E4F'},
        ]
    },
    {
        'slug': '16_release',
        'prefix': 'release',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'evidence telemetry pipelines', 'pill': (280, 680), 'targ': (473, 549), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'gsn argument hierarchy', 'pill': (480, 220), 'targ': (691, 379), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'deployment authorization seal', 'pill': (722, 60), 'targ': (710, 175), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'certified envelope threshold', 'pill': (1160, 320), 'targ': (875, 410), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'revocation interlock', 'pill': (1120, 680), 'targ': (888, 524), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '17_frontier',
        'prefix': 'frontier',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'tripartite physical stack', 'pill': (200, 140), 'targ': (390, 253), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'actuator body housing', 'pill': (340, 680), 'targ': (528, 454), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'containment barrier', 'pill': (1180, 220), 'targ': (949, 307), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'exploratory horizon prism', 'pill': (880, 60), 'targ': (869, 179), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'unobserved reality chasm', 'pill': (1200, 680), 'targ': (1088, 524), 'border': '#1C4E4F', 'color': '#1C4E4F'},
        ]
    },
]

# =============================================================================
# VALIDATION & RENDERING ENGINE
# =============================================================================

def validate_chapter(vol_name, ch):
    """Ensure zero color words and zero leader-line intersections."""
    slug = ch['slug']
    labels = ch['labels']
    
    # 1. Check for semantic color word leaks
    for item in labels:
        words = set(re.findall(r'\b[a-zA-Z]+\b', item['text'].lower()))
        matched = words.intersection(COLOR_WORDS)
        if matched:
            raise ValueError(f"[{vol_name} {slug}] Label '{item['text']}' contains forbidden color word(s): {matched}")

    # 2. Check for leader-line geometric crossings
    segs = []
    for item in labels:
        px, py = item['pill']
        tx, ty = item['targ']
        tw = len(item['text']) * 4 + 12
        if abs(tx - px) < tw:
            anchor = (px, py - 10 if ty < py else py + 10)
        elif tx < px:
            anchor = (px - tw, py)
        else:
            anchor = (px + tw, py)
        segs.append((item['text'], anchor, (tx, ty)))
        
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            t1, a1, b1 = segs[i]
            t2, a2, b2 = segs[j]
            if segments_intersect(a1, b1, a2, b2):
                raise ValueError(f"[{vol_name} {slug}] Line intersection detected: '{t1}' crosses '{t2}'")

def render_chapter(vol_name, ch):
    slug = ch['slug']
    prefix = ch['prefix']
    font_path = ch['font']
    labels = ch['labels']
    
    base_png_path = os.path.join(REPO_ROOT, 'books', vol_name, slug, 'images', 'png', f'cover_{prefix}_blueprint.png')
    if not os.path.exists(base_png_path):
        raise FileNotFoundError(f"Missing base blueprint: {base_png_path}")
        
    im_base = Image.open(base_png_path).convert('RGB')
    w, h = im_base.size
    
    # 2x supersampled vector overlay
    w2, h2 = w * 2, h * 2
    overlay = Image.new('RGBA', (w2, h2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Font size: Vol 3 uses 30 (15pt @ 1x), Vol 4 uses 32 (16pt @ 1x)
    font_size = 30 if vol_name == 'vol3' else 32
    font = ImageFont.truetype(font_path, font_size)
    
    for item in labels:
        text = item['text']
        px2, py2 = item['pill'][0] * 2, item['pill'][1] * 2
        tx2, ty2 = item['targ'][0] * 2, item['targ'][1] * 2
        border = item.get('border', '#0F172A')
        tcolor = item.get('color', '#0F172A')
        
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad_x, pad_y = 24, 12
        bx0, by0 = px2 - tw // 2 - pad_x, py2 - th // 2 - pad_y
        bx1, by1 = px2 + tw // 2 + pad_x, py2 + th // 2 + pad_y
        
        # High-precision anchor selection
        if abs(tx2 - px2) < (tw // 2 + pad_x) and ty2 > by1:
            anchor = (px2, by1)
        elif abs(tx2 - px2) < (tw // 2 + pad_x) and ty2 < by0:
            anchor = (px2, by0)
        elif tx2 < px2:
            anchor = (bx0, py2)
        else:
            anchor = (bx1, py2)
            
        # Draw leader line (width 4 @ 2x -> 2 @ 1x)
        draw.line([anchor, (tx2, ty2)], fill=border, width=4)
        
        # Draw target dot (radius 7.0 @ 2x -> 3.5 @ 1x)
        r = 7.0
        draw.ellipse([tx2 - r, ty2 - r, tx2 + r, ty2 + r], fill=border)
        
        # Draw pill container
        draw.rounded_rectangle([bx0, by0, bx1, by1], radius=16, fill=(255, 255, 255, 255), outline=border, width=4)
        draw.text((px2 - tw // 2, py2 - th // 2 - 2), text, fill=tcolor, font=font)
        
    # Downsample overlay with Lanczos for subpixel anti-aliasing
    overlay_down = overlay.resize((w, h), Image.Resampling.LANCZOS)
    
    # Composite over base image
    final_im = im_base.copy()
    final_im.paste(overlay_down, (0, 0), overlay_down)
    
    # Destination directories
    png_dir = os.path.join(REPO_ROOT, 'books', vol_name, slug, 'images', 'png')
    webp_dir = os.path.join(REPO_ROOT, 'books', vol_name, slug, 'images', 'webp')
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(webp_dir, exist_ok=True)
    
    out_png = os.path.join(png_dir, f'cover_{prefix}_blueprint_labeled_print.png')
    out_webp = os.path.join(webp_dir, f'cover_{prefix}_blueprint_labeled.webp')
    
    final_im.save(out_png, 'PNG')
    final_im.save(out_webp, 'WEBP', quality=95)
    print(f"Generated [{vol_name}] {slug} -> {out_png}")

def main():
    print("=" * 70)
    print("Pre-validating all configurations for Volume 3 & Volume 4...")
    print("=" * 70)
    
    for ch in VOL3_CHAPTERS:
        validate_chapter('vol3', ch)
    print("✓ Volume 3 validation passed: 0 color words, 0 line crossings.")
    
    for ch in VOL4_CHAPTERS:
        validate_chapter('vol4', ch)
    print("✓ Volume 4 validation passed: 0 color words, 0 line crossings.")
    
    print("\n" + "=" * 70)
    print("Rendering Volume 3 blueprints with 2x supersampling...")
    print("=" * 70)
    for ch in VOL3_CHAPTERS:
        render_chapter('vol3', ch)
        
    print("\n" + "=" * 70)
    print("Rendering Volume 4 blueprints with 2x supersampling...")
    print("=" * 70)
    for ch in VOL4_CHAPTERS:
        render_chapter('vol4', ch)
        
    print("\n" + "=" * 70)
    print("All 35 chapter blueprints successfully generated with 0 defects.")
    print("=" * 70)

if __name__ == '__main__':
    main()
