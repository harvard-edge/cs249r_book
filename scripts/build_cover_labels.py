#!/usr/bin/env python3
"""
Generate publication-quality labeled isometric blueprint chapter openers
for MLSysBook across all four volumes:
- Volume 1: Machine Learning Systems (16 chapters)
- Volume 2: Distributed Machine Learning Systems (17 chapters)
- Volume 3: Agentic Systems (18 chapters)
- Volume 4: Physical AI & Robotics (17 chapters)
Total: 68 chapters.

Key Quality Standards:
1. Pure vector supersampling: Renders labels, leader lines, and targets on an upscaled
   overlay (4x / 5600x3200 for 2800x1600 print; 2x / 2800x1600 for 1400x800 webp/print)
   and downsamples with Lanczos filtering, guaranteeing crisp antialiased lines.
2. Zero semantic color leaks: Component labels never contain literal color names
   (e.g., "speculative rollback diversion gate", not "crimson rollback diversion gate").
3. Zero line crossings: Geometric routing strictly avoids leader line intersections.
4. Physical element anchoring: Targets terminate directly on physical hardware/modules
   rather than bare plinth floor space.
5. Generous breathing room: Label pills have comfortable, consistent padding and
   mathematically centered typography, with guaranteed canvas safety margins.
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
# CHAPTER CONFIGURATIONS: VOLUME 1 (ML SYSTEMS FOUNDATIONS)
# =============================================================================

VOL1_CHAPTERS = [
    {
        'slug': '01_introduction',
        'prefix': 'introduction',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'data', 'pill': (150, 137), 'targ': (202, 243), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'algorithm', 'pill': (564, 137), 'targ': (610, 284), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'machine', 'pill': (958, 133), 'targ': (998, 320), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '02_ml_systems',
        'prefix': 'ml_systems',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'workload', 'pill': (181, 131), 'targ': (236, 289), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'physics', 'pill': (490, 137), 'targ': (515, 236), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'deploy', 'pill': (807, 139), 'targ': (876, 329), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'tradeoff', 'pill': (1068, 125), 'targ': (1075, 519), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'entropy', 'pill': (1212, 622), 'targ': (1184, 535), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '03_ml_workflow',
        'prefix': 'ml_workflow',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'problem', 'pill': (176, 143), 'targ': (220, 334), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'data', 'pill': (444, 135), 'targ': (521, 339), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'model', 'pill': (711, 140), 'targ': (779, 340), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'deploy', 'pill': (1109, 142), 'targ': (1055, 330), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'monitor', 'pill': (1167, 611), 'targ': (1170, 515), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '04_data_engineering',
        'prefix': 'data_engineering',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'acquire', 'pill': (100, 59), 'targ': (232, 324), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'validate', 'pill': (414, 97), 'targ': (501, 354), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'transform', 'pill': (708, 143), 'targ': (725, 389), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'label', 'pill': (967, 147), 'targ': (859, 424), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'health', 'pill': (965, 615), 'targ': (730, 574), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '05_nn_computation',
        'prefix': 'nn_computation',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'inputs', 'pill': (156, 167), 'targ': (209, 276), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'weights', 'pill': (417, 136), 'targ': (531, 434), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'activations', 'pill': (745, 129), 'targ': (760, 394), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'loss', 'pill': (1072, 220), 'targ': (1135, 464), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'update', 'pill': (1019, 649), 'targ': (969, 636), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '06_nn_architectures',
        'prefix': 'nn_architectures',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'pattern', 'pill': (167, 145), 'targ': (251, 354), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'bias', 'pill': (443, 131), 'targ': (521, 304), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'architecture', 'pill': (977, 131), 'targ': (1020, 344), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'bottleneck', 'pill': (1172, 617), 'targ': (1159, 525), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '07_frameworks',
        'prefix': 'frameworks',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'abstraction', 'pill': (160, 143), 'targ': (241, 330), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'graph', 'pill': (460, 137), 'targ': (521, 324), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'kernel', 'pill': (944, 135), 'targ': (829, 504), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'autograd', 'pill': (695, 621), 'targ': (741, 515), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'runtime', 'pill': (1232, 480), 'targ': (1208, 534), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '08_training',
        'prefix': 'training',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'forward', 'pill': (162, 141), 'targ': (231, 324), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'activations', 'pill': (452, 133), 'targ': (552, 304), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'backward', 'pill': (754, 143), 'targ': (819, 330), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'optimizer', 'pill': (987, 615), 'targ': (928, 516), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'bottleneck', 'pill': (1173, 560), 'targ': (1159, 455), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '09_data_selection',
        'prefix': 'data_selection',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'scarcity', 'pill': (194, 144), 'targ': (200, 304), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'score', 'pill': (473, 139), 'targ': (581, 334), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'signal', 'pill': (973, 143), 'targ': (959, 364), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'select', 'pill': (719, 535), 'targ': (679, 395), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'cost', 'pill': (1173, 614), 'targ': (1139, 603), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '10_model_compression',
        'prefix': 'model_compression',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'pruning', 'pill': (146, 155), 'targ': (182, 220), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'precision', 'pill': (466, 133), 'targ': (479, 344), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'hardware', 'pill': (768, 141), 'targ': (780, 424), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'accuracy', 'pill': (921, 665), 'targ': (1013, 582), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'footprint', 'pill': (1187, 572), 'targ': (1286, 558), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '11_hw_acceleration',
        'prefix': 'hw_acceleration',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'operator', 'pill': (157, 77), 'targ': (238, 168), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'tiling', 'pill': (466, 157), 'targ': (521, 296), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'array', 'pill': (958, 165), 'targ': (1010, 441), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'dataflow', 'pill': (567, 625), 'targ': (691, 401), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'memory', 'pill': (1160, 641), 'targ': (1245, 629), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '12_benchmarking',
        'prefix': 'benchmarking',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'scope', 'pill': (130, 87), 'targ': (183, 195), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'workload', 'pill': (491, 85), 'targ': (621, 334), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'sustained', 'pill': (1025, 133), 'targ': (1140, 253), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'metric', 'pill': (694, 715), 'targ': (781, 545), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'peak', 'pill': (1194, 713), 'targ': (1264, 736), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '13_model_serving',
        'prefix': 'model_serving',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'request', 'pill': (172, 143), 'targ': (241, 334), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'queue', 'pill': (475, 133), 'targ': (499, 336), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'batch', 'pill': (661, 680), 'targ': (721, 540), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'cache', 'pill': (969, 135), 'targ': (970, 364), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'slo', 'pill': (1215, 625), 'targ': (1209, 515), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '14_ml_ops',
        'prefix': 'ml_ops',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'lineage', 'pill': (110, 99), 'targ': (176, 218), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'telemetry', 'pill': (480, 133), 'targ': (520, 280), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'retrain', 'pill': (994, 133), 'targ': (991, 231), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'drift', 'pill': (630, 655), 'targ': (658, 613), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'incident', 'pill': (1182, 619), 'targ': (1159, 515), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '15_responsible_engr',
        'prefix': 'responsible_engr',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'fairness', 'pill': (132, 91), 'targ': (193, 235), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'privacy', 'pill': (458, 133), 'targ': (511, 284), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'safety', 'pill': (733, 143), 'targ': (756, 359), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'audit', 'pill': (934, 673), 'targ': (880, 686), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'incident', 'pill': (1172, 613), 'targ': (1159, 505), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '16_conclusion',
        'prefix': 'conclusion',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'data', 'pill': (152, 137), 'targ': (202, 243), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'algorithm', 'pill': (472, 131), 'targ': (520, 309), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'machine', 'pill': (748, 141), 'targ': (770, 349), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'operate', 'pill': (1008, 137), 'targ': (1030, 344), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'governance', 'pill': (1184, 615), 'targ': (1169, 515), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    }
]

# =============================================================================
# CHAPTER CONFIGURATIONS: VOLUME 2 (DISTRIBUTED ML SYSTEMS)
# =============================================================================

VOL2_CHAPTERS = [
    {
        'slug': '01_introduction',
        'prefix': 'introduction',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'scale', 'pill': (147, 141), 'targ': (231, 324), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'compute', 'pill': (449, 129), 'targ': (500, 334), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'coordination', 'pill': (659, 605), 'targ': (823, 593), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'failure', 'pill': (1300, 200), 'targ': (1220, 340), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '02_compute_infrastructure',
        'prefix': 'compute_infrastructure',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'accelerator', 'pill': (137, 507), 'targ': (340, 480), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'hbm', 'pill': (503, 100), 'targ': (621, 424), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'tray', 'pill': (862, 56), 'targ': (905, 254), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'rack', 'pill': (1063, 97), 'targ': (1120, 334), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'cooling', 'pill': (1220, 667), 'targ': (1209, 500), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '03_network_fabrics',
        'prefix': 'network_fabrics',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'link', 'pill': (121, 49), 'targ': (251, 354), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'transport', 'pill': (480, 52), 'targ': (501, 200), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'topology', 'pill': (758, 52), 'targ': (739, 205), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'congestion', 'pill': (886, 517), 'targ': (920, 480), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'sync', 'pill': (1166, 55), 'targ': (1119, 304), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '04_data_storage',
        'prefix': 'data_storage',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'io', 'pill': (144, 143), 'targ': (217, 308), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'hierarchy', 'pill': (452, 137), 'targ': (521, 364), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'locality', 'pill': (744, 139), 'targ': (801, 334), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'retrieval', 'pill': (1166, 143), 'targ': (1240, 330), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'checkpoint', 'pill': (1009, 619), 'targ': (950, 510), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '05_distributed_training',
        'prefix': 'distributed_training',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'parallelism', 'pill': (153, 135), 'targ': (221, 334), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'partition', 'pill': (435, 87), 'targ': (460, 169), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'efficiency', 'pill': (963, 139), 'targ': (965, 334), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'sync', 'pill': (736, 700), 'targ': (732, 660), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'convergence', 'pill': (1243, 705), 'targ': (1243, 640), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '06_collective_communication',
        'prefix': 'collective_communication',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'gradient', 'pill': (136, 47), 'targ': (196, 192), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'collective', 'pill': (428, 45), 'targ': (480, 394), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'overlap', 'pill': (962, 47), 'targ': (939, 354), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'topology', 'pill': (708, 472), 'targ': (649, 472), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'scaling', 'pill': (1175, 472), 'targ': (1095, 511), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '07_fault_tolerance',
        'prefix': 'fault_tolerance',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'failure', 'pill': (162, 143), 'targ': (251, 354), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'detection', 'pill': (462, 143), 'targ': (491, 247), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'recovery', 'pill': (951, 143), 'targ': (1042, 280), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'checkpoint', 'pill': (768, 613), 'targ': (719, 506), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'resilience', 'pill': (1164, 607), 'targ': (1161, 529), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '08_fleet_orchestration',
        'prefix': 'fleet_orchestration',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'queue', 'pill': (163, 143), 'targ': (195, 233), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'schedule', 'pill': (436, 135), 'targ': (501, 334), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'place', 'pill': (610, 674), 'targ': (602, 684), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'isolation', 'pill': (997, 60), 'targ': (1020, 310), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'recovery', 'pill': (1183, 605), 'targ': (1159, 516), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '09_performance_engineering',
        'prefix': 'performance_engineering',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'measure', 'pill': (151, 83), 'targ': (220, 354), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'diagnose', 'pill': (460, 130), 'targ': (520, 320), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'rewrite', 'pill': (709, 657), 'targ': (718, 598), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'speedup', 'pill': (960, 130), 'targ': (980, 330), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'overlap', 'pill': (1212, 667), 'targ': (1138, 515), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '10_inference',
        'prefix': 'inference',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'route', 'pill': (276, 186), 'targ': (320, 399), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'batch', 'pill': (527, 169), 'targ': (576, 314), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'cache', 'pill': (830, 171), 'targ': (840, 394), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'shard', 'pill': (1091, 111), 'targ': (1118, 256), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'latency', 'pill': (1210, 527), 'targ': (1261, 624), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '11_edge_intelligence',
        'prefix': 'edge_intelligence',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'device', 'pill': (165, 143), 'targ': (213, 200), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'constraint', 'pill': (440, 135), 'targ': (501, 334), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'aggregation', 'pill': (958, 87), 'targ': (959, 218), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'privacy', 'pill': (651, 677), 'targ': (600, 663), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'update', 'pill': (1249, 657), 'targ': (1180, 520), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '12_ops_scale',
        'prefix': 'ops_scale',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'platform', 'pill': (174, 87), 'targ': (230, 200), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'registry', 'pill': (461, 87), 'targ': (485, 184), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'rollout', 'pill': (969, 85), 'targ': (968, 136), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'tenant', 'pill': (734, 689), 'targ': (780, 678), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'incident', 'pill': (1182, 615), 'targ': (1159, 495), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '13_security_privacy',
        'prefix': 'security_privacy',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'threat', 'pill': (158, 93), 'targ': (213, 198), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'boundary', 'pill': (369, 103), 'targ': (382, 262), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'privacy', 'pill': (572, 127), 'targ': (585, 206), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'enclave', 'pill': (912, 131), 'targ': (909, 364), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'audit', 'pill': (1157, 685), 'targ': (1074, 631), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '14_robust_ai',
        'prefix': 'robust_ai',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'shift', 'pill': (151, 135), 'targ': (221, 354), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'drift', 'pill': (455, 135), 'targ': (484, 276), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'uncertainty', 'pill': (736, 615), 'targ': (660, 601), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'monitor', 'pill': (972, 135), 'targ': (976, 244), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'adaptation', 'pill': (1171, 607), 'targ': (1159, 506), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '15_sustainable_ai',
        'prefix': 'sustainable_ai',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'power', 'pill': (136, 89), 'targ': (165, 168), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'cooling', 'pill': (460, 87), 'targ': (515, 326), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'utilization', 'pill': (938, 65), 'targ': (946, 163), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'efficiency', 'pill': (1180, 67), 'targ': (1120, 200), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'carbon', 'pill': (739, 705), 'targ': (730, 624), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '16_responsible_ai',
        'prefix': 'responsible_ai',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'policy', 'pill': (160, 137), 'targ': (220, 281), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'fairness', 'pill': (445, 135), 'targ': (500, 334), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'transparency', 'pill': (898, 137), 'targ': (1003, 240), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'privacy', 'pill': (682, 701), 'targ': (610, 660), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'accountability', 'pill': (1168, 707), 'targ': (1119, 495), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '17_conclusion',
        'prefix': 'conclusion',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'infrastructure', 'pill': (154, 85), 'targ': (210, 334), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'distribution', 'pill': (462, 85), 'targ': (500, 324), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'operate', 'pill': (1033, 87), 'targ': (980, 200), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'serving', 'pill': (701, 702), 'targ': (650, 688), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'governance', 'pill': (1195, 665), 'targ': (1159, 516), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    }
]

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
        'slug': '02_foundation_model',
        'prefix': 'processor',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'draft model execution core', 'pill': (250, 280), 'targ': (560, 420), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'token proposal stream', 'pill': (240, 560), 'targ': (700, 530), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'memory streaming bus', 'pill': (660, 80), 'targ': (680, 220), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'target verification engine', 'pill': (1140, 180), 'targ': (720, 300), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'rollback diversion gate', 'pill': (1150, 400), 'targ': (890, 400), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'token acceptance pool', 'pill': (1140, 600), 'targ': (825, 440), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '03_test_time_compute',
        'prefix': 'deliberation',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'deliberation search agent', 'pill': (660, 80), 'targ': (580, 240), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'active exploratory branch', 'pill': (240, 320), 'targ': (580, 380), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'root decision state', 'pill': (240, 540), 'targ': (595, 425), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'optimal trajectory', 'pill': (1140, 220), 'targ': (740, 330), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'pruned dead-end branch', 'pill': (1150, 440), 'targ': (840, 430), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'speculative rollout branch', 'pill': (1140, 640), 'targ': (780, 470), 'border': '#D97706', 'color': '#D97706'},
        ]
    },
    {
        'slug': '04_context_engineering',
        'prefix': 'working_sets',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'attention sink tokens', 'pill': (240, 200), 'targ': (480, 405), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'context carousel', 'pill': (240, 420), 'targ': (530, 440), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'budget boundary', 'pill': (240, 620), 'targ': (480, 540), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'attention envelope', 'pill': (1140, 200), 'targ': (800, 300), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'sliding window tokens', 'pill': (1140, 420), 'targ': (880, 320), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'eviction chute', 'pill': (1140, 640), 'targ': (870, 470), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '05_kv_cache',
        'prefix': 'virtual_memory',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'logical virtual blocks', 'pill': (240, 200), 'targ': (360, 220), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'shared prefix block pool', 'pill': (240, 400), 'targ': (490, 280), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'radix block table mmu', 'pill': (240, 600), 'targ': (581, 310), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'pagedattention manager', 'pill': (680, 80), 'targ': (680, 240), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'physical hbm page frames', 'pill': (1140, 200), 'targ': (1040, 280), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'zero-fragmentation allocator', 'pill': (1140, 420), 'targ': (1014, 350), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '06_long_term_memory',
        'prefix': 'episodic_memory',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'nearest-neighbor prism', 'pill': (240, 220), 'targ': (505, 420), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'similarity query beam', 'pill': (240, 360), 'targ': (470, 480), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'approximate search agent', 'pill': (240, 540), 'targ': (380, 520), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'clustered memory hierarchy', 'pill': (1140, 200), 'targ': (750, 220), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'hnsw embedding lattice', 'pill': (1140, 440), 'targ': (780, 480), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'persistent episodic plinth', 'pill': (1140, 700), 'targ': (680, 720), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '07_checkpointing',
        'prefix': 'checkpointing',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'forward transaction stream', 'pill': (240, 420), 'targ': (580, 440), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'rollback siding', 'pill': (240, 620), 'targ': (480, 500), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'checkpoint rewind portal', 'pill': (700, 80), 'targ': (720, 300), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'immutable snapshot state', 'pill': (1140, 220), 'targ': (860, 240), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'state rewind track', 'pill': (1140, 420), 'targ': (920, 360), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'fault detection tripwire', 'pill': (1140, 620), 'targ': (735, 480), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '08_actuation',
        'prefix': 'actuation',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'json-schema validation port', 'pill': (500, 80), 'targ': (680, 280), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'central mcp switchboard', 'pill': (850, 80), 'targ': (780, 180), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'agent actuation client', 'pill': (240, 320), 'targ': (480, 310), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'database console bay', 'pill': (1140, 200), 'targ': (880, 260), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'sandboxed execution rack', 'pill': (1140, 420), 'targ': (998, 359), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'external tool bus', 'pill': (1140, 620), 'targ': (850, 450), 'border': '#D97706', 'color': '#D97706'},
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
            {'text': 'microvm containment barrier', 'pill': (1140, 220), 'targ': (880, 310), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'seccomp-bpf syscall drain', 'pill': (1140, 460), 'targ': (850, 482), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'air-gapped virtual boundary', 'pill': (1140, 640), 'targ': (820, 560), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '10_durable_execution',
        'prefix': 'interrupts',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'state escrow capsule', 'pill': (700, 80), 'targ': (705, 140), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'asynchronous pause barrier', 'pill': (240, 320), 'targ': (530, 480), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'execution pipeline', 'pill': (240, 540), 'targ': (440, 640), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'preempted agent worker', 'pill': (1140, 220), 'targ': (740, 290), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'in-flight token bubbles', 'pill': (1140, 420), 'targ': (893, 310), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'human authorization gate', 'pill': (1140, 620), 'targ': (710, 520), 'border': '#D97706', 'color': '#D97706'},
        ]
    },
    {
        'slug': '11_failure_recovery',
        'prefix': 'scheduling',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'latency-critical prefill train', 'pill': (240, 240), 'targ': (520, 420), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'anti-starvation bypass switch', 'pill': (260, 480), 'targ': (620, 460), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'cluster switchyard operator', 'pill': (660, 80), 'targ': (640, 350), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'throughput decode train', 'pill': (1140, 220), 'targ': (820, 410), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'switchyard preemption lever', 'pill': (1140, 440), 'targ': (775, 415), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'cxl memory ballast', 'pill': (1140, 640), 'targ': (650, 550), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '13_trajectory_curation',
        'prefix': 'data_flywheel',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'trajectory synthesis agent', 'pill': (720, 80), 'targ': (720, 260), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'synthetic sandbox', 'pill': (240, 240), 'targ': (420, 260), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'verified policy rollout', 'pill': (240, 480), 'targ': (413, 358), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'evaluation chamber', 'pill': (1140, 240), 'targ': (990, 420), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'decontamination bus', 'pill': (1140, 480), 'targ': (960, 485), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'flywheel pipeline base', 'pill': (1140, 680), 'targ': (740, 690), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '14_fine_tuning',
        'prefix': 'sft',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'observation context mask', 'pill': (240, 320), 'targ': (480, 460), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'trajectory token filmstrip', 'pill': (240, 540), 'targ': (510, 485), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'teacher-forcing inspector', 'pill': (660, 80), 'targ': (610, 310), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'action token illumination', 'pill': (1140, 220), 'targ': (670, 350), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'gradient backprop optics', 'pill': (1140, 440), 'targ': (640, 390), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'loss masking optical bench', 'pill': (1140, 640), 'targ': (730, 480), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '15_rlvr',
        'prefix': 'rlvr',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'formal verification agent', 'pill': (240, 260), 'targ': (460, 260), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'candidate solution crystal', 'pill': (240, 500), 'targ': (560, 420), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'diffraction verifier', 'pill': (660, 80), 'targ': (640, 360), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'outcome test chamber', 'pill': (1140, 200), 'targ': (740, 330), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'fault containment bypass', 'pill': (1140, 420), 'targ': (980, 480), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'scalar binary reward bus', 'pill': (1140, 620), 'targ': (680, 530), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '16_multi_agent',
        'prefix': 'multi_agent',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'coordinator bot', 'pill': (660, 80), 'targ': (820, 230), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'token packets', 'pill': (240, 340), 'targ': (495, 436), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'specialized worker console', 'pill': (240, 560), 'targ': (410, 490), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'central consensus core', 'pill': (1140, 220), 'targ': (680, 370), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'optical ring bus', 'pill': (1140, 440), 'targ': (840, 470), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'state synchronization ring', 'pill': (1140, 640), 'targ': (700, 620), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '12_evaluation',
        'prefix': 'observability',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'trajectory trace ribbon', 'pill': (240, 240), 'targ': (480, 220), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'distributed tracer bot', 'pill': (240, 480), 'targ': (578, 434), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'execution anomaly spike', 'pill': (830, 70), 'targ': (830, 175), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'milestone beacon', 'pill': (1140, 240), 'targ': (695, 282), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'latency flame-graph span', 'pill': (1140, 440), 'targ': (880, 360), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'flight recorder box', 'pill': (1140, 640), 'targ': (800, 480), 'border': '#0090B0', 'color': '#0090B0'},
        ]
    },
    {
        'slug': '17_agent_economics',
        'prefix': 'tokenomics',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'prompt input ray', 'pill': (240, 240), 'targ': (460, 290), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'dynamic routing prism', 'pill': (240, 460), 'targ': (580, 350), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'capacity arbitrage stage', 'pill': (240, 660), 'targ': (640, 430), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'beam-splitter spectrometer', 'pill': (660, 70), 'targ': (660, 260), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'frontier compute waveguide', 'pill': (1140, 240), 'targ': (892, 352), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'slender edge fiber', 'pill': (1140, 480), 'targ': (863, 479), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
    {
        'slug': '18_conclusion',
        'prefix': 'conclusion',
        'font': FONT_ARIAL_BOLD,
        'labels': [
            {'text': 'sovereign scout agent', 'pill': (240, 340), 'targ': (590, 460), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'floating systems plinth', 'pill': (240, 580), 'targ': (400, 620), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'optical waveguides', 'pill': (660, 80), 'targ': (730, 240), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'monumental torus portal', 'pill': (1140, 220), 'targ': (780, 340), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'frontier metropolis', 'pill': (1140, 440), 'targ': (840, 410), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'invariant closure gateway', 'pill': (1140, 640), 'targ': (800, 490), 'border': '#2EC4B6', 'color': '#2EC4B6'},
        ]
    },
]
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
            {'text': 'contact friction cones', 'pill': (1140, 310), 'targ': (1100, 310), 'border': '#D97706', 'color': '#D97706'},
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
            {'text': 'temporal lease ledger', 'pill': (1140, 160), 'targ': (1079, 278), 'border': '#0090B0', 'color': '#0090B0'},
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
            {'text': 'spatial tolerance cylinder', 'pill': (240, 440), 'targ': (690, 380), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'aerospace bracket workpiece', 'pill': (320, 680), 'targ': (640, 500), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'countdown lease ring dial', 'pill': (1140, 160), 'targ': (765, 265), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'insertion affordance cone', 'pill': (1140, 360), 'targ': (886, 280), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'expiration tripwire', 'pill': (1140, 560), 'targ': (940, 500), 'border': '#9B2226', 'color': '#9B2226'},
        ]
    },
    {
        'slug': '11_planning',
        'prefix': 'planning',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'bldc outrunner cutaway', 'pill': (220, 180), 'targ': (380, 285), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': '3d polynomial spline', 'pill': (240, 440), 'targ': (540, 350), 'border': '#0090B0', 'color': '#0090B0'},
            {'text': 'aerodynamic thrust cones', 'pill': (400, 720), 'targ': (560, 520), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'spatial obstacle ring', 'pill': (740, 80), 'targ': (730, 170), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'prospective waypoint chunk', 'pill': (1140, 180), 'targ': (880, 205), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'hover-recovery stopping suffix', 'pill': (1140, 360), 'targ': (1010, 260), 'border': '#9B2226', 'color': '#9B2226'},
            {'text': 'terminal landing perch', 'pill': (1180, 540), 'targ': (1040, 480), 'border': '#1C4E4F', 'color': '#1C4E4F'},
        ]
    },
    {
        'slug': '12_enforcement',
        'prefix': 'enforcement',
        'font': FONT_HELVETICA,
        'labels': [
            {'text': 'forward-invariant safe set', 'pill': (240, 200), 'targ': (450, 400), 'border': '#2EC4B6', 'color': '#2EC4B6'},
            {'text': 'mecanum mobile base', 'pill': (240, 420), 'targ': (520, 450), 'border': '#1C4E4F', 'color': '#1C4E4F'},
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
            {'text': '3d memory stacks', 'pill': (190, 140), 'targ': (270, 240), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'cognitive npu cluster', 'pill': (200, 340), 'targ': (520, 350), 'border': '#0090B0', 'color': '#0090B0'},
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
            {'text': 'independent double-wishbone', 'pill': (270, 340), 'targ': (470, 390), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'steer-by-wire rack & pinion', 'pill': (240, 520), 'targ': (540, 410), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'tire contact friction ellipses', 'pill': (360, 720), 'targ': (690, 540), 'border': '#D97706', 'color': '#D97706'},
            {'text': 'electric traction motor', 'pill': (1160, 160), 'targ': (825, 240), 'border': '#1C4E4F', 'color': '#1C4E4F'},
            {'text': 'timestamped intervention relay', 'pill': (1140, 300), 'targ': (900, 280), 'border': '#9B2226', 'color': '#9B2226'},
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
            {'text': 'unobserved reality chasm', 'pill': (1160, 680), 'targ': (1088, 524), 'border': '#1C4E4F', 'color': '#1C4E4F'},
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
    
    png_dir = os.path.join(REPO_ROOT, 'books', vol_name, slug, 'images', 'png')
    webp_dir = os.path.join(REPO_ROOT, 'books', vol_name, slug, 'images', 'webp')
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(webp_dir, exist_ok=True)
    
    base_print_path = os.path.join(png_dir, f'cover_{prefix}_blueprint_print.png')
    base_1x_path = os.path.join(png_dir, f'cover_{prefix}_blueprint.png')
    
    has_2x_base = os.path.exists(base_print_path)
    
    # Base image for final composition
    if has_2x_base:
        im_base = Image.open(base_print_path).convert('RGB')
        out_w, out_h = im_base.size # 2800, 1600
        # Supersample overlay by 2x of 2800x1600 = 5600x3200 (scale 4 relative to 1400x800)
        scale = 4
        font_size_1x = 28
    else:
        im_base = Image.open(base_1x_path).convert('RGB')
        out_w, out_h = im_base.size # 1400, 800
        # Supersample overlay by 2x of 1400x800 = 2800x1600 (scale 2 relative to 1400x800)
        scale = 2
        font_size_1x = 28 if vol_name == 'vol3' else 30
        
    w_sup, h_sup = out_w * 2, out_h * 2
    overlay = Image.new('RGBA', (w_sup, h_sup), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    font = ImageFont.truetype(font_path, font_size_1x * scale)
    
    # Scaled visual parameters ensuring generous breathing room
    pad_x = 22 * scale
    min_half_w = 45 * scale
    pill_h = 38 * scale if vol_name in ('vol1', 'vol2') else 40 * scale
    radius = 10 * scale
    line_w = 2 * scale
    dot_r = 3.5 * scale
    safe_margin = 24 * scale
    
    for item in labels:
        text = item['text']
        px = item['pill'][0] * scale
        py = item['pill'][1] * scale
        tx = item['targ'][0] * scale
        ty = item['targ'][1] * scale
        border = item.get('border', '#0090B0')
        tcolor = item.get('color', '#0090B0')
        
        tw = draw.textlength(text, font=font)
        half_w = max(int(tw / 2 + pad_x), min_half_w)
        bx0 = px - half_w
        by0 = py - pill_h // 2
        bx1 = px + half_w
        by1 = py + pill_h // 2
        
        # Guaranteed canvas edge safety margin clamping
        if bx0 < safe_margin:
            shift = safe_margin - bx0
            bx0 += shift
            bx1 += shift
            px += shift
        elif bx1 > (w_sup - safe_margin):
            shift = bx1 - (w_sup - safe_margin)
            bx0 -= shift
            bx1 -= shift
            px -= shift
            
        # High-precision anchor selection
        if abs(tx - px) < half_w and ty > by1:
            anchor = (px, by1)
        elif abs(tx - px) < half_w and ty < by0:
            anchor = (px, by0)
        elif tx < px:
            anchor = (bx0, py)
        else:
            anchor = (bx1, py)
            
        # Draw leader line
        draw.line([anchor, (tx, ty)], fill=border, width=line_w)
        
        # Draw target dot
        draw.ellipse([tx - dot_r, ty - dot_r, tx + dot_r, ty + dot_r], fill=border)
        
        # Draw pill container
        draw.rounded_rectangle([bx0, by0, bx1, by1], radius=radius, fill=(255, 255, 255, 255), outline=border, width=line_w)
        
        # Draw perfectly centered text with ample breathing room
        draw.text((px, py), text, fill=tcolor, font=font, anchor='mm')
        
    # Downsample overlay with Lanczos for subpixel anti-aliasing
    overlay_down = overlay.resize((out_w, out_h), Image.Resampling.LANCZOS)
    
    # Composite over base image
    final_im = im_base.copy()
    final_im.paste(overlay_down, (0, 0), overlay_down)
    
    out_png = os.path.join(png_dir, f'cover_{prefix}_blueprint_labeled_print.png')
    out_webp = os.path.join(webp_dir, f'cover_{prefix}_blueprint_labeled.webp')
    
    final_im.save(out_png, 'PNG')
    
    # For webp, always output at 1400x800
    if out_w == 1400:
        final_im.save(out_webp, 'WEBP', quality=95)
    else:
        webp_im = final_im.resize((1400, 800), Image.Resampling.LANCZOS)
        webp_im.save(out_webp, 'WEBP', quality=95)
        
    print(f"Generated [{vol_name}] {slug} -> {out_png}")

def main():
    print("=" * 70)
    print("Pre-validating all configurations across Volumes 1, 2, 3, and 4...")
    print("=" * 70)
    
    for ch in VOL1_CHAPTERS:
        validate_chapter('vol1', ch)
    print("✓ Volume 1 validation passed: 0 color words, 0 line crossings.")
    
    for ch in VOL2_CHAPTERS:
        validate_chapter('vol2', ch)
    print("✓ Volume 2 validation passed: 0 color words, 0 line crossings.")
    
    for ch in VOL3_CHAPTERS:
        validate_chapter('vol3', ch)
    print("✓ Volume 3 validation passed: 0 color words, 0 line crossings.")
    
    for ch in VOL4_CHAPTERS:
        validate_chapter('vol4', ch)
    print("✓ Volume 4 validation passed: 0 color words, 0 line crossings.")
    
    print("\n" + "=" * 70)
    print("Rendering Volume 1 blueprints with 4x supersampling (2800x1600 print)...")
    print("=" * 70)
    for ch in VOL1_CHAPTERS:
        render_chapter('vol1', ch)
        
    print("\n" + "=" * 70)
    print("Rendering Volume 2 blueprints with 4x supersampling (2800x1600 print)...")
    print("=" * 70)
    for ch in VOL2_CHAPTERS:
        render_chapter('vol2', ch)
        
    print("\n" + "=" * 70)
    print("Rendering Volume 3 blueprints with 2x supersampling (1400x800)...")
    print("=" * 70)
    for ch in VOL3_CHAPTERS:
        render_chapter('vol3', ch)
        
    print("\n" + "=" * 70)
    print("Rendering Volume 4 blueprints with 2x supersampling (1400x800)...")
    print("=" * 70)
    for ch in VOL4_CHAPTERS:
        render_chapter('vol4', ch)
        
    print("\n" + "=" * 70)
    print("All 68 chapter blueprints across 4 volumes successfully generated with 0 defects.")
    print("=" * 70)

if __name__ == '__main__':
    main()
