"""
energy.py
Energy helpers for compute and data movement.
"""

from .constants import ureg


def energy_move(bytes_moved, energy_per_byte):
    return bytes_moved * energy_per_byte


def energy_compute(flops, energy_per_flop):
    return flops * energy_per_flop


def energy_total(bytes_moved, energy_per_byte, flops, energy_per_flop):
    return energy_move(bytes_moved, energy_per_byte) + energy_compute(flops, energy_per_flop)
